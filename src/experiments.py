
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.config import RUNS_DIR
from src.data import build_loaders_val
from src.engine import fit, evaluate_topk, ensemble_eval
from src.utils import set_seed, save_run, warmup_cosine

DEFAULT_SEEDS = (0, 42, 123)
COLLAPSE_ACC = 0.05      # val acc below this gets flagged as collapsed (kept, never rerolled)


def build_experiment_metadata(arch, tier, aug="crop+flip", optimizer="Adam",
                              lr="1e-3 + warmup/cos"):
    """Build the metadata written to one run's ``config.json``.

    This helper only describes an experiment; it does not create a model or
    start training.  Keeping the metadata construction here prevents notebook
    cells from duplicating the reporting schema.  The keys are part of the
    contract consumed by ``03_final_analysis.ipynb``.

    ``tier`` records the two-stage budget (``anchor``, ``screen`` or
    ``confirm``), while ``arch``, ``optimizer``, ``lr`` and ``aug`` describe
    the controlled variables used by the run.
    """
    return {
        "arch": arch,
        "optimizer": optimizer,
        "lr": lr,
        "aug": aug,
        "tier": tier,
    }


def build_architecture_ablation_batch():
    """Return the S1 architecture-ablation plan used by batch-2 notebooks.

    The returned dictionaries are configuration only; ``run_batch`` performs
    the actual training.  Keeping this plan in ``src`` lets the preview cell
    and the execution cell construct the same batch independently, so running
    S1 never depends on a stale notebook variable.
    """
    from src.models import (
        cnn_down,
        cnn_residual,
        cnn_wide,
        cnn_deep,
        cnn_3stage,
        cnn_dw,
        cnn_dw_wide,
        cnn_groupnorm,
    )

    return [
        {"exp": "exp_200", "name": "base_down", "model": cnn_down,
         "seeds": (0, 42, 123),
         "extra": build_experiment_metadata("downsampled", "anchor")},
        {"exp": "exp_201", "name": "residual", "model": cnn_residual,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+residual", "screen")},
        {"exp": "exp_202", "name": "wide", "model": cnn_wide,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+wide", "screen")},
        {"exp": "exp_203", "name": "deep", "model": cnn_deep,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+deep", "screen")},
        {"exp": "exp_204", "name": "3stage", "model": cnn_3stage,
         "seeds": (0,),
         "extra": build_experiment_metadata("3stage", "screen")},
        {"exp": "exp_205", "name": "dw", "model": cnn_dw,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+dw", "screen")},
        {"exp": "exp_206", "name": "dw_wide", "model": cnn_dw_wide,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+dw+wide", "screen")},
        {"exp": "exp_207", "name": "groupnorm", "model": cnn_groupnorm,
         "seeds": (0,),
         "extra": build_experiment_metadata("down+gn", "screen")},
    ]


def run_experiment(exp_id, name, model_fn, optimizer_fn, extra=None, scheduler_fn=None,
                   loaders=None, seeds=DEFAULT_SEEDS, epochs=10, batch_size=128, device=None,
                   ema_decay=0.999):
    """Run one configuration across seeds; save every run; print summary + group plot.

    Run id: <exp_id>_<name>_s<seed>. `extra` carries the descriptive config fields
    (arch / optimizer / lr / aug ...) into config.json, which is what the master table reads.
    A collapsed run (val acc < 5%) is kept and flagged - never silently rerolled.
    ema_decay (default 0.999) additionally records a weight-EMA curve; pass None to switch it off.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = loaders if loaders is not None else build_loaders_val(batch_size=batch_size)[:2]
    criterion = torch.nn.CrossEntropyLoss()
    histories = {}

    for s in seeds:
        set_seed(s)
        model = model_fn().to(device)
        opt = optimizer_fn(model.parameters())
        sched = scheduler_fn(opt, epochs) if scheduler_fn else None
        hist = fit(model, tr, va, criterion, opt, epochs=epochs, device=device, scheduler=sched,
                   ema_decay=ema_decay)

        cfg = {"exp": f"{exp_id}_{name}", "epochs": epochs, "batch": batch_size,
               "seed": s, "eval_set": "val5k", **(extra or {})}
        save_run(f"{exp_id}_{name}_s{s}", hist, model, cfg)
        histories[s] = hist

        if hist["test_acc"][-1] < COLLAPSE_ACC:
            print(f"  !! seed {s}: val acc {hist['test_acc'][-1]:.4f} looks collapsed - keep it, flag it in the report")

    accs = [h["test_acc"][-1] for h in histories.values()]
    print(f"\n{exp_id}_{name}: seeds {list(seeds)} | val acc mean {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    if ema_decay:
        ema_accs = [h["ema_test_acc"][-1] for h in histories.values()]
        print(f"{' ' * len(exp_id + '_' + name)}  EMA   {list(seeds)} | val acc mean {np.mean(ema_accs):.4f} +/- {np.std(ema_accs):.4f}")

    times = [t for h in histories.values() for t in h["epoch_time"]]
    if times and max(times) > 4 * min(times):
        print("  !! epoch times vary wildly - possible GPU slow-state flip; check before trusting comparisons")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for s, h in histories.items():
        ax1.plot(h["test_loss"], alpha=0.7, label=f"seed {s}")
        ax2.plot(h["test_acc"], alpha=0.7, label=f"seed {s}")
    ax1.set_title("val loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("val acc"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.savefig(RUNS_DIR / f"{exp_id}_{name}_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return histories


def run_extra(arch, tier, aug="crop+flip", optimizer="Adam", lr="1e-3 + warmup/cos"):
    """Backward-compatible alias for :func:`build_experiment_metadata`.

    New notebook code should call ``build_experiment_metadata`` because it
    states clearly that the function builds run metadata rather than running
    an extra experiment.
    """
    return build_experiment_metadata(arch, tier, aug=aug, optimizer=optimizer, lr=lr)


def _saved_accs(exp_id, name, seeds):
    """读回已经落盘的 run 的最终 val acc —— 让跳过的配置也能出现在汇总里。"""
    accs = []
    for s in seeds:
        f = RUNS_DIR / f"{exp_id}_{name}_s{s}" / "metrics.csv"
        if f.exists():
            with open(f, newline="") as fh:
                accs.append(float(list(csv.DictReader(fh))[-1]["test_acc"]))
    return accs


def run_batch(batch, loaders=None, batch_size=128, epochs=10, skip_existing=True, device=None,
              ema_decay=0.999):
    """按顺序跑一批配置，最后打一张一行式汇总表。

    batch 里每一项是一个 dict：
        {"exp": "exp_201", "name": "residual", "model": cnn_residual,
         "extra": {...},                    # 写进 config.json 的描述字段（arch/lr/aug/tier...）
         "seeds": (0,), "epochs": 20,       # 省略则用函数参数的默认值（seeds 默认三个种子）
         "loaders": (tr, va), "scheduler": warmup_cosine}
    只有 exp / name / model 是必需的。单项可用 "ema_decay" 覆盖整批的 EMA 设置（None = 关掉）。

    skip_existing=True 时，某个配置的**全部**种子都已落盘就整项跳过 —— 长批次跑到一半中断，
    重跑这个 cell 不会把已完成的重做一遍。
    """
    summary = []
    for item in batch:
        exp_id, name, model_fn = item["exp"], item["name"], item["model"]
        seeds = item.get("seeds", DEFAULT_SEEDS)
        item_epochs = item.get("epochs", epochs)

        if skip_existing and all((RUNS_DIR / f"{exp_id}_{name}_s{s}" / "config.json").exists() for s in seeds):
            accs = _saved_accs(exp_id, name, seeds)
            print(f"[skip] {exp_id}_{name}: already saved ({len(accs)} seeds)")
        else:
            histories = run_experiment(exp_id, name, model_fn,
                                       item.get("optimizer_fn", lambda p: torch.optim.Adam(p, lr=1e-3)),
                                       extra=item.get("extra"), scheduler_fn=item.get("scheduler", warmup_cosine),
                                       loaders=item.get("loaders", loaders), seeds=seeds,
                                       epochs=item_epochs, batch_size=batch_size, device=device,
                                       ema_decay=item.get("ema_decay", ema_decay))
            accs = [h["test_acc"][-1] for h in histories.values()]

        summary.append((f"{exp_id}_{name}", accs, item_epochs, len(seeds)))

    print("\n" + "=" * 58)
    print(f"{'config':<26}{'val top1':>16}{'seeds':>7}{'epochs':>9}")
    print("-" * 58)
    for name, accs, epochs, n_seeds in summary:
        if not accs:
            print(f"{name:<26}{'--':>16}{0:>7}{epochs:>9}")
            continue
        val = f"{sum(accs) / len(accs):.4f}"
        if len(accs) > 1:
            val += f" ± {np.std(accs):.4f}"
        print(f"{name:<26}{val:>16}{len(accs):>7}{epochs:>9}")
    print("=" * 58)
    return summary


def run_final_retrain(exp_id, name, model_fn, train_loader, test_loader, monitor_loader,
                      extra=None, seeds=DEFAULT_SEEDS, epochs=60, batch_size=128, device=None,
                      tta=True, lr=1e-3):
    """最终交付：50k 全量重训，test **只测一次**（每个 seed 一次），产物写 ``runs_final/``。

    和 run_batch 的区别：
      * 没有 held-out val —— 50k 全部参与训练，每 epoch 只在 ``monitor_loader``
        （train 的一个子集、无增强）上记曲线，**它不是 test**；
      * test 只在训练结束后评估一次（可选 TTA），结果写进 config.json
        （test_top1 / test_top5 / test_top1_tta / test_top5_tta）；
      * 产物写 ``runs_final/``，不混进 ``runs/`` —— ``runs/`` 是 45k/5k 协议的，
        03_final_analysis 的主表只扫那里。
    所有 seed 跑完后额外报 3-seed 集成（±TTA），**全部分开报告、不做挑选**。
    """
    from src.config import RUNS_DIR
    from src.utils import warmup_cosine

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = RUNS_DIR.parent / "runs_final"
    criterion = torch.nn.CrossEntropyLoss()
    models, rows = [], []

    for s in seeds:
        set_seed(s)
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = warmup_cosine(opt, epochs)

        hist = fit(model, train_loader, monitor_loader, criterion, opt, epochs=epochs,
                   device=device, scheduler=sched)

        # ---- test：只在这里评估一次（训练全部结束之后）----
        _, t1, t5 = evaluate_topk(model, test_loader, criterion, device)
        row = {"seed": s, "test_top1": t1, "test_top5": t5}
        if tta:
            t1_tta, t5_tta = ensemble_eval([model], test_loader, device, tta=True)
            row.update({"test_top1_tta": t1_tta, "test_top5_tta": t5_tta})

        run_dir = out_root / f"{exp_id}_{name}_s{s}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"exp": f"{exp_id}_{name}", "epochs": epochs, "batch": batch_size, "seed": s,
               "eval_set": "test10k", "train_set": "train50k", "tta": tta,
               **row, **(extra or {})}
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        with open(run_dir / "metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "monitor_loss", "monitor_acc", "epoch_time"])
            for i in range(len(hist["train_loss"])):
                w.writerow([i + 1, hist["train_loss"][i], hist["train_acc"][i],
                            hist["test_loss"][i], hist["test_acc"][i],
                            round(hist["epoch_time"][i], 1)])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(hist["train_loss"], label="train")
        ax1.plot(hist["test_loss"], label="monitor (train subset)")
        ax1.set_title("loss"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.plot(hist["train_acc"], label="train")
        ax2.plot(hist["test_acc"], label="monitor (train subset)")
        ax2.set_title("accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.savefig(run_dir / "curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        torch.save(model.state_dict(), run_dir / "model.pt")
        models.append(model)
        rows.append(row)

        msg = f"[final] {exp_id}_{name} seed {s}: test top1 {t1:.4f} top5 {t5:.4f}"
        if tta:
            msg += f" | TTA {t1_tta:.4f}/{t5_tta:.4f}"
        print(msg)

    # ---- 3-seed 集成（概率平均；模型都还在显存里）----
    summary = {"seeds": rows}
    if len(models) > 1:
        e1, e5 = ensemble_eval(models, test_loader, device, tta=False)
        summary.update(ensemble_top1=e1, ensemble_top5=e5)
        msg = (f"[final] {exp_id}_{name}: {len(models)}-seed ensemble "
               f"test top1 {e1:.4f} top5 {e5:.4f}")
        if tta:
            e1t, e5t = ensemble_eval(models, test_loader, device, tta=True)
            summary.update(ensemble_tta_top1=e1t, ensemble_tta_top5=e5t)
            msg += f" | +TTA {e1t:.4f}/{e5t:.4f}"
        print(msg)

    t1s = [r["test_top1"] for r in rows]
    print(f"[final] {exp_id}_{name}: seeds {list(seeds)} | test top1 mean "
          f"{np.mean(t1s):.4f} +/- {np.std(t1s):.4f}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{exp_id}_{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
