import csv, json, random, time, numpy as np, torch

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run(run_name, history, model, config):
    """Save one experiment to runs/<run_name>/: config.json, metrics.csv, curves.png, model.pt.

    history: the dict returned by engine.fit(); model: the trained nn.Module.
    """
    import matplotlib.pyplot as plt
    from src.config import RUNS_DIR

    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"]
        has_ema = "ema_test_acc" in history          # older runs have no EMA columns; keep the original 6
        if has_ema:
            cols += ["ema_test_loss", "ema_test_acc"]
        w.writerow(cols)
        for i in range(len(history["train_loss"])):
            row = [i + 1, history["train_loss"][i], history["train_acc"][i],
                   history["test_loss"][i], history["test_acc"][i],
                   round(history["epoch_time"][i], 1)]
            if has_ema:
                row += [history["ema_test_loss"][i], history["ema_test_acc"][i]]
            w.writerow(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["test_loss"], label="eval")
    if has_ema:
        ax1.plot(history["ema_test_loss"], label="eval (EMA)", linestyle="--")
    ax1.set_title("loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="eval")
    if has_ema:
        ax2.plot(history["ema_test_acc"], label="eval (EMA)", linestyle="--")
    ax2.set_title("accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.savefig(run_dir / "curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)                              # keep notebook memory tidy

    torch.save(model.state_dict(), run_dir / "model.pt")
    msg = f"saved -> {run_dir} | final acc: {history['test_acc'][-1]:.4f}"
    if has_ema:
        msg += f" | EMA: {history['ema_test_acc'][-1]:.4f}"
    print(msg)
    return run_dir


def check_gpu(size=4096, iters=8, verbose=True):
    """~15 s GPU state check: a healthy 4060 does ~20-40 ms per 4096x4096 matmul.

    ~300 ms means the laptop GPU fell into its slow state (14x slower training) - check the
    performance mode / reboot before starting a long run. Returns ms per matmul, or None if
    there is no CUDA device.
    """
    if not torch.cuda.is_available():
        if verbose: print("no CUDA device - training on CPU")
        return None

    a = torch.randn(size, size, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        a @ a
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000

    if verbose:
        print(f"avg ms/matmul: {ms:.0f}  ->",
              "FAST, good to go" if ms < 100 else "SLOW - check performance mode!")
    return ms


def warmup_cosine(opt, epochs, warmup=2):
    """2-epoch linear warmup then cosine decay - the lr schedule used since exp_104.

    fit() steps the scheduler once per epoch, so T_max follows `epochs` for any run length.
    """
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))],
        milestones=[warmup])