"""Torch 参照训练（§3.5 的对照）—— 同一个网络、同样的预算，看精度与速度差多少。

用 `tests/l2_parity.py` 里**同一份 NET 配置**建 torch 网络（含 TorchSN），
喂**同一条数据管线**、同样的超参，跑同样的 epoch 数。

跑法（在 task_b/ 下，等 numpy 训练跑完再跑，避免抢 CPU 影响计时对比）：
    python tests/compare_torch_train.py [--epochs 20] [--out torch_ref]

产物：runs/<out>/{config.json, metrics.csv}
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

TASKB_ROOT = Path(__file__).resolve().parents[1]
if str(TASKB_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKB_ROOT))

import config                                   # noqa: E402
config.limit_threads()

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402

from npnn import CIFAR100, batches              # noqa: E402
from npnn.data import Split                     # noqa: E402
from tests.arch import ARCH, build_torch        # noqa: E402


def evaluate(model, ce, split, batch_size=256):
    total, correct, loss_sum = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for xb, yb in batches(split, batch_size, shuffle=False, drop_last=False):
            logits = model(torch.from_numpy(xb))
            loss_sum += float(ce(logits, torch.from_numpy(yb))) * len(yb)
            correct += int((logits.argmax(dim=1) == torch.from_numpy(yb)).sum())
            total += len(yb)
    model.train()
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--out', default='torch_ref')
    args = ap.parse_args()

    out_dir = TASKB_ROOT / 'runs' / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    ds = CIFAR100()
    train_eval = Split(ds.train.images[:5000], ds.train.labels[:5000])

    model = build_torch(ARCH, torch.float32)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'epochs={args.epochs} batch={args.batch} lr={args.lr} '
          f'threads={config.BLAS_THREADS}  (CPU, torch {torch.__version__})')

    rng = np.random.default_rng(args.seed)
    rows = []
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        for xb, yb in batches(ds.train, args.batch, rng=rng):
            logits = model(torch.from_numpy(xb))
            loss = ce(logits, torch.from_numpy(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = time.perf_counter() - t0
        tr_loss, tr_acc = evaluate(model, ce, train_eval)
        va_loss, va_acc = evaluate(model, ce, ds.val)
        rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                         val_loss=va_loss, val_acc=va_acc, epoch_time=dt))
        print(f'{epoch:>5d}{tr_loss:>12.4f}{tr_acc:>10.1%}{va_loss:>10.4f}'
              f'{va_acc:>9.1%}{dt:>8.1f}s', flush=True)

    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (out_dir / 'config.json').write_text(json.dumps({
        'arch': 'same NET as tests/l2_parity.py (conv/SN/GELU/conv/SN/GELU/conv/SN/GELU/GAP/linear)',
        'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr, 'seed': args.seed,
        'optimizer': 'torch.optim.Adam', 'dtype': 'float32', 'threads': config.BLAS_THREADS,
        'torch': torch.__version__, 'train_time_s': time.perf_counter() - t_start,
        'n_params': sum(p.numel() for p in model.parameters()),
    }, indent=2), encoding='utf-8')

    print(f'\nfinal: train acc {rows[-1]["train_acc"]:.2%}  val acc {rows[-1]["val_acc"]:.2%}')
    print(f'saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
