"""端到端训练（§3.4 / notebook 11 的后端）—— 用我们自己的模块训练真正的 CNN。

架构（3 个 stage，早期下采样；借 Task A 的架构直觉）：
    3x32x32 --conv3x3-> 32x30x30 --SN,GELU-->
    32x30x30 --conv3x3 s2-> 64x14x14 --SN,GELU-->
    64x14x14 --conv3x3 s2-> 128x6x6 --SN,GELU-->
    GAP -> 128 --Linear-> 100

跑法（在 task_b/ 下）：
    python tests/train_npnn.py                       # 默认 20 epoch、全量 45k
    python tests/train_npnn.py --epochs 2 --out smoke   # 冒烟测试

产物（runs/<out>/）：
    config.json / metrics.csv / sn_weights.json / params.npz / curves.png

⚠️ 训练用 float32（as_dtype 必须在建 Adam **之前**调）；数据/前向/反向/优化器全是我们自己的实现。
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

import config                                   # noqa: E402  （只依赖标准库）
config.limit_threads()                          # ⚠️ 必须在 import numpy 之前：BLAS 首次 import 就锁线程数

import numpy as np                              # noqa: E402
from npnn import (Adam, CIFAR100, CrossEntropyLoss, batches)   # noqa: E402
from npnn.data import Split                     # noqa: E402
from tests.arch import ARCH, build_npnn         # noqa: E402


def build_model():
    return build_npnn(ARCH)


def sn_weights(model):
    """每个 SN 层的 (w_BN, w_IN, w_LN)（对通道取平均），用于报告图。"""
    out = []
    for i, m in enumerate(model.modules):
        if isinstance(m, SwitchNorm2d):
            out.append(m.w.mean(axis=1).tolist())      # m.w 是 softmax(λ)，(3,C)
    return out


def evaluate(model, criterion, split, batch_size=256):
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in batches(split, batch_size, shuffle=False, drop_last=False):
        logits = model.forward(xb)
        loss_sum += float(criterion.forward(logits, yb)) * len(yb)
        correct += int((logits.argmax(axis=1) == yb).sum())
        total += len(yb)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--out', default='cnn_sn')
    args = ap.parse_args()

    out_dir = TASKB_ROOT / 'runs' / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = CIFAR100()
    train_eval = Split(ds.train.images[:5000], ds.train.labels[:5000])   # 只用于报 train 指标
    np.random.seed(args.seed)
    model = build_model()
    model.as_dtype(np.float32)                 # 训练精度；必须在建优化器之前
    criterion = CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr)

    print(f'arch: {len(model.modules)} layers, params: {model.n_params():,}')
    print(f'train {len(ds.train):,} / val {len(ds.val):,} / test {len(ds.test):,}')
    print(f'epochs={args.epochs} batch={args.batch} lr={args.lr} '
          f'threads={config.BLAS_THREADS}')

    rng = np.random.default_rng(args.seed)
    rows, sn_hist = [], []
    t_start = time.perf_counter()

    print(f'{"epoch":>5s}{"train loss":>12s}{"train acc":>11s}{"val loss":>10s}'
          f'{"val acc":>9s}{"time":>9s}')
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        for xb, yb in batches(ds.train, args.batch, rng=rng):
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            model.backward(criterion.backward())
            opt.step(model.gradients())
        dt = time.perf_counter() - t0

        tr_loss, tr_acc = evaluate(model, criterion, train_eval)
        va_loss, va_acc = evaluate(model, criterion, ds.val)
        sn_hist.append(sn_weights(model))
        rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                         val_loss=va_loss, val_acc=va_acc, epoch_time=dt))
        print(f'{epoch:>5d}{tr_loss:>12.4f}{tr_acc:>10.1%}{va_loss:>10.4f}'
              f'{va_acc:>9.1%}{dt:>8.1f}s', flush=True)

    total_time = time.perf_counter() - t_start

    # ---- 落盘：metrics.csv / sn_weights.json / params.npz / config.json
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    (out_dir / 'sn_weights.json').write_text(json.dumps({
        'epochs': [r['epoch'] for r in rows],
        'hist': sn_hist,                       # hist[e][layer] = [w_bn, w_in, w_ln]
    }, indent=2), encoding='utf-8')

    np.savez(out_dir / 'params.npz', **model.state_dict())

    (out_dir / 'config.json').write_text(json.dumps({
        'arch': 'conv32-sn-gelu / conv64-s2-sn-gelu / conv128-s2-sn-gelu / gap / linear',
        'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr, 'seed': args.seed,
        'optimizer': 'Adam(b1=0.9,b2=0.999,eps=1e-8)',
        'dtype': 'float32', 'threads': config.BLAS_THREADS,
        'n_params': model.n_params(), 'train_time_s': total_time,
        'data': 'CIFAR-100 45k train / 5k val (split from train)',
    }, indent=2), encoding='utf-8')

    print(f'\nfinal: train acc {rows[-1]["train_acc"]:.2%}  val acc {rows[-1]["val_acc"]:.2%}')
    print(f'total {total_time / 60:.1f} min   saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
