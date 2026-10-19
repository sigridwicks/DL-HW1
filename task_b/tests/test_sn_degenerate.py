"""SwitchNorm2d 退化测试（§2.6 ★ 本模块最高性价比的证据）。

把 λ 设成 one-hot（等价于 w 变成 one-hot），SN 必须**逐位等于** PyTorch 的：

    w_BN = 1  →  nn.BatchNorm2d(C, eps=1e-5)                          （训练模式）
    w_IN = 1  →  nn.InstanceNorm2d(C, affine=True, track_running_stats=False)
    w_LN = 1  →  nn.LayerNorm([C, H, W])                              （作用在每个样本上）

这一条测试一次性验证四件事：
  · 三种统计量的**维度**对不对
  · **广播**对不对（(1,C,1,1) / (N,C,1,1) / (N,1,1,1)）
  · **有偏 vs 无偏方差**的约定对不对
  · **ε 加在哪里**（方差上、根号内）

跑法（在 task_b/ 下）：python tests/test_sn_degenerate.py
"""

import sys
from pathlib import Path

import numpy as np

TASKB_ROOT = Path(__file__).resolve().parents[1]
if str(TASKB_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKB_ROOT))

import torch                                        # noqa: E402

from npnn import SwitchNorm2d                       # noqa: E402

BIG = 60.0          # softmax 后 exp(-60) ≈ 8.8e-27，float64 下就是精确 one-hot


def _one_hot(layer, which):
    lam = np.zeros_like(layer.lam)
    lam[which, :] = BIG
    layer.lam = lam


def _case(name, make_torch, which, shape, seed=0):
    rng = np.random.default_rng(seed)
    c = shape[1]
    x = rng.standard_normal(shape)

    layer = SwitchNorm2d(c)
    _one_hot(layer, which)
    y = layer.forward(x)

    t_layer = make_torch(c)
    x_t = torch.tensor(x)
    with torch.no_grad():
        y_t = t_layer(x_t)

    diff = float(np.abs(y - y_t.numpy()).max())
    print(f'  {name:22s} max|Δ| = {diff:.3e}   {"PASS" if diff < 1e-9 else "FAIL"}')
    return diff < 1e-9, diff


def main():
    shape = (2, 3, 5, 6)          # N=2, C=3, H=5, W=6
    print(f'退化测试输入形状 {shape}，ε = 1e-5，有偏方差\n')

    ok1, d1 = _case('w_BN = 1 → BatchNorm',
                    lambda c: torch.nn.BatchNorm2d(c, eps=1e-5).double(),
                    0, shape)
    ok2, d2 = _case('w_IN = 1 → InstanceNorm',
                    lambda c: torch.nn.InstanceNorm2d(
                        c, eps=1e-5, affine=True, track_running_stats=False).double(),
                    1, shape)
    ok3, d3 = _case('w_LN = 1 → LayerNorm',
                    lambda c: torch.nn.LayerNorm([c, shape[2], shape[3]], eps=1e-5).double(),
                    2, shape)

    # 顺带把三种统计量的形状打印出来（任务书 §2.6 坑 1 要求"先看一眼"）
    layer = SwitchNorm2d(shape[1])
    x = np.random.default_rng(1).standard_normal(shape)
    layer.forward(x)
    print('\n三种统计量的广播形状：')
    for name, stat in zip(('BN (N,H,W)', 'IN (H,W)', 'LN (C,H,W)'), layer.stats[:3]):
        print(f'  {name:12s} -> {stat.shape}')

    print(f'\n退化测试汇总: max|Δ| = {max(d1, d2, d3):.3e}')
    print('VERDICT:', 'PASS - statistics axes, broadcasting, biased variance and eps are all correct'
          if (ok1 and ok2 and ok3) else 'FAIL')
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == '__main__':
    sys.exit(main())
