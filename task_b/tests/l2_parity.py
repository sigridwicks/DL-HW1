"""L2 整网对拍（§3.2）—— 10 分的核心证据。

协议（任务书 §3.2，逐条照做）：
  · **架构**：`NET` 定义一次，两边用同一份配置生成 —— 不会一边 3 层一边 4 层
  · **权重**：numpy 侧生成 → 按顺序 copy 进 torch（形状断言过）→ 两边同一份参数
  · **数据**：同一批 uint8 图像，两边拿到的是**同一个** float64 归一化 batch
  · **优化器**：我们的 Adam vs torch.optim.Adam，同 lr / betas / eps
  · **设备与精度**：全程 CPU + float64（GPU/cuDNN 的非确定性会掩盖真 bug）
  · **判据**：N 步后 loss 相对差 < 1e-8

⚠️ torch 没有 SwitchNorm2d —— 参照实现用 `TorchSN`（torch 张量 + autograd 写的同一个公式），
   它是**独立的第二实现**，和我们的 numpy 版没有共享一行代码。

跑法（在 task_b/ 下）：python tests/l2_parity.py
产物：runs/l2/parity.json（loss 两条曲线、参数差、分叉点）
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TASKB_ROOT = Path(__file__).resolve().parents[1]
if str(TASKB_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKB_ROOT))

import torch                                                   # noqa: E402
import torch.nn as nn                                         # noqa: E402

from npnn import Adam, CIFAR100, Conv2d, CrossEntropyLoss, GELU   # noqa: E402
from npnn import Linear, Sequential, SwitchNorm2d, normalize   # noqa: E402
from npnn.module import Flatten                               # noqa: E402

BATCH, STEPS, LR = 32, 100, 1e-3

# 架构：定义一次，两边共用 -----------------------------------------------------------------
NET = [
    ('conv', 3, 8, 3, 1),        # (in, out, k, stride)   32x32 -> 30x30
    ('sn', 8),
    ('gelu',),
    ('conv', 8, 16, 3, 2),       #                        30x30 -> 14x14
    ('sn', 16),
    ('gelu',),
    ('flatten',),
    ('linear', 16 * 14 * 14, 100),
]


def build_npnn():
    layers = []
    for spec in NET:
        kind = spec[0]
        if kind == 'conv':
            layers.append(Conv2d(spec[1], spec[2], spec[3], stride=spec[4]))
        elif kind == 'sn':
            layers.append(SwitchNorm2d(spec[1]))
        elif kind == 'gelu':
            layers.append(GELU())
        elif kind == 'flatten':
            layers.append(Flatten())
        elif kind == 'linear':
            layers.append(Linear(spec[1], spec[2]))
    return Sequential(*layers)


class TorchSN(nn.Module):
    """torch 版的 SwitchNorm2d —— 独立实现（autograd 负责反向）。"""

    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))
        self.lam = nn.Parameter(torch.zeros(3, c))
        self.eps = eps

    def forward(self, x):
        w = torch.softmax(self.lam, dim=0)
        wb, wi, wl = (w[k].reshape(1, -1, 1, 1) for k in range(3))
        mu = (wb * x.mean(dim=(0, 2, 3), keepdim=True)
              + wi * x.mean(dim=(2, 3), keepdim=True)
              + wl * x.mean(dim=(1, 2, 3), keepdim=True))
        var = (wb * x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
               + wi * x.var(dim=(2, 3), keepdim=True, unbiased=False)
               + wl * x.var(dim=(1, 2, 3), keepdim=True, unbiased=False))
        z = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma.reshape(1, -1, 1, 1) * z + self.beta.reshape(1, -1, 1, 1)


def build_torch(dtype=torch.float64):
    layers = []
    for spec in NET:
        kind = spec[0]
        if kind == 'conv':
            layers.append(nn.Conv2d(spec[1], spec[2], spec[3], stride=spec[4]))
        elif kind == 'sn':
            layers.append(TorchSN(spec[1]))
        elif kind == 'gelu':
            layers.append(nn.GELU(approximate='tanh'))
        elif kind == 'flatten':
            layers.append(nn.Flatten())
        elif kind == 'linear':
            layers.append(nn.Linear(spec[1], spec[2]))
    return nn.Sequential(*layers).to(dtype)


def copy_params(model, t_model):
    """按**顺序**拷贝（两边的参数顺序由同一份 NET 决定），并断言形状一致。"""
    np_params = model.parameters()
    t_params = list(t_model.parameters())
    if len(np_params) != len(t_params):
        raise RuntimeError(f'param count mismatch: {len(np_params)} vs {len(t_params)}')
    with torch.no_grad():
        for p_np, p_t in zip(np_params, t_params):
            if p_np.shape != tuple(p_t.shape):
                raise RuntimeError(f'shape mismatch: {p_np.shape} vs {tuple(p_t.shape)}')
            p_t.copy_(torch.from_numpy(p_np))


def main():
    rng = np.random.default_rng(0)
    np.random.seed(0)                       # 建模前固定随机种子（初始化用全局 np.random）

    ds = CIFAR100()
    model = build_npnn()
    t_model = build_torch()
    copy_params(model, t_model)

    criterion = CrossEntropyLoss()
    ce_t = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=LR)
    opt_t = torch.optim.Adam(t_model.parameters(), lr=LR)

    loss_np, loss_t, loss_diff, param_diff = [], [], [], []
    first_div = None
    t0 = time.perf_counter()

    for step in range(1, STEPS + 1):
        idx = rng.integers(0, len(ds.train), BATCH)
        xb = normalize(ds.train.images[idx]).astype(np.float64)   # 两边看到同一个 batch
        yb = ds.train.labels[idx]

        # --- numpy 侧
        logits = model.forward(xb)
        l_np = float(criterion.forward(logits, yb))
        model.backward(criterion.backward())
        opt.step(model.gradients())

        # --- torch 侧
        x_t, y_t = torch.from_numpy(xb), torch.from_numpy(yb)
        logits_t = t_model(x_t)
        l_t = ce_t(logits_t, y_t)
        opt_t.zero_grad()
        l_t.backward()
        opt_t.step()

        l_t_val = l_t.detach().item()
        loss_np.append(l_np)
        loss_t.append(l_t_val)
        d = abs(l_np - l_t_val)
        loss_diff.append(d)
        if first_div is None and d > 1e-10:
            first_div = step
        param_diff.append(float(max(np.abs(a - b.detach().numpy()).max()
                                    for a, b in zip(model.parameters(), t_model.parameters()))))

    dt = time.perf_counter() - t0
    rel = max(d / max(1e-12, abs(a)) for d, a in zip(loss_diff, loss_t))

    print(f'steps={STEPS}  batch={BATCH}  lr={LR}  float64  CPU   ({dt:.1f} s)')
    print(f'loss  : first {loss_np[0]:.6f} -> last {loss_np[-1]:.6f}')
    print(f'|Δloss| max = {max(loss_diff):.3e}   (task book: < 1e-8)')
    print(f'|Δparam| max over the whole net = {param_diff[-1]:.3e}')
    print(f'first step with |Δloss| > 1e-10: {first_div if first_div else "none"}')
    print('VERDICT:', 'PASS - identical trajectory' if max(loss_diff) < 1e-8
          else 'FAIL - trajectories diverge')

    # 最终每个参数的差 → 直方图（报告图）；原始数组太大，存直方图就够了
    final_diffs = np.concatenate([np.abs(a - b.detach().numpy()).ravel()
                                  for a, b in zip(model.parameters(), t_model.parameters())])
    counts, edges = np.histogram(final_diffs, bins=60)

    out = TASKB_ROOT / 'runs' / 'l2'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'parity.json').write_text(json.dumps({
        'steps': STEPS, 'batch': BATCH, 'lr': LR, 'net': NET,
        'loss_np': loss_np, 'loss_t': loss_t,
        'loss_diff': loss_diff, 'param_diff': param_diff,
        'max_loss_diff': max(loss_diff), 'max_rel_loss_diff': rel,
        'first_divergence_step': first_div,
        'param_hist': {'counts': counts.tolist(), 'edges': edges.tolist()},
        'n_params': int(sum(p.size for p in model.parameters())),
    }, indent=2), encoding='utf-8')
    print(f'saved -> {out / "parity.json"}')
    return 0 if max(loss_diff) < 1e-8 else 1


if __name__ == '__main__':
    sys.exit(main())
