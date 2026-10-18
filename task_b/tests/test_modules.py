"""逐模块 vs PyTorch 对拍（第 2 阶段）—— 验证工具，可以 import torch。

跑法（在 task_b/ 下）：
    python tests/test_modules.py

每个模块都要过 §1.2 的三关：
  1. **形状守恒**：dX / dW / db 的形状必须与对应数组一致（先查形状，再看数值）
  2. **量级检查**：表里给出 |g|max；梯度全 0 几乎总是 bug
  3. **两条独立路径**：数值梯度（tests/grad_check.py）+ torch 对拍（本文件）

判据：前向 max|Δ| < 1e-9（float64）；反向 / 数值梯度 rel_err < 1e-6

后续模块（CE / Adam / Conv2d / SN）往本文件里加各自的 check_xxx() 即可。
"""

import sys
from pathlib import Path

import numpy as np

TASKB_ROOT = Path(__file__).resolve().parents[1]
if str(TASKB_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKB_ROOT))

from tests.grad_check import check, scalarize      # noqa: E402

FWD_TOL = 1e-9          # 前向 max|Δ|（float64）
REL_TOL = 1e-6          # 反向 / 数值梯度的相对误差


def rel_err(a, b):
    """§1.1 的判据公式，返回 max。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float((np.abs(a - b) / np.maximum(1.0, np.abs(a) + np.abs(b))).max())


def torch_copy(np_layer, t_layer):
    """把 numpy 侧参数拷进 torch 模型 —— 对拍的前提是两边同一份权重。"""
    import torch
    with torch.no_grad():
        t_layer.weight.copy_(torch.from_numpy(np.asarray(np_layer.weight)))
        if np_layer.bias is not None:
            t_layer.bias.copy_(torch.from_numpy(np.asarray(np_layer.bias)))
    return t_layer


def _set_params(layer, ps, has_bias):
    layer.weight = ps[0]
    if has_bias:
        layer.bias = ps[1]


# ---------------------------------------------------------------- Linear
def check_linear():
    """Linear：前向 / 反向 / 3 维输入 / bias=False / 数值梯度。"""
    import torch
    from npnn import Linear

    cases = [
        ('(2,8) bias',          (2, 8),         8, 4, True),
        ('(2,3,4) 3-D bias',    (2, 3, 4),      4, 5, True),     # 前导维度：头号坑
        ('(6,5) no-bias',       (6, 5),         5, 3, False),    # 参数表里真的没有 bias
        ('(2,2,3,4) 4-D bias',  (2, 2, 3, 4),   4, 2, True),
    ]
    rng = np.random.default_rng(0)
    rows = []
    for name, shape, in_f, out_f, has_bias in cases:
        layer = Linear(in_f, out_f, bias=has_bias)
        t_layer = torch_copy(layer, torch.nn.Linear(in_f, out_f, bias=has_bias).double())
        params = [layer.weight] + ([layer.bias] if has_bias else [])

        x = rng.standard_normal(shape)
        y = layer.forward(x)
        x_t = torch.tensor(x, requires_grad=True)
        y_t = t_layer(x_t)
        fwd = float(np.abs(y - y_t.detach().numpy()).max())

        dout = rng.standard_normal(y.shape)
        dx = layer.backward(dout)
        y_t.backward(torch.from_numpy(dout))
        rels = {'dx': rel_err(dx, x_t.grad.numpy()),
                'dW': rel_err(layer.grads['weight'], t_layer.weight.grad.numpy())}
        if has_bias:
            rels['db'] = rel_err(layer.grads['bias'], t_layer.bias.grad.numpy())

        def f(ps, layer=layer, x=x, has_bias=has_bias):
            _set_params(layer, ps, has_bias)
            return layer.forward(x)

        phi, cot = scalarize(f, y.shape, rng)
        _set_params(layer, params, has_bias)
        layer.forward(x)
        layer.backward(cot)
        analytic = [layer.grads['weight']] + ([layer.grads['bias']] if has_bias else [])
        num = check(phi, params, analytic)

        ok = fwd < FWD_TOL and all(v < REL_TOL for v in rels.values()) and num['verdict'] == 'PASS'
        rows.append(dict(name=name, fwd=fwd, rels=rels, num=num['max_rel_err'],
                         gmax=num['per_param'][0]['grad_absmax'], ok=ok))
    return rows


# ---------------------------------------------------------------- GELU
def check_gelu():
    """GELU：前向/反向对拍 + 数值梯度 + 与 erf 精确版的偏差（报告用）。

    ⚠️ torch 默认 `nn.GELU()` 是 **erf 版**；这里必须用 `approximate='tanh'`，
       否则你会看到 ~5e-4 的差，以为自己的实现错了。
    """
    import torch
    from npnn import GELU

    rng = np.random.default_rng(0)
    cases = [
        ('(2,3,4) uniform(-4,4)', rng.uniform(-4.0, 4.0, (2, 3, 4))),   # 覆盖负半轴（曲率都堆那儿）
        ('(64,) linspace(-6,6)',  np.linspace(-6.0, 6.0, 64)),
        ('(2,8) N(0,3) 大值',     rng.standard_normal((2, 8)) * 3.0),
    ]
    rows = []
    for name, x in cases:
        layer = GELU()
        y = layer.forward(x)

        x_t = torch.tensor(x, requires_grad=True)
        y_t = torch.nn.GELU(approximate='tanh')(x_t)
        fwd = float(np.abs(y - y_t.detach().numpy()).max())

        dout = rng.standard_normal(y.shape)
        dx = layer.backward(dout)
        y_t.backward(torch.from_numpy(dout))
        rels = {'dx': rel_err(dx, x_t.grad.numpy())}

        # 无参数模块：数值梯度直接对**输入**求
        x_copy = x.copy()
        phi, cot = scalarize(lambda ps: GELU().forward(ps[0]), y.shape, rng)
        layer.forward(x_copy)
        dx_cot = layer.backward(cot)
        num = check(phi, [x_copy], [dx_cot])

        ok = fwd < FWD_TOL and all(v < REL_TOL for v in rels.values()) and num['verdict'] == 'PASS'
        rows.append(dict(name=name, fwd=fwd, rels=rels, num=num['max_rel_err'],
                         gmax=num['per_param'][0]['grad_absmax'], ok=ok))

    # 与 erf 精确版的偏差：任务书给的预期是 ≈4.7e-4、出现在 x≈-2.7 附近
    xs = np.linspace(-6.0, 6.0, 12001)
    with torch.no_grad():
        y_erf = torch.nn.functional.gelu(torch.tensor(xs)).numpy()
    diff = np.abs(GELU().forward(xs) - y_erf)
    i = int(diff.argmax())
    erf = {'maxdiff': float(diff[i]), 'x': float(xs[i]), 'reldiff': float(diff[i] / abs(y_erf[i]))}
    return rows, erf


# ---------------------------------------------------------------- CrossEntropyLoss
def check_ce():
    """CE：前向/反向对拍 + 大 logits 数值稳定 + 手算 2x3 + N=1。"""
    import math

    import torch
    from npnn import CrossEntropyLoss

    rng = np.random.default_rng(0)
    cases = [
        ('(4,100) 常规',     rng.standard_normal((4, 100)),             np.array([0, 17, 99, 42])),
        ('(1,5) N=1',        rng.standard_normal((1, 5)),               np.array([3])),
        ('(8,10) 大 logits', rng.standard_normal((8, 10)) * 1000.0,     rng.integers(0, 10, 8)),
    ]
    rows = []
    for name, logits, labels in cases:
        fn = CrossEntropyLoss()
        loss = fn.forward(logits, labels)

        z_t = torch.tensor(logits, requires_grad=True)
        loss_t = torch.nn.CrossEntropyLoss()(z_t, torch.tensor(labels))
        fwd = float(abs(loss - loss_t.detach().numpy()))

        dz = fn.backward()
        loss_t.backward()
        rels = {'dz': rel_err(dz, z_t.grad.numpy())}

        # 无参数模块：数值梯度直接对 logits 求（loss 本身就是标量，不用 scalarize）
        l_copy = logits.copy()
        fn2 = CrossEntropyLoss()
        fn2.forward(l_copy, labels)
        dz_ana = fn2.backward()
        num = check(lambda ps: CrossEntropyLoss().forward(ps[0], labels), [l_copy], [dz_ana])

        ok = (fwd < FWD_TOL and all(v < REL_TOL for v in rels.values())
              and num['verdict'] == 'PASS' and bool(np.isfinite(loss)))
        rows.append(dict(name=name, fwd=fwd, rels=rels, num=num['max_rel_err'],
                         gmax=num['per_param'][0]['grad_absmax'], ok=ok))

    # 坑 1 的反面教材：不减 max 直接 exp（1e3 量级 logits）
    big_logits, big_labels = cases[2][1], cases[2][2]
    with np.errstate(over='ignore'):
        naive = np.exp(big_logits)
    with np.errstate(divide='ignore', invalid='ignore'):
        naive_loss = float(np.mean(np.log(naive.sum(axis=1))
                                   - big_logits[np.arange(len(big_labels)), big_labels]))
    naive_str = 'finite' if np.isfinite(naive_loss) else str(naive_loss)

    # 手算 2x3（用 math 的标量运算，独立于实现里的向量化路径）
    hand_logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    hand_labels = np.array([2, 0])
    e1 = [math.exp(1.0), math.exp(2.0), math.exp(3.0)]
    s1 = sum(e1)
    p1 = [v / s1 for v in e1]
    loss_hand = -(math.log(p1[2]) + math.log(1.0 / 3.0)) / 2.0
    grad_hand = np.array([[(p1[0]) / 2, (p1[1]) / 2, (p1[2] - 1.0) / 2],
                          [(1.0 / 3.0 - 1.0) / 2, (1.0 / 3.0) / 2, (1.0 / 3.0) / 2]])
    h = CrossEntropyLoss()
    loss_impl = float(h.forward(hand_logits, hand_labels))
    grad_impl = h.backward()
    hand = {'loss': loss_impl, 'loss_ref': loss_hand,
            'grad_err': float(np.abs(grad_impl - grad_hand).max())}
    return rows, {'naive': naive_str, 'hand': hand}


# ---------------------------------------------------------------- 报告
def print_table(rows, title):
    keys = []                                   # 收集本表出现过的所有梯度名，保持顺序
    for r in rows:
        for k in r['rels']:
            if k not in keys:
                keys.append(k)
    print(f'=== {title} ===')
    print(f'{"case":24s}{"fwd max|d|":>12s}' + ''.join(f'{k + " rel":>10s}' for k in keys)
          + f'{"num rel":>10s}{"|g|max":>10s}  verdict')
    for r in rows:
        cells = ''.join(f'{r["rels"][k]:>10.2e}' if k in r['rels'] else f'{"-":>10s}' for k in keys)
        print(f'{r["name"]:24s}{r["fwd"]:>12.2e}' + cells
              + f'{r["num"]:>10.2e}{r["gmax"]:>10.2e}  {"PASS" if r["ok"] else "FAIL"}')
    print()


def main():
    results = []

    rows = check_linear()
    print_table(rows, 'Linear vs torch.nn.Linear (float64, CPU)')
    results += rows

    rows, erf = check_gelu()
    print_table(rows, "GELU(tanh) vs torch.nn.GELU(approximate='tanh')")
    print(f'[erf vs tanh]  max|d| = {erf["maxdiff"]:.3e} at x = {erf["x"]:+.2f} '
          f'(rel {erf["reldiff"]:.1%})  -- task book expected ~4.7e-4 near x=-2.7')
    results += rows

    rows, ce = check_ce()
    print_table(rows, 'CrossEntropyLoss vs torch.nn.CrossEntropyLoss (float64, CPU)')
    print(f'[large logits] without max subtraction the same loss is: {ce["naive"]} '
          f'(that is why pitfall #1 exists)')
    print(f'[hand 2x3]     loss = {ce["hand"]["loss"]:.6f} (hand-computed {ce["hand"]["loss_ref"]:.6f}), '
          f'max|d grad| = {ce["hand"]["grad_err"]:.2e}')
    results += rows

    ok = all(r['ok'] for r in results)
    print('\nVERDICT:', 'PASS' if ok else 'FAIL',
          f'(fwd < {FWD_TOL:.0e}, rel < {REL_TOL:.0e})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
