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

import json
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


# ---------------------------------------------------------------- Adam
def check_adam():
    """Adam：100 步轨迹对拍 + 手算前 3 步 + None 梯度 + 存档续跑。"""
    import torch
    from npnn.optim import Adam

    rng = np.random.default_rng(0)
    lr = 1e-3

    # --- 1) 逐步轨迹对拍（任务书说的"最强证据"）：同初值、同人工梯度，100 步
    w_np = rng.standard_normal((3, 4))
    w_t = torch.tensor(w_np.copy(), requires_grad=True)
    opt_np, opt_t = Adam([w_np], lr=lr), torch.optim.Adam([w_t], lr=lr)
    deltas = []
    for _ in range(100):
        g = rng.standard_normal(w_np.shape)
        opt_np.step([g])
        w_t.grad = torch.from_numpy(g.copy())
        opt_t.step()
        deltas.append(float(np.abs(w_np - w_t.detach().numpy()).max()))

    # --- 2) 手算前 3 步（纯 Python 浮点，独立于 numpy 的向量化路径）
    def adam_scalar(grads, theta=1.0, lr=0.01):
        m = v = 0.0
        outs = []
        for t, g in enumerate(grads, start=1):
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta -= lr * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.999 ** t)) ** 0.5 + 1e-8)
            outs.append(theta)
        return outs

    hand = adam_scalar([0.1, 0.2, -0.3])
    th = np.array([1.0])
    opt_h = Adam([th], lr=0.01)
    got = []
    for g in (0.1, 0.2, -0.3):
        opt_h.step([np.array([g])])
        got.append(float(th[0]))
    hand_err = max(abs(a - b) for a, b in zip(got, hand))

    # --- 3) None 梯度（模拟 bias=False）：跳过、不崩、其余参数照常更新
    w1, w2 = np.array([1.0, 2.0]), np.array([3.0])
    opt_n = Adam([w1, w2], lr=0.1)
    opt_n.step([np.ones(2), None])
    none_ok = (np.allclose(w1, [0.9, 1.9], atol=1e-6) and np.array_equal(w2, [3.0]))

    # --- 4) 存档续跑：连续 100 步 vs 50 步存档 → 载入新优化器 → 再 50 步
    grads = rng.standard_normal((100, 2))
    wa = np.array([1.0, -2.0]); oa = Adam([wa], lr=0.01)
    for g in grads:
        oa.step([g.copy()])
    wb = np.array([1.0, -2.0]); ob = Adam([wb], lr=0.01)
    for g in grads[:50]:
        ob.step([g.copy()])
    state = ob.state_dict()
    ob2 = Adam([wb], lr=0.01)              # 模拟"新进程"
    ob2.load_state_dict(state)
    for g in grads[50:]:
        ob2.step([g.copy()])
    ckpt_err = float(np.abs(wa - wb).max())

    # 轨迹数据落盘（报告图用）
    out_dir = TASKB_ROOT / 'runs' / 'adam'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'trajectory.json').write_text(
        json.dumps({'deltas': deltas, 'lr': lr}), encoding='utf-8')

    return {'traj': max(deltas), 'hand': hand_err, 'none_ok': bool(none_ok),
            'ckpt': ckpt_err,
            'ok': max(deltas) < 1e-7 and hand_err < 1e-12 and none_ok and ckpt_err == 0.0}


# ---------------------------------------------------------------- Conv2d
def check_conv2d():
    """Conv2d：多形状前/反向对拍 + 数值梯度 + im2col vs 朴素 + 耗时对比。"""
    import time

    import torch
    from npnn import Conv2d

    rng = np.random.default_rng(0)
    cases = [
        # name, (N,Cin,H,W), Cout, K, S, bias, padding
        ('K3S1 (overlap!)', (2, 3, 8, 8),  4, 3, 1, True,  0),   # 重叠最严重：col2im 的照妖镜
        ('K3S2',            (2, 3, 9, 9),  4, 3, 2, True,  0),
        ('K1 pointwise',    (2, 3, 6, 6),  5, 1, 1, True,  0),
        ('K5S1',            (2, 2, 8, 8),  3, 5, 1, True,  0),
        ('H % S != 0',      (2, 2, 8, 8),  3, 3, 2, True,  0),   # (8-3)/2 不整除
        ('stride >= K',     (2, 2, 9, 9),  3, 2, 3, True,  0),   # 窗口不重叠
        ('bias=False',      (2, 2, 6, 6),  3, 3, 1, False, 0),
        ('padding=1 (ext)', (2, 2, 7, 7),  3, 3, 1, True,  1),
    ]
    rows = []
    for name, shape, cout, k, s, bias, pad in cases:
        layer = Conv2d(shape[1], cout, k, stride=s, bias=bias, padding=pad)
        t_layer = torch.nn.Conv2d(shape[1], cout, k, stride=s, bias=bias,
                                  padding=pad).double()
        torch_copy(layer, t_layer)

        x = rng.standard_normal(shape)
        y = layer.forward(x)
        x_t = torch.tensor(x, requires_grad=True)
        y_t = t_layer(x_t)
        fwd = float(np.abs(y - y_t.detach().numpy()).max())

        dout = rng.standard_normal(y.shape)
        dx = layer.backward(dout)
        w_fast = layer.grads['weight'].copy()
        b_fast = layer.grads['bias'].copy() if bias else None
        y_t.backward(torch.from_numpy(dout))
        rels = {'dx': rel_err(dx, x_t.grad.numpy()),
                'dW': rel_err(w_fast, t_layer.weight.grad.numpy())}
        if bias:
            rels['db'] = rel_err(b_fast, t_layer.bias.grad.numpy())

        # im2col vs 朴素版：前向逐位、反向一致（两版必须给出同一个卷积）
        y_naive = layer.forward(x, naive=True)
        dx_naive = layer.backward(dout)
        naive_fwd = float(np.abs(y_naive - y).max())
        naive_dx = rel_err(dx_naive, dx)
        naive_dw = rel_err(layer.grads['weight'], w_fast)

        ok = (fwd < FWD_TOL and all(v < REL_TOL for v in rels.values())
              and naive_fwd < 1e-12 and naive_dx < 1e-12 and naive_dw < 1e-12)
        rows.append(dict(name=name, fwd=fwd, rels=rels, num=None,
                         gmax=float(np.abs(w_fast).max()), ok=ok))

    # --- 数值梯度：小规模、对 weight+bias 全量做（独立于 torch 的第三条路径）
    lc = Conv2d(1, 2, 2, stride=1, bias=True)
    xc = rng.standard_normal((2, 1, 5, 5))
    yc = lc.forward(xc)
    phi, cot = scalarize(lambda ps: _conv_forward(lc, ps, xc), yc.shape, rng)
    lc.forward(xc)
    lc.backward(cot)
    numres = check(phi, [lc.weight, lc.bias],
                   [lc.grads['weight'].copy(), lc.grads['bias'].copy()])

    # --- 耗时对比：naive vs im2col（报告表用）
    lt = Conv2d(3, 8, 3, stride=1)
    xt = rng.standard_normal((8, 3, 16, 16))

    def _best(fn, reps=3):
        best = float('inf')
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_fast = _best(lambda: lt.forward(xt))
    t_naive = _best(lambda: lt.forward(xt, naive=True))

    return rows, {'num': numres['max_rel_err'], 'num_ok': numres['verdict'] == 'PASS',
                  't_fast': t_fast * 1e3, 't_naive': t_naive * 1e3,
                  'shape': '(8,3,16,16)->8ch K3S1'}


def _conv_forward(layer, ps, x):
    layer.weight = ps[0]
    layer.bias = ps[1]
    return layer.forward(x)


# ---------------------------------------------------------------- SwitchNorm2d
def _sn_torch_reference(x, gamma, beta, lam, eps=1e-5):
    """用 torch 张量运算**独立**实现同一个公式（第二条实现路径，不是照抄 numpy 代码）。"""
    import torch
    w = torch.softmax(lam, dim=0)                                 # (3,C)
    wb, wi, wl = (w[k].reshape(1, -1, 1, 1) for k in range(3))
    mu = (wb * x.mean(dim=(0, 2, 3), keepdim=True)
          + wi * x.mean(dim=(2, 3), keepdim=True)
          + wl * x.mean(dim=(1, 2, 3), keepdim=True))
    var = (wb * x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
           + wi * x.var(dim=(2, 3), keepdim=True, unbiased=False)
           + wl * x.var(dim=(1, 2, 3), keepdim=True, unbiased=False))
    return (gamma.reshape(1, -1, 1, 1) * (x - mu) / torch.sqrt(var + eps)
            + beta.reshape(1, -1, 1, 1))


def _sn_forward(layer, ps, x):
    layer.gamma, layer.beta, layer.lam = ps
    return layer.forward(x)


def check_sn():
    """SwitchNorm2d：与独立 torch 参考对比 + 数值梯度（x/γ/β/λ 四项）。"""
    import torch
    from npnn import SwitchNorm2d

    rng = np.random.default_rng(0)
    cases = [
        ('(2,3,5,6) equal w',  (2, 3, 5, 6), False),
        ('(2,3,5,6) random λ', (2, 3, 5, 6), True),
        ('(4,2,3,3) random λ', (4, 2, 3, 3), True),
    ]
    rows = []
    for name, shape, random_lam in cases:
        c = shape[1]
        layer = SwitchNorm2d(c)
        if random_lam:
            layer.lam = rng.standard_normal((3, c)) * 0.5
            layer.gamma = rng.uniform(0.5, 1.5, c)
            layer.beta = rng.standard_normal(c) * 0.1

        x = rng.standard_normal(shape)
        y = layer.forward(x)

        x_t = torch.tensor(x, requires_grad=True)
        g_t = torch.tensor(layer.gamma, requires_grad=True)
        b_t = torch.tensor(layer.beta, requires_grad=True)
        l_t = torch.tensor(layer.lam, requires_grad=True)
        y_t = _sn_torch_reference(x_t, g_t, b_t, l_t)
        fwd = float(np.abs(y - y_t.detach().numpy()).max())

        dout = rng.standard_normal(shape)
        dx = layer.backward(dout)
        y_t.backward(torch.from_numpy(dout))
        rels = {'dx': rel_err(dx, x_t.grad.numpy()),
                'dgamma': rel_err(layer.grads['gamma'], g_t.grad.numpy()),
                'dbeta': rel_err(layer.grads['beta'], b_t.grad.numpy()),
                'dlam': rel_err(layer.grads['lam'], l_t.grad.numpy())}

        ok = fwd < FWD_TOL and all(v < REL_TOL for v in rels.values())
        rows.append(dict(name=name, fwd=fwd, rels=rels, num=None,
                         gmax=float(np.abs(layer.grads['lam']).max()), ok=ok))

    # --- 数值梯度：小规模，对 γ / β / λ 三项全量做（独立于 torch 的第三条路径）
    lc = SwitchNorm2d(2)
    lc.lam = rng.standard_normal((3, 2)) * 0.5
    lc.gamma = rng.uniform(0.5, 1.5, 2)
    lc.beta = rng.standard_normal(2) * 0.1
    xc = rng.standard_normal((2, 2, 4, 4))
    yc = lc.forward(xc)
    phi, cot = scalarize(lambda ps: _sn_forward(lc, ps, xc), yc.shape, rng)
    lc.forward(xc)
    lc.backward(cot)
    numres = check(phi, [lc.gamma, lc.beta, lc.lam],
                   [lc.grads['gamma'].copy(), lc.grads['beta'].copy(), lc.grads['lam'].copy()])

    return rows, {'num': numres['max_rel_err'], 'num_ok': numres['verdict'] == 'PASS'}


# ---------------------------------------------------------------- 报告
def print_table(rows, title):
    keys = []                                   # 收集本表出现过的所有梯度名，保持顺序
    for r in rows:
        for k in r['rels']:
            if k not in keys:
                keys.append(k)
    print(f'=== {title} ===')
    print(f'{"case":24s}{"fwd max|d|":>12s}' + ''.join(f'{k + " rel":>11s}' for k in keys)
          + f'{"num rel":>11s}{"|g|max":>10s}  verdict')
    for r in rows:
        cells = ''.join(f'{r["rels"][k]:>11.2e}' if k in r['rels'] else f'{"-":>11s}' for k in keys)
        num = f'{r["num"]:>11.2e}' if r['num'] is not None else f'{"-":>11s}'
        print(f'{r["name"]:24s}{r["fwd"]:>12.2e}' + cells
              + f'{num}{r["gmax"]:>10.2e}  {"PASS" if r["ok"] else "FAIL"}')
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

    rows, cv = check_conv2d()
    print_table(rows, 'Conv2d vs torch.nn.Conv2d (float64, CPU)')
    print(f'[numeric grad] weight+bias on a small case: rel = {cv["num"]:.2e} '
          f'({"PASS" if cv["num_ok"] else "FAIL"})')
    print(f'[timing] {cv["shape"]}: naive {cv["t_naive"]:.1f} ms vs im2col {cv["t_fast"]:.1f} ms '
          f'-> {cv["t_naive"] / cv["t_fast"]:.1f}x')
    results += rows

    rows, sn = check_sn()
    print_table(rows, 'SwitchNorm2d vs independent torch reference (float64)')
    print(f'[numeric grad] x/gamma/beta/lam on a small case: rel = {sn["num"]:.2e} '
          f'({"PASS" if sn["num_ok"] else "FAIL"})')
    results += rows

    ad = check_adam()
    print('=== Adam vs torch.optim.Adam (float64) ===')
    print(f'[1] 100-step trajectory   max|dtheta| = {ad["traj"]:.2e}   (target < 1e-7)')
    print(f'[2] hand-computed 3 steps max|d|      = {ad["hand"]:.2e}')
    print(f'[3] None gradient (bias=False)        : '
          f'{"skipped ok, no crash" if ad["none_ok"] else "BROKEN"}')
    print(f'[4] checkpoint round-trip max|d|      = {ad["ckpt"]:.2e}   (must be exactly 0)')
    print('    trajectory data -> runs/adam/trajectory.json')

    ok = all(r['ok'] for r in results) and ad['ok'] and cv['num_ok'] and sn['num_ok']
    print('\nVERDICT:', 'PASS' if ok else 'FAIL',
          f'(fwd < {FWD_TOL:.0e}, rel < {REL_TOL:.0e})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
