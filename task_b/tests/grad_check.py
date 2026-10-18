"""中心差分数值梯度检验器（第 1 阶段 §1.1）—— 全任务最重要的一个工具。

它是后面**每一个模块**的通行证：解析梯度对不对，不能靠"训练能收敛"来判断，
要靠一条与实现无关的独立证据。这个文件提供那条证据。

────────────────────────────────────────────────────────────────────────
用法（三步 —— 注意第 1 步的随机标量化，它最容易写错）

    import numpy as np
    from tests.grad_check import scalarize, check

    x = rng.standard_normal((2, 3, 4, 4))            # float64！
    layer = MyConv2d(...)                            # 你写的模块

    # ① 把「前向返回数组」变成「返回标量」，权重固定住（见下方"为什么"）
    phi, dout = scalarize(lambda ps: layer_forward(ps, x), out_shape, rng)

    # ② 解析梯度：走你实现的反向（dout 就是上面那个固定权重）
    layer.forward(x); layer.backward(dout)
    analytic = [layer.grads['weight'], layer.grads['bias']]

    # ③ 检验
    print(check(phi, [layer.weight, layer.bias], analytic, h=1e-5))

────────────────────────────────────────────────────────────────────────
四条纪律（§1.1，每条都对应真实的坑）

  1. **一律 float64**：float32 只有 ~7 位有效数字，h=1e-5 的差分会被舍入噪声淹没，
     你会看到"梯度怎么全错"然后浪费一天。本模块会直接拒绝非 float64 的输入。
  2. **h=1e-5 是权衡的结果**：太大 → 截断误差 O(h²)；太小 → 舍入误差 O(ε/h)。
     用 h_sweep() 亲手扫一遍 1e-2…1e-8，画出来是 U 形曲线（报告图）。
  3. **输入要随机、要多组**：全 1 / 全 0 / 全同值的输入会掩盖广播错误 ——
     写错的实现对这类输入恰好也正确。
  4. **非标量输出要用随机线性组合标量化**：`sum(out * c)`，其中 c 是**固定的**随机向量。
     用全 1 求和（等价于 sum(out)）会让各位置的误差正负抵消，掩盖索引错误。
     ⚠️ 而且 c 必须在整轮检验里**固定不变** —— 每次扰动都换一组 c 的话，
     差分测到的是噪声的组合，什么也检验不出来。

判据：rel_err < 1e-6 通过 / 1e-6~1e-4 可疑 / >1e-4 一定有 bug

⚠️ 打印与异常消息一律 ASCII：Windows 控制台默认代码页（GBK）编不出 ²、⚠️ 这类字符，
   会直接 UnicodeEncodeError。注释和 docstring 用中文不受影响。
"""

import numpy as np

PASS, SUSPECT, FAIL = 'PASS', 'SUSPECT', 'FAIL'


def random_cotangent(shape, rng=None):
    """非标量输出用的随机线性组合权重（也就是反向传播里那个 `dout`）。"""
    rng = np.random.default_rng() if rng is None else rng
    return rng.standard_normal(shape)


def scalarize(f, out_shape, rng=None):
    """把「返回数组的 f」变成「返回标量的 phi」，权重固定。

    返回 `(phi, cotangent)`：`cotangent` 同时是：
      · 数值侧标量化用的固定权重，以及
      · 解析侧 `backward(cotangent)` 的输入
    两边面对的是**同一个标量函数** —— 这是整个检验成立的前提，也是它俩能对上的原因。
    """
    cotangent = random_cotangent(out_shape, rng)

    def phi(params):
        return float(np.sum(np.asarray(f(params)) * cotangent))

    return phi, cotangent


def _as_scalar(value):
    v = np.asarray(value)
    if v.size != 1:
        raise ValueError(f'f must return a scalar (or size-1 array), got shape={v.shape}; '
                         f'scalarize() the output first if it is non-scalar')
    return float(v)


def numerical_grad(f, params, h=1e-5):
    """中心差分：`(f(theta+h) - f(theta-h)) / 2h`，逐元素。

    ⚠️ 会**就地**扰动 params（每次循环末尾复原）—— 所以：
       · 传进来的数组必须可写、dtype 必须是 float64（本函数会检查）；
       · f 必须真的使用这些数组（而不是某个副本）。
    复杂度是 O(sum|theta|) 次函数求值 —— 这是工具，测试用小块参数就好。
    """
    for p in params:
        if p.dtype != np.float64:
            raise TypeError(f'gradient checks must use float64, got {p.dtype} '
                            f'(float32 hides the h=1e-5 difference in rounding noise)')
        if not p.flags.writeable:
            raise TypeError('parameter arrays must be writable (perturbed in place, then restored)')

    grads = [np.zeros_like(p) for p in params]
    for p, g in zip(params, grads):
        for ix in np.ndindex(p.shape):
            orig = p[ix]
            p[ix] = orig + h
            f_plus = _as_scalar(f(params))
            p[ix] = orig - h
            f_minus = _as_scalar(f(params))
            p[ix] = orig
            g[ix] = (f_plus - f_minus) / (2.0 * h)
    return grads


def relative_error(g_num, g_ana):
    """§1.1 的判据公式：|g_num - g_ana| / max(1, |g_num| + |g_ana|)，返回 (max, mean)。"""
    g_num, g_ana = np.asarray(g_num, dtype=np.float64), np.asarray(g_ana, dtype=np.float64)
    denom = np.maximum(1.0, np.abs(g_num) + np.abs(g_ana))    # 分母的 1 让接近 0 的梯度不会爆炸
    err = np.abs(g_num - g_ana) / denom
    return float(err.max()), float(err.mean())


def verdict_of(max_rel_err, pass_tol=1e-6, suspect_tol=1e-4):
    return PASS if max_rel_err < pass_tol else (SUSPECT if max_rel_err < suspect_tol else FAIL)


def check(f, params, analytic, h=1e-5):
    """完整体检：数值梯度 vs 解析梯度。

    参数
        f        : callable(list[ndarray]) -> 标量（非标量输出请先用 scalarize()）
        params   : list[ndarray float64]，会被就地扰动（用完复原）
        analytic : list[ndarray]，来自你自己实现的反向
    返回 dict：每参数的 max/mean 相对误差 + 梯度量级 + 总判据。
    """
    params = [np.asarray(p) for p in params]              # 只做视图，不拷贝（拷贝会让扰动失效）
    analytic = [np.asarray(g, dtype=np.float64) for g in analytic]
    if len(params) != len(analytic):
        raise ValueError(f'arity mismatch: {len(params)} params vs {len(analytic)} analytic grads')

    numeric = numerical_grad(f, params, h=h)

    per_param = []
    for p, g_num, g_ana in zip(params, numeric, analytic):
        if g_num.shape != g_ana.shape:
            raise ValueError(f'shape not conserved: numeric {g_num.shape} vs analytic {g_ana.shape} '
                             f'(generic check #1: check shapes before numbers)')
        mx, mean = relative_error(g_num, g_ana)
        per_param.append({
            'shape': tuple(p.shape),
            'max_rel_err': mx,
            'mean_rel_err': mean,
            'grad_absmax': float(np.abs(g_ana).max()),      # 量级检查：全 0 几乎总是 bug
            'grad_absmean': float(np.abs(g_ana).mean()),
        })

    worst = max(r['max_rel_err'] for r in per_param)
    return {'h': h, 'per_param': per_param, 'max_rel_err': worst,
            'verdict': verdict_of(worst)}


def h_sweep(f, params, analytic, hs=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)):
    """扫 h，返回 [(h, max_rel_err), ...] —— 画出来就是那条 U 形曲线（报告图）。"""
    return [(h, check(f, params, analytic, h=h)['max_rel_err']) for h in hs]


def format_result(result, tol_pass=1e-6, tol_suspect=1e-4):
    """打成 ASCII 表格（报告要贴的就是这个）。"""
    lines = [f'{"param":>10s}{"shape":>16s}{"max_rel_err":>14s}{"mean_rel_err":>14s}'
             f'{"|g|max":>12s}{"|g|mean":>12s}']
    for r in result['per_param']:
        lines.append(f'{"":>10s}{str(r["shape"]):>16s}{r["max_rel_err"]:>14.3e}'
                     f'{r["mean_rel_err"]:>14.3e}{r["grad_absmax"]:>12.3e}{r["grad_absmean"]:>12.3e}')
    mark = {PASS: 'PASS', SUSPECT: 'SUSPECT (check the boundary/branch)',
            FAIL: 'FAIL (there is a bug)'}[result['verdict']]
    lines.append(f'h={result["h"]:.0e}  verdict: {mark}   '
                 f'(<{tol_pass:.0e} pass / <{tol_suspect:.0e} suspicious / else bug)')
    return '\n'.join(lines)


# ---------------------------------------------------------------- 自检（工具本身也要被验证）
if __name__ == '__main__':
    rng = np.random.default_rng(0)

    print('=== 1) known answer: f(x) = sum(x**3), analytic grad = 3*x**2 ===')
    x = rng.standard_normal((3, 4))
    f_cube = lambda ps: float(np.sum(ps[0] ** 3))
    print(format_result(check(f_cube, [x], [3.0 * x ** 2])))

    print('\n=== 2) negative control: wrong analytic grad (2*x**2), must FAIL ===')
    res_bad = check(f_cube, [x], [2.0 * x ** 2])
    print(format_result(res_bad))
    print('[selftest]', 'PASS - the checker caught the wrong gradient'
          if res_bad['verdict'] == FAIL else 'FAIL - the checker missed it!')

    print('\n=== 3) h sweep (the U-curve data; notebook 10 plots it) ===')
    for h, err in h_sweep(f_cube, [x], [3.0 * x ** 2]):
        print(f'  h={h:.0e}   max_rel_err={err:.3e}')

    print('\n=== 4) float32 is rejected on the spot ===')
    try:
        check(f_cube, [x.astype(np.float32)], [3.0 * x ** 2])
    except TypeError as exc:
        print(f'  rejected: {exc}')
