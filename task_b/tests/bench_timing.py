"""§0.5 计时基线 —— 先量，再设计（在写 Conv2d 之前决定网络能做多大）。

在 task_b/ 下运行：python tests/bench_timing.py
产物：runs/timing/timing.json（原始数字）+ 屏幕上的表（表要进报告）

**为什么这么测，而不是先写个朴素 Conv2d：**
训练路径是 im2col + GEMM，真正决定"训不训得动"的是两类成本：

  ① **算力侧（BLAS）**：把 (N,C,H,W) 展开成 (N·H'·W', C·K²) 后做 GEMM；
     顺带量出"展开"这一步的**拷贝**成本（K² 倍数据膨胀，只受内存带宽限制）
  ② **带宽侧（逐元素/归约）**：GELU 与 SwitchNorm 都是一遍遍读写整张特征图，
     尤其是 SN 的三套统计量（多次全张量归约）—— 任务书说它们才是内存带宽受限的那一类。
     这里用**等尺寸的通用 numpy 操作**量出"一遍读写要多少毫秒"，不涉及任何模块实现细节。

所以本次不实现任何 npnn 模块；§2.5/§2.6 会用真实模块复测这两类数字。

**两个测量要点：**
  · 线程数必须在 **import numpy 之前**用环境变量设定 → 每个线程数单起一个子进程（父进程汇总）
  · 取多次的**最小值**（best-of）：调度抖动只会让耗时变长，最小值最接近真实速度

⚠️ 本文件是基准工具（不是交付实现），只用 numpy。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASKB_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = TASKB_ROOT / 'runs' / 'timing'

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
THREADS = (1, 2, 4, 8, 16)

# 候选网络的单层形状：(标签, N, C_in, H, W, C_out, K, S)
CONV_LAYERS = [
    ('head   3->32   s1', 128, 3, 32, 32, 32, 3, 1),
    ('stage 32->64   s2', 128, 32, 30, 30, 64, 3, 2),
    ('stage 64->128  s2', 128, 64, 14, 14, 128, 3, 2),
    ('deep 128->256  s2', 128, 128, 6, 6, 256, 3, 2),
]

# 逐元素/归约的代价值不值得关心，取决于张量多大 —— 用上面各层的**输出**形状来量
ACT_SHAPES = [(128, 32, 30, 30), (128, 64, 14, 14), (128, 128, 6, 6)]

REPS = 3


def _best(fn, reps=REPS):
    """跑 reps 次取最小值（抖动只会让耗时变长）。"""
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_here():
    """在当前线程设置下测量（由子进程调用，返回 JSON 可序列化的 dict）。"""
    import numpy as np

    rng = np.random.default_rng(0)

    conv = []
    for name, n, c_in, h, w, c_out, k, s in CONV_LAYERS:
        h_out, w_out = (h - k) // s + 1, (w - k) // s + 1
        m, kk = n * h_out * w_out, c_in * k * k
        x = rng.standard_normal((m, kk), dtype=np.float32)
        dst = np.empty_like(x)
        wmat = rng.standard_normal((kk, c_out), dtype=np.float32)
        t_copy = _best(lambda: np.copyto(dst, x))       # 展开拷贝的**下界**（连续→连续）
        t_gemm = _best(lambda: x @ wmat)
        conv.append(dict(name=name, m=m, kk=kk, c_out=c_out, col_mb=m * kk * 4 / 1e6,
                         copy_ms=t_copy * 1e3, gemm_ms=t_gemm * 1e3,
                         gflops=2 * m * kk * c_out / t_gemm / 1e9))

    act = []
    for shape in ACT_SHAPES:
        x = rng.standard_normal(shape, dtype=np.float32)
        y = np.empty_like(x)
        act.append(dict(shape=list(shape), elems=int(x.size),
                        mb_per_pass=x.size * 4 * 2 / 1e6,           # 读 + 写
                        pass_ms=_best(lambda: np.multiply(x, 1.000001, out=y)) * 1e3,
                        reduce_ms=_best(lambda: x.mean(axis=(0, 2, 3))) * 1e3))
    return {'conv': conv, 'act': act}


def run_child(threads):
    env = {**os.environ, **{v: str(threads) for v in THREAD_VARS}}
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'],
                          env=env, capture_output=True, text=True, encoding='utf-8')
    if proc.returncode != 0:
        print(f'threads={threads}: child failed\n{proc.stderr[-400:]}')
        return None
    return json.loads(proc.stdout.strip().splitlines()[-1])


# 场景假设（都写进报告；§2.5/§2.6 会用真实模块复测）
SN_PASSES = 25      # 每层 SN ≈ 三套统计量的归约 + 归一化/仿射的正反向，粗估 ~25 遍全张量读写
GELU_PASSES = 4     # 前向 2 遍 + 反向 2 遍


def main():
    if '--child' in sys.argv:
        print(json.dumps(measure_here()))
        return 0

    results = {}
    for threads in THREADS:
        rows = run_child(threads)
        if rows:
            results[threads] = rows
            print(f'threads={threads:<3d} done')

    print('\n=== 1) matmul throughput vs threads (GFLOP/s) ===')
    print(f'{"layer":20s}' + ''.join(f'{t:>8d}T' for t in results))
    for i, layer in enumerate(CONV_LAYERS):
        print(f'{layer[0]:20s}' + ''.join(f'{rows["conv"][i]["gflops"]:>9.2f}'
                                          for rows in results.values()))

    pick = 4 if 4 in results else next(iter(results))
    rows = results[pick]
    print(f'\n=== 2) per-layer conv cost @ {pick} threads (ms/batch) ===')
    print(f'{"layer":20s}{"col_MB":>8s}{"copy":>8s}{"gemm":>8s}{"fwd":>8s}{"bwd~2x":>8s}')
    for r in rows['conv']:
        fwd = r['copy_ms'] + r['gemm_ms']
        print(f'{r["name"]:20s}{r["col_mb"]:>8.1f}{r["copy_ms"]:>8.1f}'
              f'{r["gemm_ms"]:>8.1f}{fwd:>8.1f}{fwd * 2:>8.1f}')

    print(f'\n=== 3) memory-bound proxy @ {pick} threads（一遍全张量读写 / 一次逐通道归约）===')
    print(f'{"activation shape":22s}{"elems":>10s}{"MB/pass":>9s}{"pass":>8s}{"reduce":>8s}')
    for a in rows['act']:
        print(f'{str(tuple(a["shape"])):22s}{a["elems"]:>10d}{a["mb_per_pass"]:>9.1f}'
              f'{a["pass_ms"]:>8.2f}{a["reduce_ms"]:>8.2f}')

    print('\n=== 4) epoch extrapolation（351 个 train batch，秒/epoch）===')
    print(f'{"scenario":34s}' + ''.join(f'{t:>9d}T' for t in results))
    scenarios = {}
    for threads, rows in results.items():
        conv3x = sum(r['copy_ms'] + r['gemm_ms'] for r in rows['conv']) * 3      # fwd + bwd(~2x)
        gelu = sum((a['pass_ms'] * 2 + a['reduce_ms']) * GELU_PASSES for a in rows['act'])
        sn = sum(a['pass_ms'] * SN_PASSES for a in rows['act'])
        scenarios[threads] = {'conv': conv3x, 'conv+gelu': conv3x + gelu,
                              'conv+gelu+SN': conv3x + gelu + sn}
    for key in ('conv', 'conv+gelu', 'conv+gelu+SN'):
        line = f'{key:34s}'
        for threads in results:
            line += f'{scenarios[threads][key] * 351 / 1000:>9.1f}s'
        print(line)
    print(f'\n（SN 按每层 {SN_PASSES} 遍、GELU 按每层 {GELU_PASSES} 遍粗估；真实模块的数字在 §2.5/§2.6 复测）')
    for threads in results:
        print(f'  threads={threads:<3d}  conv+gelu+SN 跑 30 epoch ≈ '
              f'{scenarios[threads]["conv+gelu+SN"] * 351 * 30 / 1000 / 60:6.1f} min')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'timing.json').write_text(json.dumps({
        'threads': {str(k): v for k, v in results.items()},
        'conv_layers': CONV_LAYERS, 'act_shapes': ACT_SHAPES,
        'assumptions': {'sn_passes': SN_PASSES, 'gelu_passes': GELU_PASSES, 'batches_per_epoch': 351},
        'epoch_seconds': {str(k): v for k, v in scenarios.items()},
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nsaved -> {OUT_DIR / "timing.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
