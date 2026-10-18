"""Task B 的路径 / 数据口径 / 线程预算 —— 第 0 阶段（任务书 §0.1、§0.4、§0.5）。

三个路径从 __file__ 推，**不依赖 cwd**（Jupyter 的 cwd 是「启动目录」，不是 notebook 所在目录）：

    TASKB_ROOT = task_b/
    DATA_DIR   = ../data      ← 只读共享：Task B 绝不往里写任何文件
    RUNS_DIR   = task_b/runs

⛔ 本文件不许 import torch（也不 import numpy）：
   npnn/data.py 会 import 它，而纯度守卫要求 npnn/ 的**整条 import 链**里只有 numpy。
   所以这里只放常量和一个小函数 —— 以后加东西时请守住这条线。
"""

import os
import pathlib
import sys

# ------------------------------------------------------------------ 路径
TASKB_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = TASKB_ROOT.parent / 'data'
RUNS_DIR = TASKB_ROOT / 'runs'

# ------------------------------------------------- 数据口径（与 Task A 一致，抄自 ../src/config.py）
CIFAR100_DIR = DATA_DIR / 'cifar-100-python'
NUM_CLASSES = 100
VAL_SIZE = 5000          # 45k train + 5k val，与 A 的 build_loaders_val 同口径
SEED = 42
CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# ------------------------------------------------------------------ 线程预算
# §0.5 实测（2026-10-16，tests/bench_timing.py，数字在 runs/timing/timing.json）：
#   GEMM 吞吐 1T→4T 约 3.5 倍（91 / 328 / 392 GFLOP/s），4T 之后不再涨；
#   而**内存带宽受限**的部分（GELU/SN 的逐元素与归约）在 8T/16T 反而**变慢**
#   （23.1 → 27.0 / 25.4 s/epoch）。本机 32 逻辑核，但 A 的 dataloader 也在用 CPU。
#   拐点 + 让路给 A → 定在 4。
BLAS_THREADS = 4

_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def limit_threads(n: int = BLAS_THREADS) -> None:
    """把 BLAS 线程数写进 os.environ —— **必须在 import numpy 之前调用**。

    BLAS 在首次 import 时就把线程数锁死了，之后再设环境变量不会生效；
    所以这里做一个显式提醒，而不是让你静默地以为限制生效了。
    """
    if 'numpy' in sys.modules:
        print(f'[warn] numpy is already imported ({sys.modules["numpy"].__version__}) -- '
              f'setting {n} threads now has NO effect; move config.limit_threads() before import numpy')
    for var in _VARS:
        os.environ[var] = str(n)


def summary() -> dict:
    """给 notebook / 报告用的环境留痕（只读常量，不 import 任何计算库）。"""
    return {
        'sys.executable': sys.executable,
        'sys.prefix': sys.prefix,
        'TASKB_ROOT': str(TASKB_ROOT),
        'DATA_DIR': str(DATA_DIR),
        'RUNS_DIR': str(RUNS_DIR),
        'BLAS_THREADS': BLAS_THREADS,
    }


if __name__ == '__main__':
    # 自检：从任何 cwd 运行都应得到同样的三个路径（第 0 阶段验收第 1 条）
    # 打印一律用 ASCII —— Windows 控制台的默认代码页会吃掉中文，留痕不该被编码问题污染
    for key, value in summary().items():
        print(f'{key:16s}: {value}')
    print(f'{"DATA_DIR exists":16s}:', DATA_DIR.is_dir())
    print(f'{"CIFAR files":16s}:', sorted(p.name for p in CIFAR100_DIR.glob('*')))
    print(f'{"RUNS_DIR exists":16s}:', RUNS_DIR.is_dir())
