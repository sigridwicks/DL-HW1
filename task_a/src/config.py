import sys, pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_A_ROOT = PROJECT_ROOT / 'task_a'
DATA_DIR = PROJECT_ROOT / 'data'
RUNS_DIR = TASK_A_ROOT / 'runs'
SEED = 42

CIFAR100_MEAN = (0.5071, 0.4866, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)