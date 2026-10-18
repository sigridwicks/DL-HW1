"""CIFAR-100 数据管线（任务书 第 0 阶段 0.4）—— 自己用 pickle 读，不用 torchvision。

⛔ 本文件在 `npnn/` 里，只允许 import numpy（外加本项目的 `config`，它也不 import torch）。
   守卫是 `../tests/test_purity.py`。因此 `config.py` 那条"不许 import torch"的线必须守住：
   `npnn/__init__.py` → 本模块 → `config`，整条链上不能出现 torch。

两个关键决定（任务书 §0.4）：
  · **uint8 常驻内存**（50k×3072 ≈ 153 MB），取 batch 时才转 float32
    （一上来就转 float32 → 立刻 ≈ 614 MB，而且还是没归一化的）
  · 归一化统计量抄自 `config`（与 A 同口径），`(x/255 - mean) / std` 的顺序也与 A 一致

实测事实（2026-10-16 自己验的，不是照抄文档）：
  · Python 3 下键都是 bytes；**train 和 test 的键完全一样**（任务书说"不一样"，实测一样）
  · `data` 是 (N, 3072) uint8，**通道优先** → `reshape(N, 3, 32, 32)` 直接就对。
    验证方式：这样 reshape 后逐通道均值 = [0.5071, 0.4865, 0.4409]，与 `config.CIFAR100_MEAN`
    吻合（若把布局搞成 (N,32,32,3)，这组数会对不上）
  · `fine_labels` 才是 100 类（[0, 99]）；`coarse_labels` 是 20 个超类（[0, 19]）
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

# 广播成 (1, 3, 1, 1) —— 逐通道作用在 (N, 3, H, W) 上
_MEAN = np.asarray(config.CIFAR100_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.asarray(config.CIFAR100_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def _load_split(path):
    """读 train 或 test → (images (N,3,32,32) uint8, labels (N,) int64)。"""
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='bytes')       # py3 的键是 bytes
    images = raw[b'data'].reshape(-1, 3, 32, 32)     # 通道优先：view，不拷贝
    labels = np.asarray(raw[b'fine_labels'], dtype=np.int64)
    return images, labels


def load_label_names(cifar_dir=None):
    """100 个类的名字（画图与错误分析用）。"""
    cifar_dir = config.CIFAR100_DIR if cifar_dir is None else Path(cifar_dir)
    with open(Path(cifar_dir) / 'meta', 'rb') as f:
        raw = pickle.load(f, encoding='bytes')
    return [name.decode() for name in raw[b'fine_label_names']]


@dataclass(frozen=True)
class Split:
    """一份数据。`images` 是 (N,3,32,32) 的 **uint8**（没归一化、没转 float32）。"""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)


class CIFAR100:
    """train 45k / val 5k / test 10k（与 A 同口径）。构造时才读盘，import 本模块不碰磁盘。"""

    def __init__(self, cifar_dir=None, val_size=config.VAL_SIZE, seed=config.SEED):
        cifar_dir = Path(config.CIFAR100_DIR if cifar_dir is None else cifar_dir)
        train_images, train_labels = _load_split(cifar_dir / 'train')
        test_images, test_labels = _load_split(cifar_dir / 'test')

        # 固定的 45k/5k 划分：randperm(seed) 的前 val_size 个当 val（与 A 的 build_loaders_val 同口径）。
        # ⚠️ 与 A 是"同一口径、不同具体划分"：A 用的是 torch.randperm，torch 与 numpy 的 RNG
        #    不是同一个序列，两份 val 集合的期望重叠只有 5k×5k/50k = 500 张（10%）。
        #    对 B 的核心结论（numpy vs torch 在同一切分上对比）没有影响。
        perm = np.random.default_rng(seed).permutation(len(train_labels))
        train_images, train_labels = train_images[perm], train_labels[perm]   # 全流程仅此一次拷贝
        # 重排之后 train / val 都是**切片视图**，不再额外占内存
        self.val = Split(train_images[:val_size], train_labels[:val_size])
        self.train = Split(train_images[val_size:], train_labels[val_size:])
        self.test = Split(test_images, test_labels)


def normalize(images):
    """uint8 (N,3,32,32) → float32 归一化：逐通道 `(x/255 - mean) / std`。

    只在取 batch 时调用 —— 内存里永远留着 uint8。三步都是**就地**运算
    （x 是 astype 出来的新数组，改它安全），省掉两次全量临时分配。
    """
    x = images.astype(np.float32)
    x /= 255.0
    x -= _MEAN
    x /= _STD
    return x


def unnormalize(x):
    """归一化后的 float32 → [0,1]，供画图。形状仍是 (N,3,32,32)，要 imshow 需自己 transpose。"""
    return np.clip(x * _STD + _MEAN, 0.0, 1.0)


def batches(split, batch_size=128, *, shuffle=True, rng=None, drop_last=True):
    """按批产出 `(X_normalized_float32, y)`。

    随机性由调用方控制：**传一个 rng（np.random.Generator）并在训练循环外创建它**，
    这样整体可复现、每个 epoch 的顺序又不同（单变量实验的前提）。不传就每次新开一个。
    """
    rng = np.random.default_rng() if rng is None else rng
    n = len(split)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        yield normalize(split.images[idx]), split.labels[idx]
