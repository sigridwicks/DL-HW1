"""npnn —— 纯 NumPy 实现的神经网络（Task B 交付物，评分项 1）。

⛔ 本包只允许 import numpy（任务书 铁律 2）：不许 torch / tensorflow / jax / autograd /
   scipy / pandas，连 matplotlib 也不行。守卫是 `../tests/test_purity.py`。

⚠️ 由此推出一条约束：本包 import 的外部模块（目前只有项目根的 `config`）也不许 import torch ——
   `config.py` 顶部已经写明这条线。

目前已完成：`data`（CIFAR-100 管线）、`module`（基类/Sequential）、
`linear`（1b）、`activations`（1a GELU）、`losses`（1c CE）。
后面每实现一个模块，在这里补一行 re-export。
"""

from .activations import GELU
from .conv import Conv2d, conv_output_size, conv_param_count
from .data import CIFAR100, Split, batches, load_label_names, normalize, unnormalize
from .linear import Linear
from .losses import CrossEntropyLoss
from .module import Module, Sequential
from .norm import SwitchNorm2d
from .optim import Adam

__all__ = ['CIFAR100', 'Split', 'batches', 'load_label_names', 'normalize', 'unnormalize',
           'Module', 'Sequential', 'Linear', 'GELU', 'CrossEntropyLoss', 'Adam',
           'Conv2d', 'conv_output_size', 'conv_param_count', 'SwitchNorm2d']
