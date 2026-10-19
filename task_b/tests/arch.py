"""端到端网络的结构定义 —— numpy 训练脚本与 torch 参照**共用同一份**。

之前踩过的坑：torch 参照脚本复用了 L2 对拍里的另一份 NET（3→8→16 + Flatten），
结果两边跑的根本不是同一个网络，精度自然对不上。结构只在这里定义一次。
"""

ARCH = [
    ('conv', 3, 32, 3, 1), ('sn', 32), ('gelu',),
    ('conv', 32, 64, 3, 2), ('sn', 64), ('gelu',),
    ('conv', 64, 128, 3, 2), ('sn', 128), ('gelu',),
    ('gap',),
    ('linear', 128, 100),
]


def build_npnn(arch=ARCH):
    from npnn import Conv2d, GELU, Linear, Sequential, SwitchNorm2d
    from npnn.module import GlobalAvgPool

    layers = []
    for spec in arch:
        kind = spec[0]
        if kind == 'conv':
            layers.append(Conv2d(spec[1], spec[2], spec[3], stride=spec[4]))
        elif kind == 'sn':
            layers.append(SwitchNorm2d(spec[1]))
        elif kind == 'gelu':
            layers.append(GELU())
        elif kind == 'gap':
            layers.append(GlobalAvgPool())
        elif kind == 'linear':
            layers.append(Linear(spec[1], spec[2]))
        else:
            raise ValueError(kind)
    return Sequential(*layers)


def build_torch(arch=ARCH, dtype=None):
    import torch
    import torch.nn as nn

    from tests.l2_parity import TorchSN

    layers = []
    for spec in arch:
        kind = spec[0]
        if kind == 'conv':
            layers.append(nn.Conv2d(spec[1], spec[2], spec[3], stride=spec[4]))
        elif kind == 'sn':
            layers.append(TorchSN(spec[1]))
        elif kind == 'gelu':
            layers.append(nn.GELU(approximate='tanh'))
        elif kind == 'gap':
            layers.append(nn.AdaptiveAvgPool2d(1))     # (N,C,1,1)
            layers.append(nn.Flatten())                # (N,C)
        elif kind == 'linear':
            layers.append(nn.Linear(spec[1], spec[2]))
        else:
            raise ValueError(kind)
    return nn.Sequential(*layers).to(dtype or torch.float32)
