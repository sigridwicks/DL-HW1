"""Module 基类 / Sequential 容器（任务书 §0.5）—— 框架的地基。

这个文件是**脚手架**（不是评分项）；六个评分模块由你自己手敲，但它们都要按下面的约定来写。

────────────────────────────────────────────────────────────────────────
约定（写模块前先读这 5 行）

    class MyLayer(Module):
        PARAM_NAMES = ('weight', 'bias')        # ① 声明参数名（与 PyTorch 对齐）

        def __init__(self, ..., bias=True):
            super().__init__()
            self.weight = np.zeros(...)         # ② 参数存在**同名属性**上
            self.bias = None if not bias else np.zeros(...)     # ③ None = 参数不存在

        def forward(self, x):   ...  -> y       # ④ 前向
        def backward(self, dout): ... -> dx     # ⑤ 反向**必须返回对输入的梯度**

    · `params` 是**动态生成**的只读视图（PARAM_NAMES 里非 None 的项）——
      所以 `bias=False` 时参数列表里**真的没有** bias（任务书 §2.1 的硬要求），
      而且属性与参数表永远不可能脱节。
      ⚠️ 不要往 `self.params[...]` 里赋值（那是在改一个临时 dict，什么也不会发生）；
         改参数请改属性：`self.weight = ...`。
    · `grads` 是普通 dict，由 `backward()` 填充，键与 `params` 相同。
    · 容器（Sequential）把子层放进 `self.modules`；`train()/eval()` 会递归传播 ——
      SwitchNorm2d 的训练/推理双轨就靠它。

⛔ 只允许 import numpy（铁律 2，守卫是 ../tests/test_purity.py）。
"""

import numpy as np


class Module:
    """所有层的基类。子类至少要实现 forward 和 backward。"""

    PARAM_NAMES = ()          # 子类覆盖：本层参数的名字（与 PyTorch 对齐）

    def __init__(self):
        self.modules = []     # 子模块（容器用；叶子层留空）
        self.grads = {}       # backward 之后填充：name -> ndarray
        self.training = True  # 训练/推理开关

    # ---------------------------------------------------------- 子类要实现
    def forward(self, x):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # 用 *args 而不是 (x)：CrossEntropyLoss.forward(logits, labels) 需要两个参数
        return self.forward(*args, **kwargs)

    # ---------------------------------------------------------- 参数视图
    @property
    def params(self):
        """name -> ndarray（PARAM_NAMES 里非 None 的项）。只读视图，别往里赋值。"""
        return {name: getattr(self, name) for name in self.PARAM_NAMES
                if getattr(self, name) is not None}

    def named_params(self, prefix=''):
        """递归遍历本层与子层的参数（优化器、存档、参数计数都用它）。"""
        for name in self.PARAM_NAMES:
            if getattr(self, name) is not None:
                yield prefix + name, getattr(self, name)
        for i, m in enumerate(self.modules):
            yield from m.named_params(f'{prefix}{i}.')

    def named_grads(self, prefix=''):
        """与 named_params 同顺序的梯度（第一次 backward 之前是 None）。"""
        for name in self.PARAM_NAMES:
            if getattr(self, name) is not None:
                yield prefix + name, self.grads.get(name)
        for i, m in enumerate(self.modules):
            yield from m.named_grads(f'{prefix}{i}.')

    def parameters(self):
        """参数数组列表（**引用** —— 优化器就地更新它们才会生效）。"""
        return [p for _, p in self.named_params()]

    def gradients(self):
        """与 parameters() 一一对应的梯度列表。"""
        return [g for _, g in self.named_grads()]

    # ---------------------------------------------------------- 训练/推理
    def train(self, mode=True):
        self.training = mode
        for m in self.modules:
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ---------------------------------------------------------- 小工具
    def as_dtype(self, dtype):
        """就地把参数换成另一种 dtype —— 梯度检验前 `layer.as_dtype(np.float64)`。

        （训练用 float32、检验用 float64，这条线由调用方控制，模块内部不硬编码 dtype。）
        """
        for name in self.PARAM_NAMES:
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name).astype(dtype))
        return self

    def n_params(self):
        return sum(p.size for _, p in self.named_params())

    def state_dict(self):
        """扁平化的 {name: array} —— 直接喂给 np.savez。"""
        return dict(self.named_params())


class Flatten(Module):
    """(N, ...) → (N, prod(...))。接线件，不是评分模块。"""

    PARAM_NAMES = ()

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class GlobalAvgPool(Module):
    """(N,C,H,W) → (N,C)：对每个通道在 H、W 上取平均。反向把梯度均摊回每个位置。"""

    PARAM_NAMES = ()

    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, dout):
        n, c, h, w = self.shape
        return np.broadcast_to(dout[:, :, None, None] / (h * w), self.shape).copy()


class Sequential(Module):
    """顺序 forward、逆序 backward。刻意做小（任务书 §0.5：30 行级别）。"""

    def __init__(self, *layers):
        super().__init__()
        self.modules = list(layers)

    def forward(self, x):
        for m in self.modules:
            x = m.forward(x)
        return x

    def backward(self, dout):
        for m in reversed(self.modules):
            dout = m.backward(dout)      # 逆序回传，每层返回给上一层的输入梯度
        return dout
