"""SwitchNorm2d（评分项 1f，10 分）—— Switchable Normalization。

论文：arXiv:1806.10779（ICLR 2019 短版）/ 题目指定 arXiv:1907.10473（TPAMI 版）。

三种统计量，在同一个特征图上同时算，再用可学习的软权重混合：

    BN：在 (N,H,W) 上求均值 → 广播形状 (1,C,1,1)      —— 跨样本、每通道
    IN：在 (H,W)   上求均值 → 广播形状 (N,C,1,1)      —— 每样本、每通道
    LN：在 (C,H,W) 上求均值 → 广播形状 (N,1,1,1)      —— 每样本、跨通道

    w = softmax(λ)                      λ 形状 (3,C)，每个通道一组
    μ̂ = Σ_k w_k μ_k ,   σ̂² = Σ_k w_k σ_k²
    y = γ·(x − μ̂)/√(σ̂² + ε) + β

三个约定（任务书 §2.6 的坑）：
  · **有偏方差**（÷元素个数）：np.var 的 ddof=0 就是它，与 BN 训练模式一致
  · **ε 加在方差上、根号内**，默认 1e-5（与 nn.BatchNorm2d 的默认值一致，便于退化测试对拍）
  · 本作业每层只有一个 op → λ 的通用形式（ops 维度）退化成 (3,C)，即"每通道一组窗口权重"

推理统计量（任务书给了三个选项，这里选**维护 running 版本**）：
  训练时用当前 batch 的混合统计量，同时以动量 0.1 更新 running_mean/running_var；
  推理时直接用 running 版本。逐样本的 IN/LN 统计量在推理时本来就不可得，
  所以把**混合后的** μ̂、σ̂² 先对 batch 求平均再累积 —— 这是 BN 式做法在 SN 上的自然推广。

⛔ 只允许 import numpy。
"""

import numpy as np

from .module import Module


def _softmax(a, axis=0):
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class SwitchNorm2d(Module):
    PARAM_NAMES = ('gamma', 'beta', 'lam')

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.C = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)              # 初始恒等仿射
        self.beta = np.zeros(num_features)
        self.lam = np.zeros((3, num_features))          # 等权初始化 → w = (1/3, 1/3, 1/3)
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

    # ---------------------------------------------------------------- 统计量
    @staticmethod
    def _stats(x):
        """三种统计量（均值 + **有偏**方差），形状分别 (1,C,1,1) / (N,C,1,1) / (N,1,1,1)。"""
        return (x.mean(axis=(0, 2, 3), keepdims=True),
                x.mean(axis=(2, 3), keepdims=True),
                x.mean(axis=(1, 2, 3), keepdims=True),
                x.var(axis=(0, 2, 3), keepdims=True),        # ddof=0 → 有偏
                x.var(axis=(2, 3), keepdims=True),
                x.var(axis=(1, 2, 3), keepdims=True))

    # ---------------------------------------------------------------- 前向
    def forward(self, x):
        n, c, h, w = x.shape
        self.x = x
        self.x_shape = (n, c, h, w)
        self.stats = self._stats(x)
        mu_bn, mu_in, mu_ln, var_bn, var_in, var_ln = self.stats

        self.w = _softmax(self.lam, axis=0)             # (3,C)
        wf = self.w.reshape(3, c, 1, 1)

        if self.training:
            mu = wf[0] * mu_bn + wf[1] * mu_in + wf[2] * mu_ln     # (N,C,1,1) 广播相加
            var = wf[0] * var_bn + wf[1] * var_in + wf[2] * var_ln
            m = self.momentum
            self.running_mean = ((1 - m) * self.running_mean
                                 + m * mu.mean(axis=0, keepdims=True))
            self.running_var = ((1 - m) * self.running_var
                                + m * var.mean(axis=0, keepdims=True))
        else:
            mu, var = self.running_mean, self.running_var

        inv = 1.0 / np.sqrt(var + self.eps)             # ε 在方差上、根号内
        self.inv = inv
        self.z = (x - mu) * inv
        return self.gamma.reshape(1, c, 1, 1) * self.z + self.beta.reshape(1, c, 1, 1)

    # ---------------------------------------------------------------- 反向
    def backward(self, dout):
        """完整解析梯度：x / γ / β / λ 四项。

        推导要点：y 只通过 z = (x−μ̂)/√(σ̂²+ε) 依赖 x、μ̂、σ̂²。
          ① 直接把 z 对 x 求导        → A = dL/dz · inv
          ② 对 μ̂ 求导（μ̂ 广播到 (N,C,H,W)）→ dm = −Σ_{H,W} A
          ③ 对 σ̂² 求导               → dv = −½·inv²·Σ_{H,W}(dL/dz · z)
          ④ 再按 w_k 把 dm/dv 分摊到三条支路，每条支路用**各自的归约集合**回传：
               均值支路：dx += dμ_k / |S_k|
               方差支路：dx += 2(x−μ_k)·dσ²_k / |S_k|
          ⑤ λ：先 dL/dw_k，再穿过 softmax 的雅可比 diag(w) − wwᵀ
        """
        x = self.x
        n, c, h, w = self.x_shape
        mu_bn, mu_in, mu_ln, var_bn, var_in, var_ln = self.stats
        wf = self.w.reshape(3, c, 1, 1)

        # 仿射参数
        self.grads['gamma'] = (dout * self.z).sum(axis=(0, 2, 3))
        self.grads['beta'] = dout.sum(axis=(0, 2, 3))

        g = dout * self.gamma.reshape(1, c, 1, 1)       # dL/dz
        inv = self.inv
        A = g * inv
        dm = -A.sum(axis=(2, 3), keepdims=True)                                   # dL/dμ̂
        dv = -0.5 * inv ** 2 * (g * self.z).sum(axis=(2, 3), keepdims=True)       # dL/dσ̂²

        # 分摊到三条支路（形状各不相同，靠 reduce 对齐）
        dmu_bn = (wf[0] * dm).sum(axis=0, keepdims=True)      # (1,C,1,1)
        dmu_in = wf[1] * dm                                   # (N,C,1,1)
        dmu_ln = (wf[2] * dm).sum(axis=1, keepdims=True)      # (N,1,1,1)
        dvar_bn = (wf[0] * dv).sum(axis=0, keepdims=True)
        dvar_in = wf[1] * dv
        dvar_ln = (wf[2] * dv).sum(axis=1, keepdims=True)

        dx = A
        dx = dx + dmu_bn / (n * h * w)
        dx = dx + dmu_in / (h * w)
        dx = dx + dmu_ln / (c * h * w)
        dx = dx + 2.0 * (x - mu_bn) * dvar_bn / (n * h * w)
        dx = dx + 2.0 * (x - mu_in) * dvar_in / (h * w)
        dx = dx + 2.0 * (x - mu_ln) * dvar_ln / (c * h * w)

        # λ 的梯度：dL/dw_k = Σ (dm·μ_k + dv·σ²_k)，再穿过 softmax
        dw = np.stack([(dm * mu_bn).sum(axis=(0, 2, 3)),
                       (dm * mu_in).sum(axis=(0, 2, 3)),
                       (dm * mu_ln).sum(axis=(0, 2, 3))]) \
            + np.stack([(dv * var_bn).sum(axis=(0, 2, 3)),
                        (dv * var_in).sum(axis=(0, 2, 3)),
                        (dv * var_ln).sum(axis=(0, 2, 3))])            # (3,C)
        self.grads['lam'] = self.w * (dw - (self.w * dw).sum(axis=0, keepdims=True))
        return dx
