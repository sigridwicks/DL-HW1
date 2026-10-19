"""Conv2d（评分项 1e，10 分）—— 与 torch.nn.Conv2d(padding=0) 对齐。

接口（任务书 §2.5）：
    Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=True, padding=0)
    X: (N,Cin,H,W)      W: (Cout,Cin,K,K)      b: (Cout,)
    H_out = floor((H + 2P - K) / S) + 1

⚠️ 题目给的 config 里**没有 padding** —— 所以参照实现就是 nn.Conv2d(..., padding=0)。
   这里把 padding 做成**带默认值的可选扩展**（默认 0，行为与题目一致），判断写进报告。

两版实现（任务书要求，不是二选一）：
  · **朴素版**：循环输出位置 + 每个窗口 tensordot —— 慢，但几乎不可能写错 → oracle
  · **im2col + GEMM**：训练路径 —— 展开成一次大矩阵乘，吃满 BLAS
  `forward(x, naive=True)` 可切到朴素版（对拍、耗时对比用）。

im2col 的反向（col2im）是本题最难的 20 行：窗口是**重叠**的，同一个输入像素会出现在
多个窗口里，必须把所有窗口的贡献**累加**回去。这里用"按 (i,j) 偏移做切片 += "的写法：
循环 K² 次、每次一整块切片累加 —— 既正确（+= 不是 =）又比 np.add.at 快得多。

⛔ 只允许 import numpy。
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .module import Module


def conv_output_size(size, kernel, stride=1, padding=0):
    """输出边长 = floor((size + 2P - K) / S) + 1。"""
    return (size + 2 * padding - kernel) // stride + 1


def conv_param_count(in_channels, out_channels, kernel_size, bias=True):
    """参数量 = Cin·Cout·K² (+ Cout)。"""
    return in_channels * out_channels * kernel_size ** 2 + (out_channels if bias else 0)


class Conv2d(Module):
    PARAM_NAMES = ('weight', 'bias')

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.naive = False

        # 与 kaiming_uniform(a=sqrt(5)) 等价，fan_in = Cin·K²（推导见 §2.1）
        bound = 1.0 / np.sqrt(in_channels * kernel_size ** 2)
        self.weight = np.random.uniform(
            -bound, bound, (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.random.uniform(-bound, bound, out_channels) if bias else None

        # 参数量公式的自检（任务书 §2.5 第 6 条）
        assert self.n_params() == conv_param_count(in_channels, out_channels, kernel_size, bias), \
            f'param count mismatch: {self.n_params()} vs formula'

    # ---------------------------------------------------------------- 前向
    def forward(self, x, naive=False):
        """x: (N,Cin,H,W) -> (N,Cout,Ho,Wo)。naive=True 走朴素 oracle。"""
        self.naive = naive
        self.x_shape = x.shape
        xp = self._pad(x)
        self.xp_shape = xp.shape
        ho = conv_output_size(x.shape[2], self.k, self.stride, self.padding)
        wo = conv_output_size(x.shape[3], self.k, self.stride, self.padding)

        if naive:
            self.xp = xp
            out = self._naive_forward(xp, ho, wo)
        else:
            self.x_col = self._im2col(xp, ho, wo)         # (N·Ho·Wo, Cin·K²)
            out = self._im2col_forward(ho, wo)

        if self.bias is not None:
            out += self.bias.reshape(1, -1, 1, 1)
        return out

    def _pad(self, x):
        if not self.padding:
            return x
        p = self.padding
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))

    # ---------------------------------------------------------------- 反向
    def backward(self, dout):
        dxp = self._naive_backward(dout) if self.naive else self._im2col_backward(dout)
        if self.bias is not None:
            self.grads['bias'] = dout.sum(axis=(0, 2, 3))
        if self.padding:
            p = self.padding
            h, w = self.x_shape[2:]
            return dxp[:, :, p:p + h, p:p + w]
        return dxp

    # ---------------------------------------------------------------- 朴素版（oracle）
    def _naive_forward(self, xp, ho, wo):
        n = xp.shape[0]
        k, s = self.k, self.stride
        out = np.zeros((n, self.out_channels, ho, wo), dtype=xp.dtype)
        for i in range(ho):
            for j in range(wo):
                patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]          # (N,Cin,K,K)
                out[:, :, i, j] = np.tensordot(patch, self.weight,
                                               axes=([1, 2, 3], [1, 2, 3]))
        return out

    def _naive_backward(self, dout):
        n, c, hp, wp = self.xp_shape
        ho, wo = dout.shape[2:]
        k, s = self.k, self.stride
        dxp = np.zeros_like(self.xp)
        dw = np.zeros_like(self.weight)
        for i in range(ho):
            for j in range(wo):
                patch = self.xp[:, :, i * s:i * s + k, j * s:j * s + k]
                d = dout[:, :, i, j]                                       # (N,Cout)
                dw += np.tensordot(d, patch, axes=([0], [0]))              # (Cout,Cin,K,K)
                # 重叠位置的贡献必须**累加**（+=），不能赋值 —— 本题头号坑
                dxp[:, :, i * s:i * s + k, j * s:j * s + k] += np.tensordot(
                    d, self.weight, axes=([1], [0]))
        self.grads['weight'] = dw
        return dxp

    # ---------------------------------------------------------------- im2col / col2im
    def _im2col(self, xp, ho, wo):
        """(N,C,Hp,Wp) → (N·Ho·Wo, C·K²) 的连续拷贝，每行是一个输出窗口。"""
        n, c = xp.shape[:2]
        k, s = self.k, self.stride
        win = sliding_window_view(xp, (k, k), axis=(2, 3))     # (N,C,Hp-K+1,Wp-K+1,K,K)
        win = win[:, :, ::s, ::s]                              # 按 stride 取起点（视图）
        col = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)
        return np.ascontiguousarray(col)                       # 不连续会让 BLAS 走慢路径

    def _im2col_forward(self, ho, wo):
        n = self.x_shape[0]
        w_col = self.weight.reshape(self.out_channels, -1)     # (Cout, Cin·K²)
        out = (self.x_col @ w_col.T).reshape(n, ho, wo, self.out_channels)
        return np.ascontiguousarray(out.transpose(0, 3, 1, 2))

    def _im2col_backward(self, dout):
        n, c, hp, wp = self.xp_shape
        ho, wo = dout.shape[2:]
        dout2 = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)   # (M,Cout)
        self.grads['weight'] = (dout2.T @ self.x_col).reshape(self.weight.shape)
        dx_col = dout2 @ self.weight.reshape(self.out_channels, -1)         # (M,Cin·K²)
        return self._col2im(dx_col, ho, wo)

    def _col2im(self, col, ho, wo):
        """(N·Ho·Wo, Cin·K²) → (N,Cin,Hp,Wp)：把每个窗口的贡献**累加**回重叠位置。"""
        n, c, hp, wp = self.xp_shape
        k, s = self.k, self.stride
        col = col.reshape(n, ho, wo, c, k, k)                  # 还原成开窗时的布局
        dx = np.zeros(self.xp_shape, dtype=col.dtype)
        for i in range(k):
            for j in range(k):
                # 窗口内偏移 (i,j) 的元素，贡献到输入上 (oh·S+i, ow·S+j) 的位置
                dx[:, :, i:i + s * ho:s, j:j + s * wo:s] += \
                    col[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx
