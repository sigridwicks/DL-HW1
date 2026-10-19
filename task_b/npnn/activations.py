import numpy as np
from .module import Module

_C = float(np.sqrt(2.0 / np.pi))
_K = 0.044715

class GELU(Module):
      PARAM_NAMES = ()
       
      def __init__(self):
            super().__init__()
            self.x = None
            self.t = None        # 缓存 tanh(u)，反向不用重算

      def forward(self, x):
            self.x = x
            # ⚠️ 用 x*x*x 而不是 x**3：numpy 对整数幂 3 走通用 libm pow（逐元素 ~15 ns），
            #    比两次乘法慢 5-10 倍 —— 实测这是整网训练的最大热点（单层前向 57 ms）
            u = _C * (x + _K * (x * x * x))
            self.t = np.tanh(u)
            return 0.5 * x * (1.0 + self.t)

      def backward(self, dout):
            x, t = self.x, self.t
            du = _C * (1.0 + 3.0 * _K * (x * x))
            return dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)

