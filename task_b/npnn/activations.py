import numpy as np
from .module import Module

_C = float(np.sqrt(2.0 / np.pi))
_K = 0.044715

class GELU(Module):
      PARAM_NAMES = ()
       
      def __init__(self):
            super().__init__()
            self.x = None
              
      def forward(self, x):
            self.x = x
            u = _C * (x + _K * x ** 3)
            return 0.5 * x * (1.0 + np.tanh(u))
      
      def backward(self, dout):
            x = self.x
            u = _C * (x + _K * x ** 3)
            t = np.tanh(u)
            du = _C * (1.0 + 3.0 * _K * x ** 2)
            return dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t **2) * du)

