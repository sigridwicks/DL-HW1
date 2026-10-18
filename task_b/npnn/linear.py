import numpy as np
from .module import Module

class Linear(Module):
    PARAM_NAMES = ('weight', 'bias')
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        bound = 1.0 / np.sqrt(in_features)
        self.weight = np.random.uniform(-bound, bound, (out_features, in_features))
        self.bias = np.random.uniform(-bound, bound, out_features) if bias else None
        self.x = None

    def forward(self, x):
        self.x = x
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def backward(self, dout):
        x = self.x.reshape(-1, self.x.shape[-1])
        d = dout.reshape(-1, dout.shape[-1])
        self.grads['weight'] = d.T @ x
        if self.bias is not None:
            self.grads['bias'] = d.sum(axis=0)
        return dout @ self.weight