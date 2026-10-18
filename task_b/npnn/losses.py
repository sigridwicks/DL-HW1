import numpy as np
from .module import Module

class CrossEntropyLoss(Module):
      PARAM_NAMES = ()
      def __init__(self):
            super().__init__()
            self.p = None
            self.labels = None

      def forward(self, logits, labels):
            n = logits.shape[0]
            z = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(z)
            denom = exp.sum(axis=1, keepdims=True)
            log_probs = z - np.log(denom)
            loss = -log_probs[np.arange(n), labels].mean()
            self.p = exp / denom
            self.labels = labels
            return loss
      
      def backward(self, dout=1.0):
            n = self.p.shape[0]
            d = self.p.copy()
            d[np.arange(n), self.labels] -= 1.0
            return d * (dout / n)
