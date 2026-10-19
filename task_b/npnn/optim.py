import numpy as np

class Adam:
  def __init__(self, params, lr):
    self.params = list(params)
    self.lr = lr
    self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
    self.t = 0
    self.m = [np.zeros_like(p) for p in self.params]
    self.v = [np.zeros_like(p) for p in self.params]

  def step(self, grads):
    self.t += 1
    for p, g, m, v in zip(self.params, grads, self.m, self.v):
      if g is None:
        continue
      m *= self.b1
      m += (1.0 - self.b1) * g
      v *= self.b2
      v += (1.0 - self.b2) * g * g
      m_hat = m / (1.0 - self.b1 ** self.t)
      v_hat = v / (1.0 - self.b2 ** self.t)
      p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) 

  def state_dict(self):
    state = {'t': np.array(self.t)}
    for i, (m, v) in enumerate(zip(self.m, self.v)):
        state[f'm{i}'] = m
        state[f'v{i}'] = v
    return state

  def load_state_dict(self, state):
      self.t = int(state['t'])
      for i in range(len(self.params)):
          self.m[i] = np.array(state[f'm{i}'])
          self.v[i] = np.array(state[f'v{i}'])
