"""§1.3 极简训练链路（纯 SGD，不用 PyTorch）—— 把「接线错误」和「Adam 错误」分开定位。

跑法（在 task_b/ 下）：
    python tests/test_training_loop.py

做四件事：
  1. 真实 CIFAR-100 的小子集（4000 train / 1000 val）→ 我们自己的数据管线
  2. Flatten → Linear → GELU → Linear → CrossEntropyLoss（全是我们自己的模块）
  3. 反向 + **普通 SGD 三行**（不是 Adam）
  4. 断言 loss 下降、准确率高于瞎猜（100 类 = 1%）

这一步通过 = 数据/前向/反向/更新这条链是通的。
之后 Adam 出问题时，你就不用怀疑接线了。
"""

import sys
import time
from pathlib import Path

import numpy as np

TASKB_ROOT = Path(__file__).resolve().parents[1]
if str(TASKB_ROOT) not in sys.path:
    sys.path.insert(0, str(TASKB_ROOT))

from npnn import CIFAR100, CrossEntropyLoss, GELU, Linear, Sequential, batches   # noqa: E402
from npnn.data import Split                                                      # noqa: E402
from npnn.module import Module                                                   # noqa: E402


class Flatten(Module):
    """(N,3,32,32) → (N,3072)。接线件，不是评分模块。"""

    PARAM_NAMES = ()

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


def sgd_step(model, lr):
    """普通 SGD —— 就是这三行。"""
    for p, g in zip(model.parameters(), model.gradients()):
        if g is not None:
            p -= lr * g


def evaluate(model, criterion, split, batch_size=256):
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in batches(split, batch_size, shuffle=False, drop_last=False):
        logits = model.forward(xb)
        loss_sum += float(criterion.forward(logits, yb)) * len(yb)
        correct += int((logits.argmax(axis=1) == yb).sum())
        total += len(yb)
    return loss_sum / total, correct / total


def main():
    ds = CIFAR100()
    train = Split(ds.train.images[:4000], ds.train.labels[:4000])
    val = Split(ds.val.images[:1000], ds.val.labels[:1000])

    model = Sequential(Flatten(), Linear(3 * 32 * 32, 128), GELU(), Linear(128, 100))
    model.as_dtype(np.float32)                 # 训练用 float32（梯度检验才用 float64）
    criterion = CrossEntropyLoss()

    lr, epochs, batch_size = 0.05, 10, 128
    rng = np.random.default_rng(0)
    hist = []

    print(f'{"epoch":>5s}{"train loss":>12s}{"train acc":>11s}{"val loss":>10s}'
          f'{"val acc":>9s}{"time":>8s}')
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        for xb, yb in batches(train, batch_size, rng=rng):
            logits = model.forward(xb)
            loss = criterion.forward(logits, yb)
            model.backward(criterion.backward())      # dL/dlogits
            sgd_step(model, lr)
        dt = time.perf_counter() - t0

        tr_loss, tr_acc = evaluate(model, criterion, train)
        va_loss, va_acc = evaluate(model, criterion, val)
        hist.append(dict(train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc))
        print(f'{epoch:>5d}{tr_loss:>12.4f}{tr_acc:>10.1%}{va_loss:>10.4f}{va_acc:>9.1%}{dt:>7.2f}s')

    first, last = hist[0], hist[-1]
    ok_loss = last['train_loss'] < first['train_loss'] - 0.5
    ok_acc = last['train_acc'] > 0.05          # 100 类瞎猜是 1%
    print(f'\nloss {first["train_loss"]:.3f} -> {last["train_loss"]:.3f} '
          f'({last["train_loss"] - first["train_loss"]:+.3f})   '
          f'train acc {first["train_acc"]:.1%} -> {last["train_acc"]:.1%}')
    print('VERDICT:', 'PASS - the wiring works (data -> forward -> loss -> backward -> update)'
          if ok_loss and ok_acc else 'FAIL - something in the chain is wrong')
    return 0 if (ok_loss and ok_acc) else 1


if __name__ == '__main__':
    sys.exit(main())
