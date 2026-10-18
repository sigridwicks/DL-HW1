import contextlib
import torch
import time


class _EMA:
    """权重的指数滑动平均（EMA）影子。

    每个 optimizer.step() 之后更新一次：shadow = decay * shadow + (1 - decay) * weight，
    其中 decay 是**每个 step** 的保留系数。评估时临时把影子权重装进 model，评估完换回来 ——
    所以它只影响记录下来的曲线，绝不参与训练本身。
    只跟浮点条目（num_batches_tracked 这种整型 buffer 跳过）；影子一律存 float32。
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    @contextlib.contextmanager
    def applied(self, model):
        """临时把 EMA 权重换进 model（评估用），退出时恢复原权重。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
                               for k, v in model.state_dict().items()})
        try:
            yield
        finally:
            model.load_state_dict(backup)


def train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train() # 切训练模式
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:

        # 搬运到 device（配合 pin_memory，用 non_blocking=True）
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs =model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # EMA 要按 step 跟，不能按 epoch 跟（一个 epoch 351 步，按 epoch 跟等于没跟）
        if ema is not None:
            ema.update(model)

        # 累计 running_loss（注意 .item()）和 correct / total
        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) ==labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

@torch.no_grad() # 评估不建计算图
def evaluate(model, loader, criterion, device):
    """评估：返回 (平均 loss, 准确率)。不更新参数。"""
    model.eval() # 关键：评估模式
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_topk(model, loader, criterion, device):
    """评估 top-1 和 top-5（CIFAR-100 共 100 类）。返回 (loss, top1, top5)。

    和 evaluate() 分开写，避免动到已经在跑的老接口。
    """
    model.eval()
    loss_sum, top1, top5, total = 0.0, 0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss_sum += criterion(outputs, labels).item()

        idx = outputs.topk(5, dim=1).indices                 # (B, 5)
        top1 += (idx[:, 0] == labels).sum().item()
        top5 += (idx == labels[:, None]).any(dim=1).sum().item()
        total += labels.size(0)

    return loss_sum / len(loader), top1 / total, top5 / total


@torch.no_grad()
def ensemble_eval(models, loader, device, tta=False):
    """对一组模型做**概率平均**，返回 (top1, top5)。

    models  已在 device 上的模型列表（可以只有一个 —— 那就是单模型评估）
    tta     True 时每个模型额外看一次水平翻转（原图 + 翻转，2 个视角）
    3-seed 集成 = 传 3 个模型；TTA = 传 1 个模型 + tta=True；两者可叠加。
    """
    for m in models:
        m.eval()

    top1 = top5 = total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        views = [imgs, torch.flip(imgs, dims=[3])] if tta else [imgs]
        prob = 0
        for m in models:
            for v in views:
                prob = prob + m(v).softmax(dim=1)
        prob = prob / (len(models) * len(views))

        top1 += (prob.argmax(dim=1) == labels).sum().item()
        top5 += (prob.topk(5, dim=1).indices == labels[:, None]).any(dim=1).sum().item()
        total += labels.size(0)

    return top1 / total, top5 / total


def fit(model, train_loader, test_loader, criterion, optimizer, epochs, device, scheduler=None,
        ema_decay=None):
    """逐 epoch 训练 + 评估，返回 history。scheduler 若给定，每 epoch 结束后 step 一次。

    ema_decay 给定时额外维护一份权重滑动平均（decay 是每个 step 的保留系数，长训练常用 0.999），
    并且每 epoch 用 EMA 权重再评估一次 —— 结果进 history 的 ema_test_loss / ema_test_acc，
    不会改动训练本身，也不会改 model 的最终权重。
    """
    ema = _EMA(model, ema_decay) if ema_decay else None

    history = {"train_loss": [], "train_acc": [],
                "test_loss": [], "test_acc": [],
                "epoch_time": []}
    if ema is not None:
        history["ema_test_loss"] = []
        history["ema_test_acc"] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, ema=ema)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        if ema is not None:
            with ema.applied(model):          # 临时换成 EMA 权重，评估完自动换回
                ema_loss, ema_acc = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(dt)
        if ema is not None:
            history["ema_test_loss"].append(ema_loss)
            history["ema_test_acc"].append(ema_acc)

        line = (f"epoch {epoch:2d}/{epochs} | "
                f"train {train_loss:.3f}/{train_acc:.3f} | "
                f"test {test_loss:.3f}/{test_acc:.3f}")
        if ema is not None:
            line += f" | ema {ema_loss:.3f}/{ema_acc:.3f}"
        print(f"{line} | {dt:.0f}s")

        # 每 epoch 结束 step 一次（仅支持按 epoch 调度的 scheduler：CosineAnnealingLR / LinearLR / SequentialLR）
        if scheduler is not None:
            scheduler.step()

    return history