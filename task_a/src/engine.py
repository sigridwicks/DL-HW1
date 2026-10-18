import contextlib
import torch
import time


class _EMA:
    """Exponential moving average (EMA) shadow of the model weights.

    Updated after every optimizer.step():
        shadow = decay * shadow + (1 - decay) * weight
    where ``decay`` is the per-step retention factor. Evaluation temporarily swaps
    the shadow weights into the model and restores the originals afterwards, so EMA
    only affects the recorded curves and never training itself.
    Only floating-point entries are tracked (integer buffers such as
    num_batches_tracked are skipped); the shadow is stored as float32.
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
        """Temporarily load the EMA weights into the model for evaluation, then restore."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
                               for k, v in model.state_dict().items()})
        try:
            yield
        finally:
            model.load_state_dict(backup)


def train_one_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()  # training mode
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:

        # Move to the device; pin_memory + non_blocking overlaps the copy with compute.
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs =model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # EMA must follow every step, not every epoch (351 steps per epoch).
        if ema is not None:
            ema.update(model)

        # Accumulate running loss (note .item()) and correct / total.
        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) ==labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

@torch.no_grad()  # no graph needed for evaluation
def evaluate(model, loader, criterion, device):
    """Evaluate and return (mean loss, accuracy). Does not update parameters."""
    model.eval()  # evaluation mode
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
    """Evaluate top-1 and top-5 on CIFAR-100. Returns (loss, top1, top5).

    Kept separate from evaluate() to avoid changing the existing interface.
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
    """Average the probabilities of a list of models and return (top1, top5).

    models  models already on ``device`` (a single model is valid)
    tta     if True, each model also sees a horizontally flipped view
            (original + flip, two views)
    A 3-seed ensemble passes three models; TTA passes one model with tta=True;
    the two options can be combined.
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
    """Train and evaluate epoch by epoch; return the history.

    If ``scheduler`` is given, step it once after each epoch.
    If ``ema_decay`` is given, maintain a weight EMA (per-step retention factor;
    0.999 is typical for long runs) and additionally evaluate the EMA weights
    each epoch. EMA results go into ``ema_test_loss`` / ``ema_test_acc``; they do
    not affect training or the final model weights.
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
            with ema.applied(model):          # swap in EMA weights, restore on exit
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

        # Step once per epoch (supports epoch-level schedulers such as
        # CosineAnnealingLR / LinearLR / SequentialLR).
        if scheduler is not None:
            scheduler.step()

    return history
