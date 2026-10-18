import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from src.config import DATA_DIR, CIFAR100_MEAN, CIFAR100_STD, SEED


def build_transforms():
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tf, test_tf


def build_datasets():
    train_tf, test_tf = build_transforms()
    train_ds = CIFAR100(root=DATA_DIR, train=True,  transform=train_tf, download=False)
    test_ds  = CIFAR100(root=DATA_DIR, train=False, transform=test_tf,  download=False)
    return train_ds, test_ds


def build_loaders(batch_size=128, num_workers=0):
    train_ds, test_ds = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=num_workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=num_workers > 0)
    return train_loader, test_loader


def build_train_test_loaders(batch_size=128, num_workers=0, train_tf=None, test_tf=None,
                             monitor_size=2000):
    """Final-retrain protocol: full 50k train + 10k test (separate from 45k/5k).

    The third returned loader is the first ``monitor_size`` training images without
    augmentation. It is only a training-health monitor: the final retrain has no
    held-out validation set. It is not the test set; test is evaluated once after
    training finishes.
    """
    default_train_tf, default_test_tf = build_transforms()
    train_tf = train_tf if train_tf is not None else default_train_tf
    test_tf  = test_tf  if test_tf  is not None else default_test_tf

    train_ds = CIFAR100(root=DATA_DIR, train=True, transform=train_tf, download=False)
    eval_ds  = CIFAR100(root=DATA_DIR, train=True, transform=test_tf,  download=False)
    test_ds  = CIFAR100(root=DATA_DIR, train=False, transform=test_tf, download=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    monitor_loader = DataLoader(Subset(eval_ds, list(range(monitor_size))), batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True,
                                persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, test_loader, monitor_loader


def build_loaders_val(batch_size=128, num_workers=0, val_size=5000, train_tf=None, test_tf=None):
    """Val protocol: 45k train + 5k val (split from the train set) + full 10k test.

    All decisions are made on the val split; the test loader is only used for final
    reporting. The split is deterministic (own generator), so every experiment sees
    the same 45k/5k partition.

    train_tf / test_tf: optional transform overrides (default: build_transforms()),
    e.g. pass an augmented train_tf for the data-augmentation experiment.

    With num_workers > 0, data loading and augmentation move to worker processes and
    overlap with GPU compute. Measured 2026-10-16 (cnn_down + crop/flip, 45k, batch 128):
    num_workers=0 was 58.9 ms/batch (20.7 s/epoch); num_workers>=2 was 22.8 ms/batch
    (8.0 s/epoch), matching the pure-GPU floor of 22.6 ms. persistent_workers avoids
    re-spawning workers every epoch on Windows.
    """
    default_train_tf, default_test_tf = build_transforms()
    train_tf = train_tf if train_tf is not None else default_train_tf
    test_tf  = test_tf  if test_tf  is not None else default_test_tf

    # Same underlying train set, two transform variants: val must never be augmented
    ds_for_train = CIFAR100(root=DATA_DIR, train=True, transform=train_tf, download=False)
    ds_for_eval  = CIFAR100(root=DATA_DIR, train=True, transform=test_tf,  download=False)

    # Deterministic split; randperm uses its own generator and does not touch the
    # global RNG (so the DataLoader shuffle order stays reproducible)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(ds_for_train), generator=g)
    val_idx, train_idx = perm[:val_size].tolist(), perm[val_size:].tolist()

    train_loader = DataLoader(Subset(ds_for_train, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(Subset(ds_for_eval, val_idx), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    test_loader  = DataLoader(CIFAR100(root=DATA_DIR, train=False, transform=test_tf, download=False),
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader
