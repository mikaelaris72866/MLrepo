"""Fashion-MNIST from the raw idx files (gzipped or not): no torchvision needed."""
import gzip
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from config import resolve


def read_idx(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    _, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def find_file(root, name):
    for cand in (f"{name}.gz", name):
        p = root / "FashionMNIST" / "raw" / cand
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}(.gz) not found under {root}/FashionMNIST/raw - see README for the download")


def load_split(root, train):
    prefix = "train" if train else "t10k"
    images = read_idx(find_file(root, f"{prefix}-images-idx3-ubyte"))
    labels = read_idx(find_file(root, f"{prefix}-labels-idx1-ubyte"))
    return torch.from_numpy(images.copy()), torch.from_numpy(labels.astype(np.int64))


class FashionDataset(Dataset):
    """Per-sample transform in __getitem__ (like torchvision's ToTensor + Normalize [+ augmentation]),
    so that data loading has a realistic per-sample cost that num_workers can parallelise."""

    def __init__(self, images, labels, mean, std, augment=False):
        self.images, self.labels, self.mean, self.std, self.augment = images, labels, mean, std, augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float().div_(255.0).unsqueeze(0)           # (1, 28, 28) in [0, 1]
        if self.augment:
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            dy, dx = (torch.randint(-2, 3, (2,))).tolist()
            x = torch.nn.functional.pad(x, (2, 2, 2, 2))[:, 2 + dy:30 + dy, 2 + dx:30 + dx]
        return (x - self.mean) / self.std, self.labels[i]


def get_datasets(cfg):
    d = cfg["data"]
    root = resolve(d["root"])
    tr_x, tr_y = load_split(root, True)
    te_x, te_y = load_split(root, False)
    full = FashionDataset(tr_x, tr_y, d["mean"], d["std"], d["augment"])
    plain = FashionDataset(tr_x, tr_y, d["mean"], d["std"], False)     # validation never augmented
    test = FashionDataset(te_x, te_y, d["mean"], d["std"], False)
    n_val = d["val_size"]
    # Same call as Lab 3: random_split with a seeded generator -> the same 55k/5k split.
    split = random_split(range(len(full)), [len(full) - n_val, n_val],
                         generator=torch.Generator().manual_seed(cfg["seed"]))
    train_idx, val_idx = list(split[0].indices), list(split[1].indices)
    if d["subset_train"]:
        train_idx = train_idx[:d["subset_train"]]
        val_idx = val_idx[:max(1, min(n_val, d["subset_train"] // 4))]
    train, val = Subset(full, train_idx), Subset(plain, val_idx)
    if d["subset_test"]:
        test = Subset(test, list(range(d["subset_test"])))
    return train, val, test


def train_loader(train_ds, cfg, epoch):
    """A fresh loader per epoch whose shuffling is a pure function of (seed, epoch): this is what makes
    resuming from a checkpoint reproduce the uninterrupted run, and what keeps batch-64x4 and batch-256 comparable."""
    g = torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch)
    return DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, generator=g,
                      num_workers=cfg["data"]["num_workers"])


def eval_loader(ds, cfg):
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
