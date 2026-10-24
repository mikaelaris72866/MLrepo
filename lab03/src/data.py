"""Fashion-MNIST loaders: 55k train / 5k validation / 10k test, batch 128."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2] / "data"
BATCH = 128
TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])


def get_loaders(seed=42, batch=BATCH):
    full = datasets.FashionMNIST(ROOT, train=True, download=False, transform=TFM)
    test = datasets.FashionMNIST(ROOT, train=False, download=False, transform=TFM)
    train, val = random_split(full, [55_000, 5_000], generator=torch.Generator().manual_seed(seed))
    g = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, generator=g),
        DataLoader(val, batch_size=batch),
        DataLoader(test, batch_size=batch),
    )
