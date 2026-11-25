"""The Lab 3 best model: the size-matched CNN (cnn_matched, 137,226 parameters)."""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c2 * 7 * 7, 10))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(cfg):
    m = cfg["model"]
    if m["name"] != "cnn":
        raise ValueError(f"unknown model {m['name']}")
    return CNN(m["c1"], m["c2"])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def activation_bytes_per_sample(model, shape=(1, 1, 28, 28)):
    """Bytes of all layer outputs for one sample (what autograd keeps alive for backward), a
    platform-independent estimate of the activation memory that grows with batch size."""
    total, hooks = [0], []
    for mod in model.modules():
        if len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(lambda m, i, o: total.__setitem__(0, total[0] + o.numel() * o.element_size())))
    was = model.training
    model.eval()
    model(torch.zeros(shape))
    model.train(was)
    for h in hooks:
        h.remove()
    return total[0]
