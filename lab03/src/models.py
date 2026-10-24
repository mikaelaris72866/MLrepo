"""MLP, CNN and patch-attention classifiers for 28x28 Fashion-MNIST.

Specified architectures (exactly as in the lab manual):
  mlp        784 -> 256 -> ReLU -> 128 -> ReLU -> 10
  cnn        Conv(1,16,3,p1) ReLU MaxPool2 Conv(16,32,3,p1) ReLU MaxPool2 Linear(32*7*7,10)
  attention  16 patches of 7x7 -> Linear(49,64) -> 2 x TransformerEncoderLayer
             (d_model=64, nhead=4, dim_feedforward=128) -> mean pool -> Linear(64,10)

The specified models have 235,146 / 20,490 / 70,794 parameters, i.e. none is
inside the 100k-150k target of the manual.  Size-matched variants (in range)
and an attention variant with learned positional embeddings are provided as
a clearly labelled supplement.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, h1=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 10),
        )

    def forward(self, x):
        return self.net(x)


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


class PatchAttention(nn.Module):
    """Split the image into a 4x4 grid of 7x7 patches (sequence length 16)."""

    def __init__(self, d_model=64, nhead=4, dim_ff=128, layers=2, pos_emb=False):
        super().__init__()
        self.proj = nn.Linear(49, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_model)) if pos_emb else None
        if pos_emb:
            nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        B = x.shape[0]
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)       # (B, 1, 4, 4, 7, 7)
        p = p.reshape(B, 16, 49)                      # 16 patches of 49 pixels
        z = self.proj(p)
        if self.pos is not None:
            z = z + self.pos
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


SPECS = {
    # name: (factory, group)
    "mlp": (lambda: MLP(), "spec"),
    "cnn": (lambda: CNN(), "spec"),
    "attention": (lambda: PatchAttention(), "spec"),
    # --- supplement: attention with learned positional embeddings ---
    "attention_pos": (lambda: PatchAttention(pos_emb=True), "supplement"),
    # --- supplement: sizes matched to the 100k-150k target ---
    "mlp_matched": (lambda: MLP(128, 64), "supplement"),
    "cnn_matched": (lambda: CNN(64, 128), "supplement"),
    "attention_matched": (lambda: PatchAttention(dim_ff=256), "supplement"),
}


def build_models():
    """name -> (model, group).  group is 'spec' or 'supplement'."""
    return {n: (f(), g) for n, (f, g) in SPECS.items()}
