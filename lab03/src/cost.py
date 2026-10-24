"""Cost of each architecture: parameters, FLOPs (hand vs tools), latency.

Outputs in lab03/results/:
  cost.csv        per-model parameters, hand/thop/FlopCounter FLOPs, latency
  layer_flops.csv one layer of each type: hand formula vs thop vs FlopCounter
Latency protocol: eval mode, torch.inference_mode, 1 thread, real test images,
batch 1 -> median of 100 runs (10 warm-up), batch 128 -> median of 20 runs (3 warm-up).
The protocol is run 3 times; the reported latency is the median of the 3 medians
(individual repetitions are kept in cost.csv).
"""
import copy
import platform
import statistics
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from thop import profile as thop_profile
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary

from data import ROOT, TFM
from models import build_models

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------- hand formulas (per sample, 1 multiply-add = 2 FLOPs) ----------------


def linear_flops(i, o, tokens=1):
    return 2 * i * o * tokens


def conv_flops(k, cin, cout, h, w):
    return 2 * k * k * cin * cout * h * w


def attn_layer_flops(seq, d, ff):
    qkv = 2 * seq * d * 3 * d
    scores = 2 * seq * seq * d          # Q K^T
    context = 2 * seq * seq * d         # softmax(scores) V
    out = 2 * seq * d * d               # output projection
    ffn = 2 * seq * d * ff + 2 * seq * ff * d
    return dict(qkv=qkv, scores=scores, context=context, out=out, ffn=ffn,
                total=qkv + scores + context + out + ffn)


def hand_mlp(h1, h2):
    return linear_flops(784, h1) + linear_flops(h1, h2) + linear_flops(h2, 10)


def hand_cnn(c1, c2):
    return (conv_flops(3, 1, c1, 28, 28) + conv_flops(3, c1, c2, 14, 14)
            + linear_flops(c2 * 49, 10))


def hand_attn(d, ff, layers=2, seq=16):
    return (linear_flops(49, d, seq) + layers * attn_layer_flops(seq, d, ff)["total"]
            + linear_flops(d, 10))


HAND = {
    "mlp": hand_mlp(256, 128), "mlp_matched": hand_mlp(128, 64),
    "cnn": hand_cnn(16, 32), "cnn_matched": hand_cnn(64, 128),
    "attention": hand_attn(64, 128), "attention_pos": hand_attn(64, 128),
    "attention_matched": hand_attn(64, 256),
}

# ---------------- tools ----------------


def thop_flops(model, x, fastpath=True):
    """thop.profile; reports multiply-accumulates, so x2 for FLOPs.

    thop.profile always calls model.eval() and runs under no_grad.  In that
    state PyTorch's transformer layers take a fused inference fast path that
    never calls their submodules, so thop's hooks never fire (it reports 0 for
    the encoder).  fastpath=False switches the fast path off so the hooks fire;
    thop then still ignores the attention internals (QKV/out projections and
    the QK^T / AV products), which are computed functionally inside
    nn.MultiheadAttention.
    """
    m = copy.deepcopy(model)
    torch.backends.mha.set_fastpath_enabled(fastpath)
    try:
        macs, _ = thop_profile(m, inputs=(x,), verbose=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    return 2 * macs


def counter_flops(model, x, math_sdpa=False):
    """torch.utils.flop_counter in train mode (no fused fast path).

    With math_sdpa=True the attention product runs through the explicit
    matmul (MATH) backend, which the counter knows how to count; the default
    CPU scaled-dot-product kernel is not counted by this torch version.
    """
    m = copy.deepcopy(model).train()
    with FlopCounterMode(display=False) as fc:
        if math_sdpa:
            with sdpa_kernel(SDPBackend.MATH):
                m(x)
        else:
            m(x)
    return fc.get_total_flops()


def latency_ms(model, x, runs, warmup):
    model.eval()
    ts = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    from torchvision import datasets
    test = datasets.FashionMNIST(ROOT, train=False, download=False, transform=TFM)
    X = torch.stack([test[i][0] for i in range(128)])
    x1, x128 = X[:1], X

    rows = []
    for name, (model, group) in build_models().items():
        s = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        rows.append({
            "model": name, "group": group,
            "params_torchinfo": s.total_params,
            "params_manual_count": sum(p.numel() for p in model.parameters()),
            "torchinfo_mult_adds": s.total_mult_adds,
            "flops_hand": HAND[name],
            "flops_thop": thop_flops(model, x1),
            "flops_thop_nofastpath": thop_flops(model, x1, fastpath=False),
            "flops_flopcounter_default": counter_flops(model, x1),
            "flops_flopcounter_mathsdpa": counter_flops(model, x1, math_sdpa=True),
        })
        # latency protocol repeated 3 times to expose run-to-run noise
        for rep in (1, 2, 3):
            rows[-1][f"latency_b1_ms_rep{rep}"] = round(latency_ms(model, x1, 100, 10), 4)
            rows[-1][f"latency_b128_ms_rep{rep}"] = round(latency_ms(model, x128, 20, 3), 4)
        rows[-1]["latency_b1_ms"] = statistics.median(
            rows[-1][f"latency_b1_ms_rep{r}"] for r in (1, 2, 3))
        rows[-1]["latency_b128_ms"] = statistics.median(
            rows[-1][f"latency_b128_ms_rep{r}"] for r in (1, 2, 3))
        print(rows[-1])
    df = pd.DataFrame(rows)
    df["latency_b128_per_sample_us"] = (df.latency_b128_ms / 128 * 1e3).round(2)
    df.to_csv(RESULTS / "cost.csv", index=False)

    # ---- one layer of each type: hand vs thop vs FlopCounter ----
    lin = nn.Linear(784, 256)
    conv = nn.Conv2d(16, 32, 3, padding=1)
    enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True)
    a = attn_layer_flops(16, 64, 128)
    layer_rows = [
        {"layer": "Linear 784->256", "hand_flops": linear_flops(784, 256),
         "thop_flops": thop_flops(lin, torch.randn(1, 784)),
         "flopcounter_flops": counter_flops(lin, torch.randn(1, 784))},
        {"layer": "Conv2d 16->32, 3x3, 14x14 out", "hand_flops": conv_flops(3, 16, 32, 14, 14),
         "thop_flops": thop_flops(conv, torch.randn(1, 16, 14, 14)),
         "flopcounter_flops": counter_flops(conv, torch.randn(1, 16, 14, 14))},
        {"layer": "Transformer encoder layer (seq 16, d 64, ff 128)", "hand_flops": a["total"],
         "thop_flops": thop_flops(enc, torch.randn(1, 16, 64)),
         "thop_flops_nofastpath": thop_flops(enc, torch.randn(1, 16, 64), fastpath=False),
         "flopcounter_flops": counter_flops(enc, torch.randn(1, 16, 64)),
         "flopcounter_flops_mathsdpa": counter_flops(enc, torch.randn(1, 16, 64), math_sdpa=True)},
    ]
    pd.DataFrame(layer_rows).to_csv(RESULTS / "layer_flops.csv", index=False)
    pd.DataFrame([a]).to_csv(RESULTS / "attention_layer_hand_breakdown.csv", index=False)
    print(pd.DataFrame(layer_rows).to_string(index=False))
    (RESULTS / "machine.txt").write_text(
        f"torch {torch.__version__}, threads 1, {platform.processor() or platform.machine()}, "
        f"{platform.platform()}\n")


if __name__ == "__main__":
    main()
