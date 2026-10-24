"""Trade-off plots: test accuracy vs parameters and vs latency.

Outputs: results/acc_vs_params.png, results/acc_vs_latency.png, results/summary.csv
Filled markers = architectures exactly as specified; hollow = supplement
(size-matched variants and attention with positional embeddings).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
COLOR = {"mlp": "tab:blue", "cnn": "tab:orange", "attention": "tab:green"}


def family(name):
    return name.split("_")[0]


def scatter(ax, df, xcol, xlabel):
    for _, r in df.iterrows():
        c = COLOR[family(r.model)]
        filled = r.group == "spec"
        ax.scatter(r[xcol], r.test_acc * 100, s=90, marker="o" if filled else "D",
                   facecolors=c if filled else "none", edgecolors=c, linewidths=2, zorder=3)
        ax.annotate(r.model, (r[xcol], r.test_acc * 100), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3, which="both")


def main():
    cost = pd.read_csv(R / "cost.csv")
    acc = pd.read_csv(R / "test_accuracy.csv")
    df = cost.merge(acc, on=["model", "group"])
    df = df.rename(columns={"params_torchinfo": "params"})
    df[["model", "group", "params", "flops_hand", "latency_b1_ms", "latency_b128_ms",
        "latency_b128_per_sample_us", "test_acc", "train_seconds"]].round(4) \
        .to_csv(R / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter(ax, df, "params", "parameters (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Accuracy vs parameters (5 epochs, single seed)")
    fig.tight_layout()
    fig.savefig(R / "acc_vs_params.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    scatter(axes[0], df, "latency_b1_ms", "latency, batch 1 (ms, log scale)")
    scatter(axes[1], df, "latency_b128_per_sample_us", "latency, batch 128 (us per sample, log scale)")
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].set_title("Accuracy vs batch-1 latency")
    axes[1].set_title("Accuracy vs batch-128 latency")
    fig.tight_layout()
    fig.savefig(R / "acc_vs_latency.png", dpi=150)
    plt.close(fig)
    print(df[["model", "group", "params", "test_acc", "latency_b1_ms", "latency_b128_per_sample_us"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
