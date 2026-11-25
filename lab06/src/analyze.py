"""Turns the finished runs into the tables and plots of the report.  python lab06/src/analyze.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUNS, RES = REPO / "lab06" / "runs", REPO / "lab06" / "results"


def load(study, name):
    p = RUNS / study / name / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def hist(s):
    return pd.DataFrame(s["history"])


def workers():
    rows = []
    for w in (0, 1, 2, 4, 8):
        s = load("workers", f"w{w}")
        if not s:
            continue
        h = hist(s)
        e2 = h.iloc[-1]                                        # epoch 2: warm
        rows.append({"num_workers": w, "epoch1_s": h.epoch_seconds[0], "epoch2_s": e2.epoch_seconds,
                     "throughput_epoch2": e2.throughput_samples_s,
                     "data_wait_s_epoch2": e2.t_data_loading, "data_wait_pct_epoch2": 100 * e2.t_data_loading / e2.epoch_seconds,
                     "compute_s_epoch2": e2.t_forward + e2.t_backward + e2.t_optimizer})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(RES / "workers_study.csv", index=False)
    lo = pd.read_csv(RES / "loader_only.csv")
    ceil = json.loads((RES / "compute_ceiling.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(df.num_workers, df.throughput_epoch2, "o-", label="full training (epoch 2)")
    ax[0].axhline(ceil["samples_per_s"], ls="--", c="gray", label="compute-only ceiling (no loader)")
    ax[0].set_xlabel("num_workers"); ax[0].set_ylabel("samples / s"); ax[0].set_ylim(0, None); ax[0].legend(fontsize=8)
    ax[0].set_title("Training throughput"); ax[0].set_xticks(df.num_workers)
    ax[1].bar(df.num_workers.astype(str), df.compute_s_epoch2, label="compute (fwd+bwd+opt)")
    ax[1].bar(df.num_workers.astype(str), df.data_wait_s_epoch2, bottom=df.compute_s_epoch2, label="main thread waiting for data")
    ax[1].set_xlabel("num_workers"); ax[1].set_ylabel("seconds in epoch 2"); ax[1].legend(fontsize=8)
    ax[1].set_title("Where the epoch time goes")
    for aug, ls in ((False, "-"), (True, "--")):
        d = lo[lo.augment == aug]
        ax[2].plot(d.num_workers, d.samples_per_s, "o" + ls, label="augmentation on" if aug else "plain (training setting)")
    ax[2].axhline(ceil["samples_per_s"], ls=":", c="gray", label="training compute ceiling")
    ax[2].set_yscale("log"); ax[2].set_xlabel("num_workers"); ax[2].set_ylabel("samples / s (loader alone)")
    ax[2].set_title("Data pipeline alone"); ax[2].legend(fontsize=8); ax[2].set_xticks([0, 1, 2, 4, 8])
    fig.tight_layout(); fig.savefig(RES / "workers_study.png", dpi=150)
    return df, lo, ceil


def batch():
    rows = []
    for bs in (32, 64, 128, 256, 512):
        s = load("batch", f"bs{bs}")
        if not s:
            continue
        h = hist(s)
        rows.append({"batch_size": bs, "steps_per_epoch": int(np.ceil(55000 / bs)),
                     "throughput_samples_s": s["median_throughput_samples_s"], "peak_rss_mb": s["peak_rss_mb"],
                     "activation_mb_est": s["activation_mb_per_batch_est"], "val_acc_ep5": s["final_val_acc"],
                     "test_acc_ep5": s["test_acc"], "train_loss_ep5": s["final_train_loss"]})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(RES / "batch_study.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(df.batch_size, df.throughput_samples_s, "o-"); ax[0].set_ylabel("samples / s"); ax[0].set_title("Throughput")
    ax[1].plot(df.batch_size, df.peak_rss_mb, "o-", label="peak RSS of the process")
    ax[1].plot(df.batch_size, df.activation_mb_est, "s--", label="activations kept for backward (estimate)")
    ax[1].set_ylabel("MB"); ax[1].set_title("Memory"); ax[1].legend(fontsize=7)
    ax[2].plot(df.batch_size, 100 * df.val_acc_ep5, "o-", label="validation"); ax[2].plot(df.batch_size, 100 * df.test_acc_ep5, "s--", label="test")
    ax[2].set_ylabel("accuracy after 5 epochs (%)"); ax[2].set_title("Accuracy"); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(df.batch_size); a.set_xticklabels(df.batch_size); a.set_xlabel("batch size"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "batch_study.png", dpi=150)
    return df


def sched():
    names = {"sgd_constant": "SGD+mom, constant", "sgd_step": "SGD+mom, StepLR", "sgd_cosine": "SGD+mom, CosineAnnealingLR",
             "sgd_plateau": "SGD+mom, ReduceLROnPlateau", "adam_constant": "Adam, constant"}
    rows, curves = [], {}
    for k, label in names.items():
        s = load("sched", k)
        if not s:
            continue
        h = hist(s); curves[label] = h
        rows.append({"run": label, "final_lr": h.lr.iloc[-1], "train_loss_ep10": s["final_train_loss"],
                     "val_loss_ep10": s["final_val_loss"], "val_acc_ep10": s["final_val_acc"], "best_val_acc": h.val_acc.max(),
                     "test_acc_ep10": s["test_acc"]})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(RES / "sched_study.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for label, h in curves.items():
        ax[0].plot(h.epoch, h.lr, "o-", ms=3, label=label); ax[1].plot(h.epoch, h.val_loss, "o-", ms=3, label=label)
        ax[2].plot(h.epoch, 100 * h.val_acc, "o-", ms=3, label=label)
    ax[0].set_yscale("log"); ax[0].set_title("Learning rate used in each epoch"); ax[1].set_title("Validation loss"); ax[2].set_title("Validation accuracy (%)")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[2].legend(fontsize=7); fig.tight_layout(); fig.savefig(RES / "sched_study.png", dpi=150)
    return df


def accum():
    a, b, base = load("accum", "bs64x4"), load("batch", "bs256"), load("batch", "bs64")
    if not (a and b):
        return None
    ha, hb = hist(a), hist(b)
    rows = []
    for e in range(len(ha)):
        rows.append({"epoch": e + 1, "bs256_train_loss": hb.train_loss[e], "bs64x4_train_loss": ha.train_loss[e],
                     "bs256_val_acc": hb.val_acc[e], "bs64x4_val_acc": ha.val_acc[e],
                     "bs64_noaccum_val_acc": hist(base).val_acc[e] if base else np.nan})
    df = pd.DataFrame(rows)
    summ = {"bs256_test": b["test_acc"], "bs64x4_test": a["test_acc"], "bs64_test": base["test_acc"] if base else None,
            "bs256_throughput": b["median_throughput_samples_s"], "bs64x4_throughput": a["median_throughput_samples_s"],
            "bs256_peak_rss_mb": b["peak_rss_mb"], "bs64x4_peak_rss_mb": a["peak_rss_mb"],
            "max_abs_train_loss_diff": float((ha.train_loss - hb.train_loss).abs().max()),
            "max_abs_val_acc_diff": float((ha.val_acc - hb.val_acc).abs().max())}
    df.round(5).to_csv(RES / "accum_study.csv", index=False)
    (RES / "accum_summary.json").write_text(json.dumps(summ, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hb.epoch, hb.train_loss, "o-", lw=3, alpha=0.5, label="batch 256"); ax[0].plot(ha.epoch, ha.train_loss, "s--", label="batch 64 x accum 4")
    if base:
        ax[0].plot(hist(base).epoch, hist(base).train_loss, ":", label="batch 64, no accumulation")
    ax[0].set_title("Training loss"); ax[1].plot(hb.epoch, 100 * hb.val_acc, "o-", lw=3, alpha=0.5, label="batch 256")
    ax[1].plot(ha.epoch, 100 * ha.val_acc, "s--", label="batch 64 x accum 4"); ax[1].set_title("Validation accuracy (%)")
    for x in ax:
        x.set_xlabel("epoch"); x.grid(alpha=0.3)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(RES / "accum_study.png", dpi=150)
    return df, summ


def resume():
    base, res = load("item1", "base"), load("resume", "resumed")
    part = RUNS / "resume" / "interrupted" / "history_partial.csv"
    if not (base and res):
        return None
    hb, hr = hist(base), hist(res)
    d = pd.DataFrame({"epoch": hb.epoch, "uninterrupted_train_loss": hb.train_loss, "resumed_train_loss": hr.train_loss,
                      "uninterrupted_val_acc": hb.val_acc, "resumed_val_acc": hr.val_acc, "lr_resumed": hr.lr})
    d.round(6).to_csv(RES / "resume_study.csv", index=False)
    summ = {"max_abs_train_loss_diff": float((hb.train_loss - hr.train_loss).abs().max()),
            "max_abs_val_loss_diff": float((hb.val_loss - hr.val_loss).abs().max()),
            "max_abs_val_acc_diff": float((hb.val_acc - hr.val_acc).abs().max()),
            "test_uninterrupted": base["test_acc"], "test_resumed": res["test_acc"],
            "interrupted_epochs": int(len(pd.read_csv(part))) if part.exists() else None}
    (RES / "resume_summary.json").write_text(json.dumps(summ, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(hb.epoch, hb.train_loss, "o-", lw=4, alpha=0.4, label="uninterrupted"); ax[0].plot(hr.epoch, hr.train_loss, "s--", label="interrupted after epoch 3, resumed")
    ax[1].plot(hb.epoch, hb.val_loss, "o-", lw=4, alpha=0.4, label="uninterrupted"); ax[1].plot(hr.epoch, hr.val_loss, "s--", label="resumed")
    for a, t in zip(ax, ("Training loss", "Validation loss")):
        a.axvline(3, c="gray", ls=":"); a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(RES / "resume_study.png", dpi=150)
    return d, summ


if __name__ == "__main__":
    for f in (workers, batch, sched, accum, resume):
        try:
            out = f()
        except FileNotFoundError as e:
            print(f.__name__, "skipped:", e); continue
        print("=" * 20, f.__name__); print(out if not isinstance(out, tuple) else "\n".join(str(o) for o in out))
