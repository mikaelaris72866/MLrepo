"""Item 2: profile one training epoch with torch.profiler and report the share of time spent on
data loading, forward, backward and the optimizer step.

  python lab06/src/profile_epoch.py [--set key=value ...]
The step is the same code as in train.py; each phase is wrapped in record_function (so the profiler sees it)
and timed with perf_counter. Both views are reported. Outputs: lab06/results/profile_*.{csv,txt,png}
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, resolve  # noqa: E402
from data import get_datasets, train_loader  # noqa: E402
from models import build_model  # noqa: E402
from train import Timers, build_optimizer, set_seed, train_one_epoch  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHASES = ["data_loading", "forward", "backward", "optimizer"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    cfg = load_config(None, ap.parse_args(argv).set)
    cfg["data"]["num_workers"] = 0          # data loading happens inline, so it is visible in the main thread
    set_seed(cfg["seed"]); torch.set_num_threads(cfg["train"]["threads"])
    train_ds, _, _ = get_datasets(cfg)
    loss_fn, n = nn.CrossEntropyLoss(reduction="sum"), len(train_ds)

    # warm-up on a throw-away model (first calls allocate buffers / spin up the thread pool); not profiled
    bs = cfg["data"]["batch_size"]
    warm, warm_n = build_model(cfg), min(30 * bs, len(train_ds))
    warm_dl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_ds, range(warm_n)), batch_size=bs)
    train_one_epoch(warm, warm_dl, build_optimizer(cfg, warm.parameters()), loss_fn, warm_n, 1)

    set_seed(cfg["seed"])
    model = build_model(cfg); opt = build_optimizer(cfg, model.parameters())
    loader, timers = train_loader(train_ds, cfg, 1), Timers()

    # Profiling every step of the epoch buffers hundreds of thousands of tiny per-sample events (the
    # dataset transform) and ran out of memory, so the profiler records a window of 10 steps out of every
    # 20 across the whole epoch (wait 8, warm-up 2, active 10) and each finished window is aggregated.
    agg = {}

    def collect(p):
        for e in p.key_averages():
            a = agg.setdefault(e.key, [0, 0.0, 0.0])
            a[0] += e.count; a[1] += e.self_cpu_time_total; a[2] += e.cpu_time_total

    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU], on_trace_ready=collect,
                 schedule=torch.profiler.schedule(wait=8, warmup=2, active=10, repeat=0)) as prof:
        train_one_epoch(model, loader, opt, loss_fn, n, cfg["train"]["accum_steps"], None, timers, step_hook=prof.step)
    wall = time.perf_counter() - t0

    # (a) shares from the explicit phase timers (all steps of the epoch)
    tt = timers.t; tsum = sum(tt.values())
    # (b) shares from the profiler's record_function events (cpu_time_total, inclusive; profiled windows only)
    pt = {k: agg[k][2] / 1e6 for k in PHASES}
    psum = sum(pt.values())
    table = pd.DataFrame({"phase": PHASES,
                          "wall_seconds": [tt[k] for k in PHASES], "wall_share_pct": [100 * tt[k] / tsum for k in PHASES],
                          "profiler_seconds": [pt[k] for k in PHASES], "profiler_share_pct": [100 * pt[k] / psum for k in PHASES]})
    table.round(3).to_csv(RESULTS / "profile_phases.csv", index=False)

    top = [{"op": k, "calls": v[0], "self_cpu_s": v[1] / 1e6, "self_cpu_pct": 100 * v[1] / 1e6 / psum}
           for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1])
           if k not in PHASES and not k.startswith(("enumerate(DataLoader)", "Optimizer.step", "ProfilerStep"))][:12]
    ops_df = pd.DataFrame(top).round(4)
    ops_df.to_csv(RESULTS / "profile_top_ops.csv", index=False)
    ops = ops_df.to_string(index=False)
    (RESULTS / "profile_top_ops.txt").write_text(ops + "\n")

    print(f"epoch wall time under the profiler: {wall:.1f}s  ({n / wall:.0f} samples/s)   steps: {len(loader)}  (profiler windows cover about half)")
    print(table.round(2).to_string(index=False))
    print(ops)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(PHASES[::-1], [table.wall_share_pct[PHASES.index(k)] for k in PHASES[::-1]], color="#4c78a8")
    for i, k in enumerate(PHASES[::-1]):
        v = table.wall_share_pct[PHASES.index(k)]
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("share of epoch time (%)"); ax.set_title("One training epoch, batch 128, 2 threads")
    ax.set_xlim(0, max(table.wall_share_pct) * 1.15); fig.tight_layout(); fig.savefig(RESULTS / "profile_shares.png", dpi=150)

    # log it to MLflow as well
    import mlflow
    mlflow.set_tracking_uri(resolve(cfg["run"]["mlflow"]["tracking_uri"]).as_uri()); mlflow.set_experiment("lab06")
    with mlflow.start_run(run_name="profile-one-epoch"):
        mlflow.set_tag("study", "profile")
        mlflow.log_params({"batch_size": cfg["data"]["batch_size"], "threads": cfg["train"]["threads"]})
        mlflow.log_metrics({f"share_{k}_pct": float(table.wall_share_pct[i]) for i, k in enumerate(PHASES)})
        mlflow.log_metric("epoch_seconds", wall)
        for f in ("profile_phases.csv", "profile_top_ops.txt", "profile_shares.png"):
            mlflow.log_artifact(str(RESULTS / f))


if __name__ == "__main__":
    main()
