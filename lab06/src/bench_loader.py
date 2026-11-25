"""Item 3 support: how fast can the data pipeline run on its own (no model), as a function of num_workers,
and what is the compute-only ceiling of the training step (same batch re-used, no data loading)?

  python lab06/src/bench_loader.py
Outputs lab06/results/loader_only.csv and compute_ceiling.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config  # noqa: E402
from data import get_datasets, train_loader  # noqa: E402
from models import build_model  # noqa: E402
from train import build_optimizer, set_seed  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    rows = []
    for augment in (False, True):
        for w in (0, 1, 2, 4, 8):
            cfg = load_config(None, [f"data.num_workers={w}", f"data.augment={str(augment).lower()}", "train.threads=2"])
            torch.set_num_threads(2)
            train_ds, _, _ = get_datasets(cfg)
            ts = []
            for ep in range(3):
                t0 = time.perf_counter()
                for _ in train_loader(train_ds, cfg, ep):
                    pass
                ts.append(time.perf_counter() - t0)
            m = statistics.median(ts)
            rows.append({"augment": augment, "num_workers": w, "median_epoch_s": m, "samples_per_s": len(train_ds) / m,
                         "runs_s": ";".join(f"{t:.2f}" for t in ts)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "loader_only.csv", index=False)

    # compute ceiling: forward+backward+Adam on one cached batch, no loader
    cfg = load_config(None, ["train.threads=2"]); set_seed(cfg["seed"]); torch.set_num_threads(2)
    model = build_model(cfg); opt = build_optimizer(cfg, model.parameters()); loss_fn = nn.CrossEntropyLoss()
    bs = cfg["data"]["batch_size"]; x, y = torch.randn(bs, 1, 28, 28), torch.randint(0, 10, (bs,))

    def step():
        opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    for _ in range(10):
        step()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            step()
        ts.append((time.perf_counter() - t0) / 50)
    per_step = statistics.median(ts)
    out = {"median_step_ms": 1e3 * per_step, "samples_per_s": bs / per_step, "batch_size": bs,
           "epoch_s_at_55k": 55000 / (bs / per_step)}
    (RESULTS / "compute_ceiling.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
