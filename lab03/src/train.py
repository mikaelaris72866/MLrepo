"""Train every model for 5 epochs (Adam, lr 1e-3, batch 128) on 55k Fashion-MNIST.

Each model is initialised from the same seed and sees the same data order.
Validation (5k) is logged every epoch; the test set (10k) is evaluated once,
after the last epoch (no checkpoint selection).
Outputs: results/training_log.csv, results/test_accuracy.csv
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from data import get_loaders
from models import SPECS

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED, EPOCHS, LR = 42, 5, 1e-3


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        tot_loss += loss_fn(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / n, correct / n


def main(names):
    torch.set_num_threads(2)
    loss_fn = nn.CrossEntropyLoss()
    log_rows, final_rows = [], []
    for name in names:
        factory, group = SPECS[name]
        torch.manual_seed(SEED)
        model = factory()
        train_dl, val_dl, test_dl = get_loaders(SEED)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        t_start = time.perf_counter()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            t0, run_loss, n = time.perf_counter(), 0.0, 0
            for x, y in train_dl:
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                run_loss += loss.item() * len(y)
                n += len(y)
            vl, va = evaluate(model, val_dl, loss_fn)
            log_rows.append({"model": name, "epoch": epoch, "train_loss": run_loss / n,
                             "val_loss": vl, "val_acc": va,
                             "epoch_seconds": time.perf_counter() - t0})
            print(name, log_rows[-1], flush=True)
        tl, ta = evaluate(model, test_dl, loss_fn)
        final_rows.append({"model": name, "group": group, "test_loss": tl, "test_acc": ta,
                           "val_acc_final": va, "train_seconds": time.perf_counter() - t_start})
        print("FINAL", final_rows[-1], flush=True)
        pd.DataFrame(log_rows).to_csv(RESULTS / "training_log.csv", index=False)
        pd.DataFrame(final_rows).to_csv(RESULTS / "test_accuracy.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SPECS))
