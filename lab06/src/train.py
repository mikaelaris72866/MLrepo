"""YAML-driven training pipeline (Lab 6, item 1).

  python lab06/src/train.py --config lab06/configs/base.yaml --set optimizer.lr=0.003 run.name=lr3e-3
  python lab06/src/train.py --set run.out_dir=lab06/runs/x run.resume=lab06/runs/x/last.pt   # resume

Everything that affects a run comes from the config (seed, data, model, optimizer, scheduler, accumulation).
Per epoch: checkpoint (model + optimizer + scheduler + history + MLflow run id) and MLflow metrics.
Finally: test accuracy once, summary.json / history.csv, and the config, checkpoint, history as MLflow artifacts.
"""
import argparse
import json
import random
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.profiler import record_function

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config_hash, flatten, load_config, resolve  # noqa: E402
from data import eval_loader, get_datasets, train_loader  # noqa: E402
from models import activation_bytes_per_sample, build_model, count_params  # noqa: E402

warnings.filterwarnings("ignore")


class TrainingInterrupted(Exception):
    """Raised on purpose after a checkpoint (config run.interrupt_after_epoch) to simulate a crash."""


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_optimizer(cfg, params):
    o = cfg["optimizer"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])


def build_scheduler(cfg, opt):
    s = cfg["scheduler"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=s["T_max"] or cfg["train"]["epochs"])
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s["factor"], patience=s["patience"])


class Timers:
    """Wall-clock time per phase of the training step (CPU ops are synchronous, so perf_counter is exact)."""

    def __init__(self):
        self.t = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}

    def region(self, name):
        timers = self

        class _R:
            def __enter__(s):
                s.rf = record_function(name); s.rf.__enter__(); s.t0 = time.perf_counter()

            def __exit__(s, *a):
                timers.t[name] += time.perf_counter() - s.t0
                s.rf.__exit__(*a)
        return _R()


def train_one_epoch(model, loader, opt, loss_fn, n_train, accum, grad_clip=None, timers=None, step_hook=None):
    """One epoch with gradient accumulation. The loss of each micro-batch is divided by the number of
    samples in its accumulation group, so an accumulated step equals one step on the concatenated batch."""
    timers = timers or Timers()
    model.train()
    bs, n_batches = loader.batch_size, len(loader)
    tot_loss, correct, seen, group_total = 0.0, 0, 0, 0
    it = iter(loader)
    opt.zero_grad(set_to_none=True)
    for i in range(n_batches):
        with timers.region("data_loading"):
            x, y = next(it)
        if i % accum == 0:
            group_total = min(accum * bs, n_train - seen)
        with timers.region("forward"):
            out = model(x)
            loss_sum = loss_fn(out, y)           # reduction="sum"
        with timers.region("backward"):
            (loss_sum / group_total).backward()
        tot_loss += loss_sum.item()
        correct += (out.argmax(1) == y).sum().item()
        seen += len(y)
        if (i + 1) % accum == 0 or i == n_batches - 1:
            with timers.region("optimizer"):
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step_hook:
                step_hook()
    return {"train_loss": tot_loss / seen, "train_acc": correct / seen, "samples": seen}


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        tot += loss_fn(out, y).item(); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n


def peak_rss_mb():
    """Peak resident memory of this process in MB (Linux/macOS: resource; Windows: psutil if present)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2**20 if sys.platform == "darwin" else r / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2**20
        except Exception:
            return float("nan")


def save_checkpoint(path, model, opt, sched, epoch, history, cfg, mlflow_run_id):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched else None, "epoch": epoch,
                "history": history, "config": cfg, "config_hash": config_hash(cfg),
                "mlflow_run_id": mlflow_run_id}, path)


def run(cfg):
    set_seed(cfg["seed"])
    torch.set_num_threads(cfg["train"]["threads"])
    out_dir = resolve(cfg["run"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds = get_datasets(cfg)
    val_dl, test_dl = eval_loader(val_ds, cfg), eval_loader(test_ds, cfg)
    model = build_model(cfg)
    opt, sched = build_optimizer(cfg, model.parameters()), None
    sched = build_scheduler(cfg, opt)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    history, start_epoch, run_id = [], 0, None

    resume = cfg["run"]["resume"]
    if resume:
        ck = torch.load(resolve(resume), map_location="cpu")
        if ck["config_hash"] != config_hash(cfg):
            raise ValueError("checkpoint was produced with a different training config; refusing to resume")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if sched and ck["scheduler"]:
            sched.load_state_dict(ck["scheduler"])
        history, start_epoch, run_id = ck["history"], ck["epoch"], ck["mlflow_run_id"]
        print(f"resumed from {resume} after epoch {start_epoch}")

    # ---- MLflow ----
    mlf, ml_run = cfg["run"]["mlflow"], None
    if mlf["enabled"]:
        import mlflow
        mlflow.set_tracking_uri(resolve(mlf["tracking_uri"]).as_uri())
        mlflow.set_experiment(mlf["experiment"])
        ml_run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name=cfg["run"]["name"])
        run_id = ml_run.info.run_id
        if not resume:
            mlflow.log_params({k: ("None" if v is None else v) for k, v in flatten(cfg).items()
                               if not k.startswith("run.mlflow")})
            mlflow.log_params({"model.params": count_params(model), "config_hash": config_hash(cfg)})
            mlflow.set_tags({"seed": cfg["seed"], **{k: str(v) for k, v in mlf["tags"].items()}})
        else:
            mlflow.set_tag("resumed", "true")

    n_train, accum = len(train_ds), cfg["train"]["accum_steps"]
    status = "FINISHED"
    try:
        for epoch in range(start_epoch + 1, cfg["train"]["epochs"] + 1):
            lr = opt.param_groups[0]["lr"]
            timers, t0 = Timers(), time.perf_counter()
            stats = train_one_epoch(model, train_loader(train_ds, cfg, epoch), opt, loss_fn, n_train, accum,
                                    cfg["train"]["grad_clip"], timers)
            epoch_s = time.perf_counter() - t0
            vl, va = evaluate(model, val_dl, loss_fn)
            row = {"epoch": epoch, "lr": lr, "train_loss": stats["train_loss"], "train_acc": stats["train_acc"],
                   "val_loss": vl, "val_acc": va, "epoch_seconds": epoch_s,
                   "throughput_samples_s": stats["samples"] / epoch_s,
                   **{f"t_{k}": v for k, v in timers.t.items()}}
            history.append(row)
            if sched:
                sched.step(vl) if cfg["scheduler"]["name"] == "plateau" else sched.step()
            save_checkpoint(out_dir / "last.pt", model, opt, sched, epoch, history, cfg, run_id)
            if ml_run:
                mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            print(f"epoch {epoch:2d} lr {lr:.5f} train {stats['train_loss']:.4f}/{stats['train_acc']:.4f} "
                  f"val {vl:.4f}/{va:.4f}  {epoch_s:.1f}s  {row['throughput_samples_s']:.0f} samples/s", flush=True)
            if cfg["run"]["interrupt_after_epoch"] == epoch and not resume:
                raise TrainingInterrupted(f"simulated interruption after epoch {epoch}")

        tl, ta = evaluate(model, test_dl, loss_fn)          # test set touched once, after the last epoch
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        summary = {"run_name": cfg["run"]["name"], "config_hash": config_hash(cfg), "mlflow_run_id": run_id,
                   "final_val_acc": history[-1]["val_acc"], "final_val_loss": history[-1]["val_loss"],
                   "final_train_loss": history[-1]["train_loss"], "test_acc": ta, "test_loss": tl,
                   "median_throughput_samples_s": float(np.median([h["throughput_samples_s"] for h in history])),
                   "peak_rss_mb": peak_rss_mb(),
                   "activation_mb_per_batch_est": activation_bytes_per_sample(model) * cfg["data"]["batch_size"] / 2**20,
                   "params": count_params(model), "epochs": len(history), "history": history}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if ml_run:
            mlflow.log_metrics({"test_acc": ta, "test_loss": tl, "peak_rss_mb": summary["peak_rss_mb"],
                                "median_throughput_samples_s": summary["median_throughput_samples_s"]})
            for f in ("config.yaml", "history.csv", "summary.json", "last.pt"):
                mlflow.log_artifact(str(out_dir / f))
        print(f"test accuracy {ta:.4f}  (peak RSS {summary['peak_rss_mb']:.0f} MB)")
        return summary
    except TrainingInterrupted as e:
        status = "KILLED"
        pd.DataFrame(history).to_csv(out_dir / "history_partial.csv", index=False)
        print(e)
        raise
    except BaseException:
        status = "FAILED"
        raise
    finally:
        if ml_run:
            mlflow.end_run(status)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    try:
        run(cfg)
    except TrainingInterrupted:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
