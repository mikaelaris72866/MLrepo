"""Runs the Lab 6 experiments by calling train.py (one subprocess per run, so peak memory is per run).
Each run is skipped if its summary.json already exists, so an interrupted batch can simply be restarted.

  python lab06/src/studies.py timing     # items 1-4, 6, 7 (timing-sensitive: run on an otherwise idle machine)
  python lab06/src/studies.py sched      # item 5 (accuracy only; two lanes x 1 thread each)
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
RUNS = REPO / "lab06" / "runs"


def train(name, study, overrides, expect=0):
    out = RUNS / study / name
    if (out / "summary.json").exists():
        print(f"[skip] {study}/{name}", flush=True)
        return out
    cmd = [sys.executable, str(SRC / "train.py"), "--set", f"run.name={study}-{name}", f"run.out_dir=lab06/runs/{study}/{name}",
           f"run.mlflow.tags={{study: {study}}}", *overrides]
    print("[run ]", study, name, flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode
    if rc != expect:
        raise SystemExit(f"{study}/{name} exited with {rc}")
    return out


def script(name, *args):
    print("[run ]", name, flush=True)
    subprocess.run([sys.executable, str(SRC / name), *args], cwd=REPO, check=True)


def timing():
    train("base", "item1", [])                                           # item 1: the Lab 3 best model, 5 epochs
    script("profile_epoch.py")                                           # item 2
    script("bench_loader.py")                                            # item 3 (pipeline alone, compute ceiling)
    for w in (0, 1, 2, 4, 8):                                            # item 3: full training, 2 epochs each
        train(f"w{w}", "workers", [f"data.num_workers={w}", "train.epochs=2"])
    for bs in (32, 64, 128, 256, 512):                                   # item 4
        train(f"bs{bs}", "batch", [f"data.batch_size={bs}"])
    train("bs64x4", "accum", ["data.batch_size=64", "train.accum_steps=4"])   # item 6 (bs256 run is batch/bs256)
    # item 7: interrupt after epoch 3 (exit code 3 expected), then resume from that epoch-3 checkpoint
    out = train("interrupted", "resume", ["run.interrupt_after_epoch=3"], expect=3) if not (RUNS / "resume/interrupted/last.pt").exists() \
        else RUNS / "resume/interrupted"
    ck = out / "epoch3.pt"
    if not ck.exists():
        shutil.copy(out / "last.pt", ck)
    train("resumed", "resume", [f"run.resume=lab06/runs/resume/interrupted/epoch3.pt"])


def sched():
    sgd = ["optimizer.name=sgd", "optimizer.lr=0.05", "optimizer.momentum=0.9", "train.epochs=10", "train.threads=1"]
    lanes = [
        [("sgd_constant", "scheduler.name=none"), ("sgd_step", "scheduler.name=step"), ("sgd_cosine", "scheduler.name=cosine")],
        [("sgd_plateau", "scheduler.name=plateau"), ("adam_constant", None)],
    ]
    procs = []
    for lane in lanes:
        lines = []
        for name, extra in lane:
            ov = (["optimizer.name=adam", "optimizer.lr=0.001", "train.epochs=10", "train.threads=1"] if extra is None else sgd + [extra])
            lines.append(f"train({name!r}, 'sched', {ov!r})")
        code = "import sys; sys.path.insert(0, %r); from studies import train\n" % str(SRC) + "\n".join(lines)
        procs.append(subprocess.Popen([sys.executable, "-c", code], cwd=REPO))
    for p in procs:
        if p.wait() != 0:
            raise SystemExit("a scheduler lane failed")


if __name__ == "__main__":
    {"timing": timing, "sched": sched}[sys.argv[1]]()
