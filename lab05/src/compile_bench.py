"""Item 6 (part 2): eager vs torch.compile vs torch.jit.trace, and the cost of non-contiguous views.

  python lab05/src/compile_bench.py
Timing protocol: 1 thread, warm-up calls (excluded; the first compiled call = compile time, reported
separately), then the median of RUNS timed batches. Outputs CSVs in lab05/results/.
"""
import time
import warnings

import pandas as pd
import torch
from torch import nn

from timing import bench

warnings.filterwarnings("ignore")
from pathlib import Path  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = 15


def first_call(fn, *a):
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


def per_call(fn, args, reps, runs=RUNS):
    """median seconds per call, each timed sample averages `reps` calls."""
    def batch():
        for _ in range(reps):
            fn(*args)
    m, q1, q3 = bench(batch, runs, warmup=3)
    return m / reps, q1 / reps, q3 / reps


def compile_comparison():
    rows = []
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()).eval()
    big = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10)).eval()

    def chain(x):  # many cheap elementwise ops: fusion removes memory passes
        return torch.sigmoid(torch.relu(x * 1.5 + 0.3) * x - x.tanh() * 0.5) + x.exp().clamp(max=3.0)

    cases = [("tiny MLP 2-8-1, batch 256", mlp, (torch.randn(256, 2),), 300),
             ("MLP 256-512-512-10, batch 128", big, (torch.randn(128, 256),), 50),
             ("elementwise chain, 1M floats", chain, (torch.randn(1_000_000),), 5)]
    for name, fn, args, reps in cases:
        with torch.no_grad():
            variants = {"eager": fn}
            t0 = time.perf_counter()
            traced = torch.jit.trace(fn, args) if isinstance(fn, nn.Module) else torch.jit.trace(fn, args)
            variants["jit.trace"] = traced
            jit_t = first_call(traced, *args)
            comp = torch.compile(fn)
            comp_first = first_call(comp, *args)
            variants["torch.compile"] = comp
            ref = fn(*args)
            for vname, v in variants.items():
                err = (v(*args) - ref).abs().max().item()
                m, q1, q3 = per_call(v, args, reps)
                rows.append({"case": name, "variant": vname, "median_us": 1e6 * m, "q1_us": 1e6 * q1,
                             "q3_us": 1e6 * q3, "first_call_s": comp_first if vname == "torch.compile" else (jit_t if vname == "jit.trace" else 0.0),
                             "max_abs_err_vs_eager": err})
    df = pd.DataFrame(rows)
    df["speedup_vs_eager"] = df.groupby("case")["median_us"].transform(lambda s: s.iloc[0] / s)
    return df.round(4)


def train_step_comparison():
    """Forward + backward + SGD step on the moons-sized model, eager vs compiled model."""
    rows = []
    X, y = torch.randn(200, 2), torch.randint(0, 2, (200, 1)).float()
    for name, wrap in (("eager", lambda m: m), ("torch.compile(model)", torch.compile)):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
        f = wrap(net)
        opt, lf = torch.optim.SGD(net.parameters(), lr=0.1), nn.BCELoss()

        def step():
            opt.zero_grad(); lf(f(X), y).backward(); opt.step()
        t0 = time.perf_counter(); step(); first = time.perf_counter() - t0
        m, q1, q3 = per_call(lambda: step(), (), 100)
        rows.append({"variant": name, "median_us_per_train_step": 1e6 * m, "first_step_s": first})
    return pd.DataFrame(rows).round(3)


def contiguity_bench():
    rows = []
    for n in (256, 4096):  # 256x256 floats = 256 KB (cache resident), 4096x4096 = 64 MB (memory bound)
        reps = 200 if n == 256 else 3
        a = torch.randn(n, n); b = torch.randn(n, n)
        at = a.t()                      # non-contiguous view
        ac = at.contiguous()
        out = torch.empty(n, n)
        ops = {
            "a + b            (both contiguous)": lambda: torch.add(a, b, out=out),
            "a.t() + b        (one transposed view)": lambda: torch.add(at, b, out=out),
            "a.t() + b.t()    (both transposed: same layout, still strided)": lambda: torch.add(at, b.t(), out=out),
            "a.t().contiguous() + b   (copy first, then add)": lambda: torch.add(at.contiguous(), b, out=out),
            "a[:, ::2].sum()  (strided slice, half the data)": lambda: a[:, ::2].sum(),
            "a[:, :n//2].contiguous().sum()": lambda: a[:, : n // 2].contiguous().sum(),
            "a.sum()          (contiguous, all data)": lambda: a.sum(),
            "a @ b            (contiguous)": lambda: a @ b,
            "a.t() @ b        (transposed operand)": lambda: at @ b,
        }
        if n == 4096:
            ops.pop("a @ b            (contiguous)"); ops.pop("a.t() @ b        (transposed operand)")
        base = None
        for name, fn in ops.items():
            m, q1, q3 = bench(lambda fn=fn: [fn() for _ in range(reps)], 9, warmup=2)
            m /= reps
            if base is None:
                base = m
            rows.append({"size": f"{n}x{n}", "operation": name, "median_ms": 1e3 * m, "relative_to_first": m / base})
    return pd.DataFrame(rows).round(4)


def main():
    torch.set_num_threads(1)
    print("torch", torch.__version__, "| threads", torch.get_num_threads())
    c = compile_comparison()
    c.to_csv(RESULTS / "compile_comparison.csv", index=False)
    print(c.to_string(index=False))
    t = train_step_comparison()
    t.to_csv(RESULTS / "compile_train_step.csv", index=False)
    print(t.to_string(index=False))
    k = contiguity_bench()
    k.to_csv(RESULTS / "contiguity_timing.csv", index=False)
    print(k.to_string(index=False))


if __name__ == "__main__":
    main()
