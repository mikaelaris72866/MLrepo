"""Item 4: verify the engine on  ((a*b + c)**2).relu().sum()  against
PyTorch autograd and against central-difference numerical gradients.

  python lab05/src/verify.py
Writes lab05/results/verification.json and prints a summary.  Threshold: max abs difference < 1e-6.
"""
import json
import random
from pathlib import Path

import torch

from autograd import Value, graph_nodes, value_sum

RESULTS = Path(__file__).resolve().parents[1] / "results"
N = 5            # length of the vectors a, b, c
TRIALS = 200     # random inputs
EPS = 1e-5       # step of the central difference
THRESH = 1e-6


def engine_forward(a, b, c):
    """((a*b + c)**2).relu().sum() on lists of Values."""
    return value_sum([((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c)])


def engine_grads(a, b, c):
    A, B, C = ([Value(x) for x in v] for v in (a, b, c))
    out = engine_forward(A, B, C)
    out.backward()
    return out.data, [v.grad for v in A + B + C], len(graph_nodes(out))


def torch_grads(a, b, c, dtype):
    ta, tb, tc = (torch.tensor(v, dtype=dtype, requires_grad=True) for v in (a, b, c))
    out = ((ta * tb + tc) ** 2).relu().sum()
    out.backward()
    return out.item(), [g for t in (ta, tb, tc) for g in t.grad.tolist()]


def numerical_grads(a, b, c, eps=EPS):
    base = [list(a), list(b), list(c)]

    def f():
        return engine_forward(*[[Value(x) for x in v] for v in base]).data

    grads = []
    for v in base:
        for i in range(len(v)):
            old = v[i]
            v[i] = old + eps
            hi = f()
            v[i] = old - eps
            lo = f()
            v[i] = old
            grads.append((hi - lo) / (2 * eps))
    return grads


def main():
    rng = random.Random(42)
    worst = {"engine_vs_torch64": 0.0, "engine_vs_numerical": 0.0, "engine_vs_torch32": 0.0,
             "value_diff_vs_torch64": 0.0}
    first = None
    for trial in range(TRIALS):
        a, b, c = ([rng.uniform(-2, 2) for _ in range(N)] for _ in range(3))
        out_e, g_e, n_nodes = engine_grads(a, b, c)
        out_t, g_t = torch_grads(a, b, c, torch.float64)
        _, g_32 = torch_grads(a, b, c, torch.float32)
        g_n = numerical_grads(a, b, c)
        d_t = max(abs(x - y) for x, y in zip(g_e, g_t))
        d_n = max(abs(x - y) for x, y in zip(g_e, g_n))
        d_32 = max(abs(x - y) for x, y in zip(g_e, g_32))
        worst["engine_vs_torch64"] = max(worst["engine_vs_torch64"], d_t)
        worst["engine_vs_numerical"] = max(worst["engine_vs_numerical"], d_n)
        worst["engine_vs_torch32"] = max(worst["engine_vs_torch32"], d_32)
        worst["value_diff_vs_torch64"] = max(worst["value_diff_vs_torch64"], abs(out_e - out_t))
        if first is None:
            first = {"a": a, "b": b, "c": c, "output": out_e, "graph_nodes": n_nodes,
                     "grad_engine": g_e, "grad_torch64": g_t, "grad_numerical": g_n,
                     "max_abs_diff_torch64": d_t, "max_abs_diff_numerical": d_n}
    summary = {"vector_length": N, "trials": TRIALS, "numerical_eps": EPS, "threshold": THRESH,
               "worst_case_max_abs_diff": worst,
               "pass_engine_vs_torch64": worst["engine_vs_torch64"] < THRESH,
               "pass_engine_vs_numerical": worst["engine_vs_numerical"] < THRESH,
               "example_trial": first}
    (RESULTS / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"graph nodes per forward pass (n={N}): {first['graph_nodes']}")
    print(f"example: output {first['output']:.6f}")
    print("  engine grad   :", [round(x, 6) for x in first["grad_engine"][:5]], "...")
    print("  torch  grad   :", [round(x, 6) for x in first["grad_torch64"][:5]], "...")
    for k, v in worst.items():
        print(f"worst max-abs-diff over {TRIALS} random inputs, {k:24s}: {v:.3e}")
    print("PASS (< 1e-6) vs torch float64:", summary["pass_engine_vs_torch64"],
          "| vs numerical:", summary["pass_engine_vs_numerical"])


if __name__ == "__main__":
    main()
