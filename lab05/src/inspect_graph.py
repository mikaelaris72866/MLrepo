"""Item 6 (part 1): inspect the graph of a small PyTorch model and the tensors in it.

  python lab05/src/inspect_graph.py
Writes lab05/results/graph_inspection.txt and tensor_table.csv.
Shows (a) the autograd graph (grad_fn chain), (b) the torch.fx graph, (c) the TorchScript trace graph,
(d) shape / dtype / stride / contiguity of parameters and activations, and (e) which view operations
produce non-contiguous tensors, which share storage and which force a copy.
"""
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import torch
from torch import nn

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results"


def small_model():
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())


def walk_autograd(fn, out, depth=0, seen=None):
    """Print the backward graph reachable from `fn` (a grad_fn), one line per node."""
    seen = {} if seen is None else seen   # id -> object; holding the object stops id() being recycled
    tag = " (already shown)" if id(fn) in seen else ""
    extra = f"  <- parameter {tuple(fn.variable.shape)}" if hasattr(fn, "variable") else ""
    out.append("  " * depth + type(fn).__name__ + extra + tag)
    if id(fn) in seen:
        return
    seen[id(fn)] = fn
    for nxt, _ in fn.next_functions:
        if nxt is not None:
            walk_autograd(nxt, out, depth + 1, seen)


def describe(t, name):
    return {"name": name, "shape": tuple(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "stride": t.stride(), "contiguous": t.is_contiguous(), "requires_grad": t.requires_grad,
            "is_leaf": t.is_leaf, "grad_fn": type(t.grad_fn).__name__ if t.grad_fn else "-",
            "bytes": t.numel() * t.element_size()}


def main():
    lines = []
    model = small_model()
    x = torch.randn(4, 2)
    y = torch.tensor([[0.], [1.], [1.], [0.]])

    # ---------------- (a) autograd graph, recorded during the forward pass ----------------
    loss = nn.BCELoss()(model(x), y)
    lines += ["=" * 70, "(a) AUTOGRAD GRAPH of  loss = BCELoss(model(x), y)   (built during the forward pass)", "=" * 70,
              f"loss.grad_fn = {loss.grad_fn}", "backward graph (outputs first, parameters last):"]
    walk_autograd(loss.grad_fn, lines)

    # ---------------- (b) torch.fx graph ----------------
    gm = torch.fx.symbolic_trace(model)
    lines += ["", "=" * 70, "(b) torch.fx graph of the model (symbolic trace, operator level)", "=" * 70, str(gm.graph)]

    # ---------------- (c) TorchScript trace ----------------
    traced = torch.jit.trace(model, x)
    lines += ["", "=" * 70, "(c) TorchScript graph (torch.jit.trace)", "=" * 70, str(traced.graph)]

    # ---------------- (d) tensors of the model ----------------
    rows = [describe(p, f"param {n}") for n, p in model.named_parameters()]
    acts = {}
    hooks = [m.register_forward_hook(lambda mod, i, o, n=n: acts.__setitem__(n, o))
             for n, m in model.named_children()]
    out = model(x)
    for h in hooks:
        h.remove()
    rows.append(describe(x, "input x"))
    for n, a in acts.items():
        rows.append(describe(a, f"activation after layer {n} ({type(model[int(n)]).__name__})"))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "tensor_table.csv", index=False)
    lines += ["", "=" * 70, "(d) TENSORS: shape, dtype, stride, contiguity", "=" * 70, table.to_string(index=False)]

    # ---------------- (e) views, storage sharing and contiguity ----------------
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cases = {
        "base = arange(12).reshape(3, 4)": base,
        "base.t()            (transpose)": base.t(),
        "base[:, ::2]        (strided slice)": base[:, ::2],
        "base[:, :2]         (column block)": base[:, :2],
        "base[1:]            (row block)": base[1:],
        "base.permute(1, 0)": base.permute(1, 0),
        "base[:1].expand(3, 4)  (stride 0)": base[:1].expand(3, 4),
        "base.t().contiguous()  (copy)": base.t().contiguous(),
        "base.t().reshape(12)   (must copy)": base.t().reshape(12),
        "base.reshape(12)       (a view)": base.reshape(12),
    }
    rows = []
    for name, t in cases.items():
        rows.append({"tensor": name, "shape": tuple(t.shape), "stride": t.stride(), "contiguous": t.is_contiguous(),
                     "shares_storage_with_base": t.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()})
    vt = pd.DataFrame(rows)
    vt.to_csv(RESULTS / "contiguity_table.csv", index=False)
    lines += ["", "=" * 70, "(e) VIEWS: which operations keep memory layout, which are non-contiguous", "=" * 70,
              vt.to_string(index=False)]
    try:
        base.t().view(12)
    except RuntimeError as e:
        lines += ["", "base.t().view(12) fails because the tensor is not contiguous:", "  RuntimeError: " + str(e).split("\n")[0]]

    # ---------------- (f) what torch.compile captured ----------------
    try:
        import torch._dynamo as dynamo
        dynamo.reset()
        buf = io.StringIO()
        with redirect_stdout(buf):
            exp = dynamo.explain(model)(x)
        lines += ["", "=" * 70, "(f) torch._dynamo.explain(model)(x)  -- what torch.compile captures", "=" * 70,
                  f"graphs captured: {exp.graph_count}, graph breaks: {exp.graph_break_count}, ops captured: {exp.op_count}"]
        for g in exp.graphs:
            lines += ["captured FX graph:", str(g.graph)]
    except Exception as e:  # keep the rest of the report if this API differs between versions
        lines += ["", f"(f) dynamo.explain unavailable: {type(e).__name__}: {str(e)[:150]}"]

    text = "\n".join(lines) + "\n"
    (RESULTS / "graph_inspection.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
