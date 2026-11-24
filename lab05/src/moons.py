"""Item 5: train a 2 -> 8 -> 1 network (ReLU hidden layer, sigmoid output, BCE loss) on make_moons
with (a) our scalar autograd engine and (b) PyTorch, and compare speed.

  python lab05/src/moons.py
Both models start from identical weights and use full-batch gradient descent, so the two
training runs are mathematically the same computation; the comparison is about speed.
Outputs: lab05/results/moons_results.json, moons_timing.csv, moons_training.png
"""
import gc
import json
import statistics
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from autograd import Value, value_sum  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parents[1] / "results"
SEED, EPOCHS, LR, HIDDEN, RUNS = 42, 200, 0.5, 8, 5


def get_data(n_samples=300, seed=SEED):
    X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=seed)
    return train_test_split(X, y, test_size=1 / 3, stratify=y, random_state=seed)


def init_weights(seed=SEED):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 0.7, (HIDDEN, 2)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, 0.7, (1, HIDDEN)), "b2": np.zeros(1)}


# --------------------------------------------------------------------------- our engine
class EngineMLP:
    def __init__(self, w):
        self.W1 = [[Value(x) for x in row] for row in w["W1"]]
        self.b1 = [Value(x) for x in w["b1"]]
        self.W2 = [Value(x) for x in w["W2"][0]]
        self.b2 = Value(w["b2"][0])
        self.params = [v for row in self.W1 for v in row] + self.b1 + self.W2 + [self.b2]

    def forward(self, x0, x1):
        hidden = [(row[0] * x0 + row[1] * x1 + b).relu() for row, b in zip(self.W1, self.b1)]
        z = value_sum([w * h for w, h in zip(self.W2, hidden)]) + self.b2
        return z.sigmoid()

    def loss(self, inputs, labels):
        """Mean binary cross-entropy; the label is a constant, so we pick the matching term."""
        terms = []
        for (x0, x1), y in zip(inputs, labels):
            p = self.forward(x0, x1)
            terms.append(-p.log() if y == 1 else -(1.0 - p).log())
        return value_sum(terms) * (1.0 / len(terms))

    def train_epoch(self, inputs, labels, lr):
        for v in self.params:
            v.grad = 0.0
        loss = self.loss(inputs, labels)
        loss.backward()
        for v in self.params:
            v.data -= lr * v.grad
        return loss.data

    def predict(self, X):
        return np.array([self.forward(Value(a), Value(b)).data > 0.5 for a, b in X], dtype=int)

    def flat_params(self):
        return np.array([v.data for v in self.params])


def to_values(X):
    return [(Value(a), Value(b)) for a, b in X]


def train_engine(w, Xtr, ytr, epochs=EPOCHS, lr=LR):
    model, inputs = EngineMLP(w), to_values(Xtr)
    losses = [model.train_epoch(inputs, ytr, lr) for _ in range(epochs)]
    return model, losses


# --------------------------------------------------------------------------- PyTorch
def make_torch(w, dtype):
    net = torch.nn.Sequential(torch.nn.Linear(2, HIDDEN), torch.nn.ReLU(),
                              torch.nn.Linear(HIDDEN, 1), torch.nn.Sigmoid()).to(dtype)
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(w["W1"])); net[0].bias.copy_(torch.tensor(w["b1"]))
        net[2].weight.copy_(torch.tensor(w["W2"])); net[2].bias.copy_(torch.tensor(w["b2"]))
    return net


def train_torch(w, Xtr, ytr, epochs=EPOCHS, lr=LR, dtype=torch.float64):
    net = make_torch(w, dtype)
    X, y = torch.tensor(Xtr, dtype=dtype), torch.tensor(ytr, dtype=dtype).unsqueeze(1)
    opt, loss_fn, losses = torch.optim.SGD(net.parameters(), lr=lr), torch.nn.BCELoss(), []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return net, losses


def torch_flat(net):
    return np.concatenate([net[0].weight.detach().numpy().ravel(), net[0].bias.detach().numpy(),
                           net[2].weight.detach().numpy().ravel(), net[2].bias.detach().numpy()])


# --------------------------------------------------------------------------- timing
def median_time(fn, runs=RUNS):
    fn()  # one untimed warm-up run
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def median_time_nogc(fn, runs=RUNS):
    """Diagnostic: time fn with the cyclic GC disabled. The graph is full of reference cycles
    (node <-> its backward closure), so the garbage is collected outside the timed region."""
    fn()
    ts = []
    for _ in range(runs):
        gc.collect(); gc.disable()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        gc.enable()
    gc.collect()
    return statistics.median(ts), ts


def main():
    torch.set_num_threads(1)
    Xtr, Xte, ytr, yte = get_data()
    w = init_weights()

    # ---- equivalence of the two implementations ----
    eng, eng_losses = train_engine(w, Xtr, ytr)
    net, tor_losses = train_torch(w, Xtr, ytr)
    acc = lambda p, y: float((p == y).mean())
    eng_tr, eng_te = acc(eng.predict(Xtr), ytr), acc(eng.predict(Xte), yte)
    with torch.no_grad():
        tor_pred = lambda X: (net(torch.tensor(X)).squeeze(1) > 0.5).numpy().astype(int)
        tor_tr, tor_te = acc(tor_pred(Xtr), ytr), acc(tor_pred(Xte), yte)
    equiv = {
        "final_loss_engine": eng_losses[-1], "final_loss_torch64": tor_losses[-1],
        "max_abs_loss_diff_over_epochs": float(np.max(np.abs(np.array(eng_losses) - np.array(tor_losses)))),
        "max_abs_param_diff_after_training": float(np.max(np.abs(eng.flat_params() - torch_flat(net)))),
        "initial_loss": eng_losses[0], "n_train": len(Xtr), "n_test": len(Xte),
        "train_accuracy_engine": eng_tr, "test_accuracy_engine": eng_te,
        "train_accuracy_torch": tor_tr, "test_accuracy_torch": tor_te,
        "majority_baseline_test_accuracy": float(max(yte.mean(), 1 - yte.mean())),
        "epochs": EPOCHS, "lr": LR,
    }
    print(json.dumps(equiv, indent=2))

    # ---- speed: whole training run, median of 5 (after one warm-up run) ----
    t_eng, runs_eng = median_time(lambda: train_engine(w, Xtr, ytr))
    t_t64, runs_t64 = median_time(lambda: train_torch(w, Xtr, ytr, dtype=torch.float64))
    t_t32, runs_t32 = median_time(lambda: train_torch(w, Xtr, ytr, dtype=torch.float32))
    rows = [{"implementation": "own scalar engine (Python floats)", "median_s": t_eng,
             "ms_per_epoch": 1e3 * t_eng / EPOCHS, "runs_s": ";".join(f"{t:.3f}" for t in runs_eng)},
            {"implementation": "PyTorch eager float64", "median_s": t_t64,
             "ms_per_epoch": 1e3 * t_t64 / EPOCHS, "runs_s": ";".join(f"{t:.4f}" for t in runs_t64)},
            {"implementation": "PyTorch eager float32", "median_s": t_t32,
             "ms_per_epoch": 1e3 * t_t32 / EPOCHS, "runs_s": ";".join(f"{t:.4f}" for t in runs_t32)}]
    pd.DataFrame(rows).round(5).to_csv(RESULTS / "moons_timing.csv", index=False)
    print(pd.DataFrame(rows)[["implementation", "median_s", "ms_per_epoch"]].round(4).to_string(index=False))
    print(f"engine / torch64 slowdown: {t_eng / t_t64:.0f}x   (engine / torch32: {t_eng / t_t32:.0f}x)")

    # ---- scaling with the number of training samples (one epoch, median of 5 epochs) ----
    scale_rows = []
    for n in (50, 100, 200, 400, 800):
        X, y = make_moons(n_samples=n, noise=0.2, random_state=SEED)
        model, inputs = EngineMLP(w), to_values(X)
        e, _ = median_time(lambda: model.train_epoch(inputs, y, LR))
        e_ng, _ = median_time_nogc(lambda: model.train_epoch(inputs, y, LR))
        net_n = make_torch(w, torch.float64)
        Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.float64).unsqueeze(1)
        opt, lf = torch.optim.SGD(net_n.parameters(), lr=LR), torch.nn.BCELoss()

        def step():
            opt.zero_grad(); lf(net_n(Xt), yt).backward(); opt.step()
        t, _ = median_time(step)
        scale_rows.append({"n_samples": n, "engine_ms_per_epoch": 1e3 * e, "engine_nogc_ms_per_epoch": 1e3 * e_ng, "torch64_ms_per_epoch": 1e3 * t,
                           "slowdown": e / t, "slowdown_nogc": e_ng / t})
    sdf = pd.DataFrame(scale_rows).round(3)
    sdf.to_csv(RESULTS / "moons_scaling.csv", index=False)
    print(sdf.to_string(index=False))

    # node count of one training graph
    inputs = to_values(Xtr)
    loss = EngineMLP(w).loss(inputs, ytr)
    n_nodes = len(loss.topological_order())
    equiv["graph_nodes_per_epoch"] = n_nodes
    equiv["graph_nodes_per_sample"] = round(n_nodes / len(Xtr), 1)
    equiv["timing"] = {r["implementation"]: {"median_s": r["median_s"], "runs_s": r["runs_s"]} for r in rows}
    equiv["slowdown_engine_vs_torch64"] = t_eng / t_t64
    equiv["slowdown_engine_vs_torch32"] = t_eng / t_t32
    (RESULTS / "moons_results.json").write_text(json.dumps(equiv, indent=2) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(eng_losses, label="own engine", lw=3, alpha=0.5)
    ax[0].plot(tor_losses, "--", label="PyTorch float64", lw=1.2)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training BCE"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title("Same training run, two implementations")
    ax[1].semilogy(np.abs(np.array(eng_losses) - np.array(tor_losses)) + 1e-18)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("|loss difference|"); ax[1].grid(alpha=0.3)
    ax[1].set_title("Difference between the two loss curves")
    fig.tight_layout(); fig.savefig(RESULTS / "moons_training.png", dpi=150)


if __name__ == "__main__":
    main()
