# Lab 5 – Frameworks and automatic differentiation

Environment: Python 3.11, torch 2.3.1 (CPU, 1 thread), numpy 1.26.4, scikit-learn 1.5.0. Timings come from a shared 2-core VM, so treat them as indicative (medians are reported, the raw runs are in the CSVs).
Code: `src/autograd.py` (engine), `tests/test_autograd.py`, `src/verify.py`, `src/moons.py`, `src/inspect_graph.py`, `src/compile_bench.py`. Raw outputs: `results/`.

## 1–3. The engine (`src/autograd.py`)

* `Value` stores `data`, `grad` (starts at 0.0), its parent nodes `_prev`, the op name, and a `_backward` closure holding that op's local derivative rule.
* `+`, `*`, `**` (constant int/float exponent), `relu` build the graph during the forward pass. Extras for the model: `-`, `/`, `exp`, `log`, `sigmoid`, `value_sum`.
* `backward()`: topological order (iterative DFS post-order, so deep graphs do not hit the recursion limit), set output grad to 1.0, walk in reverse order and call each `_backward`. Every rule uses `+=`, so nodes used more than once (`x*x`, diamond graphs) get all contributions summed. `backward()` does not zero gradients (like PyTorch); `zero_grad` does.
* 44 tests pass: graph structure, accumulation, same node used twice, diamond graph, seed of 1, repeated backward, ReLU at 0, error cases, 25 random DAGs compared with PyTorch, a 20 000-node chain.
* Deviation to know about: `log` evaluates at `max(x, 1e-12)` to avoid `log(0)`. It differs from `torch.log` only for arguments below 1e-12 (covered by a test).

## 4. Verification on `((a*b + c)**2).relu().sum()` (200 random inputs, length-5 vectors)

| comparison | worst max abs gradient difference | < 1e-6? |
|---|---|---|
| engine vs PyTorch float64 | 0.0 | yes |
| engine vs central differences (eps 1e-5) | 9.8e-10 | yes |
| engine vs PyTorch float32 | 1.8e-6 | no – float32 rounding (about 1e-7 relative), not an engine error |

The forward value differs from PyTorch float64 by at most 7e-15.

## 5. Training 2→8→1 on `make_moons` (`results/moons_*`)

Setup: `make_moons(300, noise=0.2)`, 200 train / 100 test, 200 epochs of full-batch SGD (lr 0.5), ReLU hidden layer (the manual does not specify the hidden activation; this is my choice), sigmoid output, mean BCE, identical initial weights in both implementations.

* Same computation: loss curves differ by at most 4e-16 over all epochs and final parameters by 3e-16 (PyTorch in float64). Final loss 0.247, train accuracy 0.89, test accuracy 0.87 (majority baseline 0.50), identical for both. See `moons_training.png`.
* Speed, whole 200-epoch run, 1 untimed warm-up then median of 5:

| implementation | median s | ms / epoch |
|---|---|---|
| own scalar engine | 37.1 | 185 |
| PyTorch eager float64 | 0.053 | 0.26 |
| PyTorch eager float32 | 0.053 | 0.27 |

PyTorch is about 700x faster. One epoch builds 13 034 graph nodes (65 per sample) as Python objects, each op is a Python call and a closure; PyTorch handles the whole batch in a few vectorised kernels (~10 ops regardless of batch size). float32 and float64 are equal here because the tensors are tiny and per-call overhead dominates.

* Scaling (ms per epoch; `moons_scaling.csv`): engine time grows with the number of samples, PyTorch's barely does.

| samples | engine | engine, GC off (diagnostic) | PyTorch |
|---|---|---|---|
| 50 | 5.4 | 3.3 | 0.33 |
| 100 | 13.9 | 7.5 | 0.35 |
| 200 | 175 | 13.3 | 0.28 |
| 400 | 395 | 28.3 | 0.40 |
| 800 | 920 | 65.2 | 0.62 |

The jump between 100 and 200 samples is Python's cyclic garbage collector: nodes and their backward closures form reference cycles, so full collections repeatedly scan tens of thousands of live objects. With GC paused during the epoch (garbage collected outside the timed region), the engine is about 47x slower than PyTorch at 200 samples instead of about 600x. GC cannot simply be disabled for a full run: it leaked about 6 GB and the process was killed.

## 6. PyTorch graph, tensors, compilation (`results/graph_inspection.txt`, `compile_*.csv`, `contiguity_*.csv`)

**Graph.** For `BCELoss(model(x), y)` the autograd graph recorded in the forward pass is `BinaryCrossEntropyBackward0 → SigmoidBackward0 → AddmmBackward0 → ReluBackward0 → AddmmBackward0`, with `TBackward0` (weight transposes) and `AccumulateGrad` leaves for the 4 parameters. This is the same structure as my `Value` graph, but with one node per tensor operation instead of per scalar. `torch.fx` and `jit.trace` show the model as 4 module calls (Linear, ReLU, Linear, Sigmoid); Dynamo captured 1 graph with 0 graph breaks.

**Tensors.** Parameters and activations are float32, contiguous, row-major (weight (8,2) stride (2,1), activation (4,8) stride (8,1)); activations carry `grad_fn`, parameters are leaves with `requires_grad`.

**Views.** `t()`, `permute`, `[:, ::2]`, `[:, :2]` and `expand` are views sharing storage with the base but are non-contiguous (strides (1,4), (4,2), (4,1), (0,1)). Row slices stay contiguous. `.contiguous()` and `reshape` on a non-contiguous tensor copy; `.view(12)` on `t()` raises an error.

**Eager vs compiled** (1 thread, after warm-up, median of 15 batches; `compile_comparison.csv`):

| case | eager | jit.trace | torch.compile | compile time (first call) |
|---|---|---|---|---|
| tiny MLP 2-8-1, batch 256 | 25.4 µs | 18.8 µs (1.35x) | 37.2 µs (0.68x) | 9.4 s |
| MLP 256-512-512-10, batch 128 | 903 µs | 889 µs (1.02x) | 1480 µs (0.61x) | 2.6 s |
| elementwise chain, 1M floats | 5.03 ms | 4.89 ms (1.03x) | 3.19 ms (1.58x) | 2.0 s |
| training step (fwd+bwd+SGD), moons-sized | 414 µs | – | 751 µs (0.55x) | 5.6 s |

All variants reproduce the eager output exactly (max abs error 0.0 in every case).

**When does compilation help?** Compilation pays off when a function contains several cheap, memory-bound operations that can be fused into one kernel (the elementwise chain: 1.6x), so intermediate tensors are not written to and read from memory. It does not help when the time is dominated by already-optimised library calls such as large matmuls (MLP: no gain), or when the model is so small that per-call overhead dominates; in that case the compiled wrapper's guard checks and dispatch were slower than eager here (0.6–0.7x). Compilation also costs seconds on the first call, so it only pays back if the compiled function is called many times with stable shapes. `jit.trace` removes some Python overhead for the tiny model (1.35x) but does no fusion on CPU here.

**Non-contiguous layouts** (`contiguity_timing.csv`, float32, 1 thread):

| operation | 256×256 (cache-resident) | 4096×4096 (64 MB) |
|---|---|---|
| `a + b` (contiguous) | 11.7 µs | 15.0 ms |
| `a.t() + b` | 91 µs (7.8x) | 132 ms (8.8x) |
| `a.t() + b.t()` | 82 µs (7.0x) | 236 ms (15.8x) |
| `a.t().contiguous() + b` | 59 µs (5.0x) | 70 ms (4.7x) |
| `a[:, ::2].sum()` | 22 µs | 5.4 ms (vs 2.6 ms for the full contiguous `a.sum()`) |
| `a @ b` vs `a.t() @ b` | 260 µs vs 241 µs | – |

A transposed view is free to create but the following element-wise op reads memory with a large stride, which defeats vectorisation and cache lines: 8–16x slower on large arrays. Paying for one `.contiguous()` copy first is cheaper (about 4.7x), and worth it if the tensor is reused. Matrix multiply is not affected, because the BLAS backend handles transposed operands natively. A strided slice reads half the elements but is still slower per element than a contiguous sum.

## Caveats

* Timings were taken on a shared 2-core VM; the ratios are meaningful, the absolute numbers are not portable. The machine used torch with a CUDA build on CPU only.
* Hidden activation (ReLU) and learning rate were my choices; I did not tune them on test accuracy.
* The "when compilation helps" and speed explanations are my draft; rewrite them in your own words.
* Reading (Reddi Ch. 7 §7.3, 7.4, 7.6–7.8) was not used to produce these numbers; link the concepts (graph construction, fusion, eager vs graph execution) yourself.
