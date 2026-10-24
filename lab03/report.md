<img width="2110" height="3000" alt="CamScanner 24-10-2026 16 35_2" src="https://github.com/user-attachments/assets/45039fcd-09fb-434f-81ad-b4e402995cd2" /># Lab 3: Architectures and their costs

Code is in `src/` (`models.py`, `data.py`, `cost.py`, `train.py`, `plots.py`), unit tests in `tests/` (`python -m pytest lab03/tests`), raw outputs in `results/`. Everything ran on CPU on a shared 2-core cloud VM (torch 2.3.1); latency was measured with 1 thread, training used 2.

## 1. Data (item 1)

`torchvision.datasets.FashionMNIST` with `ToTensor()` and `Normalize((0.2860,), (0.3530,))`. The official 60k training set is split 55k train / 5k validation (`random_split`, seed 42), the official 10k test set is used for the final test accuracy, batch size 128.

The workspace could not reach torchvision's default mirror, so the four dataset files were downloaded from the dataset's GitHub repository (`zalandoresearch/fashion-mnist`) and each file's MD5 was checked against the checksums built into torchvision (all four matched). On your machine, `download=True` should work as usual.

## 2. Models (item 2)

The three architectures are implemented exactly as written in the manual. The manual also asks for "similar size, target 100k-150k parameters", but its own layer specifications give:

| Model | Parameters (torchinfo) | Hand count | In 100k-150k? |
|---|---|---|---|
| MLP 784-256-128-10 | 235,146 | 235,146 | no (too large) |
| CNN 16/32 channels | 20,490 | 20,490 | no (too small) |
| Patch attention, d=64, ff=128, 2 layers | 70,794 | 70,794 | no (too small) |

So the specification contradicts the size target. I kept the specified models as the main result, because the graded evidence refers to those layers, and added a clearly labelled supplement so the "similar size" comparison can still be made (hollow markers in the plots):

| Supplement model | Change | Parameters |
|---|---|---|
| `mlp_matched` | hidden widths 128, 64 | 109,386 |
| `cnn_matched` | 64 and 128 channels | 137,226 |
| `attention_matched` | dim_feedforward 256 | 103,818 |
| `attention_pos` | spec model plus learned positional embedding (16x64) | 71,818 |

**Hand parameter counts** (weights + biases): MLP = (784x256+256) + (256x128+128) + (128x10+10). CNN = (1x16x9+16) + (16x32x9+32) + (1568x10+10). Attention = patch projection (49x64+64 = 3,200) + 2 x encoder layer + head (64x10+10 = 650); one encoder layer = attention input projection (3x64x64+192 = 12,480) + output projection (64x64+64 = 4,160) + feed-forward (64x128+128 = 8,320 and 128x64+64 = 8,256) + two LayerNorms (256) = 33,472. These match `torchinfo.summary` and are asserted in `tests/test_models.py`.

**Choices the manual leaves open:**
- The image is cut into a 4x4 grid of 7x7 patches (sequence length 16), each flattened to 49 values.
- The specified attention model has no positional information. With mean pooling it cannot tell where in the image a patch came from, only what it contains. That is why I added `attention_pos`.
- Transformer dropout is set to 0 (PyTorch's default is 0.1) so the three families are trained alike. Layers use PyTorch's defaults otherwise (post-norm, ReLU, batch-first).

## 3. Cost (item 3)

<img width="2146" height="3000" alt="CamScanner 24-10-2026 16 35_1" src="https://github.com/user-attachments/assets/d921cbe6-f1a7-45ae-b667-a221c842ec5b" />
<img width="2110" height="3000" alt="CamScanner 24-10-2026 16 35_2" src="https://github.com/user-attachments/assets/10391a9b-9c46-4aff-9407-0f3cae088a54" />

### 3.1 FLOPs by hand

One multiply-add counts as 2 FLOPs, per sample, counting only matrix products and convolutions. Biases, ReLU, pooling, softmax, LayerNorm and residual additions are left out.

| Layer | Formula | FLOPs |
|---|---|---|
| Linear 784 -> 256 (MLP) | 2 x 784 x 256 | 401,408 |
| Conv2d 16 -> 32, 3x3, 14x14 output (CNN, 2nd conv) | 2 x 3 x 3 x 16 x 32 x 14 x 14 | 1,806,336 |
| Attention: Q, K, V projections | 2 x 16 x 64 x (3 x 64) | 393,216 |
| Attention: scores Q K^T | 2 x 16^2 x 64 | 32,768 |
| Attention: weights x V | 2 x 16^2 x 64 | 32,768 |
| Attention: output projection | 2 x 16 x 64 x 64 | 131,072 |
| Feed-forward (two linear layers) | 2 x 16 x 64 x 128 x 2 | 524,288 |
| **Encoder layer total** | | **1,114,112** |

For whole models: MLP = 2 x (784x256 + 256x128 + 128x10) = **469,504**. CNN = first conv 225,792 + second conv 1,806,336 + linear 31,360 = **2,063,488**. Attention = patch projection 100,352 + 2 x 1,114,112 + head 1,280 = **2,329,856**.

### 3.2 Hand vs tools

thop reports multiply-accumulates, so its value is doubled here. "FlopCounter" is PyTorch's own `torch.utils.flop_counter`. Raw numbers: `results/cost.csv`, `results/layer_flops.csv`.

| Model | Hand | thop (default) | thop (fast path off) | FlopCounter (default) | FlopCounter (matmul attention) |
|---|---|---|---|---|---|
| MLP | 469,504 | 469,504 | 469,504 | 469,504 | 469,504 |
| CNN | 2,063,488 | 2,063,488 | 2,063,488 | 2,063,488 | 2,063,488 |
| Attention | 2,329,856 | **101,632** | 1,182,976 | 2,198,784 | 2,329,856 |

For the MLP and CNN, hand calculation and every tool agree exactly (the tools do not count biases, ReLU or pooling either, so there is nothing to explain).

For attention there is a large gap, and I tested each explanation instead of assuming it:

1. **thop reports 101,632 (4% of the hand value).** That is exactly 2 x (50,176 + 640): the patch-projection linear layer and the final head. The whole encoder counts as zero. The cause: `thop.profile` always switches the model to eval mode and runs it without gradients, and in that state PyTorch executes a fused inference "fast path" for transformer layers that never calls the submodules thop attaches its counters to. With the fast path switched off (`torch.backends.mha.set_fastpath_enabled(False)`), a single encoder layer goes from 0 to 540,672 FLOPs.
2. **Even then, thop (1,182,976) misses about half.** It now counts the feed-forward layers (524,288) plus the LayerNorms (16,384 per layer, which my hand count excludes), but not the Q/K/V and output projections or the two attention products, which PyTorch computes inside `nn.MultiheadAttention` rather than through layers thop recognises. Missing per layer: 393,216 + 131,072 + 32,768 + 32,768 = 589,824, i.e. 1,179,648 for two layers; after subtracting the LayerNorm FLOPs thop adds (32,768), the remaining gap to my hand value is exactly 1,146,880.
3. **FlopCounter (default) is 131,072 below hand.** That is exactly 2 layers x 65,536 = the two attention products (scores and weights x V). The CPU attention kernel PyTorch uses by default is not one the counter knows. When I force the explicit matrix-multiply implementation of attention (`sdpa_kernel(SDPBackend.MATH)`), FlopCounter matches the hand value exactly (2,329,856).

Takeaway: the hand formulas are the reference here, and thop's number for attention models should not be used without checking. `torchinfo`'s own "mult-adds" figure has the same problem for attention (it reports only 3,850).

### 3.3 Latency (item 3, last bullet)

Protocol: eval mode, `torch.inference_mode()`, 1 thread, real test images, 10 warm-up calls for batch 1 and 3 for batch 128, then median of 100 runs (batch 1) or 20 runs (batch 128). I ran this whole protocol three times and report the median of the three medians; the three repetitions are in `cost.csv`.

| Model | Batch 1 (ms) | Batch 128 (ms) | Batch 128 per sample (us) | Effective GFLOP/s at batch 128 |
|---|---|---|---|---|
| mlp | 0.064 | 0.710 | 5.5 | 84.7 |
| cnn | 0.146 | 20.24 | 158.2 | 13.0 |
| attention | 0.262 | 4.67 | 36.5 | 63.9 |
| attention_pos | 0.256 | 4.59 | 35.9 | 64.9 |
| mlp_matched | 0.032 | 0.204 | 1.6 | 137.1 |
| cnn_matched | 0.624 | 77.37 | 604.5 | 49.5 |
| attention_matched | 0.296 | 6.15 | 48.0 | 70.3 |

Latency is noisy on this shared VM. Within one run the three repetitions mostly agree within a few percent, but there are outliers (CNN batch 1: 0.245, 0.146, 0.140 ms; attention batch 1: 0.262, 0.454, 0.238 ms), and separate runs of the whole script differed by up to about 50% in absolute terms (MLP batch 128: 0.46 ms in an earlier run, 0.71 ms in this one). The ordering of the families was the same in every run, with one exception: at batch 1, the spec CNN and the spec attention model are close (0.15 vs 0.26 ms here, 0.22 vs 0.24 ms in an earlier run).

## 4. Training (item 4)

Adam, learning rate 1e-3, batch 128, 5 epochs, seed 42 (same initialisation seed and same data order for all models). The test set is evaluated once, after epoch 5, with no checkpoint selection. Per-epoch logs: `results/training_log.csv`.

| Model | Parameters | Final train loss | Val acc | **Test acc** | Train time (s) |
|---|---|---|---|---|---|
| mlp | 235,146 | 0.268 | 0.878 | **0.8773** | 35 |
| cnn | 20,490 | 0.263 | 0.898 | **0.8925** | 67 |
| attention | 70,794 | 0.389 | 0.849 | **0.8430** | 65 |
| attention_pos (suppl.) | 71,818 | 0.342 | 0.869 | 0.8640 | 64 |
| mlp_matched (suppl.) | 109,386 | 0.283 | 0.881 | 0.8765 | 35 |
| cnn_matched (suppl.) | 137,226 | 0.181 | 0.919 | 0.9182 | 266 |
| attention_matched (suppl.) | 103,818 | 0.381 | 0.850 | 0.8461 | 73 |

These are single-seed results after only 5 epochs, so differences of a few tenths of a point (for example `mlp` vs `mlp_matched`, or `attention` vs `attention_matched`) should not be read as real. I did not measure seed-to-seed variation. The attention models were still improving at epoch 5 and were behind the other families at every epoch.

## 5. Trade-offs (item 5)

Plots: `results/acc_vs_params.png` and `results/acc_vs_latency.png` (batch 1 and batch 128 per sample). Filled markers are the specified models, hollow markers the supplement. Summary table: `results/summary.csv`.

What the plots show:
- **Accuracy vs parameters:** parameter count does not predict accuracy across families. The CNN with the fewest parameters (20k) beats the MLP with the most (235k) by 1.5 points, and the size-matched CNN is the best model overall at 91.8%.
- **Accuracy vs latency:** the MLPs are fastest by a wide margin, the CNNs are the most accurate but the slowest, and attention sits at middle latency with the lowest accuracy.

**Which is most cost-effective?** It depends on which cost matters:
- **Per parameter (memory footprint):** the CNN. 89.3% with 20k parameters (0.08 MB of weights) against 87.7% with 235k (0.94 MB) for the MLP.
- **Per unit of latency or compute:** the MLP. At batch 128 it is about 29 times faster per sample than the CNN (5.5 vs 158 us) for 1.5 points less accuracy.
- **Attention is not on the efficient frontier for this task.** Under the specification it is less accurate than the MLP (84.3% vs 87.7%) and also slower at both batch sizes. Adding positional embeddings helps (86.4%) but does not change that picture.

**Why (access patterns and compute, using the structure of each model; link these to the Ch. 4.6-4.7 terms from your reading):**
- **MLP:** dense matrix products with contiguous, regular memory access. Each weight is used once per sample, so at batch 1 the cost is mostly reading ~0.94 MB of weights (about 7 GFLOP/s effective); at batch 128 each weight is reused across the batch and throughput rises to about 85 GFLOP/s (about 137 for the smaller MLP). The batch-128 run is only 11x slower than batch 1 for 128x the work.
- **CNN:** weight sharing makes the model tiny (about 100 FLOPs per parameter, against 2 for the MLP), so it is compute-heavy for its size. Each filter slides over the image and the early layers produce large activation maps (16 x 28 x 28 = 12.5k values per sample) that must be written and read again by ReLU and pooling. With few channels (1 -> 16 -> 32) the convolutions use the CPU poorly: 13 GFLOP/s at batch 128, against about 50 for the 64/128-channel version. Batch 128 takes 139x as long as batch 1, so there is little to amortise.
- **Attention:** all 16 patches interact, so the score matrix is only 16 x 16 per head and costs little here. In this model about 94% of the FLOPs of an encoder layer are in the Q/K/V, output and feed-forward linear layers, so it behaves like a stack of dense matrix products, with extra small operations (softmax, LayerNorm, residuals, reshapes). At batch 1 its latency roughly tracks its FLOPs (0.262 ms vs 0.064 ms for the MLP, a ratio of 4.1, against a FLOP ratio of 5.0), so it is slower than the MLP mainly because it does about 5 times the work per sample for less accuracy. At batch 128 it runs at about 64 GFLOP/s, below the MLP's 85 but far above the CNN's 13. The all-pairs score matrix grows with the square of the sequence length, so a larger image or smaller patches would shift its cost sharply upward; at 16 tokens that term is negligible.

## 6. Conclusions (draft; rewrite in your own words)

1. FLOP counts from hand formulas agree exactly with thop and PyTorch's counter for MLP and CNN, but thop reports only 4% of the attention model's FLOPs because of the fused inference path and because it ignores the internals of multi-head attention. Tools need checking against hand formulas.
2. On Fashion-MNIST after 5 epochs the convolutional model gives the best accuracy per parameter and the best accuracy overall, the MLP gives the best accuracy per unit of latency, and the small patch-attention model is dominated by the MLP on both axes.
3. Parameters, FLOPs and latency are different costs: the CNN has the fewest parameters but the highest latency of the three specified models, because convolutions reuse each weight many times (high compute per parameter) and move large activation maps.

## Caveats

- One seed, 5 epochs, no tuning; the ranking of close pairs is not established.
- CPU only, shared 2-core VM, noisy timings (see section 3.3); on a different machine the absolute numbers will change and the CNN/attention ordering at batch 1 might change.
- The specified architectures do not meet the 100k-150k size target; the size-matched models are an addition of mine, not part of the manual.
- The Fashion-MNIST files came from GitHub, not torchvision's default mirror (checksums verified).
