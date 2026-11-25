# Lab 6 – Training pipelines and project proposal

Model throughout: the Lab 3 best model, `cnn_matched` (CNN with 64/128 channels, 137,226 parameters) on Fashion-MNIST, 55k train / 5k validation / 10k test. Everything ran on CPU on a shared 2-core VM (torch 2.3.1, 2 threads unless stated), so absolute timings are not portable; ratios and trends are what matter. All numbers are single-seed. Code: `lab06/src/`, configs: `lab06/configs/`, tests: `lab06/tests/` (9 pass), raw outputs: `lab06/results/`, MLflow runs: `lab06/mlruns` (19 runs, table in `results/mlflow_runs.csv`).

## 1. YAML-driven pipeline (`src/train.py`, `configs/base.yaml`)

* One YAML config (merged over `base.yaml`, any key overridable with `--set a.b=value`, unknown keys are rejected) drives seed, data, model, optimizer, scheduler, accumulation, checkpointing and MLflow.
* Per epoch: checkpoint (`last.pt` with model, optimizer, scheduler, history, config hash, MLflow run id) and metrics (train/val loss and accuracy, learning rate, epoch time, throughput, time per phase). At the end: test accuracy once, `summary.json`, `history.csv`. MLflow gets all params, the metrics per epoch, and the config, history, summary and checkpoint as artifacts.
* Reproducibility: the seed fixes initialisation and the train/val split; shuffling is a function of (seed, epoch). Running the same config twice (`item1/base` and `batch/bs128`) gives identical results (test accuracy 0.9127 both). The default config reproduces Lab 3's setup (Adam 1e-3, batch 128, 5 epochs); test accuracy is 91.27 % against 91.82 % in Lab 3. The difference is expected single-seed noise plus a different shuffling order (Lab 3 used one generator for all epochs).
* Tests cover config overrides and typo detection, config hashing, the 137,226 parameter count, gradient accumulation equal to a large batch, resume equal to an uninterrupted run, refusing to resume with a changed config, MLflow content, and the 55k/5k split.

## 2. Profile of one epoch (`src/profile_epoch.py`, `results/profile_*`)

Batch 128, 430 steps, `num_workers=0`. Profiling every step ran out of memory (the per-sample data transform produces hundreds of thousands of tiny events), so `torch.profiler` records 10 steps out of every 20 across the epoch and the windows are aggregated; the phase timers cover all steps.

| phase | seconds (whole epoch) | share (timers) | share (profiler windows) |
|---|---|---|---|
| data loading | 2.0 | 4.7 % | 5.8 % |
| forward | 20.7 | 48.2 % | 47.2 % |
| backward | 19.7 | 46.0 % | 45.8 % |
| optimizer step | 0.5 | 1.1 % | 1.2 % |

Compute (forward + backward) is about 94 % of the epoch. Within it, `convolution_backward` takes 33 % of profiled CPU time, `max_pool2d` forward 25 % and the forward convolution 18 % (`results/profile_top_ops.csv`); max-pooling is surprisingly expensive on this CPU.

## 3. `num_workers` in [0, 1, 2, 4, 8] (`results/workers_study.*`)

Two epochs per setting; epoch 2 (after warm-up) is reported.

| workers | epoch 2 (s) | samples/s | main thread waiting for data (s, %) |
|---|---|---|---|
| 0 | 40.6 | 1355 | 1.5 (3.7 %) |
| 1 | 41.3 | 1330 | 0.7 (1.7 %) |
| 2 | 41.2 | 1336 | 0.9 (2.1 %) |
| 4 | 40.2 | 1367 | 1.0 (2.4 %) |
| 8 | 42.0 | 1311 | 1.1 (2.6 %) |

**Bottleneck: compute, not data loading.** The data pipeline alone runs at about 42,000 samples/s with no workers (1.3 s per epoch), 30 times faster than the training step can consume (a compute-only benchmark with a cached batch gives 1,317 samples/s, which is what full training achieves). Workers therefore change nothing, and in the loader-only test extra workers are even slower (1 worker 18k, 2 workers 27k, 8 workers 13k samples/s) because inter-process transfer costs more than the tiny per-sample transform, and on 2 cores the workers compete with the 2 compute threads. Adding random flip/shift augmentation halves the loader's speed (19k samples/s) but it is still far above the training speed. Results are identical for every worker count (same test accuracy 0.8935), so workers do not change the data order. `num_workers > 0` would matter if the per-sample work were much heavier (JPEG decoding, large augmentations) or the model much faster (GPU).

## 4. Batch-size study (`results/batch_study.*`), 5 epochs, fixed lr 1e-3

| batch | steps/epoch | samples/s | peak RSS (MB) | activations kept for backward (MB, estimate) | val acc | test acc | final train loss |
|---|---|---|---|---|---|---|---|
| 32 | 1719 | 1399 | 968 | 21 | 91.96 | 91.50 | 0.161 |
| 64 | 860 | 1356 | 992 | 43 | 91.96 | 91.71 | 0.165 |
| 128 | 430 | 1298 | 1016 | 86 | 91.76 | 91.27 | 0.180 |
| 256 | 215 | 1268 | 1139 | 172 | 91.56 | 91.28 | 0.208 |
| 512 | 108 | 1128 | 1293 | 343 | 90.48 | 89.98 | 0.239 |

* Throughput falls with batch size on this CPU (1399 → 1128 samples/s), the opposite of the usual GPU behaviour: with no launch overhead to amortise, larger batches only add cache pressure.
* Memory: the peak RSS has a fixed part of about 950 MB (libraries, dataset, optimizer) and grows with the batch; from 32 to 512 it rises by 325 MB, matching the estimated growth of the activations (322 MB).
* Accuracy: the learning rate was not rescaled, so a larger batch means fewer updates in 5 epochs. Batch 512 is clearly behind (train loss 0.239 against 0.161 at batch 32: it is under-trained, not overfitting). The differences among 32–256 (up to about 0.4 points) are within what I would expect from seed noise, which I did not measure.

## 5. Schedules and optimizers, 10 epochs each, 1 thread (`results/sched_study.*`)

SGD with momentum 0.9, lr 0.05, with each schedule (StepLR step 3 / gamma 0.5; CosineAnnealingLR T_max 10; ReduceLROnPlateau factor 0.5 / patience 1 on validation loss) and without, then Adam at 1e-3 constant. The learning rates were chosen in advance and not tuned for any method.

| run | final lr | train loss | val loss | val acc | test acc |
|---|---|---|---|---|---|
| SGD+mom, constant | 0.05 | 0.130 | 0.270 | 91.52 | 90.84 |
| SGD+mom, StepLR | 0.0062 | 0.097 | 0.215 | 92.38 | 92.26 |
| SGD+mom, CosineAnnealingLR | 0.0012 | 0.089 | 0.212 | 92.46 | 92.29 |
| SGD+mom, ReduceLROnPlateau | 0.025 | 0.091 | 0.241 | 92.24 | 91.92 |
| Adam, constant | 0.001 | 0.100 | 0.228 | 92.18 | 92.05 |

* Any decay beats the constant rate: with constant lr 0.05 the validation loss rises again after epoch 8 (0.24 → 0.27). Cosine and StepLR end with the lowest validation loss; Cosine is the smoothest.
* ReduceLROnPlateau reacted late: it halved the rate once, after epoch 8, so most of the run used the high rate. With a 10-epoch budget and patience 1 it has little time to act; it suits long runs where the right decay time is unknown.
* The ranking among Cosine, StepLR, Adam and Plateau is within about 0.4 points on one seed, so I would not claim a winner among them. Adam at a constant rate matched the best decayed SGD runs without a schedule; SGD needed a schedule to reach the same level.

## 6. Gradient accumulation (`results/accum_study.*`)

Batch 64 × 4 accumulation steps against batch 256, 5 epochs, same seed. Each micro-batch loss is divided by the number of samples in its accumulation group, so one accumulated step equals one step on the 256 samples (a unit test checks this, including a short last group).

| | train loss epoch 5 | val acc epoch 5 | test acc | samples/s | peak RSS (MB) |
|---|---|---|---|---|---|
| batch 256 | 0.2075 | 91.56 | 91.28 | 1268 | 1139 |
| batch 64 × 4 | 0.2075 | 91.58 | 91.19 | 1345 | 972 |

The curves coincide: the largest per-epoch difference is 8.6e-5 in training loss and 0.14 points in validation accuracy (floating-point order of summation). Plain batch 64 without accumulation behaves differently (val 91.96), as expected since it makes 4 times more updates. Accumulation also saves memory (peak RSS 167 MB lower) at no cost in speed here.

## 7. Interrupt and resume (`results/resume_study.*`)

The config `run.interrupt_after_epoch=3` stops the run right after the epoch-3 checkpoint is written (MLflow status KILLED); a second invocation with `run.resume=<checkpoint>` continues from epoch 4 in the same MLflow run. The resumed curves are identical to the uninterrupted run (item 1): maximum difference 0.0 in training loss, validation loss and validation accuracy, and the same test accuracy (91.27 %). This works because the optimizer and scheduler state are saved and the data order depends only on (seed, epoch). The pipeline refuses to resume a checkpoint made with a different training config.

## 8. Project proposal

`project/proposal.md`: tiny spoken-language identification for Malaysian speech (Malay, English, Mandarin, Tamil, Indonesian) on a streamed subset of the 14.7M-row Malaysian-STT-Whisper dataset, distilled from a Whisper teacher and quantized to ≤ 5 MB and ≤ 50 ms per 3 s clip on CPU. **Draft:** the topic is my suggestion, and the student name and ID are placeholders that need your details before you commit it and get it approved.

## Caveats

* Single seed everywhere; the VM is shared and has 2 cores. Throughput numbers move by a few percent between identical runs.
* Peak memory is process-level RSS (Linux `ru_maxrss`), which includes about 950 MB of fixed overhead; activations are an estimate from layer output sizes.
* The profiler windows cover half of the steps because profiling all of them ran out of memory.
* The scheduler/optimizer comparison uses untuned learning rates; a different SGD rate could change the ranking.
* The reading (Reddi Ch. 8, §5.5, §13.3) was not used for these numbers; connect the results to it in your own words, and rewrite the explanations in this report in your own words before submitting.
