# Individual project proposal

**Student:** Adam Mikael Bin Aris Abdillah  **Student ID:** 72866  **Course:** ML Systems  **Date:** 19 November 2026

## Title
**High-throughput edge classifier for tropical land-use and deforestation patches in Malaysia: compressing a Sentinel-2 patch classifier to ≤ 5 MB for CPU-only inference**

## Problem
Malaysia's forests are under steady pressure from oil palm and other agricultural expansion, and monitoring agencies, plantation auditors and NGOs need to scan very large areas for forest loss and land-use change. Sentinel-2 covers the whole country every few days, so the bottleneck is not imagery but processing: tiles are cut into millions of small patches that must be classified cheaply, ideally on a laptop, a field device or a low-cost server without a GPU. Tropical scenes are hard because of persistent cloud, and because oil palm, rubber, secondary forest and logged forest look alike at 10 m resolution. The project asks: **how small and fast can a land-use / forest-loss patch classifier be made for Malaysia, on CPU, before accuracy (especially oil palm vs. natural forest and "recent clearing") breaks down, and which compression step costs the most?**

## Dataset
Images and labels are built from open sources, so the dataset is assembled rather than downloaded as one file:
- **Imagery:** Sentinel-2 L2A (free and open), cut into 64 × 64 px patches (640 m) with 4–10 bands. Cloud-filtered annual or seasonal median composites to reduce cloud contamination.
- **Land-cover labels (weak, map-derived):** a 10 m land-cover map such as ESA WorldCover or the Esri/Impact Observatory Sentinel-2 land-use/land-cover product; a nationwide 10 m oil palm map for Malaysia and Indonesia (2022–2024, CC-BY 4.0, [Zenodo DOI 10.5281/zenodo.17768444](https://doi.org/10.5281/zenodo.17768444)) to separate oil palm from other tree cover; and Hansen Global Forest Change (loss year) to mark "recent forest loss".
- **Classes (about 7):** natural forest, oil palm, mangrove / wetland, other cropland and shrub, built-up, water, recent clearing.
- **Size:** a class-balanced sample of about 200–500 k patches across Peninsular Malaysia, Sabah and Sarawak. EuroSAT and BigEarthNet are European only, so they are not used as test data (at most for pretraining).

**To verify in week 1:** (a) the licence and attribution terms of each label product and the Sentinel-2 access route (I would use everything for coursework only and cite each source); (b) label quality: labels come from maps, not from ground truth, so I will audit a random sample of about 300 patches against high-resolution imagery and report per-class label agreement; (c) the split: patches are assigned to spatial blocks (for example 10 × 10 km tiles) and whole blocks go to train, validation or test, with one region (for example part of Sarawak) held out entirely, to avoid spatial leakage.

## Success metric
- **Primary:** macro-F1 over the 7 classes on a spatially held-out, manually audited test subset, with a target **≥ 85 %**. Because the labels are noisy, the ceiling is set by label agreement, so I report this next to the audited label accuracy, not as a claim about ground truth.
- **Hard pairs, reported separately:** F1 for oil palm vs. natural forest, and recall for recent clearing.
- Accuracy no more than **2 points** below the uncompressed teacher after compression.
- Also reported: model size, throughput (patches/s), p50/p95 latency, accuracy drop per optimization step, mean ± std over 3 seeds; differences under about 0.5 points are not claimed as real.

## Deployment constraints
- **Latency ≤ 50 ms** per patch on CPU, including normalization and band stacking, batch 1, 1 thread, median and p95 of 200 runs after warm-up.
- **Throughput:** target ≥ 500 patches/s at batch 64 on 4 CPU threads, so a 100 × 100 km area (about 15 k patches at 640 m, no overlap) is classified in well under a minute.
- **Model size ≤ 5 MB** (serialized file).
- No GPU and no network at inference time.

## Planned optimization
1. **Teacher / baseline:** a ResNet-style model (for example ResNet-18 width-scaled for 64 × 64 inputs) trained directly on the labels; it also serves as the accuracy reference.
2. **Student and distillation (main method):** a small depthwise-separable CNN (about 0.3–1 M parameters) trained on the teacher's soft class probabilities plus the hard labels; temperature and mixing weight tuned on validation only.
3. **Quantization:** int8 post-training quantization, with quantization-aware training if the drop exceeds the budget.
4. **Pruning (extension):** structured channel pruning of the student, reporting the size / latency / accuracy frontier.
5. **Robustness check:** evaluate on a held-out region, on cloudy or hazy patches, and on a different year, and report where the model fails (for example young oil palm vs. regrowing forest).

The Lab 6 YAML + MLflow pipeline drives all runs so each result traces to a config. The `num_workers` and batch-size findings from Lab 6 guide the data loader: patches are cached as uint8 arrays (memory-mapped) so that loading is cheap, and the study will show whether it still is.

## Feasibility and risks
- **Label noise** (map-derived, with errors between maps): mitigated by the manual audit, by using teacher soft labels, and by reporting results on the audited subset.
- **Class overlap** (oil palm vs. forest vs. rubber): reported as its own metric, and "other tree crops" may be merged if labels are unreliable.
- **Cloud cover:** mitigated by compositing; residual clouds are part of the robustness test.
- **Compute and data volume:** a first model on 20 k patches by week 2; imagery cut once and cached.
- **Novelty:** I make no claim that land-cover classification on Sentinel-2 is new; the contribution is a careful size / latency / accuracy study for Malaysian land use under a fixed edge budget, with the oil palm vs. forest case analysed.

## Timeline
Week 1 data assembly, label audit and spatial splits; week 2 baselines and teacher; week 3 distillation; week 4 quantization, latency and throughput, ablations, seeds, write-up.
