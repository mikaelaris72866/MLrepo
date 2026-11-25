# Individual project proposal

**Student:** Adam Mikael Bin Aris Abdillah  **Student ID:** 72866  **Course:** ML Systems  **Date:** 20 November 2026

## Title
**Tiny spoken-language identification for Malaysia's multilingual speech: distilling a Whisper teacher into a ≤ 5 MB on-device model, trained on millions of Malaysian clips**

## Problem
Everyday speech in Malaysia mixes Malay, English, Mandarin, Tamil and the closely related Indonesian, often inside one conversation. Speech products (voice assistants, call-centre routing, captioning, kiosks) must first decide which language they are hearing, and a wrong decision sends the audio to the wrong recognizer. General-purpose language detectors are large, and Malay versus Indonesian is a known hard pair because the languages are very close. The project asks: **how small and fast can a spoken-language identifier for Malaysian speech be made, on-device and on CPU, before accuracy (especially Malay vs Indonesian) breaks down, and which compression step costs the most?**

## Dataset
`mesolitica/Malaysian-STT-Whisper` on Hugging Face: about **14.7 million transcribed audio rows, 139 GB**, in nine splits that include `malaysian_context_v2` (about 5.9 M rows), `imda` (1.6 M), `mandarin` (1.4 M), `tamil` (1.5 M), `indonesian` (0.9 M) and English science speech (0.9 M) ([dataset card](https://huggingface.co/datasets/mesolitica/Malaysian-STT-Whisper)). Language labels would come from the split of origin. Optional second source for a harder test: `malaysia-ai/malaysian-dialects-youtube`, 68,607 hours of Malay-dialect audio without transcripts ([card](https://huggingface.co/datasets/malaysia-ai/malaysian-dialects-youtube/raw/main/README.md)), usable as unlabelled data for distillation because the teacher provides the labels.

The full set is far too large to train on in a semester, so the plan is a streamed, balanced sample of 1–2 M clips of 3 s, with speakers/sources kept disjoint between train, validation and test. **To verify in week 1:** (a) the licence (the card states none; I would use it for coursework only and not redistribute the audio or models trained on it without checking), (b) that split names are reliable language labels (I will manually listen to and check a random sample per split, and drop the noisy ones), (c) whether clips can be assigned to speakers or sources for a leak-free split.

## Success metric
- **Primary:** macro-F1 over the 5 languages ≥ **95 %** on a held-out test set, and **Malay vs Indonesian** accuracy ≥ **90 %** (reported separately because it is the hard pair).
- Accuracy no more than **2 points** below the teacher after compression.
- Also reported: size, p50/p95 latency, accuracy drop per optimization step, mean ± std over 3 seeds; differences under about 0.5 points are not claimed as real.

## Deployment constraints
- **Latency ≤ 50 ms** per 3-second clip on CPU, including feature extraction (log-mel), batch 1, 1 thread, median and p95 of 200 runs after warm-up.
- **Model size ≤ 5 MB** (serialized file).
- No GPU and no network at inference time.

## Planned optimization
1. **Teacher:** Whisper's built-in language detection (small model) as a strong reference, plus a ResNet-style model trained directly on the labels as a second baseline.
2. **Student and distillation (main method):** a small depthwise-separable CNN on log-mel features (about 0.5–1 M parameters) trained against the teacher's soft language probabilities and the hard labels; temperature and mixing weight tuned on validation only.
3. **Quantization:** int8 post-training quantization, with quantization-aware training if the drop exceeds the budget.
4. **Pruning (extension):** structured channel pruning of the student, reporting the size/latency/accuracy frontier.
5. **Robustness check:** evaluate on noisy and phone-quality audio, and on code-switched utterances where the "right" answer is ambiguous, and report where the model fails.

The Lab 6 YAML + MLflow pipeline drives all runs, so every result is traceable to a config; streaming plus `num_workers` and batch-size findings from Lab 6 guide the data loader design (decoding audio is much heavier than the Fashion-MNIST transform, so workers are expected to matter here).

## Feasibility and risks
- **Data volume:** mitigated by streaming a balanced subset; a first model on 100 k clips by week 2.
- **Label noise** from split-based labels, and Malay/Indonesian overlap: mitigated by the manual audit and by using teacher soft labels.
- **Compute:** feature extraction and teacher scoring of 1–2 M clips is the main cost; I will cache log-mel features and teacher outputs once.
- **Novelty:** I make no claim that this has not been done; the contribution is a careful size/latency/accuracy study for Malaysian speech under a fixed budget, with the hard Malay–Indonesian case analysed.

## Timeline
Week 1 data audit and splits; week 2 baselines and teacher; week 3 distillation; week 4 quantization, latency, ablations, seeds, write-up.
