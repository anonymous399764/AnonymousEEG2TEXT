# EEG-to-Text Decoding from Word-Level EEG
> **S4D Encoder + BART Decoder** — Translating word-level EEG recordings into natural language text using Structured State Spaces and pretrained language models.
**Dataset in pickle format is available in https://drive.google.com/drive/folders/1TWMDhZFfOhglPuUnT2YZMpEHe3T_3HPu?usp=sharing**
---

## Overview

The model combines a bidirectional diagonal structured state-space (S4D) EEG encoder with a pretrained BART-base decoder. It is designed to address a common failure mode in cross-modal generation: a strong language decoder can ignore weak EEG conditioning and generate fluent but EEG-independent text.

The implementation uses:

- A six-layer bidirectional S4D encoder (hidden size 512, state size 64) over word-level EEG features.
- An adaptive gate between the EEG encoder and BART cross-attention. Its bias is initialized to 1.0 so that the EEG pathway starts open.
- BART-base (768 hidden units) with its text encoder replaced by gated EEG representations.
- Progressive decoder self-attention dampening during phase 2 (`self_attn_scale = 0.0`) and progressive decoder-input word dropout (0.35 to 0.85).
- Label-smoothed language-model loss, symmetric InfoNCE alignment, cross-attention entropy regularization, an EEG vocabulary-prior loss, and shuffled-EEG discrimination.

The goal is open-vocabulary generation. Greedy outputs are generated without reference tokens; teacher-forced outputs are reported only as a diagnostic upper bound and must not be interpreted as free-decoding performance.

## Data and protocol

Experiments use the publicly available ZuCo EEG/eye-tracking corpus.

```text
dataset/
├── task1-SR-dataset.pickle
├── task2-NR-dataset.pickle
└── task3-TSR-dataset.pickle
```

Each input position is a 840-dimensional Gaze Duration (GD) feature vector: 105 usable EEG channels x 8 frequency bands. EEG is normalized per feature using statistics fit on the training partition only. Training-time augmentation consists of Gaussian feature noise, channel dropout, and shifts of up to one word position.

### Primary split: sentence-disjoint

The primary evaluation uses a fixed seed of 42 and partitions 1,039 unique sentences with no sentence overlap. All recordings of a sentence are assigned to one partition.

| Partition | Unique sentences | Samples |
|---|---:|---:|
| Train | 831 | 9,978 |
| Development | 103 | 1,285 |
| Test | 105 | 1,316 |
| Total | 1,039 | 12,579 |

This protocol prevents text-level memorization across train, development, and test sets. The code also supports a subject-wise 9/1/2 train/development/test split and zero-shot evaluation on compatible held-out corpus files.

## Reported results

The primary paper result is the mean and population standard deviation over training seeds 42, 99, and 34, using deterministic greedy decoding on the sentence-disjoint test split. The checked-in machine-readable source is [`main results/Sentence_disjoint/summary.json`](main%20results/Sentence_disjoint/summary.json).

| Metric | Result |
|---|---:|
| BLEU-1 | 22.44 +/- 0.33 |
| BLEU-2 | 7.98 +/- 0.26 |
| BLEU-3 | 2.84 +/- 0.06 |
| BLEU-4 | 1.46 +/- 0.07 |
| ROUGE-1 F | 18.77 +/- 0.17 |
| ROUGE-L F | 15.42 +/- 0.55 |
| BERTScore F1 | 0.693 +/- 0.006 |
| WER | 1.033 +/- 0.003 |

The manuscript additionally reports teacher-forced diagnostic scores of BLEU-1/2/3/4 = 39.70/23.30/14.80/9.90 and ROUGE-1 F = 34.20. These scores use ground-truth prior tokens, so they are not comparable with the greedy result above.

### Signal-conditioning controls

The following controlled inputs are evaluated with the same decoder and test references. Their lower scores relative to real EEG are evidence that free outputs depend on input EEG rather than only on the decoder prior.

| Input | BLEU-1 | BLEU-4 | ROUGE-L F |
|---|---:|---:|---:|
| Real EEG | 22.44 | 1.46 | 15.40 |
| Shuffled EEG | 16.38 | 0.47 | 11.05 |
| Gaussian noise | 13.58 | 0.29 | 10.30 |
| Zero EEG | 11.67 | 0.16 | 7.82 |

Per-seed control summaries are available in [`main results/Sentence_disjoint/multiseed_reliance_results.json`](main%20results/Sentence_disjoint/multiseed_reliance_results.json).

### Selected ablations

All values below are free greedy decoding on the sentence-disjoint split. Full tables and raw result files are under [`ablation results/`](ablation%20results/).

| Variant | BLEU-1 | BLEU-4 |
|---|---:|---:|
| Full model | 22.44 | 1.46 |
| Without self-attention dampening | 20.14 | 0.83 |
| Without word dropout | 20.28 | 1.01 |
| Without InfoNCE | 19.97 | 1.04 |
| Without shuffled-EEG discrimination | 19.52 | 0.81 |
| Without attention-entropy regularization | 19.36 | 0.76 |
| Replace S4D with BiLSTM | 20.76 | 0.88 |
| Replace S4D with Transformer | 20.10 | 0.84 |

The self-attention sensitivity sweep and associated artifacts are in [`ablation results/sensitivity_ablation/`](ablation%20results/sensitivity_ablation/). The SSM comparison results are in [`ablation results/ssm_ablations/`](ablation%20results/ssm_ablations/).

## Installation

Use Python 3.10+ and PyTorch with a CUDA build appropriate for the local system. BART-base and BERTScore resources are downloaded by their respective libraries on first use.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r eeg_to_text/requirements.txt
```

On non-Windows systems, activate the environment with `source .venv/bin/activate`.

## Reproduction

Run all commands from the repository root. The public entry points accept an explicit data directory and write new artifacts to user-chosen output directories; do not rely on the legacy `full_eval.py` paths.

### Default training configuration

The main configuration in [`eeg_to_text/config.py`](eeg_to_text/config.py) uses the following two-stage schedule and regularization settings:

| Parameter | Default |
|---|---:|
| Phase 1 epochs / learning rate | 30 / 8e-05 |
| Phase 2 epochs / learning rate | 70 / 2.5e-05 |
| InfoNCE loss weight | 3.0 |
| Attention entropy loss weight | 0.15 |
| Phase 2 word dropout | 0.35 → 0.85 |
| EEG prior loss weight | 1.2 |
| EEG noise standard deviation | 0.05 |
| EEG channel dropout probability | 0.02 |
| Decoder self-attention scale | 0.0 |

### One sentence-disjoint training run

The following trains the base two-stage configuration used to describe the method: 30 warm-up epochs followed by 70 cross-attention fine-tuning epochs.

```bash
python -m eeg_to_text.train \
  --data_dir dataset \
  --split_mode sentence \
  --seed 42 \
  --phase1_epochs 30 \
  --phase2_epochs 70 \
  --fp16 \
  --checkpoint_dir runs/sentence/checkpoints \
  --log_dir runs/sentence/logs
```

For CPU-only testing, replace `--fp16` with `--device cpu --no_fp16`; this is useful for a smoke run but is substantially slower for full training.

### Evaluate a checkpoint

Use greedy decoding (`--num_beams 1`) for comparison to the main free-decoding result. The normalizer is automatically loaded from `eeg_norm_stats.npz` next to the checkpoint when present.

```bash
python -m eeg_to_text.evaluate \
  --checkpoint runs/sentence/checkpoints/<run-directory>/best.pt \
  --data_dir dataset \
  --split_mode sentence \
  --split test \
  --num_beams 1 \
  --output runs/sentence/eval_greedy.txt
```

### Three-seed aggregate

`run_multi_seed.py` fixes the data split while reinitializing and retraining the model for each requested seed. It saves per-seed results and an aggregate JSON/CSV/TXT report.

```bash
python run_multi_seed.py \
  --data_dir dataset \
  --split_seed 42 \
  --seeds 42 99 34 \
  --phase1_epochs 30 \
  --phase2_epochs 70 \
  --select_metric bertscore \
  --val_mode greedy \
  --fp16 \
  --out_dir runs/multiseed
```

For a quick functional smoke test, use one seed and one epoch per stage:

```bash
python run_multi_seed.py --data_dir dataset --seeds 42 --phase1_epochs 1 --phase2_epochs 1 --val_mode greedy --out_dir runs/smoke
```

### Result provenance

The repository preserves both the base method configuration in [`eeg_to_text/config.py`](eeg_to_text/config.py) and the frozen settings for the archived three-seed result in [`main results/Sentence_disjoint/config.json`](main%20results/Sentence_disjoint/config.json). The latter records the 30+70-epoch run and its selection criterion; it is retained to make the reported aggregate auditable. When reproducing an archived table, use the saved configuration, seeds, task files, split seed, and metric-selection rule together.

## Repository layout

```text
eeg_to_text/
├── config.py                 # Base hyperparameters
├── train.py                  # Single-run training entry point
├── evaluate.py               # Checkpoint evaluation entry point
├── data/                     # Pickle loading, preprocessing, splits, datasets
├── models/                   # S4D encoder, gate, BART integration
├── training/                 # Trainer, losses, scheduler
└── evaluation/               # BLEU, ROUGE, BERTScore, WER utilities

run_multi_seed.py             # Fixed-split multi-seed experiment
run_encoder_benchmark.py      # Controlled SSM encoder comparison
run_ablation.py               # Component ablations and EEG controls
main results/                 # Checked-in primary and generalization artifacts
ablation results/             # Checked-in ablation artifacts
Mat to Pickle file/           # Dataset-format conversion utilities
```

## Notes for reviewers

- Data, checkpoints, cached model downloads, and local training outputs are intentionally excluded from version control. Result summaries and prediction artifacts needed to audit reported numbers are retained where licensing permits.
- BERTScore can vary slightly across package/model versions and hardware. Record the environment and report seed aggregates rather than a single favorable run.
- The primary result is sentence-disjoint, free greedy decoding. Do not compare it to teacher-forced scores or to results generated under a split with overlapping sentences.

## Acknowledgements

This implementation builds on ZuCo, S4D, BART, and contrastive-learning research. Please cite the original dataset and method papers when using the data or underlying components.
