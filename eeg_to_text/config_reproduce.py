"""
Reproduction Config — matches the settings that produced
    Results/checkpoints_sentence_split/best.pt
    (BLEU-4 ≈ 0.025, BERTScore F1 ≈ 0.71, val BERTScore 0.6912)

Key differences vs the current config.py:
  1. Trains on ALL 5 pickle files (ZuCo 1.0 + 2.0) — more training data.
  2. No early stopping (patience = 0) — runs the full 60 epochs.
  3. disable_disc_loss = False (kept as-is; original had no such flag).

Evaluation is always done on ZuCo 1.0 only (sentence-disjoint split).
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class ReproduceConfig:
    """Config frozen to reproduce checkpoints_sentence_split/best.pt."""

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir: str = "dataset"
    # ALL 5 pickle files — this is what the original training used
    task_pickle_files: List[str] = field(default_factory=lambda: [
        "task1-SR-dataset.pickle",
        "task2-NR-dataset.pickle",
        "task3-TSR-dataset.pickle",
    ])
    eeg_type: str = "GD"
    bands: List[str] = field(default_factory=lambda: [
        "_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"
    ])
    n_channels: int = 105
    eeg_feature_dim: int = 840      # 105 × 8
    max_words: int = 56
    max_text_len: int = 56

    # ── Model ───────────────────────────────────────────────────────────────
    bart_model: str = "facebook/bart-base"
    bart_dim: int = 768
    s4d_dim: int = 512
    s4d_layers: int = 6
    s4d_state_dim: int = 64
    s4d_dropout: float = 0.1
    s4d_bidirectional: bool = True
    gate_bias_init: float = 1.0

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    phase1_epochs: int = 20
    phase1_lr: float = 1e-4
    phase1_weight_decay: float = 0.01

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    phase2_epochs: int = 40
    phase2_lr: float = 3e-5
    phase2_weight_decay: float = 0.01

    # ── Shared Training ─────────────────────────────────────────────────────
    batch_size: int = 32
    phase2_batch_size: int = 16
    grad_accum_steps: int = 4
    phase2_grad_accum_steps: int = 8
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    lambda_contrastive: float = 2.0
    lambda_attn_entropy: float = 0.1
    temperature: float = 0.07
    word_dropout: float = 0.40
    word_dropout_start: float = 0.30
    word_dropout_end: float = 0.75
    self_attn_scale: float = 0.0
    eeg_prior_alpha: float = 0.5
    eeg_prior_lambda: float = 1.0

    # EEG augmentation
    eeg_noise_std: float = 0.1
    eeg_channel_drop: float = 0.05
    eeg_time_shift: int = 1

    # ── Generation ──────────────────────────────────────────────────────────
    num_beams: int = 5
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.3
    max_gen_length: int = 56
    gen_do_sample: bool = True
    gen_top_p: float = 0.9
    gen_temperature: float = 0.8
    gen_eeg_prior_alpha: float = 0.0
    best_of_n: int = 10
    best_of_n_temperature: float = 0.9
    mbr_n: int = 0
    mbr_temperature: float = 0.8
    contrastive_alpha: float = 0.0
    contrastive_k: int = 5

    # ── I/O ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "Results/checkpoints_reproduce"
    log_dir: str = "Results/logs_reproduce"
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"
    fp16: bool = True
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_top_k: int = 3
    early_stopping_patience: int = 0   # NO early stopping — run full 60 epochs
    disable_disc_loss: bool = False
    val_primary_metric_key: str = "bleu4"

    # ── Subjects ────────────────────────────────────────────────────────────
    subject: str = "ALL"

    def total_epochs(self) -> int:
        return self.phase1_epochs + self.phase2_epochs

    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def get_checkpoint_dir(self) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return self.checkpoint_dir

    def get_log_dir(self) -> str:
        os.makedirs(self.log_dir, exist_ok=True)
        return self.log_dir
