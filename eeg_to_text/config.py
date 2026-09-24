"""
Configuration for EEG-to-Text Architecture

All hyperparameters in one place. Uses dataclass for type safety and defaults.
Matches the existing ZuCo pickle dataset format (8 bands × 105 channels = 840 features).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    """Master configuration for the EEG-to-Text system."""

    # ── Data ────────────────────────────────────────────────────────────────
    # Path to the directory containing ZuCo pickle files
    data_dir: str = "dataset"
    # Which pickle files to load (relative to data_dir)
    task_pickle_files: List[str] = field(default_factory=lambda: [
        "task1-SR-dataset.pickle",
        "task2-NR-dataset.pickle",
        "task3-TSR-dataset.pickle",
    ])
    # EEG feature type used in the pickle files
    eeg_type: str = "GD"
    # Frequency bands in the pickle data  (8 bands → 105×8 = 840 features)
    bands: List[str] = field(default_factory=lambda: [
        "_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"
    ])
    n_channels: int = 105           # Number of EEG channels
    eeg_feature_dim: int = 840      # 105 channels × 8 frequency bands
    max_words: int = 56             # Max words per sentence (padding length)
    max_text_len: int = 56          # Max tokens for BART tokenizer

    # ── Model ───────────────────────────────────────────────────────────────
    bart_model: str = "facebook/bart-base"    # d_model = 768
    bart_dim: int = 768                       # BART hidden dimension
    s4d_dim: int = 512                        # S4D internal dimension
    s4d_layers: int = 6                       # Number of S4D blocks
    s4d_state_dim: int = 64                   # State space dimension N
    s4d_dropout: float = 0.1                  # Dropout in S4D blocks
    s4d_bidirectional: bool = True            # Bidirectional S4D for full-sentence context

    # Attention gate
    gate_bias_init: float = 1.0               # Initialize gate bias >0 so gate starts open

    # ── Training Phase 1: EEG Encoder Warm-Up ──────────────────────────────
    phase1_epochs: int = 30             # Longer Phase 1 for stronger EEG↔text alignment
    phase1_lr: float = 8e-05
    phase1_weight_decay: float = 0.01

    # ── Training Phase 2: Cross-Attention Fine-Tuning ────────────────
    phase2_epochs: int = 70
    phase2_lr: float = 2.5e-05          # Lower LR for fine-tuning
    phase2_weight_decay: float = 0.01

    # ── Shared Training ─────────────────────────────────────────────────────
    batch_size: int = 32
    phase2_batch_size: int = 16       # Smaller batch in Phase 2 to fit VRAM (more grads)
    grad_accum_steps: int = 4         # Effective batch = batch_size × grad_accum_steps
    phase2_grad_accum_steps: int = 8  # Keep effective batch = 16×8 = 128 ≈ 32×4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0        # Gradient clipping
    label_smoothing: float = 0.1
    lambda_contrastive: float = 3.0   # Weight for InfoNCE alignment loss
    lambda_attn_entropy: float = 0.15 # Weight for attention entropy regularizer
    temperature: float = 0.07         # InfoNCE temperature
    word_dropout: float = 0.40        # Decoder input word dropout rate (Phase 2) — peak rate
    word_dropout_start: float = 0.35  # Word dropout at start of Phase 2
    word_dropout_end: float = 0.85    # Word dropout at end of Phase 2 (forces cross-attn hard)
    self_attn_scale: float = 0.0      # Scale self-attention by this factor in Phase 2 (0=fully EEG-dependent)
    eeg_prior_alpha: float = 0.5      # Weight for EEG vocabulary prior during generation
    eeg_prior_lambda: float = 1.2     # Weight for EEG vocabulary prior loss during training

    # EEG augmentation
    eeg_noise_std: float = 0.05      # Gaussian noise added to EEG features
    eeg_channel_drop: float = 0.02   # Probability of zeroing individual channels
    eeg_time_shift: int = 1          # Max word positions to shift EEG (data aug)

    # ── Generation ──────────────────────────────────────────────────────────
    num_beams: int = 5                 # For beam search mode
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.3    # Penalise already-generated tokens (breaks mode collapse)
    max_gen_length: int = 56
    # Generation strategy (greedy is best & deterministic; alpha=0 disables EEG prior which hurts)
    gen_do_sample: bool = True         # Also run nucleus sampling for comparison
    gen_top_p: float = 0.9             # Nucleus probability threshold
    gen_temperature: float = 0.8       # Sampling temperature
    gen_eeg_prior_alpha: float = 0.0   # EEG vocab prior at gen time (0=disabled; sweep showed it hurts)
    # Best-of-N reranking
    best_of_n: int = 10                # Number of candidates for reranking (0=disabled)
    best_of_n_temperature: float = 0.9 # Temperature for diverse candidate generation
    # MBR (Minimum Bayes Risk) decoding
    mbr_n: int = 0                     # Number of MBR candidates (0=disabled)
    mbr_temperature: float = 0.8       # Temperature for MBR candidate generation
    # Contrastive search
    contrastive_alpha: float = 0.0     # Alpha for contrastive search (0=disabled)
    contrastive_k: int = 5             # Top-k for contrastive search

    # ── I/O ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "Results/checkpoints"
    log_dir: str = "Results/logs"
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"              # "cuda" or "cpu"
    fp16: bool = True                 # Mixed precision training
    log_every_n_steps: int = 50       # Log training metrics every N steps
    eval_every_n_epochs: int = 1      # Run validation every N epochs
    save_top_k: int = 3               # Keep top-k checkpoints by BERTScore
    early_stopping_patience: int = 0  # Stop if no improvement for N evals (0=disabled)
    disable_disc_loss: bool = False   # Ablation: disable shuffled-EEG discrimination loss

    # ── Subjects ────────────────────────────────────────────────────────────
    subject: str = "ALL"              # "ALL" or specific subject ID

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