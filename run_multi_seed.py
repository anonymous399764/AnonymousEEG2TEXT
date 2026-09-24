"""
run_multi_seed.py — Multi-Seed Reproducibility Experiment for EEG-to-Text

Runs the FULL training pipeline N times with different training seeds while
keeping the dataset split FIXED.  Reports per-seed test metrics and final
mean ± std across all runs.

Usage (from repo root):
    python run_multi_seed.py
    python run_multi_seed.py --data_dir dataset --seeds 42 99 34
    python run_multi_seed.py --split_seed 42 --seeds 42 99 34

Design:
    • Split seed (default 42) → used ONLY for train/val/test split.
    • Training seeds → affect model init, dropout, batch shuffling, augmentation.
    • Data is loaded and split ONCE before the seed loop.
    • For each training seed: fresh model, optimizer, scheduler, scaler.
    • Evaluation uses greedy decoding (deterministic, fastest).

Output:
    Results/multi_seed_experiment/
        per_seed_results.csv   — one row per seed with all metrics
        summary.txt            — human-readable mean ± std
        results.json           — machine-readable full results
        seed_42/               — checkpoints for seed 42
        seed_99/               — checkpoints for seed 99
        ...
"""

import argparse
import csv
import gc
import importlib
import json
import os
import random
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

# ── Make sure we can import from the project ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import (
    ZuCoEEGDataset,
    eeg_collate_fn,
    split_samples,
)
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.training.trainer import Trainer
from eeg_to_text.evaluation.metrics import (
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    compute_wer,
)

# ════════════════════════════════════════════════════════════════════════════
# Seed utilities
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down training slightly):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════════════
# Test evaluation (greedy only — fast & deterministic)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_set(model, test_loader, tokenizer, device, cfg):
    """
    Evaluate model on the test set using greedy decoding only.
    Returns a dict with all relevant metrics.
    """
    model.eval()
    model.self_attn_scale = cfg.self_attn_scale

    all_refs = []
    all_preds = []

    for batch in test_loader:
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        raw_texts = batch["raw_text"]
        all_refs.extend(raw_texts)

        # Greedy decoding (deterministic)
        preds = model.generate_text(
            eeg=eeg,
            eeg_mask=eeg_mask,
            tokenizer=tokenizer,
            max_length=cfg.max_gen_length,
            num_beams=1,
            # Match eval_sentence_split.py reference settings for greedy decoding
            eeg_prior_alpha=0.0,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.0,
        )
        all_preds.extend(preds)

    # Compute metrics
    metrics = OrderedDict()

    # BLEU
    bleu = compute_bleu(all_preds, all_refs)
    metrics["bleu1"] = bleu["bleu1"]
    metrics["bleu2"] = bleu["bleu2"]
    metrics["bleu3"] = bleu["bleu3"]
    metrics["bleu4"] = bleu["bleu4"]
    # SacreBLEU is the same as BLEU-4 from sacrebleu
    metrics["sacrebleu"] = bleu["bleu4"]

    # ROUGE
    rouge = compute_rouge(all_preds, all_refs)
    metrics["rouge1_precision"] = rouge["rouge1_precision"]
    metrics["rouge1_recall"] = rouge["rouge1_recall"]
    metrics["rouge1_f"] = rouge["rouge1"]
    metrics["rouge2_precision"] = rouge["rouge2_precision"]
    metrics["rouge2_recall"] = rouge["rouge2_recall"]
    metrics["rouge2_f"] = rouge["rouge2"]
    metrics["rougeL_precision"] = rouge["rougeL_precision"]
    metrics["rougeL_recall"] = rouge["rougeL_recall"]
    metrics["rougeL_f"] = rouge["rougeL"]

    # BERTScore
    bertscore = compute_bertscore(all_preds, all_refs)
    metrics["bertscore_precision"] = bertscore["bertscore_precision"]
    metrics["bertscore_recall"] = bertscore["bertscore_recall"]
    metrics["bertscore_f1"] = bertscore["bertscore_f1"]

    # WER
    metrics["wer"] = compute_wer(all_preds, all_refs)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Validation eval callback (used during training)
# ════════════════════════════════════════════════════════════════════════════

def make_eval_fn(cfg, hybrid_bert_weight: float = 0.7, hybrid_bleu_scale: float = 20.0):
    """Create validation evaluation callback for the Trainer.

    Notes:
      - Full evaluation (beam + TF + BERTScore) is very slow on Windows and
        can dominate Phase-2 time.
      - For checkpoint selection, we only need a stable primary metric.
    """
    from eeg_to_text.evaluation.metrics import evaluate_model

    def eval_fn(model, loader, tok, dev):
        metrics = evaluate_model(
            model, loader, tok, dev,
            num_beams=cfg.num_beams,
            max_gen_length=cfg.max_gen_length,
            print_examples=2,            # Fewer prints during multi-seed
            eeg_prior_alpha=cfg.gen_eeg_prior_alpha,
            repetition_penalty=cfg.repetition_penalty,
            gen_do_sample=False,         # Skip nucleus during training (save time)
            gen_top_p=cfg.gen_top_p,
            gen_temperature=cfg.gen_temperature,
            best_of_n=0,                 # Skip BoN during training (too slow)
        )
        # Optional hybrid metric for checkpoint selection
        try:
            bert = float(metrics.get("bertscore_f1_free", metrics.get("greedy_bertscore_f1", -1.0)))
            bleu4 = float(metrics.get("greedy_bleu4", metrics.get("bleu4", -1.0)))
            if bert >= 0 and bleu4 >= 0:
                metrics["hybrid_score"] = hybrid_bert_weight * bert + (1.0 - hybrid_bert_weight) * (bleu4 * hybrid_bleu_scale)
        except Exception:
            pass
        return metrics

    return eval_fn


def make_eval_fn_greedy_only(cfg, hybrid_bert_weight: float = 0.7, hybrid_bleu_scale: float = 20.0):
    """Greedy-only dev evaluation (fast) used for checkpoint selection.

    Returns metrics with `bertscore_f1_free` as the primary key expected by Trainer.
    """
    @torch.no_grad()
    def eval_fn(model, loader, tok, dev):
        model.eval()
        model.self_attn_scale = cfg.self_attn_scale

        all_refs = []
        all_preds = []
        for batch in loader:
            eeg = batch["eeg"].to(dev)
            eeg_mask = batch["eeg_mask"].to(dev)
            raw_texts = batch["raw_text"]
            all_refs.extend(raw_texts)
            preds = model.generate_text(
                eeg=eeg,
                eeg_mask=eeg_mask,
                tokenizer=tok,
                max_length=cfg.max_gen_length,
                num_beams=1,
                # Match eval_sentence_split.py reference settings for greedy decoding
                eeg_prior_alpha=0.0,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                repetition_penalty=1.0,
            )
            all_preds.extend(preds)

        bleu = compute_bleu(all_preds, all_refs)
        rouge = compute_rouge(all_preds, all_refs)
        bert = compute_bertscore(all_preds, all_refs)
        wer = compute_wer(all_preds, all_refs)

        metrics = {
            **bleu,
            **rouge,
            **bert,
            "wer": wer,
            # Trainer primary metric (higher is better)
            "bertscore_f1_free": bert.get("bertscore_f1", 0.0),
        }

        # Optional hybrid metric for checkpoint selection
        try:
            bert_f1 = float(metrics.get("bertscore_f1_free", -1.0))
            bleu4 = float(metrics.get("bleu4", -1.0))
            if bert_f1 >= 0 and bleu4 >= 0:
                metrics["hybrid_score"] = hybrid_bert_weight * bert_f1 + (1.0 - hybrid_bert_weight) * (bleu4 * hybrid_bleu_scale)
        except Exception:
            pass
            
        primary_key = getattr(cfg, "val_primary_metric_key", "bertscore_f1_free")
        primary_val = metrics.get(primary_key, -1.0)
        print(f"  Val Primary ({primary_key}) = {primary_val:.4f}")
        return metrics

    return eval_fn


# ════════════════════════════════════════════════════════════════════════════
# Single seed run
# ════════════════════════════════════════════════════════════════════════════

def run_single_seed(
    training_seed: int,
    train_samples,
    dev_samples,
    test_samples,
    tokenizer,
    base_cfg: Config,
    out_dir: str,
    device: str,
    val_loss_only: bool = False,
    val_mode: str = "full",
):
    """
    Run ONE full training + test evaluation for a single training seed.

    Args:
        training_seed:  Seed for model init, dropout, batch shuffling.
        train_samples:  Fixed (preprocessed) training samples [(eeg, text), ...].
        dev_samples:    Fixed (preprocessed) validation samples.
        test_samples:   Fixed (preprocessed) test samples.
        tokenizer:      BART tokenizer (reused).
        base_cfg:       Base Config object (will be copied/modified per seed).
        out_dir:        Output directory for this seed's checkpoints.
        device:         "cuda" or "cpu".
        val_loss_only:  Use fast val loss instead of full text metrics.

    Returns:
        OrderedDict of test metrics for this seed.
    """
    print(f"\n{'#' * 80}")
    print(f"#  TRAINING SEED = {training_seed}")
    print(f"#  Checkpoint dir: {out_dir}")
    print(f"{'#' * 80}\n")

    # ── Set training seed ───────────────────────────────────────────────
    set_seed(training_seed)

    # ── Build config for this seed ──────────────────────────────────────
    cfg = Config()
    # Copy over relevant settings from base_cfg
    cfg.data_dir = base_cfg.data_dir
    cfg.seed = training_seed
    cfg.device = device
    cfg.checkpoint_dir = out_dir
    cfg.log_dir = os.path.join(out_dir, "logs")
    cfg.fp16 = base_cfg.fp16
    cfg.batch_size = base_cfg.batch_size
    cfg.phase2_batch_size = base_cfg.phase2_batch_size
    cfg.num_workers = base_cfg.num_workers
    cfg.bart_model = base_cfg.bart_model
    cfg.bart_dim = base_cfg.bart_dim
    cfg.eval_every_n_epochs = base_cfg.eval_every_n_epochs
    # Copy all hyperparams
    cfg.phase1_epochs = base_cfg.phase1_epochs
    cfg.phase2_epochs = base_cfg.phase2_epochs
    cfg.phase1_lr = base_cfg.phase1_lr
    cfg.phase2_lr = base_cfg.phase2_lr
    cfg.phase1_weight_decay = base_cfg.phase1_weight_decay
    cfg.phase2_weight_decay = base_cfg.phase2_weight_decay
    cfg.grad_accum_steps = base_cfg.grad_accum_steps
    cfg.phase2_grad_accum_steps = base_cfg.phase2_grad_accum_steps
    cfg.warmup_steps = base_cfg.warmup_steps
    cfg.max_grad_norm = base_cfg.max_grad_norm
    cfg.label_smoothing = base_cfg.label_smoothing
    cfg.lambda_contrastive = base_cfg.lambda_contrastive
    cfg.lambda_attn_entropy = base_cfg.lambda_attn_entropy
    cfg.temperature = base_cfg.temperature
    cfg.word_dropout = base_cfg.word_dropout
    cfg.word_dropout_start = base_cfg.word_dropout_start
    cfg.word_dropout_end = base_cfg.word_dropout_end
    cfg.self_attn_scale = base_cfg.self_attn_scale
    cfg.eeg_prior_alpha = base_cfg.eeg_prior_alpha
    cfg.eeg_prior_lambda = base_cfg.eeg_prior_lambda
    cfg.eeg_noise_std = base_cfg.eeg_noise_std
    cfg.eeg_channel_drop = base_cfg.eeg_channel_drop
    cfg.eeg_time_shift = base_cfg.eeg_time_shift
    cfg.num_beams = base_cfg.num_beams
    cfg.repetition_penalty = base_cfg.repetition_penalty
    cfg.max_gen_length = base_cfg.max_gen_length
    cfg.gen_eeg_prior_alpha = base_cfg.gen_eeg_prior_alpha
    cfg.gen_do_sample = base_cfg.gen_do_sample
    cfg.gen_top_p = base_cfg.gen_top_p
    cfg.gen_temperature = base_cfg.gen_temperature
    cfg.early_stopping_patience = base_cfg.early_stopping_patience
    cfg.s4d_dim = base_cfg.s4d_dim
    cfg.s4d_layers = base_cfg.s4d_layers
    cfg.s4d_state_dim = base_cfg.s4d_state_dim
    cfg.s4d_dropout = base_cfg.s4d_dropout
    cfg.s4d_bidirectional = base_cfg.s4d_bidirectional
    cfg.gate_bias_init = base_cfg.gate_bias_init
    cfg.eeg_feature_dim = base_cfg.eeg_feature_dim
    cfg.max_words = base_cfg.max_words
    cfg.max_text_len = base_cfg.max_text_len
    cfg.log_every_n_steps = base_cfg.log_every_n_steps
    cfg.save_top_k = base_cfg.save_top_k
    cfg.disable_disc_loss = base_cfg.disable_disc_loss
    cfg.val_primary_metric_key = getattr(base_cfg, "val_primary_metric_key", "bertscore_f1_free")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Copy normalisation stats into this seed's checkpoint dir
    # (so each seed is self-contained for later standalone evaluation)
    parent_stats = os.path.join(os.path.dirname(out_dir), "eeg_norm_stats.npz")
    seed_stats = os.path.join(cfg.checkpoint_dir, "eeg_norm_stats.npz")
    if os.path.isfile(parent_stats):
        import shutil
        shutil.copy2(parent_stats, seed_stats)

    # ── Datasets & DataLoaders ──────────────────────────────────────────
    train_ds = ZuCoEEGDataset(
        train_samples, tokenizer, cfg.max_words, cfg.max_text_len,
        augment=True,
        noise_std=cfg.eeg_noise_std,
        channel_drop=cfg.eeg_channel_drop,
        time_shift=cfg.eeg_time_shift,
    )
    dev_ds = ZuCoEEGDataset(dev_samples, tokenizer, cfg.max_words, cfg.max_text_len)
    test_ds = ZuCoEEGDataset(test_samples, tokenizer, cfg.max_words, cfg.max_text_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    # ── Build fresh model ───────────────────────────────────────────────
    # (seed already set → model weights initialized deterministically)
    model = EEGToTextModel(
        bart_model_name=cfg.bart_model,
        eeg_input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim,
        s4d_layers=cfg.s4d_layers,
        s4d_state_dim=cfg.s4d_state_dim,
        s4d_dropout=cfg.s4d_dropout,
        s4d_bidirectional=cfg.s4d_bidirectional,
        gate_bias_init=cfg.gate_bias_init,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Evaluation callback ─────────────────────────────────────────────
    if val_loss_only:
        eval_fn = None
    elif val_mode == "greedy":
        eval_fn = make_eval_fn_greedy_only(
            cfg,
            hybrid_bert_weight=getattr(base_cfg, "hybrid_bert_weight", 0.7),
            hybrid_bleu_scale=getattr(base_cfg, "hybrid_bleu_scale", 20.0),
        )
    else:
        eval_fn = make_eval_fn(
            cfg,
            hybrid_bert_weight=getattr(base_cfg, "hybrid_bert_weight", 0.7),
            hybrid_bleu_scale=getattr(base_cfg, "hybrid_bleu_scale", 20.0),
        )

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        tokenizer=tokenizer,
        config=cfg,
        evaluate_fn=eval_fn,
    )

    # ── Train (Phase 1 + Phase 2) ───────────────────────────────────────
    t0 = time.time()
    if not val_loss_only and hasattr(base_cfg, "eval_only") and base_cfg.eval_only:
        print(f"\n  [EVAL ONLY] Skipping training for seed {training_seed}")
        train_time = 0.0
    else:
        trainer.train(resume_path=None)
        train_time = time.time() - t0
        print(f"\n  Training completed in {train_time / 60:.1f} minutes")

    # ── Load best checkpoint ────────────────────────────────────────────
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    if os.path.isfile(best_path):
        trainer.load_checkpoint(best_path)
    else:
        print("  WARNING: No best checkpoint found, using last model state")

    # ── Evaluate on test set ────────────────────────────────────────────
    print(f"\n  Evaluating on test set (greedy decoding)...")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    test_metrics = evaluate_test_set(model, test_loader, tokenizer, dev, cfg)
    test_metrics["train_time_min"] = train_time / 60.0

    # Print per-seed test results
    print(f"\n  ┌─── Test Results (seed={training_seed}) ───┐")
    for k, v in test_metrics.items():
        print(f"  │  {k:<25s} {v:>10.4f}  │")
    print(f"  └{'─' * 40}┘\n")

    # ── Cleanup GPU memory ──────────────────────────────────────────────
    del model, trainer, train_loader, dev_loader, test_loader
    del train_ds, dev_ds, test_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_metrics


# ════════════════════════════════════════════════════════════════════════════
# Results formatting and saving
# ════════════════════════════════════════════════════════════════════════════

def save_results(all_results: dict, seeds: list, out_dir: str):
    """
    Save per-seed results and aggregated summary.

    Args:
        all_results:  {seed: OrderedDict_of_metrics}
        seeds:        list of seeds (in order)
        out_dir:      output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Determine metric names ──────────────────────────────────────────
    metric_names = list(next(iter(all_results.values())).keys())

    # Key metrics for the summary
    key_metrics = [
        "bleu1", "bleu2", "bleu3", "bleu4", "sacrebleu",
        "rouge1_f", "rouge2_f", "rougeL_f",
        "rouge1_precision", "rouge1_recall",
        "rouge2_precision", "rouge2_recall",
        "rougeL_precision", "rougeL_recall",
        "bertscore_precision", "bertscore_recall", "bertscore_f1",
        "wer",
    ]

    # ── 1. Per-seed CSV ─────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "per_seed_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + metric_names)
        for seed in seeds:
            row = [seed] + [f"{all_results[seed][m]:.6f}" for m in metric_names]
            writer.writerow(row)

        # Add mean and std rows
        writer.writerow([])
        means = ["MEAN"] + [
            f"{np.mean([all_results[s][m] for s in seeds]):.6f}" for m in metric_names
        ]
        stds = ["STD"] + [
            f"{np.std([all_results[s][m] for s in seeds]):.6f}" for m in metric_names
        ]
        writer.writerow(means)
        writer.writerow(stds)

    print(f"  Saved: {csv_path}")

    # ── 2. Summary TXT ──────────────────────────────────────────────────
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("MULTI-SEED REPRODUCIBILITY EXPERIMENT — EEG-TO-TEXT\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training seeds: {seeds}\n")
        f.write(f"Number of runs: {len(seeds)}\n")
        f.write("Split: sentence-disjoint (ZuCo 1.0 only, no sentence overlap)\n")
        f.write("Evaluation: greedy decoding (deterministic)\n")
        f.write("=" * 90 + "\n\n")

        # Per-seed results table
        f.write("PER-SEED TEST RESULTS\n")
        f.write("-" * 90 + "\n")

        # Header
        header = f"{'Metric':<28}"
        for seed in seeds:
            header += f"  Seed {seed:>5}"
        header += f"    {'Mean':>8}  {'± Std':>8}"
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")

        # Rows
        for m in key_metrics:
            row = f"{m:<28}"
            vals = [all_results[s][m] for s in seeds]
            for v in vals:
                row += f"  {v:>10.4f}"
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            row += f"    {mean_v:>8.4f}  {std_v:>8.4f}"
            f.write(row + "\n")

        f.write("─" * len(header) + "\n\n")

        # Final summary in requested format
        f.write("=" * 90 + "\n")
        f.write("FINAL AGGREGATED RESULTS (Mean ± Std)\n")
        f.write("=" * 90 + "\n\n")

        for seed in seeds:
            f.write(f"  Seed {seed} -> ")
            vals = []
            for m in ["bleu4", "rougeL_f", "bertscore_f1"]:
                vals.append(f"{m}={all_results[seed][m]:.4f}")
            f.write("  ".join(vals) + "\n")

        f.write("\n  Final:\n")
        for m_display, m_key in [
            ("BLEU-1", "bleu1"),
            ("BLEU-2", "bleu2"),
            ("BLEU-3", "bleu3"),
            ("BLEU-4", "bleu4"),
            ("SacreBLEU", "sacrebleu"),
            ("ROUGE-1 F", "rouge1_f"),
            ("ROUGE-2 F", "rouge2_f"),
            ("ROUGE-L F", "rougeL_f"),
            ("ROUGE-1 P", "rouge1_precision"),
            ("ROUGE-1 R", "rouge1_recall"),
            ("ROUGE-2 P", "rouge2_precision"),
            ("ROUGE-2 R", "rouge2_recall"),
            ("ROUGE-L P", "rougeL_precision"),
            ("ROUGE-L R", "rougeL_recall"),
            ("BERTScore P", "bertscore_precision"),
            ("BERTScore R", "bertscore_recall"),
            ("BERTScore F1", "bertscore_f1"),
            ("WER", "wer"),
        ]:
            vals = [all_results[s][m_key] for s in seeds]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            f.write(f"    {m_display:<16} = {mean_v:.4f} ± {std_v:.4f}\n")

        # Training time stats
        times = [all_results[s].get("train_time_min", 0) for s in seeds]
        f.write(f"\n  Training time per run: {np.mean(times):.1f} ± {np.std(times):.1f} min\n")
        f.write(f"  Total experiment time: {np.sum(times):.1f} min\n")

    print(f"  Saved: {txt_path}")

    # ── 3. JSON results ─────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    json_data = {
        "experiment": "multi_seed_reproducibility",
        "date": datetime.now().isoformat(),
        "seeds": seeds,
        "split_mode": "sentence_disjoint",
        "per_seed": {str(s): dict(all_results[s]) for s in seeds},
        "aggregated": {},
    }
    for m in key_metrics:
        vals = [all_results[s][m] for s in seeds]
        json_data["aggregated"][m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Saved: {json_path}")

    return csv_path, txt_path, json_path


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Seed Reproducibility Experiment for EEG-to-Text"
    )
    parser.add_argument(
        "--data_dir", type=str, default="dataset",
        help="Directory containing ZuCo pickle files (default: dataset)"
    )
    parser.add_argument(
        "--split_seed", type=int, default=42,
        help="Fixed seed for train/val/test split (default: 42)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 99, 34],
        help="Training seeds to run (default: 42 99 34)"
    )
    parser.add_argument(
        "--out_dir", type=str, default="Results/multi_seed_experiment",
        help="Output directory for results (default: Results/multi_seed_experiment)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--phase1_epochs", type=int, default=None,
        help="Override Phase 1 epochs"
    )
    parser.add_argument(
        "--phase2_epochs", type=int, default=None,
        help="Override Phase 2 epochs"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--no_fp16", action="store_true", default=False,
        help="Disable mixed precision"
    )
    parser.add_argument(
        "--val_loss_only", action="store_true", default=False,
        help="Use fast validation loss instead of full text metrics during training"
    )
    parser.add_argument(
        "--val_mode", type=str, choices=["full", "greedy"], default="full",
        help=(
            "Validation evaluation mode. 'full' runs greedy+beam+TF (slow). "
            "'greedy' runs greedy-only metrics (faster; still uses BERTScore for checkpoint selection)."
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Override DataLoader workers"
    )
    parser.add_argument(
        "--eval_every_n_epochs", type=int, default=None,
        help="Run validation every N epochs"
    )
    parser.add_argument(
        "--eval_only", action="store_true", default=False,
        help="Skip training, load existing best.pt for each seed, and evaluate"
    )
    parser.add_argument(
        "--norm_stats_path", type=str, default=None,
        help=(
            "Optional path to an existing eeg_norm_stats.npz to use for ALL splits "
            "(skips fitting on the train split). Useful to match a reference checkpoint's "
            "normalization recipe, e.g. Results/checkpoints_sentence_split/eeg_norm_stats.npz"
        ),
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=None,
        help=(
            "Override Config.early_stopping_patience. "
            "If not set, uses Config default."
        ),
    )
    parser.add_argument(
        "--task_files", nargs="+", default=None,
        help="Pickle file names to load (default: ZuCo 1.0 tasks 1-3)"
    )
    parser.add_argument(
        "--select_metric",
        type=str,
        choices=["bertscore", "bleu4", "bleu1", "hybrid"],
        default="bertscore",
        help=(
            "Which DEV metric to use for saving best.pt. "
            "'bertscore' matches the reference setup. "
            "'bleu4' optimizes BLEU-4 directly. "
            "'hybrid' uses a weighted combination (see --hybrid_*)."
        ),
    )
    parser.add_argument(
        "--hybrid_bert_weight",
        type=float,
        default=0.7,
        help="Hybrid metric weight on BERTScore (default: 0.7).",
    )
    parser.add_argument(
        "--hybrid_bleu_scale",
        type=float,
        default=20.0,
        help="Hybrid metric BLEU-4 scale factor before mixing (default: 20.0).",
    )
    return parser.parse_args()


def _require_modules():
    """Fail fast with a helpful message if metric dependencies are missing.

    This prevents long training runs from crashing only when validation/test metrics
    are computed (common if the wrong Python interpreter is used).
    """
    required = [
        ("sacrebleu", "sacrebleu"),
        ("rouge_score", "rouge-score"),
        ("bert_score", "bert-score"),
        ("jiwer", "jiwer"),
    ]
    missing = []
    for module_name, pip_name in required:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("\nERROR: Missing required evaluation packages: " + ", ".join(missing))
        print(f"Python executable: {sys.executable}")
        print("Install with:")
        print(f"  {sys.executable} -m pip install " + " ".join(missing))
        sys.exit(1)


def main():
    args = parse_args()

    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    _require_modules()

    print("=" * 90)
    print("  MULTI-SEED REPRODUCIBILITY EXPERIMENT")
    print("  EEG-to-Text (Sentence-Disjoint Split)")
    print(f"  Split seed: {args.split_seed} (FIXED across all runs)")
    print(f"  Training seeds: {args.seeds}")
    print(f"  Output: {args.out_dir}")
    print("=" * 90)

    # ── Base config ─────────────────────────────────────────────────────
    BART = "facebook/bart-base"
    base_cfg = Config(bart_model=BART, bart_dim=768)
    base_cfg.data_dir = args.data_dir
    base_cfg.device = args.device

    # ZuCo 1.0 only (sentence-disjoint split uses these 3 task files)
    if args.task_files is not None:
        base_cfg.task_pickle_files = args.task_files
    else:
        base_cfg.task_pickle_files = [
            "task1-SR-dataset.pickle",
            "task2-NR-dataset.pickle",
            "task3-TSR-dataset.pickle",
        ]

    # Apply overrides
    if args.batch_size is not None:
        base_cfg.batch_size = args.batch_size
    if args.phase1_epochs is not None:
        base_cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        base_cfg.phase2_epochs = args.phase2_epochs
    if args.num_workers is not None:
        base_cfg.num_workers = args.num_workers
    if args.eval_every_n_epochs is not None:
        base_cfg.eval_every_n_epochs = args.eval_every_n_epochs
    if args.early_stopping_patience is not None:
        base_cfg.early_stopping_patience = args.early_stopping_patience
    if args.no_fp16:
        base_cfg.fp16 = False
    elif args.fp16:
        base_cfg.fp16 = True
    base_cfg.eval_only = args.eval_only

    # Selection metric for best.pt
    if args.select_metric == "bertscore":
        base_cfg.val_primary_metric_key = "bertscore_f1_free"
    elif args.select_metric == "bleu4":
        base_cfg.val_primary_metric_key = "bleu4"
    elif args.select_metric == "bleu1":
        base_cfg.val_primary_metric_key = "bleu1"
    else:
        base_cfg.val_primary_metric_key = "hybrid_score"

    # Hybrid scoring params (stored on config so per-seed eval fn can read them)
    base_cfg.hybrid_bert_weight = float(args.hybrid_bert_weight)
    base_cfg.hybrid_bleu_scale = float(args.hybrid_bleu_scale)

    # ════════════════════════════════════════════════════════════════════
    # STEP 1: Load data ONCE with the FIXED split seed
    # ════════════════════════════════════════════════════════════════════
    print(f"\n[STEP 1] Loading data from: {base_cfg.data_dir}")
    print(f"         Task files: {base_cfg.task_pickle_files}")
    print(f"         Split seed: {args.split_seed} (FIXED — same split for all runs)\n")

    dataset_dicts = load_pickle_datasets(base_cfg.data_dir, base_cfg.task_pickle_files)
    if not dataset_dicts:
        print(f"ERROR: No datasets found in {base_cfg.data_dir}. "
              f"Expected files: {base_cfg.task_pickle_files}")
        sys.exit(1)

    # Extract EEG features
    preprocessor = EEGPreprocessor(
        eeg_type=base_cfg.eeg_type,
        bands=base_cfg.bands,
        n_channels=base_cfg.n_channels,
    )
    # For sentence-disjoint split: extract without subject info
    all_samples = preprocessor.extract_all_sentences(dataset_dicts, subject=base_cfg.subject)

    if len(all_samples) == 0:
        print("ERROR: No valid samples extracted. Check pickle files.")
        sys.exit(1)

    print(f"  Total samples extracted: {len(all_samples)}")

    # ── Split with FIXED seed ───────────────────────────────────────────
    # This split is IDENTICAL for every training seed run.
    train_raw, dev_raw, test_raw = split_samples(
        all_samples, train_ratio=0.8, dev_ratio=0.1, seed=args.split_seed
    )

    print(f"\n  Fixed split (seed={args.split_seed}):")
    print(f"    Train: {len(train_raw)} samples")
    print(f"    Dev:   {len(dev_raw)} samples")
    print(f"    Test:  {len(test_raw)} samples")

    # ── Normalisation (match recipe if requested) ──────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "eeg_norm_stats.npz")
    if args.norm_stats_path is not None:
        if not os.path.isfile(args.norm_stats_path):
            print(f"ERROR: --norm_stats_path not found: {args.norm_stats_path}")
            sys.exit(1)
        preprocessor.load_stats(args.norm_stats_path)
        # Copy into out_dir for self-contained later evaluation
        import shutil
        shutil.copy2(args.norm_stats_path, stats_path)
        print(f"  Using normalisation stats from: {args.norm_stats_path}")
        print(f"  Normalisation stats copied to: {stats_path}")
    else:
        preprocessor.fit([e for e, t in train_raw])
        preprocessor.save_stats(stats_path)
        print(f"  Normalisation stats fit on TRAIN and saved to: {stats_path}")

    # ── Transform all splits ────────────────────────────────────────────
    train_samples = [(preprocessor.transform(e), t) for e, t in train_raw]
    dev_samples = [(preprocessor.transform(e), t) for e, t in dev_raw]
    test_samples = [(preprocessor.transform(e), t) for e, t in test_raw]

    # ── Load tokenizer ONCE ─────────────────────────────────────────────
    print(f"  Loading tokenizer: {BART}")
    tokenizer = BartTokenizer.from_pretrained(BART)

    # ════════════════════════════════════════════════════════════════════
    # STEP 2: Run training for each seed
    # ════════════════════════════════════════════════════════════════════
    all_results = OrderedDict()
    experiment_t0 = time.time()

    for i, seed in enumerate(args.seeds, 1):
        print(f"\n{'█' * 90}")
        print(f"  RUN {i}/{len(args.seeds)} — Training Seed {seed}")
        print(f"{'█' * 90}")

        seed_out_dir = os.path.join(args.out_dir, f"seed_{seed}")

        metrics = run_single_seed(
            training_seed=seed,
            train_samples=train_samples,
            dev_samples=dev_samples,
            test_samples=test_samples,
            tokenizer=tokenizer,
            base_cfg=base_cfg,
            out_dir=seed_out_dir,
            device=args.device,
            val_loss_only=args.val_loss_only,
            val_mode=args.val_mode,
        )

        all_results[seed] = metrics

        # Save intermediate results after each seed (in case of crash)
        _save_intermediate(all_results, args.seeds[:i], args.out_dir)

    total_time = time.time() - experiment_t0

    # ════════════════════════════════════════════════════════════════════
    # STEP 3: Save final results
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  SAVING FINAL RESULTS")
    print(f"{'=' * 90}\n")

    csv_path, txt_path, json_path = save_results(all_results, args.seeds, args.out_dir)

    # ════════════════════════════════════════════════════════════════════
    # STEP 4: Print final summary to console
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("  FINAL MULTI-SEED RESULTS")
    print(f"{'=' * 90}\n")

    for seed in args.seeds:
        m = all_results[seed]
        print(f"  Seed {seed:>5} -> "
              f"BLEU-4={m['bleu4']:.4f}  "
              f"ROUGE-L={m['rougeL_f']:.4f}  "
              f"BERTScore={m['bertscore_f1']:.4f}")

    print(f"\n  {'─' * 60}")
    print(f"  Final (Mean ± Std):")
    for m_display, m_key in [
        ("BLEU-1", "bleu1"),
        ("BLEU-2", "bleu2"),
        ("BLEU-3", "bleu3"),
        ("BLEU-4", "bleu4"),
        ("SacreBLEU", "sacrebleu"),
        ("ROUGE-1 F", "rouge1_f"),
        ("ROUGE-2 F", "rouge2_f"),
        ("ROUGE-L F", "rougeL_f"),
        ("BERTScore P", "bertscore_precision"),
        ("BERTScore R", "bertscore_recall"),
        ("BERTScore F1", "bertscore_f1"),
        ("WER", "wer"),
    ]:
        vals = [all_results[s][m_key] for s in args.seeds]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        print(f"    {m_display:<16} = {mean_v:.4f} ± {std_v:.4f}")

    print(f"\n  Total experiment time: {total_time / 60:.1f} minutes")
    print(f"  Results saved to: {args.out_dir}/")
    print(f"    • {csv_path}")
    print(f"    • {txt_path}")
    print(f"    • {json_path}")
    print(f"\n{'=' * 90}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'=' * 90}\n")


def _save_intermediate(all_results, completed_seeds, out_dir):
    """Save intermediate results after each seed run (crash recovery)."""
    os.makedirs(out_dir, exist_ok=True)
    intermediate_path = os.path.join(out_dir, "intermediate_results.json")
    data = {
        "completed_seeds": completed_seeds,
        "timestamp": datetime.now().isoformat(),
        "per_seed": {str(s): dict(all_results[s]) for s in completed_seeds},
    }
    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
