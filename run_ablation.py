"""
Ablation Study for EEG-to-Text

Trains and evaluates model variants with individual components removed
to prove that each design decision contributes to final performance.

Ablation conditions:
    0. Full model (baseline)          — all components active
    1. No contrastive loss            — lambda_contrastive = 0
    2. No attention entropy loss      — lambda_attn_entropy = 0
    3. No attention gate              — bypass gate (identity)
    4. No word dropout                — word_dropout_start = word_dropout_end = 0
    5. No self-attention dampening    — self_attn_scale = 1.0
    6. BiLSTM encoder                 — replace S4D encoder with BiLSTM
    7. Transformer encoder            — replace S4D encoder with Transformer

All trained with sentence-disjoint split on ZuCo 1.0.
Evaluated with greedy decoding (primary metric).
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import (
    ZuCoEEGDataset, eeg_collate_fn, split_samples,
)
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.models.s4d_encoder import (
    BiLSTMEEGEncoder,
    LinearEEGEncoder,
    TransformerEEGEncoder,
)
from eeg_to_text.training.trainer import Trainer
from eeg_to_text.evaluation.metrics import evaluate_model, compute_bleu, compute_rouge, compute_bertscore, compute_wer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BART = "facebook/bart-base"
ABLATION_DIR = "Results/ablation_study"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_existing_baseline():
    """Load pre-existing full model results from sentence_split_metrics.txt."""
    # These are from Results/checkpoints_sentence_split/best.pt
    # Best epoch 43, trained 54 epochs, val BERTScore F1 = 0.6912
    # Sentence-disjoint split, 1316 test samples
    return {
        "greedy_bertscore_f1": 0.7085,
        "greedy_bertscore_precision": 0.7108,
        "greedy_bertscore_recall": 0.7067,
        "greedy_bleu1": 0.2242,
        "greedy_bleu2": 0.0892,
        "greedy_bleu3": 0.0446,
        "greedy_bleu4": 0.0253,
        "greedy_rouge1": 0.1891,
        "greedy_rouge1_precision": 0.1974,
        "greedy_rouge1_recall": 0.1896,
        "greedy_rouge2": 0.0319,
        "greedy_rouge2_precision": 0.0336,
        "greedy_rouge2_recall": 0.0318,
        "greedy_rougeL": 0.1480,
        "greedy_rougeL_precision": 0.1545,
        "greedy_rougeL_recall": 0.1485,
        "greedy_wer": 1.0166,
        "tf_bertscore_f1": 0.7535,
        "tf_bleu1": 0.3973,
        "tf_bleu4": 0.0987,
    }


def get_ablation_configs():
    """Return (name, config_overrides, model_overrides) for each condition."""
    ablations = [
        (
            "no_contrastive_loss",
            {"lambda_contrastive": 0.0},
            {},
        ),
        (
            "no_attn_entropy",
            {"lambda_attn_entropy": 0.0},
            {},
        ),
        (
            "no_attention_gate",
            {},
            {"bypass_gate": True},
        ),
        (
            "no_word_dropout",
            {"word_dropout_start": 0.0, "word_dropout_end": 0.0, "word_dropout": 0.0},
            {},
        ),
        (
            "no_self_attn_dampening",
            {"self_attn_scale": 1.0},
            {},
        ),
        (
            "no_disc_loss",
            {"disable_disc_loss": True},
            {},
        ),
        (
            "linear_encoder",
            {},
            {"use_linear_encoder": True},
        ),
        (
            "bilstm_encoder",
            {},
            {"use_bilstm_encoder": True},
        ),
        (
            "transformer_encoder",
            {},
            {"use_transformer_encoder": True},
        ),
    ]
    return ablations


def build_config(overrides: dict) -> Config:
    """Build a Config with ablation overrides."""
    cfg = Config()
    cfg.device = DEVICE
    cfg.bart_model = BART
    cfg.bart_dim = 768
    cfg.fp16 = True
    cfg.seed = 42
    cfg.data_dir = "dataset"
    cfg.num_workers = 2
    cfg.early_stopping_patience = 8  # stop if no improvement for 8 Phase-2 epochs
    # Apply overrides
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def load_and_split_data(cfg):
    """Load ZuCo data, sentence-disjoint split, normalise."""
    dicts = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    pre = EEGPreprocessor(
        eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels,
    )
    all_samples = pre.extract_all_sentences_with_subjects(dicts, subject=cfg.subject)
    plain = [(e, t) for e, t, s in all_samples]

    train_s, dev_s, test_s = split_samples(
        plain, train_ratio=0.8, dev_ratio=0.1, seed=cfg.seed,
    )
    pre.fit([s[0] for s in train_s])
    train_s = [(pre.transform(e), t) for e, t in train_s]
    dev_s = [(pre.transform(e), t) for e, t in dev_s]
    test_s = [(pre.transform(e), t) for e, t in test_s]
    return train_s, dev_s, test_s, pre


def build_dataloaders(cfg, train_s, dev_s, test_s, tokenizer):
    """Build train/dev/test DataLoaders."""
    train_ds = ZuCoEEGDataset(
        train_s, tokenizer, cfg.max_words, cfg.max_text_len,
        augment=True,
        noise_std=cfg.eeg_noise_std,
        channel_drop=cfg.eeg_channel_drop,
        time_shift=cfg.eeg_time_shift,
    )
    dev_ds = ZuCoEEGDataset(dev_s, tokenizer, cfg.max_words, cfg.max_text_len)
    test_ds = ZuCoEEGDataset(test_s, tokenizer, cfg.max_words, cfg.max_text_len)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True,
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_dl, dev_dl, test_dl


def build_model(cfg, model_overrides: dict) -> EEGToTextModel:
    """Build model, optionally bypassing the attention gate or replacing encoder."""
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
    if model_overrides.get("bypass_gate", False):
        # Replace attention gate with identity so EEG passes through unchanged
        model.attention_gate = nn.Identity()
    if model_overrides.get("use_linear_encoder", False):
        # Replace S4D encoder with minimal Linear → GELU → LayerNorm
        model.eeg_encoder = LinearEEGEncoder(
            input_dim=cfg.eeg_feature_dim,
            bart_dim=cfg.bart_dim,
        )
    if model_overrides.get("use_bilstm_encoder", False):
        # Replace S4D encoder with BiLSTM baseline
        model.eeg_encoder = BiLSTMEEGEncoder(
            input_dim=cfg.eeg_feature_dim,
            lstm_input_dim=cfg.s4d_dim,
            hidden_dim=cfg.s4d_dim // 2,
            n_layers=max(2, cfg.s4d_layers),
            dropout=cfg.s4d_dropout,
            bart_dim=cfg.bart_dim,
        )
    if model_overrides.get("use_transformer_encoder", False):
        # Replace S4D encoder with Transformer baseline
        n_heads = 8
        while n_heads > 1 and (cfg.s4d_dim % n_heads != 0):
            n_heads //= 2
        model.eeg_encoder = TransformerEEGEncoder(
            input_dim=cfg.eeg_feature_dim,
            model_dim=cfg.s4d_dim,
            n_layers=cfg.s4d_layers,
            n_heads=n_heads,
            ff_dim=cfg.s4d_dim * 4,
            dropout=cfg.s4d_dropout,
            bart_dim=cfg.bart_dim,
        )
    return model


def evaluate_checkpoint(ckpt_path, cfg, test_dl, tokenizer, model_overrides):
    """Load a checkpoint and evaluate with greedy decoding."""
    model = build_model(cfg, model_overrides)
    model.to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.self_attn_scale = cfg.self_attn_scale
    model.set_phase(2)
    model.eval()

    metrics = evaluate_model(
        model, test_dl, tokenizer, DEVICE,
        num_beams=1,
        max_gen_length=cfg.max_gen_length,
        print_examples=3,
        eeg_prior_alpha=0.0,
        repetition_penalty=cfg.repetition_penalty,
        gen_do_sample=False,  # greedy only for ablation
        best_of_n=0,
        mbr_n=0,
        contrastive_alpha=0.0,
    )
    return metrics


def run_single_ablation(name, cfg_overrides, model_overrides, tokenizer,
                        train_s, dev_s, test_s, pre):
    """Train and evaluate one ablation condition."""
    print(f"\n{'='*80}")
    print(f"  ABLATION: {name}")
    print(f"  Config overrides: {cfg_overrides}")
    print(f"  Model overrides: {model_overrides}")
    print(f"{'='*80}\n")

    set_seed(42)

    cfg = build_config(cfg_overrides)
    ckpt_dir = os.path.join(ABLATION_DIR, name)
    cfg.checkpoint_dir = ckpt_dir
    cfg.log_dir = os.path.join(ABLATION_DIR, "logs", name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Save norm stats
    pre.save_stats(os.path.join(ckpt_dir, "eeg_norm_stats.npz"))

    # Build data
    train_dl, dev_dl, test_dl = build_dataloaders(
        cfg, train_s, dev_s, test_s, tokenizer,
    )
    print(f"Train: {len(train_dl.dataset)} | Dev: {len(dev_dl.dataset)} | Test: {len(test_dl.dataset)}")

    # Build model
    model = build_model(cfg, model_overrides)
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.eeg_encoder.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"EEG encoder parameters: {encoder_params:,}")

    # Eval callback
    def eval_fn(m, loader, tok, dev):
        return evaluate_model(
            m, loader, tok, dev,
            num_beams=1,
            max_gen_length=cfg.max_gen_length,
            print_examples=2,
            eeg_prior_alpha=0.0,
            repetition_penalty=cfg.repetition_penalty,
            gen_do_sample=False,
            best_of_n=0,
            mbr_n=0,
            contrastive_alpha=0.0,
        )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=dev_dl,
        tokenizer=tokenizer,
        config=cfg,
        evaluate_fn=eval_fn,
    )

    # Auto-resume from last.pt if it exists (handles interrupted runs)
    last_pt = os.path.join(ckpt_dir, "last.pt")
    best_pt_existing = os.path.join(ckpt_dir, "best.pt")
    csv_log_path = os.path.join(ckpt_dir, "epoch_log.csv")
    resume_from = last_pt if os.path.isfile(last_pt) else None
    if resume_from:
        import torch as _torch
        _ck = _torch.load(resume_from, map_location="cpu", weights_only=True)
        _ck_state = _ck.get("model_state_dict", {})
        _model_state = model.state_dict()
        _missing = [k for k in _model_state.keys() if k not in _ck_state]
        _unexpected = [k for k in _ck_state.keys() if k not in _model_state]
        _shape_mismatch = [
            k for k in _model_state.keys()
            if k in _ck_state and _model_state[k].shape != _ck_state[k].shape
        ]

        if _missing or _unexpected or _shape_mismatch:
            print("  [Resume disabled] Checkpoint incompatible with current model variant.")
            print(f"    missing={len(_missing)}  unexpected={len(_unexpected)}  shape_mismatch={len(_shape_mismatch)}")
            print("  [Restarting] Clearing stale checkpoints/log for this ablation and training from scratch.")
            for _p in (last_pt, best_pt_existing, csv_log_path):
                if os.path.isfile(_p):
                    os.remove(_p)
            resume_from = None

        _saved_epoch = _ck.get("epoch", 0)
        total_epochs = cfg.phase1_epochs + cfg.phase2_epochs
        if resume_from is not None and _saved_epoch >= total_epochs:
            print(f"  [Skip training] last.pt is epoch {_saved_epoch}/{total_epochs} — already complete.")
            resume_from = None  # will go straight to evaluation
            trainer = None
        elif resume_from is not None:
            print(f"  [Resuming] last.pt is epoch {_saved_epoch} — resuming from epoch {_saved_epoch + 1}")
    if resume_from is not None or not os.path.isfile(best_pt_existing):
        trainer.train(resume_path=resume_from)

    # Evaluate best checkpoint
    best_path = os.path.join(ckpt_dir, "best.pt")
    if not os.path.isfile(best_path):
        print(f"WARNING: No best checkpoint found for {name}")
        return None

    metrics = evaluate_checkpoint(
        best_path, cfg, test_dl, tokenizer, model_overrides,
    )

    # Save test results
    results_path = os.path.join(ckpt_dir, "test_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        for k, v in sorted(metrics.items()):
            f.write(f"{k}: {v}\n")
    print(f"\nResults saved to {results_path}")

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return metrics


def format_ablation_table(results: dict) -> str:
    """Format results into a publication-ready ablation table."""
    # Key metrics to report
    metric_keys = [
        ("BERTScore F1", "greedy_bertscore_f1"),
        ("BERTScore P", "greedy_bertscore_precision"),
        ("BERTScore R", "greedy_bertscore_recall"),
        ("BLEU-1", "greedy_bleu1"),
        ("BLEU-2", "greedy_bleu2"),
        ("BLEU-3", "greedy_bleu3"),
        ("BLEU-4", "greedy_bleu4"),
        ("ROUGE-1 F", "greedy_rouge1"),
        ("ROUGE-2 F", "greedy_rouge2"),
        ("ROUGE-L F", "greedy_rougeL"),
        ("WER", "greedy_wer"),
        ("TF BERTScore F1", "tf_bertscore_f1"),
    ]

    ablation_display = {
        "full_model":           "Full Model (Ours)",
        "no_contrastive_loss":  "- InfoNCE Loss",
        "no_attn_entropy":      "- Attn Entropy",
        "no_attention_gate":    "- Attention Gate",
        "no_word_dropout":      "- Word Dropout",
        "no_self_attn_dampening": "- Self-Attn Damp.",
        "no_disc_loss":         "- Disc Loss",
        "linear_encoder":       "- S4D (Linear Proj.)",
        "bilstm_encoder":       "- S4D (BiLSTM)",
        "transformer_encoder":  "- S4D (Transformer)",
    }

    lines = []
    lines.append("=" * 130)
    lines.append("ABLATION STUDY — EEG-TO-TEXT (SENTENCE-DISJOINT SPLIT, ZuCo 1.0)")
    lines.append("Each row removes ONE component from the full model.")
    lines.append("All models trained with identical hyperparameters, seed=42, sentence-disjoint split.")
    lines.append("Evaluation: Greedy decoding (deterministic, primary metric).")
    lines.append("Early stopping: patience=8 epochs (Phase 2 only).")
    lines.append("=" * 130)
    lines.append("")

    # Header
    header_names = [name for name, _ in metric_keys]
    header = f"{'Condition':<25s}" + "".join(f"{n:>14s}" for n in header_names)
    lines.append(header)
    lines.append("-" * len(header))

    ablation_order = [
        "full_model",
        "no_contrastive_loss",
        "no_attn_entropy",
        "no_attention_gate",
        "no_word_dropout",
        "no_self_attn_dampening",
        "no_disc_loss",
        "linear_encoder",
        "bilstm_encoder",
        "transformer_encoder",
    ]

    for abl_name in ablation_order:
        if abl_name not in results or results[abl_name] is None:
            row = f"{ablation_display.get(abl_name, abl_name):<25s}" + "".join(f"{'N/A':>14s}" for _ in metric_keys)
        else:
            m = results[abl_name]
            vals = []
            for _, mkey in metric_keys:
                v = m.get(mkey, float("nan"))
                vals.append(f"{v:>14.4f}")
            row = f"{ablation_display.get(abl_name, abl_name):<25s}" + "".join(vals)
        lines.append(row)

    lines.append("-" * len(header))
    lines.append("")

    # Delta table (drop from full model)
    if "full_model" in results and results["full_model"] is not None:
        lines.append("")
        lines.append("PERFORMANCE DELTA (vs Full Model)")
        lines.append("-" * len(header))
        header2 = f"{'Condition':<25s}" + "".join(f"{'Δ'+n:>14s}" for n in header_names)
        lines.append(header2)
        lines.append("-" * len(header))

        base = results["full_model"]
        for abl_name in ablation_order[1:]:
            if abl_name not in results or results[abl_name] is None:
                continue
            m = results[abl_name]
            vals = []
            for _, mkey in metric_keys:
                bv = base.get(mkey, 0.0)
                av = m.get(mkey, 0.0)
                delta = av - bv
                vals.append(f"{delta:>+14.4f}")
            row = f"{ablation_display.get(abl_name, abl_name):<25s}" + "".join(vals)
            lines.append(row)
        lines.append("-" * len(header))

    lines.append("")
    lines.append("NOTE: For BERTScore, BLEU, ROUGE — higher is better. For WER — lower is better.")
    lines.append("Negative Δ for BERTScore/BLEU/ROUGE = removed component HELPED performance.")
    lines.append("Positive Δ for WER = removed component HELPED (lower WER is better).")

    return "\n".join(lines)


def format_markdown_table(results: dict) -> str:
    """Format results as a Markdown table for the README."""
    metric_keys = [
        ("BERTScore F1", "greedy_bertscore_f1"),
        ("BLEU-1", "greedy_bleu1"),
        ("BLEU-4", "greedy_bleu4"),
        ("ROUGE-1 F", "greedy_rouge1"),
        ("ROUGE-L F", "greedy_rougeL"),
        ("WER", "greedy_wer"),
    ]

    ablation_display = {
        "full_model":           "**Full Model (Ours)**",
        "no_contrastive_loss":  "− InfoNCE Loss",
        "no_attn_entropy":      "− Attn Entropy Reg.",
        "no_attention_gate":    "− Attention Gate",
        "no_word_dropout":      "− Word Dropout",
        "no_self_attn_dampening": "− Self-Attn Dampening",
        "no_disc_loss":         "− Disc. Loss",
        "linear_encoder":       "− S4D (Linear Proj.)",
        "bilstm_encoder":       "− S4D (BiLSTM)",
        "transformer_encoder":  "− S4D (Transformer)",
    }

    ablation_order = [
        "full_model",
        "no_contrastive_loss",
        "no_attn_entropy",
        "no_attention_gate",
        "no_word_dropout",
        "no_self_attn_dampening",
        "no_disc_loss",
        "linear_encoder",
        "bilstm_encoder",
        "transformer_encoder",
    ]

    lines = []
    lines.append("## Ablation Study Results")
    lines.append("")
    lines.append("Sentence-disjoint split, ZuCo 1.0, greedy decoding. Each row removes one component.")
    lines.append("")

    # Header
    header_names = [name for name, _ in metric_keys]
    header = "| Condition | " + " | ".join(header_names) + " |"
    sep = "|---|" + "|".join(["---:" for _ in header_names]) + "|"
    lines.append(header)
    lines.append(sep)

    base = results.get("full_model", {})
    for abl_name in ablation_order:
        if abl_name not in results or results[abl_name] is None:
            continue
        m = results[abl_name]
        display = ablation_display.get(abl_name, abl_name)
        vals = []
        for _, mkey in metric_keys:
            v = m.get(mkey, float("nan"))
            bv = base.get(mkey, v)
            if abl_name != "full_model" and abs(v - bv) > 0.001:
                if mkey == "greedy_wer":
                    # WER: lower is better, so increase = worse
                    if v > bv:
                        vals.append(f"{v:.4f} (↑{v-bv:+.3f})")
                    else:
                        vals.append(f"{v:.4f}")
                else:
                    if v < bv:
                        vals.append(f"{v:.4f} (↓{v-bv:+.3f})")
                    else:
                        vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v:.4f}")
        row = f"| {display} | " + " | ".join(vals) + " |"
        lines.append(row)

    lines.append("")
    lines.append("↓ = performance drop when component is removed (proves component helps)")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2 — EEG Reliance Test
# ════════════════════════════════════════════════════════════════════════════

def run_eeg_reliance_test(tokenizer, test_s):
    """
    Evaluate the full trained model under three EEG input conditions:
      1. Real EEG     — normal inference (already in full_model results)
      2. Shuffled EEG — swap EEG signals between samples in each batch
      3. Zero EEG     — replace all EEG with zeros

    No retraining. Uses checkpoints_sentence_split/best.pt.
    If performance drops strongly with shuffled/zero EEG, it proves
    the model genuinely uses the EEG signal.
    """
    CKPT = "Results/checkpoints_sentence_split/best.pt"
    NORM = "Results/checkpoints_sentence_split/eeg_norm_stats.npz"

    if not os.path.isfile(CKPT):
        print(f"[EEG Reliance] Checkpoint not found: {CKPT}")
        return None

    cfg = build_config({})
    cfg.checkpoint_dir = "Results/checkpoints_sentence_split"

    model = build_model(cfg, {})
    model.to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.self_attn_scale = cfg.self_attn_scale
    model.set_phase(2)
    model.eval()
    print(f"\n[EEG Reliance] Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"metric {ckpt.get('metric', 0):.4f})")

    test_ds = ZuCoEEGDataset(test_s, tokenizer, cfg.max_words, cfg.max_text_len)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False,
                         collate_fn=eeg_collate_fn, num_workers=cfg.num_workers,
                         pin_memory=True)

    conditions = {
        "real_eeg":     {"shuffle": False, "zero": False},
        "shuffled_eeg": {"shuffle": True,  "zero": False},
        "zero_eeg":     {"shuffle": False, "zero": True},
    }

    reliance_results = {}
    for cond_name, cond_opts in conditions.items():
        print(f"\n  [EEG Reliance] Condition: {cond_name}")
        all_preds, all_refs = [], []

        with torch.no_grad():
            for batch in tqdm(test_dl, desc=f"  {cond_name}"):
                eeg      = batch["eeg"].to(DEVICE)
                eeg_mask = batch["eeg_mask"].to(DEVICE)
                raw_texts = batch["raw_text"]
                all_refs.extend(raw_texts)

                if cond_opts["zero"]:
                    eeg = torch.zeros_like(eeg)
                elif cond_opts["shuffle"]:
                    perm = torch.randperm(eeg.size(0), device=DEVICE)
                    eeg      = eeg[perm]
                    eeg_mask = eeg_mask[perm]

                preds = model.generate_text(
                    eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                    max_length=cfg.max_gen_length, num_beams=1,
                    eeg_prior_alpha=0.0,
                    repetition_penalty=cfg.repetition_penalty,
                )
                all_preds.extend(preds)

        bleu  = compute_bleu(all_preds, all_refs)
        rouge = compute_rouge(all_preds, all_refs)
        wer   = compute_wer(all_preds, all_refs)
        print(f"  Computing BERTScore for {cond_name}...")
        bs    = compute_bertscore(all_preds, all_refs)
        reliance_results[cond_name] = {
            "greedy_bertscore_f1":        bs["bertscore_f1"],
            "greedy_bertscore_precision":  bs["bertscore_precision"],
            "greedy_bertscore_recall":     bs["bertscore_recall"],
            "greedy_bleu1":  bleu["bleu1"],
            "greedy_bleu2":  bleu["bleu2"],
            "greedy_bleu3":  bleu["bleu3"],
            "greedy_bleu4":  bleu["bleu4"],
            "greedy_rouge1":           rouge["rouge1"],
            "greedy_rouge1_precision":  rouge["rouge1_precision"],
            "greedy_rouge1_recall":     rouge["rouge1_recall"],
            "greedy_rougeL": rouge["rougeL"],
            "greedy_wer":    wer,
        }
        print(f"    BERTScore F1={bs['bertscore_f1']:.4f}  BLEU-1={bleu['bleu1']:.4f}  WER={wer:.4f}")

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return reliance_results


def format_reliance_table(reliance: dict) -> str:
    """Format EEG reliance test results as a plain-text table."""
    metric_keys = [
        ("BERTScore F1",  "greedy_bertscore_f1"),
        ("BLEU-1",        "greedy_bleu1"),
        ("BLEU-2",        "greedy_bleu2"),
        ("BLEU-3",        "greedy_bleu3"),
        ("BLEU-4",        "greedy_bleu4"),
        ("ROUGE-1 P",     "greedy_rouge1_precision"),
        ("ROUGE-1 R",     "greedy_rouge1_recall"),
        ("ROUGE-1 F",     "greedy_rouge1"),
        ("ROUGE-L F",     "greedy_rougeL"),
        ("WER",           "greedy_wer"),
    ]
    cond_display = {
        "real_eeg":     "Real EEG (Ours)",
        "shuffled_eeg": "Shuffled EEG",
        "zero_eeg":     "Zero EEG",
    }
    cond_order = ["real_eeg", "shuffled_eeg", "zero_eeg"]

    lines = []
    lines.append("=" * 90)
    lines.append("EXPERIMENT 2 — EEG RELIANCE TEST")
    lines.append("Proves the model genuinely uses EEG input (not just language model prior).")
    lines.append("Checkpoint: Results/checkpoints_sentence_split/best.pt (no retraining).")
    lines.append("Shuffled EEG: EEG from a different sample replaces the correct EEG.")
    lines.append("Zero EEG:     All-zeros EEG is fed; model has no brain signal to use.")
    lines.append("=" * 90)
    lines.append("")

    header_names = [n for n, _ in metric_keys]
    header = f"{'Condition':<22s}" + "".join(f"{n:>13s}" for n in header_names)
    lines.append(header)
    lines.append("-" * len(header))

    base = reliance.get("real_eeg", {})
    for cond in cond_order:
        if cond not in reliance:
            continue
        m = reliance[cond]
        vals = []
        for _, mkey in metric_keys:
            v = m.get(mkey, float("nan"))
            vals.append(f"{v:>13.4f}")
        row = f"{cond_display.get(cond, cond):<22s}" + "".join(vals)
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("")

    # Delta vs real EEG
    lines.append("DELTA vs Real EEG (drop proves model uses EEG)")
    lines.append("-" * len(header))
    header2 = f"{'Condition':<22s}" + "".join(f"{'Δ'+n:>13s}" for n in header_names)
    lines.append(header2)
    lines.append("-" * len(header))
    for cond in ["shuffled_eeg", "zero_eeg"]:
        if cond not in reliance:
            continue
        m = reliance[cond]
        vals = []
        for _, mkey in metric_keys:
            bv = base.get(mkey, 0.0)
            av = m.get(mkey, 0.0)
            vals.append(f"{av - bv:>+13.4f}")
        row = f"{cond_display.get(cond, cond):<22s}" + "".join(vals)
        lines.append(row)
    lines.append("-" * len(header))
    lines.append("")
    lines.append("Negative Δ for BERTScore/BLEU/ROUGE confirms EEG is the signal source.")
    lines.append("Positive Δ for WER confirms EEG removal degrades output quality.")
    return "\n".join(lines)


def format_reliance_markdown(reliance: dict) -> str:
    """Markdown table for the EEG reliance test."""
    metric_keys = [
        ("BERTScore F1", "greedy_bertscore_f1"),
        ("BLEU-1",       "greedy_bleu1"),
        ("BLEU-3",       "greedy_bleu3"),
        ("BLEU-4",       "greedy_bleu4"),
        ("ROUGE-1 P",    "greedy_rouge1_precision"),
        ("ROUGE-1 R",    "greedy_rouge1_recall"),
        ("ROUGE-1 F",    "greedy_rouge1"),
        ("WER",          "greedy_wer"),
    ]
    cond_display = {
        "real_eeg":     "**Real EEG (Ours)**",
        "shuffled_eeg": "Shuffled EEG",
        "zero_eeg":     "Zero EEG",
    }
    cond_order = ["real_eeg", "shuffled_eeg", "zero_eeg"]
    base = reliance.get("real_eeg", {})

    lines = []
    lines.append("## Experiment 2 — EEG Reliance Test")
    lines.append("")
    lines.append("Proves the model uses EEG. No retraining — only the input EEG is changed.")
    lines.append("")
    header = "| Condition | " + " | ".join(n for n, _ in metric_keys) + " |"
    sep    = "|---|" + "|".join(["---:" for _ in metric_keys]) + "|"
    lines.append(header)
    lines.append(sep)
    for cond in cond_order:
        if cond not in reliance:
            continue
        m = reliance[cond]
        display = cond_display.get(cond, cond)
        vals = []
        for _, mkey in metric_keys:
            v  = m.get(mkey, float("nan"))
            bv = base.get(mkey, v)
            if cond != "real_eeg" and abs(v - bv) > 0.001:
                if mkey == "greedy_wer":
                    arrow = f"(↑{v-bv:+.3f})" if v > bv else ""
                else:
                    arrow = f"(↓{v-bv:+.3f})" if v < bv else ""
                vals.append(f"{v:.4f} {arrow}".strip())
            else:
                vals.append(f"{v:.4f}")
        lines.append(f"| {display} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("↓ = drop versus real EEG — confirms EEG carries content signal")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("  EEG-TO-TEXT ABLATION STUDY")
    print("  Sentence-disjoint split, ZuCo 1.0")
    print("=" * 80)

    os.makedirs(ABLATION_DIR, exist_ok=True)

    # Load tokenizer
    tokenizer = BartTokenizer.from_pretrained(BART)

    # Load and split data ONCE (same split for all ablations)
    print("\nLoading and splitting data...")
    cfg_base = build_config({})
    train_s, dev_s, test_s, preprocessor = load_and_split_data(cfg_base)
    print(f"Train: {len(train_s)} | Dev: {len(dev_s)} | Test: {len(test_s)}")

    # Get ablation conditions
    ablations = get_ablation_configs()

    # Check which ablations are already done
    all_results = {}
    results_json = os.path.join(ABLATION_DIR, "ablation_results.json")
    if os.path.isfile(results_json):
        with open(results_json, "r") as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} existing results from {results_json}")

    # Full-model baseline: prefer re-evaluating checkpoint with CURRENT code/settings
    full_ckpt = "Results/checkpoints_sentence_split/best.pt"
    if os.path.isfile(full_ckpt):
        print("\nEvaluating full-model baseline from checkpoint with current evaluation pipeline...")
        cfg_full = build_config({})
        _, _, test_dl_full = build_dataloaders(
            cfg_full, train_s, dev_s, test_s, tokenizer,
        )
        all_results["full_model"] = evaluate_checkpoint(
            full_ckpt, cfg_full, test_dl_full, tokenizer, {},
        )
        print(f"  Full-model baseline (greedy BERTScore F1): "
              f"{all_results['full_model']['greedy_bertscore_f1']:.4f}")
    elif "full_model" not in all_results:
        all_results["full_model"] = load_existing_baseline()
        print("\nWarning: full-model checkpoint not found; using stored baseline values.")
        print(f"  Greedy BERTScore F1: {all_results['full_model']['greedy_bertscore_f1']:.4f}")

    # Run each ablation
    for name, cfg_overrides, model_overrides in ablations:
        if name in all_results and all_results[name] is not None:
            print(f"\n>> Skipping {name} (already completed)")
            continue

        metrics = run_single_ablation(
            name, cfg_overrides, model_overrides,
            tokenizer, train_s, dev_s, test_s, preprocessor,
        )
        all_results[name] = metrics

        # Save intermediate results (in case of crash)
        with open(results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Intermediate results saved to {results_json}")

    # ── Experiment 2: EEG Reliance Test ────────────────────────────────
    reliance_json = os.path.join(ABLATION_DIR, "eeg_reliance_results.json")
    if os.path.isfile(reliance_json):
        with open(reliance_json) as f:
            reliance_results = json.load(f)
        print(f"\n>> Loaded existing EEG reliance results from {reliance_json}")
    else:
        print("\n" + "=" * 80)
        print("  EXPERIMENT 2: EEG RELIANCE TEST")
        print("=" * 80)
        reliance_results = run_eeg_reliance_test(tokenizer, test_s)
        if reliance_results:
            with open(reliance_json, "w") as f:
                json.dump(reliance_results, f, indent=2)
            print(f"EEG reliance results saved to {reliance_json}")

    # ── Final ablation summary ──────────────────────────────────────────
    print("\n\n")
    table_txt = format_ablation_table(all_results)
    print(table_txt)

    out_txt = os.path.join(ABLATION_DIR, "ablation_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(table_txt)
    print(f"\nAblation table saved to {out_txt}")

    md_txt = format_markdown_table(all_results)
    out_md = os.path.join(ABLATION_DIR, "ablation_results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_txt)
    print(f"Ablation markdown saved to {out_md}")

    # ── EEG reliance summary ────────────────────────────────────────────
    if reliance_results:
        reliance_txt = format_reliance_table(reliance_results)
        print("\n\n" + reliance_txt)

        out_rel_txt = os.path.join(ABLATION_DIR, "eeg_reliance_results.txt")
        with open(out_rel_txt, "w", encoding="utf-8") as f:
            f.write(reliance_txt)
        print(f"\nReliance table saved to {out_rel_txt}")

        rel_md = format_reliance_markdown(reliance_results)
        out_rel_md = os.path.join(ABLATION_DIR, "eeg_reliance_results.md")
        with open(out_rel_md, "w", encoding="utf-8") as f:
            f.write(rel_md)
        print(f"Reliance markdown saved to {out_rel_md}")

    # ── Combined results file ───────────────────────────────────────────
    combined_path = os.path.join(ABLATION_DIR, "FULL_RESULTS.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(table_txt)
        f.write("\n\n\n")
        if reliance_results:
            f.write(format_reliance_table(reliance_results))
    print(f"\nCombined results saved to {combined_path}")

    # Save JSON results
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved to {results_json}")


if __name__ == "__main__":
    main()
