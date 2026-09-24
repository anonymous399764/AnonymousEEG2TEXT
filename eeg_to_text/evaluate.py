"""
evaluate.py — Entry point for standalone evaluation of a trained checkpoint.

Usage:
    python -m eeg_to_text.evaluate --checkpoint runs/ckpts/best.pt
    python -m eeg_to_text.evaluate --checkpoint runs/ckpts/best.pt --split test
    python -m eeg_to_text.evaluate --checkpoint runs/ckpts/best.pt --test_input noise

Run from the workspace root (c:/kaggle2/EEG-To-text/).
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples_by_subject, split_samples
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.evaluation.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EEG-to-Text checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="test",
                        choices=["dev", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_gen_length", type=int, default=56)
    parser.add_argument("--print_examples", type=int, default=10)
    parser.add_argument("--bart_model", type=str, default=None,
                        help="Override BART model name (must match checkpoint)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (default: alongside checkpoint)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_files", nargs="+", default=None,
                        help="Optional list of pickle files to load from data_dir")
    parser.add_argument("--split_mode", type=str, default="subject",
                        choices=["subject", "sentence"],
                        help="Split strategy to mirror training")
    parser.add_argument("--test_input", type=str, default="real",
                        choices=["real", "noise"],
                        help="Use real EEG or Gaussian noise as input")
    parser.add_argument("--noise_std", type=float, default=1.0,
                        help="Std of Gaussian noise in normalized EEG space")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.device = args.device
    cfg.seed = args.seed
    if args.task_files is not None:
        cfg.task_pickle_files = args.task_files

    if args.bart_model:
        cfg.bart_model = args.bart_model
        if "base" in args.bart_model:
            cfg.bart_dim = 768

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ── Load data ───────────────────────────────────────────────────────
    print("Loading data...")
    dataset_dicts = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    if not dataset_dicts:
        print(f"ERROR: No datasets in {cfg.data_dir}")
        sys.exit(1)

    preprocessor = EEGPreprocessor(eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels)
    all_samples = preprocessor.extract_all_sentences_with_subjects(dataset_dicts, subject=cfg.subject)

    # Split strategy (must match training split)
    if args.split_mode == "sentence":
        plain_samples = [(e, t) for e, t, _ in all_samples]
        train_samples, dev_samples, test_samples = split_samples(plain_samples, seed=cfg.seed)
    else:
        train_samples, dev_samples, test_samples = split_samples_by_subject(all_samples, seed=cfg.seed)
    eval_samples = test_samples if args.split == "test" else dev_samples

    # Try to load saved normalisation stats
    ckpt_dir = os.path.dirname(args.checkpoint)
    stats_path = os.path.join(ckpt_dir, "eeg_norm_stats.npz")
    if os.path.isfile(stats_path):
        print(f"Loading normalisation stats from {stats_path}")
        preprocessor.load_stats(stats_path)
    else:
        # Fallback: fit on training data
        print("WARNING: No saved normalisation stats found, fitting on training data")
        preprocessor.fit([s[0] for s in train_samples])

    eval_samples = [(preprocessor.transform(e), t) for e, t in eval_samples]

    if args.test_input == "noise":
        rng = np.random.RandomState(cfg.seed)
        noise_samples = []
        for eeg, text in eval_samples:
            noise = rng.normal(loc=0.0, scale=args.noise_std, size=eeg.shape).astype(np.float32)
            noise_samples.append((noise, text))
        eval_samples = noise_samples
        print("Using Gaussian noise input (normalized space)")

    # ── Tokenizer + Dataset ─────────────────────────────────────────────
    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model)
    eval_ds = ZuCoEEGDataset(eval_samples, tokenizer, cfg.max_words, cfg.max_text_len)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=eeg_collate_fn, num_workers=2, pin_memory=True,
    )
    print(f"Evaluating {len(eval_ds)} samples from '{args.split}' split")

    # ── Model ───────────────────────────────────────────────────────────
    print(f"Building model ({cfg.bart_model})...")
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
    model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Epoch: {ckpt.get('epoch', '?')}  |  Metric: {ckpt.get('metric', '?')}")

    # ── Evaluate ────────────────────────────────────────────────────────
    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.num_beams,
        max_gen_length=args.max_gen_length,
        print_examples=args.print_examples,
    )

    # ── Save results ────────────────────────────────────────────────────
    out_path = args.output or os.path.join(ckpt_dir, f"eval_{args.split}_results.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for k, v in sorted(metrics.items()):
            f.write(f"{k}: {v}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
