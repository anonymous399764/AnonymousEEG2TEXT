"""Controlled EEG encoder benchmark for NeuroLens.

This script answers one question only:

    With the data, split, BART decoder, losses, and training schedule held
    constant, which EEG encoder is best?

It compares the current S4D encoder with a second efficient diagonal SSM and
existing non-SSM baselines.  By design it does *not* mix different language
models into the same table.  A different decoder changes the task's language
prior and cross-attention behaviour, so that is a separate secondary study.

Examples (run from the Anonymized directory):

    # Quick functional smoke run: one seed, two epochs total
    python run_encoder_benchmark.py --models dss,mamba_tiny,lru_lite,s5_lite,h3_lite --seeds 42 --phase1-epochs 1 --phase2-epochs 1

    # Full experiment: three seeds, sentence-disjoint split
    python run_encoder_benchmark.py --models dss,mamba_tiny,lru_lite,s5_lite,h3_lite --seeds 42 --split-mode sentence

Results are written below Results/final_encoder_comparison_v6/.  Each variant receives
its own normalization statistics, checkpoints, epoch log, and test metrics.
The aggregate CSV/JSON files report mean and standard deviation across seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from eeg_to_text.config import Config
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples, split_samples_by_subject
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.evaluation.metrics import compute_bleu, compute_rouge, evaluate_model
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.models.ssm_baselines import BENCHMARK_ENCODERS, build_eeg_encoder
from eeg_to_text.training.trainer import Trainer


# The paper's complete deterministic metric suite.  Greedy generation is the
# pre-registered primary result; teacher-forced metrics are diagnostics, not a
# substitute for genuinely generating text from EEG.
PRIMARY_METRICS = tuple(
    f"{prefix}{metric}"
    for prefix in ("greedy_", "tf_")
    for metric in (
        "bleu1", "bleu2", "bleu3", "bleu4",
        "rouge1", "rouge1_precision", "rouge1_recall",
        "rouge2", "rouge2_precision", "rouge2_recall",
        "rougeL", "rougeL_precision", "rougeL_recall",
        "bertscore_f1", "bertscore_precision", "bertscore_recall", "wer",
    )
) + ("avg_cross_attn_entropy", "avg_eeg_text_cosine_sim")
MINIMAL_PAPER_METRICS = (
    "greedy_bleu1", "greedy_bleu2", "greedy_bleu3", "greedy_bleu4",
    "greedy_rouge1_precision", "greedy_rouge1",
)

# Capacity is not a quality metric, but it must be visible in the results.
# All variants share depth and hidden width where possible, while their exact
# parameter counts naturally differ because their mechanisms differ.
MODEL_SIZE_FIELDS = ("encoder_parameters", "total_trainable_parameters")
RUNTIME_FIELDS = ("training_wall_seconds", "test_evaluation_seconds", "total_run_seconds")
TIMING_FIELDS = (
    "dataloader_setup_seconds",
    "model_and_encoder_initialization_seconds",
    "trainer_initialization_seconds",
    "sanity_check_seconds",
    "phase1_training_seconds",
    "phase1_validation_seconds",
    "phase1_training_epochs",
    "phase1_validation_events",
    "phase2_training_seconds",
    "phase2_validation_seconds",
    "phase2_training_epochs",
    "phase2_validation_events",
    "training_loop_seconds",
    "training_wall_seconds",
    "checkpoint_reload_seconds",
    "primary_test_evaluation_seconds",
    "control_shuffled_evaluation_seconds",
    "control_zero_evaluation_seconds",
    "control_gaussian_evaluation_seconds",
    "test_evaluation_seconds",
    "total_run_seconds",
)
# v9 fixes masked teacher-forcing/discrimination bookkeeping.  It intentionally
# does not resume or reuse v8 checkpoints, which were trained with the older
# loss wiring.
IMPLEMENTATION_VERSION = "single_seed_ssm_b1_recipe_v9_masked_disc"
CHECKPOINT_SELECTION_METRIC = "validation greedy BLEU-1"

# Component ablations retain the S4D encoder and change exactly one
# NeuroLens mechanism.  Encoder replacements belong in --models and are
# reported as architecture baselines, not component ablations.
ABLATION_CONDITIONS = {
    "no_self_attn_dampening": ({"self_attn_scale": 1.0}, {}),
    "no_word_dropout": ({"word_dropout": 0.0, "word_dropout_start": 0.0, "word_dropout_end": 0.0}, {}),
    "no_contrastive_loss": ({"lambda_contrastive": 0.0}, {}),
    "no_disc_loss": ({"disable_disc_loss": True}, {}),
    "no_attn_entropy": ({"lambda_attn_entropy": 0.0}, {}),
    "no_attention_gate": ({}, {"bypass_gate": True}),
}


def format_duration(seconds: float) -> str:
    """Human-readable duration for progress messages and estimates."""
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(enabled: bool) -> None:
    """Choose a documented reproducibility/speed trade-off for all runs."""
    if enabled:
        # warn_only preserves a useful experiment rather than crashing because
        # an otherwise harmless CUDA kernel lacks a deterministic alternative.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[Reproducibility] Deterministic algorithms requested (warn-only).")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("[Reproducibility] Fast non-deterministic CUDA kernels allowed.")


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, value: object) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def collect_environment() -> dict:
    packages = {}
    for package in ("torch", "transformers", "numpy", "bert-score", "sacrebleu", "rouge-score", "jiwer"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def fingerprint_data_files(data_dir: str, task_files: list[str], include_hashes: bool) -> list[dict]:
    fingerprints = []
    for filename in task_files:
        path = os.path.abspath(os.path.join(data_dir, filename))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configured dataset file does not exist: {path}")
        entry = {"filename": filename, "path": path, "size_bytes": os.path.getsize(path)}
        if include_hashes:
            print(f"[Provenance] Hashing {filename} ...")
            entry["sha256"] = file_sha256(path)
        fingerprints.append(entry)
    return fingerprints


def fingerprint_source_tree() -> dict:
    """Hash the Python implementation used for this experiment invocation."""
    project_root = Path(__file__).resolve().parent
    source_files = [Path(__file__).resolve()] + sorted((project_root / "eeg_to_text").rglob("*.py"))
    digest = hashlib.sha256()
    entries = []
    for path in source_files:
        relative = path.relative_to(project_root).as_posix()
        file_digest = file_sha256(str(path))
        digest.update(f"{relative}:{file_digest}\n".encode("utf-8"))
        entries.append({"path": relative, "sha256": file_digest})
    return {"aggregate_sha256": digest.hexdigest(), "files": entries}


def parse_csv(value: str, valid: Iterable[str] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value")
    if valid is not None:
        unknown = sorted(set(items) - set(valid))
        if unknown:
            raise ValueError(f"Unknown values {unknown}; valid choices: {', '.join(valid)}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair NeuroLens EEG encoder benchmark")
    parser.add_argument("--data-dir", default="Data/pickle_file")
    parser.add_argument(
        "--models",
        default="dss,mamba_tiny,lru_lite,s5_lite,h3_lite",
        help=f"Comma-separated encoders. Choices: {', '.join(BENCHMARK_ENCODERS)}",
    )
    parser.add_argument(
        "--ablations",
        default=None,
        help=("Comma-separated S4D component ablations, or 'all': "
              + ", ".join(ABLATION_CONDITIONS)),
    )
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds")
    parser.add_argument("--split-mode", choices=["sentence", "subject"], default="sentence")
    parser.add_argument(
        "--results-dir", default="Results/ssm_single_seed_v1",
        help="Separate directory for the single-seed SSM study",
    )
    parser.add_argument("--bart-model", default="facebook/bart-base")
    parser.add_argument("--task-files", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 on CUDA")
    parser.add_argument(
        "--minimal-metrics", action="store_true",
        help="Evaluate only greedy BLEU-1/2/3/4 and ROUGE-1 P/F (no TF, BERTScore, controls, or beam).",
    )
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    # Validated on the project's RTX 5080 (16 GB). These larger micro-batches
    # preserve the original effective batch size of 128:
    # Phase 1: 128 x 1; Phase 2: 64 x 2.
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Phase-1 / evaluation micro-batch size (default: 128)")
    parser.add_argument("--phase2-batch-size", type=int, default=64,
                        help="Phase-2 micro-batch size (default: 64)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Phase-1 gradient accumulation (default: 1)")
    parser.add_argument("--phase2-grad-accum-steps", type=int, default=2,
                        help="Phase-2 gradient accumulation (default: 2)")
    parser.add_argument(
        "--estimate-full-seeds",
        type=int,
        default=1,
        help="Number of seeds in the planned full run used for the smoke-test time estimate",
    )
    parser.add_argument("--estimate-full-phase1-epochs", type=int, default=30)
    parser.add_argument("--estimate-full-phase2-epochs", type=int, default=70)
    # On Windows, worker processes are spawned and can duplicate the large
    # in-memory ZuCo samples.  Zero is slower but substantially safer; raise
    # this only after checking available system RAM.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true", help="Reuse an existing test_metrics.json")
    parser.add_argument(
        "--no-resume-incomplete", dest="resume_incomplete", action="store_false",
        help="Restart incomplete seeds instead of resuming a compatible checkpoints/last.pt",
    )
    parser.add_argument(
        "--no-report-controls", dest="report_controls", action="store_false",
        help="Skip shuffled, zero, and Gaussian EEG test controls (not recommended for final results)",
    )
    parser.add_argument(
        "--report-beam", action="store_true",
        help="Also save deterministic beam-search metrics/predictions; greedy remains the primary result",
    )
    parser.add_argument(
        "--no-save-predictions", dest="save_predictions", action="store_false",
        help="Do not save per-example JSONL predictions (not recommended for final results)",
    )
    parser.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false",
        help="Allow faster but non-deterministic CUDA operations",
    )
    parser.add_argument(
        "--no-hash-data-files", dest="hash_data_files", action="store_false",
        help="Skip SHA-256 checksums of source pickles",
    )
    parser.set_defaults(
        report_controls=True,
        save_predictions=True,
        deterministic=True,
        hash_data_files=True,
        resume_incomplete=True,
    )
    return parser.parse_args()


def prepare_splits(cfg: Config, split_mode: str):
    started = time.perf_counter()
    datasets = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    if not datasets:
        raise FileNotFoundError(
            f"No pickle files loaded from {cfg.data_dir}. Expected: {cfg.task_pickle_files}"
        )
    preprocessor = EEGPreprocessor(cfg.eeg_type, cfg.bands, cfg.n_channels)
    all_samples = preprocessor.extract_all_sentences_with_subjects(datasets, subject=cfg.subject)
    if not all_samples:
        raise RuntimeError("No usable EEG sentence samples were extracted")

    if split_mode == "sentence":
        plain = [(eeg, text) for eeg, text, _ in all_samples]
        train, dev, test = split_samples(plain, seed=cfg.seed)
    else:
        train, dev, test = split_samples_by_subject(all_samples, seed=cfg.seed)

    # This is done once from the training partition only, then the frozen
    # normalised samples are reused by every encoder.  That makes comparison
    # fair and prevents any validation/test leakage.
    preprocessor.fit([eeg for eeg, _ in train])

    def normalise_split(name, samples):
        return [
            (preprocessor.transform(eeg), text)
            for eeg, text in tqdm(samples, desc=f"Normalising {name}", unit="sentence")
        ]

    train = normalise_split("train", train)
    dev = normalise_split("dev", dev)
    test = normalise_split("test", test)
    elapsed = time.perf_counter() - started
    # This manifest contains only sentence hashes and membership statistics,
    # not EEG or text content.  It proves exactly which sentence instances
    # belonged to each partition without duplicating the dataset in Results/.
    split_membership = {}
    for name, samples in (("train", train), ("dev", dev), ("test", test)):
        text_counts = Counter(text for _, text in samples)
        split_membership[name] = {
            "n_samples": len(samples),
            "n_unique_sentences": len(text_counts),
            "sentence_hash_counts": [
                {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "count": count}
                for text, count in sorted(text_counts.items())
            ],
        }
    split_manifest = {
        "split_mode": split_mode,
        "split_seed": cfg.seed,
        "subject_filter": cfg.subject,
        "partitions": split_membership,
    }
    split_manifest["sha256"] = sha256_json(split_manifest)
    print(f"[Data] Preparation complete in {format_duration(elapsed)}")
    return train, dev, test, preprocessor, split_manifest, elapsed


def make_loaders(cfg: Config, train, dev, test, tokenizer):
    train_ds = ZuCoEEGDataset(
        train,
        tokenizer,
        cfg.max_words,
        cfg.max_text_len,
        augment=True,
        noise_std=cfg.eeg_noise_std,
        channel_drop=cfg.eeg_channel_drop,
        time_shift=cfg.eeg_time_shift,
        augmentation_seed=cfg.seed,
    )
    dev_ds = ZuCoEEGDataset(dev, tokenizer, cfg.max_words, cfg.max_text_len)
    test_ds = ZuCoEEGDataset(test, tokenizer, cfg.max_words, cfg.max_text_len)
    loader_options = dict(collate_fn=eeg_collate_fn, num_workers=cfg.num_workers, pin_memory=True)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed), **loader_options,
    )
    dev_dl = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, **loader_options)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_options)
    print(
        f"[DataLoader] train={len(train_dl)} batches, dev={len(dev_dl)} batches, "
        f"test={len(test_dl)} batches; batch_size={cfg.batch_size}, workers={cfg.num_workers}"
    )
    return train_dl, dev_dl, test_dl


@torch.no_grad()
def evaluate_minimal_greedy(model, loader, tokenizer, device, cfg: Config) -> dict:
    """Fast paper-table evaluator: greedy BLEU-1..4 and ROUGE-1 P/F only."""
    model.eval()
    predictions, references = [], []
    for batch in tqdm(loader, desc="Evaluating (minimal greedy)"):
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        references.extend(batch["raw_text"])
        predictions.extend(model.generate_text(
            eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
            max_length=cfg.max_gen_length, num_beams=1,
            eeg_prior_alpha=0.0, repetition_penalty=cfg.repetition_penalty,
        ))
    bleu = compute_bleu(predictions, references)
    rouge = compute_rouge(predictions, references)
    return {
        "greedy_bleu1": bleu["bleu1"], "greedy_bleu2": bleu["bleu2"],
        "greedy_bleu3": bleu["bleu3"], "greedy_bleu4": bleu["bleu4"],
        "greedy_rouge1_precision": rouge["rouge1_precision"],
        "greedy_rouge1": rouge["rouge1"],
    }


def evaluate_for_selection(model, loader, tokenizer, device, cfg: Config, minimal_metrics: bool = False):
    """Fastest valid validation setting: deterministic greedy generation only."""
    if minimal_metrics:
        return evaluate_minimal_greedy(model, loader, tokenizer, device, cfg)
    return evaluate_model(
        model,
        loader,
        tokenizer,
        device,
        num_beams=1,
        include_beam=False,
        max_gen_length=cfg.max_gen_length,
        print_examples=0,
        eeg_prior_alpha=0.0,
        repetition_penalty=cfg.repetition_penalty,
        gen_do_sample=False,
        best_of_n=0,
        mbr_n=0,
        contrastive_alpha=0.0,
    )


def evaluate_for_final_report(
    model, loader, tokenizer, device, cfg: Config, args: argparse.Namespace, run_dir: str, seed: int
) -> dict:
    """Evaluate the selected checkpoint with the paper's deterministic modes."""
    if args.minimal_metrics:
        return evaluate_minimal_greedy(model, loader, tokenizer, device, cfg)
    prediction_path = (
        os.path.join(run_dir, "reports", "predictions_real.jsonl") if args.save_predictions else None
    )
    return evaluate_model(
        model,
        loader,
        tokenizer,
        device,
        num_beams=cfg.num_beams if args.report_beam else 1,
        include_beam=args.report_beam,
        max_gen_length=cfg.max_gen_length,
        print_examples=0,
        eeg_prior_alpha=0.0,
        repetition_penalty=cfg.repetition_penalty,
        gen_do_sample=False,
        best_of_n=0,
        mbr_n=0,
        contrastive_alpha=0.0,
        eeg_condition="real",
        condition_seed=seed,
        save_predictions_path=prediction_path,
    )


def evaluate_eeg_reliance_controls(
    model, loader, tokenizer, device, cfg: Config, args: argparse.Namespace, run_dir: str, seed: int
) -> tuple[dict, dict]:
    """Run test-time controls that check whether results depend on EEG."""
    controls = {}
    timings = {}
    for condition in ("shuffled", "zero", "gaussian"):
        print(f"\n[EEG reliance control] {condition}")
        prediction_path = (
            os.path.join(run_dir, "reports", f"predictions_{condition}.jsonl")
            if args.save_predictions else None
        )
        started = time.perf_counter()
        controls[condition] = evaluate_model(
            model,
            loader,
            tokenizer,
            device,
            num_beams=1,
            include_beam=False,
            max_gen_length=cfg.max_gen_length,
            print_examples=0,
            eeg_prior_alpha=0.0,
            repetition_penalty=cfg.repetition_penalty,
            gen_do_sample=False,
            best_of_n=0,
            mbr_n=0,
            contrastive_alpha=0.0,
            eeg_condition=condition,
            condition_seed=seed,
            save_predictions_path=prediction_path,
        )
        timings[f"control_{condition}_evaluation_seconds"] = time.perf_counter() - started
    write_json(
        os.path.join(run_dir, "reports", "eeg_reliance_controls.json"),
        {"metrics": controls, "timing_seconds": timings},
    )
    return controls, timings


def run_one(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    train,
    dev,
    test,
    normalizer,
    tokenizer,
    split_manifest_sha256: str,
    environment: dict,
    ablation: str | None = None,
):
    run_started = time.perf_counter()
    set_seed(seed)
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.bart_model = args.bart_model
    cfg.seed = seed
    cfg.device = args.device
    cfg.fp16 = bool(args.fp16 and torch.cuda.is_available())
    cfg.num_workers = args.num_workers
    if args.task_files:
        cfg.task_pickle_files = args.task_files
    if args.phase1_epochs is not None:
        cfg.phase1_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg.phase2_epochs = args.phase2_epochs
    cfg.batch_size = args.batch_size
    cfg.phase2_batch_size = args.phase2_batch_size
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.phase2_grad_accum_steps = args.phase2_grad_accum_steps
    config_overrides, model_overrides = ABLATION_CONDITIONS.get(ablation, ({}, {}))
    for key, value in config_overrides.items():
        setattr(cfg, key, value)
    run_label = f"{variant}__{ablation}" if ablation else variant
    cfg.experiment_name = "final_s4d_ablation" if ablation else "final_encoder_comparison"
    cfg.encoder_variant = variant
    cfg.implementation_version = IMPLEMENTATION_VERSION
    cfg.split_manifest_sha256 = split_manifest_sha256
    cfg.checkpoint_selection_metric = CHECKPOINT_SELECTION_METRIC

    run_dir = os.path.join(args.results_dir, args.split_mode, run_label, f"seed_{seed}")
    cfg.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    cfg.log_dir = os.path.join(run_dir, "logs")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    normalizer.save_stats(os.path.join(cfg.checkpoint_dir, "eeg_norm_stats.npz"))
    metrics_path = os.path.join(run_dir, "test_metrics.json")
    if args.skip_completed and os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
        required_controls = {"shuffled", "zero", "gaussian"}
        controls_complete = (not args.report_controls) or required_controls.issubset(
            set(existing.get("eeg_reliance_controls", {}))
        )
        beam_complete = (not args.report_beam) or isinstance(existing.get("free_bertscore_f1"), (int, float))
        predictions_complete = (not args.save_predictions) or bool(existing.get("prediction_file"))
        if (
            existing.get("implementation_version") == IMPLEMENTATION_VERSION
            and controls_complete and beam_complete and predictions_complete
        ):
            print(f"[Reuse] {variant}, seed={seed}: {metrics_path}")
            return existing
        print(f"[Re-run] {variant}, seed={seed}: stored run lacks part of the requested final protocol.")

    # This is deliberately written before training.  If a run is interrupted,
    # its intended data, code settings, and hardware context are still known.
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg.experiment_name,
        "encoder": variant,
        "ablation": ablation,
        "seed": seed,
        "implementation_version": IMPLEMENTATION_VERSION,
        "split_mode": args.split_mode,
        "split_manifest_sha256": split_manifest_sha256,
        "resolved_config": asdict(cfg),
        "command_arguments": vars(args),
        "environment": environment,
        "checkpoint_policy": {
            "best_path": "checkpoints/best.pt",
            "last_path": "checkpoints/last.pt",
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "selection_partition": "development only",
        },
        "test_protocol": {
            "primary_decoding": "deterministic greedy (num_beams=1)",
            "optional_secondary_decoding": "deterministic beam" if args.report_beam else None,
            "eeg_prior_alpha": 0.0,
            "controls": ["real", "shuffled", "zero", "gaussian"] if args.report_controls else ["real"],
        },
    }
    write_json(os.path.join(run_dir, "run_manifest.json"), run_manifest)

    dataloader_setup_started = time.perf_counter()
    train_dl, dev_dl, test_dl = make_loaders(cfg, train, dev, test, tokenizer)
    dataloader_setup_seconds = time.perf_counter() - dataloader_setup_started

    model_setup_started = time.perf_counter()
    model = EEGToTextModel(
        bart_model_name=cfg.bart_model,
        eeg_input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim,
        s4d_layers=cfg.s4d_layers,
        s4d_state_dim=cfg.s4d_state_dim,
        s4d_dropout=cfg.s4d_dropout,
        s4d_bidirectional=True,
        gate_bias_init=cfg.gate_bias_init,
    )
    # Replace the default S4D only after BART has supplied its actual hidden
    # dimension.  This keeps parameter sizes valid for every BART checkpoint.
    model.eeg_encoder = build_eeg_encoder(
        variant,
        input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim,
        n_layers=cfg.s4d_layers,
        state_dim=cfg.s4d_state_dim,
        dropout=cfg.s4d_dropout,
        bart_dim=model.bart_dim,
        n_channels=cfg.n_channels,
        n_bands=len(cfg.bands),
    )
    if model_overrides.get("bypass_gate"):
        model.attention_gate = nn.Identity()
    encoder_parameters = sum(parameter.numel() for parameter in model.eeg_encoder.parameters())
    total_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model_and_encoder_initialization_seconds = time.perf_counter() - model_setup_started

    trainer_setup_started = time.perf_counter()
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=dev_dl,
        tokenizer=tokenizer,
        config=cfg,
        evaluate_fn=lambda m, dl, tok, device: evaluate_for_selection(
            m, dl, tok, device, cfg, args.minimal_metrics
        ),
    )
    trainer_initialization_seconds = time.perf_counter() - trainer_setup_started
    print(f"\n{'=' * 80}\nENCODER={variant}  SEED={seed}  SPLIT={args.split_mode}\n{'=' * 80}")
    training_started = time.perf_counter()
    resume_path = os.path.join(cfg.checkpoint_dir, "last.pt")
    compatible_resume_path = None
    if args.resume_incomplete and os.path.isfile(resume_path):
        try:
            resume_metadata = torch.load(resume_path, map_location="cpu", weights_only=True)
            compatible = (
                resume_metadata.get("implementation_version") == IMPLEMENTATION_VERSION
                and resume_metadata.get("encoder_variant") == variant
                and resume_metadata.get("split_manifest_sha256") == split_manifest_sha256
                and resume_metadata.get("config", {}).get("phase1_epochs") == cfg.phase1_epochs
                and resume_metadata.get("config", {}).get("phase2_epochs") == cfg.phase2_epochs
            )
            if compatible:
                compatible_resume_path = resume_path
                print(f"[Resume] Continuing compatible checkpoint: {resume_path}")
            else:
                print("[Resume] Ignoring incompatible last.pt; starting this seed cleanly.")
        except Exception as error:
            print(f"[Resume] Could not inspect last.pt ({error}); starting this seed cleanly.")
    trainer.train(resume_path=compatible_resume_path)
    training_wall_seconds = time.perf_counter() - training_started
    trainer_timings = dict(trainer.timing)

    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    checkpoint_phase = 2
    best_ckpt = {}
    checkpoint_reload_started = time.perf_counter()
    if os.path.isfile(best_path):
        best_ckpt = trainer.load_checkpoint(best_path)
        checkpoint_phase = best_ckpt.get("phase", 2)
    checkpoint_reload_seconds = time.perf_counter() - checkpoint_reload_started
    model.self_attn_scale = cfg.self_attn_scale
    model.set_phase(checkpoint_phase)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    test_started = time.perf_counter()
    primary_test_started = time.perf_counter()
    metrics = evaluate_for_final_report(model, test_dl, tokenizer, device, cfg, args, run_dir, seed)
    primary_test_evaluation_seconds = time.perf_counter() - primary_test_started
    controls, control_timings = evaluate_eeg_reliance_controls(
        model, test_dl, tokenizer, device, cfg, args, run_dir, seed
    ) if args.report_controls else ({}, {})
    test_evaluation_seconds = time.perf_counter() - test_started
    total_run_seconds = time.perf_counter() - run_started
    component_timings = {
        "dataloader_setup_seconds": dataloader_setup_seconds,
        "model_and_encoder_initialization_seconds": model_and_encoder_initialization_seconds,
        "trainer_initialization_seconds": trainer_initialization_seconds,
        **trainer_timings,
        "training_wall_seconds": training_wall_seconds,
        "checkpoint_reload_seconds": checkpoint_reload_seconds,
        "primary_test_evaluation_seconds": primary_test_evaluation_seconds,
        **control_timings,
        "test_evaluation_seconds": test_evaluation_seconds,
        "total_run_seconds": total_run_seconds,
    }
    metrics.update(
        {
            "encoder": run_label,
            "base_encoder": variant,
            "ablation": ablation,
            "seed": seed,
            "split_mode": args.split_mode,
            "encoder_parameters": encoder_parameters,
            "total_trainable_parameters": total_trainable_parameters,
            "epochs_completed": cfg.total_epochs(),
            "selected_checkpoint_phase": checkpoint_phase,
            "selected_checkpoint_epoch": best_ckpt.get("epoch"),
            "selected_validation_greedy_bleu1": best_ckpt.get("metric"),
            "implementation_version": IMPLEMENTATION_VERSION,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "teacher_forcing_protocol": (
                "not computed (--minimal-metrics)"
                if args.minimal_metrics else "teacher_forced_argmax_full_self_attention"
            ),
            "split_manifest_sha256": split_manifest_sha256,
            "s4d_parameterization": "stable_log_decay_v1",
            **component_timings,
            "eeg_reliance_controls": controls,
        }
    )
    os.makedirs(run_dir, exist_ok=True)
    write_json(metrics_path, metrics)
    # Compact, paper-ready artifact for the requested single-seed SSM table.
    # The complete metrics file is retained for auditability, but this file
    # contains only the six reported greedy-decoding values.
    write_json(
        os.path.join(run_dir, "paper_metrics.json"),
        {
            "encoder": run_label,
            "seed": seed,
            "split_mode": args.split_mode,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            **{key: metrics.get(key) for key in MINIMAL_PAPER_METRICS},
        },
    )
    write_json(os.path.join(run_dir, "config.json"), asdict(cfg))
    write_json(os.path.join(run_dir, "component_timings.json"), {
        "seconds": component_timings,
        "human_readable": {
            key: format_duration(value)
            for key, value in component_timings.items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
        },
    })
    checkpoint_index = {
        "best_validation_checkpoint": {
            "path": "checkpoints/best.pt",
            "selection_metric": CHECKPOINT_SELECTION_METRIC,
            "selected_epoch": best_ckpt.get("epoch"),
            "validation_metric": best_ckpt.get("metric"),
            "phase": checkpoint_phase,
        },
        "last_resume_checkpoint": {"path": "checkpoints/last.pt"},
        "normalization_statistics": "checkpoints/eeg_norm_stats.npz",
    }
    write_json(os.path.join(run_dir, "checkpoint_index.json"), checkpoint_index)
    run_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["artifacts"] = {
        "test_metrics": "test_metrics.json",
        "configuration": "config.json",
        "checkpoint_index": "checkpoint_index.json",
        "component_timings": "component_timings.json",
        "primary_predictions": "reports/predictions_real.jsonl" if args.save_predictions else None,
        "controls": "reports/eeg_reliance_controls.json" if args.report_controls else None,
    }
    write_json(os.path.join(run_dir, "run_manifest.json"), run_manifest)
    print(
        f"[Timing] {variant}, seed={seed}: training={format_duration(training_wall_seconds)}, "
        f"test={format_duration(test_evaluation_seconds)}, total={format_duration(total_run_seconds)}"
    )
    return metrics


def write_runtime_estimate(rows: list[dict], args: argparse.Namespace, variants: list[str]) -> dict:
    """Project smoke-test timings onto the planned full three-seed run."""
    target_epochs = args.estimate_full_phase1_epochs + args.estimate_full_phase2_epochs
    by_encoder: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if all(isinstance(row.get(field), (int, float)) for field in RUNTIME_FIELDS):
            by_encoder[row["encoder"]].append(row)

    encoder_estimates = {}
    total_seconds = 0.0
    for encoder in variants:
        runs = by_encoder.get(encoder, [])
        if not runs:
            continue
        training_per_epoch = mean(row["training_wall_seconds"] / row["epochs_completed"] for row in runs)
        test_seconds = mean(row["test_evaluation_seconds"] for row in runs)
        estimate = args.estimate_full_seeds * (training_per_epoch * target_epochs + test_seconds)
        encoder_estimates[encoder] = {
            "estimated_seconds": estimate,
            "estimated_human": format_duration(estimate),
            "based_on_runs": len(runs),
        }
        total_seconds += estimate

    return {
        "planned_seeds": args.estimate_full_seeds,
        "planned_phase1_epochs": args.estimate_full_phase1_epochs,
        "planned_phase2_epochs": args.estimate_full_phase2_epochs,
        "estimated_total_seconds": total_seconds,
        "estimated_total_human": format_duration(total_seconds),
        "per_encoder": encoder_estimates,
    }


def write_summary(
    rows: list[dict],
    results_dir: str,
    split_mode: str,
    args: argparse.Namespace,
    variants: list[str],
) -> None:
    summary_dir = os.path.join(results_dir, split_mode)
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, "all_seed_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["encoder"]].append(row)

    summary_rows = []
    for encoder, runs in sorted(grouped.items()):
        summary = {"encoder": encoder, "n_seeds": len(runs)}
        for metric in PRIMARY_METRICS:
            values = [float(r[metric]) for r in runs if isinstance(r.get(metric), (int, float))]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        for field in MODEL_SIZE_FIELDS:
            values = [int(r[field]) for r in runs if isinstance(r.get(field), (int, float))]
            if values:
                # Same architecture is used for every seed.  Keeping this in
                # the aggregate file makes the comparison auditable.
                summary[field] = values[0]
        for field in TIMING_FIELDS:
            values = [float(r[field]) for r in runs if isinstance(r.get(field), (int, float))]
            if values:
                summary[f"{field}_mean"] = mean(values)
                summary[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0
        values = [
            float(r["selected_validation_greedy_bleu1"])
            for r in runs if isinstance(r.get("selected_validation_greedy_bleu1"), (int, float))
        ]
        if values:
            summary["selected_validation_greedy_bleu1_mean"] = mean(values)
            summary["selected_validation_greedy_bleu1_std"] = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    with open(os.path.join(summary_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, sort_keys=True)
    fields = sorted({key for row in summary_rows for key in row})
    with open(os.path.join(summary_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Keep EEG-reliance controls in a separate compact table.  The main table
    # remains the real-EEG encoder comparison, while this table answers the
    # crucial question: did performance fall when the EEG was destroyed?
    control_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        for condition, values in row.get("eeg_reliance_controls", {}).items():
            if isinstance(values, dict):
                control_groups[(row["encoder"], condition)].append(values)
    control_rows = []
    for (encoder, condition), runs in sorted(control_groups.items()):
        control_row = {"encoder": encoder, "condition": condition, "n_seeds": len(runs)}
        for metric in PRIMARY_METRICS:
            values = [float(run[metric]) for run in runs if isinstance(run.get(metric), (int, float))]
            if values:
                control_row[f"{metric}_mean"] = mean(values)
                control_row[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        control_rows.append(control_row)
    if control_rows:
        write_json(os.path.join(summary_dir, "eeg_reliance_summary.json"), control_rows)
        control_fields = sorted({key for row in control_rows for key in row})
        with open(os.path.join(summary_dir, "eeg_reliance_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=control_fields)
            writer.writeheader()
            writer.writerows(control_rows)

    runtime_estimate = write_runtime_estimate(rows, args, variants)
    runtime_path = os.path.join(summary_dir, "runtime_estimate.json")
    with open(runtime_path, "w", encoding="utf-8") as f:
        json.dump(runtime_estimate, f, indent=2, sort_keys=True)

    print("\nENCODER BENCHMARK SUMMARY")
    for row in summary_rows:
        print(
            f"{row['encoder']:<20} "
            f"BERTScore={row.get('greedy_bertscore_f1_mean', float('nan')):.4f} ± "
            f"{row.get('greedy_bertscore_f1_std', float('nan')):.4f}  "
            f"BLEU-4={row.get('greedy_bleu4_mean', float('nan')):.4f} ± "
            f"{row.get('greedy_bleu4_std', float('nan')):.4f}"
        )
    print(
        "\nSMOKE-TEST TIME ESTIMATE "
        f"({runtime_estimate['planned_seeds']} seeds, "
        f"{runtime_estimate['planned_phase1_epochs']}+{runtime_estimate['planned_phase2_epochs']} epochs): "
        f"{runtime_estimate['estimated_total_human']}"
    )
    print(f"Detailed estimate: {runtime_path}")


def main() -> None:
    experiment_started = time.perf_counter()
    args = parse_args()
    if args.minimal_metrics:
        # A minimal paper-table study intentionally omits every expensive
        # diagnostic and decoding mode beyond greedy generation.
        args.report_controls = False
        args.report_beam = False
        args.save_predictions = False
    variants = parse_csv(args.models, BENCHMARK_ENCODERS)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    if args.ablations:
        ablations = list(ABLATION_CONDITIONS) if args.ablations.strip().lower() == "all" else parse_csv(
            args.ablations, ABLATION_CONDITIONS
        )
        if variants != ["s4d"]:
            raise ValueError("--ablations requires --models s4d so every condition keeps the same encoder.")
    else:
        ablations = [None]
    configure_reproducibility(args.deterministic)

    base_cfg = Config(data_dir=args.data_dir, bart_model=args.bart_model, seed=seeds[0])
    if args.task_files:
        base_cfg.task_pickle_files = args.task_files
    data_fingerprint_started = time.perf_counter()
    data_fingerprints = fingerprint_data_files(
        args.data_dir, base_cfg.task_pickle_files, args.hash_data_files
    )
    data_fingerprint_seconds = time.perf_counter() - data_fingerprint_started
    train, dev, test, normalizer, split_manifest, data_preparation_seconds = prepare_splits(
        base_cfg, args.split_mode
    )
    tokenizer_started = time.perf_counter()
    tokenizer = BartTokenizer.from_pretrained(args.bart_model)
    tokenizer_loading_seconds = time.perf_counter() - tokenizer_started
    summary_dir = os.path.join(args.results_dir, args.split_mode)
    write_json(os.path.join(summary_dir, "split_manifest.json"), split_manifest)
    experiment_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "final_encoder_comparison",
        "implementation_version": IMPLEMENTATION_VERSION,
        "command_arguments": vars(args),
        "encoders": variants,
        "ablations": [item for item in ablations if item],
        "seeds": seeds,
        "data_fingerprints": data_fingerprints,
        "source_fingerprint": fingerprint_source_tree(),
        "split_manifest": "split_manifest.json",
        "split_manifest_sha256": split_manifest["sha256"],
        "environment": collect_environment(),
        "setup_timing_seconds": {
            "data_fingerprint_seconds": data_fingerprint_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "tokenizer_loading_seconds": tokenizer_loading_seconds,
        },
        "primary_outcome": {
            "metric": CHECKPOINT_SELECTION_METRIC,
            "selection": "best development checkpoint only",
            "test_reporting": "single-seed report unless multiple seeds are explicitly requested",
        },
    }
    write_json(os.path.join(summary_dir, "experiment_manifest.json"), experiment_manifest)
    print(f"Data: train={len(train)}, dev={len(dev)}, test={len(test)}")
    print(f"Encoders: {variants}; ablations: {[item for item in ablations if item] or 'none'}; seeds: {seeds}; decoder fixed to {args.bart_model}")
    print(f"[Provenance] Split manifest SHA-256: {split_manifest['sha256']}")

    rows = []
    report_labels = []
    for variant in variants:
        for ablation in ablations:
            report_label = f"{variant}__{ablation}" if ablation else variant
            report_labels.append(report_label)
            for seed in seeds:
                rows.append(
                    run_one(
                        variant, seed, args, train, dev, test, normalizer, tokenizer,
                        split_manifest["sha256"], experiment_manifest["environment"], ablation,
                    )
                )
    write_summary(rows, args.results_dir, args.split_mode, args, report_labels)
    minimal_csv = os.path.join(summary_dir, "paper_metrics.csv")
    with open(minimal_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("encoder", "seed", *MINIMAL_PAPER_METRICS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(f"Minimal paper metrics: {minimal_csv}")
    write_json(os.path.join(summary_dir, "experiment_timings.json"), {
        "seconds": {
            "data_fingerprint_seconds": data_fingerprint_seconds,
            "data_preparation_seconds": data_preparation_seconds,
            "tokenizer_loading_seconds": tokenizer_loading_seconds,
            "sum_of_serial_run_seconds": sum(
                row.get("total_run_seconds", 0.0) for row in rows if isinstance(row.get("total_run_seconds"), (int, float))
            ),
            "experiment_wall_seconds": time.perf_counter() - experiment_started,
        },
    })


if __name__ == "__main__":
    main()
