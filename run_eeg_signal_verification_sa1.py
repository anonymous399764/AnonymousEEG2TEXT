"""EEG signal-verification evaluation for the requested alpha_sa=1 checkpoint.

This reproduces the previous sentence-disjoint reliance controls with greedy,
no-teacher-forcing decoding.  The held-out partition contains 105 unique
sentences represented by 1,316 EEG trials.  Outputs are written beneath
``ablation results/sensitivity_ablation/sa_scale_1.0/eeg_signal_verification``.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Evaluation is fully reproducible from the already-cached BART artefacts.
# Prevent Transformers from probing Hugging Face for newer metadata first.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import BartTokenizer

from eeg_to_text.config import Config
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.evaluation.metrics import compute_bleu, compute_rouge
from eeg_to_text.models.eeg_to_text import EEGToTextModel


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "ablation results" / "sensitivity_ablation" / "sa_scale_1.0" / "best.pt"
NORM_STATS = CHECKPOINT.parent / "eeg_norm_stats.npz"
DATA_DIR = Path(r"C:\Users\LOQ\Desktop\decoders\decoder_ablation_hpo03\data")
OUT_DIR = CHECKPOINT.parent / "eeg_signal_verification"
SEED = 42
BATCH_SIZE = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_test_loader(cfg: Config, tokenizer: BartTokenizer) -> tuple[DataLoader, int]:
    data = load_pickle_datasets(str(DATA_DIR), cfg.task_pickle_files)
    if len(data) != len(cfg.task_pickle_files):
        raise FileNotFoundError(f"Expected all ZuCo pickle files in {DATA_DIR}")

    preprocessor = EEGPreprocessor(
        eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels
    )
    samples = preprocessor.extract_all_sentences(data, subject=cfg.subject)
    train, _, test = split_samples(samples, train_ratio=0.8, dev_ratio=0.1, seed=SEED)
    del train  # Normalisation stats are loaded, never fitted on evaluation data.
    preprocessor.load_stats(str(NORM_STATS))
    test = [(preprocessor.transform(eeg), text) for eeg, text in test]

    test_dataset = ZuCoEEGDataset(test, tokenizer, cfg.max_words, cfg.max_text_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=eeg_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader, len({text for _, text in test})


@torch.no_grad()
def decode_condition(
    name: str,
    model: EEGToTextModel,
    loader: DataLoader,
    tokenizer: BartTokenizer,
    device: torch.device,
    cfg: Config,
) -> tuple[list[str], list[str]]:
    """Decode one condition using the historical batchwise shuffle control."""
    refs: list[str] = []
    predictions: list[str] = []
    noise_rng = np.random.RandomState(SEED)

    # Match the earlier verification convention: reset then use torch.randperm
    # for every batch (and apply the same permutation to its EEG padding mask).
    if name == "shuffled_eeg":
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        eeg_mask = batch["eeg_mask"].to(device, non_blocking=True)
        refs.extend(batch["raw_text"])

        if name == "shuffled_eeg":
            permutation = torch.randperm(eeg.size(0), device=device)
            eeg, eeg_mask = eeg[permutation], eeg_mask[permutation]
        elif name == "gaussian_noise":
            noise = noise_rng.normal(0.0, 1.0, size=tuple(eeg.shape)).astype(np.float32)
            eeg = torch.from_numpy(noise).to(device, non_blocking=True)
        elif name == "zero_eeg":
            eeg = torch.zeros_like(eeg)

        predictions.extend(
            model.generate_text(
                eeg=eeg,
                eeg_mask=eeg_mask,
                tokenizer=tokenizer,
                max_length=cfg.max_gen_length,
                num_beams=1,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                repetition_penalty=cfg.repetition_penalty,
                eeg_prior_alpha=0.0,
            )
        )

    return predictions, refs


def metric_subset(predictions: list[str], refs: list[str]) -> dict[str, float]:
    bleu = compute_bleu(predictions, refs)
    rouge = compute_rouge(predictions, refs)
    return OrderedDict(
        bleu1=float(bleu["bleu1"]),
        bleu2=float(bleu["bleu2"]),
        bleu3=float(bleu["bleu3"]),
        bleu4=float(bleu["bleu4"]),
        rougeL=float(rouge["rougeL"]),
    )


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not NORM_STATS.is_file():
        raise FileNotFoundError(NORM_STATS)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(bart_model="facebook/bart-base", bart_dim=768)
    tokenizer = BartTokenizer.from_pretrained(cfg.bart_model, local_files_only=True)
    loader, n_unique_sentences = build_test_loader(cfg, tokenizer)

    model = EEGToTextModel(
        bart_model_name=cfg.bart_model,
        eeg_input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim,
        s4d_layers=cfg.s4d_layers,
        s4d_state_dim=cfg.s4d_state_dim,
        s4d_dropout=cfg.s4d_dropout,
        s4d_bidirectional=cfg.s4d_bidirectional,
        gate_bias_init=cfg.gate_bias_init,
    ).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.self_attn_scale = 1.0
    model.set_phase(2)
    model.eval()

    print(
        f"Checkpoint epoch {checkpoint.get('epoch', '?')} on {device}; "
        f"{len(loader.dataset)} EEG trials / {n_unique_sentences} unique held-out sentences."
    )
    all_predictions: dict[str, list[str]] = {}
    references: list[str] | None = None
    metrics: dict[str, dict[str, float]] = {}
    for condition in ("real_eeg", "shuffled_eeg", "gaussian_noise", "zero_eeg"):
        print(f"Decoding {condition} ...", flush=True)
        preds, refs = decode_condition(condition, model, loader, tokenizer, device, cfg)
        if references is None:
            references = refs
        elif refs != references:
            raise RuntimeError("Reference order changed between conditions")
        all_predictions[condition] = preds
        metrics[condition] = metric_subset(preds, refs)
        print("  " + ", ".join(f"{key}={value:.6f}" for key, value in metrics[condition].items()))

    assert references is not None
    exact_matches = sum(
        real.strip() == shuffled.strip()
        for real, shuffled in zip(all_predictions["real_eeg"], all_predictions["shuffled_eeg"])
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "real_vs_shuffled_decoded_sentences.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "reference", "real_eeg", "shuffled_eeg", "exact_match"],
        )
        writer.writeheader()
        for index, (ref, real, shuffled) in enumerate(
            zip(references, all_predictions["real_eeg"], all_predictions["shuffled_eeg"])
        ):
            writer.writerow(
                {
                    "sample_index": index,
                    "reference": ref,
                    "real_eeg": real,
                    "shuffled_eeg": shuffled,
                    "exact_match": real.strip() == shuffled.strip(),
                }
            )

    output = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "self_attn_scale": 1.0,
        "decoding": {
            "mode": "greedy",
            "num_beams": 1,
            "teacher_forcing": False,
            "eeg_prior_alpha": 0.0,
            "repetition_penalty": cfg.repetition_penalty,
            "no_repeat_ngram_size": 3,
        },
        "dataset": {
            "data_dir": str(DATA_DIR),
            "normalization_stats": str(NORM_STATS),
            "split_seed": SEED,
            "n_eeg_trials": len(references),
            "n_unique_test_sentences": n_unique_sentences,
        },
        "controls": {
            "shuffled_eeg": "batchwise torch.randperm with seed 42; EEG masks are permuted identically",
            "gaussian_noise": "N(0, 1) in normalized EEG space, NumPy RandomState(42)",
            "zero_eeg": "all normalized EEG values set to zero; original EEG mask retained",
        },
        "metrics": metrics,
        "real_shuffled_exact_output_matches": exact_matches,
        "real_shuffled_exact_output_match_rate": exact_matches / len(references),
        "decoded_sentences_csv": str(csv_path),
    }
    results_path = OUT_DIR / "eeg_signal_verification_results.json"
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Exact real/shuffled output matches: {exact_matches}/{len(references)}")
    print(f"Saved: {results_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
