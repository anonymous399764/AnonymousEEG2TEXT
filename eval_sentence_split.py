"""
Evaluate checkpoints_sentence_split/best.pt (ZuCo 1.0 only, sentence-level split).
Trained till epoch 54, best checkpoint at epoch 43.

Outputs:
  1. Results/sentence_split_predictions_ranked.txt  — predictions ranked by greedy BERTScore
  2. Results/sentence_split_metrics.txt              — all aggregate metrics
"""
import torch, os, sys, csv
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from eeg_to_text.evaluation.metrics import compute_bleu, compute_rouge, compute_wer
from transformers import BartTokenizer
import torch.nn.functional as F

BART = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "Results/checkpoints_sentence_split"
CKPT = os.path.join(CKPT_DIR, "best.pt")
OUT_DIR = "Results"
OUT_TXT = os.path.join(OUT_DIR, "sentence_split_predictions_ranked.txt")
OUT_METRICS = os.path.join(OUT_DIR, "sentence_split_metrics.txt")

# ZuCo 1.0 only
TASK_FILES = [
    "task1-SR-dataset.pickle",
    "task2-NR-dataset.pickle",
    "task3-TSR-dataset.pickle",
]


def main():
    cfg = Config(bart_model=BART, bart_dim=768)
    tokenizer = BartTokenizer.from_pretrained(BART)

    # ── Load model ──────────────────────────────────────────────────────
    model = EEGToTextModel(
        bart_model_name=BART,
        eeg_input_dim=cfg.eeg_feature_dim,
        s4d_dim=cfg.s4d_dim, s4d_layers=cfg.s4d_layers,
        s4d_state_dim=cfg.s4d_state_dim, s4d_dropout=cfg.s4d_dropout,
        s4d_bidirectional=cfg.s4d_bidirectional,
        gate_bias_init=cfg.gate_bias_init,
    ).to(DEVICE)

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.self_attn_scale = cfg.self_attn_scale
    model.set_phase(2)
    model.eval()
    best_epoch = ckpt.get("epoch", "?")
    val_metric = ckpt.get("metric", 0)

    # Check last epoch
    last_ckpt = torch.load(os.path.join(CKPT_DIR, "last.pt"), map_location="cpu", weights_only=True)
    last_epoch = last_ckpt.get("epoch", "?")
    del last_ckpt

    print(f"Loaded best checkpoint: epoch {best_epoch}, val BERTScore {val_metric:.4f}")
    print(f"Training ran till epoch {last_epoch}")

    # ── Load test data (sentence-level split, ZuCo 1.0 only) ───────────
    dicts = load_pickle_datasets(cfg.data_dir, TASK_FILES)
    pre = EEGPreprocessor(eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels)
    # For sentence split: extract without subject info
    samples = pre.extract_all_sentences(dicts, subject=cfg.subject)

    # Sentence-level split (same as training)
    train_s, dev_s, test_s = split_samples(samples, train_ratio=0.8, dev_ratio=0.1, seed=cfg.seed)

    # Load saved normalization stats
    stats_path = os.path.join(CKPT_DIR, "eeg_norm_stats.npz")
    if os.path.isfile(stats_path):
        print(f"Loading normalisation stats from {stats_path}")
        pre.load_stats(stats_path)
    else:
        print("WARNING: No saved stats, fitting on train split")
        pre.fit([s[0] for s in train_s])

    test_s = [(pre.transform(e), t) for e, t in test_s]
    test_ds = ZuCoEEGDataset(test_s, tokenizer, cfg.max_words, cfg.max_text_len)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False,
                         collate_fn=eeg_collate_fn, num_workers=0, pin_memory=True)
    print(f"Test set: {len(test_ds)} samples\n")

    # ── Generate predictions (greedy only for ranking) ──────────────────
    all_refs = []
    all_greedy = []
    all_tf = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Generating predictions"):
            eeg = batch["eeg"].to(DEVICE)
            eeg_mask = batch["eeg_mask"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            raw_texts = batch["raw_text"]
            B = eeg.size(0)
            all_refs.extend(raw_texts)

            # Greedy
            preds_greedy = model.generate_text(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=cfg.max_gen_length, num_beams=1,
                eeg_prior_alpha=0.0,
                repetition_penalty=cfg.repetition_penalty,
            )
            all_greedy.extend(preds_greedy)

            # Teacher-forced argmax
            saved_sa = model.self_attn_scale
            model.self_attn_scale = 0.2
            outputs_tf = model(eeg=eeg, eeg_mask=eeg_mask, labels=labels, output_attentions=False)
            model.self_attn_scale = saved_sa

            tf_ids = outputs_tf.logits.argmax(dim=-1)
            for i in range(B):
                valid_mask = labels[i] != -100
                ids = tf_ids[i][valid_mask]
                all_tf.append(tokenizer.decode(ids, skip_special_tokens=True))

    N = len(all_refs)
    print(f"\nGenerated {N} predictions")

    # ── Per-sample BERTScore ────────────────────────────────────────────
    print("Computing per-sample BERTScore...")
    from bert_score import score as bscore

    all_concat = all_greedy + all_tf
    refs_concat = all_refs * 2
    preds_clean = [p if p.strip() else "<empty>" for p in all_concat]

    P, R, F1 = bscore(
        preds_clean, refs_concat,
        model_type="distilbert-base-uncased",
        verbose=True, batch_size=64,
    )

    bs_greedy_f1 = F1[:N]
    bs_greedy_p = P[:N]
    bs_greedy_r = R[:N]
    bs_tf_f1 = F1[N:]

    # ── Aggregate metrics ───────────────────────────────────────────────
    print("Computing aggregate metrics...")

    def full_metrics(preds, refs, prefix):
        m = {}
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        wer_val = compute_wer(preds, refs)
        for k, v in bleu.items():
            m[f"{prefix}_{k}"] = v
        for k, v in rouge.items():
            m[f"{prefix}_{k}"] = v
        m[f"{prefix}_wer"] = wer_val
        return m

    metrics = {}
    metrics.update(full_metrics(all_greedy, all_refs, "greedy"))
    metrics["greedy_bertscore_f1"] = bs_greedy_f1.mean().item()
    metrics["greedy_bertscore_precision"] = bs_greedy_p.mean().item()
    metrics["greedy_bertscore_recall"] = bs_greedy_r.mean().item()

    metrics.update(full_metrics(all_tf, all_refs, "tf"))
    metrics["tf_bertscore_f1"] = bs_tf_f1.mean().item()

    # ── Build ranked records ────────────────────────────────────────────
    records = []
    for i in range(N):
        records.append({
            "idx": i,
            "reference": all_refs[i],
            "greedy": all_greedy[i],
            "tf": all_tf[i],
            "greedy_bs_f1": bs_greedy_f1[i].item(),
            "greedy_bs_p": bs_greedy_p[i].item(),
            "greedy_bs_r": bs_greedy_r[i].item(),
        })
    records.sort(key=lambda x: x["greedy_bs_f1"], reverse=True)

    # ── Save ranked predictions TXT ─────────────────────────────────────
    print(f"\nWriting {OUT_TXT} ...")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("EEG-TO-TEXT — SENTENCE-SPLIT PREDICTIONS RANKED BY GREEDY BERTScore F1\n")
        f.write(f"Checkpoint: {CKPT}\n")
        f.write(f"Best epoch: {best_epoch} | Trained till epoch: {last_epoch}\n")
        f.write(f"Val BERTScore F1: {val_metric:.4f}\n")
        f.write(f"Dataset: ZuCo 1.0 only (3 tasks) | Split: sentence-level (no overlap)\n")
        f.write(f"Test samples: {N}\n")
        f.write("=" * 100 + "\n\n")

        for rank, r in enumerate(records, 1):
            f.write(f"{'━' * 100}\n")
            f.write(f"  Rank {rank:>4d}/{N}  |  Sample #{r['idx']}\n")
            f.write(f"{'─' * 100}\n")
            f.write(f"  REFERENCE : {r['reference']}\n")
            f.write(f"  GREEDY    : {r['greedy']}\n")
            f.write(f"              BERTScore F1={r['greedy_bs_f1']:.4f}  P={r['greedy_bs_p']:.4f}  R={r['greedy_bs_r']:.4f}\n")
            f.write(f"  TF        : {r['tf']}\n")

        f.write(f"\n{'━' * 100}\n")
        f.write(f"SUMMARY: {N} test samples\n")
        f.write(f"  Avg Greedy BERTScore F1: {bs_greedy_f1.mean().item():.4f}\n")
        f.write(f"  Avg TF     BERTScore F1: {bs_tf_f1.mean().item():.4f}\n")

    # ── Save metrics TXT ────────────────────────────────────────────────
    print(f"Writing {OUT_METRICS} ...")
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION METRICS — SENTENCE-SPLIT CHECKPOINT\n")
        f.write(f"Checkpoint: {CKPT}\n")
        f.write(f"Best epoch: {best_epoch} | Trained till epoch: {last_epoch}\n")
        f.write(f"Val BERTScore F1: {val_metric:.4f}\n")
        f.write(f"Dataset: ZuCo 1.0 only (3 tasks, 12 subjects)\n")
        f.write(f"Split: sentence-level (no sentence overlap between train/dev/test)\n")
        f.write(f"Test samples: {N}\n")
        f.write("=" * 80 + "\n\n")

        # Table
        header = f"{'Metric':<24}  {'GREEDY':>10}  {'TF':>10}"
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")

        rows = [
            ("BERTScore F1", "greedy_bertscore_f1", "tf_bertscore_f1"),
            ("BERTScore P", "greedy_bertscore_precision", None),
            ("BERTScore R", "greedy_bertscore_recall", None),
            ("BLEU-1", "greedy_bleu1", "tf_bleu1"),
            ("BLEU-2", "greedy_bleu2", "tf_bleu2"),
            ("BLEU-3", "greedy_bleu3", "tf_bleu3"),
            ("BLEU-4", "greedy_bleu4", "tf_bleu4"),
            ("ROUGE-1 P", "greedy_rouge1_precision", "tf_rouge1_precision"),
            ("ROUGE-1 R", "greedy_rouge1_recall", "tf_rouge1_recall"),
            ("ROUGE-1 F", "greedy_rouge1", "tf_rouge1"),
            ("ROUGE-2 P", "greedy_rouge2_precision", "tf_rouge2_precision"),
            ("ROUGE-2 R", "greedy_rouge2_recall", "tf_rouge2_recall"),
            ("ROUGE-2 F", "greedy_rouge2", "tf_rouge2"),
            ("ROUGE-L P", "greedy_rougeL_precision", "tf_rougeL_precision"),
            ("ROUGE-L R", "greedy_rougeL_recall", "tf_rougeL_recall"),
            ("ROUGE-L F", "greedy_rougeL", "tf_rougeL"),
            ("WER", "greedy_wer", "tf_wer"),
        ]

        for display, gkey, tkey in rows:
            gval = metrics.get(gkey, 0)
            tval = metrics.get(tkey, 0) if tkey else ""
            if isinstance(tval, (int, float)):
                f.write(f"{display:<24}  {gval:>10.4f}  {tval:>10.4f}\n")
            else:
                f.write(f"{display:<24}  {gval:>10.4f}  {'':>10}\n")
        f.write("─" * len(header) + "\n")

        # BERTScore distribution
        f.write(f"\n{'=' * 60}\n")
        f.write("GREEDY BERTScore F1 DISTRIBUTION\n")
        f.write(f"{'─' * 40}\n")
        bs_np = bs_greedy_f1.numpy()
        for p in [10, 25, 50, 75, 90]:
            f.write(f"  P{p:>2d}:  {np.percentile(bs_np, p):.4f}\n")
        f.write(f"  Mean: {bs_np.mean():.4f}\n")
        f.write(f"  Std:  {bs_np.std():.4f}\n")
        f.write(f"  Min:  {bs_np.min():.4f}\n")
        f.write(f"  Max:  {bs_np.max():.4f}\n")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"DONE — Files saved:")
    print(f"  1. {OUT_TXT}")
    print(f"  2. {OUT_METRICS}")
    print(f"{'=' * 60}")
    print(f"\nCheckpoint: epoch {best_epoch} (trained till {last_epoch})")
    print(f"Test samples: {N}")
    print(f"\nGREEDY METRICS:")
    print(f"  BERTScore F1:  {metrics['greedy_bertscore_f1']:.4f}")
    print(f"  BLEU-1:        {metrics['greedy_bleu1']:.4f}")
    print(f"  BLEU-2:        {metrics['greedy_bleu2']:.4f}")
    print(f"  BLEU-3:        {metrics['greedy_bleu3']:.4f}")
    print(f"  BLEU-4:        {metrics['greedy_bleu4']:.4f}")
    print(f"  ROUGE-1 P/R/F: {metrics['greedy_rouge1_precision']:.4f} / {metrics['greedy_rouge1_recall']:.4f} / {metrics['greedy_rouge1']:.4f}")
    print(f"  ROUGE-2 P/R/F: {metrics['greedy_rouge2_precision']:.4f} / {metrics['greedy_rouge2_recall']:.4f} / {metrics['greedy_rouge2']:.4f}")
    print(f"  ROUGE-L P/R/F: {metrics['greedy_rougeL_precision']:.4f} / {metrics['greedy_rougeL_recall']:.4f} / {metrics['greedy_rougeL']:.4f}")
    print(f"  WER:           {metrics['greedy_wer']:.4f}")


if __name__ == "__main__":
    main()
