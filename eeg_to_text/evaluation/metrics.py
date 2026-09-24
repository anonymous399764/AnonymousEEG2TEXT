"""
Comprehensive Evaluation for EEG-to-Text

Computes metrics BOTH with and without teacher forcing:

  Without TF (free generation):   model.generate_text() with beam search & greedy
  With TF:                        argmax over logits given ground-truth context

Metrics:
    BLEU-1 / 2 / 3 / 4       (corpus BLEU via sacrebleu)
    ROUGE-1 / 2 / L           (rouge-score)
    BERTScore F1              (semantic similarity — PRIMARY metric)
    WER                       (word error rate)
    Cross-attention entropy   (diagnostic: is decoder using EEG?)
    EEG–text cosine sim       (diagnostic: alignment quality)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict


def _safe_import(name):
    """Lazy import with helpful error message."""
    try:
        return __import__(name)
    except ImportError:
        raise ImportError(
            f"Missing package '{name}'.  Install via: "
            f"pip install bert-score rouge-score sacrebleu jiwer"
        )


# ════════════════════════════════════════════════════════════════════════════
# Individual metric helpers
# ════════════════════════════════════════════════════════════════════════════

def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Corpus BLEU-1..4 via sacrebleu."""
    _safe_import("sacrebleu")
    from sacrebleu.metrics import BLEU
    refs_wrapped = [references]  # sacrebleu expects list-of-lists
    result = {}
    for n in range(1, 5):
        metric = BLEU(max_ngram_order=n, smooth_method="exp", smooth_value=0.01)
        bleu = metric.corpus_score(predictions, refs_wrapped)
        result[f"bleu{n}"] = bleu.score / 100.0  # normalise to [0,1]
    return result


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L via rouge-score. Returns P, R, F for each."""
    _safe_import("rouge_score")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    stats = {k: {"p": [], "r": [], "f": []} for k in ["rouge1", "rouge2", "rougeL"]}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for k in ["rouge1", "rouge2", "rougeL"]:
            stats[k]["p"].append(scores[k].precision)
            stats[k]["r"].append(scores[k].recall)
            stats[k]["f"].append(scores[k].fmeasure)
    result = {}
    for k in ["rouge1", "rouge2", "rougeL"]:
        result[f"{k}"] = np.mean(stats[k]["f"])
        result[f"{k}_precision"] = np.mean(stats[k]["p"])
        result[f"{k}_recall"] = np.mean(stats[k]["r"])
    return result


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """BERTScore F1 — the PRIMARY evaluation metric.
    Uses distilbert-base-uncased instead of roberta-large to save ~1.3GB download
    and ~3x inference time.  Relative ranking between models is preserved."""
    _safe_import("bert_score")
    from bert_score import score as bscore
    # Sanitise: replace empty predictions with a placeholder so bert_score doesn't warn
    preds_clean = [p if p.strip() else "<empty>" for p in predictions]
    P, R, F1 = bscore(
        preds_clean, references,
        model_type="distilbert-base-uncased",  # ~260MB vs 1.4GB for roberta-large
        verbose=False,
        batch_size=64,
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Word Error Rate."""
    _safe_import("jiwer")
    from jiwer import wer
    # Filter out empty strings
    valid_preds, valid_refs = [], []
    for p, r in zip(predictions, references):
        if r.strip():
            valid_preds.append(p if p.strip() else "<empty>")
            valid_refs.append(r)
    if not valid_refs:
        return 1.0
    return wer(valid_refs, valid_preds)


# ════════════════════════════════════════════════════════════════════════════
# Complete metric bundle
# ════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    prefix: str = "",
    skip_bertscore: bool = False,
) -> Dict[str, float]:
    """
    Compute BLEU, ROUGE, BERTScore, and WER.
    All keys are prefixed with ``prefix`` (e.g. "free_" or "tf_").
    Set skip_bertscore=True when BERTScore is computed externally in a batch.
    """
    metrics = {}

    bleu = compute_bleu(predictions, references)
    for k, v in bleu.items():
        metrics[f"{prefix}{k}"] = v

    rouge = compute_rouge(predictions, references)
    for k, v in rouge.items():
        metrics[f"{prefix}{k}"] = v

    if not skip_bertscore:
        try:
            bert = compute_bertscore(predictions, references)
            for k, v in bert.items():
                metrics[f"{prefix}{k}"] = v
        except Exception as e:
            print(f"[WARNING] BERTScore failed: {e}")
            metrics[f"{prefix}bertscore_f1"] = 0.0

    metrics[f"{prefix}wer"] = compute_wer(predictions, references)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Diagnostic metrics
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cross_attn_entropy(model, batch, device) -> float:
    """Average entropy of cross-attention distributions (lower = more focused)."""
    eeg = batch["eeg"].to(device)
    eeg_mask = batch["eeg_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(eeg=eeg, eeg_mask=eeg_mask, labels=labels, output_attentions=True)

    total_entropy = 0.0
    count = 0
    if outputs.cross_attentions:
        for attn in outputs.cross_attentions:
            ent = -(attn * (attn + 1e-9).log()).sum(dim=-1)
            total_entropy += ent.mean().item()
            count += 1
    return total_entropy / max(count, 1)


@torch.no_grad()
def compute_alignment_cosine_sim(model, batch, device) -> float:
    """Mean cosine similarity between EEG and BART text embeddings."""
    eeg = batch["eeg"].to(device)
    eeg_mask = batch["eeg_mask"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    eeg_emb = model.get_eeg_embeddings(eeg, eeg_mask)
    text_emb = model.get_bart_text_embeddings(input_ids, attention_mask)

    cos = F.cosine_similarity(eeg_emb, text_emb, dim=-1)
    return cos.mean().item()


# ════════════════════════════════════════════════════════════════════════════
# Full evaluation function
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    tokenizer,
    device,
    num_beams: int = 5,
    max_gen_length: int = 56,
    print_examples: int = 5,
    eeg_prior_alpha: float = 0.0,
    repetition_penalty: float = 1.0,
    gen_do_sample: bool = True,
    gen_top_p: float = 0.9,
    gen_temperature: float = 0.8,
    best_of_n: int = 0,
    best_of_n_temperature: float = 0.9,
    mbr_n: int = 0,
    mbr_temperature: float = 0.8,
    contrastive_alpha: float = 0.0,
    contrastive_k: int = 5,
    return_predictions: bool = False,
) -> Dict[str, float] or (Dict[str, float], Dict[str, list]):
    """
    Comprehensive evaluation: metrics both with and without teacher forcing,
    plus diagnostic cross-attention and alignment statistics.

    Runs up to 7 generation modes:
      1. Free greedy            (PRIMARY — best & deterministic)
      2. Free beam search       (standard baseline)
      3. Free nucleus sampling  (if gen_do_sample=True)
      4. Best-of-N reranking    (if best_of_n > 0)
      5. MBR decoding           (if mbr_n > 0)
      6. Contrastive search     (if contrastive_alpha > 0)
      7. Teacher-forced argmax

    Args:
        model:           EEGToTextModel (already on device)
        dataloader:      Validation or test DataLoader
        tokenizer:       BART tokenizer
        device:          torch.device
        num_beams:       Beam width for beam search mode
        max_gen_length:  Max tokens for generation
        print_examples:  How many example predictions to print
        eeg_prior_alpha: EEG vocab prior weight during generation
        repetition_penalty: repetition penalty
        gen_do_sample:   Enable nucleus sampling mode
        gen_top_p:       Nucleus sampling top_p
        gen_temperature: Nucleus sampling temperature
        best_of_n:       Number of candidates for reranking (0=disabled)
        best_of_n_temperature: Temperature for diverse candidate generation
        mbr_n:           Number of MBR candidates (0=disabled)
        mbr_temperature: Temperature for MBR candidate generation
        contrastive_alpha: Alpha for contrastive search (0=disabled)
        contrastive_k:   Top-k for contrastive search

    Returns:
        Dict of all metrics
    """
    model.eval()

    all_targets: List[str] = []
    all_preds_nucleus: List[str] = []     # Free generation (nucleus sampling)
    all_preds_free: List[str] = []        # Free generation (beam search)
    all_preds_greedy: List[str] = []      # Free generation (greedy)
    all_preds_bon: List[str] = []         # Best-of-N reranking
    all_preds_mbr: List[str] = []         # MBR decoding
    all_preds_contrastive: List[str] = [] # Contrastive search
    all_preds_tf: List[str] = []          # Teacher-forced (argmax)
    cross_attn_entropies = []
    cosine_sims = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        raw_texts = batch["raw_text"]

        B = eeg.size(0)
        all_targets.extend(raw_texts)

        # ── 1. Free generation (greedy — PRIMARY) ──────────────────────
        preds_greedy = model.generate_text(
            eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
            max_length=max_gen_length, num_beams=1,
            eeg_prior_alpha=eeg_prior_alpha,
            repetition_penalty=repetition_penalty,
        )
        all_preds_greedy.extend(preds_greedy)

        # ── 2. Free generation (beam search) ────────────────────────────
        preds_beam = model.generate_text(
            eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
            max_length=max_gen_length, num_beams=num_beams,
            eeg_prior_alpha=eeg_prior_alpha,
            repetition_penalty=repetition_penalty,
        )
        all_preds_free.extend(preds_beam)

        # ── 3. Free generation (nucleus sampling) ───────────────────────
        if gen_do_sample:
            preds_nuc = model.generate_text(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=max_gen_length,
                do_sample=True,
                top_p=gen_top_p,
                temperature=gen_temperature,
                eeg_prior_alpha=eeg_prior_alpha,
                repetition_penalty=repetition_penalty,
            )
            all_preds_nucleus.extend(preds_nuc)

        # ── 4. Best-of-N reranking ─────────────────────────────────────
        if best_of_n > 0:
            preds_bon = model.generate_best_of_n(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                n_candidates=best_of_n,
                max_length=max_gen_length,
                top_p=gen_top_p,
                temperature=best_of_n_temperature,
                repetition_penalty=repetition_penalty,
                eeg_prior_alpha=eeg_prior_alpha,
            )
            all_preds_bon.extend(preds_bon)

        # ── 5. MBR decoding ────────────────────────────────────────────
        if mbr_n > 0:
            preds_mbr = model.generate_mbr(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                n_candidates=mbr_n,
                max_length=max_gen_length,
                top_p=gen_top_p,
                temperature=mbr_temperature,
                repetition_penalty=repetition_penalty,
                eeg_prior_alpha=eeg_prior_alpha,
            )
            all_preds_mbr.extend(preds_mbr)

        # ── 6. Contrastive search ──────────────────────────────────────
        if contrastive_alpha > 0:
            preds_cs = model.generate_contrastive(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=max_gen_length,
                top_k=contrastive_k,
                alpha=contrastive_alpha,
                repetition_penalty=repetition_penalty,
            )
            all_preds_contrastive.extend(preds_cs)

        # ── 7. Teacher-forced decoding (argmax) ────────────────────────
        # Restore fuller self-attention for TF: the decoder receives CORRECT
        # previous tokens, so dampening self-attention handicaps it unfairly.
        saved_sa_scale = model.self_attn_scale
        model.self_attn_scale = 0.2  # Slightly restore SA for TF eval (trained with 0.0)

        outputs_tf = model(
            eeg=eeg, eeg_mask=eeg_mask, labels=labels,
            output_attentions=True,
        )

        model.self_attn_scale = saved_sa_scale  # restore dampening

        tf_preds_ids = outputs_tf.logits.argmax(dim=-1)          # (B, T)
        for i in range(B):
            # Only decode at valid label positions (ignore -100 padding)
            valid_mask = labels[i] != -100
            ids = tf_preds_ids[i][valid_mask]
            pred_str = tokenizer.decode(ids, skip_special_tokens=True)
            all_preds_tf.append(pred_str)

        # ── 5. Diagnostics ──────────────────────────────────────────────
        if outputs_tf.cross_attentions:
            valid_cross_attn = [attn for attn in outputs_tf.cross_attentions if attn is not None]
            if valid_cross_attn:
                ent = 0.0
                for attn in valid_cross_attn:
                    ent += (-(attn * (attn + 1e-9).log()).sum(-1)).mean().item()
                cross_attn_entropies.append(ent / len(valid_cross_attn))

        eeg_emb = model.get_eeg_embeddings(eeg, eeg_mask)
        text_emb = model.get_bart_text_embeddings(input_ids, attention_mask)
        cos = F.cosine_similarity(eeg_emb, text_emb, dim=-1).mean().item()
        cosine_sims.append(cos)

    # ── Compute all metric suites ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Compute BLEU/ROUGE/WER per suite (fast)
    metrics_greedy = compute_all_metrics(all_preds_greedy, all_targets, prefix="greedy_", skip_bertscore=True)
    metrics_free = compute_all_metrics(all_preds_free, all_targets, prefix="free_", skip_bertscore=True)
    metrics_nucleus = compute_all_metrics(all_preds_nucleus, all_targets, prefix="nuc_", skip_bertscore=True) if gen_do_sample else {}
    metrics_bon = compute_all_metrics(all_preds_bon, all_targets, prefix="bon_", skip_bertscore=True) if best_of_n > 0 else {}
    metrics_mbr = compute_all_metrics(all_preds_mbr, all_targets, prefix="mbr_", skip_bertscore=True) if mbr_n > 0 else {}
    metrics_cs = compute_all_metrics(all_preds_contrastive, all_targets, prefix="cs_", skip_bertscore=True) if contrastive_alpha > 0 else {}
    metrics_tf = compute_all_metrics(all_preds_tf, all_targets, prefix="tf_", skip_bertscore=True)

    # Compute BERTScore in ONE batched call for all prediction sets
    print("Computing BERTScore (batched)...")
    try:
        all_preds_concat = []
        all_refs_concat = []
        segments = []  # (prefix, metrics_dict, count)

        all_preds_concat.extend(all_preds_greedy)
        all_refs_concat.extend(all_targets)
        segments.append(("greedy_", metrics_greedy, len(all_preds_greedy)))

        all_preds_concat.extend(all_preds_free)
        all_refs_concat.extend(all_targets)
        segments.append(("free_", metrics_free, len(all_preds_free)))

        if gen_do_sample:
            all_preds_concat.extend(all_preds_nucleus)
            all_refs_concat.extend(all_targets)
            segments.append(("nuc_", metrics_nucleus, len(all_preds_nucleus)))

        if best_of_n > 0:
            all_preds_concat.extend(all_preds_bon)
            all_refs_concat.extend(all_targets)
            segments.append(("bon_", metrics_bon, len(all_preds_bon)))

        if mbr_n > 0:
            all_preds_concat.extend(all_preds_mbr)
            all_refs_concat.extend(all_targets)
            segments.append(("mbr_", metrics_mbr, len(all_preds_mbr)))

        if contrastive_alpha > 0:
            all_preds_concat.extend(all_preds_contrastive)
            all_refs_concat.extend(all_targets)
            segments.append(("cs_", metrics_cs, len(all_preds_contrastive)))

        all_preds_concat.extend(all_preds_tf)
        all_refs_concat.extend(all_targets)
        segments.append(("tf_", metrics_tf, len(all_preds_tf)))

        from bert_score import score as bscore
        preds_clean = [p if p.strip() else "<empty>" for p in all_preds_concat]
        P, R, F1 = bscore(
            preds_clean, all_refs_concat,
            model_type="distilbert-base-uncased",
            verbose=False,
            batch_size=64,
        )
        # Split back
        offset = 0
        for prefix, metrics_dict, count in segments:
            seg_f1 = F1[offset:offset+count].mean().item()
            seg_p = P[offset:offset+count].mean().item()
            seg_r = R[offset:offset+count].mean().item()
            metrics_dict[f"{prefix}bertscore_f1"] = seg_f1
            metrics_dict[f"{prefix}bertscore_precision"] = seg_p
            metrics_dict[f"{prefix}bertscore_recall"] = seg_r
            offset += count
    except Exception as e:
        print(f"[WARNING] BERTScore failed: {e}")
        for m in [metrics_greedy, metrics_free, metrics_nucleus, metrics_bon, metrics_mbr, metrics_cs, metrics_tf]:
            for prefix in ("greedy_", "free_", "nuc_", "bon_", "mbr_", "cs_", "tf_"):
                m.setdefault(f"{prefix}bertscore_f1", 0.0)

    all_metrics = {**metrics_greedy, **metrics_free, **metrics_nucleus, **metrics_bon, **metrics_mbr, **metrics_cs, **metrics_tf}

    # Diagnostics
    all_metrics["avg_cross_attn_entropy"] = np.mean(cross_attn_entropies) if cross_attn_entropies else 0.0
    all_metrics["avg_eeg_text_cosine_sim"] = np.mean(cosine_sims) if cosine_sims else 0.0

    # Convenience alias — use greedy as primary (best & deterministic)
    all_metrics["bertscore_f1_free"] = metrics_greedy.get("greedy_bertscore_f1", 0.0)

    # ── Pretty print ────────────────────────────────────────────────────
    # Build columns dynamically based on which modes are active
    cols = [("Greedy*", "greedy_", metrics_greedy)]
    cols.append(("Beam", "free_", metrics_free))
    if gen_do_sample:
        cols.append(("Nucleus", "nuc_", metrics_nucleus))
    if best_of_n > 0:
        cols.append((f"BoN-{best_of_n}", "bon_", metrics_bon))
    if mbr_n > 0:
        cols.append((f"MBR-{mbr_n}", "mbr_", metrics_mbr))
    if contrastive_alpha > 0:
        cols.append(("Contr", "cs_", metrics_cs))
    cols.append(("TF", "tf_", metrics_tf))

    header = f"{'Metric':<28}" + "".join(f" {name:>10}" for name, _, _ in cols)
    print(f"\n{header}")
    print("-" * len(header))
    for base_metric in ["bleu1", "bleu2", "bleu3", "bleu4",
                        "rouge1", "rouge1_precision", "rouge1_recall",
                        "rouge2", "rouge2_precision", "rouge2_recall",
                        "rougeL", "rougeL_precision", "rougeL_recall",
                        "bertscore_f1", "wer"]:
        vals = []
        for name, prefix, mdict in cols:
            vals.append(f" {mdict.get(f'{prefix}{base_metric}', 0):>10.4f}")
        print(f"  {base_metric:<26}" + "".join(vals))

    print(f"\n  {'avg_cross_attn_entropy':<33} {all_metrics['avg_cross_attn_entropy']:>12.4f}")
    print(f"  {'avg_eeg_text_cosine_sim':<33} {all_metrics['avg_eeg_text_cosine_sim']:>12.4f}")

    # Flag cross-attention collapse
    free_bleu1 = metrics_free.get("free_bleu1", 0)
    tf_bleu1 = metrics_tf.get("tf_bleu1", 0)
    gap = tf_bleu1 - free_bleu1
    if gap > 0.15:
        print(f"\n  WARNING: TF-Free BLEU-1 gap = {gap:.2f} (>0.15) -> "
              f"possible cross-attention collapse!")

    # ── Example predictions ─────────────────────────────────────────────
    if print_examples > 0:
        print(f"\n{'-' * 70}")
        print("EXAMPLE PREDICTIONS")
        print(f"{'-' * 70}")
        for i in range(min(print_examples, len(all_targets))):
            print(f"\n[{i+1}] Target:     {all_targets[i]}")
            print(f"    Greedy*:    {all_preds_greedy[i]}")
            print(f"    Beam:       {all_preds_free[i]}")
            if gen_do_sample:
                print(f"    Nucleus:    {all_preds_nucleus[i]}")
            if best_of_n > 0 and i < len(all_preds_bon):
                print(f"    BoN-{best_of_n}:     {all_preds_bon[i]}")
            if mbr_n > 0 and i < len(all_preds_mbr):
                print(f"    MBR-{mbr_n}:     {all_preds_mbr[i]}")
            if contrastive_alpha > 0 and i < len(all_preds_contrastive):
                print(f"    Contr:      {all_preds_contrastive[i]}")
            print(f"    With TF:    {all_preds_tf[i]}")

    # Optionally return raw prediction lists for downstream saving/analysis
    if return_predictions:
        preds = {
            "targets": all_targets,
            "greedy": all_preds_greedy,
            "beam": all_preds_free,
            "nucleus": all_preds_nucleus,
            "bon": all_preds_bon,
            "mbr": all_preds_mbr,
            "contrastive": all_preds_contrastive,
            "teacher_forced": all_preds_tf,
        }
        return all_metrics, preds
    return all_metrics
