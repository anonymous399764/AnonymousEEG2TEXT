"""
Comprehensive evaluation: generate ALL test predictions with every decoding mode,
save per-sample texts ranked by BERTScore, and save a final metrics summary file.

Output files:
  1. Results/all_predictions_ranked.txt   — human-readable, all modes per sample
  2. Results/all_predictions_ranked.csv   — CSV with all modes per sample
  3. Results/final_metrics.txt            — aggregate scores for every mode
"""
import torch, os, sys, csv, json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from eeg_to_text.config import Config
from eeg_to_text.data.preprocessing import EEGPreprocessor, load_pickle_datasets
from eeg_to_text.data.dataset import ZuCoEEGDataset, eeg_collate_fn, split_samples_by_subject
from eeg_to_text.models.eeg_to_text import EEGToTextModel
from transformers import BartTokenizer
import torch.nn.functional as F

BART = "facebook/bart-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "Results/checkpoints/best.pt"
OUT_DIR = "Results"
OUT_TXT = os.path.join(OUT_DIR, "all_predictions_ranked.txt")
OUT_CSV = os.path.join(OUT_DIR, "all_predictions_ranked.csv")
OUT_METRICS = os.path.join(OUT_DIR, "final_metrics.txt")


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
    epoch = ckpt.get("epoch", "?")
    val_metric = ckpt.get("metric", 0)
    print(f"Loaded checkpoint: epoch {epoch}, val BERTScore {val_metric:.4f}")

    # ── Load test data ──────────────────────────────────────────────────
    dicts = load_pickle_datasets(cfg.data_dir, cfg.task_pickle_files)
    pre = EEGPreprocessor(eeg_type=cfg.eeg_type, bands=cfg.bands, n_channels=cfg.n_channels)
    samples = pre.extract_all_sentences_with_subjects(dicts, subject=cfg.subject)
    train_s, dev_s, test_s = split_samples_by_subject(samples, seed=cfg.seed)
    pre.fit([s[0] for s in train_s])
    stats_path = os.path.join(cfg.checkpoint_dir, "eeg_norm_stats.npz")
    if os.path.isfile(stats_path):
        pre.load_stats(stats_path)
    test_s = [(pre.transform(e), t) for e, t in test_s]
    test_ds = ZuCoEEGDataset(test_s, tokenizer, cfg.max_words, cfg.max_text_len)
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False,
                         collate_fn=eeg_collate_fn, num_workers=0, pin_memory=True)
    print(f"Test set: {len(test_ds)} samples\n")

    # ── Generate all predictions ────────────────────────────────────────
    all_refs = []
    all_greedy = []
    all_beam = []
    all_nucleus = []
    all_tf = []

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Generating (all modes)"):
            eeg = batch["eeg"].to(DEVICE)
            eeg_mask = batch["eeg_mask"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            raw_texts = batch["raw_text"]
            B = eeg.size(0)
            all_refs.extend(raw_texts)

            # 1. Greedy
            preds_greedy = model.generate_text(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=cfg.max_gen_length, num_beams=1,
                eeg_prior_alpha=0.0,
                repetition_penalty=cfg.repetition_penalty,
            )
            all_greedy.extend(preds_greedy)

            # 2. Beam search (5 beams)
            preds_beam = model.generate_text(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=cfg.max_gen_length, num_beams=cfg.num_beams,
                eeg_prior_alpha=0.0,
                repetition_penalty=cfg.repetition_penalty,
            )
            all_beam.extend(preds_beam)

            # 3. Nucleus sampling
            preds_nuc = model.generate_text(
                eeg=eeg, eeg_mask=eeg_mask, tokenizer=tokenizer,
                max_length=cfg.max_gen_length,
                do_sample=True,
                top_p=cfg.gen_top_p,
                temperature=cfg.gen_temperature,
                eeg_prior_alpha=0.0,
                repetition_penalty=cfg.repetition_penalty,
            )
            all_nucleus.extend(preds_nuc)

            # 4. Teacher-forced (argmax of logits)
            # Restore fuller self-attention for TF evaluation.
            saved_sa_scale = model.self_attn_scale
            model.self_attn_scale = 0.2  # Slightly restore SA for TF eval (trained with 0.0)

            outputs_tf = model(
                eeg=eeg, eeg_mask=eeg_mask, labels=labels,
                output_attentions=False,
            )

            model.self_attn_scale = saved_sa_scale  # restore dampening

            tf_ids = outputs_tf.logits.argmax(dim=-1)
            for i in range(B):
                # Only decode at valid label positions (ignore -100 padding)
                valid_mask = labels[i] != -100
                ids = tf_ids[i][valid_mask]
                all_tf.append(tokenizer.decode(ids, skip_special_tokens=True))

    N = len(all_refs)
    print(f"\nGenerated {N} predictions × 4 modes")

    # ── Per-sample BERTScore (batched for all modes at once) ────────────
    print("Computing per-sample BERTScore (all 4 modes batched)...")
    from bert_score import score as bscore

    # Concatenate all modes for one big batched call
    all_preds_concat = all_greedy + all_beam + all_nucleus + all_tf
    all_refs_concat = all_refs * 4
    preds_clean = [p if p.strip() else "<empty>" for p in all_preds_concat]

    P, R, F1 = bscore(
        preds_clean, all_refs_concat,
        model_type="distilbert-base-uncased",
        verbose=True,
        batch_size=64,
    )

    # Split back
    bs_greedy = F1[0:N]
    bs_beam   = F1[N:2*N]
    bs_nuc    = F1[2*N:3*N]
    bs_tf     = F1[3*N:4*N]

    # ── Per-sample ROUGE-L ──────────────────────────────────────────────
    print("Computing per-sample ROUGE-L...")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def get_rougeL(preds, refs):
        return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]

    rl_greedy = get_rougeL(all_greedy, all_refs)
    rl_beam   = get_rougeL(all_beam, all_refs)
    rl_nuc    = get_rougeL(all_nucleus, all_refs)
    rl_tf     = get_rougeL(all_tf, all_refs)

    # ── Build per-sample records, sort by greedy BERTScore ──────────────
    records = []
    for i in range(N):
        records.append({
            "idx": i,
            "reference": all_refs[i],
            "greedy": all_greedy[i],
            "beam": all_beam[i],
            "nucleus": all_nucleus[i],
            "tf": all_tf[i],
            "greedy_bs": bs_greedy[i].item(),
            "beam_bs": bs_beam[i].item(),
            "nuc_bs": bs_nuc[i].item(),
            "tf_bs": bs_tf[i].item(),
            "greedy_rl": rl_greedy[i],
            "beam_rl": rl_beam[i],
            "nuc_rl": rl_nuc[i],
            "tf_rl": rl_tf[i],
        })

    records.sort(key=lambda x: x["greedy_bs"], reverse=True)

    # ── Aggregate metrics ───────────────────────────────────────────────
    from eeg_to_text.evaluation.metrics import compute_bleu, compute_rouge, compute_wer

    def agg_metrics(preds, refs, prefix):
        m = {}
        bleu = compute_bleu(preds, refs)
        rouge = compute_rouge(preds, refs)
        wer = compute_wer(preds, refs)
        for k, v in bleu.items():
            m[f"{prefix}_{k}"] = v
        for k, v in rouge.items():
            m[f"{prefix}_{k}"] = v
        m[f"{prefix}_wer"] = wer
        return m

    metrics = {}
    metrics.update(agg_metrics(all_greedy, all_refs, "greedy"))
    metrics["greedy_bertscore_f1"] = bs_greedy.mean().item()
    metrics.update(agg_metrics(all_beam, all_refs, "beam"))
    metrics["beam_bertscore_f1"] = bs_beam.mean().item()
    metrics.update(agg_metrics(all_nucleus, all_refs, "nucleus"))
    metrics["nucleus_bertscore_f1"] = bs_nuc.mean().item()
    metrics.update(agg_metrics(all_tf, all_refs, "tf"))
    metrics["tf_bertscore_f1"] = bs_tf.mean().item()

    # Exact match accuracy (meaningful with subject-level split)
    def _exact_match(preds, refs):
        return sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip()) / len(refs)
    metrics["greedy_exact_match"] = _exact_match(all_greedy, all_refs)
    metrics["beam_exact_match"] = _exact_match(all_beam, all_refs)
    metrics["nucleus_exact_match"] = _exact_match(all_nucleus, all_refs)
    metrics["tf_exact_match"] = _exact_match(all_tf, all_refs)

    # ── Retrieval accuracy (key metric for subject-level split) ─────────
    # For each test sample, check if the generated text's nearest sentence
    # (by TF-IDF cosine similarity) in the candidate pool is the correct reference.
    print("Computing retrieval accuracy (TF-IDF cosine similarity)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    unique_refs = sorted(set(all_refs))
    ref_to_idx = {r: i for i, r in enumerate(unique_refs)}
    n_cands = len(unique_refs)
    print(f"  Candidate pool: {n_cands} unique sentences")

    # Build TF-IDF matrix for candidate sentences
    tfidf = TfidfVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b')
    cand_matrix = tfidf.fit_transform(unique_refs)  # (n_cands, vocab)

    def retrieval_acc(preds, refs, topk_list=[1, 5, 10]):
        """Vectorized retrieval: TF-IDF cosine similarity."""
        pred_matrix = tfidf.transform([p if p.strip() else '<empty>' for p in preds])
        sim = cos_sim(pred_matrix, cand_matrix)  # (N, n_cands)
        # For each sample, rank candidates by similarity
        correct = {k: 0 for k in topk_list}
        mrr_sum = 0.0
        for i, true_ref in enumerate(refs):
            true_idx = ref_to_idx[true_ref]
            # Rank by descending similarity
            ranked = np.argsort(-sim[i])
            rank = int(np.where(ranked == true_idx)[0][0]) + 1
            mrr_sum += 1.0 / rank
            for k in topk_list:
                if rank <= k:
                    correct[k] += 1
        n = len(preds)
        return {f"top{k}": correct[k] / n for k in topk_list}, mrr_sum / n

    topk_greedy, mrr_greedy = retrieval_acc(all_greedy, all_refs)
    topk_beam, mrr_beam = retrieval_acc(all_beam, all_refs)
    topk_tf, mrr_tf = retrieval_acc(all_tf, all_refs)

    for k, v in topk_greedy.items():
        metrics[f"greedy_retrieval_{k}"] = v
    metrics["greedy_retrieval_mrr"] = mrr_greedy
    for k, v in topk_beam.items():
        metrics[f"beam_retrieval_{k}"] = v
    metrics["beam_retrieval_mrr"] = mrr_beam
    for k, v in topk_tf.items():
        metrics[f"tf_retrieval_{k}"] = v
    metrics["tf_retrieval_mrr"] = mrr_tf

    print(f"  Greedy retrieval: top1={topk_greedy['top1']:.4f}, top5={topk_greedy['top5']:.4f}, "
          f"top10={topk_greedy['top10']:.4f}, MRR={mrr_greedy:.4f}")
    print(f"  Beam retrieval:   top1={topk_beam['top1']:.4f}, top5={topk_beam['top5']:.4f}, "
          f"top10={topk_beam['top10']:.4f}, MRR={mrr_beam:.4f}")
    print(f"  TF retrieval:     top1={topk_tf['top1']:.4f}, top5={topk_tf['top5']:.4f}, "
          f"top10={topk_tf['top10']:.4f}, MRR={mrr_tf:.4f}")

    # ── Save per-sample BERTScores for paired significance test later ────
    np.savez(os.path.join(OUT_DIR, "real_eeg_per_sample_scores.npz"),
             greedy_bertscore=bs_greedy.numpy(),
             beam_bertscore=bs_beam.numpy(),
             tf_bertscore=bs_tf.numpy(),
             greedy_rougeL=np.array(rl_greedy),
             refs=np.array(all_refs),
             greedy_preds=np.array(all_greedy))

    # ══════════════════════════════════════════════════════════════════════
    # SAVE FILE 1: All predictions ranked (TXT)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\nWriting {OUT_TXT} ...")
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("EEG-TO-TEXT — ALL PREDICTIONS RANKED BY GREEDY BERTScore F1 (best → worst)\n")
        f.write(f"Checkpoint: epoch {epoch} | Val BERTScore: {val_metric:.4f} | Test samples: {N}\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nGeneration modes:\n")
        f.write(f"  GREEDY  — Greedy decoding (num_beams=1, deterministic)\n")
        f.write(f"  BEAM    — Beam search (num_beams={cfg.num_beams})\n")
        f.write(f"  NUCLEUS — Nucleus sampling (top_p={cfg.gen_top_p}, temp={cfg.gen_temperature})\n")
        f.write(f"  TF      — Teacher-forced argmax (upper bound — sees reference tokens)\n")
        f.write(f"\nScoring: BS = BERTScore F1, RL = ROUGE-L F1\n")
        f.write("=" * 100 + "\n\n")

        for rank, r in enumerate(records, 1):
            f.write(f"{'━' * 100}\n")
            f.write(f"  Rank {rank:>4d}/{N}  |  Sample #{r['idx']}\n")
            f.write(f"{'─' * 100}\n")
            f.write(f"  REFERENCE : {r['reference']}\n")
            f.write(f"{'─' * 100}\n")
            f.write(f"  GREEDY    : {r['greedy']}\n")
            f.write(f"              BS={r['greedy_bs']:.4f}  RL={r['greedy_rl']:.4f}\n")
            f.write(f"  BEAM      : {r['beam']}\n")
            f.write(f"              BS={r['beam_bs']:.4f}  RL={r['beam_rl']:.4f}\n")
            f.write(f"  NUCLEUS   : {r['nucleus']}\n")
            f.write(f"              BS={r['nuc_bs']:.4f}  RL={r['nuc_rl']:.4f}\n")
            f.write(f"  TF        : {r['tf']}\n")
            f.write(f"              BS={r['tf_bs']:.4f}  RL={r['tf_rl']:.4f}\n")

        f.write(f"\n{'━' * 100}\n")
        f.write(f"SUMMARY: {N} test samples\n")
        f.write(f"  Avg Greedy  BERTScore F1: {bs_greedy.mean().item():.4f}\n")
        f.write(f"  Avg Beam    BERTScore F1: {bs_beam.mean().item():.4f}\n")
        f.write(f"  Avg Nucleus BERTScore F1: {bs_nuc.mean().item():.4f}\n")
        f.write(f"  Avg TF      BERTScore F1: {bs_tf.mean().item():.4f}\n")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE FILE 2: All predictions ranked (CSV)
    # ══════════════════════════════════════════════════════════════════════
    print(f"Writing {OUT_CSV} ...")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "sample_idx", "reference",
            "greedy_prediction", "greedy_bertscore", "greedy_rougeL",
            "beam_prediction", "beam_bertscore", "beam_rougeL",
            "nucleus_prediction", "nucleus_bertscore", "nucleus_rougeL",
            "tf_prediction", "tf_bertscore", "tf_rougeL",
        ])
        for rank, r in enumerate(records, 1):
            writer.writerow([
                rank, r["idx"], r["reference"],
                r["greedy"], f"{r['greedy_bs']:.4f}", f"{r['greedy_rl']:.4f}",
                r["beam"],   f"{r['beam_bs']:.4f}",   f"{r['beam_rl']:.4f}",
                r["nucleus"],f"{r['nuc_bs']:.4f}",    f"{r['nuc_rl']:.4f}",
                r["tf"],     f"{r['tf_bs']:.4f}",     f"{r['tf_rl']:.4f}",
            ])

    # ══════════════════════════════════════════════════════════════════════
    # SAVE FILE 3: Final aggregated metrics
    # ══════════════════════════════════════════════════════════════════════
    print(f"Writing {OUT_METRICS} ...")
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("FINAL EVALUATION METRICS — EEG-TO-TEXT\n")
        f.write(f"Checkpoint: epoch {epoch} | Val BERTScore: {val_metric:.4f}\n")
        f.write(f"Test samples: {N}\n")
        f.write("=" * 80 + "\n\n")

        # Nice comparison table
        modes = [
            ("GREEDY",  "greedy"),
            ("BEAM-5",  "beam"),
            ("NUCLEUS", "nucleus"),
            ("TF",      "tf"),
        ]
        header = f"{'Metric':<22}"
        for label, _ in modes:
            header += f"  {label:>10}"
        f.write(header + "\n")
        f.write("─" * len(header) + "\n")

        metric_keys = [
            ("BERTScore F1",       "_bertscore_f1", True),
            ("Exact Match",        "_exact_match", False),
            ("BLEU-1",             "_bleu1", False),
            ("BLEU-2",             "_bleu2", False),
            ("BLEU-3",             "_bleu3", False),
            ("BLEU-4",             "_bleu4", False),
            ("ROUGE-1 F",          "_rouge1", False),
            ("ROUGE-1 P",          "_rouge1_precision", False),
            ("ROUGE-1 R",          "_rouge1_recall", False),
            ("ROUGE-2 F",          "_rouge2", False),
            ("ROUGE-2 P",          "_rouge2_precision", False),
            ("ROUGE-2 R",          "_rouge2_recall", False),
            ("ROUGE-L F",          "_rougeL", False),
            ("ROUGE-L P",          "_rougeL_precision", False),
            ("ROUGE-L R",          "_rougeL_recall", False),
            ("WER",                "_wer", False),
        ]

        for display_name, suffix, is_bs in metric_keys:
            row = f"{display_name:<22}"
            for _, prefix in modes:
                if is_bs:
                    key = f"{prefix}_bertscore_f1"
                else:
                    key = f"{prefix}{suffix}"
                val = metrics.get(key, 0)
                row += f"  {val:>10.4f}"
            f.write(row + "\n")

        f.write("─" * len(header) + "\n")

        # Retrieval accuracy table
        f.write(f"\n{'=' * 80}\n")
        f.write(f"RETRIEVAL ACCURACY (TF-IDF cosine similarity)\n")
        f.write(f"  Candidate pool: {n_cands} unique sentences\n")
        f.write(f"  Random chance top-1: {1/n_cands:.4f} ({100/n_cands:.2f}%)\n")
        f.write("─" * 60 + "\n")
        f.write(f"  {'Mode':<12}{'Top-1':>10}{'Top-5':>10}{'Top-10':>10}{'MRR':>10}\n")
        f.write(f"  {'GREEDY':<12}{topk_greedy['top1']:>10.4f}{topk_greedy['top5']:>10.4f}{topk_greedy['top10']:>10.4f}{mrr_greedy:>10.4f}\n")
        f.write(f"  {'BEAM-5':<12}{topk_beam['top1']:>10.4f}{topk_beam['top5']:>10.4f}{topk_beam['top10']:>10.4f}{mrr_beam:>10.4f}\n")
        f.write(f"  {'TF':<12}{topk_tf['top1']:>10.4f}{topk_tf['top5']:>10.4f}{topk_tf['top10']:>10.4f}{mrr_tf:>10.4f}\n")

        f.write("\nNotes:\n")
        f.write("  GREEDY  = Greedy decoding (deterministic, num_beams=1) — PRIMARY metric\n")
        f.write(f"  BEAM-5  = Beam search (num_beams={cfg.num_beams})\n")
        f.write(f"  NUCLEUS = Nucleus sampling (top_p={cfg.gen_top_p}, temp={cfg.gen_temperature})\n")
        f.write("  TF      = Teacher-forced argmax (UPPER BOUND — model sees reference tokens)\n")
        f.write(f"\nModel: S4D encoder ({cfg.s4d_layers}L, {cfg.s4d_dim}d) + BART-base decoder\n")
        f.write(f"self_attn_scale={cfg.self_attn_scale}, repetition_penalty={cfg.repetition_penalty}\n")
        f.write(f"EEG prior alpha={cfg.gen_eeg_prior_alpha} (disabled at generation time)\n")

        # Per-sample BERTScore distribution
        f.write(f"\n{'=' * 80}\n")
        f.write("GREEDY BERTScore F1 DISTRIBUTION\n")
        f.write(f"{'─' * 40}\n")
        bs_np = bs_greedy.numpy()
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            f.write(f"  P{p:>2d}:  {np.percentile(bs_np, p):.4f}\n")
        f.write(f"  Mean: {bs_np.mean():.4f}\n")
        f.write(f"  Std:  {bs_np.std():.4f}\n")
        f.write(f"  Min:  {bs_np.min():.4f}\n")
        f.write(f"  Max:  {bs_np.max():.4f}\n")

        # Top-K / Bottom-K summary
        f.write(f"\n{'=' * 80}\n")
        f.write("TOP/BOTTOM SAMPLE ANALYSIS (by greedy BERTScore)\n")
        f.write(f"{'─' * 40}\n")
        for k in [10, 50, 100]:
            top_avg = np.mean([r["greedy_bs"] for r in records[:k]])
            bot_avg = np.mean([r["greedy_bs"] for r in records[-k:]])
            f.write(f"  Top-{k:>3d} avg: {top_avg:.4f}   Bottom-{k:>3d} avg: {bot_avg:.4f}\n")

    print(f"\n{'=' * 60}")
    print(f"DONE — 3 files saved:")
    print(f"  1. {OUT_TXT}")
    print(f"  2. {OUT_CSV}")
    print(f"  3. {OUT_METRICS}")
    print(f"{'=' * 60}")
    print(f"\nAVERAGE SCORES:")
    print(f"  Greedy  BERTScore F1: {bs_greedy.mean().item():.4f}")
    print(f"  Beam    BERTScore F1: {bs_beam.mean().item():.4f}")
    print(f"  Nucleus BERTScore F1: {bs_nuc.mean().item():.4f}")
    print(f"  TF      BERTScore F1: {bs_tf.mean().item():.4f}")


if __name__ == "__main__":
    main()
