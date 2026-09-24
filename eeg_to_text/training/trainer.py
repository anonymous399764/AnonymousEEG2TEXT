"""
Two-Phase Trainer for EEG-to-Text

Phase 1 – EEG Encoder Warm-Up (epochs 1 .. phase1_epochs):
    • Frozen:    entire BART
    • Trainable: S4D encoder, MLP projection, attention gate
    • Loss:      L_lm  +  λ_c · L_contrastive
    • Rationale: before cross-attention can work the EEG encoder must produce
                 vectors that live in BART's embedding space.

Phase 2 – Cross-Attention Fine-Tuning (epochs phase1_epochs+1 .. total):
    • Frozen:    BART text encoder, BART decoder self-attn / FFN
    • Trainable: BART cross-attention layers, S4D encoder, gate
    • Loss:      L_lm  +  λ_c · L_contrastive  +  λ_a · L_attn_entropy
    • Progressive word dropout: forces decoder to rely on cross-attention

Logs metrics to TensorBoard.
Saves best checkpoint by validation BERTScore F1.
"""

import os
import gc
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, Optional

from ..config import Config
from ..models.eeg_to_text import EEGToTextModel
from .losses import EEGToTextLoss, eeg_vocab_prior_loss
from .scheduler import get_cosine_schedule_with_warmup


class Trainer:
    """
    Two-phase curriculum trainer for EEGToTextModel.
    """

    def __init__(
        self,
        model: EEGToTextModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: Config,
        evaluate_fn=None,  # external evaluation callback
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.cfg = config
        self.evaluate_fn = evaluate_fn

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Loss function
        self.criterion = EEGToTextLoss(
            label_smoothing=config.label_smoothing,
            lambda_contrastive=config.lambda_contrastive,
            lambda_attn_entropy=config.lambda_attn_entropy,
            temperature=config.temperature,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.fp16)

        # TensorBoard writer (lazy init)
        self._writer = None

        # CSV epoch log
        self._csv_path = os.path.join(config.get_checkpoint_dir(), "epoch_log.csv")
        self._csv_fields = [
            "epoch", "phase", "lr", "word_dropout",
            "loss_total", "loss_lm", "loss_nce", "loss_attn",
            "loss_vocab", "loss_disc",
            "val_primary_key", "val_primary", "val_bleu4",
            "val_bertscore_f1", "best_metric", "epoch_time_s",
        ]

        # Best metric tracking
        self.best_metric = -1.0
        self.best_state = None
        self._early_stop_counter = 0  # consecutive epochs without improvement

        # Cache last validation details for logging
        self._last_val_primary_key = getattr(self.cfg, "val_primary_metric_key", "bertscore_f1_free")
        self._last_val_primary = -1.0
        self._last_val_bertscore_f1 = -1.0
        self._last_val_bleu4 = -1.0

    @property
    def writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self.cfg.get_log_dir())
            except ImportError:
                self._writer = None
        return self._writer

    # ────────────────────────────────────────────────────────────────────
    # Build optimiser for a given phase
    # ────────────────────────────────────────────────────────────────────
    def _build_optimizer_and_scheduler(self, phase: int, num_steps: int):
        lr = self.cfg.phase1_lr if phase == 1 else self.cfg.phase2_lr
        wd = self.cfg.phase1_weight_decay if phase == 1 else self.cfg.phase2_weight_decay

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=num_steps,
        )
        return optimizer, scheduler

    # ────────────────────────────────────────────────────────────────────
    # Single training epoch
    # ────────────────────────────────────────────────────────────────────
    def _train_one_epoch(
        self,
        optimizer,
        scheduler,
        epoch: int,
        phase: int,
        global_step: int,
    ) -> int:
        self.model.train()
        accum = (self.cfg.phase2_grad_accum_steps if phase == 2
                 else self.cfg.grad_accum_steps)
        use_attn = (phase == 2)

        # Progressive word dropout: linearly increase over Phase 2
        if phase == 2:
            p2_start = self.cfg.phase1_epochs + 1
            p2_end = self.cfg.phase1_epochs + self.cfg.phase2_epochs
            if p2_end > p2_start:
                progress = (epoch - p2_start) / (p2_end - p2_start)
            else:
                progress = 1.0
            wd_rate = (self.cfg.word_dropout_start
                       + progress * (self.cfg.word_dropout_end - self.cfg.word_dropout_start))
        else:
            wd_rate = 0.0

        running = {"loss_total": 0.0, "loss_lm": 0.0, "loss_nce": 0.0, "loss_attn": 0.0, "loss_vocab": 0.0, "loss_disc": 0.0}
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} (Phase {phase})")
        for batch_idx, batch in enumerate(pbar):
            eeg = batch["eeg"].to(self.device)
            eeg_mask = batch["eeg_mask"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast('cuda', enabled=self.cfg.fp16):
                # ── Teacher forcing with progressive word dropout ─────
                # In Phase 2, randomly replace decoder input tokens with
                # <pad> to prevent the decoder from relying solely on its
                # self-attention (LM prior).  The dropout rate increases
                # linearly over Phase 2 (word_dropout_start → word_dropout_end)
                # to progressively force cross-attention to carry content.
                if phase == 2 and wd_rate > 0:
                    dec_ids = self.model.bart.prepare_decoder_input_ids_from_labels(labels)
                    dec_ids = self._apply_word_dropout(
                        dec_ids, wd_rate,
                    )
                    outputs = self.model(
                        eeg=eeg,
                        eeg_mask=eeg_mask,
                        decoder_input_ids=dec_ids,
                        labels=labels,
                        output_attentions=use_attn,
                    )
                else:
                    outputs = self.model(
                        eeg=eeg,
                        eeg_mask=eeg_mask,
                        labels=labels,
                        output_attentions=use_attn,
                    )

                # ── Get embeddings for InfoNCE ──────────────────────
                # Reuse encoder output from forward pass, then project
                # through dedicated contrastive heads (decoupled from
                # the cross-attention representations).
                eeg_hidden_cached = self.model._last_eeg_hidden
                mask_f = eeg_mask.unsqueeze(-1).float()
                eeg_pooled = (eeg_hidden_cached * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
                eeg_emb = self.model.eeg_proj_head(eeg_pooled)
                with torch.no_grad():
                    text_emb = self.model.get_bart_text_embeddings(input_ids, attention_mask)

                # ── Composite loss ──────────────────────────────────
                loss, comps = self.criterion(
                    logits=outputs.logits,
                    labels=labels,
                    eeg_embeds=eeg_emb,
                    text_embeds=text_emb,
                    cross_attentions=outputs.cross_attentions if use_attn else None,
                    use_attn_loss=use_attn,
                )
                # ── EEG vocabulary prior loss (bag-of-words) ──────────────
                eeg_vocab_logits = self.model.get_eeg_vocab_logits(
                    eeg_hidden_cached, eeg_mask
                )
                loss_vocab = eeg_vocab_prior_loss(eeg_vocab_logits, labels)
                loss = loss + self.cfg.eeg_prior_lambda * loss_vocab
                comps["loss_vocab"] = loss_vocab.item()

                # ── Shuffled-EEG discrimination loss ──────────────────
                # Forward pass with WRONG EEG (shuffled within batch).
                # The model should produce HIGHER loss with wrong EEG.
                # We penalise when it doesn't, teaching it that EEG matters.
                if phase == 2 and eeg.size(0) > 1 and not self.cfg.disable_disc_loss:
                    perm = torch.randperm(eeg.size(0), device=eeg.device)
                    # Make sure no sample maps to itself
                    same_mask = (perm == torch.arange(eeg.size(0), device=eeg.device))
                    if same_mask.any():
                        perm[same_mask] = (perm[same_mask] + 1) % eeg.size(0)
                    shuffled_eeg = eeg[perm]
                    shuffled_mask = eeg_mask[perm]

                    # Shuffled-EEG forward (no grad — only used as reference)
                    with torch.no_grad():
                        out_shuf = self.model(
                            eeg=shuffled_eeg,
                            eeg_mask=shuffled_mask,
                            labels=labels,
                            output_attentions=False,
                        )
                        loss_fn = nn.CrossEntropyLoss(reduction='none')
                        shuf_logits = out_shuf.logits
                        shuf_loss_per_token = loss_fn(
                            shuf_logits.view(-1, shuf_logits.size(-1)),
                            labels.view(-1).clamp(min=0),
                        ).view(labels.size())
                        valid = (labels != -100).float()
                        shuf_loss = (shuf_loss_per_token * valid).sum() / valid.sum().clamp(min=1)

                    # Real-EEG loss WITH gradients (so margin loss can backprop)
                    loss_fn = nn.CrossEntropyLoss(reduction='none')
                    real_loss_per_token = loss_fn(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1).clamp(min=0),
                    ).view(labels.size())
                    valid = (labels != -100).float()
                    real_loss = (real_loss_per_token * valid).sum() / valid.sum().clamp(min=1)

                    # Margin loss: penalise when real EEG isn't significantly
                    # better than shuffled EEG — gradient flows through real_loss
                    margin = float(getattr(self.cfg, "disc_loss_margin", 1.0))
                    disc_loss = F.relu(margin - (shuf_loss.detach() - real_loss))
                    disc_w = float(getattr(self.cfg, "disc_loss_weight", 0.5))
                    loss = loss + disc_w * disc_loss
                    comps["loss_disc"] = disc_loss.item()

                loss = loss / accum

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.max_grad_norm,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Accumulate stats
            for k in running:
                running[k] += comps.get(k, 0.0)
            n_batches += 1

            # Periodic logging
            if (batch_idx + 1) % self.cfg.log_every_n_steps == 0:
                avg = {k: v / n_batches for k, v in running.items()}
                pbar.set_postfix({k: f"{v:.4f}" for k, v in avg.items()})
                if self.writer:
                    for k, v in avg.items():
                        self.writer.add_scalar(f"train/{k}", v, global_step)
                    self.writer.add_scalar("train/word_dropout", wd_rate, global_step)
                    self.writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        # Epoch summary
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(f"  Epoch {epoch} avg - " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
              + (f"  wd_rate={wd_rate:.3f}" if phase == 2 else ""))
        return global_step, avg, wd_rate

    # ────────────────────────────────────────────────────────────────────
    # Word dropout (force cross-attention reliance)
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_word_dropout(
        token_ids: torch.Tensor,
        drop_rate: float,
        pad_id: int = 1,          # BART pad token id
        special_ids: set = None,  # never drop these
    ) -> torch.Tensor:
        """
        Randomly replace tokens with <pad> to degrade decoder self-attention
        context.  Keeps BOS (position 0) and special tokens intact.

        Args:
            token_ids:  (B, T) decoder input ids
            drop_rate:  probability of replacing each token
            pad_id:     replacement token id (BART <pad> = 1)

        Returns:
            (B, T) with some tokens replaced by pad_id
        """
        if special_ids is None:
            special_ids = {0, 1, 2}  # <s>, <pad>, </s>

        dropped = token_ids.clone()
        mask = torch.rand_like(token_ids, dtype=torch.float) < drop_rate
        # Never drop position 0 (BOS) or special tokens
        mask[:, 0] = False
        for sid in special_ids:
            mask = mask & (token_ids != sid)
        dropped[mask] = pad_id
        return dropped

    # ────────────────────────────────────────────────────────────────────
    # Validation
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, epoch: int, global_step: int) -> float:
        """Run validation; return BERTScore F1 (or -loss as fallback)."""
        self.model.eval()

        if self.evaluate_fn is not None:
            metrics = self.evaluate_fn(
                self.model, self.val_loader, self.tokenizer, self.device,
            )
            # Log all metrics
            if self.writer:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f"val/{k}", v, global_step)

            # Cache common metrics for logging
            self._last_val_bertscore_f1 = float(
                metrics.get(
                    "bertscore_f1_free",
                    metrics.get("greedy_bertscore_f1", metrics.get("bertscore_f1", -1.0)),
                )
            )
            self._last_val_bleu4 = float(
                metrics.get(
                    "bleu4",
                    metrics.get("greedy_bleu4", metrics.get("free_bleu4", -1.0)),
                )
            )

            # Primary metric for checkpoint selection
            key = getattr(self.cfg, "val_primary_metric_key", "bertscore_f1_free")
            self._last_val_primary_key = key

            if key == "bleu4":
                primary = metrics.get("bleu4", metrics.get("greedy_bleu4", -1.0))
            elif key == "bleu1":
                primary = metrics.get("bleu1", metrics.get("greedy_bleu1", -1.0))
            elif key == "bertscore_f1":
                primary = metrics.get("bertscore_f1", metrics.get("greedy_bertscore_f1", -1.0))
            else:
                primary = metrics.get(key, -1.0)

            if not isinstance(primary, (int, float)) or float(primary) < 0:
                # Hard fallback to existing behavior
                self._last_val_primary_key = "bertscore_f1_free"
                primary = self._last_val_bertscore_f1

            self._last_val_primary = float(primary)
            print(
                f"  Val primary ({self._last_val_primary_key}) = {self._last_val_primary:.4f}  |  "
                f"Val BERTScore-F1 = {self._last_val_bertscore_f1:.4f}  |  Val BLEU-4 = {self._last_val_bleu4:.4f}"
            )
            return self._last_val_primary
        else:
            # Fallback: return negative validation loss
            total_loss = 0.0
            n = 0
            for batch in self.val_loader:
                eeg = batch["eeg"].to(self.device)
                eeg_mask = batch["eeg_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(eeg=eeg, eeg_mask=eeg_mask, labels=labels)
                total_loss += outputs.loss.item() * eeg.size(0)
                n += eeg.size(0)
            avg_loss = total_loss / max(n, 1)
            print(f"  Val loss = {avg_loss:.4f}")
            if self.writer:
                self.writer.add_scalar("val/loss", avg_loss, global_step)

            # Cache for logging
            self._last_val_primary_key = "val_loss"
            self._last_val_primary = float(-avg_loss)
            self._last_val_bertscore_f1 = -1.0
            self._last_val_bleu4 = -1.0
            return self._last_val_primary  # higher is better

    # ────────────────────────────────────────────────────────────────────
    # CSV epoch logging
    # ────────────────────────────────────────────────────────────────────
    def _log_epoch(self, epoch, phase, lr, wd_rate, losses, val_metric, epoch_time):
        write_header = not os.path.isfile(self._csv_path)
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch,
                "phase": phase,
                "lr": f"{lr:.2e}",
                "word_dropout": f"{wd_rate:.4f}",
                "loss_total": f"{losses.get('loss_total', 0):.6f}",
                "loss_lm": f"{losses.get('loss_lm', 0):.6f}",
                "loss_nce": f"{losses.get('loss_nce', 0):.6f}",
                "loss_attn": f"{losses.get('loss_attn', 0):.6f}",
                "loss_vocab": f"{losses.get('loss_vocab', 0):.6f}",
                "loss_disc": f"{losses.get('loss_disc', 0):.6f}",
                "val_primary_key": self._last_val_primary_key,
                "val_primary": f"{self._last_val_primary:.6f}" if self._last_val_primary > 0 else "",
                "val_bleu4": f"{self._last_val_bleu4:.6f}" if self._last_val_bleu4 > 0 else "",
                "val_bertscore_f1": f"{self._last_val_bertscore_f1:.6f}" if self._last_val_bertscore_f1 > 0 else "",
                "best_metric": f"{self.best_metric:.6f}",
                "epoch_time_s": f"{epoch_time:.1f}",
            })

    # ────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metric: float, tag: str = "best"):
        ckpt_dir = self.cfg.get_checkpoint_dir()
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metric": metric,
            "early_stop_counter": self._early_stop_counter,
        }, path)
        print(f"  Saved checkpoint: {path}  (metric={metric:.4f})")

    def _truncate_csv_to_epoch(self, keep_up_to: int):
        """Keep only CSV rows with epoch <= keep_up_to (inclusive)."""
        if not os.path.isfile(self._csv_path):
            return
        with open(self._csv_path, "r", newline="") as f:
            lines = f.readlines()
        if not lines:
            return
        header = lines[0]
        kept = [header]
        for line in lines[1:]:
            try:
                ep = int(line.split(",")[0].strip())
                if ep <= keep_up_to:
                    kept.append(line)
            except (ValueError, IndexError):
                kept.append(line)  # keep malformed lines (e.g. empty)
        with open(self._csv_path, "w", newline="") as f:
            f.writelines(kept)
        print(f"  [CSV] Truncated epoch log to epoch {keep_up_to} ({len(kept)-1} rows kept)")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')}, "
              f"metric {ckpt.get('metric', '?')})")
        return ckpt

    # ────────────────────────────────────────────────────────────────────
    # Main training loop
    # ────────────────────────────────────────────────────────────────────
    def train(self, resume_path: Optional[str] = None, phase2_only: bool = False):
        """
        Run the full two-phase training curriculum:
            Phase 1 (epochs 1..phase1_epochs)  → encoder warm-up
            Phase 2 (epochs P1+1..total)       → cross-attention fine-tuning

        If phase2_only=True, loads checkpoint and starts directly at Phase 2
        (useful for re-tuning Phase 2 with different hyperparameters while
        keeping the Phase 1 encoder weights).
        """
        start_epoch = 1
        if resume_path and os.path.isfile(resume_path):
            ckpt = self.load_checkpoint(resume_path)
            if phase2_only:
                start_epoch = self.cfg.phase1_epochs + 1
                self.best_metric = -1.0  # reset so new Phase 2 can save fresh best
                self._early_stop_counter = 0
                print(f"[Phase2-only] Starting at epoch {start_epoch}, "
                      f"best_metric reset to -1.0")
            else:
                start_epoch = ckpt.get("epoch", 0) + 1
                self.best_metric = ckpt.get("metric", -1.0)
                self._early_stop_counter = ckpt.get("early_stop_counter", 0)
                print(f"  [Resume] Starting from epoch {start_epoch}, "
                      f"best_metric={self.best_metric:.4f}, "
                      f"early_stop_counter={self._early_stop_counter}")
                # Truncate CSV so resumed epochs don't create duplicate rows
                self._truncate_csv_to_epoch(start_epoch - 1)

        global_step = 0
        total_epochs = self.cfg.phase1_epochs + self.cfg.phase2_epochs

        print("=" * 70)
        print(f"Starting training: {total_epochs} epochs "
              f"(Phase 1: {self.cfg.phase1_epochs}, Phase 2: {self.cfg.phase2_epochs})")
        print(f"Device: {self.device}  |  FP16: {self.cfg.fp16}  |  "
              f"Batch: {self.cfg.batch_size}×{self.cfg.grad_accum_steps}")
        print("=" * 70)

        for epoch in range(start_epoch, total_epochs + 1):
            phase = 1 if epoch <= self.cfg.phase1_epochs else 2
            n_phase_epochs = self.cfg.phase1_epochs if phase == 1 else self.cfg.phase2_epochs

            # ── Configure phase ─────────────────────────────────────────
            if epoch == start_epoch or epoch == self.cfg.phase1_epochs + 1:
                print(f"\n{'=' * 70}")
                print(f"  PHASE {phase}")
                print(f"{'=' * 70}")

                # Free previous optimizer states to reclaim VRAM
                if 'optimizer' in dir():
                    del optimizer, scheduler
                    gc.collect()
                    torch.cuda.empty_cache()

                # Rebuild DataLoader with smaller batch for Phase 2
                if phase == 2 and self.cfg.phase2_batch_size < self.cfg.batch_size:
                    from ..data.dataset import eeg_collate_fn
                    self.train_loader = DataLoader(
                        self.train_loader.dataset,
                        batch_size=self.cfg.phase2_batch_size,
                        shuffle=True,
                        num_workers=self.cfg.num_workers,
                        collate_fn=eeg_collate_fn,
                        pin_memory=True,
                        drop_last=True,
                    )
                    print(f"  Phase 2 batch: {self.cfg.phase2_batch_size}x"
                          f"{self.cfg.phase2_grad_accum_steps}  "
                          f"(eff={self.cfg.phase2_batch_size * self.cfg.phase2_grad_accum_steps})")

                self.model.self_attn_scale = self.cfg.self_attn_scale
                self.model.set_phase(phase)
                if phase == 2:
                    self._early_stop_counter = 0  # reset counter at phase boundary
                accum_for_phase = (self.cfg.phase2_grad_accum_steps if phase == 2
                                   else self.cfg.grad_accum_steps)
                num_steps = n_phase_epochs * len(self.train_loader) // accum_for_phase
                optimizer, scheduler = self._build_optimizer_and_scheduler(phase, num_steps)

                # Sanity check at start of Phase 1
                if phase == 1 and epoch == 1:
                    print("\nRunning sanity checks...")
                    sample_batch = next(iter(self.train_loader))
                    self.model.sanity_check(sample_batch, self.device)
                    print()

            # ── Train one epoch ─────────────────────────────────────────
            t0 = time.time()
            global_step, epoch_losses, wd_rate = self._train_one_epoch(
                optimizer, scheduler, epoch, phase, global_step,
            )
            epoch_time = time.time() - t0
            print(f"  Epoch time: {epoch_time:.1f}s")

            # ── Validate ────────────────────────────────────────────────
            val_metric = -1.0
            if epoch % self.cfg.eval_every_n_epochs == 0:
                metric = self._validate(epoch, global_step)
                val_metric = metric
                if metric > self.best_metric:
                    self.best_metric = metric
                    self._early_stop_counter = 0
                    self._save_checkpoint(epoch, metric, tag="best")
                else:
                    self._early_stop_counter += 1
                self._save_checkpoint(epoch, metric, tag="last")

                # ── Early stopping ───────────────────────────────────
                patience = self.cfg.early_stopping_patience
                # Early stopping disabled per user request
                # if patience > 0 and phase == 2 and self._early_stop_counter >= patience:
                #     print(f"\n[Early Stopping] No improvement for {patience} consecutive "
                #           f"epochs (phase 2). Best = {self.best_metric:.4f}. Stopping.")
                #     break

            # ── Log epoch to CSV ────────────────────────────────────────
            lr = scheduler.get_last_lr()[0] if scheduler else 0.0
            self._log_epoch(epoch, phase, lr, wd_rate,
                            epoch_losses, val_metric, epoch_time)

        print("\n" + "=" * 70)
        print(f"Training complete.  Best metric = {self.best_metric:.4f}")
        print("=" * 70)

        if self.writer:
            self.writer.close()
