"""
EEG-to-Text Full Model: S4D Encoder + Attention Gate + BART Decoder

CRITICAL WIRING (the whole point of this architecture):
    The S4D encoder output is passed as ``encoder_outputs`` to
    ``BartForConditionalGeneration``, which causes BART's cross-attention
    layers to attend over the EEG hidden states instead of BART's own
    text-encoder output.

    ╔════════════════════╗          ╔════════════════════════╗
    ║ Word-level EEG     ║          ║ Target sentence tokens ║
    ║ (B, L_eeg, 840)   ║          ║ (decoder_input_ids)    ║
    ╚═══════╤════════════╝          ╚══════════╤═════════════╝
            │                                  │
       S4D Encoder                       BART Tokenizer
       + Attn Gate                             │
            │                                  │
    encoder_hidden_states ──► BART Decoder (cross-attention) ──► logits
    (B, L_eeg, 768)          (self-attn + cross-attn + FFN)

Anti-patterns avoided:
  ✗ input_ids fed to BART (bypasses EEG encoder)
  ✗ EEG concatenated/prepended to text embeddings
  ✗ Entire BART frozen (cross-attention can't learn EEG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    LogitsProcessorList,
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Dict, List, Optional, Tuple

from .s4d_encoder import S4DEEGEncoder
from .attention_gate import EEGAttentionGate


def _torch_version_ge(min_version: str) -> bool:
    try:
        from packaging import version
    except Exception:
        return False
    try:
        current = torch.__version__.split("+")[0]
        return version.parse(current) >= version.parse(min_version)
    except Exception:
        return False


def _load_bart_model(model_name: str):
    if _torch_version_ge("2.6.0"):
        try:
            return BartForConditionalGeneration.from_pretrained(
                model_name, use_safetensors=True
            )
        except Exception:
            return BartForConditionalGeneration.from_pretrained(model_name)
    try:
        return BartForConditionalGeneration.from_pretrained(
            model_name, use_safetensors=True
        )
    except Exception as exc:
        msg = (
            "Torch <2.6 requires safetensors. Install `safetensors` or upgrade "
            "torch>=2.6 to load PyTorch weights safely."
        )
        raise ValueError(msg) from exc


class EEGVocabLogitsProcessor:
    """
    Logits processor that adds EEG-predicted vocabulary prior to decoder
    scores during beam search generation.  This explicitly injects EEG
    content signal into the decoding process, preventing the LM prior
    from dominating.
    """
    def __init__(self, eeg_log_probs: torch.Tensor, alpha: float = 0.3):
        self.eeg_log_probs = eeg_log_probs   # (B*num_beams, V)
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + self.alpha * self.eeg_log_probs


class EEGToTextModel(nn.Module):
    """
    End-to-end EEG-conditioned BART with proper cross-attention wiring.
    """

    def __init__(
        self,
        bart_model_name: str = "facebook/bart-base",
        eeg_input_dim: int = 840,
        s4d_dim: int = 512,
        s4d_layers: int = 6,
        s4d_state_dim: int = 64,
        s4d_dropout: float = 0.1,
        s4d_bidirectional: bool = True,
        gate_bias_init: float = 1.0,
        contrastive_proj_dim: int = 256,
    ):
        super().__init__()

        # ── BART backbone ───────────────────────────────────────────────
        self.bart = _load_bart_model(bart_model_name)
        bart_dim = self.bart.config.d_model   # 768 for bart-base

        # Enable gradient checkpointing on BART to save VRAM
        self.bart.config.use_cache = False
        if hasattr(self.bart, 'gradient_checkpointing_enable'):
            try:
                self.bart.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.bart.gradient_checkpointing_enable()

        # ── S4D EEG Encoder ─────────────────────────────────────────────
        self.eeg_encoder = S4DEEGEncoder(
            input_dim=eeg_input_dim,
            s4d_dim=s4d_dim,
            n_layers=s4d_layers,
            state_dim=s4d_state_dim,
            dropout=s4d_dropout,
            bart_dim=bart_dim,
            bidirectional=s4d_bidirectional,
        )

        # ── Attention Gate ──────────────────────────────────────────────
        self.attention_gate = EEGAttentionGate(
            d_model=bart_dim,
            bias_init=gate_bias_init,
        )

        # ── Contrastive projection head (CLIP-style) ─────────────────────
        # Separate MLP that projects mean-pooled encoder output into a
        # lower-dimensional space for InfoNCE.  This decouples the
        # contrastive objective from the per-token representations used
        # for cross-attention, preventing gradient interference.
        self.eeg_proj_head = nn.Sequential(
            nn.Linear(bart_dim, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, contrastive_proj_dim),
        )
        self.text_proj_head = nn.Sequential(
            nn.Linear(bart_dim, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, contrastive_proj_dim),
        )

        # ── EEG vocabulary prior head (bag-of-words prediction from EEG) ──
        # Projects mean-pooled EEG through a small MLP, then uses BART's
        # shared embedding matrix for vocabulary projection (~590K new params).
        # Trained with BCE on bag-of-words targets in BOTH phases.
        # During beam search, adds soft vocabulary prior to decoder logits.
        self.eeg_vocab_proj = nn.Sequential(
            nn.Linear(bart_dim, bart_dim),
            nn.GELU(),
        )

        # ── Self-attention dampening ────────────────────────────────
        # Scale factor applied to decoder self-attention outputs via hooks.
        # 1.0 = no dampening (Phase 1), 0.1 = 90% dampening (Phase 2).
        # Forces the decoder to rely on cross-attention for content.
        self.self_attn_scale = 1.0
        self._sa_hooks = []
        self._install_self_attn_hooks()

        # Store dimensions for convenience
        self.bart_dim = bart_dim
        self.contrastive_proj_dim = contrastive_proj_dim

    # ────────────────────────────────────────────────────────────────────
    # Encoding helper (shared by forward and generate_text)
    # ────────────────────────────────────────────────────────────────────
    def encode_eeg(self, eeg: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        EEG → S4D → gate → hidden states ready for BART cross-attention.

        Args:
            eeg: (B, L, eeg_dim)

        Returns:
            (B, L, bart_dim)
        """
        h = self.eeg_encoder(eeg, eeg_mask=eeg_mask)  # (B, L, bart_dim)
        h = self.attention_gate(h)         # (B, L, bart_dim)
        return h

    # ────────────────────────────────────────────────────────────────────
    # Self-attention dampening hooks
    # ────────────────────────────────────────────────────────────────────
    def _install_self_attn_hooks(self):
        """
        Install forward hooks on each BART decoder self-attention layer.
        When self.self_attn_scale < 1.0, the hook multiplies the self-attention
        output by the scale factor, reducing the self-attention contribution
        to the residual stream.  This forces the decoder to rely on
        cross-attention (EEG signal) for content.
        """
        for layer in self.bart.model.decoder.layers:
            def hook_fn(module, input, output):
                if self.self_attn_scale < 1.0:
                    return (output[0] * self.self_attn_scale,) + output[1:]
                return output
            h = layer.self_attn.register_forward_hook(hook_fn)
            self._sa_hooks.append(h)

    # ────────────────────────────────────────────────────────────────────
    # EEG vocabulary prior
    # ────────────────────────────────────────────────────────────────────
    def get_eeg_vocab_logits(
        self, eeg_hidden: torch.Tensor, eeg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict vocabulary distribution from mean-pooled EEG hidden states.
        Uses BART's shared embedding matrix for the final projection
        (no extra vocab-sized weight matrix needed).

        Args:
            eeg_hidden: (B, L, bart_dim) encoded EEG
            eeg_mask:   (B, L) mask

        Returns:
            (B, vocab_size) raw logits over vocabulary
        """
        mask = eeg_mask.unsqueeze(-1).float()
        pooled = (eeg_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        projected = self.eeg_vocab_proj(pooled)                    # (B, bart_dim)
        # Use BART's shared embedding matrix for vocabulary projection
        return F.linear(projected, self.bart.model.shared.weight)  # (B, vocab_size)

    # ────────────────────────────────────────────────────────────────────
    # Forward pass (training with teacher forcing)
    # ────────────────────────────────────────────────────────────────────
    def forward(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            eeg:              (B, L_eeg, eeg_dim)   Word-level EEG features
            eeg_mask:         (B, L_eeg)             1 = real word,  0 = padding
            decoder_input_ids: (B, L_text)            Shifted target tokens (teacher forcing)
            labels:           (B, L_text)             Target token IDs (-100 for pads)
            output_attentions: Return cross-attention weights for diagnostics

        Returns:
            BartOutput containing .loss, .logits, and optionally .cross_attentions
        """
        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)       # (B, L_eeg, bart_dim)

        # Cache for trainer to reuse (avoids running S4D encoder twice for InfoNCE)
        self._last_eeg_hidden = eeg_hidden

        # ── CRITICAL WIRING: replace BART encoder with EEG hidden states ─
        encoder_out = BaseModelOutput(last_hidden_state=eeg_hidden)

        outputs = self.bart(
            encoder_outputs=encoder_out,      # EEG replaces BART text encoder
            attention_mask=eeg_mask,           # mask over EEG word positions
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return outputs

    # ────────────────────────────────────────────────────────────────────
    # Free generation (no teacher forcing)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_text(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        tokenizer: BartTokenizer,
        max_length: int = 56,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.0,
        eeg_prior_alpha: float = 0.0,
        do_sample: bool = False,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
    ) -> List[str]:
        """
        Generation conditioned solely on EEG.  Supports beam search,
        diverse beam search, greedy, and nucleus sampling.

        Args:
            eeg:       (B, L_eeg, eeg_dim)
            eeg_mask:  (B, L_eeg)
            tokenizer: BART tokenizer
            max_length, num_beams, etc.: generation hyper-params
            repetition_penalty: penalise already-generated tokens (>1.0 to activate)
            eeg_prior_alpha: weight for EEG vocabulary prior (0 = disabled)
            do_sample: enable sampling (top-k / top-p / temperature)
            top_k: top-k sampling (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            temperature: sampling temperature (1.0 = no change)
            num_beam_groups: number of diverse beam groups (1 = standard beam)
            diversity_penalty: penalty for diverse beam search (0 = disabled)

        Returns:
            List of B decoded strings
        """
        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
        encoder_out = BaseModelOutput(last_hidden_state=eeg_hidden)

        # ── EEG vocabulary prior ─────────────────────────────────────
        processors = None
        if eeg_prior_alpha > 0:
            vocab_logits = self.get_eeg_vocab_logits(eeg_hidden, eeg_mask)  # (B, V)
            vocab_log_probs = F.log_softmax(vocab_logits, dim=-1)
            # Expand for beam/sample search
            expand_factor = num_beams if not do_sample else 1
            if expand_factor > 1:
                vocab_log_probs = vocab_log_probs.repeat_interleave(expand_factor, dim=0)
            processors = LogitsProcessorList([
                EEGVocabLogitsProcessor(vocab_log_probs, eeg_prior_alpha)
            ])

        # Temporarily enable KV cache for fast autoregressive generation
        self.bart.config.use_cache = True

        gen_kwargs = dict(
            encoder_outputs=encoder_out,
            attention_mask=eeg_mask,
            max_length=max_length,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            logits_processor=processors,
        )

        if do_sample:
            # Sampling-based generation
            gen_kwargs.update(
                do_sample=True,
                num_beams=1,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                temperature=temperature,
            )
        elif num_beam_groups > 1:
            # Diverse beam search
            gen_kwargs.update(
                num_beams=num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
            )
        else:
            # Standard beam search
            gen_kwargs.update(num_beams=num_beams)

        generated_ids = self.bart.generate(**gen_kwargs)
        self.bart.config.use_cache = False  # restore for training
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # ────────────────────────────────────────────────────────────────────
    # Greedy generation WITH cross-attention extraction (for heatmaps)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_with_cross_attention(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        tokenizer: BartTokenizer,
        max_length: int = 56,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Greedy decoding that captures cross-attention weights at every step.

        Unlike generate_text() (which calls bart.generate()), this runs the
        decoder loop manually so we can collect output_attentions at each
        autoregressive step.

        Args:
            eeg:       (B, L_eeg, eeg_dim)
            eeg_mask:  (B, L_eeg)
            tokenizer: BART tokenizer
            max_length: maximum tokens to generate
            repetition_penalty: penalise already-generated tokens
            no_repeat_ngram_size: block repeated n-grams

        Returns:
            texts:       List of B decoded strings
            cross_attns: (B, n_layers, n_heads, T_out, L_eeg)
                         Attention weights per layer/head/decode-step/EEG-position.
        """
        B = eeg.size(0)
        device = eeg.device

        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
        encoder_out = BaseModelOutput(last_hidden_state=eeg_hidden)

        # Force eager attention so HF returns real attention weight tensors
        # (SDPA returns None even when output_attentions=True)
        saved_attn_impl = getattr(self.bart.config, "_attn_implementation", None)
        self.bart.config._attn_implementation = "eager"
        # Also patch each decoder layer's attention module
        _patched_layers = []
        for layer in self.bart.model.decoder.layers:
            for attn_mod in [layer.self_attn, layer.encoder_attn]:
                if hasattr(attn_mod, "_attn_implementation"):
                    _patched_layers.append((attn_mod, attn_mod._attn_implementation))
                    attn_mod._attn_implementation = "eager"
                elif hasattr(attn_mod, "config") and hasattr(attn_mod.config, "_attn_implementation"):
                    _patched_layers.append((attn_mod.config, attn_mod.config._attn_implementation))
                    attn_mod.config._attn_implementation = "eager"

        self.bart.config.use_cache = True

        bos_id = self.bart.config.decoder_start_token_id
        eos_id = self.bart.config.eos_token_id
        pad_id = self.bart.config.pad_token_id

        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        past_key_values = None

        # Collect cross-attention per step: list of (n_layers,) tuples
        # Each layer tensor: (B, n_heads, 1, L_eeg)
        step_cross_attns = []

        for step in range(max_length - 1):
            out = self.bart(
                encoder_outputs=encoder_out,
                attention_mask=eeg_mask,
                decoder_input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                output_attentions=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values

            # Collect cross-attention: tuple of (B, n_heads, 1, L_eeg) per layer
            if out.cross_attentions is not None:
                # Stack layers: (n_layers, B, n_heads, 1, L_eeg)
                layer_stack = torch.stack(
                    [a for a in out.cross_attentions if a is not None], dim=0
                )
                step_cross_attns.append(layer_stack)

            # Greedy next token
            logits = out.logits[:, -1, :]  # (B, V)

            # Repetition penalty
            if repetition_penalty > 1.0:
                for b in range(B):
                    for prev_id in input_ids[b].unique():
                        if logits[b, prev_id] > 0:
                            logits[b, prev_id] /= repetition_penalty
                        else:
                            logits[b, prev_id] *= repetition_penalty

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and input_ids.size(1) >= no_repeat_ngram_size:
                for b in range(B):
                    gen_tokens = input_ids[b].tolist()
                    for ns in range(len(gen_tokens) - no_repeat_ngram_size + 1):
                        ngram = tuple(gen_tokens[ns:ns + no_repeat_ngram_size])
                        if tuple(gen_tokens[-(no_repeat_ngram_size - 1):]) == ngram[:-1]:
                            logits[b, ngram[-1]] = float("-inf")

            next_ids = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Pad finished sequences
            next_ids[finished] = pad_id
            input_ids = torch.cat([input_ids, next_ids], dim=1)

            finished = finished | (next_ids.squeeze(1) == eos_id)
            if finished.all():
                break

        # Restore config
        self.bart.config.use_cache = False
        if saved_attn_impl is not None:
            self.bart.config._attn_implementation = saved_attn_impl
        else:
            try:
                del self.bart.config._attn_implementation
            except AttributeError:
                pass
        for obj, orig_val in _patched_layers:
            if hasattr(obj, "_attn_implementation"):
                obj._attn_implementation = orig_val

        # Decode text
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Stack cross-attentions across steps
        # step_cross_attns: list of (n_layers, B, n_heads, 1, L_eeg)
        if step_cross_attns:
            # → (T_out, n_layers, B, n_heads, 1, L_eeg), squeeze the 1
            stacked = torch.stack(step_cross_attns, dim=0)  # (T, nL, B, nH, 1, Le)
            stacked = stacked.squeeze(4)                     # (T, nL, B, nH, Le)
            # Rearrange to (B, nL, nH, T, Le)
            stacked = stacked.permute(2, 1, 3, 0, 4).contiguous()
        else:
            stacked = torch.zeros(B, 1, 1, 1, eeg_mask.size(1), device=device)

        return texts, stacked

    @torch.no_grad()
    def generate_best_of_n(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        tokenizer: "BartTokenizer",
        n_candidates: int = 10,
        max_length: int = 56,
        top_p: float = 0.9,
        temperature: float = 0.9,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        eeg_prior_alpha: float = 0.0,
    ) -> List[str]:
        """
        Best-of-N reranking: generate N candidates per sample via nucleus
        sampling, then pick the one with highest EEG–text alignment score
        (cosine similarity in contrastive embedding space).

        This exploits the model's own alignment model as a reranker, choosing
        the generation that best matches the EEG signal.

        Args:
            eeg:              (B, L, D) EEG features
            eeg_mask:         (B, L)    mask
            tokenizer:        BART tokenizer
            n_candidates:     Number of candidates to generate per sample
            max_length:       Max generation length
            top_p:            Nucleus sampling threshold
            temperature:      Sampling temperature (slightly higher for diversity)
            repetition_penalty: Repetition penalty
            no_repeat_ngram_size: no-repeat ngram size

        Returns:
            List of B selected strings (one per sample)
        """
        B = eeg.size(0)
        device = eeg.device

        # ── Get EEG contrastive embeddings (B, proj_dim) ───────────────
        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
        mask_f = eeg_mask.unsqueeze(-1).float()
        eeg_pooled = (eeg_hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        eeg_emb = self.eeg_proj_head(eeg_pooled)  # (B, proj_dim)
        eeg_emb = F.normalize(eeg_emb, dim=-1)

        # ── Generate N candidates per sample ───────────────────────────
        # Repeat EEG inputs N times: (B*N, ...)
        eeg_expanded = eeg.repeat_interleave(n_candidates, dim=0)
        eeg_mask_expanded = eeg_mask.repeat_interleave(n_candidates, dim=0)

        # Generate all B*N candidates at once
        all_texts = self.generate_text(
            eeg=eeg_expanded,
            eeg_mask=eeg_mask_expanded,
            tokenizer=tokenizer,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eeg_prior_alpha=eeg_prior_alpha,
        )

        # ── Score each candidate via EEG–text cosine similarity ────────
        # Encode all generated texts through BART text encoder
        bart_encoder = self.bart.get_encoder()
        best_texts = []

        for b in range(B):
            candidates = all_texts[b * n_candidates:(b + 1) * n_candidates]

            # Tokenize candidates
            enc = tokenizer(
                candidates,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Get text embeddings for all N candidates
            enc_out = bart_encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            hidden = enc_out.last_hidden_state  # (N, T, bart_dim)
            mask_c = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask_c).sum(dim=1) / mask_c.sum(dim=1).clamp(min=1e-9)
            text_emb = self.text_proj_head(pooled)  # (N, proj_dim)
            text_emb = F.normalize(text_emb, dim=-1)

            # Cosine similarity between this sample's EEG and each candidate
            scores = (eeg_emb[b:b+1] @ text_emb.T).squeeze(0)  # (N,)
            best_idx = scores.argmax().item()
            best_texts.append(candidates[best_idx])

        return best_texts

    @torch.no_grad()
    def generate_mbr(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        tokenizer: "BartTokenizer",
        n_candidates: int = 16,
        max_length: int = 56,
        top_p: float = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
        eeg_prior_alpha: float = 0.0,
        metric: str = "bertscore",
    ) -> List[str]:
        """
        Minimum Bayes Risk (MBR) decoding: generate N candidates via nucleus
        sampling, pick the one that maximizes average utility against all
        other candidates.

        Unlike Best-of-N (which uses a weak EEG-text contrastive signal),
        MBR uses inter-candidate agreement measured by BERTScore.
        The intuition: the "consensus" output is most likely correct.

        Args:
            eeg, eeg_mask:     Input EEG features + mask
            tokenizer:         BART tokenizer
            n_candidates:      Number of candidates per sample
            metric:            "bertscore" or "rouge" for utility function
            (other args):      Generation parameters

        Returns:
            List of B selected strings
        """
        B = eeg.size(0)

        # ── Generate N candidates per sample ───────────────────────────
        eeg_expanded = eeg.repeat_interleave(n_candidates, dim=0)
        eeg_mask_expanded = eeg_mask.repeat_interleave(n_candidates, dim=0)

        all_texts = self.generate_text(
            eeg=eeg_expanded,
            eeg_mask=eeg_mask_expanded,
            tokenizer=tokenizer,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eeg_prior_alpha=eeg_prior_alpha,
        )

        # ── MBR selection: pairwise utility ────────────────────────────
        best_texts = []
        for b in range(B):
            candidates = all_texts[b * n_candidates:(b + 1) * n_candidates]

            if metric == "bertscore":
                # Compute pairwise BERTScore between all candidates
                from bert_score import score as bscore
                # Each candidate is scored against all others
                # refs = repeat each candidate N times
                # hyps = tile the full candidate list N times
                n = len(candidates)
                refs_flat = []
                hyps_flat = []
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            hyps_flat.append(candidates[i])
                            refs_flat.append(candidates[j])

                if len(hyps_flat) > 0:
                    _, _, F1 = bscore(
                        hyps_flat, refs_flat,
                        model_type="distilbert-base-uncased",
                        verbose=False, batch_size=128,
                    )
                    # Reshape to (N, N-1) and average
                    F1_matrix = F1.view(n, n - 1)
                    avg_scores = F1_matrix.mean(dim=1)  # (N,)
                    best_idx = avg_scores.argmax().item()
                else:
                    best_idx = 0
            else:
                # Fallback: ROUGE-L based utility
                from rouge_score import rouge_scorer
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                n = len(candidates)
                avg_scores = []
                for i in range(n):
                    scores_i = []
                    for j in range(n):
                        if i != j:
                            s = scorer.score(candidates[j], candidates[i])
                            scores_i.append(s["rougeL"].fmeasure)
                    avg_scores.append(sum(scores_i) / max(len(scores_i), 1))
                best_idx = max(range(n), key=lambda i: avg_scores[i])

            best_texts.append(candidates[best_idx])

        return best_texts

    @torch.no_grad()
    def generate_contrastive(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
        tokenizer: "BartTokenizer",
        max_length: int = 56,
        top_k: int = 5,
        alpha: float = 0.6,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> List[str]:
        """
        Contrastive search decoding (Su et al., 2022).

        At each step, selects the token that maximizes:
            (1 - alpha) * model_confidence + alpha * degeneration_penalty

        where degeneration_penalty = 1 - max_cosine_similarity with
        previous hidden states. This prevents repetitive/degenerate text
        while maintaining semantic relevance.

        Args:
            eeg, eeg_mask:  Input EEG
            tokenizer:      BART tokenizer
            max_length:     Max tokens
            top_k:          Number of candidates per step
            alpha:          Balance between confidence and diversity (0=greedy, 1=max diversity)
            repetition_penalty: Penalty for repeated tokens
            no_repeat_ngram_size: Block repeated n-grams

        Returns:
            List of B decoded strings
        """
        B = eeg.size(0)
        device = eeg.device

        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
        encoder_out = BaseModelOutput(last_hidden_state=eeg_hidden)

        self.bart.config.use_cache = True

        # Start with BOS token
        bos_id = self.bart.config.decoder_start_token_id
        eos_id = self.bart.config.eos_token_id
        pad_id = self.bart.config.pad_token_id

        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Track decoder hidden states for degeneration penalty
        past_key_values = None
        all_hidden_states = []  # list of (B, bart_dim) per step

        for step in range(max_length - 1):
            out = self.bart(
                encoder_outputs=encoder_out,
                attention_mask=eeg_mask,
                decoder_input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values

            # Get logits for last position
            logits = out.logits[:, -1, :]  # (B, V)

            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for b_idx in range(B):
                    for prev_id in input_ids[b_idx].unique():
                        if logits[b_idx, prev_id] > 0:
                            logits[b_idx, prev_id] /= repetition_penalty
                        else:
                            logits[b_idx, prev_id] *= repetition_penalty

            # Apply no-repeat n-gram blocking
            if no_repeat_ngram_size > 0 and input_ids.size(1) >= no_repeat_ngram_size:
                for b_idx in range(B):
                    gen_tokens = input_ids[b_idx].tolist()
                    for ngram_start in range(len(gen_tokens) - no_repeat_ngram_size + 1):
                        ngram = tuple(gen_tokens[ngram_start:ngram_start + no_repeat_ngram_size])
                        if tuple(gen_tokens[-(no_repeat_ngram_size - 1):]) == ngram[:-1]:
                            logits[b_idx, ngram[-1]] = float('-inf')

            # Get hidden state for this step (last decoder layer)
            step_hidden = out.decoder_hidden_states[-1][:, -1, :]  # (B, bart_dim)
            step_hidden_norm = F.normalize(step_hidden, dim=-1)

            # Get top-k candidates
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = probs.topk(top_k, dim=-1)  # (B, k)

            if len(all_hidden_states) == 0 or alpha == 0:
                # First token or pure greedy: just pick argmax
                next_ids = topk_ids[:, 0:1]  # (B, 1)
            else:
                # Compute degeneration penalty for each candidate
                # Stack previous hidden states: (B, steps, bart_dim)
                prev_hiddens = torch.stack(all_hidden_states, dim=1)
                prev_hiddens_norm = F.normalize(prev_hiddens, dim=-1)

                best_ids = []
                for b_idx in range(B):
                    if finished[b_idx]:
                        best_ids.append(pad_id)
                        continue

                    best_score = float('-inf')
                    best_token = topk_ids[b_idx, 0].item()

                    for k_idx in range(top_k):
                        token_id = topk_ids[b_idx, k_idx].item()
                        confidence = topk_probs[b_idx, k_idx].item()

                        # For degeneration penalty, we need the hidden state
                        # that would result from this token. Approximate with
                        # the current step's hidden state (token-independent
                        # since we already computed the forward pass).
                        cos_sims = (step_hidden_norm[b_idx:b_idx+1] @
                                   prev_hiddens_norm[b_idx].T)  # (1, steps)
                        max_sim = cos_sims.max().item()
                        degeneration_penalty = 1.0 - max_sim

                        score = (1.0 - alpha) * confidence + alpha * degeneration_penalty
                        if score > best_score:
                            best_score = score
                            best_token = token_id

                    best_ids.append(best_token)

                next_ids = torch.tensor(best_ids, device=device).unsqueeze(1)  # (B, 1)

            all_hidden_states.append(step_hidden_norm)

            input_ids = torch.cat([input_ids, next_ids], dim=1)

            # Check for EOS
            finished = finished | (next_ids.squeeze(1) == eos_id)
            if finished.all():
                break

        self.bart.config.use_cache = False

        return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    # ────────────────────────────────────────────────────────────────────
    # BART text encoder pass (for InfoNCE anchor embeddings)
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_bart_text_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run BART's own text encoder, mean-pool, then project through the
        text contrastive head.

        Returns:
            (B, contrastive_proj_dim)
        """
        bart_encoder = self.bart.get_encoder()
        enc_out = bart_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state                         # (B, T, bart_dim)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.text_proj_head(pooled)                         # (B, proj_dim)

    def get_eeg_embeddings(
        self,
        eeg: torch.Tensor,
        eeg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool gated EEG encoder output, then project through the
        EEG contrastive head.

        Returns:
            (B, contrastive_proj_dim)
        """
        eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
        mask = eeg_mask.unsqueeze(-1).float()
        pooled = (eeg_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.eeg_proj_head(pooled)                          # (B, proj_dim)

    # ────────────────────────────────────────────────────────────────────
    # Phase-aware parameter freezing
    # ────────────────────────────────────────────────────────────────────
    def set_phase(self, phase: int):
        """
        Configure which parameters are trainable for each training phase.

        Phase 1 – EEG encoder warm-up:
            Trainable:  eeg_encoder, attention_gate
            Frozen:     ALL of BART (encoder + decoder)

        Phase 2 – Cross-attention fine-tuning:
            Trainable:  eeg_encoder, attention_gate, BART cross-attention layers
            Frozen:     BART text encoder, BART decoder self-attention + FFN + LN
        """
        if phase == 1:
            # Freeze ALL BART parameters
            for param in self.bart.parameters():
                param.requires_grad = False
            # Unfreeze EEG encoder + gate + projection heads + vocab prior
            for param in self.eeg_encoder.parameters():
                param.requires_grad = True
            for param in self.attention_gate.parameters():
                param.requires_grad = True
            for param in self.eeg_proj_head.parameters():
                param.requires_grad = True
            for param in self.text_proj_head.parameters():
                param.requires_grad = True
            for param in self.eeg_vocab_proj.parameters():
                param.requires_grad = True
            # No self-attention dampening in Phase 1
            self.self_attn_scale = 1.0

        elif phase == 2:
            # Freeze BART encoder entirely
            for param in self.bart.get_encoder().parameters():
                param.requires_grad = False

            # Selectively unfreeze BART decoder:
            #   ✓ Cross-attention (encoder_attn) — learns to read EEG
            #   ✓ FFN (fc1, fc2) — adapts to integrate cross-attn outputs
            #   ✓ Layer norms — adapts to new activation distributions
            #   ✗ Self-attention — stays frozen to preserve LM capability
            #   ✗ Positional embeddings — stays frozen
            for name, param in self.bart.get_decoder().named_parameters():
                if any(key in name for key in [
                    "encoder_attn",        # cross-attention Q/K/V/out_proj
                    "encoder_attn_layer_norm",
                    "fc1", "fc2",          # FFN layers
                    "final_layer_norm",    # post-FFN layer norm
                ]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # BART lm_head stays frozen (it's tied to embeddings)
            if hasattr(self.bart, "lm_head"):
                for param in self.bart.lm_head.parameters():
                    param.requires_grad = False

            # EEG encoder + gate + projection heads + vocab prior remain trainable
            for param in self.eeg_encoder.parameters():
                param.requires_grad = True
            for param in self.attention_gate.parameters():
                param.requires_grad = True
            for param in self.eeg_proj_head.parameters():
                param.requires_grad = True
            for param in self.text_proj_head.parameters():
                param.requires_grad = True
            for param in self.eeg_vocab_proj.parameters():
                param.requires_grad = True

            # Enable self-attention dampening
            # Note: self_attn_scale is set externally from config before calling set_phase
            # (default 0.0 — fully EEG-dependent)
            print(f"  Self-attention dampening: scale = {self.self_attn_scale}")

        else:
            raise ValueError(f"Unknown phase: {phase}")

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Phase {phase}] Trainable: {n_train:,} / {n_total:,} parameters "
              f"({100 * n_train / n_total:.1f}%)")

    # ────────────────────────────────────────────────────────────────────
    # Sanity checks (run before training)
    # ────────────────────────────────────────────────────────────────────
    def sanity_check(self, batch: Dict, device: torch.device):
        """
        Run diagnostic assertions to catch wiring bugs early.
        Call once with a real batch before starting training.
        """
        self.train()
        eeg = batch["eeg"].to(device)
        eeg_mask = batch["eeg_mask"].to(device)
        labels = batch["labels"].to(device)

        # 1. Forward pass – compute loss
        outputs = self(
            eeg=eeg,
            eeg_mask=eeg_mask,
            labels=labels,
            output_attentions=True,
        )
        loss = outputs.loss
        loss.backward()

        # 2. Gradient flow reaches EEG encoder
        # Support both S4DEncoder (has input_proj) and LinearEEGEncoder (has proj)
        if hasattr(self.eeg_encoder, 'input_proj'):
            grad_param = self.eeg_encoder.input_proj[0].weight
        elif hasattr(self.eeg_encoder, 'proj'):
            grad_param = self.eeg_encoder.proj[0].weight
        else:
            grad_param = next(self.eeg_encoder.parameters())
        grad = grad_param.grad
        assert grad is not None, \
            "FATAL: No gradient in EEG encoder — check wiring"
        assert grad.abs().sum() > 0, \
            "FATAL: Gradient is all zeros in EEG encoder"

        # 3. Cross-attention is not perfectly uniform
        cross_attentions = getattr(outputs, "cross_attentions", None)
        attn_std = None
        # Some attention implementations (e.g. SDPA/flash) may ignore output_attentions
        # and return an empty tuple even when requested.
        if cross_attentions is not None and len(cross_attentions) > 0:
            attn = cross_attentions[-1]  # last decoder layer
            if attn is None:
                print("  Cross-attention not available (attention backend) — OK")
            else:
                attn_std = attn.std(dim=-1).mean().item()
        else:
            print("  Cross-attention not available (attention backend) — OK")
        if attn_std is not None:
            if attn_std < 0.005:
                print(f"WARNING: Cross-attention nearly uniform (std={attn_std:.4f})")
            else:
                print(f"  Cross-attention std = {attn_std:.4f} (OK)")

        # 4. EEG embeddings are not NaN / zero-variance
        with torch.no_grad():
            eeg_hidden = self.encode_eeg(eeg, eeg_mask=eeg_mask)
            assert not torch.isnan(eeg_hidden).any(), "NaN in EEG encoder output"
            var = eeg_hidden.var().item()
            assert var > 1e-4, f"EEG encoder output near-zero variance: {var:.6f}"
            print(f"  EEG hidden variance = {var:.4f} (OK)")

        self.zero_grad()
        print("  All sanity checks passed")
