"""
Loss Functions for EEG-to-Text Training

Four loss components:

1. L_lm         – Label-smoothed cross-entropy (next-token prediction)
2. L_contrastive – InfoNCE alignment loss (EEG ↔ BART text embeddings)
3. L_attn       – Cross-attention entropy regularisation
4. L_vocab      – EEG vocabulary prior (bag-of-words prediction from EEG)

Total:
    L = L_lm  +  λ_c · L_contrastive  +  λ_a · L_attn  +  λ_v · L_vocab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# 1. Label-Smoothed Cross-Entropy
# ════════════════════════════════════════════════════════════════════════════

class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing (ε).

    For vocabulary size V and target class y:
        p_smooth(c) = (1-ε) · 1[c=y]  +  ε/V

    This penalises overconfident predictions, acting as a regulariser that
    encourages the model to keep probability mass spread across the vocabulary
    rather than collapsing to pure one-hot outputs.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, T, V) or (B*T, V)
            targets: (B, T) or (B*T,)

        Returns:
            Scalar loss
        """
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.reshape(-1, V)
            targets = targets.reshape(-1)
        else:
            V = logits.size(-1)

        # Filter out ignore positions
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]

        if logits.numel() == 0:
            return logits.sum()  # zero loss, still differentiable

        log_probs = F.log_softmax(logits, dim=-1)

        # NLL component (hard targets)
        nll = F.nll_loss(log_probs, targets, reduction="mean")

        # Uniform component (label smoothing)
        smooth = -log_probs.mean(dim=-1).mean()

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss


# ════════════════════════════════════════════════════════════════════════════
# 2. InfoNCE Contrastive Alignment Loss
# ════════════════════════════════════════════════════════════════════════════

def info_nce_loss(
    eeg_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (CLIP-style) that aligns EEG sentence embeddings
    with BART text encoder sentence embeddings.

    Positive pairs:  (eeg_i, text_i)  — same sentence
    Negative pairs:  (eeg_i, text_j)  — different sentences in batch

    This is THE KEY FIX for cross-attention collapse: it forces the S4D encoder
    to produce vectors that live in BART's representational space.  Without it,
    cross-attention receives random-direction queries and collapses to uniform.

    Args:
        eeg_embeddings:   (B, d)  mean-pooled S4D output
        text_embeddings:  (B, d)  mean-pooled BART text encoder output (detached)
        temperature:      Softmax temperature (0.07 per original CLIP)

    Returns:
        Scalar loss
    """
    B = eeg_embeddings.size(0)
    if B < 2:
        # InfoNCE is meaningless with batch size 1 — return zero
        return eeg_embeddings.sum() * 0.0

    # L2 normalise
    eeg_norm = F.normalize(eeg_embeddings, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)

    # Cosine similarity matrix
    logits = torch.matmul(eeg_norm, text_norm.T) / temperature   # (B, B)
    labels = torch.arange(B, device=logits.device)

    # Symmetric cross-entropy
    loss_eeg2text = F.cross_entropy(logits, labels)
    loss_text2eeg = F.cross_entropy(logits.T, labels)
    return (loss_eeg2text + loss_text2eeg) / 2.0


# ════════════════════════════════════════════════════════════════════════════
# 3. Cross-Attention Entropy Regularisation
# ════════════════════════════════════════════════════════════════════════════

def attention_entropy_loss(
    cross_attention_weights: Tuple[torch.Tensor, ...],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Penalises HIGH entropy in cross-attention distributions.

    High entropy → the decoder is attending uniformly over all EEG positions,
    which means it is not using the EEG signal structure.  By minimising this
    loss we encourage *peaked* (focused) cross-attention distributions.

    Args:
        cross_attention_weights: Tuple of tensors, each (B, heads, tgt_len, src_len)
                                 (one per decoder layer)

    Returns:
        Scalar: mean entropy across layers / heads / positions
    """
    if cross_attention_weights is None or len(cross_attention_weights) == 0:
        return torch.tensor(0.0, device=device)

    valid_attn = [attn for attn in cross_attention_weights if attn is not None]
    if len(valid_attn) == 0:
        return torch.tensor(0.0, device=device)

    total = torch.tensor(0.0, device=valid_attn[0].device)
    count = 0

    for attn in valid_attn:
        # attn: (B, heads, tgt_len, src_len)
        # Entropy per position over source dimension
        entropy = -(attn * (attn + 1e-9).log()).sum(dim=-1)    # (B, heads, tgt_len)
        total = total + entropy.mean()
        count += 1

    if count == 0:
        return total
    return total / count


# ════════════════════════════════════════════════════════════════════════════
# 4. EEG Vocabulary Prior Loss (Bag-of-Words)
# ════════════════════════════════════════════════════════════════════════════

def eeg_vocab_prior_loss(
    eeg_vocab_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Binary cross-entropy for bag-of-words prediction from EEG.

    Trains the EEG encoder to predict which tokens appear in the target
    sentence from mean-pooled EEG hidden states.  During beam search, this
    prior is used via LogitsProcessor to boost relevant vocabulary items.

    Args:
        eeg_vocab_logits: (B, vocab_size) - raw logits from EEG vocab head
        labels:           (B, T)          - target token IDs (-100 = pad)
        ignore_index:     padding token ID to ignore

    Returns:
        Scalar BCE loss
    """
    B, V = eeg_vocab_logits.shape
    target = torch.zeros_like(eeg_vocab_logits)
    for b in range(B):
        valid_tokens = labels[b][labels[b] != ignore_index]
        if len(valid_tokens) > 0:
            unique_tokens = valid_tokens.unique()
            target[b].scatter_(0, unique_tokens, 1.0)
    return F.binary_cross_entropy_with_logits(eeg_vocab_logits, target)


# ════════════════════════════════════════════════════════════════════════════
# Composite Loss
# ════════════════════════════════════════════════════════════════════════════

class EEGToTextLoss(nn.Module):
    """
    Composite loss:  L = L_lm  +  λ_c · L_nce  +  λ_a · L_entropy

    During Phase 1 the attention entropy term is set to 0 (λ_a = 0).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        lambda_contrastive: float = 0.5,
        lambda_attn_entropy: float = 0.01,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lm_criterion = LabelSmoothedCrossEntropy(smoothing=label_smoothing)
        self.lambda_c = lambda_contrastive
        self.lambda_a = lambda_attn_entropy
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        eeg_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        cross_attentions: Tuple[torch.Tensor, ...] = None,
        use_attn_loss: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits:          (B, T, V) model output logits
            labels:          (B, T)    target token IDs
            eeg_embeds:      (B, d)    mean-pooled EEG encoder output
            text_embeds:     (B, d)    mean-pooled BART text encoder output
            cross_attentions: optional tuple of attention tensors
            use_attn_loss:   whether to include attention entropy (Phase 2 only)

        Returns:
            (total_loss, component_dict)
        """
        # 1. Language modelling loss
        l_lm = self.lm_criterion(logits, labels)

        # 2. InfoNCE contrastive alignment (skip if embeddings not provided)
        if eeg_embeds is not None and text_embeds is not None and self.lambda_c > 0:
            l_nce = info_nce_loss(eeg_embeds, text_embeds, self.temperature)
        else:
            l_nce = torch.tensor(0.0, device=logits.device)

        # 3. Attention entropy (Phase 2 only)
        if use_attn_loss and cross_attentions is not None:
            l_attn = attention_entropy_loss(cross_attentions, device=logits.device)
        else:
            l_attn = torch.tensor(0.0, device=logits.device)

        total = l_lm + self.lambda_c * l_nce + self.lambda_a * l_attn

        components = {
            "loss_total": total.item(),
            "loss_lm": l_lm.item(),
            "loss_nce": l_nce.item(),
            "loss_attn": l_attn.item() if isinstance(l_attn, torch.Tensor) else l_attn,
        }
        return total, components
