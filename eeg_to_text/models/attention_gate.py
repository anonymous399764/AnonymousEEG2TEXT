"""
EEG Attention Gate Module

A learned gating mechanism applied to EEG encoder hidden states BEFORE they
enter BART's cross-attention. Its purpose is to force the decoder to engage
with EEG content by scaling encoder outputs.

Design:
    gate = σ(W · h + bias)
    output = gate ⊙ h + (1 - gate) ⊙ null_vector

The `null_vector` is a learnable "no-signal" baseline initialised to zeros.
If the gate collapses to 0 (decoder ignoring EEG), the gradient from the
attention-entropy loss will push it back open.

The bias is initialised positive (default +1.0) so the gate starts OPEN,
giving the cross-attention real EEG signal from the very first training step.
"""

import torch
import torch.nn as nn


class EEGAttentionGate(nn.Module):
    """
    Gating module for EEG encoder outputs.

    Input:  (B, L, d_model)
    Output: (B, L, d_model)   same shape, gated
    """

    def __init__(self, d_model: int = 768, bias_init: float = 1.0):
        """
        Args:
            d_model:    Dimension of encoder hidden states.
            bias_init:  Initial value for the gate bias.  >0 means the gate
                        starts open (passes EEG through).
        """
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        # Initialise bias so sigmoid(bias_init) ≈ 0.73 → gate starts mostly open
        nn.init.constant_(self.gate_proj.bias, bias_init)

        # Learnable null vector (baseline when gate is closed)
        self.null_vector = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: EEG encoder hidden states (B, L, d_model)

        Returns:
            Gated hidden states (B, L, d_model)
        """
        gate = torch.sigmoid(self.gate_proj(h))           # (B, L, d_model)
        null = self.null_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
        return gate * h + (1.0 - gate) * null
