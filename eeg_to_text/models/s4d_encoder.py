"""
S4D-based EEG Encoder

Processes word-level EEG sequences (B, L, eeg_dim) → (B, L, bart_dim)
using stacked Diagonal State Space Model (S4D) layers.

Architecture:
    Input projection  → S4D layers  → Output MLP  → LayerNorm
    (eeg_dim → s4d_dim)  (×N-layers)  (s4d_dim → bart_dim)

Reference:
    "On the Parameterization and Initialization of Diagonal State Space Models"
    Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré (NeurIPS 2022)
    https://arxiv.org/abs/2206.11893

The S4D kernel is implemented from scratch using the diagonal SSM closed-form:
    y_k = C · diag(Ā)^k · B̄ · u_0 + D · u_k
No dependency on the `state-spaces` pip package.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ════════════════════════════════════════════════════════════════════════════
# Minimal S4D Kernel  (self-contained, no external dependency)
# ════════════════════════════════════════════════════════════════════════════

class S4DKernel(nn.Module):
    """
    Diagonal State Space Model kernel.

    State equations (continuous):
        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t) + D u(t)

    A is diagonal complex with Hurwitz (negative real) parameterisation for
    guaranteed stability.  Initialised with HiPPO-LegS for long-range memory.
    Discretised via the bilinear (Tustin) transform.
    """

    def __init__(self, d_model: int, N: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.N = N
        self.d_model = d_model

        # ── HiPPO-LegS initialisation ─────────────────────────────────
        n = torch.arange(N, dtype=torch.float32)
        A_real = -0.5 * torch.ones(N)
        A_imag = math.pi * (n + 0.5)
        B_init = torch.sqrt(2 * n + 1)

        self.A_real = nn.Parameter(A_real)              # (N,)  decay rates
        self.A_imag = nn.Parameter(A_imag)              # (N,)  frequencies

        # B: (N, d_model)
        self.B = nn.Parameter(B_init.unsqueeze(-1).expand(-1, d_model).clone())

        # C: stored as real view of complex tensor → (d_model, N, 2)
        C = torch.randn(d_model, N, dtype=torch.cfloat) * 0.5
        self.C = nn.Parameter(torch.view_as_real(C))

        # D: skip connection
        self.D = nn.Parameter(torch.randn(d_model))

        # log Δt: log-uniform initialisation
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, L: int) -> torch.Tensor:
        """
        Generate SSM convolution kernel of length L.

        Returns:
            K – real tensor of shape (d_model, L)
        """
        dt = self.log_dt.exp()                                         # (d_model,)
        A = torch.complex(self.A_real, self.A_imag)                    # (N,)

        # Bilinear discretisation
        dtA = dt.unsqueeze(-1) * A                                     # (d_model, N)
        A_bar = (1.0 + dtA / 2.0) / (1.0 - dtA / 2.0)                # (d_model, N)
        B_bar = (dt.unsqueeze(-1) * self.B.T) / (1.0 - dtA / 2.0)     # (d_model, N)

        C = torch.view_as_complex(self.C)                              # (d_model, N)

        # Power series: A_bar^0, A_bar^1, ..., A_bar^(L-1)
        ks = torch.arange(L, device=A_bar.device).float()             # (L,)  real
        A_powers = A_bar.unsqueeze(-1) ** ks.unsqueeze(0).unsqueeze(0) # (d_model, N, L)

        # K = Re[ C · (A_powers ⊙ B_bar) ]
        K = (C.unsqueeze(-1) * (A_powers * B_bar.unsqueeze(-1))).sum(dim=1).real  # (d_model, L)

        # Skip connection impulse on first timestep
        skip = torch.zeros(L, device=K.device)
        skip[0] = 1.0
        K = K + self.D.unsqueeze(-1) * skip.unsqueeze(0)

        return K  # (d_model, L)


class S4DLayer(nn.Module):
    """
    Single S4D layer: FFT convolution → GLU → dropout → residual → LayerNorm.
    Input / output shape: (B, L, d_model)   (transposed=True convention).
    """

    def __init__(self, d_model: int, N: int = 64, dropout: float = 0.1):
        super().__init__()
        self.kernel = S4DKernel(d_model, N=N)
        self.output_linear = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, D)"""
        B_size, L, D = u.shape

        # Transpose to (B, D, L) for conv
        x = u.transpose(1, 2)                                   # (B, D, L)
        K = self.kernel(L)                                       # (D, L)

        # FFT circular convolution (zero-pad to 2L to avoid aliasing)
        n = 2 * L
        x_f = torch.fft.rfft(x, n=n)
        K_f = torch.fft.rfft(K, n=n)
        y = torch.fft.irfft(x_f * K_f.unsqueeze(0), n=n)[..., :L]  # (B, D, L)

        y = y.transpose(1, 2)                                   # (B, L, D)

        # GLU gating (SiLU activation on gate)
        y = self.output_linear(y)                                # (B, L, 2D)
        y, gate = y.chunk(2, dim=-1)
        y = y * F.silu(gate)                                     # (B, L, D)

        # Residual + dropout + norm
        y = self.norm(self.dropout(y) + u)
        return y


# ════════════════════════════════════════════════════════════════════════════
# S4D EEG Encoder
# ════════════════════════════════════════════════════════════════════════════

class S4DEEGEncoder(nn.Module):
    """
    Full S4D-based EEG encoder that maps word-level EEG features to
    BART-compatible hidden states.

    Pipeline:
        Linear(eeg_dim → s4d_dim)
        → [S4D Layer × n_layers]
        → MLP projection(s4d_dim → bart_dim)
        → LayerNorm

    Bidirectional mode runs a second S4D stack on the reversed sequence
    and concatenates outputs before the MLP projection.
    """

    def __init__(
        self,
        input_dim: int = 840,
        s4d_dim: int = 512,
        n_layers: int = 6,
        state_dim: int = 64,
        dropout: float = 0.1,
        bart_dim: int = 768,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # ── Input projection ────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, s4d_dim),
            nn.GELU(),
            nn.LayerNorm(s4d_dim),
            nn.Dropout(dropout),
        )

        # ── Forward S4D stack ───────────────────────────────────────────
        self.fwd_layers = nn.ModuleList([
            S4DLayer(s4d_dim, N=state_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # ── Backward S4D stack (bidirectional) ──────────────────────────
        if bidirectional:
            self.bwd_layers = nn.ModuleList([
                S4DLayer(s4d_dim, N=state_dim, dropout=dropout)
                for _ in range(n_layers)
            ])
            mlp_input_dim = s4d_dim * 2
        else:
            self.bwd_layers = None
            mlp_input_dim = s4d_dim

        # ── Output MLP projection to BART dimension ────────────────────
        # Two-layer MLP with GELU for non-linear projection into BART space
        self.output_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, bart_dim),
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    def forward(self, x: torch.Tensor, eeg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: EEG features (batch, num_words, eeg_dim)

        Returns:
            Encoder hidden states (batch, num_words, bart_dim)
        """
        h = self.input_proj(x)                           # (B, L, s4d_dim)

        # Forward pass
        h_fwd = h
        for layer in self.fwd_layers:
            h_fwd = layer(h_fwd)

        if self.bidirectional:
            # Backward pass on time-reversed input
            h_bwd = h.flip(dims=[1])
            for layer in self.bwd_layers:
                h_bwd = layer(h_bwd)
            h_bwd = h_bwd.flip(dims=[1])                # reverse back
            h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, L, 2·s4d_dim)
        else:
            h_cat = h_fwd

        out = self.output_mlp(h_cat)                     # (B, L, bart_dim)
        out = self.output_norm(out)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Linear EEG Encoder (ablation baseline — no temporal modelling)
# ════════════════════════════════════════════════════════════════════════════

class LinearEEGEncoder(nn.Module):
    """
    Minimal EEG encoder: per-word linear projection with no temporal modelling.
    Used as ablation to prove S4D's sequential processing adds value.

    Pipeline:  Linear(eeg_dim → 768) → GELU → LayerNorm

    Each word's EEG is projected independently — no interaction across time.
    """

    def __init__(self, input_dim: int = 840, bart_dim: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, bart_dim),
            nn.GELU(),
            nn.LayerNorm(bart_dim),
        )

    def forward(self, x: torch.Tensor, eeg_mask: torch.Tensor = None) -> torch.Tensor:
        """x: (B, L, eeg_dim) → (B, L, bart_dim)"""
        return self.proj(x)


# ════════════════════════════════════════════════════════════════════════════
# BiLSTM EEG Encoder (ablation baseline)
# ════════════════════════════════════════════════════════════════════════════

class BiLSTMEEGEncoder(nn.Module):
    """
    BiLSTM-based encoder baseline for temporal EEG modelling.

    Pipeline:
        Linear(eeg_dim → lstm_input_dim)
        → BiLSTM
        → Linear(2*hidden_dim → bart_dim)
        → LayerNorm
    """

    def __init__(
        self,
        input_dim: int = 840,
        lstm_input_dim: int = 512,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        bart_dim: int = 768,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, lstm_input_dim),
            nn.GELU(),
            nn.LayerNorm(lstm_input_dim),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, bart_dim),
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    def forward(self, x: torch.Tensor, eeg_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.input_proj(x)
        if eeg_mask is not None:
            lengths = eeg_mask.sum(dim=1).clamp(min=1).to(torch.int64).cpu()
            packed = pack_padded_sequence(
                h,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            h, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1),
            )
        else:
            h, _ = self.lstm(h)
        out = self.output_proj(h)
        out = self.output_norm(out)
        return out


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        _, length, dim = x.shape
        device = x.device

        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )

        pe = torch.zeros(length, dim, device=device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)

        x = x + pe.unsqueeze(0)
        return self.dropout(x)


# ════════════════════════════════════════════════════════════════════════════
# Transformer EEG Encoder (ablation baseline)
# ════════════════════════════════════════════════════════════════════════════

class TransformerEEGEncoder(nn.Module):
    """
    Transformer encoder baseline for temporal EEG modelling.

    Pipeline:
        Linear(eeg_dim → model_dim)
        → Positional Encoding
        → TransformerEncoder × n_layers
        → Linear(model_dim → bart_dim)
        → LayerNorm
    """

    def __init__(
        self,
        input_dim: int = 840,
        model_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        bart_dim: int = 768,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=model_dim,
            dropout=dropout,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, bart_dim),
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    def forward(self, x: torch.Tensor, eeg_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.positional_encoding(h)
        src_key_padding_mask = None
        if eeg_mask is not None:
            src_key_padding_mask = (eeg_mask == 0)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        out = self.output_proj(h)
        out = self.output_norm(out)
        return out
