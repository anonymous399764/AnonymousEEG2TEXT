"""EEG encoder variants for controlled architecture comparisons.

The benchmark intentionally keeps the BART decoder, losses, data split, and
training curriculum fixed.  It changes *only* the EEG encoder, so a score
difference can be attributed to the sequence encoder rather than a different
language model.

Implemented families
--------------------
``s4d``
    The project model (imported from :mod:`s4d_encoder`).
``dss``
    A Diagonal State Space (DSS) baseline.  Unlike the project's S4D
    implementation, every hidden channel owns its diagonal modes rather than
    sharing one HiPPO-initialised spectrum across all hidden channels.
``tcn``
    A bidirectional (non-causal) dilated Temporal Convolutional Network.  It
    is a fully parallel, non-SSM temporal baseline with a receptive field
    covering the project's maximum 56-word sequence length.
``channel_graph_s4d``
    Keeps the 105 electrodes separate, mixes them through a learned graph,
    pools them into one vector per word, then applies the existing S4D
    temporal encoder.  The pickle format has no electrode names or 3-D cap
    coordinates, so this is a *learned functional channel graph*, not a claim
    of physical-distance scalp adjacency.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .s4d_encoder import (
    BiLSTMEEGEncoder,
    LinearEEGEncoder,
    S4DEEGEncoder,
    SinusoidalPositionalEncoding,
    TransformerEEGEncoder,
)


EncoderName = Literal[
    "s4d",
    "dss",
    "mamba_tiny",
    "lru_lite",
    "s5_lite",
    "h3_lite",
]


class _DSSKernel(nn.Module):
    """Parallel convolutional diagonal SSM with per-hidden-channel modes.

    This is a DSS-inspired comparison, not a wrapper around an external
    package.  Its important distinction from ``S4DKernel`` is that A, B, and
    C are independently learned for every hidden channel.  The state remains
    diagonal and the sequence operation remains an FFT convolution.
    """

    def __init__(self, d_model: int, state_dim: int, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        n = torch.arange(state_dim, dtype=torch.float32)
        self.log_a_real = nn.Parameter(torch.full((d_model, state_dim), math.log(0.5)))
        self.a_imag = nn.Parameter(math.pi * (n + 0.5).unsqueeze(0).repeat(d_model, 1))
        self.b = nn.Parameter(torch.sqrt(2 * n + 1).unsqueeze(0).repeat(d_model, 1))
        c = torch.randn(d_model, state_dim, dtype=torch.cfloat) * 0.5
        self.c = nn.Parameter(torch.view_as_real(c))
        self.d = nn.Parameter(torch.randn(d_model))
        self.log_dt = nn.Parameter(
            torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )

    def forward(self, length: int) -> torch.Tensor:
        dt = self.log_dt.exp().unsqueeze(-1)                         # (D, 1)
        a = torch.complex(-torch.exp(self.log_a_real), self.a_imag)  # (D, N)
        dt_a = dt * a
        a_bar = (1.0 + dt_a / 2.0) / (1.0 - dt_a / 2.0)
        b_bar = (dt * self.b) / (1.0 - dt_a / 2.0)
        c = torch.view_as_complex(self.c)
        powers = a_bar.unsqueeze(-1) ** torch.arange(
            length, device=a.device, dtype=a.real.dtype
        ).view(1, 1, -1)
        kernel = (c.unsqueeze(-1) * b_bar.unsqueeze(-1) * powers).sum(dim=1).real
        skip = torch.zeros(length, device=kernel.device, dtype=kernel.dtype)
        skip[0] = 1.0
        return kernel + self.d.unsqueeze(-1) * skip.unsqueeze(0)


class _DSSLayer(nn.Module):
    """DSS FFT convolution, gated projection, residual, and normalisation."""

    def __init__(self, d_model: int, state_dim: int, dropout: float):
        super().__init__()
        self.kernel = _DSSKernel(d_model, state_dim)
        self.output_linear = nn.Linear(d_model, 2 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        _, length, _ = u.shape
        x = u.transpose(1, 2)
        kernel = self.kernel(length)
        fft_length = 2 * length
        y = torch.fft.irfft(
            torch.fft.rfft(x, n=fft_length) * torch.fft.rfft(kernel, n=fft_length).unsqueeze(0),
            n=fft_length,
        )[..., :length].transpose(1, 2)
        y, gate = self.output_linear(y).chunk(2, dim=-1)
        return self.norm(u + self.dropout(y * F.silu(gate)))


class DSSEEGEncoder(nn.Module):
    """Bidirectional DSS baseline with the same width, depth, and output as S4D."""

    def __init__(
        self,
        input_dim: int = 840,
        s4d_dim: int = 512,
        n_layers: int = 6,
        state_dim: int = 64,
        dropout: float = 0.1,
        bart_dim: int = 768,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim), nn.Dropout(dropout)
        )
        self.fwd_layers = nn.ModuleList(
            [_DSSLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [_DSSLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)]
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim)
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    @staticmethod
    def _apply_layers(h: torch.Tensor, layers: nn.ModuleList, mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in layers:
            h = layer(h)
            if mask is not None:
                h = h * mask
        return h

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None:
            h = h * mask
        h_fwd = self._apply_layers(h, self.fwd_layers, mask)
        h_bwd = self._apply_layers(
            h.flip(dims=[1]), self.bwd_layers, mask.flip(dims=[1]) if mask is not None else None
        ).flip(dims=[1])
        out = self.output_norm(self.output_mlp(torch.cat([h_fwd, h_bwd], dim=-1)))
        return out * mask if mask is not None else out


class _MambaLiteLayer(nn.Module):
    """Small, dependency-free bidirectional selective SSM layer.

    This is a faithful *Mamba-style* selective recurrence: the input controls
    its step size and B/C state projections at every word position.  It is
    intentionally implemented in PyTorch rather than depending on the
    CUDA-only ``mamba_ssm`` package, which has no reliable Windows/Python 3.12
    installation path for this project.
    """

    def __init__(self, dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.b_proj = nn.Linear(dim, state_dim)
        self.c_proj = nn.Linear(dim, state_dim)
        self.log_a = nn.Parameter(torch.randn(dim, state_dim) - 2.0)
        self.d = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, dim = h.shape
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        u = F.silu(u)
        # The recurrence is short (at most 56 word positions), making this
        # transparent PyTorch implementation practical for the benchmark.
        state = h.new_zeros(batch, dim, self.log_a.size(1))
        a = -torch.exp(self.log_a).to(dtype=h.dtype)
        outputs = []
        for position in range(length):
            source = u[:, position]
            delta = F.softplus(self.delta_proj(source)).unsqueeze(-1)
            b = self.b_proj(source).unsqueeze(1)
            c = self.c_proj(source).unsqueeze(1)
            decay = torch.exp(delta * a.unsqueeze(0))
            state = decay * state + delta * b * source.unsqueeze(-1)
            outputs.append((state * c).sum(dim=-1) + self.d * source)
        y = torch.stack(outputs, dim=1)
        y = self.out_proj(y * F.silu(gate))
        return self.norm(h + self.dropout(y))


class MambaLiteEEGEncoder(nn.Module):
    """Bidirectional Mamba-style selective SSM encoder for EEG sequences."""

    def __init__(
        self, input_dim: int = 840, s4d_dim: int = 512, n_layers: int = 6,
        state_dim: int = 16, dropout: float = 0.1, bart_dim: int = 768, **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim), nn.Dropout(dropout)
        )
        self.fwd_layers = nn.ModuleList([_MambaLiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.bwd_layers = nn.ModuleList([_MambaLiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim)
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    @staticmethod
    def _run(h: torch.Tensor, layers: nn.ModuleList, mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in layers:
            h = layer(h)
            if mask is not None:
                h = h * mask
        return h

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None:
            h = h * mask
        fwd = self._run(h, self.fwd_layers, mask)
        bwd = self._run(h.flip(1), self.bwd_layers, mask.flip(1) if mask is not None else None).flip(1)
        out = self.output_norm(self.output_mlp(torch.cat([fwd, bwd], dim=-1)))
        return out * mask if mask is not None else out


class _LRULiteLayer(nn.Module):
    """Efficient real-diagonal Linear Recurrent Unit (LRU) layer."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(dim, 2 * dim)
        self.logit_decay = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, dim = h.shape
        u, gate = self.input_proj(h).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(dtype=h.dtype).unsqueeze(0)
        state = h.new_zeros(batch, dim)
        outputs = []
        for position in range(length):
            state = decay * state + (1.0 - decay) * self.b.to(h.dtype) * u[:, position]
            outputs.append(state)
        y = self.out_proj(torch.stack(outputs, dim=1) * F.silu(gate))
        return self.norm(h + self.dropout(y))


class LRULiteEEGEncoder(nn.Module):
    """Bidirectional, capacity-controlled LRU/diagonal-SSM baseline."""

    def __init__(
        self, input_dim: int = 840, s4d_dim: int = 128, n_layers: int = 2,
        dropout: float = 0.1, bart_dim: int = 768, **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim), nn.Dropout(dropout)
        )
        self.fwd_layers = nn.ModuleList([_LRULiteLayer(s4d_dim, dropout) for _ in range(n_layers)])
        self.bwd_layers = nn.ModuleList([_LRULiteLayer(s4d_dim, dropout) for _ in range(n_layers)])
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim)
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None:
            h = h * mask
        fwd, bwd = h, h.flip(1)
        for layer in self.fwd_layers:
            fwd = layer(fwd)
            if mask is not None:
                fwd = fwd * mask
        bwd_mask = mask.flip(1) if mask is not None else None
        for layer in self.bwd_layers:
            bwd = layer(bwd)
            if bwd_mask is not None:
                bwd = bwd * bwd_mask
        out = self.output_norm(self.output_mlp(torch.cat([fwd, bwd.flip(1)], dim=-1)))
        return out * mask if mask is not None else out


class _S5LiteLayer(nn.Module):
    """Small MIMO diagonal state-space layer inspired by S5.

    Unlike DSS/S4D (one diagonal bank per hidden channel), this layer has one
    shared multi-input/multi-output latent state.  It is a lightweight S5
    family baseline, not a claim of the official parallel-scan S5 codebase.
    """

    def __init__(self, dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim)
        self.b = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        self.c = nn.Parameter(torch.randn(state_dim, dim) * 0.02)
        self.logit_decay = nn.Parameter(torch.zeros(state_dim))
        self.d = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, _ = h.shape
        u, gate = self.in_proj(h).chunk(2, dim=-1)
        decay = torch.sigmoid(self.logit_decay).to(h.dtype).unsqueeze(0)
        state = h.new_zeros(batch, self.logit_decay.numel())
        outputs = []
        for position in range(length):
            state = decay * state + u[:, position] @ self.b.to(h.dtype)
            outputs.append(state @ self.c.to(h.dtype) + self.d.to(h.dtype) * u[:, position])
        y = self.out_proj(torch.stack(outputs, dim=1) * F.silu(gate))
        return self.norm(h + self.dropout(y))


class S5LiteEEGEncoder(nn.Module):
    """Bidirectional lightweight S5-style MIMO SSM encoder."""

    def __init__(self, input_dim: int = 840, s4d_dim: int = 128, n_layers: int = 2,
                 state_dim: int = 32, dropout: float = 0.1, bart_dim: int = 768, **_: object):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim))
        self.fwd_layers = nn.ModuleList([_S5LiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.bwd_layers = nn.ModuleList([_S5LiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.output = nn.Sequential(nn.Linear(2 * s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim), nn.LayerNorm(bart_dim))

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None: h = h * mask
        fwd, bwd = h, h.flip(1)
        for layer in self.fwd_layers:
            fwd = layer(fwd)
            if mask is not None: fwd = fwd * mask
        bwd_mask = mask.flip(1) if mask is not None else None
        for layer in self.bwd_layers:
            bwd = layer(bwd)
            if bwd_mask is not None: bwd = bwd * bwd_mask
        out = self.output(torch.cat([fwd, bwd.flip(1)], dim=-1))
        return out * mask if mask is not None else out


class _H3LiteLayer(nn.Module):
    """H3-inspired gated state-space plus short-convolution mixer."""

    def __init__(self, dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.ssm = _DSSLayer(dim, state_dim, dropout)
        self.short_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ssm_out = self.ssm(h)
        local = self.short_conv(h.transpose(1, 2)).transpose(1, 2)
        return self.norm(h + F.silu(self.gate(h)) * (ssm_out + local))


class H3LiteEEGEncoder(nn.Module):
    """Bidirectional H3-inspired gated SSM baseline with FFT DSS kernels."""

    def __init__(self, input_dim: int = 840, s4d_dim: int = 128, n_layers: int = 2,
                 state_dim: int = 16, dropout: float = 0.1, bart_dim: int = 768, **_: object):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim))
        self.fwd_layers = nn.ModuleList([_H3LiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.bwd_layers = nn.ModuleList([_H3LiteLayer(s4d_dim, state_dim, dropout) for _ in range(n_layers)])
        self.output = nn.Sequential(nn.Linear(2 * s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim), nn.LayerNorm(bart_dim))

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_proj(x)
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None: h = h * mask
        fwd, bwd = h, h.flip(1)
        for layer in self.fwd_layers:
            fwd = layer(fwd)
            if mask is not None: fwd = fwd * mask
        bwd_mask = mask.flip(1) if mask is not None else None
        for layer in self.bwd_layers:
            bwd = layer(bwd)
            if bwd_mask is not None: bwd = bwd * bwd_mask
        out = self.output(torch.cat([fwd, bwd.flip(1)], dim=-1))
        return out * mask if mask is not None else out


class _TCNResidualBlock(nn.Module):
    """Two non-causal dilated convolutions with a residual connection."""

    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.transpose(1, 2)
        x = self.dropout(F.gelu(self.conv1(x)))
        x = self.dropout(F.gelu(self.conv2(x))).transpose(1, 2)
        return self.norm(h + x)


class TCNEEGEncoder(nn.Module):
    """Parallel dilated-convolution encoder with a full-sentence receptive field."""

    def __init__(
        self,
        input_dim: int = 840,
        s4d_dim: int = 512,
        n_layers: int = 6,
        dropout: float = 0.1,
        bart_dim: int = 768,
        **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, s4d_dim), nn.GELU(), nn.LayerNorm(s4d_dim), nn.Dropout(dropout)
        )
        self.positional_encoding = SinusoidalPositionalEncoding(s4d_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [_TCNResidualBlock(s4d_dim, dilation=2 ** layer, dropout=dropout) for layer in range(n_layers)]
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(s4d_dim, bart_dim), nn.GELU(), nn.Linear(bart_dim, bart_dim)
        )
        self.output_norm = nn.LayerNorm(bart_dim)

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.positional_encoding(self.input_proj(x))
        mask = eeg_mask.unsqueeze(-1).to(h.dtype) if eeg_mask is not None else None
        if mask is not None:
            h = h * mask
        for layer in self.layers:
            h = layer(h)
            if mask is not None:
                h = h * mask
        out = self.output_norm(self.output_mlp(h))
        return out * mask if mask is not None else out


class _LearnedChannelGraphBlock(nn.Module):
    """Message passing over electrodes with a learned, shared adjacency matrix."""

    def __init__(self, n_channels: int, dim: int, graph_rank: int, dropout: float):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(n_channels, graph_rank) * 0.02)
        self.message = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, C, D). The learned graph remains identical across words,
        # while the messages themselves are data dependent.
        scale = math.sqrt(self.node_embedding.size(-1))
        adjacency = torch.softmax((self.node_embedding @ self.node_embedding.T) / scale, dim=-1)
        messages = torch.einsum("ij,bljd->blid", adjacency, h)
        return self.norm(h + self.dropout(F.gelu(self.message(messages))))


class ChannelGraphS4DEEGEncoder(nn.Module):
    """Channel-structured, learned-graph encoder followed by bidirectional S4D.

    This is intentionally not named a physical scalp graph: the pickle files
    contain 105 ordered channels but not channel labels or cap coordinates.
    Supply a verified montage before claiming physical-distance adjacency in a
    paper.
    """

    def __init__(
        self,
        input_dim: int = 840,
        n_channels: int = 105,
        n_bands: int = 8,
        channel_dim: int = 128,
        graph_layers: int = 2,
        graph_rank: int = 16,
        s4d_dim: int = 512,
        n_layers: int = 6,
        state_dim: int = 64,
        dropout: float = 0.1,
        bart_dim: int = 768,
    ):
        super().__init__()
        if input_dim != n_channels * n_bands:
            raise ValueError(
                f"ChannelGraphS4D needs input_dim = n_channels * n_bands; got "
                f"{input_dim} != {n_channels} * {n_bands}"
            )
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.channel_proj = nn.Sequential(nn.Linear(n_bands, channel_dim), nn.GELU(), nn.LayerNorm(channel_dim))
        self.graph_layers = nn.ModuleList(
            [_LearnedChannelGraphBlock(n_channels, channel_dim, graph_rank, dropout) for _ in range(graph_layers)]
        )
        self.pool_query = nn.Parameter(torch.randn(channel_dim) * 0.02)
        self.temporal_encoder = S4DEEGEncoder(
            input_dim=channel_dim,
            s4d_dim=s4d_dim,
            n_layers=n_layers,
            state_dim=state_dim,
            dropout=dropout,
            bart_dim=bart_dim,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor, eeg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, length, _ = x.shape
        # Current preprocessing concatenates [band_1(105), ..., band_8(105)].
        h = x.view(batch, length, self.n_bands, self.n_channels).permute(0, 1, 3, 2)
        h = self.channel_proj(h)  # (B, L, channels, channel_dim)
        mask = None
        if eeg_mask is not None:
            mask = eeg_mask[:, :, None, None].to(h.dtype)
            h = h * mask
        for layer in self.graph_layers:
            h = layer(h)
            if mask is not None:
                h = h * mask
        scores = (h * self.pool_query).sum(dim=-1) / math.sqrt(h.size(-1))
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=2)
        if eeg_mask is not None:
            pooled = pooled * eeg_mask.unsqueeze(-1).to(pooled.dtype)
        return self.temporal_encoder(pooled, eeg_mask=eeg_mask)


def build_eeg_encoder(
    name: EncoderName,
    *,
    input_dim: int = 840,
    s4d_dim: int = 512,
    n_layers: int = 6,
    state_dim: int = 64,
    dropout: float = 0.1,
    bart_dim: int = 768,
    n_channels: int = 105,
    n_bands: int = 8,
) -> nn.Module:
    """Construct one encoder for a controlled benchmark run."""
    common = dict(
        input_dim=input_dim,
        s4d_dim=s4d_dim,
        n_layers=n_layers,
        state_dim=state_dim,
        dropout=dropout,
        bart_dim=bart_dim,
    )
    if name == "s4d":
        return S4DEEGEncoder(**common, bidirectional=True)
    if name == "dss":
        return DSSEEGEncoder(**common)
    if name == "mamba_tiny":
        # A deliberately small Mamba-style selective SSM for a feasible
        # single-GPU comparison.  It must be reported as an efficiency
        # baseline, not as capacity-matched to S4D/DSS.
        return MambaLiteEEGEncoder(
            input_dim=input_dim, s4d_dim=128, n_layers=2, state_dim=4,
            dropout=dropout, bart_dim=bart_dim,
        )
    if name == "lru_lite":
        return LRULiteEEGEncoder(
            input_dim=input_dim, s4d_dim=128, n_layers=2,
            dropout=dropout, bart_dim=bart_dim,
        )
    if name == "s5_lite":
        return S5LiteEEGEncoder(
            input_dim=input_dim, s4d_dim=128, n_layers=2, state_dim=32,
            dropout=dropout, bart_dim=bart_dim,
        )
    if name == "h3_lite":
        return H3LiteEEGEncoder(
            input_dim=input_dim, s4d_dim=128, n_layers=2, state_dim=16,
            dropout=dropout, bart_dim=bart_dim,
        )
    raise ValueError(f"Unknown encoder '{name}'")


BENCHMARK_ENCODERS: tuple[str, ...] = (
    "s4d",
    "dss",
    "mamba_tiny",
    "lru_lite",
    "s5_lite",
    "h3_lite",
)
