"""Custom layers that researchers can create via the UI."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A basic residual block: output = F(x) + x."""

    def __init__(self, in_features: int, hidden_features: int | None = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excite(scale).view(b, c, 1, 1)
        return x * scale


class GatedLinearUnit(nn.Module):
    """GLU: output = (Wx + b) * sigmoid(Vx + c)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.sigmoid(self.gate(x))


class SwishLinear(nn.Module):
    """Linear layer with Swish activation: x * sigmoid(beta * x)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x * torch.sigmoid(self.beta * x)


class TransformerBlock(nn.Module):
    """Self-attention + FFN transformer encoder block for sequential use.

    Works in nn.Sequential: takes (batch, seq_len, d_model) tensors.
    Includes LayerNorm, MultiheadAttention, and a 2-layer FFN with residual connections.
    """

    def __init__(self, d_model: int, n_heads: int = 4, ffn_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs.

    Input: (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model) with positions added.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Embedding layer that reshapes flat input into (batch, seq_len, d_model).

    For tabular data: treats each feature as a "token" with learned embedding.
    Input: (batch, in_features)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, in_features: int, d_model: int, seq_len: int | None = None):
        super().__init__()
        self.seq_len = seq_len or in_features
        self.proj = nn.Linear(in_features, self.seq_len * d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self.proj(x).view(x.size(0), self.seq_len, self.d_model)
        return x


class SequencePool(nn.Module):
    """Pool sequence output to a single vector for classification.

    Input: (batch, seq_len, d_model)
    Output: (batch, d_model)
    Supports: 'mean', 'cls' (first token), 'max' pooling.
    """

    def __init__(self, d_model: int, mode: str = 'mean'):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x
        if self.mode == 'cls':
            return x[:, 0]
        elif self.mode == 'max':
            return x.max(dim=1).values
        else:
            return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Vision Components
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ViT-style patch embedding for images.

    Input: (B, C, H, W) → Output: (B, num_patches, d_model)
    """
    def __init__(self, in_channels: int = 3, d_model: int = 256,
                 patch_size: int = 16, image_size: int = 224):
        super().__init__()
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_emb[:, :x.shape[1]]


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNet-style). More efficient than standard Conv2d."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1)
        return x * w


class UpsampleBlock(nn.Module):
    """Upsample + Conv for decoder/generator architectures."""
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(self.up(x))))


class GlobalAvgPool(nn.Module):
    """Global average pooling. (B, C, H, W) → (B, C)"""
    def forward(self, x):
        return x.mean(dim=[-2, -1])


class Reshape(nn.Module):
    """Reshape tensor. Useful for connecting Conv → Linear or vice versa."""
    def __init__(self, shape: list[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


# ---------------------------------------------------------------------------
# Audio Components
# ---------------------------------------------------------------------------

class MelSpectrogram(nn.Module):
    """Convert raw waveform to mel spectrogram features.

    Input: (B, 1, T) waveform → Output: (B, n_mels, T') spectrogram
    """
    def __init__(self, n_mels: int = 80, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Learnable mel filterbank
        self.conv = nn.Conv1d(1, n_mels, kernel_size=n_fft, stride=hop_length, bias=False)

    def forward(self, x):
        # x: (B, 1, T) or (B, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return torch.abs(self.conv(x))


class AudioConvBlock(nn.Module):
    """1D convolution block for audio processing."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, groups=groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Transpose(nn.Module):
    """Transpose dimensions. Default: swap dim 1 and 2 (for Conv1d ↔ Transformer)."""
    def __init__(self, dim0: int = 1, dim1: int = 2):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


# ---------------------------------------------------------------------------
# Video Components
# ---------------------------------------------------------------------------

class Conv3dBlock(nn.Module):
    """3D convolution block for video/volumetric data.

    Input: (B, C, D, H, W) → Output: (B, out_channels, D', H', W')
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TemporalPool(nn.Module):
    """Pool across temporal dimension. (B, C, T, H, W) → (B, C, H, W)"""
    def __init__(self, mode: str = 'mean'):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if self.mode == 'mean':
            return x.mean(dim=2)
        elif self.mode == 'max':
            return x.max(dim=2).values
        return x[:, :, 0]  # first frame


# ---------------------------------------------------------------------------
# Diffusion Components
# ---------------------------------------------------------------------------

class SinusoidalTimestepEmbed(nn.Module):
    """Sinusoidal timestep embedding for diffusion models.

    Input: (B,) integer timesteps → Output: (B, d_model)
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, t):
        half = self.d_model // 2
        emb = torch.exp(-torch.arange(half, device=t.device).float() * (math.log(10000.0) / half))
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class ConditionalBatchNorm2d(nn.Module):
    """Conditional BatchNorm — modulated by an external conditioning signal.

    Used in GANs and conditional generation.
    """
    def __init__(self, num_features: int, cond_dim: int = 256):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, x, cond=None):
        out = self.bn(x)
        if cond is not None:
            gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
            beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
            out = out * (1 + gamma) + beta
        return out


class ResConvBlock(nn.Module):
    """Residual convolution block (used in UNet, ResNet, diffusion).

    Conv → Norm → Act → Conv → Norm → Add residual
    """
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class DownBlock(nn.Module):
    """Downsample block: Conv(stride=2) + ResConv. For UNet encoder / image encoder."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.res = ResConvBlock(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.res(self.act(self.down(x)))


class UpBlock(nn.Module):
    """Upsample block: Upsample + Conv + ResConv. For UNet decoder / image decoder."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res = ResConvBlock(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.res(self.act(self.conv(self.up(x))))


import math
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State-Space Models (SSM) — Mamba / S4 / S5 primitives
# ---------------------------------------------------------------------------

class SelectiveScan(nn.Module):
    """Mamba-style Selective State Space Model.

    Pure PyTorch implementation of the selective scan mechanism.
    Input: (B, L, D) → Output: (B, L, D)

    Architecture:
      x → Linear(D, 2*expand) → split → [Conv1d → SiLU → SSM] * [SiLU gate]
      → Linear(expand, D)

    The SSM computes: h_t = A * h_{t-1} + B * x_t;  y_t = C * h_t
    where A, B, C are input-dependent (selective).
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel,
                                padding=conv_kernel - 1, groups=d_inner)

        # SSM parameters — input-dependent (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A is structured (diagonal, log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Input projection + gate split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Conv1d (causal)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters from input
        x_params = self.x_proj(x_conv)  # (B, L, d_state*2 + 1)
        B_ssm = x_params[:, :, :self.d_state]  # (B, L, d_state)
        C_ssm = x_params[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        dt = F.softplus(self.dt_proj(x_params[:, :, -1:]))  # (B, L, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (sequential, pure PyTorch)
        d_inner = x_conv.shape[-1]
        h = torch.zeros(B, d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            # h = exp(A * dt) * h + B * x * dt
            dt_t = dt[:, t, :]  # (B, d_inner)
            A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (B, d_inner, d_state)
            B_bar = B_ssm[:, t, :].unsqueeze(1) * dt_t.unsqueeze(-1)  # (B, d_inner, d_state)

            h = A_bar * h + B_bar * x_conv[:, t, :].unsqueeze(-1)
            y_t = (h * C_ssm[:, t, :].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + x_conv * self.D  # Skip connection with D

        # Gate and project out
        y = y * F.silu(z)
        return self.drop(self.out_proj(y))


class MambaBlock(nn.Module):
    """Complete Mamba block: Norm → SelectiveScan → Residual.

    Drop-in replacement for a Transformer block.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)  # Using RMSNorm would be better but LayerNorm is simpler
        self.ssm = SelectiveScan(d_model, d_state, expand, conv_kernel, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# RWKV — Receptance Weighted Key Value (linear attention RNN)
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """RWKV-style block: Time-mixing (linear attention) + Channel-mixing (FFN).

    Pure PyTorch implementation of the WKV (weighted key-value) mechanism.
    O(n) complexity — processes sequences as a linear recurrence.
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Time-mixing (linear attention via recurrence)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # Shift sequence by 1
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Learnable time decay
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.1)
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.1)

        # Time mix ratios
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)

        # Channel-mixing (FFN)
        self.chan_norm = nn.LayerNorm(d_model)
        self.ffn_key = nn.Linear(d_model, d_model * 4, bias=False)
        self.ffn_value = nn.Linear(d_model * 4, d_model, bias=False)
        self.ffn_receptance = nn.Linear(d_model, d_model, bias=False)

    def _time_mixing(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_prev = self.time_shift(x)  # Shift by 1 timestep

        # Mix current and previous
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # WKV computation (linear recurrence)
        w = -torch.exp(self.time_decay)
        u = self.time_first

        # Sequential scan
        wkv = torch.zeros(B, D, device=x.device)
        wk_sum = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(L):
            kt = k[:, t]
            vt = v[:, t]

            # First step uses time_first bonus
            if t == 0:
                wkv = torch.exp(u + kt) * vt
                wk_sum = torch.exp(u + kt)
            else:
                wkv = torch.exp(w) * wkv + torch.exp(kt) * vt
                wk_sum = torch.exp(w) * wk_sum + torch.exp(kt)

            outputs.append(wkv / (wk_sum + 1e-8))

        out = torch.stack(outputs, dim=1)  # (B, L, D)
        return self.output(r * out)

    def _channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = self.time_shift(x)
        xk = x * 0.5 + x_prev * 0.5

        k = torch.relu(self.ffn_key(xk)) ** 2  # Squared ReLU
        r = torch.sigmoid(self.ffn_receptance(x))
        return r * self.ffn_value(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._time_mixing(self.time_norm(x))
        x = x + self._channel_mixing(self.chan_norm(x))
        return x


# ---------------------------------------------------------------------------
# RetNet — Retentive Network (parallel/recurrent/chunk modes)
# ---------------------------------------------------------------------------

class RetentionLayer(nn.Module):
    """Multi-Scale Retention (RetNet core mechanism).

    Combines benefits of Transformers (parallel training) with RNNs (O(1) inference).
    This implements the parallel mode for training.

    Formula: Retention(X) = (QK^T ⊙ D) V, where D is the causal decay mask.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)  # Swish gate

        self.norm = nn.LayerNorm(self.head_dim)

        # Per-head decay rates (log-space for stability)
        decay = 1.0 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), n_heads))
        self.register_buffer("decay", decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Build decay mask D: D[i,j] = gamma^(i-j) for i >= j, 0 otherwise
        positions = torch.arange(L, device=x.device).float()
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        # Per-head decay
        decay_mask = self.decay.view(-1, 1, 1) ** distance.unsqueeze(0)  # (n_heads, L, L)
        decay_mask = decay_mask * (distance >= 0).float().unsqueeze(0)  # Causal

        # Retention: (Q K^T ⊙ D) V
        retention = torch.matmul(q, k.transpose(-2, -1)) * decay_mask.unsqueeze(0)
        out = torch.matmul(retention, v)

        # GroupNorm per head + swish gate
        out = self.norm(out)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        gate = F.silu(self.g_proj(x))
        return self.o_proj(out * gate)


class RetNetBlock(nn.Module):
    """Complete RetNet block: Norm → Retention → Residual → Norm → FFN → Residual."""

    def __init__(self, d_model: int, n_heads: int = 4, ffn_dim: int = None,
                 dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.retention = RetentionLayer(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.retention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Hyena — Long Convolution with Implicit Parametrization
# ---------------------------------------------------------------------------

class HyenaOperator(nn.Module):
    """Hyena operator: sub-quadratic attention alternative via long convolutions.

    Uses implicitly parametrized convolution filters instead of attention.
    Complexity: O(N log N) via FFT.
    """

    def __init__(self, d_model: int, max_len: int = 2048, order: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.order = order  # Number of implicit convolution layers

        # Input projections (order+1 projections for interleaved gating)
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model, bias=False)

        # Implicit convolution filters (parametrized by a small network)
        self.filter_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, d_model),
            )
            for _ in range(order)
        ])

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _generate_filter(self, filter_net: nn.Module, L: int, device) -> torch.Tensor:
        """Generate convolution filter from implicit network."""
        t = torch.linspace(0, 1, L, device=device).unsqueeze(-1)  # (L, 1)
        h = filter_net(t)  # (L, d_model)
        return h.transpose(0, 1)  # (d_model, L)

    def _fft_conv(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Causal convolution via FFT. x: (B, D, L), h: (D, L)"""
        L = x.shape[-1]
        fft_size = 2 * L  # Pad to avoid circular convolution
        x_f = torch.fft.rfft(x, n=fft_size)
        h_f = torch.fft.rfft(h, n=fft_size)
        out = torch.fft.irfft(x_f * h_f.unsqueeze(0), n=fft_size)
        return out[:, :, :L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Project input into (order+1) streams
        projected = self.in_proj(x)  # (B, L, (order+1)*D)
        splits = projected.chunk(self.order + 1, dim=-1)  # List of (B, L, D)

        # First split is the value (v)
        v = splits[0].transpose(1, 2)  # (B, D, L)

        # Apply interleaved long convolution + gating
        for i in range(self.order):
            h = self._generate_filter(self.filter_nets[i], L, x.device)
            v = self._fft_conv(v, h)
            gate = splits[i + 1].transpose(1, 2)  # (B, D, L)
            v = v * gate

        out = v.transpose(1, 2)  # (B, L, D)
        return self.drop(self.out_proj(out))


class HyenaBlock(nn.Module):
    """Complete Hyena block: Norm → HyenaOperator → Residual → Norm → FFN → Residual."""

    def __init__(self, d_model: int, max_len: int = 2048, order: int = 2,
                 ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.hyena = HyenaOperator(d_model, max_len, order, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.hyena(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# xLSTM — Extended LSTM with Exponential Gating
# ---------------------------------------------------------------------------

class SLSTMCell(nn.Module):
    """sLSTM cell with exponential gating (scalar LSTM variant from xLSTM paper).

    Key difference from standard LSTM:
    - Exponential input gate: i = exp(W_i x + R_i h)
    - Exponential forget gate: f = exp(W_f x + R_f h)
    - Normalizer stabilization for numerical stability
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size, 4 * hidden_size)  # i, f, o, z gates
        self.R = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x: torch.Tensor, state=None):
        """x: (B, input_size), state: tuple of (h, c, n) or None"""
        B = x.shape[0]
        H = self.hidden_size

        if state is None:
            h = torch.zeros(B, H, device=x.device)
            c = torch.zeros(B, H, device=x.device)
            n = torch.ones(B, H, device=x.device)  # normalizer
        else:
            h, c, n = state

        gates = self.W(x) + self.R(h)
        i_gate, f_gate, o_gate, z_gate = gates.chunk(4, dim=-1)

        # Exponential gates (clamped for stability)
        i_gate = torch.exp(torch.clamp(i_gate, max=20))
        f_gate = torch.exp(torch.clamp(f_gate, max=20))
        o_gate = torch.sigmoid(o_gate)
        z_gate = torch.tanh(z_gate)

        # Update cell state with exponential gating
        c_new = f_gate * c + i_gate * z_gate
        n_new = f_gate * n + i_gate  # normalizer tracks scale
        h_new = o_gate * (c_new / (n_new + 1e-8))

        return h_new, (h_new, c_new, n_new)


class XLSTM(nn.Module):
    """xLSTM sequence model — stacks sLSTM cells for sequence processing.

    Input: (B, L, D) → Output: (B, L, D)
    Drop-in replacement for Transformer/Mamba blocks.
    """

    def __init__(self, d_model: int, hidden_size: int = None, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        hidden_size = hidden_size or d_model
        self.cells = nn.ModuleList([
            SLSTMCell(d_model if i == 0 else hidden_size, hidden_size)
            for i in range(n_layers)
        ])
        self.proj = nn.Linear(hidden_size, d_model) if hidden_size != d_model else nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_normed = self.norm(x)

        outputs = []
        states = [None] * len(self.cells)

        for t in range(L):
            h = x_normed[:, t, :]
            for i, cell in enumerate(self.cells):
                h, states[i] = cell(h, states[i])
            outputs.append(h)

        out = torch.stack(outputs, dim=1)  # (B, L, hidden_size)
        return residual + self.drop(self.proj(out))


# ---------------------------------------------------------------------------
# Griffin — Gated Linear Recurrence (Google DeepMind, 2024)
# ---------------------------------------------------------------------------

class GatedLinearRecurrence(nn.Module):
    """Gated Linear Recurrence Unit (from Griffin / Hawk papers).

    Combines linear recurrence with gating for efficient sequence modeling.
    h_t = a_t ⊙ h_{t-1} + (1 - a_t) ⊙ x_t, where a_t is input-dependent.
    """

    def __init__(self, d_model: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.gate_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal conv
        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Linear recurrence with input-dependent gate
        a = torch.sigmoid(self.gate_proj(x_conv))  # Forget gate

        # Sequential recurrence
        h = torch.zeros_like(x_conv[:, 0, :])
        outputs = []
        for t in range(L):
            h = a[:, t, :] * h + (1 - a[:, t, :]) * x_conv[:, t, :]
            outputs.append(h)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return residual + self.drop(self.out_proj(y))


# ---------------------------------------------------------------------------
# CNN Backbone Blocks (ResNet, EfficientNet, ConvNeXt)
# ---------------------------------------------------------------------------

class ResNetBlock(nn.Module):
    """Standard ResNet bottleneck block.

    Architecture: Conv1x1→BN→ReLU → Conv3x3→BN→ReLU → Conv1x1→BN + Skip → ReLU
    Used as vision backbone encoder for multimodal models.
    """
    def __init__(self, in_channels: int, mid_channels: int = None, out_channels: int = None,
                 stride: int = 1):
        super().__init__()
        mid_channels = mid_channels or in_channels // 4
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity() if (in_channels == out_channels and stride == 1) else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + identity)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block — modernized ConvNet design.

    Architecture: DWConv7x7 → LayerNorm → Linear → GELU → Linear + Skip
    """
    def __init__(self, channels: int, expand_ratio: int = 4):
        super().__init__()
        hidden = channels * expand_ratio
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x + identity


class MBConvBlock(nn.Module):
    """MBConv (Mobile Inverted Bottleneck) — EfficientNet building block.

    Architecture: Conv1x1(expand) → DWConv → SE → Conv1x1(project) + Skip
    """
    def __init__(self, in_channels: int, out_channels: int = None, expand_ratio: int = 4,
                 kernel_size: int = 3, stride: int = 1, se_ratio: float = 0.25):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden = in_channels * expand_ratio
        self.use_skip = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, hidden, 1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = out * self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = out + identity
        return out


# ---------------------------------------------------------------------------
# Vision Encoder (stackable backbone for multimodal models)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Configurable vision encoder backbone.

    Stacks conv blocks to encode images into feature sequences.
    Input: (B, C, H, W) → Output: (B, num_features, d_model)

    backbone: "resnet", "convnext", "mbconv", "simple"
    """
    def __init__(self, d_model: int = 512, in_channels: int = 3,
                 backbone: str = "simple", n_stages: int = 4,
                 base_channels: int = 64):
        super().__init__()
        self.d_model = d_model

        layers = []
        ch_in = in_channels
        ch = base_channels

        for i in range(n_stages):
            ch_out = ch * (2 ** i)
            if backbone == "resnet":
                layers.append(ResNetBlock(ch_in if i == 0 else ch * (2 ** (i-1)),
                                         out_channels=ch_out, stride=2 if i > 0 else 1))
                if i == 0:
                    layers.insert(0, nn.Sequential(
                        nn.Conv2d(in_channels, ch_in if i == 0 else ch, 7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(ch_in if i == 0 else ch),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2, padding=1),
                    ))
                    ch_in = ch
            elif backbone == "convnext":
                if i == 0:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, ch_out, 4, stride=4),
                        nn.GroupNorm(1, ch_out),
                    ))
                else:
                    layers.append(nn.Sequential(
                        nn.GroupNorm(1, ch * (2 ** (i-1))),
                        nn.Conv2d(ch * (2 ** (i-1)), ch_out, 2, stride=2),
                    ))
                layers.append(ConvNeXtBlock(ch_out))
            elif backbone == "mbconv":
                if i == 0:
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_channels, ch_out, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(ch_out),
                        nn.SiLU(inplace=True),
                    ))
                else:
                    layers.append(MBConvBlock(ch * (2 ** (i-1)), ch_out, stride=2))
            else:  # simple
                layers.append(nn.Sequential(
                    nn.Conv2d(ch_in if i == 0 else ch * (2 ** (i-1)), ch_out, 3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.GELU(),
                ))
            ch_in = ch_out

        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(ch * (2 ** (n_stages - 1)), d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, H, W)
        features = self.backbone(x)  # (B, C', H', W')
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', C')
        return self.norm(self.proj(features))  # (B, N, d_model)


# ---------------------------------------------------------------------------
# Diffusion Components (Full)
# ---------------------------------------------------------------------------

class DiffusionTimestepBlock(nn.Module):
    """ResNet block conditioned on diffusion timestep.

    Used in UNet diffusion models. Injects timestep embedding via AdaGN.
    """
    def __init__(self, channels: int, time_dim: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Inject timestep: scale and shift
        t = self.time_proj(self.act(t_emb))
        scale, shift = t.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialAttentionBlock(nn.Module):
    """Self-attention over spatial dimensions for diffusion UNet.

    Input: (B, C, H, W) → reshape to (B, H*W, C) → attention → reshape back
    """
    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)  # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


class CrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning (text→image in diffusion models).

    Query from image features, Key/Value from text embeddings.
    """
    def __init__(self, channels: int, context_dim: int = 512, n_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.to_out = nn.Linear(channels, channels)
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        ctx = self.norm_ctx(context)  # (B, L, context_dim)

        q = self.to_q(h).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.to_out(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class DiffusionUNet(nn.Module):
    """Complete UNet for diffusion models (text-to-image / text-to-video).

    Architecture:
      Encoder: [DownBlock with ResBlocks + Attention] x N
      Middle: ResBlock + Attention + ResBlock
      Decoder: [UpBlock with ResBlocks + Attention + Skip-connections] x N

    Supports text conditioning via cross-attention.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 base_channels: int = 128, channel_mults: tuple = (1, 2, 4, 8),
                 n_res_blocks: int = 2, attn_resolutions: tuple = (2, 3),
                 time_dim: int = 512, context_dim: int = 512,
                 n_heads: int = 8):
        super().__init__()
        self.time_dim = time_dim

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = base_channels
        encoder_channels = [ch]

        for level, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            for _ in range(n_res_blocks):
                block = nn.ModuleList([DiffusionTimestepBlock(ch, time_dim, ch_out)])
                if level in attn_resolutions:
                    block.append(SpatialAttentionBlock(ch_out, n_heads))
                    block.append(CrossAttentionBlock(ch_out, context_dim, n_heads))
                self.encoder_blocks.append(block)
                ch = ch_out
                encoder_channels.append(ch)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                encoder_channels.append(ch)

        # Middle
        self.mid_block1 = DiffusionTimestepBlock(ch, time_dim)
        self.mid_attn = SpatialAttentionBlock(ch, n_heads)
        self.mid_cross = CrossAttentionBlock(ch, context_dim, n_heads)
        self.mid_block2 = DiffusionTimestepBlock(ch, time_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(len(channel_mults))):
            ch_out = base_channels * channel_mults[level]
            for i in range(n_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                block = nn.ModuleList([DiffusionTimestepBlock(ch + skip_ch, time_dim, ch_out)])
                if level in attn_resolutions:
                    block.append(SpatialAttentionBlock(ch_out, n_heads))
                    block.append(CrossAttentionBlock(ch_out, context_dim, n_heads))
                self.decoder_blocks.append(block)
                ch = ch_out
            if level > 0:
                self.upsamplers.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def _timestep_embedding(self, t, dim):
        half = dim // 2
        emb = torch.exp(-torch.arange(half, device=t.device).float() * (math.log(10000.0) / half))
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, x, timesteps, context=None):
        t_emb = self._timestep_embedding(timesteps, self.time_dim // 4)
        t_emb = self.time_embed(t_emb)

        if context is None:
            context = torch.zeros(x.shape[0], 1, 512, device=x.device)

        h = self.input_conv(x)
        skips = [h]

        # Encoder
        ds_idx = 0
        for block in self.encoder_blocks:
            h = block[0](h, t_emb)
            for layer in block[1:]:
                if isinstance(layer, CrossAttentionBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)
            skips.append(h)
            if ds_idx < len(self.downsamplers) and len(skips) % 3 == 0:
                h = self.downsamplers[ds_idx](h)
                skips.append(h)
                ds_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_cross(h, context)
        h = self.mid_block2(h, t_emb)

        # Decoder
        us_idx = 0
        for block in self.decoder_blocks:
            skip = skips.pop() if skips else torch.zeros_like(h)
            if h.shape != skip.shape:
                skip = torch.nn.functional.interpolate(skip, size=h.shape[2:])
                if skip.shape[1] != h.shape[1]:
                    skip = skip[:, :h.shape[1]] if skip.shape[1] > h.shape[1] else torch.nn.functional.pad(skip, (0, 0, 0, 0, 0, h.shape[1] - skip.shape[1]))
            h = torch.cat([h, skip], dim=1)
            h = block[0](h, t_emb)
            for layer in block[1:]:
                if isinstance(layer, CrossAttentionBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)
            if us_idx < len(self.upsamplers):
                h = self.upsamplers[us_idx](h)
                us_idx += 1

        return self.out_conv(torch.nn.functional.silu(self.out_norm(h)))


class VAE(nn.Module):
    """Variational Autoencoder for latent diffusion.

    Encodes images to latent space, decodes latent back to images.
    Used as the first stage in latent diffusion models (Stable Diffusion, VeO3).
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 base_channels: int = 64, channel_mults: tuple = (1, 2, 4)):
        super().__init__()

        # Encoder
        enc_layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for mult in channel_mults:
            ch_out = base_channels * mult
            enc_layers.extend([
                ResConvBlock(ch, ch_out),
                ResConvBlock(ch_out, ch_out),
                nn.Conv2d(ch_out, ch_out, 3, stride=2, padding=1),
            ])
            ch = ch_out
        enc_layers.extend([
            ResConvBlock(ch, ch),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_channels * 2, 1),  # mu + log_var
        ])
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = [nn.Conv2d(latent_channels, ch, 1)]
        for mult in reversed(channel_mults):
            ch_out = base_channels * mult
            dec_layers.extend([
                ResConvBlock(ch, ch_out),
                ResConvBlock(ch_out, ch_out),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_out, ch_out, 3, padding=1),
            ])
            ch = ch_out
        dec_layers.extend([
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return {"reconstruction": recon, "mu": mu, "log_var": log_var, "kl_loss": kl_loss, "latent": z}


class NoiseScheduler:
    """DDPM / DDIM noise scheduler for diffusion training.

    Manages the forward (noise addition) and reverse (denoising) process.
    """
    def __init__(self, n_steps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, schedule: str = "linear"):
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == "cosine":
            steps = torch.linspace(0, n_steps, n_steps + 1)
            alpha_bar = torch.cos(((steps / n_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            self.betas = torch.linspace(beta_start, beta_end, n_steps)

        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.n_steps = n_steps

    def add_noise(self, x, noise, timesteps):
        """Forward process: add noise to clean data."""
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].view(-1, *([1] * (x.dim() - 1))).to(x.device)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, *([1] * (x.dim() - 1))).to(x.device)
        return sqrt_alpha * x + sqrt_one_minus * noise

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.n_steps, (batch_size,), device=device)


# ---------------------------------------------------------------------------
# Video Generation Components
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Attention across the temporal dimension for video models.

    Input: (B, T, C, H, W) → attention over T → Output: (B, T, C, H, W)
    Used in video diffusion models like VeO3, AnimateDiff.
    """
    def __init__(self, channels: int, n_heads: int = 8, n_frames: int = 16):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.pos_emb = nn.Parameter(torch.randn(1, n_frames, channels) * 0.02)
        self.n_frames = n_frames

    def forward(self, x):
        # x: (B*T, C, H, W) or (B, T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            reshape_5d = True
        else:
            BT, C, H, W = x.shape
            T = min(self.n_frames, BT)
            B = BT // T
            reshape_5d = False

        # Reshape to (B*H*W, T, C) for temporal attention
        if reshape_5d:
            h = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        else:
            h = x.reshape(B, T, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        h = h + self.pos_emb[:, :T]
        out, _ = self.attn(h, h, h)

        if reshape_5d:
            return out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        else:
            return out.reshape(B, H, W, T, C).permute(0, 3, 4, 1, 2).reshape(BT, C, H, W)


class TemporalConv3d(nn.Module):
    """Temporal convolution for video — convolves along time dimension only.

    Input: (B, C, T, H, W) → Output: (B, C, T, H, W)
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                              padding=(kernel_size // 2, 0, 0))
        self.norm = nn.GroupNorm(32, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x))) + x


class VideoVAE(nn.Module):
    """3D VAE for video — encodes video frames to spatial-temporal latent space.

    Input: (B, C, T, H, W) → Output latent: (B, latent_ch, T', H', W')
    Used in VeO3, Sora-style video generation.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4,
                 base_channels: int = 64, temporal_downsample: bool = True):
        super().__init__()
        t_stride = 2 if temporal_downsample else 1

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, (t_stride, 2, 2),
                      stride=(t_stride, 2, 2), padding=(0, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, (1, 2, 2),
                      stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, latent_channels * 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * 4, 1),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, (1, 2, 2),
                               stride=(1, 2, 2)),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels, (t_stride, 2, 2),
                               stride=(t_stride, 2, 2)),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, 3, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return {"reconstruction": recon, "latent": z, "kl_loss": kl_loss}


# ---------------------------------------------------------------------------
# Perceiver Resampler (for flexible multimodal fusion)
# ---------------------------------------------------------------------------

class PerceiverResampler(nn.Module):
    """Perceiver-style resampler for multimodal models.

    Maps variable-length input (image patches, audio frames, video tokens)
    to a fixed number of latent tokens via cross-attention.
    Used in Flamingo, Gemini, and other multimodal models.

    Input: (B, N_input, d_input) → Output: (B, n_latents, d_model)
    """
    def __init__(self, d_input: int, d_model: int = 512, n_latents: int = 64,
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.input_proj = nn.Linear(d_input, d_model) if d_input != d_model else nn.Identity()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "cross_norm": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "ffn_norm": nn.LayerNorm(d_model),
            }))

    def forward(self, x):
        x = self.input_proj(x)
        latents = self.latents.expand(x.shape[0], -1, -1)

        for layer in self.layers:
            # Cross-attention: latents attend to input
            normed = layer["cross_norm"](latents)
            attn_out, _ = layer["cross_attn"](normed, x, x)
            latents = latents + attn_out
            # FFN
            latents = latents + layer["ffn"](layer["ffn_norm"](latents))

        return latents


# ---------------------------------------------------------------------------
# Knowledge Distillation Module
# ---------------------------------------------------------------------------

class DistillationWrapper(nn.Module):
    """Wraps a student model for knowledge distillation from a teacher.

    Combines hard label loss + soft KL-divergence loss from teacher logits.
    """
    def __init__(self, student: nn.Module, teacher: nn.Module = None,
                 temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha  # weight for distillation loss vs hard loss
        if teacher is not None:
            for p in teacher.parameters():
                p.requires_grad = False

    def forward(self, input_ids, labels=None):
        student_out = self.student(input_ids, labels=labels)

        if self.teacher is not None and self.training and labels is not None:
            with torch.no_grad():
                teacher_out = self.teacher(input_ids)

            # Soft distillation loss
            T = self.temperature
            student_logits = student_out["logits"] / T
            teacher_logits = teacher_out["logits"] / T

            soft_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            ) * (T * T)

            hard_loss = student_out.get("loss", torch.tensor(0.0))
            if hard_loss is not None:
                student_out["loss"] = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            else:
                student_out["loss"] = soft_loss
            student_out["distill_loss"] = soft_loss

        return student_out


# Register all custom layers
from state_graph.core.registry import Registry

# Transformer / Attention
Registry.register_layer("TransformerBlock", TransformerBlock, category="Transformer")
Registry.register_layer("PositionalEncoding", PositionalEncoding, category="Transformer")
Registry.register_layer("TokenEmbedding", TokenEmbedding, category="Transformer")
Registry.register_layer("SequencePool", SequencePool, category="Transformer")

# Building Blocks
Registry.register_layer("ResidualBlock", ResidualBlock, category="Building Blocks")
Registry.register_layer("SqueezeExcite", SqueezeExcite, category="Building Blocks")
Registry.register_layer("GatedLinearUnit", GatedLinearUnit, category="Building Blocks")
Registry.register_layer("SwishLinear", SwishLinear, category="Building Blocks")
Registry.register_layer("Reshape", Reshape, category="Building Blocks")
Registry.register_layer("GlobalAvgPool", GlobalAvgPool, category="Building Blocks")

# Vision
Registry.register_layer("PatchEmbed", PatchEmbed, category="Vision")
Registry.register_layer("DepthwiseSeparableConv", DepthwiseSeparableConv, category="Vision")
Registry.register_layer("ChannelAttention", ChannelAttention, category="Vision")
Registry.register_layer("UpsampleBlock", UpsampleBlock, category="Vision")
Registry.register_layer("ResConvBlock", ResConvBlock, category="Vision")
Registry.register_layer("DownBlock", DownBlock, category="Vision")
Registry.register_layer("UpBlock", UpBlock, category="Vision")

# Audio
Registry.register_layer("MelSpectrogram", MelSpectrogram, category="Audio")
Registry.register_layer("AudioConvBlock", AudioConvBlock, category="Audio")
Registry.register_layer("Transpose", Transpose, category="Audio")

# Video
Registry.register_layer("Conv3dBlock", Conv3dBlock, category="Video")
Registry.register_layer("TemporalPool", TemporalPool, category="Video")

# Diffusion / Generative
Registry.register_layer("SinusoidalTimestepEmbed", SinusoidalTimestepEmbed, category="Diffusion")
Registry.register_layer("ConditionalBatchNorm2d", ConditionalBatchNorm2d, category="Diffusion")

# State-Space Models (Mamba / S4)
Registry.register_layer("SelectiveScan", SelectiveScan, category="SSM / Alternative")
Registry.register_layer("MambaBlock", MambaBlock, category="SSM / Alternative")

# RWKV
Registry.register_layer("RWKVBlock", RWKVBlock, category="SSM / Alternative")

# RetNet
Registry.register_layer("RetentionLayer", RetentionLayer, category="SSM / Alternative")
Registry.register_layer("RetNetBlock", RetNetBlock, category="SSM / Alternative")

# Hyena
Registry.register_layer("HyenaOperator", HyenaOperator, category="SSM / Alternative")
Registry.register_layer("HyenaBlock", HyenaBlock, category="SSM / Alternative")

# xLSTM
Registry.register_layer("XLSTM", XLSTM, category="SSM / Alternative")

# Griffin / Hawk
Registry.register_layer("GatedLinearRecurrence", GatedLinearRecurrence, category="SSM / Alternative")

# CNN Backbones
Registry.register_layer("ResNetBlock", ResNetBlock, category="Vision")
Registry.register_layer("ConvNeXtBlock", ConvNeXtBlock, category="Vision")
Registry.register_layer("MBConvBlock", MBConvBlock, category="Vision")
Registry.register_layer("VisionEncoder", VisionEncoder, category="Vision")

# Diffusion (Full)
Registry.register_layer("DiffusionTimestepBlock", DiffusionTimestepBlock, category="Diffusion")
Registry.register_layer("SpatialAttentionBlock", SpatialAttentionBlock, category="Diffusion")
Registry.register_layer("CrossAttentionBlock", CrossAttentionBlock, category="Diffusion")
Registry.register_layer("DiffusionUNet", DiffusionUNet, category="Diffusion")
Registry.register_layer("VAE", VAE, category="Diffusion")
Registry.register_layer("NoiseScheduler", NoiseScheduler, category="Diffusion")

# Video Generation
Registry.register_layer("TemporalAttention", TemporalAttention, category="Video")
Registry.register_layer("TemporalConv3d", TemporalConv3d, category="Video")
Registry.register_layer("VideoVAE", VideoVAE, category="Video")

# Multimodal
Registry.register_layer("PerceiverResampler", PerceiverResampler, category="Multimodal")

# Knowledge Distillation
Registry.register_layer("DistillationWrapper", DistillationWrapper, category="Training")
