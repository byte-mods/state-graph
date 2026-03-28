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


# Register all custom layers
from state_graph.core.registry import Registry

# Original custom layers
Registry.register_layer("ResidualBlock", ResidualBlock)
Registry.register_layer("SqueezeExcite", SqueezeExcite)
Registry.register_layer("GatedLinearUnit", GatedLinearUnit)
Registry.register_layer("SwishLinear", SwishLinear)
Registry.register_layer("TransformerBlock", TransformerBlock)
Registry.register_layer("PositionalEncoding", PositionalEncoding)
Registry.register_layer("TokenEmbedding", TokenEmbedding)
Registry.register_layer("SequencePool", SequencePool)

# Vision
Registry.register_layer("PatchEmbed", PatchEmbed)
Registry.register_layer("DepthwiseSeparableConv", DepthwiseSeparableConv)
Registry.register_layer("ChannelAttention", ChannelAttention)
Registry.register_layer("UpsampleBlock", UpsampleBlock)
Registry.register_layer("GlobalAvgPool", GlobalAvgPool)
Registry.register_layer("Reshape", Reshape)
Registry.register_layer("ResConvBlock", ResConvBlock)
Registry.register_layer("DownBlock", DownBlock)
Registry.register_layer("UpBlock", UpBlock)

# Audio
Registry.register_layer("MelSpectrogram", MelSpectrogram)
Registry.register_layer("AudioConvBlock", AudioConvBlock)
Registry.register_layer("Transpose", Transpose)

# Video
Registry.register_layer("Conv3dBlock", Conv3dBlock)
Registry.register_layer("TemporalPool", TemporalPool)

# Diffusion / Generative
Registry.register_layer("SinusoidalTimestepEmbed", SinusoidalTimestepEmbed)
Registry.register_layer("ConditionalBatchNorm2d", ConditionalBatchNorm2d)

# State-Space Models (Mamba / S4)
Registry.register_layer("SelectiveScan", SelectiveScan)
Registry.register_layer("MambaBlock", MambaBlock)

# RWKV
Registry.register_layer("RWKVBlock", RWKVBlock)

# RetNet
Registry.register_layer("RetentionLayer", RetentionLayer)
Registry.register_layer("RetNetBlock", RetNetBlock)

# Hyena
Registry.register_layer("HyenaOperator", HyenaOperator)
Registry.register_layer("HyenaBlock", HyenaBlock)

# xLSTM
Registry.register_layer("XLSTM", XLSTM)

# Griffin / Hawk
Registry.register_layer("GatedLinearRecurrence", GatedLinearRecurrence)
