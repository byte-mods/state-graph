"""LLM-specific layers: RMSNorm, RoPE, Flash/Grouped-Query Attention, SwiGLU, MoE, KV-Cache.

These layers implement modern LLM architecture components that can be composed
via the graph builder or the dedicated LLM builder to create GPT/Llama-style models.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm  (Llama, Gemma, etc.)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Formula: x_norm = x / sqrt(mean(x^2) + eps) * gamma
    Used in Llama, Gemma, and modern LLMs instead of LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE) — Llama, GPT-NeoX, Mistral
# ---------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Formula:
      theta_i = base^(-2i/d) for i in [0, d/2)
      R(pos) = [[cos(pos*theta), -sin(pos*theta)],
                 [sin(pos*theta),  cos(pos*theta)]]
      Apply rotation to pairs of dimensions in Q and K.
    """

    def __init__(self, d_model: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1] if x.dim() >= 2 else self.max_len
        if seq_len > self.cos_cached.shape[1]:
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Multi-Head Attention with Flash Attention support + GQA
# ---------------------------------------------------------------------------
class LLMAttention(nn.Module):
    """Multi-Head Attention with RoPE, optional Flash Attention, and Grouped Query Attention.

    Supports:
    - Standard Multi-Head Attention (n_kv_heads == n_heads)
    - Grouped-Query Attention (n_kv_heads < n_heads)  [Llama 2 70B, Mistral]
    - Multi-Query Attention (n_kv_heads == 1)  [PaLM, Falcon]
    - Flash Attention 2 (via PyTorch scaled_dot_product_attention)
    - KV-Cache for autoregressive generation

    Formula:
      Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
      Attn(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
      Output = Concat(head_1, ..., head_h) @ W_o
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
        max_len: int = 4096,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_groups = n_heads // self.n_kv_heads
        self.use_flash = use_flash
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_len=max_len, base=rope_base)

        # KV cache for generation
        self._kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def clear_kv_cache(self) -> None:
        self._kv_cache = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(x, seq_len=L)
        cos = cos[:, :L, :self.head_dim]
        sin = sin[:, :L, :self.head_dim]
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        # KV-Cache for autoregressive generation
        if use_cache:
            if self._kv_cache is not None:
                prev_k, prev_v = self._kv_cache
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            self._kv_cache = (k, v)

        # Grouped-Query Attention: repeat KV heads
        if self.n_kv_groups > 1:
            k = k.repeat_interleave(self.n_kv_groups, dim=1)
            v = v.repeat_interleave(self.n_kv_groups, dim=1)

        # Flash Attention via PyTorch's SDPA
        if self.use_flash:
            use_causal = (mask is None and L > 1 and not use_cache)
            attn_mask = None
            if mask is not None:
                attn_mask = mask
            elif L > 1 and not use_causal:
                # Only create explicit mask when not using is_causal flag
                attn_mask = torch.triu(
                    torch.full((L, L), float("-inf"), device=x.device), diagonal=1
                )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=use_causal,
            )
        else:
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if mask is not None:
                scores = scores + mask
            elif L > 1:
                causal = torch.triu(
                    torch.full((L, L), float("-inf"), device=x.device), diagonal=1
                )
                scores = scores + causal
            attn_weights = F.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network  (Llama, PaLM)
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Formula:
      gate = Swish(x @ W_gate) = (x @ W_gate) * sigmoid(x @ W_gate)
      up   = x @ W_up
      FFN(x) = (gate * up) @ W_down

    Used in Llama, PaLM. More expressive than standard GELU FFN.
    Hidden dim typically = (4 * d_model * 2/3) rounded to multiple of 256.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * d_model * 2 / 3)
            # Round to multiple of 64 for hardware efficiency
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# GeGLU Feed-Forward Network  (alternative to SwiGLU)
# ---------------------------------------------------------------------------
class GeGLUFFN(nn.Module):
    """GeGLU Feed-Forward Network.

    Formula: FFN(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down
    Used in some T5 variants, Gemma.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * d_model * 2 / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# ReGLU Feed-Forward Network  (ReLU-gated, used in some T5 variants)
# ---------------------------------------------------------------------------
class ReGLUFFN(nn.Module):
    """ReGLU Feed-Forward Network.

    Formula: FFN(x) = (ReLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(4 * d_model * 2 / 3)
            hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.relu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Standard MLP Feed-Forward Network  (GPT-2 style)
# ---------------------------------------------------------------------------
class StandardFFN(nn.Module):
    """Standard Feed-Forward Network (GPT-2/BERT style).

    Formula: FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# FFN type registry for easy lookup
FFN_TYPES = {
    "swiglu": SwiGLUFFN,
    "geglu": GeGLUFFN,
    "reglu": ReGLUFFN,
    "standard": StandardFFN,
}


# ---------------------------------------------------------------------------
# Mixture of Experts (MoE) — Mixtral, Switch Transformer, DeepSeek
# ---------------------------------------------------------------------------
class MoERouter(nn.Module):
    """Top-K router for Mixture of Experts.

    Formula:
      logits = x @ W_gate
      top_k_indices, top_k_weights = TopK(softmax(logits), k)
      Optionally adds load-balancing noise during training.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2,
                 noise_std: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.n_experts = n_experts
        self.top_k = top_k
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)  # (B, L, n_experts)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return top_k_probs, top_k_indices, probs


class MoELayer(nn.Module):
    """Mixture of Experts layer with Top-K routing.

    Architecture:
      1. Router selects top-K experts per token
      2. Each expert is a SwiGLU FFN
      3. Output = sum(weight_i * expert_i(x)) for selected experts
      4. Load-balancing loss encourages uniform expert usage

    Formula:
      router_probs = softmax(x @ W_gate)
      top_k = TopK(router_probs, k)
      output = sum_{i in top_k} weight_i * Expert_i(x)
      L_balance = n_experts * sum(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = mean probability assigned to expert i
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = MoERouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, hidden_dim=expert_hidden_dim, dropout=dropout)
            for _ in range(n_experts)
        ])
        self.aux_loss: float = 0.0  # Load-balancing loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # (B*L, D)

        top_k_probs, top_k_indices, all_probs = self.router(x_flat.unsqueeze(1))
        top_k_probs = top_k_probs.squeeze(1)  # (B*L, top_k)
        top_k_indices = top_k_indices.squeeze(1)
        all_probs = all_probs.squeeze(1)

        # Compute load-balancing loss
        if self.training:
            # f_i: fraction of tokens routed to each expert
            one_hot = F.one_hot(top_k_indices[:, 0], self.n_experts).float()
            f = one_hot.mean(dim=0)
            # P_i: mean router probability for each expert
            P = all_probs.mean(dim=0)
            self.aux_loss = self.n_experts * (f * P).sum()

        # Process tokens through selected experts
        output = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            expert_indices = top_k_indices[:, k_idx]
            weights = top_k_probs[:, k_idx]
            for e_idx in range(self.n_experts):
                mask = expert_indices == e_idx
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        return output.view(B, L, D)


# ---------------------------------------------------------------------------
# LLM Decoder Block  (full Llama-style block)
# ---------------------------------------------------------------------------
class LLMDecoderBlock(nn.Module):
    """Full LLM Decoder Block (Llama/Mistral-style).

    Architecture:
      x -> Norm -> Attention(RoPE + GQA + Flash) -> + residual
      -> Norm -> FFN (SwiGLU/GeGLU/ReGLU/Standard or MoE) -> + residual

    Components: RMSNorm/LayerNorm, RoPE, Multi-Head/GQA Attention, FFN/MoE
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash: bool = True,
        use_moe: bool = False,
        n_experts: int = 8,
        moe_top_k: int = 2,
        max_len: int = 4096,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.ffn_type = ffn_type

        # Normalization layers
        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)

        self.attn = LLMAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout, use_flash=use_flash, max_len=max_len,
            rope_base=rope_base,
        )

        if use_moe:
            self.ffn = MoELayer(
                d_model, n_experts=n_experts, top_k=moe_top_k,
                expert_hidden_dim=ffn_hidden_dim, dropout=dropout,
            )
        else:
            FFNClass = FFN_TYPES.get(ffn_type, SwiGLUFFN)
            self.ffn = FFNClass(d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Complete LLM Model
# ---------------------------------------------------------------------------
class LLMModel(nn.Module):
    """Complete decoder-only LLM (GPT/Llama-style).

    Architecture:
      Token Embedding -> [LLMDecoderBlock x N] -> RMSNorm -> LM Head

    Supports: RoPE, Flash Attention, GQA, MoE, SwiGLU, KV-Cache, Weight Tying.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        max_len: int = 2048,
        dropout: float = 0.0,
        use_flash: bool = True,
        use_moe: bool = False,
        moe_layers: Optional[list[int]] = None,
        n_experts: int = 8,
        moe_top_k: int = 2,
        tie_weights: bool = True,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        rope_base: float = 10000.0,
        bias: bool = False,
        # Per-layer overrides: list of dicts with layer-specific config
        layer_configs: Optional[list[dict]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.norm_type = norm_type
        self.ffn_type = ffn_type

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Determine which layers use MoE (e.g., every other layer in Mixtral)
        if moe_layers is None and use_moe:
            moe_layers = list(range(1, n_layers, 2))  # every other layer
        moe_set = set(moe_layers or [])

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Per-layer overrides
            lc = (layer_configs[i] if layer_configs and i < len(layer_configs) else {}) or {}
            self.layers.append(LLMDecoderBlock(
                d_model=d_model,
                n_heads=lc.get("n_heads", n_heads),
                n_kv_heads=lc.get("n_kv_heads", n_kv_heads),
                ffn_hidden_dim=lc.get("ffn_hidden_dim", ffn_hidden_dim),
                dropout=lc.get("dropout", dropout),
                use_flash=lc.get("use_flash", use_flash),
                use_moe=lc.get("use_moe", i in moe_set),
                n_experts=lc.get("n_experts", n_experts),
                moe_top_k=lc.get("moe_top_k", moe_top_k),
                max_len=max_len,
                norm_type=lc.get("norm_type", norm_type),
                ffn_type=lc.get("ffn_type", ffn_type),
                rope_base=lc.get("rope_base", rope_base),
                bias=lc.get("bias", bias),
            ))

        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm = NormClass(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def clear_kv_cache(self) -> None:
        for layer in self.layers:
            layer.attn.clear_kv_cache()

    def get_moe_aux_loss(self) -> torch.Tensor:
        total = 0.0
        for layer in self.layers:
            if isinstance(layer.ffn, MoELayer):
                total += layer.ffn.aux_loss
        return total

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            moe_loss = self.get_moe_aux_loss()
            if moe_loss > 0:
                loss = loss + 0.01 * moe_loss

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.clear_kv_cache()
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.max_len:])["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        self.clear_kv_cache()
        return input_ids

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.1f}M",
            "embedding": self.tok_emb.weight.numel(),
            "per_layer": sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0,
        }


# ---------------------------------------------------------------------------
# Custom Component System — create novel layers from code or formulas
# ---------------------------------------------------------------------------

class CustomComponent(nn.Module):
    """A component created from user-provided Python code.

    The code must define a class `CustomModule(nn.Module)` with
    __init__(self, d_model, **kwargs) and forward(self, x).
    """

    @staticmethod
    def create_from_code(code: str, d_model: int, **kwargs) -> nn.Module:
        safe_globals = {
            "torch": torch,
            "nn": nn,
            "F": F,
            "math": math,
            "Optional": Optional,
        }
        local_ns: dict = {}
        exec(code, safe_globals, local_ns)  # noqa: S102

        # Find the class defined in the code
        module_cls = None
        for obj in local_ns.values():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                module_cls = obj
                break

        if module_cls is None:
            raise ValueError("Code must define an nn.Module subclass")

        return module_cls(d_model, **kwargs)


class CustomFFN(nn.Module):
    """FFN created from a formula string.

    Formula uses variables: x, d_model, hidden_dim
    The formula defines the forward computation.
    For complex FFNs, use CustomComponent.create_from_code instead.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None,
                 formula: str = "F.gelu(self.fc1(x)) @ self.fc2.weight.t()"):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.formula = formula
        # Default layers available for formula use
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gate = nn.Linear(d_model, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        safe_globals = {
            "torch": torch, "F": F, "math": math,
            "self": self, "x": x,
        }
        return eval(self.formula, safe_globals)  # noqa: S307


# ---------------------------------------------------------------------------
# Alternative Attention Mechanisms
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention (Mistral/Longformer-style).

    Only attends to the last `window_size` tokens instead of full sequence.
    """

    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Create sliding window mask
        mask = torch.full((L, L), float("-inf"), device=x.device)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0.0

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class LinearAttention(nn.Module):
    """Linear Attention (O(n) complexity).

    Formula: Attn(Q,K,V) = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))
    where phi is ELU + 1.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self._feature_map(self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2))
        k = self._feature_map(self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2))
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Linear attention: O(n) via kernel trick
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)  # (B, H, D, D)
        qkv = torch.einsum("bhnd,bhdm->bhnm", q, kv)  # (B, H, N, D)
        normalizer = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1).clamp(min=1e-6)
        out = qkv / normalizer

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class ALiBiAttention(nn.Module):
    """Attention with Linear Biases (ALiBi) — no positional embeddings needed.

    Adds a linear bias based on distance: bias = -m * |i - j|
    where m varies per head geometrically.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # Compute ALiBi slopes: geometric sequence
        ratio = 2.0 ** (-8.0 / n_heads)
        slopes = torch.tensor([ratio ** (i + 1) for i in range(n_heads)])
        self.register_buffer("slopes", slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # ALiBi bias
        positions = torch.arange(L, device=x.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        alibi = -distance.unsqueeze(0) * self.slopes.view(-1, 1, 1)
        scores = scores + alibi.unsqueeze(0)

        # Causal mask
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        scores = scores + causal

        attn = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Alternative Positional Encodings
# ---------------------------------------------------------------------------

class AbsolutePositionalEncoding(nn.Module):
    """Learned absolute positional encoding (GPT-2 style)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        return x + self.pos_emb(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Transformer/Vaswani-style)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


class NoPE(nn.Module):
    """No positional encoding (for ALiBi or other position-free approaches)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# Attention type registry
ATTENTION_TYPES = {
    "standard": LLMAttention,
    "sliding_window": SlidingWindowAttention,
    "linear": LinearAttention,
    "alibi": ALiBiAttention,
}

# Positional encoding registry
POS_ENCODING_TYPES = {
    "rope": None,  # Handled inside LLMAttention
    "absolute": AbsolutePositionalEncoding,
    "sinusoidal": SinusoidalPositionalEncoding,
    "none": NoPE,
    "alibi": NoPE,  # ALiBi builds position into attention
}


# ---------------------------------------------------------------------------
# Parallel Branch (for PaLM-style parallel attn+ffn)
# ---------------------------------------------------------------------------

class ParallelBranch(nn.Module):
    """Runs two sub-modules in parallel and merges their outputs.

    merge_mode: "add" (sum outputs), "concat" (concatenate + project), "gate" (learned gate)
    """

    def __init__(self, branch_a: nn.Module, branch_b: nn.Module,
                 d_model: int, merge_mode: str = "add"):
        super().__init__()
        self.branch_a = branch_a
        self.branch_b = branch_b
        self.merge_mode = merge_mode
        if merge_mode == "concat":
            self.merge_proj = nn.Linear(d_model * 2, d_model, bias=False)
        elif merge_mode == "gate":
            self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        if self.merge_mode == "add":
            return out_a + out_b
        elif self.merge_mode == "concat":
            return self.merge_proj(torch.cat([out_a, out_b], dim=-1))
        elif self.merge_mode == "gate":
            g = torch.sigmoid(self.gate_proj(x))
            return g * out_a + (1 - g) * out_b
        return out_a + out_b


class ComposableBlock(nn.Module):
    """A fully composable decoder block built from a list of sub-components.

    Each step is a dict: {
        "type": "norm" | "attention" | "ffn" | "residual" | "custom" | "parallel" | ...,
        "config": {...},  # component-specific config
        "residual_from": int | None,  # step index to add residual from
    }

    This allows arbitrary wiring:
    - Sequential: norm → attn → norm → ffn (standard)
    - Parallel: norm → [attn, ffn] → merge (PaLM-style)
    - Custom: norm → attn → custom_layer → ffn → norm
    """

    def __init__(
        self,
        d_model: int,
        steps: list[dict],
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        max_len: int = 4096,
        dropout: float = 0.0,
        use_flash: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.steps_config = steps
        self.modules_list = nn.ModuleList()
        self.step_types = []
        self.residual_sources = []

        for step in steps:
            stype = step.get("type", "")
            cfg = step.get("config", {})
            self.step_types.append(stype)
            self.residual_sources.append(step.get("residual_from"))

            if stype == "norm":
                norm_kind = cfg.get("norm_type", "rmsnorm")
                if norm_kind == "layernorm":
                    self.modules_list.append(nn.LayerNorm(d_model))
                else:
                    self.modules_list.append(RMSNorm(d_model))

            elif stype == "attention":
                self.modules_list.append(LLMAttention(
                    d_model,
                    n_heads=cfg.get("n_heads", n_heads),
                    n_kv_heads=cfg.get("n_kv_heads", n_kv_heads),
                    dropout=cfg.get("dropout", dropout),
                    use_flash=cfg.get("use_flash", use_flash),
                    max_len=max_len,
                    rope_base=cfg.get("rope_base", 10000.0),
                ))

            elif stype == "ffn":
                ffn_kind = cfg.get("ffn_type", "swiglu")
                hidden = cfg.get("hidden_dim")
                FFNClass = FFN_TYPES.get(ffn_kind, SwiGLUFFN)
                self.modules_list.append(FFNClass(d_model, hidden_dim=hidden, dropout=cfg.get("dropout", dropout)))

            elif stype == "moe":
                self.modules_list.append(MoELayer(
                    d_model,
                    n_experts=cfg.get("n_experts", 8),
                    top_k=cfg.get("top_k", 2),
                    expert_hidden_dim=cfg.get("hidden_dim"),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "residual":
                # Placeholder — residual add uses stored activations
                self.modules_list.append(nn.Identity())

            elif stype == "dropout":
                self.modules_list.append(nn.Dropout(cfg.get("p", dropout)))

            elif stype == "linear":
                self.modules_list.append(nn.Linear(
                    d_model, cfg.get("out_features", d_model),
                    bias=cfg.get("bias", False),
                ))

            elif stype == "activation":
                act_name = cfg.get("name", "silu")
                act_map = {
                    "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU(),
                    "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "mish": nn.Mish(),
                }
                self.modules_list.append(act_map.get(act_name, nn.SiLU()))

            elif stype == "custom_code":
                code = cfg.get("code", "")
                module = CustomComponent.create_from_code(code, d_model, **cfg.get("kwargs", {}))
                self.modules_list.append(module)

            elif stype == "custom_formula":
                formula = cfg.get("formula", "x")
                self.modules_list.append(CustomFFN(d_model, formula=formula))

            elif stype == "sliding_window_attention":
                self.modules_list.append(SlidingWindowAttention(
                    d_model,
                    n_heads=cfg.get("n_heads", n_heads),
                    window_size=cfg.get("window_size", 256),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "linear_attention":
                self.modules_list.append(LinearAttention(
                    d_model,
                    n_heads=cfg.get("n_heads", n_heads),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "alibi_attention":
                self.modules_list.append(ALiBiAttention(
                    d_model,
                    n_heads=cfg.get("n_heads", n_heads),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "pos_encoding":
                enc_type = cfg.get("encoding_type", "absolute")
                PEClass = POS_ENCODING_TYPES.get(enc_type, AbsolutePositionalEncoding)
                if PEClass is NoPE:
                    self.modules_list.append(NoPE())
                else:
                    self.modules_list.append(PEClass(d_model, max_len=max_len))

            elif stype == "parallel":
                # Parallel branches: two sub-step configs run in parallel
                branch_a_cfg = cfg.get("branch_a", {"type": "attention", "config": {}})
                branch_b_cfg = cfg.get("branch_b", {"type": "ffn", "config": {"ffn_type": "swiglu"}})
                merge = cfg.get("merge", "add")

                # Build branch A
                a_block = ComposableBlock(d_model, [branch_a_cfg], n_heads, n_kv_heads, max_len, dropout, use_flash)
                # Build branch B
                b_block = ComposableBlock(d_model, [branch_b_cfg], n_heads, n_kv_heads, max_len, dropout, use_flash)

                self.modules_list.append(ParallelBranch(a_block, b_block, d_model, merge))

            elif stype == "cross_attention":
                # Cross-attention uses the same projections but attends to a stored context
                self.modules_list.append(nn.MultiheadAttention(
                    d_model,
                    num_heads=cfg.get("n_heads", n_heads),
                    dropout=cfg.get("dropout", dropout),
                    batch_first=True,
                ))

            elif stype == "conv1d":
                # 1D convolution over sequence (used in Mamba, Hyena, etc.)
                kernel = cfg.get("kernel_size", 3)
                self.modules_list.append(nn.Conv1d(
                    d_model, d_model,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=cfg.get("groups", 1),
                ))

            elif stype == "embedding":
                # Learned or sinusoidal positional encoding (standalone step)
                enc_type = cfg.get("type", "absolute")
                if enc_type == "sinusoidal":
                    self.modules_list.append(SinusoidalPositionalEncoding(d_model, max_len))
                else:
                    self.modules_list.append(AbsolutePositionalEncoding(d_model, max_len))

            elif stype == "mamba":
                from state_graph.layers.custom import SelectiveScan
                self.modules_list.append(SelectiveScan(
                    d_model,
                    d_state=cfg.get("d_state", 16),
                    expand=cfg.get("expand", 2),
                    conv_kernel=cfg.get("conv_kernel", 4),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "rwkv":
                from state_graph.layers.custom import RWKVBlock
                self.modules_list.append(RWKVBlock(d_model, dropout=cfg.get("dropout", dropout)))

            elif stype == "retention":
                from state_graph.layers.custom import RetentionLayer
                self.modules_list.append(RetentionLayer(
                    d_model,
                    n_heads=cfg.get("n_heads", n_heads),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "hyena":
                from state_graph.layers.custom import HyenaOperator
                self.modules_list.append(HyenaOperator(
                    d_model, max_len=max_len,
                    order=cfg.get("order", 2),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "xlstm":
                from state_graph.layers.custom import XLSTM
                self.modules_list.append(XLSTM(
                    d_model,
                    n_layers=cfg.get("n_layers", 1),
                    dropout=cfg.get("dropout", dropout),
                ))

            elif stype == "gated_recurrence":
                from state_graph.layers.custom import GatedLinearRecurrence
                self.modules_list.append(GatedLinearRecurrence(
                    d_model,
                    expand=cfg.get("expand", 2),
                    dropout=cfg.get("dropout", dropout),
                ))

            else:
                raise ValueError(f"Unknown step type: {stype}")

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        activations = {}
        current = x

        for i, (stype, module) in enumerate(zip(self.step_types, self.modules_list)):
            activations[i] = current

            if stype == "residual":
                src = self.residual_sources[i]
                if src is not None and src in activations:
                    current = current + activations[src]
                else:
                    current = current + x  # default: residual from input
            elif stype == "conv1d":
                # Conv1d expects (B, C, L), but we have (B, L, C)
                current = module(current.transpose(1, 2)).transpose(1, 2)
            elif stype == "cross_attention":
                # nn.MultiheadAttention: (query, key, value)
                ctx = context if context is not None else current
                current, _ = module(current, ctx, ctx)
            else:
                current = module(current)

            # Auto-add residual if specified (skip for "residual" type, already handled)
            if stype != "residual":
                src = self.residual_sources[i]
                if src is not None and src in activations:
                    current = current + activations[src]

        return current

    def get_step_info(self) -> list[dict]:
        """Return info about each step for visualization."""
        info = []
        for i, (stype, module) in enumerate(zip(self.step_types, self.modules_list)):
            step = {
                "index": i,
                "type": stype,
                "residual_from": self.residual_sources[i],
                "params": sum(p.numel() for p in module.parameters()),
            }
            info.append(step)
        return info


class ComposableLLM(nn.Module):
    """LLM built from ComposableBlocks — fully customizable architecture.

    Each layer can have a different internal structure defined by its steps.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 6,
        max_len: int = 2048,
        dropout: float = 0.0,
        tie_weights: bool = True,
        norm_type: str = "rmsnorm",
        # Each entry defines a block's internal structure
        block_designs: Optional[list[list[dict]]] = None,
        # Default block design (used if block_designs not specified)
        default_block: Optional[list[dict]] = None,
        # Defaults for attention
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        use_flash: bool = True,
        # Positional encoding
        pos_encoding: str = "rope",  # "rope", "absolute", "sinusoidal", "none", "alibi"
        # Custom embedding code (if provided, replaces standard nn.Embedding)
        custom_embedding_code: Optional[str] = None,
        # Custom loss formula
        custom_loss: Optional[str] = None,
        # Extra output heads (multi-task)
        extra_heads: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.custom_loss_formula = custom_loss

        # Embedding
        if custom_embedding_code:
            self.tok_emb = CustomComponent.create_from_code(custom_embedding_code, d_model)
        else:
            self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Positional encoding (applied after embedding, before blocks)
        self.pos_encoding_type = pos_encoding
        PEClass = POS_ENCODING_TYPES.get(pos_encoding)
        if PEClass is not None and PEClass is not NoPE:
            self.pos_enc = PEClass(d_model, max_len)
        elif pos_encoding == "none" or pos_encoding == "alibi" or pos_encoding == "rope":
            self.pos_enc = NoPE()  # RoPE is inside attention, ALiBi is inside attention
        else:
            self.pos_enc = NoPE()

        if default_block is None:
            # Standard Llama-style block
            default_block = [
                {"type": "norm", "config": {"norm_type": norm_type}},
                {"type": "attention", "config": {"n_heads": n_heads, "n_kv_heads": n_kv_heads, "use_flash": use_flash}},
                {"type": "residual", "residual_from": -1},  # -1 = block input
                {"type": "norm", "config": {"norm_type": norm_type}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 2},  # after first residual
            ]

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            design = default_block
            if block_designs and i < len(block_designs) and block_designs[i]:
                design = block_designs[i]
            self.layers.append(ComposableBlock(
                d_model=d_model,
                steps=design,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_len=max_len,
                dropout=dropout,
                use_flash=use_flash,
            ))

        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm = NormClass(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights and isinstance(self.tok_emb, nn.Embedding):
            self.lm_head.weight = self.tok_emb.weight

        # Extra output heads (multi-task)
        self.extra_heads = nn.ModuleDict()
        if extra_heads:
            for head_name, head_size in extra_heads.items():
                self.extra_heads[head_name] = nn.Linear(d_model, head_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        B, L = input_ids.shape
        x = self.tok_emb(input_ids)
        x = self.pos_enc(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            if self.custom_loss_formula:
                # Custom loss via formula
                safe_globals = {
                    "torch": torch, "F": F, "math": math,
                    "logits": logits, "labels": labels,
                    "vocab_size": self.vocab_size,
                }
                loss = eval(self.custom_loss_formula, safe_globals)  # noqa: S307
            else:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        result = {"logits": logits, "loss": loss}

        # Extra heads
        for head_name, head in self.extra_heads.items():
            result[head_name] = head(x)

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.max_len:])["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.1f}M",
            "embedding": self.tok_emb.weight.numel(),
            "per_layer": sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0,
        }


# ---------------------------------------------------------------------------
# Feature 1: Encoder-Decoder Model (T5/BART-style)
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Non-causal encoder block (no causal masking in attention)."""

    def __init__(self, d_model: int, n_heads: int = 8, ffn_hidden_dim: Optional[int] = None,
                 dropout: float = 0.0, norm_type: str = "layernorm", ffn_type: str = "standard"):
        super().__init__()
        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = NormClass(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = NormClass(d_model)
        FFNClass = FFN_TYPES.get(ffn_type, StandardFFN)
        self.ffn = FFNClass(d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlockWithCrossAttn(nn.Module):
    """Decoder block with self-attention + cross-attention to encoder output."""

    def __init__(self, d_model: int, n_heads: int = 8, ffn_hidden_dim: Optional[int] = None,
                 dropout: float = 0.0, norm_type: str = "layernorm", ffn_type: str = "standard"):
        super().__init__()
        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm1 = NormClass(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = NormClass(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = NormClass(d_model)
        FFNClass = FFN_TYPES.get(ffn_type, StandardFFN)
        self.ffn = FFNClass(d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (causal)
        normed = self.norm1(x)
        L = x.shape[1]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal)
        x = x + sa_out
        # Cross-attention to encoder
        normed = self.norm2(x)
        ca_out, _ = self.cross_attn(normed, encoder_out, encoder_out)
        x = x + ca_out
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class EncoderDecoderLLM(nn.Module):
    """Full encoder-decoder model (T5/BART-style).

    Architecture: Encoder(input) -> Decoder(target, encoder_output) -> LM Head
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        ffn_hidden_dim: Optional[int] = None,
        max_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        ffn_type: str = "standard",
        tie_weights: bool = True,
        pos_encoding: str = "sinusoidal",
        share_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.encoder_emb = nn.Embedding(vocab_size, d_model)
        if share_embeddings:
            self.decoder_emb = self.encoder_emb
        else:
            self.decoder_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        elif pos_encoding == "absolute":
            self.pos_enc = AbsolutePositionalEncoding(d_model, max_len)
        else:
            self.pos_enc = NoPE()

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, ffn_hidden_dim, dropout, norm_type, ffn_type)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlockWithCrossAttn(d_model, n_heads, ffn_hidden_dim, dropout, norm_type, ffn_type)
            for _ in range(n_decoder_layers)
        ])

        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.encoder_norm = NormClass(d_model)
        self.decoder_norm = NormClass(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.encoder_emb.weight

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.drop(self.encoder_emb(src_ids)))
        for layer in self.encoder_layers:
            x = layer(x)
        return self.encoder_norm(x)

    def decode(self, tgt_ids: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.drop(self.decoder_emb(tgt_ids)))
        for layer in self.decoder_layers:
            x = layer(x, encoder_out)
        return self.decoder_norm(x)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        encoder_out = self.encode(input_ids)

        if decoder_input_ids is None:
            # For training: shift labels right to create decoder input
            if labels is not None:
                decoder_input_ids = labels[:, :-1]
            else:
                decoder_input_ids = input_ids

        decoded = self.decode(decoder_input_ids, encoder_out)
        logits = self.lm_head(decoded)

        loss = None
        if labels is not None:
            target = labels[:, 1:] if decoder_input_ids.shape[1] == labels.shape[1] - 1 else labels
            logits_for_loss = logits[:, :target.shape[1], :]
            loss = F.cross_entropy(
                logits_for_loss.contiguous().view(-1, self.vocab_size),
                target.contiguous().view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "encoder_output": encoder_out}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        encoder_out = self.encode(input_ids)
        # Start with BOS token (0)
        generated = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=input_ids.device)

        for _ in range(max_new_tokens):
            decoded = self.decode(generated, encoder_out)
            logits = self.lm_head(decoded[:, -1:, :]).squeeze(1)
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total, "trainable": trainable,
            "total_M": f"{total / 1e6:.1f}M",
            "embedding": self.encoder_emb.weight.numel(),
            "encoder_params": sum(p.numel() for p in self.encoder_layers.parameters()),
            "decoder_params": sum(p.numel() for p in self.decoder_layers.parameters()),
        }


# ---------------------------------------------------------------------------
# Feature 2: Custom Tokenizer Training
# ---------------------------------------------------------------------------

class TokenizerTrainer:
    """Train custom tokenizers from text using the `tokenizers` library."""

    @staticmethod
    def train(text: str, vocab_size: int = 8000, algorithm: str = "bpe",
              min_frequency: int = 2, special_tokens: Optional[list[str]] = None) -> dict:
        """Train a tokenizer and return it.

        Args:
            text: Training text
            vocab_size: Target vocabulary size
            algorithm: "bpe", "wordpiece", "unigram", or "char_bpe"
            min_frequency: Minimum token frequency
            special_tokens: List of special tokens (default: [PAD, UNK, BOS, EOS, MASK])

        Returns: dict with tokenizer info and encode/decode functions
        """
        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]

        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        except ImportError:
            # Fallback: simple character or whitespace tokenizer
            return TokenizerTrainer._train_simple(text, vocab_size, algorithm)

        if algorithm == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size, min_frequency=min_frequency,
                special_tokens=special_tokens,
            )
        elif algorithm == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size, min_frequency=min_frequency,
                special_tokens=special_tokens,
            )
        elif algorithm == "unigram":
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
            tokenizer.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size, special_tokens=special_tokens,
            )
        else:
            return TokenizerTrainer._train_simple(text, vocab_size, algorithm)

        # Train from text
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            tmp_path = f.name
        try:
            tokenizer.train([tmp_path], trainer)
        finally:
            os.unlink(tmp_path)

        return {
            "type": "custom_trained",
            "algorithm": algorithm,
            "vocab_size": tokenizer.get_vocab_size(),
            "tokenizer": tokenizer,
            "encode": lambda t: tokenizer.encode(t).ids,
            "decode": lambda ids: tokenizer.decode(ids),
        }

    @staticmethod
    def _train_simple(text: str, vocab_size: int, algorithm: str) -> dict:
        """Fallback: simple whitespace or character tokenizer."""
        if algorithm == "char" or algorithm == "char_bpe":
            chars = sorted(set(text))
            char2idx = {c: i + 5 for i, c in enumerate(chars)}  # Reserve 0-4 for special
            char2idx["[PAD]"] = 0
            char2idx["[UNK]"] = 1
            char2idx["[BOS]"] = 2
            char2idx["[EOS]"] = 3
            char2idx["[MASK]"] = 4
            idx2char = {i: c for c, i in char2idx.items()}
            return {
                "type": "custom_trained",
                "algorithm": "char",
                "vocab_size": len(char2idx),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "encode": lambda t: [char2idx.get(c, 1) for c in t],
                "decode": lambda ids: ''.join(idx2char.get(i, '?') for i in ids),
            }
        else:
            # Whitespace tokenizer
            words = sorted(set(text.split()))[:vocab_size - 5]
            word2idx = {w: i + 5 for i, w in enumerate(words)}
            word2idx["[PAD]"] = 0
            word2idx["[UNK]"] = 1
            word2idx["[BOS]"] = 2
            word2idx["[EOS]"] = 3
            word2idx["[MASK]"] = 4
            idx2word = {i: w for w, i in word2idx.items()}
            return {
                "type": "custom_trained",
                "algorithm": "whitespace",
                "vocab_size": len(word2idx),
                "word2idx": word2idx,
                "idx2word": idx2word,
                "encode": lambda t: [word2idx.get(w, 1) for w in t.split()],
                "decode": lambda ids: ' '.join(idx2word.get(i, '?') for i in ids),
            }


# ---------------------------------------------------------------------------
# Feature 3: Dynamic Computation (Early Exit / Adaptive Depth)
# ---------------------------------------------------------------------------

class EarlyExitClassifier(nn.Module):
    """Exit classifier attached to intermediate layers."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.confidence = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm(x)
        logits = self.head(normed)
        conf = torch.sigmoid(self.confidence(normed[:, -1:, :]))  # confidence on last token
        return logits, conf


class AdaptiveDepthLLM(nn.Module):
    """LLM with early exit capability — skips remaining layers when confident.

    During training: runs all layers, computes auxiliary loss from exit classifiers.
    During inference: exits early when confidence exceeds threshold.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        max_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        exit_interval: int = 2,  # Place exit every N layers
        exit_threshold: float = 0.9,  # Confidence threshold for early exit
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.exit_threshold = exit_threshold
        self.exit_interval = exit_interval

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.layers = nn.ModuleList([
            LLMDecoderBlock(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                norm_type=norm_type, ffn_type=ffn_type, max_len=max_len,
            )
            for _ in range(n_layers)
        ])

        # Exit classifiers at regular intervals
        self.exits = nn.ModuleDict()
        for i in range(exit_interval - 1, n_layers, exit_interval):
            self.exits[str(i)] = EarlyExitClassifier(d_model, vocab_size)

        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm = NormClass(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> dict:
        x = self.drop(self.tok_emb(input_ids))

        exit_logits = {}
        exit_losses = []
        exited_at = self.n_layers  # Default: full depth

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if str(i) in self.exits:
                logits_i, conf_i = self.exits[str(i)](x)
                exit_logits[i] = logits_i

                # Compute exit loss during training
                if labels is not None:
                    shift = logits_i[:, :-1, :].contiguous()
                    target = labels[:, 1:].contiguous()
                    eloss = F.cross_entropy(shift.view(-1, self.vocab_size), target.view(-1), ignore_index=-100)
                    exit_losses.append(eloss)

                # Early exit during inference
                if not self.training and conf_i.mean() > self.exit_threshold:
                    exited_at = i + 1
                    final_logits = logits_i
                    break

        if exited_at == self.n_layers:
            x = self.norm(x)
            final_logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift = final_logits[:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            main_loss = F.cross_entropy(shift.view(-1, self.vocab_size), target.view(-1), ignore_index=-100)
            # Auxiliary exit losses (weighted average)
            if exit_losses:
                aux_loss = sum(exit_losses) / len(exit_losses)
                loss = main_loss + 0.1 * aux_loss
            else:
                loss = main_loss

        return {
            "logits": final_logits,
            "loss": loss,
            "exited_at": exited_at,
            "exit_logits": exit_logits,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            out = self(input_ids[:, -self.max_len:])
            logits = out["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "total_M": f"{total / 1e6:.1f}M",
                "embedding": self.tok_emb.weight.numel(),
                "per_layer": sum(p.numel() for p in self.layers[0].parameters()) if self.layers else 0}


# ---------------------------------------------------------------------------
# Feature 4: Multi-Modal Fusion
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings (ViT-style).

    Input: (B, C, H, W) image tensor
    Output: (B, num_patches, d_model) embedding sequence
    """

    def __init__(self, d_model: int = 512, patch_size: int = 16,
                 in_channels: int = 3, image_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, d_model)
        x = x + self.pos_emb[:, :x.shape[1]]
        return x


class AudioEmbedding(nn.Module):
    """Convert audio waveform/spectrogram into embeddings.

    Input: (B, 1, T) waveform or (B, n_mels, T) spectrogram
    Output: (B, seq_len, d_model) embedding sequence
    """

    def __init__(self, d_model: int = 512, n_mels: int = 80, stride: int = 160):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T', d_model)
        return self.proj(x)


class ModalityProjector(nn.Module):
    """Projects one modality's embeddings into the LLM's embedding space."""

    def __init__(self, input_dim: int, output_dim: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = input_dim if i == 0 else output_dim
            layers.extend([nn.Linear(d_in, output_dim), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalLLM(nn.Module):
    """LLM that can process text + images + audio.

    Architecture:
      - Text: Token Embedding → blocks
      - Image: PatchEmbedding → Projector → prepend/interleave with text
      - Audio: AudioEmbedding → Projector → prepend/interleave with text
      - Combined sequence → decoder blocks → LM Head

    Fusion modes: "prepend" (put image/audio before text), "interleave" (alternate),
                  "cross_attention" (attend to modality via cross-attn layers)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        ffn_type: str = "swiglu",
        tie_weights: bool = True,
        # Image config
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        # Audio config
        n_mels: int = 80,
        # Fusion
        fusion_mode: str = "prepend",  # "prepend", "cross_attention"
        modalities: Optional[list[str]] = None,  # ["text", "image", "audio"]
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.fusion_mode = fusion_mode
        self.modalities = modalities or ["text"]

        # Text embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Image encoder + projector
        if "image" in self.modalities:
            self.image_encoder = PatchEmbedding(d_model, patch_size, in_channels, image_size)
            self.image_projector = ModalityProjector(d_model, d_model)

        # Audio encoder + projector
        if "audio" in self.modalities:
            self.audio_encoder = AudioEmbedding(d_model, n_mels)
            self.audio_projector = ModalityProjector(d_model, d_model)

        # Decoder blocks
        self.layers = nn.ModuleList([
            LLMDecoderBlock(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
                norm_type=norm_type, ffn_type=ffn_type, max_len=max_len,
            )
            for _ in range(n_layers)
        ])

        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        self.norm = NormClass(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                audio: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        # Text embeddings
        text_emb = self.drop(self.tok_emb(input_ids))

        # Collect all modality embeddings
        prefix_embeds = []

        if images is not None and "image" in self.modalities:
            img_emb = self.image_projector(self.image_encoder(images))
            prefix_embeds.append(img_emb)

        if audio is not None and "audio" in self.modalities:
            aud_emb = self.audio_projector(self.audio_encoder(audio))
            prefix_embeds.append(aud_emb)

        # Fuse: prepend modality tokens before text
        if prefix_embeds:
            all_prefix = torch.cat(prefix_embeds, dim=1)
            x = torch.cat([all_prefix, text_emb], dim=1)
        else:
            x = text_emb

        # Decoder blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        # Only compute loss on text portion
        loss = None
        if labels is not None:
            n_prefix = sum(e.shape[1] for e in prefix_embeds) if prefix_embeds else 0
            text_logits = logits[:, n_prefix:, :]
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, images: Optional[torch.Tensor] = None,
                 audio: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 100, temperature: float = 0.8,
                 top_k: int = 50) -> torch.Tensor:
        for _ in range(max_new_tokens):
            out = self(input_ids[:, -self.max_len:], images=images, audio=audio)
            logits = out["logits"][:, -1, :]
            logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            images = None  # Only encode images on first pass
            audio = None
        return input_ids

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result = {"total": total, "trainable": trainable, "total_M": f"{total / 1e6:.1f}M",
                  "embedding": self.tok_emb.weight.numel()}
        if "image" in self.modalities:
            result["image_encoder"] = sum(p.numel() for p in self.image_encoder.parameters())
        if "audio" in self.modalities:
            result["audio_encoder"] = sum(p.numel() for p in self.audio_encoder.parameters())
        return result


# Pre-built block designs users can use as starting points
BLOCK_DESIGNS = {
    "llama": [
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        {"type": "residual", "residual_from": 2},
    ],
    "gpt2": [
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "ffn", "config": {"ffn_type": "standard"}},
        {"type": "residual", "residual_from": 2},
    ],
    "palm": [
        # PaLM-style: parallel attention + FFN
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "attention", "config": {}},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        {"type": "residual", "residual_from": -1},
    ],
    "gemma": [
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "ffn", "config": {"ffn_type": "geglu"}},
        {"type": "residual", "residual_from": 2},
    ],
    "minimal": [
        {"type": "attention", "config": {}},
        {"type": "ffn", "config": {"ffn_type": "standard"}},
    ],
    "deep_norm": [
        # Post-norm architecture (used in some deep models)
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        {"type": "residual", "residual_from": 2},
        {"type": "norm", "config": {"norm_type": "layernorm"}},
    ],
    "moe_block": [
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "moe", "config": {"n_experts": 8, "top_k": 2}},
        {"type": "residual", "residual_from": 2},
    ],
    "mamba": [
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "mamba", "config": {"d_state": 16, "expand": 2}},
        {"type": "residual", "residual_from": -1},
    ],
    "rwkv": [
        {"type": "rwkv", "config": {}},
    ],
    "retnet": [
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "retention", "config": {}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "ffn", "config": {"ffn_type": "standard"}},
        {"type": "residual", "residual_from": 3},
    ],
    "hyena": [
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "hyena", "config": {"order": 2}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "layernorm"}},
        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        {"type": "residual", "residual_from": 3},
    ],
    "xlstm": [
        {"type": "xlstm", "config": {"n_layers": 1}},
    ],
    "griffin": [
        {"type": "gated_recurrence", "config": {"expand": 2}},
    ],
    "hybrid_mamba_attn": [
        # Alternating Mamba + Attention (like Jamba)
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "mamba", "config": {"d_state": 16}},
        {"type": "residual", "residual_from": -1},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "attention", "config": {}},
        {"type": "residual", "residual_from": 3},
        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        {"type": "residual", "residual_from": 5},
    ],
}


# Component library for users to browse available step types
COMPONENT_CATALOG = {
    "norm": {
        "name": "Normalization",
        "description": "Normalize activations (RMSNorm or LayerNorm)",
        "config_schema": {
            "norm_type": {"type": "select", "options": ["rmsnorm", "layernorm"], "default": "rmsnorm"},
        },
    },
    "attention": {
        "name": "Self-Attention",
        "description": "Multi-head attention with RoPE, GQA, Flash support",
        "config_schema": {
            "n_heads": {"type": "int", "default": 8, "min": 1},
            "n_kv_heads": {"type": "int", "default": None, "min": 1, "description": "KV heads for GQA (null = n_heads)"},
            "use_flash": {"type": "bool", "default": True},
            "dropout": {"type": "float", "default": 0.0, "min": 0, "max": 0.5},
            "rope_base": {"type": "float", "default": 10000.0},
        },
    },
    "ffn": {
        "name": "Feed-Forward Network",
        "description": "FFN layer (SwiGLU, GeGLU, ReGLU, or Standard)",
        "config_schema": {
            "ffn_type": {"type": "select", "options": ["swiglu", "geglu", "reglu", "standard"], "default": "swiglu"},
            "hidden_dim": {"type": "int", "default": None, "description": "Auto = ~2.7x d_model"},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "moe": {
        "name": "Mixture of Experts",
        "description": "Top-K routing to multiple FFN experts",
        "config_schema": {
            "n_experts": {"type": "int", "default": 8, "min": 2},
            "top_k": {"type": "int", "default": 2, "min": 1},
            "hidden_dim": {"type": "int", "default": None},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "residual": {
        "name": "Residual Connection",
        "description": "Add output from a previous step (skip connection)",
        "config_schema": {
            "residual_from": {"type": "int", "default": -1, "description": "-1 = block input"},
        },
    },
    "dropout": {
        "name": "Dropout",
        "description": "Randomly zero elements during training",
        "config_schema": {
            "p": {"type": "float", "default": 0.1, "min": 0, "max": 1},
        },
    },
    "linear": {
        "name": "Linear Projection",
        "description": "Linear transform: y = xW + b",
        "config_schema": {
            "out_features": {"type": "int", "default": None, "description": "Default = d_model"},
            "bias": {"type": "bool", "default": False},
        },
    },
    "activation": {
        "name": "Activation Function",
        "description": "Non-linear activation",
        "config_schema": {
            "name": {"type": "select", "options": ["relu", "gelu", "silu", "tanh", "sigmoid", "mish"], "default": "silu"},
        },
    },
    "custom_code": {
        "name": "Custom Layer (Code)",
        "description": "Write your own nn.Module in Python",
        "config_schema": {
            "code": {"type": "code", "default": "class CustomModule(nn.Module):\n    def __init__(self, d_model, **kwargs):\n        super().__init__()\n        self.linear = nn.Linear(d_model, d_model)\n\n    def forward(self, x):\n        return self.linear(x)\n"},
        },
    },
    "custom_formula": {
        "name": "Custom FFN (Formula)",
        "description": "Define FFN using a math formula. Available: self.fc1, self.fc2, self.gate, self.norm, F, torch",
        "config_schema": {
            "formula": {"type": "code", "default": "self.fc2(F.gelu(self.fc1(x)))"},
        },
    },
    "sliding_window_attention": {
        "name": "Sliding Window Attention",
        "description": "Attends only to the last N tokens (Mistral/Longformer-style). O(n*w) instead of O(n^2)",
        "config_schema": {
            "n_heads": {"type": "int", "default": 8, "min": 1},
            "window_size": {"type": "int", "default": 256, "min": 1},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "linear_attention": {
        "name": "Linear Attention",
        "description": "O(n) attention using kernel feature maps. Trades quality for speed on long sequences",
        "config_schema": {
            "n_heads": {"type": "int", "default": 8, "min": 1},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "alibi_attention": {
        "name": "ALiBi Attention",
        "description": "Attention with Linear Biases — no positional embeddings needed. Extrapolates to longer sequences",
        "config_schema": {
            "n_heads": {"type": "int", "default": 8, "min": 1},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "parallel": {
        "name": "Parallel Branches",
        "description": "Run two sub-components in parallel and merge (PaLM-style). Merge modes: add, concat, gate",
        "config_schema": {
            "branch_a": {"type": "step", "default": {"type": "attention", "config": {}}},
            "branch_b": {"type": "step", "default": {"type": "ffn", "config": {"ffn_type": "swiglu"}}},
            "merge": {"type": "select", "options": ["add", "concat", "gate"], "default": "add"},
        },
    },
    "cross_attention": {
        "name": "Cross-Attention",
        "description": "Attend to external context (encoder output, memory, etc.)",
        "config_schema": {
            "n_heads": {"type": "int", "default": 8, "min": 1},
            "dropout": {"type": "float", "default": 0.0},
        },
    },
    "conv1d": {
        "name": "1D Convolution",
        "description": "Convolution over sequence dimension (used in Mamba, Hyena, local mixing)",
        "config_schema": {
            "kernel_size": {"type": "int", "default": 3, "min": 1},
            "groups": {"type": "int", "default": 1, "min": 1},
        },
    },
    "pos_encoding": {
        "name": "Positional Encoding",
        "description": "Add positional information: absolute (learned), sinusoidal (fixed), or none",
        "config_schema": {
            "encoding_type": {"type": "select", "options": ["absolute", "sinusoidal", "none"], "default": "absolute"},
        },
    },
}


# ---------------------------------------------------------------------------
# Register all LLM layers with the Registry
# ---------------------------------------------------------------------------
from state_graph.core.registry import Registry

Registry.register_layer("RMSNorm", RMSNorm)
Registry.register_layer("RotaryPositionalEmbedding", RotaryPositionalEmbedding)
Registry.register_layer("LLMAttention", LLMAttention)
Registry.register_layer("SwiGLUFFN", SwiGLUFFN)
Registry.register_layer("GeGLUFFN", GeGLUFFN)
Registry.register_layer("ReGLUFFN", ReGLUFFN)
Registry.register_layer("StandardFFN", StandardFFN)
Registry.register_layer("MoELayer", MoELayer)
Registry.register_layer("LLMDecoderBlock", LLMDecoderBlock)
Registry.register_layer("LLMModel", LLMModel)
Registry.register_layer("ComposableBlock", ComposableBlock)
Registry.register_layer("ComposableLLM", ComposableLLM)
Registry.register_layer("CustomFFN", CustomFFN)
Registry.register_layer("SlidingWindowAttention", SlidingWindowAttention)
Registry.register_layer("LinearAttention", LinearAttention)
Registry.register_layer("ALiBiAttention", ALiBiAttention)
Registry.register_layer("ParallelBranch", ParallelBranch)
Registry.register_layer("AbsolutePositionalEncoding", AbsolutePositionalEncoding)
Registry.register_layer("SinusoidalPositionalEncoding", SinusoidalPositionalEncoding)
Registry.register_layer("EncoderBlock", EncoderBlock)
Registry.register_layer("DecoderBlockWithCrossAttn", DecoderBlockWithCrossAttn)
Registry.register_layer("EncoderDecoderLLM", EncoderDecoderLLM)
Registry.register_layer("AdaptiveDepthLLM", AdaptiveDepthLLM)
Registry.register_layer("PatchEmbedding", PatchEmbedding)
Registry.register_layer("AudioEmbedding", AudioEmbedding)
Registry.register_layer("MultiModalLLM", MultiModalLLM)
