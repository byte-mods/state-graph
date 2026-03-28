"""Tests for LLM layers, model building, modification, and training endpoints."""

import torch
import torch.nn as nn
import pytest

from state_graph.layers.llm import (
    RMSNorm, RotaryPositionalEmbedding, LLMAttention,
    SwiGLUFFN, GeGLUFFN, ReGLUFFN, StandardFFN,
    MoERouter, MoELayer, LLMDecoderBlock, LLMModel,
    FFN_TYPES, apply_rotary_pos_emb, _rotate_half,
    CustomComponent, CustomFFN, ComposableBlock, ComposableLLM,
    BLOCK_DESIGNS, COMPONENT_CATALOG,
    SlidingWindowAttention, LinearAttention, ALiBiAttention,
    AbsolutePositionalEncoding, SinusoidalPositionalEncoding, NoPE,
    ParallelBranch, ATTENTION_TYPES, POS_ENCODING_TYPES,
    EncoderBlock, DecoderBlockWithCrossAttn, EncoderDecoderLLM,
    TokenizerTrainer, EarlyExitClassifier, AdaptiveDepthLLM,
    PatchEmbedding, AudioEmbedding, ModalityProjector, MultiModalLLM,
)


# ── Helper Function Tests ──

class TestRotateHalf:
    def test_shape(self):
        x = torch.randn(2, 4, 8, 32)
        out = _rotate_half(x)
        assert out.shape == x.shape

    def test_rotation_correctness(self):
        # For [a, b], _rotate_half should give [-b, a]
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = _rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_double_rotation(self):
        # Rotating twice = negation
        x = torch.randn(1, 4)
        out = _rotate_half(_rotate_half(x))
        assert torch.allclose(out, -x, atol=1e-6)


# ── Layer Unit Tests ──

class TestRMSNorm:
    def test_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_normalization_effect(self):
        norm = RMSNorm(32)
        x = torch.randn(1, 5, 32) * 100
        out = norm(x)
        # Output should have smaller scale than input
        assert out.abs().mean() < x.abs().mean()

    def test_weight_init(self):
        norm = RMSNorm(16)
        assert torch.allclose(norm.weight, torch.ones(16))

    def test_custom_eps(self):
        norm = RMSNorm(32, eps=1e-8)
        assert norm.eps == 1e-8
        x = torch.randn(1, 5, 32)
        out = norm(x)
        assert out.shape == (1, 5, 32)

    def test_2d_input(self):
        norm = RMSNorm(16)
        x = torch.randn(4, 16)
        out = norm(x)
        assert out.shape == (4, 16)

    def test_gradient_flow(self):
        norm = RMSNorm(16)
        x = torch.randn(1, 5, 16, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


class TestRotaryPositionalEmbedding:
    def test_shape(self):
        rope = RotaryPositionalEmbedding(64, max_len=128)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (1, 10, 64)
        assert sin.shape == (1, 10, 64)

    def test_cache_extension(self):
        rope = RotaryPositionalEmbedding(32, max_len=64)
        x = torch.randn(1, 100, 32)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape[1] == 100

    def test_custom_base(self):
        rope = RotaryPositionalEmbedding(32, max_len=64, base=500000.0)
        assert rope.base == 500000.0
        x = torch.randn(1, 10, 32)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (1, 10, 32)

    def test_default_seq_len(self):
        rope = RotaryPositionalEmbedding(32, max_len=64)
        x = torch.randn(1, 10, 32)
        cos, sin = rope(x)  # No seq_len, should use x.shape[1]
        assert cos.shape[1] == 10

    def test_apply_rotary(self):
        q = torch.randn(1, 4, 10, 32)
        k = torch.randn(1, 4, 10, 32)
        cos = torch.ones(1, 10, 32)
        sin = torch.zeros(1, 10, 32)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        # cos=1, sin=0 should keep q and k unchanged
        assert torch.allclose(q_out, q, atol=1e-6)
        assert torch.allclose(k_out, k, atol=1e-6)

    def test_apply_rotary_nontrivial(self):
        # With actual non-trivial cos/sin, output should differ from input
        q = torch.randn(1, 4, 10, 32)
        k = torch.randn(1, 4, 10, 32)
        rope = RotaryPositionalEmbedding(32, max_len=64)
        cos, sin = rope(q, seq_len=10)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        assert not torch.allclose(q_out, q, atol=1e-3)


class TestLLMAttention:
    def test_basic_forward(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == (2, 10, 64)

    def test_gqa(self):
        attn = LLMAttention(d_model=64, n_heads=8, n_kv_heads=2, max_len=32)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)
        assert attn.n_kv_groups == 4

    def test_mqa(self):
        attn = LLMAttention(d_model=64, n_heads=8, n_kv_heads=1, max_len=32)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_no_flash(self):
        attn = LLMAttention(d_model=64, n_heads=4, use_flash=False, max_len=32)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_kv_cache(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        attn.clear_kv_cache()
        x1 = torch.randn(1, 1, 64)
        out1 = attn(x1, use_cache=True)
        assert out1.shape == (1, 1, 64)
        assert attn._kv_cache is not None
        attn.clear_kv_cache()
        assert attn._kv_cache is None

    def test_kv_cache_incremental(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        attn.clear_kv_cache()
        x1 = torch.randn(1, 1, 64)
        attn(x1, use_cache=True)
        k1_len = attn._kv_cache[0].shape[2]
        x2 = torch.randn(1, 1, 64)
        attn(x2, use_cache=True)
        k2_len = attn._kv_cache[0].shape[2]
        assert k2_len == k1_len + 1  # Cache grows

    def test_with_dropout_training(self):
        attn = LLMAttention(d_model=64, n_heads=4, dropout=0.1, max_len=32)
        attn.train()
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_with_explicit_mask(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
        out = attn(x, mask=mask)
        assert out.shape == (1, 8, 64)

    def test_single_token(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == (1, 1, 64)

    def test_gradient_flow(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_custom_rope_base(self):
        attn = LLMAttention(d_model=64, n_heads=4, max_len=32, rope_base=500000.0)
        assert attn.rope.base == 500000.0


class TestSwiGLUFFN:
    def test_shape(self):
        ffn = SwiGLUFFN(64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_custom_hidden(self):
        ffn = SwiGLUFFN(64, hidden_dim=128)
        assert ffn.gate_proj.out_features == 128
        x = torch.randn(1, 5, 64)
        out = ffn(x)
        assert out.shape == (1, 5, 64)

    def test_auto_hidden_dim_alignment(self):
        # Auto-calculated hidden dim should be multiple of 64
        ffn = SwiGLUFFN(100)
        assert ffn.gate_proj.out_features % 64 == 0

    def test_with_dropout(self):
        ffn = SwiGLUFFN(64, dropout=0.1)
        ffn.train()
        x = torch.randn(1, 5, 64)
        out = ffn(x)
        assert out.shape == (1, 5, 64)

    def test_gradient_flow(self):
        ffn = SwiGLUFFN(64)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestGeGLUFFN:
    def test_shape(self):
        ffn = GeGLUFFN(64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_custom_hidden(self):
        ffn = GeGLUFFN(64, hidden_dim=256)
        assert ffn.gate_proj.out_features == 256

    def test_with_dropout(self):
        ffn = GeGLUFFN(64, dropout=0.2)
        ffn.train()
        x = torch.randn(1, 5, 64)
        out = ffn(x)
        assert out.shape == (1, 5, 64)

    def test_gradient_flow(self):
        ffn = GeGLUFFN(64)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_auto_hidden_dim_alignment(self):
        ffn = GeGLUFFN(100)
        assert ffn.gate_proj.out_features % 64 == 0


class TestReGLUFFN:
    def test_shape(self):
        ffn = ReGLUFFN(64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_custom_hidden(self):
        ffn = ReGLUFFN(64, hidden_dim=128)
        assert ffn.gate_proj.out_features == 128

    def test_with_dropout(self):
        ffn = ReGLUFFN(64, dropout=0.15)
        ffn.train()
        x = torch.randn(1, 5, 64)
        out = ffn(x)
        assert out.shape == (1, 5, 64)

    def test_gradient_flow(self):
        ffn = ReGLUFFN(64)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_auto_hidden_dim_alignment(self):
        ffn = ReGLUFFN(100)
        assert ffn.gate_proj.out_features % 64 == 0


class TestStandardFFN:
    def test_shape(self):
        ffn = StandardFFN(64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)

    def test_default_hidden(self):
        ffn = StandardFFN(64)
        assert ffn.fc1.out_features == 256  # 4 * 64

    def test_custom_hidden(self):
        ffn = StandardFFN(64, hidden_dim=128)
        assert ffn.fc1.out_features == 128

    def test_has_bias(self):
        # Standard FFN uses bias (unlike gated variants)
        ffn = StandardFFN(64)
        assert ffn.fc1.bias is not None
        assert ffn.fc2.bias is not None

    def test_with_dropout(self):
        ffn = StandardFFN(64, dropout=0.1)
        ffn.train()
        x = torch.randn(1, 5, 64)
        out = ffn(x)
        assert out.shape == (1, 5, 64)

    def test_gradient_flow(self):
        ffn = StandardFFN(64)
        x = torch.randn(1, 5, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestFFNTypesRegistry:
    def test_all_types_present(self):
        assert "swiglu" in FFN_TYPES
        assert "geglu" in FFN_TYPES
        assert "reglu" in FFN_TYPES
        assert "standard" in FFN_TYPES

    def test_correct_classes(self):
        assert FFN_TYPES["swiglu"] is SwiGLUFFN
        assert FFN_TYPES["geglu"] is GeGLUFFN
        assert FFN_TYPES["reglu"] is ReGLUFFN
        assert FFN_TYPES["standard"] is StandardFFN


class TestMoERouter:
    def test_shape(self):
        router = MoERouter(64, n_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        probs, indices, all_probs = router(x)
        assert probs.shape == (2, 10, 2)
        assert indices.shape == (2, 10, 2)
        assert all_probs.shape == (2, 10, 4)

    def test_probs_sum_to_one(self):
        router = MoERouter(64, n_experts=4, top_k=2)
        x = torch.randn(1, 5, 64)
        probs, _, _ = router(x)
        # top_k probs are renormalized
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_top_k_1(self):
        router = MoERouter(64, n_experts=4, top_k=1)
        x = torch.randn(1, 5, 64)
        probs, indices, _ = router(x)
        assert probs.shape == (1, 5, 1)
        assert indices.shape == (1, 5, 1)

    def test_training_noise(self):
        router = MoERouter(64, n_experts=4, top_k=2, noise_std=0.1)
        router.train()
        x = torch.randn(1, 5, 64)
        # Should run without error with noise
        probs, indices, all_probs = router(x)
        assert probs.shape == (1, 5, 2)

    def test_eval_no_noise(self):
        router = MoERouter(64, n_experts=4, top_k=2, noise_std=0.1)
        router.eval()
        x = torch.randn(1, 5, 64)
        # Same input should give same output in eval mode (deterministic)
        p1, _, _ = router(x)
        p2, _, _ = router(x)
        assert torch.allclose(p1, p2)


class TestMoELayer:
    def test_shape(self):
        moe = MoELayer(64, n_experts=4, top_k=2)
        x = torch.randn(2, 10, 64)
        out = moe(x)
        assert out.shape == (2, 10, 64)

    def test_aux_loss(self):
        moe = MoELayer(64, n_experts=4, top_k=2)
        moe.train()
        x = torch.randn(1, 8, 64)
        moe(x)
        assert moe.aux_loss > 0

    def test_eval_no_aux_loss_update(self):
        moe = MoELayer(64, n_experts=4, top_k=2)
        moe.eval()
        moe.aux_loss = 0.0
        x = torch.randn(1, 8, 64)
        moe(x)
        assert moe.aux_loss == 0.0  # Not updated in eval mode

    def test_custom_expert_hidden_dim(self):
        moe = MoELayer(64, n_experts=4, top_k=2, expert_hidden_dim=128)
        assert moe.experts[0].gate_proj.out_features == 128

    def test_gradient_flow(self):
        moe = MoELayer(64, n_experts=4, top_k=2)
        moe.train()
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None

    def test_expert_count(self):
        moe = MoELayer(64, n_experts=8, top_k=2)
        assert len(moe.experts) == 8
        assert moe.n_experts == 8
        assert moe.top_k == 2


class TestLLMDecoderBlock:
    def test_basic(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_rmsnorm(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, norm_type="rmsnorm", max_len=32)
        assert isinstance(block.norm1, RMSNorm)

    def test_layernorm(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, norm_type="layernorm", max_len=32)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)

    def test_ffn_swiglu(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_type="swiglu", max_len=32)
        assert isinstance(block.ffn, SwiGLUFFN)

    def test_ffn_geglu(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_type="geglu", max_len=32)
        assert isinstance(block.ffn, GeGLUFFN)

    def test_ffn_reglu(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_type="reglu", max_len=32)
        assert isinstance(block.ffn, ReGLUFFN)

    def test_ffn_standard(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_type="standard", max_len=32)
        assert isinstance(block.ffn, StandardFFN)

    def test_moe(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, use_moe=True, n_experts=4, max_len=32)
        assert isinstance(block.ffn, MoELayer)

    def test_gqa(self):
        block = LLMDecoderBlock(d_model=64, n_heads=8, n_kv_heads=2, max_len=32)
        assert block.attn.n_kv_heads == 2

    def test_custom_rope_base(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, rope_base=500000.0, max_len=32)
        assert block.attn.rope.base == 500000.0

    def test_attributes_stored(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, norm_type="layernorm", ffn_type="geglu", max_len=32)
        assert block.norm_type == "layernorm"
        assert block.ffn_type == "geglu"

    def test_forward_all_ffn_types(self):
        """Verify forward pass works for every FFN type."""
        x = torch.randn(1, 8, 64)
        for ffn_type in ["swiglu", "geglu", "reglu", "standard"]:
            block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_type=ffn_type, max_len=32)
            out = block(x)
            assert out.shape == (1, 8, 64), f"Forward failed for ffn_type={ffn_type}"

    def test_forward_all_norm_types(self):
        x = torch.randn(1, 8, 64)
        for norm_type in ["rmsnorm", "layernorm"]:
            block = LLMDecoderBlock(d_model=64, n_heads=4, norm_type=norm_type, max_len=32)
            out = block(x)
            assert out.shape == (1, 8, 64), f"Forward failed for norm_type={norm_type}"

    def test_residual_connection(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        # Output should differ from input (not identity), but also not be wildly different
        assert not torch.allclose(out, x, atol=1e-3)

    def test_with_dropout(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, dropout=0.1, max_len=32)
        block.train()
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_gradient_flow(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_custom_ffn_hidden_dim(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, ffn_hidden_dim=256, max_len=32)
        assert block.ffn.gate_proj.out_features == 256

    def test_moe_top_k(self):
        block = LLMDecoderBlock(d_model=64, n_heads=4, use_moe=True, n_experts=4, moe_top_k=1, max_len=32)
        assert block.ffn.top_k == 1


class TestLLMModel:
    def test_basic_forward(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (1, 16))
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (1, 16, 100)
        assert out["loss"] is None

    def test_with_labels(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))
        out = model(x, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].dim() == 0  # scalar

    def test_generate(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        model.eval()
        x = torch.randint(0, 100, (1, 5))
        gen = model.generate(x, max_new_tokens=10)
        assert gen.shape == (1, 15)  # 5 + 10

    def test_norm_type_rmsnorm(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, norm_type="rmsnorm")
        assert isinstance(model.norm, RMSNorm)

    def test_norm_type_layernorm(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, norm_type="layernorm")
        assert isinstance(model.norm, nn.LayerNorm)
        assert isinstance(model.layers[0].norm1, nn.LayerNorm)

    def test_ffn_type(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, ffn_type="geglu")
        assert isinstance(model.layers[0].ffn, GeGLUFFN)
        assert isinstance(model.layers[1].ffn, GeGLUFFN)

    def test_moe(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=4, n_heads=4, max_len=32,
                         use_moe=True, n_experts=4, moe_top_k=2)
        # MoE on odd layers by default
        assert isinstance(model.layers[1].ffn, MoELayer)
        assert isinstance(model.layers[3].ffn, MoELayer)
        assert not isinstance(model.layers[0].ffn, MoELayer)

    def test_weight_tying(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, tie_weights=True)
        assert model.lm_head.weight is model.tok_emb.weight

    def test_no_weight_tying(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, tie_weights=False)
        assert model.lm_head.weight is not model.tok_emb.weight

    def test_count_parameters(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert "total_M" in params

    def test_rope_base(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, rope_base=500000.0)
        assert model.layers[0].attn.rope.base == 500000.0

    def test_gqa(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=8, n_kv_heads=2, max_len=32)
        assert model.layers[0].attn.n_kv_heads == 2

    def test_layer_configs_override(self):
        layer_configs = [
            {"ffn_type": "geglu", "norm_type": "layernorm"},
            {"ffn_type": "standard", "norm_type": "rmsnorm"},
        ]
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
                         layer_configs=layer_configs)
        assert isinstance(model.layers[0].ffn, GeGLUFFN)
        assert isinstance(model.layers[0].norm1, nn.LayerNorm)
        assert isinstance(model.layers[1].ffn, StandardFFN)
        assert isinstance(model.layers[1].norm1, RMSNorm)

    def test_moe_aux_loss(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
                         use_moe=True, n_experts=4, moe_layers=[0, 1])
        model.train()
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["loss"] is not None

    def test_backward(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        # Check gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_clear_kv_cache(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        # Manually set a cache and verify clear works
        model.layers[0].attn._kv_cache = (torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16))
        model.clear_kv_cache()
        for layer in model.layers:
            assert layer.attn._kv_cache is None

    def test_generate_with_temperature(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        # Low temperature (more deterministic)
        gen_low = model.generate(x, max_new_tokens=5, temperature=0.01)
        assert gen_low.shape == (1, 8)
        # High temperature
        gen_high = model.generate(x, max_new_tokens=5, temperature=2.0)
        assert gen_high.shape == (1, 8)

    def test_generate_with_top_p(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        gen = model.generate(x, max_new_tokens=5, top_p=0.9)
        assert gen.shape == (1, 8)

    def test_generate_with_top_k_0(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        gen = model.generate(x, max_new_tokens=5, top_k=0)  # No top-k filtering
        assert gen.shape == (1, 8)

    def test_moe_explicit_layers(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=4, n_heads=4, max_len=32,
                         use_moe=True, moe_layers=[0, 2])
        assert isinstance(model.layers[0].ffn, MoELayer)
        assert not isinstance(model.layers[1].ffn, MoELayer)
        assert isinstance(model.layers[2].ffn, MoELayer)
        assert not isinstance(model.layers[3].ffn, MoELayer)

    def test_init_weights(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        # Weights should be initialized with small values (std=0.02)
        for name, p in model.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                assert p.data.std() < 0.1, f"Weight {name} has std {p.data.std()}"

    def test_dropout_layer(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, dropout=0.1)
        assert isinstance(model.drop, nn.Dropout)

    def test_no_dropout_layer(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, dropout=0.0)
        assert isinstance(model.drop, nn.Identity)

    def test_forward_all_architectures(self):
        """End-to-end forward pass with all FFN/norm combinations."""
        configs = [
            {"norm_type": "rmsnorm", "ffn_type": "swiglu"},
            {"norm_type": "layernorm", "ffn_type": "standard"},
            {"norm_type": "rmsnorm", "ffn_type": "geglu"},
            {"norm_type": "layernorm", "ffn_type": "reglu"},
        ]
        for cfg in configs:
            model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, **cfg)
            x = torch.randint(0, 100, (1, 8))
            labels = torch.randint(0, 100, (1, 8))
            out = model(x, labels=labels)
            assert out["loss"] is not None, f"Failed for config {cfg}"
            assert out["logits"].shape == (1, 8, 100), f"Wrong shape for config {cfg}"

    def test_batch_forward(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (4, 16))
        out = model(x)
        assert out["logits"].shape == (4, 16, 100)

    def test_get_moe_aux_loss_no_moe(self):
        model = LLMModel(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        assert model.get_moe_aux_loss() == 0.0

    def test_count_parameters_embedding(self):
        model = LLMModel(vocab_size=200, d_model=64, n_layers=2, n_heads=4, max_len=32, tie_weights=False)
        params = model.count_parameters()
        assert params["embedding"] == 200 * 64
        assert params["per_layer"] > 0

    def test_partial_layer_configs(self):
        # Fewer layer_configs than n_layers — remaining use defaults
        layer_configs = [{"ffn_type": "geglu"}]
        model = LLMModel(vocab_size=100, d_model=64, n_layers=3, n_heads=4, max_len=32,
                         layer_configs=layer_configs)
        assert isinstance(model.layers[0].ffn, GeGLUFFN)
        assert isinstance(model.layers[1].ffn, SwiGLUFFN)  # default
        assert isinstance(model.layers[2].ffn, SwiGLUFFN)  # default


# ── Composable System Unit Tests ──

class TestCustomComponent:
    def test_create_from_code(self):
        code = """
class CustomModule(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)
"""
        module = CustomComponent.create_from_code(code, 64)
        x = torch.randn(1, 8, 64)
        out = module(x)
        assert out.shape == (1, 8, 64)

    def test_create_complex_module(self):
        code = """
class GatedMLP(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.up = nn.Linear(d_model, d_model)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.up(x)
"""
        module = CustomComponent.create_from_code(code, 64)
        x = torch.randn(1, 4, 64)
        out = module(x)
        assert out.shape == (1, 4, 64)

    def test_invalid_code_no_module(self):
        with pytest.raises(ValueError, match="nn.Module subclass"):
            CustomComponent.create_from_code("x = 5", 64)

    def test_code_with_kwargs(self):
        code = """
class ScaledLinear(nn.Module):
    def __init__(self, d_model, scale=2.0, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.scale = scale

    def forward(self, x):
        return self.linear(x) * self.scale
"""
        module = CustomComponent.create_from_code(code, 64, scale=3.0)
        assert module.scale == 3.0


class TestCustomFFN:
    def test_default_formula(self):
        ffn = CustomFFN(64)
        x = torch.randn(1, 4, 64)
        out = ffn(x)
        assert out.shape == (1, 4, 64)

    def test_custom_formula(self):
        ffn = CustomFFN(64, formula="self.fc2(F.gelu(self.fc1(x)))")
        x = torch.randn(1, 4, 64)
        out = ffn(x)
        assert out.shape == (1, 4, 64)

    def test_gated_formula(self):
        ffn = CustomFFN(64, formula="self.fc2(F.silu(self.gate(x)) * self.fc1(x))")
        x = torch.randn(1, 4, 64)
        out = ffn(x)
        assert out.shape == (1, 4, 64)

    def test_formula_with_norm(self):
        ffn = CustomFFN(64, formula="self.fc2(self.norm(F.gelu(self.fc1(x))))")
        x = torch.randn(1, 4, 64)
        out = ffn(x)
        assert out.shape == (1, 4, 64)

    def test_custom_hidden_dim(self):
        ffn = CustomFFN(64, hidden_dim=128)
        assert ffn.fc1.out_features == 128
        assert ffn.hidden_dim == 128

    def test_gradient_flow(self):
        ffn = CustomFFN(64, formula="self.fc2(F.gelu(self.fc1(x)))")
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


class TestComposableBlock:
    def test_basic_llama_style(self):
        steps = BLOCK_DESIGNS["llama"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_gpt2_style(self):
        steps = BLOCK_DESIGNS["gpt2"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_palm_style(self):
        steps = BLOCK_DESIGNS["palm"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_minimal(self):
        steps = BLOCK_DESIGNS["minimal"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_deep_norm(self):
        steps = BLOCK_DESIGNS["deep_norm"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_moe_block(self):
        steps = BLOCK_DESIGNS["moe_block"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_custom_steps(self):
        steps = [
            {"type": "norm", "config": {"norm_type": "layernorm"}},
            {"type": "attention", "config": {"n_heads": 4}},
            {"type": "activation", "config": {"name": "gelu"}},
            {"type": "ffn", "config": {"ffn_type": "geglu"}},
        ]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_with_dropout_step(self):
        steps = [
            {"type": "attention", "config": {}},
            {"type": "dropout", "config": {"p": 0.1}},
            {"type": "ffn", "config": {"ffn_type": "swiglu"}},
        ]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        block.train()
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_with_linear_step(self):
        steps = [
            {"type": "linear", "config": {"out_features": 64}},
            {"type": "activation", "config": {"name": "relu"}},
        ]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_custom_code_step(self):
        code = """
class IdentityModule(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
    def forward(self, x):
        return x * 2
"""
        steps = [{"type": "custom_code", "config": {"code": code}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert torch.allclose(out, x * 2, atol=1e-6)

    def test_custom_formula_step(self):
        steps = [{"type": "custom_formula", "config": {"formula": "self.fc2(F.gelu(self.fc1(x)))"}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_get_step_info(self):
        steps = BLOCK_DESIGNS["llama"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        info = block.get_step_info()
        assert len(info) == len(steps)
        for s in info:
            assert "index" in s
            assert "type" in s
            assert "params" in s

    def test_gradient_flow(self):
        steps = BLOCK_DESIGNS["llama"]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None

    def test_unknown_step_raises(self):
        with pytest.raises(ValueError, match="Unknown step type"):
            ComposableBlock(d_model=64, steps=[{"type": "nonexistent"}], n_heads=4, max_len=32)

    def test_all_activation_types(self):
        for act in ["relu", "gelu", "silu", "tanh", "sigmoid", "mish"]:
            steps = [{"type": "activation", "config": {"name": act}}]
            block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
            x = torch.randn(1, 4, 64)
            out = block(x)
            assert out.shape == (1, 4, 64), f"Failed for activation {act}"


class TestComposableLLM:
    def test_basic_forward(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (1, 16))
        out = model(x)
        assert out["logits"].shape == (1, 16, 100)

    def test_with_labels(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        x = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))
        out = model(x, labels=labels)
        assert out["loss"] is not None

    def test_generate(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        model.eval()
        x = torch.randint(0, 100, (1, 5))
        gen = model.generate(x, max_new_tokens=10)
        assert gen.shape == (1, 15)

    def test_custom_default_block(self):
        block = BLOCK_DESIGNS["gpt2"]
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
                              default_block=block)
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_per_layer_block_designs(self):
        designs = [
            BLOCK_DESIGNS["llama"],
            BLOCK_DESIGNS["gpt2"],
            BLOCK_DESIGNS["minimal"],
        ]
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=3, n_heads=4, max_len=32,
                              block_designs=designs)
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_mixed_architectures_backward(self):
        designs = [
            BLOCK_DESIGNS["llama"],
            BLOCK_DESIGNS["palm"],
        ]
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
                              block_designs=designs)
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_weight_tying(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32, tie_weights=True)
        assert model.lm_head.weight is model.tok_emb.weight

    def test_count_parameters(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32)
        params = model.count_parameters()
        assert params["total"] > 0
        assert "total_M" in params


class TestBlockDesigns:
    def test_all_presets_exist(self):
        expected = ["llama", "gpt2", "palm", "gemma", "minimal", "deep_norm", "moe_block"]
        for name in expected:
            assert name in BLOCK_DESIGNS, f"Missing preset: {name}"

    def test_all_presets_buildable(self):
        for name, design in BLOCK_DESIGNS.items():
            block = ComposableBlock(d_model=64, steps=design, n_heads=4, max_len=32)
            x = torch.randn(1, 4, 64)
            out = block(x)
            assert out.shape == (1, 4, 64), f"Preset {name} failed"

    def test_all_presets_in_full_model(self):
        for name, design in BLOCK_DESIGNS.items():
            model = ComposableLLM(vocab_size=50, d_model=64, n_layers=1, n_heads=4, max_len=32,
                                  default_block=design)
            x = torch.randint(0, 50, (1, 8))
            out = model(x)
            assert out["logits"].shape == (1, 8, 50), f"Preset {name} failed in full model"


class TestSlidingWindowAttention:
    def test_shape(self):
        attn = SlidingWindowAttention(64, n_heads=4, window_size=4)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_window_size(self):
        attn = SlidingWindowAttention(64, n_heads=4, window_size=2)
        assert attn.window_size == 2

    def test_gradient_flow(self):
        attn = SlidingWindowAttention(64, n_heads=4, window_size=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestLinearAttention:
    def test_shape(self):
        attn = LinearAttention(64, n_heads=4)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_gradient_flow(self):
        attn = LinearAttention(64, n_heads=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestALiBiAttention:
    def test_shape(self):
        attn = ALiBiAttention(64, n_heads=4)
        x = torch.randn(1, 8, 64)
        out = attn(x)
        assert out.shape == (1, 8, 64)

    def test_slopes_computed(self):
        attn = ALiBiAttention(64, n_heads=4)
        assert attn.slopes.shape == (4,)
        assert (attn.slopes > 0).all()

    def test_gradient_flow(self):
        attn = ALiBiAttention(64, n_heads=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


class TestPositionalEncodings:
    def test_absolute(self):
        pe = AbsolutePositionalEncoding(64, max_len=32)
        x = torch.randn(1, 8, 64)
        out = pe(x)
        assert out.shape == (1, 8, 64)
        assert not torch.allclose(out, x)

    def test_sinusoidal(self):
        pe = SinusoidalPositionalEncoding(64, max_len=32)
        x = torch.randn(1, 8, 64)
        out = pe(x)
        assert out.shape == (1, 8, 64)
        assert not torch.allclose(out, x)

    def test_none(self):
        pe = NoPE()
        x = torch.randn(1, 8, 64)
        out = pe(x)
        assert torch.allclose(out, x)

    def test_registry(self):
        assert "rope" in POS_ENCODING_TYPES
        assert "absolute" in POS_ENCODING_TYPES
        assert "sinusoidal" in POS_ENCODING_TYPES
        assert "none" in POS_ENCODING_TYPES


class TestAttentionTypes:
    def test_registry(self):
        assert "standard" in ATTENTION_TYPES
        assert "sliding_window" in ATTENTION_TYPES
        assert "linear" in ATTENTION_TYPES
        assert "alibi" in ATTENTION_TYPES


class TestParallelBranch:
    def test_add_merge(self):
        a = nn.Linear(64, 64)
        b = nn.Linear(64, 64)
        pb = ParallelBranch(a, b, 64, merge_mode="add")
        x = torch.randn(1, 8, 64)
        out = pb(x)
        assert out.shape == (1, 8, 64)

    def test_concat_merge(self):
        a = nn.Linear(64, 64)
        b = nn.Linear(64, 64)
        pb = ParallelBranch(a, b, 64, merge_mode="concat")
        x = torch.randn(1, 8, 64)
        out = pb(x)
        assert out.shape == (1, 8, 64)

    def test_gate_merge(self):
        a = nn.Linear(64, 64)
        b = nn.Linear(64, 64)
        pb = ParallelBranch(a, b, 64, merge_mode="gate")
        x = torch.randn(1, 8, 64)
        out = pb(x)
        assert out.shape == (1, 8, 64)


class TestComposableBlockNewSteps:
    def test_sliding_window_step(self):
        steps = [{"type": "sliding_window_attention", "config": {"n_heads": 4, "window_size": 4}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_linear_attention_step(self):
        steps = [{"type": "linear_attention", "config": {"n_heads": 4}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_alibi_attention_step(self):
        steps = [{"type": "alibi_attention", "config": {"n_heads": 4}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_parallel_step(self):
        steps = [{"type": "parallel", "config": {
            "branch_a": {"type": "attention", "config": {}},
            "branch_b": {"type": "ffn", "config": {"ffn_type": "swiglu"}},
            "merge": "add",
        }}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_conv1d_step(self):
        steps = [{"type": "conv1d", "config": {"kernel_size": 3}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_cross_attention_step(self):
        steps = [{"type": "cross_attention", "config": {"n_heads": 4}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_pos_encoding_step(self):
        steps = [{"type": "pos_encoding", "config": {"encoding_type": "absolute"}}]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_complex_custom_architecture(self):
        """Build a truly novel architecture combining multiple new components."""
        steps = [
            {"type": "norm", "config": {"norm_type": "rmsnorm"}},
            {"type": "parallel", "config": {
                "branch_a": {"type": "attention", "config": {}},
                "branch_b": {"type": "conv1d", "config": {"kernel_size": 3}},
                "merge": "gate",
            }},
            {"type": "residual", "residual_from": -1},
            {"type": "norm", "config": {"norm_type": "rmsnorm"}},
            {"type": "ffn", "config": {"ffn_type": "geglu"}},
            {"type": "residual", "residual_from": 2},
        ]
        block = ComposableBlock(d_model=64, steps=steps, n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)


class TestComposableLLMExtended:
    def test_absolute_pos_encoding(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=1, n_heads=4,
                              max_len=32, pos_encoding="absolute")
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_sinusoidal_pos_encoding(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=1, n_heads=4,
                              max_len=32, pos_encoding="sinusoidal")
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_no_pos_encoding(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=1, n_heads=4,
                              max_len=32, pos_encoding="none",
                              default_block=[
                                  {"type": "alibi_attention", "config": {"n_heads": 4}},
                                  {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                              ])
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_extra_heads(self):
        model = ComposableLLM(vocab_size=100, d_model=64, n_layers=1, n_heads=4,
                              max_len=32, extra_heads={"classifier": 10, "reward": 1})
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)
        assert out["classifier"].shape == (1, 8, 10)
        assert out["reward"].shape == (1, 8, 1)

    def test_custom_loss(self):
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=1, n_heads=4, max_len=32,
            custom_loss="F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, vocab_size), labels[:, 1:].contiguous().view(-1), ignore_index=-100, label_smoothing=0.1)",
        )
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["loss"] is not None

    def test_alibi_model(self):
        """Build a full ALiBi-based model (no positional encoding)."""
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
            pos_encoding="none",
            default_block=[
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "alibi_attention", "config": {"n_heads": 4}},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 2},
            ],
        )
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_hybrid_attention_model(self):
        """Mix standard and sliding window attention across layers."""
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
            block_designs=[
                [  # Layer 0: standard attention
                    {"type": "norm", "config": {}},
                    {"type": "attention", "config": {}},
                    {"type": "residual", "residual_from": -1},
                    {"type": "norm", "config": {}},
                    {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                    {"type": "residual", "residual_from": 2},
                ],
                [  # Layer 1: sliding window
                    {"type": "norm", "config": {}},
                    {"type": "sliding_window_attention", "config": {"n_heads": 4, "window_size": 4}},
                    {"type": "residual", "residual_from": -1},
                    {"type": "norm", "config": {}},
                    {"type": "ffn", "config": {"ffn_type": "geglu"}},
                    {"type": "residual", "residual_from": 2},
                ],
            ],
        )
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)


# ── Feature 1: Encoder-Decoder Tests ──

class TestEncoderBlock:
    def test_shape(self):
        block = EncoderBlock(64, n_heads=4)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_gradient(self):
        block = EncoderBlock(64, n_heads=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestDecoderBlockWithCrossAttn:
    def test_shape(self):
        block = DecoderBlockWithCrossAttn(64, n_heads=4)
        x = torch.randn(1, 8, 64)
        enc = torch.randn(1, 12, 64)
        out = block(x, enc)
        assert out.shape == (1, 8, 64)

    def test_gradient(self):
        block = DecoderBlockWithCrossAttn(64, n_heads=4)
        x = torch.randn(1, 8, 64, requires_grad=True)
        enc = torch.randn(1, 12, 64)
        out = block(x, enc)
        out.sum().backward()
        assert x.grad is not None


class TestEncoderDecoderLLM:
    def test_forward(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        src = torch.randint(0, 100, (1, 8))
        tgt = torch.randint(0, 100, (1, 6))
        out = model(src, decoder_input_ids=tgt)
        assert out["logits"].shape == (1, 6, 100)

    def test_with_labels(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        src = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(src, labels=labels)
        assert out["loss"] is not None

    def test_generate(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        model.eval()
        src = torch.randint(0, 100, (1, 8))
        gen = model.generate(src, max_new_tokens=5)
        assert gen.shape[1] > 1

    def test_encode_decode_separate(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        src = torch.randint(0, 100, (1, 8))
        enc_out = model.encode(src)
        assert enc_out.shape == (1, 8, 64)
        tgt = torch.randint(0, 100, (1, 4))
        dec_out = model.decode(tgt, enc_out)
        assert dec_out.shape == (1, 4, 64)

    def test_count_params(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        params = model.count_parameters()
        assert params["encoder_params"] > 0
        assert params["decoder_params"] > 0

    def test_backward(self):
        model = EncoderDecoderLLM(vocab_size=100, d_model=64, n_encoder_layers=2,
                                   n_decoder_layers=2, n_heads=4, max_len=32)
        src = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(src, labels=labels)
        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"


# ── Feature 2: Tokenizer Training Tests ──

class TestTokenizerTrainer:
    def test_char_tokenizer(self):
        text = "hello world this is a test " * 20
        result = TokenizerTrainer.train(text, vocab_size=50, algorithm="char")
        assert result["type"] == "custom_trained"
        assert result["vocab_size"] > 0
        encoded = result["encode"]("hello")
        assert len(encoded) == 5
        decoded = result["decode"](encoded)
        assert decoded == "hello"

    def test_whitespace_tokenizer(self):
        text = "the quick brown fox jumps over the lazy dog " * 20
        result = TokenizerTrainer.train(text, vocab_size=100, algorithm="whitespace")
        assert result["type"] == "custom_trained"
        assert result["vocab_size"] > 0

    def test_encode_decode_roundtrip(self):
        text = "hello world test " * 50
        result = TokenizerTrainer.train(text, vocab_size=50, algorithm="char")
        original = "hello"
        encoded = result["encode"](original)
        decoded = result["decode"](encoded)
        assert decoded == original


# ── Feature 3: Adaptive Depth Tests ──

class TestEarlyExitClassifier:
    def test_shape(self):
        exit_cls = EarlyExitClassifier(64, 100)
        x = torch.randn(1, 8, 64)
        logits, conf = exit_cls(x)
        assert logits.shape == (1, 8, 100)
        assert conf.shape == (1, 1, 1)
        assert (conf >= 0).all() and (conf <= 1).all()


class TestAdaptiveDepthLLM:
    def test_forward_training(self):
        model = AdaptiveDepthLLM(vocab_size=100, d_model=64, n_layers=6,
                                  n_heads=4, max_len=32, exit_interval=2)
        model.train()
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["logits"].shape[2] == 100
        assert out["loss"] is not None
        assert out["exited_at"] == 6  # Training: always full depth

    def test_forward_inference(self):
        model = AdaptiveDepthLLM(vocab_size=100, d_model=64, n_layers=6,
                                  n_heads=4, max_len=32, exit_interval=2,
                                  exit_threshold=0.0)  # Low threshold = always exit early
        model.eval()
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape[2] == 100
        assert out["exited_at"] <= 6  # Should exit early

    def test_exits_created(self):
        model = AdaptiveDepthLLM(vocab_size=100, d_model=64, n_layers=6,
                                  n_heads=4, max_len=32, exit_interval=2)
        # Exit classifiers at layers 1, 3, 5
        assert "1" in model.exits
        assert "3" in model.exits
        assert "5" in model.exits

    def test_generate(self):
        model = AdaptiveDepthLLM(vocab_size=100, d_model=64, n_layers=4,
                                  n_heads=4, max_len=32, exit_interval=2)
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        gen = model.generate(x, max_new_tokens=5)
        assert gen.shape == (1, 8)

    def test_backward(self):
        model = AdaptiveDepthLLM(vocab_size=100, d_model=64, n_layers=4,
                                  n_heads=4, max_len=32, exit_interval=2)
        model.train()
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        out["loss"].backward()


# ── Feature 4: Multi-Modal Tests ──

class TestPatchEmbedding:
    def test_shape(self):
        pe = PatchEmbedding(d_model=64, patch_size=16, image_size=64)
        img = torch.randn(1, 3, 64, 64)
        out = pe(img)
        n_patches = (64 // 16) ** 2  # 16
        assert out.shape == (1, n_patches + 1, 64)  # +1 for CLS

    def test_different_sizes(self):
        pe = PatchEmbedding(d_model=32, patch_size=8, image_size=32)
        img = torch.randn(2, 3, 32, 32)
        out = pe(img)
        assert out.shape[0] == 2
        assert out.shape[2] == 32


class TestAudioEmbedding:
    def test_shape(self):
        ae = AudioEmbedding(d_model=64, n_mels=80)
        audio = torch.randn(1, 80, 100)
        out = ae(audio)
        assert out.shape[0] == 1
        assert out.shape[2] == 64


class TestModalityProjector:
    def test_shape(self):
        proj = ModalityProjector(64, 128)
        x = torch.randn(1, 10, 64)
        out = proj(x)
        assert out.shape == (1, 10, 128)


class TestMultiModalLLM:
    def test_text_only(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=32, modalities=["text"])
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_text_and_image(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=64, modalities=["text", "image"],
                               image_size=32, patch_size=8)
        x = torch.randint(0, 100, (1, 8))
        img = torch.randn(1, 3, 32, 32)
        out = model(x, images=img)
        assert out["logits"].shape[0] == 1
        assert out["logits"].shape[2] == 100

    def test_text_image_audio(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=128, modalities=["text", "image", "audio"],
                               image_size=32, patch_size=8, n_mels=40)
        x = torch.randint(0, 100, (1, 8))
        img = torch.randn(1, 3, 32, 32)
        audio = torch.randn(1, 40, 50)
        out = model(x, images=img, audio=audio)
        assert out["logits"].shape[0] == 1

    def test_with_labels(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=64, modalities=["text", "image"],
                               image_size=32, patch_size=8)
        x = torch.randint(0, 100, (1, 8))
        img = torch.randn(1, 3, 32, 32)
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, images=img, labels=labels)
        assert out["loss"] is not None

    def test_generate(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=64, modalities=["text", "image"],
                               image_size=32, patch_size=8)
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        img = torch.randn(1, 3, 32, 32)
        gen = model.generate(x, images=img, max_new_tokens=5)
        assert gen.shape[1] == 8

    def test_count_params(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=32, modalities=["text", "image"])
        params = model.count_parameters()
        assert params["image_encoder"] > 0

    def test_backward(self):
        model = MultiModalLLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4,
                               max_len=64, modalities=["text", "image"],
                               image_size=32, patch_size=8)
        x = torch.randint(0, 100, (1, 8))
        img = torch.randn(1, 3, 32, 32)
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, images=img, labels=labels)
        out["loss"].backward()


class TestComponentCatalog:
    def test_catalog_has_all_types(self):
        expected = ["norm", "attention", "ffn", "moe", "residual", "dropout",
                    "linear", "activation", "custom_code", "custom_formula"]
        for t in expected:
            assert t in COMPONENT_CATALOG, f"Missing type: {t}"

    def test_catalog_entries_have_schema(self):
        for name, entry in COMPONENT_CATALOG.items():
            assert "name" in entry
            assert "description" in entry
            assert "config_schema" in entry


# ── Server Endpoint Tests ──

from fastapi.testclient import TestClient
from state_graph.server.app import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


class TestLLMBuildEndpoint:
    def test_build_basic(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "built"
        assert data["params"]["total"] > 0
        assert "architecture" in data

    def test_build_with_norm_type(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "norm_type": "layernorm",
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_build_with_ffn_type(self):
        for ffn_type in ["swiglu", "geglu", "reglu", "standard"]:
            resp = client.post("/api/llm/build", json={
                "vocab_size": 100, "d_model": 64, "n_layers": 2,
                "n_heads": 4, "max_len": 32, "ffn_type": ffn_type,
            })
            data = resp.json()
            assert data["status"] == "built", f"Failed for ffn_type={ffn_type}"

    def test_build_with_moe(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 4,
            "n_heads": 4, "max_len": 32, "use_moe": True,
            "n_experts": 4, "moe_top_k": 2,
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_build_gqa(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 8, "n_kv_heads": 2, "max_len": 32,
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_build_rope_base(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "rope_base": 500000.0,
        })
        data = resp.json()
        assert data["status"] == "built"


class TestLLMPresetsEndpoint:
    def test_get_presets(self):
        resp = client.get("/api/llm/presets")
        assert resp.status_code == 200
        data = resp.json()
        presets = data["presets"]
        assert "tiny" in presets
        assert "small" in presets
        assert "medium" in presets
        assert "large" in presets
        assert "mixtral_tiny" in presets
        assert "gqa_small" in presets
        assert "gpt2_style" in presets
        assert "gemma_tiny" in presets

    def test_gpt2_preset_has_correct_config(self):
        resp = client.get("/api/llm/presets")
        gpt2 = resp.json()["presets"]["gpt2_style"]
        assert gpt2["norm_type"] == "layernorm"
        assert gpt2["ffn_type"] == "standard"

    def test_gemma_preset_has_correct_config(self):
        resp = client.get("/api/llm/presets")
        gemma = resp.json()["presets"]["gemma_tiny"]
        assert gemma["ffn_type"] == "geglu"


class TestLLMModifyEndpoint:
    def _build_model(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 4,
            "n_heads": 4, "max_len": 32,
        })

    def test_add_layer(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "add_layer", "position": 2,
            "config": {"n_heads": 4, "ffn_type": "geglu"},
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert data["config"]["n_layers"] == 5

    def test_remove_layer(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "remove_layer", "index": 0,
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert data["config"]["n_layers"] == 3

    def test_remove_invalid_index(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "remove_layer", "index": 99,
        })
        data = resp.json()
        assert data["status"] == "error"

    def test_update_layer(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "update_layer", "index": 1,
            "config": {"ffn_type": "standard", "norm_type": "layernorm"},
        })
        data = resp.json()
        assert data["status"] == "modified"

    def test_freeze_layer(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "freeze_layer", "index": 0,
        })
        data = resp.json()
        assert data["status"] == "modified"
        # Verify frozen
        for p in engine.model.layers[0].parameters():
            assert not p.requires_grad

    def test_unfreeze_layer(self):
        self._build_model()
        # Freeze then unfreeze
        client.post("/api/llm/modify", json={"action": "freeze_layer", "index": 0})
        resp = client.post("/api/llm/modify", json={"action": "unfreeze_layer", "index": 0})
        data = resp.json()
        assert data["status"] == "modified"
        for p in engine.model.layers[0].parameters():
            assert p.requires_grad

    def test_freeze_component(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "freeze_layer", "component": "embedding",
        })
        data = resp.json()
        assert data["status"] == "modified"
        for p in engine.model.tok_emb.parameters():
            assert not p.requires_grad

    def test_change_vocab(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "change_vocab", "vocab_size": 200,
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert data["config"]["vocab_size"] == 200
        assert engine.model.vocab_size == 200

    def test_change_norm(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "change_norm", "norm_type": "layernorm",
        })
        data = resp.json()
        assert data["status"] == "modified"

    def test_reorder_layer(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "reorder_layer", "from_index": 0, "to_index": 3,
        })
        data = resp.json()
        assert data["status"] == "modified"

    def test_unknown_action(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={"action": "invalid"})
        data = resp.json()
        assert data["status"] == "error"

    def test_no_model_loaded(self):
        resp = client.post("/api/llm/modify", json={"action": "add_layer"})
        data = resp.json()
        assert data["status"] == "error"

    def test_cannot_remove_last_layer(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 1,
            "n_heads": 4, "max_len": 32,
        })
        resp = client.post("/api/llm/modify", json={
            "action": "remove_layer", "index": 0,
        })
        data = resp.json()
        assert data["status"] == "error"


class TestLLMModifyEdgeCases:
    def _build_model(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 4,
            "n_heads": 4, "max_len": 32,
        })

    def test_add_layer_at_position_0(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "add_layer", "position": 0,
            "config": {"ffn_type": "standard"},
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert data["config"]["n_layers"] == 5

    def test_add_layer_at_end(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "add_layer", "position": 999,  # beyond end
            "config": {"ffn_type": "reglu"},
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert data["config"]["n_layers"] == 5

    def test_freeze_and_unfreeze_lm_head(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "freeze_layer", "component": "lm_head",
        })
        assert resp.json()["status"] == "modified"
        for p in engine.model.lm_head.parameters():
            assert not p.requires_grad
        resp = client.post("/api/llm/modify", json={
            "action": "unfreeze_layer", "component": "lm_head",
        })
        assert resp.json()["status"] == "modified"
        for p in engine.model.lm_head.parameters():
            assert p.requires_grad

    def test_freeze_and_unfreeze_norm(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "freeze_layer", "component": "norm",
        })
        assert resp.json()["status"] == "modified"

    def test_freeze_invalid_component(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "freeze_layer", "component": "nonexistent",
        })
        assert resp.json()["status"] == "error"

    def test_change_vocab_smaller(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "change_vocab", "vocab_size": 50,
        })
        data = resp.json()
        assert data["status"] == "modified"
        assert engine.model.vocab_size == 50
        assert engine.model.tok_emb.num_embeddings == 50

    def test_change_vocab_invalid(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "change_vocab", "vocab_size": 0,
        })
        assert resp.json()["status"] == "error"

    def test_reorder_invalid_indices(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "reorder_layer", "from_index": -1, "to_index": 0,
        })
        assert resp.json()["status"] == "error"

    def test_update_layer_invalid_index(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "update_layer", "index": 99,
            "config": {"ffn_type": "geglu"},
        })
        assert resp.json()["status"] == "error"

    def test_modify_returns_architecture(self):
        self._build_model()
        resp = client.post("/api/llm/modify", json={
            "action": "add_layer", "position": 0,
        })
        data = resp.json()
        assert "architecture" in data
        assert len(data["architecture"]) > 0

    def test_sequential_modifications(self):
        self._build_model()
        # Add, update, freeze, remove in sequence
        client.post("/api/llm/modify", json={"action": "add_layer", "position": 0})
        assert len(engine.model.layers) == 5
        client.post("/api/llm/modify", json={"action": "update_layer", "index": 0, "config": {"ffn_type": "geglu"}})
        client.post("/api/llm/modify", json={"action": "freeze_layer", "index": 0})
        client.post("/api/llm/modify", json={"action": "remove_layer", "index": 0})
        assert len(engine.model.layers) == 4

    def test_model_still_works_after_modification(self):
        """Verify model forward pass works after modifications."""
        self._build_model()
        client.post("/api/llm/modify", json={
            "action": "add_layer", "position": 1,
            "config": {"ffn_type": "geglu", "norm_type": "layernorm"},
        })
        client.post("/api/llm/modify", json={
            "action": "update_layer", "index": 3,
            "config": {"ffn_type": "standard"},
        })
        # Model should still work
        x = torch.randint(0, 100, (1, 8))
        out = engine.model(x)
        assert out["logits"].shape == (1, 8, 100)


class TestLLMLayerEndpoint:
    def test_get_layer_info(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        resp = client.get("/api/llm/layer/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        info = data["layer"]
        assert info["index"] == 0
        assert info["n_heads"] == 4
        assert info["param_count"] > 0

    def test_get_layer_invalid_index(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        resp = client.get("/api/llm/layer/99")
        data = resp.json()
        assert data["status"] == "error"

    def test_get_layer_no_model(self):
        resp = client.get("/api/llm/layer/0")
        data = resp.json()
        assert data["status"] == "error"


class TestLLMLayerEndpointExtended:
    def test_layer_shows_frozen_status(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        client.post("/api/llm/modify", json={"action": "freeze_layer", "index": 0})
        resp = client.get("/api/llm/layer/0")
        data = resp.json()
        assert data["layer"]["frozen"] is True
        assert data["layer"]["trainable_params"] == 0

    def test_layer_shows_norm_and_ffn_type(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "norm_type": "layernorm", "ffn_type": "geglu",
        })
        resp = client.get("/api/llm/layer/0")
        data = resp.json()
        assert data["layer"]["norm_type"] == "layernorm"
        assert data["layer"]["ffn_type"] == "geglu"

    def test_layer_shows_moe_info(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "use_moe": True, "n_experts": 4, "moe_top_k": 2,
            "moe_layers": [0, 1],
        })
        resp = client.get("/api/llm/layer/0")
        data = resp.json()
        assert data["layer"]["has_moe"] is True
        assert data["layer"]["n_experts"] == 4
        assert data["layer"]["moe_top_k"] == 2


class TestLLMTrainEndpoint:
    def _build_and_get_text(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        # Generate enough text
        return "The quick brown fox jumps over the lazy dog. " * 50

    def test_train_char_tokenizer(self):
        text = self._build_and_get_text()
        resp = client.post("/api/llm/train", json={
            "text": text, "tokenizer": "char",
            "max_len": 32, "batch_size": 2, "epochs": 1,
            "learning_rate": 0.001,
        })
        data = resp.json()
        assert data["status"] == "started"
        assert data["n_train"] >= 1
        assert data["n_val"] >= 1

    def test_train_response_contains_stats(self):
        text = self._build_and_get_text()
        resp = client.post("/api/llm/train", json={
            "text": text, "tokenizer": "char",
            "max_len": 32, "batch_size": 2, "epochs": 1,
            "learning_rate": 0.001,
        })
        data = resp.json()
        assert "n_sequences" in data
        assert "n_train" in data
        assert "n_val" in data
        assert "seq_len" in data
        assert data["seq_len"] == 32

    def test_train_no_text(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        resp = client.post("/api/llm/train", json={
            "text": "", "tokenizer": "char",
        })
        data = resp.json()
        assert data["status"] == "error"

    def test_train_insufficient_data(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        # Very short text with long seq_len
        resp = client.post("/api/llm/train", json={
            "text": "ab", "tokenizer": "char",
            "max_len": 256,
        })
        data = resp.json()
        assert data["status"] == "error"
        assert "Not enough data" in data["message"]

    def test_train_boundary_data(self):
        """Test with exactly enough data for 2 sequences."""
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        # Need at least 2*max_len + 1 = 65 chars for 2 sequences
        text = "a" * 65
        resp = client.post("/api/llm/train", json={
            "text": text, "tokenizer": "char",
            "max_len": 32, "batch_size": 1, "epochs": 1,
        })
        data = resp.json()
        assert data["status"] == "started"
        assert data["n_sequences"] == 2

    def test_train_no_model(self):
        resp = client.post("/api/llm/train", json={
            "text": "hello world " * 100, "tokenizer": "char",
        })
        data = resp.json()
        assert data["status"] == "error"

    def test_train_with_short_seq_len(self):
        text = self._build_and_get_text()
        resp = client.post("/api/llm/train", json={
            "text": text, "tokenizer": "char",
            "max_len": 16, "batch_size": 4, "epochs": 1,
        })
        data = resp.json()
        assert data["status"] == "started"
        assert data["seq_len"] == 16


class TestLLMGenerateEndpoint:
    def test_generate_no_model(self):
        resp = client.post("/api/llm/generate", json={
            "prompt": "hello", "max_tokens": 5,
        })
        data = resp.json()
        assert data["status"] == "error"

    def test_generate_no_tokenizer(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        # Ensure no tokenizer is set
        if hasattr(engine, '_llm_tokenizer'):
            del engine._llm_tokenizer
        resp = client.post("/api/llm/generate", json={
            "prompt": "hello", "max_tokens": 5,
        })
        data = resp.json()
        assert data["status"] == "error"
        assert "tokenizer" in data["message"].lower()


class TestArchitectureVisualization:
    def test_architecture_with_new_norm(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "norm_type": "layernorm",
        })
        data = resp.json()
        arch = data["architecture"]
        # Check that LayerNorm appears in architecture
        block = arch[1]  # First decoder block
        assert "LayerNorm" in block["children"][0]["name"]

    def test_architecture_with_new_ffn(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "ffn_type": "geglu",
        })
        data = resp.json()
        arch = data["architecture"]
        block = arch[1]
        # FFN child should mention GeGLU
        ffn_child = block["children"][4]  # After norm, attn, residual, norm
        assert "GeGLU" in ffn_child["name"]

    def test_architecture_standard_ffn(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "ffn_type": "standard",
        })
        data = resp.json()
        arch = data["architecture"]
        block = arch[1]
        ffn_child = block["children"][4]
        assert "Standard" in ffn_child["name"]
        # Standard FFN has FC1, GELU, FC2
        assert len(ffn_child["children"]) == 3

    def test_architecture_reglu_ffn(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "ffn_type": "reglu",
        })
        data = resp.json()
        arch = data["architecture"]
        block = arch[1]
        ffn_child = block["children"][4]
        assert "ReGLU" in ffn_child["name"]

    def test_architecture_swiglu_ffn_children(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "ffn_type": "swiglu",
        })
        data = resp.json()
        block = data["architecture"][1]
        ffn_child = block["children"][4]
        assert "SwiGLU" in ffn_child["name"]
        # SwiGLU has Gate Proj, Up Proj, SiLU Gate, Down Proj
        assert len(ffn_child["children"]) == 4

    def test_architecture_moe_tree(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "use_moe": True, "n_experts": 4, "moe_top_k": 2,
            "moe_layers": [0, 1],
        })
        data = resp.json()
        block = data["architecture"][1]  # First decoder block
        ffn_child = block["children"][4]
        assert "MoE" in ffn_child["name"]
        assert "4 experts" in ffn_child["name"]
        # Router + 4 experts = 5 children
        assert len(ffn_child["children"]) == 5

    def test_architecture_frozen_label(self):
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        client.post("/api/llm/modify", json={
            "action": "freeze_layer", "index": 0,
        })
        # Get architecture via visualize endpoint
        resp = client.get("/api/architecture/visualize")
        data = resp.json()
        arch = data["architecture"]
        assert "[frozen]" in arch[1]["name"]

    def test_architecture_structure(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        data = resp.json()
        arch = data["architecture"]
        # Token Embedding + 2 decoder blocks + RMSNorm (final) + LM Head = 5
        assert len(arch) == 5
        assert arch[0]["name"] == "Token Embedding"
        assert "Decoder Block 0" in arch[1]["name"]
        assert "Decoder Block 1" in arch[2]["name"]
        assert "RMSNorm (final)" in arch[3]["name"]
        assert arch[4]["name"] == "LM Head"

    def test_architecture_attention_children(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        data = resp.json()
        block = data["architecture"][1]
        attn = block["children"][1]
        assert attn["name"] == "Attention"
        assert attn["params"]["n_heads"] == 4
        # Q, K, V, RoPE, SDPA, O = 6 children
        assert len(attn["children"]) == 6

    def test_architecture_gqa_kv_heads(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 8, "n_kv_heads": 2, "max_len": 32,
        })
        data = resp.json()
        block = data["architecture"][1]
        attn = block["children"][1]
        assert attn["params"]["n_heads"] == 8
        assert attn["params"]["n_kv_heads"] == 2

    def test_architecture_weight_tying_note(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "tie_weights": True,
        })
        data = resp.json()
        lm_head = data["architecture"][-1]
        assert "Weight-tied" in lm_head.get("note", "")

    def test_architecture_no_weight_tying_note(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "tie_weights": False,
        })
        data = resp.json()
        lm_head = data["architecture"][-1]
        assert "Weight-tied" not in lm_head.get("note", "")

    def test_architecture_param_counts(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        data = resp.json()
        # Embedding and blocks should have param_count
        assert data["architecture"][0]["param_count"] == 100 * 64
        assert data["architecture"][1]["param_count"] > 0


class TestLLMBuildEndpointExtended:
    def test_build_with_layer_configs(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 3,
            "n_heads": 4, "max_len": 32,
            "layer_configs": [
                {"ffn_type": "geglu", "norm_type": "layernorm"},
                None,
                {"ffn_type": "standard"},
            ],
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_build_no_weight_tying(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "tie_weights": False,
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_build_stores_config_on_engine(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32, "ffn_type": "geglu",
        })
        assert engine._llm_config["ffn_type"] == "geglu"
        assert engine._llm_config["d_model"] == 64
        assert engine.model_source == "llm"

    def test_build_broadcasts_model_type(self):
        resp = client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        data = resp.json()
        assert data["device"] in ["cpu", "cuda", "mps"]


class TestRegistryIntegration:
    def test_new_layers_registered(self):
        from state_graph.core.registry import Registry
        layers = Registry.list_layers()
        assert "ReGLUFFN" in layers
        assert "StandardFFN" in layers
        assert "SwiGLUFFN" in layers
        assert "GeGLUFFN" in layers
        assert "LLMModel" in layers

    def test_all_llm_layers_registered(self):
        from state_graph.core.registry import Registry
        layers = Registry.list_layers()
        expected = [
            "RMSNorm", "RotaryPositionalEmbedding", "LLMAttention",
            "SwiGLUFFN", "GeGLUFFN", "ReGLUFFN", "StandardFFN",
            "MoELayer", "LLMDecoderBlock", "LLMModel",
            "ComposableBlock", "ComposableLLM", "CustomFFN",
        ]
        for name in expected:
            assert name in layers, f"{name} not registered"

    def test_registered_layers_are_instantiable(self):
        from state_graph.core.registry import Registry
        cls = Registry.get_layer("RMSNorm")
        obj = cls(64)
        assert obj is not None
        cls = Registry.get_layer("SwiGLUFFN")
        obj = cls(64)
        assert obj is not None


# ── Composable Endpoint Tests ──

class TestComposeEndpoint:
    def test_compose_basic(self):
        resp = client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "default_block_name": "llama",
        })
        data = resp.json()
        assert data["status"] == "built"
        assert data["params"]["total"] > 0
        assert "architecture" in data

    def test_compose_gpt2_style(self):
        resp = client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "default_block_name": "gpt2",
        })
        assert resp.json()["status"] == "built"

    def test_compose_all_presets(self):
        for preset in ["llama", "gpt2", "palm", "gemma", "minimal", "deep_norm", "moe_block"]:
            resp = client.post("/api/llm/compose", json={
                "vocab_size": 50, "d_model": 64, "n_layers": 1,
                "n_heads": 4, "max_len": 32,
                "default_block_name": preset,
            })
            assert resp.json()["status"] == "built", f"Preset {preset} failed"

    def test_compose_custom_block(self):
        resp = client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "default_block": [
                {"type": "norm", "config": {"norm_type": "layernorm"}},
                {"type": "attention", "config": {}},
                {"type": "ffn", "config": {"ffn_type": "geglu"}},
            ],
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_compose_with_custom_formula(self):
        resp = client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 1,
            "n_heads": 4, "max_len": 32,
            "default_block": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "attention", "config": {}},
                {"type": "custom_formula", "config": {"formula": "self.fc2(F.gelu(self.fc1(x)))"}},
            ],
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_compose_sets_engine_state(self):
        client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "default_block_name": "llama",
        })
        assert engine.model is not None
        assert engine.model_source == "llm"
        assert engine._llm_config.get("composable") is True

    def test_compose_model_forward_works(self):
        client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
            "default_block_name": "llama",
        })
        x = torch.randint(0, 100, (1, 8))
        out = engine.model(x)
        assert out["logits"].shape == (1, 8, 100)


class TestComposeBlockEndpoint:
    def _build_composable(self):
        client.post("/api/llm/compose", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 3,
            "n_heads": 4, "max_len": 32,
            "default_block_name": "llama",
        })

    def test_update_single_block(self):
        self._build_composable()
        resp = client.post("/api/llm/compose/block/1", json={
            "steps": [
                {"type": "norm", "config": {"norm_type": "layernorm"}},
                {"type": "attention", "config": {}},
                {"type": "ffn", "config": {"ffn_type": "standard"}},
            ],
        })
        data = resp.json()
        assert data["status"] == "modified"

    def test_update_block_invalid_index(self):
        self._build_composable()
        resp = client.post("/api/llm/compose/block/99", json={
            "steps": [{"type": "attention", "config": {}}],
        })
        assert resp.json()["status"] == "error"

    def test_update_block_no_steps(self):
        self._build_composable()
        resp = client.post("/api/llm/compose/block/0", json={})
        assert resp.json()["status"] == "error"

    def test_update_block_model_still_works(self):
        self._build_composable()
        client.post("/api/llm/compose/block/0", json={
            "steps": [
                {"type": "attention", "config": {}},
                {"type": "ffn", "config": {"ffn_type": "geglu"}},
            ],
        })
        x = torch.randint(0, 100, (1, 8))
        out = engine.model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_not_composable_model(self):
        # Build a regular LLM first
        client.post("/api/llm/build", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 32,
        })
        resp = client.post("/api/llm/compose/block/0", json={
            "steps": [{"type": "attention", "config": {}}],
        })
        assert resp.json()["status"] == "error"


class TestComponentsEndpoint:
    def test_get_components(self):
        resp = client.get("/api/llm/components")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "block_designs" in data
        assert "llama" in data["block_designs"]
        assert "norm" in data["components"]
        assert "attention" in data["components"]
        assert "custom_code" in data["components"]

    def test_components_have_schemas(self):
        resp = client.get("/api/llm/components")
        for name, comp in resp.json()["components"].items():
            assert "name" in comp, f"{name} missing name"
            assert "description" in comp, f"{name} missing description"
            assert "config_schema" in comp, f"{name} missing config_schema"


class TestValidateEndpoint:
    def test_validate_custom_code(self):
        code = """
class TestModule(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.linear(x)
"""
        resp = client.post("/api/llm/component/validate", json={
            "type": "custom_code", "code": code, "d_model": 64,
        })
        data = resp.json()
        assert data["status"] == "valid"
        assert data["output_shape"] == [1, 4, 64]
        assert data["params"] > 0

    def test_validate_invalid_code(self):
        resp = client.post("/api/llm/component/validate", json={
            "type": "custom_code", "code": "x = 5", "d_model": 64,
        })
        assert resp.json()["status"] == "error"

    def test_validate_custom_formula(self):
        resp = client.post("/api/llm/component/validate", json={
            "type": "custom_formula",
            "formula": "self.fc2(F.gelu(self.fc1(x)))",
            "d_model": 64,
        })
        data = resp.json()
        assert data["status"] == "valid"

    def test_validate_bad_formula(self):
        resp = client.post("/api/llm/component/validate", json={
            "type": "custom_formula",
            "formula": "self.nonexistent(x)",
            "d_model": 64,
        })
        assert resp.json()["status"] == "error"

    def test_validate_unknown_type(self):
        resp = client.post("/api/llm/component/validate", json={
            "type": "invalid_type",
        })
        assert resp.json()["status"] == "error"


# ── Feature Endpoint Tests ──

class TestEncoderDecoderEndpoint:
    def test_build(self):
        resp = client.post("/api/llm/encoder-decoder", json={
            "vocab_size": 100, "d_model": 64, "n_encoder_layers": 2,
            "n_decoder_layers": 2, "n_heads": 4, "max_len": 32,
        })
        data = resp.json()
        assert data["status"] == "built"
        assert data["params"]["encoder_params"] > 0
        assert data["params"]["decoder_params"] > 0

    def test_model_works(self):
        client.post("/api/llm/encoder-decoder", json={
            "vocab_size": 100, "d_model": 64, "n_encoder_layers": 2,
            "n_decoder_layers": 2, "n_heads": 4, "max_len": 32,
        })
        src = torch.randint(0, 100, (1, 8))
        tgt = torch.randint(0, 100, (1, 6))
        out = engine.model(src, decoder_input_ids=tgt)
        assert out["logits"].shape == (1, 6, 100)


class TestAdaptiveDepthEndpoint:
    def test_build(self):
        resp = client.post("/api/llm/adaptive-depth", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 6,
            "n_heads": 4, "max_len": 32, "exit_interval": 2,
        })
        data = resp.json()
        assert data["status"] == "built"

    def test_model_works(self):
        client.post("/api/llm/adaptive-depth", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 4,
            "n_heads": 4, "max_len": 32, "exit_interval": 2,
        })
        x = torch.randint(0, 100, (1, 8))
        out = engine.model(x)
        assert "exited_at" in out


class TestMultiModalEndpoint:
    def test_build_text_image(self):
        resp = client.post("/api/llm/multimodal", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 64, "modalities": ["text", "image"],
            "image_size": 32, "patch_size": 8,
        })
        data = resp.json()
        assert data["status"] == "built"
        assert data["params"]["image_encoder"] > 0

    def test_build_all_modalities(self):
        resp = client.post("/api/llm/multimodal", json={
            "vocab_size": 100, "d_model": 64, "n_layers": 2,
            "n_heads": 4, "max_len": 128, "modalities": ["text", "image", "audio"],
            "image_size": 32, "patch_size": 8, "n_mels": 40,
        })
        assert resp.json()["status"] == "built"


class TestTokenizerTrainEndpoint:
    def test_train_bpe(self):
        text = "the quick brown fox " * 100
        resp = client.post("/api/llm/tokenizer/train", json={
            "text": text, "algorithm": "bpe", "vocab_size": 500,
        })
        data = resp.json()
        assert data["status"] == "trained"
        assert data["vocab_size"] > 0

    def test_train_char(self):
        text = "hello world test " * 50
        resp = client.post("/api/llm/tokenizer/train", json={
            "text": text, "algorithm": "char", "vocab_size": 100,
        })
        data = resp.json()
        assert data["status"] == "trained"

    def test_train_too_short(self):
        resp = client.post("/api/llm/tokenizer/train", json={
            "text": "hi", "algorithm": "bpe",
        })
        assert resp.json()["status"] == "error"

    def test_stored_on_engine(self):
        text = "hello world test data " * 50
        client.post("/api/llm/tokenizer/train", json={
            "text": text, "algorithm": "char", "vocab_size": 100,
        })
        tok = getattr(engine, '_llm_tokenizer', None)
        assert tok is not None
        assert tok["type"] == "custom_trained"
