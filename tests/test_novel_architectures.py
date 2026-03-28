"""Tests for novel architecture primitives: Mamba, RWKV, RetNet, Hyena, xLSTM, Griffin."""

import torch
import torch.nn as nn
import pytest

from state_graph.layers.custom import (
    SelectiveScan, MambaBlock,
    RWKVBlock,
    RetentionLayer, RetNetBlock,
    HyenaOperator, HyenaBlock,
    SLSTMCell, XLSTM,
    GatedLinearRecurrence,
)
from state_graph.layers.llm import ComposableBlock, ComposableLLM, BLOCK_DESIGNS
from state_graph.core.registry import Registry


class TestSelectiveScan:
    def test_shape(self):
        ssm = SelectiveScan(64, d_state=16, expand=2)
        x = torch.randn(2, 16, 64)
        out = ssm(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        ssm = SelectiveScan(32, d_state=8, expand=2)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = ssm(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_d_state(self):
        for d_state in [4, 8, 16, 32]:
            ssm = SelectiveScan(64, d_state=d_state)
            x = torch.randn(1, 8, 64)
            out = ssm(x)
            assert out.shape == (1, 8, 64)


class TestMambaBlock:
    def test_shape(self):
        block = MambaBlock(64, d_state=16)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_residual(self):
        block = MambaBlock(64)
        x = torch.randn(1, 8, 64)
        out = block(x)
        # Should not be identical (residual + transformation)
        assert not torch.allclose(out, x, atol=1e-3)

    def test_gradient(self):
        block = MambaBlock(32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestRWKVBlock:
    def test_shape(self):
        block = RWKVBlock(64)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        block = RWKVBlock(32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestRetentionLayer:
    def test_shape(self):
        ret = RetentionLayer(64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out = ret(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        ret = RetentionLayer(32, n_heads=4)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = ret(x)
        out.sum().backward()
        assert x.grad is not None


class TestRetNetBlock:
    def test_shape(self):
        block = RetNetBlock(64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        block = RetNetBlock(32, n_heads=4)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestHyenaOperator:
    def test_shape(self):
        hyena = HyenaOperator(64, max_len=32, order=2)
        x = torch.randn(2, 16, 64)
        out = hyena(x)
        assert out.shape == (2, 16, 64)

    def test_different_orders(self):
        for order in [1, 2, 3]:
            hyena = HyenaOperator(32, max_len=16, order=order)
            x = torch.randn(1, 8, 32)
            out = hyena(x)
            assert out.shape == (1, 8, 32)

    def test_gradient(self):
        hyena = HyenaOperator(32, max_len=16)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = hyena(x)
        out.sum().backward()
        assert x.grad is not None


class TestHyenaBlock:
    def test_shape(self):
        block = HyenaBlock(64, max_len=32)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)


class TestSLSTMCell:
    def test_output(self):
        cell = SLSTMCell(32, 64)
        x = torch.randn(2, 32)
        h, state = cell(x)
        assert h.shape == (2, 64)
        assert len(state) == 3  # h, c, n

    def test_sequential(self):
        cell = SLSTMCell(32, 64)
        x1 = torch.randn(2, 32)
        x2 = torch.randn(2, 32)
        h1, state1 = cell(x1)
        h2, state2 = cell(x2, state1)
        assert h2.shape == (2, 64)


class TestXLSTM:
    def test_shape(self):
        xlstm = XLSTM(64, n_layers=2)
        x = torch.randn(2, 16, 64)
        out = xlstm(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        xlstm = XLSTM(32, n_layers=1)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = xlstm(x)
        out.sum().backward()
        assert x.grad is not None


class TestGatedLinearRecurrence:
    def test_shape(self):
        glr = GatedLinearRecurrence(64, expand=2)
        x = torch.randn(2, 16, 64)
        out = glr(x)
        assert out.shape == (2, 16, 64)

    def test_gradient(self):
        glr = GatedLinearRecurrence(32)
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = glr(x)
        out.sum().backward()
        assert x.grad is not None


class TestRegistration:
    def test_all_registered(self):
        layers = Registry.list_layers()
        expected = [
            "SelectiveScan", "MambaBlock", "RWKVBlock",
            "RetentionLayer", "RetNetBlock",
            "HyenaOperator", "HyenaBlock",
            "XLSTM", "GatedLinearRecurrence",
        ]
        for name in expected:
            assert name in layers, f"{name} not registered"


class TestBlockDesignPresets:
    def test_mamba_preset(self):
        assert "mamba" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["mamba"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_rwkv_preset(self):
        assert "rwkv" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["rwkv"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_retnet_preset(self):
        assert "retnet" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["retnet"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_hyena_preset(self):
        assert "hyena" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["hyena"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_xlstm_preset(self):
        assert "xlstm" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["xlstm"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_griffin_preset(self):
        assert "griffin" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["griffin"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)

    def test_hybrid_mamba_attn_preset(self):
        assert "hybrid_mamba_attn" in BLOCK_DESIGNS
        block = ComposableBlock(d_model=64, steps=BLOCK_DESIGNS["hybrid_mamba_attn"], n_heads=4, max_len=32)
        x = torch.randn(1, 8, 64)
        out = block(x)
        assert out.shape == (1, 8, 64)


class TestFullModelsWithNovelArchitectures:
    def test_mamba_llm(self):
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
            default_block=BLOCK_DESIGNS["mamba"],
        )
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["logits"].shape == (1, 8, 100)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_retnet_llm(self):
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
            default_block=BLOCK_DESIGNS["retnet"],
        )
        x = torch.randint(0, 100, (1, 8))
        out = model(x)
        assert out["logits"].shape == (1, 8, 100)

    def test_hybrid_llm(self):
        """Mix different architectures per layer — Jamba-style."""
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=3, n_heads=4, max_len=32,
            block_designs=[
                BLOCK_DESIGNS["mamba"],     # Layer 0: Mamba
                BLOCK_DESIGNS["llama"],     # Layer 1: Transformer
                BLOCK_DESIGNS["mamba"],     # Layer 2: Mamba
            ],
        )
        x = torch.randint(0, 100, (1, 8))
        labels = torch.randint(0, 100, (1, 8))
        out = model(x, labels=labels)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_generate_mamba(self):
        model = ComposableLLM(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4, max_len=32,
            default_block=BLOCK_DESIGNS["mamba"],
        )
        model.eval()
        x = torch.randint(0, 100, (1, 3))
        gen = model.generate(x, max_new_tokens=5)
        assert gen.shape == (1, 8)
