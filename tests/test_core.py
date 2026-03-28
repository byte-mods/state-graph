"""Tests for StateGraph core modules."""

import torch
import torch.nn as nn
import pytest

from state_graph.core.registry import Registry
from state_graph.core.graph import StateGraph
from state_graph.core.engine import TrainingEngine
from state_graph.core.metrics import MetricsCollector
from state_graph.core.scheduler import SchedulerRegistry
from state_graph.core.data import DataManager


# ── Registry ──

class TestRegistry:
    def test_list_layers(self):
        layers = Registry.list_layers()
        assert "Linear" in layers
        assert "Conv2d" in layers

    def test_list_activations(self):
        acts = Registry.list_activations()
        assert "ReLU" in acts
        assert "GELU" in acts

    def test_list_losses(self):
        losses = Registry.list_losses()
        assert "CrossEntropyLoss" in losses

    def test_list_optimizers(self):
        opts = Registry.list_optimizers()
        assert "Adam" in opts

    def test_get_layer(self):
        cls = Registry.get_layer("Linear")
        assert cls is nn.Linear

    def test_register_formula(self):
        Registry.register_formula_from_string("TestAct", "x * torch.sigmoid(x)")
        assert "TestAct" in Registry.list_activations()
        cls = Registry.get_activation("TestAct")
        model = cls()
        out = model(torch.randn(4, 8))
        assert out.shape == (4, 8)

    def test_list_all(self):
        result = Registry.list_all()
        assert "layers" in result
        assert "activations" in result
        assert "losses" in result
        assert "optimizers" in result


# ── Graph ──

class TestStateGraph:
    def test_add_remove_layer(self):
        g = StateGraph()
        nid = g.add_layer("Linear", {"in_features": 10, "out_features": 5})
        assert nid in g.nodes
        g.remove_layer(nid)
        assert nid not in g.nodes

    def test_build_model(self):
        g = StateGraph()
        g.add_layer("Linear", {"in_features": 10, "out_features": 5}, activation="ReLU")
        g.add_layer("Linear", {"in_features": 5, "out_features": 2})
        model = g.build_model()
        assert isinstance(model, nn.Sequential)
        out = model(torch.randn(4, 10))
        assert out.shape == (4, 2)

    def test_reorder_layer(self):
        g = StateGraph()
        id1 = g.add_layer("Linear", {"in_features": 10, "out_features": 5})
        id2 = g.add_layer("Linear", {"in_features": 5, "out_features": 2})
        assert g.nodes[id1].position == 0
        assert g.nodes[id2].position == 1
        g.reorder_layer(id2, 0)
        assert g.nodes[id2].position == 0
        assert g.nodes[id1].position == 1

    def test_update_layer(self):
        g = StateGraph()
        nid = g.add_layer("Linear", {"in_features": 10, "out_features": 5})
        g.update_layer(nid, params={"in_features": 20, "out_features": 10})
        assert g.nodes[nid].params["in_features"] == 20

    def test_to_dict(self):
        g = StateGraph()
        g.add_layer("Linear", {"in_features": 10, "out_features": 5})
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 1

    def test_param_count(self):
        g = StateGraph()
        g.add_layer("Linear", {"in_features": 10, "out_features": 5})
        counts = g.get_param_count()
        assert len(counts) == 1
        info = list(counts.values())[0]
        assert info["total"] == 55  # 10*5 + 5 bias
        assert info["trainable"] == 55


# ── Custom Layers ──

class TestCustomLayers:
    def test_residual_block(self):
        from state_graph.layers.custom import ResidualBlock
        block = ResidualBlock(64)
        x = torch.randn(8, 64)
        out = block(x)
        assert out.shape == (8, 64)

    def test_gated_linear_unit(self):
        from state_graph.layers.custom import GatedLinearUnit
        glu = GatedLinearUnit(32, 16)
        out = glu(torch.randn(8, 32))
        assert out.shape == (8, 16)

    def test_transformer_block(self):
        from state_graph.layers.custom import TransformerBlock
        block = TransformerBlock(d_model=64, n_heads=4)
        x = torch.randn(8, 10, 64)  # batch, seq, dim
        out = block(x)
        assert out.shape == (8, 10, 64)

    def test_positional_encoding(self):
        from state_graph.layers.custom import PositionalEncoding
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(8, 10, 64)
        out = pe(x)
        assert out.shape == (8, 10, 64)

    def test_token_embedding(self):
        from state_graph.layers.custom import TokenEmbedding
        embed = TokenEmbedding(in_features=2, d_model=64, seq_len=4)
        x = torch.randn(8, 2)
        out = embed(x)
        assert out.shape == (8, 4, 64)

    def test_sequence_pool(self):
        from state_graph.layers.custom import SequencePool
        pool = SequencePool(d_model=64, mode="mean")
        x = torch.randn(8, 10, 64)
        out = pool(x)
        assert out.shape == (8, 64)

    def test_transformer_sequential_pipeline(self):
        """Full transformer pipeline in nn.Sequential."""
        from state_graph.layers.custom import (
            TokenEmbedding, PositionalEncoding, TransformerBlock, SequencePool,
        )
        model = nn.Sequential(
            TokenEmbedding(2, 64, 4),
            PositionalEncoding(64),
            TransformerBlock(64, 4),
            SequencePool(64),
            nn.Linear(64, 2),
        )
        out = model(torch.randn(8, 2))
        assert out.shape == (8, 2)

    def test_custom_layers_registered(self):
        layers = Registry.list_layers()
        for name in ["ResidualBlock", "GatedLinearUnit", "SwishLinear",
                      "TransformerBlock", "PositionalEncoding", "TokenEmbedding", "SequencePool"]:
            assert name in layers, f"{name} not registered"


# ── Scheduler ──

class TestScheduler:
    def test_list_all(self):
        schedulers = SchedulerRegistry.list_all()
        assert "StepLR" in schedulers
        assert "CosineAnnealingLR" in schedulers
        assert len(schedulers) == 10

    def test_get_defaults(self):
        defaults = SchedulerRegistry.get_default_params("StepLR")
        assert "step_size" in defaults
        assert "gamma" in defaults

    def test_create(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters())
        sched = SchedulerRegistry.create("StepLR", opt)
        assert sched is not None
        sched.step()  # should not raise


# ── DataManager ──

class TestDataManager:
    def test_load_xor(self):
        dm = DataManager()
        result = dm.load_builtin("xor", 200)
        assert result["status"] == "loaded"
        assert result["n_classes"] == 2
        assert dm.x_train is not None

    def test_load_spiral(self):
        dm = DataManager()
        result = dm.load_builtin("spiral", 200)
        assert result["status"] == "loaded"

    def test_load_all_synthetic(self):
        dm = DataManager()
        for name in ["random", "xor", "spiral", "circles", "moons", "blobs", "checkerboard", "regression_sin"]:
            result = dm.load_builtin(name, 100)
            assert result["status"] == "loaded", f"Failed to load {name}"

    def test_get_info(self):
        dm = DataManager()
        dm.load_builtin("xor", 100)
        info = dm.get_info()
        assert info["dataset"] == "xor"
        assert info["n_train"] > 0


# ── Metrics ──

class TestMetrics:
    def test_collect_step(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        opt = torch.optim.Adam(model.parameters())
        mc = MetricsCollector()
        mc.attach_hooks(model)

        # Forward + backward to populate grads
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        loss.backward()

        step_data = mc.collect_step(model, loss.item(), opt)
        assert "step" in step_data
        assert "loss" in step_data
        assert "layers" in step_data

    def test_collect_epoch(self):
        mc = MetricsCollector()
        data = mc.collect_epoch(0, 0.5, 0.6, 0.8, 0.75)
        assert data["epoch"] == 0
        assert data["train_loss"] == 0.5

    def test_reset(self):
        mc = MetricsCollector()
        mc.collect_epoch(0, 0.5)
        mc.reset()
        assert mc._step == 0
        assert len(mc._epoch_metrics) == 0

    def test_snapshot(self):
        mc = MetricsCollector()
        snap = mc.get_snapshot()
        assert "step" in snap
        assert "loss_history" in snap


# ── Engine ──

class TestEngine:
    def test_init(self):
        e = TrainingEngine()
        assert e.model_source == "graph"
        assert e.model is None

    def test_build_graph(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 16}, "ReLU")
        e.graph.add_layer("Linear", {"in_features": 16, "out_features": 2})
        result = e.build()
        assert result["status"] == "built"
        assert result["total_params"] > 0
        assert e.model is not None

    def test_build_with_scheduler(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 2})
        e.config["scheduler"] = "StepLR"
        e.config["scheduler_params"] = {"step_size": 5}
        result = e.build()
        assert result["status"] == "built"
        assert e.scheduler is not None

    def test_set_data(self):
        e = TrainingEngine()
        e.set_data(torch.randn(100, 2), torch.randint(0, 2, (100,)))
        assert e._train_loader is not None

    def test_export_architecture(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 2})
        arch = e.export_architecture()
        assert "version" in arch
        assert "graph" in arch
        assert len(arch["graph"]["nodes"]) == 1

    def test_import_architecture(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 2})
        arch = e.export_architecture()

        e2 = TrainingEngine()
        result = e2.import_architecture(arch)
        assert result["status"] == "imported"
        assert len(e2.graph.nodes) == 1

    def test_export_python_standard_layers(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 16}, "ReLU")
        e.graph.add_layer("Linear", {"in_features": 16, "out_features": 2})
        code = e.export_python()
        assert "nn.Linear" in code
        assert "nn.ReLU" in code
        assert "state_graph" not in code  # No custom imports needed

    def test_export_python_custom_layers(self):
        e = TrainingEngine()
        e.graph.add_layer("TransformerBlock", {"d_model": 64, "n_heads": 4})
        code = e.export_python()
        assert "from state_graph.layers.custom import TransformerBlock" in code
        assert "TransformerBlock(d_model=64" in code
        assert "nn.TransformerBlock" not in code  # Should NOT use nn. prefix

    def test_reset(self):
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 2})
        e.build()
        result = e.reset()
        assert result["status"] == "reset"
        assert len(e.graph.nodes) == 0
        assert e.model is None
        assert e.model_source == "graph"

    def test_get_status(self):
        e = TrainingEngine()
        status = e.get_status()
        assert "is_training" in status
        assert "config" in status
        assert "model_source" in status

    def test_train_xor(self):
        """Integration test: train a small model on XOR."""
        e = TrainingEngine()
        e.graph.add_layer("Linear", {"in_features": 2, "out_features": 16}, "ReLU")
        e.graph.add_layer("Linear", {"in_features": 16, "out_features": 2})
        e.config["epochs"] = 2
        e.config["batch_size"] = 32

        # Load XOR data
        dm = DataManager()
        dm.load_builtin("xor", 200)
        e.set_data(dm.x_train, dm.y_train, dm.x_val, dm.y_val)

        e.build()
        result = e.start_training()
        assert result["status"] == "started"

        # Wait for completion
        import time
        for _ in range(50):
            if not e._is_training:
                break
            time.sleep(0.1)

        assert not e._is_training
        assert len(e.metrics.get_epoch_metrics()) == 2
