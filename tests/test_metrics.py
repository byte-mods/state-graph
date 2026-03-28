"""Tests for extended metrics, profiling, and bottleneck detection."""

import torch
import torch.nn as nn
import pytest
import time

from state_graph.core.metrics import MetricsCollector, LayerMetrics


class TestLayerMetrics:
    def test_extended_fields(self):
        m = LayerMetrics()
        d = m.to_dict()
        assert "activation_min" in d
        assert "activation_max" in d
        assert "dead_neuron_pct" in d
        assert "forward_time_ms" in d
        assert "memory_mb" in d
        assert "grad_to_weight_ratio" in d


class TestMetricsCollectorExtended:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def test_timing_hooks(self):
        model = self._make_model()
        mc = MetricsCollector()
        mc.attach_hooks(model)

        x = torch.randn(2, 4)
        model(x)

        # Timing should be recorded for linear layers
        assert any("_start" not in k and mc._timing_cache.get(k, 0) > 0
                    for k in mc._timing_cache)
        mc.detach_hooks()

    def test_activation_distribution(self):
        model = self._make_model()
        mc = MetricsCollector()
        mc.attach_hooks(model)

        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()

        mc.mark_step_start()
        data = mc.collect_step(model, loss.item(), opt)

        # Check extended activation metrics exist
        for name, metrics in data["layers"].items():
            if "activation_min" in metrics:
                # Should have real values
                assert isinstance(metrics["activation_min"], float)
                assert isinstance(metrics["activation_max"], float)
                assert isinstance(metrics["activation_abs_max"], float)

        mc.detach_hooks()

    def test_dead_neuron_detection(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        mc = MetricsCollector()
        mc.attach_hooks(model)

        # Use input that makes many ReLU outputs zero
        x = torch.randn(2, 4) * 0.001
        out = model(x)
        loss = out.sum()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()

        mc.mark_step_start()
        data = mc.collect_step(model, loss.item(), opt)

        # dead_neuron_pct should be present
        found_dead = False
        for name, metrics in data["layers"].items():
            if "dead_neuron_pct" in metrics:
                found_dead = True
                assert isinstance(metrics["dead_neuron_pct"], float)
        assert found_dead
        mc.detach_hooks()

    def test_grad_to_weight_ratio(self):
        model = nn.Linear(4, 2)
        mc = MetricsCollector()

        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()

        data = mc.collect_step(model, loss.item(), opt)
        for name, metrics in data["layers"].items():
            assert "grad_to_weight_ratio" in metrics
            assert metrics["grad_to_weight_ratio"] >= 0

    def test_throughput_metrics(self):
        model = nn.Linear(4, 2)
        mc = MetricsCollector()

        x = torch.randn(2, 4)
        out = model(x)
        loss = out.sum()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss.backward()

        mc.mark_step_start()
        time.sleep(0.01)  # Ensure measurable time
        data = mc.collect_step(model, loss.item(), opt)

        assert "throughput" in data
        assert data["throughput"]["step_time_ms"] > 0
        assert data["throughput"]["steps_per_sec"] > 0

    def test_token_throughput(self):
        model = nn.Linear(4, 2)
        mc = MetricsCollector()

        mc.add_tokens(1000)
        mc.mark_step_start()
        time.sleep(0.01)

        x = torch.randn(2, 4)
        out = model(x)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        out.sum().backward()

        data = mc.collect_step(model, 0.5, opt)
        assert "tokens_per_sec" in data["throughput"]
        assert data["throughput"]["tokens_per_sec"] > 0

    def test_memory_snapshot(self):
        mc = MetricsCollector()
        mem = mc._get_memory_snapshot()
        assert "process_ram_mb" in mem
        assert mem["process_ram_mb"] > 0

    def test_collect_step_returns_memory(self):
        model = nn.Linear(4, 2)
        mc = MetricsCollector()
        x = torch.randn(2, 4)
        out = model(x)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        out.sum().backward()

        data = mc.collect_step(model, 0.5, opt)
        assert "memory" in data

    def test_collect_step_returns_bottleneck(self):
        model = self._make_model()
        mc = MetricsCollector()
        mc.attach_hooks(model)

        x = torch.randn(2, 4)
        out = model(x)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        out.sum().backward()

        mc.mark_step_start()
        data = mc.collect_step(model, 0.5, opt)
        assert "bottleneck" in data
        assert "issues" in data["bottleneck"]
        mc.detach_hooks()


class TestBottleneckDetection:
    def test_vanishing_gradient(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "layer0", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 1e-9, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
        ]
        result = mc._analyze_bottlenecks(candidates)
        issues = [i for i in result["issues"] if i["type"] == "vanishing_gradient"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "critical"

    def test_exploding_gradient(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "layer0", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 500.0, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
        ]
        result = mc._analyze_bottlenecks(candidates)
        issues = [i for i in result["issues"] if i["type"] == "exploding_gradient"]
        assert len(issues) == 1

    def test_dead_neurons_detected(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "relu_layer", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 0.1, "dead_neuron_pct": 80.0, "grad_to_weight_ratio": 0.001},
        ]
        result = mc._analyze_bottlenecks(candidates)
        issues = [i for i in result["issues"] if i["type"] == "dead_neurons"]
        assert len(issues) == 1

    def test_slow_layer_detected(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "fast", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 0.1, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
            {"name": "slow", "forward_time_ms": 50.0, "memory_mb": 1.0,
             "grad_norm": 0.1, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
        ]
        result = mc._analyze_bottlenecks(candidates)
        assert result["slowest_layer"] == "slow"
        issues = [i for i in result["issues"] if i["type"] == "slow_layer"]
        assert len(issues) == 1

    def test_memory_hog_detected(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "small", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 0.1, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
            {"name": "big", "forward_time_ms": 1.0, "memory_mb": 100.0,
             "grad_norm": 0.1, "dead_neuron_pct": 0, "grad_to_weight_ratio": 0.001},
        ]
        result = mc._analyze_bottlenecks(candidates)
        assert result["largest_memory"] == "big"

    def test_high_grad_ratio(self):
        mc = MetricsCollector()
        candidates = [
            {"name": "layer0", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 0.1, "dead_neuron_pct": 0, "grad_to_weight_ratio": 5.0},
        ]
        result = mc._analyze_bottlenecks(candidates)
        issues = [i for i in result["issues"] if i["type"] == "high_grad_ratio"]
        assert len(issues) == 1

    def test_no_issues_healthy_model(self):
        mc = MetricsCollector()
        # 4 balanced layers — none should trigger "slow_layer" (>30%) or "memory_hog" (>40%)
        candidates = [
            {"name": f"layer{i}", "forward_time_ms": 1.0, "memory_mb": 1.0,
             "grad_norm": 0.01, "dead_neuron_pct": 5.0, "grad_to_weight_ratio": 0.005}
            for i in range(4)
        ]
        result = mc._analyze_bottlenecks(candidates)
        assert len(result["issues"]) == 0

    def test_empty_candidates(self):
        mc = MetricsCollector()
        result = mc._analyze_bottlenecks([])
        assert result == {}


class TestModelProfiling:
    def test_profile_sequential(self):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        mc = MetricsCollector()
        result = mc.profile_model(model, (1, 4), n_runs=5)

        assert result["total_forward_ms"] > 0
        assert result["total_params"] > 0
        assert len(result["layers"]) > 0

        # Layers should have timing and memory
        for layer in result["layers"]:
            assert "avg_time_ms" in layer
            assert "params" in layer
            assert "memory_mb" in layer
            assert "time_pct" in layer
            assert "type" in layer

    def test_profile_conv_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        mc = MetricsCollector()
        result = mc.profile_model(model, (1, 3, 8, 8), n_runs=5)
        assert result["total_forward_ms"] > 0
        assert len(result["layers"]) > 0

    def test_profile_sorted_by_time(self):
        model = nn.Sequential(
            nn.Linear(4, 128),
            nn.Linear(128, 4),
        )
        mc = MetricsCollector()
        result = mc.profile_model(model, (1, 4), n_runs=5)
        times = [l["avg_time_ms"] for l in result["layers"]]
        assert times == sorted(times, reverse=True)  # Slowest first


# Server endpoint tests

from fastapi.testclient import TestClient
from state_graph.server.app import app, engine

@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()

client = TestClient(app)


class TestProfileEndpoint:
    def test_profile_no_model(self):
        resp = client.post("/api/profile", json={"input_shape": [1, 4]})
        assert resp.json()["status"] == "error"

    def test_profile_no_shape(self):
        # Build a model first
        client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 4, "out_features": 8}
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 2}
        })
        client.post("/api/build")
        resp = client.post("/api/profile", json={})
        assert resp.json()["status"] == "error"

    def test_profile_works(self):
        client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 4, "out_features": 8}
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 2}
        })
        client.post("/api/build")

        resp = client.post("/api/profile", json={"input_shape": [1, 4], "n_runs": 3})
        data = resp.json()
        assert data["status"] == "ok"
        assert data["total_forward_ms"] > 0
        assert len(data["layers"]) > 0


class TestBottleneckEndpoint:
    def test_bottleneck_no_data(self):
        resp = client.get("/api/bottleneck")
        assert resp.json()["status"] == "no_data"


class TestMetricsSnapshotEndpoint:
    def test_snapshot(self):
        resp = client.get("/api/metrics/snapshot")
        data = resp.json()
        assert "step" in data
        assert "loss_history" in data
        assert "layer_names" in data


class TestLayerMetricsEndpoint:
    def test_layer_metrics_empty(self):
        resp = client.get("/api/metrics/layer/nonexistent")
        data = resp.json()
        assert data["layer"] == "nonexistent"
        assert data["history"] == []
