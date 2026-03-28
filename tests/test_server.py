"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient

from state_graph.server.app import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset engine state before each test."""
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


class TestBasicEndpoints:
    def test_index(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "StateGraph" in resp.text

    def test_registry(self):
        resp = client.get("/api/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "layers" in data
        assert "activations" in data
        assert "schedulers" in data
        assert "scheduler_defaults" in data

    def test_status(self):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data
        assert "model_source" in data

    def test_templates(self):
        resp = client.get("/api/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "mlp_classifier" in data
        assert "transformer_classifier" in data


class TestGraphAPI:
    def test_add_layer(self):
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 5},
            "activation": "ReLU",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data
        assert len(data["graph"]["nodes"]) == 1

    def test_remove_layer(self):
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 10, "out_features": 5},
        })
        node_id = resp.json()["node_id"]
        resp = client.delete(f"/api/graph/layer/{node_id}")
        assert resp.status_code == 200
        assert len(resp.json()["graph"]["nodes"]) == 0

    def test_update_layer(self):
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 10, "out_features": 5},
        })
        node_id = resp.json()["node_id"]
        resp = client.put(f"/api/graph/layer/{node_id}", json={
            "params": {"in_features": 20, "out_features": 10},
        })
        assert resp.status_code == 200

    def test_reorder_layer(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 10, "out_features": 5}})
        resp2 = client.post("/api/graph/layer", json={"layer_type": "Dropout", "params": {"p": 0.5}})
        node_id = resp2.json()["node_id"]
        resp = client.post(f"/api/graph/layer/{node_id}/reorder", json={"position": 0})
        assert resp.status_code == 200

    def test_get_graph(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 10, "out_features": 5}})
        resp = client.get("/api/graph")
        assert resp.status_code == 200
        assert len(resp.json()["nodes"]) == 1


class TestConfig:
    def test_update_config(self):
        resp = client.post("/api/config", json={"epochs": 20, "learning_rate": 0.01})
        assert resp.status_code == 200
        assert resp.json()["config"]["epochs"] == 20


class TestFormula:
    def test_register_formula(self):
        resp = client.post("/api/formula", json={
            "name": "TestSwish",
            "expression": "x * torch.sigmoid(x)",
        })
        assert resp.status_code == 200
        assert "TestSwish" in resp.json()["activations"]


class TestDataset:
    def test_load_synthetic(self):
        resp = client.post("/api/data/load", json={"dataset": "xor", "type": "synthetic"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "loaded"

    def test_load_sample_backward_compat(self):
        resp = client.post("/api/data/sample", json={"dataset": "spiral"})
        assert resp.status_code == 200

    def test_data_info(self):
        client.post("/api/data/load", json={"dataset": "xor", "type": "synthetic"})
        resp = client.get("/api/data/info")
        assert resp.status_code == 200

    def test_augmentation(self):
        resp = client.post("/api/data/augmentation", json={
            "augmentations": [{"name": "gaussian_noise", "sigma": 0.1}]
        })
        assert resp.status_code == 200

    def test_list_augmentations(self):
        resp = client.get("/api/data/augmentations")
        assert resp.status_code == 200
        assert "available" in resp.json()


class TestBuildAndTrain:
    def test_build_model(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        resp = client.post("/api/build")
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_train_no_data(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        client.post("/api/build")
        resp = client.post("/api/train/start")
        assert resp.json()["status"] == "error"


class TestExportImport:
    def test_export_architecture(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        resp = client.get("/api/export/architecture")
        assert resp.status_code == 200
        data = resp.json()
        assert "graph" in data
        assert "config" in data

    def test_import_architecture(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        arch = client.get("/api/export/architecture").json()
        engine.reset()
        resp = client.post("/api/import/architecture", json=arch)
        assert resp.json()["status"] == "imported"

    def test_export_python(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        resp = client.get("/api/export/python")
        assert resp.status_code == 200
        assert "nn.Linear" in resp.text


class TestTemplates:
    def test_apply_template(self):
        resp = client.post("/api/templates/mlp_classifier/apply")
        assert resp.status_code == 200
        assert resp.json()["status"] == "applied"
        assert len(resp.json()["graph"]["nodes"]) > 0

    def test_apply_unknown_template(self):
        resp = client.post("/api/templates/nonexistent/apply")
        assert resp.json()["status"] == "error"


class TestSystemResources:
    def test_get_resources(self):
        resp = client.get("/api/system/resources")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "ram_total_gb" in data


class TestExperiments:
    def test_list_experiments_empty(self):
        resp = client.get("/api/experiments")
        assert resp.status_code == 200

    def test_save_experiment(self):
        # Setup: add layer, build, load data
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        client.post("/api/data/load", json={"dataset": "xor", "type": "synthetic"})
        client.post("/api/build")
        resp = client.post("/api/experiments/save", json={"name": "Test Exp"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "saved"


class TestReset:
    def test_reset(self):
        client.post("/api/graph/layer", json={"layer_type": "Linear", "params": {"in_features": 2, "out_features": 2}})
        resp = client.post("/api/reset")
        assert resp.json()["status"] == "reset"
        # Graph should be empty
        resp = client.get("/api/graph")
        assert len(resp.json()["nodes"]) == 0


class TestModelSource:
    def test_set_model_source(self):
        resp = client.post("/api/model_source", json={"source": "hf"})
        assert resp.json()["model_source"] == "hf"
        resp = client.post("/api/model_source", json={"source": "graph"})
        assert resp.json()["model_source"] == "graph"
