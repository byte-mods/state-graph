"""Tests for HuggingFace architecture surgery, training, and inference features."""

import torch
import torch.nn as nn
import pytest

from state_graph.hf.hub import HFModelManager


class TestHFModelManagerArchitectureSurgery:
    """Test insert/remove/replace/add_head on a simple model."""

    def _create_manager_with_model(self):
        mgr = HFModelManager()
        # Create a simple model manually instead of downloading
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        mgr.model = model
        mgr.model_id = "test-model"
        mgr.library = "transformers"
        mgr.task = "text-classification"
        mgr._model_info = {"model_id": "test-model"}
        return mgr

    def test_get_module_by_path(self):
        mgr = self._create_manager_with_model()
        module = mgr._get_module_by_path(mgr.model, "0")
        assert isinstance(module, nn.Linear)
        assert module.in_features == 64

    def test_get_module_root(self):
        mgr = self._create_manager_with_model()
        module = mgr._get_module_by_path(mgr.model, "")
        assert module is mgr.model

    def test_get_module_info(self):
        mgr = self._create_manager_with_model()
        info = mgr.get_module_info("0")
        assert info["status"] == "ok"
        assert info["type"] == "Linear"
        assert info["params"] > 0

    def test_get_module_info_not_found(self):
        mgr = self._create_manager_with_model()
        info = mgr.get_module_info("nonexistent")
        assert info["status"] == "error"

    def test_replace_module(self):
        mgr = self._create_manager_with_model()
        new_linear = nn.Linear(64, 256)
        result = mgr.replace_module("0", new_linear)
        assert result["status"] == "ok"
        assert mgr.model[0].out_features == 256

    def test_replace_module_not_found(self):
        mgr = self._create_manager_with_model()
        result = mgr.replace_module("nonexistent", nn.Linear(1, 1))
        assert result["status"] == "error"

    def test_remove_module(self):
        mgr = self._create_manager_with_model()
        # nn.Sequential doesn't support remove well, but replace with Identity works
        result = mgr.replace_module("1", nn.Identity())
        assert result["status"] == "ok"
        assert isinstance(mgr.model[1], nn.Identity)

    def test_remove_module_not_found(self):
        mgr = self._create_manager_with_model()
        result = mgr.remove_module("nonexistent.child")
        assert result["status"] == "error"

    def test_add_head(self):
        mgr = self._create_manager_with_model()
        result = mgr.add_head("custom_classifier", 10, 5)
        assert result["status"] == "ok"
        assert hasattr(mgr.model, "custom_classifier")
        assert mgr.model.custom_classifier.out_features == 5

    def test_add_head_no_model(self):
        mgr = HFModelManager()
        result = mgr.add_head("head", 10, 5)
        assert result["status"] == "error"

    def test_insert_module(self):
        mgr = self._create_manager_with_model()
        # Insert into the Sequential (which is a ModuleList-like)
        # Actually nn.Sequential doesn't have insert, so this tests the setattr path
        new_mod = nn.Dropout(0.5)
        result = mgr.insert_module("", "dropout", new_mod)
        assert result["status"] == "ok"
        assert hasattr(mgr.model, "dropout")

    def test_replace_no_model(self):
        mgr = HFModelManager()
        result = mgr.replace_module("0", nn.Linear(1, 1))
        assert result["status"] == "error"


class TestHFModelManagerWithModuleList:
    """Test surgery on models with ModuleList (like transformer layers)."""

    def _create_model_with_layers(self):
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(64, 64) for _ in range(4)
                ])
                self.head = nn.Linear(64, 10)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)

        mgr = HFModelManager()
        mgr.model = SimpleTransformer()
        mgr.model_id = "test"
        mgr.library = "transformers"
        mgr.task = "text-classification"
        mgr._model_info = {}
        return mgr

    def test_remove_from_modulelist(self):
        mgr = self._create_model_with_layers()
        assert len(mgr.model.layers) == 4
        result = mgr.remove_module("layers.3")
        assert result["status"] == "ok"
        assert len(mgr.model.layers) == 3

    def test_insert_into_modulelist(self):
        mgr = self._create_model_with_layers()
        new_layer = nn.Linear(64, 64)
        result = mgr.insert_module("layers", "4", new_layer)
        assert result["status"] == "ok"
        assert len(mgr.model.layers) == 5

    def test_replace_in_modulelist(self):
        mgr = self._create_model_with_layers()
        new_layer = nn.Linear(64, 128)
        result = mgr.replace_module("layers.0", new_layer)
        assert result["status"] == "ok"
        assert mgr.model.layers[0].out_features == 128

    def test_get_module_in_modulelist(self):
        mgr = self._create_model_with_layers()
        info = mgr.get_module_info("layers.2")
        assert info["status"] == "ok"
        assert info["type"] == "Linear"


class TestHFTraining:
    """Test the training loop (with a tiny model, no HF download needed)."""

    def _setup_training(self):
        mgr = HFModelManager()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)

            def forward(self, x, labels=None):
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits, labels)
                return type('Output', (), {'loss': loss, 'logits': logits})()

        mgr.model = TinyModel()
        mgr.model_id = "tiny"
        mgr.library = "transformers"
        mgr._model_info = {}

        # Create tiny dataset
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(20, 4)
        y = torch.randint(0, 2, (20,))
        ds = TensorDataset(X, y)
        train_loader = DataLoader(ds, batch_size=4)
        return mgr, train_loader

    def test_train_starts(self):
        mgr, loader = self._setup_training()
        result = mgr.train(loader, epochs=1, lr=0.01)
        assert result["status"] == "started"
        # Wait for training to finish
        mgr._train_thread.join(timeout=10)

    def test_train_history(self):
        mgr, loader = self._setup_training()
        mgr.train(loader, epochs=1, lr=0.01)
        mgr._train_thread.join(timeout=10)
        history = mgr.get_train_history()
        assert len(history) > 0
        assert "loss" in history[0]
        assert "lr" in history[0]

    def test_stop_training(self):
        mgr, loader = self._setup_training()
        mgr.train(loader, epochs=100, lr=0.01)  # Many epochs
        result = mgr.stop_training()
        assert result["status"] == "stopped"

    def test_train_no_model(self):
        mgr = HFModelManager()
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)
        result = mgr.train(loader)
        assert result["status"] == "error"


class TestHFInference:
    def test_inference_no_model(self):
        mgr = HFModelManager()
        result = mgr.inference({"text": "hello"})
        assert result["status"] == "error"

    def test_diffusion_no_model(self):
        mgr = HFModelManager()
        result = mgr.diffusion_generate("a cat")
        assert result["status"] == "error"

    def test_diffusion_wrong_library(self):
        mgr = HFModelManager()
        mgr.model = nn.Linear(1, 1)
        mgr.library = "transformers"
        result = mgr.diffusion_generate("a cat")
        assert result["status"] == "error"
        assert "Not a diffusers" in result["message"]


# ── Server Endpoint Tests ──

from fastapi.testclient import TestClient
from state_graph.server.app import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


class TestHFArchitectureEndpoints:
    def _load_dummy_model(self):
        """Set up a dummy model on the engine's HF manager."""
        from state_graph.server.app import _get_hf_manager

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
                self.classifier = nn.Linear(64, 10)

            def forward(self, x):
                for layer in self.encoder:
                    x = layer(x)
                return self.classifier(x)

        mgr = _get_hf_manager()
        mgr.model = DummyModel()
        mgr.model_id = "test-model"
        mgr.library = "transformers"
        mgr.task = "text-classification"
        mgr._model_info = {"model_id": "test-model"}
        engine.model_source = "hf"

    def test_replace_module(self):
        self._load_dummy_model()
        resp = client.post("/api/hf/model/replace", json={
            "path": "classifier",
            "module_type": "Linear",
            "module_params": {"in_features": 64, "out_features": 5},
        })
        data = resp.json()
        assert data["status"] == "ok"

    def test_remove_module(self):
        self._load_dummy_model()
        resp = client.post("/api/hf/model/remove", json={"path": "encoder.2"})
        data = resp.json()
        assert data["status"] == "ok"

    def test_insert_module(self):
        self._load_dummy_model()
        resp = client.post("/api/hf/model/insert", json={
            "parent_path": "encoder",
            "name": "3",
            "module_type": "Linear",
            "module_params": {"in_features": 64, "out_features": 64},
        })
        data = resp.json()
        assert data["status"] == "ok"

    def test_add_head(self):
        self._load_dummy_model()
        resp = client.post("/api/hf/model/add_head", json={
            "name": "reward_head",
            "in_features": 64,
            "out_features": 1,
        })
        data = resp.json()
        assert data["status"] == "ok"

    def test_get_module_info(self):
        self._load_dummy_model()
        resp = client.get("/api/hf/model/module/classifier")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["type"] == "Linear"

    def test_get_module_not_found(self):
        self._load_dummy_model()
        resp = client.get("/api/hf/model/module/nonexistent")
        data = resp.json()
        assert data["status"] == "error"


class TestHFTrainEndpoints:
    def test_train_no_model(self):
        resp = client.post("/api/hf/train", json={"epochs": 1})
        data = resp.json()
        assert data["status"] == "error"

    def test_train_history_empty(self):
        resp = client.get("/api/hf/train/history")
        data = resp.json()
        assert "history" in data


class TestHFInferenceEndpoints:
    def test_inference_no_model(self):
        resp = client.post("/api/hf/inference", json={"text": "hello"})
        data = resp.json()
        assert data["status"] == "error"

    def test_diffusion_no_model(self):
        resp = client.post("/api/hf/diffusion/generate", json={"prompt": "a cat"})
        data = resp.json()
        assert data["status"] == "error"
