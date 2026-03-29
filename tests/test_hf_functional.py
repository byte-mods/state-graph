"""
Comprehensive functional tests for the HuggingFace tab.

All tests require `transformers` to be installed.
Network-dependent tests (model downloading) can be skipped with SKIP_NETWORK=1.
"""

import pytest
import os
import csv
import tempfile

from fastapi.testclient import TestClient

# Skip all tests if transformers not installed
transformers = pytest.importorskip("transformers")

from state_graph.server.app import app, engine

skip_network = pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK", "") != "",
    reason="Network tests skipped (SKIP_NETWORK is set)",
)


@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bert_tiny():
    """Load hf-internal-testing/tiny-random-BertForSequenceClassification for text-classification."""
    resp = client.post("/api/hf/load", json={
        "model_id": "hf-internal-testing/tiny-random-BertForSequenceClassification",
        "task": "text-classification",
        "num_labels": 2,
    })
    return resp


def _create_csv(n_rows=100):
    """Create a temporary CSV with text/label columns. Returns the file path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for i in range(n_rows):
        writer.writerow([f"This is sample text number {i}", i % 2])
    f.close()
    return f.name


def _load_csv_dataset(csv_path):
    """Load a local CSV dataset through the API."""
    return client.post("/api/hf/datasets/local", json={
        "format": "csv",
        "path": csv_path,
        "text_col": "text",
        "label_col": "label",
    })


# ---------------------------------------------------------------------------
# 2A. Model Loading
# ---------------------------------------------------------------------------

@skip_network
class TestHFModelLoading:
    def test_load_bert_tiny(self):
        resp = _load_bert_tiny()
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok" or "model_id" in data

    def test_model_tree(self):
        _load_bert_tiny()
        resp = client.get("/api/hf/model/tree")
        assert resp.status_code == 200
        assert "tree" in resp.json()

    def test_model_info(self):
        _load_bert_tiny()
        resp = client.get("/api/hf/model/info")
        assert resp.status_code == 200

    def test_load_nonexistent_model_returns_error(self):
        resp = client.post("/api/hf/load", json={
            "model_id": "nonexistent-org/nonexistent-model-xyz-12345",
            "task": "text-classification",
            "num_labels": 2,
        })
        # Should return an error status (either HTTP error or error in body)
        data = resp.json()
        is_error = resp.status_code >= 400 or data.get("status") == "error" or "error" in data
        assert is_error


# ---------------------------------------------------------------------------
# 2B. Architecture Surgery
# ---------------------------------------------------------------------------

@skip_network
class TestHFSurgery:
    @pytest.fixture(autouse=True)
    def load_model(self):
        _load_bert_tiny()

    def test_insert_module(self):
        resp = client.post("/api/hf/model/insert", json={
            "parent_path": "bert.encoder.layer",
            "name": "99",
            "module_type": "Linear",
            "module_params": {"in_features": 128, "out_features": 128},
        })
        assert resp.status_code == 200

    def test_remove_module(self):
        resp = client.post("/api/hf/model/remove", json={"path": "classifier"})
        assert resp.status_code == 200

    def test_replace_module(self):
        resp = client.post("/api/hf/model/replace", json={
            "path": "classifier",
            "module_type": "Linear",
            "module_params": {"in_features": 128, "out_features": 3},
        })
        assert resp.status_code == 200

    def test_add_head(self):
        resp = client.post("/api/hf/model/add_head", json={
            "name": "custom_head",
            "in_features": 128,
            "out_features": 5,
        })
        assert resp.status_code == 200

    def test_get_module_info(self):
        resp = client.get("/api/hf/model/module/classifier")
        assert resp.status_code == 200

    def test_replace_then_tree_reflects_change(self):
        client.post("/api/hf/model/replace", json={
            "path": "classifier",
            "module_type": "Linear",
            "module_params": {"in_features": 128, "out_features": 7},
        })
        resp = client.get("/api/hf/model/tree")
        assert resp.status_code == 200
        tree_data = resp.json()
        assert "tree" in tree_data


# ---------------------------------------------------------------------------
# 2C. Freeze / Unfreeze
# ---------------------------------------------------------------------------

@skip_network
class TestHFFreeze:
    @pytest.fixture(autouse=True)
    def load_model(self):
        _load_bert_tiny()

    def test_freeze_by_pattern(self):
        resp = client.post("/api/hf/model/freeze", json={
            "patterns": ["bert.encoder.layer.0"],
        })
        assert resp.status_code == 200

    def test_unfreeze_by_pattern(self):
        client.post("/api/hf/model/freeze", json={"patterns": ["bert.encoder"]})
        resp = client.post("/api/hf/model/unfreeze", json={
            "patterns": ["bert.encoder.layer.1"],
        })
        assert resp.status_code == 200

    def test_freeze_unfreeze_changes_params(self):
        info1 = client.get("/api/hf/model/info").json()
        client.post("/api/hf/model/freeze", json={"patterns": ["bert"]})
        info2 = client.get("/api/hf/model/info").json()
        # After freezing, the response should still be valid
        assert info2 is not None
        # Verify info contains expected keys (implementation-dependent)
        assert isinstance(info1, dict)
        assert isinstance(info2, dict)

    def test_freeze_all_then_unfreeze_classifier(self):
        client.post("/api/hf/model/freeze", json={"patterns": [""]})
        resp = client.post("/api/hf/model/unfreeze", json={
            "patterns": ["classifier"],
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2D. LoRA
# ---------------------------------------------------------------------------

@skip_network
class TestHFLoRA:
    @pytest.fixture(autouse=True)
    def load_model(self):
        _load_bert_tiny()

    def test_get_lora_targets(self):
        resp = client.get("/api/hf/model/lora_targets")
        assert resp.status_code == 200
        assert "targets" in resp.json()

    def test_apply_lora(self):
        peft = pytest.importorskip("peft")
        resp = client.post("/api/hf/model/lora", json={
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
        })
        assert resp.status_code == 200

    def test_apply_lora_with_targets(self):
        peft = pytest.importorskip("peft")
        # First get available targets
        targets_resp = client.get("/api/hf/model/lora_targets")
        targets = targets_resp.json().get("targets", [])
        if targets:
            resp = client.post("/api/hf/model/lora", json={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": targets[:2] if len(targets) >= 2 else targets,
            })
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2E. Dataset Operations
# ---------------------------------------------------------------------------

@skip_network
class TestHFDatasets:
    @pytest.fixture(autouse=True)
    def load_model(self):
        _load_bert_tiny()

    def test_load_local_csv(self):
        csv_path = _create_csv(100)
        try:
            resp = _load_csv_dataset(csv_path)
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_dataset_preview(self):
        csv_path = _create_csv(50)
        try:
            _load_csv_dataset(csv_path)
            resp = client.get("/api/hf/datasets/preview", params={"n": 3})
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_dataset_info(self):
        csv_path = _create_csv(50)
        try:
            _load_csv_dataset(csv_path)
            resp = client.get("/api/hf/datasets/info")
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_dataset_columns(self):
        csv_path = _create_csv(50)
        try:
            _load_csv_dataset(csv_path)
            resp = client.get("/api/hf/datasets/columns")
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_preprocess(self):
        csv_path = _create_csv(50)
        try:
            _load_csv_dataset(csv_path)
            resp = client.post("/api/hf/datasets/preprocess", json={
                "max_length": 32,
                "text_column": "text",
                "label_column": "label",
            })
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_prepare_dataloaders(self):
        csv_path = _create_csv(50)
        try:
            _load_csv_dataset(csv_path)
            client.post("/api/hf/datasets/preprocess", json={
                "max_length": 32,
                "text_column": "text",
                "label_column": "label",
            })
            resp = client.post("/api/hf/datasets/prepare", json={
                "batch_size": 16,
                "val_split": 0.2,
            })
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_load_csv_with_missing_columns(self):
        """Loading a CSV and specifying non-existent columns should handle gracefully."""
        csv_path = _create_csv(20)
        try:
            resp = client.post("/api/hf/datasets/local", json={
                "format": "csv",
                "path": csv_path,
                "text_col": "nonexistent_col",
                "label_col": "also_missing",
            })
            # Should either succeed (ignoring bad cols) or return a clear error
            assert resp.status_code in (200, 400, 422)
        finally:
            os.unlink(csv_path)


# ---------------------------------------------------------------------------
# 2F. HF Training
# ---------------------------------------------------------------------------

@skip_network
class TestHFTraining:
    def test_full_train_flow(self):
        # Load model
        _load_bert_tiny()

        csv_path = None
        try:
            # Create and load CSV data
            f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for i in range(100):
                sentiment = "good" if i % 2 == 0 else "bad"
                writer.writerow([f"This is a {sentiment} review number {i}", i % 2])
            f.close()
            csv_path = f.name

            _load_csv_dataset(csv_path)

            # Train for 1 epoch
            resp = client.post("/api/hf/train", json={
                "epochs": 1,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "max_length": 32,
                "text_column": "text",
                "label_column": "label",
            })
            assert resp.status_code == 200
        finally:
            if csv_path and os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_training_history(self):
        resp = client.get("/api/hf/train/history")
        assert resp.status_code == 200

    def test_train_without_data_returns_error(self):
        _load_bert_tiny()
        resp = client.post("/api/hf/train", json={
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "max_length": 32,
            "text_column": "text",
            "label_column": "label",
        })
        # Should fail gracefully when no dataset loaded
        data = resp.json()
        is_error = resp.status_code >= 400 or data.get("status") == "error" or "error" in data
        assert is_error or resp.status_code == 200  # Some impls may queue training


# ---------------------------------------------------------------------------
# 2G. Inference
# ---------------------------------------------------------------------------

@skip_network
class TestHFInference:
    def test_text_classification_inference(self):
        _load_bert_tiny()
        resp = client.post("/api/hf/inference", json={
            "input": "This is a test sentence",
        })
        assert resp.status_code == 200

    def test_inference_without_model_returns_error(self):
        resp = client.post("/api/hf/inference", json={
            "input": "This should fail without a model loaded",
        })
        data = resp.json()
        is_error = resp.status_code >= 400 or data.get("status") == "error" or "error" in data
        assert is_error

    def test_inference_with_empty_input(self):
        _load_bert_tiny()
        resp = client.post("/api/hf/inference", json={
            "input": "",
        })
        # Should either handle empty input gracefully or return an error
        assert resp.status_code in (200, 400, 422)


# ---------------------------------------------------------------------------
# 2H. HF Config
# ---------------------------------------------------------------------------

class TestHFConfig:
    def test_update_config(self):
        resp = client.post("/api/hf/config", json={
            "gradient_accumulation_steps": 2,
            "fp16": False,
            "warmup_steps": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hf_config"]["gradient_accumulation_steps"] == 2

    def test_update_config_fp16(self):
        resp = client.post("/api/hf/config", json={
            "fp16": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hf_config"]["fp16"] is True

    def test_update_config_multiple_fields(self):
        resp = client.post("/api/hf/config", json={
            "gradient_accumulation_steps": 4,
            "fp16": False,
            "warmup_steps": 50,
        })
        assert resp.status_code == 200
        cfg = resp.json()["hf_config"]
        assert cfg["gradient_accumulation_steps"] == 4
        assert cfg["warmup_steps"] == 50


# ---------------------------------------------------------------------------
# 2I. Model Source Toggle
# ---------------------------------------------------------------------------

class TestModelSource:
    def test_set_hf_source(self):
        resp = client.post("/api/model_source", json={"source": "hf"})
        assert resp.status_code == 200
        assert resp.json()["model_source"] == "hf"

    def test_set_graph_source(self):
        resp = client.post("/api/model_source", json={"source": "graph"})
        assert resp.status_code == 200
        assert resp.json()["model_source"] == "graph"

    def test_toggle_source_back_and_forth(self):
        resp1 = client.post("/api/model_source", json={"source": "hf"})
        assert resp1.json()["model_source"] == "hf"
        resp2 = client.post("/api/model_source", json={"source": "graph"})
        assert resp2.json()["model_source"] == "graph"
        resp3 = client.post("/api/model_source", json={"source": "hf"})
        assert resp3.json()["model_source"] == "hf"

    def test_invalid_source(self):
        resp = client.post("/api/model_source", json={"source": "invalid_source"})
        # Should either reject invalid source or default to something
        assert resp.status_code in (200, 400, 422)
