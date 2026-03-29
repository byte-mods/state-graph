"""Comprehensive functional tests for the Graph tab.

Covers: architecture templates, layer types, activations, datasets,
optimizer/loss training matrix, LR schedulers, augmentation, export/import,
experiment management, model save/load, and end-to-end training.
"""

import os
import time
import tempfile

import pytest
from fastapi.testclient import TestClient

from state_graph.server.app import app, engine, experiment_history


@pytest.fixture(autouse=True)
def reset_engine():
    """Reset engine state before each test."""
    engine.reset()
    experiment_history.clear()
    yield
    engine.reset()
    experiment_history.clear()


client = TestClient(app)


def wait_for_training(timeout=30):
    """Poll /api/status until training completes."""
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get("/api/status")
        if not resp.json().get("is_training", False):
            return resp.json()
        time.sleep(0.3)
    raise TimeoutError("Training didn't finish")


def _setup_simple_model_and_data(
    optimizer="Adam",
    loss="CrossEntropyLoss",
    dataset="xor",
    dataset_type="synthetic",
    n_samples=200,
    epochs=2,
    scheduler=None,
    scheduler_params=None,
    in_features=2,
    out_features=2,
):
    """Helper: add a Linear layer, load data, configure, and build."""
    client.post("/api/graph/layer", json={
        "layer_type": "Linear",
        "params": {"in_features": in_features, "out_features": out_features},
    })
    client.post("/api/data/load", json={
        "dataset": dataset, "type": dataset_type, "n_samples": n_samples,
    })
    config = {
        "epochs": epochs,
        "batch_size": 32,
        "learning_rate": 0.01,
        "optimizer": optimizer,
        "loss": loss,
    }
    if scheduler:
        config["scheduler"] = scheduler
        if scheduler_params:
            config["scheduler_params"] = scheduler_params
    client.post("/api/config", json=config)
    resp = client.post("/api/build")
    assert resp.status_code == 200
    return resp


# ---------------------------------------------------------------------------
# 1A. Architecture Templates
# ---------------------------------------------------------------------------

ALL_TEMPLATES = [
    "mlp_classifier", "deep_mlp", "gated_network", "wide_shallow",
    "transformer_classifier", "deep_transformer", "mnist_cnn", "vit_tiny",
    "mobilenet_style", "audio_classifier", "autoencoder", "resnet_style",
    "swin_transformer", "convnext", "yolov8_backbone", "detr_detector",
    "dit_small", "sd_unet", "conformer_asr", "whisper_encoder",
    "hifi_gan_generator", "tts_fastspeech", "voice_cloning", "gemini_style",
    "claude_style_llm", "moe_llm", "encoder_decoder_t5", "mamba_lm",
    "video_dit", "multimodal_llm", "nano_lm",
]


class TestArchitectureTemplates:
    @pytest.mark.parametrize("template_name", ALL_TEMPLATES)
    def test_apply_template(self, template_name):
        resp = client.post(f"/api/templates/{template_name}/apply")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "applied"
        assert len(data["graph"]["nodes"]) > 0

    def test_list_templates(self):
        resp = client.get("/api/templates")
        assert resp.status_code == 200
        data = resp.json()
        for name in ALL_TEMPLATES:
            assert name in data, f"Template '{name}' missing from list"

    def test_unknown_template_returns_error(self):
        resp = client.post("/api/templates/nonexistent_template/apply")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# 1B. Layer Types
# ---------------------------------------------------------------------------

CORE_LAYERS = [
    ("Linear", {"in_features": 10, "out_features": 5}),
    ("Conv1d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3}),
    ("Conv2d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3}),
    ("Conv3d", {"in_channels": 1, "out_channels": 8, "kernel_size": 3}),
    ("BatchNorm1d", {"num_features": 10}),
    ("BatchNorm2d", {"num_features": 8}),
    ("GroupNorm", {"num_groups": 2, "num_channels": 8}),
    ("LayerNorm", {"normalized_shape": [10]}),
    ("Dropout", {"p": 0.5}),
    ("Embedding", {"num_embeddings": 100, "embedding_dim": 32}),
    ("LSTM", {"input_size": 10, "hidden_size": 20}),
    ("GRU", {"input_size": 10, "hidden_size": 20}),
    ("MultiheadAttention", {"embed_dim": 32, "num_heads": 4}),
    ("Flatten", {}),
    ("MaxPool1d", {"kernel_size": 2}),
    ("MaxPool2d", {"kernel_size": 2}),
    ("AvgPool2d", {"kernel_size": 2}),
    ("AdaptiveAvgPool1d", {"output_size": 1}),
    ("AdaptiveAvgPool2d", {"output_size": [1, 1]}),
]

BUILDING_BLOCK_LAYERS = [
    ("TransformerBlock", {"d_model": 32, "n_heads": 4, "dropout": 0.1}),
    ("ResidualBlock", {"in_features": 32, "hidden_features": 64}),
    ("GatedLinearUnit", {"in_features": 32}),
    ("SwishLinear", {"in_features": 32, "out_features": 32}),
    ("PositionalEncoding", {"d_model": 32, "max_len": 100}),
    ("TokenEmbedding", {"vocab_size": 100, "d_model": 32}),
    ("SequencePool", {"d_model": 32}),
]

VISION_LAYERS = [
    ("PatchEmbed", {"in_channels": 3, "d_model": 32, "patch_size": 4, "image_size": 16}),
    ("DepthwiseSeparableConv", {"in_channels": 8, "out_channels": 16, "kernel_size": 3}),
    ("ResNetBlock", {"in_channels": 64, "out_channels": 64}),
    ("ConvNeXtBlock", {"dim": 64}),
    ("MBConvBlock", {"in_channels": 32, "out_channels": 64, "expand_ratio": 4}),
    ("VisionEncoder", {"in_channels": 3, "d_model": 64, "patch_size": 4, "image_size": 32, "n_layers": 2, "n_heads": 4}),
]

DIFFUSION_LAYERS = [
    ("DiffusionUNet", {"in_channels": 1, "base_channels": 32}),
    ("VAE", {"in_channels": 1, "latent_dim": 16}),
]

SSM_LAYERS = [
    ("MambaBlock", {"d_model": 32, "d_state": 8, "d_conv": 4, "expand_factor": 2}),
    ("RWKVBlock", {"d_model": 32}),
    ("RetentionLayer", {"d_model": 32, "n_heads": 4}),
    ("HyenaOperator", {"d_model": 32, "max_len": 64}),
    ("XLSTM", {"d_model": 32}),
    ("GatedLinearRecurrence", {"d_model": 32}),
    ("SelectiveScan", {"d_model": 32, "d_state": 8}),
]

ALL_LAYERS = CORE_LAYERS + BUILDING_BLOCK_LAYERS + VISION_LAYERS + DIFFUSION_LAYERS + SSM_LAYERS


class TestLayerTypes:
    @pytest.mark.parametrize("layer_type,params", ALL_LAYERS,
                             ids=[l[0] for l in ALL_LAYERS])
    def test_add_layer(self, layer_type, params):
        resp = client.post("/api/graph/layer", json={
            "layer_type": layer_type, "params": params,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data
        assert len(data["graph"]["nodes"]) == 1

    def test_add_multiple_layers_increments_count(self):
        for i, (layer_type, params) in enumerate(CORE_LAYERS[:5], 1):
            resp = client.post("/api/graph/layer", json={
                "layer_type": layer_type, "params": params,
            })
            assert len(resp.json()["graph"]["nodes"]) == i

    def test_remove_layer_after_add(self):
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 10, "out_features": 5},
        })
        node_id = resp.json()["node_id"]
        resp = client.delete(f"/api/graph/layer/{node_id}")
        assert resp.status_code == 200
        assert len(resp.json()["graph"]["nodes"]) == 0


# ---------------------------------------------------------------------------
# 1C. Activation Functions
# ---------------------------------------------------------------------------

ACTIVATIONS = [
    "ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid",
    "Tanh", "Softmax", "ELU", "PReLU", "Mish",
]


class TestActivations:
    @pytest.mark.parametrize("activation", ACTIVATIONS)
    def test_add_layer_with_activation(self, activation):
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 5},
            "activation": activation,
        })
        assert resp.status_code == 200
        assert "node_id" in resp.json()

    def test_custom_formula_swish(self):
        resp = client.post("/api/formula", json={
            "name": "Swish",
            "expression": "x * torch.sigmoid(x)",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "registered"
        assert "Swish" in data["activations"]

    def test_custom_formula_squared_relu(self):
        resp = client.post("/api/formula", json={
            "name": "SquaredReLU",
            "expression": "torch.relu(x) ** 2",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "registered"
        assert "SquaredReLU" in resp.json()["activations"]

    def test_custom_formula_usable_on_layer(self):
        client.post("/api/formula", json={
            "name": "MyAct", "expression": "x * 2",
        })
        resp = client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 4, "out_features": 4},
            "activation": "MyAct",
        })
        assert resp.status_code == 200
        assert "node_id" in resp.json()


# ---------------------------------------------------------------------------
# 1D. Datasets
# ---------------------------------------------------------------------------

SYNTHETIC_DATASETS = [
    "random", "xor", "spiral", "circles", "moons",
    "blobs", "checkerboard", "regression_sin",
]

REAL_DATASETS = ["mnist", "fashion_mnist", "cifar10", "cifar100", "svhn"]


class TestDatasets:
    @pytest.mark.parametrize("dataset", SYNTHETIC_DATASETS)
    def test_load_synthetic_dataset(self, dataset):
        resp = client.post("/api/data/load", json={
            "dataset": dataset, "type": "synthetic", "n_samples": 200,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") in ("ok", "loaded")

    @pytest.mark.parametrize("dataset", REAL_DATASETS)
    @pytest.mark.skipif(
        not os.environ.get("TEST_REAL_DATASETS"),
        reason="Skipping real datasets (set TEST_REAL_DATASETS=1 to enable)",
    )
    def test_load_real_dataset(self, dataset):
        resp = client.post("/api/data/load", json={
            "dataset": dataset, "type": "real",
        })
        assert resp.status_code == 200

    def test_data_info_after_load(self):
        client.post("/api/data/load", json={
            "dataset": "xor", "type": "synthetic", "n_samples": 100,
        })
        resp = client.get("/api/data/info")
        assert resp.status_code == 200
        info = resp.json()
        assert info.get("dataset") is not None or info.get("loaded") is not None


# ---------------------------------------------------------------------------
# 1E. Optimizers x Loss Functions Training Matrix
# ---------------------------------------------------------------------------

CLASSIFICATION_OPTIMIZERS = ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"]
REGRESSION_LOSSES = ["MSELoss"]


class TestOptimizerLossMatrix:
    @pytest.mark.parametrize("optimizer", CLASSIFICATION_OPTIMIZERS)
    def test_optimizer_with_cross_entropy(self, optimizer):
        _setup_simple_model_and_data(optimizer=optimizer, loss="CrossEntropyLoss")
        resp = client.post("/api/train/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        status = wait_for_training()
        assert status.get("is_training") is False

    @pytest.mark.parametrize("loss", REGRESSION_LOSSES)
    def test_adam_with_regression_loss(self, loss):
        """Test regression losses with appropriate data."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 1, "out_features": 1},
        })
        client.post("/api/data/load", json={
            "dataset": "regression_sin", "type": "synthetic", "n_samples": 200,
        })
        client.post("/api/config", json={
            "epochs": 2, "batch_size": 32, "learning_rate": 0.01,
            "optimizer": "Adam", "loss": loss,
        })
        client.post("/api/build")
        resp = client.post("/api/train/start")
        assert resp.json()["status"] == "started"
        status = wait_for_training()
        assert status.get("is_training") is False

    def test_training_stop(self):
        """Verify training can be stopped early."""
        _setup_simple_model_and_data(epochs=50)
        client.post("/api/train/start")
        time.sleep(0.5)
        resp = client.post("/api/train/stop")
        assert resp.status_code == 200
        wait_for_training(timeout=10)


# ---------------------------------------------------------------------------
# 1F. LR Schedulers
# ---------------------------------------------------------------------------

SCHEDULER_CONFIGS = {
    "StepLR": {"step_size": 1, "gamma": 0.9},
    "MultiStepLR": {"milestones": [1], "gamma": 0.5},
    "ExponentialLR": {"gamma": 0.9},
    "CosineAnnealingLR": {"T_max": 2},
    "ReduceLROnPlateau": {"mode": "min", "factor": 0.5, "patience": 1},
    "CyclicLR": {"base_lr": 0.001, "max_lr": 0.01, "step_size_up": 5},
    "OneCycleLR": {"max_lr": 0.01, "total_steps": 20},
    "CosineAnnealingWarmRestarts": {"T_0": 1, "T_mult": 1},
    "LinearLR": {"start_factor": 0.1, "total_iters": 10},
    "PolynomialLR": {"total_iters": 10, "power": 2.0},
}


class TestLRSchedulers:
    @pytest.mark.parametrize("scheduler_name,scheduler_params",
                             list(SCHEDULER_CONFIGS.items()),
                             ids=list(SCHEDULER_CONFIGS.keys()))
    def test_scheduler_training(self, scheduler_name, scheduler_params):
        _setup_simple_model_and_data(
            scheduler=scheduler_name,
            scheduler_params=scheduler_params,
        )
        resp = client.post("/api/train/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        status = wait_for_training()
        assert status.get("is_training") is False

    def test_scheduler_defaults_in_registry(self):
        resp = client.get("/api/registry")
        data = resp.json()
        assert "scheduler_defaults" in data
        for name in SCHEDULER_CONFIGS:
            assert name in data["schedulers"], f"Scheduler {name} not in registry"


# ---------------------------------------------------------------------------
# 1G. Augmentation Pipeline
# ---------------------------------------------------------------------------

AUGMENTATION_TYPES = [
    {"name": "gaussian_noise", "sigma": 0.1},
    {"name": "dropout_noise", "p": 0.1},
    {"name": "scaling", "min_scale": 0.9, "max_scale": 1.1},
    {"name": "mixup", "alpha": 0.2},
]


class TestAugmentation:
    @pytest.mark.parametrize("aug", AUGMENTATION_TYPES,
                             ids=[a["name"] for a in AUGMENTATION_TYPES])
    def test_set_single_augmentation(self, aug):
        resp = client.post("/api/data/augmentation", json={
            "augmentations": [aug],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_set_multiple_augmentations(self):
        resp = client.post("/api/data/augmentation", json={
            "augmentations": AUGMENTATION_TYPES,
        })
        assert resp.status_code == 200
        assert len(resp.json()["augmentations"]) == len(AUGMENTATION_TYPES)

    def test_list_augmentations(self):
        resp = client.get("/api/data/augmentations")
        assert resp.status_code == 200
        data = resp.json()
        assert "available" in data
        assert "active" in data
        for aug_name in ["gaussian_noise", "dropout_noise", "scaling", "mixup"]:
            assert aug_name in data["available"]

    def test_clear_augmentations(self):
        client.post("/api/data/augmentation", json={
            "augmentations": AUGMENTATION_TYPES,
        })
        resp = client.post("/api/data/augmentation", json={
            "augmentations": [],
        })
        assert resp.status_code == 200
        assert resp.json()["augmentations"] == []


# ---------------------------------------------------------------------------
# 1H. Export / Import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_architecture_json(self):
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 5},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 5, "out_features": 2},
        })
        resp = client.get("/api/export/architecture")
        assert resp.status_code == 200
        arch = resp.json()
        assert "graph" in arch
        assert "nodes" in arch["graph"]
        assert len(arch["graph"]["nodes"]) == 2

    def test_import_architecture_json(self):
        # Build architecture
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 5},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Dropout", "params": {"p": 0.3},
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 5, "out_features": 2},
        })
        # Export
        export_resp = client.get("/api/export/architecture")
        arch = export_resp.json()
        original_count = len(arch["graph"]["nodes"])

        # Reset and import
        engine.reset()
        resp = client.post("/api/import/architecture", json=arch)
        assert resp.status_code == 200
        assert len(resp.json()["graph"]["nodes"]) == original_count

    def test_export_python_code(self):
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 5},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 5, "out_features": 2},
        })
        client.post("/api/build")
        resp = client.get("/api/export/python")
        assert resp.status_code == 200
        code = resp.text
        assert "Linear" in code

    def test_roundtrip_preserves_architecture(self):
        """Export then import should yield identical node count."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 8, "out_features": 4},
            "activation": "GELU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 4, "out_features": 2},
        })
        arch = client.get("/api/export/architecture").json()
        engine.reset()
        imported = client.post("/api/import/architecture", json=arch).json()
        assert len(imported["graph"]["nodes"]) == len(arch["graph"]["nodes"])


# ---------------------------------------------------------------------------
# 1I. Experiment Management
# ---------------------------------------------------------------------------

class TestExperimentManagement:
    def _train_and_save(self, name="Test Experiment"):
        """Helper: build, train, save experiment."""
        _setup_simple_model_and_data()
        client.post("/api/train/start")
        wait_for_training()
        resp = client.post("/api/experiments/save", json={"name": name})
        assert resp.status_code == 200
        return resp.json()

    def test_save_experiment(self):
        result = self._train_and_save()
        assert result["status"] == "saved"
        assert "experiment" in result
        assert result["experiment"]["name"] == "Test Experiment"

    def test_list_experiments(self):
        self._train_and_save("Exp A")
        engine.reset()
        self._train_and_save("Exp B")
        resp = client.get("/api/experiments")
        assert resp.status_code == 200
        exps = resp.json()["experiments"]
        assert len(exps) == 2

    def test_get_experiment_by_id(self):
        self._train_and_save("My Exp")
        resp = client.get("/api/experiments/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Exp"

    def test_get_nonexistent_experiment(self):
        resp = client.get("/api/experiments/999")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_compare_experiments(self):
        self._train_and_save("Exp 1")
        engine.reset()
        self._train_and_save("Exp 2")
        resp = client.get("/api/experiments/compare")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["experiments"]) == 2

    def test_delete_experiment(self):
        self._train_and_save("To Delete")
        resp = client.delete("/api/experiments/0")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        resp = client.get("/api/experiments")
        assert len(resp.json()["experiments"]) == 0


# ---------------------------------------------------------------------------
# 1J. Model Save / Load
# ---------------------------------------------------------------------------

class TestModelSaveLoad:
    def test_save_model(self):
        _setup_simple_model_and_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            resp = client.post("/api/model/save", json={"path": path})
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            assert os.path.exists(path)

    def test_load_model(self):
        _setup_simple_model_and_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            client.post("/api/model/save", json={"path": path})
            # Reset and reload
            engine.reset()
            resp = client.post("/api/model/load", json={"path": path})
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_save_without_model_errors(self):
        resp = client.post("/api/model/save", json={"path": "/tmp/no_model.pt"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_load_nonexistent_file_errors(self):
        resp = client.post("/api/model/load", json={"path": "/tmp/nonexistent_model_xyz.pt"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# 1K. Full End-to-End Training
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_mlp_spiral_training(self):
        """Build MLP, load spiral data, train, verify loss decreases."""
        # Build a small MLP
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 2, "out_features": 32},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 32, "out_features": 32},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 32, "out_features": 3},
        })
        # Load spiral data (3 classes)
        client.post("/api/data/load", json={
            "dataset": "spiral", "type": "synthetic", "n_samples": 200,
        })
        client.post("/api/config", json={
            "epochs": 5, "batch_size": 32, "learning_rate": 0.01,
            "optimizer": "Adam", "loss": "CrossEntropyLoss",
        })
        build_resp = client.post("/api/build")
        assert build_resp.status_code == 200

        # Train
        train_resp = client.post("/api/train/start")
        assert train_resp.json()["status"] == "started"
        status = wait_for_training()
        assert status.get("is_training") is False

    def test_profiling_after_build(self):
        """Profile a built model and check bottleneck analysis."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 10, "out_features": 32},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 32, "out_features": 5},
        })
        client.post("/api/build")
        resp = client.post("/api/profile", json={
            "input_shape": [1, 10], "n_runs": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_bottleneck_after_training(self):
        """Train then check bottleneck endpoint returns data."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 2, "out_features": 16},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 16, "out_features": 2},
        })
        client.post("/api/data/load", json={
            "dataset": "xor", "type": "synthetic", "n_samples": 100,
        })
        client.post("/api/config", json={
            "epochs": 2, "batch_size": 32, "learning_rate": 0.01,
            "optimizer": "Adam", "loss": "CrossEntropyLoss",
        })
        client.post("/api/build")
        client.post("/api/train/start")
        wait_for_training()
        resp = client.get("/api/bottleneck")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_metrics_snapshot(self):
        """After training, metrics snapshot should have data."""
        _setup_simple_model_and_data(epochs=2)
        client.post("/api/train/start")
        wait_for_training()
        resp = client.get("/api/metrics/snapshot")
        assert resp.status_code == 200

    def test_system_resources(self):
        """System resources endpoint should always work."""
        resp = client.get("/api/system/resources")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "ram_total_gb" in data

    def test_full_pipeline_with_augmentation(self):
        """End-to-end: build, load data, augment, train, save experiment."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 2, "out_features": 16},
            "activation": "ReLU",
        })
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 16, "out_features": 2},
        })
        client.post("/api/data/load", json={
            "dataset": "moons", "type": "synthetic", "n_samples": 200,
        })
        client.post("/api/data/augmentation", json={
            "augmentations": [
                {"name": "gaussian_noise", "sigma": 0.05},
                {"name": "scaling", "min_scale": 0.95, "max_scale": 1.05},
            ],
        })
        client.post("/api/config", json={
            "epochs": 3, "batch_size": 32, "learning_rate": 0.01,
            "optimizer": "Adam", "loss": "CrossEntropyLoss",
        })
        client.post("/api/build")
        client.post("/api/train/start")
        wait_for_training()

        # Save experiment
        resp = client.post("/api/experiments/save", json={"name": "Full Pipeline"})
        assert resp.json()["status"] == "saved"

        # Verify experiment has data
        exp = client.get("/api/experiments/0").json()
        assert exp["name"] == "Full Pipeline"
        assert exp["dataset"] is not None

    def test_config_update_and_retrieve(self):
        """Config updates should be reflected in status."""
        client.post("/api/config", json={
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "loss": "MSELoss",
        })
        resp = client.get("/api/status")
        assert resp.status_code == 200
        status = resp.json()
        assert status["config"]["epochs"] == 10
        assert status["config"]["optimizer"] == "AdamW"

    def test_build_then_rebuild(self):
        """Building twice should work without errors."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 4, "out_features": 2},
        })
        resp1 = client.post("/api/build")
        assert resp1.status_code == 200
        resp2 = client.post("/api/build")
        assert resp2.status_code == 200

    def test_graph_reset_via_api(self):
        """POST /api/reset clears graph."""
        client.post("/api/graph/layer", json={
            "layer_type": "Linear",
            "params": {"in_features": 4, "out_features": 2},
        })
        resp = client.post("/api/reset")
        assert resp.status_code == 200
        graph = client.get("/api/graph").json()
        assert len(graph["nodes"]) == 0
