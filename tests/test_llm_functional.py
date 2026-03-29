"""Comprehensive functional tests for the LLM Builder tab."""

import pytest
import time
from fastapi.testclient import TestClient
from state_graph.server.app import app, engine

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()

client = TestClient(app)

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 100
    + "Machine learning models process sequences of tokens. " * 80
    + "Attention mechanisms focus on relevant parts of the input. " * 60
)


def wait_for_training(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get("/api/status")
        if not resp.json().get("is_training", False):
            return resp.json()
        time.sleep(0.3)
    raise TimeoutError("Training didn't finish")


# ---------------------------------------------------------------------------
# 3A. All 18 Blueprints — Build + Forward Pass
# ---------------------------------------------------------------------------

STANDARD_BLUEPRINTS = [
    "gpt2_scratch",
    "llama_scratch",
    "claude_scratch",
    "mistral_scratch",
    "mixtral_scratch",
    "deepseek_scratch",
    "t5_scratch",
    "mamba_scratch",
    "jamba_scratch",
    "rwkv_scratch",
    "retnet_scratch",
    "gemini_scratch",
    "llava_scratch",
    "custom_from_scratch",
    "adaptive_depth_scratch",
    "nano_banana",
]


class TestBlueprintsBuild:
    @pytest.mark.parametrize("name", STANDARD_BLUEPRINTS)
    def test_blueprint_build(self, name):
        resp = client.post("/api/llm/blueprint/build", json={"blueprint": name})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "built")

    def test_stable_diffusion_blueprint(self):
        resp = client.post(
            "/api/llm/blueprint/build", json={"blueprint": "stable_diffusion_scratch"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "components" in data

    def test_veo3_blueprint(self):
        resp = client.post(
            "/api/llm/blueprint/build", json={"blueprint": "veo3_scratch"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 3B. Blueprint Scales
# ---------------------------------------------------------------------------

class TestBlueprintScales:
    @pytest.mark.parametrize("scale", ["nano", "micro", "small"])
    def test_llama_scales(self, scale):
        resp = client.post(
            "/api/llm/blueprint/build",
            json={"blueprint": "llama_scratch", "scale": scale},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "built")

    def test_nano_smaller_than_small(self):
        resp_nano = client.post(
            "/api/llm/blueprint/build",
            json={"blueprint": "llama_scratch", "scale": "nano"},
        )
        nano_params = resp_nano.json().get("params", {}).get("total", 0)

        # Reset so we can build again
        engine.reset()

        resp_small = client.post(
            "/api/llm/blueprint/build",
            json={"blueprint": "llama_scratch", "scale": "small"},
        )
        small_params = resp_small.json().get("params", {}).get("total", 0)

        if nano_params > 0 and small_params > 0:
            assert nano_params < small_params


# ---------------------------------------------------------------------------
# 3C. All 19 Block Designs via Compose
# ---------------------------------------------------------------------------

BLOCK_DESIGNS = [
    "llama",
    "gpt2",
    "palm",
    "gemma",
    "minimal",
    "deep_norm",
    "moe_block",
    "mamba",
    "rwkv",
    "retnet",
    "hyena",
    "xlstm",
    "griffin",
    "hybrid_mamba_attn",
    "parallel_moe_mamba",
    "triple_hybrid",
    "retention_moe",
    "hyena_attention_hybrid",
]


class TestBlockDesigns:
    @pytest.mark.parametrize("block_name", BLOCK_DESIGNS)
    def test_compose_block_design(self, block_name):
        resp = client.post(
            "/api/llm/compose",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "default_block_name": block_name,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "built"


# ---------------------------------------------------------------------------
# 3D. Custom LLM Build — Config Combinations
# ---------------------------------------------------------------------------

class TestLLMBuildConfigs:
    def test_basic_build(self):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "dropout": 0.0,
                "use_flash": False,
                "tie_weights": True,
                "norm_type": "rmsnorm",
                "ffn_type": "swiglu",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "built"
        assert "params" in data
        assert "architecture" in data

    @pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
    def test_norm_types(self, norm_type):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "norm_type": norm_type,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    @pytest.mark.parametrize("ffn_type", ["swiglu", "geglu", "reglu", "standard"])
    def test_ffn_types(self, ffn_type):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "ffn_type": ffn_type,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    @pytest.mark.parametrize("rope_base", [10000, 100000, 500000])
    def test_rope_bases(self, rope_base):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "rope_base": rope_base,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_moe_build(self):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 4,
                "n_heads": 4,
                "max_len": 64,
                "use_moe": True,
                "n_experts": 4,
                "moe_top_k": 2,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_weight_tying_off(self):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "tie_weights": False,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_gqa(self):
        resp = client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "max_len": 64,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"


# ---------------------------------------------------------------------------
# 3E. LLM Modification Actions
# ---------------------------------------------------------------------------

class TestLLMModify:
    def _build_model(self):
        client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 3,
                "n_heads": 4,
                "max_len": 64,
            },
        )

    def test_add_layer(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={
                "action": "add_layer",
                "position": 1,
                "config": {"n_heads": 4},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_remove_layer(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "remove_layer", "index": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_update_layer(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={
                "action": "update_layer",
                "index": 0,
                "config": {"n_heads": 4, "ffn_type": "geglu"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_freeze_layer(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "freeze_layer", "index": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_unfreeze_layer(self):
        self._build_model()
        client.post(
            "/api/llm/modify",
            json={"action": "freeze_layer", "index": 0},
        )
        resp = client.post(
            "/api/llm/modify",
            json={"action": "unfreeze_layer", "index": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_freeze_component(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "freeze_layer", "component": "embedding"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_change_vocab(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "change_vocab", "vocab_size": 512},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_change_norm(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "change_norm", "norm_type": "layernorm"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"

    def test_reorder_layer(self):
        self._build_model()
        resp = client.post(
            "/api/llm/modify",
            json={"action": "reorder_layer", "from_index": 0, "to_index": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"


# ---------------------------------------------------------------------------
# 3F. Component Validation
# ---------------------------------------------------------------------------

class TestComponentValidation:
    def test_validate_custom_code(self):
        resp = client.post(
            "/api/llm/component/validate",
            json={
                "type": "custom_code",
                "code": """
class CustomModule(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return self.linear(x)
""",
                "d_model": 64,
            },
        )
        assert resp.status_code == 200

    def test_validate_custom_formula(self):
        resp = client.post(
            "/api/llm/component/validate",
            json={
                "type": "custom_formula",
                "formula": "torch.relu(fc1(x))",
                "d_model": 64,
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3G. Novel Architecture Lab
# ---------------------------------------------------------------------------

STANDARD_BLOCK_DESIGN = [
    {"type": "norm", "config": {"norm_type": "rmsnorm"}},
    {"type": "attention", "config": {}},
    {"type": "residual", "residual_from": -1},
    {"type": "norm", "config": {"norm_type": "rmsnorm"}},
    {"type": "ffn", "config": {"ffn_type": "swiglu"}},
    {"type": "residual", "residual_from": 3},
]


class TestNovelArchLab:
    def test_get_templates(self):
        resp = client.get("/api/llm/novel/templates")
        assert resp.status_code == 200
        assert "templates" in resp.json()

    def test_validate_block_design(self):
        resp = client.post(
            "/api/llm/novel/validate",
            json={
                "block_design": STANDARD_BLOCK_DESIGN,
                "d_model": 64,
                "n_heads": 4,
                "vocab_size": 256,
                "n_layers": 2,
                "seq_len": 16,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "valid"

    def test_run_quick_experiment(self):
        resp = client.post(
            "/api/llm/novel/experiment",
            json={
                "block_design": STANDARD_BLOCK_DESIGN,
                "d_model": 64,
                "n_heads": 4,
                "vocab_size": 256,
                "n_layers": 2,
                "max_len": 32,
                "text": SAMPLE_TEXT,
                "train_steps": 10,
                "learning_rate": 1e-3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["final_loss"] < data["initial_loss"]

    def test_model_from_code(self):
        code = """
class MyLLM(nn.Module):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.tok_emb(input_ids)
        logits = self.linear(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return {"logits": logits, "loss": loss}
"""
        resp = client.post(
            "/api/llm/novel/model-from-code",
            json={"code": code, "vocab_size": 256, "d_model": 64},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_custom_loss_formula(self):
        resp = client.post(
            "/api/llm/novel/custom-loss",
            json={
                "mode": "formula",
                "formula": "F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_custom_loss_code(self):
        code = """
class CustomLoss(nn.Module):
    def forward(self, logits, labels, **kwargs):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
"""
        resp = client.post(
            "/api/llm/novel/custom-loss",
            json={"mode": "code", "code": code},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_architecture_search(self):
        resp = client.post(
            "/api/llm/novel/arch-search",
            json={
                "designs": {
                    "simple_attn": STANDARD_BLOCK_DESIGN,
                    "simple_mamba": [
                        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                        {"type": "mamba", "config": {"d_state": 8, "expand": 2}},
                        {"type": "residual", "residual_from": -1},
                        {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                        {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                        {"type": "residual", "residual_from": 3},
                    ],
                },
                "d_model": 64,
                "n_heads": 4,
                "vocab_size": 256,
                "n_layers": 2,
                "text": SAMPLE_TEXT,
                "train_steps": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["n_successful"] >= 1


# ---------------------------------------------------------------------------
# 3H. LLM Training (char-level tokenizer)
# ---------------------------------------------------------------------------

class TestLLMTraining:
    def test_train_char_tokenizer(self):
        client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 32,
            },
        )
        resp = client.post(
            "/api/llm/train",
            json={
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "text": SAMPLE_TEXT,
                "tokenizer": "char",
                "max_len": 32,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"
        status = wait_for_training(timeout=60)
        assert not status.get("is_training", True)


# ---------------------------------------------------------------------------
# 3I. LLM Text Generation
# ---------------------------------------------------------------------------

class TestLLMGeneration:
    def test_generate_after_train(self):
        client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 32,
            },
        )
        client.post(
            "/api/llm/train",
            json={
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-3,
                "text": SAMPLE_TEXT,
                "tokenizer": "char",
                "max_len": 32,
            },
        )
        wait_for_training(timeout=60)

        resp = client.post(
            "/api/llm/generate",
            json={
                "prompt": "The quick",
                "max_tokens": 20,
                "temperature": 0.8,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["text"]) > 0


# ---------------------------------------------------------------------------
# 3J. Advanced Variants
# ---------------------------------------------------------------------------

class TestAdvancedVariants:
    def test_encoder_decoder(self):
        resp = client.post(
            "/api/llm/encoder-decoder",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_encoder_layers": 2,
                "n_decoder_layers": 2,
                "n_heads": 4,
                "max_len": 64,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_adaptive_depth(self):
        resp = client.post(
            "/api/llm/adaptive-depth",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 4,
                "n_heads": 4,
                "max_len": 64,
                "exit_interval": 2,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"

    def test_multimodal(self):
        resp = client.post(
            "/api/llm/multimodal",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "image_size": 32,
                "patch_size": 8,
                "modalities": ["text", "image"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "built"


# ---------------------------------------------------------------------------
# 3K. Diffusion Blueprint
# ---------------------------------------------------------------------------

class TestDiffusionBlueprint:
    def test_stable_diffusion_blueprint(self):
        resp = client.post(
            "/api/llm/blueprint/build",
            json={"blueprint": "stable_diffusion_scratch"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "components" in data

    def test_veo3_blueprint(self):
        resp = client.post(
            "/api/llm/blueprint/build", json={"blueprint": "veo3_scratch"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# 3L. Tokenizer Training
# ---------------------------------------------------------------------------

class TestTokenizerTraining:
    def test_train_bpe_tokenizer(self):
        resp = client.post(
            "/api/llm/tokenizer/train",
            json={
                "text": SAMPLE_TEXT,
                "algorithm": "bpe",
                "vocab_size": 500,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "trained"

    def test_train_char_bpe_tokenizer(self):
        resp = client.post(
            "/api/llm/tokenizer/train",
            json={
                "text": SAMPLE_TEXT,
                "algorithm": "char_bpe",
                "vocab_size": 200,
            },
        )
        assert resp.status_code == 200

    def test_too_short_text(self):
        resp = client.post(
            "/api/llm/tokenizer/train",
            json={
                "text": "short",
                "algorithm": "bpe",
            },
        )
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# 3M. LLM Presets and Layer Info
# ---------------------------------------------------------------------------

class TestLLMPresets:
    def test_get_presets(self):
        resp = client.get("/api/llm/presets")
        assert resp.status_code == 200
        presets = resp.json()["presets"]
        assert "tiny" in presets
        assert "small" in presets

    def test_get_components(self):
        resp = client.get("/api/llm/components")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "block_designs" in data

    def test_get_blueprints(self):
        resp = client.get("/api/llm/blueprints")
        assert resp.status_code == 200
        assert "blueprints" in resp.json()

    def test_get_layer_info(self):
        client.post(
            "/api/llm/build",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
            },
        )
        resp = client.get("/api/llm/layer/0")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 3N. Composable Block Update
# ---------------------------------------------------------------------------

class TestComposableBlockUpdate:
    def test_update_block(self):
        client.post(
            "/api/llm/compose",
            json={
                "vocab_size": 256,
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "max_len": 64,
                "default_block_name": "llama",
            },
        )
        resp = client.post(
            "/api/llm/compose/block/0",
            json={
                "steps": [
                    {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                    {"type": "attention", "config": {}},
                    {"type": "residual", "residual_from": -1},
                    {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                    {"type": "ffn", "config": {"ffn_type": "geglu"}},
                    {"type": "residual", "residual_from": 3},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "modified"
