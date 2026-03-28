"""Tests for extended graph builder: branching, skip connections, new components."""

import torch
import torch.nn as nn
import pytest

from state_graph.core.graph import StateGraph, _BranchingModel
from state_graph.core.registry import Registry


class TestGraphBranching:
    def test_sequential_backward_compatible(self):
        g = StateGraph()
        g.add_layer("Linear", {"in_features": 4, "out_features": 8}, "ReLU")
        g.add_layer("Linear", {"in_features": 8, "out_features": 2})
        model = g.build_model()
        assert isinstance(model, nn.Sequential)
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 2)

    def test_skip_connection(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8}, "ReLU")
        id1 = g.add_layer("Linear", {"in_features": 8, "out_features": 8}, "ReLU")
        id2 = g.add_layer("Linear", {"in_features": 8, "out_features": 8},
                           inputs=[id0], merge_mode="add")  # skip from id0
        id3 = g.add_layer("Linear", {"in_features": 8, "out_features": 2})

        model = g.build_model()
        assert isinstance(model, _BranchingModel)
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 2)

    def test_add_skip_connection_method(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8}, "ReLU")
        id1 = g.add_layer("Linear", {"in_features": 8, "out_features": 8}, "ReLU")
        id2 = g.add_layer("Linear", {"in_features": 8, "out_features": 2})
        g.add_skip_connection(id0, id2, "add")

        assert g._has_branching()
        model = g.build_model()
        assert isinstance(model, _BranchingModel)

    def test_has_branching_false(self):
        g = StateGraph()
        g.add_layer("Linear", {"in_features": 4, "out_features": 8})
        g.add_layer("Linear", {"in_features": 8, "out_features": 2})
        assert not g._has_branching()

    def test_branching_gradient_flow(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8}, "ReLU")
        id1 = g.add_layer("Linear", {"in_features": 8, "out_features": 8}, "ReLU")
        g.add_layer("Linear", {"in_features": 8, "out_features": 2},
                     inputs=[id0], merge_mode="add")

        model = g.build_model()
        x = torch.randn(1, 4, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_branching_param_count(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8})
        g.add_layer("Linear", {"in_features": 8, "out_features": 2},
                     inputs=[id0], merge_mode="add")
        model = g.build_model()
        counts = g.get_param_count()
        assert len(counts) == 2
        for nid, c in counts.items():
            assert c["total"] > 0

    def test_remove_cleans_inputs(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8})
        id1 = g.add_layer("Linear", {"in_features": 8, "out_features": 8})
        id2 = g.add_layer("Linear", {"in_features": 8, "out_features": 2},
                           inputs=[id0], merge_mode="add")
        g.remove_layer(id0)
        # id2's inputs should be cleaned
        assert g.nodes[id2].inputs is None or id0 not in g.nodes[id2].inputs

    def test_to_dict_includes_inputs(self):
        g = StateGraph()
        id0 = g.add_layer("Linear", {"in_features": 4, "out_features": 8})
        g.add_layer("Linear", {"in_features": 8, "out_features": 2},
                     inputs=[id0], merge_mode="add")
        d = g.to_dict()
        skip_node = [n for n in d["nodes"] if "inputs" in n][0]
        assert id0 in skip_node["inputs"]
        assert skip_node["merge_mode"] == "add"


class TestNewVisionComponents:
    def test_patch_embed(self):
        from state_graph.layers.custom import PatchEmbed
        pe = PatchEmbed(in_channels=3, d_model=64, patch_size=8, image_size=32)
        img = torch.randn(2, 3, 32, 32)
        out = pe(img)
        n_patches = (32 // 8) ** 2
        assert out.shape == (2, n_patches + 1, 64)

    def test_depthwise_separable_conv(self):
        from state_graph.layers.custom import DepthwiseSeparableConv
        conv = DepthwiseSeparableConv(3, 64)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape == (1, 64, 32, 32)

    def test_channel_attention(self):
        from state_graph.layers.custom import ChannelAttention
        ca = ChannelAttention(64, reduction=16)
        x = torch.randn(1, 64, 8, 8)
        out = ca(x)
        assert out.shape == (1, 64, 8, 8)

    def test_upsample_block(self):
        from state_graph.layers.custom import UpsampleBlock
        up = UpsampleBlock(64, 32)
        x = torch.randn(1, 64, 8, 8)
        out = up(x)
        assert out.shape == (1, 32, 16, 16)

    def test_global_avg_pool(self):
        from state_graph.layers.custom import GlobalAvgPool
        gap = GlobalAvgPool()
        x = torch.randn(1, 64, 8, 8)
        out = gap(x)
        assert out.shape == (1, 64)

    def test_reshape(self):
        from state_graph.layers.custom import Reshape
        r = Reshape([8, 4, 4])
        x = torch.randn(2, 128)
        out = r(x)
        assert out.shape == (2, 8, 4, 4)

    def test_res_conv_block(self):
        from state_graph.layers.custom import ResConvBlock
        rcb = ResConvBlock(32, 64)
        x = torch.randn(1, 32, 16, 16)
        out = rcb(x)
        assert out.shape == (1, 64, 16, 16)

    def test_down_block(self):
        from state_graph.layers.custom import DownBlock
        db = DownBlock(3, 32)
        x = torch.randn(1, 3, 32, 32)
        out = db(x)
        assert out.shape == (1, 32, 16, 16)

    def test_up_block(self):
        from state_graph.layers.custom import UpBlock
        ub = UpBlock(64, 32)
        x = torch.randn(1, 64, 8, 8)
        out = ub(x)
        assert out.shape == (1, 32, 16, 16)


class TestNewAudioComponents:
    def test_mel_spectrogram(self):
        from state_graph.layers.custom import MelSpectrogram
        mel = MelSpectrogram(n_mels=40, n_fft=256, hop_length=64)
        x = torch.randn(1, 1, 1600)
        out = mel(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 40

    def test_audio_conv_block(self):
        from state_graph.layers.custom import AudioConvBlock
        acb = AudioConvBlock(40, 64, stride=2)
        x = torch.randn(1, 40, 100)
        out = acb(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 64

    def test_transpose(self):
        from state_graph.layers.custom import Transpose
        t = Transpose(1, 2)
        x = torch.randn(1, 40, 100)
        out = t(x)
        assert out.shape == (1, 100, 40)


class TestNewVideoComponents:
    def test_conv3d_block(self):
        from state_graph.layers.custom import Conv3dBlock
        cb = Conv3dBlock(3, 32)
        x = torch.randn(1, 3, 8, 16, 16)
        out = cb(x)
        assert out.shape == (1, 32, 8, 16, 16)

    def test_temporal_pool(self):
        from state_graph.layers.custom import TemporalPool
        tp = TemporalPool(mode='mean')
        x = torch.randn(1, 32, 8, 16, 16)
        out = tp(x)
        assert out.shape == (1, 32, 16, 16)


class TestDiffusionComponents:
    def test_timestep_embed(self):
        from state_graph.layers.custom import SinusoidalTimestepEmbed
        te = SinusoidalTimestepEmbed(d_model=64)
        t = torch.tensor([0, 50, 100])
        out = te(t)
        assert out.shape == (3, 64)

    def test_res_conv_block_residual(self):
        from state_graph.layers.custom import ResConvBlock
        rcb = ResConvBlock(32)
        x = torch.randn(1, 32, 16, 16)
        out = rcb(x)
        assert out.shape == (1, 32, 16, 16)
        # Should not be identity
        assert not torch.allclose(out, x, atol=1e-3)


class TestRegistryNewLayers:
    def test_vision_layers_registered(self):
        expected = ["PatchEmbed", "DepthwiseSeparableConv", "ChannelAttention",
                    "UpsampleBlock", "GlobalAvgPool", "Reshape",
                    "ResConvBlock", "DownBlock", "UpBlock"]
        layers = Registry.list_layers()
        for name in expected:
            assert name in layers, f"{name} not registered"

    def test_audio_layers_registered(self):
        expected = ["MelSpectrogram", "AudioConvBlock", "Transpose"]
        layers = Registry.list_layers()
        for name in expected:
            assert name in layers, f"{name} not registered"

    def test_video_layers_registered(self):
        expected = ["Conv3dBlock", "TemporalPool"]
        layers = Registry.list_layers()
        for name in expected:
            assert name in layers, f"{name} not registered"

    def test_diffusion_layers_registered(self):
        expected = ["SinusoidalTimestepEmbed", "ConditionalBatchNorm2d"]
        layers = Registry.list_layers()
        for name in expected:
            assert name in layers, f"{name} not registered"


# Server endpoint tests

from fastapi.testclient import TestClient
from state_graph.server.app import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


class TestGraphSkipEndpoint:
    def test_add_skip(self):
        # Build a 3-layer graph
        r1 = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 4, "out_features": 8}, "activation": "ReLU"
        })
        id0 = r1.json()["node_id"]
        r2 = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 8}, "activation": "ReLU"
        })
        r3 = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 2}
        })
        id2 = r3.json()["node_id"]

        # Add skip connection
        resp = client.post("/api/graph/skip", json={
            "from_id": id0, "to_id": id2, "merge_mode": "add"
        })
        data = resp.json()
        assert "graph" in data
        # Verify skip is in the graph
        skip_node = [n for n in data["graph"]["nodes"] if n["id"] == id2][0]
        assert id0 in skip_node.get("inputs", [])

    def test_build_with_skip(self):
        r1 = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 4, "out_features": 8}, "activation": "ReLU"
        })
        id0 = r1.json()["node_id"]
        client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 8}, "activation": "ReLU"
        })
        r3 = client.post("/api/graph/layer", json={
            "layer_type": "Linear", "params": {"in_features": 8, "out_features": 2}
        })
        id2 = r3.json()["node_id"]
        client.post("/api/graph/skip", json={"from_id": id0, "to_id": id2, "merge_mode": "add"})

        resp = client.post("/api/build")
        assert resp.status_code == 200


class TestNewTemplates:
    def test_templates_exist(self):
        resp = client.get("/api/templates")
        data = resp.json()
        assert "vit_tiny" in data
        assert "mobilenet_style" in data
        assert "audio_classifier" in data
        assert "autoencoder" in data
        assert "resnet_style" in data

    def test_vit_template(self):
        resp = client.get("/api/templates")
        vit = resp.json()["vit_tiny"]
        assert "ViT" in vit["name"]

    def test_audio_template(self):
        resp = client.get("/api/templates")
        audio = resp.json()["audio_classifier"]
        assert "Audio" in audio["name"]
