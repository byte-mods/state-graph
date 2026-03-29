"""Registry for layers, activations, loss functions, and optimizers.

Provides a plug-and-play system where researchers can register custom
components and have them immediately available in the UI.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.nn as nn


class Registry:
    """Central registry for all pluggable components."""

    _layers: dict[str, type[nn.Module]] = {}
    _activations: dict[str, type[nn.Module] | Callable] = {}
    _losses: dict[str, type[nn.Module] | Callable] = {}
    _optimizers: dict[str, type[torch.optim.Optimizer]] = {}
    _custom_formulas: dict[str, Callable] = {}
    _layer_categories: dict[str, list[str]] = {}

    @classmethod
    def reset(cls) -> None:
        cls._layers = {}
        cls._activations = {}
        cls._losses = {}
        cls._optimizers = {}
        cls._custom_formulas = {}
        cls._layer_categories = {}
        cls._register_defaults()

    @classmethod
    def _register_defaults(cls) -> None:
        # Built-in layers (with categories)
        _core = {
            "Linear": nn.Linear,
            "Conv1d": nn.Conv1d,
            "Conv2d": nn.Conv2d,
            "Conv3d": nn.Conv3d,
            "ConvTranspose2d": nn.ConvTranspose2d,
            "BatchNorm1d": nn.BatchNorm1d,
            "BatchNorm2d": nn.BatchNorm2d,
            "GroupNorm": nn.GroupNorm,
            "InstanceNorm2d": nn.InstanceNorm2d,
            "LayerNorm": nn.LayerNorm,
            "Dropout": nn.Dropout,
            "Dropout2d": nn.Dropout2d,
            "Embedding": nn.Embedding,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
            "MultiheadAttention": nn.MultiheadAttention,
            "Flatten": nn.Flatten,
            "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
            "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
            "MaxPool1d": nn.MaxPool1d,
            "MaxPool2d": nn.MaxPool2d,
            "AvgPool2d": nn.AvgPool2d,
        }
        cls._layers.update(_core)
        for name in _core:
            cls.register_layer(name, _core[name], category="Core")

        # Built-in activations
        cls._activations.update({
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "Softmax": nn.Softmax,
            "ELU": nn.ELU,
            "PReLU": nn.PReLU,
            "Mish": nn.Mish,
        })

        # Built-in losses
        cls._losses.update({
            "CrossEntropyLoss": nn.CrossEntropyLoss,
            "MSELoss": nn.MSELoss,
            "L1Loss": nn.L1Loss,
            "BCELoss": nn.BCELoss,
            "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
            "NLLLoss": nn.NLLLoss,
            "SmoothL1Loss": nn.SmoothL1Loss,
            "HuberLoss": nn.HuberLoss,
            "KLDivLoss": nn.KLDivLoss,
            "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
            "CTCLoss": nn.CTCLoss,
            "TripletMarginLoss": nn.TripletMarginLoss,
            "MultiMarginLoss": nn.MultiMarginLoss,
            "MarginRankingLoss": nn.MarginRankingLoss,
        })

        # Built-in optimizers
        cls._optimizers.update({
            "SGD": torch.optim.SGD,
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "RMSprop": torch.optim.RMSprop,
            "Adagrad": torch.optim.Adagrad,
        })

        cls._register_custom_layers()

    @classmethod
    def _register_custom_layers(cls) -> None:
        """Register custom layers, LLM layers, and advanced components.

        Uses lazy imports to avoid loading heavy dependencies at startup.
        """
        try:
            from state_graph.layers.custom import (
                ResidualBlock, SqueezeExcite, GatedLinearUnit, SwishLinear,
                TransformerBlock, PositionalEncoding, TokenEmbedding, SequencePool,
                PatchEmbed, DepthwiseSeparableConv, ChannelAttention,
                UpsampleBlock, GlobalAvgPool, Reshape,
                MelSpectrogram, AudioConvBlock, Transpose,
                Conv3dBlock, TemporalPool,
                SinusoidalTimestepEmbed, ConditionalBatchNorm2d,
                ResConvBlock, DownBlock, UpBlock,
                SelectiveScan, MambaBlock, RWKVBlock, RetentionLayer, RetNetBlock,
                HyenaOperator, HyenaBlock, SLSTMCell, XLSTM, GatedLinearRecurrence,
                # New components
                ResNetBlock, ConvNeXtBlock, MBConvBlock, VisionEncoder,
                DiffusionTimestepBlock, SpatialAttentionBlock, CrossAttentionBlock,
                DiffusionUNet, VAE,
                TemporalAttention, TemporalConv3d, VideoVAE,
                PerceiverResampler, DistillationWrapper,
            )

            # Transformer & Building Blocks
            _transformer = {
                "TransformerBlock": TransformerBlock,
                "PositionalEncoding": PositionalEncoding,
                "TokenEmbedding": TokenEmbedding,
                "SequencePool": SequencePool,
                "ResidualBlock": ResidualBlock,
                "SqueezeExcite": SqueezeExcite,
                "GatedLinearUnit": GatedLinearUnit,
                "SwishLinear": SwishLinear,
            }
            for name, layer in _transformer.items():
                cls.register_layer(name, layer, category="Transformer")

            # Vision
            _vision = {
                "PatchEmbed": PatchEmbed,
                "DepthwiseSeparableConv": DepthwiseSeparableConv,
                "ChannelAttention": ChannelAttention,
                "UpsampleBlock": UpsampleBlock,
                "GlobalAvgPool": GlobalAvgPool,
                "Reshape": Reshape,
                "ResNetBlock": ResNetBlock,
                "ConvNeXtBlock": ConvNeXtBlock,
                "MBConvBlock": MBConvBlock,
                "VisionEncoder": VisionEncoder,
            }
            for name, layer in _vision.items():
                cls.register_layer(name, layer, category="Vision")

            # Audio
            _audio = {
                "MelSpectrogram": MelSpectrogram,
                "AudioConvBlock": AudioConvBlock,
                "Transpose": Transpose,
            }
            for name, layer in _audio.items():
                cls.register_layer(name, layer, category="Audio")

            # Video
            _video = {
                "Conv3dBlock": Conv3dBlock,
                "TemporalPool": TemporalPool,
                "TemporalAttention": TemporalAttention,
                "TemporalConv3d": TemporalConv3d,
                "VideoVAE": VideoVAE,
            }
            for name, layer in _video.items():
                cls.register_layer(name, layer, category="Video")

            # Diffusion
            _diffusion = {
                "SinusoidalTimestepEmbed": SinusoidalTimestepEmbed,
                "ConditionalBatchNorm2d": ConditionalBatchNorm2d,
                "ResConvBlock": ResConvBlock,
                "DownBlock": DownBlock,
                "UpBlock": UpBlock,
                "DiffusionTimestepBlock": DiffusionTimestepBlock,
                "SpatialAttentionBlock": SpatialAttentionBlock,
                "CrossAttentionBlock": CrossAttentionBlock,
                "DiffusionUNet": DiffusionUNet,
                "VAE": VAE,
            }
            for name, layer in _diffusion.items():
                cls.register_layer(name, layer, category="Diffusion")

            # State-Space Models
            _ssm = {
                "SelectiveScan": SelectiveScan,
                "MambaBlock": MambaBlock,
                "RWKVBlock": RWKVBlock,
                "RetentionLayer": RetentionLayer,
                "RetNetBlock": RetNetBlock,
                "HyenaOperator": HyenaOperator,
                "HyenaBlock": HyenaBlock,
                "SLSTMCell": SLSTMCell,
                "XLSTM": XLSTM,
                "GatedLinearRecurrence": GatedLinearRecurrence,
            }
            for name, layer in _ssm.items():
                cls.register_layer(name, layer, category="SSM")

            # Multimodal & Training
            _multimodal = {
                "PerceiverResampler": PerceiverResampler,
                "DistillationWrapper": DistillationWrapper,
            }
            for name, layer in _multimodal.items():
                cls.register_layer(name, layer, category="Multimodal")

        except ImportError:
            pass  # Custom layers not available

        try:
            from state_graph.layers.llm import (
                RMSNorm, LLMAttention, SwiGLUFFN, GeGLUFFN, ReGLUFFN, StandardFFN,
                MoELayer, LLMDecoderBlock, LLMModel,
                SlidingWindowAttention, LinearAttention, ALiBiAttention,
                ComposableBlock, ComposableLLM, ParallelBranch,
                EncoderBlock, DecoderBlockWithCrossAttn, EncoderDecoderLLM,
                EarlyExitClassifier, AdaptiveDepthLLM,
                PatchEmbedding, AudioEmbedding, ModalityProjector,
                MultiModalLLM, VideoEmbedding, UnifiedMultiModalLLM,
            )

            _llm = {
                "RMSNorm": RMSNorm,
                "LLMAttention": LLMAttention,
                "SwiGLUFFN": SwiGLUFFN,
                "GeGLUFFN": GeGLUFFN,
                "ReGLUFFN": ReGLUFFN,
                "StandardFFN": StandardFFN,
                "MoELayer": MoELayer,
                "LLMDecoderBlock": LLMDecoderBlock,
                "LLMModel": LLMModel,
                "SlidingWindowAttention": SlidingWindowAttention,
                "LinearAttention": LinearAttention,
                "ALiBiAttention": ALiBiAttention,
                "ComposableBlock": ComposableBlock,
                "ComposableLLM": ComposableLLM,
                "ParallelBranch": ParallelBranch,
                "EncoderBlock": EncoderBlock,
                "DecoderBlockWithCrossAttn": DecoderBlockWithCrossAttn,
                "EncoderDecoderLLM": EncoderDecoderLLM,
                "EarlyExitClassifier": EarlyExitClassifier,
                "AdaptiveDepthLLM": AdaptiveDepthLLM,
                "PatchEmbedding": PatchEmbedding,
                "AudioEmbedding": AudioEmbedding,
                "ModalityProjector": ModalityProjector,
                "MultiModalLLM": MultiModalLLM,
                "VideoEmbedding": VideoEmbedding,
                "UnifiedMultiModalLLM": UnifiedMultiModalLLM,
            }
            for name, layer in _llm.items():
                cls.register_layer(name, layer, category="LLM")

        except ImportError:
            pass  # LLM layers not available

    @classmethod
    def register_layer(cls, name: str, layer_cls: type[nn.Module], category: str = "General") -> None:
        cls._layers[name] = layer_cls
        if category not in cls._layer_categories:
            cls._layer_categories[category] = []
        if name not in cls._layer_categories[category]:
            cls._layer_categories[category].append(name)

    @classmethod
    def register_activation(cls, name: str, act: type[nn.Module] | Callable) -> None:
        cls._activations[name] = act

    @classmethod
    def register_loss(cls, name: str, loss: type[nn.Module] | Callable) -> None:
        cls._losses[name] = loss

    @classmethod
    def register_optimizer(cls, name: str, opt: type[torch.optim.Optimizer]) -> None:
        cls._optimizers[name] = opt

    @classmethod
    def register_formula(cls, name: str, formula_fn: Callable) -> None:
        """Register a custom formula as an activation or transform.

        The formula_fn should accept a tensor and return a tensor.
        It gets wrapped into an nn.Module automatically.
        """
        cls._custom_formulas[name] = formula_fn

        class FormulaModule(nn.Module):
            def __init__(self):
                super().__init__()
                self._name = name

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return formula_fn(x)

            def __repr__(self) -> str:
                return f"Formula({self._name})"

        cls._activations[name] = FormulaModule

    @classmethod
    def register_formula_from_string(cls, name: str, expr: str) -> None:
        """Register a formula from a math expression string.

        Supports: x, torch.*, math.*, abs, min, max, pow, sqrt, exp, log, sin, cos, tan
        Example: "torch.clamp(x * 0.5 + 0.5, 0, 1)"
        """
        safe_globals = {
            "torch": torch,
            "math": math,
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
        }

        def formula_fn(x: torch.Tensor) -> torch.Tensor:
            return eval(expr, safe_globals, {"x": x})  # noqa: S307

        cls.register_formula(name, formula_fn)

    @classmethod
    def get_layer(cls, name: str) -> type[nn.Module]:
        return cls._layers[name]

    @classmethod
    def get_activation(cls, name: str) -> type[nn.Module] | Callable:
        return cls._activations[name]

    @classmethod
    def get_loss(cls, name: str) -> type[nn.Module] | Callable:
        return cls._losses[name]

    @classmethod
    def get_optimizer(cls, name: str) -> type[torch.optim.Optimizer]:
        return cls._optimizers[name]

    @classmethod
    def list_layers(cls) -> list[str]:
        return sorted(cls._layers.keys())

    @classmethod
    def list_activations(cls) -> list[str]:
        return sorted(cls._activations.keys())

    @classmethod
    def list_losses(cls) -> list[str]:
        return sorted(cls._losses.keys())

    @classmethod
    def list_optimizers(cls) -> list[str]:
        return sorted(cls._optimizers.keys())

    @classmethod
    def list_layer_categories(cls) -> dict[str, list[str]]:
        """Return layers organized by category."""
        return {cat: sorted(names) for cat, names in cls._layer_categories.items()}

    @classmethod
    def list_all(cls) -> dict[str, list[str] | dict]:
        return {
            "layers": cls.list_layers(),
            "activations": cls.list_activations(),
            "losses": cls.list_losses(),
            "optimizers": cls.list_optimizers(),
            "layer_categories": cls.list_layer_categories(),
        }


# Initialize defaults on import
Registry._register_defaults()
