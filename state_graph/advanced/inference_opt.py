"""Inference optimization — TensorRT, ONNX Runtime, quantization, benchmarking.

Optimize trained models for production deployment speed.
"""

from __future__ import annotations

import os
import time
from typing import Any


OPTIMIZATION_METHODS = {
    "onnx_runtime": {
        "name": "ONNX Runtime",
        "description": "Cross-platform optimized inference. 2-5x speedup over PyTorch.",
        "pip": "onnxruntime-gpu",
        "supports": ["CPU", "CUDA", "TensorRT", "DirectML"],
    },
    "tensorrt": {
        "name": "NVIDIA TensorRT",
        "description": "NVIDIA GPU-optimized inference. Up to 10x speedup.",
        "pip": "tensorrt",
        "supports": ["CUDA"],
    },
    "torch_compile": {
        "name": "torch.compile",
        "description": "PyTorch 2.0 compiler. Zero-effort 1.5-3x speedup.",
        "pip": None,
        "supports": ["CPU", "CUDA"],
    },
    "quantization_dynamic": {
        "name": "Dynamic Quantization (INT8)",
        "description": "Quantize weights to INT8 at runtime. 2-4x smaller, faster on CPU.",
        "pip": None,
        "supports": ["CPU"],
    },
    "quantization_static": {
        "name": "Static Quantization (INT8)",
        "description": "Calibrated INT8. Best accuracy/speed tradeoff.",
        "pip": None,
        "supports": ["CPU", "CUDA"],
    },
    "quantization_gptq": {
        "name": "GPTQ (4-bit LLM)",
        "description": "4-bit weight quantization for LLMs.",
        "pip": "auto-gptq",
        "supports": ["CUDA"],
    },
    "quantization_awq": {
        "name": "AWQ (4-bit LLM)",
        "description": "Activation-aware weight quantization. Better quality than GPTQ.",
        "pip": "autoawq",
        "supports": ["CUDA"],
    },
    "bettertransformer": {
        "name": "BetterTransformer",
        "description": "Optimized attention for HF transformers. Free speedup.",
        "pip": "optimum",
        "supports": ["CPU", "CUDA"],
    },
    "openvino": {
        "name": "OpenVINO",
        "description": "Intel hardware optimization (CPU, iGPU, VPU).",
        "pip": "openvino",
        "supports": ["CPU", "Intel GPU"],
    },
    "coreml": {
        "name": "CoreML",
        "description": "Apple Silicon optimization (M1/M2/M3).",
        "pip": "coremltools",
        "supports": ["Apple Neural Engine", "Apple GPU"],
    },
}


def benchmark_model(model, dummy_input, n_runs: int = 100, warmup: int = 10) -> dict:
    """Benchmark PyTorch model latency."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input.to(device))

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy_input.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return {
        "mean_ms": round(sum(times) / len(times), 3),
        "median_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
        "p99_ms": round(times[int(len(times) * 0.99)], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "throughput_per_sec": round(1000 / (sum(times) / len(times)), 1),
        "n_runs": n_runs,
        "device": str(device),
    }


def benchmark_onnx(onnx_path: str, dummy_input_shape: list, n_runs: int = 100) -> dict:
    """Benchmark ONNX Runtime inference."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    dummy = np.random.randn(*dummy_input_shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return {
        "engine": "onnxruntime",
        "providers": session.get_providers(),
        "mean_ms": round(sum(times) / len(times), 3),
        "median_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[int(len(times) * 0.95)], 3),
        "throughput_per_sec": round(1000 / (sum(times) / len(times)), 1),
    }


def quantize_dynamic(model, output_path: str = "./quantized_model.pt") -> dict:
    """Apply dynamic INT8 quantization."""
    import torch
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), output_path)

    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quant_size = sum(p.numel() * p.element_size() for p in quantized.parameters())

    return {
        "status": "quantized",
        "method": "dynamic_int8",
        "original_size_mb": round(orig_size / 1e6, 2),
        "quantized_size_mb": round(quant_size / 1e6, 2),
        "compression_ratio": round(orig_size / max(quant_size, 1), 2),
        "path": output_path,
    }


def generate_optimization_script(method: str, model_path: str = "./model.pt", params: dict = None) -> str:
    """Generate optimization script for the IDE."""
    params = params or {}

    if method == "torch_compile":
        return f'''"""Optimize with torch.compile (PyTorch 2.0+)"""
import torch

model = torch.load("{model_path}")
compiled = torch.compile(model, mode="{params.get('mode', 'reduce-overhead')}")

# Benchmark
x = torch.randn(1, {params.get('input_dim', 784)})
import time
times = []
for _ in range(100):
    start = time.perf_counter()
    compiled(x)
    times.append((time.perf_counter() - start) * 1000)
print(f"Mean: {{sum(times)/len(times):.2f}}ms, Median: {{sorted(times)[50]:.2f}}ms")
'''

    elif method == "onnx_runtime":
        return f'''"""Optimize with ONNX Runtime"""
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("{model_path}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(f"Providers: {{session.get_providers()}}")

# Benchmark
x = np.random.randn(1, {params.get('input_dim', 784)}).astype(np.float32)
import time
times = []
for _ in range(100):
    start = time.perf_counter()
    session.run(None, {{session.get_inputs()[0].name: x}})
    times.append((time.perf_counter() - start) * 1000)
print(f"Mean: {{sum(times)/len(times):.2f}}ms, Throughput: {{1000/(sum(times)/len(times)):.0f}}/sec")
'''

    return f"# {method} optimization — see docs for setup\n"
