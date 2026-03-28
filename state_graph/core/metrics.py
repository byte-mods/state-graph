"""Metrics collector for real-time training monitoring.

Collects per-layer gradient stats, weight distributions, loss curves,
learning rate schedules, per-layer timing, activation distributions,
memory tracking, and bottleneck detection. Broadcasts updates via WebSocket.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class LayerMetrics:
    """Metrics tracked per layer per step."""

    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_norm: float = 0.0
    grad_mean: float = 0.0
    grad_std: float = 0.0
    grad_norm: float = 0.0
    activation_mean: float = 0.0
    activation_std: float = 0.0
    # Extended metrics
    activation_min: float = 0.0
    activation_max: float = 0.0
    activation_abs_max: float = 0.0
    dead_neuron_pct: float = 0.0  # % of neurons with zero activation
    forward_time_ms: float = 0.0  # Forward pass time for this layer
    memory_mb: float = 0.0  # Parameter memory in MB
    grad_to_weight_ratio: float = 0.0  # grad_norm / weight_norm — healthy ≈ 0.001-0.01

    def to_dict(self) -> dict:
        return {
            "weight_mean": self.weight_mean,
            "weight_std": self.weight_std,
            "weight_norm": self.weight_norm,
            "grad_mean": self.grad_mean,
            "grad_std": self.grad_std,
            "grad_norm": self.grad_norm,
            "activation_mean": self.activation_mean,
            "activation_std": self.activation_std,
            "activation_min": self.activation_min,
            "activation_max": self.activation_max,
            "activation_abs_max": self.activation_abs_max,
            "dead_neuron_pct": self.dead_neuron_pct,
            "forward_time_ms": self.forward_time_ms,
            "memory_mb": self.memory_mb,
            "grad_to_weight_ratio": self.grad_to_weight_ratio,
        }


class MetricsCollector:
    """Collects and stores training metrics for visualization."""

    def __init__(self, max_history: int = 5000) -> None:
        self.max_history = max_history
        self._loss_history: list[dict[str, float]] = []
        self._lr_history: list[dict[str, float]] = []
        self._layer_history: dict[str, list[dict]] = defaultdict(list)
        self._epoch_metrics: list[dict[str, Any]] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._activation_cache: dict[str, torch.Tensor] = {}
        self._timing_cache: dict[str, float] = {}  # Per-layer forward time
        self._listeners: list[Callable] = []
        self._step = 0
        self._step_start_time: float = 0.0
        self._tokens_processed: int = 0
        # Bottleneck analysis
        self._bottleneck_history: list[dict] = []

    def add_listener(self, fn: Callable) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Callable) -> None:
        self._listeners.remove(fn)

    async def _notify(self, event: str, data: dict) -> None:
        for fn in self._listeners:
            try:
                await fn(event, data)
            except Exception:
                pass

    def attach_hooks(self, model: nn.Module) -> None:
        """Attach forward hooks to capture activations AND timing per layer."""
        self.detach_hooks()
        self._activation_cache.clear()
        self._timing_cache.clear()

        for name, module in model.named_modules():
            if name == "":
                continue
            # Forward pre-hook: record start time
            pre_hook = module.register_forward_pre_hook(self._make_timing_pre_hook(name))
            self._hooks.append(pre_hook)
            # Forward hook: record activation + end time
            hook = module.register_forward_hook(self._make_activation_hook(name))
            self._hooks.append(hook)

    def detach_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_timing_pre_hook(self, layer_name: str):
        def hook(module, input):
            self._timing_cache[f"{layer_name}_start"] = time.perf_counter()
        return hook

    def _make_activation_hook(self, layer_name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._activation_cache[layer_name] = output.detach()
            # Record timing
            start_key = f"{layer_name}_start"
            if start_key in self._timing_cache:
                elapsed = (time.perf_counter() - self._timing_cache[start_key]) * 1000  # ms
                self._timing_cache[layer_name] = elapsed
        return hook

    def mark_step_start(self) -> None:
        """Call at the beginning of each training step to track step time."""
        self._step_start_time = time.perf_counter()

    def add_tokens(self, n_tokens: int) -> None:
        """Track tokens processed (for throughput calculation)."""
        self._tokens_processed += n_tokens

    def collect_step(
        self,
        model: nn.Module,
        loss_value: float,
        optimizer: torch.optim.Optimizer,
        extra: dict[str, float] | None = None,
    ) -> dict:
        """Collect metrics for a single training step."""
        self._step += 1
        timestamp = time.time()
        step_elapsed = (time.perf_counter() - self._step_start_time) * 1000 if self._step_start_time else 0

        # Loss
        loss_entry = {"step": self._step, "loss": loss_value, "time": timestamp,
                      "step_time_ms": step_elapsed}
        if extra:
            loss_entry.update(extra)
        self._loss_history.append(loss_entry)

        # Learning rate
        lr_entry = {"step": self._step}
        for i, pg in enumerate(optimizer.param_groups):
            lr_entry[f"group_{i}"] = pg["lr"]
        self._lr_history.append(lr_entry)

        # Per-layer metrics with extended data
        layer_metrics = {}
        bottleneck_candidates = []

        for name, module in model.named_modules():
            if name == "" or not list(module.parameters(recurse=False)):
                continue

            metrics = LayerMetrics()

            # Weight stats
            for pname, param in module.named_parameters(recurse=False):
                if param.data is not None:
                    metrics.weight_mean = param.data.mean().item()
                    metrics.weight_std = param.data.std().item()
                    metrics.weight_norm = param.data.norm().item()
                    metrics.memory_mb = param.nelement() * param.element_size() / (1024 * 1024)
                if param.grad is not None:
                    metrics.grad_mean = param.grad.mean().item()
                    metrics.grad_std = param.grad.std().item()
                    metrics.grad_norm = param.grad.norm().item()

            # Grad-to-weight ratio
            if metrics.weight_norm > 0:
                metrics.grad_to_weight_ratio = metrics.grad_norm / metrics.weight_norm

            # Activation stats (extended)
            if name in self._activation_cache:
                act = self._activation_cache[name]
                metrics.activation_mean = act.mean().item()
                metrics.activation_std = act.std().item()
                metrics.activation_min = act.min().item()
                metrics.activation_max = act.max().item()
                metrics.activation_abs_max = act.abs().max().item()
                # Dead neuron detection
                if act.dim() >= 2:
                    dead = (act.abs() < 1e-6).float().mean().item() * 100
                    metrics.dead_neuron_pct = dead

            # Per-layer timing
            if name in self._timing_cache:
                metrics.forward_time_ms = self._timing_cache[name]

            entry = {"step": self._step, **metrics.to_dict()}
            self._layer_history[name].append(entry)
            layer_metrics[name] = metrics.to_dict()

            # Collect bottleneck data
            bottleneck_candidates.append({
                "name": name,
                "forward_time_ms": metrics.forward_time_ms,
                "memory_mb": metrics.memory_mb,
                "grad_norm": metrics.grad_norm,
                "dead_neuron_pct": metrics.dead_neuron_pct,
                "grad_to_weight_ratio": metrics.grad_to_weight_ratio,
            })

        # Bottleneck analysis
        bottleneck = self._analyze_bottlenecks(bottleneck_candidates)

        # Trim history
        if len(self._loss_history) > self.max_history:
            self._loss_history = self._loss_history[-self.max_history:]
        for key in self._layer_history:
            if len(self._layer_history[key]) > self.max_history:
                self._layer_history[key] = self._layer_history[key][-self.max_history:]

        # Throughput
        throughput = {}
        if step_elapsed > 0:
            throughput["step_time_ms"] = step_elapsed
            throughput["steps_per_sec"] = 1000.0 / step_elapsed
        if self._tokens_processed > 0 and step_elapsed > 0:
            throughput["tokens_per_sec"] = self._tokens_processed / (step_elapsed / 1000.0)
            self._tokens_processed = 0  # Reset per step

        # Memory snapshot
        memory = self._get_memory_snapshot()

        return {
            "step": self._step,
            "loss": loss_value,
            "lr": lr_entry,
            "layers": layer_metrics,
            "throughput": throughput,
            "memory": memory,
            "bottleneck": bottleneck,
        }

    def _analyze_bottlenecks(self, candidates: list[dict]) -> dict:
        """Analyze per-layer data to detect bottlenecks and issues."""
        if not candidates:
            return {}

        issues = []

        # Find slowest layer
        by_time = sorted(candidates, key=lambda c: c["forward_time_ms"], reverse=True)
        if by_time and by_time[0]["forward_time_ms"] > 0:
            total_time = sum(c["forward_time_ms"] for c in candidates)
            slowest = by_time[0]
            if total_time > 0:
                pct = (slowest["forward_time_ms"] / total_time) * 100
                if pct > 30:  # One layer takes >30% of total time
                    issues.append({
                        "type": "slow_layer",
                        "severity": "warning" if pct < 50 else "critical",
                        "layer": slowest["name"],
                        "detail": f"Takes {pct:.0f}% of forward pass ({slowest['forward_time_ms']:.1f}ms)",
                    })

        # Find largest memory consumer
        by_mem = sorted(candidates, key=lambda c: c["memory_mb"], reverse=True)
        if by_mem and by_mem[0]["memory_mb"] > 0:
            total_mem = sum(c["memory_mb"] for c in candidates)
            if total_mem > 0:
                largest = by_mem[0]
                pct = (largest["memory_mb"] / total_mem) * 100
                if pct > 40:
                    issues.append({
                        "type": "memory_hog",
                        "severity": "info",
                        "layer": largest["name"],
                        "detail": f"Uses {pct:.0f}% of parameter memory ({largest['memory_mb']:.1f}MB)",
                    })

        # Detect vanishing gradients
        for c in candidates:
            if c["grad_norm"] > 0 and c["grad_norm"] < 1e-7:
                issues.append({
                    "type": "vanishing_gradient",
                    "severity": "critical",
                    "layer": c["name"],
                    "detail": f"Gradient norm = {c['grad_norm']:.2e} — effectively zero",
                })
            elif c["grad_norm"] > 100:
                issues.append({
                    "type": "exploding_gradient",
                    "severity": "critical",
                    "layer": c["name"],
                    "detail": f"Gradient norm = {c['grad_norm']:.1f} — too large",
                })

        # Detect dead neurons
        for c in candidates:
            if c["dead_neuron_pct"] > 50:
                issues.append({
                    "type": "dead_neurons",
                    "severity": "warning",
                    "layer": c["name"],
                    "detail": f"{c['dead_neuron_pct']:.0f}% of neurons are dead (activation ≈ 0)",
                })

        # Detect unhealthy grad-to-weight ratio
        for c in candidates:
            ratio = c["grad_to_weight_ratio"]
            if ratio > 1.0:
                issues.append({
                    "type": "high_grad_ratio",
                    "severity": "warning",
                    "layer": c["name"],
                    "detail": f"Grad/weight ratio = {ratio:.3f} — gradients may be too large relative to weights",
                })

        return {
            "issues": issues[:10],  # Cap at 10 issues
            "slowest_layer": by_time[0]["name"] if by_time and by_time[0]["forward_time_ms"] > 0 else None,
            "largest_memory": by_mem[0]["name"] if by_mem and by_mem[0]["memory_mb"] > 0 else None,
            "total_forward_ms": sum(c["forward_time_ms"] for c in candidates),
            "total_memory_mb": sum(c["memory_mb"] for c in candidates),
        }

    def _get_memory_snapshot(self) -> dict:
        """Get current memory usage."""
        import psutil
        proc = psutil.Process()
        mem = {
            "process_ram_mb": proc.memory_info().rss / (1024 * 1024),
        }
        if torch.cuda.is_available():
            mem["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            mem["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            mem["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mem["gpu_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
        return mem

    def collect_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
        train_acc: float | None = None,
        val_acc: float | None = None,
    ) -> dict:
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "time": time.time(),
        }
        self._epoch_metrics.append(entry)
        return entry

    def profile_model(self, model: nn.Module, input_shape: tuple,
                      n_runs: int = 10, device: str = "cpu") -> dict:
        """Profile a model: per-layer timing, memory, FLOPS estimation.

        Returns detailed per-layer profiling data.
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device(device)
        x = torch.randn(*input_shape, device=device)

        # Attach timing hooks
        timing_data: dict[str, list[float]] = defaultdict(list)
        hooks = []

        for name, module in model.named_modules():
            if name == "" or not list(module.parameters(recurse=False)):
                continue

            def make_hooks(n):
                times = []
                def pre_hook(mod, inp):
                    times.append(time.perf_counter())
                def post_hook(mod, inp, out):
                    if times:
                        elapsed = (time.perf_counter() - times.pop()) * 1000
                        timing_data[n].append(elapsed)
                return pre_hook, post_hook

            pre_h, post_h = make_hooks(name)
            hooks.append(module.register_forward_pre_hook(pre_h))
            hooks.append(module.register_forward_hook(post_h))

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(x)

        # Benchmark
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)

        # Cleanup hooks
        for h in hooks:
            h.remove()

        # Compute results
        layers = []
        total_time = 0.0
        for name, module in model.named_modules():
            if name not in timing_data:
                continue
            times = timing_data[name]
            avg_ms = sum(times) / len(times) if times else 0
            total_time += avg_ms
            params = sum(p.numel() for p in module.parameters(recurse=False))
            mem_mb = sum(p.nelement() * p.element_size() for p in module.parameters(recurse=False)) / (1024 * 1024)

            layers.append({
                "name": name,
                "type": type(module).__name__,
                "avg_time_ms": round(avg_ms, 3),
                "params": params,
                "memory_mb": round(mem_mb, 3),
            })

        # Sort by time (slowest first)
        layers.sort(key=lambda l: l["avg_time_ms"], reverse=True)

        # Add percentage
        for layer in layers:
            layer["time_pct"] = round((layer["avg_time_ms"] / total_time * 100) if total_time > 0 else 0, 1)

        # Total model stats
        total_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model(x)
        total_model_ms = (time.perf_counter() - total_start) * 1000 / n_runs

        return {
            "layers": layers,
            "total_forward_ms": round(total_model_ms, 3),
            "total_params": sum(p.numel() for p in model.parameters()),
            "input_shape": list(input_shape),
            "n_runs": n_runs,
            "device": str(device),
        }

    def get_attention_weights(self, model: nn.Module, input_ids: torch.Tensor) -> dict:
        """Extract attention weights from transformer models for visualization."""
        attention_maps = {}
        hooks = []

        for name, module in model.named_modules():
            if hasattr(module, 'n_heads') or 'attn' in name.lower() or 'attention' in name.lower():
                def make_hook(n):
                    def hook(mod, inp, out):
                        # Try to capture attention weights
                        if isinstance(out, tuple) and len(out) >= 2:
                            attn_w = out[1]
                            if attn_w is not None and isinstance(attn_w, torch.Tensor):
                                attention_maps[n] = attn_w.detach().cpu()
                    return hook
                hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        # Convert to serializable format
        result = {}
        for name, weights in attention_maps.items():
            result[name] = {
                "shape": list(weights.shape),
                "mean_attention": weights.mean(dim=1)[0].tolist() if weights.dim() >= 3 else [],
            }

        return result

    def get_loss_history(self, last_n: int | None = None) -> list[dict]:
        if last_n:
            return self._loss_history[-last_n:]
        return self._loss_history

    def get_layer_history(self, layer_name: str, last_n: int | None = None) -> list[dict]:
        history = self._layer_history.get(layer_name, [])
        if last_n:
            return history[-last_n:]
        return history

    def get_all_layer_names(self) -> list[str]:
        return list(self._layer_history.keys())

    def get_epoch_metrics(self) -> list[dict]:
        return self._epoch_metrics

    def get_snapshot(self) -> dict:
        """Get current state snapshot for UI initialization."""
        return {
            "step": self._step,
            "loss_history": self._loss_history[-200:],
            "lr_history": self._lr_history[-200:],
            "layer_names": self.get_all_layer_names(),
            "layer_history": {
                name: entries[-200:]
                for name, entries in self._layer_history.items()
            },
            "epoch_metrics": self._epoch_metrics,
        }

    def reset(self) -> None:
        self.detach_hooks()
        self._loss_history.clear()
        self._lr_history.clear()
        self._layer_history.clear()
        self._epoch_metrics.clear()
        self._activation_cache.clear()
        self._timing_cache.clear()
        self._bottleneck_history.clear()
        self._step = 0
        self._tokens_processed = 0
