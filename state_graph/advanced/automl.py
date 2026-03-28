"""AutoML — Neural Architecture Search, auto feature engineering, model selection.

Searches over architectures, hyperparameters, and preprocessing pipelines.
"""

from __future__ import annotations

import itertools
import math
import random
import time
import threading
from typing import Any, Callable

import torch
import torch.nn as nn


# ── Search Spaces ──

LAYER_SEARCH_SPACE = {
    "linear": {"type": "Linear", "out_features": [32, 64, 128, 256, 512, 1024]},
    "conv2d": {"type": "Conv2d", "out_channels": [16, 32, 64, 128], "kernel_size": [3, 5]},
    "activation": {"type": "activation", "options": ["ReLU", "GELU", "SiLU", "Mish", "LeakyReLU"]},
    "normalization": {"type": "norm", "options": ["BatchNorm1d", "LayerNorm", "none"]},
    "dropout": {"type": "Dropout", "p": [0.0, 0.1, 0.2, 0.3, 0.5]},
    "residual": {"type": "ResidualBlock", "hidden_features": [64, 128, 256]},
}

HYPERPARAM_SPACE = {
    "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-1, "log": True},
    "batch_size": {"type": "choice", "options": [8, 16, 32, 64, 128, 256]},
    "optimizer": {"type": "choice", "options": ["Adam", "AdamW", "SGD", "RMSprop"]},
    "weight_decay": {"type": "float", "min": 0.0, "max": 0.1},
    "scheduler": {"type": "choice", "options": ["none", "CosineAnnealingLR", "StepLR", "OneCycleLR"]},
}


class NASEngine:
    """Neural Architecture Search engine."""

    def __init__(self):
        self._running = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._broadcast: Callable | None = None
        self._loop = None
        self._results: list[dict] = []
        self._best: dict | None = None

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def search(
        self,
        input_dim: int,
        output_dim: int,
        train_data: tuple | None = None,
        strategy: str = "random",
        n_trials: int = 20,
        epochs_per_trial: int = 5,
        task: str = "classification",
    ) -> dict:
        """Start NAS in background thread."""
        if self._running:
            return {"status": "already_running"}

        self._stop_event.clear()
        self._running = True
        self._results = []
        self._best = None

        self._thread = threading.Thread(
            target=self._search_loop,
            args=(input_dim, output_dim, train_data, strategy, n_trials, epochs_per_trial, task),
            daemon=True,
        )
        self._thread.start()
        return {"status": "started", "strategy": strategy, "n_trials": n_trials}

    def stop(self) -> dict:
        self._stop_event.set()
        self._running = False
        return {"status": "stopped"}

    def _search_loop(self, input_dim, output_dim, train_data, strategy, n_trials, epochs_per_trial, task):
        try:
            # Generate training data if not provided
            if train_data is None:
                x = torch.randn(500, input_dim)
                if task == "classification":
                    y = torch.randint(0, output_dim, (500,))
                else:
                    y = torch.randn(500, output_dim)
            else:
                x, y = train_data

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for trial in range(n_trials):
                if self._stop_event.is_set():
                    break

                # Generate architecture
                arch = self._sample_architecture(input_dim, output_dim, strategy, trial)

                # Build model
                try:
                    model = self._build_model(arch, input_dim, output_dim)
                except Exception as e:
                    continue

                model = model.to(device)

                # Sample hyperparameters
                hp = self._sample_hyperparams()

                # Quick train
                loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
                opt_cls = getattr(torch.optim, hp["optimizer"])
                optimizer = opt_cls(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

                # Train
                model.train()
                total_params = sum(p.numel() for p in model.parameters())
                batch_size = hp["batch_size"]
                final_loss = float("inf")
                accuracy = 0.0

                for epoch in range(epochs_per_trial):
                    if self._stop_event.is_set():
                        break
                    perm = torch.randperm(len(x))
                    epoch_loss = 0
                    correct = 0
                    total = 0
                    for i in range(0, len(x), batch_size):
                        idx = perm[i:i + batch_size]
                        xb = x[idx].to(device)
                        yb = y[idx].to(device)
                        optimizer.zero_grad()
                        out = model(xb)
                        loss = loss_fn(out, yb)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        if task == "classification" and out.dim() > 1:
                            correct += (out.argmax(1) == yb).sum().item()
                            total += len(yb)
                    final_loss = epoch_loss / max(len(x) // batch_size, 1)
                    if total > 0:
                        accuracy = correct / total

                # Score (lower is better for loss, higher for accuracy)
                score = accuracy if task == "classification" else -final_loss

                result = {
                    "trial": trial,
                    "architecture": arch,
                    "hyperparams": hp,
                    "total_params": total_params,
                    "final_loss": round(final_loss, 6),
                    "accuracy": round(accuracy, 4) if task == "classification" else None,
                    "score": round(score, 6),
                    "layers": [l["type"] for l in arch],
                }

                self._results.append(result)

                if self._best is None or score > self._best["score"]:
                    self._best = result

                self._emit("nas_trial", result)

            self._emit("nas_complete", {
                "total_trials": len(self._results),
                "best": self._best,
            })

        except Exception as e:
            self._emit("error", {"message": f"NAS error: {e}"})
        finally:
            self._running = False

    def _sample_architecture(self, in_dim: int, out_dim: int, strategy: str, trial: int) -> list[dict]:
        """Sample a random architecture."""
        n_layers = random.randint(2, 6)
        arch = []
        current_dim = in_dim

        for i in range(n_layers):
            layer_type = random.choice(["linear", "linear", "residual", "linear"])

            if layer_type == "linear":
                out = random.choice([32, 64, 128, 256, 512])
                arch.append({"type": "Linear", "in_features": current_dim, "out_features": out})
                current_dim = out

                # Activation
                act = random.choice(["ReLU", "GELU", "SiLU", "Mish", "LeakyReLU"])
                arch.append({"type": act})

                # Maybe norm
                if random.random() > 0.5:
                    arch.append({"type": "BatchNorm1d", "num_features": current_dim})

                # Maybe dropout
                if random.random() > 0.5:
                    p = random.choice([0.1, 0.2, 0.3, 0.5])
                    arch.append({"type": "Dropout", "p": p})

            elif layer_type == "residual":
                if current_dim >= 32:
                    arch.append({"type": "ResidualBlock", "in_features": current_dim})

        # Output layer
        arch.append({"type": "Linear", "in_features": current_dim, "out_features": out_dim})
        return arch

    def _build_model(self, arch: list[dict], in_dim: int, out_dim: int) -> nn.Module:
        """Build nn.Sequential from architecture spec."""
        from state_graph.core.registry import Registry

        layers = []
        for spec in arch:
            t = spec["type"]
            if t in ("ReLU", "GELU", "SiLU", "Mish", "LeakyReLU"):
                layers.append(getattr(nn, t)())
            elif t == "Linear":
                layers.append(nn.Linear(spec["in_features"], spec["out_features"]))
            elif t == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(spec["num_features"]))
            elif t == "Dropout":
                layers.append(nn.Dropout(spec.get("p", 0.5)))
            elif t == "ResidualBlock":
                cls = Registry.get_layer("ResidualBlock")
                layers.append(cls(spec["in_features"]))
            else:
                try:
                    cls = Registry.get_layer(t)
                    params = {k: v for k, v in spec.items() if k != "type"}
                    layers.append(cls(**params))
                except Exception:
                    pass

        return nn.Sequential(*layers)

    def _sample_hyperparams(self) -> dict:
        return {
            "learning_rate": 10 ** random.uniform(-5, -1),
            "batch_size": random.choice([16, 32, 64, 128]),
            "optimizer": random.choice(["Adam", "AdamW", "SGD"]),
            "weight_decay": random.uniform(0, 0.1),
        }

    def get_results(self) -> dict:
        return {
            "results": sorted(self._results, key=lambda r: r["score"], reverse=True),
            "best": self._best,
            "total_trials": len(self._results),
            "running": self._running,
        }

    def apply_best(self) -> dict:
        """Return the best architecture in a format the graph builder can import."""
        if not self._best:
            return {"status": "error", "message": "No search results"}

        nodes = []
        for i, spec in enumerate(self._best["architecture"]):
            t = spec["type"]
            if t in ("ReLU", "GELU", "SiLU", "Mish", "LeakyReLU"):
                if nodes:
                    nodes[-1]["activation"] = t
                continue
            params = {k: v for k, v in spec.items() if k != "type"}
            nodes.append({"layer_type": t, "params": params, "activation": None, "position": len(nodes)})

        return {
            "status": "ok",
            "architecture": nodes,
            "hyperparams": self._best["hyperparams"],
            "score": self._best["score"],
        }
