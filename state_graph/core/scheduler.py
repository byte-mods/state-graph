"""Learning rate scheduler registry and management."""

from __future__ import annotations

from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class SchedulerRegistry:
    """Registry for learning rate schedulers."""

    _schedulers: dict[str, type] = {}

    @classmethod
    def _register_defaults(cls) -> None:
        cls._schedulers = {
            "StepLR": optim.lr_scheduler.StepLR,
            "MultiStepLR": optim.lr_scheduler.MultiStepLR,
            "ExponentialLR": optim.lr_scheduler.ExponentialLR,
            "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
            "CyclicLR": optim.lr_scheduler.CyclicLR,
            "OneCycleLR": optim.lr_scheduler.OneCycleLR,
            "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
            "LinearLR": optim.lr_scheduler.LinearLR,
            "PolynomialLR": optim.lr_scheduler.PolynomialLR,
        }

    @classmethod
    def register(cls, name: str, scheduler_cls: type) -> None:
        cls._schedulers[name] = scheduler_cls

    @classmethod
    def get(cls, name: str) -> type:
        return cls._schedulers[name]

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._schedulers.keys())

    @classmethod
    def get_default_params(cls, name: str) -> dict[str, Any]:
        """Return sensible default params for each scheduler."""
        defaults = {
            "StepLR": {"step_size": 10, "gamma": 0.1},
            "MultiStepLR": {"milestones": [30, 60, 90], "gamma": 0.1},
            "ExponentialLR": {"gamma": 0.95},
            "CosineAnnealingLR": {"T_max": 50, "eta_min": 1e-6},
            "ReduceLROnPlateau": {"mode": "min", "factor": 0.1, "patience": 10},
            "CyclicLR": {"base_lr": 1e-4, "max_lr": 0.01, "step_size_up": 200},
            "OneCycleLR": {"max_lr": 0.01, "total_steps": 1000},
            "CosineAnnealingWarmRestarts": {"T_0": 10, "T_mult": 2},
            "LinearLR": {"start_factor": 0.1, "total_iters": 100},
            "PolynomialLR": {"total_iters": 100, "power": 2.0},
        }
        return defaults.get(name, {})

    @classmethod
    def create(cls, name: str, optimizer: optim.Optimizer, params: dict[str, Any] | None = None) -> LRScheduler:
        """Create a scheduler instance."""
        scheduler_cls = cls._schedulers[name]
        p = cls.get_default_params(name)
        if params:
            p.update(params)
        return scheduler_cls(optimizer, **p)


SchedulerRegistry._register_defaults()
