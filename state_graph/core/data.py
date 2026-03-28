"""Dataset management with real datasets, augmentation, and preprocessing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class DataManager:
    """Handles dataset loading, augmentation, and splitting."""

    # Available augmentations for tabular/flat data
    AUGMENTATIONS = {
        "gaussian_noise": {"sigma": 0.1},
        "dropout_noise": {"p": 0.1},
        "scaling": {"min_scale": 0.9, "max_scale": 1.1},
        "mixup": {"alpha": 0.2},
    }

    def __init__(self) -> None:
        self.x_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.x_val: torch.Tensor | None = None
        self.y_val: torch.Tensor | None = None
        self.x_test: torch.Tensor | None = None
        self.y_test: torch.Tensor | None = None
        self.input_shape: tuple = ()
        self.n_classes: int = 0
        self.dataset_name: str = ""
        self.augmentations: list[dict] = []
        self._is_image: bool = False

    def load_builtin(self, name: str, n_samples: int = 1000) -> dict:
        """Load a built-in synthetic dataset."""
        if name == "random":
            return self._load_random(n_samples, 784, 10)
        elif name == "xor":
            return self._load_xor(n_samples)
        elif name == "spiral":
            return self._load_spiral(n_samples)
        elif name == "circles":
            return self._load_circles(n_samples)
        elif name == "moons":
            return self._load_moons(n_samples)
        elif name == "blobs":
            return self._load_blobs(n_samples)
        elif name == "checkerboard":
            return self._load_checkerboard(n_samples)
        elif name == "regression_sin":
            return self._load_regression_sin(n_samples)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def load_real(self, name: str, data_dir: str = "./data") -> dict:
        """Load a real dataset via torchvision."""
        try:
            import torchvision
            import torchvision.transforms as T
        except ImportError:
            raise ImportError(
                "torchvision is required for real datasets. "
                "Install it: pip install torchvision"
            )

        os.makedirs(data_dir, exist_ok=True)

        if name == "mnist":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            train_ds = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
            self._is_image = True
            self.input_shape = (1, 28, 28)
            self.n_classes = 10

        elif name == "fashion_mnist":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
            train_ds = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
            self._is_image = True
            self.input_shape = (1, 28, 28)
            self.n_classes = 10

        elif name == "cifar10":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ])
            train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
            self._is_image = True
            self.input_shape = (3, 32, 32)
            self.n_classes = 10

        elif name == "cifar100":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            train_ds = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
            self._is_image = True
            self.input_shape = (3, 32, 32)
            self.n_classes = 100

        elif name == "svhn":
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
            train_ds = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=transform)
            test_ds = torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=transform)
            self._is_image = True
            self.input_shape = (3, 32, 32)
            self.n_classes = 10

        else:
            raise ValueError(f"Unknown real dataset: {name}")

        # Convert to tensors
        train_loader = DataLoader(train_ds, batch_size=len(train_ds))
        test_loader = DataLoader(test_ds, batch_size=len(test_ds))

        self.x_train, self.y_train = next(iter(train_loader))
        self.x_test, self.y_test = next(iter(test_loader))

        # Split train into train/val (90/10)
        n_val = int(0.1 * len(self.x_train))
        n_train = len(self.x_train) - n_val
        indices = torch.randperm(len(self.x_train))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        self.x_val = self.x_train[val_idx]
        self.y_val = self.y_train[val_idx]
        self.x_train = self.x_train[train_idx]
        self.y_train = self.y_train[train_idx]

        self.dataset_name = name

        return {
            "status": "loaded",
            "dataset": name,
            "n_train": len(self.x_train),
            "n_val": len(self.x_val),
            "n_test": len(self.x_test),
            "input_shape": list(self.input_shape),
            "n_classes": self.n_classes,
            "is_image": self._is_image,
        }

    def set_augmentations(self, augs: list[dict]) -> None:
        """Set augmentation pipeline. Each dict has 'name' and optional params."""
        self.augmentations = augs

    def apply_augmentation(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to a batch."""
        for aug in self.augmentations:
            name = aug["name"]
            params = {k: v for k, v in aug.items() if k != "name"}

            if name == "gaussian_noise":
                sigma = params.get("sigma", 0.1)
                x = x + torch.randn_like(x) * sigma

            elif name == "dropout_noise":
                p = params.get("p", 0.1)
                mask = torch.bernoulli(torch.full_like(x, 1 - p))
                x = x * mask

            elif name == "scaling":
                min_s = params.get("min_scale", 0.9)
                max_s = params.get("max_scale", 1.1)
                scale = torch.empty(x.shape[0], 1).uniform_(min_s, max_s)
                if x.dim() > 2:
                    scale = scale.view(x.shape[0], 1, 1, 1)
                x = x * scale.to(x.device)

            elif name == "mixup":
                alpha = params.get("alpha", 0.2)
                lam = torch.distributions.Beta(alpha, alpha).sample()
                idx = torch.randperm(x.shape[0])
                x = lam * x + (1 - lam) * x[idx]
                # For mixup, y stays the same (simplified - real mixup needs label mixing)

            elif name == "cutout" and x.dim() == 4:
                size = params.get("size", 8)
                b, c, h, w = x.shape
                cy = torch.randint(0, h, (b,))
                cx = torch.randint(0, w, (b,))
                for i in range(b):
                    y1 = max(0, cy[i] - size // 2)
                    y2 = min(h, cy[i] + size // 2)
                    x1 = max(0, cx[i] - size // 2)
                    x2 = min(w, cx[i] + size // 2)
                    x[i, :, y1:y2, x1:x2] = 0

            elif name == "random_flip" and x.dim() == 4:
                if torch.rand(1).item() > 0.5:
                    x = x.flip(-1)

        return x, y

    def get_data_loaders(self, batch_size: int = 32, flatten: bool = False) -> dict:
        """Get DataLoaders for train/val/test."""
        loaders = {}

        if self.x_train is not None:
            x = self.x_train.view(self.x_train.shape[0], -1) if flatten else self.x_train
            loaders["train"] = DataLoader(
                TensorDataset(x, self.y_train),
                batch_size=batch_size, shuffle=True,
            )
        if self.x_val is not None:
            x = self.x_val.view(self.x_val.shape[0], -1) if flatten else self.x_val
            loaders["val"] = DataLoader(
                TensorDataset(x, self.y_val),
                batch_size=batch_size,
            )
        if self.x_test is not None:
            x = self.x_test.view(self.x_test.shape[0], -1) if flatten else self.x_test
            loaders["test"] = DataLoader(
                TensorDataset(x, self.y_test),
                batch_size=batch_size,
            )
        return loaders

    def get_info(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n_train": len(self.x_train) if self.x_train is not None else 0,
            "n_val": len(self.x_val) if self.x_val is not None else 0,
            "n_test": len(self.x_test) if self.x_test is not None else 0,
            "input_shape": list(self.input_shape),
            "n_classes": self.n_classes,
            "is_image": self._is_image,
            "augmentations": self.augmentations,
        }

    # --- Synthetic dataset generators ---

    def _load_random(self, n: int, dim: int, classes: int) -> dict:
        x = torch.randn(n, dim)
        y = torch.randint(0, classes, (n,))
        self.input_shape = (dim,)
        self.n_classes = classes
        return self._finalize_synthetic("random", x, y)

    def _load_xor(self, n: int) -> dict:
        x = torch.randn(n, 2)
        y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long()
        self.input_shape = (2,)
        self.n_classes = 2
        return self._finalize_synthetic("xor", x, y)

    def _load_spiral(self, n: int) -> dict:
        half = n // 2
        theta = torch.linspace(0, 4 * 3.14159, half)
        r = torch.linspace(0.5, 2, half)
        x0 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        x1 = torch.stack([r * torch.cos(theta + 3.14159), r * torch.sin(theta + 3.14159)], dim=1)
        x = torch.cat([x0, x1]) + torch.randn(n, 2) * 0.1
        y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
        self.input_shape = (2,)
        self.n_classes = 2
        return self._finalize_synthetic("spiral", x, y)

    def _load_circles(self, n: int) -> dict:
        half = n // 2
        theta = torch.linspace(0, 2 * 3.14159, half)
        x0 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * 1.0
        x1 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * 2.0
        x = torch.cat([x0, x1]) + torch.randn(n, 2) * 0.1
        y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
        self.input_shape = (2,)
        self.n_classes = 2
        return self._finalize_synthetic("circles", x, y)

    def _load_moons(self, n: int) -> dict:
        half = n // 2
        theta = torch.linspace(0, 3.14159, half)
        x0 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        x1 = torch.stack([1 - torch.cos(theta), 1 - torch.sin(theta) - 0.5], dim=1)
        x = torch.cat([x0, x1]) + torch.randn(n, 2) * 0.1
        y = torch.cat([torch.zeros(half), torch.ones(half)]).long()
        self.input_shape = (2,)
        self.n_classes = 2
        return self._finalize_synthetic("moons", x, y)

    def _load_blobs(self, n: int) -> dict:
        k = 4
        per_class = n // k
        xs, ys = [], []
        centers = [(0, 0), (3, 3), (0, 3), (3, 0)]
        for i, (cx, cy) in enumerate(centers):
            xs.append(torch.randn(per_class, 2) * 0.5 + torch.tensor([cx, cy]))
            ys.append(torch.full((per_class,), i, dtype=torch.long))
        x = torch.cat(xs)
        y = torch.cat(ys)
        self.input_shape = (2,)
        self.n_classes = k
        return self._finalize_synthetic("blobs", x, y)

    def _load_checkerboard(self, n: int) -> dict:
        x = torch.rand(n, 2) * 4
        y = ((x[:, 0].long() + x[:, 1].long()) % 2).long()
        self.input_shape = (2,)
        self.n_classes = 2
        return self._finalize_synthetic("checkerboard", x, y)

    def _load_regression_sin(self, n: int) -> dict:
        x = torch.linspace(-3, 3, n).unsqueeze(1)
        y = torch.sin(x * 2) + torch.randn(n, 1) * 0.1
        self.input_shape = (1,)
        self.n_classes = 0  # regression
        self._is_image = False
        return self._finalize_synthetic("regression_sin", x, y.squeeze())

    def _finalize_synthetic(self, name: str, x: torch.Tensor, y: torch.Tensor) -> dict:
        self.dataset_name = name
        self._is_image = False
        split = int(0.8 * len(x))
        perm = torch.randperm(len(x))
        x, y = x[perm], y[perm]
        self.x_train, self.y_train = x[:split], y[:split]
        self.x_val, self.y_val = x[split:], y[split:]
        self.x_test = None
        self.y_test = None
        return {
            "status": "loaded",
            "dataset": name,
            "n_train": split,
            "n_val": len(x) - split,
            "input_shape": list(self.input_shape),
            "n_classes": self.n_classes,
            "is_image": False,
        }
