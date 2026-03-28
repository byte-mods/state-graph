"""HuggingFace datasets integration — search, load, create, preprocess."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader


class HFDataManager:
    """Manages HF datasets for training: load, preprocess, create custom datasets."""

    def __init__(self) -> None:
        self.dataset: Any = None
        self.dataset_id: str = ""
        self.task: str = ""
        self.tokenizer: Any = None
        self.processor: Any = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self._info: dict = {}

    def load_hf_dataset(
        self,
        dataset_id: str,
        split: str | None = None,
        subset: str | None = None,
        streaming: bool = False,
    ) -> dict:
        """Load a dataset from HF Hub."""
        from datasets import load_dataset

        kwargs: dict[str, Any] = {}
        if subset:
            kwargs["name"] = subset
        if streaming:
            kwargs["streaming"] = True

        ds = load_dataset(dataset_id, split=split, **kwargs)

        self.dataset = ds
        self.dataset_id = dataset_id

        # Get info
        info = self._extract_info(ds)
        self._info = info
        return info

    def load_local_text(
        self,
        texts: list[str],
        labels: list[int | str],
        label_names: list[str] | None = None,
    ) -> dict:
        """Create a text classification dataset from raw data."""
        from datasets import Dataset, ClassLabel

        data = {"text": texts, "label": labels}
        ds = Dataset.from_dict(data)

        if label_names:
            ds = ds.cast_column("label", ClassLabel(names=label_names))

        self.dataset = ds
        self.dataset_id = "custom_text"
        self.task = "text-classification"
        self._info = {
            "status": "loaded",
            "dataset": "custom_text",
            "task": "text-classification",
            "n_samples": len(ds),
            "columns": list(ds.column_names),
            "n_classes": len(set(labels)),
        }
        return self._info

    def load_local_images(
        self,
        image_dir: str,
        label_map: dict[str, int] | None = None,
    ) -> dict:
        """Create an image classification dataset from a folder structure.

        Expected structure: image_dir/class_name/image.jpg
        """
        from datasets import load_dataset

        ds = load_dataset("imagefolder", data_dir=image_dir)
        self.dataset = ds
        self.dataset_id = "custom_images"
        self.task = "image-classification"

        info = self._extract_info(ds)
        self._info = info
        return info

    def load_local_audio(
        self,
        audio_dir: str,
    ) -> dict:
        """Create an audio dataset from a folder structure."""
        from datasets import load_dataset

        ds = load_dataset("audiofolder", data_dir=audio_dir)
        self.dataset = ds
        self.dataset_id = "custom_audio"
        self.task = "audio-classification"

        info = self._extract_info(ds)
        self._info = info
        return info

    def load_csv(self, file_path: str, text_col: str | None = None, label_col: str | None = None) -> dict:
        """Load dataset from CSV file."""
        from datasets import load_dataset

        ds = load_dataset("csv", data_files=file_path)
        self.dataset = ds
        self.dataset_id = os.path.basename(file_path)

        info = self._extract_info(ds)
        self._info = info
        return info

    def load_json(self, file_path: str) -> dict:
        """Load dataset from JSON/JSONL file."""
        from datasets import load_dataset

        ds = load_dataset("json", data_files=file_path)
        self.dataset = ds
        self.dataset_id = os.path.basename(file_path)

        info = self._extract_info(ds)
        self._info = info
        return info

    def set_preprocessing(
        self,
        tokenizer: Any = None,
        processor: Any = None,
        max_length: int = 128,
        text_column: str = "text",
        label_column: str = "label",
        image_column: str = "image",
        audio_column: str = "audio",
    ) -> dict:
        """Configure preprocessing for the loaded dataset."""
        self.tokenizer = tokenizer
        self.processor = processor

        if self.dataset is None:
            return {"status": "error", "message": "No dataset loaded"}

        ds = self.dataset
        # Handle DatasetDict (train/test splits)
        if hasattr(ds, "keys"):
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            sample_ds = ds[split_name]
        else:
            sample_ds = ds

        columns = sample_ds.column_names

        # Text preprocessing
        if tokenizer and text_column in columns:
            def tokenize_fn(examples):
                return tokenizer(
                    examples[text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )

            ds = ds.map(tokenize_fn, batched=True)

            # Handle label column — may not exist in all datasets
            format_cols = ["input_ids", "attention_mask"]
            if label_column in columns:
                if label_column != "labels":
                    ds = ds.rename_column(label_column, "labels")
                format_cols.append("labels")
            elif "labels" in (sample_ds.column_names if not hasattr(ds, "keys") else ds[split_name].column_names):
                format_cols.append("labels")

            ds.set_format("torch", columns=format_cols)

        # Image preprocessing
        elif processor and image_column in columns:
            def process_images(examples):
                images = examples[image_column]
                inputs = processor(images=images, return_tensors="pt")
                inputs["labels"] = examples.get(label_column, [0] * len(images))
                return inputs

            ds = ds.map(process_images, batched=True, remove_columns=columns)
            ds.set_format("torch")

        self.dataset = ds
        return {"status": "preprocessed", "columns": list(ds.column_names) if not hasattr(ds, "keys") else {k: list(v.column_names) for k, v in ds.items()}}

    def get_dataloaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.1,
    ) -> dict[str, DataLoader]:
        """Convert the dataset to PyTorch DataLoaders."""
        if self.dataset is None:
            return {}

        ds = self.dataset

        # Handle DatasetDict
        if hasattr(ds, "keys"):
            loaders = {}
            if "train" in ds:
                loaders["train"] = DataLoader(
                    ds["train"],
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=self._collate_fn,
                )
            if "validation" in ds:
                loaders["val"] = DataLoader(
                    ds["validation"],
                    batch_size=batch_size,
                    collate_fn=self._collate_fn,
                )
            elif "test" in ds:
                loaders["val"] = DataLoader(
                    ds["test"],
                    batch_size=batch_size,
                    collate_fn=self._collate_fn,
                )
            self.train_loader = loaders.get("train")
            self.val_loader = loaders.get("val")
            return loaders
        else:
            # Single split — do train/val split
            split = ds.train_test_split(test_size=val_split)
            train_loader = DataLoader(
                split["train"],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self._collate_fn,
            )
            val_loader = DataLoader(
                split["test"],
                batch_size=batch_size,
                collate_fn=self._collate_fn,
            )
            self.train_loader = train_loader
            self.val_loader = val_loader
            return {"train": train_loader, "val": val_loader}

    def _collate_fn(self, batch):
        """Default collation that handles dicts and lists."""
        if isinstance(batch[0], dict):
            result = {}
            for key in batch[0]:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    result[key] = torch.stack(values)
                elif isinstance(values[0], (int, float)):
                    result[key] = torch.tensor(values)
                else:
                    result[key] = values
            return result
        return torch.utils.data.default_collate(batch)

    def preview(self, n: int = 5) -> list[dict]:
        """Get sample rows for UI preview."""
        if self.dataset is None:
            return []

        ds = self.dataset
        if hasattr(ds, "keys"):
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            ds = ds[split_name]

        samples = []
        for i in range(min(n, len(ds))):
            row = {}
            for col in ds.column_names:
                val = ds[i][col]
                if isinstance(val, torch.Tensor):
                    val = f"Tensor{list(val.shape)}"
                elif hasattr(val, "size"):
                    val = f"Image({val.size})" if hasattr(val, "mode") else str(val)[:100]
                else:
                    val = str(val)[:200]
                row[col] = val
            samples.append(row)
        return samples

    def _extract_info(self, ds) -> dict:
        """Extract dataset info for UI."""
        if hasattr(ds, "keys"):
            splits = {k: len(v) for k, v in ds.items()}
            sample_ds = ds[list(ds.keys())[0]]
            columns = sample_ds.column_names
            n_total = sum(splits.values())
        else:
            splits = {"all": len(ds)}
            columns = ds.column_names
            n_total = len(ds)

        return {
            "status": "loaded",
            "dataset": self.dataset_id,
            "task": self.task,
            "n_samples": n_total,
            "splits": splits,
            "columns": columns,
        }

    def get_info(self) -> dict:
        if self.dataset is None:
            return {"loaded": False}
        return {**self._info, "loaded": True}
