"""Project and file management for StateGraph workspace."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("./sg_workspace")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Project:
    """A workspace project containing code, data, configs, and outputs."""

    def __init__(self, name: str, description: str = "", template: str = "empty"):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.created_at = _now()
        self.path = WORKSPACE_ROOT / self.id
        self.path.mkdir(parents=True, exist_ok=True)

        # Create initial structure based on template
        self._scaffold(template)

        # Save project metadata
        self._save_meta()

    def _scaffold(self, template: str) -> None:
        """Create initial project files based on template."""
        (self.path / "data").mkdir(exist_ok=True)
        (self.path / "outputs").mkdir(exist_ok=True)
        (self.path / "scripts").mkdir(exist_ok=True)

        if template == "llm_finetune":
            self._write("scripts/train.py", LLM_FINETUNE_TEMPLATE)
            self._write("scripts/prepare_data.py", DATA_PREP_TEMPLATE)
            self._write("config.json", json.dumps({
                "model_id": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                "max_seq_length": 2048,
                "lora_r": 16,
                "lora_alpha": 16,
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 2e-4,
            }, indent=2))
            self._write("README.md", "# LLM Fine-Tuning Project\n\nEdit `config.json` and run `scripts/train.py`.\n")

        elif template == "vision":
            self._write("scripts/train.py", VISION_TEMPLATE)
            self._write("config.json", json.dumps({"model": "resnet50", "epochs": 10, "lr": 0.001}, indent=2))

        elif template == "dataset":
            self._write("scripts/create_dataset.py", DATASET_TEMPLATE)
            self._write("data/samples.jsonl", "")

        elif template == "yolo":
            self._write("scripts/train_yolo.py", YOLO_TEMPLATE)
            (self.path / "data" / "images").mkdir(exist_ok=True)
            (self.path / "data" / "labels").mkdir(exist_ok=True)
            self._write("data/classes.txt", "")

        else:
            self._write("main.py", '"""StateGraph Project"""\n\nimport torch\nimport torch.nn as nn\n\nprint("Hello from StateGraph!")\n')

        self._write(".gitignore", "outputs/\n__pycache__/\n*.pyc\n.DS_Store\n")

    def _write(self, rel_path: str, content: str) -> None:
        p = self.path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def _save_meta(self) -> None:
        meta = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }
        (self.path / ".sg_project.json").write_text(json.dumps(meta, indent=2))

    def list_files(self, rel_dir: str = "") -> list[dict]:
        """List files and dirs in a directory."""
        target = self.path / rel_dir
        if not target.exists():
            return []

        items = []
        for item in sorted(target.iterdir()):
            if item.name.startswith(".sg_"):
                continue
            info = {
                "name": item.name,
                "path": str(item.relative_to(self.path)),
                "type": "directory" if item.is_dir() else "file",
            }
            if item.is_file():
                info["size"] = item.stat().st_size
                info["extension"] = item.suffix
            elif item.is_dir():
                info["children_count"] = len(list(item.iterdir()))
            items.append(info)
        return items

    def read_file(self, rel_path: str) -> dict:
        p = self.path / rel_path
        if not p.exists():
            return {"status": "error", "message": f"File not found: {rel_path}"}
        if not p.is_file():
            return {"status": "error", "message": "Not a file"}

        # Detect if binary
        try:
            content = p.read_text(encoding="utf-8")
            return {"status": "ok", "path": rel_path, "content": content, "size": len(content)}
        except UnicodeDecodeError:
            return {"status": "ok", "path": rel_path, "content": None, "binary": True, "size": p.stat().st_size}

    def write_file(self, rel_path: str, content: str) -> dict:
        p = self.path / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "saved", "path": rel_path, "size": len(content)}

    def create_file(self, rel_path: str, content: str = "") -> dict:
        p = self.path / rel_path
        if p.exists():
            return {"status": "error", "message": "File already exists"}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"status": "created", "path": rel_path}

    def create_dir(self, rel_path: str) -> dict:
        p = self.path / rel_path
        p.mkdir(parents=True, exist_ok=True)
        return {"status": "created", "path": rel_path}

    def delete_file(self, rel_path: str) -> dict:
        p = self.path / rel_path
        if not p.exists():
            return {"status": "error", "message": "Not found"}
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return {"status": "deleted", "path": rel_path}

    def rename_file(self, old_path: str, new_path: str) -> dict:
        src = self.path / old_path
        dst = self.path / new_path
        if not src.exists():
            return {"status": "error", "message": "Source not found"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return {"status": "renamed", "old": old_path, "new": new_path}

    def get_file_tree(self, rel_dir: str = "", max_depth: int = 5, _depth: int = 0) -> list[dict]:
        """Recursive file tree."""
        if _depth > max_depth:
            return []
        target = self.path / rel_dir
        if not target.exists():
            return []

        items = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            if item.name == "__pycache__":
                continue
            node = {
                "name": item.name,
                "path": str(item.relative_to(self.path)),
                "type": "directory" if item.is_dir() else "file",
            }
            if item.is_file():
                node["extension"] = item.suffix
                node["size"] = item.stat().st_size
            elif item.is_dir():
                node["children"] = self.get_file_tree(
                    str(item.relative_to(self.path)), max_depth, _depth + 1
                )
            items.append(node)
        return items

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "path": str(self.path),
        }


class WorkspaceManager:
    """Manages multiple projects."""

    def __init__(self):
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        self.projects: dict[str, Project] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing projects from disk."""
        for d in WORKSPACE_ROOT.iterdir():
            meta_file = d / ".sg_project.json"
            if d.is_dir() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    p = object.__new__(Project)
                    p.id = meta["id"]
                    p.name = meta["name"]
                    p.description = meta.get("description", "")
                    p.created_at = meta.get("created_at", "")
                    p.path = d
                    self.projects[p.id] = p
                except Exception:
                    pass

    def create(self, name: str, description: str = "", template: str = "empty") -> Project:
        p = Project(name, description, template)
        self.projects[p.id] = p
        return p

    def get(self, project_id: str) -> Project | None:
        return self.projects.get(project_id)

    def list_all(self) -> list[dict]:
        return [p.to_dict() for p in sorted(self.projects.values(), key=lambda x: x.created_at, reverse=True)]

    def delete(self, project_id: str) -> dict:
        p = self.projects.pop(project_id, None)
        if not p:
            return {"status": "error", "message": "Not found"}
        if p.path.exists():
            shutil.rmtree(p.path, ignore_errors=True)
        return {"status": "deleted"}


# ── Project Templates ──

LLM_FINETUNE_TEMPLATE = '''"""LLM Fine-Tuning with Unsloth"""

import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load config
with open("config.json") as f:
    cfg = json.load(f)

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model_id"],
    max_seq_length=cfg["max_seq_length"],
    load_in_4bit=True,
)

# 2. Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# 3. Load dataset
# Replace with your dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# 4. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=cfg["max_seq_length"],
    args=TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        gradient_accumulation_steps=4,
        warmup_steps=5,
        logging_steps=1,
        fp16=True,
        save_steps=100,
    ),
)

trainer.train()

# 5. Save
model.save_pretrained("./outputs/lora_model")
tokenizer.save_pretrained("./outputs/lora_model")
print("Training complete! Model saved to ./outputs/lora_model")
'''

DATA_PREP_TEMPLATE = '''"""Prepare dataset for fine-tuning."""

import json

# Example: convert your data to Alpaca format
samples = [
    {
        "instruction": "What is machine learning?",
        "input": "",
        "output": "Machine learning is a subset of AI..."
    },
    # Add more samples here
]

# Save as JSONL
with open("data/train.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\\n")

print(f"Saved {len(samples)} samples to data/train.jsonl")
'''

VISION_TEMPLATE = '''"""Vision model training."""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import json

with open("config.json") as f:
    cfg = json.load(f)

# Load pretrained model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 10)  # Adjust for your number of classes

optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
loss_fn = nn.CrossEntropyLoss()

print(f"Model ready: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print("Add your dataset loading code here!")
'''

DATASET_TEMPLATE = '''"""Create and export a dataset."""

import json

# Define your dataset samples
samples = []

# Example: text classification
samples.append({"text": "This is great!", "label": "positive"})
samples.append({"text": "This is terrible.", "label": "negative"})

# Save as JSONL
with open("data/samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\\n")

print(f"Created dataset with {len(samples)} samples")
'''

YOLO_TEMPLATE = '''"""YOLO object detection training setup."""

# 1. Organize your data:
#    data/images/  - put your images here
#    data/labels/  - YOLO format label files (one per image)
#    data/classes.txt - one class name per line
#
# YOLO label format (one line per object):
#    class_id x_center y_center width height
#    (all normalized 0-1)
#
# 2. Install ultralytics: pip install ultralytics
# 3. Run this script

from pathlib import Path
import yaml

# Create YOLO dataset config
classes = Path("data/classes.txt").read_text().strip().split("\\n")
config = {
    "path": str(Path("data").resolve()),
    "train": "images",
    "val": "images",
    "names": {i: name for i, name in enumerate(classes)},
}

with open("data/dataset.yaml", "w") as f:
    yaml.dump(config, f)

print(f"Dataset config created with {len(classes)} classes")
print("Run: yolo train model=yolov8n.pt data=data/dataset.yaml epochs=50")
'''
