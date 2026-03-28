"""Hugging Face Hub model management — search, load, inspect, fine-tune."""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
import torch.nn as nn


# Task → AutoModel class name mapping for transformers
TASK_TO_AUTO_MODEL = {
    "text-classification": "AutoModelForSequenceClassification",
    "token-classification": "AutoModelForTokenClassification",
    "question-answering": "AutoModelForQuestionAnswering",
    "text-generation": "AutoModelForCausalLM",
    "text2text-generation": "AutoModelForSeq2SeqLM",
    "summarization": "AutoModelForSeq2SeqLM",
    "translation": "AutoModelForSeq2SeqLM",
    "fill-mask": "AutoModelForMaskedLM",
    "image-classification": "AutoModelForImageClassification",
    "object-detection": "AutoModelForObjectDetection",
    "image-segmentation": "AutoModelForSemanticSegmentation",
    "audio-classification": "AutoModelForAudioClassification",
    "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
    "feature-extraction": "AutoModel",
}


class HFModelManager:
    """Manages Hugging Face model lifecycle: search, load, inspect, LoRA."""

    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.tokenizer: Any | None = None
        self.processor: Any | None = None
        self.model_id: str = ""
        self.library: str = ""
        self.task: str = ""
        self.lora_applied: bool = False
        self._model_info: dict = {}

    def search_models(
        self,
        query: str,
        library: str | None = None,
        task: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search Hugging Face Hub for models."""
        from huggingface_hub import HfApi

        api = HfApi()
        kwargs: dict[str, Any] = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
        }
        if library:
            kwargs["library"] = library
        if task:
            kwargs["pipeline_tag"] = task

        models = api.list_models(**kwargs)
        return [
            {
                "model_id": m.modelId,
                "downloads": m.downloads,
                "likes": m.likes,
                "library": getattr(m, "library_name", None),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "tags": list(m.tags) if m.tags else [],
            }
            for m in models
        ]

    def search_datasets(
        self,
        query: str,
        task: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search HF Hub for datasets."""
        from huggingface_hub import HfApi

        api = HfApi()
        kwargs: dict[str, Any] = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
        }
        if task:
            kwargs["task_categories"] = task

        datasets = api.list_datasets(**kwargs)
        return [
            {
                "dataset_id": d.id,
                "downloads": d.downloads,
                "likes": d.likes,
                "tags": list(d.tags) if d.tags else [],
            }
            for d in datasets
        ]

    def load_model(
        self,
        model_id: str,
        library: str | None = None,
        task: str | None = None,
        num_labels: int | None = None,
        dtype: str | None = None,
    ) -> dict:
        """Load a pretrained model from HF Hub.

        Supports: transformers, timm, diffusers.
        """
        self.model_id = model_id
        self.lora_applied = False
        torch_dtype = getattr(torch, dtype) if dtype else None

        # Auto-detect library
        if library is None:
            library = self._detect_library(model_id)
        self.library = library
        self.task = task or ""

        if library == "timm":
            return self._load_timm(model_id, num_labels)
        elif library == "diffusers":
            return self._load_diffusers(model_id, torch_dtype)
        else:
            return self._load_transformers(model_id, task, num_labels, torch_dtype)

    def _detect_library(self, model_id: str) -> str:
        """Detect which library a model belongs to."""
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            lib = getattr(info, "library_name", None)
            if lib:
                return lib
            tags = set(info.tags) if info.tags else set()
            if "timm" in tags:
                return "timm"
            if "diffusers" in tags:
                return "diffusers"
        except Exception:
            pass
        return "transformers"

    def _load_transformers(
        self, model_id: str, task: str | None, num_labels: int | None, torch_dtype: Any
    ) -> dict:
        import transformers

        # Determine AutoModel class
        auto_cls_name = TASK_TO_AUTO_MODEL.get(task, "AutoModel")
        auto_cls = getattr(transformers, auto_cls_name)

        kwargs: dict[str, Any] = {}
        if torch_dtype:
            kwargs["torch_dtype"] = torch_dtype
        if num_labels and "Classification" in auto_cls_name:
            kwargs["num_labels"] = num_labels

        self.model = auto_cls.from_pretrained(model_id, **kwargs)

        # Load tokenizer or processor
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor
        except Exception:
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
            except Exception:
                self.tokenizer = None

        total, trainable = self._count_params()
        self._model_info = {
            "model_id": model_id,
            "library": "transformers",
            "task": task or auto_cls_name,
            "model_class": self.model.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "has_tokenizer": self.tokenizer is not None,
            "has_processor": self.processor is not None,
        }
        return self._model_info

    def _load_timm(self, model_id: str, num_labels: int | None) -> dict:
        import timm

        kwargs: dict[str, Any] = {"pretrained": True}
        if num_labels:
            kwargs["num_classes"] = num_labels

        self.model = timm.create_model(model_id, **kwargs)
        self.tokenizer = None

        # timm models often have a data_config
        try:
            data_cfg = timm.data.resolve_model_data_config(self.model)
            self.processor = timm.data.create_transform(**data_cfg, is_training=False)
        except Exception:
            self.processor = None

        total, trainable = self._count_params()
        self._model_info = {
            "model_id": model_id,
            "library": "timm",
            "task": "image-classification",
            "model_class": self.model.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "has_tokenizer": False,
            "has_processor": self.processor is not None,
        }
        return self._model_info

    def _load_diffusers(self, model_id: str, torch_dtype: Any) -> dict:
        import diffusers

        kwargs: dict[str, Any] = {}
        if torch_dtype:
            kwargs["torch_dtype"] = torch_dtype

        # Try loading as pipeline first
        self.model = diffusers.DiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.tokenizer = getattr(self.model, "tokenizer", None)
        self.processor = None

        # For diffusers, we track the unet params
        unet = getattr(self.model, "unet", None)
        if unet:
            total = sum(p.numel() for p in unet.parameters())
            trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        else:
            total, trainable = 0, 0

        self._model_info = {
            "model_id": model_id,
            "library": "diffusers",
            "task": "text-to-image",
            "model_class": self.model.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "has_tokenizer": self.tokenizer is not None,
            "has_processor": False,
        }
        return self._model_info

    def get_model_tree(self, max_depth: int = 3) -> list[dict]:
        """Get model architecture as a tree of modules."""
        if self.model is None:
            return []

        # For diffusers pipelines, inspect the unet
        model = self.model
        if hasattr(model, "unet"):
            model = model.unet

        tree = []
        for name, module in model.named_modules():
            depth = name.count(".") if name else 0
            if depth > max_depth:
                continue
            if not name:
                name = "(root)"

            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )
            frozen = params > 0 and trainable == 0

            tree.append({
                "name": name,
                "type": module.__class__.__name__,
                "depth": depth,
                "params": params,
                "trainable": trainable,
                "frozen": frozen,
            })

        return tree

    def freeze_layers(self, patterns: list[str]) -> dict:
        """Freeze layers matching any of the given regex patterns."""
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.model
        if hasattr(model, "unet"):
            model = model.unet

        frozen_count = 0
        for name, param in model.named_parameters():
            for pattern in patterns:
                if re.search(pattern, name):
                    param.requires_grad = False
                    frozen_count += 1
                    break

        total, trainable = self._count_params()
        return {
            "status": "ok",
            "frozen_count": frozen_count,
            "total_params": total,
            "trainable_params": trainable,
        }

    def unfreeze_layers(self, patterns: list[str]) -> dict:
        """Unfreeze layers matching any of the given regex patterns."""
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.model
        if hasattr(model, "unet"):
            model = model.unet

        unfrozen_count = 0
        for name, param in model.named_parameters():
            for pattern in patterns:
                if re.search(pattern, name):
                    param.requires_grad = True
                    unfrozen_count += 1
                    break

        total, trainable = self._count_params()
        return {
            "status": "ok",
            "unfrozen_count": unfrozen_count,
            "total_params": total,
            "trainable_params": trainable,
        }

    def apply_lora(
        self,
        target_modules: list[str] | None = None,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        task_type: str | None = None,
    ) -> dict:
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        from peft import LoraConfig, get_peft_model, TaskType

        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = self._suggest_lora_targets()

        # Detect task type
        peft_task = None
        if task_type:
            peft_task = getattr(TaskType, task_type, None)
        elif self.task:
            task_map = {
                "text-classification": TaskType.SEQ_CLS,
                "token-classification": TaskType.TOKEN_CLS,
                "question-answering": TaskType.QUESTION_ANS,
                "text-generation": TaskType.CAUSAL_LM,
                "text2text-generation": TaskType.SEQ_2_SEQ_LM,
            }
            peft_task = task_map.get(self.task)

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=peft_task,
        )

        self.model = get_peft_model(self.model, config)
        self.lora_applied = True

        total, trainable = self._count_params()
        self._model_info.update({
            "total_params": total,
            "trainable_params": trainable,
            "lora_applied": True,
            "lora_config": {
                "r": r,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "target_modules": target_modules,
            },
        })
        return self._model_info

    def _suggest_lora_targets(self) -> list[str]:
        """Auto-suggest LoRA target modules based on model architecture."""
        if self.model is None:
            return ["q_proj", "v_proj"]

        module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Extract the last part of the name
                parts = name.split(".")
                if parts:
                    module_names.add(parts[-1])

        # Common attention projection names
        common_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Llama, Mistral
            "query", "key", "value",  # BERT style
            "q_lin", "k_lin", "v_lin",  # DistilBERT
            "Wqkv", "out_proj",  # MPT
            "c_attn", "c_proj",  # GPT-2
            "query_key_value",  # Falcon
        ]

        targets = [t for t in common_targets if t in module_names]
        if not targets:
            # Fallback: pick all linear layer names
            targets = list(module_names)[:4]

        return targets

    def _count_params(self) -> tuple[int, int]:
        """Count total and trainable parameters."""
        if self.model is None:
            return 0, 0
        model = self.model
        if hasattr(model, "unet"):
            model = model.unet
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def get_info(self) -> dict:
        """Get current model info."""
        if self.model is None:
            return {"loaded": False}
        total, trainable = self._count_params()
        return {
            **self._model_info,
            "loaded": True,
            "total_params": total,
            "trainable_params": trainable,
            "lora_applied": self.lora_applied,
        }

    def get_model(self) -> nn.Module | None:
        """Get the underlying nn.Module for training."""
        if self.model is None:
            return None
        # For diffusers pipelines, return unet
        if hasattr(self.model, "unet"):
            return self.model.unet
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_processor(self):
        return self.processor

    # ------------------------------------------------------------------
    # Architecture Surgery — modify loaded HF models
    # ------------------------------------------------------------------

    def insert_module(self, parent_path: str, name: str, module: nn.Module) -> dict:
        """Insert a new module into the model at the specified path.

        Example: insert_module("encoder.layer", "12", nn.Linear(768, 768))
        """
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        parent = self._get_module_by_path(model, parent_path)
        if parent is None:
            return {"status": "error", "message": f"Module path '{parent_path}' not found"}

        if isinstance(parent, nn.ModuleList):
            idx = int(name) if name.isdigit() else len(parent)
            parent.insert(idx, module)
        else:
            setattr(parent, name, module)

        total, trainable = self._count_params()
        return {"status": "ok", "total_params": total, "trainable_params": trainable}

    def remove_module(self, path: str) -> dict:
        """Remove a module from the model by path.

        Example: remove_module("encoder.layer.11")
        """
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        parts = path.rsplit(".", 1)
        if len(parts) == 1:
            return {"status": "error", "message": "Cannot remove root module"}

        parent_path, child_name = parts
        parent = self._get_module_by_path(model, parent_path)
        if parent is None:
            return {"status": "error", "message": f"Parent '{parent_path}' not found"}

        if isinstance(parent, nn.ModuleList) and child_name.isdigit():
            idx = int(child_name)
            if idx < len(parent):
                del parent[idx]
            else:
                return {"status": "error", "message": f"Index {idx} out of range"}
        elif hasattr(parent, child_name):
            setattr(parent, child_name, nn.Identity())
        else:
            return {"status": "error", "message": f"Module '{child_name}' not found in '{parent_path}'"}

        total, trainable = self._count_params()
        return {"status": "ok", "total_params": total, "trainable_params": trainable}

    def replace_module(self, path: str, new_module: nn.Module) -> dict:
        """Replace a module in the model.

        Example: replace_module("classifier", nn.Linear(768, 10))
        """
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        parts = path.rsplit(".", 1)

        if len(parts) == 1:
            # Top-level attribute
            if hasattr(model, path):
                setattr(model, path, new_module)
            else:
                return {"status": "error", "message": f"Module '{path}' not found"}
        else:
            parent_path, child_name = parts
            parent = self._get_module_by_path(model, parent_path)
            if parent is None:
                return {"status": "error", "message": f"Parent '{parent_path}' not found"}

            if isinstance(parent, nn.ModuleList) and child_name.isdigit():
                parent[int(child_name)] = new_module
            elif hasattr(parent, child_name):
                setattr(parent, child_name, new_module)
            else:
                return {"status": "error", "message": f"Module '{child_name}' not found"}

        total, trainable = self._count_params()
        return {"status": "ok", "total_params": total, "trainable_params": trainable}

    def add_head(self, name: str, in_features: int, out_features: int) -> dict:
        """Add a new classification/output head to the model."""
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        head = nn.Linear(in_features, out_features)
        setattr(model, name, head)

        total, trainable = self._count_params()
        return {"status": "ok", "head_name": name, "total_params": total, "trainable_params": trainable}

    def _get_module_by_path(self, model: nn.Module, path: str) -> nn.Module | None:
        """Navigate to a module by dot-separated path."""
        if not path or path == "(root)":
            return model
        current = model
        for part in path.split("."):
            if isinstance(current, nn.ModuleList) and part.isdigit():
                idx = int(part)
                if idx < len(current):
                    current = current[idx]
                else:
                    return None
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def get_module_info(self, path: str) -> dict:
        """Get info about a specific module by path."""
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        module = self._get_module_by_path(model, path)
        if module is None:
            return {"status": "error", "message": f"Module '{path}' not found"}

        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        children = [(n, type(m).__name__) for n, m in module.named_children()]

        return {
            "status": "ok",
            "path": path,
            "type": type(module).__name__,
            "params": params,
            "trainable": trainable,
            "frozen": params > 0 and trainable == 0,
            "children": [{"name": n, "type": t} for n, t in children],
        }

    # ------------------------------------------------------------------
    # HF Trainer Integration — train any transformers/timm model
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader=None, epochs: int = 3,
              lr: float = 2e-5, weight_decay: float = 0.01,
              scheduler_type: str = "linear", warmup_steps: int = 0,
              max_grad_norm: float = 1.0, fp16: bool = False,
              broadcast_fn=None) -> dict:
        """Train the loaded HF model using PyTorch training loop.

        Works with any model (transformers, timm, custom) — not just Unsloth.
        """
        import threading

        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        device = next(model.parameters()).device

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * epochs
        from torch.optim.lr_scheduler import OneCycleLR, LinearLR
        if scheduler_type == "onecycle":
            scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)
        else:
            scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1))

        scaler = torch.amp.GradScaler() if fp16 else None

        self._stop_training = False
        self._train_history = []

        def _loop():
            model.train()
            global_step = 0

            for epoch in range(epochs):
                if self._stop_training:
                    break

                total_loss = 0.0
                n_batches = 0

                for batch_idx, batch in enumerate(train_loader):
                    if self._stop_training:
                        break

                    # Move batch to device
                    if isinstance(batch, dict):
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                 for k, v in batch.items()}
                    elif isinstance(batch, (list, tuple)):
                        batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

                    optimizer.zero_grad()

                    if fp16:
                        with torch.amp.autocast(device_type=str(device)):
                            if isinstance(batch, dict):
                                outputs = model(**batch)
                            else:
                                outputs = model(*batch)
                            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if isinstance(batch, dict):
                            outputs = model(**batch)
                        else:
                            outputs = model(*batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                    scheduler.step()
                    total_loss += loss.item()
                    n_batches += 1
                    global_step += 1

                    step_data = {
                        "step": global_step,
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }
                    self._train_history.append(step_data)

                    if broadcast_fn and global_step % 5 == 0:
                        broadcast_fn("hf_train_step", step_data)

                avg_loss = total_loss / max(n_batches, 1)

                # Validation
                val_loss = None
                if val_loader:
                    model.eval()
                    vl = 0.0
                    vn = 0
                    with torch.no_grad():
                        for vbatch in val_loader:
                            if isinstance(vbatch, dict):
                                vbatch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                           for k, v in vbatch.items()}
                            elif isinstance(vbatch, (list, tuple)):
                                vbatch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in vbatch]

                            if isinstance(vbatch, dict):
                                vout = model(**vbatch)
                            else:
                                vout = model(*vbatch)
                            vloss = vout.loss if hasattr(vout, 'loss') else vout[0]
                            vl += vloss.item()
                            vn += 1
                    val_loss = vl / max(vn, 1)
                    model.train()

                epoch_data = {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                }
                if broadcast_fn:
                    broadcast_fn("hf_train_epoch", epoch_data)

            if broadcast_fn:
                broadcast_fn("hf_train_complete", {"final_loss": avg_loss if n_batches else 0})

        self._train_thread = threading.Thread(target=_loop, daemon=True)
        self._train_thread.start()

        return {"status": "started", "epochs": epochs, "lr": lr}

    def stop_training(self):
        self._stop_training = True
        if hasattr(self, '_train_thread') and self._train_thread.is_alive():
            self._train_thread.join(timeout=10)
        return {"status": "stopped"}

    def get_train_history(self) -> list[dict]:
        return getattr(self, '_train_history', [])

    # ------------------------------------------------------------------
    # Diffusion Inference
    # ------------------------------------------------------------------

    def diffusion_generate(self, prompt: str, num_inference_steps: int = 50,
                           guidance_scale: float = 7.5, height: int = 512,
                           width: int = 512, seed: Optional[int] = None) -> dict:
        """Generate image from text using loaded diffusion pipeline."""
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}
        if self.library != "diffusers":
            return {"status": "error", "message": "Not a diffusers model"}

        try:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.model.device).manual_seed(seed)

            result = self.model(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

            # Save image to temp file
            import tempfile, base64
            from io import BytesIO
            image = result.images[0]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            return {
                "status": "ok",
                "image_base64": img_b64,
                "width": width,
                "height": height,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Inference for any model
    # ------------------------------------------------------------------

    def inference(self, inputs: dict[str, Any]) -> dict:
        """Run inference on any loaded model.

        For text models: {"text": "input text"}
        For image models: {"image_path": "/path/to/image"}
        For text-generation: {"text": "prompt", "max_length": 100}
        """
        if self.model is None:
            return {"status": "error", "message": "No model loaded"}

        model = self.get_model()
        device = next(model.parameters()).device
        model.eval()

        try:
            with torch.no_grad():
                if self.task == "text-generation":
                    text = inputs.get("text", "")
                    max_length = inputs.get("max_length", 100)
                    if self.tokenizer:
                        enc = self.tokenizer(text, return_tensors="pt").to(device)
                        out = model.generate(**enc, max_new_tokens=max_length)
                        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
                        return {"status": "ok", "text": decoded}

                elif self.task in ("text-classification", "fill-mask", "token-classification",
                                   "question-answering", "text2text-generation"):
                    text = inputs.get("text", "")
                    if self.tokenizer:
                        enc = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                        out = model(**enc)
                        if hasattr(out, "logits"):
                            preds = out.logits.argmax(dim=-1).tolist()
                            return {"status": "ok", "predictions": preds, "logits_shape": list(out.logits.shape)}
                        return {"status": "ok", "output_keys": list(out.keys()) if isinstance(out, dict) else "tensor"}

                elif self.task == "image-classification" and self.library == "timm":
                    # For timm models
                    import PIL.Image
                    img_path = inputs.get("image_path", "")
                    if img_path and self.processor:
                        img = PIL.Image.open(img_path).convert("RGB")
                        tensor = self.processor(img).unsqueeze(0).to(device)
                        out = model(tensor)
                        preds = out.argmax(dim=-1).tolist()
                        return {"status": "ok", "predictions": preds}

                # Fallback: try generic forward
                text = inputs.get("text", "")
                if text and self.tokenizer:
                    enc = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
                    out = model(**enc)
                    return {"status": "ok", "output_type": type(out).__name__}

            return {"status": "error", "message": f"Inference not supported for task '{self.task}'"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
