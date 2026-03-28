"""Training engine that ties together the graph, model, and metrics.

Runs training in a background thread so the UI stays responsive.
Broadcasts real-time updates via an async callback.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from state_graph.core.graph import StateGraph
from state_graph.core.metrics import MetricsCollector
from state_graph.core.registry import Registry
from state_graph.core.scheduler import SchedulerRegistry
from state_graph.core.data import DataManager


class TrainingEngine:
    """Manages the full training lifecycle."""

    def __init__(self) -> None:
        self.graph = StateGraph()
        self.metrics = MetricsCollector()
        self.data_manager = DataManager()
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.loss_fn: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._train_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_training = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # Broadcast callback: async fn(event_type: str, data: dict)
        self._broadcast: Callable | None = None

        # Model source: "graph" or "hf"
        self.model_source: str = "graph"

        # HF integration (lazy loaded)
        self.hf_manager: Any = None
        self.hf_data: Any = None

        # Training config
        self.config = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "scheduler": None,
            "scheduler_params": {},
        }

        # HF-specific training config
        self.hf_config = {
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }

        # Data
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def set_broadcast(self, fn: Callable) -> None:
        self._broadcast = fn

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def _emit(self, event: str, data: dict) -> None:
        if self._broadcast:
            try:
                await self._broadcast(event, data)
            except Exception:
                pass

    def _emit_from_thread(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(event, data), self._loop
            )

    def set_data(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> None:
        """Set training (and optional validation) data."""
        batch_size = self.config["batch_size"]
        self._train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        if x_val is not None and y_val is not None:
            self._val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=batch_size,
            )

    def build(self) -> dict:
        """Build model from the current graph state or HF model."""
        if self.model_source == "hf" and self.hf_manager:
            return self._build_hf()
        return self._build_graph()

    def _build_graph(self) -> dict:
        """Build model from the graph."""
        self.model = self.graph.build_model().to(self.device)

        # Set up optimizer
        opt_cls = Registry.get_optimizer(self.config["optimizer"])
        self.optimizer = opt_cls(
            self.model.parameters(), lr=self.config["learning_rate"]
        )

        # Set up scheduler
        self.scheduler = None
        if self.config.get("scheduler"):
            self.scheduler = SchedulerRegistry.create(
                self.config["scheduler"],
                self.optimizer,
                self.config.get("scheduler_params"),
            )

        # Set up loss
        loss_cls = Registry.get_loss(self.config["loss"])
        self.loss_fn = loss_cls()

        # Attach metric hooks
        self.metrics.attach_hooks(self.model)

        param_count = self.graph.get_param_count()
        total_params = sum(
            info["trainable"] for info in param_count.values()
        )

        return {
            "status": "built",
            "total_params": total_params,
            "param_details": param_count,
            "device": str(self.device),
            "model_source": "graph",
        }

    def _build_hf(self) -> dict:
        """Build from HF model."""
        model = self.hf_manager.get_model()
        if model is None:
            return {"status": "error", "message": "No HF model loaded"}

        self.model = model.to(self.device)

        # Only optimize trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return {"status": "error", "message": "No trainable parameters. Apply LoRA or unfreeze some layers."}

        opt_cls = Registry.get_optimizer(self.config["optimizer"])
        self.optimizer = opt_cls(
            trainable_params,
            lr=self.config["learning_rate"],
            weight_decay=self.hf_config.get("weight_decay", 0.01),
        )

        # Scheduler
        self.scheduler = None
        if self.config.get("scheduler"):
            self.scheduler = SchedulerRegistry.create(
                self.config["scheduler"],
                self.optimizer,
                self.config.get("scheduler_params"),
            )

        # Loss — for HF models, loss may come from the model itself
        loss_cls = Registry.get_loss(self.config["loss"])
        self.loss_fn = loss_cls()

        # Attach metric hooks
        self.metrics.attach_hooks(self.model)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "built",
            "total_params": total,
            "trainable_params": trainable,
            "device": str(self.device),
            "model_source": "hf",
            "model_id": self.hf_manager.model_id,
        }

    def start_training(self) -> dict:
        """Start training in a background thread."""
        if self._is_training:
            return {"status": "already_training"}
        if self.model is None:
            self.build()

        # Check data source
        if self.model_source == "hf" and self.hf_data:
            if self.hf_data.train_loader is None:
                return {"status": "error", "message": "No HF training data set"}
            self._train_loader = self.hf_data.train_loader
            self._val_loader = self.hf_data.val_loader

        if self._train_loader is None:
            return {"status": "error", "message": "No training data set"}

        self._stop_event.clear()
        self._is_training = True

        # Choose training loop based on model source
        target = self._hf_train_loop if self.model_source == "hf" else self._train_loop
        self._train_thread = threading.Thread(target=target, daemon=True)
        self._train_thread.start()
        return {"status": "started"}

    def stop_training(self) -> dict:
        """Stop training."""
        if not self._is_training:
            return {"status": "not_training"}
        self._stop_event.set()
        if self._train_thread:
            self._train_thread.join(timeout=5)
        self._is_training = False
        return {"status": "stopped"}

    def _train_loop(self) -> None:
        """Main training loop running in a background thread."""
        try:
            epochs = self.config["epochs"]

            for epoch in range(epochs):
                if self._stop_event.is_set():
                    break

                self.model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (x_batch, y_batch) in enumerate(self._train_loader):
                    if self._stop_event.is_set():
                        break

                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = self.loss_fn(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    loss_val = loss.item()
                    epoch_loss += loss_val

                    # Accuracy for classification
                    if output.dim() > 1 and output.shape[1] > 1:
                        pred = output.argmax(dim=1)
                        correct += (pred == y_batch).sum().item()
                        total += y_batch.size(0)

                    # Collect and broadcast step metrics
                    step_data = self.metrics.collect_step(
                        self.model, loss_val, self.optimizer,
                        extra={"epoch": epoch, "batch": batch_idx},
                    )
                    step_data["epoch"] = epoch
                    step_data["total_epochs"] = epochs
                    step_data["batch"] = batch_idx
                    step_data["total_batches"] = len(self._train_loader)
                    self._emit_from_thread("step", step_data)

                # Epoch metrics
                avg_loss = epoch_loss / len(self._train_loader)
                train_acc = correct / total if total > 0 else None

                val_loss = None
                val_acc = None
                if self._val_loader:
                    val_loss, val_acc = self._evaluate()

                # Step scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss if val_loss is not None else avg_loss)
                    else:
                        self.scheduler.step()

                epoch_data = self.metrics.collect_epoch(
                    epoch, avg_loss, val_loss, train_acc, val_acc
                )
                self._emit_from_thread("epoch", epoch_data)

            self._emit_from_thread("training_complete", {
                "final_loss": self.metrics.get_loss_history()[-1]["loss"]
                if self.metrics.get_loss_history() else None,
            })

        except Exception as e:
            self._emit_from_thread("error", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
        finally:
            self._is_training = False

    def _evaluate(self) -> tuple[float, float | None]:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in self._val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(x_batch)
                loss = self.loss_fn(output, y_batch)
                total_loss += loss.item()

                if output.dim() > 1 and output.shape[1] > 1:
                    pred = output.argmax(dim=1)
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)

        avg_loss = total_loss / len(self._val_loader)
        acc = correct / total if total > 0 else None
        return avg_loss, acc

    def _hf_train_loop(self) -> None:
        """Training loop for HF models — handles dict batches, model.loss, grad accumulation."""
        try:
            epochs = self.config["epochs"]
            grad_accum = self.hf_config.get("gradient_accumulation_steps", 1)
            max_grad_norm = self.hf_config.get("max_grad_norm", 1.0)
            use_fp16 = self.hf_config.get("fp16", False) and self.device.type == "cuda"

            scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

            for epoch in range(epochs):
                if self._stop_event.is_set():
                    break

                self.model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, batch in enumerate(self._train_loader):
                    if self._stop_event.is_set():
                        break

                    # Handle both dict batches (HF) and tuple batches (standard)
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            outputs = self.model(**batch)

                        # HF models return loss if labels are provided
                        if hasattr(outputs, "loss") and outputs.loss is not None:
                            loss = outputs.loss
                        else:
                            logits = outputs.logits if hasattr(outputs, "logits") else outputs
                            labels = batch.get("labels", batch.get("label"))
                            loss = self.loss_fn(logits, labels)

                        # Accuracy
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        if isinstance(logits, torch.Tensor) and logits.dim() > 1 and logits.shape[-1] > 1:
                            pred = logits.argmax(dim=-1)
                            labels = batch.get("labels", batch.get("label"))
                            if labels is not None:
                                correct += (pred == labels).sum().item()
                                total += labels.numel()
                    else:
                        # Standard tuple batch
                        x_batch, y_batch = batch
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            output = self.model(x_batch)

                        loss = self.loss_fn(output, y_batch)

                        if output.dim() > 1 and output.shape[1] > 1:
                            pred = output.argmax(dim=1)
                            correct += (pred == y_batch).sum().item()
                            total += y_batch.size(0)

                    # Gradient accumulation
                    loss_scaled = loss / grad_accum
                    if scaler:
                        scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    if (batch_idx + 1) % grad_accum == 0:
                        if scaler:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        if scaler:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                    loss_val = loss.item()
                    epoch_loss += loss_val

                    step_data = self.metrics.collect_step(
                        self.model, loss_val, self.optimizer,
                        extra={"epoch": epoch, "batch": batch_idx},
                    )
                    self._emit_from_thread("step", step_data)

                avg_loss = epoch_loss / max(len(self._train_loader), 1)
                train_acc = correct / total if total > 0 else None

                val_loss = None
                val_acc = None
                if self._val_loader:
                    val_loss, val_acc = self._evaluate_hf()

                # Step scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss if val_loss is not None else avg_loss)
                    else:
                        self.scheduler.step()

                epoch_data = self.metrics.collect_epoch(
                    epoch, avg_loss, val_loss, train_acc, val_acc
                )
                self._emit_from_thread("epoch", epoch_data)

            self._emit_from_thread("training_complete", {
                "final_loss": self.metrics.get_loss_history()[-1]["loss"]
                if self.metrics.get_loss_history() else None,
            })

        except Exception as e:
            self._emit_from_thread("error", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
        finally:
            self._is_training = False

    def _evaluate_hf(self) -> tuple[float, float | None]:
        """Evaluate HF model on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self._val_loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = self.model(**batch)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    if isinstance(logits, torch.Tensor) and logits.dim() > 1 and logits.shape[-1] > 1:
                        pred = logits.argmax(dim=-1)
                        labels = batch.get("labels", batch.get("label"))
                        if labels is not None:
                            correct += (pred == labels).sum().item()
                            total += labels.numel()
                else:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    output = self.model(x_batch)
                    loss = self.loss_fn(output, y_batch)
                    total_loss += loss.item()
                    if output.dim() > 1 and output.shape[1] > 1:
                        pred = output.argmax(dim=1)
                        correct += (pred == y_batch).sum().item()
                        total += y_batch.size(0)

        avg_loss = total_loss / max(len(self._val_loader), 1)
        acc = correct / total if total > 0 else None
        return avg_loss, acc

    def export_architecture(self) -> dict:
        """Export architecture as JSON for save/load."""
        return {
            "version": 1,
            "config": self.config,
            "graph": self.graph.to_dict(),
        }

    def import_architecture(self, data: dict) -> dict:
        """Import architecture from JSON."""
        # Clear existing
        for node_id in list(self.graph.nodes.keys()):
            self.graph.remove_layer(node_id)

        # Restore config
        if "config" in data:
            self.config.update(data["config"])

        # Restore layers
        graph_data = data.get("graph", {})
        for node in sorted(graph_data.get("nodes", []), key=lambda n: n["position"]):
            self.graph.add_layer(
                layer_type=node["layer_type"],
                params=node.get("params", {}),
                activation=node.get("activation"),
                position=node.get("position"),
            )

        return {"status": "imported", "graph": self.graph.to_dict(), "config": self.config}

    # Standard PyTorch layer types (exist in torch.nn)
    _PYTORCH_LAYERS = {
        "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d",
        "BatchNorm1d", "BatchNorm2d", "LayerNorm",
        "Dropout", "Dropout2d", "Embedding",
        "LSTM", "GRU", "MultiheadAttention",
        "Flatten", "AdaptiveAvgPool2d", "MaxPool2d", "AvgPool2d",
    }

    _PYTORCH_ACTIVATIONS = {
        "ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh",
        "Softmax", "ELU", "PReLU", "Mish",
    }

    def export_python(self) -> str:
        """Export the current architecture as Python code."""
        if self.model_source == "hf" and self.hf_manager:
            return self._export_python_hf()
        return self._export_python_graph()

    def _export_python_graph(self) -> str:
        nodes = self.graph.get_sorted_nodes()

        # Detect which custom layers are used
        custom_layers = set()
        custom_activations = set()
        for node in nodes:
            if node.layer_type not in self._PYTORCH_LAYERS:
                custom_layers.add(node.layer_type)
            if node.activation and node.activation not in self._PYTORCH_ACTIVATIONS:
                custom_activations.add(node.activation)

        lines = [
            "import torch",
            "import torch.nn as nn",
        ]

        if custom_layers:
            imports = ", ".join(sorted(custom_layers))
            lines.append(f"from state_graph.layers.custom import {imports}")

        lines += ["", "", "class GeneratedModel(nn.Module):", "    def __init__(self):", "        super().__init__()", "        self.model = nn.Sequential("]

        for node in nodes:
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in node.params.items())
            if node.layer_type in self._PYTORCH_LAYERS:
                lines.append(f"            nn.{node.layer_type}({params_str}),")
            else:
                lines.append(f"            {node.layer_type}({params_str}),")

            if node.activation:
                if node.activation in self._PYTORCH_ACTIVATIONS:
                    lines.append(f"            nn.{node.activation}(),")
                else:
                    lines.append(f"            # Custom activation: {node.activation}")

        lines.append("        )")
        lines.append("")
        lines.append("    def forward(self, x):")
        lines.append("        return self.model(x)")
        lines.append("")
        lines.append("")

        # Training snippet
        cfg = self.config
        lines.append("# --- Training Setup ---")
        lines.append("model = GeneratedModel()")
        lines.append(f"optimizer = torch.optim.{cfg['optimizer']}(model.parameters(), lr={cfg['learning_rate']})")
        lines.append(f"loss_fn = nn.{cfg['loss']}()")
        if cfg.get("scheduler"):
            sched_params = cfg.get("scheduler_params", {})
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in sched_params.items())
            lines.append(f"scheduler = torch.optim.lr_scheduler.{cfg['scheduler']}(optimizer, {params_str})")
        lines.append("")
        lines.append(f"# Train for {cfg['epochs']} epochs with batch_size={cfg['batch_size']}")
        lines.append("for epoch in range({epochs}):".format(epochs=cfg['epochs']))
        lines.append("    for x_batch, y_batch in train_loader:")
        lines.append("        optimizer.zero_grad()")
        lines.append("        output = model(x_batch)")
        lines.append("        loss = loss_fn(output, y_batch)")
        lines.append("        loss.backward()")
        lines.append("        optimizer.step()")
        if cfg.get("scheduler"):
            lines.append("    scheduler.step()")
        lines.append("")

        return "\n".join(lines)

    def _export_python_hf(self) -> str:
        """Export HF fine-tuning code."""
        info = self.hf_manager.get_info()
        cfg = self.config
        hf_cfg = self.hf_config

        lines = [
            "import torch",
            "from transformers import AutoModelForSequenceClassification, AutoTokenizer",
        ]

        if info.get("lora_applied"):
            lines.append("from peft import LoraConfig, get_peft_model")

        lines += [
            "",
            f"# Load pretrained model",
            f'model = AutoModelForSequenceClassification.from_pretrained("{info.get("model_id", "model_name")}")',
            f'tokenizer = AutoTokenizer.from_pretrained("{info.get("model_id", "model_name")}")',
            "",
        ]

        if info.get("lora_applied") and info.get("lora_config"):
            lc = info["lora_config"]
            lines += [
                "# Apply LoRA",
                f"lora_config = LoraConfig(",
                f"    r={lc['r']},",
                f"    lora_alpha={lc['alpha']},",
                f"    lora_dropout={lc['dropout']},",
                f"    target_modules={lc['target_modules']},",
                ")",
                "model = get_peft_model(model, lora_config)",
                "",
            ]

        lines += [
            "# Training setup",
            f"optimizer = torch.optim.{cfg['optimizer']}(",
            f"    [p for p in model.parameters() if p.requires_grad],",
            f"    lr={cfg['learning_rate']},",
            f"    weight_decay={hf_cfg.get('weight_decay', 0.01)},",
            ")",
            "",
            f"# Train for {cfg['epochs']} epochs",
            f"for epoch in range({cfg['epochs']}):",
            "    model.train()",
            "    for batch in train_loader:",
            "        batch = {k: v.to(device) for k, v in batch.items()}",
            "        outputs = model(**batch)",
            "        loss = outputs.loss",
            f"        loss = loss / {hf_cfg.get('gradient_accumulation_steps', 1)}",
            "        loss.backward()",
            f"        torch.nn.utils.clip_grad_norm_(model.parameters(), {hf_cfg.get('max_grad_norm', 1.0)})",
            "        optimizer.step()",
            "        optimizer.zero_grad()",
            "",
        ]

        return "\n".join(lines)

    def reset(self) -> dict:
        """Full reset — stop training, clear graph, metrics, HF state."""
        self.stop_training()
        # Clear graph
        for node_id in list(self.graph.nodes.keys()):
            self.graph.remove_layer(node_id)
        # Reset metrics
        self.metrics.reset()
        # Reset model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        # Reset data
        self._train_loader = None
        self._val_loader = None
        self.data_manager = DataManager()
        # Reset HF
        self.hf_manager = None
        self.hf_data = None
        self.model_source = "graph"
        return {"status": "reset"}

    def get_status(self) -> dict:
        status = {
            "is_training": self._is_training,
            "config": self.config,
            "hf_config": self.hf_config,
            "graph": self.graph.to_dict(),
            "metrics_snapshot": self.metrics.get_snapshot() if self.metrics else {},
            "device": str(self.device),
            "model_source": self.model_source,
        }
        if self.hf_manager:
            status["hf_model"] = self.hf_manager.get_info()
        if self.hf_data:
            status["hf_data"] = self.hf_data.get_info()
        return status
