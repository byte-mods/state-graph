"""Unsloth integration — fast LLM fine-tuning with LoRA, RLHF, DPO, ORPO.

Provides 2-4x faster training and 70% less memory vs standard HF training.
Hooks into StateGraph's real-time metrics broadcast for live visualization.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any, Callable


# Supported models for quick reference
UNSLOTH_MODELS = {
    "llama": [
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/llama-3-8b-bnb-4bit",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
    ],
    "mistral": [
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    ],
    "gemma": [
        "unsloth/gemma-2-9b-bnb-4bit",
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "unsloth/gemma-2-2b-bnb-4bit",
        "unsloth/gemma-2-2b-it-bnb-4bit",
    ],
    "phi": [
        "unsloth/Phi-3.5-mini-instruct",
        "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    ],
    "qwen": [
        "unsloth/Qwen2.5-7B-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
    ],
    "deepseek": [
        "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
    ],
    "tinyllama": [
        "unsloth/tinyllama-bnb-4bit",
    ],
}

TRAINING_METHODS = {
    "sft": {
        "name": "Supervised Fine-Tuning (SFT)",
        "description": "Standard instruction tuning with input-output pairs",
        "trainer": "SFTTrainer",
        "dataset_format": "instruction/chat",
    },
    "dpo": {
        "name": "Direct Preference Optimization (DPO)",
        "description": "Learn from preference pairs (chosen vs rejected)",
        "trainer": "DPOTrainer",
        "dataset_format": "prompt/chosen/rejected",
    },
    "orpo": {
        "name": "Odds Ratio Preference Optimization (ORPO)",
        "description": "Combined SFT + preference learning, no reference model needed",
        "trainer": "ORPOTrainer",
        "dataset_format": "prompt/chosen/rejected",
    },
    "kto": {
        "name": "Kahneman-Tversky Optimization (KTO)",
        "description": "Binary feedback (thumbs up/down) instead of pairwise preferences",
        "trainer": "KTOTrainer",
        "dataset_format": "prompt/completion/label",
    },
    "reward": {
        "name": "Reward Model Training",
        "description": "Train a reward model for RLHF pipeline",
        "trainer": "RewardTrainer",
        "dataset_format": "prompt/chosen/rejected",
    },
}

CHAT_TEMPLATES = [
    "chatml", "llama-3", "mistral", "gemma", "phi-3", "qwen-2.5",
    "alpaca", "vicuna", "zephyr", "unsloth",
]

EXPORT_FORMATS = [
    {"id": "lora", "name": "LoRA Adapters", "description": "Save just the LoRA weights"},
    {"id": "merged_16bit", "name": "Merged (16-bit)", "description": "Full model merged in float16"},
    {"id": "merged_4bit", "name": "Merged (4-bit)", "description": "Full model merged in 4-bit quantized"},
    {"id": "gguf_q4", "name": "GGUF Q4_K_M", "description": "For llama.cpp / Ollama (4-bit)"},
    {"id": "gguf_q5", "name": "GGUF Q5_K_M", "description": "For llama.cpp / Ollama (5-bit)"},
    {"id": "gguf_q8", "name": "GGUF Q8_0", "description": "For llama.cpp / Ollama (8-bit)"},
    {"id": "gguf_f16", "name": "GGUF F16", "description": "For llama.cpp / Ollama (16-bit)"},
    {"id": "ollama", "name": "Push to Ollama", "description": "Create Ollama model directly"},
    {"id": "hub", "name": "Push to HF Hub", "description": "Upload to Hugging Face Hub"},
]


class UnslothManager:
    """Manages Unsloth model lifecycle: load, configure LoRA, train, export."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.model_id: str = ""
        self.is_loaded: bool = False
        self.lora_applied: bool = False
        self._train_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_training: bool = False
        self._broadcast: Callable | None = None
        self._loop = None
        self._train_history: list[dict] = []
        self._config: dict = {}

    def set_broadcast(self, fn: Callable, loop) -> None:
        self._broadcast = fn
        self._loop = loop

    def _emit(self, event: str, data: dict) -> None:
        if self._broadcast and self._loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(self._broadcast(event, data), self._loop)

    def load_model(
        self,
        model_id: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        dtype: str | None = None,
    ) -> dict:
        """Load a model using Unsloth's FastLanguageModel."""
        from unsloth import FastLanguageModel
        import torch

        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=torch_dtype,
        )

        self.model_id = model_id
        self.is_loaded = True
        self.lora_applied = False

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "loaded",
            "model_id": model_id,
            "model_class": self.model.__class__.__name__,
            "total_params": total,
            "trainable_params": trainable,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
        }

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        target_modules: list[str] | None = None,
        use_gradient_checkpointing: str = "unsloth",
        use_rslora: bool = False,
        loftq_config: dict | None = None,
    ) -> dict:
        """Apply LoRA adapters using Unsloth's optimized method."""
        from unsloth import FastLanguageModel

        if not self.is_loaded:
            return {"status": "error", "message": "No model loaded"}

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_rslora=use_rslora,
            loftq_config=loftq_config or {},
            random_state=42,
        )

        self.lora_applied = True
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "lora_applied",
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(trainable / total * 100, 2),
            "r": r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
        }

    def set_chat_template(self, template: str) -> dict:
        """Apply a chat template to the tokenizer."""
        from unsloth.chat_templates import get_chat_template

        self.tokenizer = get_chat_template(self.tokenizer, chat_template=template)
        return {"status": "ok", "template": template}

    def prepare_dataset(
        self,
        dataset_source: str,
        dataset_id: str = "",
        dataset_path: str = "",
        text_column: str = "text",
        formatting: str = "instruction",
        max_seq_length: int = 2048,
        split: str = "train",
    ) -> dict:
        """Prepare a dataset for training."""
        from datasets import load_dataset

        if dataset_source == "hub":
            ds = load_dataset(dataset_id, split=split)
        elif dataset_source == "local_jsonl":
            ds = load_dataset("json", data_files=dataset_path, split="train")
        elif dataset_source == "local_csv":
            ds = load_dataset("csv", data_files=dataset_path, split="train")
        else:
            return {"status": "error", "message": f"Unknown source: {dataset_source}"}

        self._dataset = ds
        self._dataset_config = {
            "source": dataset_source,
            "id": dataset_id or dataset_path,
            "formatting": formatting,
            "text_column": text_column,
            "max_seq_length": max_seq_length,
        }

        return {
            "status": "prepared",
            "n_samples": len(ds),
            "columns": ds.column_names,
            "preview": [ds[i] for i in range(min(3, len(ds)))],
        }

    def start_training(self, method: str, config: dict) -> dict:
        """Start training in a background thread with real-time metrics."""
        if self._is_training:
            return {"status": "already_training"}
        if not self.is_loaded or not self.lora_applied:
            return {"status": "error", "message": "Load model and apply LoRA first"}
        if not hasattr(self, "_dataset"):
            return {"status": "error", "message": "Prepare dataset first"}

        self._config = {"method": method, **config}
        self._stop_event.clear()
        self._is_training = True
        self._train_history = []

        self._train_thread = threading.Thread(
            target=self._train_loop, args=(method, config), daemon=True
        )
        self._train_thread.start()

        return {"status": "started", "method": method}

    def stop_training(self) -> dict:
        if not self._is_training:
            return {"status": "not_training"}
        self._stop_event.set()
        if self._train_thread:
            self._train_thread.join(timeout=10)
        self._is_training = False
        return {"status": "stopped"}

    def _train_loop(self, method: str, config: dict) -> None:
        """Run training with metrics broadcasting."""
        try:
            import torch
            from transformers import TrainingArguments
            from unsloth import FastLanguageModel

            # Build training arguments
            output_dir = config.get("output_dir", "./sg_outputs/unsloth")
            args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config.get("epochs", 1),
                per_device_train_batch_size=config.get("batch_size", 2),
                gradient_accumulation_steps=config.get("gradient_accumulation", 4),
                learning_rate=config.get("learning_rate", 2e-4),
                lr_scheduler_type=config.get("lr_scheduler", "linear"),
                warmup_steps=config.get("warmup_steps", 5),
                weight_decay=config.get("weight_decay", 0.01),
                max_grad_norm=config.get("max_grad_norm", 1.0),
                fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
                logging_steps=config.get("logging_steps", 1),
                save_steps=config.get("save_steps", 100),
                seed=42,
                report_to="none",
            )

            ds = self._dataset
            ds_config = self._dataset_config

            if method == "sft":
                trainer = self._create_sft_trainer(args, ds, ds_config, config)
            elif method == "dpo":
                trainer = self._create_dpo_trainer(args, ds, config)
            elif method == "orpo":
                trainer = self._create_orpo_trainer(args, ds, config)
            elif method == "kto":
                trainer = self._create_kto_trainer(args, ds, config)
            elif method == "reward":
                trainer = self._create_reward_trainer(args, ds, config)
            else:
                self._emit("error", {"message": f"Unknown method: {method}"})
                return

            self.trainer = trainer

            # Add custom callback for real-time metrics
            from transformers import TrainerCallback

            mgr = self

            class MetricsBroadcastCallback(TrainerCallback):
                def on_log(self, _args, state, control, logs=None, **kwargs):
                    if mgr._stop_event.is_set():
                        control.should_training_stop = True
                        return
                    if logs:
                        step_data = {
                            "step": state.global_step,
                            "epoch": round(state.epoch, 2) if state.epoch else 0,
                            "loss": logs.get("loss", 0),
                            "learning_rate": logs.get("learning_rate", 0),
                            "grad_norm": logs.get("grad_norm", 0),
                        }
                        # Add method-specific metrics
                        for key in ["rewards/chosen", "rewards/rejected", "rewards/accuracies",
                                     "rewards/margins", "logps/chosen", "logps/rejected",
                                     "kl", "kto_loss", "reward_loss"]:
                            if key in logs:
                                step_data[key.replace("/", "_")] = logs[key]

                        mgr._train_history.append(step_data)
                        mgr._emit("unsloth_step", step_data)

                def on_epoch_end(self, _args, state, control, **kwargs):
                    mgr._emit("unsloth_epoch", {
                        "epoch": round(state.epoch, 2) if state.epoch else 0,
                        "global_step": state.global_step,
                    })

            trainer.add_callback(MetricsBroadcastCallback())

            self._emit("unsloth_training_status", {"status": "started", "method": method})

            # Train
            trainer.train()

            self._emit("unsloth_training_complete", {
                "total_steps": len(self._train_history),
                "final_loss": self._train_history[-1]["loss"] if self._train_history else None,
            })

        except Exception as e:
            self._emit("error", {"message": str(e), "traceback": traceback.format_exc()})
        finally:
            self._is_training = False

    def _create_sft_trainer(self, args, dataset, ds_config, config):
        from trl import SFTTrainer
        from unsloth import is_bfloat16_supported

        formatting = ds_config.get("formatting", "instruction")

        if formatting == "chat":
            from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

            dataset = standardize_sharegpt(dataset)

            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=args,
                max_seq_length=ds_config.get("max_seq_length", 2048),
                packing=config.get("packing", False),
            )

            if config.get("train_on_responses_only", True):
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )

            return trainer
        else:
            # Simple text column formatting
            text_col = ds_config.get("text_column", "text")

            return SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field=text_col,
                args=args,
                max_seq_length=ds_config.get("max_seq_length", 2048),
                packing=config.get("packing", False),
            )

    def _create_dpo_trainer(self, args, dataset, config):
        from trl import DPOTrainer, DPOConfig
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()

        dpo_args = DPOConfig(
            **{k: v for k, v in args.to_dict().items() if k in DPOConfig.__dataclass_fields__},
            beta=config.get("beta", 0.1),
        )

        return DPOTrainer(
            model=self.model,
            ref_model=None,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=dpo_args,
        )

    def _create_orpo_trainer(self, args, dataset, config):
        from trl import ORPOTrainer, ORPOConfig

        orpo_args = ORPOConfig(
            **{k: v for k, v in args.to_dict().items() if k in ORPOConfig.__dataclass_fields__},
            beta=config.get("beta", 0.1),
        )

        return ORPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=orpo_args,
        )

    def _create_kto_trainer(self, args, dataset, config):
        from trl import KTOTrainer, KTOConfig

        kto_args = KTOConfig(
            **{k: v for k, v in args.to_dict().items() if k in KTOConfig.__dataclass_fields__},
            beta=config.get("beta", 0.1),
        )

        return KTOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=kto_args,
        )

    def _create_reward_trainer(self, args, dataset, config):
        from trl import RewardTrainer

        return RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=args,
        )

    def save_model(self, format: str, path: str = "./sg_outputs/unsloth_model", **kwargs) -> dict:
        """Save/export the fine-tuned model."""
        if not self.is_loaded:
            return {"status": "error", "message": "No model loaded"}

        from unsloth import FastLanguageModel

        if format == "lora":
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            return {"status": "saved", "format": "lora", "path": path}

        elif format == "merged_16bit":
            self.model.save_pretrained_merged(path, self.tokenizer, save_method="merged_16bit")
            return {"status": "saved", "format": "merged_16bit", "path": path}

        elif format == "merged_4bit":
            self.model.save_pretrained_merged(path, self.tokenizer, save_method="merged_4bit")
            return {"status": "saved", "format": "merged_4bit", "path": path}

        elif format.startswith("gguf"):
            quant_map = {
                "gguf_q4": "q4_k_m",
                "gguf_q5": "q5_k_m",
                "gguf_q8": "q8_0",
                "gguf_f16": "f16",
            }
            quant = quant_map.get(format, "q4_k_m")
            self.model.save_pretrained_gguf(path, self.tokenizer, quantization_method=quant)
            return {"status": "saved", "format": format, "path": path, "quantization": quant}

        elif format == "hub":
            hub_id = kwargs.get("hub_id", "")
            if not hub_id:
                return {"status": "error", "message": "hub_id required"}
            self.model.push_to_hub_merged(hub_id, self.tokenizer, save_method="merged_16bit")
            return {"status": "pushed", "format": "hub", "hub_id": hub_id}

        elif format == "ollama":
            model_name = kwargs.get("model_name", "stategraph-model")
            self.model.save_pretrained_gguf(path, self.tokenizer, quantization_method="q4_k_m")
            # Create Modelfile
            modelfile = f'FROM {path}/unsloth.Q4_K_M.gguf\n'
            modelfile += f'TEMPLATE """{{{{ .System }}}} {{{{ .Prompt }}}}"""\n'
            import subprocess
            mf_path = f"{path}/Modelfile"
            with open(mf_path, "w") as f:
                f.write(modelfile)
            try:
                subprocess.run(["ollama", "create", model_name, "-f", mf_path], check=True)
                return {"status": "created", "format": "ollama", "model_name": model_name}
            except Exception as e:
                return {"status": "error", "message": f"GGUF saved but Ollama create failed: {e}", "path": path}

        return {"status": "error", "message": f"Unknown format: {format}"}

    def run_inference(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> dict:
        """Run inference with the loaded model."""
        if not self.is_loaded:
            return {"status": "error", "message": "No model loaded"}

        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Switch back to training mode
        FastLanguageModel.for_training(self.model)

        return {"status": "ok", "output": text, "prompt": prompt}

    def get_info(self) -> dict:
        if not self.is_loaded:
            return {"loaded": False}

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "loaded": True,
            "model_id": self.model_id,
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": round(trainable / total * 100, 2) if total > 0 else 0,
            "lora_applied": self.lora_applied,
            "is_training": self._is_training,
            "train_steps": len(self._train_history),
        }

    def get_train_history(self) -> list[dict]:
        return self._train_history
