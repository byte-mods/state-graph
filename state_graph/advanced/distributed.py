"""Distributed training — multi-GPU, multi-node, DeepSpeed, FSDP, Accelerate.

Generates configs and launch scripts. Manages distributed runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STRATEGIES = {
    "ddp": {
        "name": "PyTorch DDP",
        "description": "Data-parallel across GPUs. Simplest multi-GPU.",
        "min_gpus": 2,
        "config_params": {
            "backend": {"default": "nccl", "options": ["nccl", "gloo"]},
            "find_unused_parameters": {"default": False, "type": "bool"},
        },
    },
    "fsdp": {
        "name": "Fully Sharded Data Parallel (FSDP)",
        "description": "Shard model + optimizer + gradients across GPUs. For large models.",
        "min_gpus": 2,
        "config_params": {
            "sharding_strategy": {"default": "FULL_SHARD", "options": ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]},
            "cpu_offload": {"default": False, "type": "bool"},
            "mixed_precision": {"default": "bf16", "options": ["no", "fp16", "bf16"]},
            "activation_checkpointing": {"default": True, "type": "bool"},
        },
    },
    "deepspeed_zero2": {
        "name": "DeepSpeed ZeRO Stage 2",
        "description": "Partition optimizer states + gradients. Good for medium models.",
        "min_gpus": 1,
        "config_params": {
            "offload_optimizer": {"default": False, "type": "bool"},
            "overlap_comm": {"default": True, "type": "bool"},
            "reduce_bucket_size": {"default": 5e8, "type": "float"},
            "allgather_bucket_size": {"default": 5e8, "type": "float"},
        },
    },
    "deepspeed_zero3": {
        "name": "DeepSpeed ZeRO Stage 3",
        "description": "Partition everything. Train 100B+ models across GPUs/nodes.",
        "min_gpus": 1,
        "config_params": {
            "offload_optimizer": {"default": True, "type": "bool"},
            "offload_param": {"default": False, "type": "bool"},
            "overlap_comm": {"default": True, "type": "bool"},
            "sub_group_size": {"default": 1e9, "type": "float"},
            "stage3_max_live_parameters": {"default": 1e9, "type": "float"},
            "stage3_prefetch_bucket_size": {"default": 5e7, "type": "float"},
        },
    },
    "accelerate": {
        "name": "HuggingFace Accelerate",
        "description": "Auto-detects hardware and applies best strategy.",
        "min_gpus": 1,
        "config_params": {
            "mixed_precision": {"default": "bf16", "options": ["no", "fp16", "bf16"]},
            "gradient_accumulation_steps": {"default": 4, "type": "int"},
            "gradient_clipping": {"default": 1.0, "type": "float"},
        },
    },
}


def detect_gpus() -> dict:
    """Detect available GPUs."""
    import torch
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "memory_gb": round(props.total_mem / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    return {
        "gpu_count": len(gpus),
        "gpus": gpus,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }


def generate_deepspeed_config(stage: int = 2, params: dict = None) -> dict:
    """Generate a DeepSpeed JSON config."""
    params = params or {}
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": params.get("gradient_clipping", 1.0),
        "fp16": {"enabled": "auto"},
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": params.get("overlap_comm", True),
            "contiguous_gradients": True,
            "reduce_bucket_size": int(params.get("reduce_bucket_size", 5e8)),
            "allgather_bucket_size": int(params.get("allgather_bucket_size", 5e8)),
        },
    }

    if stage >= 2 and params.get("offload_optimizer"):
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if stage >= 3:
        config["zero_optimization"]["stage3_max_live_parameters"] = int(params.get("stage3_max_live_parameters", 1e9))
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = int(params.get("stage3_prefetch_bucket_size", 5e7))
        if params.get("offload_param"):
            config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

    return config


def generate_accelerate_config(params: dict = None) -> dict:
    """Generate an Accelerate YAML config."""
    params = params or {}
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "mixed_precision": params.get("mixed_precision", "bf16"),
        "num_machines": params.get("num_machines", 1),
        "num_processes": params.get("num_gpus", detect_gpus()["gpu_count"] or 1),
        "use_cpu": False,
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 4),
    }


def generate_launch_command(strategy: str, script: str, num_gpus: int = None, params: dict = None) -> str:
    """Generate the shell command to launch distributed training."""
    params = params or {}
    ngpu = num_gpus or detect_gpus()["gpu_count"] or 1

    if strategy == "ddp":
        return f"torchrun --nproc_per_node={ngpu} {script}"
    elif strategy.startswith("deepspeed"):
        return f"deepspeed --num_gpus={ngpu} {script} --deepspeed ds_config.json"
    elif strategy == "fsdp":
        return f"torchrun --nproc_per_node={ngpu} {script}"
    elif strategy == "accelerate":
        return f"accelerate launch --num_processes={ngpu} {script}"
    return f"python {script}"


def generate_training_script(strategy: str, model_code: str = "", params: dict = None) -> str:
    """Generate a complete distributed training script."""
    params = params or {}

    if strategy == "accelerate":
        return f'''"""Distributed training with HuggingFace Accelerate."""
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

accelerator = Accelerator(
    mixed_precision="{params.get('mixed_precision', 'bf16')}",
    gradient_accumulation_steps={params.get('gradient_accumulation_steps', 4)},
)

# Model
{model_code or "model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))"}

# Data (replace with your dataset)
dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# Accelerate prepares everything
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

# Train
for epoch in range({params.get('epochs', 10)}):
    model.train()
    for batch in loader:
        with accelerator.accumulate(model):
            x, y = batch
            loss = nn.CrossEntropyLoss()(model(x), y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    if accelerator.is_main_process:
        print(f"Epoch {{epoch}} complete")

accelerator.save_model(model, "./distributed_model")
print("Training complete!")
'''

    elif strategy.startswith("deepspeed"):
        stage = 3 if "zero3" in strategy else 2
        return f'''"""Distributed training with DeepSpeed ZeRO Stage {stage}."""
import deepspeed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# Model
{model_code or "model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))"}

# Data
dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))

# DeepSpeed init
model_engine, optimizer, loader, _ = deepspeed.initialize(
    args=args,
    model=model,
    training_data=dataset,
)

# Train
for epoch in range({params.get('epochs', 10)}):
    for batch in loader:
        x, y = batch[0].to(model_engine.device), batch[1].to(model_engine.device)
        loss = nn.CrossEntropyLoss()(model_engine(x), y)
        model_engine.backward(loss)
        model_engine.step()

    if model_engine.local_rank == 0:
        print(f"Epoch {{epoch}} complete")

print("Training complete!")
'''

    else:  # DDP / FSDP
        return f'''"""Distributed training with PyTorch {'FSDP' if strategy == 'fsdp' else 'DDP'}."""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def main():
    dist.init_process_group(backend="{params.get('backend', 'nccl')}")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{{rank}}")

    # Model
    {model_code or "model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))"}
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Data
    dataset = TensorDataset(torch.randn(1000, 784), torch.randint(0, 10, (1000,)))
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range({params.get('epochs', 10)}):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.CrossEntropyLoss()(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if rank == 0:
            print(f"Epoch {{epoch}} complete")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''
