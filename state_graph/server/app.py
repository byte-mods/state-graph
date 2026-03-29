"""FastAPI application with WebSocket support for real-time updates."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from state_graph.core.engine import TrainingEngine
from state_graph.core.registry import Registry
from state_graph.core.scheduler import SchedulerRegistry
from state_graph.core.data import DataManager

# Import custom layers to register them
import state_graph.layers.custom  # noqa: F401
import state_graph.layers.llm  # noqa: F401
try:
    import state_graph.layers.vision_advanced  # noqa: F401
except ImportError:
    pass
try:
    import state_graph.layers.diffusion_advanced  # noqa: F401
except ImportError:
    pass
try:
    import state_graph.layers.audio_advanced  # noqa: F401
except ImportError:
    pass

app = FastAPI(title="StateGraph", version="0.4.0")
engine = TrainingEngine()

# Serve uploaded files for annotation tools
_UPLOAD_DIR = Path("./sg_datasets/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_UPLOAD_DIR)), name="uploads")

# Experiment history for comparison
experiment_history: list[dict] = []
_EXPERIMENTS_FILE = Path("./experiments.json")


def _save_experiments():
    """Persist experiments to disk."""
    try:
        _EXPERIMENTS_FILE.write_text(json.dumps(experiment_history, default=str))
    except Exception:
        pass


def _load_experiments():
    """Load experiments from disk on startup."""
    global experiment_history
    if _EXPERIMENTS_FILE.exists():
        try:
            experiment_history = json.loads(_EXPERIMENTS_FILE.read_text())
        except Exception:
            experiment_history = []


_load_experiments()

# WebSocket connection manager
connected_clients: set[WebSocket] = set()


async def broadcast(event: str, data: dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    global connected_clients
    message = json.dumps({"event": event, "data": data}, default=str)
    disconnected = set()
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    connected_clients -= disconnected


engine.set_broadcast(broadcast)


@app.on_event("startup")
async def startup():
    engine.set_event_loop(asyncio.get_event_loop())


# --- REST API ---

@app.get("/")
async def index():
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    return HTMLResponse(ui_path.read_text())


@app.get("/api/registry")
async def get_registry():
    result = Registry.list_all()
    result["schedulers"] = SchedulerRegistry.list_all()
    result["scheduler_defaults"] = {
        name: SchedulerRegistry.get_default_params(name)
        for name in SchedulerRegistry.list_all()
    }
    return result


@app.get("/api/status")
async def get_status():
    status = engine.get_status()
    status["data_info"] = engine.data_manager.get_info()
    return status


@app.get("/api/graph")
async def get_graph():
    return engine.graph.to_dict()


@app.post("/api/graph/layer")
async def add_layer(body: dict[str, Any]):
    node_id = engine.graph.add_layer(
        layer_type=body["layer_type"],
        params=body.get("params", {}),
        activation=body.get("activation"),
        position=body.get("position"),
        group=body.get("group", "main"),
        inputs=body.get("inputs"),
        merge_mode=body.get("merge_mode"),
    )
    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)
    return {"node_id": node_id, "graph": graph_data}


@app.delete("/api/graph/layer/{node_id}")
async def remove_layer(node_id: str):
    engine.graph.remove_layer(node_id)
    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)
    return {"graph": graph_data}


@app.put("/api/graph/layer/{node_id}")
async def update_layer(node_id: str, body: dict[str, Any]):
    engine.graph.update_layer(
        node_id=node_id,
        layer_type=body.get("layer_type"),
        params=body.get("params"),
        activation=body.get("activation"),
        inputs=body.get("inputs"),
        merge_mode=body.get("merge_mode"),
    )
    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)
    return {"graph": graph_data}


@app.post("/api/graph/skip")
async def add_skip_connection(body: dict[str, Any]):
    """Add a skip/residual connection between two layers."""
    engine.graph.add_skip_connection(
        from_id=body["from_id"],
        to_id=body["to_id"],
        merge_mode=body.get("merge_mode", "add"),
    )
    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)
    return {"graph": graph_data}


@app.post("/api/graph/layer/{node_id}/reorder")
async def reorder_layer(node_id: str, body: dict[str, Any]):
    new_position = body["position"]
    engine.graph.reorder_layer(node_id, new_position)
    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)
    return {"graph": graph_data}


@app.post("/api/config")
async def update_config(body: dict[str, Any]):
    engine.config.update(body)
    return {"config": engine.config}


@app.post("/api/formula")
async def register_formula(body: dict[str, Any]):
    name = body["name"]
    expr = body["expression"]
    Registry.register_formula_from_string(name, expr)
    return {"status": "registered", "name": name, "activations": Registry.list_activations()}


@app.post("/api/build")
async def build_model():
    result = engine.build()
    await broadcast("model_built", result)
    return result


# --- Dataset endpoints ---

@app.post("/api/data/load")
async def load_data(body: dict[str, Any]):
    """Unified data loading for synthetic and real datasets."""
    dataset = body.get("dataset", "random")
    n_samples = body.get("n_samples", 1000)
    dataset_type = body.get("type", "synthetic")

    dm = engine.data_manager

    if dataset_type == "real":
        result = dm.load_real(dataset, data_dir=body.get("data_dir", "./data"))
    else:
        result = dm.load_builtin(dataset, n_samples)

    # Set data on engine
    flatten = not dm._is_image
    engine.set_data(
        dm.x_train.view(dm.x_train.shape[0], -1) if flatten else dm.x_train,
        dm.y_train,
        dm.x_val.view(dm.x_val.shape[0], -1) if flatten and dm.x_val is not None else dm.x_val,
        dm.y_val,
    )

    return result


@app.post("/api/data/sample")
async def load_sample_data(body: dict[str, Any]):
    """Load a sample dataset for quick experimentation (backwards compatible)."""
    body["type"] = "synthetic"
    return await load_data(body)


@app.get("/api/data/info")
async def get_data_info():
    return engine.data_manager.get_info()


@app.post("/api/data/augmentation")
async def set_augmentation(body: dict[str, Any]):
    """Set data augmentation pipeline."""
    augs = body.get("augmentations", [])
    engine.data_manager.set_augmentations(augs)
    return {"status": "ok", "augmentations": augs}


@app.get("/api/data/augmentations")
async def list_augmentations():
    return {
        "available": DataManager.AUGMENTATIONS,
        "active": engine.data_manager.augmentations,
    }


# --- Export/Import ---

@app.get("/api/export/architecture")
async def export_architecture():
    return engine.export_architecture()


@app.post("/api/import/architecture")
async def import_architecture(body: dict[str, Any]):
    result = engine.import_architecture(body)
    await broadcast("graph_updated", result["graph"])
    return result


@app.get("/api/export/python")
async def export_python():
    code = engine.export_python()
    return PlainTextResponse(code, media_type="text/plain")


# --- System Resources ---

@app.get("/api/system/resources")
async def get_system_resources():
    """Get current CPU, RAM, and GPU usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()

    result = {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_used_gb": round(mem.used / (1024**3), 1),
        "ram_percent": mem.percent,
        "process_ram_mb": round(proc_mem.rss / (1024**2), 1),
    }

    # GPU info if available
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_mem_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
        result["gpu_mem_used_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        result["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        result["gpu_utilization"] = None  # Would need pynvml for this
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        result["gpu_name"] = "Apple MPS"
        result["gpu_mem_total_gb"] = None
        result["gpu_mem_used_gb"] = None

    return result


@app.post("/api/profile")
async def profile_model(body: dict[str, Any]):
    """Profile the current model: per-layer timing, memory, bottleneck detection.

    Body: {input_shape: [1, 784], n_runs: 10}
    """
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}

    input_shape = body.get("input_shape")
    if not input_shape:
        return {"status": "error", "message": "Provide input_shape, e.g. [1, 784] or [1, 3, 32, 32]"}

    n_runs = body.get("n_runs", 10)

    try:
        result = engine.metrics.profile_model(engine.model, tuple(input_shape), n_runs)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/bottleneck")
async def get_bottleneck():
    """Get the latest bottleneck analysis from training metrics."""
    if not engine.metrics._loss_history:
        return {"status": "no_data", "message": "No training data yet. Start training first."}

    last_step = engine.metrics._loss_history[-1] if engine.metrics._loss_history else {}

    # Collect current bottleneck data
    candidates = []
    if engine.model:
        for name, module in engine.model.named_modules():
            if name == "" or not list(module.parameters(recurse=False)):
                continue
            params = sum(p.numel() for p in module.parameters(recurse=False))
            grad_norm = 0.0
            for p in module.parameters(recurse=False):
                if p.grad is not None:
                    grad_norm = p.grad.norm().item()
            mem = sum(p.nelement() * p.element_size() for p in module.parameters(recurse=False)) / (1024 * 1024)

            fwd_time = engine.metrics._timing_cache.get(name, 0.0)

            candidates.append({
                "name": name,
                "type": type(module).__name__,
                "params": params,
                "forward_time_ms": fwd_time,
                "memory_mb": mem,
                "grad_norm": grad_norm,
                "dead_neuron_pct": 0.0,
                "grad_to_weight_ratio": 0.0,
            })

    analysis = engine.metrics._analyze_bottlenecks(candidates)
    analysis["layers"] = candidates
    return {"status": "ok", **analysis}


@app.get("/api/metrics/snapshot")
async def get_metrics_snapshot():
    """Get current metrics snapshot for UI initialization."""
    return engine.metrics.get_snapshot()


@app.get("/api/metrics/layer/{name:path}")
async def get_layer_metrics(name: str, last_n: int = 100):
    """Get metrics history for a specific layer."""
    return {"layer": name, "history": engine.metrics.get_layer_history(name, last_n)}


# --- Experiment Comparison ---

@app.post("/api/experiments/save")
async def save_experiment(body: dict[str, Any]):
    """Save current training run as an experiment for comparison."""
    name = body.get("name", f"Experiment {len(experiment_history) + 1}")
    epochs = engine.metrics.get_epoch_metrics()
    loss_hist = engine.metrics.get_loss_history()

    exp = {
        "id": len(experiment_history),
        "name": name,
        "timestamp": time.time(),
        "architecture": [n.to_dict() for n in engine.graph.get_sorted_nodes()],
        "config": {**engine.config},
        "dataset": engine.data_manager.dataset_name,
        "total_params": sum(
            info["trainable"]
            for info in engine.graph.get_param_count().values()
        ) if engine.model else 0,
        "epochs": epochs,
        "final_train_loss": epochs[-1]["train_loss"] if epochs else None,
        "final_val_loss": epochs[-1]["val_loss"] if epochs else None,
        "final_train_acc": epochs[-1]["train_acc"] if epochs else None,
        "final_val_acc": epochs[-1]["val_acc"] if epochs else None,
        "best_val_loss": min((e["val_loss"] for e in epochs if e["val_loss"] is not None), default=None),
        "best_val_acc": max((e["val_acc"] for e in epochs if e["val_acc"] is not None), default=None),
        "loss_curve": [{"step": h["step"], "loss": h["loss"]} for h in loss_hist[::max(1, len(loss_hist)//200)]],
    }
    experiment_history.append(exp)
    _save_experiments()
    return {"status": "saved", "experiment": exp}


@app.get("/api/experiments")
async def list_experiments():
    """List all saved experiments."""
    return {"experiments": [
        {
            "id": e["id"],
            "name": e["name"],
            "dataset": e["dataset"],
            "total_params": e["total_params"],
            "final_train_loss": e["final_train_loss"],
            "final_val_loss": e["final_val_loss"],
            "final_train_acc": e["final_train_acc"],
            "final_val_acc": e["final_val_acc"],
            "best_val_loss": e["best_val_loss"],
            "best_val_acc": e["best_val_acc"],
            "arch_summary": " -> ".join(n["layer_type"] for n in e["architecture"]),
            "config_summary": f"{e['config']['optimizer']} lr={e['config']['learning_rate']}",
        }
        for e in experiment_history
    ]}


@app.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: int):
    if exp_id < 0 or exp_id >= len(experiment_history):
        return {"status": "error", "message": "Experiment not found"}
    return experiment_history[exp_id]


@app.delete("/api/experiments/{exp_id}")
async def delete_experiment(exp_id: int):
    if exp_id < 0 or exp_id >= len(experiment_history):
        return {"status": "error", "message": "Experiment not found"}
    experiment_history.pop(exp_id)
    # Reindex
    for i, e in enumerate(experiment_history):
        e["id"] = i
    _save_experiments()
    return {"status": "deleted"}


@app.get("/api/experiments/compare")
async def compare_experiments():
    """Get all experiments formatted for comparison charts."""
    return {
        "experiments": [
            {
                "id": e["id"],
                "name": e["name"],
                "loss_curve": e["loss_curve"],
                "epochs": e["epochs"],
                "total_params": e["total_params"],
                "best_val_acc": e["best_val_acc"],
                "best_val_loss": e["best_val_loss"],
            }
            for e in experiment_history
        ]
    }


# --- LLM Reasoning Benchmark ---

REASONING_BENCHMARKS = {
    "math": [
        {"prompt": "What is 15% of 240?", "answer": "36", "type": "arithmetic"},
        {"prompt": "If x + 3 = 10, what is x?", "answer": "7", "type": "algebra"},
        {"prompt": "A train travels 120 miles in 2 hours. What is its average speed in mph?", "answer": "60", "type": "word_problem"},
        {"prompt": "What is the sum of the first 10 positive integers?", "answer": "55", "type": "series"},
        {"prompt": "If a rectangle has length 8 and width 5, what is its area?", "answer": "40", "type": "geometry"},
        {"prompt": "What is 3^4?", "answer": "81", "type": "arithmetic"},
        {"prompt": "Solve: 2x - 5 = 11", "answer": "8", "type": "algebra"},
        {"prompt": "A store offers 20% off a $50 item. What is the sale price?", "answer": "40", "type": "word_problem"},
        {"prompt": "What is the factorial of 5?", "answer": "120", "type": "arithmetic"},
        {"prompt": "If f(x) = 2x + 1, what is f(4)?", "answer": "9", "type": "function"},
    ],
    "logic": [
        {"prompt": "All cats are animals. Whiskers is a cat. Is Whiskers an animal? Answer yes or no.", "answer": "yes", "type": "syllogism"},
        {"prompt": "If it rains, the ground is wet. The ground is wet. Did it rain? Answer: yes, no, or uncertain.", "answer": "uncertain", "type": "affirming_consequent"},
        {"prompt": "A is taller than B. B is taller than C. Is A taller than C? Answer yes or no.", "answer": "yes", "type": "transitive"},
        {"prompt": "If all dogs bark and Rex is a dog, does Rex bark? Answer yes or no.", "answer": "yes", "type": "modus_ponens"},
        {"prompt": "None of the birds are fish. A penguin is a bird. Is a penguin a fish? Answer yes or no.", "answer": "no", "type": "negation"},
        {"prompt": "If P implies Q and Q is false, is P true or false?", "answer": "false", "type": "modus_tollens"},
        {"prompt": "Some A are B. Some B are C. Must some A be C? Answer yes or no.", "answer": "no", "type": "quantifier"},
        {"prompt": "If today is Monday, tomorrow is Tuesday. Today is Monday. What day is tomorrow?", "answer": "tuesday", "type": "modus_ponens"},
    ],
    "code": [
        {"prompt": "What does `len([1,2,3])` return in Python?", "answer": "3", "type": "python"},
        {"prompt": "What is the output of `print(2**3)` in Python?", "answer": "8", "type": "python"},
        {"prompt": "In Python, what does `'hello'.upper()` return?", "answer": "HELLO", "type": "python"},
        {"prompt": "What is the time complexity of binary search? Answer: O(log n) or O(n)", "answer": "o(log n)", "type": "complexity"},
        {"prompt": "What does `bool([])` return in Python?", "answer": "false", "type": "python"},
        {"prompt": "What is `list(range(5))`?", "answer": "[0, 1, 2, 3, 4]", "type": "python"},
        {"prompt": "What does `'abc'[1]` return in Python?", "answer": "b", "type": "python"},
        {"prompt": "What is `type(3.14).__name__` in Python?", "answer": "float", "type": "python"},
    ],
    "commonsense": [
        {"prompt": "Water freezes at what temperature in Celsius?", "answer": "0", "type": "science"},
        {"prompt": "How many days are in a leap year?", "answer": "366", "type": "calendar"},
        {"prompt": "Which is heavier: a kilogram of steel or a kilogram of feathers?", "answer": "same", "type": "trick"},
        {"prompt": "If you drop a ball, does it go up or down?", "answer": "down", "type": "physics"},
        {"prompt": "How many continents are there?", "answer": "7", "type": "geography"},
        {"prompt": "What comes after Wednesday?", "answer": "thursday", "type": "sequence"},
        {"prompt": "Is the sun a star or a planet?", "answer": "star", "type": "astronomy"},
        {"prompt": "How many legs does a spider have?", "answer": "8", "type": "biology"},
    ],
    "reading": [
        {"prompt": "Read: 'The cat sat on the mat.' Question: Where did the cat sit?", "answer": "mat", "type": "extraction"},
        {"prompt": "Read: 'John is 10 years old. Mary is 2 years older.' How old is Mary?", "answer": "12", "type": "inference"},
        {"prompt": "Read: 'It was raining so Tom took an umbrella.' Why did Tom take an umbrella?", "answer": "raining", "type": "causal"},
        {"prompt": "Read: 'The store closes at 9pm. It is now 10pm.' Is the store open? Answer yes or no.", "answer": "no", "type": "inference"},
        {"prompt": "Read: 'Alice has 3 apples, Bob has 5.' Who has more apples?", "answer": "bob", "type": "comparison"},
    ],
    "chain_of_thought": [
        {"prompt": "If I have 3 boxes with 4 balls each, and I remove 2 balls, how many balls remain? Think step by step.", "answer": "10", "type": "multi_step"},
        {"prompt": "A shop sells pens for $2 each. I buy 5 pens and pay with a $20 bill. How much change do I get?", "answer": "10", "type": "multi_step"},
        {"prompt": "There are 12 eggs. I use 3 for breakfast and give half the rest to my neighbor. How many do I have left?", "answer": "4.5", "type": "multi_step"},
        {"prompt": "A car uses 5 liters per 100km. How many liters for 250km?", "answer": "12.5", "type": "proportion"},
        {"prompt": "I start with 100 coins. I spend 30, earn 20, then spend 15. How many coins do I have?", "answer": "75", "type": "sequential"},
    ],
}


def _check_answer(generated: str, expected: str) -> bool:
    """Check if generated text contains the expected answer."""
    gen_lower = generated.lower().strip()
    exp_lower = expected.lower().strip()
    # Direct containment
    if exp_lower in gen_lower:
        return True
    # Check for numeric answers
    import re
    numbers = re.findall(r'-?\d+\.?\d*', gen_lower)
    if numbers and exp_lower in [n.rstrip('0').rstrip('.') if '.' in n else n for n in numbers]:
        return True
    if numbers and exp_lower in numbers:
        return True
    return False


@app.post("/api/benchmark/reasoning")
async def benchmark_reasoning(body: dict[str, Any]):
    """Run LLM reasoning benchmark on the current model."""
    if engine.model is None:
        return {"status": "error", "message": "No model loaded. Train or load a model first."}

    import torch, time as t

    suite = body.get("suite", "all")
    max_samples = body.get("samples", 50)
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.1)

    # Gather benchmark questions
    if suite == "all":
        questions = []
        for cat, qs in REASONING_BENCHMARKS.items():
            for q in qs:
                questions.append({**q, "category": cat})
    elif suite in REASONING_BENCHMARKS:
        questions = [{**q, "category": suite} for q in REASONING_BENCHMARKS[suite]]
    else:
        return {"status": "error", "message": f"Unknown benchmark suite: {suite}"}

    # Limit samples
    import random
    if len(questions) > max_samples:
        questions = random.sample(questions, max_samples)

    results_by_cat = {}
    total_correct = 0
    total_count = 0
    total_latency = 0.0

    model = engine.model
    model.eval()

    # Check if model has a generate method (LLM) or is a classifier
    has_generate = hasattr(model, 'generate')
    has_tokenizer = hasattr(engine, 'tokenizer') and engine.tokenizer is not None
    # Also check hf_manager
    tokenizer = None
    if has_tokenizer:
        tokenizer = engine.tokenizer
    elif engine.hf_manager and hasattr(engine.hf_manager, 'tokenizer') and engine.hf_manager.tokenizer:
        tokenizer = engine.hf_manager.tokenizer
        has_tokenizer = True

    if not has_generate or not has_tokenizer:
        # For non-LLM models, do a simulated benchmark based on model confidence
        for q in questions:
            cat = q["category"]
            if cat not in results_by_cat:
                results_by_cat[cat] = {"correct": 0, "total": 0, "score": 0.0}

            start = t.time()
            try:
                dummy = torch.randn(1, *([model.config.hidden_size] if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else [64])).to(engine.device)
                with torch.no_grad():
                    out = model(dummy) if not isinstance(dummy, dict) else model(**dummy)
                latency = (t.time() - start) * 1000
                total_latency += latency
            except Exception:
                latency = 0
            results_by_cat[cat]["total"] += 1
            total_count += 1

        # Non-LLM models can't really do reasoning
        for cat in results_by_cat:
            results_by_cat[cat]["score"] = 0.0

        return {
            "status": "completed",
            "results": {
                "overall_score": 0.0,
                "categories": results_by_cat,
                "avg_latency_ms": total_latency / max(total_count, 1),
                "total_time_sec": total_latency / 1000,
                "note": "Model does not support text generation. Load an LLM for reasoning benchmarks.",
            }
        }

    # LLM benchmark with tokenizer + generate
    for q in questions:
        cat = q["category"]
        if cat not in results_by_cat:
            results_by_cat[cat] = {"correct": 0, "total": 0, "score": 0.0}

        start = t.time()
        try:
            inputs = tokenizer(q["prompt"], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(engine.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
                if temperature > 0:
                    gen_kwargs["temperature"] = temperature
                output_ids = model.generate(**inputs, **gen_kwargs)

            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            latency = (t.time() - start) * 1000
            total_latency += latency

            correct = _check_answer(generated, q["answer"])
            if correct:
                results_by_cat[cat]["correct"] += 1
                total_correct += 1
        except Exception:
            latency = (t.time() - start) * 1000
            total_latency += latency

        results_by_cat[cat]["total"] += 1
        total_count += 1

    # Compute scores
    for cat in results_by_cat:
        d = results_by_cat[cat]
        d["score"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    overall = total_correct / total_count if total_count > 0 else 0.0

    return {
        "status": "completed",
        "results": {
            "overall_score": overall,
            "categories": results_by_cat,
            "avg_latency_ms": total_latency / max(total_count, 1),
            "total_time_sec": total_latency / 1000,
        }
    }


# --- LLM Builder ---

# Formula descriptions for architecture visualization
COMPONENT_FORMULAS = {
    # Layers
    "Linear": {
        "formula": "y = xW^T + b",
        "description": "Affine transformation. Multiplies input by weight matrix and adds bias.",
        "params": "W ∈ R^{out×in}, b ∈ R^{out}",
    },
    "Conv2d": {
        "formula": "y[i,j] = Σ_k Σ_m Σ_n W[k,m,n] · x[i+m, j+n] + b",
        "description": "2D convolution with learnable kernels. Slides filter across spatial dims.",
        "params": "W ∈ R^{C_out×C_in×kH×kW}",
    },
    "Conv1d": {
        "formula": "y[i] = Σ_k Σ_j W[k,j] · x[i+j] + b",
        "description": "1D convolution across temporal/sequence dimension.",
        "params": "W ∈ R^{C_out×C_in×kW}",
    },
    "BatchNorm1d": {
        "formula": "y = γ · (x - μ_B) / √(σ²_B + ε) + β",
        "description": "Normalizes per-feature across the batch. μ_B, σ²_B are batch statistics.",
        "params": "γ, β ∈ R^{C} (learnable scale and shift)",
    },
    "BatchNorm2d": {
        "formula": "y = γ · (x - μ_B) / √(σ²_B + ε) + β",
        "description": "Batch normalization for 2D (image) inputs, per-channel statistics.",
        "params": "γ, β ∈ R^{C}",
    },
    "LayerNorm": {
        "formula": "y = γ · (x - μ) / √(σ² + ε) + β",
        "description": "Normalizes across the last dimension. μ, σ² are per-sample statistics.",
        "params": "γ, β ∈ R^{d} (learnable)",
    },
    "Dropout": {
        "formula": "y_i = x_i · m_i / (1-p), where m_i ~ Bernoulli(1-p)",
        "description": "Randomly zeros elements during training for regularization.",
        "params": "p = drop probability",
    },
    "Embedding": {
        "formula": "y = E[input_ids], E ∈ R^{V×d}",
        "description": "Lookup table mapping integer indices to dense vectors.",
        "params": "E ∈ R^{vocab_size × embedding_dim}",
    },
    "LSTM": {
        "formula": "f_t = σ(W_f·[h_{t-1},x_t]+b_f)\ni_t = σ(W_i·[h_{t-1},x_t]+b_i)\ng_t = tanh(W_g·[h_{t-1},x_t]+b_g)\no_t = σ(W_o·[h_{t-1},x_t]+b_o)\nc_t = f_t⊙c_{t-1} + i_t⊙g_t\nh_t = o_t⊙tanh(c_t)",
        "description": "Long Short-Term Memory. Gates control information flow through cell state.",
        "params": "W_f, W_i, W_g, W_o ∈ R^{4h×(h+x)}",
    },
    "GRU": {
        "formula": "z_t = σ(W_z·[h_{t-1},x_t])\nr_t = σ(W_r·[h_{t-1},x_t])\nh̃_t = tanh(W·[r_t⊙h_{t-1},x_t])\nh_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t",
        "description": "Gated Recurrent Unit. Simplified LSTM with update and reset gates.",
        "params": "W_z, W_r, W ∈ R^{3h×(h+x)}",
    },
    "MultiheadAttention": {
        "formula": "Attn(Q,K,V) = softmax(QK^T/√d_k)V\nMultiHead = Concat(head_1,...,head_h)W^O\nhead_i = Attn(xW^Q_i, xW^K_i, xW^V_i)",
        "description": "Scaled dot-product attention across multiple parallel heads.",
        "params": "W^Q, W^K, W^V ∈ R^{d×d_k}, W^O ∈ R^{hd_k×d}",
    },
    "Flatten": {
        "formula": "y = reshape(x, [batch, -1])",
        "description": "Flattens all dimensions except batch into a single vector.",
        "params": "No learnable parameters",
    },
    "MaxPool2d": {
        "formula": "y[i,j] = max(x[i·s:i·s+k, j·s:j·s+k])",
        "description": "Takes the maximum value in each pooling window.",
        "params": "kernel_size, stride",
    },
    "AdaptiveAvgPool2d": {
        "formula": "y = mean(x over adaptive regions)",
        "description": "Average pools to a fixed output spatial size regardless of input size.",
        "params": "output_size",
    },
    # Custom layers
    "ResidualBlock": {
        "formula": "y = F(x) + x\nF(x) = W_2 · ReLU(W_1 · x + b_1) + b_2",
        "description": "Skip connection enables gradient flow. Core building block of ResNets.",
        "params": "W_1 ∈ R^{h×d}, W_2 ∈ R^{d×h}",
    },
    "GatedLinearUnit": {
        "formula": "y = (Wx + b) ⊙ σ(Vx + c)",
        "description": "GLU gates information flow with element-wise sigmoid gating.",
        "params": "W, V ∈ R^{out×in}",
    },
    "SwishLinear": {
        "formula": "y = (Wx+b) · σ(β · (Wx+b))\nSwish(x) = x · sigmoid(βx)",
        "description": "Swish activation with learnable β parameter.",
        "params": "W ∈ R^{out×in}, β ∈ R^{1}",
    },
    "TransformerBlock": {
        "formula": "h = x + MultiHeadAttn(LayerNorm(x))\ny = h + FFN(LayerNorm(h))\nFFN = W_2·GELU(W_1·x+b_1)+b_2",
        "description": "Pre-norm Transformer encoder block with self-attention and feed-forward.",
        "params": "Attention + FFN params, LayerNorm γ,β",
    },
    "PositionalEncoding": {
        "formula": "PE(pos,2i) = sin(pos/10000^{2i/d})\nPE(pos,2i+1) = cos(pos/10000^{2i/d})\ny = x + PE[:seq_len]",
        "description": "Injects position information via fixed sinusoidal patterns.",
        "params": "No learnable params (fixed sin/cos buffer)",
    },
    "TokenEmbedding": {
        "formula": "y = reshape(W·x, [batch, seq_len, d_model])\nW ∈ R^{(seq·d)×in}",
        "description": "Projects flat features into sequence of token embeddings.",
        "params": "W ∈ R^{(seq_len·d_model)×in_features}",
    },
    "SequencePool": {
        "formula": "mean: y = (1/L)Σ_t x_t\ncls: y = x_0\nmax: y = max_t(x_t)",
        "description": "Reduces sequence to single vector for classification head.",
        "params": "No learnable parameters",
    },
    "SqueezeExcite": {
        "formula": "s = σ(W_2·ReLU(W_1·GAP(x)))\ny = x ⊙ s",
        "description": "Channel attention: squeeze via global avg pool, excite via FC + sigmoid.",
        "params": "W_1 ∈ R^{C/r×C}, W_2 ∈ R^{C×C/r}",
    },
    # LLM layers
    "RMSNorm": {
        "formula": "y = x / √(mean(x²) + ε) · γ\nRMS(x) = √(1/d · Σx_i²)",
        "description": "Root Mean Square normalization. Simpler than LayerNorm (no mean subtraction/bias).",
        "params": "γ ∈ R^{d} (learnable scale)",
    },
    "RotaryPositionalEmbedding": {
        "formula": "θ_i = base^{-2i/d}\nR(pos) = [[cos(pos·θ), -sin(pos·θ)],\n           [sin(pos·θ),  cos(pos·θ)]]\nq' = R(pos)·q,  k' = R(pos)·k",
        "description": "Rotary Position Embedding. Encodes position by rotating Q,K in pairs of dimensions.",
        "params": "No learnable params (precomputed sin/cos)",
    },
    "LLMAttention": {
        "formula": "Q=xW_Q, K=xW_K, V=xW_V (with RoPE on Q,K)\nAttn = softmax(QK^T/√d_k + causal_mask)·V\n[Flash Attn: fused O(N) memory]\n[GQA: K,V have fewer heads, repeated for Q]",
        "description": "Multi-Head Attention with RoPE, Flash Attention (SDPA), and Grouped-Query Attention.",
        "params": "W_Q ∈ R^{d×(h·d_k)}, W_K,W_V ∈ R^{d×(h_kv·d_k)}, W_O ∈ R^{(h·d_k)×d}",
    },
    "SwiGLUFFN": {
        "formula": "gate = SiLU(x·W_gate) = (x·W_gate)·σ(x·W_gate)\nup = x·W_up\ny = (gate ⊙ up)·W_down",
        "description": "SwiGLU FFN used in Llama/PaLM. Gated linear unit with Swish activation.",
        "params": "W_gate, W_up ∈ R^{d×h}, W_down ∈ R^{h×d}",
    },
    "GeGLUFFN": {
        "formula": "gate = GELU(x·W_gate)\nup = x·W_up\ny = (gate ⊙ up)·W_down",
        "description": "GELU-Gated Linear Unit FFN. Alternative to SwiGLU used in T5/Gemma variants.",
        "params": "W_gate, W_up ∈ R^{d×h}, W_down ∈ R^{h×d}",
    },
    "MoELayer": {
        "formula": "probs = softmax(x·W_gate)\ntop_k = TopK(probs, k)\ny = Σ_{i∈top_k} w_i · Expert_i(x)\nL_balance = n·Σ(f_i·P_i)",
        "description": "Mixture of Experts with top-k routing. Each expert is a SwiGLU FFN.",
        "params": "W_gate ∈ R^{d×n_experts}, each expert has SwiGLU params",
    },
    "LLMDecoderBlock": {
        "formula": "h = x + Attn(RMSNorm(x))\ny = h + FFN(RMSNorm(h))\n[Attn: RoPE + GQA + Flash]\n[FFN: SwiGLU or MoE]",
        "description": "Full Llama-style decoder block. Pre-norm with RMSNorm, residual connections.",
        "params": "RMSNorm γ + Attention W_Q,K,V,O + FFN W_gate,up,down",
    },
    "LLMModel": {
        "formula": "x = Embedding(tokens)\nfor layer in layers:\n  x = DecoderBlock(x)\nlogits = RMSNorm(x) · W_head\nloss = CrossEntropy(logits[:-1], tokens[1:])",
        "description": "Full decoder-only LLM. Supports RoPE, Flash Attn, GQA, MoE, weight tying.",
        "params": "E ∈ R^{V×d}, N×DecoderBlock, W_head = E^T (tied)",
    },
    # Activations
    "ReLU": {
        "formula": "y = max(0, x)",
        "description": "Rectified Linear Unit. Zero for negative, identity for positive.",
    },
    "LeakyReLU": {
        "formula": "y = max(αx, x), α=0.01",
        "description": "Allows small gradient for negative values to avoid dead neurons.",
    },
    "GELU": {
        "formula": "y = x · Φ(x) ≈ x · σ(1.702x)\nΦ(x) = CDF of N(0,1)",
        "description": "Gaussian Error Linear Unit. Smooth approximation used in BERT/GPT.",
    },
    "SiLU": {
        "formula": "y = x · σ(x) = x / (1 + e^{-x})",
        "description": "Sigmoid Linear Unit (Swish). Self-gated activation used in modern LLMs.",
    },
    "Sigmoid": {
        "formula": "y = 1 / (1 + e^{-x})",
        "description": "Squashes input to (0, 1) range. Used for binary classification/gating.",
    },
    "Tanh": {
        "formula": "y = (e^x - e^{-x}) / (e^x + e^{-x})",
        "description": "Hyperbolic tangent. Squashes to (-1, 1). Used in RNNs.",
    },
    "Softmax": {
        "formula": "y_i = e^{x_i} / Σ_j e^{x_j}",
        "description": "Converts logits to probability distribution summing to 1.",
    },
    "ELU": {
        "formula": "y = x if x>0, else α(e^x - 1)",
        "description": "Exponential Linear Unit. Smooth for negative values, reduces bias shift.",
    },
    "PReLU": {
        "formula": "y = max(0,x) + a·min(0,x)\na is learnable",
        "description": "Parametric ReLU with learnable negative slope.",
    },
    "Mish": {
        "formula": "y = x · tanh(softplus(x))\n  = x · tanh(ln(1 + e^x))",
        "description": "Self-regularized non-monotonic activation. Smooth and bounded below.",
    },
    # Losses
    "CrossEntropyLoss": {
        "formula": "L = -Σ_c y_c · log(softmax(x)_c)\n  = -log(e^{x_y} / Σ_j e^{x_j})",
        "description": "Standard classification loss. Combines LogSoftmax + NLLLoss.",
    },
    "MSELoss": {
        "formula": "L = (1/n) Σ (y_i - ŷ_i)²",
        "description": "Mean Squared Error for regression tasks.",
    },
    "BCELoss": {
        "formula": "L = -(y·log(p) + (1-y)·log(1-p))",
        "description": "Binary Cross-Entropy. Input must be probabilities (after sigmoid).",
    },
    "NLLLoss": {
        "formula": "L = -log(p_y)",
        "description": "Negative Log-Likelihood. Input should be log-probabilities.",
    },
    "HuberLoss": {
        "formula": "L = 0.5(y-ŷ)² if |y-ŷ|<δ\n  = δ(|y-ŷ| - 0.5δ) otherwise",
        "description": "Smooth L1 loss. Less sensitive to outliers than MSE.",
    },
    "KLDivLoss": {
        "formula": "L = Σ p(x)·log(p(x)/q(x))",
        "description": "Kullback-Leibler divergence between two distributions.",
    },
}


@app.post("/api/llm/build")
async def build_llm(body: dict[str, Any]):
    """Build a custom LLM from configuration."""
    from state_graph.layers.llm import LLMModel

    config = {
        "vocab_size": body.get("vocab_size", 32000),
        "d_model": body.get("d_model", 512),
        "n_layers": body.get("n_layers", 6),
        "n_heads": body.get("n_heads", 8),
        "n_kv_heads": body.get("n_kv_heads"),
        "ffn_hidden_dim": body.get("ffn_hidden_dim"),
        "max_len": body.get("max_len", 2048),
        "dropout": body.get("dropout", 0.0),
        "use_flash": body.get("use_flash", True),
        "use_moe": body.get("use_moe", False),
        "n_experts": body.get("n_experts", 8),
        "moe_top_k": body.get("moe_top_k", 2),
        "moe_layers": body.get("moe_layers"),
        "tie_weights": body.get("tie_weights", True),
        "norm_type": body.get("norm_type", "rmsnorm"),
        "ffn_type": body.get("ffn_type", "swiglu"),
        "rope_base": body.get("rope_base", 10000.0),
        "bias": body.get("bias", False),
        "layer_configs": body.get("layer_configs"),
    }

    model = LLMModel(**config)
    engine.model = model.to(engine.device)
    engine.model_source = "llm"

    # Store config for reference
    engine._llm_config = config

    param_info = model.count_parameters()
    arch = _get_llm_architecture(model, config)

    await broadcast("model_built", {
        "status": "built",
        "model_type": "llm",
        "config": config,
        "total_params": param_info["total"],
        "total_params_M": param_info["total_M"],
        "device": str(engine.device),
        "architecture": arch,
    })

    return {
        "status": "built",
        "config": config,
        "params": param_info,
        "device": str(engine.device),
        "architecture": arch,
    }


def _get_llm_architecture(model, config: dict) -> list[dict]:
    """Extract architecture tree for visualization."""
    arch = []
    arch.append({
        "name": "Token Embedding",
        "type": "Embedding",
        "params": {"vocab_size": config["vocab_size"], "d_model": config["d_model"]},
        "shape": f"R^{{{config['vocab_size']}×{config['d_model']}}}",
        "param_count": model.tok_emb.weight.numel(),
    })

    for i, layer in enumerate(model.layers):
        is_moe = hasattr(layer.ffn, 'n_experts')
        norm_type = getattr(layer, 'norm_type', config.get('norm_type', 'rmsnorm'))
        ffn_type = getattr(layer, 'ffn_type', config.get('ffn_type', 'swiglu'))
        norm_name = "LayerNorm" if norm_type == "layernorm" else "RMSNorm"
        layer_n_heads = layer.attn.n_heads
        layer_n_kv_heads = layer.attn.n_kv_heads
        layer_head_dim = layer.attn.head_dim
        frozen = all(not p.requires_grad for p in layer.parameters())

        block = {
            "name": f"Decoder Block {i}" + (" [frozen]" if frozen else ""),
            "type": "LLMDecoderBlock",
            "children": [
                {
                    "name": f"{norm_name} (pre-attn)",
                    "type": norm_name,
                    "params": {"d_model": config["d_model"]},
                },
                {
                    "name": "Attention",
                    "type": "LLMAttention",
                    "params": {
                        "n_heads": layer_n_heads,
                        "n_kv_heads": layer_n_kv_heads,
                        "head_dim": layer_head_dim,
                        "use_flash": layer.attn.use_flash,
                    },
                    "children": [
                        {"name": "Q Projection", "type": "Linear",
                         "params": {"in": config["d_model"], "out": layer_n_heads * layer_head_dim}},
                        {"name": "K Projection", "type": "Linear",
                         "params": {"in": config["d_model"], "out": layer_n_kv_heads * layer_head_dim}},
                        {"name": "V Projection", "type": "Linear",
                         "params": {"in": config["d_model"], "out": layer_n_kv_heads * layer_head_dim}},
                        {"name": "RoPE", "type": "RotaryPositionalEmbedding", "params": {}},
                        {"name": "Scaled Dot-Product" + (" (Flash)" if layer.attn.use_flash else ""), "type": "attention_core", "params": {}},
                        {"name": "O Projection", "type": "Linear",
                         "params": {"in": layer_n_heads * layer_head_dim, "out": config["d_model"]}},
                    ],
                },
                {"name": "Residual Add", "type": "residual", "params": {}},
                {
                    "name": f"{norm_name} (pre-FFN)",
                    "type": norm_name,
                    "params": {"d_model": config["d_model"]},
                },
            ],
            "param_count": sum(p.numel() for p in layer.parameters()),
        }

        if is_moe:
            ffn_child = {
                "name": f"MoE ({layer.ffn.n_experts} experts, top-{layer.ffn.top_k})",
                "type": "MoELayer",
                "params": {
                    "n_experts": layer.ffn.n_experts,
                    "top_k": layer.ffn.top_k,
                },
                "children": [
                    {"name": "Router", "type": "Linear", "params": {"in": config["d_model"], "out": layer.ffn.n_experts}},
                ] + [
                    {"name": f"Expert {j} (SwiGLU)", "type": "SwiGLUFFN", "params": {}}
                    for j in range(layer.ffn.n_experts)
                ],
            }
        else:
            ffn_labels = {
                "swiglu": ("SwiGLU FFN", "SwiGLUFFN", "SiLU Gate"),
                "geglu": ("GeGLU FFN", "GeGLUFFN", "GELU Gate"),
                "reglu": ("ReGLU FFN", "ReGLUFFN", "ReLU Gate"),
                "standard": ("Standard FFN", "StandardFFN", "GELU"),
            }
            label, ftype, act_name = ffn_labels.get(ffn_type, ffn_labels["swiglu"])

            if ffn_type == "standard":
                ffn_child = {
                    "name": label,
                    "type": ftype,
                    "params": {},
                    "children": [
                        {"name": "FC1 (Linear)", "type": "Linear", "params": {}},
                        {"name": act_name, "type": "activation", "params": {}},
                        {"name": "FC2 (Linear)", "type": "Linear", "params": {}},
                    ],
                }
            else:
                ffn_child = {
                    "name": label,
                    "type": ftype,
                    "params": {},
                    "children": [
                        {"name": "Gate Proj", "type": "Linear", "params": {}},
                        {"name": "Up Proj", "type": "Linear", "params": {}},
                        {"name": act_name, "type": "activation", "params": {}},
                        {"name": "Down Proj", "type": "Linear", "params": {}},
                    ],
                }

        block["children"].append(ffn_child)
        block["children"].append({"name": "Residual Add", "type": "residual", "params": {}})
        arch.append(block)

    arch.append({
        "name": "RMSNorm (final)",
        "type": "RMSNorm",
        "params": {"d_model": config["d_model"]},
    })
    arch.append({
        "name": "LM Head",
        "type": "Linear",
        "params": {"in": config["d_model"], "out": config["vocab_size"]},
        "note": "Weight-tied with embedding" if config.get("tie_weights", True) else "",
    })

    return arch


@app.post("/api/llm/modify")
async def modify_llm(body: dict[str, Any]):
    """Modify LLM architecture: add/remove/update layers, change components."""
    if engine.model is None or engine.model_source != "llm":
        return {"status": "error", "message": "No LLM model loaded. Build one first."}

    from state_graph.layers.llm import (
        LLMDecoderBlock, RMSNorm, LLMAttention, SwiGLUFFN, GeGLUFFN,
        ReGLUFFN, StandardFFN, MoELayer, FFN_TYPES,
    )

    model = engine.model
    config = getattr(engine, '_llm_config', {})
    action = body.get("action")

    if action == "add_layer":
        position = body.get("position", len(model.layers))
        layer_cfg = body.get("config", {})
        d_model = config.get("d_model", model.d_model)
        block = LLMDecoderBlock(
            d_model=d_model,
            n_heads=layer_cfg.get("n_heads", config.get("n_heads", 8)),
            n_kv_heads=layer_cfg.get("n_kv_heads", config.get("n_kv_heads")),
            ffn_hidden_dim=layer_cfg.get("ffn_hidden_dim", config.get("ffn_hidden_dim")),
            dropout=layer_cfg.get("dropout", config.get("dropout", 0.0)),
            use_flash=layer_cfg.get("use_flash", config.get("use_flash", True)),
            use_moe=layer_cfg.get("use_moe", False),
            n_experts=layer_cfg.get("n_experts", config.get("n_experts", 8)),
            moe_top_k=layer_cfg.get("moe_top_k", config.get("moe_top_k", 2)),
            max_len=config.get("max_len", 2048),
            norm_type=layer_cfg.get("norm_type", config.get("norm_type", "rmsnorm")),
            ffn_type=layer_cfg.get("ffn_type", config.get("ffn_type", "swiglu")),
            rope_base=layer_cfg.get("rope_base", config.get("rope_base", 10000.0)),
        ).to(engine.device)
        layers_list = list(model.layers)
        position = max(0, min(position, len(layers_list)))
        layers_list.insert(position, block)
        model.layers = nn.ModuleList(layers_list)
        model.n_layers = len(model.layers)
        config["n_layers"] = model.n_layers

    elif action == "remove_layer":
        idx = body.get("index", -1)
        if idx < 0 or idx >= len(model.layers):
            return {"status": "error", "message": f"Invalid layer index {idx}. Model has {len(model.layers)} layers."}
        if len(model.layers) <= 1:
            return {"status": "error", "message": "Cannot remove the last layer."}
        layers_list = list(model.layers)
        layers_list.pop(idx)
        model.layers = nn.ModuleList(layers_list)
        model.n_layers = len(model.layers)
        config["n_layers"] = model.n_layers

    elif action == "update_layer":
        idx = body.get("index", 0)
        if idx < 0 or idx >= len(model.layers):
            return {"status": "error", "message": f"Invalid layer index {idx}."}
        layer_cfg = body.get("config", {})
        d_model = config.get("d_model", model.d_model)
        new_block = LLMDecoderBlock(
            d_model=d_model,
            n_heads=layer_cfg.get("n_heads", config.get("n_heads", 8)),
            n_kv_heads=layer_cfg.get("n_kv_heads", config.get("n_kv_heads")),
            ffn_hidden_dim=layer_cfg.get("ffn_hidden_dim", config.get("ffn_hidden_dim")),
            dropout=layer_cfg.get("dropout", config.get("dropout", 0.0)),
            use_flash=layer_cfg.get("use_flash", config.get("use_flash", True)),
            use_moe=layer_cfg.get("use_moe", False),
            n_experts=layer_cfg.get("n_experts", config.get("n_experts", 8)),
            moe_top_k=layer_cfg.get("moe_top_k", config.get("moe_top_k", 2)),
            max_len=config.get("max_len", 2048),
            norm_type=layer_cfg.get("norm_type", config.get("norm_type", "rmsnorm")),
            ffn_type=layer_cfg.get("ffn_type", config.get("ffn_type", "swiglu")),
            rope_base=layer_cfg.get("rope_base", config.get("rope_base", 10000.0)),
        ).to(engine.device)
        model.layers[idx] = new_block

    elif action == "freeze_layer":
        idx = body.get("index")
        if idx is not None:
            if idx < 0 or idx >= len(model.layers):
                return {"status": "error", "message": f"Invalid layer index {idx}."}
            for p in model.layers[idx].parameters():
                p.requires_grad = False
        else:
            component = body.get("component", "")
            target = {"embedding": model.tok_emb, "lm_head": model.lm_head, "norm": model.norm}.get(component)
            if target:
                for p in target.parameters():
                    p.requires_grad = False
            else:
                return {"status": "error", "message": f"Unknown component: {component}"}

    elif action == "unfreeze_layer":
        idx = body.get("index")
        if idx is not None:
            if idx < 0 or idx >= len(model.layers):
                return {"status": "error", "message": f"Invalid layer index {idx}."}
            for p in model.layers[idx].parameters():
                p.requires_grad = True
        else:
            component = body.get("component", "")
            target = {"embedding": model.tok_emb, "lm_head": model.lm_head, "norm": model.norm}.get(component)
            if target:
                for p in target.parameters():
                    p.requires_grad = True
            else:
                return {"status": "error", "message": f"Unknown component: {component}"}

    elif action == "reorder_layer":
        from_idx = body.get("from_index", 0)
        to_idx = body.get("to_index", 0)
        if from_idx < 0 or from_idx >= len(model.layers) or to_idx < 0 or to_idx >= len(model.layers):
            return {"status": "error", "message": "Invalid indices for reorder."}
        layers_list = list(model.layers)
        layer = layers_list.pop(from_idx)
        layers_list.insert(to_idx, layer)
        model.layers = nn.ModuleList(layers_list)

    elif action == "change_vocab":
        new_vocab = body.get("vocab_size")
        if not new_vocab or new_vocab < 1:
            return {"status": "error", "message": "Invalid vocab_size."}
        old_emb = model.tok_emb
        model.tok_emb = nn.Embedding(new_vocab, model.d_model).to(engine.device)
        copy_size = min(old_emb.num_embeddings, new_vocab)
        model.tok_emb.weight.data[:copy_size] = old_emb.weight.data[:copy_size]
        model.lm_head = nn.Linear(model.d_model, new_vocab, bias=False).to(engine.device)
        if config.get("tie_weights", True):
            model.lm_head.weight = model.tok_emb.weight
        model.vocab_size = new_vocab
        config["vocab_size"] = new_vocab

    elif action == "change_norm":
        norm_type = body.get("norm_type", "rmsnorm")
        NormClass = nn.LayerNorm if norm_type == "layernorm" else RMSNorm
        model.norm = NormClass(model.d_model).to(engine.device)
        config["norm_type"] = norm_type

    else:
        return {"status": "error", "message": f"Unknown action: {action}. Use: add_layer, remove_layer, update_layer, freeze_layer, unfreeze_layer, reorder_layer, change_vocab, change_norm"}

    param_info = model.count_parameters()
    arch = _get_llm_architecture(model, config)

    await broadcast("model_modified", {
        "action": action,
        "config": config,
        "total_params": param_info["total"],
        "total_params_M": param_info["total_M"],
        "architecture": arch,
    })

    return {
        "status": "modified",
        "action": action,
        "config": config,
        "params": param_info,
        "architecture": arch,
    }


@app.get("/api/llm/layer/{idx}")
async def get_llm_layer(idx: int):
    """Get details of a specific decoder block."""
    if engine.model is None or engine.model_source != "llm":
        return {"status": "error", "message": "No LLM model loaded."}
    if idx < 0 or idx >= len(engine.model.layers):
        return {"status": "error", "message": f"Invalid layer index {idx}."}

    layer = engine.model.layers[idx]
    frozen = all(not p.requires_grad for p in layer.parameters())
    has_moe = hasattr(layer.ffn, 'n_experts')

    info = {
        "index": idx,
        "n_heads": layer.attn.n_heads,
        "n_kv_heads": layer.attn.n_kv_heads,
        "head_dim": layer.attn.head_dim,
        "use_flash": layer.attn.use_flash,
        "norm_type": layer.norm_type if hasattr(layer, 'norm_type') else "rmsnorm",
        "ffn_type": layer.ffn_type if hasattr(layer, 'ffn_type') else "swiglu",
        "has_moe": has_moe,
        "frozen": frozen,
        "param_count": sum(p.numel() for p in layer.parameters()),
        "trainable_params": sum(p.numel() for p in layer.parameters() if p.requires_grad),
    }
    if has_moe:
        info["n_experts"] = layer.ffn.n_experts
        info["moe_top_k"] = layer.ffn.top_k

    return {"status": "ok", "layer": info}


@app.get("/api/llm/presets")
async def llm_presets():
    """Return LLM architecture presets."""
    return {
        "presets": {
            "tiny": {
                "name": "Tiny (1M)", "vocab_size": 32000, "d_model": 128,
                "n_layers": 4, "n_heads": 4, "max_len": 512, "description": "Tiny model for learning/testing",
            },
            "small": {
                "name": "Small (25M)", "vocab_size": 32000, "d_model": 512,
                "n_layers": 8, "n_heads": 8, "max_len": 1024, "description": "Small GPT-2 scale",
            },
            "medium": {
                "name": "Medium (125M)", "vocab_size": 32000, "d_model": 768,
                "n_layers": 12, "n_heads": 12, "max_len": 2048, "description": "GPT-2 small scale",
            },
            "large": {
                "name": "Large (350M)", "vocab_size": 32000, "d_model": 1024,
                "n_layers": 24, "n_heads": 16, "max_len": 2048, "description": "GPT-2 medium scale",
            },
            "mixtral_tiny": {
                "name": "MoE Tiny (8M)", "vocab_size": 32000, "d_model": 256,
                "n_layers": 4, "n_heads": 4, "max_len": 1024,
                "use_moe": True, "n_experts": 4, "moe_top_k": 2,
                "description": "Tiny Mixtral-style MoE model",
            },
            "gqa_small": {
                "name": "GQA Small (20M)", "vocab_size": 32000, "d_model": 512,
                "n_layers": 6, "n_heads": 8, "n_kv_heads": 2, "max_len": 1024,
                "description": "Grouped-Query Attention (Llama 2 style)",
            },
            "gpt2_style": {
                "name": "GPT-2 Style (25M)", "vocab_size": 50257, "d_model": 512,
                "n_layers": 8, "n_heads": 8, "max_len": 1024,
                "norm_type": "layernorm", "ffn_type": "standard",
                "description": "GPT-2 architecture with LayerNorm + Standard FFN",
            },
            "gemma_tiny": {
                "name": "Gemma Tiny (5M)", "vocab_size": 32000, "d_model": 256,
                "n_layers": 6, "n_heads": 4, "max_len": 1024,
                "ffn_type": "geglu", "rope_base": 10000,
                "description": "Gemma-style with GeGLU FFN",
            },
        }
    }


@app.post("/api/llm/compose")
async def compose_llm(body: dict[str, Any]):
    """Build a fully composable LLM from block designs.

    Body: {
        vocab_size, d_model, n_layers, n_heads, max_len, dropout, tie_weights,
        norm_type, n_kv_heads, use_flash,
        default_block: [...steps...],  # default block design for all layers
        block_designs: [[...steps...], ...],  # per-layer overrides
    }
    """
    from state_graph.layers.llm import ComposableLLM, BLOCK_DESIGNS

    default_block_name = body.get("default_block_name")
    default_block = body.get("default_block")

    # If a preset name is given, use it
    if default_block_name and default_block_name in BLOCK_DESIGNS:
        default_block = BLOCK_DESIGNS[default_block_name]

    config = {
        "vocab_size": body.get("vocab_size", 32000),
        "d_model": body.get("d_model", 512),
        "n_layers": body.get("n_layers", 6),
        "n_heads": body.get("n_heads", 8),
        "n_kv_heads": body.get("n_kv_heads"),
        "max_len": body.get("max_len", 2048),
        "dropout": body.get("dropout", 0.0),
        "use_flash": body.get("use_flash", True),
        "tie_weights": body.get("tie_weights", True),
        "norm_type": body.get("norm_type", "rmsnorm"),
    }

    try:
        model = ComposableLLM(
            **config,
            default_block=default_block,
            block_designs=body.get("block_designs"),
            pos_encoding=body.get("pos_encoding", "rope"),
            custom_embedding_code=body.get("custom_embedding_code"),
            custom_loss=body.get("custom_loss"),
            extra_heads=body.get("extra_heads"),
        )
    except Exception as e:
        return {"status": "error", "message": f"Failed to build: {e}"}

    engine.model = model.to(engine.device)
    engine.model_source = "llm"
    engine._llm_config = config
    engine._llm_config["composable"] = True
    engine._llm_config["default_block"] = default_block
    engine._llm_config["block_designs"] = body.get("block_designs")

    param_info = model.count_parameters()

    # Build architecture tree
    arch = _get_composable_architecture(model, config)

    await broadcast("model_built", {
        "status": "built",
        "model_type": "composable_llm",
        "config": config,
        "total_params": param_info["total"],
        "total_params_M": param_info["total_M"],
        "device": str(engine.device),
        "architecture": arch,
    })

    return {
        "status": "built",
        "config": config,
        "params": param_info,
        "device": str(engine.device),
        "architecture": arch,
    }


def _get_composable_architecture(model, config: dict) -> list[dict]:
    """Extract architecture tree from a ComposableLLM."""
    arch = []
    arch.append({
        "name": "Token Embedding",
        "type": "Embedding",
        "params": {"vocab_size": config["vocab_size"], "d_model": config["d_model"]},
        "param_count": model.tok_emb.weight.numel(),
    })

    for i, layer in enumerate(model.layers):
        frozen = all(not p.requires_grad for p in layer.parameters())
        children = []
        if hasattr(layer, 'get_step_info'):
            for step_info in layer.get_step_info():
                stype = step_info["type"]
                res_from = step_info.get("residual_from")
                name = stype.replace("_", " ").title()
                if stype == "residual":
                    name = f"Residual Add (from {'input' if res_from == -1 else f'step {res_from}'})"
                child = {
                    "name": name,
                    "type": stype,
                    "param_count": step_info["params"],
                }
                if res_from is not None and stype != "residual":
                    child["residual_from"] = res_from
                children.append(child)

        block = {
            "name": f"Composable Block {i}" + (" [frozen]" if frozen else ""),
            "type": "ComposableBlock",
            "children": children,
            "param_count": sum(p.numel() for p in layer.parameters()),
        }
        arch.append(block)

    norm_name = "LayerNorm" if config.get("norm_type") == "layernorm" else "RMSNorm"
    arch.append({"name": f"{norm_name} (final)", "type": norm_name, "params": {"d_model": config["d_model"]}})
    arch.append({
        "name": "LM Head", "type": "Linear",
        "params": {"in": config["d_model"], "out": config["vocab_size"]},
        "note": "Weight-tied with embedding" if config.get("tie_weights", True) else "",
    })
    return arch


@app.post("/api/llm/compose/block/{idx}")
async def update_composable_block(idx: int, body: dict[str, Any]):
    """Replace a single block's design in a composable LLM."""
    if engine.model is None or engine.model_source != "llm":
        return {"status": "error", "message": "No LLM model loaded."}
    config = getattr(engine, '_llm_config', {})
    if not config.get("composable"):
        return {"status": "error", "message": "Model is not a composable LLM. Use /api/llm/compose first."}
    if idx < 0 or idx >= len(engine.model.layers):
        return {"status": "error", "message": f"Invalid block index {idx}."}

    from state_graph.layers.llm import ComposableBlock
    steps = body.get("steps", [])
    if not steps:
        return {"status": "error", "message": "No steps provided."}

    try:
        new_block = ComposableBlock(
            d_model=config["d_model"],
            steps=steps,
            n_heads=config.get("n_heads", 8),
            n_kv_heads=config.get("n_kv_heads"),
            max_len=config.get("max_len", 2048),
            dropout=config.get("dropout", 0.0),
            use_flash=config.get("use_flash", True),
        ).to(engine.device)
    except Exception as e:
        return {"status": "error", "message": f"Failed to create block: {e}"}

    engine.model.layers[idx] = new_block

    param_info = engine.model.count_parameters()
    arch = _get_composable_architecture(engine.model, config)

    await broadcast("model_modified", {
        "action": "update_block",
        "block_index": idx,
        "total_params": param_info["total"],
        "total_params_M": param_info["total_M"],
        "architecture": arch,
    })

    return {"status": "modified", "params": param_info, "architecture": arch}


@app.get("/api/llm/components")
async def get_component_catalog():
    """Return available component types and block design presets."""
    from state_graph.layers.llm import COMPONENT_CATALOG, BLOCK_DESIGNS
    return {
        "components": COMPONENT_CATALOG,
        "block_designs": {k: v for k, v in BLOCK_DESIGNS.items()},
    }


@app.post("/api/llm/component/validate")
async def validate_custom_component(body: dict[str, Any]):
    """Validate custom code or formula without building a full model."""
    from state_graph.layers.llm import CustomComponent, CustomFFN
    comp_type = body.get("type", "custom_code")
    d_model = body.get("d_model", 64)

    if comp_type == "custom_code":
        code = body.get("code", "")
        try:
            module = CustomComponent.create_from_code(code, d_model)
            x = torch.randn(1, 4, d_model)
            out = module(x)
            return {
                "status": "valid",
                "output_shape": list(out.shape),
                "params": sum(p.numel() for p in module.parameters()),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif comp_type == "custom_formula":
        formula = body.get("formula", "x")
        try:
            ffn = CustomFFN(d_model, formula=formula)
            x = torch.randn(1, 4, d_model)
            out = ffn(x)
            return {
                "status": "valid",
                "output_shape": list(out.shape),
                "params": sum(p.numel() for p in ffn.parameters()),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": f"Unknown type: {comp_type}"}


@app.post("/api/llm/novel/validate")
async def validate_novel_architecture(body: dict[str, Any]):
    """Validate a novel block design by building, running forward pass, and benchmarking.

    Body: {
        block_design: [...steps...],
        d_model: 128,
        n_heads: 4,
        vocab_size: 1000,
        n_layers: 2,
        seq_len: 32,
        benchmark: true,  // run speed comparison against standard Llama block
    }
    """
    from state_graph.layers.llm import ComposableBlock, ComposableLLM, BLOCK_DESIGNS
    import time

    steps = body.get("block_design", [])
    d_model = body.get("d_model", 128)
    n_heads = body.get("n_heads", 4)
    n_kv_heads = body.get("n_kv_heads")
    vocab_size = body.get("vocab_size", 1000)
    n_layers = body.get("n_layers", 2)
    seq_len = body.get("seq_len", 32)
    do_benchmark = body.get("benchmark", False)

    if not steps:
        return {"status": "error", "message": "No block_design steps provided"}

    # 1. Validate: build single block
    try:
        block = ComposableBlock(
            d_model=d_model, steps=steps, n_heads=n_heads,
            n_kv_heads=n_kv_heads, max_len=512,
        )
    except Exception as e:
        return {"status": "error", "stage": "block_build", "message": str(e)}

    # 2. Test forward pass
    try:
        x = torch.randn(1, seq_len, d_model)
        out = block(x)
        if out.shape != x.shape:
            return {"status": "warning", "message": f"Output shape {list(out.shape)} != input shape {list(x.shape)}. Block may not stack correctly.",
                    "output_shape": list(out.shape), "block_params": sum(p.numel() for p in block.parameters())}
    except Exception as e:
        return {"status": "error", "stage": "forward_pass", "message": str(e)}

    block_params = sum(p.numel() for p in block.parameters())

    # 3. Build full model
    try:
        model = ComposableLLM(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, n_kv_heads=n_kv_heads, max_len=512,
            default_block=steps,
        )
        ids = torch.randint(0, vocab_size, (1, seq_len))
        model_out = model(ids, labels=ids)
        total_params = sum(p.numel() for p in model.parameters())
    except Exception as e:
        return {"status": "error", "stage": "model_build", "message": str(e)}

    result = {
        "status": "valid",
        "block_params": block_params,
        "total_params": total_params,
        "total_params_M": f"{total_params / 1e6:.2f}M",
        "output_shape": list(out.shape),
        "loss": model_out["loss"].item() if model_out.get("loss") is not None else None,
        "n_steps": len(steps),
        "step_types": [s.get("type", "?") for s in steps],
    }

    # 4. Benchmark against baselines
    if do_benchmark:
        benchmarks = {}
        for baseline_name in ["llama", "mamba", "minimal"]:
            baseline_steps = BLOCK_DESIGNS.get(baseline_name, [])
            if not baseline_steps:
                continue
            try:
                baseline_model = ComposableLLM(
                    vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                    n_heads=n_heads, max_len=512, default_block=baseline_steps,
                )
                baseline_params = sum(p.numel() for p in baseline_model.parameters())

                # Speed test
                ids_bench = torch.randint(0, vocab_size, (2, seq_len))
                start = time.time()
                for _ in range(5):
                    baseline_model(ids_bench)
                baseline_time = (time.time() - start) / 5

                start = time.time()
                for _ in range(5):
                    model(ids_bench)
                novel_time = (time.time() - start) / 5

                benchmarks[baseline_name] = {
                    "baseline_params": baseline_params,
                    "baseline_ms": round(baseline_time * 1000, 2),
                    "novel_ms": round(novel_time * 1000, 2),
                    "speedup": round(baseline_time / max(novel_time, 1e-9), 2),
                    "param_ratio": round(total_params / max(baseline_params, 1), 2),
                }
            except Exception:
                continue

        result["benchmarks"] = benchmarks

    return result


@app.post("/api/llm/novel/experiment")
async def novel_architecture_experiment(body: dict[str, Any]):
    """Run a quick experiment: build model, train briefly, and report metrics.

    Body: {
        block_design: [...steps...],
        d_model: 128, n_heads: 4, vocab_size: 1000, n_layers: 2,
        text: "training text...",
        train_steps: 50,
        learning_rate: 1e-3,
    }
    """
    from state_graph.layers.llm import ComposableLLM

    steps = body.get("block_design", [])
    if not steps:
        return {"status": "error", "message": "No block_design provided"}

    d_model = body.get("d_model", 128)
    n_heads = body.get("n_heads", 4)
    vocab_size = body.get("vocab_size", 1000)
    n_layers = body.get("n_layers", 2)
    max_len = body.get("max_len", 64)
    text = body.get("text", "")
    train_steps = body.get("train_steps", 50)
    lr = body.get("learning_rate", 1e-3)

    if len(text) < 100:
        return {"status": "error", "message": "Need at least 100 characters of text"}

    try:
        model = ComposableLLM(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, max_len=max_len, default_block=steps,
        ).to(engine.device)
    except Exception as e:
        return {"status": "error", "message": f"Build failed: {e}"}

    # Quick char-level tokenization
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    encoded = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    if len(chars) > vocab_size:
        return {"status": "error", "message": f"Text has {len(chars)} unique chars but vocab_size={vocab_size}"}

    # Create sequences
    n_seqs = (len(encoded) - 1) // max_len
    if n_seqs < 2:
        return {"status": "error", "message": "Not enough text for training sequences"}
    trim = n_seqs * max_len + 1
    data = encoded[:trim]
    input_ids = data[:-1].view(n_seqs, max_len).to(engine.device)
    labels = data[1:].view(n_seqs, max_len).to(engine.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_params = sum(p.numel() for p in model.parameters())

    # Train
    model.train()
    losses = []
    import time
    start = time.time()
    for step in range(train_steps):
        idx = step % n_seqs
        out = model(input_ids[idx:idx+1], labels=labels[idx:idx+1])
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    elapsed = time.time() - start

    return {
        "status": "ok",
        "total_params": total_params,
        "total_params_M": f"{total_params / 1e6:.2f}M",
        "train_steps": train_steps,
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "loss_reduction": round(losses[0] - losses[-1], 4),
        "loss_history": [round(l, 4) for l in losses[::max(1, len(losses)//20)]],
        "time_seconds": round(elapsed, 2),
        "steps_per_second": round(train_steps / elapsed, 1),
        "step_types": [s.get("type", "?") for s in steps],
    }


@app.get("/api/llm/novel/templates")
async def novel_architecture_templates():
    """Return starter templates for inventing novel architectures.

    Each template is a research direction with a starting block design
    and suggestions for what to experiment with.
    """
    return {"templates": {
        "sparse_attention_ssm": {
            "name": "Sparse Attention + SSM Fusion",
            "description": "Combine sparse/windowed attention for precision with Mamba SSM for long-range. Research question: what's the optimal ratio?",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "sliding_window_attention", "config": {"window_size": 256}},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "mamba", "config": {"d_state": 16, "expand": 2}},
                {"type": "residual", "residual_from": 3},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 5},
            ],
            "experiments": [
                "Try window_size: 64, 128, 256, 512 — which gives best quality/speed?",
                "Swap order: SSM first, then attention — does it matter?",
                "Replace SwiGLU with MoE — does expert specialization help?",
                "Use linear_attention instead of sliding_window — faster but how much quality loss?",
            ],
        },
        "parallel_everything": {
            "name": "Fully Parallel Block",
            "description": "Run attention, SSM, and convolution in parallel then merge. Maximizes information flow per layer.",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "parallel", "config": {
                    "branch_a": {"type": "attention", "config": {}},
                    "branch_b": {"type": "mamba", "config": {"d_state": 16}},
                    "merge": "gate",
                }},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 3},
            ],
            "experiments": [
                "Try merge modes: add vs gate vs concat — which preserves most info?",
                "Add a third parallel branch with conv1d",
                "Use MoE as the FFN to add expert routing",
                "Make different layers use different parallel combos",
            ],
        },
        "recursive_depth": {
            "name": "Recursive / Shared-Weight Layers",
            "description": "Same block repeated with shared weights — like Universal Transformer. More compute per parameter.",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "attention", "config": {}},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 2},
            ],
            "use_tip": "Build with n_layers=1 then in custom code, loop through the same block N times. Or use block_designs to repeat the same design.",
            "experiments": [
                "Compare: 12 unique layers vs 4 layers × 3 passes each (same params, more compute)",
                "Add layer-specific scaling: multiply by learned alpha per pass",
                "Combine with early exit — exit when representation stabilizes",
            ],
        },
        "gated_expert_ssm": {
            "name": "Gated Expert SSM",
            "description": "Route tokens to specialized SSM experts — each expert handles different sequence patterns.",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "moe", "config": {"n_experts": 4, "top_k": 1}},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "mamba", "config": {"d_state": 32, "expand": 1}},
                {"type": "residual", "residual_from": 3},
            ],
            "experiments": [
                "MoE first vs Mamba first — does order matter for specialization?",
                "Increase d_state to 32, 64 — does larger state help experts?",
                "Add attention every 4th layer for global information exchange",
                "Try top_k=2 vs top_k=1 — more experts per token vs specialization",
            ],
        },
        "multi_scale_hybrid": {
            "name": "Multi-Scale Processing",
            "description": "Different components at different scales: local conv, medium attention, global SSM.",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "conv1d", "config": {"kernel_size": 3, "groups": 1}},
                {"type": "activation", "config": {"name": "silu"}},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "sliding_window_attention", "config": {"window_size": 128}},
                {"type": "residual", "residual_from": 4},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "mamba", "config": {"d_state": 16}},
                {"type": "residual", "residual_from": 6},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 9},
            ],
            "experiments": [
                "Which scale matters most? Remove one component at a time",
                "Try conv kernel sizes: 3, 7, 15 — larger = more local context",
                "Replace Mamba with Hyena for a different long-range mechanism",
                "Use this for only deep layers, use simple attention for early layers",
            ],
        },
        "custom_code_layer": {
            "name": "Write Your Own Layer",
            "description": "Define a completely new mechanism in Python. Full access to PyTorch + all StateGraph primitives.",
            "block_design": [
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "custom_code", "config": {
                    "code": "class CustomModule(nn.Module):\n    def __init__(self, d_model, **kwargs):\n        super().__init__()\n        # YOUR NOVEL MECHANISM HERE\n        # Available: torch, nn, F, math, RMSNorm, LLMAttention,\n        #   SwiGLUFFN, MoELayer, SelectiveScan, PerceiverResampler, etc.\n        self.gate = nn.Linear(d_model, d_model)\n        self.transform = nn.Linear(d_model, d_model)\n        self.mix = nn.Linear(d_model * 2, d_model)\n\n    def forward(self, x):\n        # Example: gated cross-dimensional mixing\n        g = torch.sigmoid(self.gate(x))\n        t = F.silu(self.transform(x))\n        mixed = self.mix(torch.cat([g * x, (1-g) * t], dim=-1))\n        return mixed\n",
                }},
                {"type": "residual", "residual_from": -1},
                {"type": "norm", "config": {"norm_type": "rmsnorm"}},
                {"type": "ffn", "config": {"ffn_type": "swiglu"}},
                {"type": "residual", "residual_from": 3},
            ],
            "experiments": [
                "Replace the example with your own attention variant",
                "Try combining multiple existing primitives in a new way",
                "Implement a novel gating mechanism",
                "Create a differentiable memory module",
            ],
        },
    }}


@app.post("/api/llm/novel/model-from-code")
async def build_model_from_code(body: dict[str, Any]):
    """Build an entirely custom model from Python code and load into the training engine.

    The code must define a class with:
    - __init__(self, vocab_size, d_model, **kwargs)
    - forward(self, input_ids, labels=None) -> dict with "logits" and "loss"
    - Optionally: generate(), count_parameters()

    Available in code: torch, nn, F, math, + all StateGraph primitives (RMSNorm,
    LLMAttention, SwiGLUFFN, MoELayer, SelectiveScan, PerceiverResampler, etc.)

    Body: {
        code: "class MyModel(nn.Module): ...",
        vocab_size: 32000,
        d_model: 512,
        kwargs: {},  // extra init args
    }
    """
    code = body.get("code", "")
    vocab_size = body.get("vocab_size", 32000)
    d_model = body.get("d_model", 512)
    extra_kwargs = body.get("kwargs", {})

    if not code.strip():
        return {"status": "error", "message": "No code provided"}

    import torch.nn as tnn

    # Build safe execution environment with all primitives
    safe_globals = {
        "torch": torch, "nn": tnn, "F": torch.nn.functional,
        "math": __import__("math"), "Optional": __import__("typing").Optional,
    }

    # Inject all custom layer primitives
    try:
        import state_graph.layers.custom as _custom
        for attr in dir(_custom):
            obj = getattr(_custom, attr)
            if isinstance(obj, type) and issubclass(obj, tnn.Module):
                safe_globals[attr] = obj
            elif attr == "NoiseScheduler":
                safe_globals[attr] = obj
    except ImportError:
        pass

    # Inject LLM components
    try:
        from state_graph.layers.llm import (
            RMSNorm, LLMAttention, SwiGLUFFN, GeGLUFFN, ReGLUFFN, StandardFFN,
            MoELayer, MoERouter, LLMDecoderBlock, RotaryPositionalEmbedding,
            apply_rotary_pos_emb, SlidingWindowAttention, LinearAttention,
            ALiBiAttention, ComposableBlock, ParallelBranch,
            EncoderBlock, DecoderBlockWithCrossAttn,
            PatchEmbedding, AudioEmbedding, ModalityProjector,
        )
        for name, obj in locals().items():
            if isinstance(obj, type):
                safe_globals[name] = obj
            elif callable(obj):
                safe_globals[name] = obj
    except ImportError:
        pass

    # Execute the code
    local_ns: dict = {}
    try:
        exec(code, safe_globals, local_ns)  # noqa: S102
    except Exception as e:
        return {"status": "error", "stage": "parse", "message": f"Code execution error: {e}"}

    # Find the model class
    model_cls = None
    for obj in local_ns.values():
        if isinstance(obj, type) and issubclass(obj, tnn.Module) and obj is not tnn.Module:
            model_cls = obj
            break

    if model_cls is None:
        return {"status": "error", "stage": "class", "message": "Code must define an nn.Module subclass"}

    # Instantiate
    try:
        model = model_cls(vocab_size=vocab_size, d_model=d_model, **extra_kwargs)
    except TypeError as e:
        # Try without keyword args
        try:
            model = model_cls(vocab_size, d_model, **extra_kwargs)
        except Exception:
            return {"status": "error", "stage": "init", "message": f"Failed to instantiate: {e}"}

    # Validate forward pass
    try:
        test_ids = torch.randint(0, min(vocab_size, 100), (1, 16))
        out = model(test_ids)
        if not isinstance(out, dict) or "logits" not in out:
            return {"status": "error", "stage": "forward",
                    "message": "forward() must return a dict with 'logits' key. Got: " + str(type(out))}
    except Exception as e:
        return {"status": "error", "stage": "forward", "message": f"Forward pass failed: {e}"}

    # Validate with labels
    try:
        out_with_labels = model(test_ids, labels=test_ids)
        has_loss = out_with_labels.get("loss") is not None
    except Exception:
        has_loss = False

    # Load into engine
    model = model.to(engine.device)
    engine.model = model
    engine.model_source = "llm"
    engine._llm_config = {
        "model_class": "custom_code",
        "vocab_size": vocab_size,
        "d_model": d_model,
        "code": code,
    }

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check for generate method
    has_generate = hasattr(model, "generate") and callable(model.generate)

    return {
        "status": "ok",
        "model_class": model_cls.__name__,
        "parameters": {
            "total": total, "trainable": trainable,
            "total_M": f"{total / 1e6:.1f}M",
        },
        "has_loss": has_loss,
        "has_generate": has_generate,
        "output_shape": list(out["logits"].shape),
        "message": f"Custom model '{model_cls.__name__}' loaded: {total/1e6:.1f}M params. "
                   + ("Loss function: OK. " if has_loss else "WARNING: No loss from forward(labels=...). ")
                   + ("Generation: OK." if has_generate else "No generate() method."),
    }


@app.post("/api/llm/novel/custom-loss")
async def set_custom_loss(body: dict[str, Any]):
    """Set a custom loss function for training.

    Supports two modes:
    1. Formula: "F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1)) + 0.1 * F.mse_loss(logits, ...)"
    2. Code: Full Python class definition

    Formula vars: logits, labels, vocab_size, torch, F, math, model
    Code must define: class CustomLoss(nn.Module) with forward(self, logits, labels, **kwargs) -> tensor

    Body: {
        mode: "formula" | "code",
        formula: "...",   // for formula mode
        code: "...",      // for code mode
    }
    """
    mode = body.get("mode", "formula")

    if mode == "formula":
        formula = body.get("formula", "")
        if not formula:
            return {"status": "error", "message": "No formula provided"}

        # Validate
        try:
            import torch.nn as tnn
            logits = torch.randn(1, 8, 100)
            labels = torch.randint(0, 100, (1, 8))
            safe = {"torch": torch, "F": torch.nn.functional, "math": __import__("math"),
                    "logits": logits, "labels": labels, "vocab_size": 100}
            result = eval(formula, safe)  # noqa: S307
            if not isinstance(result, torch.Tensor):
                return {"status": "error", "message": f"Formula must return a tensor, got {type(result)}"}
        except Exception as e:
            return {"status": "error", "message": f"Formula validation failed: {e}"}

        engine._custom_loss = {"mode": "formula", "formula": formula}
        return {"status": "ok", "mode": "formula", "formula": formula,
                "message": "Custom loss formula set. Will be used in next training run."}

    elif mode == "code":
        code = body.get("code", "")
        if not code:
            return {"status": "error", "message": "No code provided"}

        try:
            import torch.nn as tnn
            safe_globals = {"torch": torch, "nn": tnn, "F": torch.nn.functional,
                           "math": __import__("math")}
            local_ns: dict = {}
            exec(code, safe_globals, local_ns)  # noqa: S102

            loss_cls = None
            for obj in local_ns.values():
                if isinstance(obj, type) and issubclass(obj, tnn.Module) and obj is not tnn.Module:
                    loss_cls = obj
                    break

            if loss_cls is None:
                return {"status": "error", "message": "Code must define an nn.Module subclass"}

            # Validate
            loss_fn = loss_cls()
            logits = torch.randn(1, 8, 100)
            labels = torch.randint(0, 100, (1, 8))
            result = loss_fn(logits, labels)
            if not isinstance(result, torch.Tensor):
                return {"status": "error", "message": f"Loss must return a tensor, got {type(result)}"}

        except Exception as e:
            return {"status": "error", "message": f"Code validation failed: {e}"}

        engine._custom_loss = {"mode": "code", "code": code, "loss_cls": loss_cls}
        return {"status": "ok", "mode": "code", "class_name": loss_cls.__name__,
                "message": f"Custom loss '{loss_cls.__name__}' set. Will be used in next training run."}

    return {"status": "error", "message": f"Unknown mode: {mode}"}


@app.post("/api/llm/novel/arch-search")
async def architecture_search(body: dict[str, Any]):
    """Run an architecture search: try multiple block designs and rank by loss reduction.

    Body: {
        designs: {
            "name1": [...steps...],
            "name2": [...steps...],
        },
        // OR use: auto_search: true  (will try common combinations)
        d_model: 128,
        n_heads: 4,
        vocab_size: 256,
        n_layers: 2,
        text: "training text...",
        train_steps: 30,
    }
    """
    from state_graph.layers.llm import ComposableLLM, BLOCK_DESIGNS
    import time as _time

    d_model = body.get("d_model", 128)
    n_heads = body.get("n_heads", 4)
    vocab_size = body.get("vocab_size", 256)
    n_layers = body.get("n_layers", 2)
    train_steps = body.get("train_steps", 30)
    lr = body.get("learning_rate", 1e-3)
    text = body.get("text", "")

    if len(text) < 100:
        text = ("The quick brown fox jumps over the lazy dog. " * 50 +
                "Machine learning models process sequences of tokens. " * 30 +
                "Attention mechanisms focus on relevant input parts. " * 30)

    # Tokenize
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    encoded = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    actual_vocab = len(chars)
    if actual_vocab > vocab_size:
        vocab_size = actual_vocab + 1

    max_len = 64
    n_seqs = (len(encoded) - 1) // max_len
    if n_seqs < 2:
        return {"status": "error", "message": "Not enough text"}
    data = encoded[:n_seqs * max_len + 1]
    input_ids = data[:-1].view(n_seqs, max_len)
    labels = data[1:].view(n_seqs, max_len)

    # Get designs to test
    designs = body.get("designs", {})
    if body.get("auto_search", False) or not designs:
        designs = {
            "llama": BLOCK_DESIGNS.get("llama", []),
            "mamba": BLOCK_DESIGNS.get("mamba", []),
            "palm": BLOCK_DESIGNS.get("palm", []),
            "hybrid_mamba_attn": BLOCK_DESIGNS.get("hybrid_mamba_attn", []),
            "retnet": BLOCK_DESIGNS.get("retnet", []),
            "griffin": BLOCK_DESIGNS.get("griffin", []),
            "minimal": BLOCK_DESIGNS.get("minimal", []),
            "moe_block": BLOCK_DESIGNS.get("moe_block", []),
            "parallel_moe_mamba": BLOCK_DESIGNS.get("parallel_moe_mamba", []),
            "triple_hybrid": BLOCK_DESIGNS.get("triple_hybrid", []),
        }
        # Add user's custom designs
        designs.update(body.get("designs", {}))

    results = []

    for name, steps in designs.items():
        if not steps:
            continue
        try:
            model = ComposableLLM(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, max_len=max_len + 64, default_block=steps,
            )
            total_params = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            model.train()
            losses_list = []
            start = _time.time()
            for step in range(train_steps):
                idx = step % n_seqs
                out = model(input_ids[idx:idx+1], labels=labels[idx:idx+1])
                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses_list.append(loss.item())
            elapsed = _time.time() - start

            results.append({
                "name": name,
                "status": "ok",
                "params": total_params,
                "params_M": f"{total_params/1e6:.2f}M",
                "initial_loss": round(losses_list[0], 4),
                "final_loss": round(losses_list[-1], 4),
                "loss_reduction": round(losses_list[0] - losses_list[-1], 4),
                "time_seconds": round(elapsed, 2),
                "steps_per_sec": round(train_steps / elapsed, 1),
                "efficiency": round((losses_list[0] - losses_list[-1]) / (total_params / 1e6), 4),  # loss reduction per M params
                "step_types": [s.get("type", "?") for s in steps],
            })
        except Exception as e:
            results.append({"name": name, "status": "error", "message": str(e)})

    # Sort by loss reduction (best first)
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["loss_reduction"], reverse=True)
    err_results = [r for r in results if r["status"] != "ok"]

    ranked = ok_results + err_results

    return {
        "status": "ok",
        "n_designs": len(designs),
        "n_successful": len(ok_results),
        "ranking": ranked,
        "best": ok_results[0] if ok_results else None,
        "summary": {
            "best_loss_reduction": ok_results[0]["name"] if ok_results else "N/A",
            "best_efficiency": max(ok_results, key=lambda r: r["efficiency"])["name"] if ok_results else "N/A",
            "fastest": min(ok_results, key=lambda r: r["time_seconds"])["name"] if ok_results else "N/A",
            "smallest": min(ok_results, key=lambda r: r["params"])["name"] if ok_results else "N/A",
        },
    }


@app.get("/api/llm/blueprints")
async def llm_blueprints():
    """Return all model blueprints organized by category."""
    from state_graph.layers.llm import MODEL_BLUEPRINTS, get_blueprint_categories
    return {
        "blueprints": {k: {
            "name": v["name"],
            "category": v.get("category", "other"),
            "description": v["description"],
            "model_class": v["model_class"],
            "config": v["config"],
            "scalable_configs": v.get("scalable_configs", {}),
            "training_tips": v.get("training_tips", ""),
            "default_block": v.get("default_block"),
            "architecture": v.get("architecture"),
        } for k, v in MODEL_BLUEPRINTS.items()},
        "categories": get_blueprint_categories(),
    }


@app.post("/api/llm/blueprint/build")
async def build_from_blueprint(body: dict[str, Any]):
    """Build a model from a blueprint.

    Body: {
        blueprint: "gemini_scratch" | "claude_scratch" | "veo3_scratch" | ...,
        scale: "nano" | "small" | "medium" | "large" | "xl",  // optional
        overrides: { ... },  // optional: override any config parameter
    }
    """
    from state_graph.layers.llm import (
        MODEL_BLUEPRINTS, LLMModel, ComposableLLM, EncoderDecoderLLM,
        AdaptiveDepthLLM, MultiModalLLM, UnifiedMultiModalLLM,
    )

    blueprint_key = body.get("blueprint", "")
    if blueprint_key not in MODEL_BLUEPRINTS:
        return {"status": "error", "message": f"Unknown blueprint: {blueprint_key}. Available: {list(MODEL_BLUEPRINTS.keys())}"}

    bp = MODEL_BLUEPRINTS[blueprint_key]
    config = dict(bp["config"])

    # Apply scale preset if specified
    scale = body.get("scale")
    if scale and scale in bp.get("scalable_configs", {}):
        config.update(bp["scalable_configs"][scale])

    # Apply user overrides
    overrides = body.get("overrides", {})
    config.update(overrides)

    model_class_name = bp["model_class"]

    # Handle custom multi-component architectures (VeO3, Stable Diffusion)
    if model_class_name == "custom":
        arch = bp.get("architecture", {})
        components = {}
        total_params = 0

        for comp_name, comp_info in arch.items():
            components[comp_name] = {
                "class": comp_info["class"],
                "description": comp_info["description"],
                "config": comp_info["config"],
            }

        # Store blueprint info in engine
        engine._llm_config = {
            "blueprint": blueprint_key,
            "model_class": "custom",
            "architecture": arch,
            "config": config,
        }

        return {
            "status": "ok",
            "model_type": f"blueprint:{blueprint_key}",
            "name": bp["name"],
            "description": bp["description"],
            "components": components,
            "training_tips": bp.get("training_tips", ""),
            "message": f"Blueprint '{bp['name']}' loaded. This is a multi-component architecture. Use /api/llm/blueprint/build-component to instantiate individual components.",
        }

    # Map model class name to actual class
    class_map = {
        "LLMModel": LLMModel,
        "ComposableLLM": ComposableLLM,
        "EncoderDecoderLLM": EncoderDecoderLLM,
        "AdaptiveDepthLLM": AdaptiveDepthLLM,
        "MultiModalLLM": MultiModalLLM,
        "UnifiedMultiModalLLM": UnifiedMultiModalLLM,
    }

    ModelClass = class_map.get(model_class_name)
    if ModelClass is None:
        return {"status": "error", "message": f"Unknown model class: {model_class_name}"}

    # Handle ComposableLLM default_block
    build_config = dict(config)
    if model_class_name == "ComposableLLM" and "default_block" in bp:
        build_config["default_block"] = bp["default_block"]

    # Remove non-model params
    for key in ["moe_layers"]:
        if key in build_config and model_class_name not in ["LLMModel"]:
            build_config.pop(key, None)

    try:
        model = ModelClass(**build_config)
    except TypeError as e:
        # Some params may not apply to all model classes — filter them
        valid_keys = set(ModelClass.__init__.__code__.co_varnames)
        filtered = {k: v for k, v in build_config.items() if k in valid_keys}
        model = ModelClass(**filtered)

    engine.model = model
    engine._llm_config = {
        "blueprint": blueprint_key,
        "model_class": model_class_name,
        "config": config,
    }

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "status": "ok",
        "model_type": f"blueprint:{blueprint_key}",
        "name": bp["name"],
        "description": bp["description"],
        "model_class": model_class_name,
        "config": config,
        "parameters": {
            "total": total,
            "trainable": trainable,
            "total_M": f"{total / 1e6:.1f}M",
        },
        "training_tips": bp.get("training_tips", ""),
        "scalable_configs": list(bp.get("scalable_configs", {}).keys()),
    }


@app.post("/api/llm/blueprint/modify")
async def modify_blueprint_model(body: dict[str, Any]):
    """Modify the current blueprint model's configuration.

    Body: {
        changes: { "d_model": 1024, "n_layers": 24, ... },
        rebuild: true  // whether to rebuild the model
    }
    """
    config = getattr(engine, '_llm_config', None)
    if not config or "blueprint" not in config:
        return {"status": "error", "message": "No blueprint model loaded. Use /api/llm/blueprint/build first."}

    from state_graph.layers.llm import (
        MODEL_BLUEPRINTS, LLMModel, ComposableLLM, EncoderDecoderLLM,
        AdaptiveDepthLLM, MultiModalLLM, UnifiedMultiModalLLM,
    )

    changes = body.get("changes", {})
    config["config"].update(changes)

    if body.get("rebuild", True) and config["model_class"] != "custom":
        bp = MODEL_BLUEPRINTS.get(config["blueprint"], {})
        model_class_name = config["model_class"]

        class_map = {
            "LLMModel": LLMModel,
            "ComposableLLM": ComposableLLM,
            "EncoderDecoderLLM": EncoderDecoderLLM,
            "AdaptiveDepthLLM": AdaptiveDepthLLM,
            "MultiModalLLM": MultiModalLLM,
            "UnifiedMultiModalLLM": UnifiedMultiModalLLM,
        }

        ModelClass = class_map.get(model_class_name)
        if ModelClass:
            build_config = dict(config["config"])
            if model_class_name == "ComposableLLM" and "default_block" in bp:
                build_config["default_block"] = bp["default_block"]

            try:
                model = ModelClass(**build_config)
            except TypeError:
                valid_keys = set(ModelClass.__init__.__code__.co_varnames)
                filtered = {k: v for k, v in build_config.items() if k in valid_keys}
                model = ModelClass(**filtered)

            engine.model = model

            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return {
                "status": "ok",
                "message": "Model rebuilt with updated config",
                "config": config["config"],
                "parameters": {"total": total, "trainable": trainable, "total_M": f"{total / 1e6:.1f}M"},
            }

    return {"status": "ok", "message": "Config updated (model not rebuilt)", "config": config["config"]}


@app.post("/api/llm/blueprint/build-components")
async def build_blueprint_components(body: dict[str, Any]):
    """Build all components of a multi-component blueprint (VeO3, Stable Diffusion).

    Body: {
        blueprint: "veo3_scratch" | "stable_diffusion_scratch",
        scale: "nano" | "small" | "medium",
        overrides: {},
    }
    """
    from state_graph.layers.llm import MODEL_BLUEPRINTS, EncoderDecoderLLM
    from state_graph.layers.custom import DiffusionUNet, VAE, VideoVAE, NoiseScheduler

    blueprint_key = body.get("blueprint", "")
    bp = MODEL_BLUEPRINTS.get(blueprint_key)
    if not bp or bp.get("model_class") != "custom":
        return {"status": "error", "message": f"Blueprint '{blueprint_key}' is not a multi-component architecture"}

    arch = bp.get("architecture", {})
    class_map = {
        "DiffusionUNet": DiffusionUNet,
        "VAE": VAE,
        "VideoVAE": VideoVAE,
        "NoiseScheduler": NoiseScheduler,
        "EncoderDecoderLLM": EncoderDecoderLLM,
    }

    built = {}
    total_params = 0

    for comp_name, comp_info in arch.items():
        cls_name = comp_info["class"]
        cls = class_map.get(cls_name)
        if cls is None:
            built[comp_name] = {"status": "skipped", "reason": f"Unknown class: {cls_name}"}
            continue

        cfg = dict(comp_info["config"])
        # Apply scale overrides
        scale = body.get("scale")
        if scale:
            scale_cfg = bp.get("scalable_configs", {}).get(scale, {})
            for k, v in scale_cfg.items():
                if k.startswith(comp_name + "."):
                    cfg[k.split(".", 1)[1]] = v

        # Apply user overrides
        for k, v in body.get("overrides", {}).items():
            if k.startswith(comp_name + "."):
                cfg[k.split(".", 1)[1]] = v

        try:
            if cls_name == "NoiseScheduler":
                obj = cls(**cfg)
                built[comp_name] = {"status": "built", "class": cls_name, "config": cfg}
                if not hasattr(engine, '_diffusion_components'):
                    engine._diffusion_components = {}
                engine._diffusion_components[comp_name] = obj
            elif cls_name == "EncoderDecoderLLM":
                # Text encoder: only build encoder layers
                if cfg.get("n_decoder_layers", 0) == 0:
                    cfg["n_decoder_layers"] = 1  # Need at least 1 for the class
                obj = cls(**cfg)
                params = sum(p.numel() for p in obj.parameters())
                total_params += params
                built[comp_name] = {"status": "built", "class": cls_name, "params": params, "params_M": f"{params/1e6:.1f}M"}
                if not hasattr(engine, '_diffusion_components'):
                    engine._diffusion_components = {}
                engine._diffusion_components[comp_name] = obj.to(engine.device)
            else:
                obj = cls(**cfg)
                params = sum(p.numel() for p in obj.parameters())
                total_params += params
                built[comp_name] = {"status": "built", "class": cls_name, "params": params, "params_M": f"{params/1e6:.1f}M"}
                if not hasattr(engine, '_diffusion_components'):
                    engine._diffusion_components = {}
                engine._diffusion_components[comp_name] = obj.to(engine.device)
        except Exception as e:
            built[comp_name] = {"status": "error", "message": str(e)}

    engine._llm_config = {
        "blueprint": blueprint_key,
        "model_class": "custom",
        "architecture": arch,
        "components": built,
    }

    return {
        "status": "ok",
        "blueprint": blueprint_key,
        "name": bp["name"],
        "components": built,
        "total_params": total_params,
        "total_params_M": f"{total_params/1e6:.1f}M",
        "training_tips": bp.get("training_tips", ""),
    }


@app.post("/api/diffusion/train")
async def train_diffusion(body: dict[str, Any]):
    """Train a diffusion model (image or video generation).

    Body: {
        mode: "image" | "video",
        epochs: 10,
        batch_size: 4,
        learning_rate: 1e-4,
        image_size: 64,
        n_steps: 1000,
        schedule: "cosine",
        use_vae: true,       // train in latent space
        data_source: "random" | "folder_path",
    }
    """
    import threading
    from state_graph.layers.custom import DiffusionUNet, VAE, VideoVAE, NoiseScheduler

    mode = body.get("mode", "image")
    epochs = body.get("epochs", 10)
    batch_size = body.get("batch_size", 4)
    lr = body.get("learning_rate", 1e-4)
    image_size = body.get("image_size", 64)
    n_steps = body.get("n_steps", 1000)
    schedule = body.get("schedule", "cosine")
    use_vae = body.get("use_vae", False)

    # Get or create components
    components = getattr(engine, '_diffusion_components', {})

    # Noise scheduler
    if "noise_scheduler" not in components:
        components["noise_scheduler"] = NoiseScheduler(n_steps=n_steps, schedule=schedule)

    sched = components["noise_scheduler"]

    # UNet denoiser
    if "denoiser" not in components:
        in_ch = 4 if use_vae else 3
        components["denoiser"] = DiffusionUNet(
            in_channels=in_ch, out_channels=in_ch,
            base_channels=body.get("base_channels", 64),
            channel_mults=tuple(body.get("channel_mults", [1, 2, 4])),
            n_res_blocks=body.get("n_res_blocks", 2),
            time_dim=body.get("time_dim", 256),
            context_dim=body.get("context_dim", 256),
            n_heads=body.get("n_heads", 4),
        ).to(engine.device)

    unet = components["denoiser"]

    # VAE (optional)
    vae = None
    if use_vae:
        if "vae" not in components:
            components["vae"] = VAE(
                in_channels=3, latent_channels=4,
                base_channels=32, channel_mults=(1, 2, 4),
            ).to(engine.device)
        vae = components["vae"]
        vae.eval()

    engine._diffusion_components = components

    # Create synthetic data for demo (or load from folder)
    data_source = body.get("data_source", "random")
    if data_source == "random":
        n_samples = body.get("n_samples", max(batch_size * 10, 64))
        data = torch.randn(n_samples, 3, image_size, image_size)
    else:
        return {"status": "error", "message": "Folder data loading not yet implemented. Use data_source='random' for demo."}

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=0.01)

    if engine._is_training:
        return {"status": "error", "message": "Training already in progress"}

    engine._stop_event.clear()
    engine._is_training = True

    total_params = sum(p.numel() for p in unet.parameters())

    def train():
        try:
            unet.train()
            step = 0
            for epoch in range(epochs):
                if engine._stop_event.is_set():
                    break
                epoch_loss = 0.0
                n_batches = 0
                for (batch,) in loader:
                    if engine._stop_event.is_set():
                        break
                    batch = batch.to(engine.device)

                    # Encode to latent if using VAE
                    if vae is not None:
                        with torch.no_grad():
                            mu, log_var = vae.encode(batch)
                            batch = mu  # Use mean (no reparameterization for training UNet)

                    # Sample noise and timesteps
                    noise = torch.randn_like(batch)
                    t = sched.sample_timesteps(batch.shape[0], engine.device)
                    noisy = sched.add_noise(batch, noise, t)

                    # Predict noise
                    pred_noise = unet(noisy, t)

                    # MSE loss
                    loss = torch.nn.functional.mse_loss(pred_noise, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    step += 1

                avg_loss = epoch_loss / max(n_batches, 1)

                engine._emit_from_thread("training_step", {
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "loss": avg_loss, "step": step,
                    "model_type": "diffusion",
                })

            engine._emit_from_thread("training_complete", {
                "epochs": epochs, "final_loss": avg_loss if 'avg_loss' in dir() else 0,
                "model_type": "diffusion",
            })
        except Exception as e:
            import traceback
            engine._emit_from_thread("training_error", {"error": str(e), "traceback": traceback.format_exc()})
        finally:
            engine._is_training = False

    engine._train_thread = threading.Thread(target=train, daemon=True)
    engine._train_thread.start()

    return {
        "status": "started",
        "mode": mode,
        "model_params": total_params,
        "model_params_M": f"{total_params/1e6:.1f}M",
        "epochs": epochs,
        "n_samples": len(data),
        "image_size": image_size,
        "use_vae": use_vae,
    }


@app.post("/api/diffusion/generate")
async def diffusion_generate(body: dict[str, Any]):
    """Generate images using a trained diffusion model.

    Body: {
        n_images: 4,
        n_steps: 50,
        guidance_scale: 7.5,
        prompt: "optional text prompt",
    }
    """
    components = getattr(engine, '_diffusion_components', {})
    unet = components.get("denoiser")
    if unet is None:
        return {"status": "error", "message": "No diffusion model loaded. Build a blueprint first."}

    sched = components.get("noise_scheduler")
    vae = components.get("vae")

    n_images = body.get("n_images", 4)
    n_steps = body.get("n_steps", 50)

    unet.eval()
    device = engine.device

    # Determine latent shape from the UNet input
    in_ch = 4 if vae else 3
    image_size = body.get("image_size", 64)
    latent_size = image_size // 8 if vae else image_size

    # DDPM sampling (simplified)
    with torch.no_grad():
        x = torch.randn(n_images, in_ch, latent_size, latent_size, device=device)

        # Simple linear schedule for sampling
        timesteps = torch.linspace(sched.n_steps - 1, 0, n_steps, dtype=torch.long, device=device) if sched else torch.linspace(999, 0, n_steps, dtype=torch.long, device=device)

        for t_val in timesteps:
            t = torch.full((n_images,), t_val.long().item(), device=device, dtype=torch.long)
            pred_noise = unet(x, t)
            # Simplified DDPM step
            alpha = sched.alphas[t_val.long().item()] if sched else 0.99
            alpha_bar = sched.alpha_cumprod[t_val.long().item()] if sched else 0.5
            beta = 1 - alpha
            x = (1 / alpha**0.5) * (x - (beta / (1 - alpha_bar)**0.5) * pred_noise)
            if t_val > 0:
                x = x + (beta**0.5) * torch.randn_like(x) * 0.5

        # Decode from latent if VAE
        if vae is not None:
            x = vae.decode(x)

        # Clamp to valid range
        x = x.clamp(-1, 1)

    return {
        "status": "ok",
        "n_images": n_images,
        "image_shape": list(x.shape),
        "message": f"Generated {n_images} images of shape {list(x.shape[1:])}",
    }


@app.post("/api/diffusion/train-vae")
async def train_vae(body: dict[str, Any]):
    """Train the VAE component for latent diffusion.

    Body: {
        epochs: 20,
        batch_size: 8,
        learning_rate: 1e-4,
        image_size: 64,
        kl_weight: 0.01,
        data_source: "random",
    }
    """
    import threading
    from state_graph.layers.custom import VAE

    epochs = body.get("epochs", 20)
    batch_size = body.get("batch_size", 8)
    lr = body.get("learning_rate", 1e-4)
    image_size = body.get("image_size", 64)
    kl_weight = body.get("kl_weight", 0.01)

    components = getattr(engine, '_diffusion_components', {})
    if "vae" not in components:
        components["vae"] = VAE(
            in_channels=3, latent_channels=4,
            base_channels=body.get("base_channels", 32),
            channel_mults=tuple(body.get("channel_mults", [1, 2, 4])),
        ).to(engine.device)
    engine._diffusion_components = components

    vae = components["vae"]
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01)

    # Synthetic data
    n_samples = body.get("n_samples", max(batch_size * 10, 64))
    data = torch.randn(n_samples, 3, image_size, image_size)

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    if engine._is_training:
        return {"status": "error", "message": "Training already in progress"}

    engine._stop_event.clear()
    engine._is_training = True
    total_params = sum(p.numel() for p in vae.parameters())

    def train():
        try:
            vae.train()
            step = 0
            for epoch in range(epochs):
                if engine._stop_event.is_set():
                    break
                epoch_loss = 0.0
                n_batches = 0
                for (batch,) in loader:
                    if engine._stop_event.is_set():
                        break
                    batch = batch.to(engine.device)
                    out = vae(batch)
                    recon_loss = torch.nn.functional.mse_loss(out["reconstruction"], batch)
                    loss = recon_loss + kl_weight * out["kl_loss"]

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1
                    step += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                engine._emit_from_thread("training_step", {
                    "epoch": epoch + 1, "total_epochs": epochs,
                    "loss": avg_loss, "step": step,
                    "model_type": "vae",
                })

            engine._emit_from_thread("training_complete", {
                "epochs": epochs, "final_loss": avg_loss if 'avg_loss' in dir() else 0,
                "model_type": "vae",
            })
        except Exception as e:
            import traceback
            engine._emit_from_thread("training_error", {"error": str(e), "traceback": traceback.format_exc()})
        finally:
            engine._is_training = False

    engine._train_thread = threading.Thread(target=train, daemon=True)
    engine._train_thread.start()

    return {
        "status": "started",
        "model_type": "vae",
        "model_params_M": f"{total_params/1e6:.1f}M",
        "epochs": epochs,
        "image_size": image_size,
        "kl_weight": kl_weight,
    }


@app.post("/api/llm/encoder-decoder")
async def build_encoder_decoder(body: dict[str, Any]):
    """Build an encoder-decoder model (T5/BART-style)."""
    from state_graph.layers.llm import EncoderDecoderLLM

    config = {
        "vocab_size": body.get("vocab_size", 32000),
        "d_model": body.get("d_model", 512),
        "n_encoder_layers": body.get("n_encoder_layers", 6),
        "n_decoder_layers": body.get("n_decoder_layers", 6),
        "n_heads": body.get("n_heads", 8),
        "ffn_hidden_dim": body.get("ffn_hidden_dim"),
        "max_len": body.get("max_len", 2048),
        "dropout": body.get("dropout", 0.0),
        "norm_type": body.get("norm_type", "layernorm"),
        "ffn_type": body.get("ffn_type", "standard"),
        "tie_weights": body.get("tie_weights", True),
        "pos_encoding": body.get("pos_encoding", "sinusoidal"),
        "share_embeddings": body.get("share_embeddings", True),
    }

    try:
        model = EncoderDecoderLLM(**config)
    except Exception as e:
        return {"status": "error", "message": f"Failed to build: {e}"}

    engine.model = model.to(engine.device)
    engine.model_source = "llm"
    engine._llm_config = config
    engine._llm_config["model_type"] = "encoder_decoder"

    param_info = model.count_parameters()

    await broadcast("model_built", {
        "status": "built", "model_type": "encoder_decoder",
        "config": config, "total_params": param_info["total"],
        "total_params_M": param_info["total_M"], "device": str(engine.device),
    })

    return {"status": "built", "config": config, "params": param_info, "device": str(engine.device)}


@app.post("/api/llm/adaptive-depth")
async def build_adaptive_depth(body: dict[str, Any]):
    """Build an LLM with early exit / adaptive depth."""
    from state_graph.layers.llm import AdaptiveDepthLLM

    config = {
        "vocab_size": body.get("vocab_size", 32000),
        "d_model": body.get("d_model", 512),
        "n_layers": body.get("n_layers", 12),
        "n_heads": body.get("n_heads", 8),
        "max_len": body.get("max_len", 2048),
        "dropout": body.get("dropout", 0.0),
        "norm_type": body.get("norm_type", "rmsnorm"),
        "ffn_type": body.get("ffn_type", "swiglu"),
        "exit_interval": body.get("exit_interval", 2),
        "exit_threshold": body.get("exit_threshold", 0.9),
        "tie_weights": body.get("tie_weights", True),
    }

    try:
        model = AdaptiveDepthLLM(**config)
    except Exception as e:
        return {"status": "error", "message": f"Failed to build: {e}"}

    engine.model = model.to(engine.device)
    engine.model_source = "llm"
    engine._llm_config = config
    engine._llm_config["model_type"] = "adaptive_depth"

    param_info = model.count_parameters()

    await broadcast("model_built", {
        "status": "built", "model_type": "adaptive_depth",
        "config": config, "total_params": param_info["total"],
        "total_params_M": param_info["total_M"], "device": str(engine.device),
    })

    return {"status": "built", "config": config, "params": param_info, "device": str(engine.device)}


@app.post("/api/llm/multimodal")
async def build_multimodal(body: dict[str, Any]):
    """Build a multi-modal LLM (text + image + audio)."""
    from state_graph.layers.llm import MultiModalLLM

    config = {
        "vocab_size": body.get("vocab_size", 32000),
        "d_model": body.get("d_model", 512),
        "n_layers": body.get("n_layers", 6),
        "n_heads": body.get("n_heads", 8),
        "max_len": body.get("max_len", 2048),
        "dropout": body.get("dropout", 0.0),
        "norm_type": body.get("norm_type", "rmsnorm"),
        "ffn_type": body.get("ffn_type", "swiglu"),
        "tie_weights": body.get("tie_weights", True),
        "image_size": body.get("image_size", 224),
        "patch_size": body.get("patch_size", 16),
        "in_channels": body.get("in_channels", 3),
        "n_mels": body.get("n_mels", 80),
        "fusion_mode": body.get("fusion_mode", "prepend"),
        "modalities": body.get("modalities", ["text", "image"]),
    }

    try:
        model = MultiModalLLM(**config)
    except Exception as e:
        return {"status": "error", "message": f"Failed to build: {e}"}

    engine.model = model.to(engine.device)
    engine.model_source = "llm"
    engine._llm_config = config
    engine._llm_config["model_type"] = "multimodal"

    param_info = model.count_parameters()

    await broadcast("model_built", {
        "status": "built", "model_type": "multimodal",
        "config": config, "total_params": param_info["total"],
        "total_params_M": param_info["total_M"], "device": str(engine.device),
    })

    return {"status": "built", "config": config, "params": param_info, "device": str(engine.device)}


@app.post("/api/llm/tokenizer/train")
async def train_tokenizer(body: dict[str, Any]):
    """Train a custom tokenizer from text."""
    from state_graph.layers.llm import TokenizerTrainer

    text = body.get("text", "")
    if len(text) < 100:
        return {"status": "error", "message": "Need at least 100 characters to train a tokenizer"}

    algorithm = body.get("algorithm", "bpe")
    vocab_size = body.get("vocab_size", 8000)
    min_frequency = body.get("min_frequency", 2)

    try:
        result = TokenizerTrainer.train(text, vocab_size, algorithm, min_frequency)
    except Exception as e:
        return {"status": "error", "message": f"Failed to train tokenizer: {e}"}

    # Store on engine for use in training
    engine._llm_tokenizer = {
        "type": "custom_trained",
        "algorithm": result["algorithm"],
        "vocab_size": result["vocab_size"],
        "encode": result["encode"],
        "decode": result["decode"],
    }
    # Store raw tokenizer object if available
    if "tokenizer" in result:
        engine._llm_tokenizer["tokenizer_obj"] = result["tokenizer"]
    if "char2idx" in result:
        engine._llm_tokenizer["char2idx"] = result["char2idx"]
        engine._llm_tokenizer["idx2char"] = result["idx2char"]

    return {
        "status": "trained",
        "algorithm": result["algorithm"],
        "vocab_size": result["vocab_size"],
    }


@app.get("/api/architecture/formulas")
async def get_formulas():
    """Return formula descriptions for all registered components."""
    return {"formulas": COMPONENT_FORMULAS}


@app.get("/api/architecture/visualize")
async def visualize_architecture():
    """Return current model architecture tree for visualization."""
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}

    if hasattr(engine, '_llm_config'):
        from state_graph.layers.llm import LLMModel
        if isinstance(engine.model, LLMModel):
            return {
                "status": "ok",
                "model_type": "llm",
                "architecture": _get_llm_architecture(engine.model, engine._llm_config),
            }

    # Check if built from graph
    graph_nodes = engine.graph.get_sorted_nodes()
    if graph_nodes:
        arch = _extract_graph_architecture(engine.model, graph_nodes)
        return {"status": "ok", "model_type": "graph", "architecture": arch}

    # Generic model architecture extraction (HF models, etc.)
    arch = _extract_generic_architecture(engine.model)
    model_name = type(engine.model).__name__
    return {"status": "ok", "model_type": model_name, "architecture": arch}


def _extract_graph_architecture(model, graph_nodes) -> list[dict]:
    """Extract architecture from graph-built model with node info."""
    arch = []
    modules = list(model.children()) if hasattr(model, 'children') else []
    mod_idx = 0
    for node in graph_nodes:
        entry = {
            "name": f"{node.layer_type}" + (f" ({node.activation})" if node.activation else ""),
            "type": node.layer_type,
            "params": node.params,
            "param_count": 0,
        }
        # Get actual param count from module
        if mod_idx < len(modules):
            entry["param_count"] = sum(p.numel() for p in modules[mod_idx].parameters())
            # Extract children for complex layers
            children = list(modules[mod_idx].named_children())
            if children:
                entry["children"] = _extract_generic_architecture(modules[mod_idx])
            mod_idx += 1
            # Skip activation module
            if node.activation and mod_idx < len(modules):
                mod_idx += 1
        arch.append(entry)
    return arch


def _extract_generic_architecture(model: torch.nn.Module, depth: int = 0) -> list[dict]:
    """Extract architecture tree from any PyTorch model (HF, YOLO, diffusers, ViT, etc.)."""
    arch = []
    MAX_DEPTH = 6  # Prevent excessive recursion

    for name, module in model.named_children():
        mod_type = type(module).__name__
        node = {
            "name": name,
            "type": mod_type,
            "params": {},
            "param_count": sum(p.numel() for p in module.parameters()),
        }

        # Extract common params from any PyTorch module
        _extract_module_params(module, node)

        # Extract weight dtype and shape info
        for pname, param in module.named_parameters(recurse=False):
            if pname == 'weight':
                shape = list(param.shape)
                node["params"]["weight_shape"] = shape
                node["params"]["dtype"] = str(param.dtype).replace("torch.", "")
                break

        # Recurse into children (with depth limit)
        children = list(module.named_children())
        if children and depth < MAX_DEPTH:
            node["children"] = _extract_generic_architecture(module, depth + 1)

        arch.append(node)

    # Handle nn.Sequential with indexed children
    if isinstance(model, torch.nn.Sequential) and not arch:
        for i, module in enumerate(model):
            mod_type = type(module).__name__
            node = {
                "name": f"layer_{i}",
                "type": mod_type,
                "params": {},
                "param_count": sum(p.numel() for p in module.parameters()),
            }
            _extract_module_params(module, node)
            children = list(module.named_children())
            if children and depth < MAX_DEPTH:
                node["children"] = _extract_generic_architecture(module, depth + 1)
            arch.append(node)

    return arch


def _extract_module_params(module, node: dict) -> None:
    """Extract key parameters from a PyTorch module into the node dict."""
    param_attrs = [
        'in_features', 'out_features', 'in_channels', 'out_channels',
        'kernel_size', 'stride', 'padding', 'dilation', 'groups',
        'num_features', 'normalized_shape', 'embed_dim', 'num_heads',
        'hidden_size', 'intermediate_size', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads',
        'd_model', 'nhead', 'dim_feedforward',
        'num_embeddings', 'embedding_dim', 'bias',
    ]
    for attr in param_attrs:
        val = getattr(module, attr, None)
        if val is not None:
            # Convert tensors/non-serializable to basic types
            if isinstance(val, (int, float, str, bool, list, tuple)):
                node["params"][attr] = val
    # Dropout probability
    if hasattr(module, 'p') and isinstance(getattr(module, 'p', None), float):
        node["params"]["p"] = module.p


@app.post("/api/llm/train")
async def start_llm_training(body: dict[str, Any]):
    """Start LLM training with text data."""
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}

    # Update config
    engine.config["epochs"] = body.get("epochs", engine.config["epochs"])
    engine.config["batch_size"] = body.get("batch_size", engine.config["batch_size"])
    engine.config["learning_rate"] = body.get("learning_rate", engine.config["learning_rate"])

    text_data = body.get("text", "")
    tokenizer_name = body.get("tokenizer", "char")  # "char" or HF tokenizer name
    max_len = body.get("max_len", 256)

    if not text_data:
        return {"status": "error", "message": "No training text provided"}

    # Tokenize
    if tokenizer_name == "char":
        chars = sorted(set(text_data))
        char2idx = {c: i for i, c in enumerate(chars)}
        encoded = [char2idx[c] for c in text_data]
        vocab_size = len(chars)
        engine._llm_tokenizer = {"type": "char", "char2idx": char2idx, "idx2char": {i: c for c, i in char2idx.items()}, "vocab_size": vocab_size}
    elif tokenizer_name == "custom_trained":
        # Use previously trained custom tokenizer
        tok_info = getattr(engine, '_llm_tokenizer', None)
        if not tok_info or tok_info.get("type") != "custom_trained":
            return {"status": "error", "message": "No custom tokenizer trained. Use /api/llm/tokenizer/train first."}
        encoded = tok_info["encode"](text_data)
    else:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            encoded = tok.encode(text_data)
            engine._llm_tokenizer = {"type": "hf", "tokenizer": tok, "vocab_size": tok.vocab_size}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load tokenizer: {e}"}

    # Determine actual vocab size from tokenizer and resize model if needed
    tok_info = getattr(engine, '_llm_tokenizer', {})
    data_vocab = tok_info.get("vocab_size") or (max(encoded) + 1)
    model_vocab = getattr(engine.model, 'vocab_size', None)
    if model_vocab is not None and data_vocab > model_vocab:
        _resize_llm_embeddings(engine.model, data_vocab)
        engine.model.vocab_size = data_vocab
        if engine._llm_config:
            engine._llm_config["vocab_size"] = data_vocab

    # Create sequences
    data = torch.tensor(encoded, dtype=torch.long)
    if len(data) - 1 < max_len:
        return {"status": "error", "message": f"Not enough data: {len(data)} tokens but need at least {max_len + 1}. Provide more text or reduce sequence length."}
    n_seqs = (len(data) - 1) // max_len
    if n_seqs < 2:
        return {"status": "error", "message": f"Not enough data: only {n_seqs} sequence(s). Provide more text or reduce sequence length."}
    trim = n_seqs * max_len + 1
    data = data[:trim]

    input_ids = data[:-1].view(n_seqs, max_len)
    labels = data[1:].view(n_seqs, max_len)
    n_val = max(1, n_seqs // 10)
    n_train = n_seqs - n_val
    if n_train < 1:
        n_train = 1
        n_val = n_seqs - 1

    from torch.utils.data import DataLoader, TensorDataset
    batch_size = engine.config["batch_size"]
    engine._train_loader = DataLoader(
        TensorDataset(input_ids[:n_train], labels[:n_train]),
        batch_size=batch_size, shuffle=True,
    )
    engine._val_loader = DataLoader(
        TensorDataset(input_ids[n_train:], labels[n_train:]),
        batch_size=batch_size,
    )

    # Setup optimizer
    engine.optimizer = torch.optim.AdamW(
        engine.model.parameters(),
        lr=engine.config["learning_rate"],
        weight_decay=0.01,
    )

    # Setup scheduler
    if engine.config.get("scheduler"):
        from state_graph.core.scheduler import SchedulerRegistry
        engine.scheduler = SchedulerRegistry.create(
            engine.config["scheduler"], engine.optimizer, engine.config.get("scheduler_params", {})
        )

    engine.loss_fn = None  # LLM computes its own loss

    # Start training thread
    import threading
    if engine._is_training:
        return {"status": "error", "message": "Training already in progress"}

    engine._stop_event.clear()
    engine._is_training = True

    def train():
        try:
            _llm_train_loop()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            engine._emit_from_thread("training_error", {
                "error": str(e),
                "traceback": tb,
            })
        finally:
            engine._is_training = False

    engine._train_thread = threading.Thread(target=train, daemon=True)
    engine._train_thread.start()

    return {
        "status": "started",
        "n_sequences": n_seqs,
        "n_train": n_train,
        "n_val": n_val,
        "seq_len": max_len,
    }


def _resize_llm_embeddings(model, new_vocab_size: int):
    """Resize embedding and lm_head layers to match a new vocab size."""
    import torch.nn as nn
    old_emb = model.tok_emb
    if isinstance(old_emb, nn.Embedding):
        old_vocab, d_model = old_emb.weight.shape
        if new_vocab_size != old_vocab:
            new_emb = nn.Embedding(new_vocab_size, d_model)
            # Copy existing weights
            copy_size = min(old_vocab, new_vocab_size)
            new_emb.weight.data[:copy_size] = old_emb.weight.data[:copy_size]
            model.tok_emb = new_emb
            # Resize lm_head
            if hasattr(model, 'lm_head'):
                old_head = model.lm_head
                new_head = nn.Linear(d_model, new_vocab_size, bias=old_head.bias is not None)
                copy_size_head = min(old_head.out_features, new_vocab_size)
                new_head.weight.data[:copy_size_head] = old_head.weight.data[:copy_size_head]
                if old_head.bias is not None:
                    new_head.bias.data[:copy_size_head] = old_head.bias.data[:copy_size_head]
                model.lm_head = new_head
                # Re-tie weights if they were tied
                if getattr(model, 'tie_weights', False) or (hasattr(model, '_tie_weights') and model._tie_weights):
                    model.lm_head.weight = model.tok_emb.weight
            model.to(next(model.parameters()).device)


def _llm_train_loop():
    """Training loop for LLM models."""
    model = engine.model
    optimizer = engine.optimizer
    scheduler = engine.scheduler
    device = engine.device

    for epoch in range(engine.config["epochs"]):
        if engine._stop_event.is_set():
            break

        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(engine._train_loader):
            if engine._stop_event.is_set():
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            step = epoch * len(engine._train_loader) + batch_idx

            # Collect per-layer metrics
            layer_metrics = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    short_name = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
                    layer_metrics[short_name] = {
                        "weight_mean": param.data.mean().item(),
                        "weight_std": param.data.std().item(),
                        "grad_norm": param.grad.data.norm().item(),
                    }

            lr_dict = {f"group_{i}": g["lr"] for i, g in enumerate(optimizer.param_groups)}

            engine._emit_from_thread("step", {
                "step": step,
                "loss": loss.item(),
                "lr": lr_dict,
                "layers": dict(list(layer_metrics.items())[:20]),
                "perplexity": min(math.exp(loss.item()), 1e6),
            })

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for input_ids, labels in (engine._val_loader or []):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                output = model(input_ids, labels=labels)
                val_loss += output["loss"].item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        if scheduler:
            scheduler.step()

        engine._emit_from_thread("epoch", {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_perplexity": min(math.exp(avg_train_loss), 1e6),
            "val_perplexity": min(math.exp(avg_val_loss), 1e6),
        })

    engine._emit_from_thread("training_complete", {
        "final_loss": avg_train_loss if 'avg_train_loss' in dir() else 0,
    })


@app.post("/api/llm/generate")
async def llm_generate(body: dict[str, Any]):
    """Generate text from the trained LLM."""
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}
    if not hasattr(engine.model, 'generate'):
        return {"status": "error", "message": "Model does not support generation"}

    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 100)
    temperature = body.get("temperature", 0.8)
    top_k = body.get("top_k", 50)

    tokenizer_info = getattr(engine, '_llm_tokenizer', None)
    if not tokenizer_info:
        return {"status": "error", "message": "No tokenizer configured"}

    if tokenizer_info["type"] == "char":
        char2idx = tokenizer_info["char2idx"]
        idx2char = tokenizer_info["idx2char"]
        input_ids = torch.tensor([[char2idx.get(c, 0) for c in prompt]], dtype=torch.long).to(engine.device)

        output_ids = engine.model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        generated = ''.join(idx2char.get(t.item(), '?') for t in output_ids[0])
        return {"status": "ok", "text": generated, "prompt_len": len(prompt)}
    else:
        tok = tokenizer_info["tokenizer"]
        input_ids = tok.encode(prompt, return_tensors="pt").to(engine.device)
        output_ids = engine.model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        generated = tok.decode(output_ids[0], skip_special_tokens=True)
        return {"status": "ok", "text": generated, "prompt_len": len(prompt)}


# --- Hugging Face Integration ---

def _get_hf_manager():
    """Lazy-init HF model manager."""
    if engine.hf_manager is None:
        try:
            from state_graph.hf.hub import HFModelManager
            engine.hf_manager = HFModelManager()
        except ImportError:
            raise RuntimeError("HuggingFace not installed. Run: pip install -e '.[hf]'")
    return engine.hf_manager


def _get_hf_data():
    """Lazy-init HF data manager."""
    if engine.hf_data is None:
        try:
            from state_graph.hf.datasets import HFDataManager
            engine.hf_data = HFDataManager()
        except ImportError:
            raise RuntimeError("HuggingFace datasets not installed. Run: pip install -e '.[hf]'")
    return engine.hf_data


@app.get("/api/hf/search")
async def hf_search_models(query: str, library: str | None = None, task: str | None = None, limit: int = 20):
    mgr = _get_hf_manager()
    return {"models": mgr.search_models(query, library=library, task=task, limit=limit)}


@app.get("/api/hf/datasets/search")
async def hf_search_datasets(query: str, task: str | None = None, limit: int = 20):
    mgr = _get_hf_manager()
    return {"datasets": mgr.search_datasets(query, task=task, limit=limit)}


@app.post("/api/hf/load")
async def hf_load_model(body: dict[str, Any]):
    mgr = _get_hf_manager()
    result = mgr.load_model(
        model_id=body["model_id"],
        library=body.get("library"),
        task=body.get("task"),
        num_labels=body.get("num_labels"),
        dtype=body.get("dtype"),
    )
    engine.model_source = "hf"
    await broadcast("hf_model_loaded", result)
    return result


@app.get("/api/hf/model/tree")
async def hf_model_tree(max_depth: int = 3):
    mgr = _get_hf_manager()
    return {"tree": mgr.get_model_tree(max_depth)}


@app.get("/api/hf/model/info")
async def hf_model_info():
    mgr = _get_hf_manager()
    return mgr.get_info()


@app.post("/api/hf/model/freeze")
async def hf_freeze_layers(body: dict[str, Any]):
    mgr = _get_hf_manager()
    return mgr.freeze_layers(body.get("patterns", []))


@app.post("/api/hf/model/unfreeze")
async def hf_unfreeze_layers(body: dict[str, Any]):
    mgr = _get_hf_manager()
    return mgr.unfreeze_layers(body.get("patterns", []))


@app.post("/api/hf/model/lora")
async def hf_apply_lora(body: dict[str, Any]):
    mgr = _get_hf_manager()
    result = mgr.apply_lora(
        target_modules=body.get("target_modules"),
        r=body.get("r", 8),
        lora_alpha=body.get("lora_alpha", 16),
        lora_dropout=body.get("lora_dropout", 0.1),
        task_type=body.get("task_type"),
    )
    await broadcast("hf_lora_applied", result)
    return result


@app.get("/api/hf/model/lora_targets")
async def hf_lora_targets():
    mgr = _get_hf_manager()
    return {"targets": mgr._suggest_lora_targets()}


@app.post("/api/hf/datasets/load")
async def hf_load_dataset(body: dict[str, Any]):
    dm = _get_hf_data()
    result = dm.load_hf_dataset(
        dataset_id=body["dataset_id"],
        split=body.get("split"),
        subset=body.get("subset"),
    )
    return result


@app.post("/api/hf/datasets/local")
async def hf_load_local_dataset(body: dict[str, Any]):
    dm = _get_hf_data()
    fmt = body.get("format", "csv")
    path = body["path"]
    if fmt == "csv":
        return dm.load_csv(path, body.get("text_col"), body.get("label_col"))
    elif fmt == "json":
        return dm.load_json(path)
    elif fmt == "imagefolder":
        return dm.load_local_images(path)
    elif fmt == "audiofolder":
        return dm.load_local_audio(path)
    return {"status": "error", "message": f"Unknown format: {fmt}"}


@app.post("/api/hf/datasets/preprocess")
async def hf_preprocess_dataset(body: dict[str, Any]):
    dm = _get_hf_data()
    mgr = _get_hf_manager()
    return dm.set_preprocessing(
        tokenizer=mgr.get_tokenizer(),
        processor=mgr.get_processor(),
        max_length=body.get("max_length", 128),
        text_column=body.get("text_column", "text"),
        label_column=body.get("label_column", "label"),
    )


@app.post("/api/hf/datasets/prepare")
async def hf_prepare_dataloaders(body: dict[str, Any]):
    dm = _get_hf_data()
    loaders = dm.get_dataloaders(
        batch_size=engine.config.get("batch_size", 32),
        val_split=body.get("val_split", 0.1),
    )
    return {"status": "ready", "splits": list(loaders.keys())}


@app.get("/api/hf/datasets/preview")
async def hf_preview_dataset(n: int = 5):
    dm = _get_hf_data()
    return {"samples": dm.preview(n)}


@app.get("/api/hf/datasets/info")
async def hf_dataset_info():
    dm = _get_hf_data()
    return dm.get_info()


@app.get("/api/hf/datasets/columns")
async def hf_dataset_columns():
    """Get column names and auto-detected column mapping suggestions."""
    dm = _get_hf_data()
    return dm.suggest_columns()


@app.post("/api/hf/config")
async def hf_update_config(body: dict[str, Any]):
    engine.hf_config.update(body)
    return {"hf_config": engine.hf_config}


# --- HF Architecture Surgery ---

@app.post("/api/hf/model/insert")
async def hf_insert_module(body: dict[str, Any]):
    """Insert a module into the loaded HF model.

    Body: {parent_path, name, module_type, module_params}
    Example: {"parent_path": "encoder.layer", "name": "12",
              "module_type": "Linear", "module_params": {"in_features": 768, "out_features": 768}}
    """
    mgr = _get_hf_manager()
    from state_graph.core.registry import Registry
    mod_type = body.get("module_type", "Linear")
    mod_params = body.get("module_params", {})
    try:
        ModClass = Registry.get_layer(mod_type)
        module = ModClass(**mod_params)
    except Exception as e:
        return {"status": "error", "message": f"Failed to create module: {e}"}

    result = mgr.insert_module(body.get("parent_path", ""), body.get("name", "new"), module)
    if result.get("status") == "ok":
        await broadcast("hf_model_modified", result)
    return result


@app.post("/api/hf/model/remove")
async def hf_remove_module(body: dict[str, Any]):
    """Remove a module from the loaded HF model by path."""
    mgr = _get_hf_manager()
    result = mgr.remove_module(body.get("path", ""))
    if result.get("status") == "ok":
        await broadcast("hf_model_modified", result)
    return result


@app.post("/api/hf/model/replace")
async def hf_replace_module(body: dict[str, Any]):
    """Replace a module in the loaded HF model.

    Body: {path, module_type, module_params}
    """
    mgr = _get_hf_manager()
    from state_graph.core.registry import Registry
    mod_type = body.get("module_type", "Linear")
    mod_params = body.get("module_params", {})
    try:
        ModClass = Registry.get_layer(mod_type)
        module = ModClass(**mod_params)
    except Exception as e:
        return {"status": "error", "message": f"Failed to create module: {e}"}

    result = mgr.replace_module(body.get("path", ""), module)
    if result.get("status") == "ok":
        await broadcast("hf_model_modified", result)
    return result


@app.post("/api/hf/model/add_head")
async def hf_add_head(body: dict[str, Any]):
    """Add a new output head to the HF model."""
    mgr = _get_hf_manager()
    return mgr.add_head(
        name=body.get("name", "custom_head"),
        in_features=body.get("in_features", 768),
        out_features=body.get("out_features", 2),
    )


@app.get("/api/hf/model/module/{path:path}")
async def hf_get_module(path: str):
    """Get info about a specific module by path."""
    mgr = _get_hf_manager()
    return mgr.get_module_info(path)


# --- HF Training (any model) ---

@app.post("/api/hf/train")
async def hf_start_training(body: dict[str, Any]):
    """Start training any loaded HF model (transformers, timm, etc.)."""
    mgr = _get_hf_manager()
    dm = _get_hf_data()

    if mgr.model is None:
        return {"status": "error", "message": "No model loaded. Use /api/hf/load first."}

    if dm.dataset is None:
        return {"status": "error", "message": "No dataset loaded. Load a dataset first."}

    # Auto-preprocess if not yet done (check if format is set)
    ds = dm.dataset
    sample_ds = ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds
    if "input_ids" not in (sample_ds.column_names or []):
        tokenizer = mgr.get_tokenizer()
        if tokenizer:
            suggestions = dm.suggest_columns()
            text_col = body.get("text_column") or suggestions.get("text") or "text"
            label_col = body.get("label_column") or suggestions.get("label") or "label"
            dm.set_preprocessing(
                tokenizer=tokenizer,
                processor=mgr.get_processor(),
                max_length=body.get("max_length", 128),
                text_column=text_col,
                label_column=label_col,
            )

    loaders = dm.get_dataloaders(
        batch_size=body.get("batch_size", engine.config.get("batch_size", 16)),
        val_split=body.get("val_split", 0.1),
    )

    train_loader = loaders.get("train")
    val_loader = loaders.get("validation") or loaders.get("val")

    if train_loader is None:
        return {"status": "error", "message": "No training data. Load and prepare a dataset first."}

    def _broadcast(event, data):
        engine._emit_from_thread(event, data)

    result = mgr.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=body.get("epochs", 3),
        lr=body.get("learning_rate", 2e-5),
        weight_decay=body.get("weight_decay", 0.01),
        warmup_steps=body.get("warmup_steps", 0),
        max_grad_norm=body.get("max_grad_norm", 1.0),
        fp16=body.get("fp16", False),
        broadcast_fn=_broadcast,
    )

    return result


@app.post("/api/hf/train/stop")
async def hf_stop_training():
    """Stop HF model training."""
    mgr = _get_hf_manager()
    return mgr.stop_training()


@app.get("/api/hf/train/history")
async def hf_train_history():
    """Get training history."""
    mgr = _get_hf_manager()
    return {"history": mgr.get_train_history()}


# --- HF Inference ---

@app.post("/api/hf/inference")
async def hf_inference(body: dict[str, Any]):
    """Run inference on any loaded HF model."""
    mgr = _get_hf_manager()
    return mgr.inference(body)


@app.post("/api/hf/diffusion/generate")
async def hf_diffusion_generate(body: dict[str, Any]):
    """Generate image from text using a loaded diffusion model."""
    mgr = _get_hf_manager()
    return mgr.diffusion_generate(
        prompt=body.get("prompt", ""),
        num_inference_steps=body.get("num_inference_steps", 50),
        guidance_scale=body.get("guidance_scale", 7.5),
        height=body.get("height", 512),
        width=body.get("width", 512),
        seed=body.get("seed"),
    )


@app.post("/api/model_source")
async def set_model_source(body: dict[str, Any]):
    """Switch between 'graph' and 'hf' model source."""
    engine.model_source = body.get("source", "graph")
    return {"model_source": engine.model_source}


# --- Data Engineering / Pipelines ---

from state_graph.dataeng.connectors import CONNECTOR_REGISTRY, DataConnector
from state_graph.dataeng.pipeline import (
    PipelineManager, TRANSFORM_REGISTRY, compute_stats,
)

_pipe_mgr = PipelineManager()


@app.get("/api/dataeng/connectors")
async def de_list_connectors():
    result = {}
    for cid, cfg in CONNECTOR_REGISTRY.items():
        result[cid] = {"name": cfg["name"], "category": cfg["category"], "params": cfg["params"], "pip": cfg.get("pip")}
    return {"connectors": result}


@app.get("/api/dataeng/transforms")
async def de_list_transforms():
    return {"transforms": TRANSFORM_REGISTRY}


@app.post("/api/dataeng/test_connection")
async def de_test_connection(body: dict[str, Any]):
    conn = DataConnector(body["connector_type"], body["params"])
    return conn.test_connection()


@app.post("/api/dataeng/list_tables")
async def de_list_tables(body: dict[str, Any]):
    conn = DataConnector(body["connector_type"], body["params"])
    return conn.list_tables()


@app.post("/api/dataeng/preview")
async def de_preview_data(body: dict[str, Any]):
    conn = DataConnector(body["connector_type"], body["params"])
    return conn.preview(body.get("query"), body.get("limit", 50))


@app.post("/api/dataeng/stats")
async def de_data_stats(body: dict[str, Any]):
    """Compute stats on provided rows or from a connector."""
    if "rows" in body:
        return compute_stats(body["rows"])
    conn = DataConnector(body["connector_type"], body["params"])
    result = conn.load(body.get("query"), body.get("limit", 5000))
    if result["status"] != "ok":
        return result
    return compute_stats(result["rows"])


# Pipeline CRUD
@app.post("/api/dataeng/pipelines")
async def de_create_pipeline(body: dict[str, Any]):
    p = _pipe_mgr.create(body.get("name", ""))
    return {"status": "created", "pipeline": p.to_dict()}


@app.get("/api/dataeng/pipelines")
async def de_list_pipelines():
    return {"pipelines": _pipe_mgr.list_all()}


@app.get("/api/dataeng/pipelines/{pid}")
async def de_get_pipeline(pid: str):
    p = _pipe_mgr.get(pid)
    return p.to_dict() if p else {"status": "error", "message": "Not found"}


@app.delete("/api/dataeng/pipelines/{pid}")
async def de_delete_pipeline(pid: str):
    return _pipe_mgr.delete(pid)


# Pipeline sources
@app.post("/api/dataeng/pipelines/{pid}/sources")
async def de_add_source(pid: str, body: dict[str, Any]):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    sid = body.get("source_id", str(uuid.uuid4())[:6])
    p.add_source(sid, body["connector_type"], body["params"])
    return {"status": "added", "source_id": sid}


@app.post("/api/dataeng/pipelines/{pid}/sources/{sid}/load")
async def de_load_source(pid: str, sid: str, body: dict[str, Any] = {}):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.load_source(sid, body.get("query"), body.get("limit", 10000))


# Pipeline transforms
@app.post("/api/dataeng/pipelines/{pid}/transforms")
async def de_add_transform(pid: str, body: dict[str, Any]):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.add_transform(body["op"], body.get("params", {}))


@app.delete("/api/dataeng/pipelines/{pid}/transforms/{tid}")
async def de_remove_transform(pid: str, tid: str):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.remove_transform(tid)


@app.post("/api/dataeng/pipelines/{pid}/transforms/{tid}/toggle")
async def de_toggle_transform(pid: str, tid: str):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.toggle_transform(tid)


# Run pipeline
@app.post("/api/dataeng/pipelines/{pid}/run")
async def de_run_pipeline(pid: str, body: dict[str, Any] = {}):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.run(body.get("primary_source_id"))


@app.get("/api/dataeng/pipelines/{pid}/stats")
async def de_pipeline_stats(pid: str):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.get_stats()


@app.get("/api/dataeng/pipelines/{pid}/result")
async def de_pipeline_result(pid: str, offset: int = 0, limit: int = 100):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    rows = p.get_result()
    return {"rows": rows[offset:offset + limit], "total": len(rows)}


# Sink
@app.post("/api/dataeng/pipelines/{pid}/sink")
async def de_sink(pid: str, body: dict[str, Any]):
    p = _pipe_mgr.get(pid)
    if not p:
        return {"status": "error", "message": "Pipeline not found"}
    return p.sink(body["connector_type"], body["params"], body.get("target"))


# --- Multi-User Collaboration ---

from state_graph.server.collaboration import CollaborationManager

_collab = CollaborationManager()


@app.post("/api/collab/users")
async def collab_create_user(body: dict[str, Any]):
    u = _collab.create_user(body.get("name", "Anonymous"))
    return {"user_id": u.id, "name": u.name, "color": u.color}


@app.post("/api/collab/rooms")
async def collab_create_room(body: dict[str, Any]):
    r = _collab.create_room(body.get("name", "Room"), body.get("project_id"))
    return {"status": "created", "room": r.get_state()}


@app.get("/api/collab/rooms")
async def collab_list_rooms():
    return {"rooms": _collab.list_rooms()}


@app.post("/api/collab/rooms/{room_id}/join")
async def collab_join(room_id: str, body: dict[str, Any]):
    result = _collab.join_room(body["user_id"], room_id)
    if result.get("status") == "joined":
        await broadcast("collab_user_joined", result)
    return result


@app.post("/api/collab/rooms/{room_id}/leave")
async def collab_leave(room_id: str, body: dict[str, Any]):
    result = _collab.leave_room(body["user_id"])
    await broadcast("collab_user_left", {"user_id": body["user_id"], "room_id": room_id})
    return result


@app.post("/api/collab/cursor")
async def collab_cursor(body: dict[str, Any]):
    room = _collab.get_user_room(body["user_id"])
    if room:
        result = room.update_cursor(body["user_id"], body.get("cursor", {}))
        await broadcast("collab_cursors", result)
        return result
    return {"status": "error"}


@app.post("/api/collab/chat")
async def collab_chat(body: dict[str, Any]):
    room = _collab.get_user_room(body["user_id"])
    if room:
        result = room.send_chat(body["user_id"], body["message"])
        if result.get("chat"):
            await broadcast("collab_chat", result["chat"])
        return result
    return {"status": "error"}


@app.post("/api/collab/lock")
async def collab_lock(body: dict[str, Any]):
    room = _collab.get_user_room(body["user_id"])
    if room:
        return room.lock_file(body["user_id"], body["file"])
    return {"status": "error"}


@app.post("/api/collab/unlock")
async def collab_unlock(body: dict[str, Any]):
    room = _collab.get_user_room(body["user_id"])
    if room:
        return room.unlock_file(body["user_id"], body["file"])
    return {"status": "error"}


@app.get("/api/collab/rooms/{room_id}")
async def collab_room_state(room_id: str):
    r = _collab.rooms.get(room_id)
    return r.get_state() if r else {"status": "error"}


@app.get("/api/collab/rooms/{room_id}/chat")
async def collab_chat_history(room_id: str):
    r = _collab.rooms.get(room_id)
    return {"messages": r.chat_history[-100:]} if r else {"status": "error"}


# --- Hardware-in-the-Loop ---

from state_graph.robotics.hardware import HardwareBridge

_hardware = HardwareBridge()


@app.get("/api/hardware/ports")
async def hw_list_ports():
    return {"ports": _hardware.list_ports()}


@app.post("/api/hardware/connect")
async def hw_connect(body: dict[str, Any]):
    result = _hardware.connect(body["port"], body.get("baud", 115200))
    if result["status"] == "connected":
        _hardware.set_broadcast(broadcast, engine._loop)
    return result


@app.post("/api/hardware/disconnect")
async def hw_disconnect():
    return _hardware.disconnect()


@app.post("/api/hardware/command")
async def hw_send_command(body: dict[str, Any]):
    return _hardware.send_command(body)


@app.post("/api/hardware/joints")
async def hw_send_joints(body: dict[str, Any]):
    return _hardware.send_joint_angles(body.get("angles", {}))


@app.post("/api/hardware/motors")
async def hw_send_motors(body: dict[str, Any]):
    return _hardware.send_motor_speeds(body.get("speeds", {}))


@app.post("/api/hardware/raw")
async def hw_send_raw(body: dict[str, Any]):
    return _hardware.send_raw(body.get("data", ""))


@app.get("/api/hardware/sensors")
async def hw_get_sensors():
    return _hardware.get_sensor_data()


@app.get("/api/hardware/info")
async def hw_info():
    return _hardware.get_info()


@app.post("/api/hardware/firmware/generate")
async def hw_generate_firmware(body: dict[str, Any]):
    r = _robot_mgr.get(body.get("robot_id", ""))
    if not r:
        return {"status": "error", "message": "Robot not found"}
    code = _hardware.generate_firmware({"components": r.components})
    return {"status": "ok", "code": code}


@app.post("/api/hardware/firmware/upload")
async def hw_upload_firmware(body: dict[str, Any]):
    return _hardware.upload_firmware(body["board"], body["sketch_path"])


# --- Server-Side Physics (MuJoCo) ---

from state_graph.robotics.physics_server import PhysicsServer

_physics = PhysicsServer()


@app.post("/api/physics/load")
async def phys_load(body: dict[str, Any]):
    _physics.set_broadcast(broadcast, engine._loop)
    return _physics.load_scene(
        bodies=body.get("bodies", []),
        joints=body.get("joints", []),
        use_mujoco=body.get("use_mujoco", True),
    )


@app.post("/api/physics/start")
async def phys_start():
    return _physics.start()


@app.post("/api/physics/stop")
async def phys_stop():
    return _physics.stop()


@app.post("/api/physics/step")
async def phys_step():
    return _physics.step_once()


@app.post("/api/physics/force")
async def phys_apply_force(body: dict[str, Any]):
    return _physics.apply_force(body["body_index"], body["force"])


@app.post("/api/physics/joint")
async def phys_set_joint(body: dict[str, Any]):
    return _physics.set_joint_target(body["joint_index"], body["target"])


@app.get("/api/physics/info")
async def phys_info():
    return _physics.get_info()


# --- Robotics Simulation ---

from state_graph.robotics.simulator import (
    COMPONENT_CATALOG, ROBOT_TEMPLATES, RobotManager, solve_circuit,
)

_robot_mgr = RobotManager()


@app.get("/api/robotics/components")
async def rob_list_components():
    result = {}
    for cid, info in COMPONENT_CATALOG.items():
        cat = info["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append({"id": cid, "name": info["name"], "subcategory": info.get("subcategory", ""),
            "specs": info["specs"], "dimensions": info["dimensions"], "color": info["color"]})
    return {"components": result}


@app.get("/api/robotics/templates")
async def rob_list_templates():
    return {"templates": {k: {"name": v["name"], "description": v["description"],
        "component_count": len(v["components"])} for k, v in ROBOT_TEMPLATES.items()}}


@app.post("/api/robotics/robots")
async def rob_create(body: dict[str, Any]):
    r = _robot_mgr.create(body.get("name", "Robot"), body.get("template"))
    return {"status": "created", "robot": r.to_dict()}


@app.get("/api/robotics/robots")
async def rob_list():
    return {"robots": _robot_mgr.list_all()}


@app.get("/api/robotics/robots/{rid}")
async def rob_get(rid: str):
    r = _robot_mgr.get(rid)
    return r.to_dict() if r else {"status": "error"}


@app.delete("/api/robotics/robots/{rid}")
async def rob_delete(rid: str):
    return _robot_mgr.delete(rid)


@app.post("/api/robotics/robots/{rid}/components")
async def rob_add_component(rid: str, body: dict[str, Any]):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error", "message": "Robot not found"}
    return r.add_component(body["type"], body.get("position", [0, 0, 0]),
        body.get("role", ""), body.get("duty_cycle", 0.5))


@app.delete("/api/robotics/robots/{rid}/components/{cid}")
async def rob_remove_component(rid: str, cid: str):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.remove_component(cid)


@app.put("/api/robotics/robots/{rid}/components/{cid}")
async def rob_update_component(rid: str, cid: str, body: dict[str, Any]):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.update_component(cid, body)


@app.get("/api/robotics/robots/{rid}/circuit")
async def rob_analyze_circuit(rid: str):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.analyze_circuit()


@app.get("/api/robotics/robots/{rid}/scene")
async def rob_get_scene(rid: str):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.get_3d_scene()


@app.post("/api/robotics/robots/{rid}/joints")
async def rob_set_joints(rid: str, body: dict[str, Any]):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.set_all_joints(body.get("angles", {}))


@app.post("/api/robotics/robots/{rid}/joint/{cid}")
async def rob_set_joint(rid: str, cid: str, body: dict[str, Any]):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    result = r.set_joint_angle(cid, body.get("angle", 0))
    # Broadcast for real-time sync
    await broadcast("robot_joint_update", {"robot_id": rid, "comp_id": cid, "angle": body.get("angle", 0)})
    return result


@app.get("/api/robotics/robots/{rid}/joints")
async def rob_get_joints(rid: str):
    r = _robot_mgr.get(rid)
    if not r:
        return {"status": "error"}
    return r.get_joint_states()


# --- ROS2 Bridge ---

from state_graph.robotics.ros_bridge import ROS2Bridge, ROS2_MSG_TYPES, ROS2_PACKAGES

_ros2 = ROS2Bridge()


@app.get("/api/ros2/check")
async def ros2_check():
    return _ros2.check_ros2()


@app.get("/api/ros2/msg_types")
async def ros2_msg_types():
    return {"msg_types": ROS2_MSG_TYPES}


@app.get("/api/ros2/packages")
async def ros2_packages():
    return {"packages": ROS2_PACKAGES}


@app.post("/api/ros2/init")
async def ros2_init(body: dict[str, Any] = {}):
    _ros2.set_broadcast(broadcast, engine._loop)
    return _ros2.init_node(body.get("node_name", "stategraph_bridge"))


@app.post("/api/ros2/shutdown")
async def ros2_shutdown():
    return _ros2.shutdown()


@app.get("/api/ros2/topics")
async def ros2_topics():
    return _ros2.list_topics()


@app.get("/api/ros2/nodes")
async def ros2_nodes():
    return _ros2.list_nodes()


@app.get("/api/ros2/services")
async def ros2_services():
    return _ros2.list_services()


@app.post("/api/ros2/publish")
async def ros2_publish(body: dict[str, Any]):
    return _ros2.publish(body["topic"], body["msg_type"], body["data"])


@app.post("/api/ros2/cmd_vel")
async def ros2_cmd_vel(body: dict[str, Any]):
    return _ros2.publish_cmd_vel(body.get("linear_x", 0), body.get("angular_z", 0), body.get("topic", "/cmd_vel"))


@app.post("/api/ros2/joint_state")
async def ros2_joint_state(body: dict[str, Any]):
    return _ros2.publish_joint_state(body["names"], body["positions"], body.get("topic", "/joint_commands"))


@app.post("/api/ros2/echo")
async def ros2_echo(body: dict[str, Any]):
    return _ros2.echo_topic(body["topic"], body.get("count", 1), body.get("timeout", 5))


@app.post("/api/ros2/topic_info")
async def ros2_topic_info(body: dict[str, Any]):
    return _ros2.get_topic_info(body["topic"])


@app.post("/api/ros2/urdf")
async def ros2_generate_urdf(body: dict[str, Any]):
    r = _robot_mgr.get(body.get("robot_id", ""))
    if not r:
        return {"status": "error", "message": "Robot not found"}
    urdf = _ros2.generate_urdf({"name": r.name, "components": r.components})
    return {"status": "ok", "urdf": urdf}


@app.post("/api/ros2/launch")
async def ros2_generate_launch(body: dict[str, Any]):
    r = _robot_mgr.get(body.get("robot_id", ""))
    if not r:
        return {"status": "error", "message": "Robot not found"}
    launch = _ros2.generate_launch_file({"name": r.name}, body.get("packages", []))
    return {"status": "ok", "launch": launch}


@app.post("/api/ros2/package")
async def ros2_generate_package(body: dict[str, Any]):
    r = _robot_mgr.get(body.get("robot_id", ""))
    if not r:
        return {"status": "error", "message": "Robot not found"}
    return _ros2.generate_package({"name": r.name, "components": r.components}, body.get("output_dir", "./sg_ros2_ws/src"))


@app.get("/api/ros2/info")
async def ros2_info():
    return _ros2.get_info()


# --- Reinforcement Learning ---

_rl_engine = None

def _get_rl():
    global _rl_engine
    if _rl_engine is None:
        try:
            from state_graph.rl.engine import RLEngine
        except ImportError:
            raise RuntimeError("RL dependencies not installed. Run: pip install -e '.[rl]'")
        _rl_engine = RLEngine()
        if engine._loop:
            _rl_engine.set_broadcast(broadcast, engine._loop)
    return _rl_engine


@app.get("/api/rl/envs")
async def rl_list_envs():
    from state_graph.rl.engine import BUILTIN_ENVS
    return {"environments": BUILTIN_ENVS}


@app.get("/api/rl/algorithms")
async def rl_list_algorithms():
    from state_graph.rl.engine import ALGORITHMS
    return {"algorithms": ALGORITHMS}


@app.post("/api/rl/env")
async def rl_create_env(body: dict[str, Any]):
    rl = _get_rl()
    return rl.create_env(body["env_id"], body.get("params"))


@app.post("/api/rl/agent")
async def rl_create_agent(body: dict[str, Any]):
    rl = _get_rl()
    return rl.create_agent(body["algorithm"], body.get("params"))


@app.post("/api/rl/train")
async def rl_train(body: dict[str, Any]):
    rl = _get_rl()
    result = rl.start_training(
        total_timesteps=body.get("total_timesteps", 100000),
        eval_freq=body.get("eval_freq", 1000),
    )
    await broadcast("rl_training_status", result)
    return result


@app.post("/api/rl/stop")
async def rl_stop():
    rl = _get_rl()
    result = rl.stop_training()
    await broadcast("rl_training_status", result)
    return result


@app.post("/api/rl/episode")
async def rl_run_episode():
    rl = _get_rl()
    return rl.run_episode()


@app.get("/api/rl/info")
async def rl_info():
    rl = _get_rl()
    return rl.get_info()


@app.get("/api/rl/history")
async def rl_history():
    rl = _get_rl()
    return {"history": rl.get_history()}


@app.post("/api/rl/save")
async def rl_save(body: dict[str, Any]):
    rl = _get_rl()
    return rl.save_model(body.get("path", "./sg_outputs/rl_model"))


@app.post("/api/rl/load")
async def rl_load(body: dict[str, Any]):
    rl = _get_rl()
    return rl.load_model(body["path"], body["algorithm"])


# --- Advanced: Distributed, AutoML, Federated, Video, Inference, Cloud, Embeddings ---

from state_graph.advanced.distributed import (
    STRATEGIES as DIST_STRATEGIES, detect_gpus,
    generate_deepspeed_config, generate_accelerate_config,
    generate_launch_command, generate_training_script as gen_dist_script,
)
from state_graph.advanced.automl import NASEngine
from state_graph.advanced.federated import FL_STRATEGIES, generate_fl_server, generate_fl_client
from state_graph.advanced.video_training import VIDEO_MODELS, generate_video_training_script
from state_graph.advanced.inference_opt import (
    OPTIMIZATION_METHODS, generate_optimization_script,
    benchmark_model as bench_model, quantize_dynamic,
)
from state_graph.advanced.cloud import (
    CLOUD_PROVIDERS, generate_sagemaker_script, generate_modal_script,
    generate_runpod_script, generate_vertex_ai_script, generate_docker_compose,
)
from state_graph.advanced.embeddings import (
    EMBEDDING_MODELS, generate_embedding_training_script,
)

_nas = NASEngine()


# Distributed
@app.get("/api/advanced/distributed/strategies")
async def adv_dist_strategies():
    return {"strategies": DIST_STRATEGIES}

@app.get("/api/advanced/distributed/gpus")
async def adv_dist_gpus():
    return detect_gpus()

@app.post("/api/advanced/distributed/config")
async def adv_dist_config(body: dict[str, Any]):
    strategy = body.get("strategy", "accelerate")
    params = body.get("params", {})
    if "deepspeed" in strategy:
        stage = 3 if "zero3" in strategy else 2
        return {"config": generate_deepspeed_config(stage, params)}
    return {"config": generate_accelerate_config(params)}

@app.post("/api/advanced/distributed/script")
async def adv_dist_script(body: dict[str, Any]):
    code = gen_dist_script(body.get("strategy", "accelerate"), body.get("model_code", ""), body.get("params"))
    return {"code": code, "launch": generate_launch_command(body.get("strategy", "accelerate"), "train.py", body.get("num_gpus"))}


# AutoML / NAS
@app.post("/api/advanced/nas/search")
async def adv_nas_search(body: dict[str, Any]):
    _nas.set_broadcast(broadcast, engine._loop)
    return _nas.search(
        input_dim=body.get("input_dim", 2),
        output_dim=body.get("output_dim", 2),
        strategy=body.get("strategy", "random"),
        n_trials=body.get("n_trials", 20),
        epochs_per_trial=body.get("epochs_per_trial", 5),
        task=body.get("task", "classification"),
    )

@app.post("/api/advanced/nas/stop")
async def adv_nas_stop():
    return _nas.stop()

@app.get("/api/advanced/nas/results")
async def adv_nas_results():
    return _nas.get_results()

@app.post("/api/advanced/nas/apply")
async def adv_nas_apply():
    result = _nas.apply_best()
    if result.get("architecture"):
        # Clear current graph and import best architecture
        for nid in list(engine.graph.nodes.keys()):
            engine.graph.remove_layer(nid)
        for node in result["architecture"]:
            engine.graph.add_layer(node["layer_type"], node.get("params", {}), node.get("activation"), node.get("position"))
        graph_data = engine.graph.to_dict()
        await broadcast("graph_updated", graph_data)
    return result


# Federated Learning
@app.get("/api/advanced/federated/strategies")
async def adv_fl_strategies():
    return {"strategies": FL_STRATEGIES}

@app.post("/api/advanced/federated/server")
async def adv_fl_server(body: dict[str, Any]):
    return {"code": generate_fl_server(body.get("strategy", "FedAvg"), body.get("num_rounds", 10), body.get("min_clients", 2))}

@app.post("/api/advanced/federated/client")
async def adv_fl_client(body: dict[str, Any]):
    return {"code": generate_fl_client(body.get("model_code", ""))}


# Video Training
@app.get("/api/advanced/video/models")
async def adv_video_models():
    return {"models": VIDEO_MODELS}

@app.post("/api/advanced/video/script")
async def adv_video_script(body: dict[str, Any]):
    return {"code": generate_video_training_script(body["model"], body.get("dataset_path", ""), body.get("params"))}


# Inference Optimization
@app.get("/api/advanced/inference/methods")
async def adv_inf_methods():
    return {"methods": OPTIMIZATION_METHODS}

@app.post("/api/advanced/inference/benchmark")
async def adv_inf_benchmark(body: dict[str, Any]):
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}
    import torch
    try:
        dummy = torch.randn(1, *engine.data_manager.input_shape).to(engine.device)
    except Exception:
        dummy = torch.randn(1, body.get("input_dim", 2)).to(engine.device)
    return bench_model(engine.model, dummy, body.get("n_runs", 100))

@app.post("/api/advanced/inference/quantize")
async def adv_inf_quantize(body: dict[str, Any]):
    if engine.model is None:
        return {"status": "error", "message": "No model loaded"}
    return quantize_dynamic(engine.model, body.get("output_path", "./sg_outputs/quantized.pt"))

@app.post("/api/advanced/inference/script")
async def adv_inf_script(body: dict[str, Any]):
    return {"code": generate_optimization_script(body["method"], body.get("model_path", "./model.pt"), body.get("params"))}


# Cloud
@app.get("/api/advanced/cloud/providers")
async def adv_cloud_providers():
    return {"providers": CLOUD_PROVIDERS}

@app.post("/api/advanced/cloud/script")
async def adv_cloud_script(body: dict[str, Any]):
    provider = body.get("provider", "sagemaker")
    params = body.get("params", {})
    generators = {
        "sagemaker": lambda: generate_sagemaker_script(body.get("script", "train.py"), params),
        "vertex_ai": lambda: generate_vertex_ai_script(params),
        "modal": lambda: generate_modal_script(params),
        "runpod": lambda: generate_runpod_script(params),
    }
    gen = generators.get(provider)
    return {"code": gen() if gen else f"# {provider} script generation not available"}

@app.post("/api/advanced/cloud/docker_compose")
async def adv_cloud_docker(body: dict[str, Any]):
    return {"code": generate_docker_compose(body.get("params"))}


# Embeddings
@app.get("/api/advanced/embeddings/models")
async def adv_emb_models():
    return {"models": EMBEDDING_MODELS}

@app.post("/api/advanced/embeddings/script")
async def adv_emb_script(body: dict[str, Any]):
    return {"code": generate_embedding_training_script(
        body.get("base_model", "sentence-transformers/all-MiniLM-L6-v2"),
        body.get("task", "similarity"),
        body.get("dataset", ""),
        body.get("params"),
    )}


# --- Evaluation, Deploy, Libraries, Hyperparam Search ---

from state_graph.core.evaluator import (
    evaluate_classification, evaluate_regression, evaluate_text_generation,
    grid_search, random_search,
)
from state_graph.core.deploy import (
    export_onnx, export_torchscript, generate_inference_server,
    generate_dockerfile, generate_gradio_app,
)
from state_graph.core.libraries import LIBRARY_CATALOG, get_code_template


@app.post("/api/eval/classification")
async def eval_classification(body: dict[str, Any]):
    return evaluate_classification(body["y_true"], body["y_pred"], body.get("labels"))


@app.post("/api/eval/regression")
async def eval_regression(body: dict[str, Any]):
    return evaluate_regression(body["y_true"], body["y_pred"])


@app.post("/api/eval/generation")
async def eval_generation(body: dict[str, Any]):
    return evaluate_text_generation(body["references"], body["predictions"])


@app.post("/api/eval/auto")
async def eval_auto():
    """Auto-evaluate the current trained model on validation set."""
    if engine.model is None:
        return {"status": "error", "message": "No model trained"}
    if engine._val_loader is None:
        return {"status": "error", "message": "No validation data"}

    import torch
    y_true, y_pred = [], []
    engine.model.eval()
    with torch.no_grad():
        for batch in engine._val_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(engine.device), batch[1]
                out = engine.model(x)
            else:
                batch = {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = engine.model(**batch)
                out = out.logits if hasattr(out, "logits") else out
                y = batch.get("labels", batch.get("label", torch.zeros(1)))

            if out.dim() > 1 and out.shape[-1] > 1:
                preds = out.argmax(dim=-1).cpu().tolist()
            else:
                preds = out.cpu().squeeze().tolist()

            y_true.extend(y.cpu().tolist() if isinstance(y, torch.Tensor) else y)
            y_pred.extend(preds if isinstance(preds, list) else [preds])

    # Detect task type
    if len(set(y_true)) <= 50:
        return evaluate_classification(y_true, y_pred)
    else:
        return evaluate_regression(y_true, y_pred)


@app.post("/api/deploy/onnx")
async def deploy_onnx(body: dict[str, Any]):
    if engine.model is None:
        return {"status": "error", "message": "No model"}
    import torch
    # Create dummy input based on model's first layer
    try:
        dummy = torch.randn(1, *engine.data_manager.input_shape).to(engine.device)
    except Exception:
        dummy = torch.randn(1, 2).to(engine.device)
    path = body.get("path", "./sg_outputs/model.onnx")
    return export_onnx(engine.model, dummy, path)


@app.post("/api/deploy/torchscript")
async def deploy_torchscript(body: dict[str, Any]):
    if engine.model is None:
        return {"status": "error", "message": "No model"}
    import torch
    try:
        dummy = torch.randn(1, *engine.data_manager.input_shape).to(engine.device)
    except Exception:
        dummy = torch.randn(1, 2).to(engine.device)
    path = body.get("path", "./sg_outputs/model.pt")
    return export_torchscript(engine.model, dummy, path, body.get("method", "trace"))


@app.post("/api/deploy/server")
async def deploy_generate_server(body: dict[str, Any]):
    code = generate_inference_server(
        body.get("model_path", "./model.onnx"),
        body.get("model_type", "onnx"),
        body.get("port", 8080),
    )
    return {"status": "ok", "code": code}


@app.post("/api/deploy/dockerfile")
async def deploy_dockerfile(body: dict[str, Any]):
    code = generate_dockerfile(
        body.get("model_path", "./model.onnx"),
        body.get("model_type", "onnx"),
    )
    return {"status": "ok", "dockerfile": code}


@app.post("/api/deploy/gradio")
async def deploy_gradio(body: dict[str, Any]):
    code = generate_gradio_app(
        body.get("model_path", "./model.pt"),
        body.get("model_type", "pytorch"),
    )
    return {"status": "ok", "code": code}


@app.get("/api/libraries")
async def list_libraries():
    result = {}
    for lid, lib in LIBRARY_CATALOG.items():
        cat = lib["category"]
        if cat not in result:
            result[cat] = []
        result[cat].append({"id": lid, **lib})
    return {"libraries": result, "total": len(LIBRARY_CATALOG)}


@app.get("/api/libraries/{library_id}/template")
async def get_library_template(library_id: str, task: str = ""):
    if library_id not in LIBRARY_CATALOG:
        return {"status": "error", "message": f"Unknown library: {library_id}"}
    code = get_code_template(library_id, task)
    return {"status": "ok", "code": code, "library": library_id, "task": task}


@app.post("/api/hyperparam/grid")
async def hyperparam_grid(body: dict[str, Any]):
    configs = grid_search(body["param_grid"])
    return {"configs": configs, "total": len(configs)}


@app.post("/api/hyperparam/random")
async def hyperparam_random(body: dict[str, Any]):
    configs = random_search(body["param_ranges"], body.get("n_trials", 20))
    return {"configs": configs, "total": len(configs)}


# --- Unsloth Integration ---

_unsloth_mgr = None

def _get_unsloth():
    global _unsloth_mgr
    if _unsloth_mgr is None:
        try:
            from state_graph.hf.unsloth import UnslothManager
        except ImportError:
            raise RuntimeError("Unsloth not installed. Run: pip install -e '.[unsloth]'")
        _unsloth_mgr = UnslothManager()
        if engine._loop:
            _unsloth_mgr.set_broadcast(broadcast, engine._loop)
    return _unsloth_mgr


@app.get("/api/unsloth/models")
async def unsloth_list_models():
    from state_graph.hf.unsloth import UNSLOTH_MODELS
    return {"models": UNSLOTH_MODELS}


@app.get("/api/unsloth/methods")
async def unsloth_list_methods():
    from state_graph.hf.unsloth import TRAINING_METHODS
    return {"methods": TRAINING_METHODS}


@app.get("/api/unsloth/templates")
async def unsloth_list_templates():
    from state_graph.hf.unsloth import CHAT_TEMPLATES
    return {"templates": CHAT_TEMPLATES}


@app.get("/api/unsloth/export_formats")
async def unsloth_export_formats():
    from state_graph.hf.unsloth import EXPORT_FORMATS
    return {"formats": EXPORT_FORMATS}


@app.post("/api/unsloth/load")
async def unsloth_load(body: dict[str, Any]):
    mgr = _get_unsloth()
    result = mgr.load_model(
        model_id=body["model_id"],
        max_seq_length=body.get("max_seq_length", 2048),
        load_in_4bit=body.get("load_in_4bit", True),
        dtype=body.get("dtype"),
    )
    await broadcast("unsloth_model_loaded", result)
    return result


@app.post("/api/unsloth/lora")
async def unsloth_lora(body: dict[str, Any]):
    mgr = _get_unsloth()
    result = mgr.apply_lora(
        r=body.get("r", 16),
        lora_alpha=body.get("lora_alpha", 16),
        lora_dropout=body.get("lora_dropout", 0),
        target_modules=body.get("target_modules"),
        use_rslora=body.get("use_rslora", False),
        use_gradient_checkpointing=body.get("gradient_checkpointing", "unsloth"),
    )
    await broadcast("unsloth_lora_applied", result)
    return result


@app.post("/api/unsloth/chat_template")
async def unsloth_chat_template(body: dict[str, Any]):
    mgr = _get_unsloth()
    return mgr.set_chat_template(body["template"])


@app.post("/api/unsloth/dataset")
async def unsloth_prepare_dataset(body: dict[str, Any]):
    mgr = _get_unsloth()
    return mgr.prepare_dataset(
        dataset_source=body.get("source", "hub"),
        dataset_id=body.get("dataset_id", ""),
        dataset_path=body.get("dataset_path", ""),
        text_column=body.get("text_column", "text"),
        formatting=body.get("formatting", "instruction"),
        max_seq_length=body.get("max_seq_length", 2048),
        split=body.get("split", "train"),
    )


@app.post("/api/unsloth/train")
async def unsloth_train(body: dict[str, Any]):
    mgr = _get_unsloth()
    method = body.get("method", "sft")
    config = body.get("config", {})
    result = mgr.start_training(method, config)
    await broadcast("unsloth_training_status", result)
    return result


@app.post("/api/unsloth/stop")
async def unsloth_stop():
    mgr = _get_unsloth()
    result = mgr.stop_training()
    await broadcast("unsloth_training_status", result)
    return result


@app.get("/api/unsloth/info")
async def unsloth_info():
    mgr = _get_unsloth()
    return mgr.get_info()


@app.get("/api/unsloth/history")
async def unsloth_history():
    mgr = _get_unsloth()
    return {"history": mgr.get_train_history()}


@app.post("/api/unsloth/save")
async def unsloth_save(body: dict[str, Any]):
    mgr = _get_unsloth()
    return mgr.save_model(
        format=body.get("format", "lora"),
        path=body.get("path", "./sg_outputs/unsloth_model"),
        **{k: v for k, v in body.items() if k not in ("format", "path")},
    )


@app.post("/api/unsloth/inference")
async def unsloth_inference(body: dict[str, Any]):
    mgr = _get_unsloth()
    return mgr.run_inference(
        prompt=body["prompt"],
        max_tokens=body.get("max_tokens", 256),
        temperature=body.get("temperature", 0.7),
    )


# --- Workspace / IDE ---

from state_graph.workspace.manager import WorkspaceManager
from state_graph.workspace.executor import CodeExecutor

_workspace = WorkspaceManager()
_executor = CodeExecutor()

PROJECT_TEMPLATES = [
    {"id": "empty", "name": "Empty Project", "description": "Blank project with main.py"},
    {"id": "llm_finetune", "name": "LLM Fine-Tuning", "description": "Unsloth + LoRA training setup"},
    {"id": "vision", "name": "Vision Model", "description": "Image classification with torchvision"},
    {"id": "dataset", "name": "Dataset Creator", "description": "Build and export datasets"},
    {"id": "yolo", "name": "YOLO Detection", "description": "Object detection with YOLOv8"},
]


@app.get("/api/workspace/templates")
async def ws_templates():
    return {"templates": PROJECT_TEMPLATES}


@app.post("/api/workspace/projects")
async def ws_create_project(body: dict[str, Any]):
    p = _workspace.create(
        name=body["name"],
        description=body.get("description", ""),
        template=body.get("template", "empty"),
    )
    return {"status": "created", "project": p.to_dict()}


@app.get("/api/workspace/projects")
async def ws_list_projects():
    return {"projects": _workspace.list_all()}


@app.delete("/api/workspace/projects/{project_id}")
async def ws_delete_project(project_id: str):
    return _workspace.delete(project_id)


@app.get("/api/workspace/projects/{project_id}/tree")
async def ws_file_tree(project_id: str):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return {"tree": p.get_file_tree()}


@app.get("/api/workspace/projects/{project_id}/files")
async def ws_list_files(project_id: str, dir: str = ""):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return {"files": p.list_files(dir)}


@app.post("/api/workspace/projects/{project_id}/read")
async def ws_read_file(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.read_file(body["path"])


@app.post("/api/workspace/projects/{project_id}/write")
async def ws_write_file(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.write_file(body["path"], body["content"])


@app.post("/api/workspace/projects/{project_id}/create_file")
async def ws_create_file(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.create_file(body["path"], body.get("content", ""))


@app.post("/api/workspace/projects/{project_id}/create_dir")
async def ws_create_dir(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.create_dir(body["path"])


@app.post("/api/workspace/projects/{project_id}/delete")
async def ws_delete_file(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.delete_file(body["path"])


@app.post("/api/workspace/projects/{project_id}/rename")
async def ws_rename_file(project_id: str, body: dict[str, Any]):
    p = _workspace.get(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.rename_file(body["old_path"], body["new_path"])


@app.post("/api/workspace/run")
async def ws_run_code(body: dict[str, Any]):
    """Execute Python code or file."""
    project_id = body.get("project_id")
    cwd = None
    if project_id:
        p = _workspace.get(project_id)
        if p:
            cwd = str(p.path)

    if "code" in body:
        return _executor.run_code(body["code"], cwd=cwd, timeout=body.get("timeout", 300))
    elif "file" in body:
        file_path = body["file"]
        if cwd and not os.path.isabs(file_path):
            file_path = os.path.join(cwd, file_path)
        return _executor.run_file(file_path, cwd=cwd, timeout=body.get("timeout", 300))
    return {"status": "error", "message": "Provide 'code' or 'file'"}


@app.post("/api/workspace/stop")
async def ws_stop_execution():
    return _executor.stop()


@app.post("/api/workspace/pip")
async def ws_pip_install(body: dict[str, Any]):
    return _executor.install_package(body["package"])


# --- Paper Writer ---

from state_graph.workspace.paper_writer import generate_paper


@app.post("/api/paper/write")
async def paper_write(body: dict[str, Any]):
    """Generate a research paper from current architecture + results."""
    # Collect data from engine
    arch = engine.graph.to_dict()
    tc = engine.config
    eval_res = body.get("eval_results", {})

    # Get experiment comparisons
    exps = []
    for e in experiment_history:
        exps.append({
            "name": e.get("name", ""),
            "best_val_acc": e.get("best_val_acc"),
            "best_val_loss": e.get("best_val_loss"),
            "total_params": e.get("total_params"),
        })

    return generate_paper(
        title=body.get("title", "Novel Neural Network Architecture"),
        architecture=arch,
        training_config=tc,
        eval_results=eval_res,
        experiments=exps,
        abstract_hint=body.get("abstract", ""),
        key_contribution=body.get("contribution", ""),
        ai_assistant=_ai if _ai._configured else None,
    )


# --- Paper-to-Model ---

from state_graph.workspace.paper_to_model import fetch_paper_text, paper_to_architecture, apply_paper_config


@app.post("/api/paper/fetch")
async def paper_fetch(body: dict[str, Any]):
    """Fetch paper content from URL."""
    return fetch_paper_text(body["url"])


@app.post("/api/paper/analyze")
async def paper_analyze(body: dict[str, Any]):
    """Fetch paper + send to AI + get architecture."""
    url = body["url"]

    # Fetch paper
    paper = fetch_paper_text(url)
    if paper.get("status") == "error":
        return paper

    # Build text for AI
    paper_text = f"Title: {paper.get('title', '')}\n\n"
    paper_text += f"Abstract: {paper.get('abstract', '')}\n\n"
    paper_text += f"Full text:\n{paper.get('full_text', '')}"

    # Send to AI
    result = paper_to_architecture(paper_text, _ai)
    if result.get("status") != "ok":
        return result

    result["paper_info"] = {
        "title": paper.get("title", ""),
        "source": paper.get("source", ""),
        "url": paper.get("url", url),
        "authors": paper.get("authors", []),
    }
    return result


@app.post("/api/paper/build")
async def paper_build(body: dict[str, Any]):
    """Apply the AI-extracted architecture to the graph."""
    config = body.get("config")
    if not config:
        return {"status": "error", "message": "No config provided"}
    result = apply_paper_config(engine, config)
    if result.get("graph"):
        await broadcast("graph_updated", result["graph"])
    return result


@app.post("/api/paper/full")
async def paper_full_pipeline(body: dict[str, Any]):
    """One-shot: URL → fetch → AI analyze → build architecture."""
    url = body["url"]

    # Step 1: Fetch
    paper = fetch_paper_text(url)
    if paper.get("status") == "error":
        return {"status": "error", "step": "fetch", "message": paper.get("message", "Failed to fetch")}

    paper_text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}\n\n{paper.get('full_text', '')}"

    # Step 2: AI analyze
    result = paper_to_architecture(paper_text, _ai)
    if result.get("status") != "ok":
        return {"status": "error", "step": "analyze", "message": result.get("message", "AI failed"), "raw": result.get("raw_response")}

    config = result["config"]

    # Step 3: Build
    build_result = apply_paper_config(engine, config)
    if build_result.get("graph"):
        await broadcast("graph_updated", build_result["graph"])

    return {
        "status": "built",
        "paper_title": config.get("paper_title", paper.get("title", "")),
        "paper_summary": config.get("paper_summary", ""),
        "layers_added": build_result.get("layers_added", 0),
        "key_innovations": config.get("key_innovations", []),
        "custom_formulas": build_result.get("custom_formulas", []),
        "dataset_suggestion": config.get("dataset_suggestion", ""),
        "training_config": config.get("training_config", {}),
        "notes": config.get("notes", ""),
        "paper_url": url,
    }


# --- AI Assistant ---

from state_graph.workspace.ai_assistant import AIAssistant

_ai = AIAssistant()


@app.post("/api/ai/configure")
async def ai_configure(body: dict[str, Any]):
    return _ai.configure(
        provider=body.get("provider", "claude"),
        api_key=body.get("api_key", ""),
        model=body.get("model", ""),
        base_url=body.get("base_url", ""),
    )


@app.post("/api/ai/chat")
async def ai_chat(body: dict[str, Any]):
    # Optionally read file content from project
    file_content = body.get("file_content")
    file_path = body.get("file_path")
    if body.get("project_id") and file_path and not file_content:
        p = _workspace.get(body["project_id"])
        if p:
            r = p.read_file(file_path)
            if r.get("content"):
                file_content = r["content"]

    return _ai.chat(
        message=body["message"],
        file_content=file_content,
        file_path=file_path,
        error_output=body.get("error"),
        project_files=body.get("project_files"),
    )


@app.post("/api/ai/fix")
async def ai_fix_error(body: dict[str, Any]):
    return _ai.fix_error(
        code=body["code"],
        error=body["error"],
        file_path=body.get("file_path", ""),
    )


@app.post("/api/ai/improve")
async def ai_improve(body: dict[str, Any]):
    return _ai.improve_code(
        code=body["code"],
        instruction=body.get("instruction", ""),
        file_path=body.get("file_path", ""),
    )


@app.post("/api/ai/generate")
async def ai_generate(body: dict[str, Any]):
    return _ai.generate_code(
        description=body["description"],
        project_context=body.get("context", ""),
    )


@app.post("/api/ai/explain")
async def ai_explain(body: dict[str, Any]):
    return _ai.explain_code(body["code"])


@app.get("/api/ai/info")
async def ai_info():
    return _ai.get_info()


@app.post("/api/ai/clear")
async def ai_clear():
    return _ai.clear_history()


# --- File Upload for Annotation ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = file.filename.replace(" ", "_")
    dest = _UPLOAD_DIR / safe_name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 1
        while dest.exists():
            dest = _UPLOAD_DIR / f"{stem}_{i}{suffix}"
            i += 1
    content = await file.read()
    dest.write_bytes(content)
    return {"status": "uploaded", "filename": dest.name, "path": str(dest), "url": f"/uploads/{dest.name}", "size": len(content)}


@app.post("/api/upload/multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for file in files:
        safe_name = file.filename.replace(" ", "_")
        dest = _UPLOAD_DIR / safe_name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = _UPLOAD_DIR / f"{stem}_{i}{suffix}"
                i += 1
        content = await file.read()
        dest.write_bytes(content)
        results.append({"filename": dest.name, "path": str(dest), "url": f"/uploads/{dest.name}", "size": len(content)})
    return {"status": "uploaded", "files": results}


# --- Dataset Factory ---

from state_graph.datasets.creator import DatasetManager as DSManager, TEMPLATES as DS_TEMPLATES
from state_graph.datasets.sources import KaggleSource, URLSource, LocalSource
from state_graph.datasets import converters as ds_converters

_ds_manager = DSManager()


@app.get("/api/ds/templates")
async def ds_list_templates():
    return DSManager.list_templates_by_category()


@app.get("/api/ds/templates/all")
async def ds_list_all_templates():
    return DSManager.list_templates()


@app.post("/api/ds/projects")
async def ds_create_project(body: dict[str, Any]):
    return _ds_manager.create_project(
        name=body["name"],
        template_id=body["template_id"],
        labels=body.get("labels", []),
    )


@app.get("/api/ds/projects")
async def ds_list_projects():
    return {"projects": _ds_manager.list_projects()}


@app.get("/api/ds/projects/{project_id}")
async def ds_get_project(project_id: str):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.get_stats()


@app.delete("/api/ds/projects/{project_id}")
async def ds_delete_project(project_id: str):
    return _ds_manager.delete_project(project_id)


@app.post("/api/ds/projects/{project_id}/samples")
async def ds_add_sample(project_id: str, body: dict[str, Any]):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.add_sample(body)


@app.post("/api/ds/projects/{project_id}/samples/bulk")
async def ds_add_bulk(project_id: str, body: dict[str, Any]):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.add_samples_bulk(body.get("samples", []))


@app.get("/api/ds/projects/{project_id}/samples")
async def ds_get_samples(project_id: str, offset: int = 0, limit: int = 50):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.get_samples(offset, limit)


@app.delete("/api/ds/projects/{project_id}/samples/{sample_id}")
async def ds_remove_sample(project_id: str, sample_id: str):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.remove_sample(sample_id)


@app.put("/api/ds/projects/{project_id}/samples/{sample_id}")
async def ds_update_sample(project_id: str, sample_id: str, body: dict[str, Any]):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.update_sample(sample_id, body)


@app.post("/api/ds/projects/{project_id}/export")
async def ds_export(project_id: str, body: dict[str, Any]):
    p = _ds_manager.get_project(project_id)
    if not p:
        return {"status": "error", "message": "Project not found"}
    return p.export(body.get("format", "jsonl"))


# --- Dataset Sources ---

@app.get("/api/ds/kaggle/search")
async def ds_kaggle_search(query: str, limit: int = 20):
    try:
        return {"datasets": KaggleSource.search(query, limit)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/ds/kaggle/download")
async def ds_kaggle_download(body: dict[str, Any]):
    try:
        return KaggleSource.download(body["dataset_id"], body.get("dest"))
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/ds/url/download")
async def ds_url_download(body: dict[str, Any]):
    try:
        return URLSource.download(body["url"], body.get("filename"))
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/ds/local/scan")
async def ds_local_scan(body: dict[str, Any]):
    return LocalSource.scan_directory(body["path"])


@app.post("/api/ds/local/load")
async def ds_local_load(body: dict[str, Any]):
    path = body["path"]
    fmt = body.get("format")
    if not fmt:
        ext = Path(path).suffix.lower()
        fmt = {".csv": "csv", ".json": "json", ".jsonl": "jsonl"}.get(ext, "csv")
    loaders = {"csv": LocalSource.load_csv, "json": LocalSource.load_json, "jsonl": LocalSource.load_jsonl}
    if fmt not in loaders:
        return {"status": "error", "message": f"Unsupported format: {fmt}"}
    return loaders[fmt](path)


# --- Format Converters ---

@app.post("/api/ds/convert")
async def ds_convert(body: dict[str, Any]):
    conversion = body.get("conversion")
    converters_map = {
        "csv_to_jsonl": lambda: ds_converters.csv_to_jsonl(body["input"], body.get("output")),
        "jsonl_to_csv": lambda: ds_converters.jsonl_to_csv(body["input"], body.get("output")),
        "yolo_to_coco": lambda: ds_converters.yolo_to_coco(body["images_dir"], body["labels_dir"], body.get("classes", []), body.get("output")),
        "coco_to_yolo": lambda: ds_converters.coco_to_yolo(body["input"], body.get("output")),
        "alpaca_to_sharegpt": lambda: ds_converters.alpaca_to_sharegpt(body["input"], body.get("output")),
        "sharegpt_to_alpaca": lambda: ds_converters.sharegpt_to_alpaca(body["input"], body.get("output")),
    }
    if conversion not in converters_map:
        return {"status": "error", "message": f"Unknown conversion: {conversion}", "available": list(converters_map.keys())}
    try:
        return converters_map[conversion]()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- Architecture templates ---

TEMPLATES = {
    "mlp_classifier": {
        "name": "MLP Classifier",
        "description": "Simple feedforward network for tabular data",
        "dataset": "spiral",
        "nodes": [
            {"layer_type": "Linear", "params": {"in_features": 2, "out_features": 64}, "activation": "ReLU", "position": 0},
            {"layer_type": "BatchNorm1d", "params": {"num_features": 64}, "activation": None, "position": 1},
            {"layer_type": "Dropout", "params": {"p": 0.3}, "activation": None, "position": 2},
            {"layer_type": "Linear", "params": {"in_features": 64, "out_features": 32}, "activation": "ReLU", "position": 3},
            {"layer_type": "Linear", "params": {"in_features": 32, "out_features": 2}, "activation": None, "position": 4},
        ],
    },
    "deep_mlp": {
        "name": "Deep MLP",
        "description": "Deeper network with residual-style blocks",
        "dataset": "spiral",
        "nodes": [
            {"layer_type": "Linear", "params": {"in_features": 2, "out_features": 128}, "activation": "ReLU", "position": 0},
            {"layer_type": "BatchNorm1d", "params": {"num_features": 128}, "activation": None, "position": 1},
            {"layer_type": "Dropout", "params": {"p": 0.2}, "activation": None, "position": 2},
            {"layer_type": "ResidualBlock", "params": {"in_features": 128}, "activation": None, "position": 3},
            {"layer_type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "GELU", "position": 4},
            {"layer_type": "Dropout", "params": {"p": 0.2}, "activation": None, "position": 5},
            {"layer_type": "Linear", "params": {"in_features": 64, "out_features": 2}, "activation": None, "position": 6},
        ],
    },
    "gated_network": {
        "name": "Gated Network",
        "description": "Uses GLU for gating mechanism",
        "dataset": "xor",
        "nodes": [
            {"layer_type": "Linear", "params": {"in_features": 2, "out_features": 64}, "activation": "ReLU", "position": 0},
            {"layer_type": "GatedLinearUnit", "params": {"in_features": 64, "out_features": 32}, "activation": None, "position": 1},
            {"layer_type": "Linear", "params": {"in_features": 32, "out_features": 32}, "activation": "SiLU", "position": 2},
            {"layer_type": "Linear", "params": {"in_features": 32, "out_features": 2}, "activation": None, "position": 3},
        ],
    },
    "wide_shallow": {
        "name": "Wide & Shallow",
        "description": "Wide hidden layer, minimal depth",
        "dataset": "circles",
        "nodes": [
            {"layer_type": "Linear", "params": {"in_features": 2, "out_features": 256}, "activation": "ReLU", "position": 0},
            {"layer_type": "Dropout", "params": {"p": 0.5}, "activation": None, "position": 1},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 2}, "activation": None, "position": 2},
        ],
    },
    "transformer_classifier": {
        "name": "Transformer",
        "description": "Self-attention transformer for tabular data",
        "dataset": "spiral",
        "nodes": [
            {"layer_type": "TokenEmbedding", "params": {"in_features": 2, "d_model": 64, "seq_len": 4}, "activation": None, "position": 0},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 64}, "activation": None, "position": 1},
            {"layer_type": "TransformerBlock", "params": {"d_model": 64, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 2},
            {"layer_type": "TransformerBlock", "params": {"d_model": 64, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 3},
            {"layer_type": "SequencePool", "params": {"d_model": 64, "mode": "mean"}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 64, "out_features": 2}, "activation": None, "position": 5},
        ],
    },
    "deep_transformer": {
        "name": "Deep Transformer",
        "description": "4-layer transformer with larger model dim",
        "dataset": "blobs",
        "nodes": [
            {"layer_type": "TokenEmbedding", "params": {"in_features": 2, "d_model": 128, "seq_len": 8}, "activation": None, "position": 0},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 128}, "activation": None, "position": 1},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 2},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 3},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "SequencePool", "params": {"d_model": 128, "mode": "mean"}, "activation": None, "position": 6},
            {"layer_type": "Linear", "params": {"in_features": 128, "out_features": 32}, "activation": "ReLU", "position": 7},
            {"layer_type": "Linear", "params": {"in_features": 32, "out_features": 4}, "activation": None, "position": 8},
        ],
    },
    "mnist_cnn": {
        "name": "MNIST CNN",
        "description": "Convolutional network for 28x28 images",
        "dataset": "mnist",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "Conv2d", "params": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1}, "activation": "ReLU", "position": 0},
            {"layer_type": "MaxPool2d", "params": {"kernel_size": 2}, "activation": None, "position": 1},
            {"layer_type": "Conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1}, "activation": "ReLU", "position": 2},
            {"layer_type": "MaxPool2d", "params": {"kernel_size": 2}, "activation": None, "position": 3},
            {"layer_type": "Flatten", "params": {}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 3136, "out_features": 128}, "activation": "ReLU", "position": 5},
            {"layer_type": "Dropout", "params": {"p": 0.5}, "activation": None, "position": 6},
            {"layer_type": "Linear", "params": {"in_features": 128, "out_features": 10}, "activation": None, "position": 7},
        ],
    },
    "vit_tiny": {
        "name": "ViT Tiny",
        "description": "Vision Transformer for image classification",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "PatchEmbed", "params": {"in_channels": 3, "d_model": 128, "patch_size": 4, "image_size": 32}, "activation": None, "position": 0},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 1},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 2},
            {"layer_type": "TransformerBlock", "params": {"d_model": 128, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 3},
            {"layer_type": "SequencePool", "params": {"d_model": 128, "mode": "cls"}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 128, "out_features": 10}, "activation": None, "position": 5},
        ],
    },
    "mobilenet_style": {
        "name": "MobileNet-style",
        "description": "Efficient CNN with depthwise separable convolutions",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "ReLU", "position": 0},
            {"layer_type": "DepthwiseSeparableConv", "params": {"in_channels": 32, "out_channels": 64}, "activation": "ReLU", "position": 1},
            {"layer_type": "DepthwiseSeparableConv", "params": {"in_channels": 64, "out_channels": 128, "stride": 2}, "activation": "ReLU", "position": 2},
            {"layer_type": "DepthwiseSeparableConv", "params": {"in_channels": 128, "out_channels": 256, "stride": 2}, "activation": "ReLU", "position": 3},
            {"layer_type": "GlobalAvgPool", "params": {}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 10}, "activation": None, "position": 5},
        ],
    },
    "audio_classifier": {
        "name": "Audio Classifier",
        "description": "Conv1d-based audio classification pipeline",
        "dataset": "random",
        "nodes": [
            {"layer_type": "MelSpectrogram", "params": {"n_mels": 40, "n_fft": 512, "hop_length": 128}, "activation": None, "position": 0},
            {"layer_type": "AudioConvBlock", "params": {"in_channels": 40, "out_channels": 64}, "activation": None, "position": 1},
            {"layer_type": "AudioConvBlock", "params": {"in_channels": 64, "out_channels": 128, "stride": 2}, "activation": None, "position": 2},
            {"layer_type": "AudioConvBlock", "params": {"in_channels": 128, "out_channels": 256, "stride": 2}, "activation": None, "position": 3},
            {"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4}, "activation": None, "position": 5},
            {"layer_type": "SequencePool", "params": {"d_model": 256, "mode": "mean"}, "activation": None, "position": 6},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 10}, "activation": None, "position": 7},
        ],
    },
    "autoencoder": {
        "name": "Autoencoder",
        "description": "Conv autoencoder with encoder-decoder structure",
        "dataset": "mnist",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "DownBlock", "params": {"in_channels": 1, "out_channels": 32}, "activation": None, "position": 0},
            {"layer_type": "DownBlock", "params": {"in_channels": 32, "out_channels": 64}, "activation": None, "position": 1},
            {"layer_type": "ResConvBlock", "params": {"in_channels": 64}, "activation": None, "position": 2},
            {"layer_type": "UpBlock", "params": {"in_channels": 64, "out_channels": 32}, "activation": None, "position": 3},
            {"layer_type": "UpBlock", "params": {"in_channels": 32, "out_channels": 1}, "activation": None, "position": 4},
        ],
    },
    "resnet_style": {
        "name": "ResNet-style CNN",
        "description": "CNN with residual convolution blocks",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1}, "activation": "ReLU", "position": 0},
            {"layer_type": "ResConvBlock", "params": {"in_channels": 64, "out_channels": 64}, "activation": None, "position": 1},
            {"layer_type": "ResConvBlock", "params": {"in_channels": 64, "out_channels": 128}, "activation": None, "position": 2},
            {"layer_type": "ResConvBlock", "params": {"in_channels": 128, "out_channels": 256}, "activation": None, "position": 3},
            {"layer_type": "GlobalAvgPool", "params": {}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 10}, "activation": None, "position": 5},
        ],
    },
    # ========================= ADVANCED VISION =========================
    "swin_transformer": {
        "name": "Swin Transformer",
        "description": "Shifted window transformer for image classification (Swin-T style)",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "PatchEmbed", "params": {"in_channels": 3, "d_model": 96, "patch_size": 4, "image_size": 32}, "activation": None, "position": 0},
            {"layer_type": "SwinStage", "params": {"d_model": 96, "depth": 2, "n_heads": 3, "window_size": 7, "downsample": True}, "activation": None, "position": 1},
            {"layer_type": "SwinStage", "params": {"d_model": 192, "depth": 2, "n_heads": 6, "window_size": 7, "downsample": True}, "activation": None, "position": 2},
            {"layer_type": "SwinStage", "params": {"d_model": 384, "depth": 6, "n_heads": 12, "window_size": 7, "downsample": True}, "activation": None, "position": 3},
            {"layer_type": "SwinStage", "params": {"d_model": 768, "depth": 2, "n_heads": 24, "window_size": 7, "downsample": False}, "activation": None, "position": 4},
            {"layer_type": "GlobalAvgPool", "params": {}, "activation": None, "position": 5},
            {"layer_type": "Linear", "params": {"in_features": 768, "out_features": 10}, "activation": None, "position": 6},
        ],
    },
    "convnext": {
        "name": "ConvNeXt",
        "description": "Modernized ConvNet competing with transformers",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 96, "kernel_size": 4, "stride": 4}, "activation": None, "position": 0},
            {"layer_type": "ConvNeXtBlock", "params": {"dim": 96}, "activation": None, "position": 1},
            {"layer_type": "ConvNeXtBlock", "params": {"dim": 96}, "activation": None, "position": 2},
            {"layer_type": "ConvNeXtBlock", "params": {"dim": 96}, "activation": None, "position": 3},
            {"layer_type": "GlobalAvgPool", "params": {}, "activation": None, "position": 4},
            {"layer_type": "Linear", "params": {"in_features": 96, "out_features": 10}, "activation": None, "position": 5},
        ],
    },
    "yolov8_backbone": {
        "name": "YOLOv8 Backbone",
        "description": "YOLOv8-style backbone with CSP and C2f blocks for object detection",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU", "position": 0},
            {"layer_type": "Conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU", "position": 1},
            {"layer_type": "C2fBlock", "params": {"in_channels": 128, "out_channels": 128, "n_bottlenecks": 3}, "activation": None, "position": 2},
            {"layer_type": "Conv2d", "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU", "position": 3},
            {"layer_type": "C2fBlock", "params": {"in_channels": 256, "out_channels": 256, "n_bottlenecks": 6}, "activation": None, "position": 4},
            {"layer_type": "Conv2d", "params": {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU", "position": 5},
            {"layer_type": "C2fBlock", "params": {"in_channels": 512, "out_channels": 512, "n_bottlenecks": 6}, "activation": None, "position": 6},
            {"layer_type": "SPPFBlock", "params": {"in_channels": 512, "out_channels": 512}, "activation": None, "position": 7},
            {"layer_type": "GlobalAvgPool", "params": {}, "activation": None, "position": 8},
            {"layer_type": "DetectionHead", "params": {"in_channels": 512, "num_classes": 80}, "activation": None, "position": 9},
        ],
    },
    "detr_detector": {
        "name": "DETR Detector",
        "description": "DEtection TRansformer — end-to-end object detection with transformer",
        "dataset": "cifar10",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "MultiScaleFeatureExtractor", "params": {"in_channels": 3, "channels": [64, 128, 256, 512]}, "activation": None, "position": 0},
            {"layer_type": "Conv2d", "params": {"in_channels": 512, "out_channels": 256, "kernel_size": 1}, "activation": None, "position": 1},
            {"layer_type": "Flatten", "params": {"start_dim": 2}, "activation": None, "position": 2},
            {"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}, "activation": None, "position": 3},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 6},
            {"layer_type": "DETRTransformerDecoder", "params": {"d_model": 256, "n_heads": 8, "num_layers": 6, "num_queries": 100}, "activation": None, "position": 7},
        ],
    },
    # ========================= ADVANCED DIFFUSION =========================
    "dit_small": {
        "name": "DiT-S (Diffusion Transformer)",
        "description": "Diffusion Transformer — scalable diffusion with transformer backbone",
        "dataset": "mnist",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "PatchEmbedDiT", "params": {"in_channels": 1, "d_model": 384, "patch_size": 4, "image_size": 28}, "activation": None, "position": 0},
            {"layer_type": "TimestepMLP", "params": {"d_model": 384, "time_embed_dim": 384}, "activation": None, "position": 1},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 2},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 3},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 4},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 5},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 6},
            {"layer_type": "DiTBlock", "params": {"d_model": 384, "n_heads": 6, "cond_dim": 384}, "activation": None, "position": 7},
            {"layer_type": "DiTFinalLayer", "params": {"d_model": 384, "patch_size": 4, "out_channels": 1}, "activation": None, "position": 8},
        ],
    },
    "sd_unet": {
        "name": "Stable Diffusion UNet",
        "description": "UNet with cross-attention for text-conditioned image generation",
        "dataset": "mnist",
        "dataset_type": "real",
        "nodes": [
            {"layer_type": "TimestepMLP", "params": {"d_model": 128, "time_embed_dim": 512}, "activation": None, "position": 0},
            {"layer_type": "UNetDownBlock", "params": {"in_channels": 1, "out_channels": 128, "time_embed_dim": 512, "num_layers": 2, "has_attention": False}, "activation": None, "position": 1},
            {"layer_type": "UNetDownBlock", "params": {"in_channels": 128, "out_channels": 256, "time_embed_dim": 512, "num_layers": 2, "has_attention": True, "context_dim": 256}, "activation": None, "position": 2},
            {"layer_type": "UNetDownBlock", "params": {"in_channels": 256, "out_channels": 512, "time_embed_dim": 512, "num_layers": 2, "has_attention": True, "context_dim": 256}, "activation": None, "position": 3},
            {"layer_type": "UNetMidBlock", "params": {"channels": 512, "time_embed_dim": 512, "context_dim": 256}, "activation": None, "position": 4},
            {"layer_type": "UNetUpBlock", "params": {"in_channels": 512, "out_channels": 256, "skip_channels": 512, "time_embed_dim": 512, "num_layers": 2, "has_attention": True, "context_dim": 256}, "activation": None, "position": 5},
            {"layer_type": "UNetUpBlock", "params": {"in_channels": 256, "out_channels": 128, "skip_channels": 256, "time_embed_dim": 512, "num_layers": 2, "has_attention": True, "context_dim": 256}, "activation": None, "position": 6},
            {"layer_type": "UNetUpBlock", "params": {"in_channels": 128, "out_channels": 64, "skip_channels": 128, "time_embed_dim": 512, "num_layers": 2, "has_attention": False}, "activation": None, "position": 7},
        ],
    },
    # ========================= ADVANCED AUDIO / SPEECH =========================
    "conformer_asr": {
        "name": "Conformer (ASR)",
        "description": "Conformer encoder for automatic speech recognition — state-of-the-art ASR",
        "dataset": "random",
        "nodes": [
            {"layer_type": "MelSpectrogram", "params": {"n_mels": 80, "n_fft": 512, "hop_length": 160}, "activation": None, "position": 0},
            {"layer_type": "AudioConvBlock", "params": {"in_channels": 80, "out_channels": 256}, "activation": None, "position": 1},
            {"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}, "activation": None, "position": 2},
            {"layer_type": "ConformerBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_dim": 1024, "kernel_size": 31}, "activation": None, "position": 3},
            {"layer_type": "ConformerBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_dim": 1024, "kernel_size": 31}, "activation": None, "position": 4},
            {"layer_type": "ConformerBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_dim": 1024, "kernel_size": 31}, "activation": None, "position": 5},
            {"layer_type": "ConformerBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_dim": 1024, "kernel_size": 31}, "activation": None, "position": 6},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 5000}, "activation": None, "position": 7},
        ],
    },
    "whisper_encoder": {
        "name": "Whisper-style Encoder",
        "description": "Whisper-style audio encoder with conv stem + transformer layers",
        "dataset": "random",
        "nodes": [
            {"layer_type": "MelSpectrogram", "params": {"n_mels": 80, "n_fft": 400, "hop_length": 160}, "activation": None, "position": 0},
            {"layer_type": "Conv1d", "params": {"in_channels": 80, "out_channels": 256, "kernel_size": 3, "padding": 1}, "activation": "GELU", "position": 1},
            {"layer_type": "Conv1d", "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "GELU", "position": 2},
            {"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}, "activation": None, "position": 3},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 256, "max_len": 1500}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 6},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 7},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 8},
            {"layer_type": "SequencePool", "params": {"d_model": 256, "mode": "mean"}, "activation": None, "position": 9},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 5000}, "activation": None, "position": 10},
        ],
    },
    "hifi_gan_generator": {
        "name": "HiFi-GAN Generator",
        "description": "Neural vocoder for high-fidelity speech synthesis",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Conv1d", "params": {"in_channels": 80, "out_channels": 512, "kernel_size": 7, "padding": 3}, "activation": None, "position": 0},
            {"layer_type": "VocoderUpsampleBlock", "params": {"in_channels": 512, "out_channels": 256, "upsample_rate": 8, "kernel_size": 16}, "activation": None, "position": 1},
            {"layer_type": "VocoderUpsampleBlock", "params": {"in_channels": 256, "out_channels": 128, "upsample_rate": 8, "kernel_size": 16}, "activation": None, "position": 2},
            {"layer_type": "VocoderUpsampleBlock", "params": {"in_channels": 128, "out_channels": 64, "upsample_rate": 2, "kernel_size": 4}, "activation": None, "position": 3},
            {"layer_type": "VocoderUpsampleBlock", "params": {"in_channels": 64, "out_channels": 32, "upsample_rate": 2, "kernel_size": 4}, "activation": None, "position": 4},
            {"layer_type": "Conv1d", "params": {"in_channels": 32, "out_channels": 1, "kernel_size": 7, "padding": 3}, "activation": "Tanh", "position": 5},
        ],
    },
    "tts_fastspeech": {
        "name": "FastSpeech2 TTS",
        "description": "Non-autoregressive text-to-speech with duration/pitch/energy prediction",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 256, "embedding_dim": 256}, "activation": None, "position": 0},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 256}, "activation": None, "position": 1},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 2},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 3},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "DurationPredictor", "params": {"d_model": 256, "kernel_size": 3, "num_layers": 2}, "activation": None, "position": 6},
            {"layer_type": "VariancePredictor", "params": {"d_model": 256, "kernel_size": 3, "num_layers": 2}, "activation": None, "position": 7},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 80}, "activation": None, "position": 8},
        ],
    },
    "voice_cloning": {
        "name": "Voice Cloning Encoder",
        "description": "Speaker encoder + mel decoder for voice cloning / conversion",
        "dataset": "random",
        "nodes": [
            {"layer_type": "MelSpectrogram", "params": {"n_mels": 40, "n_fft": 512, "hop_length": 160}, "activation": None, "position": 0},
            {"layer_type": "SpeakerEncoder", "params": {"in_channels": 40, "d_model": 256, "num_layers": 3}, "activation": None, "position": 1},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 512}, "activation": "ReLU", "position": 2},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 256}, "activation": None, "position": 3},
        ],
    },
    # ========================= ADVANCED LLM =========================
    "gemini_style": {
        "name": "Gemini-style Multi-Modal LLM",
        "description": "Multi-modal transformer LLM with vision + audio + text (Gemini/GPT-4 architecture)",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 512}, "activation": None, "position": 0},
            {"layer_type": "RotaryPositionalEmbedding", "params": {"d_model": 512}, "activation": None, "position": 1},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 2},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 3},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 4},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 5},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 6},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm", "dropout": 0.0}, "activation": None, "position": 7},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 8},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 32000}, "activation": None, "position": 9},
        ],
    },
    "claude_style_llm": {
        "name": "Claude/GPT-style Decoder LLM",
        "description": "Large decoder-only transformer with RoPE, SwiGLU, RMSNorm — Claude/Llama architecture",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 768}, "activation": None, "position": 0},
            {"layer_type": "RotaryPositionalEmbedding", "params": {"d_model": 768}, "activation": None, "position": 1},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 2},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 3},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 4},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 5},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 6},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 7},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 8},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 9},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 10},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 768, "n_heads": 12, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 11},
            {"layer_type": "RMSNorm", "params": {"d_model": 768}, "activation": None, "position": 12},
            {"layer_type": "Linear", "params": {"in_features": 768, "out_features": 32000}, "activation": None, "position": 13},
        ],
    },
    "moe_llm": {
        "name": "Mixture-of-Experts LLM",
        "description": "Mixtral/Switch-style MoE transformer — sparse expert routing for efficiency",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 512}, "activation": None, "position": 0},
            {"layer_type": "RotaryPositionalEmbedding", "params": {"d_model": 512}, "activation": None, "position": 1},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 2},
            {"layer_type": "LLMAttention", "params": {"d_model": 512, "n_heads": 8}, "activation": None, "position": 3},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 4},
            {"layer_type": "MoELayer", "params": {"d_model": 512, "num_experts": 8, "top_k": 2, "ffn_dim": 2048}, "activation": None, "position": 5},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 6},
            {"layer_type": "LLMAttention", "params": {"d_model": 512, "n_heads": 8}, "activation": None, "position": 7},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 8},
            {"layer_type": "MoELayer", "params": {"d_model": 512, "num_experts": 8, "top_k": 2, "ffn_dim": 2048}, "activation": None, "position": 9},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 10},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 32000}, "activation": None, "position": 11},
        ],
    },
    "encoder_decoder_t5": {
        "name": "T5/BART Encoder-Decoder",
        "description": "Encoder-decoder transformer (T5/BART style) for seq2seq, translation, summarization",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 512}, "activation": None, "position": 0},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 512, "max_len": 512}, "activation": None, "position": 1},
            {"layer_type": "EncoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 2},
            {"layer_type": "EncoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 3},
            {"layer_type": "EncoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 4},
            {"layer_type": "DecoderBlockWithCrossAttn", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "DecoderBlockWithCrossAttn", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 6},
            {"layer_type": "DecoderBlockWithCrossAttn", "params": {"d_model": 512, "n_heads": 8, "ffn_dim": 2048, "dropout": 0.1}, "activation": None, "position": 7},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 32000}, "activation": None, "position": 8},
        ],
    },
    "mamba_lm": {
        "name": "Mamba Language Model",
        "description": "State-space model for language — linear-time alternative to transformers",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 512}, "activation": None, "position": 0},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 1},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 2},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 3},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 4},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 5},
            {"layer_type": "MambaBlock", "params": {"d_model": 512, "d_state": 16, "d_conv": 4, "expand_factor": 2}, "activation": None, "position": 6},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 7},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 32000}, "activation": None, "position": 8},
        ],
    },
    # ========================= VIDEO GENERATION =========================
    "video_dit": {
        "name": "Video DiT (Veo-style)",
        "description": "Video diffusion transformer — generate video frames with 3D attention (Veo/Sora-style)",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Conv3d", "params": {"in_channels": 3, "out_channels": 256, "kernel_size": 3}, "activation": "SiLU", "position": 0},
            {"layer_type": "Conv3dBlock", "params": {"in_channels": 256, "out_channels": 256}, "activation": None, "position": 1},
            {"layer_type": "TemporalPool", "params": {"d_model": 256, "mode": "mean"}, "activation": None, "position": 2},
            {"layer_type": "PositionalEncoding", "params": {"d_model": 256, "max_len": 1024}, "activation": None, "position": 3},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 4},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 5},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 6},
            {"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 8, "dropout": 0.1}, "activation": None, "position": 7},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 768}, "activation": None, "position": 8},
        ],
    },
    "multimodal_llm": {
        "name": "Multi-Modal LLM",
        "description": "Vision + Audio + Text multi-modal transformer (Gemini/GPT-4V/Pixtral architecture)",
        "dataset": "random",
        "nodes": [
            {"layer_type": "PatchEmbedding", "params": {"in_channels": 3, "d_model": 512, "patch_size": 16, "image_size": 224}, "activation": None, "position": 0},
            {"layer_type": "AudioEmbedding", "params": {"n_mels": 80, "d_model": 512}, "activation": None, "position": 1},
            {"layer_type": "Embedding", "params": {"num_embeddings": 32000, "embedding_dim": 512}, "activation": None, "position": 2},
            {"layer_type": "RotaryPositionalEmbedding", "params": {"d_model": 512}, "activation": None, "position": 3},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 4},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 5},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 6},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 512, "n_heads": 8, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 7},
            {"layer_type": "RMSNorm", "params": {"d_model": 512}, "activation": None, "position": 8},
            {"layer_type": "Linear", "params": {"in_features": 512, "out_features": 32000}, "activation": None, "position": 9},
        ],
    },
    "nano_lm": {
        "name": "Nano Language Model",
        "description": "Tiny but complete LLM for learning and experimentation — nanoGPT/TinyLlama style",
        "dataset": "random",
        "nodes": [
            {"layer_type": "Embedding", "params": {"num_embeddings": 8000, "embedding_dim": 256}, "activation": None, "position": 0},
            {"layer_type": "RotaryPositionalEmbedding", "params": {"d_model": 256}, "activation": None, "position": 1},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 2},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 3},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 4},
            {"layer_type": "LLMDecoderBlock", "params": {"d_model": 256, "n_heads": 4, "ffn_type": "swiglu", "norm_type": "rmsnorm"}, "activation": None, "position": 5},
            {"layer_type": "RMSNorm", "params": {"d_model": 256}, "activation": None, "position": 6},
            {"layer_type": "Linear", "params": {"in_features": 256, "out_features": 8000}, "activation": None, "position": 7},
        ],
    },
}


@app.get("/api/templates")
async def list_templates():
    return {
        name: {"name": t["name"], "description": t["description"], "dataset": t.get("dataset")}
        for name, t in TEMPLATES.items()
    }


@app.post("/api/templates/{template_name}/apply")
async def apply_template(template_name: str):
    if template_name not in TEMPLATES:
        return {"status": "error", "message": f"Unknown template: {template_name}"}

    template = TEMPLATES[template_name]

    # Clear existing graph
    for node_id in list(engine.graph.nodes.keys()):
        engine.graph.remove_layer(node_id)

    # Add template layers
    for node in template["nodes"]:
        engine.graph.add_layer(
            layer_type=node["layer_type"],
            params=node.get("params", {}),
            activation=node.get("activation"),
            position=node.get("position"),
        )

    graph_data = engine.graph.to_dict()
    await broadcast("graph_updated", graph_data)

    return {
        "status": "applied",
        "template": template_name,
        "graph": graph_data,
        "suggested_dataset": template.get("dataset"),
        "suggested_dataset_type": template.get("dataset_type", "synthetic"),
    }


# --- Training ---

@app.post("/api/train/start")
async def start_training():
    result = engine.start_training()
    await broadcast("training_status", result)
    return result


@app.post("/api/train/stop")
async def stop_training():
    result = engine.stop_training()
    await broadcast("training_status", result)
    return result


@app.post("/api/model/save")
async def save_model(body: dict[str, Any]):
    """Save trained model weights to disk."""
    if engine.model is None:
        return {"status": "error", "message": "No model built. Build and train a model first."}
    import os
    path = body.get("path", "./sg_outputs/model_weights.pt")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": engine.model.state_dict(),
        "architecture": engine.export_architecture(),
    }, path)
    return {"status": "ok", "path": path, "message": f"Model saved to {path}"}


@app.post("/api/model/load")
async def load_model_weights(body: dict[str, Any]):
    """Load trained model weights from disk."""
    path = body.get("path", "./sg_outputs/model_weights.pt")
    if not os.path.exists(path):
        return {"status": "error", "message": f"File not found: {path}"}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Restore architecture first if present
    if "architecture" in checkpoint:
        engine.import_architecture(checkpoint["architecture"])
        engine.build()
    if engine.model is None:
        return {"status": "error", "message": "No model to load weights into. Build or include architecture."}
    engine.model.load_state_dict(checkpoint["model_state_dict"])
    await broadcast("graph_updated", engine.graph.to_dict())
    return {"status": "ok", "path": path, "message": f"Model loaded from {path}"}


@app.post("/api/reset")
async def reset_all():
    """Full server-side reset."""
    result = engine.reset()
    await broadcast("graph_updated", {"nodes": [], "edges": []})
    await broadcast("training_status", {"status": "stopped"})
    return result


@app.get("/api/metrics/snapshot")
async def get_metrics_snapshot():
    return engine.metrics.get_snapshot()


@app.get("/api/metrics/loss")
async def get_loss_history(last_n: int | None = None):
    return engine.metrics.get_loss_history(last_n)


@app.get("/api/metrics/layer/{layer_name}")
async def get_layer_metrics(layer_name: str, last_n: int | None = None):
    return engine.metrics.get_layer_history(layer_name, last_n)


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.add(ws)
    try:
        reg = Registry.list_all()
        reg["schedulers"] = SchedulerRegistry.list_all()
        reg["scheduler_defaults"] = {
            name: SchedulerRegistry.get_default_params(name)
            for name in SchedulerRegistry.list_all()
        }

        status = engine.get_status()
        status["data_info"] = engine.data_manager.get_info()

        await ws.send_text(json.dumps({
            "event": "init",
            "data": {
                "registry": reg,
                "status": status,
                "templates": {
                    name: {"name": t["name"], "description": t["description"], "dataset": t.get("dataset")}
                    for name, t in TEMPLATES.items()
                },
            },
        }, default=str))

        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            # Handle ping/pong keepalive
            if msg.get("event") == "ping":
                await ws.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)


def main():
    import uvicorn
    uvicorn.run(
        "state_graph.server.app:app",
        host="0.0.0.0",
        port=8765,
        reload=True,
    )


if __name__ == "__main__":
    main()
