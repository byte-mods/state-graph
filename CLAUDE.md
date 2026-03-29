# StateGraph

Full-stack ML research platform with real-time web UI — architecture builder, HuggingFace fine-tuning, Unsloth, RL training, robotics simulation, data pipelines, dataset factory, and in-browser IDE.

## Quick Start

```bash
pip install -e .                # Base (PyTorch + FastAPI)
pip install -e ".[hf]"          # + HuggingFace (transformers, peft, datasets, timm, diffusers, trl)
pip install -e ".[unsloth]"     # + Unsloth fast fine-tuning
pip install -e ".[rl]"          # + RL (gymnasium, stable-baselines3)
pip install -e ".[rl-robotics]" # + MuJoCo/PyBullet robotics
pip install -e ".[all]"         # Everything
state-graph                     # Opens UI at http://localhost:8765
```

## Architecture

### Core (Neural Network Builder)
- `state_graph/core/registry.py` — Plug-and-play registry for layers, activations, losses, optimizers
- `state_graph/core/graph.py` — State graph: add/remove/reorder layers, build nn.Sequential
- `state_graph/core/engine.py` — Training engine: background thread, async broadcast, graph + HF model support
- `state_graph/core/metrics.py` — Per-layer metrics (gradients, weights, activations)
- `state_graph/core/scheduler.py` — 10 LR schedulers with defaults
- `state_graph/core/data.py` — 8 synthetic + 5 real datasets, augmentation pipeline
- `state_graph/layers/custom.py` — TransformerBlock, ResidualBlock, GLU, ResNetBlock, ConvNeXtBlock, MBConvBlock, VisionEncoder, DiffusionUNet, VAE, NoiseScheduler, VideoVAE, TemporalAttention, PerceiverResampler, DistillationWrapper, SelectiveScan, MambaBlock, RWKVBlock, RetentionLayer, HyenaOperator, XLSTM, GatedLinearRecurrence
- `state_graph/layers/llm.py` — RMSNorm, RoPE, LLMAttention (GQA + Flash), SwiGLU/GeGLU/ReGLU, MoE, LLMModel, ComposableLLM, EncoderDecoderLLM, AdaptiveDepthLLM, MultiModalLLM, UnifiedMultiModalLLM, VideoEmbedding, 18 MODEL_BLUEPRINTS, 19 BLOCK_DESIGNS, CustomComponent

### HuggingFace Integration
- `state_graph/hf/hub.py` — Model search, load (transformers/timm/diffusers), freeze/unfreeze, LoRA
- `state_graph/hf/datasets.py` — HF datasets search, load, create (text/image/audio/CSV/JSON)
- `state_graph/hf/unsloth.py` — Unsloth fast fine-tuning: SFT, DPO, ORPO, KTO, Reward model, GGUF export

### Reinforcement Learning
- `state_graph/rl/engine.py` — RL training with Stable-Baselines3: PPO, A2C, DQN, SAC, TD3, DDPG; 26 Gymnasium envs + custom grid world

### Robotics Simulation
- `state_graph/robotics/simulator.py` — 27 real components, circuit solver, 5 robot templates, joint articulation, Three.js 3D + Cannon.js physics

### Data Engineering
- `state_graph/dataeng/connectors.py` — 15 connectors: MySQL, PostgreSQL, MSSQL, Oracle, SQLite, MongoDB, Redis, Elasticsearch, Kafka, S3, BigQuery, ClickHouse, CSV, JSON, Parquet
- `state_graph/dataeng/pipeline.py` — 17 transforms, merge/join, stats, visualization, persistent pipelines

### Dataset Factory
- `state_graph/datasets/creator.py` — 21 templates (text/image/audio/video/multimodal), export to JSONL/CSV/Alpaca/ShareGPT/YOLO/COCO/HF
- `state_graph/datasets/sources.py` — Kaggle, URL download, local file scanning
- `state_graph/datasets/converters.py` — YOLO↔COCO, CSV↔JSONL, Alpaca↔ShareGPT

### Workspace / IDE
- `state_graph/workspace/manager.py` — Project management with templates (LLM, Vision, YOLO, Dataset)
- `state_graph/workspace/executor.py` — Python code execution with stdout/stderr capture

### Server & UI
- `state_graph/server/app.py` — FastAPI + WebSocket, 291 endpoints (blueprints, novel architecture lab, diffusion training)
- `state_graph/ui/index.html` — Single-page app: CodeMirror IDE, Three.js 3D, Chart.js, annotation tools

## Key Patterns

- All optional dependencies are lazy-loaded — base install only needs PyTorch + FastAPI
- WebSocket broadcasts real-time metrics for training, RL, and Unsloth
- Dual model source: `graph` (visual builder) or `hf` (HuggingFace pretrained)
- Experiment comparison: save runs, overlay loss/accuracy curves
- System resource monitoring: CPU, RAM, GPU in real-time

## Tests

```bash
pytest tests/ -v  # 598 tests covering core, server, datasets, LLM, blueprints
```
