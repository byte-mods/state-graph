<p align="center">
  <h1 align="center">StateGraph</h1>
  <p align="center">
    <strong>The complete ML research platform. Design, train, evaluate, and deploy AI models — entirely from your browser.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#features">Features</a> &bull;
    <a href="#installation">Installation</a> &bull;
    <a href="#screenshots">Screenshots</a> &bull;
    <a href="#architecture">Architecture</a> &bull;
    <a href="#license">License</a>
  </p>
</p>

---

StateGraph is a full-stack ML research platform with a real-time web UI. Researchers and engineers can build neural network architectures, fine-tune LLMs, train RL agents, simulate robots, create datasets, build data pipelines, and deploy models — all without writing code. When you do want code, there's a full Python IDE with AI assistance built in.

## Quick Start

```bash
# Install
git clone https://github.com/yourusername/state-graph.git
cd state-graph
pip install -e .

# Launch
state-graph
```

Open **http://localhost:8765** in your browser. That's it.

## Features

### Neural Network Architecture Builder
Build PyTorch models visually with drag-and-drop. 28+ layer types including Transformer blocks. 7 architecture templates. Real-time training charts (loss, learning rate, gradients, weight distributions). Experiment comparison across architectures.

- Drag layers from palette, reorder with handles, edit params in inspector
- Quick-add chips for common patterns (Linear+ReLU, Conv2d+ReLU, etc.)
- Custom activation formulas: type a math expression, register as activation
- 10 learning rate schedulers with auto-populated defaults
- Save/load architectures as JSON, export as Python code

### LLM Fine-Tuning (Unsloth + HuggingFace)
Fine-tune any LLM 2-4x faster with 70% less memory using Unsloth. Full HuggingFace ecosystem integration.

| Method | Description |
|--------|-------------|
| **SFT** | Supervised fine-tuning with instruction data |
| **DPO** | Direct Preference Optimization |
| **ORPO** | Odds Ratio Preference (no reference model needed) |
| **KTO** | Binary feedback (thumbs up/down) |
| **Reward** | Train reward models for RLHF |

- 18 pre-configured Unsloth models (LLaMA, Mistral, Gemma, Phi, Qwen, DeepSeek)
- LoRA configuration with auto-detected target modules
- Chat template support (ChatML, LLaMA-3, Mistral, etc.)
- Export to GGUF (for Ollama/llama.cpp), HF Hub, merged weights
- Real-time loss, learning rate, and RL reward charts during training
- Built-in inference testing

### Reinforcement Learning
Train RL agents with Stable-Baselines3. 6 algorithms, 26+ environments, real-time visualization.

| Algorithm | Type | Best For |
|-----------|------|----------|
| PPO | On-policy | General purpose |
| A2C | On-policy | Fast training |
| DQN | Off-policy | Discrete actions (games) |
| SAC | Off-policy | Continuous control (robotics) |
| TD3 | Off-policy | Robotics |
| DDPG | Off-policy | Continuous control |

- Classic control (CartPole, LunarLander, Pendulum)
- Box2D (BipedalWalker, CarRacing)
- MuJoCo robotics (Humanoid, Ant, HalfCheetah, Hopper, Reacher)
- PyBullet (Kuka arm, humanoid)
- Custom grid world (configurable size/obstacles)
- Canvas-based episode replay animation
- Live reward and episode length charts

### Robotics Simulation
Build robots from real components, simulate physics, analyze circuits, connect to hardware.

**27 real components** with actual specs (voltage, current, torque, weight, dimensions):
- Servo motors (SG90, MG996R, DS3225)
- DC motors, steppers, BLDC
- Sensors (IMU, ultrasonic, LiDAR, camera, force, encoder)
- Batteries (LiPo 1S/2S/3S, 18650)
- Controllers (Arduino, Raspberry Pi, ESP32, Jetson Nano)
- Structural (aluminum plates, carbon fiber tubes, wheels)
- Electronics (motor drivers, voltage regulators, BECs)

**5 robot templates**: 2WD wheeled, 3-DOF arm, quadruped, humanoid (17-DOF), quadcopter drone

**3D simulation** (Three.js + Cannon.js):
- PBR materials, soft shadows, 3-point lighting
- Rigid body physics with gravity, collisions, constraints
- Joint sliders for real-time articulation
- Mouse orbit + pan + zoom

**Circuit analysis**:
- Total weight, power draw, current consumption
- Battery life estimation
- Current limit warnings
- Per-component power breakdown
- Voltage rail mapping

**Hardware bridge**:
- Serial/USB connection to Arduino, ESP32, RPi
- Auto-generated firmware (Arduino sketch)
- Send joint angles to physical robot from UI
- Read sensor data in real-time

**Server-side physics** (MuJoCo):
- 500Hz timestep (0.002s) for research-grade accuracy
- Streams state to browser at 60fps
- Falls back to Euler integration if MuJoCo not installed

### Dataset Factory
Create, annotate, and export datasets for any ML task. 21 templates across 5 categories.

**Interactive annotation tools**:
- **Image bounding box annotator** — draw YOLO/COCO boxes on canvas
- **Text span annotator** — highlight entities for NER
- **Tool calling builder** — visual tool schema + expected call builder
- **Media captioner** — upload images/video/audio, write captions (for Stable Diffusion, SORA, Veo3)

**Data sources**: Kaggle, direct URL, local files, HuggingFace Hub

**Export formats**: JSONL, CSV, JSON, Alpaca, ShareGPT, YOLO, COCO, ImageFolder, HuggingFace Arrow

### Data Engineering Pipelines
Connect to any database, clean/transform data, sink to any destination.

**15 connectors**: MySQL, PostgreSQL, MSSQL, Oracle, SQLite, MongoDB, Redis, Elasticsearch, Kafka, S3, BigQuery, ClickHouse, CSV, JSON/JSONL, Parquet

**17 transforms**: select/drop/rename columns, filter rows, drop/fill nulls, cast types, deduplicate, sort, limit, sample, text cleaning, text chunking, JSON flattening, merge/join, concatenate

**Visualization**: per-column statistics, histograms, value distributions, null analysis

### Python IDE
Full in-browser code editor with syntax highlighting, file management, and code execution.

- **CodeMirror editor** with Python, JSON, YAML, Markdown syntax highlighting
- Material Darker theme, line numbers, bracket matching, auto-close
- **Multi-tab** file editing with modified indicators
- **File tree** sidebar with recursive directory navigation
- **Console** with stdout (white), stderr (red), system messages (gray)
- **Ctrl+S** save, **Ctrl+Enter** run
- **pip install** packages from UI
- Project templates: Empty, LLM Fine-Tuning, Vision, Dataset, YOLO

### AI Assistant (Claude / GPT-4 / Ollama)
LLM-powered code editing, error fixing, and project guidance — integrated into the IDE.

- **Chat panel** — ask anything about your code in natural language
- **Auto-context** — AI sees your current file, project structure, and errors
- **Fix with AI** — when code fails, one click sends error to AI, get fix, apply to editor
- **Error line highlighting** — auto-scrolls to and highlights the error line
- **Explain / Improve / Generate** — quick action buttons
- **Apply Changes** — AI's code edits applied to editor with one button
- Supports: Claude (Anthropic), GPT-4o (OpenAI), Ollama (local), any OpenAI-compatible API

### Model Evaluation
One-click evaluation with automatic metric computation.

- **Classification**: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix heatmap
- **Regression**: R2, RMSE, MAE, MAPE
- **Text generation**: BLEU-1/2/3/4, ROUGE-1/2/L, exact match

### Model Deployment
Export and serve models with one click.

- **ONNX export** with dynamic batch axes
- **TorchScript export** (trace or script)
- **FastAPI inference server** generation (ONNX or PyTorch)
- **Dockerfile** generation for containerized deployment
- **Gradio demo app** generation with shareable link
- **30 ML library templates** with runnable code: YOLO, scikit-learn, Optuna, Whisper, Diffusers, vLLM, LangChain, and more

### Multi-User Collaboration
Work together in real-time.

- Named rooms with join/leave
- Live cursor broadcasting
- File locking (prevents edit conflicts)
- In-room chat
- User session management with unique colors

### System Monitoring
Real-time resource tracking in the header bar.

- CPU usage with color-coded bar
- RAM usage (used/total GB)
- Process memory
- GPU memory (CUDA) or MPS detection

## Installation

### Base (architecture builder + training + IDE)
```bash
pip install -e .
```

### With HuggingFace (transformers, peft, datasets, timm, diffusers, trl)
```bash
pip install -e ".[hf]"
```

### With Unsloth (fast LLM fine-tuning)
```bash
pip install -e ".[unsloth]"
```

### With RL (gymnasium, stable-baselines3)
```bash
pip install -e ".[rl]"
```

### With Robotics Physics (MuJoCo + PyBullet)
```bash
pip install -e ".[rl-robotics]"
```

### With Hardware Bridge (serial/USB for Arduino/ESP32)
```bash
pip install -e ".[hardware]"
```

### Everything
```bash
pip install -e ".[all]"
```

### For AI Assistant
Set one of these environment variables:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # For Claude
export OPENAI_API_KEY=sk-...          # For GPT-4
# Or use Ollama locally (no key needed)
```

## Usage

```bash
# Start the server
state-graph

# Open in browser
open http://localhost:8765
```

### From the UI

1. **Build architecture** — use templates or drag-drop layers
2. **Load data** — pick a dataset from the left panel
3. **Train** — click Build, then Train. Watch charts update live
4. **Evaluate** — go to Eval tab, click "Evaluate Now"
5. **Deploy** — go to Deploy tab, export ONNX or generate server code
6. **Fine-tune LLMs** — switch to HuggingFace mode, load model, apply LoRA, train
7. **Train RL** — click RL button, pick env + algorithm, train
8. **Build robots** — click Robot button, pick template, adjust joints, run physics
9. **Create datasets** — use Dataset Factory in left panel
10. **Write code** — click IDE button, create project, edit + run

## Architecture

```
state_graph/
  core/           # Neural net builder, training engine, metrics, schedulers
    engine.py     #   Training with graph + HF model support
    evaluator.py  #   Classification, regression, generation metrics
    deploy.py     #   ONNX, TorchScript, server generation
    libraries.py  #   30 ML library templates
  hf/             # HuggingFace integration
    hub.py        #   Model search, load, LoRA, freeze/unfreeze
    datasets.py   #   HF dataset loading and preprocessing
    unsloth.py    #   Fast fine-tuning: SFT, DPO, ORPO, KTO
  rl/             # Reinforcement learning
    engine.py     #   SB3 training with real-time metrics
  robotics/       # Robot simulation
    simulator.py  #   27 components, circuit solver, 3D scene
    hardware.py   #   Serial bridge, firmware generation
    physics_server.py  # MuJoCo server-side physics
  datasets/       # Dataset factory
    creator.py    #   21 templates, project management
    sources.py    #   Kaggle, URL, local file connectors
    converters.py #   YOLO<>COCO, CSV<>JSONL, Alpaca<>ShareGPT
  dataeng/        # Data engineering
    connectors.py #   15 database/file connectors
    pipeline.py   #   17 transforms, merge/join, stats
  workspace/      # IDE and project management
    manager.py    #   Project CRUD, file operations
    executor.py   #   Python code execution
    ai_assistant.py  # Claude/GPT-4/Ollama integration
  layers/         # Custom PyTorch layers
    custom.py     #   TransformerBlock, ResidualBlock, GLU, etc.
  server/         # Web server
    app.py        #   FastAPI + WebSocket (185 endpoints)
    collaboration.py  # Multi-user rooms, cursors, chat
  ui/
    index.html    #   Single-page app (Three.js, Chart.js, CodeMirror)
```

## API

185 REST + WebSocket endpoints. Full OpenAPI docs at `http://localhost:8765/docs`.

Key endpoint groups:
- `/api/graph/*` — architecture CRUD
- `/api/train/*` — training lifecycle
- `/api/hf/*` — HuggingFace operations
- `/api/unsloth/*` — Unsloth fine-tuning
- `/api/rl/*` — RL training
- `/api/robotics/*` — robot builder
- `/api/physics/*` — server-side physics
- `/api/hardware/*` — serial bridge
- `/api/dataeng/*` — data pipelines
- `/api/ds/*` — dataset factory
- `/api/workspace/*` — IDE / projects
- `/api/ai/*` — AI assistant
- `/api/eval/*` — model evaluation
- `/api/deploy/*` — model export
- `/api/collab/*` — collaboration
- `/ws` — WebSocket for real-time updates

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

168 tests covering core, server, datasets, robotics, data engineering, and workspace.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- A modern browser (Chrome, Firefox, Safari, Edge)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built for researchers who want to focus on math and architecture, not boilerplate code.
</p>
