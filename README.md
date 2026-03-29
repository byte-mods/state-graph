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
Build PyTorch models visually with drag-and-drop. 45+ layer types including Transformer, Mamba, RWKV, and Vision blocks. 31 architecture templates. Real-time training charts (loss, learning rate, gradients, weight distributions). Experiment comparison across architectures.

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
    app.py        #   FastAPI + WebSocket (285+ endpoints)
    collaboration.py  # Multi-user rooms, cursors, chat
  ui/
    index.html    #   Single-page app (Three.js, Chart.js, CodeMirror)
```

## API

285+ REST + WebSocket endpoints. Full OpenAPI docs at `http://localhost:8765/docs`.

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

# Tutorials

## Tutorial 1: Graph — Build & Train Models from Scratch

The **Graph** tab is the visual neural network builder. You design architectures by stacking layers, configure training parameters, load datasets, and train — all without writing code. This tutorial covers every component in detail with real-world examples.

---

### 1.1 Understanding the Graph Tab Layout

When you open StateGraph and click the **Graph** button in the top bar, you see three areas:

| Area | What It Contains |
|------|-----------------|
| **Left Sidebar** | Architecture templates, quick-add layer chips, layer palette, dataset selector, augmentation options, custom activation formulas |
| **Center Canvas** | Your model architecture — layer cards stacked vertically, drag handles to reorder, live gradient indicators during training |
| **Right Sidebar** | 7 tabs: **Config** (training params), **Inspector** (edit selected layer), **Metrics** (live training stats), **Compare** (experiment comparison), **Eval** (model evaluation), **Bench** (benchmarks), **Deploy** (export & serve) |

**Top Header Bar** contains action buttons:
- **Build** (Ctrl+B) — Compile your layer stack into a PyTorch model
- **Train** (Ctrl+T) — Start training
- **Stop** (Ctrl+Q) — Stop training
- **Save Run** — Save current experiment for comparison
- **Save / Load** — Export/import architecture as JSON
- **Export .py** (Ctrl+E) — Generate standalone Python code
- **Save Model / Load Model** — Save/load trained weights
- **Visualize** — Show architecture diagram with formulas
- **Reset** — Clear everything

---

### 1.2 Architecture Templates (Quick Start)

The left sidebar has an **Architecture Templates** section with 31 pre-built architectures. Click any template to instantly load it. Here are some key ones:

| Template | Description | Default Dataset |
|----------|-------------|-----------------|
| **MLP Classifier** | Simple feedforward network for tabular data | spiral |
| **Deep MLP** | Deeper network with residual blocks | spiral |
| **Gated Network** | Uses GLU gating mechanism | xor |
| **Wide & Shallow** | Single wide hidden layer, minimal depth | circles |
| **Transformer** | Self-attention transformer for tabular data | spiral |
| **Deep Transformer** | 4-layer transformer with larger model dim | blobs |
| **MNIST CNN** | Convolutional network for 28x28 images | mnist |
| **ViT Tiny** | Vision Transformer for image classification | cifar10 |
| **MobileNet-style** | Efficient CNN with depthwise separable convolutions | cifar10 |
| **Audio Classifier** | Conv1d + Transformer audio classification pipeline | random |
| **Autoencoder** | Conv autoencoder with encoder-decoder structure | mnist |
| **ResNet-style CNN** | CNN with residual convolution blocks | cifar10 |

> **Tip**: Templates auto-load the recommended dataset. After applying a template, just click **Build** then **Train**.

---

### 1.3 Quick Add Chips

Below templates, the **Quick Add** section provides one-click buttons for common layer combinations:

| Chip | What It Adds |
|------|-------------|
| `Linear+ReLU` | Linear layer with ReLU activation |
| `Linear+GELU` | Linear layer with GELU activation |
| `Conv2d+ReLU` | 2D convolution with ReLU |
| `BatchNorm1d` | 1D batch normalization |
| `BatchNorm2d` | 2D batch normalization |
| `Dropout` | Dropout regularization |
| `Flatten` | Flatten spatial dims to 1D |
| `MaxPool2d` | 2D max pooling |
| `LSTM` | Long short-term memory |
| `ResidualBlock` | Residual/skip connection block |
| `GLU` | Gated linear unit |
| `SwishLinear` | Linear with Swish activation |
| `TokenEmbedding` | Token embedding for transformers |
| `PositionalEncoding` | Sinusoidal position encoding |
| `TransformerBlock` | Full transformer encoder block |
| `SequencePool` | Pool sequence to single vector |

---

### 1.4 All Available Layers — Complete Reference

Click any layer from the **All Layers** palette in the left sidebar. A modal appears where you set parameters and activation.

#### 1.4.1 Standard PyTorch Layers (18 types)

**Linear Layers:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `Linear` | `in_features` (int), `out_features` (int) | Fully connected layer. The building block of MLPs |
| `Embedding` | `num_embeddings` (int), `embedding_dim` (int) | Lookup table for discrete tokens |

**Convolution Layers:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `Conv1d` | `in_channels`, `out_channels`, `kernel_size`, `stride`=1, `padding`=0 | 1D convolution (audio, time-series) |
| `Conv2d` | `in_channels`, `out_channels`, `kernel_size`, `stride`=1, `padding`=0 | 2D convolution (images) |
| `Conv3d` | `in_channels`, `out_channels`, `kernel_size`, `stride`=1, `padding`=0 | 3D convolution (video, volumetric) |
| `ConvTranspose2d` | `in_channels`, `out_channels`, `kernel_size`, `stride`=1, `padding`=0 | Transposed convolution (upsampling) |

**Normalization Layers:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `BatchNorm1d` | `num_features` (int) | Batch norm for 1D inputs (after Linear) |
| `BatchNorm2d` | `num_features` (int) | Batch norm for 2D inputs (after Conv2d) |
| `LayerNorm` | `normalized_shape` (int) | Layer normalization (transformers) |

**Regularization:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `Dropout` | `p` (float, 0.0–1.0) | Randomly zero elements during training |
| `Dropout2d` | `p` (float, 0.0–1.0) | Drop entire channels (after Conv2d) |

**Pooling Layers:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `MaxPool2d` | `kernel_size` (int) | Max pooling (downsamples spatial dims) |
| `AvgPool2d` | `kernel_size` (int) | Average pooling |
| `AdaptiveAvgPool2d` | `output_size` (int or tuple) | Adaptive average pooling to fixed output size |

**Recurrent Layers:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `LSTM` | `input_size`, `hidden_size`, `num_layers`=1, `batch_first`=True | Long Short-Term Memory |
| `GRU` | `input_size`, `hidden_size`, `num_layers`=1, `batch_first`=True | Gated Recurrent Unit |

**Attention:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `MultiheadAttention` | `embed_dim`, `num_heads` | Multi-head self/cross attention |

**Utility:**

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `Flatten` | *(none)* | Flatten all dims except batch to 1D vector |

#### 1.4.2 Custom Layers — Core (4 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `ResidualBlock` | `in_features` (int), `hidden_features` (int, optional) | Two linear layers with skip connection: `output = x + f(x)` |
| `SqueezeExcite` | `channels` (int), `reduction` (int, default=4) | Channel attention: learns per-channel importance weights |
| `GatedLinearUnit` | `in_features` (int), `out_features` (int) | GLU gating: splits into two halves, one gates the other via sigmoid |
| `SwishLinear` | `in_features` (int), `out_features` (int) | Linear + SiLU (Swish) activation in one layer |

#### 1.4.3 Custom Layers — Transformer Components (4 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `TokenEmbedding` | `in_features` (int), `d_model` (int), `seq_len` (int, optional) | Projects input features to model dimension, reshapes to sequence |
| `PositionalEncoding` | `d_model` (int), `max_len` (int, default=512), `dropout` (float, default=0.1) | Sinusoidal position encoding added to embeddings |
| `TransformerBlock` | `d_model` (int), `n_heads` (int, default=4), `ffn_dim` (int, optional), `dropout` (float, default=0.1) | Full encoder block: multi-head attention + FFN + LayerNorm + residual |
| `SequencePool` | `d_model` (int), `mode` ('mean'/'cls'/'max', default='mean') | Reduces sequence to single vector for classification head |

#### 1.4.4 Custom Layers — Vision (11 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `PatchEmbed` | `in_channels` (default=3), `d_model` (default=256), `patch_size` (default=16), `image_size` (default=224) | Splits image into patches and projects to embeddings (ViT) |
| `DepthwiseSeparableConv` | `in_channels`, `out_channels`, `kernel_size` (default=3), `stride` (default=1), `padding` (default=1) | MobileNet-style efficient convolution |
| `ChannelAttention` | `channels` (int), `reduction` (int, default=16) | SE-Net style channel attention module |
| `UpsampleBlock` | `in_channels`, `out_channels`, `scale_factor` (default=2) | Bilinear upsample + conv (for decoders) |
| `GlobalAvgPool` | *(none)* | Global average pooling: (B,C,H,W) → (B,C) |
| `Reshape` | `shape` (list[int]) | Reshape tensor to arbitrary shape |
| `ResConvBlock` | `in_channels`, `out_channels` (optional) | Residual 2D convolution block with batch norm |
| `DownBlock` | `in_channels`, `out_channels` | Conv + BatchNorm + ReLU + MaxPool2d (encoder) |
| `UpBlock` | `in_channels`, `out_channels` | ConvTranspose2d + BatchNorm + ReLU (decoder) |

#### 1.4.5 Custom Layers — Audio (3 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `MelSpectrogram` | `n_mels` (default=80), `n_fft` (default=1024), `hop_length` (default=256) | Converts raw audio waveform to mel spectrogram |
| `AudioConvBlock` | `in_channels`, `out_channels`, `kernel_size` (default=3), `stride` (default=1), `groups` (default=1) | 1D conv + BatchNorm + ReLU for audio features |
| `Transpose` | `dim0` (default=1), `dim1` (default=2) | Swap tensor dimensions (e.g., channels ↔ time) |

#### 1.4.6 Custom Layers — Video (2 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `Conv3dBlock` | `in_channels`, `out_channels`, `kernel_size` (default=3), `stride` (default=1), `padding` (default=1) | 3D conv + BatchNorm3d + ReLU for video |
| `TemporalPool` | `mode` ('mean'/'max', default='mean') | Pool across temporal dimension |

#### 1.4.7 Custom Layers — Diffusion/Generative (2 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `SinusoidalTimestepEmbed` | `d_model` (default=256) | Timestep embedding for diffusion models |
| `ConditionalBatchNorm2d` | `num_features` (int), `cond_dim` (default=256) | Conditional batch norm (condition modulates scale/shift) |

#### 1.4.8 Custom Layers — State-Space Models (2 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `SelectiveScan` | `d_model`, `d_state` (default=16), `expand` (default=2), `conv_kernel` (default=4), `dropout` (default=0.0) | Mamba core operation: selective state-space scanning |
| `MambaBlock` | `d_model`, `d_state` (default=16), `expand` (default=2), `conv_kernel` (default=4), `dropout` (default=0.0) | Full Mamba block with conv, SSM, and gating |

#### 1.4.9 Custom Layers — Advanced Architectures (7 types)

| Layer | Parameters | Description |
|-------|-----------|-------------|
| `RWKVBlock` | `d_model`, `n_heads` (default=1), `dropout` (default=0.0) | RWKV attention-free block (linear complexity) |
| `RetentionLayer` | `d_model`, `n_heads` (default=4), `dropout` (default=0.0) | Retentive self-attention (RetNet) |
| `RetNetBlock` | `d_model`, `n_heads` (default=4), `ffn_dim` (optional), `dropout` (default=0.0) | Full RetNet block with retention + FFN |
| `HyenaOperator` | `d_model`, `max_len` (default=2048), `order` (default=2), `dropout` (default=0.0) | Hyena sub-quadratic attention replacement |
| `HyenaBlock` | `d_model`, `max_len` (default=2048), `order` (default=2), `ffn_dim` (optional), `dropout` (default=0.0) | Full Hyena block with FFN |
| `XLSTM` | `d_model`, `hidden_size` (optional), `n_layers` (default=1), `dropout` (default=0.0) | Extended LSTM with exponential gating |
| `GatedLinearRecurrence` | `d_model`, `expand` (default=2), `dropout` (default=0.0) | Griffin-style gated linear recurrence |

---

### 1.5 Activations

When adding or editing any layer, you can attach a **post-activation function** from the dropdown:

| Activation | Formula | Best For |
|------------|---------|----------|
| `ReLU` | max(0, x) | Default choice for most networks |
| `LeakyReLU` | max(0.01x, x) | Avoids dying neurons |
| `GELU` | x · Φ(x) | Transformers, modern architectures |
| `SiLU` (Swish) | x · σ(x) | Efficient networks, MobileNet |
| `Sigmoid` | 1/(1+e^(-x)) | Binary classification output |
| `Tanh` | (e^x - e^(-x))/(e^x + e^(-x)) | RNNs, bounded output |
| `Softmax` | e^xi / Σe^xj | Multi-class output (usually auto-applied by loss) |
| `ELU` | x if x>0, α(e^x - 1) if x≤0 | Smoother than ReLU |
| `PReLU` | max(αx, x), α learned | Learnable leaky slope |
| `Mish` | x · tanh(softplus(x)) | Self-regularizing |

#### Custom Activation Formulas

In the left sidebar, expand **Custom Activation Formula** to create your own:

1. Enter a **Name** (e.g., `SquaredReLU`)
2. Enter a **PyTorch expression** using `x` as the input tensor (e.g., `torch.relu(x) ** 2`)
3. Click **Register**

**Preset formulas** available as one-click chips:
- `Swish` → `x * torch.sigmoid(x)`
- `HardSwish` → `x * torch.clamp(x + 3, 0, 6) / 6`
- `SquaredReLU` → `torch.relu(x) ** 2`
- `SoftSign` → `x / (1 + torch.abs(x))`
- `CELU` → `torch.where(x > 0, x, torch.exp(x) - 1)`
- `SiLU*2` → `2 * x * torch.sigmoid(x)`
- `Mish` → `x * torch.tanh(torch.nn.functional.softplus(x))`
- `LogSigmoid` → `torch.nn.functional.logsigmoid(x)`
- `SymReLU` → `torch.sign(x) * torch.relu(torch.abs(x) - 0.5)`
- `GaussAct` → `x * torch.exp(-x ** 2)`

---

### 1.6 Right Sidebar — Config Tab (Training Configuration)

The **Config** tab in the right sidebar controls how your model trains.

#### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Epochs** | 10 | Number of full passes through the training data |
| **Batch Size** | 32 | Samples per gradient update |
| **Learning Rate** | 0.001 | Step size for optimizer |

#### Optimizers

| Optimizer | Best For |
|-----------|----------|
| `Adam` | Default choice. Adaptive learning rates per parameter |
| `AdamW` | Adam + decoupled weight decay. Best for transformers |
| `SGD` | Simple, often best with momentum for CNNs |
| `RMSprop` | Good for RNNs and non-stationary objectives |
| `Adagrad` | Sparse data (NLP, embeddings) |

#### Loss Functions

| Loss | Use Case |
|------|----------|
| `CrossEntropyLoss` | Multi-class classification (default) |
| `MSELoss` | Regression |
| `L1Loss` | Regression (robust to outliers) |
| `BCELoss` | Binary classification (after sigmoid) |
| `BCEWithLogitsLoss` | Binary classification (raw logits, numerically stable) |
| `NLLLoss` | Classification (after log_softmax) |
| `SmoothL1Loss` | Regression (Huber loss variant) |
| `HuberLoss` | Regression (smooth transition between L1/L2) |
| `KLDivLoss` | Distribution matching (VAEs, knowledge distillation) |
| `CosineEmbeddingLoss` | Similarity/contrastive learning |

#### Learning Rate Schedulers

Select a scheduler from the dropdown — parameter fields appear automatically with defaults:

| Scheduler | Parameters | What It Does |
|-----------|-----------|--------------|
| `StepLR` | step_size=10, gamma=0.1 | Multiply LR by gamma every step_size epochs |
| `MultiStepLR` | milestones=[30,60,90], gamma=0.1 | Multiply LR by gamma at each milestone epoch |
| `ExponentialLR` | gamma=0.95 | Multiply LR by gamma every epoch |
| `CosineAnnealingLR` | T_max=50, eta_min=1e-6 | Cosine curve from initial LR to eta_min |
| `ReduceLROnPlateau` | factor=0.1, patience=10 | Reduce LR when val loss stops improving |
| `CyclicLR` | base_lr=1e-4, max_lr=0.01, step_size_up=200 | Cycle LR between base and max |
| `OneCycleLR` | max_lr=0.01, total_steps=1000 | Super-convergence: warmup → high LR → anneal |
| `CosineAnnealingWarmRestarts` | T_0=10, T_mult=2 | Cosine with periodic warm restarts |
| `LinearLR` | start_factor=0.1, total_iters=100 | Linear warmup from start_factor × LR to LR |
| `PolynomialLR` | total_iters=100, power=2.0 | Polynomial decay |

---

### 1.7 Right Sidebar — Inspector Tab

Click any layer card in the center canvas to inspect and edit it:

- **Layer Type** dropdown — change the layer type entirely
- **Parameter fields** — all parameters for the selected layer, with current values
- **Activation** dropdown — change or remove the post-activation
- **Update** — apply changes
- **Duplicate** — create a copy of this layer below it
- **Delete** — remove this layer from the graph

---

### 1.8 Right Sidebar — Metrics Tab (Live Training)

During training, this tab shows:

- **Progress bar** — current epoch/batch with percentage
- **Metric cards** — Loss, Step count, Train Accuracy, Val Accuracy, Perplexity
- **Epoch history table** — per-epoch train loss, val loss, train acc, val acc
- **Per-layer gradient norms** — color-coded health indicators:
  - **Red**: Exploding gradients (norm > 100)
  - **Orange**: High gradients
  - **Green**: Healthy
  - **Blue**: Low gradients
  - **Gray**: Vanishing gradients (norm < 1e-7)

**Bottom Charts Panel** (toggle with Ctrl+G):
1. **Training Loss** — loss over steps
2. **Learning Rate** — LR schedule visualization
3. **Gradient Norms** — per-layer gradient norms over time
4. **Weights** — weight statistics over time

---

### 1.9 Datasets

In the left sidebar **Dataset** section, select from:

#### Synthetic Datasets (2D)

| Dataset | Input Shape | Classes | Description |
|---------|-------------|---------|-------------|
| `spiral` | (2,) | 2 | Two interleaving spirals |
| `xor` | (2,) | 2 | XOR pattern — linearly inseparable |
| `circles` | (2,) | 2 | Concentric circles |
| `moons` | (2,) | 2 | Two half-moon shapes |
| `blobs` | (2,) | 4 | Four Gaussian clusters |
| `checkerboard` | (2,) | 2 | Checkerboard grid pattern |
| `regression_sin` | (1,) | — | Sine wave regression |
| `random` | (784,) | 10 | Random data (for testing pipelines) |

- **Samples**: Set from 100 to any number (default 1000)

#### Real Datasets (requires torchvision)

| Dataset | Input Shape | Classes | Samples | Description |
|---------|-------------|---------|---------|-------------|
| `mnist` | (1, 28, 28) | 10 | 70,000 | Handwritten digits |
| `fashion_mnist` | (1, 28, 28) | 10 | 70,000 | Clothing items |
| `cifar10` | (3, 32, 32) | 10 | 60,000 | Natural images (10 categories) |
| `cifar100` | (3, 32, 32) | 100 | 60,000 | Natural images (100 categories) |
| `svhn` | (3, 32, 32) | 10 | ~600,000 | Street view house numbers |

#### Data Augmentations

Expand the **Augmentation** section and check any combination:

| Augmentation | Parameter | Description |
|-------------|-----------|-------------|
| `gaussian_noise` | sigma=0.1 | Add Gaussian noise |
| `dropout_noise` | p=0.1 | Randomly zero elements |
| `scaling` | min=0.9, max=1.1 | Random scale factor |
| `mixup` | alpha=0.2 | Blend pairs of samples |
| `cutout` | size=8 | Zero-out random square patch (images only) |
| `random_flip` | — | Random horizontal flip (images only) |

---

### 1.10 The Complete Workflow

The basic workflow for every model:

```
1. Select/build architecture  →  2. Load dataset  →  3. Configure training
→  4. Build model  →  5. Train  →  6. Evaluate  →  7. Save/Export
```

---

## Example 1: MLP Classifier (Tabular Data — Beginner)

**Goal**: Classify 2D spiral data with a feedforward network.

### Step 1 — Load Template or Build Manually

**Option A: Use template**
- Left sidebar → Architecture Templates → Click **"MLP Classifier"**

**Option B: Build manually**
- Click **Linear+ReLU** quick-add chip → Set `in_features=2`, `out_features=64`
- Click **BatchNorm1d** → Set `num_features=64`
- Click **Dropout** → Set `p=0.3`
- Click **Linear+ReLU** → Set `in_features=64`, `out_features=32`
- From All Layers, add **Linear** → Set `in_features=32`, `out_features=2`, activation=None

### Step 2 — Load Dataset
- Left sidebar → Dataset → **Synthetic (2D)** → Select **spiral**
- Samples: **1000**
- Click **Load Dataset**
- Info shows: `spiral | train: 800 / val: 200 | shape: (2,) | 2 classes`

### Step 3 — Configure Training (Right Sidebar → Config)
| Setting | Value |
|---------|-------|
| Epochs | **50** |
| Batch Size | **32** |
| Learning Rate | **0.001** |
| Optimizer | **Adam** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **CosineAnnealingLR** (T_max=50, eta_min=1e-6) |

Click **Apply Config**.

### Step 4 — Build & Train
1. Click **Build** → Status shows parameter count and device
2. Click **Train** → Watch live charts:
   - Loss drops from ~0.7 to ~0.01
   - Train accuracy climbs to ~99%
   - Val accuracy reaches ~97-99%
3. Gradient norm bars on layer cards show healthy green

### Step 5 — Evaluate
- Right sidebar → **Eval** tab → Click **"Auto-Evaluate Model"**
- See per-class precision, recall, F1 and confusion matrix

### Step 6 — Save
- Click **Save Run** to save for comparison
- Click **Export .py** to get standalone Python code

---

## Example 2: MNIST CNN (Image Classification)

**Goal**: Classify handwritten digits (0-9) with a convolutional network.

### Architecture
Use template **"MNIST CNN"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `Conv2d` | in_channels=1, out_channels=32, kernel_size=3, padding=1 | ReLU |
| 1 | `MaxPool2d` | kernel_size=2 | — |
| 2 | `Conv2d` | in_channels=32, out_channels=64, kernel_size=3, padding=1 | ReLU |
| 3 | `MaxPool2d` | kernel_size=2 | — |
| 4 | `Flatten` | — | — |
| 5 | `Linear` | in_features=3136, out_features=128 | ReLU |
| 6 | `Dropout` | p=0.5 | — |
| 7 | `Linear` | in_features=128, out_features=10 | — |

> **Note**: `in_features=3136` comes from 64 channels × 7 × 7 (after two 2× poolings of 28×28).

### Dataset
- **Real** → **mnist** → Load Dataset

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **15** |
| Batch Size | **64** |
| Learning Rate | **0.001** |
| Optimizer | **Adam** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **StepLR** (step_size=5, gamma=0.5) |

### Expected Results
- Train accuracy: **~99.5%**
- Val accuracy: **~99.0%**
- Loss: drops from ~2.3 to ~0.02

---

## Example 3: Vision Transformer (ViT) for CIFAR-10

**Goal**: Classify 32×32 color images using self-attention (no convolutions in the core).

### Architecture
Use template **"ViT Tiny"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `PatchEmbed` | in_channels=3, d_model=128, patch_size=4, image_size=32 | — |
| 1 | `TransformerBlock` | d_model=128, n_heads=4, dropout=0.1 | — |
| 2 | `TransformerBlock` | d_model=128, n_heads=4, dropout=0.1 | — |
| 3 | `TransformerBlock` | d_model=128, n_heads=4, dropout=0.1 | — |
| 4 | `SequencePool` | d_model=128, mode='cls' | — |
| 5 | `Linear` | in_features=128, out_features=10 | — |

**How it works**:
1. `PatchEmbed` splits each 32×32 image into 8×8=64 patches of 4×4 pixels, projects each to 128-dim
2. Three `TransformerBlock`s apply self-attention across the 64 patch tokens
3. `SequencePool(mode='cls')` takes the CLS token representation
4. Final `Linear` maps to 10 classes

### Dataset
- **Real** → **cifar10** → Load Dataset

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **30** |
| Batch Size | **64** |
| Learning Rate | **0.0003** |
| Optimizer | **AdamW** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **CosineAnnealingLR** (T_max=30, eta_min=1e-6) |

### Augmentations
- Enable **random_flip** and **cutout** (size=4) for better generalization

### Expected Results
- Val accuracy: **~75-82%** (small ViT, no pretraining)
- Watch gradient norms — transformers should show uniform green across layers

---

## Example 4: ResNet-Style CNN for CIFAR-10

**Goal**: Build a residual convolutional network — skip connections prevent vanishing gradients.

### Architecture
Use template **"ResNet-style CNN"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `Conv2d` | in_channels=3, out_channels=64, kernel_size=3, padding=1 | ReLU |
| 1 | `ResConvBlock` | in_channels=64, out_channels=64 | — |
| 2 | `ResConvBlock` | in_channels=64, out_channels=128 | — |
| 3 | `ResConvBlock` | in_channels=128, out_channels=256 | — |
| 4 | `GlobalAvgPool` | — | — |
| 5 | `Linear` | in_features=256, out_features=10 | — |

### Dataset
- **Real** → **cifar10** → Load Dataset

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **40** |
| Batch Size | **64** |
| Learning Rate | **0.01** |
| Optimizer | **SGD** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **CosineAnnealingLR** (T_max=40, eta_min=1e-5) |

### Augmentations
- Enable **random_flip**, **cutout** (size=8)

### Expected Results
- Val accuracy: **~85-90%**
- ResConvBlocks show stable gradients thanks to skip connections

---

## Example 5: MobileNet-Style Efficient CNN

**Goal**: Build a lightweight model using depthwise separable convolutions (ideal for edge/mobile deployment).

### Architecture
Use template **"MobileNet-style"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `Conv2d` | in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1 | ReLU |
| 1 | `DepthwiseSeparableConv` | in_channels=32, out_channels=64 | ReLU |
| 2 | `DepthwiseSeparableConv` | in_channels=64, out_channels=128, stride=2 | ReLU |
| 3 | `DepthwiseSeparableConv` | in_channels=128, out_channels=256, stride=2 | ReLU |
| 4 | `GlobalAvgPool` | — | — |
| 5 | `Linear` | in_features=256, out_features=10 | — |

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **30** |
| Batch Size | **128** |
| Learning Rate | **0.01** |
| Optimizer | **SGD** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **CosineAnnealingLR** (T_max=30) |

### Why It Matters
- ~3-5x fewer parameters than standard Conv2d networks
- Perfect for ONNX export → mobile/edge deployment (see Deploy tab)

---

## Example 6: Transformer Language Model (Nano LLM)

**Goal**: Build a small GPT-style transformer from scratch for learning how LLMs work.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=256, d_model=128, seq_len=64 | — |
| 1 | `PositionalEncoding` | d_model=128, max_len=512, dropout=0.1 | — |
| 2 | `TransformerBlock` | d_model=128, n_heads=4, ffn_dim=512, dropout=0.1 | — |
| 3 | `TransformerBlock` | d_model=128, n_heads=4, ffn_dim=512, dropout=0.1 | — |
| 4 | `TransformerBlock` | d_model=128, n_heads=4, ffn_dim=512, dropout=0.1 | — |
| 5 | `TransformerBlock` | d_model=128, n_heads=4, ffn_dim=512, dropout=0.1 | — |
| 6 | `SequencePool` | d_model=128, mode='mean' | — |
| 7 | `Linear` | in_features=128, out_features=10 | — |

**Scaling up** — to build a larger LLM:
- Increase `d_model` to 256, 512, or 768
- Increase `n_heads` to 8 or 16 (must divide d_model evenly)
- Set `ffn_dim` to 4× d_model (standard ratio)
- Add more TransformerBlock layers (6, 12, or 24)
- Increase `seq_len` for longer context

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **50** |
| Batch Size | **32** |
| Learning Rate | **0.0003** |
| Optimizer | **AdamW** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **CosineAnnealingWarmRestarts** (T_0=10, T_mult=2) |

---

## Example 7: Mamba (State-Space Model)

**Goal**: Build a Mamba-style model — the attention-free architecture that matches Transformer quality at linear complexity.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=8 | — |
| 1 | `MambaBlock` | d_model=64, d_state=16, expand=2, conv_kernel=4, dropout=0.1 | — |
| 2 | `MambaBlock` | d_model=64, d_state=16, expand=2, conv_kernel=4, dropout=0.1 | — |
| 3 | `MambaBlock` | d_model=64, d_state=16, expand=2, conv_kernel=4, dropout=0.1 | — |
| 4 | `SequencePool` | d_model=64, mode='mean' | — |
| 5 | `Linear` | in_features=64, out_features=2 | — |

**Key parameters**:
- `d_state=16` — state dimension (higher = more memory per token)
- `expand=2` — inner dimension multiplier
- `conv_kernel=4` — local convolution kernel size

### Dataset & Config
- Dataset: **spiral** (1000 samples)
- Epochs: **40**, Batch Size: **32**, LR: **0.001**, Optimizer: **AdamW**
- Loss: **CrossEntropyLoss**, Scheduler: **CosineAnnealingLR** (T_max=40)

---

## Example 8: RWKV Block (Linear Attention)

**Goal**: Build an RWKV-style model — O(1) inference per token, no attention matrix.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=8 | — |
| 1 | `RWKVBlock` | d_model=64, n_heads=1, dropout=0.1 | — |
| 2 | `RWKVBlock` | d_model=64, n_heads=1, dropout=0.1 | — |
| 3 | `SequencePool` | d_model=64, mode='mean' | — |
| 4 | `Linear` | in_features=64, out_features=2 | — |

### Training Config
Same as Mamba example. RWKV trains with parallel scan but infers with constant memory.

---

## Example 9: RetNet (Retentive Network)

**Goal**: Build a RetNet — combines the training parallelism of Transformers with the O(1) inference of RNNs.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=8 | — |
| 1 | `RetNetBlock` | d_model=64, n_heads=4, dropout=0.1 | — |
| 2 | `RetNetBlock` | d_model=64, n_heads=4, dropout=0.1 | — |
| 3 | `RetNetBlock` | d_model=64, n_heads=4, dropout=0.1 | — |
| 4 | `SequencePool` | d_model=64, mode='mean' | — |
| 5 | `Linear` | in_features=64, out_features=2 | — |

---

## Example 10: Hyena (Sub-Quadratic Attention)

**Goal**: Replace attention with long convolutions — handles sequences up to 100K+ tokens.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=16 | — |
| 1 | `HyenaBlock` | d_model=64, max_len=2048, order=2, dropout=0.1 | — |
| 2 | `HyenaBlock` | d_model=64, max_len=2048, order=2, dropout=0.1 | — |
| 3 | `SequencePool` | d_model=64, mode='mean' | — |
| 4 | `Linear` | in_features=64, out_features=2 | — |

---

## Example 11: xLSTM (Extended LSTM)

**Goal**: Build with the modernized LSTM that uses exponential gating and matrix memory.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=8 | — |
| 1 | `XLSTM` | d_model=64, n_layers=2, dropout=0.1 | — |
| 2 | `SequencePool` | d_model=64, mode='mean' | — |
| 3 | `Linear` | in_features=64, out_features=2 | — |

---

## Example 12: Griffin (Gated Linear Recurrence)

**Goal**: Build Google's Griffin architecture — gated linear recurrences for efficient sequence modeling.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=64, seq_len=8 | — |
| 1 | `GatedLinearRecurrence` | d_model=64, expand=2, dropout=0.1 | — |
| 2 | `GatedLinearRecurrence` | d_model=64, expand=2, dropout=0.1 | — |
| 3 | `SequencePool` | d_model=64, mode='mean' | — |
| 4 | `Linear` | in_features=64, out_features=4 | — |

### Dataset & Config
- Dataset: **blobs** (4 classes)
- Epochs: **40**, Batch Size: **32**, LR: **0.001**, Optimizer: **AdamW**

---

## Example 13: Convolutional Autoencoder (Generative)

**Goal**: Learn compressed representations of images — encoder compresses, decoder reconstructs.

### Architecture
Use template **"Autoencoder"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `DownBlock` | in_channels=1, out_channels=32 | — |
| 1 | `DownBlock` | in_channels=32, out_channels=64 | — |
| 2 | `ResConvBlock` | in_channels=64 | — |
| 3 | `UpBlock` | in_channels=64, out_channels=32 | — |
| 4 | `UpBlock` | in_channels=32, out_channels=1 | — |

### Dataset & Config
- Dataset: **Real** → **mnist**
- Epochs: **20**, Batch Size: **64**, LR: **0.001**, Optimizer: **Adam**
- Loss: **MSELoss** (reconstruction loss, not classification)

---

## Example 14: Nano Diffusion Model

**Goal**: Build a tiny diffusion model architecture with timestep conditioning.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `SinusoidalTimestepEmbed` | d_model=128 | — |
| 1 | `Linear` | in_features=128, out_features=256 | GELU |
| 2 | `Linear` | in_features=256, out_features=784 | — |
| 3 | `Reshape` | shape=[-1, 1, 28, 28] | — |
| 4 | `ConditionalBatchNorm2d` | num_features=1, cond_dim=128 | — |
| 5 | `Conv2d` | in_channels=1, out_channels=16, kernel_size=3, padding=1 | GELU |
| 6 | `ResConvBlock` | in_channels=16, out_channels=16 | — |
| 7 | `Conv2d` | in_channels=16, out_channels=1, kernel_size=3, padding=1 | — |

**How it works**:
1. `SinusoidalTimestepEmbed` encodes the diffusion timestep
2. Two `Linear` layers map timestep embedding to image space
3. `ConditionalBatchNorm2d` conditions the image features on timestep
4. Conv layers refine the output
5. During training, the model learns to denoise images at each timestep

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **50** |
| Batch Size | **64** |
| Learning Rate | **0.0002** |
| Optimizer | **AdamW** |
| Loss | **MSELoss** |
| Scheduler | **CosineAnnealingLR** (T_max=50) |

---

## Example 15: Audio Classification Pipeline

**Goal**: Classify audio using mel spectrograms + Conv1d + Transformer attention.

### Architecture
Use template **"Audio Classifier"** or build manually:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `MelSpectrogram` | n_mels=40, n_fft=512, hop_length=128 | — |
| 1 | `AudioConvBlock` | in_channels=40, out_channels=64 | — |
| 2 | `AudioConvBlock` | in_channels=64, out_channels=128, stride=2 | — |
| 3 | `AudioConvBlock` | in_channels=128, out_channels=256, stride=2 | — |
| 4 | `Transpose` | dim0=1, dim1=2 | — |
| 5 | `TransformerBlock` | d_model=256, n_heads=4 | — |
| 6 | `SequencePool` | d_model=256, mode='mean' | — |
| 7 | `Linear` | in_features=256, out_features=10 | — |

**How it works**:
1. `MelSpectrogram` converts raw waveform to mel-frequency spectrogram
2. `AudioConvBlock` layers extract local features with stride=2 downsampling
3. `Transpose` flips (B, channels, time) → (B, time, channels) for transformer input
4. `TransformerBlock` applies self-attention across time steps
5. `SequencePool` + `Linear` produces classification

### Training Config
| Setting | Value |
|---------|-------|
| Epochs | **30** |
| Batch Size | **32** |
| Learning Rate | **0.0005** |
| Optimizer | **AdamW** |
| Loss | **CrossEntropyLoss** |
| Scheduler | **OneCycleLR** (max_lr=0.005, total_steps=1000) |

---

## Example 16: Video Classification (3D CNN)

**Goal**: Classify video clips using 3D convolutions across spatial and temporal dimensions.

### Architecture (Manual Build)

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `Conv3dBlock` | in_channels=3, out_channels=32, kernel_size=3, padding=1 | — |
| 1 | `Conv3dBlock` | in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1 | — |
| 2 | `Conv3dBlock` | in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1 | — |
| 3 | `TemporalPool` | mode='mean' | — |
| 4 | `GlobalAvgPool` | — | — |
| 5 | `Linear` | in_features=128, out_features=10 | — |

**How it works**:
1. `Conv3dBlock` applies 3D convolutions across (time, height, width)
2. stride=2 downsamples spatially
3. `TemporalPool(mode='mean')` collapses the temporal dimension
4. `GlobalAvgPool` collapses spatial dims → (B, 128)
5. `Linear` classifies

---

## Example 17: Deep Transformer (Advanced LLM-style, 4+ layers)

**Goal**: Build a deeper transformer with larger dimensions — closer to real LLM architecture.

### Architecture
Use template **"Deep Transformer"** or build:

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `TokenEmbedding` | in_features=2, d_model=256, seq_len=16 | — |
| 1 | `PositionalEncoding` | d_model=256, max_len=512, dropout=0.1 | — |
| 2 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 3 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 4 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 5 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 6 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 7 | `TransformerBlock` | d_model=256, n_heads=8, ffn_dim=1024, dropout=0.1 | — |
| 8 | `SequencePool` | d_model=256, mode='mean' | — |
| 9 | `Linear` | in_features=256, out_features=64 | ReLU |
| 10 | `Dropout` | p=0.1 | — |
| 11 | `Linear` | in_features=64, out_features=4 | — |

### Dataset & Config
- Dataset: **blobs** (4 classes, 2000 samples)
- Epochs: **40**, Batch Size: **32**, LR: **0.0001**
- Optimizer: **AdamW**, Loss: **CrossEntropyLoss**
- Scheduler: **CosineAnnealingWarmRestarts** (T_0=10, T_mult=2)

### LLM Architecture Scaling Guide

| Scale | d_model | n_heads | ffn_dim | Layers | Params |
|-------|---------|---------|---------|--------|--------|
| Nano | 64 | 4 | 256 | 2 | ~100K |
| Tiny | 128 | 4 | 512 | 4 | ~1M |
| Small | 256 | 8 | 1024 | 6 | ~10M |
| Medium | 512 | 8 | 2048 | 8 | ~50M |
| Base | 768 | 12 | 3072 | 12 | ~125M |

> **Rule of thumb**: `ffn_dim = 4 × d_model`, `n_heads` must divide `d_model` evenly.

---

## Example 18: Gated Network with GLU

**Goal**: Build a network using gated linear units — the gating mechanism used in PaLM, LLaMA, and modern LLMs.

### Architecture

| # | Layer | Parameters | Activation |
|---|-------|-----------|------------|
| 0 | `Linear` | in_features=2, out_features=64 | ReLU |
| 1 | `GatedLinearUnit` | in_features=64, out_features=32 | — |
| 2 | `SwishLinear` | in_features=32, out_features=32 | — |
| 3 | `Linear` | in_features=32, out_features=2 | — |

### Dataset & Config
- Dataset: **xor** → Epochs: **30**, LR: **0.001**, Optimizer: **Adam**

---

## Post-Training: Evaluate, Compare, and Deploy

### Evaluate (Eval Tab)
After training completes:
1. Go to **Eval** tab in right sidebar
2. Click **"Auto-Evaluate Model"**
3. See: accuracy, F1 (macro/weighted), per-class precision/recall, confusion matrix heatmap

### Compare Experiments (Compare Tab)
1. After each training run, click **Save Run** in the header
2. Modify architecture or config → train again → **Save Run**
3. Go to **Compare** tab → see all saved experiments
4. View overlay charts comparing loss curves and accuracy across experiments

### Deploy (Deploy Tab)
| Action | How |
|--------|-----|
| **Export ONNX** | Deploy tab → Export Model → ONNX |
| **Export TorchScript** | Deploy tab → Export Model → TorchScript |
| **FastAPI Server** | Deploy tab → Generate Server → ONNX Server or PyTorch Server |
| **Gradio Demo** | Deploy tab → Generate Server → Gradio Demo |
| **Dockerfile** | Deploy tab → Generate Server → Dockerfile |
| **Hyperparameter Search** | Deploy tab → Grid/Random search with configurable trials |
| **AutoML/NAS** | Deploy tab → Neural Architecture Search with trial budget |
| **Distributed Training** | Deploy tab → Accelerate, DeepSpeed, FSDP, or DDP scripts |
| **Cloud Training** | Deploy tab → SageMaker, Vertex AI, Modal, or RunPod scripts |

---

### Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Ctrl+B` | Build model |
| `Ctrl+T` | Start training |
| `Ctrl+Q` | Stop training |
| `Ctrl+E` | Export Python code |
| `Ctrl+S` | Save experiment |
| `Ctrl+G` | Toggle charts panel |

---

<p align="center">
  Built for researchers who want to focus on math and architecture, not boilerplate code.
</p>

---

## Tutorial 2: LLM Builder — Build & Train Language Models from Scratch

The **LLM Builder** lets you create any language model architecture from scratch — GPT-2, Llama, Claude-style, Mixtral MoE, Mamba, or entirely novel designs. No pretrained weights needed.

---

### 2.1 Using Blueprints (Fastest Way)

Blueprints are complete architecture configurations. Pick one, choose a scale, and build.

**Example: Build a Llama-style LLM**

```
UI: LLM Builder → Model Blueprints → Select "Llama (from scratch)"
     → Scale: "small" → Click "Build This Model"
```

```bash
# API equivalent
curl -X POST http://localhost:8765/api/llm/blueprint/build \
  -H "Content-Type: application/json" \
  -d '{
    "blueprint": "llama_scratch",
    "scale": "small"
  }'

# Response: {status: "ok", parameters: {total_M: "25.2M"}, ...}
```

**Available blueprints (18 total):**

| Blueprint | Category | What It Builds |
|-----------|----------|---------------|
| `gpt2_scratch` | Text LLM | GPT-2 with LayerNorm + Standard FFN |
| `llama_scratch` | Text LLM | Llama with RMSNorm + SwiGLU + GQA + RoPE |
| `claude_scratch` | Text LLM | Claude-style with extended context RoPE (base=500K) |
| `mistral_scratch` | Text LLM | Mistral with Sliding Window Attention |
| `mixtral_scratch` | Text LLM | Mixtral MoE (8 experts, top-2 routing) |
| `deepseek_scratch` | Text LLM | DeepSeek (dense early layers + MoE deep layers) |
| `t5_scratch` | Enc-Dec | T5 encoder-decoder with cross-attention |
| `mamba_scratch` | Alternative | Mamba SSM — O(n) linear-time, no attention |
| `jamba_scratch` | Alternative | Jamba hybrid (alternating Mamba + Attention) |
| `rwkv_scratch` | Alternative | RWKV — trains like transformer, runs like RNN |
| `retnet_scratch` | Alternative | RetNet with multi-scale retention |
| `gemini_scratch` | Multimodal | Gemini-style: text + image + audio + video + MoE |
| `llava_scratch` | Multimodal | LLaVA vision-language model |
| `veo3_scratch` | Video Gen | VeO3-style text-to-video diffusion |
| `stable_diffusion_scratch` | Image Gen | Latent diffusion (VAE + UNet + text encoder) |
| `nano_banana` | Experimental | Ultra-compact hybrid (Mamba + Attention + MoE) |
| `custom_from_scratch` | Experimental | Blank canvas with composable blocks |
| `adaptive_depth_scratch` | Efficient | Early-exit LLM (skips layers when confident) |

Each blueprint has **scalable configs**: `nano`, `micro`, `small`, `medium`, `large`, `xl`.

---

### 2.2 Example: Build Llama 4 from Scratch

This walks through building a Llama 4-class model (dense + MoE hybrid, GQA, RoPE, SwiGLU) from scratch, creating a dataset, training, and generating text.

#### Step 1: Build the Architecture

```bash
# Build a Llama 4-style model: dense layers + MoE layers, GQA, long context
curl -X POST http://localhost:8765/api/llm/build \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 32000,
    "d_model": 1024,
    "n_layers": 16,
    "n_heads": 16,
    "n_kv_heads": 4,
    "max_len": 8192,
    "dropout": 0.0,
    "norm_type": "rmsnorm",
    "ffn_type": "swiglu",
    "use_flash": true,
    "tie_weights": true,
    "use_moe": true,
    "n_experts": 8,
    "moe_top_k": 2,
    "rope_base": 500000
  }'
```

**What each parameter does (Llama 4 architecture):**

| Parameter | Value | Why |
|-----------|-------|-----|
| `d_model: 1024` | Hidden dimension | Larger = more capacity |
| `n_layers: 16` | Decoder blocks | More layers = deeper reasoning |
| `n_heads: 16` | Attention heads | Parallel attention patterns |
| `n_kv_heads: 4` | GQA key-value heads | 4x less KV cache memory (16/4 = 4 groups) |
| `norm_type: rmsnorm` | RMS normalization | Faster than LayerNorm, same quality |
| `ffn_type: swiglu` | Gated FFN | `FFN(x) = (SiLU(xW_gate) * xW_up) @ W_down` |
| `use_moe: true` | Mixture of Experts | 8 FFN experts, only 2 active per token |
| `rope_base: 500000` | Extended RoPE | Supports 8K+ context without quality loss |

**Or use the blueprint with overrides:**

```bash
# Same thing, but starting from the claude_scratch blueprint
curl -X POST http://localhost:8765/api/llm/blueprint/build \
  -H "Content-Type: application/json" \
  -d '{
    "blueprint": "claude_scratch",
    "scale": "medium",
    "overrides": {
      "use_moe": true,
      "n_experts": 8,
      "moe_top_k": 2,
      "moe_layers": [4,5,6,7,8,9,10,11,12,13,14,15]
    }
  }'
```

#### Step 2: Train a Custom Tokenizer

```bash
# Train a BPE tokenizer on your data
curl -X POST http://localhost:8765/api/llm/tokenizer/train \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your training corpus goes here... (paste thousands of words)",
    "algorithm": "bpe",
    "vocab_size": 32000,
    "min_frequency": 2
  }'
```

#### Step 3: Train the Model

```bash
# Start training with text data
curl -X POST http://localhost:8765/api/llm/train \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your full training text corpus...",
    "tokenizer": "char",
    "max_len": 256,
    "epochs": 20,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "scheduler": "cosine",
    "scheduler_params": {"T_max": 20}
  }'
```

Training broadcasts real-time metrics via WebSocket:
- Loss per epoch
- Learning rate schedule
- Validation loss
- Training speed (tokens/sec)

#### Step 4: Generate Text

```bash
curl -X POST http://localhost:8765/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of artificial intelligence",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_k": 50
  }'
```

#### Step 5: Modify the Architecture After Building

```bash
# Add a layer at position 8
curl -X POST http://localhost:8765/api/llm/modify \
  -H "Content-Type: application/json" \
  -d '{
    "action": "add_layer",
    "position": 8,
    "config": {"n_heads": 16, "n_kv_heads": 4, "norm_type": "rmsnorm", "ffn_type": "swiglu"}
  }'

# Freeze first 4 layers (for fine-tuning)
curl -X POST http://localhost:8765/api/llm/modify \
  -d '{"action": "freeze_layer", "index": 0}'
curl -X POST http://localhost:8765/api/llm/modify \
  -d '{"action": "freeze_layer", "index": 1}'
curl -X POST http://localhost:8765/api/llm/modify \
  -d '{"action": "freeze_layer", "index": 2}'
curl -X POST http://localhost:8765/api/llm/modify \
  -d '{"action": "freeze_layer", "index": 3}'
```

---

### 2.3 Example: Build a Gemini-style Multimodal Model

```bash
# Gemini-style: text + image + audio + video with MoE
curl -X POST http://localhost:8765/api/llm/blueprint/build \
  -d '{
    "blueprint": "gemini_scratch",
    "scale": "small",
    "overrides": {
      "modalities": ["text", "image", "audio", "video"],
      "n_image_tokens": 64,
      "n_video_tokens": 128
    }
  }'
```

The model automatically includes:
- Image encoder (ViT patch embedding) + Perceiver Resampler (64 tokens)
- Audio encoder (mel spectrogram + conv) + Perceiver Resampler (32 tokens)
- Video encoder (per-frame patches + temporal pos) + Perceiver Resampler (128 tokens)
- MoE decoder blocks (8 experts, top-2)
- All modality tokens prepended to text for unified processing

---

### 2.4 Example: Build a VeO3-style Video Generation Model

```bash
# Step 1: Build all components
curl -X POST http://localhost:8765/api/llm/blueprint/build-components \
  -d '{
    "blueprint": "veo3_scratch",
    "scale": "small"
  }'

# Step 2: Train the VAE first (encodes video to latent space)
curl -X POST http://localhost:8765/api/diffusion/train-vae \
  -d '{
    "epochs": 20,
    "batch_size": 4,
    "image_size": 64,
    "learning_rate": 1e-4,
    "kl_weight": 0.01
  }'

# Step 3: Train the UNet denoiser (in latent space)
curl -X POST http://localhost:8765/api/diffusion/train \
  -d '{
    "mode": "image",
    "epochs": 50,
    "batch_size": 4,
    "image_size": 64,
    "use_vae": true,
    "n_steps": 1000,
    "schedule": "cosine"
  }'

# Step 4: Generate
curl -X POST http://localhost:8765/api/diffusion/generate \
  -d '{"n_images": 4, "n_steps": 50, "image_size": 64}'
```

---

## Tutorial 3: Create Datasets from Scratch

### 3.1 Using the Dataset Factory

```bash
# Create a text instruction dataset
curl -X POST http://localhost:8765/api/ds/create \
  -d '{
    "name": "my_instruct_data",
    "template": "instruct",
    "format": "alpaca"
  }'

# Add samples
curl -X POST http://localhost:8765/api/ds/my_instruct_data/add \
  -d '{
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing uses qubits that can exist in superposition..."
  }'
```

**21 dataset templates:**

| Template | Category | Output Format |
|----------|----------|--------------|
| `instruct` | Text | Alpaca (instruction/input/output) |
| `chat` | Text | ShareGPT (multi-turn conversations) |
| `qa` | Text | Question/answer pairs |
| `summarization` | Text | Article/summary pairs |
| `translation` | Text | Source/target language pairs |
| `sentiment` | Text | Text + positive/negative/neutral label |
| `ner` | Text | Tokens + BIO entity tags |
| `classification` | Image | Image path + class label |
| `detection` | Image | Image + YOLO bounding boxes |
| `segmentation` | Image | Image + COCO segmentation masks |
| `captioning` | Image | Image + text description |
| `asr` | Audio | Audio file + transcription |
| `tts` | Audio | Text + audio file |
| `audio_classification` | Audio | Audio + label |
| `video_classification` | Video | Video + label |
| `video_captioning` | Video | Video + text description |
| `vqa` | Multimodal | Image + question + answer |
| `tool_calling` | Text | Instruction + tool schema + expected call |
| `preference` | Text | Prompt + chosen + rejected (for DPO/RLHF) |
| `reward` | Text | Prompt + response + score |
| `custom` | Any | Define your own schema |

### 3.2 Export Formats

```bash
# Export to different formats
curl -X POST http://localhost:8765/api/ds/my_instruct_data/export \
  -d '{"format": "jsonl", "path": "./data/train.jsonl"}'

curl -X POST http://localhost:8765/api/ds/my_instruct_data/export \
  -d '{"format": "alpaca", "path": "./data/alpaca.json"}'

curl -X POST http://localhost:8765/api/ds/my_instruct_data/export \
  -d '{"format": "sharegpt", "path": "./data/sharegpt.json"}'
```

### 3.3 Load Data from External Sources

```bash
# From HuggingFace
curl -X POST http://localhost:8765/api/hf/datasets/load \
  -d '{"dataset_id": "tatsu-lab/alpaca", "split": "train"}'

# From Kaggle
curl -X POST http://localhost:8765/api/ds/sources/kaggle \
  -d '{"dataset": "username/dataset-name"}'

# From URL
curl -X POST http://localhost:8765/api/ds/sources/download \
  -d '{"url": "https://example.com/data.csv"}'
```

### 3.4 Convert Between Formats

```bash
# YOLO to COCO
curl -X POST http://localhost:8765/api/ds/convert \
  -d '{"from": "yolo", "to": "coco", "input_path": "./yolo_labels/", "output_path": "./coco.json"}'

# Alpaca to ShareGPT
curl -X POST http://localhost:8765/api/ds/convert \
  -d '{"from": "alpaca", "to": "sharegpt", "input_path": "./alpaca.json", "output_path": "./sharegpt.json"}'
```

---

## Tutorial 4: Create Entirely New Model Architectures

This is for researchers who want to invent architectures that **don't exist yet**.

### 4.1 Method A: Block Composer (No Code Required)

Mix any of the 25+ composable block primitives:

```bash
# Build a novel block: Mamba + Sliding Window Attention + MoE FFN
curl -X POST http://localhost:8765/api/llm/compose \
  -d '{
    "vocab_size": 32000,
    "d_model": 512,
    "n_layers": 8,
    "n_heads": 8,
    "default_block": [
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "mamba", "config": {"d_state": 16, "expand": 2}},
      {"type": "residual", "residual_from": -1},
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "sliding_window_attention", "config": {"window_size": 512}},
      {"type": "residual", "residual_from": 3},
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "moe", "config": {"n_experts": 8, "top_k": 2}},
      {"type": "residual", "residual_from": 5}
    ]
  }'
```

**All available step types:**

| Type | Description |
|------|-------------|
| `norm` | RMSNorm or LayerNorm |
| `attention` | Multi-head attention with RoPE, GQA, Flash |
| `sliding_window_attention` | Local attention (Mistral-style) |
| `linear_attention` | O(n) kernel-based attention |
| `alibi_attention` | Attention with Linear Biases |
| `ffn` | Feed-forward (SwiGLU / GeGLU / ReGLU / Standard) |
| `moe` | Mixture of Experts with Top-K routing |
| `mamba` | Selective State Space Model |
| `rwkv` | RWKV linear attention |
| `retention` | Multi-scale retention (RetNet) |
| `hyena` | Long convolution via FFT |
| `xlstm` | Extended LSTM with exponential gating |
| `gated_recurrence` | Griffin/Hawk gated linear recurrence |
| `parallel` | Run two components in parallel + merge |
| `cross_attention` | Attend to external context |
| `conv1d` | 1D convolution over sequence |
| `residual` | Skip connection from previous step |
| `dropout` | Regularization |
| `linear` | Linear projection |
| `activation` | Non-linear activation (relu/gelu/silu/...) |
| `custom_code` | Write any nn.Module in Python |
| `custom_formula` | Define FFN with a math formula |
| `pos_encoding` | Positional encoding (absolute/sinusoidal) |
| `embedding` | Standalone positional embedding |

### 4.2 Method B: Full Custom Model from Python Code

Define the **entire model** in Python with access to all 101+ building blocks:

```bash
curl -X POST http://localhost:8765/api/llm/novel/model-from-code \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_size": 32000,
    "d_model": 512,
    "code": "class Llama4(nn.Module):\n    def __init__(self, vocab_size=32000, d_model=512, n_layers=12, n_heads=8, n_kv_heads=4, **kw):\n        super().__init__()\n        self.vocab_size = vocab_size\n        self.tok_emb = nn.Embedding(vocab_size, d_model)\n        self.layers = nn.ModuleList()\n        for i in range(n_layers):\n            use_moe = i >= n_layers // 2  # Dense first half, MoE second half\n            self.layers.append(nn.ModuleDict({\n                \"norm1\": RMSNorm(d_model),\n                \"attn\": LLMAttention(d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, max_len=8192, rope_base=500000),\n                \"norm2\": RMSNorm(d_model),\n                \"ffn\": MoELayer(d_model, n_experts=8, top_k=2) if use_moe else SwiGLUFFN(d_model),\n            }))\n        self.norm = RMSNorm(d_model)\n        self.head = nn.Linear(d_model, vocab_size, bias=False)\n        self.head.weight = self.tok_emb.weight\n    def forward(self, input_ids, labels=None):\n        x = self.tok_emb(input_ids.clamp(0, self.vocab_size-1))\n        for L in self.layers:\n            x = x + L[\"attn\"](L[\"norm1\"](x))\n            x = x + L[\"ffn\"](L[\"norm2\"](x))\n        logits = self.head(self.norm(x))\n        loss = None\n        if labels is not None:\n            loss = F.cross_entropy(logits[:,:-1].reshape(-1,self.vocab_size), labels[:,1:].reshape(-1), ignore_index=-100)\n        return {\"logits\": logits, \"loss\": loss}"
  }'
```

**Available building blocks inside custom code:**

| Category | Components |
|----------|-----------|
| **LLM Core** | `RMSNorm`, `LLMAttention`, `SwiGLUFFN`, `GeGLUFFN`, `MoELayer`, `MoERouter`, `RotaryPositionalEmbedding` |
| **SSM** | `SelectiveScan`, `MambaBlock`, `RWKVBlock`, `RetentionLayer`, `HyenaOperator`, `XLSTM`, `GatedLinearRecurrence` |
| **Vision** | `ResNetBlock`, `ConvNeXtBlock`, `MBConvBlock`, `VisionEncoder`, `PatchEmbed` |
| **Diffusion** | `DiffusionUNet`, `VAE`, `DiffusionTimestepBlock`, `SpatialAttentionBlock`, `CrossAttentionBlock` |
| **Video** | `VideoVAE`, `TemporalAttention`, `TemporalConv3d` |
| **Multimodal** | `PerceiverResampler`, `PatchEmbedding`, `AudioEmbedding`, `ModalityProjector` |
| **Training** | `DistillationWrapper` |
| **PyTorch** | `torch`, `nn`, `F`, `math` + all standard modules |

### 4.3 Method C: Validate, Benchmark, and Search

```bash
# Validate a novel design
curl -X POST http://localhost:8765/api/llm/novel/validate \
  -d '{
    "block_design": [
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "mamba", "config": {"d_state": 16}},
      {"type": "residual", "residual_from": -1},
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "attention", "config": {}},
      {"type": "residual", "residual_from": 3}
    ],
    "d_model": 128,
    "n_heads": 4,
    "benchmark": true
  }'

# Response shows speed comparison vs Llama/Mamba/Minimal baselines

# Architecture search: compare 10+ designs automatically
curl -X POST http://localhost:8765/api/llm/novel/arch-search \
  -d '{
    "auto_search": true,
    "text": "Your training text for comparison...",
    "d_model": 128,
    "train_steps": 30
  }'

# Response ranks all designs by: loss reduction, speed, param efficiency
```

### 4.4 Custom Loss Functions

```bash
# Formula-based loss (label smoothing)
curl -X POST http://localhost:8765/api/llm/novel/custom-loss \
  -d '{
    "mode": "formula",
    "formula": "0.9 * F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100) + 0.1 * (-F.log_softmax(logits, dim=-1).mean())"
  }'

# Full Python class loss
curl -X POST http://localhost:8765/api/llm/novel/custom-loss \
  -d '{
    "mode": "code",
    "code": "class FocalLoss(nn.Module):\n    def __init__(self, gamma=2.0, alpha=0.25):\n        super().__init__()\n        self.gamma = gamma\n        self.alpha = alpha\n    def forward(self, logits, labels, **kw):\n        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction=\"none\", ignore_index=-100)\n        pt = torch.exp(-ce)\n        return (self.alpha * (1-pt)**self.gamma * ce).mean()"
  }'
```

---

## Tutorial 5: Complete Example — Build Llama 4 from Scratch (End to End)

This is the full walkthrough: architecture design, dataset creation, tokenizer training, model training, evaluation, and generation.

### Step 1: Design the Llama 4 Architecture

Llama 4 features: dense initial layers + MoE deep layers, GQA (4 KV heads), RoPE with extended base, SwiGLU, RMSNorm.

```
UI Flow:
1. Click "LLM Builder" in the top bar
2. Open "Model Blueprints" section
3. Select category: "Text LLMs"
4. Click "Claude-style LLM (from scratch)" (closest to Llama 4)
5. Select scale: "medium" (1024 d_model, 24 layers)
6. In the config fields below, enable MoE checkbox
7. Set Experts: 8, Top-K: 2
8. Click "Build This Model"
```

```bash
# API: Build Llama 4 (medium scale, ~350M params)
curl -X POST http://localhost:8765/api/llm/build -d '{
  "vocab_size": 32000, "d_model": 1024, "n_layers": 24,
  "n_heads": 16, "n_kv_heads": 4, "max_len": 8192,
  "norm_type": "rmsnorm", "ffn_type": "swiglu",
  "use_moe": true, "n_experts": 8, "moe_top_k": 2,
  "rope_base": 500000, "use_flash": true, "tie_weights": true
}'
```

### Step 2: Create Training Dataset

```
UI Flow:
1. In the left panel, expand "Dataset Factory"
2. Select template: "instruct"
3. Add samples using the form, or paste bulk data
4. Export as Alpaca format
```

```bash
# Or load an existing dataset from HuggingFace
curl -X POST http://localhost:8765/api/hf/datasets/load -d '{
  "dataset_id": "tatsu-lab/alpaca",
  "split": "train"
}'
```

For pre-training on raw text, just paste your corpus directly into the training text area.

### Step 3: Train Tokenizer

```
UI Flow:
1. Expand "Train Tokenizer" section in LLM Builder
2. Select algorithm: "BPE (GPT-style)"
3. Set vocab size: 32000
4. Paste your training text (or it uses the LLM training text)
5. Click "Train Tokenizer"
```

### Step 4: Train the Model

```
UI Flow:
1. Expand "Training Data" section
2. Paste your text corpus (or upload a .txt file)
3. Set epochs: 20, batch size: 8, learning rate: 3e-4
4. Select scheduler: Cosine
5. Click "Start LLM Training"
6. Watch real-time loss charts via WebSocket
```

```bash
# API
curl -X POST http://localhost:8765/api/llm/train -d '{
  "text": "<your training corpus>",
  "tokenizer": "custom_trained",
  "max_len": 256,
  "epochs": 20,
  "batch_size": 8,
  "learning_rate": 3e-4,
  "scheduler": "cosine"
}'
```

### Step 5: Generate Text

```
UI Flow:
1. Expand "Generation" section
2. Type your prompt
3. Set temperature, top-k, max tokens
4. Click "Generate"
```

```bash
curl -X POST http://localhost:8765/api/llm/generate -d '{
  "prompt": "Explain the theory of relativity",
  "max_tokens": 200,
  "temperature": 0.8,
  "top_k": 50
}'
```

### Step 6: Iterate on Architecture

```bash
# Try making some layers use attention, others use Mamba
curl -X POST http://localhost:8765/api/llm/compose -d '{
  "vocab_size": 32000, "d_model": 1024, "n_layers": 24, "n_heads": 16,
  "block_designs": [
    null, null, null, null, null, null, null, null,
    [{"type":"norm","config":{"norm_type":"rmsnorm"}},
     {"type":"mamba","config":{"d_state":16}},
     {"type":"residual","residual_from":-1},
     {"type":"norm","config":{"norm_type":"rmsnorm"}},
     {"type":"moe","config":{"n_experts":8,"top_k":2}},
     {"type":"residual","residual_from":2}],
    null, null, null, null, null, null, null,
    null, null, null, null, null, null, null, null
  ]
}'
# Layers 0-7: standard Llama blocks
# Layer 8: hybrid Mamba + MoE (experimental)
# Layers 9-23: standard Llama blocks
```

---

## Tutorial 6: Diffusion Model Training (Image/Video Generation)

### 6.1 Train a Stable Diffusion Model from Scratch

```
UI Flow:
1. LLM Builder → Advanced Models → Diffusion Model section
2. Set Image Size: 64, Base Channels: 64
3. Check "Latent Diffusion (VAE)"
4. Click "Train VAE" first (20 epochs)
5. After VAE training completes, click "Train UNet"
6. After UNet training, click "Generate"
```

```bash
# Step 1: Train VAE
curl -X POST http://localhost:8765/api/diffusion/train-vae -d '{
  "epochs": 20, "batch_size": 8, "image_size": 64,
  "learning_rate": 1e-4, "kl_weight": 0.01
}'

# Step 2: Train UNet denoiser
curl -X POST http://localhost:8765/api/diffusion/train -d '{
  "mode": "image", "epochs": 50, "batch_size": 4,
  "image_size": 64, "use_vae": true,
  "n_steps": 1000, "schedule": "cosine"
}'

# Step 3: Generate images
curl -X POST http://localhost:8765/api/diffusion/generate -d '{
  "n_images": 4, "n_steps": 50, "image_size": 64
}'
```

---

## Tutorial 7: Invent a Frontier Model (Research Guide)

### 7.1 Use the Novel Architecture Lab

```
UI Flow:
1. LLM Builder → Novel Architecture Lab
2. Pick a research template (e.g., "Multi-Scale Processing")
3. Click "Validate" to verify shapes
4. Click "Benchmark" to compare vs Llama/Mamba
5. Click "Quick Train" to see if it learns
6. Click "Architecture Search" to rank all designs
7. Modify the design and repeat
```

### 7.2 Research Templates

| Template | Key Idea | Components |
|----------|----------|-----------|
| Sparse Attention + SSM | Windowed attention for precision, Mamba for long-range | `sliding_window_attention` + `mamba` + `swiglu` |
| Fully Parallel Block | Everything runs simultaneously with learned gating | `parallel(attention, mamba)` + `ffn` |
| Recursive Layers | Same weights repeated N times (Universal Transformer) | Shared `attention` + `ffn` blocks |
| Gated Expert SSM | Route tokens to specialized SSM experts | `moe` + `mamba` |
| Multi-Scale Processing | Local conv + medium attention + global SSM | `conv1d` + `sliding_window` + `mamba` + `ffn` |
| Write Your Own | Full Python code with all primitives | `custom_code` with any nn.Module |

### 7.3 Architecture Search

```bash
# Compare your design against 10+ baselines automatically
curl -X POST http://localhost:8765/api/llm/novel/arch-search -d '{
  "auto_search": true,
  "designs": {
    "my_frontier_design": [
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "parallel", "config": {
        "branch_a": {"type": "attention", "config": {}},
        "branch_b": {"type": "mamba", "config": {"d_state": 16}},
        "merge": "gate"
      }},
      {"type": "residual", "residual_from": -1},
      {"type": "norm", "config": {"norm_type": "rmsnorm"}},
      {"type": "moe", "config": {"n_experts": 4, "top_k": 1}},
      {"type": "residual", "residual_from": 3}
    ]
  },
  "text": "Training text for comparison...",
  "d_model": 128, "train_steps": 30
}'

# Response: ranked table of all designs by loss reduction, speed, efficiency
```

---

## Tutorial 8: Reinforcement Learning — Train RL Agents

The **RL** tab lets you train reinforcement learning agents using Stable-Baselines3 across 27 environments with 6 algorithms.

---

### 8.1 Quick Start: Train PPO on CartPole

```
UI Flow:
1. Click "RL" in the top bar
2. Select environment: CartPole-v1
3. Select algorithm: PPO
4. Set Total Timesteps: 50000
5. Click "Train" — watch reward chart climb in real time
6. Click "Run Episode" to see the trained agent in action
```

```bash
# Step 1: Create environment
curl -X POST http://localhost:8765/api/rl/env -d '{"env_id": "CartPole-v1"}'

# Step 2: Create agent
curl -X POST http://localhost:8765/api/rl/agent -d '{
  "algorithm": "PPO",
  "params": {"learning_rate": 3e-4, "n_steps": 2048}
}'

# Step 3: Train (runs in background)
curl -X POST http://localhost:8765/api/rl/train -d '{
  "total_timesteps": 50000, "eval_freq": 1000
}'

# Step 4: Check progress
curl http://localhost:8765/api/rl/info

# Step 5: Run evaluation episode
curl -X POST http://localhost:8765/api/rl/episode

# Step 6: Save model
curl -X POST http://localhost:8765/api/rl/save -d '{"path": "./my_rl_model"}'
```

### 8.2 Available Environments

| Category | Environments | Action Type |
|----------|-------------|-------------|
| **Classic Control** | CartPole-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3, Pendulum-v1 | Discrete / Continuous |
| **Box2D** | BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3 | Continuous |
| **MuJoCo** | HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5, Swimmer-v5, Reacher-v5 | Continuous |
| **PyBullet** | HumanoidBulletEnv-v0, AntBulletEnv-v0, KukaBulletEnv-v0 | Continuous |
| **Custom** | custom_grid (configurable size, obstacles, max steps) | Discrete |

### 8.3 Algorithm Selection Guide

| Algorithm | Best For | Action Space |
|-----------|----------|-------------|
| **PPO** | General purpose — start here | Both |
| **A2C** | Fast training, simple tasks | Both |
| **DQN** | Discrete actions (games, grid worlds) | Discrete only |
| **SAC** | Continuous control, robotics | Continuous only |
| **TD3** | Robotics, precision control | Continuous only |
| **DDPG** | Continuous control (simpler than SAC) | Continuous only |

### 8.4 Custom Grid World

The built-in grid world needs no Gymnasium:

```bash
curl -X POST http://localhost:8765/api/rl/env -d '{
  "env_id": "custom_grid",
  "params": {"size": 10, "n_obstacles": 8, "max_steps": 300}
}'
```

The agent navigates from (0,0) to (size-1, size-1) avoiding obstacles. Reward: +1 for reaching goal, -0.01 per step. Episode frames include grid state for canvas visualization.

---

## Tutorial 9: Robotics — Build, Simulate & Deploy Robots

The **Robotics** tab is a complete robot engineering environment. Build robots from 27 real components, analyze circuits, visualize in 3D, articulate joints, and run physics simulations.

---

### 9.1 Quick Start: Build a Wheeled Robot

```
UI Flow:
1. Click "Robot" in the top bar
2. Select template: "2WD Wheeled Robot"
3. View 3D model — orbit with mouse, zoom with scroll
4. Check circuit analysis — battery life, power draw, weight
5. Add sensors (drag from component palette)
6. Click "Simulate" to run physics
```

```bash
# Step 1: Create robot from template
curl -X POST http://localhost:8765/api/robotics/robots -d '{
  "name": "My Robot", "template": "wheeled_2wd"
}'
# Response: {"status": "created", "robot": {"id": "abc123", "component_count": 9, ...}}

# Step 2: Add an IMU sensor
curl -X POST http://localhost:8765/api/robotics/robots/abc123/components -d '{
  "type": "imu_mpu6050", "position": [0, 0.05, 0], "role": "imu"
}'

# Step 3: Analyze circuit
curl http://localhost:8765/api/robotics/robots/abc123/circuit
# Response: total weight, current draw, battery life, voltage rails, warnings

# Step 4: Get 3D scene (for Three.js rendering)
curl http://localhost:8765/api/robotics/robots/abc123/scene
# Response: objects with positions/dimensions/colors, joints, links
```

### 9.2 All 27 Components

| Category | Components |
|----------|-----------|
| **Actuators** (7) | Micro Servo SG90, Standard Servo MG996R, High-Torque Servo DS3225, DC Motor N20, DC Motor 775, Stepper NEMA17, BLDC Motor 2212 |
| **Sensors** (7) | IMU MPU6050, Ultrasonic HC-SR04, LiDAR RPLiDAR, Camera OV5647, Force Sensor, Rotary Encoder |
| **Power** (4) | LiPo 1S 500mAh, LiPo 2S 2200mAh, LiPo 3S 5000mAh, 18650 Cell |
| **Controllers** (4) | Arduino Nano, Raspberry Pi 4, ESP32, Jetson Nano |
| **Structure** (3) | Aluminum Plate, Carbon Fiber Tube, Rubber Wheel |
| **Electronics** (2) | Motor Driver L298N, Voltage Regulator, BEC 5V 3A |

### 9.3 Robot Templates

| Template | DOF | Components | Description |
|----------|-----|-----------|-------------|
| `wheeled_2wd` | 2 | 9 | Differential-drive mobile robot with ultrasonic sensor |
| `robot_arm_3dof` | 3 | 8 | Articulated arm with 3 servo joints |
| `quadruped` | 12 | 16 | 4-legged walker (4 hips + 8 knees/ankles) |
| `humanoid` | 17 | 17 | Bipedal humanoid with camera, IMU, full articulation |
| `drone_quad` | 4 | 11 | Quadcopter with 4 BLDC motors and IMU |

### 9.4 Circuit Analysis

Every robot gets automatic circuit analysis:

```bash
curl http://localhost:8765/api/robotics/robots/{id}/circuit
```

Returns:
- **Total weight** (grams) — sum of all components
- **Current draw** (idle + active) — based on component specs and duty cycles
- **Battery life** (minutes) — capacity / active current
- **Current within limits** — warns if total draw exceeds battery max discharge
- **Voltage rails** — maps which components share voltage levels
- **Per-component breakdown** — weight, current, power for each part

### 9.5 Joint Control & Articulation

```bash
# Set a single joint
curl -X POST http://localhost:8765/api/robotics/robots/{id}/joint/{comp_id} \
  -d '{"angle": 45.0}'

# Set all joints at once
curl -X POST http://localhost:8765/api/robotics/robots/{id}/joints \
  -d '{"angles": {"comp_001": 45, "comp_002": -30, "comp_003": 90}}'

# Read current joint states
curl http://localhost:8765/api/robotics/robots/{id}/joints
```

Joint angles are automatically clamped to the component's physical range (e.g., servos: ±90°, DC motors: ±180°).

### 9.6 Deploy Robot to Physics Simulation

This is the full workflow to take a robot from design to simulation:

```bash
# Step 1: Get the 3D scene
scene=$(curl -s http://localhost:8765/api/robotics/robots/{id}/scene)

# Step 2: Load into physics engine
curl -X POST http://localhost:8765/api/physics/load -d '{
  "bodies": [
    {"position": [0, 0.1, 0], "dimensions": {"x": 0.1, "y": 0.05, "z": 0.05},
     "mass": 0.45, "name": "chassis"},
    {"position": [-0.05, 0, 0], "dimensions": {"x": 0.065, "y": 0.026, "z": 0.065},
     "mass": 0.028, "name": "left_wheel"}
  ],
  "use_mujoco": true
}'
# Falls back to Euler integration if MuJoCo is not installed

# Step 3: Start real-time simulation
curl -X POST http://localhost:8765/api/physics/start
# Physics runs at 500Hz, broadcasts state to browser at 60fps via WebSocket

# Step 4: Apply forces (e.g., drive forward)
curl -X POST http://localhost:8765/api/physics/force -d '{
  "body_index": 0, "force": [5.0, 0, 0]
}'

# Step 5: Step-by-step mode (for debugging)
curl -X POST http://localhost:8765/api/physics/step

# Step 6: Stop simulation
curl -X POST http://localhost:8765/api/physics/stop
```

### 9.7 Connect to Hardware

Bridge your simulated robot to real hardware via serial/USB:

```bash
# List available serial ports
curl http://localhost:8765/api/hardware/ports

# Connect to Arduino
curl -X POST http://localhost:8765/api/hardware/connect -d '{
  "port": "/dev/ttyUSB0", "baud_rate": 115200
}'

# Send joint angles to physical servos
curl -X POST http://localhost:8765/api/hardware/joints -d '{
  "angles": [45, -30, 90, 0]
}'

# Generate Arduino firmware for your robot
curl -X POST http://localhost:8765/api/hardware/firmware/generate -d '{
  "robot_id": "abc123"
}'
```

---

## Tutorial 10: HuggingFace — Fine-Tune Pretrained Models

The **HuggingFace** tab connects to the HF ecosystem. Load any model, apply LoRA, load datasets, train, and run inference.

---

### 10.1 Quick Start: Fine-Tune BERT for Classification

```
UI Flow:
1. Click "HuggingFace" in the top bar
2. Search for "bert-base-uncased" → click Load
3. Set task: "text-classification", num_labels: 2
4. Load dataset (CSV, HF Hub, or local)
5. Click "Apply LoRA" for efficient fine-tuning
6. Click "Train" — watch loss chart
7. Click "Inference" to test with custom text
```

```bash
# Step 1: Load model
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "bert-base-uncased",
  "task": "text-classification",
  "num_labels": 2
}'

# Step 2: Load CSV dataset
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "csv", "path": "./data/reviews.csv",
  "text_col": "text", "label_col": "label"
}'

# Step 3: Apply LoRA (reduces trainable params by 90%+)
curl -X POST http://localhost:8765/api/hf/model/lora -d '{
  "r": 8, "lora_alpha": 16, "lora_dropout": 0.1
}'

# Step 4: Train
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 3, "learning_rate": 2e-5, "batch_size": 16
}'

# Step 5: Run inference
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "This movie was absolutely fantastic!"
}'
```

### 10.2 Architecture Surgery

Modify any loaded model's architecture:

```bash
# Freeze encoder layers 0-8
curl -X POST http://localhost:8765/api/hf/model/freeze -d '{
  "patterns": ["encoder.layer.[0-8]"]
}'

# Insert a new dropout layer
curl -X POST http://localhost:8765/api/hf/model/insert -d '{
  "parent_path": "classifier", "name": "dropout",
  "module_type": "Dropout", "module_params": {"p": 0.3}
}'

# Replace the classification head
curl -X POST http://localhost:8765/api/hf/model/replace -d '{
  "path": "classifier",
  "module_type": "Linear",
  "module_params": {"in_features": 768, "out_features": 5}
}'

# View model architecture tree
curl http://localhost:8765/api/hf/model/tree
```

---

## Tutorial 11: Data Engineering — Build ETL Pipelines

The **Data** tab provides a visual pipeline builder for ETL workflows. Connect to databases, apply transforms, and sink results.

---

### 11.1 Quick Start: CSV → Filter → Export

```
UI Flow:
1. Click "Data" in the top bar
2. Click "New Pipeline"
3. Drag a CSV source → set file path
4. Add Filter transform: age > 25
5. Add Select transform: keep only name, score columns
6. Click "Run Pipeline"
7. View results table and column statistics
```

```bash
# Step 1: Create pipeline
curl -X POST http://localhost:8765/api/dataeng/pipelines -d '{"name": "My Pipeline"}'
# Response: {"pipeline": {"id": "pipe_123"}}

# Step 2: Add CSV source
curl -X POST http://localhost:8765/api/dataeng/pipelines/pipe_123/sources -d '{
  "connector_type": "csv_file",
  "params": {"path": "./data/users.csv"},
  "source_id": "src1"
}'

# Step 3: Load data from source
curl -X POST http://localhost:8765/api/dataeng/pipelines/pipe_123/sources/src1/load

# Step 4: Add filter transform
curl -X POST http://localhost:8765/api/dataeng/pipelines/pipe_123/transforms -d '{
  "type": "filter", "params": {"column": "age", "op": ">", "value": 25}
}'

# Step 5: Run pipeline
curl -X POST http://localhost:8765/api/dataeng/pipelines/pipe_123/run

# Step 6: View results
curl http://localhost:8765/api/dataeng/pipelines/pipe_123/result

# Step 7: Get column statistics
curl http://localhost:8765/api/dataeng/pipelines/pipe_123/stats
```

### 11.2 Supported Connectors (15)

| Category | Connectors |
|----------|-----------|
| **SQL** | MySQL, PostgreSQL, MSSQL, Oracle, SQLite |
| **NoSQL** | MongoDB, Redis, Elasticsearch |
| **Streaming** | Kafka |
| **Cloud** | S3, BigQuery, ClickHouse |
| **File** | CSV, JSON/JSONL, Parquet |

### 11.3 Available Transforms (17)

| Transform | Description |
|-----------|-------------|
| `select` | Keep only specified columns |
| `drop` | Remove specified columns |
| `rename` | Rename columns |
| `filter` | Filter rows by condition (>, <, ==, !=, contains, etc.) |
| `fill_null` | Fill null values (mean, median, mode, constant) |
| `drop_null` | Remove rows with nulls |
| `cast` | Change column data types |
| `deduplicate` | Remove duplicate rows |
| `sort` | Sort by column(s) |
| `limit` | Take first N rows |
| `sample` | Random sample of N rows |
| `text_clean` | Lowercase, strip, remove HTML/URLs |
| `text_chunk` | Split text into chunks |
| `json_flatten` | Flatten nested JSON columns |
| `merge` | Join with another source |
| `concat` | Concatenate multiple sources |
| `add_column` | Add computed column |

---

## Tutorial 12: Workspace/IDE — Write & Run Code

The **IDE** tab is a full Python development environment in your browser.

---

### 12.1 Quick Start

```
UI Flow:
1. Click "IDE" in the top bar
2. Click "New Project" → select "LLM Fine-Tuning" template
3. Browse file tree on the left
4. Edit code in the CodeMirror editor
5. Press Ctrl+Enter to run
6. View stdout/stderr in the console
```

```bash
# Create a project
curl -X POST http://localhost:8765/api/workspace/projects -d '{
  "name": "my_experiment", "template": "llm_finetune"
}'

# Write a file
curl -X POST http://localhost:8765/api/workspace/projects/{id}/write -d '{
  "path": "train.py", "content": "import torch\nprint(torch.cuda.is_available())"
}'

# Run code directly
curl -X POST http://localhost:8765/api/workspace/run -d '{
  "code": "print(sum(range(100)))"
}'
# Response: {"stdout": "4950\n", "stderr": "", "exit_code": 0}

# Install packages
curl -X POST http://localhost:8765/api/workspace/pip -d '{"action": "install", "package": "scipy"}'
```

### 12.2 Project Templates

| Template | Contents |
|----------|---------|
| **Empty** | `main.py` + `.gitignore` |
| **LLM Fine-Tuning** | Train/eval scripts, config.json, data preparation |
| **Vision** | Image classification training with augmentation |
| **YOLO** | Object detection training with YOLO format data |
| **Dataset** | Dataset creation and processing scripts |

---

## Tutorial 13: Evaluation & Deployment

### 13.1 Model Evaluation

After training any model, evaluate with one click:

```bash
# Auto-evaluate (detects classification vs regression)
curl -X POST http://localhost:8765/api/eval/auto

# Manual classification evaluation
curl -X POST http://localhost:8765/api/eval/classification -d '{
  "y_true": [0, 1, 1, 0, 1, 0],
  "y_pred": [0, 1, 0, 0, 1, 1]
}'
# Response: accuracy, f1, precision, recall, confusion matrix

# Regression evaluation
curl -X POST http://localhost:8765/api/eval/regression -d '{
  "y_true": [1.0, 2.0, 3.0], "y_pred": [1.1, 2.2, 2.8]
}'
# Response: mse, rmse, mae, r2

# Text generation evaluation
curl -X POST http://localhost:8765/api/eval/generation -d '{
  "references": ["The cat sat on the mat"],
  "predictions": ["A cat was sitting on the mat"]
}'
# Response: bleu-1/2/3/4, rouge-1/2/L
```

### 13.2 Model Export

```bash
# Export to ONNX (for fast inference, cross-platform)
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./model.onnx"}'

# Export to TorchScript
curl -X POST http://localhost:8765/api/deploy/torchscript -d '{
  "path": "./model.pt", "method": "trace"
}'
```

### 13.3 Generate Deployment Code

```bash
# FastAPI inference server
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./model.onnx", "model_type": "onnx", "port": 8080
}'

# Dockerfile
curl -X POST http://localhost:8765/api/deploy/dockerfile -d '{
  "model_path": "./model.onnx", "model_type": "onnx"
}'

# Gradio demo app (shareable link)
curl -X POST http://localhost:8765/api/deploy/gradio -d '{
  "model_path": "./model.pt", "model_type": "pytorch"
}'
```

---

## Tutorial 14: Voice Cloning — Production Pipeline

Build a production voice cloning system that captures a speaker's identity from a few seconds of audio and generates speech in their voice. This tutorial covers the full pipeline: speaker encoder, synthesizer, and vocoder.

---

### 14.1 Architecture Overview

A production voice cloning system has 3 components:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Speaker Encoder  │     │ Synthesizer      │     │ Vocoder          │
│ (voice identity) │────>│ (text→mel specs) │────>│ (mel→waveform)   │
│                  │     │                  │     │                  │
│ Input: 3-10s     │     │ Input: text +    │     │ Input: mel specs │
│ reference audio  │     │ speaker embedding│     │ Output: waveform │
│ Output: 256-d    │     │ Output: mel specs│     │ (24kHz audio)    │
│ speaker vector   │     │                  │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

### 14.2 Method A: Fine-Tune a Pretrained Voice Cloning Model (Recommended)

The fastest path to production. Use XTTS v2 or SpeechT5 via the HuggingFace tab.

#### Step 1: Prepare Your Voice Dataset

```
UI Flow:
1. Click "Datasets" in left panel → "New Project"
2. Select template: "text_to_speech"
3. Add samples: upload audio clips (3-30 seconds each, WAV/MP3)
4. Each sample needs: audio file path + transcription text
5. Aim for 30-60 minutes of clean single-speaker audio
6. Export as JSONL
```

```bash
# Create dataset project
curl -X POST http://localhost:8765/api/ds/projects -d '{
  "name": "voice_clone_data",
  "template_id": "text_to_speech"
}'

# Add samples (audio path + transcription)
curl -X POST http://localhost:8765/api/ds/projects/{id}/samples/bulk -d '{
  "samples": [
    {"audio_path": "./audio/clip_001.wav", "text": "The quick brown fox jumps over the lazy dog.", "speaker": "target"},
    {"audio_path": "./audio/clip_002.wav", "text": "Machine learning models process sequences of tokens.", "speaker": "target"}
  ]
}'

# Export
curl -X POST http://localhost:8765/api/ds/projects/{id}/export -d '{"format": "jsonl"}'
```

**Data quality requirements for production:**

| Requirement | Minimum | Recommended |
|------------|---------|------------|
| Total audio | 5 minutes | 30-60 minutes |
| Sample rate | 16kHz | 22.05kHz or 24kHz |
| Format | WAV (16-bit PCM) | WAV (16-bit PCM) |
| Noise floor | < -40dB | < -60dB |
| Clipping | None | None |
| Room reverb | Minimal | Treated room / close mic |
| Speaker count | 1 per voice | 1 per voice |

#### Step 2: Fine-Tune with XTTS v2

```
UI Flow:
1. Click "HuggingFace" in top bar
2. Search: "coqui/XTTS-v2" → Load
3. Load your voice dataset (JSONL from Step 1)
4. Apply LoRA (r=16, alpha=32) to reduce training cost
5. Train: epochs=5, lr=1e-5, batch_size=4
6. Test: type text → generate speech in cloned voice
```

```bash
# Load XTTS v2
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "coqui/XTTS-v2"
}'

# Apply LoRA for efficient fine-tuning
curl -X POST http://localhost:8765/api/hf/model/lora -d '{
  "r": 16, "lora_alpha": 32, "lora_dropout": 0.05
}'

# Load dataset
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "json", "path": "./voice_clone_data.jsonl"
}'

# Train (LoRA fine-tuning — much faster than full fine-tuning)
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 5, "learning_rate": 1e-5, "batch_size": 4,
  "warmup_steps": 100, "max_grad_norm": 1.0
}'

# Inference: generate speech in the cloned voice
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "Hello, this is my cloned voice speaking.",
  "reference_audio": "./audio/reference_clip.wav"
}'
```

#### Step 3: Alternative — SpeechT5 (Lighter Weight)

```bash
# Load Microsoft SpeechT5 TTS
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "microsoft/speecht5_tts"
}'

# Load speaker embeddings dataset
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "json", "path": "./speaker_embeddings.jsonl"
}'

# Fine-tune for your target speaker
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 10, "learning_rate": 5e-5, "batch_size": 8
}'
```

### 14.3 Method B: Build Voice Cloning from Scratch

For researchers who want full control. Build each component in the Graph tab.

#### Component 1: Speaker Encoder (GE2E)

The speaker encoder maps any audio clip to a fixed-size speaker embedding vector.

```
UI Flow:
1. Graph tab → Clear canvas
2. Add layers manually:
   - MelSpectrogram (n_mels=40, n_fft=512, hop_length=160)
   - AudioConvBlock (in=40, out=256)
   - AudioConvBlock (in=256, out=256, stride=2)
   - Transpose (dim0=1, dim1=2)
   - LSTM (input_size=256, hidden_size=256, num_layers=3)
   - Linear (in=256, out=256)
3. Dataset: Load audio classification dataset (speaker IDs as labels)
4. Loss: CrossEntropyLoss (for speaker classification)
5. Train: epochs=50, lr=1e-4, optimizer=Adam
```

```bash
# Build speaker encoder architecture
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "MelSpectrogram",
  "params": {"n_mels": 40, "n_fft": 512, "hop_length": 160}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "AudioConvBlock",
  "params": {"in_channels": 40, "out_channels": 256}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "AudioConvBlock",
  "params": {"in_channels": 256, "out_channels": 256, "stride": 2}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "LSTM",
  "params": {"input_size": 256, "hidden_size": 256, "num_layers": 3}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "Linear",
  "params": {"in_features": 256, "out_features": 256}
}'

# Build and train
curl -X POST http://localhost:8765/api/build
curl -X POST http://localhost:8765/api/config -d '{
  "epochs": 50, "learning_rate": 1e-4, "optimizer": "Adam",
  "loss": "CrossEntropyLoss"
}'
curl -X POST http://localhost:8765/api/train/start
```

#### Component 2: Mel Synthesizer (Tacotron2 / FastSpeech2)

Use the Workspace IDE to write the synthesizer since it needs custom attention:

```
UI Flow:
1. Click "IDE" in top bar
2. New Project → "Empty"
3. Write synthesizer code (see below)
4. Ctrl+Enter to run
```

```python
# In the IDE — synthesizer.py
import torch
import torch.nn as nn

class VoiceCloneSynthesizer(nn.Module):
    """Text + speaker embedding → mel spectrogram."""
    def __init__(self, vocab_size=256, d_model=256, n_heads=4,
                 n_layers=4, speaker_dim=256, n_mels=80):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       batch_first=True),
            num_layers=n_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4,
                                       batch_first=True),
            num_layers=n_layers
        )
        self.mel_proj = nn.Linear(d_model, n_mels)

    def forward(self, text_ids, speaker_emb, mel_target=None):
        B, T = text_ids.shape
        x = self.text_embed(text_ids) + self.pos_enc[:, :T, :]
        # Add speaker identity
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk
        memory = self.encoder(x)

        if mel_target is not None:
            # Teacher forcing during training
            mel_in = mel_target[:, :-1, :]  # shift right
            decoder_out = self.decoder(mel_in, memory)
        else:
            # Autoregressive generation
            decoder_out = memory  # simplified

        mel_out = self.mel_proj(decoder_out)
        return mel_out

# Train using: MSELoss between predicted mel and target mel
```

#### Component 3: Vocoder (HiFi-GAN)

```bash
# Option 1: Load pretrained HiFi-GAN vocoder
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "facebook/encodec_24khz"
}'

# Option 2: Use Workspace IDE to train your own
# (see HiFi-GAN paper implementation in IDE)
```

### 14.4 Production Deployment

```bash
# Export the fine-tuned model
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./voice_clone.onnx"}'

# Generate FastAPI inference server
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./voice_clone.onnx",
  "model_type": "onnx", "port": 8080
}'

# Generate Docker container
curl -X POST http://localhost:8765/api/deploy/dockerfile -d '{
  "model_path": "./voice_clone.onnx", "model_type": "onnx"
}'

# Generate Gradio demo
curl -X POST http://localhost:8765/api/deploy/gradio -d '{
  "model_path": "./voice_clone.pt", "model_type": "pytorch"
}'
```

### 14.5 Production Quality Checklist

| Item | How to Verify |
|------|--------------|
| Speaker similarity (SIM-O) > 0.85 | Compare embeddings of generated vs reference audio |
| Naturalness (MOS) > 3.8 | Human listening test on 1-5 scale |
| No clipping or artifacts | Check waveform peak < 0.95 |
| Consistent pacing | Compare speaking rate to reference |
| Works on unseen text | Test with text not in training data |
| Latency < 500ms per sentence | Profile with `/api/deploy/server` |

---

## Tutorial 15: YOLO Training — Object Detection for Production

Train YOLOv8/v9/v11 models for production object detection, from dataset creation through deployment.

---

### 15.1 Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Backbone         │     │ Neck (FPN/PANet) │     │ Detection Heads  │
│ (feature extract)│────>│ (multi-scale     │────>│ (bbox + class    │
│                  │     │  feature fusion)  │     │  predictions)    │
│ CSPDarknet /     │     │                  │     │                  │
│ EfficientRep    │     │ P3: small objects │     │ Small: 80x80     │
│ Input: 640x640  │     │ P4: medium        │     │ Medium: 40x40    │
│                  │     │ P5: large objects │     │ Large: 20x20     │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

### 15.2 Step 1: Create YOLO Dataset

```
UI Flow:
1. Click "Datasets" in left panel → "New Project"
2. Select template: "object_detection_yolo"
3. Upload images and draw bounding boxes using the annotation tool
4. Or import existing annotations (YOLO format or COCO → convert)
5. Export in YOLO format
```

```bash
# Create YOLO dataset project
curl -X POST http://localhost:8765/api/ds/projects -d '{
  "name": "my_detector",
  "template_id": "object_detection_yolo",
  "labels": ["car", "person", "bicycle", "traffic_light"]
}'

# Add annotated samples
curl -X POST http://localhost:8765/api/ds/projects/{id}/samples/bulk -d '{
  "samples": [
    {
      "image_path": "./images/img_001.jpg",
      "annotations": [
        {"class": 0, "x_center": 0.5, "y_center": 0.3, "width": 0.2, "height": 0.4},
        {"class": 1, "x_center": 0.8, "y_center": 0.6, "width": 0.1, "height": 0.3}
      ]
    }
  ]
}'

# Convert from COCO format if you have existing annotations
curl -X POST http://localhost:8765/api/ds/convert -d '{
  "input_path": "./annotations/coco.json",
  "input_format": "coco",
  "output_format": "yolo"
}'

# Export dataset
curl -X POST http://localhost:8765/api/ds/projects/{id}/export -d '{"format": "yolo"}'
```

**YOLO dataset structure:**
```
dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       └── img_100.jpg
├── labels/
│   ├── train/
│   │   ├── img_001.txt    # class x_center y_center width height
│   │   └── img_002.txt
│   └── val/
│       └── img_100.txt
└── data.yaml              # class names, paths
```

### 15.3 Step 2: Train with Workspace IDE

```
UI Flow:
1. Click "IDE" in top bar
2. New Project → "YOLO Detection" template
3. The template generates ready-to-run training scripts
4. Edit data.yaml with your dataset paths and class names
5. Ctrl+Enter to run training
```

```python
# In the IDE — train_yolo.py (auto-generated by YOLO template)
from ultralytics import YOLO

# Load base model (pretrained on COCO)
model = YOLO("yolov8n.pt")  # nano (fastest)
# model = YOLO("yolov8s.pt")  # small (balanced)
# model = YOLO("yolov8m.pt")  # medium (accurate)
# model = YOLO("yolov8l.pt")  # large (most accurate)

# Train on your dataset
results = model.train(
    data="./data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.01,           # final LR = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,             # box loss gain
    cls=0.5,             # classification loss gain
    dfl=1.5,             # distribution focal loss gain
    close_mosaic=10,     # disable mosaic last 10 epochs
    augment=True,
    # GPU settings
    device=0,            # GPU 0 (or "cpu", or "0,1" for multi-GPU)
    workers=8,
    project="./runs",
    name="my_detector",
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Export for deployment
model.export(format="onnx", dynamic=True, simplify=True)
model.export(format="torchscript")
model.export(format="tflite")  # for mobile
```

```bash
# Run from IDE
curl -X POST http://localhost:8765/api/workspace/run -d '{
  "code": "from ultralytics import YOLO; model = YOLO(\"yolov8n.pt\"); model.train(data=\"data.yaml\", epochs=3, imgsz=640)"
}'
```

### 15.4 Step 3: Model Selection Guide

| Model | Params | mAP50-95 | Speed (ms) | Use Case |
|-------|--------|----------|-----------|----------|
| YOLOv8n | 3.2M | 37.3 | 1.2 | Edge devices, real-time video |
| YOLOv8s | 11.2M | 44.9 | 2.1 | Balanced speed/accuracy |
| YOLOv8m | 25.9M | 50.2 | 4.7 | General production |
| YOLOv8l | 43.7M | 52.9 | 7.1 | High accuracy required |
| YOLOv8x | 68.2M | 53.9 | 12.1 | Maximum accuracy |
| YOLOv9c | 25.3M | 53.0 | 5.4 | Latest architecture |
| YOLO11n | 2.6M | 39.5 | 1.5 | Latest nano model |

### 15.5 Step 4: Production Deployment

```bash
# Export trained model to ONNX
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./detector.onnx"}'

# Generate inference server with YOLO-specific preprocessing
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./detector.onnx", "model_type": "onnx", "port": 8080
}'

# Generate Docker container
curl -X POST http://localhost:8765/api/deploy/dockerfile -d '{
  "model_path": "./detector.onnx", "model_type": "onnx"
}'
```

### 15.6 Production Quality Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| mAP50 | > 0.60 | > 0.80 | > 0.90 |
| mAP50-95 | > 0.40 | > 0.55 | > 0.65 |
| Precision | > 0.70 | > 0.85 | > 0.95 |
| Recall | > 0.70 | > 0.85 | > 0.90 |
| Inference (GPU) | < 20ms | < 10ms | < 5ms |
| Inference (CPU) | < 100ms | < 50ms | < 20ms |

### 15.7 Advanced: Train YOLO Backbone from Scratch in Graph Tab

```bash
# Use the yolov8_backbone template
curl -X POST http://localhost:8765/api/templates/yolov8_backbone/apply

# Or build the DETR detector template
curl -X POST http://localhost:8765/api/templates/detr_detector/apply
```

---

## Tutorial 16: OCR — Optical Character Recognition for Production

Build production OCR systems for document processing, license plates, receipts, handwriting, and scene text recognition.

---

### 16.1 Architecture Overview

Modern OCR has two stages:

```
┌──────────────────┐     ┌──────────────────────────────┐
│ Text Detection    │     │ Text Recognition              │
│ (find text boxes) │────>│ (read characters in each box) │
│                   │     │                               │
│ YOLO / CRAFT /   │     │ CRNN / TrOCR / PaddleOCR     │
│ EAST / DBNet     │     │                               │
│ Input: image     │     │ Input: cropped text region    │
│ Output: bboxes   │     │ Output: text string           │
└──────────────────┘     └──────────────────────────────┘
```

### 16.2 Method A: Fine-Tune TrOCR (Transformer OCR — SOTA)

TrOCR is the current state-of-the-art for text recognition. It uses a ViT encoder + GPT-2 decoder.

```
UI Flow:
1. Click "HuggingFace" in top bar
2. Search: "microsoft/trocr-base-handwritten" → Load
3. Load your OCR dataset (images + text labels)
4. Apply LoRA for efficient fine-tuning
5. Train → Inference
```

```bash
# Load TrOCR (choose variant based on your use case)
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "microsoft/trocr-base-printed"
}'
# Other options:
# "microsoft/trocr-base-handwritten"  — handwritten text
# "microsoft/trocr-large-printed"     — highest accuracy
# "microsoft/trocr-small-printed"     — fastest

# Load OCR dataset (CSV: image_path, text)
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "csv", "path": "./ocr_data.csv",
  "text_col": "text", "label_col": "text"
}'

# Apply LoRA
curl -X POST http://localhost:8765/api/hf/model/lora -d '{
  "r": 16, "lora_alpha": 32, "lora_dropout": 0.1
}'

# Train
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 10, "learning_rate": 5e-5, "batch_size": 8
}'

# Run OCR on new images
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "./test_image.png"
}'
```

### 16.3 Method B: Detection + Recognition Pipeline

For production document processing, use a two-stage pipeline:

**Stage 1: Text Detection with YOLO**

```python
# In the IDE — detect_text.py
from ultralytics import YOLO

# Train text detector on your document images
model = YOLO("yolov8s.pt")
model.train(
    data="text_detection.yaml",  # class: ["text"]
    epochs=50,
    imgsz=1024,  # higher resolution for small text
    batch=8,
)

# Detect text regions
results = model.predict("document.jpg", conf=0.5)
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    # Crop each text region for recognition
```

**Stage 2: Text Recognition**

```bash
# Load recognition model
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "microsoft/trocr-base-printed"
}'

# For each detected region, run recognition
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "./cropped_text_region.png"
}'
```

### 16.4 Method C: Build OCR from Scratch (CRNN Architecture)

```
UI Flow:
1. Graph tab → Build CRNN architecture:
   - Conv2d (1→32, k=3, p=1) + ReLU + MaxPool2d(2)
   - Conv2d (32→64, k=3, p=1) + ReLU + MaxPool2d(2)
   - Conv2d (64→128, k=3, p=1) + ReLU
   - Conv2d (128→128, k=3, p=1) + ReLU + MaxPool2d((2,1))
   - Flatten spatial → LSTM (input=128*H, hidden=256, bidirectional=True)
   - Linear (512 → n_classes)
2. Train with CTC loss (CTCLoss is available in the loss selector)
3. Dataset: image patches with text labels
```

```bash
# Build CRNN for OCR
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "Conv2d",
  "params": {"in_channels": 1, "out_channels": 32, "kernel_size": 3, "padding": 1},
  "activation": "ReLU"
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "MaxPool2d", "params": {"kernel_size": 2}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "Conv2d",
  "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1},
  "activation": "ReLU"
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "MaxPool2d", "params": {"kernel_size": 2}
}'
curl -X POST http://localhost:8765/api/graph/layer -d '{
  "layer_type": "Conv2d",
  "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "padding": 1},
  "activation": "ReLU"
}'

# Configure with CTC loss
curl -X POST http://localhost:8765/api/config -d '{
  "epochs": 50, "learning_rate": 1e-4, "optimizer": "Adam",
  "loss": "CTCLoss"
}'
```

### 16.5 OCR Model Selection Guide

| Model | Type | Best For | Accuracy | Speed |
|-------|------|----------|----------|-------|
| TrOCR-large | Transformer | Printed + handwritten | SOTA (CER < 3%) | Medium |
| TrOCR-base | Transformer | General purpose | Very good | Fast |
| PaddleOCR | CNN+LSTM | Multi-language | Excellent | Very fast |
| EasyOCR | CRAFT+CRNN | Scene text | Good | Fast |
| Tesseract | Classic | Clean documents | Moderate | Very fast |
| CRNN (custom) | CNN+LSTM | Domain-specific | Depends on data | Fast |

### 16.6 Production Deployment

```bash
# Export detection model
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./text_detector.onnx"}'

# Export recognition model
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./text_recognizer.onnx"}'

# Generate combined inference server
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./text_recognizer.onnx", "model_type": "onnx", "port": 8080
}'

# Docker deployment
curl -X POST http://localhost:8765/api/deploy/dockerfile -d '{
  "model_path": "./text_recognizer.onnx", "model_type": "onnx"
}'
```

---

## Tutorial 17: ASR — Automatic Speech Recognition for Production

Build production speech-to-text systems. Fine-tune Whisper or build from scratch with Conformer.

---

### 17.1 Architecture Overview

```
┌───────────────┐     ┌──────────────┐     ┌────────────────┐
│ Audio Frontend │     │ Encoder      │     │ Decoder        │
│               │     │              │     │                │
│ Waveform →    │────>│ Conformer /  │────>│ Transformer /  │
│ Mel Spectrogram│     │ Transformer  │     │ CTC / Hybrid  │
│ (80 mel bins)  │     │ (contextual) │     │ (text output)  │
└───────────────┘     └──────────────┘     └────────────────┘
```

### 17.2 Method A: Fine-Tune Whisper (Recommended for Production)

OpenAI Whisper is the current SOTA for general ASR. Fine-tuning on your domain data dramatically improves accuracy.

```
UI Flow:
1. Click "HuggingFace" in top bar
2. Search: "openai/whisper-small" → Load
3. Task: "automatic-speech-recognition"
4. Load your audio dataset (audio files + transcriptions)
5. Apply LoRA → Train → Inference
```

```bash
# Load Whisper (choose size based on your needs)
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "openai/whisper-small",
  "task": "automatic-speech-recognition"
}'
# Options:
# "openai/whisper-tiny"     — 39M params, fastest, WER ~7.6%
# "openai/whisper-base"     — 74M params, fast, WER ~5.0%
# "openai/whisper-small"    — 244M params, balanced, WER ~3.4%
# "openai/whisper-medium"   — 769M params, accurate, WER ~2.9%
# "openai/whisper-large-v3" — 1.5B params, SOTA, WER ~2.0%

# Prepare ASR dataset (CSV: audio_path, transcription)
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "csv", "path": "./asr_data.csv",
  "text_col": "transcription", "label_col": "transcription"
}'

# Apply LoRA for efficient fine-tuning
curl -X POST http://localhost:8765/api/hf/model/lora -d '{
  "r": 16, "lora_alpha": 32, "lora_dropout": 0.1
}'

# Freeze encoder, train decoder only (most efficient)
curl -X POST http://localhost:8765/api/hf/model/freeze -d '{
  "patterns": ["model.encoder"]
}'

# Train
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 5, "learning_rate": 1e-5, "batch_size": 4,
  "warmup_steps": 500, "max_grad_norm": 1.0, "fp16": true
}'

# Transcribe audio
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "./test_audio.wav"
}'
```

### 17.3 Dataset Preparation for ASR

```bash
# Create ASR dataset project
curl -X POST http://localhost:8765/api/ds/projects -d '{
  "name": "asr_training_data",
  "template_id": "speech_to_text"
}'

# Add samples
curl -X POST http://localhost:8765/api/ds/projects/{id}/samples/bulk -d '{
  "samples": [
    {"audio_path": "./audio/001.wav", "transcription": "hello world"},
    {"audio_path": "./audio/002.wav", "transcription": "how are you today"}
  ]
}'
```

**Production ASR data requirements:**

| Requirement | Minimum | Recommended |
|------------|---------|------------|
| Total audio | 10 hours | 100+ hours |
| Sample rate | 16kHz | 16kHz (Whisper native) |
| Format | WAV, FLAC, MP3 | WAV 16-bit PCM |
| Transcription accuracy | > 95% | > 99% (human-verified) |
| Speaker diversity | 10+ speakers | 100+ speakers |
| Noise conditions | Clean | Clean + noisy + reverberant |
| Domain coverage | Target domain | Target + general |

### 17.4 Method B: Build Conformer ASR from Scratch

```bash
# Use the built-in Conformer ASR template
curl -X POST http://localhost:8765/api/templates/conformer_asr/apply
# This creates: MelSpectrogram → AudioConvBlock → 4x ConformerBlock → Linear

# Or use the Whisper encoder template
curl -X POST http://localhost:8765/api/templates/whisper_encoder/apply
# This creates: MelSpectrogram → Conv stems → 4x TransformerBlock → SequencePool → Linear
```

### 17.5 Whisper Model Selection Guide

| Model | Params | English WER | Multilingual WER | VRAM | Speed (RTF) |
|-------|--------|-------------|------------------|------|-------------|
| whisper-tiny | 39M | 7.6% | 14.5% | 1GB | 0.03x |
| whisper-base | 74M | 5.0% | 10.3% | 1GB | 0.05x |
| whisper-small | 244M | 3.4% | 7.6% | 2GB | 0.12x |
| whisper-medium | 769M | 2.9% | 5.8% | 5GB | 0.35x |
| whisper-large-v3 | 1.5B | 2.0% | 4.2% | 10GB | 0.7x |

*RTF = Real-Time Factor (lower = faster). RTF < 1 means faster than real-time.*

### 17.6 Production Deployment

```bash
# Export to ONNX for fast inference
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./whisper_finetuned.onnx"}'

# Generate streaming inference server
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./whisper_finetuned.onnx", "model_type": "onnx", "port": 8080
}'

# Generate Gradio demo
curl -X POST http://localhost:8765/api/deploy/gradio -d '{
  "model_path": "./whisper_finetuned.pt", "model_type": "pytorch"
}'
```

### 17.7 Production Quality Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Word Error Rate (WER) | < 15% | < 8% | < 3% |
| Character Error Rate (CER) | < 10% | < 5% | < 2% |
| Real-Time Factor (RTF) | < 1.0 | < 0.3 | < 0.1 |
| Latency (first word) | < 2s | < 500ms | < 200ms |
| Streaming support | Chunked | VAD + chunked | WebSocket streaming |

---

## Tutorial 18: TTS — Text-to-Speech for Production

Build natural-sounding speech synthesis systems. From single-speaker to multi-speaker, emotional to expressive speech.

---

### 18.1 Architecture Overview

Modern TTS has evolved through several generations:

```
Gen 1: Concatenative (Festival, eSpeak)          — robotic
Gen 2: Statistical Parametric (HTS)               — smoother but metallic
Gen 3: Neural seq2seq (Tacotron 2 + WaveGlow)     — natural but slow
Gen 4: Non-autoregressive (FastSpeech 2 + HiFi-GAN) — natural + fast
Gen 5: Codec-based (VALL-E, Bark, XTTS)          — zero-shot voice cloning
```

### 18.2 Method A: Fine-Tune Bark (Zero-Shot TTS — Recommended)

Bark by Suno can generate speech, music, and sound effects from text prompts.

```bash
# Load Bark
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "suno/bark-small"
}'

# Or use the full model for highest quality
# "suno/bark" — 1.3B params, very high quality
# "suno/bark-small" — 400M params, faster

# Generate speech
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "Hello! [laughs] This is a text-to-speech demonstration.",
  "voice_preset": "v2/en_speaker_6"
}'
```

### 18.3 Method B: Fine-Tune SpeechT5 (Microsoft — Production Grade)

```bash
# Load SpeechT5
curl -X POST http://localhost:8765/api/hf/load -d '{
  "model_id": "microsoft/speecht5_tts"
}'

# Create TTS dataset
curl -X POST http://localhost:8765/api/ds/projects -d '{
  "name": "tts_training",
  "template_id": "text_to_speech"
}'

# Add training samples (text + audio pairs)
curl -X POST http://localhost:8765/api/ds/projects/{id}/samples/bulk -d '{
  "samples": [
    {"text": "The weather is nice today.", "audio_path": "./wavs/001.wav"},
    {"text": "Please turn left at the intersection.", "audio_path": "./wavs/002.wav"}
  ]
}'

# Load dataset
curl -X POST http://localhost:8765/api/hf/datasets/local -d '{
  "format": "json", "path": "./tts_data.jsonl"
}'

# Fine-tune
curl -X POST http://localhost:8765/api/hf/train -d '{
  "epochs": 20, "learning_rate": 1e-5, "batch_size": 4
}'

# Generate speech
curl -X POST http://localhost:8765/api/hf/inference -d '{
  "input": "This is my custom trained TTS voice."
}'
```

### 18.4 Method C: Build TTS from Scratch (FastSpeech2 + HiFi-GAN)

Use the Workspace IDE for full control:

```python
# In the IDE — fastspeech2.py
import torch
import torch.nn as nn

class FastSpeech2(nn.Module):
    """Non-autoregressive TTS: text → mel spectrogram."""
    def __init__(self, vocab_size=256, d_model=256, n_heads=4,
                 n_encoder_layers=4, n_decoder_layers=4, n_mels=80):
        super().__init__()
        self.phoneme_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 2000, d_model) * 0.02)

        # Encoder (phoneme → hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=n_encoder_layers
        )

        # Variance adaptor (duration, pitch, energy prediction)
        self.duration_pred = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.pitch_pred = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.energy_pred = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # Decoder (hidden → mel)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=n_decoder_layers
        )
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, phoneme_ids, durations=None):
        x = self.phoneme_embed(phoneme_ids)
        T = x.size(1)
        x = x + self.pos_enc[:, :T, :]
        encoder_out = self.encoder(x)

        # Predict variance
        dur_pred = self.duration_pred(encoder_out).squeeze(-1)
        pitch_pred = self.pitch_pred(encoder_out)
        energy_pred = self.energy_pred(encoder_out)

        # Length regulation (expand phonemes to mel frames)
        if durations is not None:
            expanded = self._length_regulate(encoder_out, durations)
        else:
            expanded = self._length_regulate(encoder_out,
                                             dur_pred.round().long().clamp(min=1))

        decoder_out = self.decoder(expanded)
        mel_out = self.mel_linear(decoder_out)
        return mel_out, dur_pred, pitch_pred, energy_pred

    def _length_regulate(self, x, durations):
        """Expand each phoneme embedding by its predicted duration."""
        output = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], durations[i], dim=0)
            output.append(expanded)
        return nn.utils.rnn.pad_sequence(output, batch_first=True)

# Training: MSE loss on mel + duration loss + pitch loss + energy loss
```

### 18.5 TTS Model Selection Guide

| Model | Quality (MOS) | Speed | Zero-Shot | Languages | Best For |
|-------|---------------|-------|-----------|-----------|----------|
| **Bark** | 4.0 | Slow | Yes | 13 | Expressive, emotions, sound effects |
| **XTTS v2** | 4.2 | Medium | Yes | 17 | Voice cloning, multilingual |
| **SpeechT5** | 3.8 | Fast | No | 1 (en) | Single-speaker, fine-tuning |
| **FastSpeech2** | 3.7 | Very fast | No | 1 | Low-latency production |
| **VITS** | 4.0 | Fast | No | 1 | End-to-end, high quality |
| **Tortoise** | 4.3 | Very slow | Yes | 1 (en) | Maximum quality, offline |

*MOS = Mean Opinion Score (1-5 scale, 5 = human quality)*

### 18.6 Production TTS Data Requirements

| Requirement | Minimum | Production |
|------------|---------|-----------|
| Total audio | 1 hour | 10-20 hours |
| Sample rate | 22.05kHz | 24kHz |
| Speaker | Single | Single (per voice) |
| Transcription | Phoneme-aligned | Phoneme-aligned + prosody |
| Recording quality | Studio | Professional studio |
| Text coverage | Common phonemes | Full phoneme set + rare words |

### 18.7 Production Deployment

```bash
# Export model
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./tts_model.onnx"}'

# Generate streaming TTS server
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./tts_model.onnx", "model_type": "onnx", "port": 8080
}'

# Docker container
curl -X POST http://localhost:8765/api/deploy/dockerfile -d '{
  "model_path": "./tts_model.onnx", "model_type": "onnx"
}'

# Gradio demo with audio player
curl -X POST http://localhost:8765/api/deploy/gradio -d '{
  "model_path": "./tts_model.pt", "model_type": "pytorch"
}'
```

### 18.8 Production Quality Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| MOS (naturalness) | > 3.5 | > 4.0 | > 4.3 |
| MOS (intelligibility) | > 4.0 | > 4.5 | > 4.8 |
| Latency (first audio) | < 1s | < 300ms | < 100ms |
| Real-Time Factor | < 1.0 | < 0.3 | < 0.1 |
| Character Error Rate | < 5% | < 2% | < 1% |
| Speaker similarity (clone) | > 0.80 | > 0.85 | > 0.90 |

---

## Tutorial 19: Build Everything from Scratch — Complete Architectures, Training & Deployment

This tutorial shows how to design, train, evaluate, and deploy **complete systems from zero** — no pretrained models, no fine-tuning. Every weight is randomly initialized and trained on your data. This is the path for researchers, custom domains, and edge cases where no pretrained model fits.

Each section is a complete, self-contained walkthrough using only the Graph tab (visual builder), LLM Builder, and Workspace IDE.

---

### 19.1 Voice Cloning System from Scratch

A complete 3-component system: Speaker Encoder → Synthesizer → Vocoder.

#### Architecture

```
Reference Audio (3-10s)                    Text Input
       │                                       │
       ▼                                       ▼
┌──────────────┐                    ┌──────────────────┐
│ Speaker       │                    │ Phoneme Encoder   │
│ Encoder       │                    │ (text→hidden)     │
│              │                    │                    │
│ Mel→Conv→    │  speaker_emb       │ Embed→Pos→         │
│ LSTM→L2Norm  │───────────────────>│ 4x Transformer     │
│              │                    │ + speaker_emb      │
│ Output:       │                    │                    │
│ 256-d vector  │                    │ Output: hidden seq │
└──────────────┘                    └────────┬───────────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Mel Decoder        │
                                   │ (hidden→mel)       │
                                   │                    │
                                   │ 4x Transformer     │
                                   │ → Linear(→80 mels) │
                                   └────────┬───────────┘
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ Vocoder (HiFi-GAN)│
                                   │ (mel→waveform)     │
                                   │                    │
                                   │ Conv→4x Upsample   │
                                   │ → Conv→Tanh        │
                                   │ Output: 24kHz wav  │
                                   └──────────────────┘
```

#### Step 1: Build Speaker Encoder in Graph Tab

This component learns to map any speaker's voice to a 256-d embedding vector.

```
UI Flow:
1. Graph tab → Reset canvas
2. Add layers in order (click each from "All Layers" palette):
   a. MelSpectrogram — n_mels: 40, n_fft: 512, hop_length: 160
   b. AudioConvBlock — in_channels: 40, out_channels: 256
   c. AudioConvBlock — in_channels: 256, out_channels: 256, stride: 2
   d. Transpose — dim0: 1, dim1: 2
   e. LSTM — input_size: 256, hidden_size: 256, num_layers: 3
   f. Linear — in_features: 256, out_features: 256
3. Config tab (right sidebar):
   - Epochs: 100
   - Learning Rate: 0.0001
   - Optimizer: Adam
   - Loss: CrossEntropyLoss
   - Batch Size: 32
   - Scheduler: CosineAnnealingLR (T_max: 100)
4. Dataset: Load speaker classification dataset (each speaker = 1 class)
   - Use "random" or load a CSV with audio paths + speaker IDs
5. Click Build → Click Train
6. Watch loss decrease in Metrics tab
7. Save Model when loss plateaus
```

```bash
# Complete API flow for speaker encoder

# 1. Build architecture
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "MelSpectrogram", "params": {"n_mels": 40, "n_fft": 512, "hop_length": 160}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "AudioConvBlock", "params": {"in_channels": 40, "out_channels": 256}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "AudioConvBlock", "params": {"in_channels": 256, "out_channels": 256, "stride": 2}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "LSTM", "params": {"input_size": 256, "hidden_size": 256, "num_layers": 3}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Linear", "params": {"in_features": 256, "out_features": 256}}'

# 2. Configure training
curl -X POST http://localhost:8765/api/config -d '{
  "epochs": 100, "learning_rate": 0.0001, "optimizer": "Adam",
  "loss": "CrossEntropyLoss", "batch_size": 32,
  "scheduler": "CosineAnnealingLR", "scheduler_params": {"T_max": 100}
}'

# 3. Load dataset
curl -X POST http://localhost:8765/api/data/load -d '{"dataset": "random", "type": "synthetic", "n_samples": 1000}'

# 4. Build model
curl -X POST http://localhost:8765/api/build

# 5. Train
curl -X POST http://localhost:8765/api/train/start

# 6. Monitor
curl http://localhost:8765/api/status
curl http://localhost:8765/api/metrics/loss

# 7. Save trained encoder
curl -X POST http://localhost:8765/api/model/save -d '{"path": "./models/speaker_encoder.pt"}'

# 8. Export architecture
curl http://localhost:8765/api/export/architecture > speaker_encoder_arch.json
curl http://localhost:8765/api/export/python > speaker_encoder.py
```

#### Step 2: Build Mel Synthesizer in the IDE

The synthesizer takes text + speaker embedding and generates mel spectrograms.

```
UI Flow:
1. Click "IDE" in top bar
2. New Project → "Empty"
3. Create file: synthesizer.py
4. Paste the code below
5. Create file: train_synth.py
6. Paste training loop
7. Ctrl+Enter to run training
```

```python
# synthesizer.py — Complete Tacotron2-style mel synthesizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DurationPredictor(nn.Module):
    """Predicts how many mel frames each phoneme should last."""
    def __init__(self, d_model, kernel_size=3, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            ])
        self.convs = nn.ModuleList(layers)
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D)
        h = x.transpose(1, 2)  # (B, D, T)
        for i in range(0, len(self.convs), 3):
            h = self.convs[i](h)        # Conv1d
            h = self.convs[i+1](h)      # ReLU
            h_t = h.transpose(1, 2)      # (B, T, D)
            h_t = self.convs[i+2](h_t)  # LayerNorm
            h = h_t.transpose(1, 2)      # (B, D, T)
        return self.proj(h.transpose(1, 2)).squeeze(-1)  # (B, T)

class LengthRegulator(nn.Module):
    """Expand phoneme embeddings by predicted durations."""
    def forward(self, x, durations):
        # x: (B, T_text, D), durations: (B, T_text) integers
        outputs = []
        for i in range(x.size(0)):
            dur = durations[i].clamp(min=1)
            expanded = torch.repeat_interleave(x[i], dur, dim=0)
            outputs.append(expanded)
        # Pad to same length
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(len(outputs), max_len, x.size(2), device=x.device)
        for i, o in enumerate(outputs):
            padded[i, :o.size(0)] = o
        return padded

class MelSynthesizer(nn.Module):
    """Complete text-to-mel synthesizer with speaker conditioning.

    Architecture:
        Phoneme Embedding + Speaker Projection + Positional Encoding
        → Transformer Encoder (4 layers)
        → Duration Predictor + Length Regulator
        → Transformer Decoder (4 layers)
        → Linear projection to 80 mel bins
    """
    def __init__(self, vocab_size=256, d_model=256, n_heads=4,
                 n_enc_layers=4, n_dec_layers=4, speaker_dim=256, n_mels=80):
        super().__init__()
        self.d_model = d_model
        self.phoneme_embed = nn.Embedding(vocab_size, d_model)
        self.speaker_proj = nn.Linear(speaker_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=n_enc_layers,
        )

        self.duration_pred = DurationPredictor(d_model)
        self.length_reg = LengthRegulator()

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=n_dec_layers,
        )

        self.mel_proj = nn.Linear(d_model, n_mels)

    def forward(self, phoneme_ids, speaker_emb, durations=None):
        """
        Args:
            phoneme_ids: (B, T_text) phoneme token IDs
            speaker_emb: (B, 256) speaker embedding from encoder
            durations: (B, T_text) ground-truth durations (training only)
        Returns:
            mel_out: (B, T_mel, 80) predicted mel spectrogram
            dur_pred: (B, T_text) predicted durations
        """
        x = self.phoneme_embed(phoneme_ids)
        x = self.pos_enc(x)

        # Add speaker identity to every position
        spk = self.speaker_proj(speaker_emb).unsqueeze(1)
        x = x + spk

        encoded = self.encoder(x)
        dur_pred = self.duration_pred(encoded)

        if durations is not None:
            regulated = self.length_reg(encoded, durations)
        else:
            regulated = self.length_reg(encoded, dur_pred.round().long().clamp(min=1))

        decoded = self.decoder(self.pos_enc(regulated))
        mel_out = self.mel_proj(decoded)
        return mel_out, dur_pred
```

```python
# train_synth.py — Training loop for the synthesizer
import torch
import torch.nn.functional as F

# Initialize
model = MelSynthesizer(vocab_size=256, d_model=256, n_mels=80)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training loop
for epoch in range(200):
    model.train()
    # Your dataloader provides: phoneme_ids, speaker_emb, target_mel, durations
    # For demo, use synthetic data:
    B, T_text, T_mel = 8, 50, 200
    phoneme_ids = torch.randint(0, 256, (B, T_text))
    speaker_emb = torch.randn(B, 256)
    target_mel = torch.randn(B, T_mel, 80)
    durations = torch.ones(B, T_text, dtype=torch.long) * (T_mel // T_text)

    mel_pred, dur_pred = model(phoneme_ids, speaker_emb, durations)

    # Mel reconstruction loss
    mel_loss = F.mse_loss(mel_pred[:, :T_mel], target_mel)
    # Duration prediction loss
    dur_loss = F.mse_loss(dur_pred.float(), durations.float())
    # Total loss
    loss = mel_loss + 0.1 * dur_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: mel_loss={mel_loss:.4f} dur_loss={dur_loss:.4f}")

torch.save(model.state_dict(), "synthesizer.pt")
print("Training complete!")
```

#### Step 3: Build Vocoder (HiFi-GAN) in the IDE

The vocoder converts mel spectrograms to audible waveforms.

```python
# vocoder.py — HiFi-GAN generator (mel → waveform)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=(kernel_size * d - d) // 2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1,
                          padding=(kernel_size - 1) // 2),
            ))

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x

class HiFiGANGenerator(nn.Module):
    """Converts 80-bin mel spectrogram to 24kHz waveform.

    Architecture:
        Conv1d(80→512) → 4x [Upsample → ResBlocks] → Conv1d(32→1) → Tanh
        Upsample rates: [8, 8, 2, 2] → total 256x upsampling
        Input mel hop_length should be 256 to match.
    """
    def __init__(self, n_mels=80, upsample_rates=[8, 8, 2, 2],
                 upsample_channels=[512, 256, 128, 64, 32],
                 resblock_kernel_sizes=[3, 7, 11]):
        super().__init__()
        self.conv_pre = nn.Conv1d(n_mels, upsample_channels[0], 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (rate, ch_out) in enumerate(zip(upsample_rates, upsample_channels[1:])):
            ch_in = upsample_channels[i]
            self.ups.append(nn.ConvTranspose1d(
                ch_in, ch_out, rate * 2, stride=rate, padding=rate // 2
            ))
            for k in resblock_kernel_sizes:
                self.resblocks.append(ResBlock(ch_out, k, [1, 3, 5]))

        self.conv_post = nn.Conv1d(upsample_channels[-1], 1, 7, padding=3)

    def forward(self, mel):
        """mel: (B, 80, T_mel) → waveform: (B, 1, T_mel * 256)"""
        x = self.conv_pre(mel)
        n_res_per_up = len([3, 7, 11])  # resblock_kernel_sizes
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            res_sum = None
            for j in range(n_res_per_up):
                rb = self.resblocks[i * n_res_per_up + j]
                if res_sum is None:
                    res_sum = rb(x)
                else:
                    res_sum = res_sum + rb(x)
            x = res_sum / n_res_per_up
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)

# Train with: L1 mel loss + multi-scale discriminator loss
# (discriminator code omitted for brevity — use adversarial training)
```

#### Step 4: Inference Pipeline

```python
# inference.py — Put it all together
import torch

# Load all 3 components
speaker_encoder = torch.load("speaker_encoder.pt")
synthesizer = MelSynthesizer(256, 256, n_mels=80)
synthesizer.load_state_dict(torch.load("synthesizer.pt"))
vocoder = HiFiGANGenerator()
vocoder.load_state_dict(torch.load("vocoder.pt"))

# Step 1: Get speaker embedding from reference audio
ref_audio = torch.randn(1, 1, 48000)  # 3 seconds at 16kHz
speaker_emb = speaker_encoder(ref_audio)  # (1, 256)

# Step 2: Convert text to phoneme IDs
text = "Hello, this is my cloned voice."
phoneme_ids = torch.tensor([[ord(c) for c in text]])  # simplified

# Step 3: Generate mel spectrogram
mel, _ = synthesizer(phoneme_ids, speaker_emb)  # (1, T, 80)

# Step 4: Convert mel to waveform
waveform = vocoder(mel.transpose(1, 2))  # (1, 1, T*256)

# Save as WAV
import torchaudio
torchaudio.save("output.wav", waveform.squeeze(0).cpu(), 24000)
print("Generated: output.wav")
```

#### Step 5: Deploy

```bash
# Export all models
curl -X POST http://localhost:8765/api/deploy/onnx -d '{"path": "./models/encoder.onnx"}'
curl -X POST http://localhost:8765/api/deploy/server -d '{
  "model_path": "./models/encoder.onnx", "model_type": "onnx", "port": 8080
}'
curl -X POST http://localhost:8765/api/deploy/gradio -d '{
  "model_path": "./models/synthesizer.pt", "model_type": "pytorch"
}'
```

---

### 19.2 YOLO Object Detector from Scratch

Build a complete object detection system without using any pretrained backbone.

#### Architecture

```
Input Image (640×640×3)
       │
       ▼
┌──────────────────────┐
│ Backbone (CSP-like)   │
│                       │
│ Stem: Conv(3→32,k=3)  │
│ Stage 1: ResNetBlock   │  → P3 features (80×80×128)
│   Conv(32→64,s=2)     │
│   2× ResNetBlock(64)   │
│ Stage 2: ResNetBlock   │  → P4 features (40×40×256)
│   Conv(64→128,s=2)    │
│   2× ResNetBlock(128)  │
│ Stage 3: ResNetBlock   │  → P5 features (20×20×512)
│   Conv(128→256,s=2)   │
│   2× ResNetBlock(256)  │
│ Stage 4: Conv(256→512) │
│   2× ResNetBlock(512)  │
└────┬──────┬──────┬────┘
     │      │      │
     P3     P4     P5
     │      │      │
     ▼      ▼      ▼
┌──────────────────────┐
│ FPN Neck (top-down)   │
│ P5 → Upsample+Conv   │
│     + P4 → Concat     │ → N4
│ N4 → Upsample+Conv   │
│     + P3 → Concat     │ → N3
└────┬──────┬──────┬────┘
     N3     N4     P5
     │      │      │
     ▼      ▼      ▼
┌──────────────────────┐
│ Detection Heads       │
│ Each: Conv→Conv→Conv  │
│ Output per head:       │
│  (B, n_anchors, H, W, │
│   5+n_classes)         │
│  [x, y, w, h, obj,    │
│   class_probs...]      │
└──────────────────────┘
```

#### Build in Graph Tab

```
UI Flow:
1. Graph tab → Reset
2. Build backbone:
   a. Conv2d (3→32, k=3, s=1, p=1) + SiLU
   b. Conv2d (32→64, k=3, s=2, p=1) + SiLU
   c. ResNetBlock (in=64, out=64)
   d. ResNetBlock (in=64, out=64)
   e. Conv2d (64→128, k=3, s=2, p=1) + SiLU
   f. ResNetBlock (in=128, out=128)
   g. ResNetBlock (in=128, out=128)
   h. Conv2d (128→256, k=3, s=2, p=1) + SiLU
   i. ResNetBlock (in=256, out=256)
   j. Conv2d (256→512, k=3, s=2, p=1) + SiLU
   k. ResNetBlock (in=512, out=512)
   l. Flatten
   m. Linear (512 → n_classes * n_anchors * 5)
3. Config: loss=MSELoss, optimizer=AdamW, lr=0.001
4. Load CIFAR-10 dataset (for classification backbone pretraining)
5. Build → Train backbone for 20 epochs
```

```bash
# Build YOLO-like backbone via Graph API
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}, "activation": "SiLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 64, "out_channels": 64}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 64, "out_channels": 64}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 128, "out_channels": 128}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "SiLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 256, "out_channels": 256}}'

# Or use the built-in template
curl -X POST http://localhost:8765/api/templates/yolov8_backbone/apply
```

#### Full YOLO Training in IDE

```python
# yolo_from_scratch.py — Complete YOLO implementation
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 2
        self.conv1 = ConvBNSiLU(channels, mid, k=1, p=0)
        self.conv2 = ConvBNSiLU(mid, channels, k=3, p=1)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class YOLOBackbone(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512]):
        super().__init__()
        self.stem = ConvBNSiLU(3, channels[0], k=3, s=1, p=1)
        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            stage = nn.Sequential(
                ConvBNSiLU(channels[i], channels[i+1], k=3, s=2, p=1),
                ResBlock(channels[i+1]),
                ResBlock(channels[i+1]),
            )
            self.stages.append(stage)

    def forward(self, x):
        features = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features  # [P2, P3, P4, P5]

class FPN(nn.Module):
    """Feature Pyramid Network — fuses multi-scale features."""
    def __init__(self, in_channels=[128, 256, 512], out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            ConvBNSiLU(out_channels, out_channels) for _ in in_channels
        ])

    def forward(self, features):
        # features = [P3, P4, P5] (small to large stride)
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + up
        return [s(l) for s, l in zip(self.smooth, laterals)]

class DetectionHead(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors=3):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels),
            ConvBNSiLU(in_channels, in_channels),
        )
        self.pred = nn.Conv2d(in_channels, n_anchors * (5 + n_classes), 1)
        self.n_anchors = n_anchors
        self.n_classes = n_classes

    def forward(self, x):
        x = self.conv(x)
        pred = self.pred(x)
        B, _, H, W = pred.shape
        pred = pred.view(B, self.n_anchors, 5 + self.n_classes, H, W)
        pred = pred.permute(0, 1, 3, 4, 2)  # (B, A, H, W, 5+C)
        return pred

class YOLOv8Scratch(nn.Module):
    """Complete YOLOv8-style detector built from scratch."""
    def __init__(self, n_classes=80):
        super().__init__()
        self.backbone = YOLOBackbone([32, 64, 128, 256, 512])
        self.fpn = FPN([128, 256, 512], 256)
        self.heads = nn.ModuleList([
            DetectionHead(256, n_classes) for _ in range(3)
        ])

    def forward(self, x):
        features = self.backbone(x)
        # Use P3, P4, P5 (skip P2 — too fine-grained)
        fpn_features = self.fpn(features[1:])
        outputs = [head(f) for head, f in zip(self.heads, fpn_features)]
        return outputs  # 3 scales of predictions

# Training
model = YOLOv8Scratch(n_classes=20)  # VOC has 20 classes
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
# Train with: bbox regression loss (CIoU) + objectness (BCE) + classification (BCE)
```

---

### 19.3 OCR System from Scratch

A complete text detection + recognition system with no pretrained weights.

#### Text Recognition: CRNN with CTC Loss

```
UI Flow (Graph tab):
1. Conv2d (1→64, k=3, p=1) + ReLU → MaxPool2d(2)
2. Conv2d (64→128, k=3, p=1) + ReLU → MaxPool2d(2)
3. Conv2d (128→256, k=3, p=1) + ReLU
4. Conv2d (256→256, k=3, p=1) + ReLU → MaxPool2d((2,1))
5. Conv2d (256→512, k=3, p=1) + ReLU
6. BatchNorm2d(512)
7. Conv2d (512→512, k=3, p=1) + ReLU → MaxPool2d((2,1))
8. Flatten
9. Linear (512*H_remaining → 256)
10. LSTM (input=256, hidden=256, num_layers=2, bidirectional)
11. Linear (512 → n_chars + 1)  # +1 for CTC blank
Loss: CTCLoss
```

#### Full CRNN in IDE

```python
# crnn_ocr.py — Complete OCR from scratch
import torch
import torch.nn as nn

class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for text recognition.

    Input: grayscale image (B, 1, 32, W) — height normalized to 32
    Output: (T, B, n_classes) — CTC-compatible logits
    """
    def __init__(self, n_classes=37):  # 26 letters + 10 digits + blank
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1: (1, 32, W) → (64, 16, W/2)
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2: (64, 16, W/2) → (128, 8, W/4)
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3: (128, 8, W/4) → (256, 8, W/4)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # pool height only
            # Block 4: (256, 4, W/4) → (512, 4, W/4)
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # pool height only
            # Block 5: (512, 2, W/4) → (512, 1, W/4)
            nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True),
        )

        # Bidirectional LSTM
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=False)

        # Output projection
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B, 1, 32, W)
        conv = self.cnn(x)          # (B, 512, 1, W/4)
        conv = conv.squeeze(2)       # (B, 512, W/4)
        conv = conv.permute(2, 0, 1) # (T, B, 512) — T = W/4
        rnn_out, _ = self.rnn(conv)  # (T, B, 512)
        output = self.fc(rnn_out)    # (T, B, n_classes)
        return output

# Training with CTC Loss
model = CRNN(n_classes=37)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# Training loop
for epoch in range(100):
    # Simulated batch: images (B, 1, 32, 128), labels, input_lengths, target_lengths
    B = 16
    images = torch.randn(B, 1, 32, 128)
    # Each label is a sequence of character indices (1-36, 0=blank)
    targets = torch.randint(1, 37, (B, 10))  # max 10 chars per word
    target_lengths = torch.full((B,), 10, dtype=torch.long)
    input_lengths = torch.full((B,), 32, dtype=torch.long)  # T = 128/4 = 32

    logits = model(images)  # (T=32, B, 37)
    log_probs = logits.log_softmax(2)

    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# CTC Decoding
def ctc_decode(logits, idx_to_char):
    """Greedy CTC decoding."""
    preds = logits.argmax(dim=2).permute(1, 0)  # (B, T)
    results = []
    for pred in preds:
        chars = []
        prev = -1
        for idx in pred:
            if idx != 0 and idx != prev:  # skip blank and repeated
                chars.append(idx_to_char[idx.item()])
            prev = idx
        results.append(''.join(chars))
    return results
```

#### Text Detection: Train a Text Detector in Graph Tab

```bash
# Use DETR detector template (works for text detection too)
curl -X POST http://localhost:8765/api/templates/detr_detector/apply

# Or build simpler CNN detector:
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 3, "out_channels": 64, "kernel_size": 3, "padding": 1}, "activation": "ReLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "MaxPool2d", "params": {"kernel_size": 2}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 64, "out_channels": 64}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv2d", "params": {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "ReLU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "ResNetBlock", "params": {"in_channels": 128, "out_channels": 128}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Flatten"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Linear", "params": {"in_features": 128, "out_features": 5}, "activation": null}'
# Output: [x, y, w, h, confidence] per text region
```

---

### 19.4 ASR (Speech-to-Text) from Scratch

Build a complete ASR system using the encoder-decoder architecture.

#### Architecture: Whisper-style (Conv Stem + Transformer)

```
Audio Waveform
      │
      ▼
┌───────────────┐
│ Mel Frontend   │  MelSpectrogram(n_mels=80)
└───────┬───────┘
        │  (B, 80, T)
        ▼
┌───────────────┐
│ Conv Stem      │  Conv1d(80→256, k=3) + GELU
│               │  Conv1d(256→256, k=3, s=2) + GELU
│               │  → downsamples 2x
└───────┬───────┘
        │  (B, 256, T/2)
        ▼
   Transpose → (B, T/2, 256)
        │
        ▼
┌───────────────┐
│ Positional     │  PositionalEncoding(d=256)
│ Encoding       │
└───────┬───────┘
        ▼
┌───────────────┐
│ Transformer    │  6× TransformerBlock(d=256, h=4)
│ Encoder        │
└───────┬───────┘
        │  (B, T/2, 256)
        ▼
┌───────────────┐
│ CTC Head       │  Linear(256 → vocab_size)
│               │  + CTC Loss during training
└───────────────┘
```

#### Build in Graph Tab

```
UI Flow:
1. Graph tab → Reset
2. Add layers:
   a. MelSpectrogram (n_mels=80, n_fft=400, hop_length=160)
   b. Conv1d (80→256, k=3, p=1) + GELU
   c. Conv1d (256→256, k=3, stride=2, p=1) + GELU
   d. Transpose (dim0=1, dim1=2)
   e. PositionalEncoding (d_model=256, max_len=1500)
   f. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   g. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   h. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   i. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   j. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   k. TransformerBlock (d_model=256, n_heads=4, dropout=0.1)
   l. Linear (256 → 5000)  # vocab size
3. Loss: CTCLoss
4. Optimizer: AdamW, lr=3e-4
5. Scheduler: CosineAnnealingWarmRestarts (T_0=10)
6. Build → Train
```

```bash
# Build via API — or use the template:
curl -X POST http://localhost:8765/api/templates/whisper_encoder/apply

# Or build layer by layer:
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "MelSpectrogram", "params": {"n_mels": 80, "n_fft": 400, "hop_length": 160}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv1d", "params": {"in_channels": 80, "out_channels": 256, "kernel_size": 3, "padding": 1}, "activation": "GELU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Conv1d", "params": {"in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1}, "activation": "GELU"}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Transpose", "params": {"dim0": 1, "dim1": 2}}'
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "PositionalEncoding", "params": {"d_model": 256, "max_len": 1500}}'
# Add 6 transformer blocks
for i in $(seq 1 6); do
  curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "TransformerBlock", "params": {"d_model": 256, "n_heads": 4, "dropout": 0.1}}'
done
curl -X POST http://localhost:8765/api/graph/layer -d '{"layer_type": "Linear", "params": {"in_features": 256, "out_features": 5000}}'

# Configure
curl -X POST http://localhost:8765/api/config -d '{
  "epochs": 50, "learning_rate": 3e-4, "optimizer": "AdamW",
  "loss": "CTCLoss", "batch_size": 8,
  "scheduler": "CosineAnnealingWarmRestarts", "scheduler_params": {"T_0": 10}
}'

# Build and train
curl -X POST http://localhost:8765/api/build
curl -X POST http://localhost:8765/api/train/start
```

#### Full ASR in IDE (with Encoder-Decoder)

For better accuracy, use an encoder-decoder model with attention (like Whisper):

```python
# asr_from_scratch.py — Encoder-Decoder ASR
import torch
import torch.nn as nn
import math

class ASRModel(nn.Module):
    """Complete encoder-decoder ASR model.

    Encoder: MelSpectrogram → Conv stems → 6 Transformer layers
    Decoder: Token Embedding → 6 Transformer decoder layers → Linear
    Loss: Cross-entropy on next-token prediction (teacher forcing)
    """
    def __init__(self, n_mels=80, d_model=256, n_heads=4,
                 n_enc_layers=6, n_dec_layers=6, vocab_size=5000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Audio encoder
        self.mel = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=2, padding=1), nn.GELU(),
        )
        self.enc_pos = nn.Parameter(torch.randn(1, 1500, d_model) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.1, batch_first=True),
            num_layers=n_enc_layers,
        )

        # Text decoder
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode(self, mel_input):
        """mel_input: (B, n_mels, T) → (B, T/2, d_model)"""
        x = self.mel(mel_input).transpose(1, 2)
        T = x.size(1)
        x = x + self.enc_pos[:, :T]
        return self.encoder(x)

    def decode(self, token_ids, memory):
        """token_ids: (B, S), memory: (B, T, D) → logits: (B, S, vocab)"""
        S = token_ids.size(1)
        x = self.tok_emb(token_ids) + self.dec_pos[:, :S]
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        out = self.decoder(x, memory, tgt_mask=mask)
        return self.lm_head(out)

    def forward(self, mel_input, token_ids):
        memory = self.encode(mel_input)
        logits = self.decode(token_ids[:, :-1], memory)
        return logits

# Training
model = ASRModel(vocab_size=5000)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # pad=0

for epoch in range(100):
    # Simulated batch
    mel_input = torch.randn(4, 80, 400)  # 4 samples, 80 mels, 400 frames
    token_ids = torch.randint(1, 5000, (4, 50))  # target transcriptions

    logits = model(mel_input, token_ids)  # (B, 49, 5000)
    loss = loss_fn(logits.reshape(-1, 5000), token_ids[:, 1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, ppl={math.exp(loss.item()):.1f}")
```

Or use the LLM Builder's encoder-decoder endpoint:

```bash
# Build encoder-decoder ASR model via LLM Builder
curl -X POST http://localhost:8765/api/llm/encoder-decoder -d '{
  "vocab_size": 5000, "d_model": 256,
  "n_encoder_layers": 6, "n_decoder_layers": 6,
  "n_heads": 4, "max_len": 1500,
  "norm_type": "layernorm", "ffn_type": "standard"
}'

# Train with text data (the LLM trainer works for seq2seq too)
curl -X POST http://localhost:8765/api/llm/train -d '{
  "text": "your transcription corpus here...",
  "tokenizer": "char", "max_len": 256,
  "epochs": 50, "learning_rate": 3e-4
}'
```

---

### 19.5 TTS (Text-to-Speech) from Scratch

Build a complete non-autoregressive TTS system.

#### Architecture: FastSpeech2 + HiFi-GAN

```
Text Input: "Hello world"
      │
      ▼
┌──────────────┐
│ Phoneme       │  Embedding(vocab→256)
│ Encoder       │  + PositionalEncoding
│              │  4× TransformerBlock
└───────┬──────┘
        │  (B, T_text, 256)
        ├─────────────────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│ Duration      │         │ Pitch        │
│ Predictor     │         │ Predictor    │
│ Conv→ReLU→    │         │ Conv→ReLU→   │
│ Conv→Linear(1)│         │ Linear(1)    │
└───────┬──────┘         └──────┬───────┘
        │ durations               │ pitch
        ▼                         ▼
┌──────────────────────────────────┐
│ Length Regulator                   │
│ Expand each phoneme by its        │
│ predicted duration                 │
│ T_text → T_mel frames             │
└───────────────┬──────────────────┘
                │  (B, T_mel, 256)
                ▼
┌──────────────┐
│ Mel Decoder   │  4× TransformerBlock
│              │  Linear(256→80)
│ Output:       │
│ (B, T_mel, 80)│
└───────┬──────┘
        │  mel spectrogram
        ▼
┌──────────────┐
│ HiFi-GAN      │  Conv→4×[Upsample+ResBlocks]→Conv→Tanh
│ Vocoder       │  256x upsampling (mel→24kHz waveform)
│              │
│ Output:       │
│ (B, 1, T_wav) │
└──────────────┘
```

#### Complete TTS in IDE

```python
# tts_from_scratch.py — Complete FastSpeech2 + vocoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ═══════════════════════════════════════════════════════
# Component 1: FastSpeech2 (text → mel spectrogram)
# ═══════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class VariancePredictor(nn.Module):
    """Predicts duration / pitch / energy for each phoneme."""
    def __init__(self, d_model, kernel_size=3, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            ))
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        h = x.transpose(1, 2)
        for layer in self.layers:
            conv, relu, ln, drop = layer[0], layer[1], layer[2], layer[3]
            h = conv(h)
            h = relu(h)
            h = ln(h.transpose(1, 2)).transpose(1, 2)
            h = drop(h)
        return self.proj(h.transpose(1, 2)).squeeze(-1)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_heads=4,
                 n_layers=4, n_mels=80, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=dropout, batch_first=True),
            num_layers=n_layers,
        )
        self.duration_pred = VariancePredictor(d_model)
        self.pitch_pred = VariancePredictor(d_model)
        self.energy_pred = VariancePredictor(d_model)
        self.pitch_embed = nn.Conv1d(1, d_model, 3, padding=1)
        self.energy_embed = nn.Conv1d(1, d_model, 3, padding=1)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=dropout, batch_first=True),
            num_layers=n_layers,
        )
        self.mel_out = nn.Linear(d_model, n_mels)

    def _length_regulate(self, x, durations):
        outputs = []
        for i in range(x.size(0)):
            expanded = torch.repeat_interleave(x[i], durations[i].clamp(min=1), dim=0)
            outputs.append(expanded)
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(len(outputs), max_len, x.size(2), device=x.device)
        for i, o in enumerate(outputs):
            padded[i, :o.size(0)] = o
        return padded

    def forward(self, phonemes, durations=None, pitch=None, energy=None):
        x = self.pos_enc(self.embed(phonemes))
        enc_out = self.encoder(x)

        dur_pred = self.duration_pred(enc_out)
        pitch_pred = self.pitch_pred(enc_out)
        energy_pred = self.energy_pred(enc_out)

        # Add pitch and energy embeddings
        if pitch is not None:
            p_emb = self.pitch_embed(pitch.unsqueeze(1)).transpose(1, 2)
            enc_out = enc_out + p_emb
        if energy is not None:
            e_emb = self.energy_embed(energy.unsqueeze(1)).transpose(1, 2)
            enc_out = enc_out + e_emb

        # Length regulation
        if durations is not None:
            regulated = self._length_regulate(enc_out, durations)
        else:
            regulated = self._length_regulate(
                enc_out, dur_pred.round().long().clamp(min=1)
            )

        decoded = self.decoder(self.pos_enc(regulated))
        mel = self.mel_out(decoded)
        return mel, dur_pred, pitch_pred, energy_pred

# ═══════════════════════════════════════════════════════
# Component 2: HiFi-GAN Vocoder (mel → waveform)
# ═══════════════════════════════════════════════════════

class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d,
                          padding=(kernel_size * d - d) // 2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size,
                          padding=(kernel_size - 1) // 2),
            ))
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x

class Vocoder(nn.Module):
    def __init__(self, n_mels=80):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, 512, 7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
        ])
        self.resblocks = nn.ModuleList([
            ResBlock1d(256, 3, [1, 3, 5]),
            ResBlock1d(128, 3, [1, 3, 5]),
            ResBlock1d(64, 3, [1, 3, 5]),
            ResBlock1d(32, 3, [1, 3, 5]),
        ])
        self.post = nn.Conv1d(32, 1, 7, padding=3)

    def forward(self, mel):
        x = self.pre(mel)
        for up, rb in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = rb(x)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))

# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

# Initialize models
tts = FastSpeech2(vocab_size=256, d_model=256, n_mels=80)
vocoder = Vocoder(n_mels=80)

print(f"FastSpeech2: {sum(p.numel() for p in tts.parameters())/1e6:.1f}M params")
print(f"Vocoder: {sum(p.numel() for p in vocoder.parameters())/1e6:.1f}M params")

# Train FastSpeech2
opt_tts = torch.optim.AdamW(tts.parameters(), lr=1e-4)
for epoch in range(200):
    B, T_text, T_mel = 8, 30, 150
    phonemes = torch.randint(0, 256, (B, T_text))
    target_mel = torch.randn(B, T_mel, 80)
    durations = torch.ones(B, T_text, dtype=torch.long) * (T_mel // T_text)

    mel_pred, dur_p, pitch_p, energy_p = tts(phonemes, durations)
    mel_loss = F.l1_loss(mel_pred[:, :T_mel], target_mel)
    dur_loss = F.mse_loss(dur_p.float(), durations.float())
    loss = mel_loss + 0.1 * dur_loss

    opt_tts.zero_grad()
    loss.backward()
    opt_tts.step()
    if epoch % 20 == 0:
        print(f"[TTS] Epoch {epoch}: mel={mel_loss:.4f} dur={dur_loss:.4f}")

# Train Vocoder (simplified — production uses adversarial training)
opt_voc = torch.optim.AdamW(vocoder.parameters(), lr=2e-4)
for epoch in range(200):
    mel = torch.randn(8, 80, 150)
    target_wav = torch.randn(8, 1, 150 * 256)  # 256x upsampling
    wav_pred = vocoder(mel)
    loss = F.l1_loss(wav_pred[:, :, :target_wav.size(2)], target_wav)
    opt_voc.zero_grad()
    loss.backward()
    opt_voc.step()
    if epoch % 20 == 0:
        print(f"[Vocoder] Epoch {epoch}: loss={loss:.4f}")

# Inference
tts.eval()
vocoder.eval()
with torch.no_grad():
    text = "Hello world"
    tokens = torch.tensor([[ord(c) for c in text]])
    mel, _, _, _ = tts(tokens)
    wav = vocoder(mel.transpose(1, 2))
    print(f"Generated {wav.shape[-1] / 24000:.2f}s of audio")
```

---

### 19.6 From-Scratch Checklist

For any model you build from scratch, follow this production checklist:

| Step | What | Tool |
|------|------|------|
| 1. Design | Draw architecture, choose layers | Graph tab or IDE |
| 2. Prototype | Build smallest viable model, verify forward pass | `POST /api/build` |
| 3. Data | Create/load dataset, verify shapes | Dataset Factory or Data Eng |
| 4. Train | Small run first (2 epochs), check loss decreases | `POST /api/train/start` |
| 5. Scale | Increase model size, data, epochs | Config tab |
| 6. Monitor | Watch loss, gradients, LR in real-time | Metrics tab + WebSocket |
| 7. Evaluate | Run eval metrics (accuracy, WER, MOS, mAP) | `POST /api/eval/auto` |
| 8. Compare | Save experiments, overlay curves | Compare tab |
| 9. Export | ONNX for inference, TorchScript for mobile | `POST /api/deploy/onnx` |
| 10. Deploy | Generate server, Dockerfile, Gradio demo | Deploy tab |
| 11. Profile | Measure latency, memory, throughput | `POST /api/profile` |
| 12. Iterate | Adjust architecture, retrain, compare | Repeat 1-11 |

---

## API Quick Reference

| Endpoint Group | Method | Key Endpoints |
|----------------|--------|---------------|
| **Graph** | CRUD | `/api/graph/layer`, `/api/build`, `/api/train/start`, `/api/export/*` |
| **LLM Builder** | Build+Train | `/api/llm/build`, `/api/llm/compose`, `/api/llm/blueprint/build`, `/api/llm/train`, `/api/llm/generate` |
| **Novel Arch Lab** | Research | `/api/llm/novel/validate`, `/api/llm/novel/experiment`, `/api/llm/novel/arch-search` |
| **HuggingFace** | Fine-tune | `/api/hf/load`, `/api/hf/model/lora`, `/api/hf/train`, `/api/hf/inference` |
| **Unsloth** | Fast tune | `/api/unsloth/load`, `/api/unsloth/train`, `/api/unsloth/save` |
| **RL** | Train agents | `/api/rl/env`, `/api/rl/agent`, `/api/rl/train`, `/api/rl/episode` |
| **Robotics** | Simulate | `/api/robotics/robots`, `/api/robotics/robots/{id}/circuit`, `/api/robotics/robots/{id}/scene` |
| **Physics** | Simulate | `/api/physics/load`, `/api/physics/start`, `/api/physics/force` |
| **Hardware** | Bridge | `/api/hardware/connect`, `/api/hardware/joints`, `/api/hardware/firmware/generate` |
| **Data Eng** | ETL | `/api/dataeng/pipelines`, `/api/dataeng/pipelines/{id}/run` |
| **Dataset Factory** | Create | `/api/ds/projects`, `/api/ds/projects/{id}/samples`, `/api/ds/projects/{id}/export` |
| **Workspace** | IDE | `/api/workspace/projects`, `/api/workspace/run` |
| **Eval** | Metrics | `/api/eval/classification`, `/api/eval/regression`, `/api/eval/auto` |
| **Deploy** | Export | `/api/deploy/onnx`, `/api/deploy/torchscript`, `/api/deploy/server`, `/api/deploy/gradio` |
| **Diffusion** | Train | `/api/diffusion/train`, `/api/diffusion/generate`, `/api/diffusion/train-vae` |
| **Collaboration** | Multi-user | `/api/collab/rooms`, `/api/collab/chat`, `/api/collab/cursor` |

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v   # 994 tests
```

994 tests covering all modules:

| Test File | Tests | Coverage |
|-----------|-------|---------|
| `test_core.py` | Core layers, registry, graph, engine, metrics, schedulers |
| `test_server.py` | Basic API endpoints, config, templates |
| `test_llm.py` | LLM layers, model building, modification, training |
| `test_graph_functional.py` | 151 tests — 31 templates, 45 layers, 10 activations, 5 optimizers, 10 schedulers, training |
| `test_llm_functional.py` | 86 tests — 18 blueprints, 18 block designs, configs, modify, novel arch lab, training, generation |
| `test_hf_functional.py` | 37 tests — model loading, surgery, LoRA, datasets, training, inference |
| `test_rl_functional.py` | 37 tests — environments, algorithms, custom grid, training flow, save/load |
| `test_robotics_functional.py` | 51 tests — components, templates, circuits, 3D scenes, joints, physics, full simulation |
| `test_topbar_functional.py` | 58 tests — data engineering, dataset factory, workspace, evaluation, deployment |
| `test_robotics.py` | Unit tests for simulator, circuit solver, robot builder |
| `test_dataeng.py` | Unit tests for connectors, pipelines, transforms |
| `test_datasets.py` | Unit tests for dataset factory templates, projects, converters |
| `test_workspace.py` | Unit tests for project manager, code executor |
| `test_novel_architectures.py` | Unit tests for composable blocks, novel designs |
| `test_graph_extended.py` | Extended graph builder tests |
| `test_hf_features.py` | HuggingFace unit tests |
| `test_metrics.py` | Metrics collector tests |
