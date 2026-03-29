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

The left sidebar has an **Architecture Templates** section with 12 pre-built architectures. Click any template to instantly load it:

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

## API Quick Reference (New Endpoints)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/llm/blueprints` | GET | List all 18 model blueprints |
| `/api/llm/blueprint/build` | POST | Build model from blueprint + scale |
| `/api/llm/blueprint/modify` | POST | Modify blueprint model config |
| `/api/llm/blueprint/build-components` | POST | Build multi-component models (VeO3, SD) |
| `/api/llm/novel/validate` | POST | Validate novel block design |
| `/api/llm/novel/experiment` | POST | Quick-train novel architecture |
| `/api/llm/novel/templates` | GET | Get 6 research starter templates |
| `/api/llm/novel/model-from-code` | POST | Build full model from Python code |
| `/api/llm/novel/custom-loss` | POST | Set custom loss function |
| `/api/llm/novel/arch-search` | POST | Compare multiple architectures |
| `/api/diffusion/train` | POST | Train diffusion UNet |
| `/api/diffusion/train-vae` | POST | Train VAE for latent diffusion |
| `/api/diffusion/generate` | POST | Generate images from trained model |

---

## Tests

```bash
pip install pytest httpx
pytest tests/ -v   # 598 tests
```

598 tests covering core, server, datasets, LLM, blueprints, novel architectures, diffusion, and more.
