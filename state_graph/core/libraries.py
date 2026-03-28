"""ML Library Registry — quick access to all major frameworks from the UI.

Each library has: templates, model loaders, and code generators.
Researcher picks a library + task → gets runnable code in the IDE.
"""

from __future__ import annotations

LIBRARY_CATALOG = {
    # ── Computer Vision ──
    "torchvision": {
        "name": "TorchVision",
        "category": "vision",
        "description": "PyTorch official vision models and transforms",
        "pip": "torchvision",
        "models": ["ResNet", "EfficientNet", "ViT", "Swin", "ConvNeXt", "RegNet", "MobileNet", "DenseNet"],
        "tasks": ["image_classification", "object_detection", "segmentation"],
    },
    "ultralytics": {
        "name": "Ultralytics (YOLO)",
        "category": "vision",
        "description": "YOLOv8/v9/v11 for detection, segmentation, pose, classification",
        "pip": "ultralytics",
        "models": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x", "YOLOv9", "YOLO11"],
        "tasks": ["object_detection", "instance_segmentation", "pose_estimation", "classification"],
    },
    "mmdetection": {
        "name": "MMDetection",
        "category": "vision",
        "description": "OpenMMLab detection toolbox (Faster R-CNN, DETR, Mask R-CNN, etc.)",
        "pip": "mmdet",
        "models": ["Faster R-CNN", "DETR", "Mask R-CNN", "FCOS", "RetinaNet", "Cascade R-CNN"],
        "tasks": ["object_detection", "instance_segmentation"],
    },
    "segment_anything": {
        "name": "Segment Anything (SAM)",
        "category": "vision",
        "description": "Meta's zero-shot image segmentation",
        "pip": "segment-anything",
        "models": ["SAM-ViT-H", "SAM-ViT-L", "SAM-ViT-B", "SAM-2"],
        "tasks": ["segmentation", "interactive_segmentation"],
    },
    "detectron2": {
        "name": "Detectron2",
        "category": "vision",
        "description": "Meta's detection/segmentation platform",
        "pip": "detectron2",
        "models": ["Faster R-CNN", "Mask R-CNN", "Panoptic FPN", "PointRend"],
        "tasks": ["object_detection", "segmentation", "keypoint_detection"],
    },
    "open_clip": {
        "name": "OpenCLIP",
        "category": "vision",
        "description": "Open-source CLIP for vision-language tasks",
        "pip": "open_clip_torch",
        "models": ["ViT-B/32", "ViT-L/14", "ViT-H/14", "ViT-G/14"],
        "tasks": ["zero_shot_classification", "image_retrieval", "embedding"],
    },

    # ── NLP / LLM ──
    "transformers": {
        "name": "HuggingFace Transformers",
        "category": "nlp",
        "description": "State-of-the-art NLP models",
        "pip": "transformers",
        "models": ["BERT", "GPT-2", "T5", "LLaMA", "Mistral", "Phi", "Gemma", "Qwen"],
        "tasks": ["text_classification", "ner", "qa", "summarization", "translation", "text_generation"],
    },
    "vllm": {
        "name": "vLLM",
        "category": "nlp",
        "description": "High-throughput LLM serving with PagedAttention",
        "pip": "vllm",
        "models": ["Any HF model"],
        "tasks": ["inference_serving", "batch_inference"],
    },
    "llamacpp": {
        "name": "llama.cpp (Python)",
        "category": "nlp",
        "description": "Run GGUF models locally on CPU/GPU",
        "pip": "llama-cpp-python",
        "models": ["Any GGUF model"],
        "tasks": ["local_inference", "chat"],
    },
    "langchain": {
        "name": "LangChain",
        "category": "nlp",
        "description": "LLM application framework — chains, agents, RAG",
        "pip": "langchain",
        "models": ["Any LLM"],
        "tasks": ["rag", "agents", "chains", "tool_use"],
    },
    "sentence_transformers": {
        "name": "Sentence Transformers",
        "category": "nlp",
        "description": "Text embeddings for similarity, search, clustering",
        "pip": "sentence-transformers",
        "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "e5-large-v2"],
        "tasks": ["embedding", "similarity", "clustering", "search"],
    },
    "spacy": {
        "name": "spaCy",
        "category": "nlp",
        "description": "Industrial NLP — NER, POS, dependency parsing",
        "pip": "spacy",
        "models": ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"],
        "tasks": ["ner", "pos_tagging", "dependency_parsing", "text_classification"],
    },

    # ── Audio ──
    "whisper": {
        "name": "OpenAI Whisper",
        "category": "audio",
        "description": "Speech recognition / ASR",
        "pip": "openai-whisper",
        "models": ["tiny", "base", "small", "medium", "large-v3"],
        "tasks": ["speech_to_text", "translation"],
    },
    "torchaudio": {
        "name": "TorchAudio",
        "category": "audio",
        "description": "Audio processing and models",
        "pip": "torchaudio",
        "models": ["Wav2Vec2", "HuBERT", "WavLM"],
        "tasks": ["audio_classification", "speech_recognition", "speaker_verification"],
    },
    "bark": {
        "name": "Bark",
        "category": "audio",
        "description": "Text-to-speech generation",
        "pip": "git+https://github.com/suno-ai/bark.git",
        "models": ["bark-small", "bark"],
        "tasks": ["text_to_speech"],
    },

    # ── Generative / Diffusion ──
    "diffusers": {
        "name": "HuggingFace Diffusers",
        "category": "generative",
        "description": "Stable Diffusion, SDXL, Flux, PixArt, Kandinsky",
        "pip": "diffusers",
        "models": ["SD 1.5", "SDXL", "SD 3", "Flux", "PixArt", "Kandinsky"],
        "tasks": ["text_to_image", "image_to_image", "inpainting", "controlnet"],
    },
    "comfyui": {
        "name": "ComfyUI",
        "category": "generative",
        "description": "Node-based UI for diffusion workflows",
        "pip": "comfyui",
        "models": ["Any diffusion model"],
        "tasks": ["text_to_image", "workflow_builder"],
    },

    # ── Tabular / Classical ML ──
    "sklearn": {
        "name": "scikit-learn",
        "category": "tabular",
        "description": "Classical ML — classification, regression, clustering",
        "pip": "scikit-learn",
        "models": ["RandomForest", "XGBoost", "SVM", "KMeans", "PCA", "LogisticRegression"],
        "tasks": ["classification", "regression", "clustering", "dimensionality_reduction"],
    },
    "xgboost": {
        "name": "XGBoost",
        "category": "tabular",
        "description": "Gradient boosting for tabular data",
        "pip": "xgboost",
        "models": ["XGBClassifier", "XGBRegressor", "XGBRanker"],
        "tasks": ["classification", "regression", "ranking"],
    },
    "lightgbm": {
        "name": "LightGBM",
        "category": "tabular",
        "description": "Fast gradient boosting by Microsoft",
        "pip": "lightgbm",
        "models": ["LGBMClassifier", "LGBMRegressor"],
        "tasks": ["classification", "regression"],
    },
    "catboost": {
        "name": "CatBoost",
        "category": "tabular",
        "description": "Gradient boosting with categorical feature support",
        "pip": "catboost",
        "models": ["CatBoostClassifier", "CatBoostRegressor"],
        "tasks": ["classification", "regression"],
    },

    # ── Experiment Tracking ──
    "wandb": {
        "name": "Weights & Biases",
        "category": "tracking",
        "description": "Experiment tracking, visualization, model registry",
        "pip": "wandb",
        "models": [],
        "tasks": ["experiment_tracking", "hyperparameter_search", "model_registry"],
    },
    "mlflow": {
        "name": "MLflow",
        "category": "tracking",
        "description": "Open-source ML lifecycle management",
        "pip": "mlflow",
        "models": [],
        "tasks": ["experiment_tracking", "model_registry", "deployment"],
    },

    # ── Optimization / Training ──
    "optuna": {
        "name": "Optuna",
        "category": "optimization",
        "description": "Bayesian hyperparameter optimization",
        "pip": "optuna",
        "models": [],
        "tasks": ["hyperparameter_search", "pruning"],
    },
    "ray_tune": {
        "name": "Ray Tune",
        "category": "optimization",
        "description": "Distributed hyperparameter tuning",
        "pip": "ray[tune]",
        "models": [],
        "tasks": ["hyperparameter_search", "distributed_training"],
    },
    "accelerate": {
        "name": "HuggingFace Accelerate",
        "category": "optimization",
        "description": "Multi-GPU, mixed precision, distributed training",
        "pip": "accelerate",
        "models": [],
        "tasks": ["distributed_training", "mixed_precision"],
    },
    "deepspeed": {
        "name": "DeepSpeed",
        "category": "optimization",
        "description": "Microsoft's distributed training optimization (ZeRO, offloading)",
        "pip": "deepspeed",
        "models": [],
        "tasks": ["distributed_training", "model_parallelism", "zero_optimization"],
    },

    # ── Multimodal ──
    "llava": {
        "name": "LLaVA",
        "category": "multimodal",
        "description": "Visual instruction tuning — image + text understanding",
        "pip": "llava",
        "models": ["LLaVA-1.5-7B", "LLaVA-1.5-13B"],
        "tasks": ["visual_qa", "image_captioning", "multimodal_chat"],
    },

    # ── Graph Neural Networks ──
    "pyg": {
        "name": "PyTorch Geometric",
        "category": "graph",
        "description": "GNN library — GCN, GAT, GraphSAGE, etc.",
        "pip": "torch_geometric",
        "models": ["GCN", "GAT", "GraphSAGE", "GIN", "PNA"],
        "tasks": ["node_classification", "graph_classification", "link_prediction"],
    },

    # ── Time Series ──
    "pytorch_forecasting": {
        "name": "PyTorch Forecasting",
        "category": "timeseries",
        "description": "Time series forecasting with deep learning",
        "pip": "pytorch-forecasting",
        "models": ["TFT", "N-BEATS", "DeepAR", "NHiTS"],
        "tasks": ["forecasting", "anomaly_detection"],
    },
}


def get_code_template(library: str, task: str) -> str:
    """Generate ready-to-run code for a library + task combination."""
    templates = {
        ("ultralytics", "object_detection"): '''from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # nano model, fast

# Train on your dataset
results = model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
)

# Evaluate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")

# Predict
results = model.predict("path/to/image.jpg", save=True)
''',
        ("sklearn", "classification"): '''from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# Load your data
# X, y = ...

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
''',
        ("optuna", "hyperparameter_search"): '''import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 512, step=32)

    # Build and train your model with these hyperparameters
    # ...
    # Return the metric to optimize
    accuracy = 0.0  # Replace with actual training
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.params}")
print(f"Best accuracy: {study.best_value:.4f}")
''',
        ("whisper", "speech_to_text"): '''import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large-v3

# Transcribe
result = model.transcribe("audio.mp3")
print(result["text"])

# With language detection
result = model.transcribe("audio.mp3", task="translate")  # translate to English
''',
        ("diffusers", "text_to_image"): '''from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A photo of a robot in a futuristic lab",
    negative_prompt="blurry, low quality",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

image.save("output.png")
''',
        ("sentence_transformers", "embedding"): '''from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["This is a test", "Another sentence"]
embeddings = model.encode(sentences)

# Similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity: {sim[0][0]:.4f}")
''',
        ("vllm", "inference_serving"): '''from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

prompts = ["Explain quantum computing in simple terms."]
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(prompts, params)
for output in outputs:
    print(output.outputs[0].text)
''',
    }

    key = (library, task)
    if key in templates:
        return templates[key]

    # Generic template
    lib = LIBRARY_CATALOG.get(library, {})
    return f'# {lib.get("name", library)} — {task}\n# pip install {lib.get("pip", library)}\n\n# Add your code here\n'
