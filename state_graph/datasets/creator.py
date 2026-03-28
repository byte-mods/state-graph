"""Dataset Creator — build datasets for any ML task from the UI.

Supports: text, image, audio, video, multimodal, tool-calling, reasoning, etc.
Output formats: HuggingFace, JSONL, CSV, YOLO, COCO, Alpaca, ShareGPT.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

DATA_DIR = Path("./sg_datasets")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Dataset Templates ──
# Each template defines the schema, UI fields, and export format for a task type.

TEMPLATES = {
    # ── TEXT / NLP ──
    "text_classification": {
        "name": "Text Classification",
        "category": "text",
        "description": "Labeled text for sentiment, topic, intent classification",
        "models": ["BERT", "RoBERTa", "DistilBERT", "Any Transformer"],
        "fields": [
            {"name": "text", "type": "textarea", "required": True, "label": "Text"},
            {"name": "label", "type": "select", "required": True, "label": "Label"},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {"text": "This movie was amazing!", "label": "positive"},
    },
    "text_generation": {
        "name": "Instruction Tuning (Alpaca)",
        "category": "text",
        "description": "Instruction-input-output triplets for LLM fine-tuning",
        "models": ["LLaMA", "Mistral", "GPT-2", "Any Causal LM"],
        "fields": [
            {"name": "instruction", "type": "textarea", "required": True, "label": "Instruction"},
            {"name": "input", "type": "textarea", "required": False, "label": "Input (optional context)"},
            {"name": "output", "type": "textarea", "required": True, "label": "Expected Output"},
        ],
        "export_formats": ["jsonl", "alpaca", "huggingface"],
        "example": {"instruction": "Summarize the following text", "input": "Long article...", "output": "Brief summary."},
    },
    "conversation": {
        "name": "Multi-turn Chat (ShareGPT)",
        "category": "text",
        "description": "Conversational data for chat model fine-tuning",
        "models": ["ChatGPT", "LLaMA-Chat", "Mistral-Instruct"],
        "fields": [
            {"name": "conversations", "type": "chat", "required": True, "label": "Conversation Turns"},
        ],
        "export_formats": ["jsonl", "sharegpt", "huggingface"],
        "example": {"conversations": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
    },
    "tool_calling": {
        "name": "Tool/Function Calling",
        "category": "text",
        "description": "Training data for tool-use and function-calling LLMs",
        "models": ["GPT-4", "Claude", "Mistral", "Any Tool-Use LM"],
        "fields": [
            {"name": "query", "type": "textarea", "required": True, "label": "User Query"},
            {"name": "tools", "type": "json", "required": True, "label": "Available Tools (JSON array)"},
            {"name": "tool_call", "type": "json", "required": True, "label": "Expected Tool Call (JSON)"},
            {"name": "tool_result", "type": "textarea", "required": False, "label": "Tool Result"},
            {"name": "final_answer", "type": "textarea", "required": False, "label": "Final Answer"},
        ],
        "export_formats": ["jsonl", "huggingface"],
        "example": {
            "query": "What's the weather in London?",
            "tools": [{"name": "get_weather", "parameters": {"location": "string"}}],
            "tool_call": {"name": "get_weather", "arguments": {"location": "London"}},
            "tool_result": "15°C, partly cloudy",
            "final_answer": "The weather in London is 15°C and partly cloudy.",
        },
    },
    "reasoning": {
        "name": "Math & Reasoning",
        "category": "text",
        "description": "Step-by-step reasoning and math problem datasets",
        "models": ["Any LLM", "Math-specialized models"],
        "fields": [
            {"name": "problem", "type": "textarea", "required": True, "label": "Problem"},
            {"name": "solution_steps", "type": "textarea", "required": True, "label": "Step-by-step Solution"},
            {"name": "answer", "type": "text", "required": True, "label": "Final Answer"},
            {"name": "difficulty", "type": "select", "required": False, "label": "Difficulty",
             "options": ["easy", "medium", "hard"]},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {
            "problem": "If x + 3 = 7, what is x?",
            "solution_steps": "Step 1: Subtract 3 from both sides\nx + 3 - 3 = 7 - 3\nx = 4",
            "answer": "4",
            "difficulty": "easy",
        },
    },
    "qa": {
        "name": "Question Answering",
        "category": "text",
        "description": "Context-question-answer triplets for extractive/generative QA",
        "models": ["BERT", "T5", "Any Seq2Seq"],
        "fields": [
            {"name": "context", "type": "textarea", "required": True, "label": "Context"},
            {"name": "question", "type": "textarea", "required": True, "label": "Question"},
            {"name": "answer", "type": "textarea", "required": True, "label": "Answer"},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {"context": "Paris is the capital of France.", "question": "What is the capital of France?", "answer": "Paris"},
    },
    "ner": {
        "name": "Named Entity Recognition",
        "category": "text",
        "description": "Token-level entity annotations",
        "models": ["BERT", "SpaCy", "Any Token Classifier"],
        "fields": [
            {"name": "text", "type": "textarea", "required": True, "label": "Text"},
            {"name": "entities", "type": "json", "required": True, "label": "Entities (JSON array of {start, end, label})"},
        ],
        "export_formats": ["jsonl", "huggingface"],
        "example": {"text": "John lives in New York", "entities": [{"start": 0, "end": 4, "label": "PERSON"}, {"start": 14, "end": 22, "label": "LOCATION"}]},
    },
    "summarization": {
        "name": "Summarization",
        "category": "text",
        "description": "Document-summary pairs",
        "models": ["T5", "BART", "PEGASUS", "Any Seq2Seq"],
        "fields": [
            {"name": "document", "type": "textarea", "required": True, "label": "Full Document"},
            {"name": "summary", "type": "textarea", "required": True, "label": "Summary"},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {"document": "Long article text...", "summary": "Brief summary."},
    },

    # ── IMAGE / VISION ──
    "image_classification": {
        "name": "Image Classification",
        "category": "image",
        "description": "Images organized by class labels",
        "models": ["ResNet", "ViT", "EfficientNet", "Any Vision Transformer"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "label", "type": "select", "required": True, "label": "Class Label"},
        ],
        "export_formats": ["imagefolder", "csv", "huggingface"],
        "example": {"image_path": "cats/img001.jpg", "label": "cat"},
    },
    "object_detection_yolo": {
        "name": "Object Detection (YOLO)",
        "category": "image",
        "description": "Bounding box annotations in YOLO format",
        "models": ["YOLOv5", "YOLOv8", "YOLOv9", "YOLO-World"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "annotations", "type": "bbox", "required": True, "label": "Bounding Boxes",
             "format": "yolo"},
        ],
        "export_formats": ["yolo", "coco"],
        "example": {"image_path": "img001.jpg", "annotations": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.4}]},
    },
    "object_detection_coco": {
        "name": "Object Detection (COCO)",
        "category": "image",
        "description": "Bounding box annotations in COCO JSON format",
        "models": ["Faster R-CNN", "DETR", "Mask R-CNN"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "annotations", "type": "bbox", "required": True, "label": "Bounding Boxes",
             "format": "coco"},
        ],
        "export_formats": ["coco", "yolo"],
        "example": {"image_path": "img001.jpg", "annotations": [{"bbox": [100, 100, 200, 150], "category_id": 1}]},
    },
    "image_segmentation": {
        "name": "Image Segmentation",
        "category": "image",
        "description": "Pixel-level segmentation masks",
        "models": ["U-Net", "DeepLab", "Segment Anything"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "mask_path", "type": "file", "required": True, "label": "Mask Image", "accept": "image/*"},
        ],
        "export_formats": ["imagefolder", "huggingface"],
        "example": {"image_path": "images/001.png", "mask_path": "masks/001.png"},
    },
    "image_text_pairs": {
        "name": "Image-Text Pairs",
        "category": "image",
        "description": "Image-caption pairs for CLIP, diffusers, image generation",
        "models": ["Stable Diffusion", "DALL-E", "CLIP", "BLIP"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "caption", "type": "textarea", "required": True, "label": "Caption/Prompt"},
            {"name": "negative_prompt", "type": "textarea", "required": False, "label": "Negative Prompt"},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {"image_path": "photo.jpg", "caption": "A golden retriever playing in a park", "negative_prompt": "blurry, low quality"},
    },
    "dreambooth": {
        "name": "DreamBooth / LoRA Images",
        "category": "image",
        "description": "Instance images for personalized generation (DreamBooth, LoRA)",
        "models": ["Stable Diffusion", "SDXL", "Flux"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Instance Image", "accept": "image/*"},
            {"name": "prompt", "type": "textarea", "required": True, "label": "Instance Prompt (e.g. 'a photo of sks dog')"},
        ],
        "export_formats": ["imagefolder", "jsonl"],
        "example": {"image_path": "dog_01.jpg", "prompt": "a photo of sks dog"},
    },

    # ── AUDIO ──
    "audio_classification": {
        "name": "Audio Classification",
        "category": "audio",
        "description": "Audio clips with class labels",
        "models": ["Wav2Vec2", "HuBERT", "Audio Spectrogram Transformer"],
        "fields": [
            {"name": "audio_path", "type": "file", "required": True, "label": "Audio File", "accept": "audio/*"},
            {"name": "label", "type": "select", "required": True, "label": "Label"},
        ],
        "export_formats": ["audiofolder", "csv", "huggingface"],
        "example": {"audio_path": "clip_001.wav", "label": "speech"},
    },
    "speech_to_text": {
        "name": "Speech-to-Text / ASR",
        "category": "audio",
        "description": "Audio with transcription text for ASR training",
        "models": ["Whisper", "Wav2Vec2", "DeepSpeech"],
        "fields": [
            {"name": "audio_path", "type": "file", "required": True, "label": "Audio File", "accept": "audio/*"},
            {"name": "transcription", "type": "textarea", "required": True, "label": "Transcription"},
            {"name": "language", "type": "text", "required": False, "label": "Language Code (e.g. en)"},
        ],
        "export_formats": ["jsonl", "csv", "huggingface"],
        "example": {"audio_path": "audio_001.wav", "transcription": "Hello, how are you?", "language": "en"},
    },
    "text_to_speech": {
        "name": "Text-to-Speech",
        "category": "audio",
        "description": "Text with corresponding audio for TTS training",
        "models": ["Bark", "XTTS", "Tacotron"],
        "fields": [
            {"name": "text", "type": "textarea", "required": True, "label": "Text"},
            {"name": "audio_path", "type": "file", "required": True, "label": "Audio File", "accept": "audio/*"},
            {"name": "speaker_id", "type": "text", "required": False, "label": "Speaker ID"},
        ],
        "export_formats": ["jsonl", "huggingface"],
        "example": {"text": "Hello world", "audio_path": "tts_001.wav", "speaker_id": "speaker_0"},
    },

    # ── VIDEO ──
    "video_classification": {
        "name": "Video Classification",
        "category": "video",
        "description": "Video clips with class labels",
        "models": ["TimeSformer", "VideoMAE", "X3D"],
        "fields": [
            {"name": "video_path", "type": "file", "required": True, "label": "Video File", "accept": "video/*"},
            {"name": "label", "type": "select", "required": True, "label": "Label"},
        ],
        "export_formats": ["csv", "jsonl", "huggingface"],
        "example": {"video_path": "clip_001.mp4", "label": "dancing"},
    },
    "video_captioning": {
        "name": "Video Captioning",
        "category": "video",
        "description": "Videos with text descriptions for video understanding",
        "models": ["Video-LLaVA", "InternVideo", "CogVideo"],
        "fields": [
            {"name": "video_path", "type": "file", "required": True, "label": "Video File", "accept": "video/*"},
            {"name": "caption", "type": "textarea", "required": True, "label": "Caption"},
            {"name": "timestamps", "type": "json", "required": False, "label": "Timestamps (JSON, optional)"},
        ],
        "export_formats": ["jsonl", "huggingface"],
        "example": {"video_path": "video_001.mp4", "caption": "A person cooking pasta", "timestamps": [{"start": 0, "end": 5, "text": "Boiling water"}]},
    },

    # ── MULTIMODAL ──
    "visual_qa": {
        "name": "Visual Question Answering",
        "category": "multimodal",
        "description": "Image + question → answer for VQA models",
        "models": ["LLaVA", "BLIP-2", "InstructBLIP", "GPT-4V"],
        "fields": [
            {"name": "image_path", "type": "file", "required": True, "label": "Image", "accept": "image/*"},
            {"name": "question", "type": "textarea", "required": True, "label": "Question"},
            {"name": "answer", "type": "textarea", "required": True, "label": "Answer"},
        ],
        "export_formats": ["jsonl", "huggingface"],
        "example": {"image_path": "photo.jpg", "question": "How many people are in this image?", "answer": "Three people"},
    },
    "multimodal_chat": {
        "name": "Multimodal Chat",
        "category": "multimodal",
        "description": "Interleaved image+text conversations for multimodal LLMs",
        "models": ["LLaVA", "GPT-4o", "Claude", "Gemini"],
        "fields": [
            {"name": "conversations", "type": "chat_multimodal", "required": True, "label": "Conversation"},
        ],
        "export_formats": ["jsonl", "sharegpt"],
        "example": {"conversations": [
            {"role": "user", "content": [{"type": "image", "path": "img.jpg"}, {"type": "text", "text": "What is this?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "This is a cat."}]},
        ]},
    },
}


class DatasetProject:
    """A dataset project that accumulates samples and exports to various formats."""

    def __init__(self, name: str, template_id: str, labels: list[str] | None = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.template_id = template_id
        self.template = TEMPLATES[template_id]
        self.labels = labels or []
        self.samples: list[dict] = []
        self.created_at = _now()
        self.path = _ensure_dir(DATA_DIR / "projects" / self.id)

    def add_sample(self, data: dict) -> dict:
        """Add a single sample to the dataset."""
        sample = {"_id": str(uuid.uuid4())[:8], **data}
        self.samples.append(sample)
        return {"status": "added", "sample_id": sample["_id"], "total": len(self.samples)}

    def add_samples_bulk(self, samples: list[dict]) -> dict:
        """Add multiple samples at once."""
        for s in samples:
            s["_id"] = str(uuid.uuid4())[:8]
        self.samples.extend(samples)
        return {"status": "added", "count": len(samples), "total": len(self.samples)}

    def remove_sample(self, sample_id: str) -> dict:
        self.samples = [s for s in self.samples if s.get("_id") != sample_id]
        return {"status": "removed", "total": len(self.samples)}

    def update_sample(self, sample_id: str, data: dict) -> dict:
        for i, s in enumerate(self.samples):
            if s.get("_id") == sample_id:
                self.samples[i] = {"_id": sample_id, **data}
                return {"status": "updated"}
        return {"status": "error", "message": "Sample not found"}

    def get_samples(self, offset: int = 0, limit: int = 50) -> dict:
        return {
            "samples": self.samples[offset:offset + limit],
            "total": len(self.samples),
            "offset": offset,
            "limit": limit,
        }

    def get_stats(self) -> dict:
        label_counts = {}
        for s in self.samples:
            lbl = s.get("label", s.get("difficulty", "unknown"))
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        return {
            "id": self.id,
            "name": self.name,
            "template": self.template_id,
            "category": self.template["category"],
            "total_samples": len(self.samples),
            "labels": self.labels,
            "label_distribution": label_counts,
        }

    # ── Export Methods ──

    def export(self, format: str) -> dict:
        """Export the dataset in the specified format."""
        exporters = {
            "jsonl": self._export_jsonl,
            "csv": self._export_csv,
            "json": self._export_json,
            "alpaca": self._export_alpaca,
            "sharegpt": self._export_sharegpt,
            "yolo": self._export_yolo,
            "coco": self._export_coco,
            "imagefolder": self._export_imagefolder,
            "huggingface": self._export_huggingface,
        }
        if format not in exporters:
            return {"status": "error", "message": f"Unknown format: {format}"}
        return exporters[format]()

    def _export_jsonl(self) -> dict:
        out_path = self.path / f"{self.name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in self.samples:
                row = {k: v for k, v in s.items() if not k.startswith("_")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"status": "exported", "format": "jsonl", "path": str(out_path), "count": len(self.samples)}

    def _export_csv(self) -> dict:
        import csv as csv_mod
        out_path = self.path / f"{self.name}.csv"
        if not self.samples:
            return {"status": "error", "message": "No samples"}

        fields = [k for k in self.samples[0].keys() if not k.startswith("_")]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for s in self.samples:
                row = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in s.items() if not k.startswith("_")}
                writer.writerow(row)
        return {"status": "exported", "format": "csv", "path": str(out_path), "count": len(self.samples)}

    def _export_json(self) -> dict:
        out_path = self.path / f"{self.name}.json"
        data = [{k: v for k, v in s.items() if not k.startswith("_")} for s in self.samples]
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "exported", "format": "json", "path": str(out_path), "count": len(self.samples)}

    def _export_alpaca(self) -> dict:
        """Export in Alpaca format for LLM fine-tuning."""
        out_path = self.path / f"{self.name}_alpaca.json"
        data = []
        for s in self.samples:
            data.append({
                "instruction": s.get("instruction", ""),
                "input": s.get("input", ""),
                "output": s.get("output", ""),
            })
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "exported", "format": "alpaca", "path": str(out_path), "count": len(self.samples)}

    def _export_sharegpt(self) -> dict:
        """Export in ShareGPT format for chat model fine-tuning."""
        out_path = self.path / f"{self.name}_sharegpt.json"
        data = []
        for s in self.samples:
            convs = s.get("conversations", [])
            data.append({"conversations": convs})
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "exported", "format": "sharegpt", "path": str(out_path), "count": len(self.samples)}

    def _export_yolo(self) -> dict:
        """Export in YOLO format (images/ + labels/ directories)."""
        img_dir = _ensure_dir(self.path / "yolo" / "images")
        lbl_dir = _ensure_dir(self.path / "yolo" / "labels")

        for s in self.samples:
            img_path = s.get("image_path", "")
            if img_path and os.path.exists(img_path):
                shutil.copy2(img_path, img_dir / Path(img_path).name)

            # Write label file
            annotations = s.get("annotations", [])
            lbl_name = Path(img_path).stem + ".txt" if img_path else f"{s.get('_id', 'unknown')}.txt"
            with open(lbl_dir / lbl_name, "w") as f:
                for ann in annotations:
                    if "class_id" in ann:
                        f.write(f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}\n")

        # Write classes file
        classes_path = self.path / "yolo" / "classes.txt"
        classes_path.write_text("\n".join(self.labels))

        return {"status": "exported", "format": "yolo", "path": str(self.path / "yolo"), "count": len(self.samples)}

    def _export_coco(self) -> dict:
        """Export in COCO JSON format."""
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(self.labels)],
        }
        ann_id = 0
        for img_id, s in enumerate(self.samples):
            coco["images"].append({
                "id": img_id,
                "file_name": s.get("image_path", f"img_{img_id}"),
                "width": s.get("width", 0),
                "height": s.get("height", 0),
            })
            for ann in s.get("annotations", []):
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann.get("category_id", 0),
                    "bbox": ann.get("bbox", [0, 0, 0, 0]),
                    "area": ann.get("area", 0),
                    "iscrowd": 0,
                })
                ann_id += 1

        out_path = self.path / f"{self.name}_coco.json"
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        return {"status": "exported", "format": "coco", "path": str(out_path), "count": len(self.samples)}

    def _export_imagefolder(self) -> dict:
        """Export as imagefolder structure (class_name/image.jpg)."""
        base = _ensure_dir(self.path / "imagefolder")
        for s in self.samples:
            label = s.get("label", "unknown")
            label_dir = _ensure_dir(base / label)
            img_path = s.get("image_path", "")
            if img_path and os.path.exists(img_path):
                shutil.copy2(img_path, label_dir / Path(img_path).name)
        return {"status": "exported", "format": "imagefolder", "path": str(base), "count": len(self.samples)}

    def _export_huggingface(self) -> dict:
        """Export as HuggingFace Dataset (arrow format)."""
        try:
            from datasets import Dataset
            data = {k: [] for k in self.samples[0].keys() if not k.startswith("_")} if self.samples else {}
            for s in self.samples:
                for k in data:
                    val = s.get(k, None)
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val)
                    data[k].append(val)
            ds = Dataset.from_dict(data)
            out_path = self.path / "hf_dataset"
            ds.save_to_disk(str(out_path))
            return {"status": "exported", "format": "huggingface", "path": str(out_path), "count": len(self.samples)}
        except ImportError:
            return {"status": "error", "message": "Install datasets: pip install datasets"}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "template_id": self.template_id,
            "template_name": self.template["name"],
            "category": self.template["category"],
            "labels": self.labels,
            "total_samples": len(self.samples),
            "created_at": self.created_at,
            "export_formats": self.template["export_formats"],
        }


class DatasetManager:
    """Manages multiple dataset projects."""

    def __init__(self):
        self.projects: dict[str, DatasetProject] = {}

    def create_project(self, name: str, template_id: str, labels: list[str] | None = None) -> dict:
        if template_id not in TEMPLATES:
            return {"status": "error", "message": f"Unknown template: {template_id}"}
        project = DatasetProject(name, template_id, labels)
        self.projects[project.id] = project
        return {"status": "created", "project": project.to_dict()}

    def get_project(self, project_id: str) -> DatasetProject | None:
        return self.projects.get(project_id)

    def list_projects(self) -> list[dict]:
        return [p.to_dict() for p in self.projects.values()]

    def delete_project(self, project_id: str) -> dict:
        if project_id in self.projects:
            p = self.projects.pop(project_id)
            if p.path.exists():
                shutil.rmtree(p.path, ignore_errors=True)
            return {"status": "deleted"}
        return {"status": "error", "message": "Project not found"}

    @staticmethod
    def list_templates() -> dict:
        result = {}
        for tid, t in TEMPLATES.items():
            result[tid] = {
                "name": t["name"],
                "category": t["category"],
                "description": t["description"],
                "models": t["models"],
                "fields": t["fields"],
                "export_formats": t["export_formats"],
                "example": t.get("example"),
            }
        return result

    @staticmethod
    def list_templates_by_category() -> dict:
        categories: dict[str, list] = {}
        for tid, t in TEMPLATES.items():
            cat = t["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "id": tid,
                "name": t["name"],
                "description": t["description"],
                "models": t["models"],
            })
        return categories


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
