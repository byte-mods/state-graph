"""Model deployment — serve as API, export to ONNX/TorchScript/CoreML, Docker generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def export_onnx(model, dummy_input, path: str = "./model.onnx", opset: int = 14) -> dict:
    """Export PyTorch model to ONNX."""
    import torch
    try:
        torch.onnx.export(
            model, dummy_input, path,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        size = os.path.getsize(path)
        return {"status": "exported", "format": "onnx", "path": path, "size_mb": round(size / 1e6, 2)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def export_torchscript(model, dummy_input, path: str = "./model.pt", method: str = "trace") -> dict:
    """Export to TorchScript (trace or script)."""
    import torch
    try:
        if method == "script":
            scripted = torch.jit.script(model)
        else:
            scripted = torch.jit.trace(model, dummy_input)
        scripted.save(path)
        size = os.path.getsize(path)
        return {"status": "exported", "format": "torchscript", "path": path, "size_mb": round(size / 1e6, 2)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_inference_server(model_path: str, model_type: str = "pytorch", port: int = 8080) -> str:
    """Generate a standalone FastAPI inference server script."""
    if model_type == "onnx":
        return f'''"""Auto-generated inference server for ONNX model."""
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Model Server")
session = ort.InferenceSession("{model_path}")

class PredictRequest(BaseModel):
    input: list

class PredictResponse(BaseModel):
    output: list

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    inp = np.array(req.input, dtype=np.float32)
    if inp.ndim == 1:
        inp = inp.reshape(1, -1)
    result = session.run(None, {{"input": inp}})
    return PredictResponse(output=result[0].tolist())

@app.get("/health")
async def health():
    return {{"status": "ok"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''
    else:
        return f'''"""Auto-generated inference server for PyTorch model."""
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Model Server")

# Load model
model = torch.jit.load("{model_path}")
model.eval()

class PredictRequest(BaseModel):
    input: list

class PredictResponse(BaseModel):
    output: list

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    inp = torch.tensor(req.input, dtype=torch.float32)
    if inp.ndim == 1:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        result = model(inp)
    return PredictResponse(output=result.tolist())

@app.get("/health")
async def health():
    return {{"status": "ok"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''


def generate_dockerfile(model_path: str, model_type: str = "pytorch", port: int = 8080) -> str:
    """Generate Dockerfile for model deployment."""
    if model_type == "onnx":
        pip_extra = "onnxruntime"
    else:
        pip_extra = "torch"

    return f'''FROM python:3.11-slim

WORKDIR /app
COPY {model_path} /app/model
COPY server.py /app/server.py

RUN pip install --no-cache-dir fastapi uvicorn {pip_extra} numpy

EXPOSE {port}
CMD ["python", "server.py"]
'''


def generate_gradio_app(model_path: str, model_type: str = "pytorch", task: str = "classification") -> str:
    """Generate a Gradio demo app."""
    return f'''"""Auto-generated Gradio demo."""
import gradio as gr
import torch
import numpy as np

model = torch.jit.load("{model_path}")
model.eval()

def predict(input_text):
    # Adapt this to your model's input format
    # For classification: tokenize input_text, run model, return label
    return "Prediction placeholder — adapt predict() to your model"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Label(label="Prediction"),
    title="Model Demo",
    description="Auto-generated from StateGraph",
)

if __name__ == "__main__":
    demo.launch(share=True)
'''
