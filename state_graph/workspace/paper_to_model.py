"""Paper-to-Model — paste a paper URL, AI reads it, builds the architecture.

Flow: URL → fetch paper text → send to LLM → extract architecture → build in StateGraph graph → ready to train.
Supports: arXiv, OpenReview, Semantic Scholar, any PDF URL.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from typing import Any


SYSTEM_PROMPT = """You are an expert ML researcher. Given a research paper, extract the neural network architecture and convert it to a StateGraph configuration.

You MUST return a JSON object with this exact structure:
{
  "paper_title": "string",
  "paper_summary": "1-2 sentence summary of the paper's contribution",
  "architecture": {
    "nodes": [
      {"layer_type": "Linear|Conv2d|TransformerBlock|...", "params": {"in_features": N, ...}, "activation": "ReLU|GELU|null", "position": 0},
      ...
    ]
  },
  "training_config": {
    "epochs": N,
    "batch_size": N,
    "learning_rate": float,
    "optimizer": "Adam|AdamW|SGD",
    "loss": "CrossEntropyLoss|MSELoss|...",
    "scheduler": "CosineAnnealingLR|null",
    "scheduler_params": {}
  },
  "dataset_suggestion": "spiral|xor|mnist|cifar10|custom",
  "key_innovations": ["innovation 1", "innovation 2"],
  "custom_formulas": [
    {"name": "FormulaName", "expression": "torch expression using x"}
  ],
  "notes": "Any implementation notes"
}

Available layer types: Linear, Conv1d, Conv2d, Conv3d, BatchNorm1d, BatchNorm2d, LayerNorm, Dropout, Dropout2d, LSTM, GRU, MultiheadAttention, Flatten, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Embedding, ConvTranspose2d, ResidualBlock, GatedLinearUnit, SwishLinear, TransformerBlock, PositionalEncoding, TokenEmbedding, SequencePool, SqueezeExcite

Available activations: ReLU, LeakyReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, ELU, PReLU, Mish
Custom formulas can define new activations using torch operations on x.

For TransformerBlock: params are {d_model, n_heads, ffn_dim, dropout}
For TokenEmbedding: params are {in_features, d_model, seq_len}
For PositionalEncoding: params are {d_model, max_len, dropout}
For SequencePool: params are {d_model, mode: "mean"|"cls"|"max"}

IMPORTANT: Use realistic dimensions. Scale down for demonstration (e.g., d_model=64 or 128 instead of 768).
Return ONLY the JSON, no markdown fences, no explanation outside the JSON."""


def fetch_paper_text(url: str) -> dict:
    """Fetch paper content from URL. Supports arXiv, OpenReview, generic PDFs."""

    # arXiv — use abstract API
    arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
    if arxiv_match:
        paper_id = arxiv_match.group(1)
        return _fetch_arxiv(paper_id)

    # Semantic Scholar
    s2_match = re.search(r"semanticscholar\.org/paper/([a-f0-9]+)", url)
    if s2_match:
        return _fetch_semantic_scholar(s2_match.group(1))

    # OpenReview
    if "openreview.net" in url:
        return _fetch_openreview(url)

    # Generic — try to fetch HTML and extract text
    return _fetch_generic(url)


def _fetch_arxiv(paper_id: str) -> dict:
    """Fetch from arXiv API."""
    api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "StateGraph/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read().decode()

    # Parse XML — skip the feed title, get the entry title
    titles = re.findall(r"<title[^>]*>(.*?)</title>", data, re.DOTALL)
    title = None
    for t in titles:
        if "arXiv" not in t and "Query" not in t and len(t.strip()) > 5:
            title = type("", (), {"group": lambda self, n=1: t.strip()})()
            break
    summary = re.search(r"<summary>(.*?)</summary>", data, re.DOTALL)
    authors = re.findall(r"<name>(.*?)</name>", data)

    title_text = title.group(1).strip() if title else "Unknown"
    summary_text = summary.group(1).strip() if summary else ""

    # Also try to get the full paper HTML from ar5iv
    full_text = summary_text
    try:
        html_url = f"https://ar5iv.labs.arxiv.org/html/{paper_id}"
        req2 = urllib.request.Request(html_url, headers={"User-Agent": "StateGraph/1.0"})
        with urllib.request.urlopen(req2, timeout=20) as resp2:
            html = resp2.read().decode("utf-8", errors="replace")
        # Extract text from HTML (strip tags)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Limit to relevant sections
        if len(text) > 15000:
            text = text[:15000]
        full_text = text
    except Exception as e:
        import logging
        logging.getLogger("state_graph").debug(
            "Could not fetch full paper HTML from ar5iv for %s: %s", paper_id, e
        )

    return {
        "status": "ok",
        "source": "arxiv",
        "paper_id": paper_id,
        "title": title_text,
        "authors": authors[:5],
        "abstract": summary_text,
        "full_text": full_text,
        "url": f"https://arxiv.org/abs/{paper_id}",
    }


def _fetch_semantic_scholar(paper_id: str) -> dict:
    """Fetch from Semantic Scholar API."""
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title,abstract,authors,tldr"
    req = urllib.request.Request(api_url, headers={"User-Agent": "StateGraph/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    return {
        "status": "ok",
        "source": "semantic_scholar",
        "title": data.get("title", ""),
        "abstract": data.get("abstract", ""),
        "authors": [a.get("name", "") for a in data.get("authors", [])[:5]],
        "full_text": data.get("abstract", "") + "\n" + (data.get("tldr", {}) or {}).get("text", ""),
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
    }


def _fetch_openreview(url: str) -> dict:
    """Fetch from OpenReview."""
    # Extract forum ID
    forum_match = re.search(r"id=([A-Za-z0-9_-]+)", url)
    if not forum_match:
        return {"status": "error", "message": "Cannot parse OpenReview URL"}

    forum_id = forum_match.group(1)
    api_url = f"https://api.openreview.net/notes?id={forum_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "StateGraph/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    notes = data.get("notes", [])
    if not notes:
        return {"status": "error", "message": "Paper not found"}

    note = notes[0]
    content = note.get("content", {})

    return {
        "status": "ok",
        "source": "openreview",
        "title": content.get("title", {}).get("value", ""),
        "abstract": content.get("abstract", {}).get("value", ""),
        "full_text": content.get("abstract", {}).get("value", ""),
        "url": url,
    }


def _fetch_generic(url: str) -> dict:
    """Fetch any URL and extract text."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StateGraph/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

        if "pdf" in content_type:
            return {"status": "ok", "source": "pdf", "full_text": "(PDF — text extraction requires pdfplumber. Paste abstract manually.)", "url": url}

        html = data.decode("utf-8", errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        title = re.search(r"<title>(.*?)</title>", html)

        return {
            "status": "ok",
            "source": "web",
            "title": title.group(1) if title else "",
            "full_text": text[:15000],
            "url": url,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def paper_to_architecture(paper_text: str, ai_assistant) -> dict:
    """Send paper text to AI and get back architecture config."""
    if not ai_assistant._configured:
        return {"status": "error", "message": "AI assistant not configured. Set API key first."}

    message = f"""Read this research paper and extract the neural network architecture. Return ONLY a JSON object following the schema I specified.

Paper content:
{paper_text[:12000]}

Extract:
1. The main model architecture (layers, dimensions, activations)
2. Training configuration (optimizer, LR, scheduler, loss)
3. Any custom activation functions or novel components
4. Dataset recommendations

Return the JSON now:"""

    result = ai_assistant.chat(message)
    if result["status"] != "ok":
        return result

    # Parse JSON from response
    response = result["response"]
    try:
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            config = json.loads(json_match.group())
            return {"status": "ok", "config": config, "raw_response": response}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"AI returned invalid JSON: {e}", "raw_response": response}

    return {"status": "error", "message": "AI did not return valid JSON", "raw_response": response}


def apply_paper_config(engine, config: dict) -> dict:
    """Apply the AI-extracted architecture to the StateGraph engine."""
    from state_graph.core.registry import Registry

    # Clear existing graph
    for nid in list(engine.graph.nodes.keys()):
        engine.graph.remove_layer(nid)

    # Register custom formulas
    registered_formulas = []
    failed_formulas = []
    for formula in config.get("custom_formulas", []):
        try:
            Registry.register_formula_from_string(formula["name"], formula["expression"])
            registered_formulas.append(formula["name"])
        except Exception as e:
            failed_formulas.append({"name": formula.get("name", "?"), "error": str(e)})

    # Add layers
    arch = config.get("architecture", {})
    nodes = arch.get("nodes", [])
    added_layers = []
    failed_layers = []
    for node in nodes:
        try:
            nid = engine.graph.add_layer(
                layer_type=node["layer_type"],
                params=node.get("params", {}),
                activation=node.get("activation"),
                position=node.get("position"),
            )
            added_layers.append(nid)
        except Exception as e:
            failed_layers.append({"layer_type": node.get("layer_type", "?"), "error": str(e)})

    # Apply training config
    tc = config.get("training_config", {})
    if tc:
        engine.config.update({k: v for k, v in tc.items() if v is not None})

    result = {
        "status": "applied",
        "paper_title": config.get("paper_title", ""),
        "paper_summary": config.get("paper_summary", ""),
        "layers_added": len(added_layers),
        "custom_formulas": registered_formulas,
        "key_innovations": config.get("key_innovations", []),
        "dataset_suggestion": config.get("dataset_suggestion", ""),
        "notes": config.get("notes", ""),
        "training_config": tc,
        "graph": engine.graph.to_dict(),
    }
    if failed_formulas:
        result["failed_formulas"] = failed_formulas
    if failed_layers:
        result["failed_layers"] = failed_layers
        result["warnings"] = f"{len(failed_layers)} layer(s) could not be added"
    return result
