"""Dataset sources — pull from Kaggle, URLs, local files, HuggingFace."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any


DATA_DIR = Path("./sg_datasets")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class KaggleSource:
    """Pull datasets from Kaggle."""

    @staticmethod
    def search(query: str, limit: int = 20) -> list[dict]:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        results = api.dataset_list(search=query, page_size=limit)
        return [
            {
                "id": str(ds.ref),
                "title": ds.title,
                "size": str(ds.size),
                "downloads": ds.downloadCount,
                "votes": ds.voteCount,
                "source": "kaggle",
            }
            for ds in results
        ]

    @staticmethod
    def download(dataset_id: str, dest: str | None = None) -> dict:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        dest_path = Path(dest) if dest else _ensure_dir(DATA_DIR / "kaggle" / dataset_id.replace("/", "_"))
        _ensure_dir(dest_path)

        api.dataset_download_files(dataset_id, path=str(dest_path), unzip=True)

        files = list(dest_path.rglob("*"))
        file_list = [
            {"name": f.name, "size": f.stat().st_size, "type": f.suffix}
            for f in files if f.is_file()
        ]

        return {
            "status": "downloaded",
            "dataset_id": dataset_id,
            "path": str(dest_path),
            "files": file_list,
            "total_files": len(file_list),
        }

    @staticmethod
    def search_competitions(query: str, limit: int = 10) -> list[dict]:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        results = api.competitions_list(search=query, page_size=limit)
        return [
            {
                "id": str(c.ref),
                "title": c.title,
                "category": c.category,
                "source": "kaggle_competition",
            }
            for c in results
        ]


class URLSource:
    """Download datasets from direct URLs."""

    @staticmethod
    def download(url: str, dest_name: str | None = None) -> dict:
        import urllib.request

        dest_dir = _ensure_dir(DATA_DIR / "downloads")
        fname = dest_name or url.split("/")[-1].split("?")[0] or "download"
        dest_path = dest_dir / fname

        urllib.request.urlretrieve(url, str(dest_path))

        # Auto-extract zip files
        extracted_path = str(dest_path)
        if dest_path.suffix == ".zip":
            extract_dir = dest_dir / dest_path.stem
            _ensure_dir(extract_dir)
            with zipfile.ZipFile(dest_path, "r") as zf:
                zf.extractall(extract_dir)
            extracted_path = str(extract_dir)

        return {
            "status": "downloaded",
            "url": url,
            "path": extracted_path,
            "size": os.path.getsize(str(dest_path)),
        }


class LocalSource:
    """Scan and load local files/directories."""

    @staticmethod
    def scan_directory(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"status": "error", "message": f"Path not found: {path}"}

        if p.is_file():
            return {
                "status": "ok",
                "path": str(p),
                "type": "file",
                "name": p.name,
                "size": p.stat().st_size,
                "extension": p.suffix,
            }

        # Directory scan
        files_by_ext: dict[str, int] = {}
        total_size = 0
        file_count = 0
        subdirs = []

        for item in p.iterdir():
            if item.is_dir():
                subdirs.append(item.name)
            elif item.is_file():
                ext = item.suffix.lower()
                files_by_ext[ext] = files_by_ext.get(ext, 0) + 1
                total_size += item.stat().st_size
                file_count += 1

        # Auto-detect dataset type
        detected_type = _detect_dataset_type(files_by_ext, subdirs)

        return {
            "status": "ok",
            "path": str(p),
            "type": "directory",
            "file_count": file_count,
            "subdirs": subdirs,
            "files_by_extension": files_by_ext,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "detected_type": detected_type,
        }

    @staticmethod
    def load_csv(path: str, preview_rows: int = 5) -> dict:
        p = Path(path)
        with open(p, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            rows = []
            total = 0
            for row in reader:
                total += 1
                if len(rows) < preview_rows:
                    rows.append(row)

        return {
            "status": "ok",
            "path": str(p),
            "format": "csv",
            "columns": columns,
            "n_rows": total,
            "preview": rows,
        }

    @staticmethod
    def load_json(path: str, preview_rows: int = 5) -> dict:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))

        if isinstance(data, list):
            columns = list(data[0].keys()) if data else []
            return {
                "status": "ok",
                "path": str(p),
                "format": "json_array",
                "columns": columns,
                "n_rows": len(data),
                "preview": data[:preview_rows],
            }
        elif isinstance(data, dict):
            return {
                "status": "ok",
                "path": str(p),
                "format": "json_object",
                "keys": list(data.keys()),
                "preview": {k: str(v)[:200] for k, v in list(data.items())[:5]},
            }

        return {"status": "ok", "format": "json", "path": str(p)}

    @staticmethod
    def load_jsonl(path: str, preview_rows: int = 5) -> dict:
        p = Path(path)
        rows = []
        total = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                if len(rows) < preview_rows:
                    rows.append(json.loads(line))

        columns = list(rows[0].keys()) if rows else []
        return {
            "status": "ok",
            "path": str(p),
            "format": "jsonl",
            "columns": columns,
            "n_rows": total,
            "preview": rows,
        }


def _detect_dataset_type(files_by_ext: dict, subdirs: list) -> str:
    """Auto-detect what kind of dataset this directory contains."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    text_exts = {".txt", ".csv", ".json", ".jsonl", ".tsv", ".parquet"}

    img_count = sum(v for k, v in files_by_ext.items() if k in img_exts)
    audio_count = sum(v for k, v in files_by_ext.items() if k in audio_exts)
    video_count = sum(v for k, v in files_by_ext.items() if k in video_exts)
    text_count = sum(v for k, v in files_by_ext.items() if k in text_exts)

    if img_count > 0 and len(subdirs) > 1:
        return "image_classification"
    if img_count > 0 and any(e in files_by_ext for e in [".txt", ".json", ".xml"]):
        return "object_detection"
    if img_count > 0:
        return "images"
    if audio_count > 0:
        return "audio"
    if video_count > 0:
        return "video"
    if ".csv" in files_by_ext:
        return "tabular"
    if ".jsonl" in files_by_ext or ".json" in files_by_ext:
        return "text"
    return "unknown"
