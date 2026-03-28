"""Tests for the dataset factory."""

import json
import pytest
from pathlib import Path
from state_graph.datasets.creator import DatasetManager, TEMPLATES, DatasetProject
from state_graph.datasets.sources import LocalSource, _detect_dataset_type
from state_graph.datasets import converters


class TestTemplates:
    def test_all_templates_have_required_fields(self):
        for tid, t in TEMPLATES.items():
            assert "name" in t, f"{tid} missing name"
            assert "category" in t, f"{tid} missing category"
            assert "fields" in t, f"{tid} missing fields"
            assert "export_formats" in t, f"{tid} missing export_formats"
            assert "description" in t, f"{tid} missing description"

    def test_template_categories(self):
        categories = {t["category"] for t in TEMPLATES.values()}
        assert "text" in categories
        assert "image" in categories
        assert "audio" in categories
        assert "video" in categories
        assert "multimodal" in categories

    def test_21_templates(self):
        assert len(TEMPLATES) == 21

    def test_list_by_category(self):
        cats = DatasetManager.list_templates_by_category()
        assert "text" in cats
        assert len(cats["text"]) >= 6


class TestDatasetProject:
    def test_create(self):
        dm = DatasetManager()
        res = dm.create_project("test", "text_classification", ["pos", "neg"])
        assert res["status"] == "created"
        assert res["project"]["template_id"] == "text_classification"

    def test_add_sample(self):
        dm = DatasetManager()
        res = dm.create_project("test", "text_classification")
        p = dm.get_project(res["project"]["id"])
        r = p.add_sample({"text": "hello", "label": "pos"})
        assert r["status"] == "added"
        assert r["total"] == 1

    def test_bulk_add(self):
        dm = DatasetManager()
        res = dm.create_project("test", "text_classification")
        p = dm.get_project(res["project"]["id"])
        r = p.add_samples_bulk([
            {"text": "a", "label": "pos"},
            {"text": "b", "label": "neg"},
            {"text": "c", "label": "pos"},
        ])
        assert r["total"] == 3

    def test_remove_sample(self):
        dm = DatasetManager()
        res = dm.create_project("test", "qa")
        p = dm.get_project(res["project"]["id"])
        r = p.add_sample({"context": "c", "question": "q", "answer": "a"})
        sid = r["sample_id"]
        p.remove_sample(sid)
        assert p.get_stats()["total_samples"] == 0

    def test_get_samples(self):
        dm = DatasetManager()
        res = dm.create_project("test", "reasoning")
        p = dm.get_project(res["project"]["id"])
        for i in range(10):
            p.add_sample({"problem": f"p{i}", "solution_steps": "s", "answer": str(i)})
        result = p.get_samples(offset=5, limit=3)
        assert len(result["samples"]) == 3
        assert result["total"] == 10

    def test_stats(self):
        dm = DatasetManager()
        res = dm.create_project("test", "text_classification", ["pos", "neg"])
        p = dm.get_project(res["project"]["id"])
        p.add_sample({"text": "a", "label": "pos"})
        p.add_sample({"text": "b", "label": "neg"})
        p.add_sample({"text": "c", "label": "pos"})
        stats = p.get_stats()
        assert stats["label_distribution"]["pos"] == 2
        assert stats["label_distribution"]["neg"] == 1

    def test_export_jsonl(self):
        dm = DatasetManager()
        res = dm.create_project("test_export", "text_classification")
        p = dm.get_project(res["project"]["id"])
        p.add_sample({"text": "hello", "label": "pos"})
        result = p.export("jsonl")
        assert result["status"] == "exported"
        assert Path(result["path"]).exists()

    def test_export_csv(self):
        dm = DatasetManager()
        res = dm.create_project("test_csv", "summarization")
        p = dm.get_project(res["project"]["id"])
        p.add_sample({"document": "long text", "summary": "short"})
        result = p.export("csv")
        assert result["status"] == "exported"

    def test_export_alpaca(self):
        dm = DatasetManager()
        res = dm.create_project("test_alpaca", "text_generation")
        p = dm.get_project(res["project"]["id"])
        p.add_sample({"instruction": "Summarize", "input": "text", "output": "summary"})
        result = p.export("alpaca")
        assert result["status"] == "exported"
        data = json.loads(Path(result["path"]).read_text())
        assert data[0]["instruction"] == "Summarize"

    def test_export_sharegpt(self):
        dm = DatasetManager()
        res = dm.create_project("test_sgpt", "conversation")
        p = dm.get_project(res["project"]["id"])
        p.add_sample({"conversations": [{"role": "user", "content": "Hi"}]})
        result = p.export("sharegpt")
        assert result["status"] == "exported"

    def test_all_text_templates_export(self):
        dm = DatasetManager()
        text_templates = [t for t, v in TEMPLATES.items() if v["category"] == "text"]
        for tid in text_templates:
            res = dm.create_project(f"test_{tid}", tid)
            p = dm.get_project(res["project"]["id"])
            example = TEMPLATES[tid].get("example", {})
            if example:
                p.add_sample(example)
                result = p.export("jsonl")
                assert result["status"] == "exported", f"Failed to export {tid}"

    def test_tool_calling_dataset(self):
        dm = DatasetManager()
        res = dm.create_project("tools", "tool_calling")
        p = dm.get_project(res["project"]["id"])
        p.add_sample({
            "query": "Weather?",
            "tools": [{"name": "weather"}],
            "tool_call": {"name": "weather", "args": {}},
        })
        assert p.get_stats()["total_samples"] == 1


class TestDetectType:
    def test_image_classification(self):
        assert _detect_dataset_type({".jpg": 100}, ["cats", "dogs"]) == "image_classification"

    def test_tabular(self):
        assert _detect_dataset_type({".csv": 1}, []) == "tabular"

    def test_audio(self):
        assert _detect_dataset_type({".wav": 50}, []) == "audio"

    def test_video(self):
        assert _detect_dataset_type({".mp4": 20}, []) == "video"


class TestConverters:
    def test_alpaca_sharegpt_roundtrip(self, tmp_path):
        alpaca = [
            {"instruction": "Do X", "input": "context", "output": "result"},
            {"instruction": "Do Y", "input": "", "output": "answer"},
        ]
        alpaca_path = tmp_path / "alpaca.json"
        alpaca_path.write_text(json.dumps(alpaca))

        # Alpaca → ShareGPT
        res = converters.alpaca_to_sharegpt(str(alpaca_path))
        assert res["status"] == "converted"
        assert res["count"] == 2

        # ShareGPT → Alpaca
        res2 = converters.sharegpt_to_alpaca(res["path"])
        assert res2["status"] == "converted"
        assert res2["count"] == 2

    def test_csv_jsonl_roundtrip(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,age\nAlice,30\nBob,25\n")

        res = converters.csv_to_jsonl(str(csv_path))
        assert res["status"] == "converted"
        assert res["rows"] == 2

        res2 = converters.jsonl_to_csv(res["path"])
        assert res2["status"] == "converted"
        assert res2["rows"] == 2
