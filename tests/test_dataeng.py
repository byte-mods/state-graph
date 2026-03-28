"""Tests for data engineering — connectors, pipelines, transforms."""

import json
import csv
import tempfile
import pytest
from pathlib import Path

from state_graph.dataeng.connectors import CONNECTOR_REGISTRY, DataConnector
from state_graph.dataeng.pipeline import (
    PipelineManager, Pipeline, apply_transform, compute_stats, TRANSFORM_REGISTRY,
)


class TestConnectorRegistry:
    def test_count(self):
        assert len(CONNECTOR_REGISTRY) == 15

    def test_all_have_required_fields(self):
        for cid, cfg in CONNECTOR_REGISTRY.items():
            assert "name" in cfg
            assert "category" in cfg
            assert "params" in cfg

    def test_categories(self):
        cats = set(cfg["category"] for cfg in CONNECTOR_REGISTRY.values())
        assert "sql" in cats
        assert "nosql" in cats
        assert "file" in cats


class TestCSVConnector:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        conn = DataConnector("csv_file", {"path": str(csv_file)})
        result = conn.load()
        assert result["status"] == "ok"
        assert result["count"] == 2
        assert "name" in result["columns"]

    def test_sink_csv(self, tmp_path):
        out_file = tmp_path / "output.csv"
        conn = DataConnector("csv_file", {"path": str(out_file)})
        result = conn.sink([{"a": 1, "b": 2}, {"a": 3, "b": 4}], str(out_file))
        assert result["status"] == "ok"
        assert result["written"] == 2
        assert out_file.exists()


class TestJSONConnector:
    def test_load_json_array(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text(json.dumps([{"x": 1}, {"x": 2}]))
        conn = DataConnector("json_file", {"path": str(f)})
        result = conn.load()
        assert result["count"] == 2

    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
        conn = DataConnector("json_file", {"path": str(f), "lines": True})
        result = conn.load()
        assert result["count"] == 3


class TestSQLiteConnector:
    def test_sqlite_roundtrip(self, tmp_path):
        db = tmp_path / "test.db"
        import sqlite3
        cx = sqlite3.connect(str(db))
        cx.execute("CREATE TABLE t (name TEXT, val INTEGER)")
        cx.execute("INSERT INTO t VALUES ('a', 1)")
        cx.execute("INSERT INTO t VALUES ('b', 2)")
        cx.commit()
        cx.close()

        conn = DataConnector("sqlite", {"path": str(db)})
        result = conn.load("SELECT * FROM t")
        assert result["status"] == "ok"
        assert result["count"] == 2

    def test_list_tables(self, tmp_path):
        db = tmp_path / "test.db"
        import sqlite3
        cx = sqlite3.connect(str(db))
        cx.execute("CREATE TABLE users (id INTEGER)")
        cx.execute("CREATE TABLE orders (id INTEGER)")
        cx.commit()
        cx.close()

        conn = DataConnector("sqlite", {"path": str(db)})
        result = conn.list_tables()
        assert "users" in result["tables"]
        assert "orders" in result["tables"]


class TestTransforms:
    @pytest.fixture
    def sample_data(self):
        return [
            {"name": "Alice", "age": "30", "city": "NYC", "score": "85"},
            {"name": "Bob", "age": "25", "city": "LA", "score": "92"},
            {"name": "Charlie", "age": "", "city": "NYC", "score": "78"},
            {"name": "Diana", "age": "28", "city": "SF", "score": ""},
        ]

    def test_select_columns(self, sample_data):
        result = apply_transform(sample_data, "select_columns", {"columns": "name, city"})
        assert list(result[0].keys()) == ["name", "city"]

    def test_drop_columns(self, sample_data):
        result = apply_transform(sample_data, "drop_columns", {"columns": "score"})
        assert "score" not in result[0]

    def test_filter_equals(self, sample_data):
        result = apply_transform(sample_data, "filter_rows", {"column": "city", "operator": "==", "value": "NYC"})
        assert len(result) == 2

    def test_filter_contains(self, sample_data):
        result = apply_transform(sample_data, "filter_rows", {"column": "name", "operator": "contains", "value": "li"})
        assert len(result) == 2  # Alice, Charlie

    def test_drop_nulls(self, sample_data):
        result = apply_transform(sample_data, "drop_nulls", {"columns": "age"})
        assert len(result) == 3  # Charlie's age is empty

    def test_fill_nulls(self, sample_data):
        result = apply_transform(sample_data, "fill_nulls", {"column": "age", "value": "0", "strategy": "value"})
        assert all(r["age"] != "" for r in result)

    def test_cast_type(self, sample_data):
        result = apply_transform(sample_data[:2], "cast_type", {"column": "age", "dtype": "int"})
        assert result[0]["age"] == 30

    def test_deduplicate(self):
        rows = [{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = apply_transform(rows, "deduplicate", {"columns": ""})
        assert len(result) == 2

    def test_sort(self, sample_data):
        result = apply_transform(sample_data, "sort", {"column": "name", "descending": True})
        assert result[0]["name"] == "Diana"

    def test_limit(self, sample_data):
        result = apply_transform(sample_data, "limit", {"n": 2})
        assert len(result) == 2

    def test_text_clean(self):
        rows = [{"text": "  Hello WORLD <b>html</b> https://url.com  "}]
        result = apply_transform(rows, "text_clean", {
            "column": "text", "lowercase": True, "strip": True,
            "remove_html": True, "remove_urls": True,
        })
        assert "hello world" in result[0]["text"]
        assert "<b>" not in result[0]["text"]
        assert "https" not in result[0]["text"]

    def test_text_split(self):
        rows = [{"text": " ".join(["word"] * 100)}]
        result = apply_transform(rows, "text_split", {"column": "text", "max_length": 20, "overlap": 5})
        assert len(result) > 1

    def test_add_column(self):
        rows = [{"a": 10, "b": 20}]
        result = apply_transform(rows, "add_column", {"name": "c", "expression": "row['a'] + row['b']"})
        assert result[0]["c"] == 30

    def test_rename_columns(self):
        rows = [{"old_name": 1}]
        result = apply_transform(rows, "rename_columns", {"mapping": {"old_name": "new_name"}})
        assert "new_name" in result[0]


class TestTransformRegistry:
    def test_count(self):
        assert len(TRANSFORM_REGISTRY) == 17


class TestComputeStats:
    def test_numeric_stats(self):
        rows = [{"val": 10}, {"val": 20}, {"val": 30}]
        stats = compute_stats(rows)
        assert stats["columns"]["val"]["type"] == "numeric"
        assert stats["columns"]["val"]["mean"] == 20.0

    def test_categorical_stats(self):
        rows = [{"cat": "a"}, {"cat": "b"}, {"cat": "a"}]
        stats = compute_stats(rows)
        assert stats["columns"]["cat"]["type"] == "categorical"
        assert stats["columns"]["cat"]["top_values"][0]["value"] == "a"

    def test_null_count(self):
        rows = [{"x": 1}, {"x": None}, {"x": ""}]
        stats = compute_stats(rows)
        assert stats["columns"]["x"]["null_count"] == 2

    def test_empty(self):
        stats = compute_stats([])
        assert stats["row_count"] == 0


class TestPipeline:
    def test_full_pipeline(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\nCharlie,35\n")

        p = Pipeline("test")
        p.add_source("s1", "csv_file", {"path": str(csv_file)})
        p.load_source("s1")
        p.add_transform("cast_type", {"column": "age", "dtype": "int"})
        p.add_transform("filter_rows", {"column": "age", "operator": ">", "value": "26"})
        result = p.run("s1")
        assert result["status"] == "ok"
        assert result["count"] == 2  # Alice(30) and Charlie(35)

    def test_pipeline_to_dict(self):
        p = Pipeline("test")
        d = p.to_dict()
        assert "id" in d
        assert "name" in d
        assert "transforms" in d


class TestPipelineManager:
    def test_create_and_list(self):
        pm = PipelineManager()
        p = pm.create("test")
        assert len(pm.list_all()) >= 1
        pm.delete(p.id)
