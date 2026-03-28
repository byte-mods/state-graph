"""Data pipeline engine — load, clean, transform, merge, visualize, sink."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from state_graph.dataeng.connectors import DataConnector, CONNECTOR_REGISTRY


# ── Transform Operations ──

TRANSFORM_REGISTRY = {
    "select_columns": {
        "name": "Select Columns",
        "description": "Keep only specified columns",
        "params": [{"name": "columns", "type": "list", "label": "Column names (comma-sep)"}],
    },
    "drop_columns": {
        "name": "Drop Columns",
        "description": "Remove specified columns",
        "params": [{"name": "columns", "type": "list", "label": "Column names (comma-sep)"}],
    },
    "rename_columns": {
        "name": "Rename Columns",
        "description": "Rename columns (JSON: {old: new})",
        "params": [{"name": "mapping", "type": "json", "label": "Rename map"}],
    },
    "filter_rows": {
        "name": "Filter Rows",
        "description": "Keep rows matching condition",
        "params": [
            {"name": "column", "type": "str", "label": "Column"},
            {"name": "operator", "type": "select", "options": ["==", "!=", ">", "<", ">=", "<=", "contains", "not_contains", "regex", "is_null", "not_null"]},
            {"name": "value", "type": "str", "label": "Value"},
        ],
    },
    "drop_nulls": {
        "name": "Drop Null Rows",
        "description": "Remove rows with null values",
        "params": [{"name": "columns", "type": "list", "label": "Columns (blank=all)", "default": ""}],
    },
    "fill_nulls": {
        "name": "Fill Null Values",
        "description": "Replace null values",
        "params": [
            {"name": "column", "type": "str"},
            {"name": "value", "type": "str", "label": "Fill value"},
            {"name": "strategy", "type": "select", "options": ["value", "mean", "median", "mode", "ffill", "bfill"]},
        ],
    },
    "cast_type": {
        "name": "Cast Column Type",
        "description": "Convert column to different type",
        "params": [
            {"name": "column", "type": "str"},
            {"name": "dtype", "type": "select", "options": ["str", "int", "float", "bool"]},
        ],
    },
    "add_column": {
        "name": "Add Column (Expression)",
        "description": "Add computed column using Python expression",
        "params": [
            {"name": "name", "type": "str", "label": "New column name"},
            {"name": "expression", "type": "str", "label": "Expression (use row['col'])"},
        ],
    },
    "deduplicate": {
        "name": "Remove Duplicates",
        "description": "Remove duplicate rows",
        "params": [{"name": "columns", "type": "list", "label": "Key columns (blank=all)", "default": ""}],
    },
    "sort": {
        "name": "Sort",
        "description": "Sort by column(s)",
        "params": [
            {"name": "column", "type": "str"},
            {"name": "descending", "type": "bool", "default": False},
        ],
    },
    "limit": {
        "name": "Limit Rows",
        "description": "Keep only first N rows",
        "params": [{"name": "n", "type": "int", "default": 1000}],
    },
    "sample": {
        "name": "Random Sample",
        "description": "Take random sample of N rows",
        "params": [{"name": "n", "type": "int", "default": 100}],
    },
    "text_clean": {
        "name": "Clean Text",
        "description": "Clean text column (lowercase, strip, remove special chars)",
        "params": [
            {"name": "column", "type": "str"},
            {"name": "lowercase", "type": "bool", "default": True},
            {"name": "strip", "type": "bool", "default": True},
            {"name": "remove_html", "type": "bool", "default": False},
            {"name": "remove_urls", "type": "bool", "default": False},
            {"name": "remove_special", "type": "bool", "default": False},
        ],
    },
    "text_split": {
        "name": "Split Text to Chunks",
        "description": "Split long text into chunks for LLM training",
        "params": [
            {"name": "column", "type": "str"},
            {"name": "max_length", "type": "int", "default": 512},
            {"name": "overlap", "type": "int", "default": 50},
        ],
    },
    "flatten_json": {
        "name": "Flatten JSON Column",
        "description": "Expand a JSON/dict column into multiple columns",
        "params": [{"name": "column", "type": "str"}],
    },
    "merge": {
        "name": "Merge / Join",
        "description": "Merge with another loaded dataset",
        "params": [
            {"name": "right_source_id", "type": "str", "label": "Source ID to merge with"},
            {"name": "on", "type": "str", "label": "Join column"},
            {"name": "how", "type": "select", "options": ["inner", "left", "right", "outer"]},
        ],
    },
    "concat": {
        "name": "Concatenate",
        "description": "Stack rows from another loaded dataset",
        "params": [{"name": "source_id", "type": "str", "label": "Source ID to concat"}],
    },
}


def apply_transform(rows: list[dict], op: str, params: dict, context: dict | None = None) -> list[dict]:
    """Apply a single transform operation to rows."""
    if op == "select_columns":
        cols = _parse_list(params.get("columns", ""))
        return [{k: r.get(k) for k in cols} for r in rows]

    elif op == "drop_columns":
        cols = set(_parse_list(params.get("columns", "")))
        return [{k: v for k, v in r.items() if k not in cols} for r in rows]

    elif op == "rename_columns":
        mapping = params.get("mapping", {})
        if isinstance(mapping, str):
            mapping = json.loads(mapping)
        return [{mapping.get(k, k): v for k, v in r.items()} for r in rows]

    elif op == "filter_rows":
        col = params["column"]
        operator = params["operator"]
        val = params.get("value", "")
        return [r for r in rows if _filter_match(r.get(col), operator, val)]

    elif op == "drop_nulls":
        cols = _parse_list(params.get("columns", ""))
        if not cols:
            return [r for r in rows if all(v is not None and v != "" for v in r.values())]
        return [r for r in rows if all(r.get(c) is not None and r.get(c) != "" for c in cols)]

    elif op == "fill_nulls":
        col = params["column"]
        strategy = params.get("strategy", "value")
        fill_val = params.get("value", "")
        if strategy == "mean":
            vals = [float(r[col]) for r in rows if r.get(col) is not None and r[col] != ""]
            fill_val = sum(vals) / len(vals) if vals else 0
        elif strategy == "median":
            vals = sorted(float(r[col]) for r in rows if r.get(col) is not None and r[col] != "")
            fill_val = vals[len(vals) // 2] if vals else 0
        elif strategy == "mode":
            from collections import Counter
            vals = [r[col] for r in rows if r.get(col) is not None and r[col] != ""]
            fill_val = Counter(vals).most_common(1)[0][0] if vals else ""
        for r in rows:
            if r.get(col) is None or r[col] == "":
                r[col] = fill_val
        return rows

    elif op == "cast_type":
        col = params["column"]
        dtype = params["dtype"]
        casters = {"str": str, "int": lambda x: int(float(x)), "float": float, "bool": lambda x: str(x).lower() in ("true", "1", "yes")}
        cast_fn = casters.get(dtype, str)
        for r in rows:
            try:
                r[col] = cast_fn(r[col])
            except (ValueError, TypeError):
                pass
        return rows

    elif op == "add_column":
        name = params["name"]
        expr = params["expression"]
        for r in rows:
            try:
                r[name] = eval(expr, {"__builtins__": {}}, {"row": r, "len": len, "str": str, "int": int, "float": float})
            except Exception:
                r[name] = None
        return rows

    elif op == "deduplicate":
        cols = _parse_list(params.get("columns", ""))
        seen = set()
        result = []
        for r in rows:
            key = tuple(r.get(c) for c in cols) if cols else tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key)
                result.append(r)
        return result

    elif op == "sort":
        col = params["column"]
        desc = params.get("descending", False)
        return sorted(rows, key=lambda r: r.get(col, ""), reverse=desc)

    elif op == "limit":
        return rows[:int(params.get("n", 1000))]

    elif op == "sample":
        import random
        n = min(int(params.get("n", 100)), len(rows))
        return random.sample(rows, n)

    elif op == "text_clean":
        col = params["column"]
        for r in rows:
            text = str(r.get(col, ""))
            if params.get("strip"):
                text = text.strip()
            if params.get("lowercase"):
                text = text.lower()
            if params.get("remove_html"):
                text = re.sub(r"<[^>]+>", "", text)
            if params.get("remove_urls"):
                text = re.sub(r"https?://\S+", "", text)
            if params.get("remove_special"):
                text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)
            r[col] = text
        return rows

    elif op == "text_split":
        col = params["column"]
        max_len = int(params.get("max_length", 512))
        overlap = int(params.get("overlap", 50))
        result = []
        for r in rows:
            text = str(r.get(col, ""))
            words = text.split()
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i + max_len])
                new_row = {**r, col: chunk, "_chunk_idx": i}
                result.append(new_row)
                i += max_len - overlap
        return result

    elif op == "flatten_json":
        col = params["column"]
        result = []
        for r in rows:
            val = r.get(col, {})
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    val = {}
            if isinstance(val, dict):
                new_row = {k: v for k, v in r.items() if k != col}
                for k, v in val.items():
                    new_row[f"{col}_{k}"] = v
                result.append(new_row)
            else:
                result.append(r)
        return result

    elif op == "merge":
        if not context or "sources" not in context:
            return rows
        right_id = params.get("right_source_id")
        right_rows = context["sources"].get(right_id, [])
        on_col = params["on"]
        how = params.get("how", "inner")
        right_map = {}
        for rr in right_rows:
            key = rr.get(on_col)
            right_map.setdefault(key, []).append(rr)

        result = []
        matched_keys = set()
        for lr in rows:
            key = lr.get(on_col)
            if key in right_map:
                for rr in right_map[key]:
                    merged = {**lr}
                    for k, v in rr.items():
                        if k != on_col:
                            merged[f"{k}_right" if k in lr else k] = v
                    result.append(merged)
                matched_keys.add(key)
            elif how in ("left", "outer"):
                result.append(lr)

        if how in ("right", "outer"):
            for rr in right_rows:
                if rr.get(on_col) not in matched_keys:
                    result.append(rr)

        return result

    elif op == "concat":
        if not context or "sources" not in context:
            return rows
        other_rows = context["sources"].get(params.get("source_id"), [])
        return rows + other_rows

    return rows


def _parse_list(val: str) -> list[str]:
    if not val:
        return []
    return [s.strip() for s in val.split(",") if s.strip()]


def _filter_match(val, op, target) -> bool:
    if op == "is_null":
        return val is None or val == ""
    if op == "not_null":
        return val is not None and val != ""
    if val is None:
        return False
    s = str(val)
    if op == "==":
        return s == target
    elif op == "!=":
        return s != target
    elif op == "contains":
        return target.lower() in s.lower()
    elif op == "not_contains":
        return target.lower() not in s.lower()
    elif op == "regex":
        return bool(re.search(target, s))
    try:
        nv, nt = float(val), float(target)
        if op == ">": return nv > nt
        elif op == "<": return nv < nt
        elif op == ">=": return nv >= nt
        elif op == "<=": return nv <= nt
    except (ValueError, TypeError):
        pass
    return False


def compute_stats(rows: list[dict]) -> dict:
    """Compute column-level stats for visualization."""
    if not rows:
        return {"columns": {}, "row_count": 0}

    columns = list(rows[0].keys())
    stats = {}

    for col in columns:
        vals = [r.get(col) for r in rows]
        non_null = [v for v in vals if v is not None and v != ""]
        null_count = len(vals) - len(non_null)

        col_stat = {
            "count": len(vals),
            "null_count": null_count,
            "null_pct": round(null_count / len(vals) * 100, 1) if vals else 0,
            "unique": len(set(str(v) for v in non_null)),
        }

        # Try numeric stats
        try:
            nums = [float(v) for v in non_null]
            if nums:
                col_stat["type"] = "numeric"
                col_stat["min"] = min(nums)
                col_stat["max"] = max(nums)
                col_stat["mean"] = round(sum(nums) / len(nums), 4)
                sorted_nums = sorted(nums)
                col_stat["median"] = sorted_nums[len(sorted_nums) // 2]
                # Distribution buckets for chart
                n_buckets = min(20, len(set(nums)))
                if n_buckets > 1:
                    step = (max(nums) - min(nums)) / n_buckets
                    if step > 0:
                        buckets = [0] * n_buckets
                        for n in nums:
                            idx = min(int((n - min(nums)) / step), n_buckets - 1)
                            buckets[idx] += 1
                        col_stat["histogram"] = {
                            "labels": [round(min(nums) + i * step, 2) for i in range(n_buckets)],
                            "counts": buckets,
                        }
        except (ValueError, TypeError):
            # Categorical stats
            col_stat["type"] = "categorical"
            from collections import Counter
            counter = Counter(str(v) for v in non_null)
            top = counter.most_common(20)
            col_stat["top_values"] = [{"value": v, "count": c} for v, c in top]
            col_stat["value_counts"] = {"labels": [v for v, _ in top], "counts": [c for _, c in top]}

        # String length stats for text
        str_vals = [str(v) for v in non_null]
        if str_vals and col_stat.get("type") != "numeric":
            lengths = [len(s) for s in str_vals]
            col_stat["avg_length"] = round(sum(lengths) / len(lengths), 1)
            col_stat["max_length"] = max(lengths)

        stats[col] = col_stat

    return {"columns": stats, "row_count": len(rows), "col_count": len(columns)}


class Pipeline:
    """A data pipeline: source → transforms → sink."""

    def __init__(self, name: str = ""):
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"Pipeline {self.id}"
        self.sources: dict[str, dict] = {}  # id -> {connector_type, params, loaded_rows}
        self.transforms: list[dict] = []  # [{op, params, enabled}]
        self.sink_config: dict | None = None
        self._data: dict[str, list[dict]] = {}  # source_id -> rows
        self._result: list[dict] = []  # After transforms

    def add_source(self, source_id: str, connector_type: str, params: dict) -> dict:
        self.sources[source_id] = {"connector_type": connector_type, "params": params}
        return {"status": "added", "source_id": source_id}

    def load_source(self, source_id: str, query: str | None = None, limit: int = 10000) -> dict:
        src = self.sources.get(source_id)
        if not src:
            return {"status": "error", "message": "Source not found"}
        conn = DataConnector(src["connector_type"], src["params"])
        result = conn.load(query, limit)
        if result["status"] == "ok":
            self._data[source_id] = result["rows"]
        return result

    def add_transform(self, op: str, params: dict) -> dict:
        tid = str(uuid.uuid4())[:6]
        self.transforms.append({"id": tid, "op": op, "params": params, "enabled": True})
        return {"status": "added", "transform_id": tid}

    def remove_transform(self, transform_id: str) -> dict:
        self.transforms = [t for t in self.transforms if t["id"] != transform_id]
        return {"status": "removed"}

    def reorder_transform(self, transform_id: str, new_index: int) -> dict:
        t = next((t for t in self.transforms if t["id"] == transform_id), None)
        if not t:
            return {"status": "error"}
        self.transforms.remove(t)
        self.transforms.insert(new_index, t)
        return {"status": "reordered"}

    def toggle_transform(self, transform_id: str) -> dict:
        for t in self.transforms:
            if t["id"] == transform_id:
                t["enabled"] = not t["enabled"]
                return {"status": "toggled", "enabled": t["enabled"]}
        return {"status": "error"}

    def run(self, primary_source_id: str | None = None) -> dict:
        """Execute the pipeline: load sources, apply transforms."""
        if not primary_source_id:
            primary_source_id = list(self._data.keys())[0] if self._data else None
        if not primary_source_id or primary_source_id not in self._data:
            return {"status": "error", "message": "No data loaded"}

        rows = list(self._data[primary_source_id])
        context = {"sources": self._data}

        for t in self.transforms:
            if not t.get("enabled", True):
                continue
            try:
                rows = apply_transform(rows, t["op"], t["params"], context)
            except Exception as e:
                return {"status": "error", "message": f"Transform '{t['op']}' failed: {e}", "transform_id": t["id"]}

        self._result = rows
        columns = list(rows[0].keys()) if rows else []
        return {
            "status": "ok",
            "count": len(rows),
            "columns": columns,
            "preview": rows[:50],
        }

    def get_result(self) -> list[dict]:
        return self._result

    def get_stats(self) -> dict:
        return compute_stats(self._result)

    def sink(self, connector_type: str, params: dict, target: str | None = None) -> dict:
        if not self._result:
            return {"status": "error", "message": "No data to sink. Run the pipeline first."}
        conn = DataConnector(connector_type, params)
        return conn.sink(self._result, target)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "sources": {k: {"connector_type": v["connector_type"], "row_count": len(self._data.get(k, []))} for k, v in self.sources.items()},
            "transforms": self.transforms,
            "result_count": len(self._result),
        }


class PipelineManager:
    """Manages multiple pipelines with JSON persistence."""

    _SAVE_FILE = "pipelines.json"

    def __init__(self):
        self.pipelines: dict[str, Pipeline] = {}
        self._load()

    def _save(self):
        try:
            from pathlib import Path
            data = []
            for p in self.pipelines.values():
                data.append({
                    "id": p.id, "name": p.name,
                    "sources": {k: {"connector_type": v["connector_type"], "params": v["params"]} for k, v in p.sources.items()},
                    "transforms": p.transforms,
                })
            Path(self._SAVE_FILE).write_text(json.dumps(data, default=str))
        except Exception:
            pass

    def _load(self):
        try:
            from pathlib import Path
            f = Path(self._SAVE_FILE)
            if not f.exists():
                return
            for item in json.loads(f.read_text()):
                p = Pipeline(item.get("name", ""))
                p.id = item["id"]
                for sid, src in item.get("sources", {}).items():
                    p.sources[sid] = src
                p.transforms = item.get("transforms", [])
                self.pipelines[p.id] = p
        except Exception:
            pass

    def create(self, name: str = "") -> Pipeline:
        p = Pipeline(name)
        self.pipelines[p.id] = p
        self._save()
        return p

    def get(self, pid: str) -> Pipeline | None:
        return self.pipelines.get(pid)

    def list_all(self) -> list[dict]:
        return [p.to_dict() for p in self.pipelines.values()]

    def delete(self, pid: str) -> dict:
        if pid in self.pipelines:
            del self.pipelines[pid]
            self._save()
            return {"status": "deleted"}
        return {"status": "error", "message": "Not found"}
