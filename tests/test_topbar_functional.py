import pytest
import json
import csv
import tempfile
import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from state_graph.server.app import app, engine


@pytest.fixture(autouse=True)
def reset_engine():
    engine.reset()
    yield
    engine.reset()


client = TestClient(app)


# ============================================================
# 1. Data Engineering — Connectors
# ============================================================


class TestDataEngConnectors:
    def test_list_connectors(self):
        resp = client.get("/api/dataeng/connectors")
        assert resp.status_code == 200
        data = resp.json()
        assert "connectors" in data
        # Should have 15 connectors (returned as a dict keyed by connector id)
        assert len(data["connectors"]) == 15
        # Check key connectors exist
        for conn in ["csv_file", "json_file", "sqlite", "parquet"]:
            assert conn in data["connectors"]

    def test_list_transforms(self):
        resp = client.get("/api/dataeng/transforms")
        assert resp.status_code == 200
        data = resp.json()
        assert "transforms" in data
        assert len(data["transforms"]) >= 15  # 17 transforms

    def test_preview_csv(self):
        # Create a temp CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])
            csv_path = f.name
        try:
            resp = client.post(
                "/api/dataeng/preview",
                json={
                    "connector_type": "csv_file",
                    "params": {"path": csv_path},
                    "limit": 10,
                },
            )
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_stats_with_rows(self):
        rows = [
            {"name": "Alice", "age": 30, "score": 85.5},
            {"name": "Bob", "age": 25, "score": 92.3},
            {"name": "Charlie", "age": 35, "score": 78.1},
        ]
        resp = client.post("/api/dataeng/stats", json={"rows": rows})
        assert resp.status_code == 200

    def test_connector_categories(self):
        resp = client.get("/api/dataeng/connectors")
        data = resp.json()["connectors"]
        # Each connector should have name, category, params
        for cid, info in data.items():
            assert "name" in info
            assert "category" in info
            assert "params" in info

    def test_preview_json(self):
        # Create a temp JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
                f,
            )
            json_path = f.name
        try:
            resp = client.post(
                "/api/dataeng/preview",
                json={
                    "connector_type": "json_file",
                    "params": {"path": json_path},
                    "limit": 10,
                },
            )
            assert resp.status_code == 200
        finally:
            os.unlink(json_path)


# ============================================================
# 2. Data Engineering — Pipelines
# ============================================================


class TestDataEngPipelines:
    @pytest.fixture(autouse=True)
    def reset_pipe_mgr(self):
        from state_graph.server.app import _pipe_mgr

        _pipe_mgr.pipelines.clear()
        yield
        _pipe_mgr.pipelines.clear()

    def test_create_pipeline(self):
        resp = client.post("/api/dataeng/pipelines", json={"name": "Test Pipeline"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    def test_list_pipelines(self):
        client.post("/api/dataeng/pipelines", json={"name": "Pipe 1"})
        client.post("/api/dataeng/pipelines", json={"name": "Pipe 2"})
        resp = client.get("/api/dataeng/pipelines")
        assert resp.status_code == 200
        assert len(resp.json()["pipelines"]) == 2

    def test_get_pipeline(self):
        resp = client.post("/api/dataeng/pipelines", json={"name": "Get Test"})
        pid = resp.json()["pipeline"]["id"]
        resp = client.get(f"/api/dataeng/pipelines/{pid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pid

    def test_delete_pipeline(self):
        resp = client.post("/api/dataeng/pipelines", json={"name": "Delete Me"})
        pid = resp.json()["pipeline"]["id"]
        resp = client.delete(f"/api/dataeng/pipelines/{pid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_get_nonexistent_pipeline(self):
        resp = client.get("/api/dataeng/pipelines/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["status"] == "error"

    def test_full_csv_pipeline(self):
        # Create CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "score"])
            for i in range(20):
                writer.writerow([f"person_{i}", 20 + i, 50 + i * 2.5])
            csv_path = f.name
        try:
            # Create pipeline
            resp = client.post("/api/dataeng/pipelines", json={"name": "CSV Pipeline"})
            pid = resp.json()["pipeline"]["id"]
            # Add CSV source
            resp = client.post(
                f"/api/dataeng/pipelines/{pid}/sources",
                json={
                    "connector_type": "csv_file",
                    "params": {"path": csv_path},
                    "source_id": "src1",
                },
            )
            assert resp.json()["status"] == "added"
            # Load source data
            resp = client.post(f"/api/dataeng/pipelines/{pid}/sources/src1/load")
            assert resp.status_code == 200
            # Add transform: filter age > 25
            resp = client.post(
                f"/api/dataeng/pipelines/{pid}/transforms",
                json={
                    "op": "filter",
                    "params": {"column": "age", "op": ">", "value": 25},
                },
            )
            assert resp.status_code == 200
            # Run pipeline
            resp = client.post(f"/api/dataeng/pipelines/{pid}/run")
            assert resp.status_code == 200
            # Get result
            resp = client.get(f"/api/dataeng/pipelines/{pid}/result")
            assert resp.status_code == 200
            result = resp.json()
            assert "rows" in result
            assert "total" in result
            # Get stats
            resp = client.get(f"/api/dataeng/pipelines/{pid}/stats")
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_add_and_remove_transform(self):
        resp = client.post("/api/dataeng/pipelines", json={"name": "Transform Test"})
        pid = resp.json()["pipeline"]["id"]
        resp = client.post(
            f"/api/dataeng/pipelines/{pid}/transforms",
            json={
                "op": "select",
                "params": {"columns": ["name", "age"]},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "added"
        # Get transform id from pipeline
        pipe = client.get(f"/api/dataeng/pipelines/{pid}").json()
        tid = pipe["transforms"][0]["id"]
        # Remove it
        resp = client.delete(f"/api/dataeng/pipelines/{pid}/transforms/{tid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

    def test_sqlite_pipeline(self):
        import sqlite3

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            cx = sqlite3.connect(db_path)
            cx.execute("CREATE TABLE users (name TEXT, age INT, score REAL)")
            for i in range(10):
                cx.execute(
                    "INSERT INTO users VALUES (?, ?, ?)",
                    (f"user_{i}", 20 + i, 50.0 + i),
                )
            cx.commit()
            cx.close()

            resp = client.post("/api/dataeng/pipelines", json={"name": "SQLite Pipe"})
            pid = resp.json()["pipeline"]["id"]
            resp = client.post(
                f"/api/dataeng/pipelines/{pid}/sources",
                json={
                    "connector_type": "sqlite",
                    "params": {"path": db_path},
                    "source_id": "db1",
                },
            )
            assert resp.json()["status"] == "added"
        finally:
            os.unlink(db_path)

    def test_pipeline_add_source_to_nonexistent(self):
        resp = client.post(
            "/api/dataeng/pipelines/nonexistent/sources",
            json={
                "connector_type": "csv_file",
                "params": {"path": "/tmp/fake.csv"},
                "source_id": "s1",
            },
        )
        assert resp.json()["status"] == "error"

    def test_pipeline_run_without_data(self):
        resp = client.post("/api/dataeng/pipelines", json={"name": "Empty Pipe"})
        pid = resp.json()["pipeline"]["id"]
        resp = client.post(f"/api/dataeng/pipelines/{pid}/run")
        assert resp.status_code == 200


# ============================================================
# 3. Dataset Factory
# ============================================================


class TestDatasetFactory:
    @pytest.fixture(autouse=True)
    def reset_ds(self):
        from state_graph.server.app import _ds_manager

        _ds_manager.projects.clear()
        yield
        _ds_manager.projects.clear()

    def test_list_templates(self):
        resp = client.get("/api/ds/templates")
        assert resp.status_code == 200

    def test_list_all_templates(self):
        resp = client.get("/api/ds/templates/all")
        assert resp.status_code == 200
        data = resp.json()
        # list_templates returns a dict keyed by template id, not wrapped
        assert len(data) == 21

    def test_create_project(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Test Dataset",
                "template_id": "text_classification",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"
        assert "project" in resp.json()

    def test_list_projects(self):
        client.post(
            "/api/ds/projects",
            json={"name": "DS 1", "template_id": "text_classification"},
        )
        client.post(
            "/api/ds/projects",
            json={"name": "DS 2", "template_id": "qa"},
        )
        resp = client.get("/api/ds/projects")
        assert resp.status_code == 200
        assert len(resp.json()["projects"]) == 2

    def test_add_and_get_samples(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Samples Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        # Add sample — the endpoint passes the body dict directly to add_sample
        resp = client.post(
            f"/api/ds/projects/{pid}/samples",
            json={"text": "Hello world", "label": "positive"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "added"
        # Get samples
        resp = client.get(f"/api/ds/projects/{pid}/samples")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_bulk_add_samples(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Bulk Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        resp = client.post(
            f"/api/ds/projects/{pid}/samples/bulk",
            json={
                "samples": [
                    {
                        "text": f"Sample {i}",
                        "label": "pos" if i % 2 == 0 else "neg",
                    }
                    for i in range(10)
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "added"
        assert resp.json()["count"] == 10

    def test_delete_sample(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Del Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        resp = client.post(
            f"/api/ds/projects/{pid}/samples",
            json={"text": "delete me", "label": "neg"},
        )
        sid = resp.json()["sample_id"]
        resp = client.delete(f"/api/ds/projects/{pid}/samples/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "removed"

    def test_update_sample(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Update Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        resp = client.post(
            f"/api/ds/projects/{pid}/samples",
            json={"text": "original", "label": "neg"},
        )
        sid = resp.json()["sample_id"]
        resp = client.put(
            f"/api/ds/projects/{pid}/samples/{sid}",
            json={"text": "updated", "label": "pos"},
        )
        assert resp.status_code == 200

    def test_delete_project(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Delete Project",
                "template_id": "qa",
            },
        )
        pid = resp.json()["project"]["id"]
        resp = client.delete(f"/api/ds/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_export_project(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Export Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        client.post(
            f"/api/ds/projects/{pid}/samples/bulk",
            json={
                "samples": [
                    {"text": f"sample {i}", "label": "pos"} for i in range(5)
                ],
            },
        )
        resp = client.post(
            f"/api/ds/projects/{pid}/export",
            json={"format": "jsonl"},
        )
        assert resp.status_code == 200

    def test_format_conversion(self):
        # Create a temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow(["hello", "pos"])
            writer.writerow(["bye", "neg"])
            csv_path = f.name
        try:
            resp = client.post(
                "/api/ds/convert",
                json={
                    "conversion": "csv_to_jsonl",
                    "input": csv_path,
                },
            )
            assert resp.status_code == 200
        finally:
            os.unlink(csv_path)

    def test_get_project_stats(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Stats Test",
                "template_id": "text_classification",
            },
        )
        pid = resp.json()["project"]["id"]
        # Add some samples
        for i in range(3):
            client.post(
                f"/api/ds/projects/{pid}/samples",
                json={"text": f"text {i}", "label": "pos"},
            )
        resp = client.get(f"/api/ds/projects/{pid}")
        assert resp.status_code == 200

    def test_create_project_invalid_template(self):
        resp = client.post(
            "/api/ds/projects",
            json={
                "name": "Bad Template",
                "template_id": "nonexistent_template",
            },
        )
        assert resp.json()["status"] == "error"

    def test_get_nonexistent_project(self):
        resp = client.get("/api/ds/projects/nonexistent")
        assert resp.json()["status"] == "error"


# ============================================================
# 4. Workspace / IDE
# ============================================================


class TestWorkspace:
    @pytest.fixture(autouse=True)
    def reset_workspace(self):
        from state_graph.server.app import _workspace

        for pid in list(_workspace.projects.keys()):
            _workspace.delete(pid)
        yield
        for pid in list(_workspace.projects.keys()):
            _workspace.delete(pid)

    def test_list_templates(self):
        resp = client.get("/api/workspace/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert len(data["templates"]) >= 4

    def test_create_project(self):
        resp = client.post(
            "/api/workspace/projects",
            json={
                "name": "test_project",
                "template": "empty",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"
        assert "project" in resp.json()

    def test_list_projects(self):
        client.post("/api/workspace/projects", json={"name": "proj1"})
        resp = client.get("/api/workspace/projects")
        assert resp.status_code == 200
        assert "projects" in resp.json()

    def test_file_operations(self):
        resp = client.post("/api/workspace/projects", json={"name": "file_test"})
        pid = resp.json()["project"]["id"]
        # Write file
        resp = client.post(
            f"/api/workspace/projects/{pid}/write",
            json={
                "path": "test.py",
                "content": "print('hello')",
            },
        )
        assert resp.status_code == 200
        # Read file
        resp = client.post(
            f"/api/workspace/projects/{pid}/read",
            json={"path": "test.py"},
        )
        assert resp.status_code == 200
        assert "hello" in resp.json()["content"]
        # Get tree
        resp = client.get(f"/api/workspace/projects/{pid}/tree")
        assert resp.status_code == 200
        assert "tree" in resp.json()

    def test_create_and_delete_file(self):
        resp = client.post("/api/workspace/projects", json={"name": "crud_files"})
        pid = resp.json()["project"]["id"]
        # Create file
        resp = client.post(
            f"/api/workspace/projects/{pid}/create_file",
            json={"path": "new_file.py", "content": "x = 1"},
        )
        assert resp.status_code == 200
        # Delete file
        resp = client.post(
            f"/api/workspace/projects/{pid}/delete",
            json={"path": "new_file.py"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_rename_file(self):
        resp = client.post("/api/workspace/projects", json={"name": "rename_test"})
        pid = resp.json()["project"]["id"]
        # Write a file first
        client.post(
            f"/api/workspace/projects/{pid}/write",
            json={"path": "old_name.py", "content": "# old"},
        )
        # Rename
        resp = client.post(
            f"/api/workspace/projects/{pid}/rename",
            json={"old_path": "old_name.py", "new_path": "new_name.py"},
        )
        assert resp.status_code == 200

    def test_create_directory(self):
        resp = client.post("/api/workspace/projects", json={"name": "dir_test"})
        pid = resp.json()["project"]["id"]
        resp = client.post(
            f"/api/workspace/projects/{pid}/create_dir",
            json={"path": "subdir"},
        )
        assert resp.status_code == 200

    def test_list_files(self):
        resp = client.post("/api/workspace/projects", json={"name": "list_files"})
        pid = resp.json()["project"]["id"]
        # Write some files
        client.post(
            f"/api/workspace/projects/{pid}/write",
            json={"path": "a.py", "content": "# a"},
        )
        client.post(
            f"/api/workspace/projects/{pid}/write",
            json={"path": "b.py", "content": "# b"},
        )
        resp = client.get(f"/api/workspace/projects/{pid}/files")
        assert resp.status_code == 200
        assert "files" in resp.json()

    def test_run_code(self):
        resp = client.post(
            "/api/workspace/run",
            json={"code": "print('hello from workspace')"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "hello from workspace" in data.get("stdout", "")

    def test_run_code_with_error(self):
        resp = client.post(
            "/api/workspace/run",
            json={"code": "raise ValueError('test error')"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should capture stderr or error info
        assert data.get("stderr", "") != "" or data.get("status") == "error"

    def test_delete_project(self):
        resp = client.post("/api/workspace/projects", json={"name": "del_proj"})
        pid = resp.json()["project"]["id"]
        resp = client.delete(f"/api/workspace/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_project(self):
        resp = client.delete("/api/workspace/projects/nonexistent")
        assert resp.json()["status"] == "error"

    def test_get_nonexistent_project_tree(self):
        resp = client.get("/api/workspace/projects/nonexistent/tree")
        assert resp.json()["status"] == "error"


# ============================================================
# 5. Evaluation
# ============================================================


class TestEvaluation:
    def test_classification_eval(self):
        resp = client.post(
            "/api/eval/classification",
            json={
                "y_true": [0, 1, 1, 0, 1, 0, 1, 0],
                "y_pred": [0, 1, 0, 0, 1, 1, 1, 0],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "accuracy" in data

    def test_regression_eval(self):
        resp = client.post(
            "/api/eval/regression",
            json={
                "y_true": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y_pred": [1.1, 2.2, 2.8, 4.1, 5.3],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should contain at least one regression metric
        assert any(
            k in data for k in ["mse", "rmse", "mae", "r2", "mean_squared_error"]
        )

    def test_classification_eval_with_labels(self):
        resp = client.post(
            "/api/eval/classification",
            json={
                "y_true": [0, 1, 2, 0, 1, 2],
                "y_pred": [0, 2, 1, 0, 1, 2],
                "labels": [0, 1, 2],
            },
        )
        assert resp.status_code == 200
        assert "accuracy" in resp.json()

    def test_classification_eval_perfect(self):
        labels = [0, 1, 0, 1, 0, 1]
        resp = client.post(
            "/api/eval/classification",
            json={"y_true": labels, "y_pred": labels},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accuracy"] == 1.0

    def test_regression_eval_perfect(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        resp = client.post(
            "/api/eval/regression",
            json={"y_true": values, "y_pred": values},
        )
        assert resp.status_code == 200

    def test_auto_eval_without_model(self):
        resp = client.post("/api/eval/auto")
        assert resp.json()["status"] == "error"
        assert "No model" in resp.json()["message"]


# ============================================================
# 6. Deployment Code Generation
# ============================================================


class TestDeployment:
    def test_generate_server_code(self):
        resp = client.post(
            "/api/deploy/server",
            json={
                "model_path": "./model.onnx",
                "model_type": "onnx",
                "port": 8080,
            },
        )
        assert resp.status_code == 200
        assert "code" in resp.json()
        assert len(resp.json()["code"]) > 0

    def test_generate_dockerfile(self):
        resp = client.post(
            "/api/deploy/dockerfile",
            json={
                "model_path": "./model.onnx",
                "model_type": "onnx",
            },
        )
        assert resp.status_code == 200
        assert "dockerfile" in resp.json()
        assert len(resp.json()["dockerfile"]) > 0

    def test_generate_gradio(self):
        resp = client.post(
            "/api/deploy/gradio",
            json={
                "model_path": "./model.pt",
                "model_type": "pytorch",
            },
        )
        assert resp.status_code == 200
        assert "code" in resp.json()
        assert len(resp.json()["code"]) > 0

    def test_onnx_export_without_model(self):
        resp = client.post("/api/deploy/onnx", json={})
        assert resp.json()["status"] == "error"
        assert "No model" in resp.json()["message"]

    def test_torchscript_export_without_model(self):
        resp = client.post("/api/deploy/torchscript", json={})
        assert resp.json()["status"] == "error"
        assert "No model" in resp.json()["message"]

    def test_generate_server_pytorch(self):
        resp = client.post(
            "/api/deploy/server",
            json={
                "model_path": "./model.pt",
                "model_type": "pytorch",
                "port": 9090,
            },
        )
        assert resp.status_code == 200
        assert "code" in resp.json()

    def test_generate_dockerfile_pytorch(self):
        resp = client.post(
            "/api/deploy/dockerfile",
            json={
                "model_path": "./model.pt",
                "model_type": "pytorch",
            },
        )
        assert resp.status_code == 200
        assert "dockerfile" in resp.json()

    def test_onnx_export_with_model(self):
        # Build a simple model first
        client.post(
            "/api/graph/layer",
            json={
                "layer_type": "Linear",
                "params": {"in_features": 2, "out_features": 2},
            },
        )
        client.post(
            "/api/data/load",
            json={"dataset": "xor", "type": "synthetic"},
        )
        client.post("/api/build")
        output_path = os.path.join(
            tempfile.gettempdir(), "test_topbar_onnx_model.onnx"
        )
        try:
            resp = client.post(
                "/api/deploy/onnx",
                json={"path": output_path},
            )
            assert resp.status_code == 200
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_torchscript_export_with_model(self):
        # Build a simple model first
        client.post(
            "/api/graph/layer",
            json={
                "layer_type": "Linear",
                "params": {"in_features": 2, "out_features": 2},
            },
        )
        client.post(
            "/api/data/load",
            json={"dataset": "xor", "type": "synthetic"},
        )
        client.post("/api/build")
        output_path = os.path.join(
            tempfile.gettempdir(), "test_topbar_ts_model.pt"
        )
        try:
            resp = client.post(
                "/api/deploy/torchscript",
                json={
                    "path": output_path,
                    "method": "trace",
                },
            )
            assert resp.status_code == 200
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
