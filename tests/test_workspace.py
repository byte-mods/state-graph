"""Tests for workspace manager and code executor."""

import pytest
import shutil
from pathlib import Path

from state_graph.workspace.manager import WorkspaceManager, Project, WORKSPACE_ROOT
from state_graph.workspace.executor import CodeExecutor


@pytest.fixture(autouse=True)
def clean_workspace():
    yield
    # Clean up test workspaces
    if WORKSPACE_ROOT.exists():
        for d in WORKSPACE_ROOT.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)


class TestProject:
    def test_create_empty(self):
        p = Project("test_project")
        assert p.path.exists()
        assert (p.path / "main.py").exists()
        assert (p.path / ".gitignore").exists()

    def test_create_llm_template(self):
        p = Project("llm_test", template="llm_finetune")
        assert (p.path / "scripts" / "train.py").exists()
        assert (p.path / "config.json").exists()
        assert (p.path / "scripts" / "prepare_data.py").exists()

    def test_create_yolo_template(self):
        p = Project("yolo_test", template="yolo")
        assert (p.path / "scripts" / "train_yolo.py").exists()
        assert (p.path / "data" / "images").is_dir()
        assert (p.path / "data" / "labels").is_dir()

    def test_read_write_file(self):
        p = Project("rw_test")
        p.write_file("test.txt", "hello world")
        result = p.read_file("test.txt")
        assert result["status"] == "ok"
        assert result["content"] == "hello world"

    def test_create_file(self):
        p = Project("cf_test")
        result = p.create_file("new_file.py", "print('hi')")
        assert result["status"] == "created"
        assert (p.path / "new_file.py").exists()

    def test_create_file_duplicate(self):
        p = Project("dup_test")
        p.create_file("x.py", "")
        result = p.create_file("x.py", "")
        assert result["status"] == "error"

    def test_create_dir(self):
        p = Project("dir_test")
        p.create_dir("subdir/nested")
        assert (p.path / "subdir" / "nested").is_dir()

    def test_delete_file(self):
        p = Project("del_test")
        p.write_file("to_delete.txt", "bye")
        result = p.delete_file("to_delete.txt")
        assert result["status"] == "deleted"
        assert not (p.path / "to_delete.txt").exists()

    def test_rename_file(self):
        p = Project("ren_test")
        p.write_file("old.txt", "content")
        result = p.rename_file("old.txt", "new.txt")
        assert result["status"] == "renamed"
        assert (p.path / "new.txt").exists()
        assert not (p.path / "old.txt").exists()

    def test_file_tree(self):
        p = Project("tree_test", template="llm_finetune")
        tree = p.get_file_tree()
        assert len(tree) > 0
        names = [n["name"] for n in tree]
        assert "scripts" in names
        assert "config.json" in names

    def test_list_files(self):
        p = Project("list_test")
        files = p.list_files()
        assert len(files) > 0

    def test_to_dict(self):
        p = Project("dict_test")
        d = p.to_dict()
        assert "id" in d
        assert "name" in d
        assert "path" in d


class TestWorkspaceManager:
    def test_create_project(self):
        wm = WorkspaceManager()
        p = wm.create("test")
        assert wm.get(p.id) is not None
        assert len(wm.list_all()) >= 1

    def test_delete_project(self):
        wm = WorkspaceManager()
        p = wm.create("to_delete")
        path = p.path
        wm.delete(p.id)
        assert wm.get(p.id) is None
        assert not path.exists()

    def test_persist_and_reload(self):
        wm1 = WorkspaceManager()
        p = wm1.create("persist_test")
        pid = p.id
        # Simulate restart
        wm2 = WorkspaceManager()
        assert wm2.get(pid) is not None


class TestCodeExecutor:
    def test_run_code(self):
        ex = CodeExecutor()
        result = ex.run_code("print('hello from test')")
        assert result["status"] == "ok"
        assert "hello from test" in result["stdout"]
        assert result["returncode"] == 0

    def test_run_code_error(self):
        ex = CodeExecutor()
        result = ex.run_code("raise ValueError('boom')")
        assert result["returncode"] != 0
        assert "boom" in result["stderr"]

    def test_run_code_with_imports(self):
        ex = CodeExecutor()
        result = ex.run_code("import math\nprint(math.pi)")
        assert "3.14" in result["stdout"]

    def test_run_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('file execution works')")
        ex = CodeExecutor()
        result = ex.run_file(str(f))
        assert "file execution works" in result["stdout"]

    def test_timeout(self):
        ex = CodeExecutor()
        result = ex.run_code("import time; time.sleep(10)", timeout=1)
        assert result["status"] == "timeout"

    def test_install_package(self):
        # Test with a package that's likely already installed
        ex = CodeExecutor()
        result = ex.install_package("pip")
        assert result["status"] == "ok"
