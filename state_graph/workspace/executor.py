"""Python code execution with output capture for the in-browser IDE."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class CodeExecutor:
    """Execute Python code in a subprocess with stdout/stderr capture."""

    def __init__(self):
        self._process: subprocess.Popen | None = None

    def run_code(self, code: str, cwd: str | None = None, timeout: int = 300) -> dict:
        """Execute Python code string. Returns stdout, stderr, return code."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return {
                "status": "ok",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": f"Execution timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            os.unlink(tmp_path)

    def run_file(self, file_path: str, cwd: str | None = None, timeout: int = 300) -> dict:
        """Execute a Python file."""
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        try:
            result = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or str(p.parent),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return {
                "status": "ok",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": f"Execution timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def run_file_streaming(self, file_path: str, cwd: str | None = None, callback=None) -> dict:
        """Execute a file with line-by-line output streaming."""
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(p),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or str(p.parent),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        self._process = proc
        stdout_lines = []
        stderr_lines = []

        async def read_stream(stream, lines, stream_type):
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace")
                lines.append(text)
                if callback:
                    await callback(stream_type, text)

        await asyncio.gather(
            read_stream(proc.stdout, stdout_lines, "stdout"),
            read_stream(proc.stderr, stderr_lines, "stderr"),
        )

        await proc.wait()
        self._process = None

        return {
            "status": "ok",
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
            "returncode": proc.returncode,
        }

    def stop(self) -> dict:
        """Kill the running process."""
        if self._process:
            self._process.kill()
            self._process = None
            return {"status": "killed"}
        return {"status": "not_running"}

    def install_package(self, package: str) -> dict:
        """Install a pip package."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
