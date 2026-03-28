"""AI Assistant — LLM-powered code editing, error fixing, and project guidance.

Supports: Claude (Anthropic), OpenAI (GPT-4), Ollama (local), any OpenAI-compatible API.
Reads project files, understands context, makes direct edits.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


SYSTEM_PROMPT = """You are an expert ML/AI engineer assistant integrated into StateGraph IDE.
You help researchers write, fix, and improve Python code for machine learning projects.

When asked to edit code:
- Return the COMPLETE modified file content inside <code> tags
- Explain what you changed and why
- If the code has errors, fix them and explain the root cause

When asked questions:
- Give concise, practical answers
- Include code examples when helpful
- Reference specific files in the project when relevant

You have access to the project's file tree and can read any file.
The researcher may not be a programmer — explain things clearly."""


class AIAssistant:
    """Multi-backend LLM assistant for code editing and project guidance."""

    def __init__(self):
        self.provider: str = ""  # claude, openai, ollama, custom
        self.api_key: str = ""
        self.model: str = ""
        self.base_url: str = ""
        self.history: list[dict] = []
        self._configured = False

    def configure(
        self,
        provider: str = "claude",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> dict:
        """Configure the LLM backend."""
        self.provider = provider

        if provider == "claude":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self.model = model or "claude-sonnet-4-20250514"
            self.base_url = "https://api.anthropic.com"
        elif provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.model = model or "gpt-4o"
            self.base_url = base_url or "https://api.openai.com/v1"
        elif provider == "ollama":
            self.model = model or "llama3.1"
            self.base_url = base_url or "http://localhost:11434"
            self.api_key = ""
        elif provider == "custom":
            self.api_key = api_key
            self.model = model
            self.base_url = base_url
        else:
            return {"status": "error", "message": f"Unknown provider: {provider}"}

        if provider in ("claude", "openai") and not self.api_key:
            return {"status": "error", "message": f"API key required for {provider}. Set {provider.upper()}_API_KEY env var or provide in config."}

        self._configured = True
        return {"status": "configured", "provider": provider, "model": self.model}

    def chat(
        self,
        message: str,
        file_content: str | None = None,
        file_path: str | None = None,
        error_output: str | None = None,
        project_files: list[str] | None = None,
    ) -> dict:
        """Send a message to the AI with optional code context."""
        if not self._configured:
            return {"status": "error", "message": "AI not configured. Set provider and API key first."}

        # Build context
        context_parts = []
        if file_path and file_content:
            context_parts.append(f"Current file ({file_path}):\n```python\n{file_content}\n```")
        if error_output:
            context_parts.append(f"Error output:\n```\n{error_output}\n```")
        if project_files:
            context_parts.append(f"Project files: {', '.join(project_files)}")

        full_message = message
        if context_parts:
            full_message = "\n\n".join(context_parts) + "\n\n" + message

        # Add to history
        self.history.append({"role": "user", "content": full_message})

        # Call LLM
        try:
            if self.provider == "claude":
                response = self._call_claude(full_message)
            elif self.provider == "openai" or self.provider == "custom":
                response = self._call_openai(full_message)
            elif self.provider == "ollama":
                response = self._call_ollama(full_message)
            else:
                return {"status": "error", "message": "Provider not configured"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

        self.history.append({"role": "assistant", "content": response})

        # Extract code blocks if present
        code_blocks = self._extract_code(response)

        return {
            "status": "ok",
            "response": response,
            "code_blocks": code_blocks,
            "has_code_edit": len(code_blocks) > 0,
        }

    def fix_error(self, code: str, error: str, file_path: str = "") -> dict:
        """Specifically ask the AI to fix an error in code."""
        message = f"""The following code has an error. Fix it and return the complete corrected code.

File: {file_path}
```python
{code}
```

Error:
```
{error}
```

Fix the error. Return the COMPLETE fixed file content inside <code> tags. Then explain what was wrong."""

        return self.chat(message)

    def improve_code(self, code: str, instruction: str = "", file_path: str = "") -> dict:
        """Ask the AI to improve or modify code."""
        message = f"""Modify this code as instructed. Return the COMPLETE modified file inside <code> tags.

File: {file_path}
```python
{code}
```

Instruction: {instruction or 'Improve this code — add error handling, type hints, and documentation.'}"""

        return self.chat(message)

    def explain_code(self, code: str) -> dict:
        """Ask the AI to explain code."""
        return self.chat(f"Explain this code in detail:\n```python\n{code}\n```")

    def generate_code(self, description: str, project_context: str = "") -> dict:
        """Generate new code from a description."""
        msg = f"Generate Python code for: {description}"
        if project_context:
            msg += f"\n\nProject context:\n{project_context}"
        msg += "\n\nReturn complete, runnable code inside <code> tags."
        return self.chat(msg)

    def _call_claude(self, message: str) -> str:
        import urllib.request

        messages = [{"role": "user", "content": message}]

        body = json.dumps({
            "model": self.model,
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]

    def _call_openai(self, message: str) -> str:
        import urllib.request

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]

    def _call_ollama(self, message: str) -> str:
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]

    def _extract_code(self, text: str) -> list[dict]:
        """Extract code blocks from AI response."""
        blocks = []

        # <code>...</code> tags
        for match in re.finditer(r"<code>(.*?)</code>", text, re.DOTALL):
            blocks.append({"source": "code_tag", "content": match.group(1).strip()})

        # ```python ... ``` blocks
        for match in re.finditer(r"```(?:python)?\n(.*?)```", text, re.DOTALL):
            content = match.group(1).strip()
            if content and len(content) > 50:  # Only substantial blocks
                blocks.append({"source": "markdown", "content": content})

        return blocks

    def clear_history(self):
        self.history = []
        return {"status": "cleared"}

    def get_info(self) -> dict:
        return {
            "configured": self._configured,
            "provider": self.provider,
            "model": self.model,
            "history_length": len(self.history),
        }
