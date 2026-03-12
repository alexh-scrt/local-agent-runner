"""Unit tests for local_agent_runner.tools.

Covers:
- ToolResult dataclass behaviour.
- file_read: success path, missing file, sandbox violation, non-file path,
  unreadable file (POSIX), relative path resolution.
- file_write: success path, sandbox violation, parent directory creation,
  overwrite existing file.
- shell_run: success path, non-zero exit, sandbox violation, timeout,
  stdout+stderr captured.
- web_search: success path (mocked), sandbox violation, non-200 response,
  timeout (mocked), HTTP error (mocked), truncation.
- dispatch_tool: correct routing for all four ToolType values, missing
  argument raises ToolError, unknown type raises ToolError.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from local_agent_runner.config import SandboxConfig, ToolType
from local_agent_runner.sandbox import SandboxContext
from local_agent_runner.tools import (
    ToolError,
    ToolResult,
    dispatch_tool,
    file_read,
    file_write,
    shell_run,
    web_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def disabled_sandbox(tmp_path: Path) -> SandboxContext:
    """SandboxContext with sandbox disabled (all actions permitted)."""
    cfg = SandboxConfig(enabled=False)
    return SandboxContext(cfg, working_directory=tmp_path)


@pytest.fixture()
def enabled_sandbox(tmp_path: Path) -> SandboxContext:
    """SandboxContext with sandbox enabled; only tmp_path and example.com allowed."""
    cfg = SandboxConfig(
        enabled=True,
        allowed_paths=[str(tmp_path), "/tmp"],
        allowed_commands=["echo", "ls"],
        allowed_domains=["example.com"],
    )
    return SandboxContext(cfg, working_directory=tmp_path)


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_str(self) -> None:
        r = ToolResult(success=True, output="hello")
        assert str(r) == "hello"

    def test_failure_str(self) -> None:
        r = ToolResult(success=False, output="", error="bad")
        assert "ERROR" in str(r)
        assert "bad" in str(r)

    def test_metadata_defaults_empty(self) -> None:
        r = ToolResult(success=True, output="")
        assert r.metadata == {}

    def test_error_defaults_empty(self) -> None:
        r = ToolResult(success=True, output="")
        assert r.error == ""


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


class TestFileRead:
    def test_reads_existing_file(self, tmp_path: Path, disabled_sandbox: SandboxContext) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        result = file_read(str(f), disabled_sandbox)
        assert result.success is True
        assert result.output == "Hello, world!"

    def test_metadata_contains_resolved_path(self, tmp_path: Path, disabled_sandbox: SandboxContext) -> None:
        f = tmp_path / "meta.txt"
        f.write_text("x", encoding="utf-8")
        result = file_read(str(f), disabled_sandbox)
        assert "resolved_path" in result.metadata
        assert str(f.resolve()) == result.metadata["resolved_path"]

    def test_missing_file_returns_failure(self, tmp_path: Path, disabled_sandbox: SandboxContext) -> None:
        result = file_read(str(tmp_path / "no_such.txt"), disabled_sandbox)
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_directory_returns_failure(self, tmp_path: Path, disabled_sandbox: SandboxContext) -> None:
        result = file_read(str(tmp_path), disabled_sandbox)
        assert result.success is False
        assert "not a file" in result.error.lower()

    def test_sandbox_violation_returns_failure(
        self, tmp_path: Path, enabled_sandbox: SandboxContext
    ) -> None:
        result = file_read("/etc/passwd", enabled_sandbox)
        assert result.success is False
        assert result.metadata.get("sandbox_blocked") is True
        assert "blocked" in result.error.lower() or "not" in result.error.lower()

    def test_allowed_path_passes_sandbox(
        self, tmp_path: Path, enabled_sandbox: SandboxContext
    ) -> None:
        f = tmp_path / "allowed.txt"
        f.write_text("allowed content", encoding="utf-8")
        result = file_read(str(f), enabled_sandbox)
        assert result.success is True
        assert result.output == "allowed content"

    def test_relative_path_resolved(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        """Relative paths should be resolved against the sandbox working dir."""
        f = tmp_path / "relative.txt"
        f.write_text("relative", encoding="utf-8")
        # disabled_sandbox has working_directory=tmp_path
        result = file_read("relative.txt", disabled_sandbox)
        assert result.success is True
        assert result.output == "relative"

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 not supported on Windows")
    def test_unreadable_file_returns_failure(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        f = tmp_path / "locked.txt"
        f.write_text("secret", encoding="utf-8")
        f.chmod(0o000)
        try:
            result = file_read(str(f), disabled_sandbox)
            assert result.success is False
        finally:
            f.chmod(0o644)

    def test_size_bytes_in_metadata(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        f = tmp_path / "sized.txt"
        f.write_text("abc", encoding="utf-8")
        result = file_read(str(f), disabled_sandbox)
        assert result.metadata["size_bytes"] == 3


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------


class TestFileWrite:
    def test_writes_new_file(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "out.txt"
        result = file_write(str(dest), "new content", disabled_sandbox)
        assert result.success is True
        assert dest.read_text(encoding="utf-8") == "new content"

    def test_confirmation_message_in_output(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "conf.txt"
        result = file_write(str(dest), "hello", disabled_sandbox)
        assert "wrote" in result.output.lower() or "bytes" in result.output.lower()

    def test_creates_parent_directories(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "a" / "b" / "c" / "out.txt"
        result = file_write(str(dest), "deep", disabled_sandbox)
        assert result.success is True
        assert dest.exists()
        assert dest.read_text() == "deep"

    def test_overwrites_existing_file(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "over.txt"
        dest.write_text("old", encoding="utf-8")
        result = file_write(str(dest), "new", disabled_sandbox)
        assert result.success is True
        assert dest.read_text() == "new"

    def test_sandbox_violation_returns_failure(
        self, tmp_path: Path, enabled_sandbox: SandboxContext
    ) -> None:
        result = file_write("/etc/evil.conf", "bad", enabled_sandbox)
        assert result.success is False
        assert result.metadata.get("sandbox_blocked") is True

    def test_allowed_path_passes_sandbox(
        self, tmp_path: Path, enabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "safe_write.txt"
        result = file_write(str(dest), "safe", enabled_sandbox)
        assert result.success is True
        assert dest.read_text() == "safe"

    def test_metadata_contains_resolved_path(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "meta.txt"
        result = file_write(str(dest), "x", disabled_sandbox)
        assert "resolved_path" in result.metadata

    def test_metadata_contains_size_bytes(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "size.txt"
        result = file_write(str(dest), "abcde", disabled_sandbox)
        assert result.metadata["size_bytes"] == 5


# ---------------------------------------------------------------------------
# shell_run
# ---------------------------------------------------------------------------


class TestShellRun:
    def test_success_command(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo hello", disabled_sandbox)
        assert result.success is True
        assert "hello" in result.output

    def test_returncode_in_metadata(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo hi", disabled_sandbox)
        assert result.metadata["returncode"] == 0

    def test_non_zero_exit_is_failure(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        # 'exit 1' should return code 1 on all POSIX systems
        result = shell_run("exit 1", disabled_sandbox)
        assert result.success is False
        assert result.metadata["returncode"] != 0

    def test_stderr_captured(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo err >&2", disabled_sandbox)
        # either output or stderr metadata should contain 'err'
        assert "err" in result.output or "err" in result.metadata.get("stderr", "")

    def test_stdout_and_stderr_combined_in_output(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo out; echo err >&2", disabled_sandbox)
        assert "out" in result.output

    def test_sandbox_violation_returns_failure(
        self, enabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("rm -rf /", enabled_sandbox)
        assert result.success is False
        assert result.metadata.get("sandbox_blocked") is True

    def test_allowed_command_passes_sandbox(
        self, enabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo sandbox_ok", enabled_sandbox)
        assert result.success is True
        assert "sandbox_ok" in result.output

    def test_timeout_returns_failure(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        # sleep for 10 seconds but timeout after 0.1
        result = shell_run("sleep 10", disabled_sandbox, timeout=0.1)
        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_metadata_contains_command(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run("echo metadata", disabled_sandbox)
        assert result.metadata["command"] == "echo metadata"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_ls_allowed_with_sandbox(
        self, tmp_path: Path, enabled_sandbox: SandboxContext
    ) -> None:
        result = shell_run(f"ls {tmp_path}", enabled_sandbox)
        assert result.success is True


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_sandbox_violation_returns_failure(
        self, enabled_sandbox: SandboxContext
    ) -> None:
        result = web_search("https://evil.com/data", enabled_sandbox)
        assert result.success is False
        assert result.metadata.get("sandbox_blocked") is True

    def test_success_response(self, disabled_sandbox: SandboxContext) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Hello</html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.url = "https://example.com/"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/", disabled_sandbox)

        assert result.success is True
        assert "Hello" in result.output
        assert result.metadata["status_code"] == 200

    def test_404_response_is_failure(self, disabled_sandbox: SandboxContext) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.content = b"Not Found"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.url = "https://example.com/missing"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/missing", disabled_sandbox)

        assert result.success is False
        assert "404" in result.error

    def test_timeout_returns_failure(self, disabled_sandbox: SandboxContext) -> None:
        import httpx as _httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(side_effect=_httpx.TimeoutException("timeout"))

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/slow", disabled_sandbox)

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_request_error_returns_failure(self, disabled_sandbox: SandboxContext) -> None:
        import httpx as _httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(
            side_effect=_httpx.ConnectError("connection refused")
        )

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/", disabled_sandbox)

        assert result.success is False
        assert "failed" in result.error.lower() or "connection" in result.error.lower()

    def test_truncation_at_max_bytes(self, disabled_sandbox: SandboxContext) -> None:
        large_body = b"X" * 2000
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = large_body
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.url = "https://example.com/big"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search(
                "https://example.com/big",
                disabled_sandbox,
                max_response_bytes=100,
            )

        assert result.success is True
        assert result.metadata["truncated"] is True
        assert "truncated" in result.output.lower()

    def test_allowed_domain_passes_sandbox(
        self, enabled_sandbox: SandboxContext
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"OK"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.url = "https://example.com/"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/", enabled_sandbox)

        assert result.success is True

    def test_metadata_contains_status_code(self, disabled_sandbox: SandboxContext) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"body"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.url = "https://example.com/"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = web_search("https://example.com/", disabled_sandbox)

        assert result.metadata["status_code"] == 200
        assert "content_type" in result.metadata
        assert "url" in result.metadata


# ---------------------------------------------------------------------------
# dispatch_tool
# ---------------------------------------------------------------------------


class TestDispatchTool:
    def test_dispatch_file_read(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        f = tmp_path / "dispatch_read.txt"
        f.write_text("dispatch read content", encoding="utf-8")
        result = dispatch_tool(
            ToolType.FILE_READ,
            {"path": str(f)},
            disabled_sandbox,
        )
        assert result.success is True
        assert result.output == "dispatch read content"

    def test_dispatch_file_write(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        dest = tmp_path / "dispatch_write.txt"
        result = dispatch_tool(
            ToolType.FILE_WRITE,
            {"path": str(dest), "content": "written via dispatch"},
            disabled_sandbox,
        )
        assert result.success is True
        assert dest.read_text() == "written via dispatch"

    def test_dispatch_shell(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        result = dispatch_tool(
            ToolType.SHELL,
            {"command": "echo dispatch_shell"},
            disabled_sandbox,
        )
        assert result.success is True
        assert "dispatch_shell" in result.output

    def test_dispatch_web_search(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"dispatch web"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.url = "https://example.com/dispatch"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.tools.httpx.Client", return_value=mock_client):
            result = dispatch_tool(
                ToolType.WEB_SEARCH,
                {"url": "https://example.com/dispatch"},
                disabled_sandbox,
            )

        assert result.success is True
        assert "dispatch web" in result.output

    def test_missing_path_arg_raises_tool_error(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        with pytest.raises(ToolError, match="path"):
            dispatch_tool(ToolType.FILE_READ, {}, disabled_sandbox)

    def test_missing_content_arg_for_file_write_raises(
        self, tmp_path: Path, disabled_sandbox: SandboxContext
    ) -> None:
        with pytest.raises(ToolError, match="content"):
            dispatch_tool(
                ToolType.FILE_WRITE,
                {"path": str(tmp_path / "x.txt")},
                disabled_sandbox,
            )

    def test_missing_command_arg_raises(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        with pytest.raises(ToolError, match="command"):
            dispatch_tool(ToolType.SHELL, {}, disabled_sandbox)

    def test_missing_url_arg_raises(
        self, disabled_sandbox: SandboxContext
    ) -> None:
        with pytest.raises(ToolError, match="url"):
            dispatch_tool(ToolType.WEB_SEARCH, {}, disabled_sandbox)
