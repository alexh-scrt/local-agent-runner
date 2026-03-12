"""Tool stubs for local_agent_runner.

This module implements the four tool types supported by the agentic runner:

- ``file_read``   — read the text content of a file.
- ``file_write``  — write text content to a file, creating parent directories
  as necessary.
- ``shell``       — execute a shell command and capture stdout/stderr.
- ``web_search``  — perform an HTTP GET request and return the response body.

Every tool function accepts a :class:`~local_agent_runner.sandbox.SandboxContext`
and uses it to enforce path/command/domain allow-lists before performing any
side effects.  All actions are recorded in the context regardless of whether
they are permitted.

Public API
----------
- ``ToolError``        — base exception for tool failures.
- ``ToolResult``       — typed return value carrying success flag, output text,
  and optional metadata.
- ``file_read()``      — read a file's text content.
- ``file_write()``     — write text to a file.
- ``shell_run()``      — execute a shell command.
- ``web_search()``     — HTTP GET a URL and return the body.
- ``dispatch_tool()``  — route a tool call by :class:`~local_agent_runner.config.ToolType`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from local_agent_runner.config import ToolType
from local_agent_runner.sandbox import SandboxContext, SandboxViolation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    """Raised when a tool fails for a reason other than a sandbox violation.

    Attributes
    ----------
    tool_name:
        The name of the tool that failed (e.g. ``"file_read"``).
    detail:
        Human-readable description of the failure.
    """

    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"[{tool_name}] {detail}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Typed return value from any tool function.

    Attributes
    ----------
    success:
        ``True`` if the tool completed without error.
    output:
        The primary text output of the tool (file contents, command stdout,
        HTTP body, etc.).
    error:
        Non-empty when ``success`` is ``False``; describes what went wrong.
    metadata:
        Optional extra key/value pairs (e.g. return code, HTTP status code,
        resolved path).
    """

    success: bool
    output: str
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # noqa: D105
        if self.success:
            return self.output
        return f"ERROR: {self.error}"


# ---------------------------------------------------------------------------
# file_read
# ---------------------------------------------------------------------------


def file_read(
    path: str | Path,
    sandbox: SandboxContext,
    encoding: str = "utf-8",
) -> ToolResult:
    """Read the text content of a file.

    The path is checked against the sandbox allow-list before any I/O is
    attempted.  If the sandbox blocks the operation a :class:`ToolResult` with
    ``success=False`` is returned (the violation is already recorded in the
    sandbox context).

    Parameters
    ----------
    path:
        Filesystem path of the file to read.  May be absolute or relative;
        relative paths are resolved against the sandbox's working directory.
    sandbox:
        Active :class:`~local_agent_runner.sandbox.SandboxContext`.
    encoding:
        Text encoding to use when reading the file.  Defaults to ``"utf-8"``.

    Returns
    -------
    ToolResult
        On success, ``output`` contains the full text of the file.
        On failure, ``success`` is ``False`` and ``error`` describes the
        problem.
    """
    try:
        resolved = sandbox.check_file_read(path)
    except SandboxViolation as exc:
        return ToolResult(
            success=False,
            output="",
            error=str(exc),
            metadata={"sandbox_blocked": True, "path": str(path)},
        )

    if not resolved.exists():
        return ToolResult(
            success=False,
            output="",
            error=f"File not found: {resolved}",
            metadata={"resolved_path": str(resolved)},
        )
    if not resolved.is_file():
        return ToolResult(
            success=False,
            output="",
            error=f"Path is not a file: {resolved}",
            metadata={"resolved_path": str(resolved)},
        )

    try:
        content = resolved.read_text(encoding=encoding)
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot read file {resolved}: {exc}",
            metadata={"resolved_path": str(resolved)},
        )
    except UnicodeDecodeError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot decode file {resolved} as {encoding}: {exc}",
            metadata={"resolved_path": str(resolved), "encoding": encoding},
        )

    return ToolResult(
        success=True,
        output=content,
        metadata={
            "resolved_path": str(resolved),
            "size_bytes": len(content.encode(encoding, errors="replace")),
        },
    )


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------


def file_write(
    path: str | Path,
    content: str,
    sandbox: SandboxContext,
    encoding: str = "utf-8",
    create_parents: bool = True,
) -> ToolResult:
    """Write text content to a file.

    Parent directories are created automatically when *create_parents* is
    ``True`` (the default).  The path is checked against the sandbox
    allow-list before any I/O is attempted.

    Parameters
    ----------
    path:
        Filesystem path of the file to write.  May be absolute or relative.
    content:
        Text to write to the file.  Existing content is overwritten.
    sandbox:
        Active :class:`~local_agent_runner.sandbox.SandboxContext`.
    encoding:
        Text encoding.  Defaults to ``"utf-8"``.
    create_parents:
        When ``True``, any missing parent directories are created via
        :func:`pathlib.Path.mkdir`.

    Returns
    -------
    ToolResult
        On success, ``output`` is a human-readable confirmation message.
        On failure, ``success`` is ``False`` and ``error`` describes the
        problem.
    """
    try:
        resolved = sandbox.check_file_write(path)
    except SandboxViolation as exc:
        return ToolResult(
            success=False,
            output="",
            error=str(exc),
            metadata={"sandbox_blocked": True, "path": str(path)},
        )

    try:
        if create_parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding=encoding)
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Cannot write file {resolved}: {exc}",
            metadata={"resolved_path": str(resolved)},
        )

    size_bytes = len(content.encode(encoding, errors="replace"))
    return ToolResult(
        success=True,
        output=f"Successfully wrote {size_bytes} bytes to {resolved}",
        metadata={
            "resolved_path": str(resolved),
            "size_bytes": size_bytes,
        },
    )


# ---------------------------------------------------------------------------
# shell_run
# ---------------------------------------------------------------------------


def shell_run(
    command: str,
    sandbox: SandboxContext,
    timeout: float = 30.0,
    cwd: Optional[str | Path] = None,
) -> ToolResult:
    """Execute a shell command and return its stdout and stderr.

    The command string is checked against the sandbox allow-list before
    execution.  Execution uses :func:`subprocess.run` with
    ``shell=True`` so that shell built-ins and pipelines work as expected.

    Parameters
    ----------
    command:
        The shell command to execute.
    sandbox:
        Active :class:`~local_agent_runner.sandbox.SandboxContext`.
    timeout:
        Maximum number of seconds to wait for the command to complete.
        Defaults to ``30.0``.
    cwd:
        Working directory for the subprocess.  Defaults to the process CWD.

    Returns
    -------
    ToolResult
        ``output`` contains the combined stdout and stderr text.
        ``metadata`` includes ``returncode``, ``stdout``, and ``stderr``.
        ``success`` is ``True`` when the return code is ``0``.
    """
    try:
        sandbox.check_shell_command(command)
    except SandboxViolation as exc:
        return ToolResult(
            success=False,
            output="",
            error=str(exc),
            metadata={"sandbox_blocked": True, "command": command},
        )

    effective_cwd = str(cwd) if cwd is not None else None

    try:
        proc = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=effective_cwd,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Command timed out after {timeout}s: {command!r}",
            metadata={"command": command, "timeout": timeout},
        )
    except OSError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"Failed to execute command {command!r}: {exc}",
            metadata={"command": command},
        )

    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += proc.stderr

    return ToolResult(
        success=proc.returncode == 0,
        output=combined,
        error="" if proc.returncode == 0 else f"Command exited with code {proc.returncode}",
        metadata={
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        },
    )


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def web_search(
    url: str,
    sandbox: SandboxContext,
    timeout: float = 30.0,
    max_response_bytes: int = 1_000_000,
    headers: Optional[Dict[str, str]] = None,
    follow_redirects: bool = True,
) -> ToolResult:
    """Fetch the content of a URL via HTTP GET.

    The URL's domain is checked against the sandbox allow-list before any
    network request is made.  The response body is returned as text; binary
    content is decoded with ``errors='replace'`` so it never raises.

    Parameters
    ----------
    url:
        Full URL to fetch (must include scheme, e.g. ``"https://..."``).
    sandbox:
        Active :class:`~local_agent_runner.sandbox.SandboxContext`.
    timeout:
        Maximum number of seconds to wait for the HTTP response.
        Defaults to ``30.0``.
    max_response_bytes:
        Maximum number of bytes to read from the response body.  Content
        beyond this limit is silently truncated.  Defaults to 1 MB.
    headers:
        Optional extra HTTP headers to include in the request.
    follow_redirects:
        When ``True`` (the default) HTTP redirects are followed.

    Returns
    -------
    ToolResult
        ``output`` contains the response body as text.
        ``metadata`` includes ``status_code``, ``url``, and
        ``content_type``.
    """
    try:
        sandbox.check_web_request(url)
    except SandboxViolation as exc:
        return ToolResult(
            success=False,
            output="",
            error=str(exc),
            metadata={"sandbox_blocked": True, "url": url},
        )

    request_headers: Dict[str, str] = {
        "User-Agent": "local-agent-runner/0.1.0 (httpx)",
    }
    if headers:
        request_headers.update(headers)

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=follow_redirects,
        ) as client:
            response = client.get(url, headers=request_headers)
    except httpx.TimeoutException:
        return ToolResult(
            success=False,
            output="",
            error=f"HTTP request timed out after {timeout}s: {url}",
            metadata={"url": url, "timeout": timeout},
        )
    except httpx.TooManyRedirects:
        return ToolResult(
            success=False,
            output="",
            error=f"Too many redirects fetching: {url}",
            metadata={"url": url},
        )
    except httpx.RequestError as exc:
        return ToolResult(
            success=False,
            output="",
            error=f"HTTP request failed for {url!r}: {exc}",
            metadata={"url": url, "exception_type": type(exc).__name__},
        )

    content_type = response.headers.get("content-type", "")
    raw_bytes = response.content[:max_response_bytes]

    # Attempt to decode as UTF-8; fall back to latin-1 which never fails
    try:
        body = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        body = raw_bytes.decode("latin-1", errors="replace")

    truncated = len(response.content) > max_response_bytes
    if truncated:
        body += f"\n\n[Response truncated at {max_response_bytes} bytes]"

    success = 200 <= response.status_code < 400
    error = "" if success else f"HTTP {response.status_code} for {url}"

    return ToolResult(
        success=success,
        output=body,
        error=error,
        metadata={
            "url": str(response.url),
            "status_code": response.status_code,
            "content_type": content_type,
            "response_bytes": len(response.content),
            "truncated": truncated,
        },
    )


# ---------------------------------------------------------------------------
# dispatch_tool
# ---------------------------------------------------------------------------


def dispatch_tool(
    tool_type: ToolType,
    arguments: Dict[str, Any],
    sandbox: SandboxContext,
    *,
    shell_timeout: float = 30.0,
    web_timeout: float = 30.0,
    max_response_bytes: int = 1_000_000,
) -> ToolResult:
    """Route a tool call to the appropriate tool function by *tool_type*.

    This is the primary entry point used by the agentic loop.  It unpacks
    the ``arguments`` dict and delegates to the corresponding tool function.

    Parameters
    ----------
    tool_type:
        The :class:`~local_agent_runner.config.ToolType` enum value that
        identifies which tool to invoke.
    arguments:
        Key/value arguments extracted from the LLM's tool-call response.
        Expected keys per tool:

        - ``file_read``  : ``path`` (str)
        - ``file_write`` : ``path`` (str), ``content`` (str)
        - ``shell``      : ``command`` (str)
        - ``web_search`` : ``url`` (str)
    sandbox:
        Active :class:`~local_agent_runner.sandbox.SandboxContext`.
    shell_timeout:
        Timeout in seconds for shell commands.  Defaults to ``30.0``.
    web_timeout:
        Timeout in seconds for HTTP requests.  Defaults to ``30.0``.
    max_response_bytes:
        Maximum response body size for web requests.  Defaults to 1 MB.

    Returns
    -------
    ToolResult
        The result from whichever tool function was invoked.

    Raises
    ------
    ToolError
        If *tool_type* is unknown or a required argument is missing from
        *arguments*.
    """
    if tool_type is ToolType.FILE_READ:
        path = _require_arg(arguments, "path", tool_type.value)
        return file_read(path=path, sandbox=sandbox)

    elif tool_type is ToolType.FILE_WRITE:
        path = _require_arg(arguments, "path", tool_type.value)
        content = _require_arg(arguments, "content", tool_type.value)
        return file_write(path=path, content=content, sandbox=sandbox)

    elif tool_type is ToolType.SHELL:
        command = _require_arg(arguments, "command", tool_type.value)
        return shell_run(command=command, sandbox=sandbox, timeout=shell_timeout)

    elif tool_type is ToolType.WEB_SEARCH:
        url = _require_arg(arguments, "url", tool_type.value)
        return web_search(
            url=url,
            sandbox=sandbox,
            timeout=web_timeout,
            max_response_bytes=max_response_bytes,
        )

    else:
        raise ToolError(
            "dispatch_tool",
            f"Unknown tool type: {tool_type!r}",
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_arg(
    arguments: Dict[str, Any],
    key: str,
    tool_name: str,
) -> str:
    """Extract a required string argument from *arguments*.

    Parameters
    ----------
    arguments:
        The arguments mapping from the LLM's tool-call response.
    key:
        The argument key to extract.
    tool_name:
        Name used in error messages.

    Returns
    -------
    str
        The argument value coerced to ``str``.

    Raises
    ------
    ToolError
        If *key* is absent from *arguments*.
    """
    if key not in arguments:
        raise ToolError(
            tool_name,
            f"Missing required argument {key!r} for tool {tool_name!r}. "
            f"Provided arguments: {list(arguments.keys())}",
        )
    return str(arguments[key])
