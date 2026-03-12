"""Structured JSON and Rich terminal logger for local_agent_runner.

This module provides a single :class:`RunLogger` that writes every significant
event — LLM prompt/response, tool invocations, step start/end, errors, and
sandbox actions — to:

1. A **Rich** terminal panel (colour-coded, human-readable).
2. An optional **JSON log file** on disk (one JSON object per line, i.e.
   newline-delimited JSON / NDJSON).

All log entries carry a monotonically increasing sequence number, a UTC
timestamp, and a structured ``payload`` dict so that they can be machine-
processed downstream.

Public API
----------
- ``EventType``    — enum of all loggable event kinds.
- ``LogEntry``     — immutable record of a single event.
- ``RunLogger``    — main logger object; write events and flush to disk/terminal.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Kinds of events that :class:`RunLogger` can record."""

    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SANDBOX_ACTION = "sandbox_action"
    ERROR = "error"
    INFO = "info"
    WARNING = "warning"
    DEBUG = "debug"


# ---------------------------------------------------------------------------
# Colour / style mapping for Rich terminal output
# ---------------------------------------------------------------------------

_STYLE_MAP: Dict[EventType, str] = {
    EventType.RUN_START: "bold green",
    EventType.RUN_END: "bold green",
    EventType.STEP_START: "bold cyan",
    EventType.STEP_END: "cyan",
    EventType.LLM_PROMPT: "yellow",
    EventType.LLM_RESPONSE: "white",
    EventType.TOOL_CALL: "bold magenta",
    EventType.TOOL_RESULT: "magenta",
    EventType.SANDBOX_ACTION: "bold red",
    EventType.ERROR: "bold red",
    EventType.INFO: "dim white",
    EventType.WARNING: "bold yellow",
    EventType.DEBUG: "dim",
}

_BORDER_MAP: Dict[EventType, str] = {
    EventType.RUN_START: "green",
    EventType.RUN_END: "green",
    EventType.STEP_START: "cyan",
    EventType.STEP_END: "cyan",
    EventType.LLM_PROMPT: "yellow",
    EventType.LLM_RESPONSE: "white",
    EventType.TOOL_CALL: "magenta",
    EventType.TOOL_RESULT: "magenta",
    EventType.SANDBOX_ACTION: "red",
    EventType.ERROR: "red",
    EventType.INFO: "dim",
    EventType.WARNING: "yellow",
    EventType.DEBUG: "dim",
}


# ---------------------------------------------------------------------------
# LogEntry dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEntry:
    """Immutable record of a single logged event.

    Attributes
    ----------
    seq:
        Monotonically increasing sequence number within a run.
    event_type:
        The :class:`EventType` of this entry.
    message:
        Short human-readable summary of the event.
    payload:
        Arbitrary structured data associated with the event.
    timestamp:
        UTC time at which the entry was created.
    run_id:
        Identifier of the run this entry belongs to.
    step_name:
        Name of the workflow step, if applicable.
    """

    seq: int
    event_type: EventType
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    run_id: str = ""
    step_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "seq": self.seq,
            "event_type": self.event_type.value,
            "message": self.message,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "step_name": self.step_name,
        }

    def to_json(self) -> str:
        """Serialise this entry to a compact JSON string (no trailing newline).

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


class RunLogger:
    """Structured logger that writes events to the terminal and optionally to
    a JSON log file.

    Usage example::

        logger = RunLogger(run_id="abc123", verbose=True, log_file="run.jsonl")
        logger.log_run_start(workflow_name="My Workflow", model="llama3")
        logger.log_step_start(step_name="step_one")
        logger.log_llm_prompt(step_name="step_one", prompt="Hello")
        logger.log_llm_response(step_name="step_one", response="Hi!")
        logger.log_step_end(step_name="step_one", output_variable="result")
        logger.log_run_end(success=True)
        logger.close()

    The logger is thread-safe: a :class:`threading.Lock` serialises all writes.

    Parameters
    ----------
    run_id:
        Unique identifier for this run (used in every log entry).
    verbose:
        When ``True``, ``DEBUG`` and ``INFO`` events are printed to the
        terminal in addition to being written to the JSON file.
        When ``False``, only ``WARNING``, ``ERROR``, and key lifecycle events
        are shown.
    log_file:
        Optional path to an NDJSON file.  The file is created (or appended to)
        on first write.  Pass ``None`` to disable file logging.
    console:
        Optional :class:`rich.console.Console` instance.  A default stderr
        console is created if not supplied.
    """

    def __init__(
        self,
        run_id: str = "",
        verbose: bool = False,
        log_file: Optional[str | Path] = None,
        console: Optional[Console] = None,
    ) -> None:
        self._run_id = run_id
        self._verbose = verbose
        self._log_file_path = Path(log_file) if log_file is not None else None
        self._console = console or Console(stderr=True, highlight=False)
        self._lock = threading.Lock()
        self._seq = 0
        self._entries: List[LogEntry] = []
        self._file_handle: Optional[TextIO] = None
        self._current_step: str = ""

        if self._log_file_path is not None:
            self._open_log_file()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """The run identifier supplied at construction time."""
        return self._run_id

    @property
    def entries(self) -> List[LogEntry]:
        """Ordered list of all :class:`LogEntry` objects recorded so far.

        Returns a copy to prevent external mutation.
        """
        with self._lock:
            return list(self._entries)

    @property
    def verbose(self) -> bool:
        """``True`` when verbose (debug-level) terminal output is enabled."""
        return self._verbose

    # ------------------------------------------------------------------
    # High-level lifecycle log methods
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        workflow_name: str,
        model: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log the start of a workflow run.

        Parameters
        ----------
        workflow_name:
            Human-readable name of the workflow being executed.
        model:
            Ollama model identifier used for this run.
        extra:
            Additional key/value pairs to include in the payload.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {
            "workflow_name": workflow_name,
            "model": model,
            "run_id": self._run_id,
        }
        if extra:
            payload.update(extra)
        return self._emit(
            EventType.RUN_START,
            f"Run started — workflow: {workflow_name!r}, model: {model!r}",
            payload=payload,
            always_show=True,
        )

    def log_run_end(
        self,
        success: bool,
        duration_seconds: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log the end of a workflow run.

        Parameters
        ----------
        success:
            Whether the run completed without errors.
        duration_seconds:
            Optional wall-clock duration of the run.
        extra:
            Additional key/value pairs to include in the payload.

        Returns
        -------
        LogEntry
        """
        status = "success" if success else "failure"
        payload: Dict[str, Any] = {"success": success, "status": status}
        if duration_seconds is not None:
            payload["duration_seconds"] = round(duration_seconds, 3)
        if extra:
            payload.update(extra)
        msg = f"Run ended — {status}"
        if duration_seconds is not None:
            msg += f" ({duration_seconds:.2f}s)"
        return self._emit(
            EventType.RUN_END,
            msg,
            payload=payload,
            always_show=True,
        )

    def log_step_start(
        self,
        step_name: str,
        step_index: int = 0,
        description: str = "",
    ) -> LogEntry:
        """Log the start of a workflow step.

        Parameters
        ----------
        step_name:
            Name of the step as declared in the workflow YAML.
        step_index:
            Zero-based position of the step in the workflow.
        description:
            Optional prose description of the step.

        Returns
        -------
        LogEntry
        """
        with self._lock:
            self._current_step = step_name
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "step_index": step_index,
        }
        if description:
            payload["description"] = description
        return self._emit(
            EventType.STEP_START,
            f"Step [{step_index + 1}] started: {step_name!r}",
            payload=payload,
            step_name=step_name,
            always_show=True,
        )

    def log_step_end(
        self,
        step_name: str,
        output_variable: Optional[str] = None,
        iterations_used: int = 0,
        success: bool = True,
    ) -> LogEntry:
        """Log the end of a workflow step.

        Parameters
        ----------
        step_name:
            Name of the step.
        output_variable:
            The variable name under which the step output was stored, if any.
        iterations_used:
            Number of tool-call / response iterations consumed.
        success:
            Whether the step completed successfully.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "success": success,
            "iterations_used": iterations_used,
        }
        if output_variable:
            payload["output_variable"] = output_variable
        status = "ok" if success else "failed"
        msg = f"Step ended: {step_name!r} — {status}"
        if iterations_used:
            msg += f" ({iterations_used} iteration(s))"
        return self._emit(
            EventType.STEP_END,
            msg,
            payload=payload,
            step_name=step_name,
            always_show=True,
        )

    def log_llm_prompt(
        self,
        step_name: str,
        prompt: str,
        model: str = "",
        iteration: int = 0,
    ) -> LogEntry:
        """Log an LLM prompt being sent.

        Parameters
        ----------
        step_name:
            Name of the current step.
        prompt:
            The full prompt text sent to the LLM.
        model:
            Model identifier (if known at call site).
        iteration:
            Current iteration number within the step.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "prompt_length": len(prompt),
            "iteration": iteration,
        }
        if model:
            payload["model"] = model
        # Truncate long prompts in the message but keep full text in payload
        preview = prompt[:200].replace("\n", " ")
        if len(prompt) > 200:
            preview += " …"
        payload["prompt_preview"] = preview
        payload["prompt"] = prompt
        return self._emit(
            EventType.LLM_PROMPT,
            f"LLM prompt ({len(prompt)} chars) — step: {step_name!r}, iter: {iteration}",
            payload=payload,
            step_name=step_name,
            always_show=self._verbose,
        )

    def log_llm_response(
        self,
        step_name: str,
        response: str,
        model: str = "",
        iteration: int = 0,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> LogEntry:
        """Log an LLM response received.

        Parameters
        ----------
        step_name:
            Name of the current step.
        response:
            The raw response text from the LLM.
        model:
            Model identifier.
        iteration:
            Current iteration number.
        tool_calls:
            Parsed tool-call objects extracted from the response, if any.

        Returns
        -------
        LogEntry
        """
        preview = response[:200].replace("\n", " ")
        if len(response) > 200:
            preview += " …"
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "response_length": len(response),
            "iteration": iteration,
            "response_preview": preview,
            "response": response,
        }
        if model:
            payload["model"] = model
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
            payload["tool_call_count"] = len(tool_calls)
        return self._emit(
            EventType.LLM_RESPONSE,
            f"LLM response ({len(response)} chars) — step: {step_name!r}, iter: {iteration}",
            payload=payload,
            step_name=step_name,
            always_show=self._verbose,
        )

    def log_tool_call(
        self,
        step_name: str,
        tool_name: str,
        tool_type: str,
        arguments: Dict[str, Any],
        iteration: int = 0,
    ) -> LogEntry:
        """Log a tool invocation.

        Parameters
        ----------
        step_name:
            Name of the current step.
        tool_name:
            Identifier of the tool being called.
        tool_type:
            Tool type string, e.g. ``"file_read"``.
        arguments:
            The arguments dict passed to the tool.
        iteration:
            Current iteration number.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "tool_name": tool_name,
            "tool_type": tool_type,
            "arguments": arguments,
            "iteration": iteration,
        }
        return self._emit(
            EventType.TOOL_CALL,
            f"Tool call: {tool_name!r} ({tool_type}) — step: {step_name!r}",
            payload=payload,
            step_name=step_name,
            always_show=True,
        )

    def log_tool_result(
        self,
        step_name: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        error_message: str = "",
        iteration: int = 0,
    ) -> LogEntry:
        """Log the result of a tool invocation.

        Parameters
        ----------
        step_name:
            Name of the current step.
        tool_name:
            Identifier of the tool.
        result:
            The return value of the tool (will be coerced to string for
            preview).
        success:
            Whether the tool completed without error.
        error_message:
            Error description if ``success`` is ``False``.
        iteration:
            Current iteration number.

        Returns
        -------
        LogEntry
        """
        result_str = str(result)
        preview = result_str[:200].replace("\n", " ")
        if len(result_str) > 200:
            preview += " …"
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "tool_name": tool_name,
            "success": success,
            "result_preview": preview,
            "result": result_str,
            "iteration": iteration,
        }
        if error_message:
            payload["error_message"] = error_message
        status = "ok" if success else "error"
        return self._emit(
            EventType.TOOL_RESULT,
            f"Tool result: {tool_name!r} — {status} — step: {step_name!r}",
            payload=payload,
            step_name=step_name,
            always_show=self._verbose,
        )

    def log_sandbox_action(
        self,
        step_name: str,
        action_type: str,
        target: str,
        permitted: bool,
        reason: str = "",
    ) -> LogEntry:
        """Log a sandbox allow/deny decision.

        Parameters
        ----------
        step_name:
            Name of the current step.
        action_type:
            Sandbox action type string, e.g. ``"file_read"``.
        target:
            The path, command, or URL that was checked.
        permitted:
            Whether the sandbox allowed the action.
        reason:
            Human-readable note from the sandbox.

        Returns
        -------
        LogEntry
        """
        verdict = "ALLOWED" if permitted else "BLOCKED"
        payload: Dict[str, Any] = {
            "step_name": step_name,
            "action_type": action_type,
            "target": target,
            "permitted": permitted,
            "verdict": verdict,
        }
        if reason:
            payload["reason"] = reason
        return self._emit(
            EventType.SANDBOX_ACTION,
            f"Sandbox {verdict}: {action_type} → {target!r}",
            payload=payload,
            step_name=step_name,
            always_show=not permitted,  # always show denials
        )

    def log_error(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        step_name: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log an error event.

        Parameters
        ----------
        message:
            Short description of the error.
        exception:
            Optional exception object; its type and str representation are
            captured in the payload.
        step_name:
            Name of the step during which the error occurred, if applicable.
        extra:
            Additional key/value pairs.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {"message": message}
        if exception is not None:
            payload["exception_type"] = type(exception).__name__
            payload["exception_message"] = str(exception)
        if step_name:
            payload["step_name"] = step_name
        if extra:
            payload.update(extra)
        return self._emit(
            EventType.ERROR,
            message,
            payload=payload,
            step_name=step_name or self._current_step,
            always_show=True,
        )

    def log_warning(
        self,
        message: str,
        step_name: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log a warning event.

        Parameters
        ----------
        message:
            Warning description.
        step_name:
            Name of the current step, if applicable.
        extra:
            Additional key/value pairs.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {"message": message}
        if step_name:
            payload["step_name"] = step_name
        if extra:
            payload.update(extra)
        return self._emit(
            EventType.WARNING,
            message,
            payload=payload,
            step_name=step_name or self._current_step,
            always_show=True,
        )

    def log_info(
        self,
        message: str,
        step_name: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log an informational event.

        Parameters
        ----------
        message:
            Informational message.
        step_name:
            Name of the current step, if applicable.
        extra:
            Additional key/value pairs.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {"message": message}
        if step_name:
            payload["step_name"] = step_name
        if extra:
            payload.update(extra)
        return self._emit(
            EventType.INFO,
            message,
            payload=payload,
            step_name=step_name or self._current_step,
            always_show=self._verbose,
        )

    def log_debug(
        self,
        message: str,
        step_name: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Log a debug event (only printed to terminal when verbose=True).

        Parameters
        ----------
        message:
            Debug message.
        step_name:
            Name of the current step, if applicable.
        extra:
            Additional key/value pairs.

        Returns
        -------
        LogEntry
        """
        payload: Dict[str, Any] = {"message": message}
        if step_name:
            payload["step_name"] = step_name
        if extra:
            payload.update(extra)
        return self._emit(
            EventType.DEBUG,
            message,
            payload=payload,
            step_name=step_name or self._current_step,
            always_show=self._verbose,
        )

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the JSON log file, if one was opened.

        It is safe to call this method multiple times.
        """
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.flush()
                    self._file_handle.close()
                except OSError:
                    pass
                finally:
                    self._file_handle = None

    def flush(self) -> None:
        """Flush the JSON log file without closing it."""
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.flush()
                except OSError:
                    pass

    def get_log_path(self) -> Optional[Path]:
        """Return the path to the JSON log file, or ``None`` if not logging to file.

        Returns
        -------
        Optional[Path]
        """
        return self._log_file_path

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_: Any) -> None:  # noqa: ANN002
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        """Return the next sequence number (must be called under lock)."""
        self._seq += 1
        return self._seq

    def _emit(
        self,
        event_type: EventType,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        step_name: str = "",
        always_show: bool = False,
    ) -> LogEntry:
        """Create a :class:`LogEntry`, store it, write to file, and optionally
        print it to the terminal.

        Parameters
        ----------
        event_type:
            The event kind.
        message:
            Short summary message.
        payload:
            Structured data dict.
        step_name:
            Step context for this entry.
        always_show:
            When ``True`` the entry is printed to the terminal regardless of
            the ``verbose`` flag.

        Returns
        -------
        LogEntry
        """
        with self._lock:
            seq = self._next_seq()
            entry = LogEntry(
                seq=seq,
                event_type=event_type,
                message=message,
                payload=payload or {},
                run_id=self._run_id,
                step_name=step_name,
            )
            self._entries.append(entry)
            self._write_to_file(entry)

        if always_show or self._verbose:
            self._print_entry(entry)

        return entry

    def _open_log_file(self) -> None:
        """Open (or create) the JSON log file for appending."""
        assert self._log_file_path is not None
        try:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(  # noqa: WPS515
                self._log_file_path, "a", encoding="utf-8", buffering=1
            )
        except OSError as exc:
            # Non-fatal: warn via Rich but continue without file logging
            self._console.print(
                f"[bold yellow]Warning:[/bold yellow] cannot open log file "
                f"{self._log_file_path}: {exc}"
            )
            self._log_file_path = None
            self._file_handle = None

    def _write_to_file(self, entry: LogEntry) -> None:
        """Write *entry* as a single JSON line to the log file.

        Must be called while holding :attr:`_lock`.
        """
        if self._file_handle is None:
            return
        try:
            self._file_handle.write(entry.to_json() + "\n")
        except OSError:
            pass

    def _print_entry(self, entry: LogEntry) -> None:
        """Render *entry* as a Rich panel to the terminal."""
        style = _STYLE_MAP.get(entry.event_type, "white")
        border_style = _BORDER_MAP.get(entry.event_type, "white")
        title_parts = [f"[{style}]{entry.event_type.value.upper()}[/{style}]"]
        if entry.step_name:
            title_parts.append(f"[dim]step: {escape(entry.step_name)}[/dim]")
        title_parts.append(f"[dim]#{entry.seq}[/dim]")
        title = "  ".join(title_parts)

        # Build body text
        body = Text()
        body.append(entry.message, style=style)

        # For key event types, append a few important payload fields
        if entry.event_type in (
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.SANDBOX_ACTION,
            EventType.ERROR,
            EventType.WARNING,
        ):
            extras = _format_payload_extras(entry)
            if extras:
                body.append("\n" + extras, style="dim")

        panel = Panel(
            body,
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )
        self._console.print(panel)


# ---------------------------------------------------------------------------
# Private formatting helpers
# ---------------------------------------------------------------------------


def _format_payload_extras(entry: LogEntry) -> str:
    """Return a short multi-line string of notable payload fields for display.

    Parameters
    ----------
    entry:
        The log entry whose payload to format.

    Returns
    -------
    str
        Formatted string, possibly empty.
    """
    lines: List[str] = []
    p = entry.payload

    if entry.event_type == EventType.TOOL_CALL:
        if "tool_type" in p:
            lines.append(f"type: {p['tool_type']}")
        if "arguments" in p:
            try:
                args_str = json.dumps(p["arguments"], ensure_ascii=False)
                if len(args_str) > 120:
                    args_str = args_str[:120] + " …"
                lines.append(f"args: {args_str}")
            except (TypeError, ValueError):
                pass

    elif entry.event_type == EventType.TOOL_RESULT:
        if "result_preview" in p:
            lines.append(f"result: {p['result_preview']}")
        if not p.get("success") and "error_message" in p:
            lines.append(f"error: {p['error_message']}")

    elif entry.event_type == EventType.SANDBOX_ACTION:
        if "verdict" in p:
            lines.append(f"verdict: {p['verdict']}")
        if "reason" in p:
            lines.append(f"reason: {p['reason']}")

    elif entry.event_type in (EventType.ERROR, EventType.WARNING):
        if "exception_type" in p:
            lines.append(f"exception: {p['exception_type']}: {p.get('exception_message', '')}")

    return "\n".join(lines)
