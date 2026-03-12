"""Unit tests for local_agent_runner.logger.

Covers:
- EventType enum values.
- LogEntry.to_dict and to_json serialisation.
- RunLogger construction (no file, with file).
- All log_* methods produce correct EventType and payload keys.
- JSON log file output (NDJSON format, one entry per line).
- context-manager protocol (close called on exit).
- entries property returns a copy.
- verbose flag controls terminal output (no exception raised).
- close / flush lifecycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_agent_runner.logger import EventType, LogEntry, RunLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def logger() -> RunLogger:
    """A RunLogger that writes nothing to disk and nothing to terminal."""
    from rich.console import Console
    import io

    console = Console(file=io.StringIO(), highlight=False)
    return RunLogger(run_id="test-run", verbose=False, log_file=None, console=console)


@pytest.fixture()
def verbose_logger() -> RunLogger:
    """A verbose RunLogger that suppresses terminal to StringIO."""
    from rich.console import Console
    import io

    console = Console(file=io.StringIO(), highlight=False)
    return RunLogger(run_id="verbose-run", verbose=True, log_file=None, console=console)


# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class TestEventType:
    def test_all_values(self) -> None:
        expected = {
            "run_start", "run_end", "step_start", "step_end",
            "llm_prompt", "llm_response", "tool_call", "tool_result",
            "sandbox_action", "error", "info", "warning", "debug",
        }
        actual = {e.value for e in EventType}
        assert actual == expected


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------


class TestLogEntry:
    def test_to_dict_keys(self) -> None:
        entry = LogEntry(
            seq=1,
            event_type=EventType.INFO,
            message="hello",
            run_id="run1",
            step_name="step",
        )
        d = entry.to_dict()
        assert set(d.keys()) >= {"seq", "event_type", "message", "payload",
                                  "timestamp", "run_id", "step_name"}

    def test_to_dict_event_type_is_string(self) -> None:
        entry = LogEntry(seq=1, event_type=EventType.ERROR, message="oops")
        assert entry.to_dict()["event_type"] == "error"

    def test_to_json_is_valid_json(self) -> None:
        entry = LogEntry(seq=2, event_type=EventType.DEBUG, message="dbg")
        parsed = json.loads(entry.to_json())
        assert parsed["seq"] == 2
        assert parsed["event_type"] == "debug"

    def test_frozen(self) -> None:
        entry = LogEntry(seq=1, event_type=EventType.INFO, message="hi")
        with pytest.raises((AttributeError, TypeError)):
            entry.message = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RunLogger construction
# ---------------------------------------------------------------------------


class TestRunLoggerConstruction:
    def test_default_run_id(self) -> None:
        import io
        from rich.console import Console
        console = Console(file=io.StringIO())
        lg = RunLogger(console=console)
        assert lg.run_id == ""
        lg.close()

    def test_run_id_stored(self, logger: RunLogger) -> None:
        assert logger.run_id == "test-run"

    def test_entries_empty_initially(self, logger: RunLogger) -> None:
        assert logger.entries == []

    def test_no_log_file(self, logger: RunLogger) -> None:
        assert logger.get_log_path() is None

    def test_log_file_created(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        log_path = tmp_path / "run.jsonl"
        console = Console(file=io.StringIO())
        lg = RunLogger(log_file=log_path, console=console)
        lg.log_info("test message")
        lg.close()
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "info"


# ---------------------------------------------------------------------------
# Log methods
# ---------------------------------------------------------------------------


class TestRunLoggerMethods:
    def test_log_run_start(self, logger: RunLogger) -> None:
        entry = logger.log_run_start(workflow_name="My WF", model="llama3")
        assert entry.event_type == EventType.RUN_START
        assert entry.payload["workflow_name"] == "My WF"
        assert entry.payload["model"] == "llama3"
        assert entry.run_id == "test-run"

    def test_log_run_end_success(self, logger: RunLogger) -> None:
        entry = logger.log_run_end(success=True, duration_seconds=1.23)
        assert entry.event_type == EventType.RUN_END
        assert entry.payload["success"] is True
        assert entry.payload["duration_seconds"] == pytest.approx(1.23, rel=1e-3)

    def test_log_run_end_failure(self, logger: RunLogger) -> None:
        entry = logger.log_run_end(success=False)
        assert entry.payload["status"] == "failure"

    def test_log_step_start(self, logger: RunLogger) -> None:
        entry = logger.log_step_start(step_name="my_step", step_index=0)
        assert entry.event_type == EventType.STEP_START
        assert entry.payload["step_name"] == "my_step"
        assert entry.step_name == "my_step"

    def test_log_step_end(self, logger: RunLogger) -> None:
        entry = logger.log_step_end(
            step_name="my_step",
            output_variable="result",
            iterations_used=3,
            success=True,
        )
        assert entry.event_type == EventType.STEP_END
        assert entry.payload["output_variable"] == "result"
        assert entry.payload["iterations_used"] == 3

    def test_log_llm_prompt(self, logger: RunLogger) -> None:
        entry = logger.log_llm_prompt(
            step_name="s", prompt="Say hello", model="llama3", iteration=1
        )
        assert entry.event_type == EventType.LLM_PROMPT
        assert entry.payload["prompt"] == "Say hello"
        assert entry.payload["model"] == "llama3"
        assert entry.payload["iteration"] == 1

    def test_log_llm_prompt_long_text_preview_truncated(self, logger: RunLogger) -> None:
        long_prompt = "X" * 500
        entry = logger.log_llm_prompt(step_name="s", prompt=long_prompt)
        preview = entry.payload["prompt_preview"]
        assert len(preview) <= 210  # 200 chars + " …"
        assert "…" in preview

    def test_log_llm_response(self, logger: RunLogger) -> None:
        entry = logger.log_llm_response(
            step_name="s",
            response="Hello there!",
            tool_calls=[{"tool": "read_file", "args": {}}],
        )
        assert entry.event_type == EventType.LLM_RESPONSE
        assert entry.payload["response"] == "Hello there!"
        assert entry.payload["tool_call_count"] == 1

    def test_log_tool_call(self, logger: RunLogger) -> None:
        entry = logger.log_tool_call(
            step_name="s",
            tool_name="read_file",
            tool_type="file_read",
            arguments={"path": "/tmp/x"},
            iteration=2,
        )
        assert entry.event_type == EventType.TOOL_CALL
        assert entry.payload["tool_name"] == "read_file"
        assert entry.payload["arguments"] == {"path": "/tmp/x"}

    def test_log_tool_result_success(self, logger: RunLogger) -> None:
        entry = logger.log_tool_result(
            step_name="s",
            tool_name="read_file",
            result="file contents here",
            success=True,
        )
        assert entry.event_type == EventType.TOOL_RESULT
        assert entry.payload["success"] is True
        assert "file contents" in entry.payload["result"]

    def test_log_tool_result_failure(self, logger: RunLogger) -> None:
        entry = logger.log_tool_result(
            step_name="s",
            tool_name="read_file",
            result="",
            success=False,
            error_message="File not found",
        )
        assert entry.payload["success"] is False
        assert entry.payload["error_message"] == "File not found"

    def test_log_sandbox_action_allowed(self, logger: RunLogger) -> None:
        entry = logger.log_sandbox_action(
            step_name="s",
            action_type="file_read",
            target="/tmp/ok.txt",
            permitted=True,
            reason="in allowed_paths",
        )
        assert entry.event_type == EventType.SANDBOX_ACTION
        assert entry.payload["permitted"] is True
        assert entry.payload["verdict"] == "ALLOWED"

    def test_log_sandbox_action_denied(self, logger: RunLogger) -> None:
        entry = logger.log_sandbox_action(
            step_name="s",
            action_type="shell",
            target="rm -rf /",
            permitted=False,
            reason="not in allowed_commands",
        )
        assert entry.payload["verdict"] == "BLOCKED"
        assert entry.payload["permitted"] is False

    def test_log_error_with_exception(self, logger: RunLogger) -> None:
        exc = ValueError("bad value")
        entry = logger.log_error(
            message="Something failed",
            exception=exc,
            step_name="step1",
        )
        assert entry.event_type == EventType.ERROR
        assert entry.payload["exception_type"] == "ValueError"
        assert entry.payload["exception_message"] == "bad value"

    def test_log_error_without_exception(self, logger: RunLogger) -> None:
        entry = logger.log_error(message="Plain error")
        assert entry.event_type == EventType.ERROR
        assert "exception_type" not in entry.payload

    def test_log_warning(self, logger: RunLogger) -> None:
        entry = logger.log_warning(message="careful", step_name="s")
        assert entry.event_type == EventType.WARNING
        assert entry.payload["message"] == "careful"

    def test_log_info(self, logger: RunLogger) -> None:
        entry = logger.log_info(message="all good")
        assert entry.event_type == EventType.INFO

    def test_log_debug(self, logger: RunLogger) -> None:
        entry = logger.log_debug(message="internal state")
        assert entry.event_type == EventType.DEBUG


# ---------------------------------------------------------------------------
# entries accumulation
# ---------------------------------------------------------------------------


class TestRunLoggerEntries:
    def test_entries_accumulate(self, logger: RunLogger) -> None:
        logger.log_info("a")
        logger.log_info("b")
        logger.log_warning("w")
        assert len(logger.entries) == 3

    def test_entries_returns_copy(self, logger: RunLogger) -> None:
        logger.log_info("a")
        e1 = logger.entries
        e2 = logger.entries
        assert e1 == e2
        assert e1 is not e2

    def test_seq_monotonically_increases(self, logger: RunLogger) -> None:
        for _ in range(5):
            logger.log_debug("x")
        seqs = [e.seq for e in logger.entries]
        assert seqs == sorted(seqs)
        assert seqs == list(range(1, 6))


# ---------------------------------------------------------------------------
# JSON file output
# ---------------------------------------------------------------------------


class TestRunLoggerFileOutput:
    def test_ndjson_format(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        log_path = tmp_path / "test.jsonl"
        console = Console(file=io.StringIO())
        with RunLogger(log_file=log_path, console=console) as lg:
            lg.log_run_start(workflow_name="WF", model="llama3")
            lg.log_step_start(step_name="step1")
            lg.log_run_end(success=True)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "event_type" in obj
            assert "seq" in obj
            assert "timestamp" in obj

    def test_seq_in_file_is_correct(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        log_path = tmp_path / "seq.jsonl"
        console = Console(file=io.StringIO())
        with RunLogger(log_file=log_path, console=console) as lg:
            lg.log_info("one")
            lg.log_info("two")

        lines = log_path.read_text().strip().splitlines()
        seq_values = [json.loads(l)["seq"] for l in lines]
        assert seq_values == [1, 2]

    def test_log_file_appends_on_reopen(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        log_path = tmp_path / "append.jsonl"
        for i in range(2):
            console = Console(file=io.StringIO())
            with RunLogger(log_file=log_path, console=console) as lg:
                lg.log_info(f"message {i}")

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestRunLoggerContextManager:
    def test_close_called_on_exit(self, tmp_path: Path) -> None:
        import io
        from rich.console import Console
        log_path = tmp_path / "cm.jsonl"
        console = Console(file=io.StringIO())
        with RunLogger(log_file=log_path, console=console) as lg:
            lg.log_info("inside")
        # After exit the file handle should be closed; calling close again is safe
        lg.close()  # should not raise

    def test_flush_does_not_raise_when_no_file(self, logger: RunLogger) -> None:
        logger.flush()  # should not raise


# ---------------------------------------------------------------------------
# verbose flag
# ---------------------------------------------------------------------------


class TestRunLoggerVerbose:
    def test_verbose_flag_stored(self, verbose_logger: RunLogger) -> None:
        assert verbose_logger.verbose is True

    def test_non_verbose_flag_stored(self, logger: RunLogger) -> None:
        assert logger.verbose is False

    def test_debug_event_still_recorded_when_not_verbose(
        self, logger: RunLogger
    ) -> None:
        entry = logger.log_debug("hidden debug")
        assert entry in logger.entries
