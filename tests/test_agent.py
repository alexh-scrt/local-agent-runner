"""Integration tests for the agentic loop in local_agent_runner.agent.

Covers:
- OllamaClient.chat success and error paths (mocked httpx).
- OllamaClient.health_check.
- parse_tool_calls: JSON code block format, bare JSON, multiple calls,
  unknown tool names filtered out, empty response.
- StepRunner: dry-run mode, successful tool call round-trip, max_iterations
  exceeded, prompt template substitution error.
- AgentRunner.run: all steps succeed, step failure stops execution.
- run_workflow convenience function with mocked Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from local_agent_runner.agent import (
    AgentError,
    AgentRunner,
    OllamaClient,
    OllamaMessage,
    ParsedToolCall,
    StepRunner,
    _build_ollama_tools,
    _build_system_prompt,
    _parse_ollama_native_tool_calls,
    parse_tool_calls,
    run_workflow,
)
from local_agent_runner.config import (
    SandboxConfig,
    StepConfig,
    ToolDefinition,
    ToolType,
    WorkflowConfig,
    load_workflow_from_string,
)
from local_agent_runner.logger import RunLogger

import io
from rich.console import Console


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _silent_logger(run_id: str = "test") -> RunLogger:
    """Return a RunLogger that suppresses all terminal output."""
    console = Console(file=io.StringIO(), highlight=False)
    return RunLogger(run_id=run_id, verbose=False, log_file=None, console=console)


def _minimal_workflow(
    tool_type: ToolType = ToolType.FILE_READ,
    max_iterations: int = 3,
) -> WorkflowConfig:
    """Build a minimal single-step WorkflowConfig for testing."""
    return load_workflow_from_string(
        f"""\
name: Test Workflow
model: llama3
tools:
  - name: read_file
    type: file_read
    description: Read a file.
  - name: write_file
    type: file_write
    description: Write a file.
steps:
  - name: step_one
    prompt: "Hello {{name}}"
    tools:
      - read_file
      - write_file
    output_variable: result
    max_iterations: {max_iterations}
variables:
  name: world
"""
    )


def _mock_ollama_response(
    content: str,
    tool_calls: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a fake Ollama /api/chat response dict."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "llama3", "message": message, "done": True}


def _make_ollama_client_mock(return_values: List[Dict[str, Any]]) -> OllamaClient:
    """Return an OllamaClient whose .chat() returns values from the list in order."""
    client = MagicMock(spec=OllamaClient)
    client.chat = MagicMock(side_effect=return_values)
    return client


# ---------------------------------------------------------------------------
# OllamaClient tests
# ---------------------------------------------------------------------------


class TestOllamaClient:
    def test_chat_success(self) -> None:
        """chat() should return parsed JSON on 200 response."""
        fake_response_data = _mock_ollama_response("Hello!")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json = MagicMock(return_value=fake_response_data)

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.post = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient(base_url="http://localhost:11434")
            result = client.chat(
                model="llama3",
                messages=[OllamaMessage(role="user", content="hi")],
            )

        assert result["message"]["content"] == "Hello!"

    def test_chat_non_200_raises_agent_error(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.post = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient()
            with pytest.raises(AgentError, match="500"):
                client.chat(model="llama3", messages=[])

    def test_chat_timeout_raises_agent_error(self) -> None:
        import httpx as _httpx

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.post = MagicMock(
            side_effect=_httpx.TimeoutException("timeout")
        )

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient()
            with pytest.raises(AgentError, match="timed out"):
                client.chat(model="llama3", messages=[])

    def test_chat_connection_error_raises_agent_error(self) -> None:
        import httpx as _httpx

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.post = MagicMock(
            side_effect=_httpx.ConnectError("refused")
        )

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient()
            with pytest.raises(AgentError, match="connect"):
                client.chat(model="llama3", messages=[])

    def test_health_check_true_on_200(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get = MagicMock(return_value=mock_response)

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient()
            assert client.health_check() is True

    def test_health_check_false_on_exception(self) -> None:
        import httpx as _httpx

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get = MagicMock(
            side_effect=_httpx.ConnectError("refused")
        )

        with patch("local_agent_runner.agent.httpx.Client", return_value=mock_http_client):
            client = OllamaClient()
            assert client.health_check() is False


# ---------------------------------------------------------------------------
# OllamaMessage
# ---------------------------------------------------------------------------


class TestOllamaMessage:
    def test_to_dict(self) -> None:
        msg = OllamaMessage(role="user", content="hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_no_tools_returns_generic_prompt(self) -> None:
        prompt = _build_system_prompt([])
        assert "helpful assistant" in prompt.lower()

    def test_tools_listed_in_prompt(self) -> None:
        td = ToolDefinition(
            name="read_file",
            tool_type=ToolType.FILE_READ,
            description="Read a file.",
        )
        prompt = _build_system_prompt([td])
        assert "read_file" in prompt
        assert "file_read" in prompt

    def test_json_instruction_present(self) -> None:
        td = ToolDefinition(
            name="my_tool", tool_type=ToolType.SHELL, description=""
        )
        prompt = _build_system_prompt([td])
        assert "```json" in prompt
        assert '"tool"' in prompt


# ---------------------------------------------------------------------------
# _build_ollama_tools
# ---------------------------------------------------------------------------


class TestBuildOllamaTools:
    def test_file_read_schema(self) -> None:
        td = ToolDefinition(
            name="r", tool_type=ToolType.FILE_READ, description=""
        )
        tools = _build_ollama_tools([td])
        assert len(tools) == 1
        fn = tools[0]["function"]
        assert fn["name"] == "r"
        assert "path" in fn["parameters"]["properties"]

    def test_file_write_schema(self) -> None:
        td = ToolDefinition(
            name="w", tool_type=ToolType.FILE_WRITE, description=""
        )
        tools = _build_ollama_tools([td])
        fn = tools[0]["function"]
        assert "path" in fn["parameters"]["properties"]
        assert "content" in fn["parameters"]["properties"]

    def test_shell_schema(self) -> None:
        td = ToolDefinition(
            name="sh", tool_type=ToolType.SHELL, description=""
        )
        tools = _build_ollama_tools([td])
        fn = tools[0]["function"]
        assert "command" in fn["parameters"]["properties"]

    def test_web_search_schema(self) -> None:
        td = ToolDefinition(
            name="ws", tool_type=ToolType.WEB_SEARCH, description=""
        )
        tools = _build_ollama_tools([td])
        fn = tools[0]["function"]
        assert "url" in fn["parameters"]["properties"]

    def test_multiple_tools(self) -> None:
        tds = [
            ToolDefinition("r", ToolType.FILE_READ, ""),
            ToolDefinition("w", ToolType.FILE_WRITE, ""),
        ]
        tools = _build_ollama_tools(tds)
        assert len(tools) == 2


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_json_code_block_parsed(self) -> None:
        text = '```json\n{"tool": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        calls = parse_tool_calls(text, ["read_file"])
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"
        assert calls[0].arguments == {"path": "/tmp/x"}

    def test_unknown_tool_filtered_out(self) -> None:
        text = '```json\n{"tool": "unknown_tool", "arguments": {}}\n```'
        calls = parse_tool_calls(text, ["read_file"])
        assert calls == []

    def test_empty_text_returns_empty(self) -> None:
        calls = parse_tool_calls("", ["read_file"])
        assert calls == []

    def test_plain_text_no_json_returns_empty(self) -> None:
        calls = parse_tool_calls("Here is my final answer.", ["read_file"])
        assert calls == []

    def test_bare_json_object_parsed(self) -> None:
        text = 'I will call {"tool": "read_file", "arguments": {"path": "/tmp"}}'
        calls = parse_tool_calls(text, ["read_file"])
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"

    def test_multiple_code_blocks(self) -> None:
        text = (
            '```json\n{"tool": "read_file", "arguments": {"path": "/a"}}\n```\n'
            '```json\n{"tool": "write_file", "arguments": {"path": "/b", "content": "hi"}}\n```'
        )
        calls = parse_tool_calls(text, ["read_file", "write_file"])
        assert len(calls) == 2
        assert calls[0].tool_name == "read_file"
        assert calls[1].tool_name == "write_file"

    def test_invalid_json_in_code_block_ignored(self) -> None:
        text = "```json\n{not valid json}\n```"
        calls = parse_tool_calls(text, ["read_file"])
        assert calls == []

    def test_arguments_defaults_to_empty_dict(self) -> None:
        text = '```json\n{"tool": "read_file"}\n```'
        calls = parse_tool_calls(text, ["read_file"])
        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_available_tools_empty_blocks_all(self) -> None:
        text = '```json\n{"tool": "read_file", "arguments": {}}\n```'
        calls = parse_tool_calls(text, [])
        assert calls == []

    def test_code_block_without_json_label(self) -> None:
        text = '```\n{"tool": "read_file", "arguments": {"path": "/x"}}\n```'
        calls = parse_tool_calls(text, ["read_file"])
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _parse_ollama_native_tool_calls
# ---------------------------------------------------------------------------


class TestParseOllamaNativeToolCalls:
    def test_native_tool_call_parsed(self) -> None:
        response_data = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "/tmp/x"},
                        }
                    }
                ],
            }
        }
        calls = _parse_ollama_native_tool_calls(response_data, ["read_file"])
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"
        assert calls[0].arguments == {"path": "/tmp/x"}

    def test_unknown_tool_filtered(self) -> None:
        response_data = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "bad_tool", "arguments": {}}}
                ]
            }
        }
        calls = _parse_ollama_native_tool_calls(response_data, ["read_file"])
        assert calls == []

    def test_no_tool_calls_key(self) -> None:
        response_data = {"message": {"role": "assistant", "content": "hello"}}
        calls = _parse_ollama_native_tool_calls(response_data, ["read_file"])
        assert calls == []

    def test_string_arguments_decoded(self) -> None:
        response_data = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/x"}),
                        }
                    }
                ]
            }
        }
        calls = _parse_ollama_native_tool_calls(response_data, ["read_file"])
        assert calls[0].arguments == {"path": "/x"}


# ---------------------------------------------------------------------------
# StepRunner
# ---------------------------------------------------------------------------


class TestStepRunner:
    def test_dry_run_returns_placeholder(self, tmp_path: Path) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
            dry_run=True,
        )
        result = runner.run()
        assert "dry-run" in result.lower() or "skipped" in result.lower()
        client.chat.assert_not_called()
        logger.close()

    def test_dry_run_sets_output_variable(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
            dry_run=True,
        )
        runner.run()
        assert "result" in variables
        logger.close()

    def test_final_answer_no_tool_calls(self) -> None:
        """LLM returns plain text → returned as final response."""
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = _make_ollama_client_mock(
            [_mock_ollama_response("This is the final answer.")]
        )
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
        )
        result = runner.run()
        assert result == "This is the final answer."
        assert variables["result"] == "This is the final answer."
        logger.close()

    def test_tool_call_then_final_answer(self, tmp_path: Path) -> None:
        """LLM calls a tool, then returns a final answer."""
        # Create a real file for the tool to read
        test_file = tmp_path / "data.txt"
        test_file.write_text("file contents here", encoding="utf-8")

        workflow = _minimal_workflow()
        logger = _silent_logger()

        # First response: request a tool call
        tool_call_response = _mock_ollama_response(
            f'```json\n{{"tool": "read_file", "arguments": {{"path": "{test_file}"}}}}\n```'
        )
        # Second response: final answer
        final_response = _mock_ollama_response("Done reading the file.")

        client = _make_ollama_client_mock([tool_call_response, final_response])
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
        )
        result = runner.run()
        assert result == "Done reading the file."
        assert client.chat.call_count == 2
        logger.close()

    def test_max_iterations_exceeded_uses_last_response(self) -> None:
        """When max_iterations is exhausted, the last response is used."""
        workflow = _minimal_workflow(max_iterations=2)
        logger = _silent_logger()

        # Always return a tool call (never a final answer)
        tool_call_response = _mock_ollama_response(
            '{"tool": "read_file", "arguments": {"path": "/tmp/x"}}'
        )
        client = _make_ollama_client_mock(
            [tool_call_response, tool_call_response]
        )
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
        )
        # Should not raise; should return the last response
        result = runner.run()
        assert isinstance(result, str)
        logger.close()

    def test_prompt_template_missing_variable_raises(self) -> None:
        """Missing template variable raises AgentError."""
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)
        variables: Dict[str, Any] = {}  # 'name' is missing

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
        )
        with pytest.raises(AgentError, match="name"):
            runner.run()
        logger.close()

    def test_ollama_error_propagates(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)
        client.chat = MagicMock(side_effect=AgentError("connection refused"))
        variables: Dict[str, Any] = {"name": "world"}

        runner = StepRunner(
            workflow=workflow,
            step=workflow.steps[0],
            client=client,
            logger=logger,
            variables=variables,
        )
        with pytest.raises(AgentError, match="connection refused"):
            runner.run()
        logger.close()


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class TestAgentRunner:
    def test_single_step_success(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = _make_ollama_client_mock(
            [_mock_ollama_response("Final output.")]
        )

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
        )
        result = runner.run()
        assert result["success"] is True
        assert result["steps_completed"] == 1
        assert result["error"] == ""
        logger.close()

    def test_variables_populated_after_run(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = _make_ollama_client_mock(
            [_mock_ollama_response("The answer.")]
        )

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
        )
        result = runner.run()
        assert result["variables"]["result"] == "The answer."
        logger.close()

    def test_step_failure_stops_execution(self) -> None:
        """When a step raises AgentError, remaining steps should not run."""
        yaml_text = """\
name: Two Step
model: llama3
tools:
  - name: read_file
    type: file_read
steps:
  - name: step_one
    prompt: First step
    tools: [read_file]
  - name: step_two
    prompt: Second step
    tools: [read_file]
"""
        workflow = load_workflow_from_string(yaml_text)
        logger = _silent_logger()

        # First call raises an error
        client = MagicMock(spec=OllamaClient)
        client.chat = MagicMock(side_effect=AgentError("ollama down"))

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
        )
        result = runner.run()
        assert result["success"] is False
        assert result["steps_completed"] == 0
        assert "ollama down" in result["error"]
        logger.close()

    def test_dry_run_all_steps_skipped(self) -> None:
        yaml_text = """\
name: Multi Step
model: llama3
tools:
  - name: read_file
    type: file_read
steps:
  - name: s1
    prompt: Step 1
    tools: [read_file]
  - name: s2
    prompt: Step 2
    tools: [read_file]
"""
        workflow = load_workflow_from_string(yaml_text)
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
            dry_run=True,
        )
        result = runner.run()
        assert result["success"] is True
        assert result["steps_completed"] == 2
        client.chat.assert_not_called()
        logger.close()

    def test_variables_property_returns_copy(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
            dry_run=True,
        )
        v1 = runner.variables
        v2 = runner.variables
        assert v1 == v2
        assert v1 is not v2
        logger.close()

    def test_initial_variables_from_workflow(self) -> None:
        workflow = _minimal_workflow()
        logger = _silent_logger()
        client = MagicMock(spec=OllamaClient)

        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
        )
        assert runner.variables.get("name") == "world"
        logger.close()


# ---------------------------------------------------------------------------
# run_workflow convenience function
# ---------------------------------------------------------------------------


class TestRunWorkflow:
    def test_run_workflow_dry_run(self, tmp_path: Path) -> None:
        """run_workflow with dry_run=True should not call Ollama at all."""
        workflow = _minimal_workflow()

        with patch(
            "local_agent_runner.agent.OllamaClient"
        ) as MockClient:
            mock_instance = MagicMock(spec=OllamaClient)
            mock_instance.chat = MagicMock()
            MockClient.return_value = mock_instance

            result = run_workflow(
                workflow,
                ollama_url="http://localhost:11434",
                dry_run=True,
                verbose=False,
            )

        assert result["success"] is True
        mock_instance.chat.assert_not_called()

    def test_run_workflow_writes_log_file(self, tmp_path: Path) -> None:
        """run_workflow should write NDJSON to log_file when specified."""
        import json as _json

        workflow = _minimal_workflow()
        log_path = tmp_path / "run.jsonl"

        with patch(
            "local_agent_runner.agent.OllamaClient"
        ) as MockClient:
            mock_instance = MagicMock(spec=OllamaClient)
            mock_instance.chat = MagicMock()
            MockClient.return_value = mock_instance

            result = run_workflow(
                workflow,
                dry_run=True,
                log_file=log_path,
            )

        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1  # at least run_start
        for line in lines:
            obj = _json.loads(line)
            assert "event_type" in obj
