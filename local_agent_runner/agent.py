"""Core agentic loop for local_agent_runner.

This module implements the Ollama-driven agentic loop that:

1. Builds a system prompt describing available tools for each step.
2. Sends the user prompt to the Ollama ``/api/chat`` endpoint.
3. Parses tool-call requests from the LLM's response (both native Ollama
   tool-call format and a JSON-code-block fallback format).
4. Dispatches tool calls via :func:`~local_agent_runner.tools.dispatch_tool`.
5. Feeds tool results back into the conversation and repeats until the LLM
   produces a final answer or ``max_iterations`` is reached.
6. Stores step outputs into a shared variable registry for use by later steps.

Public API
----------
- ``AgentError``      — exception raised when the agentic loop encounters an
  unrecoverable error.
- ``OllamaClient``    — thin async HTTP wrapper around the Ollama API.
- ``AgentRunner``     — orchestrates multi-step workflow execution.
- ``run_workflow()``  — convenience function to build and run an
  ``AgentRunner`` from a :class:`~local_agent_runner.config.WorkflowConfig`.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from local_agent_runner.config import (
    SandboxConfig,
    StepConfig,
    ToolDefinition,
    ToolType,
    WorkflowConfig,
)
from local_agent_runner.logger import RunLogger
from local_agent_runner.sandbox import SandboxContext
from local_agent_runner.tools import ToolResult, dispatch_tool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL: str = "http://localhost:11434"
_TOOL_CALL_JSON_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AgentError(RuntimeError):
    """Raised when the agentic loop encounters an unrecoverable error.

    Attributes
    ----------
    step_name:
        Name of the step during which the error occurred, or empty string.
    detail:
        Human-readable description of the error.
    """

    def __init__(self, detail: str, step_name: str = "") -> None:
        self.detail = detail
        self.step_name = step_name
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Ollama message / response types
# ---------------------------------------------------------------------------


@dataclass
class OllamaMessage:
    """A single message in the Ollama chat conversation.

    Attributes
    ----------
    role:
        One of ``"system"``, ``"user"``, or ``"assistant"``.
    content:
        The text content of the message.
    """

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serialisable dict for the Ollama API."""
        return {"role": self.role, "content": self.content}


@dataclass
class ParsedToolCall:
    """A tool-call request parsed from an LLM response.

    Attributes
    ----------
    tool_name:
        The name of the tool to invoke (as declared in the workflow YAML).
    arguments:
        Key/value arguments for the tool.
    raw:
        The raw string fragment from which this call was parsed.
    """

    tool_name: str
    arguments: Dict[str, Any]
    raw: str = ""


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin synchronous HTTP client for the Ollama ``/api/chat`` endpoint.

    Parameters
    ----------
    base_url:
        Base URL of the Ollama server, e.g. ``"http://localhost:11434"``.
    timeout:
        HTTP request timeout in seconds.  Defaults to ``120.0``.

    Attributes
    ----------
    base_url:
        The base URL used for all requests.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[OllamaMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama and return the parsed JSON response.

        Parameters
        ----------
        model:
            Ollama model identifier, e.g. ``"llama3"``.
        messages:
            Conversation history as a list of :class:`OllamaMessage` objects.
        tools:
            Optional list of tool definitions in Ollama's function-calling
            format.  When provided, the model may return tool_calls in the
            response.
        stream:
            When ``True`` the endpoint streams responses; this client
            currently only supports ``stream=False``.

        Returns
        -------
        Dict[str, Any]
            The parsed JSON response body from Ollama.

        Raises
        ------
        AgentError
            On connection failure, timeout, or non-200 HTTP response.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise AgentError(
                f"Ollama request timed out after {self._timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise AgentError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AgentError(
                f"Ollama returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise AgentError(
                f"Failed to parse Ollama JSON response: {exc}"
            ) from exc

    def list_models(self) -> List[str]:
        """Return a list of model names available on the Ollama server.

        Returns
        -------
        List[str]
            Model name strings, e.g. ``["llama3", "mistral"]``.

        Raises
        ------
        AgentError
            On connection failure or non-200 response.
        """
        url = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url)
        except httpx.RequestError as exc:
            raise AgentError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise AgentError(
                f"Ollama /api/tags returned HTTP {response.status_code}"
            )

        data = response.json()
        return [m.get("name", "") for m in data.get("models", [])]

    def health_check(self) -> bool:
        """Return ``True`` if the Ollama server is reachable.

        Returns
        -------
        bool
        """
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
            return response.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


def _build_ollama_tools(
    tool_defs: List[ToolDefinition],
) -> List[Dict[str, Any]]:
    """Convert :class:`~local_agent_runner.config.ToolDefinition` objects into
    the Ollama function-calling schema.

    Parameters
    ----------
    tool_defs:
        Tool definitions from the workflow YAML.

    Returns
    -------
    List[Dict[str, Any]]
        Tool descriptors in Ollama / OpenAI function-calling format.
    """
    result: List[Dict[str, Any]] = []
    for td in tool_defs:
        # Build a minimal but correct function schema per tool type
        if td.tool_type is ToolType.FILE_READ:
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path of the file to read.",
                    }
                },
                "required": ["path"],
            }
        elif td.tool_type is ToolType.FILE_WRITE:
            parameters = {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write.",
                    },
                },
                "required": ["path", "content"],
            }
        elif td.tool_type is ToolType.SHELL:
            parameters = {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    }
                },
                "required": ["command"],
            }
        else:  # WEB_SEARCH
            parameters = {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch via HTTP GET.",
                    }
                },
                "required": ["url"],
            }

        result.append(
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description or f"{td.tool_type.value} tool",
                    "parameters": parameters,
                },
            }
        )
    return result


def _build_system_prompt(tool_defs: List[ToolDefinition]) -> str:
    """Build a system prompt that describes the available tools to the LLM.

    Parameters
    ----------
    tool_defs:
        Tool definitions available in the current step.

    Returns
    -------
    str
        System prompt text.
    """
    if not tool_defs:
        return (
            "You are a helpful assistant. "
            "Complete the user's request to the best of your ability."
        )

    lines = [
        "You are a helpful assistant that can use the following tools:",
        "",
    ]
    for td in tool_defs:
        lines.append(f"Tool name: {td.name}")
        lines.append(f"Type: {td.tool_type.value}")
        if td.description:
            lines.append(f"Description: {td.description.strip()}")
        lines.append("")

    lines += [
        "When you need to use a tool, respond with a JSON object in a code block:",
        "",
        "```json",
        '{"tool": "<tool_name>", "arguments": {<key>: <value>, ...}}',
        "```",
        "",
        "You may call multiple tools in sequence. After receiving tool results,",
        "continue reasoning and either call more tools or provide your final answer.",
        "When you have a complete answer, respond with plain text (no JSON code block).",
    ]
    return "\n".join(lines)


def parse_tool_calls(
    response_text: str,
    available_tool_names: List[str],
) -> List[ParsedToolCall]:
    """Extract tool-call requests from an LLM response string.

    Supports two formats:

    1. **JSON code block** — the model wraps a JSON object in a
       ``\`\`\`json ... \`\`\`` fence.  The object must have ``"tool"``
       and ``"arguments"`` keys.
    2. **Inline JSON** — a bare JSON object in the text with ``"tool"``
       and ``"arguments"`` keys (no fences).

    Only tool names that are in *available_tool_names* are accepted.

    Parameters
    ----------
    response_text:
        The raw text response from the LLM.
    available_tool_names:
        Names of tools that are valid in the current step.

    Returns
    -------
    List[ParsedToolCall]
        Zero or more parsed tool-call objects, in the order they appear.
    """
    calls: List[ParsedToolCall] = []
    seen_raws: set[str] = set()

    # --- Strategy 1: JSON code blocks ---
    for match in _TOOL_CALL_JSON_RE.finditer(response_text):
        raw_json = match.group(1).strip()
        call = _try_parse_tool_json(raw_json, available_tool_names)
        if call is not None and raw_json not in seen_raws:
            call = ParsedToolCall(
                tool_name=call.tool_name,
                arguments=call.arguments,
                raw=match.group(0),
            )
            calls.append(call)
            seen_raws.add(raw_json)

    if calls:
        return calls

    # --- Strategy 2: Bare JSON objects anywhere in the text ---
    # Find all {...} blobs and try to parse them
    brace_re = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    for match in brace_re.finditer(response_text):
        raw_json = match.group(0).strip()
        if raw_json in seen_raws:
            continue
        call = _try_parse_tool_json(raw_json, available_tool_names)
        if call is not None:
            calls.append(call)
            seen_raws.add(raw_json)

    return calls


def _try_parse_tool_json(
    raw_json: str,
    available_tool_names: List[str],
) -> Optional[ParsedToolCall]:
    """Attempt to parse a single JSON string as a tool-call object.

    Parameters
    ----------
    raw_json:
        Candidate JSON string.
    available_tool_names:
        Valid tool names for the current step.

    Returns
    -------
    Optional[ParsedToolCall]
        A parsed call if *raw_json* is valid tool-call JSON, otherwise
        ``None``.
    """
    try:
        obj = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None

    tool_name = obj.get("tool") or obj.get("name") or obj.get("function")
    if not isinstance(tool_name, str):
        return None

    # Also handle OpenAI-style {"function": {"name": ..., "arguments": ...}}
    if tool_name not in available_tool_names:
        # Try nested format: {"function": {"name": ..., "arguments": ...}}
        nested = obj.get("function")
        if isinstance(nested, dict):
            tool_name = nested.get("name", "")
            if tool_name not in available_tool_names:
                return None
            arguments = nested.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
            return ParsedToolCall(
                tool_name=tool_name,
                arguments=arguments if isinstance(arguments, dict) else {},
                raw=raw_json,
            )
        return None

    arguments = obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    return ParsedToolCall(
        tool_name=tool_name,
        arguments=arguments,
        raw=raw_json,
    )


def _parse_ollama_native_tool_calls(
    response_data: Dict[str, Any],
    available_tool_names: List[str],
) -> List[ParsedToolCall]:
    """Extract native Ollama tool_calls from a ``/api/chat`` response.

    Ollama (≥ 0.2) may return a ``message.tool_calls`` array when the model
    invokes a function-calling tool.  This function extracts those calls.

    Parameters
    ----------
    response_data:
        Parsed JSON response from ``/api/chat``.
    available_tool_names:
        Valid tool names for the current step.

    Returns
    -------
    List[ParsedToolCall]
    """
    calls: List[ParsedToolCall] = []
    message = response_data.get("message", {})
    raw_calls = message.get("tool_calls") or []

    for raw in raw_calls:
        fn = raw.get("function", {})
        tool_name = fn.get("name", "")
        if tool_name not in available_tool_names:
            continue
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ParsedToolCall(
                tool_name=tool_name,
                arguments=arguments,
                raw=json.dumps(raw),
            )
        )
    return calls


# ---------------------------------------------------------------------------
# StepRunner
# ---------------------------------------------------------------------------


class StepRunner:
    """Executes a single workflow step by driving the Ollama agentic loop.

    Parameters
    ----------
    workflow:
        The parent :class:`~local_agent_runner.config.WorkflowConfig`.
    step:
        The :class:`~local_agent_runner.config.StepConfig` to execute.
    client:
        An :class:`OllamaClient` used to communicate with Ollama.
    logger:
        A :class:`~local_agent_runner.logger.RunLogger` for structured output.
    variables:
        Shared variable registry (mutated in-place when
        ``step.output_variable`` is set).
    dry_run:
        When ``True`` the LLM is never called; the step returns a placeholder
        response immediately.
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        step: StepConfig,
        client: OllamaClient,
        logger: RunLogger,
        variables: Dict[str, Any],
        dry_run: bool = False,
    ) -> None:
        self._workflow = workflow
        self._step = step
        self._client = client
        self._logger = logger
        self._variables = variables
        self._dry_run = dry_run

    def run(self) -> str:
        """Execute the step and return the final LLM response text.

        Returns
        -------
        str
            The final textual response from the LLM (or a dry-run placeholder).

        Raises
        ------
        AgentError
            On unrecoverable errors (Ollama connection failure, max iterations
            exceeded without a final answer, etc.).
        """
        step = self._step
        workflow = self._workflow

        # Resolve prompt template
        try:
            prompt = step.prompt.format_map(self._variables)
        except KeyError as exc:
            raise AgentError(
                f"Prompt template references undefined variable {exc} "
                f"in step {step.name!r}.",
                step_name=step.name,
            ) from exc

        # Determine effective sandbox for this step
        effective_sandbox_cfg = workflow.effective_sandbox(step)
        sandbox = SandboxContext(effective_sandbox_cfg)

        # Gather tool definitions available to this step
        step_tool_defs: List[ToolDefinition] = []
        for tool_name in step.tools:
            td = workflow.get_tool(tool_name)
            if td is not None:
                step_tool_defs.append(td)

        available_tool_names = [td.name for td in step_tool_defs]
        ollama_tools = _build_ollama_tools(step_tool_defs)
        system_prompt = _build_system_prompt(step_tool_defs)

        # --- Dry-run fast path ---
        if self._dry_run:
            self._logger.log_info(
                f"[dry-run] Skipping LLM call for step {step.name!r}.",
                step_name=step.name,
            )
            output = f"[dry-run] Step {step.name!r} skipped."
            if step.output_variable:
                self._variables[step.output_variable] = output
            return output

        # Build initial conversation
        messages: List[OllamaMessage] = [
            OllamaMessage(role="system", content=system_prompt),
            OllamaMessage(role="user", content=prompt),
        ]

        self._logger.log_llm_prompt(
            step_name=step.name,
            prompt=prompt,
            model=workflow.model,
            iteration=0,
        )

        final_response = ""
        iterations_used = 0

        for iteration in range(1, step.max_iterations + 1):
            iterations_used = iteration

            # Call Ollama
            try:
                response_data = self._client.chat(
                    model=workflow.model,
                    messages=messages,
                    tools=ollama_tools if step_tool_defs else None,
                )
            except AgentError:
                raise

            message_obj = response_data.get("message", {})
            response_text: str = message_obj.get("content") or ""

            self._logger.log_llm_response(
                step_name=step.name,
                response=response_text,
                model=workflow.model,
                iteration=iteration,
            )

            # Check for native Ollama tool_calls first
            native_calls = _parse_ollama_native_tool_calls(
                response_data, available_tool_names
            )

            # Fall back to text-based parsing
            text_calls: List[ParsedToolCall] = []
            if not native_calls and available_tool_names:
                text_calls = parse_tool_calls(response_text, available_tool_names)

            tool_calls = native_calls or text_calls

            if not tool_calls:
                # No tool calls — this is the final answer
                final_response = response_text
                break

            # Log and execute each tool call
            tool_results_for_log: List[Dict[str, Any]] = []
            tool_result_messages: List[str] = []

            for tc in tool_calls:
                td = self._workflow.get_tool(tc.tool_name)
                if td is None:
                    self._logger.log_warning(
                        f"LLM called unknown tool {tc.tool_name!r}; skipping.",
                        step_name=step.name,
                    )
                    continue

                self._logger.log_tool_call(
                    step_name=step.name,
                    tool_name=tc.tool_name,
                    tool_type=td.tool_type.value,
                    arguments=tc.arguments,
                    iteration=iteration,
                )

                # Log sandbox decision (pre-emptively via dispatch which
                # internally calls sandbox.check_*)
                result = dispatch_tool(
                    tool_type=td.tool_type,
                    arguments=tc.arguments,
                    sandbox=sandbox,
                )

                # Log sandbox records produced during this dispatch
                for rec in sandbox.records[-1:]:
                    self._logger.log_sandbox_action(
                        step_name=step.name,
                        action_type=rec.action_type.value,
                        target=rec.target,
                        permitted=rec.permitted,
                        reason=rec.reason,
                    )

                self._logger.log_tool_result(
                    step_name=step.name,
                    tool_name=tc.tool_name,
                    result=result.output if result.success else result.error,
                    success=result.success,
                    error_message=result.error if not result.success else "",
                    iteration=iteration,
                )

                tool_results_for_log.append(
                    {
                        "tool": tc.tool_name,
                        "success": result.success,
                        "output_preview": str(result)[:100],
                    }
                )

                # Build the tool-result message to feed back into conversation
                if result.success:
                    tool_result_messages.append(
                        f"Tool '{tc.tool_name}' result:\n{result.output}"
                    )
                else:
                    tool_result_messages.append(
                        f"Tool '{tc.tool_name}' failed: {result.error}"
                    )

            if not tool_result_messages:
                # All tool calls were skipped/unknown
                final_response = response_text
                break

            # Add assistant turn (with the tool call text) and tool results
            messages.append(
                OllamaMessage(role="assistant", content=response_text)
            )
            combined_results = "\n\n".join(tool_result_messages)
            messages.append(
                OllamaMessage(role="user", content=combined_results)
            )

            self._logger.log_llm_prompt(
                step_name=step.name,
                prompt=combined_results,
                model=workflow.model,
                iteration=iteration,
            )

        else:
            # Exhausted max_iterations without a final answer
            self._logger.log_warning(
                f"Step {step.name!r} reached max_iterations "
                f"({step.max_iterations}) without a final answer. "
                "Using last response as output.",
                step_name=step.name,
            )
            final_response = response_text if 'response_text' in dir() else ""

        # Store output variable
        if step.output_variable:
            self._variables[step.output_variable] = final_response

        return final_response


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------


class AgentRunner:
    """Orchestrates the execution of an entire multi-step workflow.

    Parameters
    ----------
    workflow:
        The parsed and validated :class:`~local_agent_runner.config.WorkflowConfig`.
    client:
        An :class:`OllamaClient` used to communicate with Ollama.
    logger:
        A :class:`~local_agent_runner.logger.RunLogger` for structured output.
    dry_run:
        When ``True`` all LLM calls are skipped; tool stubs are still
        invoked (unless the sandbox blocks them).
    """

    def __init__(
        self,
        workflow: WorkflowConfig,
        client: OllamaClient,
        logger: RunLogger,
        dry_run: bool = False,
    ) -> None:
        self._workflow = workflow
        self._client = client
        self._logger = logger
        self._dry_run = dry_run
        # Initialise variable registry from workflow defaults
        self._variables: Dict[str, Any] = dict(workflow.variables)

    @property
    def variables(self) -> Dict[str, Any]:
        """Current variable registry (snapshot)."""
        return dict(self._variables)

    def run(self) -> Dict[str, Any]:
        """Execute every step in the workflow in order.

        Returns
        -------
        Dict[str, Any]
            A summary dict containing:

            - ``success`` (bool): whether all steps completed without error.
            - ``variables`` (dict): final variable registry.
            - ``steps_completed`` (int): number of steps that finished.
            - ``error`` (str): non-empty on failure.

        Raises
        ------
        AgentError
            Only if ``raise_on_error`` is ``True`` (not yet exposed; for now
            errors are captured in the return dict).
        """
        import time

        workflow = self._workflow
        self._logger.log_run_start(
            workflow_name=workflow.name,
            model=workflow.model,
        )
        start_time = time.monotonic()

        steps_completed = 0
        last_error = ""
        success = True

        for step_index, step in enumerate(workflow.steps):
            self._logger.log_step_start(
                step_name=step.name,
                step_index=step_index,
                description=step.description,
            )

            step_runner = StepRunner(
                workflow=workflow,
                step=step,
                client=self._client,
                logger=self._logger,
                variables=self._variables,
                dry_run=self._dry_run,
            )

            try:
                output = step_runner.run()
                steps_completed += 1
                self._logger.log_step_end(
                    step_name=step.name,
                    output_variable=step.output_variable,
                    success=True,
                )
            except AgentError as exc:
                last_error = str(exc)
                success = False
                self._logger.log_error(
                    message=f"Step {step.name!r} failed: {exc}",
                    exception=exc,
                    step_name=step.name,
                )
                self._logger.log_step_end(
                    step_name=step.name,
                    success=False,
                )
                break
            except Exception as exc:
                last_error = f"Unexpected error in step {step.name!r}: {exc}"
                success = False
                self._logger.log_error(
                    message=last_error,
                    exception=exc,
                    step_name=step.name,
                )
                self._logger.log_step_end(
                    step_name=step.name,
                    success=False,
                )
                break

        duration = time.monotonic() - start_time
        self._logger.log_run_end(
            success=success,
            duration_seconds=duration,
        )

        return {
            "success": success,
            "variables": dict(self._variables),
            "steps_completed": steps_completed,
            "error": last_error,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_workflow(
    workflow: WorkflowConfig,
    *,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    log_file: Optional[str | Path] = None,
    verbose: bool = False,
    dry_run: bool = False,
    ollama_timeout: float = 120.0,
) -> Dict[str, Any]:
    """Build an :class:`AgentRunner` and execute the workflow.

    This is the primary high-level entry point used by the CLI.

    Parameters
    ----------
    workflow:
        Validated :class:`~local_agent_runner.config.WorkflowConfig`.
    ollama_url:
        Base URL of the Ollama server.
    log_file:
        Optional path for the NDJSON log file.
    verbose:
        When ``True``, debug-level events are printed to the terminal.
    dry_run:
        When ``True``, LLM calls are skipped.
    ollama_timeout:
        HTTP timeout for Ollama requests in seconds.

    Returns
    -------
    Dict[str, Any]
        Summary dict from :meth:`AgentRunner.run`.
    """
    run_id = str(uuid.uuid4())[:8]
    client = OllamaClient(base_url=ollama_url, timeout=ollama_timeout)

    with RunLogger(
        run_id=run_id,
        verbose=verbose,
        log_file=log_file,
    ) as logger:
        runner = AgentRunner(
            workflow=workflow,
            client=client,
            logger=logger,
            dry_run=dry_run,
        )
        return runner.run()
