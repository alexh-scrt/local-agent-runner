"""Configuration loader and typed data models for local_agent_runner.

This module is responsible for:
- Defining typed dataclasses that represent every element of a YAML workflow file.
- Loading a YAML file from disk and deserialising it into those dataclasses.
- Validating required fields, allowed enum values, and structural constraints so
  that downstream components receive well-formed objects and never have to
  repeat validation logic.

Public API
----------
- ``ToolType``          — enum of supported tool kinds.
- ``SandboxConfig``     — sandbox allow-list settings.
- ``ToolDefinition``    — a single tool entry from the ``tools:`` block.
- ``StepConfig``        — one step in the ``steps:`` list.
- ``WorkflowConfig``    — the top-level workflow document.
- ``load_workflow()``   — load + validate a YAML file and return a
  ``WorkflowConfig``.
- ``validate_workflow()`` — validate an already-loaded ``WorkflowConfig``.
- ``ConfigError``       — exception raised on any validation failure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a workflow YAML file fails to load or validate."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ToolType(str, Enum):
    """Supported tool types that a workflow step may invoke."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL = "shell"
    WEB_SEARCH = "web_search"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SandboxConfig:
    """Sandbox allow-list settings.

    All lists are empty by default, meaning *no* operations are permitted when
    the sandbox is enabled unless explicitly allowed.

    Attributes
    ----------
    enabled:
        Whether the sandbox is active.
    allowed_paths:
        Path prefixes that file I/O tools are allowed to access.
    allowed_commands:
        Shell command prefixes that the shell tool is allowed to execute.
    allowed_domains:
        Hostname suffixes that the web-search tool is allowed to contact.
    """

    enabled: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    allowed_commands: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SandboxConfig":
        """Construct a :class:`SandboxConfig` from a raw mapping.

        Parameters
        ----------
        data:
            Mapping as parsed from the YAML ``sandbox:`` block.  Any missing
            key is silently defaulted.

        Returns
        -------
        SandboxConfig
        """
        if not isinstance(data, dict):
            raise ConfigError(
                f"'sandbox' must be a mapping, got {type(data).__name__!r}"
            )
        return cls(
            enabled=bool(data.get("enabled", False)),
            allowed_paths=_str_list(data, "allowed_paths", "sandbox.allowed_paths"),
            allowed_commands=_str_list(
                data, "allowed_commands", "sandbox.allowed_commands"
            ),
            allowed_domains=_str_list(
                data, "allowed_domains", "sandbox.allowed_domains"
            ),
        )

    def merge(self, override: "SandboxConfig") -> "SandboxConfig":
        """Return a new :class:`SandboxConfig` that merges *override* on top.

        The *override*'s ``enabled`` flag always wins.  Lists from both configs
        are **unioned** so that step-level overrides can only *extend* the
        global allow-lists, never narrow them.

        Parameters
        ----------
        override:
            Step-level sandbox configuration to merge.

        Returns
        -------
        SandboxConfig
        """
        return SandboxConfig(
            enabled=override.enabled,
            allowed_paths=_union(self.allowed_paths, override.allowed_paths),
            allowed_commands=_union(
                self.allowed_commands, override.allowed_commands
            ),
            allowed_domains=_union(self.allowed_domains, override.allowed_domains),
        )


@dataclass
class ToolDefinition:
    """A single tool entry declared in the workflow's ``tools:`` block.

    Attributes
    ----------
    name:
        Unique identifier for this tool.  Used by step ``tools:`` lists.
    tool_type:
        The :class:`ToolType` enum value indicating which implementation to use.
    description:
        Human-readable description passed to the LLM in the system prompt.
    """

    name: str
    tool_type: ToolType
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int) -> "ToolDefinition":
        """Construct a :class:`ToolDefinition` from a raw mapping.

        Parameters
        ----------
        data:
            Mapping as parsed from a single entry in the YAML ``tools:`` list.
        index:
            Zero-based position in the list; used in error messages.

        Returns
        -------
        ToolDefinition

        Raises
        ------
        ConfigError
            If required fields are missing or ``type`` is not a valid
            :class:`ToolType`.
        """
        if not isinstance(data, dict):
            raise ConfigError(
                f"tools[{index}] must be a mapping, got {type(data).__name__!r}"
            )
        name = _require_str(data, "name", f"tools[{index}].name")
        raw_type = _require_str(data, "type", f"tools[{index}].type")
        try:
            tool_type = ToolType(raw_type)
        except ValueError:
            valid = ", ".join(t.value for t in ToolType)
            raise ConfigError(
                f"tools[{index}].type {raw_type!r} is not valid; "
                f"must be one of: {valid}"
            ) from None
        description = str(data.get("description", ""))
        return cls(name=name, tool_type=tool_type, description=description)


@dataclass
class StepConfig:
    """Configuration for a single step in the workflow.

    Attributes
    ----------
    name:
        Human-readable step name; must be unique within the workflow.
    prompt:
        Prompt template string.  Use ``{variable_name}`` placeholders which are
        resolved against the workflow's ``variables`` dict at runtime.
    description:
        Optional prose description of what this step does.
    tools:
        List of tool *names* (referring to entries in ``WorkflowConfig.tools``)
        that this step is allowed to invoke.
    output_variable:
        If set, the LLM's final response for this step is stored in the
        workflow variable registry under this key for use in later steps.
    max_iterations:
        Maximum number of tool-call / LLM-response rounds before the step is
        considered failed.  Defaults to 5.
    sandbox:
        Optional step-level :class:`SandboxConfig` override.
    """

    name: str
    prompt: str
    description: str = ""
    tools: List[str] = field(default_factory=list)
    output_variable: Optional[str] = None
    max_iterations: int = 5
    sandbox: Optional[SandboxConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int) -> "StepConfig":
        """Construct a :class:`StepConfig` from a raw mapping.

        Parameters
        ----------
        data:
            Mapping as parsed from a single entry in the YAML ``steps:`` list.
        index:
            Zero-based position in the list; used in error messages.

        Returns
        -------
        StepConfig

        Raises
        ------
        ConfigError
            If required fields are missing or values are of wrong types.
        """
        if not isinstance(data, dict):
            raise ConfigError(
                f"steps[{index}] must be a mapping, got {type(data).__name__!r}"
            )
        name = _require_str(data, "name", f"steps[{index}].name")
        prompt = _require_str(data, "prompt", f"steps[{index}].prompt")
        description = str(data.get("description", ""))
        tools = _str_list(data, "tools", f"steps[{index}].tools")
        output_variable: Optional[str] = None
        if "output_variable" in data:
            output_variable = _require_str(
                data, "output_variable", f"steps[{index}].output_variable"
            )
        max_iter_raw = data.get("max_iterations", 5)
        try:
            max_iterations = int(max_iter_raw)
        except (TypeError, ValueError):
            raise ConfigError(
                f"steps[{index}].max_iterations must be an integer, "
                f"got {max_iter_raw!r}"
            ) from None
        if max_iterations < 1:
            raise ConfigError(
                f"steps[{index}].max_iterations must be >= 1, "
                f"got {max_iterations}"
            )
        sandbox: Optional[SandboxConfig] = None
        if "sandbox" in data:
            sandbox = SandboxConfig.from_dict(data["sandbox"])
        return cls(
            name=name,
            prompt=prompt,
            description=description,
            tools=tools,
            output_variable=output_variable,
            max_iterations=max_iterations,
            sandbox=sandbox,
        )


@dataclass
class WorkflowConfig:
    """Top-level workflow document parsed from a YAML file.

    Attributes
    ----------
    name:
        Human-readable workflow name.
    model:
        Ollama model identifier, e.g. ``"llama3"``.
    description:
        Optional prose description of the workflow.
    sandbox:
        Global :class:`SandboxConfig`.  Step-level configs are merged on top.
    tools:
        List of :class:`ToolDefinition` objects declared by this workflow.
    steps:
        Ordered list of :class:`StepConfig` objects.
    variables:
        Initial variable registry for prompt template substitution.
    source_path:
        Absolute path to the YAML file this config was loaded from, or
        ``None`` if built programmatically.
    """

    name: str
    model: str
    description: str = ""
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    tools: List[ToolDefinition] = field(default_factory=list)
    steps: List[StepConfig] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Return the :class:`ToolDefinition` with *name*, or ``None``.

        Parameters
        ----------
        name:
            Tool identifier to look up.

        Returns
        -------
        Optional[ToolDefinition]
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def effective_sandbox(self, step: StepConfig) -> SandboxConfig:
        """Return the effective :class:`SandboxConfig` for *step*.

        If *step* has its own sandbox block, it is merged on top of the global
        config (lists are unioned, ``enabled`` flag comes from the step).
        Otherwise the global config is returned unchanged.

        Parameters
        ----------
        step:
            The step whose effective sandbox config to compute.

        Returns
        -------
        SandboxConfig
        """
        if step.sandbox is not None:
            return self.sandbox.merge(step.sandbox)
        return self.sandbox


# ---------------------------------------------------------------------------
# Public loading / validation helpers
# ---------------------------------------------------------------------------


def load_workflow(path: str | Path) -> WorkflowConfig:
    """Load a YAML workflow file from *path* and return a validated
    :class:`WorkflowConfig`.

    Parameters
    ----------
    path:
        Filesystem path to the ``.yaml`` workflow file.

    Returns
    -------
    WorkflowConfig

    Raises
    ------
    ConfigError
        On any file I/O error, YAML parse failure, or schema validation
        problem.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Workflow file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Workflow path is not a file: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read workflow file {path}: {exc}") from exc
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc
    if raw_data is None:
        raise ConfigError(f"Workflow file is empty: {path}")
    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Workflow file must be a YAML mapping at the top level: {path}"
        )
    config = _parse_workflow(raw_data)
    config.source_path = path.resolve()
    validate_workflow(config)
    return config


def load_workflow_from_string(yaml_text: str) -> WorkflowConfig:
    """Parse a YAML workflow from a string and return a validated
    :class:`WorkflowConfig`.

    This is primarily useful for testing.

    Parameters
    ----------
    yaml_text:
        Raw YAML text representing a workflow document.

    Returns
    -------
    WorkflowConfig

    Raises
    ------
    ConfigError
        On any YAML parse failure or schema validation problem.
    """
    try:
        raw_data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}") from exc
    if raw_data is None:
        raise ConfigError("Workflow document is empty.")
    if not isinstance(raw_data, dict):
        raise ConfigError(
            "Workflow document must be a YAML mapping at the top level."
        )
    config = _parse_workflow(raw_data)
    validate_workflow(config)
    return config


def validate_workflow(config: WorkflowConfig) -> None:
    """Validate a :class:`WorkflowConfig` for semantic consistency.

    Checks performed:

    1. ``name`` and ``model`` are non-empty strings.
    2. At least one step is defined.
    3. All step tool references resolve to declared tools.
    4. Step names are unique.
    5. ``output_variable`` names must be valid Python identifiers.

    Parameters
    ----------
    config:
        The workflow configuration to validate.

    Raises
    ------
    ConfigError
        If any consistency check fails.
    """
    if not config.name.strip():
        raise ConfigError("Workflow 'name' must not be blank.")
    if not config.model.strip():
        raise ConfigError("Workflow 'model' must not be blank.")
    if not config.steps:
        raise ConfigError("Workflow must define at least one step.")

    declared_tool_names = {t.name for t in config.tools}
    seen_step_names: set[str] = set()

    for i, step in enumerate(config.steps):
        label = f"steps[{i}] ({step.name!r})"
        if step.name in seen_step_names:
            raise ConfigError(f"Duplicate step name {step.name!r} at {label}.")
        seen_step_names.add(step.name)

        for tool_ref in step.tools:
            if tool_ref not in declared_tool_names:
                raise ConfigError(
                    f"{label} references undefined tool {tool_ref!r}. "
                    f"Declared tools: {sorted(declared_tool_names)}"
                )

        if step.output_variable is not None:
            if not step.output_variable.isidentifier():
                raise ConfigError(
                    f"{label}.output_variable {step.output_variable!r} is not "
                    "a valid Python identifier."
                )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_workflow(data: Dict[str, Any]) -> WorkflowConfig:
    """Deserialise a raw YAML mapping into a :class:`WorkflowConfig`.

    Does **not** run semantic validation; call :func:`validate_workflow` after
    this function.

    Parameters
    ----------
    data:
        Top-level mapping as returned by ``yaml.safe_load``.

    Returns
    -------
    WorkflowConfig
    """
    name = _require_str(data, "name", "name")
    model = _require_str(data, "model", "model")
    description = str(data.get("description", ""))

    sandbox = SandboxConfig()
    if "sandbox" in data:
        sandbox = SandboxConfig.from_dict(data["sandbox"])

    tools: List[ToolDefinition] = []
    raw_tools = data.get("tools", []) or []
    if not isinstance(raw_tools, list):
        raise ConfigError(
            f"'tools' must be a list, got {type(raw_tools).__name__!r}"
        )
    for i, raw_tool in enumerate(raw_tools):
        tools.append(ToolDefinition.from_dict(raw_tool, i))

    steps: List[StepConfig] = []
    raw_steps = data.get("steps", []) or []
    if not isinstance(raw_steps, list):
        raise ConfigError(
            f"'steps' must be a list, got {type(raw_steps).__name__!r}"
        )
    for i, raw_step in enumerate(raw_steps):
        steps.append(StepConfig.from_dict(raw_step, i))

    variables: Dict[str, Any] = {}
    raw_vars = data.get("variables", {}) or {}
    if not isinstance(raw_vars, dict):
        raise ConfigError(
            f"'variables' must be a mapping, got {type(raw_vars).__name__!r}"
        )
    variables = {str(k): v for k, v in raw_vars.items()}

    return WorkflowConfig(
        name=name,
        model=model,
        description=description,
        sandbox=sandbox,
        tools=tools,
        steps=steps,
        variables=variables,
    )


def _require_str(data: Dict[str, Any], key: str, label: str) -> str:
    """Extract a non-empty string value from *data[key]*.

    Parameters
    ----------
    data:
        Source mapping.
    key:
        Key to look up.
    label:
        Human-readable field path used in error messages.

    Returns
    -------
    str

    Raises
    ------
    ConfigError
        If the key is missing or the value is not a non-empty string.
    """
    if key not in data:
        raise ConfigError(f"Missing required field: '{label}'")
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(
            f"'{label}' must be a string, got {type(value).__name__!r}"
        )
    if not value.strip():
        raise ConfigError(f"'{label}' must not be blank")
    return value


def _str_list(data: Dict[str, Any], key: str, label: str) -> List[str]:
    """Extract an optional list of strings from *data[key]*.

    Returns an empty list if the key is absent or the value is ``None``.

    Parameters
    ----------
    data:
        Source mapping.
    key:
        Key to look up.
    label:
        Human-readable field path used in error messages.

    Returns
    -------
    List[str]

    Raises
    ------
    ConfigError
        If the value is present but not a list, or if any element is not a
        string.
    """
    raw = data.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError(
            f"'{label}' must be a list, got {type(raw).__name__!r}"
        )
    result: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ConfigError(
                f"'{label}[{i}]' must be a string, got {type(item).__name__!r}"
            )
        result.append(item)
    return result


def _union(a: List[str], b: List[str]) -> List[str]:
    """Return a deduplicated union of lists *a* and *b*, preserving order.

    Parameters
    ----------
    a:
        First list.
    b:
        Second list.

    Returns
    -------
    List[str]
    """
    seen: set[str] = set()
    result: List[str] = []
    for item in (*a, *b):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
