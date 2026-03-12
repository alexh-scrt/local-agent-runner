"""Unit tests for local_agent_runner.config.

Covers:
- Loading valid YAML from file and from string.
- Correct deserialisation of all dataclass fields.
- Validation errors for missing / malformed required fields.
- SandboxConfig.merge behaviour.
- WorkflowConfig.get_tool and effective_sandbox helpers.
- ToolType enum values.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from local_agent_runner.config import (
    ConfigError,
    SandboxConfig,
    StepConfig,
    ToolDefinition,
    ToolType,
    WorkflowConfig,
    load_workflow,
    load_workflow_from_string,
    validate_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent("""\
    name: Test Workflow
    model: llama3
    tools:
      - name: read_file
        type: file_read
        description: Read a file.
    steps:
      - name: step_one
        prompt: "Hello {name}"
        tools:
          - read_file
        output_variable: result
    variables:
      name: world
""")


def _minimal() -> WorkflowConfig:
    """Return a parsed minimal WorkflowConfig for reuse."""
    return load_workflow_from_string(MINIMAL_YAML)


# ---------------------------------------------------------------------------
# ToolType enum
# ---------------------------------------------------------------------------


class TestToolType:
    """Tests for ToolType enum."""

    def test_all_values_exist(self) -> None:
        assert ToolType.FILE_READ.value == "file_read"
        assert ToolType.FILE_WRITE.value == "file_write"
        assert ToolType.SHELL.value == "shell"
        assert ToolType.WEB_SEARCH.value == "web_search"

    def test_enum_from_string(self) -> None:
        assert ToolType("file_read") is ToolType.FILE_READ

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ToolType("nonexistent")


# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass and helpers."""

    def test_defaults(self) -> None:
        sc = SandboxConfig()
        assert sc.enabled is False
        assert sc.allowed_paths == []
        assert sc.allowed_commands == []
        assert sc.allowed_domains == []

    def test_from_dict_full(self) -> None:
        data = {
            "enabled": True,
            "allowed_paths": ["/tmp", "./output"],
            "allowed_commands": ["echo"],
            "allowed_domains": ["example.com"],
        }
        sc = SandboxConfig.from_dict(data)
        assert sc.enabled is True
        assert sc.allowed_paths == ["/tmp", "./output"]
        assert sc.allowed_commands == ["echo"]
        assert sc.allowed_domains == ["example.com"]

    def test_from_dict_partial(self) -> None:
        sc = SandboxConfig.from_dict({"enabled": True})
        assert sc.enabled is True
        assert sc.allowed_paths == []

    def test_from_dict_not_mapping_raises(self) -> None:
        with pytest.raises(ConfigError, match="sandbox.*mapping"):
            SandboxConfig.from_dict("not-a-dict")  # type: ignore[arg-type]

    def test_merge_unions_lists(self) -> None:
        base = SandboxConfig(
            enabled=False,
            allowed_paths=["/tmp"],
            allowed_commands=["echo"],
            allowed_domains=["a.com"],
        )
        override = SandboxConfig(
            enabled=True,
            allowed_paths=["/tmp", "./output"],
            allowed_commands=["ls"],
            allowed_domains=["b.com"],
        )
        merged = base.merge(override)
        assert merged.enabled is True
        assert "/tmp" in merged.allowed_paths
        assert "./output" in merged.allowed_paths
        assert "echo" in merged.allowed_commands
        assert "ls" in merged.allowed_commands
        assert "a.com" in merged.allowed_domains
        assert "b.com" in merged.allowed_domains

    def test_merge_deduplicates(self) -> None:
        base = SandboxConfig(allowed_paths=["/tmp"])
        override = SandboxConfig(allowed_paths=["/tmp", "./out"])
        merged = base.merge(override)
        assert merged.allowed_paths.count("/tmp") == 1

    def test_merge_enabled_comes_from_override(self) -> None:
        base = SandboxConfig(enabled=True)
        override = SandboxConfig(enabled=False)
        merged = base.merge(override)
        assert merged.enabled is False


# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


class TestToolDefinition:
    """Tests for ToolDefinition.from_dict."""

    def test_valid_tool(self) -> None:
        td = ToolDefinition.from_dict(
            {"name": "my_tool", "type": "shell", "description": "Run a command."},
            index=0,
        )
        assert td.name == "my_tool"
        assert td.tool_type is ToolType.SHELL
        assert td.description == "Run a command."

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ConfigError, match="name"):
            ToolDefinition.from_dict({"type": "shell"}, index=0)

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ConfigError, match="type"):
            ToolDefinition.from_dict({"name": "t"}, index=0)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ConfigError, match="not valid"):
            ToolDefinition.from_dict({"name": "t", "type": "unknown_type"}, index=1)

    def test_not_mapping_raises(self) -> None:
        with pytest.raises(ConfigError, match="tools\\[0\\].*mapping"):
            ToolDefinition.from_dict("bad", index=0)  # type: ignore[arg-type]

    def test_description_optional(self) -> None:
        td = ToolDefinition.from_dict({"name": "t", "type": "web_search"}, index=0)
        assert td.description == ""


# ---------------------------------------------------------------------------
# StepConfig
# ---------------------------------------------------------------------------


class TestStepConfig:
    """Tests for StepConfig.from_dict."""

    def test_valid_step_minimal(self) -> None:
        sc = StepConfig.from_dict(
            {"name": "s1", "prompt": "Do something"}, index=0
        )
        assert sc.name == "s1"
        assert sc.prompt == "Do something"
        assert sc.tools == []
        assert sc.output_variable is None
        assert sc.max_iterations == 5
        assert sc.sandbox is None

    def test_valid_step_full(self) -> None:
        sc = StepConfig.from_dict(
            {
                "name": "full_step",
                "description": "A full step",
                "prompt": "Hello {x}",
                "tools": ["read_file"],
                "output_variable": "my_var",
                "max_iterations": 10,
                "sandbox": {"enabled": True, "allowed_paths": ["/tmp"]},
            },
            index=0,
        )
        assert sc.description == "A full step"
        assert sc.tools == ["read_file"]
        assert sc.output_variable == "my_var"
        assert sc.max_iterations == 10
        assert sc.sandbox is not None
        assert sc.sandbox.enabled is True

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ConfigError, match="name"):
            StepConfig.from_dict({"prompt": "hi"}, index=0)

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ConfigError, match="prompt"):
            StepConfig.from_dict({"name": "s"}, index=0)

    def test_invalid_max_iterations_raises(self) -> None:
        with pytest.raises(ConfigError, match="max_iterations"):
            StepConfig.from_dict(
                {"name": "s", "prompt": "p", "max_iterations": "bad"}, index=0
            )

    def test_zero_max_iterations_raises(self) -> None:
        with pytest.raises(ConfigError, match=">= 1"):
            StepConfig.from_dict(
                {"name": "s", "prompt": "p", "max_iterations": 0}, index=0
            )

    def test_invalid_output_variable_raises(self) -> None:
        with pytest.raises(ConfigError, match="valid Python identifier"):
            StepConfig.from_dict(
                {"name": "s", "prompt": "p", "output_variable": "not-valid!"},
                index=0,
            )

    def test_not_mapping_raises(self) -> None:
        with pytest.raises(ConfigError, match="steps\\[2\\].*mapping"):
            StepConfig.from_dict(123, index=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_workflow_from_string
# ---------------------------------------------------------------------------


class TestLoadWorkflowFromString:
    """Tests for load_workflow_from_string."""

    def test_minimal_parses_correctly(self) -> None:
        cfg = _minimal()
        assert cfg.name == "Test Workflow"
        assert cfg.model == "llama3"
        assert len(cfg.tools) == 1
        assert len(cfg.steps) == 1
        assert cfg.variables == {"name": "world"}

    def test_tool_parsed_correctly(self) -> None:
        cfg = _minimal()
        tool = cfg.tools[0]
        assert tool.name == "read_file"
        assert tool.tool_type is ToolType.FILE_READ
        assert "Read a file" in tool.description

    def test_step_parsed_correctly(self) -> None:
        cfg = _minimal()
        step = cfg.steps[0]
        assert step.name == "step_one"
        assert "{name}" in step.prompt
        assert step.output_variable == "result"
        assert step.max_iterations == 5

    def test_sandbox_defaults_to_disabled(self) -> None:
        cfg = _minimal()
        assert cfg.sandbox.enabled is False

    def test_empty_document_raises(self) -> None:
        with pytest.raises(ConfigError, match="empty"):
            load_workflow_from_string("")

    def test_non_mapping_top_level_raises(self) -> None:
        with pytest.raises(ConfigError, match="mapping"):
            load_workflow_from_string("- just\n- a\n- list")

    def test_invalid_yaml_raises(self) -> None:
        with pytest.raises(ConfigError, match="YAML parse error"):
            load_workflow_from_string("{bad: yaml: document}")

    def test_missing_name_raises(self) -> None:
        yaml_text = textwrap.dedent("""\
            model: llama3
            steps:
              - name: s
                prompt: p
        """)
        with pytest.raises(ConfigError, match="name"):
            load_workflow_from_string(yaml_text)

    def test_missing_model_raises(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            steps:
              - name: s
                prompt: p
        """)
        with pytest.raises(ConfigError, match="model"):
            load_workflow_from_string(yaml_text)

    def test_tools_not_list_raises(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            model: llama3
            tools: not-a-list
            steps:
              - name: s
                prompt: p
        """)
        with pytest.raises(ConfigError, match="'tools' must be a list"):
            load_workflow_from_string(yaml_text)

    def test_steps_not_list_raises(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            model: llama3
            steps: not-a-list
        """)
        with pytest.raises(ConfigError, match="'steps' must be a list"):
            load_workflow_from_string(yaml_text)

    def test_variables_not_mapping_raises(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            model: llama3
            variables:
              - a
              - b
            steps:
              - name: s
                prompt: p
        """)
        with pytest.raises(ConfigError, match="'variables' must be a mapping"):
            load_workflow_from_string(yaml_text)

    def test_sandbox_block_parsed(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            model: llama3
            sandbox:
              enabled: true
              allowed_paths:
                - /tmp
            steps:
              - name: s
                prompt: p
        """)
        cfg = load_workflow_from_string(yaml_text)
        assert cfg.sandbox.enabled is True
        assert "/tmp" in cfg.sandbox.allowed_paths

    def test_all_tool_types_parse(self) -> None:
        yaml_text = textwrap.dedent("""\
            name: wf
            model: llama3
            tools:
              - name: t1
                type: file_read
              - name: t2
                type: file_write
              - name: t3
                type: shell
              - name: t4
                type: web_search
            steps:
              - name: s
                prompt: p
                tools: [t1, t2, t3, t4]
        """)
        cfg = load_workflow_from_string(yaml_text)
        types = [t.tool_type for t in cfg.tools]
        assert ToolType.FILE_READ in types
        assert ToolType.FILE_WRITE in types
        assert ToolType.SHELL in types
        assert ToolType.WEB_SEARCH in types


# ---------------------------------------------------------------------------
# validate_workflow
# ---------------------------------------------------------------------------


class TestValidateWorkflow:
    """Tests for validate_workflow semantic checks."""

    def test_blank_name_raises(self) -> None:
        cfg = _minimal()
        cfg.name = "   "
        with pytest.raises(ConfigError, match="name.*blank"):
            validate_workflow(cfg)

    def test_blank_model_raises(self) -> None:
        cfg = _minimal()
        cfg.model = ""
        with pytest.raises(ConfigError, match="model.*blank"):
            validate_workflow(cfg)

    def test_no_steps_raises(self) -> None:
        cfg = _minimal()
        cfg.steps = []
        with pytest.raises(ConfigError, match="at least one step"):
            validate_workflow(cfg)

    def test_undefined_tool_ref_raises(self) -> None:
        cfg = _minimal()
        cfg.steps[0].tools = ["nonexistent_tool"]
        with pytest.raises(ConfigError, match="undefined tool"):
            validate_workflow(cfg)

    def test_duplicate_step_names_raises(self) -> None:
        cfg = _minimal()
        duplicate = StepConfig(name="step_one", prompt="Another prompt")
        cfg.steps.append(duplicate)
        with pytest.raises(ConfigError, match="Duplicate step name"):
            validate_workflow(cfg)

    def test_invalid_output_variable_in_validate(self) -> None:
        """Programmatically constructed bad output_variable caught by validate."""
        cfg = _minimal()
        cfg.steps[0].output_variable = "123bad"
        with pytest.raises(ConfigError, match="valid Python identifier"):
            validate_workflow(cfg)

    def test_valid_config_passes(self) -> None:
        cfg = _minimal()
        validate_workflow(cfg)  # should not raise


# ---------------------------------------------------------------------------
# WorkflowConfig helpers
# ---------------------------------------------------------------------------


class TestWorkflowConfigHelpers:
    """Tests for WorkflowConfig.get_tool and effective_sandbox."""

    def test_get_tool_found(self) -> None:
        cfg = _minimal()
        tool = cfg.get_tool("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_get_tool_not_found(self) -> None:
        cfg = _minimal()
        assert cfg.get_tool("does_not_exist") is None

    def test_effective_sandbox_no_step_override(self) -> None:
        cfg = _minimal()
        cfg.sandbox = SandboxConfig(enabled=True, allowed_paths=["/tmp"])
        step = cfg.steps[0]
        step.sandbox = None
        eff = cfg.effective_sandbox(step)
        assert eff.enabled is True
        assert "/tmp" in eff.allowed_paths

    def test_effective_sandbox_with_step_override(self) -> None:
        cfg = _minimal()
        cfg.sandbox = SandboxConfig(
            enabled=False,
            allowed_paths=["/tmp"],
            allowed_commands=["echo"],
        )
        cfg.steps[0].sandbox = SandboxConfig(
            enabled=True,
            allowed_paths=["./output"],
            allowed_commands=[],
        )
        eff = cfg.effective_sandbox(cfg.steps[0])
        assert eff.enabled is True
        # Paths unioned
        assert "/tmp" in eff.allowed_paths
        assert "./output" in eff.allowed_paths
        # Commands come from base only (override list is empty)
        assert "echo" in eff.allowed_commands


# ---------------------------------------------------------------------------
# load_workflow (file-based)
# ---------------------------------------------------------------------------


class TestLoadWorkflowFile:
    """Tests for load_workflow (reading from a real file)."""

    def test_load_valid_file(self, tmp_path: Path) -> None:
        wf_file = tmp_path / "workflow.yaml"
        wf_file.write_text(MINIMAL_YAML, encoding="utf-8")
        cfg = load_workflow(wf_file)
        assert cfg.name == "Test Workflow"
        assert cfg.source_path == wf_file.resolve()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_workflow(tmp_path / "no_such_file.yaml")

    def test_directory_as_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a file"):
            load_workflow(tmp_path)

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        """On POSIX systems a file with mode 000 cannot be read."""
        import sys

        if sys.platform == "win32":
            pytest.skip("Permission mode 000 not supported on Windows")
        wf_file = tmp_path / "locked.yaml"
        wf_file.write_text(MINIMAL_YAML, encoding="utf-8")
        wf_file.chmod(0o000)
        try:
            with pytest.raises(ConfigError, match="Cannot read"):
                load_workflow(wf_file)
        finally:
            wf_file.chmod(0o644)

    def test_load_example_summarize_files(self) -> None:
        """The bundled example should parse without errors."""
        example_path = (
            Path(__file__).parent.parent / "examples" / "summarize_files.yaml"
        )
        if not example_path.exists():
            pytest.skip("examples/summarize_files.yaml not present")
        cfg = load_workflow(example_path)
        assert cfg.name
        assert cfg.model
        assert len(cfg.steps) >= 1

    def test_load_example_research_and_report(self) -> None:
        """The bundled example should parse without errors."""
        example_path = (
            Path(__file__).parent.parent / "examples" / "research_and_report.yaml"
        )
        if not example_path.exists():
            pytest.skip("examples/research_and_report.yaml not present")
        cfg = load_workflow(example_path)
        assert cfg.name
        assert cfg.model
        assert len(cfg.steps) >= 1
