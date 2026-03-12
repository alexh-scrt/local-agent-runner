# local_agent_runner

A minimal, open-source CLI harness for defining and executing multi-step agentic
workflows powered entirely by a **local LLM via [Ollama](https://ollama.ai)**.
No cloud dependencies. No API keys. Just YAML, Python, and your local hardware.

---

## Features

- **YAML-defined workflows** — declare model, tools, and prompt templates per step
- **Native Ollama integration** — fully local LLM inference via the Ollama HTTP API
- **Built-in tool stubs** — file read/write, shell commands, and HTTP-based web search
- **Sandboxed execution** — configurable allow-lists block dangerous paths/commands
- **Structured logging** — every LLM response and tool call captured as JSON + Rich terminal output

---

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Ollama** installed and running locally: <https://ollama.ai/download>
3. At least one model pulled, e.g. `ollama pull llama3`

### Installation

```bash
# From source
git clone https://github.com/example/local_agent_runner.git
cd local_agent_runner
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

### Run Your First Workflow

```bash
# Run a built-in example
local-agent-runner run examples/summarize_files.yaml

# Enable sandbox mode (blocks writes outside allowed paths)
local-agent-runner run examples/research_and_report.yaml --sandbox

# Override the Ollama base URL
local-agent-runner run examples/summarize_files.yaml --ollama-url http://localhost:11434

# Save structured log to a file
local-agent-runner run examples/summarize_files.yaml --log-file run_log.json

# Increase verbosity
local-agent-runner run examples/summarize_files.yaml --verbose
```

---

## YAML Schema Reference

Every workflow is a single YAML file. Here is the complete schema:

```yaml
# workflow.yaml
name: string                  # Human-readable workflow name (required)
description: string           # Optional description
model: string                 # Ollama model name, e.g. "llama3" (required)

# Global sandbox settings (can be overridden per step)
sandbox:
  enabled: bool               # Default: false
  allowed_paths:              # List of path prefixes permitted for file I/O
    - "/tmp"
    - "./output"
  allowed_commands:           # List of shell command prefixes permitted
    - "echo"
    - "ls"
  allowed_domains:            # List of domains permitted for web search/fetch
    - "example.com"

# Tool definitions available to all steps
tools:
  - name: string              # Unique tool identifier
    description: string       # Shown to the LLM in the system prompt
    type: file_read | file_write | shell | web_search

# Ordered list of steps
steps:
  - name: string              # Step name (required)
    description: string       # Optional
    prompt: string            # Prompt template; use {variable} for substitution
    tools:                    # Subset of top-level tools available in this step
      - string
    output_variable: string   # Store LLM response under this variable name
    max_iterations: int       # Max tool-call rounds per step (default: 5)
    sandbox:                  # Step-level sandbox override
      enabled: bool
      allowed_paths: []
      allowed_commands: []
      allowed_domains: []

# Initial variables available for prompt substitution
variables:
  key: value
```

### Supported Tool Types

| Type | Description |
|------|-------------|
| `file_read` | Read the contents of a file at a given path |
| `file_write` | Write text content to a file at a given path |
| `shell` | Execute a shell command and return stdout/stderr |
| `web_search` | Perform an HTTP GET and return the response body |

---

## Example Workflows

### Summarize Files (`examples/summarize_files.yaml`)

Reads one or more source files and asks the LLM to produce a Markdown summary.

```bash
local-agent-runner run examples/summarize_files.yaml
```

### Research and Report (`examples/research_and_report.yaml`)

Combines web search and file-write tools to research a topic and save a report.

```bash
local-agent-runner run examples/research_and_report.yaml --sandbox
```

---

## CLI Reference

```
usage: local-agent-runner [-h] {run,validate,version} ...

Local Agent Runner — multi-step agentic workflows via Ollama

positional arguments:
  {run,validate,version}
    run                 Execute a YAML workflow
    validate            Validate a YAML workflow without running it
    version             Print version information

run flags:
  workflow              Path to the YAML workflow file
  --sandbox             Enable sandboxed execution (overrides YAML setting)
  --no-sandbox          Disable sandboxed execution
  --ollama-url URL      Ollama base URL (default: http://localhost:11434)
  --log-file FILE       Write structured JSON log to FILE
  --verbose, -v         Increase output verbosity
  --dry-run             Parse and validate workflow without calling Ollama
```

---

## Project Structure

```
local_agent_runner/
├── __init__.py        # Package version
├── cli.py             # Argparse entry point
├── config.py          # YAML loader and typed dataclasses
├── agent.py           # Core agentic loop (Ollama + tool orchestration)
├── tools.py           # Tool stubs: file I/O, shell, web search
├── sandbox.py         # Sandbox context with allow-lists
└── logger.py          # JSON + Rich structured logger
examples/
├── summarize_files.yaml
└── research_and_report.yaml
tests/
├── __init__.py
├── test_config.py
├── test_tools.py
└── test_agent.py
```

---

## Development

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with verbose output
pytest -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
