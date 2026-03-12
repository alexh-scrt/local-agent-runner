# local_agent_runner

> Run multi-step AI agents entirely on your own hardware — no API keys, no cloud, no compromise.

`local_agent_runner` is a minimal CLI harness for defining and executing multi-step agentic workflows powered by a local LLM via [Ollama](https://ollama.ai). You describe your workflow in a simple YAML file, and the runner handles prompt chaining, tool execution, and structured logging — all on your local machine. Built for developers who want to prototype and iterate on agentic pipelines without cloud dependencies or vendor lock-in.

---

## Quick Start

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.ai) installed and running, and at least one model pulled (e.g. `ollama pull llama3`).

```bash
# Install from source
git clone https://github.com/your-org/local_agent_runner.git
cd local_agent_runner
pip install -e .

# Validate a workflow definition
local-agent-runner validate examples/summarize_files.yaml

# Run a workflow
local-agent-runner run examples/summarize_files.yaml

# Run in sandboxed mode (enforces path/command allow-lists)
local-agent-runner run examples/summarize_files.yaml --sandbox

# Check version
local-agent-runner version
```

That's it. If Ollama is running and `llama3` is available, the workflow will execute and write structured logs to your terminal.

---

## Features

- **YAML-defined workflows** — declare model, tools, and prompt templates per step; chain step outputs as variables into subsequent prompts
- **Native Ollama integration** — fully local LLM inference via the Ollama HTTP API with zero cloud dependency
- **Built-in tool stubs** — file read/write, shell command execution, and HTTP-based web search, all callable by the LLM
- **Sandboxed execution mode** — configurable path, command, and domain allow-lists that block dangerous operations before they run
- **Structured per-run logging** — every LLM prompt, response, and tool invocation captured as NDJSON on disk and rendered with Rich in the terminal

---

## Usage Examples

### Running the bundled examples

```bash
# Summarize files in a directory and write output/summary.md
local-agent-runner run examples/summarize_files.yaml

# Web research pipeline: search → synthesize → write output/report.md
local-agent-runner run examples/research_and_report.yaml --sandbox
```

### Writing your own workflow YAML

```yaml
# my_workflow.yaml
name: Extract TODOs
description: Scan source files and extract all TODO comments into a report.
model: llama3

sandbox:
  enabled: true
  allowed_paths:
    - ./src
    - ./output
  allowed_commands: []       # no shell commands permitted
  allowed_domains: []        # no web requests permitted

steps:
  - name: read_source
    description: Read the main source file.
    tools:
      - name: read_file
        type: file_read
    prompt: |
      Read the file at ./src/main.py and return its full contents.

  - name: extract_todos
    description: Extract TODO comments and write a report.
    tools:
      - name: write_file
        type: file_write
    prompt: |
      Here is the source code:

      {{ read_source }}

      Find every TODO comment and write a Markdown checklist to
      ./output/todos.md.
```

```bash
local-agent-runner run my_workflow.yaml
```

### Validating without running

```bash
local-agent-runner validate my_workflow.yaml
# ✓ Workflow 'Extract TODOs' is valid (2 steps, 2 tools)
```

### Dry-run mode (parse + plan, no LLM calls)

```bash
local-agent-runner run my_workflow.yaml --dry-run
```

---

## Project Structure

```
local_agent_runner/
├── pyproject.toml                  # Project metadata, dependencies, CLI entry point
├── README.md
│
├── local_agent_runner/
│   ├── __init__.py                 # Package initializer; exposes __version__
│   ├── cli.py                      # Argparse CLI: run / validate / version sub-commands
│   ├── config.py                   # YAML loader and typed dataclass models
│   ├── agent.py                    # Core agentic loop + OllamaClient
│   ├── tools.py                    # Tool stubs: file_read, file_write, shell, web_search
│   ├── sandbox.py                  # Sandboxed execution context and allow-list enforcement
│   └── logger.py                   # Structured NDJSON + Rich terminal logger
│
├── examples/
│   ├── summarize_files.yaml        # Read files → produce a Markdown summary
│   └── research_and_report.yaml   # Web search → synthesize → write report
│
└── tests/
    ├── __init__.py
    ├── test_config.py              # YAML loading, validation, error handling
    ├── test_tools.py               # Tool stubs, sandbox rejection, allowed paths
    ├── test_sandbox.py             # SandboxContext allow-lists and action recording
    ├── test_logger.py              # RunLogger output, NDJSON format, lifecycle
    └── test_agent.py               # Agentic loop integration (mocked Ollama)
```

---

## Configuration

Workflow files are standard YAML. The top-level fields and their defaults are:

| Field | Required | Description |
|---|---|---|
| `name` | ✅ | Human-readable workflow name |
| `description` | ✅ | Brief description of what the workflow does |
| `model` | ✅ | Ollama model name to use (e.g. `llama3`, `mistral`) |
| `steps` | ✅ | Ordered list of step definitions (see below) |
| `sandbox` | ❌ | Global sandbox settings (can be overridden per step) |
| `max_iterations` | ❌ | Max tool-call iterations per step (default: `10`) |

**Sandbox block:**

```yaml
sandbox:
  enabled: true                    # false = no restrictions (default)
  allowed_paths:                   # file read/write restricted to these paths
    - ./data
    - ./output
  allowed_commands:                # shell commands restricted to this list
    - grep
    - cat
  allowed_domains:                 # web requests restricted to these domains
    - api.example.com
    - en.wikipedia.org
```

**Step block:**

```yaml
steps:
  - name: my_step                  # unique step identifier (used as variable name)
    description: What this step does
    tools:                         # tools the LLM may call in this step
      - name: read_file
        type: file_read            # file_read | file_write | shell | web_search
    prompt: |                      # prompt template; use {{ step_name }} for prior outputs
      Do something useful.
    sandbox:                       # optional per-step sandbox override
      allowed_paths:
        - ./data
```

**Runtime flags:**

```bash
local-agent-runner run <workflow.yaml> [OPTIONS]

  --sandbox          Enable sandboxed execution (overrides workflow setting)
  --dry-run          Parse and plan the workflow without calling Ollama
  --log-file PATH    Write structured NDJSON logs to this file
  --verbose          Print full LLM responses to the terminal
  --ollama-url URL   Ollama base URL (default: http://localhost:11434)
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
