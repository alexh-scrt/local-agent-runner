"""local_agent_runner — A minimal CLI harness for multi-step agentic workflows
powered by a local LLM via Ollama.

This package provides:
- YAML-based workflow definition and validation (config)
- Sandboxed tool execution with allow-lists (sandbox, tools)
- Structured logging to JSON and Rich terminal (logger)
- Core agentic loop with Ollama integration (agent)
- Argparse-based CLI entry point (cli)
"""

__version__ = "0.1.0"
__author__ = "local_agent_runner contributors"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
