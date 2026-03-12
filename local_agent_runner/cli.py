"""Argparse-based CLI entry point for local_agent_runner.

This module implements the ``local-agent-runner`` command with three
sub-commands:

- ``run``      — load a YAML workflow and execute it via the agentic loop.
- ``validate`` — load and validate a YAML workflow without running it.
- ``version``  — print the installed package version.

All user-facing output is produced via Rich; structured run logs are written
via :class:`~local_agent_runner.logger.RunLogger`.

Entry point
-----------
The ``main()`` function is registered as the ``local-agent-runner`` console
script in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from local_agent_runner import __version__
from local_agent_runner.agent import (
    DEFAULT_OLLAMA_URL,
    AgentError,
    OllamaClient,
    run_workflow,
)
from local_agent_runner.config import ConfigError, load_workflow

# Rich console for stderr output (all CLI status messages go here).
_console = Console(stderr=True, highlight=False)
# Rich console for stdout (used for plain-text output like version).
_stdout_console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="local-agent-runner",
        description="Local Agent Runner — multi-step agentic workflows via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  local-agent-runner run examples/summarize_files.yaml\n"
            "  local-agent-runner run examples/research_and_report.yaml --sandbox\n"
            "  local-agent-runner validate examples/summarize_files.yaml\n"
            "  local-agent-runner version\n"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{run,validate,version}",
    )

    # ------------------------------------------------------------------
    # run sub-command
    # ------------------------------------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a YAML workflow",
        description="Load a YAML workflow file and execute it using the local Ollama LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "workflow",
        metavar="WORKFLOW",
        type=str,
        help="Path to the YAML workflow file.",
    )
    sandbox_group = run_parser.add_mutually_exclusive_group()
    sandbox_group.add_argument(
        "--sandbox",
        action="store_true",
        default=None,
        dest="sandbox",
        help="Enable sandboxed execution (overrides YAML sandbox.enabled setting).",
    )
    sandbox_group.add_argument(
        "--no-sandbox",
        action="store_false",
        dest="sandbox",
        help="Disable sandboxed execution (overrides YAML sandbox.enabled setting).",
    )
    run_parser.add_argument(
        "--ollama-url",
        metavar="URL",
        default=DEFAULT_OLLAMA_URL,
        dest="ollama_url",
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    run_parser.add_argument(
        "--log-file",
        metavar="FILE",
        default=None,
        dest="log_file",
        help="Write structured NDJSON log to FILE.",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Increase output verbosity (show all events in terminal).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Parse and validate workflow without calling Ollama.",
    )
    run_parser.add_argument(
        "--timeout",
        metavar="SECONDS",
        type=float,
        default=120.0,
        dest="timeout",
        help="Ollama HTTP request timeout in seconds (default: 120).",
    )

    # ------------------------------------------------------------------
    # validate sub-command
    # ------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a YAML workflow without running it",
        description="Load and validate a YAML workflow file, reporting any errors.",
    )
    validate_parser.add_argument(
        "workflow",
        metavar="WORKFLOW",
        type=str,
        help="Path to the YAML workflow file.",
    )
    validate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print full workflow summary on success.",
    )

    # ------------------------------------------------------------------
    # version sub-command
    # ------------------------------------------------------------------
    subparsers.add_parser(
        "version",
        help="Print version information",
        description="Print the installed local_agent_runner version and exit.",
    )

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    """Handler for the ``run`` sub-command.

    Parameters
    ----------
    args:
        Parsed argument namespace.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    workflow_path = Path(args.workflow)

    # --- Load workflow ---
    _console.print(
        f"[dim]Loading workflow:[/dim] [bold]{workflow_path}[/bold]"
    )
    try:
        workflow = load_workflow(workflow_path)
    except ConfigError as exc:
        _console.print(
            Panel(
                f"[bold red]Configuration error:[/bold red] {exc}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        return 1

    # --- Apply CLI sandbox override ---
    if args.sandbox is True:
        workflow.sandbox.enabled = True  # type: ignore[attr-defined]
        # SandboxConfig is a dataclass so we rebuild with enabled=True
        from local_agent_runner.config import SandboxConfig
        workflow.sandbox = SandboxConfig(
            enabled=True,
            allowed_paths=workflow.sandbox.allowed_paths,
            allowed_commands=workflow.sandbox.allowed_commands,
            allowed_domains=workflow.sandbox.allowed_domains,
        )
    elif args.sandbox is False:
        from local_agent_runner.config import SandboxConfig
        workflow.sandbox = SandboxConfig(
            enabled=False,
            allowed_paths=workflow.sandbox.allowed_paths,
            allowed_commands=workflow.sandbox.allowed_commands,
            allowed_domains=workflow.sandbox.allowed_domains,
        )

    # --- Print workflow summary ---
    _console.print(
        Panel(
            Text.assemble(
                ("Workflow:  ", "dim"),
                (workflow.name, "bold cyan"),
                ("\nModel:     ", "dim"),
                (workflow.model, "bold"),
                ("\nSteps:     ", "dim"),
                (str(len(workflow.steps)), "bold"),
                ("\nTools:     ", "dim"),
                (", ".join(t.name for t in workflow.tools) or "(none)", "bold"),
                ("\nSandbox:   ", "dim"),
                (
                    "[green]enabled[/green]" if workflow.sandbox.enabled
                    else "[yellow]disabled[/yellow]",
                    "",
                ),
                ("\nDry-run:   ", "dim"),
                (
                    "[green]yes[/green]" if args.dry_run else "[yellow]no[/yellow]",
                    "",
                ),
            ),
            title="[bold cyan]local-agent-runner[/bold cyan]",
            border_style="cyan",
        )
    )

    # --- Health-check Ollama (unless dry-run) ---
    if not args.dry_run:
        client = OllamaClient(base_url=args.ollama_url)
        _console.print(
            f"[dim]Checking Ollama at:[/dim] [bold]{args.ollama_url}[/bold]"
        )
        if not client.health_check():
            _console.print(
                Panel(
                    f"[bold red]Cannot reach Ollama at {args.ollama_url}.[/bold red]\n"
                    "Make sure Ollama is running: https://ollama.ai/download\n"
                    "Or pass a different URL with --ollama-url.",
                    title="[bold red]Ollama Unreachable[/bold red]",
                    border_style="red",
                )
            )
            return 1

    # --- Execute ---
    try:
        result = run_workflow(
            workflow,
            ollama_url=args.ollama_url,
            log_file=args.log_file,
            verbose=args.verbose,
            dry_run=args.dry_run,
            ollama_timeout=args.timeout,
        )
    except AgentError as exc:
        _console.print(
            Panel(
                f"[bold red]Agent error:[/bold red] {exc}",
                title="[bold red]Run Failed[/bold red]",
                border_style="red",
            )
        )
        return 1
    except KeyboardInterrupt:
        _console.print("\n[bold yellow]Interrupted by user.[/bold yellow]")
        return 130
    except Exception as exc:
        _console.print(
            Panel(
                f"[bold red]Unexpected error:[/bold red] {exc}",
                title="[bold red]Run Failed[/bold red]",
                border_style="red",
            )
        )
        return 1

    # --- Report outcome ---
    if result["success"]:
        _console.print(
            Panel(
                Text.assemble(
                    ("[bold green]Workflow completed successfully.[/bold green]\n", ""),
                    (f"Steps completed: {result['steps_completed']}", "dim"),
                    (
                        f"\nLog file: {args.log_file}" if args.log_file else "",
                        "dim",
                    ),
                ),
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )
        return 0
    else:
        _console.print(
            Panel(
                Text.assemble(
                    ("[bold red]Workflow did not complete successfully.[/bold red]\n", ""),
                    (f"Steps completed: {result['steps_completed']} / "
                     f"{len(workflow.steps)}\n", "dim"),
                    (result.get("error", ""), "red"),
                ),
                title="[bold red]Failure[/bold red]",
                border_style="red",
            )
        )
        return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handler for the ``validate`` sub-command.

    Parameters
    ----------
    args:
        Parsed argument namespace.

    Returns
    -------
    int
        Exit code (0 = valid, 1 = invalid).
    """
    workflow_path = Path(args.workflow)

    _console.print(
        f"[dim]Validating:[/dim] [bold]{workflow_path}[/bold]"
    )

    try:
        workflow = load_workflow(workflow_path)
    except ConfigError as exc:
        _console.print(
            Panel(
                f"[bold red]Validation failed:[/bold red]\n{exc}",
                title="[bold red]Invalid Workflow[/bold red]",
                border_style="red",
            )
        )
        return 1

    # Build success message
    step_names = "\n".join(
        f"  [{i + 1}] {s.name}" + (f" → {s.output_variable}" if s.output_variable else "")
        for i, s in enumerate(workflow.steps)
    )
    tool_names = ", ".join(t.name for t in workflow.tools) or "(none)"

    details = (
        f"Name:    {workflow.name}\n"
        f"Model:   {workflow.model}\n"
        f"Tools:   {tool_names}\n"
        f"Steps ({len(workflow.steps)}):\n{step_names}"
    )
    if workflow.description:
        details = f"Desc:    {workflow.description.strip()}\n" + details

    _console.print(
        Panel(
            f"[bold green]Workflow is valid.[/bold green]\n\n{details}"
            if args.verbose
            else "[bold green]Workflow is valid.[/bold green]",
            title="[bold green]Valid[/bold green]",
            border_style="green",
        )
    )
    return 0


def _cmd_version() -> int:
    """Handler for the ``version`` sub-command.

    Returns
    -------
    int
        Always ``0``.
    """
    _stdout_console.print(
        f"local-agent-runner [bold cyan]{__version__}[/bold cyan]"
    )
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point registered in ``pyproject.toml``.

    Parameters
    ----------
    argv:
        Optional list of argument strings (defaults to ``sys.argv[1:]`` when
        ``None``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return _cmd_run(args)
    elif args.command == "validate":
        return _cmd_validate(args)
    elif args.command == "version":
        return _cmd_version()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
