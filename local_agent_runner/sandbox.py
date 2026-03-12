"""Sandboxed execution context for local_agent_runner.

This module provides a :class:`SandboxContext` that wraps tool execution with
allow-list enforcement and action recording.  When sandbox mode is active,
every file I/O, shell, and web-search operation is checked against configured
allow-lists before being permitted; violations raise :class:`SandboxViolation`.
All attempted actions (permitted or denied) are recorded as structured
:class:`ActionRecord` objects for later inspection and logging.

Public API
----------
- ``SandboxViolation``   — exception raised when an action is blocked.
- ``ActionType``         — enum of recordable action kinds.
- ``ActionRecord``       — immutable record of a single sandbox action.
- ``SandboxContext``     — main context object; enforces rules and records
  actions.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from local_agent_runner.config import SandboxConfig


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SandboxViolation(PermissionError):
    """Raised when a tool action is blocked by the sandbox allow-lists.

    Attributes
    ----------
    action_type:
        The :class:`ActionType` of the blocked action.
    detail:
        Human-readable description of what was blocked and why.
    """

    def __init__(self, action_type: "ActionType", detail: str) -> None:
        self.action_type = action_type
        self.detail = detail
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Kinds of actions that the sandbox tracks."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL = "shell"
    WEB_REQUEST = "web_request"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """Immutable record of a single action attempted within a sandbox context.

    Attributes
    ----------
    action_type:
        The kind of action attempted.
    target:
        The path, command string, or URL that was the subject of the action.
    permitted:
        ``True`` if the sandbox allowed the action, ``False`` if it was
        blocked.
    reason:
        Human-readable note explaining why the action was permitted or denied.
    timestamp:
        UTC timestamp of when the record was created.
    metadata:
        Optional extra key/value pairs captured alongside the record (e.g.
        resolved absolute path, matched allow-list entry).
    """

    action_type: ActionType
    target: str
    permitted: bool
    reason: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "permitted": self.permitted,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# SandboxContext
# ---------------------------------------------------------------------------


class SandboxContext:
    """Enforces sandbox allow-lists and records every action attempt.

    The context can be used as a plain object or as a context manager::

        with SandboxContext(config) as ctx:
            ctx.check_file_read("/tmp/foo.txt")

    When ``config.enabled`` is ``False`` all checks pass immediately (the
    action is still recorded with ``permitted=True``).

    Parameters
    ----------
    config:
        :class:`~local_agent_runner.config.SandboxConfig` that supplies the
        ``enabled`` flag and the three allow-lists.
    working_directory:
        The working directory used to resolve relative paths.  Defaults to
        the process CWD at construction time.

    Attributes
    ----------
    records:
        Chronological list of :class:`ActionRecord` objects accumulated
        during this context's lifetime.
    """

    def __init__(
        self,
        config: SandboxConfig,
        working_directory: Optional[str | Path] = None,
    ) -> None:
        self._config = config
        self._cwd = Path(
            working_directory if working_directory is not None else os.getcwd()
        ).resolve()
        self._records: List[ActionRecord] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SandboxConfig:
        """The :class:`SandboxConfig` this context enforces."""
        return self._config

    @property
    def enabled(self) -> bool:
        """``True`` when sandbox enforcement is active."""
        return self._config.enabled

    @property
    def records(self) -> List[ActionRecord]:
        """List of all :class:`ActionRecord` objects recorded so far.

        Returns a *copy* to prevent external mutation.
        """
        return list(self._records)

    @property
    def denied_records(self) -> List[ActionRecord]:
        """Subset of :attr:`records` where ``permitted`` is ``False``."""
        return [r for r in self._records if not r.permitted]

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "SandboxContext":
        return self

    def __exit__(self, *_: Any) -> None:  # noqa: ANN002
        pass

    # ------------------------------------------------------------------
    # Public check methods
    # ------------------------------------------------------------------

    def check_file_read(self, path: str | Path) -> Path:
        """Assert that reading *path* is permitted, then return its resolved form.

        Parameters
        ----------
        path:
            The filesystem path the tool wants to read.

        Returns
        -------
        Path
            The resolved absolute path.

        Raises
        ------
        SandboxViolation
            If the sandbox is enabled and *path* is not under any entry in
            ``allowed_paths``.
        """
        resolved = self._resolve_path(path)
        if not self._config.enabled:
            self._record(
                ActionType.FILE_READ,
                str(path),
                permitted=True,
                reason="sandbox disabled",
                metadata={"resolved": str(resolved)},
            )
            return resolved

        matched = self._match_path(resolved)
        if matched is not None:
            self._record(
                ActionType.FILE_READ,
                str(path),
                permitted=True,
                reason=f"path matches allow-list entry: {matched!r}",
                metadata={"resolved": str(resolved), "matched_rule": matched},
            )
            return resolved

        self._record(
            ActionType.FILE_READ,
            str(path),
            permitted=False,
            reason=(
                f"path {str(resolved)!r} is not under any allowed path; "
                f"allowed: {self._config.allowed_paths}"
            ),
            metadata={"resolved": str(resolved)},
        )
        raise SandboxViolation(
            ActionType.FILE_READ,
            f"file_read blocked: {str(resolved)!r} is not under any allowed path. "
            f"Allowed paths: {self._config.allowed_paths}",
        )

    def check_file_write(self, path: str | Path) -> Path:
        """Assert that writing *path* is permitted, then return its resolved form.

        Parameters
        ----------
        path:
            The filesystem path the tool wants to write.

        Returns
        -------
        Path
            The resolved absolute path.

        Raises
        ------
        SandboxViolation
            If the sandbox is enabled and *path* is not under any entry in
            ``allowed_paths``.
        """
        resolved = self._resolve_path(path)
        if not self._config.enabled:
            self._record(
                ActionType.FILE_WRITE,
                str(path),
                permitted=True,
                reason="sandbox disabled",
                metadata={"resolved": str(resolved)},
            )
            return resolved

        matched = self._match_path(resolved)
        if matched is not None:
            self._record(
                ActionType.FILE_WRITE,
                str(path),
                permitted=True,
                reason=f"path matches allow-list entry: {matched!r}",
                metadata={"resolved": str(resolved), "matched_rule": matched},
            )
            return resolved

        self._record(
            ActionType.FILE_WRITE,
            str(path),
            permitted=False,
            reason=(
                f"path {str(resolved)!r} is not under any allowed path; "
                f"allowed: {self._config.allowed_paths}"
            ),
            metadata={"resolved": str(resolved)},
        )
        raise SandboxViolation(
            ActionType.FILE_WRITE,
            f"file_write blocked: {str(resolved)!r} is not under any allowed path. "
            f"Allowed paths: {self._config.allowed_paths}",
        )

    def check_shell_command(self, command: str) -> str:
        """Assert that executing *command* is permitted.

        Matching is performed against the first token (executable name) of the
        command string, compared with prefix matching against every entry in
        ``allowed_commands``.

        Parameters
        ----------
        command:
            The shell command string the tool wants to execute.

        Returns
        -------
        str
            The original *command* string (unchanged).

        Raises
        ------
        SandboxViolation
            If the sandbox is enabled and *command* does not match any entry
            in ``allowed_commands``.
        """
        if not self._config.enabled:
            self._record(
                ActionType.SHELL,
                command,
                permitted=True,
                reason="sandbox disabled",
            )
            return command

        matched = self._match_command(command)
        if matched is not None:
            self._record(
                ActionType.SHELL,
                command,
                permitted=True,
                reason=f"command matches allow-list entry: {matched!r}",
                metadata={"matched_rule": matched},
            )
            return command

        self._record(
            ActionType.SHELL,
            command,
            permitted=False,
            reason=(
                f"command {command!r} does not match any allowed command; "
                f"allowed: {self._config.allowed_commands}"
            ),
        )
        raise SandboxViolation(
            ActionType.SHELL,
            f"shell blocked: {command!r} does not match any allowed command. "
            f"Allowed commands: {self._config.allowed_commands}",
        )

    def check_web_request(self, url: str) -> str:
        """Assert that fetching *url* is permitted.

        Domain matching extracts the hostname from *url* and checks it against
        every entry in ``allowed_domains`` using suffix matching (so
        ``"example.com"`` allows ``"www.example.com"`` and
        ``"api.example.com"``).

        Parameters
        ----------
        url:
            The full URL the tool wants to fetch.

        Returns
        -------
        str
            The original *url* string (unchanged).

        Raises
        ------
        SandboxViolation
            If the sandbox is enabled and the hostname of *url* does not match
            any entry in ``allowed_domains``.
        """
        if not self._config.enabled:
            self._record(
                ActionType.WEB_REQUEST,
                url,
                permitted=True,
                reason="sandbox disabled",
            )
            return url

        hostname = self._extract_hostname(url)
        matched = self._match_domain(hostname)
        if matched is not None:
            self._record(
                ActionType.WEB_REQUEST,
                url,
                permitted=True,
                reason=f"domain {hostname!r} matches allow-list entry: {matched!r}",
                metadata={"hostname": hostname, "matched_rule": matched},
            )
            return url

        self._record(
            ActionType.WEB_REQUEST,
            url,
            permitted=False,
            reason=(
                f"domain {hostname!r} is not in allowed_domains; "
                f"allowed: {self._config.allowed_domains}"
            ),
            metadata={"hostname": hostname},
        )
        raise SandboxViolation(
            ActionType.WEB_REQUEST,
            f"web_request blocked: domain {hostname!r} is not allowed. "
            f"Allowed domains: {self._config.allowed_domains}",
        )

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    def clear_records(self) -> None:
        """Discard all accumulated :class:`ActionRecord` objects."""
        self._records.clear()

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of recorded actions.

        Returns
        -------
        Dict[str, Any]
            Keys: ``total``, ``permitted``, ``denied``, ``records``.
        """
        permitted = [r for r in self._records if r.permitted]
        denied = [r for r in self._records if not r.permitted]
        return {
            "total": len(self._records),
            "permitted": len(permitted),
            "denied": len(denied),
            "records": [r.to_dict() for r in self._records],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        action_type: ActionType,
        target: str,
        *,
        permitted: bool,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionRecord:
        """Create and store an :class:`ActionRecord`."""
        rec = ActionRecord(
            action_type=action_type,
            target=target,
            permitted=permitted,
            reason=reason,
            metadata=metadata or {},
        )
        self._records.append(rec)
        return rec

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve *path* relative to :attr:`_cwd`."""
        p = Path(path)
        if not p.is_absolute():
            p = self._cwd / p
        return p.resolve()

    def _match_path(self, resolved: Path) -> Optional[str]:
        """Return the first matching allow-list entry for *resolved*, or ``None``.

        Each entry in ``allowed_paths`` is itself resolved relative to
        :attr:`_cwd` before comparison.  A match occurs when *resolved* equals
        the allow-list path or is a descendant of it.

        Parameters
        ----------
        resolved:
            Absolute resolved target path.

        Returns
        -------
        Optional[str]
            The raw allow-list string that matched, or ``None``.
        """
        for entry in self._config.allowed_paths:
            allowed = self._resolve_path(entry)
            try:
                resolved.relative_to(allowed)
                return entry
            except ValueError:
                continue
        return None

    def _match_command(self, command: str) -> Optional[str]:
        """Return the first matching allow-list entry for *command*, or ``None``.

        The executable name is extracted via ``shlex.split`` and compared as a
        simple string prefix against each entry in ``allowed_commands``.

        Parameters
        ----------
        command:
            Raw shell command string.

        Returns
        -------
        Optional[str]
            The raw allow-list string that matched, or ``None``.
        """
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return None
        executable = os.path.basename(tokens[0])
        for entry in self._config.allowed_commands:
            # Allow exact match on the executable or full command prefix match
            entry_exe = os.path.basename(entry.split()[0]) if entry.strip() else ""
            if executable == entry_exe:
                return entry
            if command.startswith(entry):
                return entry
        return None

    @staticmethod
    def _extract_hostname(url: str) -> str:
        """Extract the hostname from *url*.

        Handles ``http://``, ``https://``, and bare hostnames gracefully.

        Parameters
        ----------
        url:
            Full URL string.

        Returns
        -------
        str
            Lower-cased hostname, without port.
        """
        # Strip scheme
        match = re.match(r"^https?://([^/:?#]+)", url, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        # No scheme — treat the first path component as the host
        return url.split("/")[0].split(":")[0].lower()

    def _match_domain(self, hostname: str) -> Optional[str]:
        """Return the first matching allowed-domain entry for *hostname*.

        Matching uses suffix logic: an allow-list entry of ``"example.com"``
        permits ``"example.com"`` and ``"sub.example.com"`` but not
        ``"notexample.com"``.

        Parameters
        ----------
        hostname:
            Lower-cased hostname to check.

        Returns
        -------
        Optional[str]
            The raw allow-list string that matched, or ``None``.
        """
        for entry in self._config.allowed_domains:
            entry_lower = entry.lower()
            if hostname == entry_lower:
                return entry
            if hostname.endswith("." + entry_lower):
                return entry
        return None
