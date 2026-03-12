"""Unit tests for local_agent_runner.sandbox.

Covers:
- ActionType enum values.
- ActionRecord.to_dict serialisation.
- SandboxContext with sandbox disabled (all actions pass).
- SandboxContext with sandbox enabled:
  - file_read: allowed and blocked paths.
  - file_write: allowed and blocked paths.
  - shell: allowed and blocked commands.
  - web_request: allowed and blocked domains, sub-domain matching.
- Action recording (records list, denied_records, summary).
- SandboxViolation attributes.
- Context-manager protocol.
- clear_records helper.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_agent_runner.config import SandboxConfig
from local_agent_runner.sandbox import (
    ActionRecord,
    ActionType,
    SandboxContext,
    SandboxViolation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def disabled_ctx(tmp_path: Path) -> SandboxContext:
    """A SandboxContext with sandbox disabled."""
    cfg = SandboxConfig(enabled=False)
    return SandboxContext(cfg, working_directory=tmp_path)


@pytest.fixture()
def enabled_ctx(tmp_path: Path) -> SandboxContext:
    """A SandboxContext with sandbox enabled and a simple allow-list."""
    cfg = SandboxConfig(
        enabled=True,
        allowed_paths=[str(tmp_path), "/tmp"],
        allowed_commands=["echo", "ls"],
        allowed_domains=["example.com", "api.test.org"],
    )
    return SandboxContext(cfg, working_directory=tmp_path)


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------


class TestActionType:
    def test_values(self) -> None:
        assert ActionType.FILE_READ.value == "file_read"
        assert ActionType.FILE_WRITE.value == "file_write"
        assert ActionType.SHELL.value == "shell"
        assert ActionType.WEB_REQUEST.value == "web_request"


# ---------------------------------------------------------------------------
# ActionRecord
# ---------------------------------------------------------------------------


class TestActionRecord:
    def test_to_dict_contains_required_keys(self) -> None:
        rec = ActionRecord(
            action_type=ActionType.FILE_READ,
            target="/tmp/foo.txt",
            permitted=True,
            reason="sandbox disabled",
        )
        d = rec.to_dict()
        assert d["action_type"] == "file_read"
        assert d["target"] == "/tmp/foo.txt"
        assert d["permitted"] is True
        assert d["reason"] == "sandbox disabled"
        assert "timestamp" in d
        assert "metadata" in d

    def test_frozen(self) -> None:
        rec = ActionRecord(
            action_type=ActionType.SHELL,
            target="echo hi",
            permitted=True,
            reason="ok",
        )
        with pytest.raises((AttributeError, TypeError)):
            rec.target = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SandboxContext — disabled mode
# ---------------------------------------------------------------------------


class TestSandboxContextDisabled:
    def test_file_read_allowed(self, disabled_ctx: SandboxContext, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        result = disabled_ctx.check_file_read(str(p))
        assert result == p.resolve()

    def test_file_write_allowed(self, disabled_ctx: SandboxContext, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        result = disabled_ctx.check_file_write(str(p))
        assert result == p.resolve()

    def test_shell_allowed(self, disabled_ctx: SandboxContext) -> None:
        cmd = disabled_ctx.check_shell_command("rm -rf /")
        assert cmd == "rm -rf /"

    def test_web_request_allowed(self, disabled_ctx: SandboxContext) -> None:
        url = disabled_ctx.check_web_request("https://evil.com/data")
        assert url == "https://evil.com/data"

    def test_records_created(self, disabled_ctx: SandboxContext, tmp_path: Path) -> None:
        disabled_ctx.check_file_read(str(tmp_path / "a.txt"))
        disabled_ctx.check_shell_command("rm -rf /")
        assert len(disabled_ctx.records) == 2
        assert all(r.permitted for r in disabled_ctx.records)


# ---------------------------------------------------------------------------
# SandboxContext — enabled / file paths
# ---------------------------------------------------------------------------


class TestSandboxContextFilePaths:
    def test_file_read_allowed_under_allowed_path(
        self, enabled_ctx: SandboxContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "sub" / "file.txt"
        result = enabled_ctx.check_file_read(str(p))
        assert result == p.resolve()

    def test_file_read_blocked_outside_allowed_path(
        self, enabled_ctx: SandboxContext
    ) -> None:
        with pytest.raises(SandboxViolation) as exc_info:
            enabled_ctx.check_file_read("/etc/passwd")
        assert exc_info.value.action_type == ActionType.FILE_READ
        assert "blocked" in str(exc_info.value).lower()

    def test_file_write_allowed_under_allowed_path(
        self, enabled_ctx: SandboxContext, tmp_path: Path
    ) -> None:
        p = tmp_path / "output.txt"
        result = enabled_ctx.check_file_write(str(p))
        assert result == p.resolve()

    def test_file_write_blocked_outside_allowed_path(
        self, enabled_ctx: SandboxContext
    ) -> None:
        with pytest.raises(SandboxViolation) as exc_info:
            enabled_ctx.check_file_write("/etc/cron.d/evil")
        assert exc_info.value.action_type == ActionType.FILE_WRITE

    def test_allowed_path_slash_tmp(
        self, enabled_ctx: SandboxContext
    ) -> None:
        # /tmp is explicitly in allowed_paths
        result = enabled_ctx.check_file_read("/tmp/safe.txt")
        assert str(result).startswith("/tmp")

    def test_relative_path_resolved_against_cwd(
        self, enabled_ctx: SandboxContext, tmp_path: Path
    ) -> None:
        # enabled_ctx cwd is tmp_path which is in allowed_paths
        result = enabled_ctx.check_file_read("relative.txt")
        assert result == (tmp_path / "relative.txt").resolve()

    def test_denied_record_added_on_violation(
        self, enabled_ctx: SandboxContext
    ) -> None:
        with pytest.raises(SandboxViolation):
            enabled_ctx.check_file_read("/secret")
        assert len(enabled_ctx.denied_records) == 1
        assert enabled_ctx.denied_records[0].action_type == ActionType.FILE_READ


# ---------------------------------------------------------------------------
# SandboxContext — enabled / shell commands
# ---------------------------------------------------------------------------


class TestSandboxContextShell:
    def test_allowed_command_exact(self, enabled_ctx: SandboxContext) -> None:
        cmd = enabled_ctx.check_shell_command("echo hello world")
        assert cmd == "echo hello world"

    def test_allowed_command_ls(self, enabled_ctx: SandboxContext) -> None:
        cmd = enabled_ctx.check_shell_command("ls -la /tmp")
        assert cmd == "ls -la /tmp"

    def test_blocked_command(self, enabled_ctx: SandboxContext) -> None:
        with pytest.raises(SandboxViolation) as exc_info:
            enabled_ctx.check_shell_command("rm -rf /")
        assert exc_info.value.action_type == ActionType.SHELL

    def test_blocked_command_records_denial(
        self, enabled_ctx: SandboxContext
    ) -> None:
        with pytest.raises(SandboxViolation):
            enabled_ctx.check_shell_command("curl https://evil.com")
        denied = enabled_ctx.denied_records
        assert len(denied) == 1
        assert denied[0].action_type == ActionType.SHELL

    def test_empty_allowed_commands_blocks_all(
        self, tmp_path: Path
    ) -> None:
        cfg = SandboxConfig(enabled=True, allowed_commands=[])
        ctx = SandboxContext(cfg, working_directory=tmp_path)
        with pytest.raises(SandboxViolation):
            ctx.check_shell_command("echo hi")


# ---------------------------------------------------------------------------
# SandboxContext — enabled / web requests
# ---------------------------------------------------------------------------


class TestSandboxContextWebRequest:
    def test_exact_domain_allowed(self, enabled_ctx: SandboxContext) -> None:
        url = enabled_ctx.check_web_request("https://example.com/page")
        assert url == "https://example.com/page"

    def test_subdomain_allowed(self, enabled_ctx: SandboxContext) -> None:
        url = enabled_ctx.check_web_request("https://sub.example.com/data")
        assert url == "https://sub.example.com/data"

    def test_unrelated_domain_blocked(self, enabled_ctx: SandboxContext) -> None:
        with pytest.raises(SandboxViolation) as exc_info:
            enabled_ctx.check_web_request("https://evil.com/steal")
        assert exc_info.value.action_type == ActionType.WEB_REQUEST

    def test_partial_domain_not_matched(self, enabled_ctx: SandboxContext) -> None:
        # 'notexample.com' should NOT match allow-list entry 'example.com'
        with pytest.raises(SandboxViolation):
            enabled_ctx.check_web_request("https://notexample.com/")

    def test_http_url_allowed(self, enabled_ctx: SandboxContext) -> None:
        url = enabled_ctx.check_web_request("http://example.com/resource")
        assert url == "http://example.com/resource"

    def test_second_allowed_domain(self, enabled_ctx: SandboxContext) -> None:
        url = enabled_ctx.check_web_request("https://api.test.org/v1")
        assert url == "https://api.test.org/v1"

    def test_subdomain_of_second_domain_allowed(
        self, enabled_ctx: SandboxContext
    ) -> None:
        url = enabled_ctx.check_web_request("https://deep.api.test.org/")
        assert url == "https://deep.api.test.org/"

    def test_empty_allowed_domains_blocks_all(
        self, tmp_path: Path
    ) -> None:
        cfg = SandboxConfig(enabled=True, allowed_domains=[])
        ctx = SandboxContext(cfg, working_directory=tmp_path)
        with pytest.raises(SandboxViolation):
            ctx.check_web_request("https://example.com/")


# ---------------------------------------------------------------------------
# SandboxContext — records and summary
# ---------------------------------------------------------------------------


class TestSandboxContextRecords:
    def test_records_returns_copy(self, disabled_ctx: SandboxContext, tmp_path: Path) -> None:
        disabled_ctx.check_file_read(str(tmp_path / "f.txt"))
        r1 = disabled_ctx.records
        r2 = disabled_ctx.records
        assert r1 == r2
        assert r1 is not r2

    def test_clear_records(self, disabled_ctx: SandboxContext, tmp_path: Path) -> None:
        disabled_ctx.check_file_read(str(tmp_path / "f.txt"))
        assert len(disabled_ctx.records) == 1
        disabled_ctx.clear_records()
        assert len(disabled_ctx.records) == 0

    def test_summary_counts(self, enabled_ctx: SandboxContext, tmp_path: Path) -> None:
        enabled_ctx.check_file_read(str(tmp_path / "ok.txt"))
        try:
            enabled_ctx.check_file_read("/etc/shadow")
        except SandboxViolation:
            pass
        s = enabled_ctx.summary()
        assert s["total"] == 2
        assert s["permitted"] == 1
        assert s["denied"] == 1
        assert len(s["records"]) == 2

    def test_denied_records_property(self, enabled_ctx: SandboxContext) -> None:
        try:
            enabled_ctx.check_file_write("/etc/cron")
        except SandboxViolation:
            pass
        try:
            enabled_ctx.check_shell_command("rm -rf /")
        except SandboxViolation:
            pass
        denied = enabled_ctx.denied_records
        assert len(denied) == 2
        assert all(not r.permitted for r in denied)


# ---------------------------------------------------------------------------
# SandboxContext — context manager
# ---------------------------------------------------------------------------


class TestSandboxContextManager:
    def test_context_manager(self, tmp_path: Path) -> None:
        cfg = SandboxConfig(enabled=False)
        with SandboxContext(cfg, working_directory=tmp_path) as ctx:
            ctx.check_file_read(str(tmp_path / "x.txt"))
        assert len(ctx.records) == 1


# ---------------------------------------------------------------------------
# SandboxViolation
# ---------------------------------------------------------------------------


class TestSandboxViolation:
    def test_attributes(self) -> None:
        exc = SandboxViolation(ActionType.SHELL, "rm -rf / is blocked")
        assert exc.action_type == ActionType.SHELL
        assert "blocked" in exc.detail
        assert isinstance(exc, PermissionError)

    def test_str(self) -> None:
        exc = SandboxViolation(ActionType.FILE_READ, "not allowed")
        assert "not allowed" in str(exc)
