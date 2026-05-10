"""Tests for serena.memory_sync."""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from serena.memory_sync import (
    R2_REQUIRED_KEYS,
    R2_SCOPE,
    MemorySyncService,
    _quote_env,
)
from serena.util.secrets_store import SecretsStore


@pytest.fixture
def fake_qdrant(tmp_path: Path) -> Path:
    """Build a fake qdrant-memory layout (scripts/ + data/) under tmp_path."""
    root = tmp_path / "qdrant-memory"
    (root / "scripts").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "scripts" / "r2-push.sh").write_text("#!/usr/bin/env bash\necho ok\n")
    os.chmod(root / "scripts" / "r2-push.sh", 0o755)
    return root


@pytest.fixture
def service(tmp_path: Path, fake_qdrant: Path) -> MemorySyncService:
    secrets = SecretsStore(path=tmp_path / "secrets.sqlite3")
    return MemorySyncService(secrets_store=secrets, qdrant_memory_dir=fake_qdrant)


class TestEnvQuoting:
    def test_safe_value_unquoted(self) -> None:
        assert _quote_env("abc-123_DEF") == "abc-123_DEF"
        assert _quote_env("https://x.example/y") == "https://x.example/y"
        assert _quote_env("") == ""

    def test_unsafe_value_single_quoted(self) -> None:
        out = _quote_env("a b c")
        assert out.startswith("'") and out.endswith("'")
        assert "a b c" in out

    def test_embedded_single_quote_escaped(self) -> None:
        out = _quote_env("it's tricky")
        # Bash idiom: end quote, escape, reopen — never an unescaped ' inside ''.
        assert "'\\''" in out


class TestCredentials:
    def test_set_and_mask_round_trip(self, service: MemorySyncService) -> None:
        service.set_r2_credentials(
            {
                "R2_ACCOUNT_ID": "acct-123",
                "R2_ACCESS_KEY_ID": "AKIA" + "X" * 16,
                "R2_SECRET_ACCESS_KEY": "S" * 64,
                "R2_BUCKET": "bucket-1",
                "R2_ENDPOINT": "https://acct-123.r2.cloudflarestorage.com",
            },
            write_through_env=False,
        )
        masked = service.get_r2_credentials_masked()
        for key in R2_REQUIRED_KEYS:
            assert masked.get(key, ""), f"missing {key} after set"

    def test_required_keys_surfaced_when_missing(self, service: MemorySyncService) -> None:
        masked = service.get_r2_credentials_masked()
        for k in R2_REQUIRED_KEYS:
            assert k in masked

    def test_write_env_r2_atomic_and_chmod(self, service: MemorySyncService, fake_qdrant: Path) -> None:
        service.set_r2_credentials(
            {
                "R2_ACCOUNT_ID": "acct",
                "R2_ACCESS_KEY_ID": "AK",
                "R2_SECRET_ACCESS_KEY": "shh has spaces",
                "R2_BUCKET": "b",
                "R2_ENDPOINT": "https://x",
            },
            write_through_env=False,
        )
        target = service.write_env_r2_file()
        assert target == fake_qdrant / ".env.r2"
        body = target.read_text()
        # Quoted (spaces).
        assert "R2_SECRET_ACCESS_KEY='shh has spaces'" in body
        # Unquoted safe value.
        assert "R2_BUCKET=b" in body
        # File should be 0600 on POSIX.
        if os.name != "nt":
            assert stat.S_IMODE(target.stat().st_mode) == 0o600


class TestStateReadback:
    def test_missing_state_reports_gracefully(self, service: MemorySyncService) -> None:
        summary = service.read_state()
        assert summary.state_file_present is False
        assert summary.error is not None and "no sync" in summary.error

    def test_present_state_parses(self, service: MemorySyncService, fake_qdrant: Path) -> None:
        state = {
            "last_push": {"snapshot_id": "snap-1", "pushed_at": "2026-05-10T12:00:00Z"},
            "last_pull": None,
        }
        (fake_qdrant / "data" / "r2-state.json").write_text(json.dumps(state))
        summary = service.read_state()
        assert summary.state_file_present is True
        assert summary.last_push and summary.last_push["snapshot_id"] == "snap-1"


class TestTriggerSync:
    def _full_creds(self, service: MemorySyncService) -> None:
        service.set_r2_credentials(
            {k: f"value-{k}" for k in R2_REQUIRED_KEYS},
            write_through_env=False,
        )

    def test_missing_credentials_fails_fast(self, service: MemorySyncService) -> None:
        # No credentials set.
        result = service.trigger_sync(project="serena")
        assert result.outcome == "failed"
        assert "missing R2 credentials" in result.notes

    def test_missing_qdrant_dir_unavailable(self, tmp_path: Path) -> None:
        secrets = SecretsStore(path=tmp_path / "secrets.sqlite3")
        svc = MemorySyncService(secrets_store=secrets, qdrant_memory_dir=tmp_path / "nope")
        # Credentials don't matter — preflight fails on missing dir first.
        svc.set_r2_credentials({k: "x" for k in R2_REQUIRED_KEYS}, write_through_env=False)
        result = svc.trigger_sync()
        assert result.outcome == "unavailable"

    def test_successful_invocation_records_ok(self, service: MemorySyncService) -> None:
        self._full_creds(service)
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="pushed", stderr="")
        with patch("serena.memory_sync.subprocess.run", return_value=completed) as m:
            result = service.trigger_sync(project="serena")
            assert m.call_count == 1
            args = m.call_args.args[0]
            assert args[:2] == ["bash", "scripts/r2-push.sh"]
            assert "--project" in args and "serena" in args
        assert result.outcome == "ok"
        assert result.exit_code == 0
        assert "pushed" in result.stdout_tail

    def test_failed_invocation_records_failed(self, service: MemorySyncService) -> None:
        self._full_creds(service)
        completed = subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")
        with patch("serena.memory_sync.subprocess.run", return_value=completed):
            result = service.trigger_sync()
        assert result.outcome == "failed"
        assert result.exit_code == 2
        assert "boom" in result.stderr_tail

    def test_dry_run_flag_propagates(self, service: MemorySyncService) -> None:
        self._full_creds(service)
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch("serena.memory_sync.subprocess.run", return_value=completed) as m:
            service.trigger_sync(dry_run=True)
        assert "--dry-run" in m.call_args.args[0]

    def test_concurrent_trigger_returns_running(self, service: MemorySyncService) -> None:
        self._full_creds(service)
        # Pre-seed an in-progress run so the early return path is exercised.
        from serena.memory_sync import SyncRunResult
        service._current_run = SyncRunResult(started_at="now", outcome="running")
        result = service.trigger_sync()
        assert result.outcome == "running"
