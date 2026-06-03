"""
Operator-facing memory sync orchestration.

The qdrant-memory MCP server already exposes ``memory_sync_now`` and ships
``scripts/r2-push.sh`` to push Qdrant collections + Claude file memories to
the Cloudflare R2 bucket. This module is the Serena-side facade that lets a
dashboard operator:

  * read the most recent push/pull outcome (from ``data/r2-state.json``)
  * trigger a fresh push when automation failed (subprocess wrapper around
    the canonical ``r2-push.sh``)
  * inspect and edit the R2 credentials, which are stored in Serena's
    SQLite secrets store and projected to qdrant-memory's ``.env.r2``
    before each push

Why a Serena-side wrapper at all? Because when the automated
``memory_sync_now`` MCP call fails (Qdrant unreachable, expired token,
network blip), the operator currently has to drop to a terminal in the
qdrant-memory repo to recover. Surfacing the same primitives in the
Serena dashboard removes that context-switch — the operator sees the
failed sync inline with the project they were working in.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from serena.memory_sync_history import MemorySyncHistory, MemorySyncWatcher
from serena.util.atomic_io import atomic_write_text
from serena.util.secrets_store import SecretsStore

log = logging.getLogger(__name__)

R2_SCOPE = "r2"
"""SecretsStore scope used for the R2 credential keys."""

R2_REQUIRED_KEYS = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_ENDPOINT",
)
"""Keys the qdrant-memory r2-push.sh script validates as required."""

DEFAULT_QDRANT_MEMORY_DIR = "~/Aerolambda/qdrant-memory"
"""Sibling repo path; override with the SERENA_QDRANT_MEMORY_DIR env var."""

_SAFE_VALUE_RE = re.compile(r"\A[\w.:/=+@%-]*\Z")
"""Conservative whitelist for .env.r2 values; falls back to shell-quoting otherwise."""


@dataclass
class SyncSummary:
    """Latest push/pull state read from qdrant-memory's r2-state.json."""

    last_push: dict[str, Any] | None = None
    last_pull: dict[str, Any] | None = None
    state_file_path: str | None = None
    state_file_present: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyncRunResult:
    """Outcome of a single trigger_sync invocation."""

    started_at: str
    finished_at: str | None = None
    outcome: str = "running"  # running | ok | failed | unavailable
    project: str | None = None
    dry_run: bool = False
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    notes: str = ""
    snapshot_id: str | None = None
    binding: str = "bash scripts/r2-push.sh"
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemorySyncService:
    """Facade over qdrant-memory's R2 sync.

    Single-instance per Serena agent. Holds the SecretsStore reference and
    tracks the in-progress / most-recent run so the dashboard can show
    status without re-reading subprocess output.
    """

    _OUTPUT_TAIL_BYTES = 4096

    def __init__(
        self,
        secrets_store: SecretsStore | None = None,
        qdrant_memory_dir: str | os.PathLike[str] | None = None,
        history: MemorySyncHistory | None = None,
        active_project_provider: Callable[[], str | None] | None = None,
    ) -> None:
        # Default to the encrypted store; tests can pass an explicit plain
        # SecretsStore via the keyword arg.
        self._secrets = secrets_store if secrets_store is not None else SecretsStore.default()
        self._qdrant_dir = self._resolve_qdrant_dir(qdrant_memory_dir)
        self._lock = threading.Lock()
        self._last_run: SyncRunResult | None = None
        self._current_run: SyncRunResult | None = None
        self._history = history if history is not None else MemorySyncHistory()
        # The watcher needs to know which project to tag external syncs with.
        # The dashboard wires this to the agent's active project; tests can
        # pass an explicit provider.
        self._active_project_provider: Callable[[], str | None] = (
            active_project_provider if active_project_provider is not None else lambda: None
        )
        self._watcher: MemorySyncWatcher | None = None

    @staticmethod
    def _resolve_qdrant_dir(override: str | os.PathLike[str] | None) -> Path:
        if override is not None:
            return Path(override).expanduser().resolve()
        env = os.environ.get("SERENA_QDRANT_MEMORY_DIR")
        if env:
            return Path(env).expanduser().resolve()
        return Path(DEFAULT_QDRANT_MEMORY_DIR).expanduser().resolve()

    @property
    def qdrant_memory_dir(self) -> Path:
        return self._qdrant_dir

    @property
    def secrets(self) -> SecretsStore:
        return self._secrets

    @property
    def history(self) -> MemorySyncHistory:
        return self._history

    def start_watcher(self) -> None:
        """Start polling the state file for externally-triggered syncs.

        Idempotent; safe to call multiple times. Stops on agent shutdown
        because the thread is daemon-flagged.
        """
        if self._watcher is None:
            self._watcher = MemorySyncWatcher(
                state_file=self._qdrant_dir / "data" / "r2-state.json",
                history=self._history,
                active_project_provider=self._active_project_provider,
            )
        self._watcher.start()

    def stop_watcher(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()

    # ---------- credentials ---------------------------------------------------

    def get_r2_credentials_masked(self) -> dict[str, str]:
        stored = self._secrets.list_scope_masked(R2_SCOPE)
        # Surface missing required keys so the UI can prompt for them.
        for k in R2_REQUIRED_KEYS:
            stored.setdefault(k, "")
        return stored

    def set_r2_credentials(self, values: dict[str, str], *, write_through_env: bool = True) -> None:
        for k, v in values.items():
            if v is None:
                continue
            self._secrets.set_secret(R2_SCOPE, k, str(v))
        if write_through_env:
            self.write_env_r2_file()

    def write_env_r2_file(self, target_path: str | os.PathLike[str] | None = None) -> Path:
        """Project the SQLite-resident credentials into qdrant-memory/.env.r2.

        Done as an atomic temp-rename so a partial write never leaves the
        consumer script reading a half-populated file.
        """
        target = Path(target_path) if target_path is not None else (self._qdrant_dir / ".env.r2")
        values = self._secrets.list_scope(R2_SCOPE)
        lines = [
            "# Auto-generated by Serena MemorySyncService — do not edit by hand.",
            "# Source of truth: ~/.serena/secrets.sqlite3 (scope=r2).",
            "",
        ]
        for k in R2_REQUIRED_KEYS:
            v = values.get(k, "")
            lines.append(f"{k}={_quote_env(v)}")
        # Preserve any caller-supplied keys outside the required set
        for k in sorted(set(values) - set(R2_REQUIRED_KEYS)):
            lines.append(f"{k}={_quote_env(values[k])}")
        atomic_write_text(target, "\n".join(lines) + "\n", encoding="utf-8")
        # The credentials are sensitive — drop the projected file to 0600
        try:
            os.chmod(target, 0o600)
        except OSError:
            log.warning("Could not chmod 600 on %s; check filesystem support", target)
        return target

    # ---------- state ---------------------------------------------------------

    def read_state(self) -> SyncSummary:
        state_path = self._qdrant_dir / "data" / "r2-state.json"
        summary = SyncSummary(state_file_path=str(state_path))
        if not state_path.exists():
            summary.error = "r2-state.json not present yet — no sync has run"
            return summary
        summary.state_file_present = True
        try:
            with open(state_path, encoding="utf-8") as f:
                data = json.load(f)
            summary.last_push = data.get("last_push")
            summary.last_pull = data.get("last_pull")
        except (OSError, ValueError) as e:
            summary.error = f"could not read state file: {e}"
        return summary

    def current_run(self) -> SyncRunResult | None:
        with self._lock:
            return self._current_run

    def last_run(self) -> SyncRunResult | None:
        with self._lock:
            return self._last_run

    # ---------- trigger -------------------------------------------------------

    def trigger_sync(self, project: str | None = None, *, dry_run: bool = False) -> SyncRunResult:
        """Run r2-push.sh synchronously. Callers should invoke from a worker thread.

        The dashboard wires this through the existing TaskExecutor so the HTTP
        request returns immediately and the operator polls ``current_run`` /
        ``last_run`` for completion.
        """
        with self._lock:
            if self._current_run is not None and self._current_run.outcome == "running":
                # One in-flight sync at a time, full stop.
                return self._current_run
            run = SyncRunResult(
                started_at=_utcnow_iso(),
                outcome="running",
                project=project,
                dry_run=dry_run,
            )
            self._current_run = run

        try:
            self._validate_preconditions(run)
            if run.outcome != "running":
                return run
            self.write_env_r2_file()
            self._invoke_script(run)
            self._reconcile_state(run)
        except Exception as e:  # last-resort safety net
            log.exception("memory sync raised")
            run.outcome = "failed"
            run.notes = f"{run.notes}\nunexpected exception: {e}".strip()
        finally:
            run.finished_at = _utcnow_iso()
            with self._lock:
                self._last_run = run
                self._current_run = None
            # Always record self-events: the operator wants the history to
            # show failed/unavailable attempts too, not only successful pushes.
            if self._watcher is not None:
                self._watcher.record_self_event(
                    project=run.project,
                    snapshot_id=run.snapshot_id,
                    outcome=run.outcome,
                    notes=run.notes,
                )
            else:
                # If the watcher is not started (e.g. headless test harness),
                # still write to history directly so dashboards picking up
                # later can see this attempt.
                from serena.memory_sync_history import SyncEvent

                self._history.append(
                    SyncEvent(
                        ts=run.finished_at or _utcnow_iso(),
                        project=run.project,
                        snapshot_id=run.snapshot_id,
                        source="self",
                        outcome=run.outcome,
                        notes=run.notes,
                    )
                )
        return run

    def _validate_preconditions(self, run: SyncRunResult) -> None:
        if not self._qdrant_dir.exists():
            run.outcome = "unavailable"
            run.notes = f"qdrant-memory dir not found at {self._qdrant_dir}"
            return
        script = self._qdrant_dir / "scripts" / "r2-push.sh"
        if not script.exists():
            run.outcome = "unavailable"
            run.notes = f"r2-push.sh not found at {script}"
            return
        # Missing required keys is an actionable, non-transient failure —
        # surface it before launching the subprocess.
        missing = [k for k in R2_REQUIRED_KEYS if not self._secrets.get_secret(R2_SCOPE, k)]
        if missing:
            run.outcome = "failed"
            run.notes = f"missing R2 credentials: {', '.join(missing)}"

    def _invoke_script(self, run: SyncRunResult) -> None:
        cmd = ["bash", "scripts/r2-push.sh"]
        if run.project:
            cmd += ["--project", run.project]
        if run.dry_run:
            cmd += ["--dry-run"]
        log.info("memory sync: %s (cwd=%s)", shlex.join(cmd), self._qdrant_dir)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self._qdrant_dir),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            run.outcome = "failed"
            run.notes = f"timeout after {e.timeout}s"
            return
        except FileNotFoundError as e:
            run.outcome = "unavailable"
            run.notes = f"bash not found: {e}"
            return
        run.exit_code = proc.returncode
        run.stdout_tail = (proc.stdout or "")[-self._OUTPUT_TAIL_BYTES :]
        run.stderr_tail = (proc.stderr or "")[-self._OUTPUT_TAIL_BYTES :]
        run.outcome = "ok" if proc.returncode == 0 else "failed"

    def _reconcile_state(self, run: SyncRunResult) -> None:
        summary = self.read_state()
        if summary.last_push:
            run.snapshot_id = summary.last_push.get("snapshot_id")
            run.sources = sorted(
                {
                    *(summary.last_push.get("sources") or []),
                    *self._derive_sources(),
                }
            )

    def _derive_sources(self) -> list[str]:
        """Best-effort enumeration of in-scope memory roots.

        Used purely for operator visibility ("what does this sync include?") —
        the actual scoping is owned by ``r2-push.sh``.
        """
        sources: list[str] = []
        home = Path.home()
        candidates = [
            home / ".claude" / "projects",
            home / ".serena" / "memories",
        ]
        for c in candidates:
            if c.exists():
                sources.append(str(c))
        return sources


def _quote_env(value: str) -> str:
    """Return a value safe to drop after ``KEY=`` in a bash-sourced env file."""
    if value == "":
        return ""
    if _SAFE_VALUE_RE.fullmatch(value):
        return value
    # Single-quote and escape any embedded single quotes.
    escaped = value.replace("'", "'\\''")
    return f"'{escaped}'"


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "R2_REQUIRED_KEYS",
    "R2_SCOPE",
    "MemorySyncService",
    "SyncRunResult",
    "SyncSummary",
]
