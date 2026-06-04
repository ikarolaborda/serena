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

from serena.memory_mirror import LocalMemoryMirror
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
"""Keys projected into .env.r2 / shown in the credentials UI."""

R2_AUTH_KEYS = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_ENDPOINT",
)
"""The R2 credentials proper. R2_BUCKET is a *target*, not a secret: r2-push.sh
resolves the bucket per project, so it must not gate "credentials present"."""

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
        # Always-on local backup so memories survive R2 being unreachable.
        self._mirror = LocalMemoryMirror(qdrant_memory_dir=self._qdrant_dir)
        # Bootstrap the encrypted store from any credentials already projected to
        # .env.r2 (e.g. configured directly in qdrant-memory) so the dashboard
        # recognises them instead of falsely reporting "missing".
        self.import_r2_credentials_from_env()

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

    @property
    def mirror(self) -> LocalMemoryMirror:
        return self._mirror

    def local_backup_state(self) -> dict[str, Any] | None:
        """Latest local-backup outcome, for dashboard status (None if never run)."""
        return self._mirror.latest_state()

    def _refresh_local_mirror(self) -> None:
        """Watcher callback: refresh the local mirror, discarding the result.

        Wrapped in a method (rather than passing the MirrorResult-returning
        ``mirror_now`` directly) so the on_change signature stays typed as
        ``Callable[[], None]``.
        """
        self._mirror.mirror_now("watcher")

    def start_watcher(self) -> None:
        """Start polling the state file for externally-triggered syncs.

        Idempotent; safe to call multiple times. Stops on agent shutdown
        because the thread is daemon-flagged. The watcher also refreshes the
        local backup mirror whenever it observes a change, so the local copy
        stays current even between explicit syncs.
        """
        if self._watcher is None:
            self._watcher = MemorySyncWatcher(
                state_file=self._qdrant_dir / "data" / "r2-state.json",
                history=self._history,
                active_project_provider=self._active_project_provider,
                on_change=self._refresh_local_mirror,
                change_watch_paths=[self._mirror.serena_memories_dir],
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

    @staticmethod
    def _parse_env_file(path: Path) -> dict[str, str]:
        """Parse a shell-style ``KEY=value`` env file into a dict (values unquoted)."""
        result: dict[str, str] = {}
        if not path.exists():
            return result
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
        except OSError as e:
            log.warning("could not read env file %s: %s", path, e)
        return result

    def read_env_r2_values(self) -> dict[str, str]:
        """Non-empty ``R2_*`` values from the projected ``.env.r2`` (fallback source)."""
        env = self._qdrant_dir / ".env.r2"
        return {k: v for k, v in self._parse_env_file(env).items() if k.startswith("R2_") and v}

    def import_r2_credentials_from_env(self) -> list[str]:
        """Backfill the encrypted store from ``.env.r2`` for keys not already stored.

        Idempotent and best-effort: only fills keys that are absent/empty in the
        SecretsStore, never overwrites, and never raises into construction.
        """
        imported: list[str] = []
        for key, value in self.read_env_r2_values().items():
            try:
                if not self._secrets.get_secret(R2_SCOPE, key):
                    self._secrets.set_secret(R2_SCOPE, key, value)
                    imported.append(key)
            except Exception as e:
                log.warning("could not import %s from .env.r2: %s", key, e)
        if imported:
            log.info("imported %d R2 credential(s) from .env.r2 into the encrypted store", len(imported))
        return imported

    def credential_status(self) -> dict[str, dict[str, Any]]:
        """Per-key presence + source across the SecretsStore and ``.env.r2``."""
        env_values = self.read_env_r2_values()
        status: dict[str, dict[str, Any]] = {}
        for k in R2_REQUIRED_KEYS:
            in_store = bool(self._secrets.get_secret(R2_SCOPE, k))
            in_env = bool(env_values.get(k))
            status[k] = {
                "present": in_store or in_env,
                "source": "secrets" if in_store else ("env" if in_env else None),
                "required": k in R2_AUTH_KEYS,
            }
        return status

    def credentials_present(self) -> bool:
        """True when all R2 auth keys are available (encrypted store or .env.r2).

        R2_BUCKET is intentionally excluded: it is resolved per-project by
        r2-push.sh and is not an authentication secret.
        """
        status = self.credential_status()
        return all(status[k]["present"] for k in R2_AUTH_KEYS)

    def set_r2_credentials(self, values: dict[str, str], *, write_through_env: bool = True) -> None:
        for k, v in values.items():
            self._secrets.set_secret(R2_SCOPE, k, str(v))
        if write_through_env:
            self.write_env_r2_file()

    def write_env_r2_file(self, target_path: str | os.PathLike[str] | None = None) -> Path:
        """Project the encrypted credentials into qdrant-memory/.env.r2.

        Merges with the existing file rather than overwriting it: keys already
        present in .env.r2 (e.g. QDRANT_URL, or a manually-set R2_BUCKET) are
        preserved, and a stored value is only written when non-empty. This
        guarantees an empty SecretsStore can never clobber working credentials.
        Done as an atomic temp-rename so a partial write is never observed.
        """
        target = Path(target_path) if target_path is not None else (self._qdrant_dir / ".env.r2")
        merged = self._parse_env_file(target)
        for k, v in self._secrets.list_scope(R2_SCOPE).items():
            if v:
                merged[k] = v
        lines = [
            "# Managed by Serena MemorySyncService (merged with existing values).",
            "# Credentials source of truth: ~/.serena/secrets.sqlite3 (scope=r2).",
            "",
        ]
        ordered = [k for k in R2_REQUIRED_KEYS if k in merged] + sorted(k for k in merged if k not in R2_REQUIRED_KEYS)
        for k in ordered:
            lines.append(f"{k}={_quote_env(merged[k])}")
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
            # Refresh the local backup regardless of the R2 outcome: when R2 is
            # unreachable this is exactly the copy that keeps memories durable.
            self._mirror.mirror_now(f"post-sync:{run.outcome}")
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
        # Missing auth credentials is an actionable, non-transient failure —
        # surface it before launching the subprocess. Checks both the encrypted
        # store and .env.r2; R2_BUCKET is excluded (resolved per-project).
        status = self.credential_status()
        missing = [k for k in R2_AUTH_KEYS if not status[k]["present"]]
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
