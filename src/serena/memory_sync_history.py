"""
Per-project R2 sync history with a polling watcher.

The qdrant-memory R2 push is triggered from many places — the Serena
dashboard, the MCP tool (``memory_sync_now``), and the canonical
``bash scripts/r2-push.sh``. All of them update the same
``qdrant-memory/data/r2-state.json``. Operators inside a particular
project want to see "the last N syncs that affected my project" without
having to remember to refresh the dashboard.

This module persists a tiny JSONL history at
``~/.serena/memory_sync_history.jsonl`` and exposes a background watcher
that polls ``r2-state.json`` for changes. Each entry is project-tagged:

* When Serena itself triggered the sync, the project is whatever the
  operator passed to ``MemorySyncService.trigger_sync``.
* When an external tool (e.g. ``memory_sync_now`` from another Claude
  Code session) updates ``r2-state.json``, the watcher tags the event
  with the currently-active project at observation time — best-effort,
  since the state file does not record the project itself.

Concurrency: every Serena instance shares the same JSONL, so appends go
through ``cross_process_lock`` on a sibling ``.lock`` file. We never
mutate older rows — append-only is enough for "show last N" and is
robust against torn writes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from serena.config.serena_config import SerenaPaths
from serena.util.atomic_io import cross_process_lock

log = logging.getLogger(__name__)

HISTORY_FILENAME = "memory_sync_history.jsonl"
"""Sibling of secrets.sqlite3 under ``~/.serena/``."""

MAX_HISTORY_BYTES = 1_000_000
"""Rotate-on-write threshold; old entries archived to ``.1`` for one round."""

WATCHER_POLL_SECONDS_DEFAULT = 10.0
"""Foreground polling cadence for the r2-state.json watcher."""


@dataclass
class SyncEvent:
    """A single record in the per-project history JSONL."""

    ts: str
    project: str | None
    snapshot_id: str | None
    source: str  # "self" (triggered through Serena) | "external" (observed)
    outcome: str  # "ok" | "failed" | "unavailable"
    archive_size_bytes: int | None = None
    archive_sha256: str | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemorySyncHistory:
    """Append-only JSONL history with cross-process safe writes + simple reads."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        if path is None:
            path = Path(SerenaPaths().serena_user_home_dir) / HISTORY_FILENAME
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._in_proc_lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: SyncEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
        with self._in_proc_lock, cross_process_lock(self._lock_path):
            self._maybe_rotate()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def _maybe_rotate(self) -> None:
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < MAX_HISTORY_BYTES:
            return
        rotated = self._path.with_suffix(self._path.suffix + ".1")
        # Replace any prior .1 silently — we only keep one generation.
        try:
            os.replace(self._path, rotated)
        except OSError as e:
            log.warning("history rotation failed: %s", e)

    def read_all(self) -> list[SyncEvent]:
        if not self._path.exists():
            return []
        events: list[SyncEvent] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(
                    SyncEvent(
                        ts=row.get("ts", ""),
                        project=row.get("project"),
                        snapshot_id=row.get("snapshot_id"),
                        source=row.get("source", "external"),
                        outcome=row.get("outcome", "ok"),
                        archive_size_bytes=row.get("archive_size_bytes"),
                        archive_sha256=row.get("archive_sha256"),
                        notes=row.get("notes", ""),
                        extra=row.get("extra", {}) or {},
                    )
                )
        return events

    def recent(self, *, project: str | None = None, limit: int = 25) -> list[SyncEvent]:
        events = self.read_all()
        if project:
            events = [e for e in events if (e.project or "") == project]
        # JSONL is ordered append; show newest first.
        return list(reversed(events))[:limit]


class MemorySyncWatcher:
    """Polls the qdrant-memory state file and records project-tagged events.

    Self-triggered events go through ``record_self_event`` from
    ``MemorySyncService`` directly. The polling loop catches external
    syncs (other Claude Code sessions, ``bash scripts/r2-push.sh``, etc.)
    by comparing ``last_push.snapshot_id`` to the last observed value.
    """

    def __init__(
        self,
        state_file: str | os.PathLike[str],
        history: MemorySyncHistory,
        active_project_provider: Callable[[], str | None],
        poll_seconds: float = WATCHER_POLL_SECONDS_DEFAULT,
        on_change: Callable[[], None] | None = None,
        change_watch_paths: Sequence[str | os.PathLike[str]] | None = None,
    ) -> None:
        self._state_file = Path(state_file)
        self._history = history
        self._active_project_provider = active_project_provider
        self._poll_seconds = poll_seconds
        # Fired when a sync is observed or a watched memory path changes, so a
        # consumer (the local backup mirror) can refresh itself promptly.
        self._on_change = on_change
        self._change_watch_paths = [Path(p) for p in (change_watch_paths or [])]
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Seed from disk so a watcher restart does not flood the JSONL with
        # the same already-recorded snapshot id.
        self._last_snapshot_id = self._seed_snapshot_id()
        self._last_change_sig = self._change_signature()

    def _seed_snapshot_id(self) -> str | None:
        push = self._read_last_push()
        return push.get("snapshot_id") if push else None

    def _read_last_push(self) -> dict[str, Any] | None:
        if not self._state_file.exists():
            return None
        try:
            with open(self._state_file, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("watcher: state file unreadable: %s", e)
            return None
        return data.get("last_push") if isinstance(data, dict) else None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="MemorySyncWatcher",
            daemon=True,
        )
        self._thread.start()
        log.info("MemorySyncWatcher started; polling %s every %.1fs", self._state_file, self._poll_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._poll_changes()
            self._stop.wait(self._poll_seconds)

    def _change_signature(self) -> float:
        """Newest mtime across the watched memory paths (0.0 if none/absent).

        Used to detect local memory edits cheaply; the watched trees are small
        (markdown memories). Directory renames from atomic writes bump mtimes.
        """
        newest = 0.0
        for root in self._change_watch_paths:
            if not root.exists():
                continue
            try:
                newest = max(newest, root.stat().st_mtime)
                for p in root.rglob("*"):
                    try:
                        newest = max(newest, p.stat().st_mtime)
                    except OSError:
                        continue
            except OSError:
                continue
        return newest

    def _poll_changes(self) -> bool:
        """Fire ``on_change`` when a watched memory path has been modified.

        Returns True iff a change was detected. Never raises into the loop.
        """
        if self._on_change is None or not self._change_watch_paths:
            return False
        sig = self._change_signature()
        if sig <= self._last_change_sig:
            return False
        self._last_change_sig = sig
        try:
            self._on_change()
        except Exception as e:
            log.warning("watcher on_change (memory edit) failed: %s", e)
        return True

    def poll_once(self) -> bool:
        """Returns True iff a new sync was observed and appended to history."""
        push = self._read_last_push()
        if not push:
            return False
        snap = push.get("snapshot_id")
        if not snap or snap == self._last_snapshot_id:
            return False
        # New snapshot id since the last observation.
        self._last_snapshot_id = snap
        self._history.append(
            SyncEvent(
                ts=_utcnow_iso(),
                project=self._active_project_provider(),
                snapshot_id=snap,
                source="external",
                outcome="ok",
                archive_size_bytes=push.get("archive_size_bytes"),
                archive_sha256=push.get("archive_sha256"),
                notes="observed via r2-state.json mtime poll",
                extra={
                    "archive_key": push.get("archive_key"),
                    "r2_bucket": push.get("r2_bucket"),
                    "pushed_from": push.get("pushed_from"),
                    "pushed_at": push.get("pushed_at"),
                },
            )
        )
        log.info("watcher recorded external sync %s", snap)
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception as e:
                log.warning("watcher on_change (new snapshot) failed: %s", e)
        return True

    def record_self_event(
        self,
        *,
        project: str | None,
        snapshot_id: str | None,
        outcome: str,
        notes: str = "",
    ) -> None:
        """Called by MemorySyncService when a sync was triggered through Serena."""
        if snapshot_id:
            # Prevent the polling loop from double-recording the same id as "external".
            self._last_snapshot_id = snapshot_id
        self._history.append(
            SyncEvent(
                ts=_utcnow_iso(),
                project=project,
                snapshot_id=snapshot_id,
                source="self",
                outcome=outcome,
                notes=notes,
            )
        )


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "HISTORY_FILENAME",
    "WATCHER_POLL_SECONDS_DEFAULT",
    "MemorySyncHistory",
    "MemorySyncWatcher",
    "SyncEvent",
]
