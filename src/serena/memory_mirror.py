"""
Always-on local backup mirror of the memory stores.

Cloud durability for memories is provided by ``qdrant-memory``'s R2 push
(see :mod:`serena.memory_sync`). R2 can be unreachable (offline, expired
token, outage), and in those windows nothing keeps a *current* local copy
of the memories. This module fills that gap: it maintains a serena-managed
local backup that is refreshed on every sync attempt and whenever a change
is observed, so that when R2 is not reachable the memories still survive
locally and can be restored.

Design (deliberately serena-side and decoupled from ``r2-push.sh``):

* Sources backed up (best-effort, each labeled):
    - ``serena-memories`` — ``~/.serena/memories`` (serena's own markdown
      memories; always current because serena owns these files).
    - ``qdrant-snapshot`` — the most recent ``memory-sync-*.tar.gz`` found
      under the qdrant-memory data dir, if one is staged locally. This is a
      best-effort capture of the qdrant collections + file-memory bundle; it
      is only as fresh as the last local snapshot (its real age is recorded
      in the manifest, never presented as "current").
* Destination: ``~/.serena/memory-backups/<UTC-timestamp>/`` with a
  ``manifest.json`` per backup and a ``latest-state.json`` pointer.
* Writes are atomic (copy into ``.tmp`` then ``os.replace``) so a partial
  copy is never published, and pruning/publishing hold a
  ``cross_process_lock`` because every serena instance shares the dir.

This module never raises into its callers: a missing source yields an
``unavailable``/``partial`` outcome, a copy error yields ``failed``, and
previously-published backups are always preserved until a new one is
committed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from serena.config.serena_config import SerenaPaths
from serena.util.atomic_io import atomic_write_text, cross_process_lock

log = logging.getLogger(__name__)

BACKUP_DIRNAME = "memory-backups"
"""Sub-directory of ``~/.serena`` that holds the local memory backups."""

LATEST_STATE_FILENAME = "latest-state.json"
"""Pointer file (under the backup root) describing the most recent backup."""

DEFAULT_RETENTION = 8
"""Number of timestamped backups to keep; older ones are pruned."""

_QDRANT_SNAPSHOT_GLOB = "memory-sync-*.tar.gz"
"""Filename pattern of the portable qdrant snapshot archive staged by r2-push."""


@dataclass
class MirrorSource:
    label: str
    path: Path
    kind: str  # "dir" | "file"


@dataclass
class MirrorResult:
    ts: str
    outcome: str  # "ok" | "partial" | "skipped" | "unavailable" | "failed"
    dest: str | None = None
    snapshot_id: str | None = None
    total_bytes: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""
    reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalMemoryMirror:
    """Maintains a serena-managed local backup of the memory sources.

    Thread-safe and cross-process safe. Intended to be triggered both on
    every R2 sync attempt (regardless of its outcome) and whenever the
    sync watcher observes a change, so the local copy is always current.
    """

    def __init__(
        self,
        *,
        backup_root: str | os.PathLike[str] | None = None,
        serena_memories_dir: str | os.PathLike[str] | None = None,
        qdrant_memory_dir: str | os.PathLike[str] | None = None,
        retention: int = DEFAULT_RETENTION,
        history_recorder: Callable[[MirrorResult], None] | None = None,
    ) -> None:
        if backup_root is None:
            backup_root = Path(SerenaPaths().serena_user_home_dir) / BACKUP_DIRNAME
        self._root = Path(backup_root)
        self._root.mkdir(parents=True, exist_ok=True)

        if serena_memories_dir is None:
            serena_memories_dir = Path(SerenaPaths().serena_user_home_dir) / "memories"
        self._serena_memories_dir = Path(serena_memories_dir)

        self._qdrant_dir = Path(qdrant_memory_dir).expanduser().resolve() if qdrant_memory_dir is not None else None
        self._retention = max(1, retention)
        self._history_recorder = history_recorder

        self._lock_path = self._root / ".mirror.lock"
        self._in_proc_lock = threading.Lock()

    @property
    def backup_root(self) -> Path:
        return self._root

    @property
    def serena_memories_dir(self) -> Path:
        return self._serena_memories_dir

    def _resolve_sources(self) -> list[MirrorSource]:
        sources: list[MirrorSource] = []
        if self._serena_memories_dir.exists():
            sources.append(MirrorSource("serena-memories", self._serena_memories_dir, "dir"))
        snapshot = self._latest_qdrant_snapshot()
        if snapshot is not None:
            sources.append(MirrorSource("qdrant-snapshot", snapshot, "file"))
        return sources

    def _latest_qdrant_snapshot(self) -> Path | None:
        if self._qdrant_dir is None:
            return None
        data_dir = self._qdrant_dir / "data"
        if not data_dir.exists():
            return None
        candidates = list(data_dir.rglob(_QDRANT_SNAPSHOT_GLOB))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _read_snapshot_id(self) -> str | None:
        if self._qdrant_dir is None:
            return None
        state = self._qdrant_dir / "data" / "r2-state.json"
        if not state.exists():
            return None
        try:
            import json

            with open(state, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        push = data.get("last_push") if isinstance(data, dict) else None
        return push.get("snapshot_id") if isinstance(push, dict) else None

    def mirror_now(self, reason: str = "") -> MirrorResult:
        """Capture the current sources into a fresh timestamped backup.

        Best-effort and non-raising: outcome reflects what was achievable.
        """
        ts = _utcnow_tag()
        result = MirrorResult(ts=ts, outcome="unavailable", reason=reason, snapshot_id=self._read_snapshot_id())

        sources = self._resolve_sources()
        if not sources:
            result.notes = "no memory sources present to back up"
            self._record(result)
            return result

        result.fingerprint = self._fingerprint(sources)

        with self._in_proc_lock, cross_process_lock(self._lock_path):
            # Skip redundant full copies when nothing changed since the last
            # backup (the watcher and per-sync triggers both call us).
            prev = self.latest_state()
            if prev and prev.get("fingerprint") == result.fingerprint and prev.get("outcome") in {"ok", "partial"}:
                result.outcome = "skipped"
                result.dest = prev.get("dest")
                result.notes = "unchanged since last backup"
                return result
            try:
                self._capture(ts, sources, result)
                self._prune()
            except Exception as e:  # never propagate into sync/watcher callers
                log.exception("local memory mirror failed")
                result.outcome = "failed"
                result.notes = (result.notes + f"\nunexpected error: {e}").strip()
            finally:
                self._write_latest_pointer(result)

        self._record(result)
        return result

    def _fingerprint(self, sources: list[MirrorSource]) -> str:
        """Content signature used to skip redundant backups.

        Cheap and good enough: newest mtime for directories (atomic writes bump
        it), and path+mtime+size for files. Not a cryptographic guarantee.
        """
        parts: list[str] = []
        for src in sources:
            try:
                if src.kind == "dir":
                    newest = src.path.stat().st_mtime
                    for p in src.path.rglob("*"):
                        try:
                            newest = max(newest, p.stat().st_mtime)
                        except OSError:
                            continue
                    parts.append(f"{src.label}:{newest:.3f}")
                else:
                    st = src.path.stat()
                    parts.append(f"{src.label}:{src.path}:{st.st_mtime:.3f}:{st.st_size}")
            except OSError:
                parts.append(f"{src.label}:missing")
        return "|".join(parts)

    def _capture(self, ts: str, sources: list[MirrorSource], result: MirrorResult) -> None:
        # pid-suffixed temp dir so concurrent processes never target the same path.
        tmp_dir = self._root / f"{ts}.{os.getpid()}.tmp"
        final_dir = self._unique_final_dir(ts)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)

        copied = 0
        for src in sources:
            entry: dict[str, Any] = {"label": src.label, "kind": src.kind, "src": str(src.path)}
            try:
                target = tmp_dir / src.label
                if src.kind == "dir":
                    shutil.copytree(src.path, target, ignore=_ignore_transient)
                    size = _dir_size(target)
                    entry["bytes"] = size
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    dst_file = target / src.path.name
                    shutil.copy2(src.path, dst_file)
                    size = dst_file.stat().st_size
                    entry["bytes"] = size
                    entry["sha256"] = _sha256_file(dst_file)
                    entry["age_seconds"] = max(0, int(_utcnow().timestamp() - src.path.stat().st_mtime))
                result.total_bytes += size
                copied += 1
            except Exception as e:
                entry["error"] = str(e)
                log.warning("mirror: failed to copy source %s: %s", src.label, e)
            result.sources.append(entry)

        manifest = {
            "ts": ts,
            "reason": result.reason,
            "snapshot_id": result.snapshot_id,
            "total_bytes": result.total_bytes,
            "sources": result.sources,
        }
        atomic_write_text(tmp_dir / "manifest.json", _dumps(manifest))

        # Atomic publish: replace only after the full copy succeeded.
        os.replace(tmp_dir, final_dir)
        result.dest = str(final_dir)

        if copied == len(sources):
            result.outcome = "ok"
        elif copied > 0:
            result.outcome = "partial"
            result.notes = (result.notes + " some sources failed to copy").strip()
        else:
            # nothing copied — drop the empty published dir, report failed
            shutil.rmtree(final_dir, ignore_errors=True)
            result.dest = None
            result.outcome = "failed"
            result.notes = (result.notes + " all sources failed to copy").strip()

    def _unique_final_dir(self, ts: str) -> Path:
        final_dir = self._root / ts
        if not final_dir.exists():
            return final_dir
        i = 1
        while (self._root / f"{ts}-{i}").exists():
            i += 1
        return self._root / f"{ts}-{i}"

    def _prune(self) -> None:
        backups = sorted(
            (p for p in self._root.iterdir() if p.is_dir() and not p.name.endswith(".tmp")),
            key=lambda p: p.name,
        )
        for old in backups[: max(0, len(backups) - self._retention)]:
            shutil.rmtree(old, ignore_errors=True)
            log.info("mirror: pruned old backup %s", old.name)

    def _write_latest_pointer(self, result: MirrorResult) -> None:
        try:
            atomic_write_text(self._root / LATEST_STATE_FILENAME, _dumps(result.to_dict()))
        except OSError as e:
            log.warning("mirror: failed to write latest pointer: %s", e)

    def latest_state(self) -> dict[str, Any] | None:
        pointer = self._root / LATEST_STATE_FILENAME
        if not pointer.exists():
            return None
        try:
            import json

            with open(pointer, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _record(self, result: MirrorResult) -> None:
        if self._history_recorder is None:
            return
        try:
            self._history_recorder(result)
        except Exception as e:
            log.warning("mirror: history recorder failed: %s", e)


def _ignore_transient(_dir: str, names: list[str]) -> set[str]:
    # Skip lock files and editor cruft so backups stay clean/idempotent.
    return {n for n in names if n.endswith(".lock") or n in {".DS_Store"}}


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, sort_keys=True)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _utcnow_tag() -> str:
    return _utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


__all__ = [
    "BACKUP_DIRNAME",
    "DEFAULT_RETENTION",
    "LocalMemoryMirror",
    "MirrorResult",
    "MirrorSource",
]
