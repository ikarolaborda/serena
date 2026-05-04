"""
Cross-process safe file I/O helpers for shared Serena state.

Multiple Serena instances (one per agent) often share the same on-disk locations
under ``$projectDir/.serena`` and ``~/.serena``. The standard ``open(path, "w")``
pattern truncates before writing, which is neither atomic for concurrent readers
nor safe against two processes racing on a read-modify-write cycle. The helpers
here wrap the canonical "write to temp, fsync, rename" pattern and the
``filelock`` cross-process advisory lock so callers do not have to reinvent
either.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Iterator

from filelock import FileLock, Timeout

DEFAULT_LOCK_TIMEOUT = 30.0
"""Seconds to wait for a cross-process file lock before raising Timeout."""


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write ``content`` to ``path`` atomically: write to a sibling temp file, fsync, then rename.
    Concurrent readers either see the old content or the new content, never a partial write.
    Concurrent writers still need a cross-process lock if read-modify-write semantics matter.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


@contextlib.contextmanager
def cross_process_lock(lock_path: str | Path, timeout: float = DEFAULT_LOCK_TIMEOUT) -> Iterator[None]:
    """
    Cross-process advisory lock around a critical section. Lock files are created
    next to the resource they protect (e.g. ``foo.md.lock`` next to ``foo.md``)
    so that recursive listings of the resource directory naturally surface them.

    Re-raises ``filelock.Timeout`` so callers can decide between waiting longer
    or falling back to a non-locking read.
    """
    lock_target = Path(lock_path)
    lock_target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_target), timeout=timeout)
    try:
        with lock:
            yield
    except Timeout:
        raise


__all__ = ["DEFAULT_LOCK_TIMEOUT", "atomic_write_text", "atomic_write_bytes", "cross_process_lock"]
