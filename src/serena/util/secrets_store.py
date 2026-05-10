"""
SQLite-backed secrets store for operator-managed credentials.

Mirrors the protection level of the existing ``qdrant-memory/.env.r2`` file
(plaintext at rest, gitignored, ``chmod 600``). The store is intentionally
single-file SQLite so it can be wiped / inspected / backed up without
introducing an extra runtime dependency. Encryption-at-rest is out of scope
for v1; if a future need arises, the schema is forward-compatible — the
``value`` column can carry an encrypted blob alongside a ``cipher`` tag.

Threading: each call opens its own short-lived connection so the store is
safe to use from both the dashboard request threads and the background
sync task. SQLite handles serialization at the file level.

Storage location: ``~/.serena/secrets.sqlite3`` (or the path passed to
``SecretsStore(path=...)``). The parent directory is created with mode 700
and the database file with mode 600 on first creation.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from serena.config.serena_config import SerenaPaths

SECRETS_DB_FILENAME = "secrets.sqlite3"
"""Default basename under SerenaPaths().serena_user_home_dir."""

MASK_VISIBLE_PREFIX = 4
MASK_VISIBLE_SUFFIX = 4
"""When masking, keep the leading and trailing N chars verbatim."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS secrets (
    scope       TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (scope, key)
) WITHOUT ROWID;
"""


def mask_secret(value: str | None) -> str:
    """Return a masked representation safe for UI / logs.

    Short values (<= visible prefix + suffix + 4) collapse to ``"****"`` to
    avoid leaking via length-revealing patterns on tiny secrets.
    """
    if value is None or value == "":
        return ""
    threshold = MASK_VISIBLE_PREFIX + MASK_VISIBLE_SUFFIX + 4
    if len(value) <= threshold:
        return "****"
    return value[:MASK_VISIBLE_PREFIX] + "****" + value[-MASK_VISIBLE_SUFFIX:]


class SecretsStore:
    """SQLite-backed key/value store scoped by ``scope`` (e.g. ``"r2"``)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        if path is None:
            path = Path(SerenaPaths().serena_user_home_dir) / SECRETS_DB_FILENAME
        self._path = Path(path)
        # In-process lock so concurrent writers in the same Serena instance
        # serialise their schema-init + insert/update path without depending
        # on SQLite's BEGIN IMMEDIATE upgrade dance.
        self._lock = threading.Lock()
        self._ensure_storage()

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_storage(self) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Best-effort tighten parent perms; ignore if the FS doesn't support
        # POSIX modes (Windows, network mounts).
        with contextlib.suppress(OSError):
            os.chmod(parent, 0o700)
        new_db = not self._path.exists()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        if new_db:
            with contextlib.suppress(OSError):
                os.chmod(self._path, 0o600)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path), timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def set_secret(self, scope: str, key: str, value: str) -> None:
        if not scope or not key:
            raise ValueError("scope and key must be non-empty")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO secrets(scope, key, value) VALUES(?, ?, ?) "
                "ON CONFLICT(scope, key) DO UPDATE SET "
                "value = excluded.value, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (scope, key, value),
            )

    def get_secret(self, scope: str, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM secrets WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
        return row[0] if row else None

    def list_scope(self, scope: str) -> dict[str, str]:
        """Return all key→value pairs in a scope. Raw values; mask at boundary."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM secrets WHERE scope = ? ORDER BY key",
                (scope,),
            ).fetchall()
        return {k: v for k, v in rows}

    def list_scope_masked(self, scope: str) -> dict[str, str]:
        return {k: mask_secret(v) for k, v in self.list_scope(scope).items()}

    def list_scopes(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT scope FROM secrets ORDER BY scope").fetchall()
        return [r[0] for r in rows]

    def delete_secret(self, scope: str, key: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM secrets WHERE scope = ? AND key = ?",
                (scope, key),
            )
            return cur.rowcount > 0

    def delete_scope(self, scope: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM secrets WHERE scope = ?", (scope,))
            return cur.rowcount


__all__ = ["SecretsStore", "mask_secret", "SECRETS_DB_FILENAME"]
