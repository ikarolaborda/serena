"""Tests for the always-on local backup mirror (serena.memory_mirror)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serena.memory_mirror import LocalMemoryMirror


def _make_memories(root: Path) -> Path:
    mem = root / "memories"
    (mem / "global").mkdir(parents=True)
    (mem / "core.md").write_text("# core\nmem:tech_stack\n", encoding="utf-8")
    (mem / "global" / "g.md").write_text("global note\n", encoding="utf-8")
    return mem


def _make_qdrant_with_snapshot(root: Path, snapshot_id: str = "2026-05-28T09-52-21Z-host") -> Path:
    q = root / "qdrant"
    staging = q / "data" / "r2-staging"
    staging.mkdir(parents=True)
    (staging / f"memory-sync-{snapshot_id}.tar.gz").write_bytes(b"FAKE" * 2000)
    (q / "data" / "r2-state.json").write_text(json.dumps({"last_push": {"snapshot_id": snapshot_id}}), encoding="utf-8")
    return q


def _mirror(tmp_path: Path, *, with_qdrant: bool = True, retention: int = 8) -> LocalMemoryMirror:
    mem = _make_memories(tmp_path)
    qdrant = _make_qdrant_with_snapshot(tmp_path) if with_qdrant else tmp_path / "absent-qdrant"
    return LocalMemoryMirror(
        backup_root=tmp_path / "backups",
        serena_memories_dir=mem,
        qdrant_memory_dir=qdrant,
        retention=retention,
    )


def test_mirror_creates_backup_with_both_sources(tmp_path: Path) -> None:
    m = _mirror(tmp_path)
    result = m.mirror_now("test")
    assert result.outcome == "ok"
    assert result.snapshot_id == "2026-05-28T09-52-21Z-host"
    dest = Path(result.dest)
    assert dest.exists()
    assert (dest / "serena-memories" / "core.md").read_text(encoding="utf-8").startswith("# core")
    assert any((dest / "qdrant-snapshot").glob("*.tar.gz"))
    assert (dest / "manifest.json").exists()
    labels = {s["label"] for s in result.sources}
    assert labels == {"serena-memories", "qdrant-snapshot"}


def test_unchanged_backup_is_skipped(tmp_path: Path) -> None:
    m = _mirror(tmp_path)
    assert m.mirror_now("first").outcome == "ok"
    second = m.mirror_now("again")
    assert second.outcome == "skipped"
    backups = [p for p in (tmp_path / "backups").iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
    assert len(backups) == 1


def test_memory_edit_triggers_new_backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mirror(tmp_path)
    assert m.mirror_now("first").outcome == "ok"
    # change content; bump mtime deterministically rather than sleeping
    edited = m.serena_memories_dir / "core.md"
    edited.write_text("# core changed\n", encoding="utf-8")
    import os as _os

    future = edited.stat().st_mtime + 10
    _os.utime(edited, (future, future))
    assert m.mirror_now("after-edit").outcome == "ok"
    backups = [p for p in (tmp_path / "backups").iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
    assert len(backups) == 2


def test_retention_prunes_oldest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    import serena.memory_mirror as mod

    m = _mirror(tmp_path, retention=3)
    f = m.serena_memories_dir / "core.md"
    base = f.stat().st_mtime
    for i in range(6):
        # distinct content + mtime so the fingerprint changes each round
        f.write_text(f"rev {i}\n", encoding="utf-8")
        _os.utime(f, (base + i + 1, base + i + 1))
        # distinct backup dir names require distinct timestamp tags
        monkeypatch.setattr(mod, "_utcnow_tag", lambda i=i: f"2026-05-28T10-00-0{i}Z")
        m.mirror_now(f"e{i}")
    kept = [p for p in (tmp_path / "backups").iterdir() if p.is_dir() and not p.name.endswith(".tmp")]
    assert len(kept) == 3


def test_no_sources_is_unavailable(tmp_path: Path) -> None:
    m = LocalMemoryMirror(
        backup_root=tmp_path / "b",
        serena_memories_dir=tmp_path / "nope",
        qdrant_memory_dir=tmp_path / "noq",
    )
    assert m.mirror_now("x").outcome == "unavailable"


def test_qdrant_snapshot_absent_still_backs_up_memories(tmp_path: Path) -> None:
    m = _mirror(tmp_path, with_qdrant=False)
    result = m.mirror_now("mem-only")
    assert result.outcome == "ok"
    assert [s["label"] for s in result.sources] == ["serena-memories"]


def test_snapshot_entry_records_age_and_sha256(tmp_path: Path) -> None:
    m = _mirror(tmp_path)
    result = m.mirror_now("test")
    snap = next(s for s in result.sources if s["label"] == "qdrant-snapshot")
    assert "sha256" in snap and len(snap["sha256"]) == 64
    assert "age_seconds" in snap and snap["age_seconds"] >= 0


def test_latest_state_matches_last_backup(tmp_path: Path) -> None:
    m = _mirror(tmp_path)
    result = m.mirror_now("test")
    state = m.latest_state()
    assert state is not None
    assert state["outcome"] == "ok"
    assert state["dest"] == result.dest
    assert state["fingerprint"] == result.fingerprint


def test_mirror_never_raises_when_source_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mirror(tmp_path, with_qdrant=False)

    import shutil as _shutil

    def boom(*_a, **_k):
        raise OSError("copy blew up")

    monkeypatch.setattr(_shutil, "copytree", boom)
    # Must not raise; with the only source failing, outcome is "failed".
    result = m.mirror_now("boom")
    assert result.outcome == "failed"
