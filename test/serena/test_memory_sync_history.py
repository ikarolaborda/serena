"""Tests for serena.memory_sync_history (history + watcher)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serena.memory_sync_history import (
    MemorySyncHistory,
    MemorySyncWatcher,
    SyncEvent,
)


@pytest.fixture
def history(tmp_path: Path) -> MemorySyncHistory:
    return MemorySyncHistory(path=tmp_path / "history.jsonl")


def _write_state(path: Path, snapshot_id: str, **extra: object) -> None:
    payload = {
        "last_push": {
            "snapshot_id": snapshot_id,
            "archive_size_bytes": 123,
            "archive_sha256": "deadbeef",
            "pushed_from": "TEST",
            "pushed_at": "2026-05-11T00:00:00Z",
            **extra,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestHistoryAppend:
    def test_append_and_read_back(self, history: MemorySyncHistory) -> None:
        history.append(SyncEvent(ts="2026-05-11T00:00:00Z", project="a", snapshot_id="s1", source="self", outcome="ok"))
        history.append(SyncEvent(ts="2026-05-11T00:01:00Z", project="b", snapshot_id="s2", source="external", outcome="ok"))
        events = history.read_all()
        assert [e.snapshot_id for e in events] == ["s1", "s2"]

    def test_project_filter(self, history: MemorySyncHistory) -> None:
        history.append(SyncEvent(ts="t1", project="a", snapshot_id="s1", source="self", outcome="ok"))
        history.append(SyncEvent(ts="t2", project="b", snapshot_id="s2", source="self", outcome="ok"))
        history.append(SyncEvent(ts="t3", project="a", snapshot_id="s3", source="external", outcome="ok"))
        only_a = history.recent(project="a")
        assert [e.snapshot_id for e in only_a] == ["s3", "s1"]  # newest first

    def test_recent_orders_newest_first(self, history: MemorySyncHistory) -> None:
        for i in range(5):
            history.append(SyncEvent(ts=f"t{i}", project="a", snapshot_id=f"s{i}", source="self", outcome="ok"))
        out = history.recent(project="a", limit=3)
        assert [e.snapshot_id for e in out] == ["s4", "s3", "s2"]

    def test_recent_no_filter_returns_all_projects(self, history: MemorySyncHistory) -> None:
        history.append(SyncEvent(ts="t1", project="a", snapshot_id="s1", source="self", outcome="ok"))
        history.append(SyncEvent(ts="t2", project="b", snapshot_id="s2", source="external", outcome="ok"))
        out = history.recent(project=None)
        assert len(out) == 2

    def test_empty_history_returns_empty_list(self, history: MemorySyncHistory) -> None:
        assert history.read_all() == []
        assert history.recent(project="a") == []

    def test_corrupt_line_is_skipped(self, history: MemorySyncHistory) -> None:
        history.append(SyncEvent(ts="t1", project="a", snapshot_id="s1", source="self", outcome="ok"))
        # Append a malformed line directly so the JSONL reader has to skip it.
        with open(history.path, "a", encoding="utf-8") as f:
            f.write("this is not json\n")
        history.append(SyncEvent(ts="t2", project="a", snapshot_id="s2", source="self", outcome="ok"))
        ids = [e.snapshot_id for e in history.read_all()]
        assert ids == ["s1", "s2"]


class TestWatcher:
    def test_seeds_existing_snapshot_silently(self, tmp_path: Path, history: MemorySyncHistory) -> None:
        state = tmp_path / "data" / "r2-state.json"
        _write_state(state, "snap-0")
        watcher = MemorySyncWatcher(
            state_file=state,
            history=history,
            active_project_provider=lambda: "a",
            poll_seconds=0.0,
        )
        # First poll must NOT record the seed snapshot — the watcher already
        # knew about it at start time.
        assert watcher.poll_once() is False
        assert history.read_all() == []

    def test_detects_new_snapshot_and_tags_project(self, tmp_path: Path, history: MemorySyncHistory) -> None:
        state = tmp_path / "data" / "r2-state.json"
        _write_state(state, "snap-0")
        watcher = MemorySyncWatcher(
            state_file=state,
            history=history,
            active_project_provider=lambda: "greenmetrics",
            poll_seconds=0.0,
        )
        _write_state(state, "snap-1", archive_key="archives/a.tar.gz")
        assert watcher.poll_once() is True
        rows = history.read_all()
        assert len(rows) == 1
        assert rows[0].snapshot_id == "snap-1"
        assert rows[0].project == "greenmetrics"
        assert rows[0].source == "external"
        assert rows[0].outcome == "ok"
        assert rows[0].extra.get("archive_key") == "archives/a.tar.gz"

    def test_repeated_poll_without_change_is_noop(self, tmp_path: Path, history: MemorySyncHistory) -> None:
        state = tmp_path / "data" / "r2-state.json"
        _write_state(state, "snap-0")
        watcher = MemorySyncWatcher(
            state_file=state,
            history=history,
            active_project_provider=lambda: "a",
            poll_seconds=0.0,
        )
        _write_state(state, "snap-1")
        assert watcher.poll_once() is True
        assert watcher.poll_once() is False
        assert len(history.read_all()) == 1

    def test_missing_state_file_is_noop(self, tmp_path: Path, history: MemorySyncHistory) -> None:
        watcher = MemorySyncWatcher(
            state_file=tmp_path / "nope.json",
            history=history,
            active_project_provider=lambda: "a",
            poll_seconds=0.0,
        )
        assert watcher.poll_once() is False
        assert history.read_all() == []

    def test_record_self_event_suppresses_external_double_record(
        self, tmp_path: Path, history: MemorySyncHistory
    ) -> None:
        # When Serena triggers a sync, we record it as "self". The next
        # state-file poll must NOT record the same snapshot id as "external".
        state = tmp_path / "data" / "r2-state.json"
        _write_state(state, "snap-0")
        watcher = MemorySyncWatcher(
            state_file=state,
            history=history,
            active_project_provider=lambda: "a",
            poll_seconds=0.0,
        )
        watcher.record_self_event(project="a", snapshot_id="snap-1", outcome="ok")
        _write_state(state, "snap-1")
        assert watcher.poll_once() is False
        rows = history.read_all()
        assert len(rows) == 1 and rows[0].source == "self"
