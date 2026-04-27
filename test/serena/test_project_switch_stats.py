"""Unit tests for ProjectSwitchEvent and ProjectSwitchStats."""

from __future__ import annotations

import threading

from serena.analytics import ProjectSwitchEvent, ProjectSwitchStats


def _make_event(activation_id: int, from_project: str | None, to_project: str) -> ProjectSwitchEvent:
    return ProjectSwitchEvent(
        activation_id=activation_id,
        from_project=from_project,
        to_project=to_project,
        shutdown_ms=0.0,
        ls_init_started=True,
    )


def test_first_activation_event_flags() -> None:
    event = _make_event(1, None, "alpha")
    assert event.is_first_activation is True
    assert event.is_switch is False


def test_switch_event_flags() -> None:
    event = _make_event(2, "alpha", "beta")
    assert event.is_first_activation is False
    assert event.is_switch is True


def test_same_project_reactivation_event_flags() -> None:
    event = _make_event(3, "alpha", "alpha")
    assert event.is_first_activation is False
    assert event.is_switch is False


def test_record_preserves_order_and_summary() -> None:
    stats = ProjectSwitchStats()
    stats.record(_make_event(1, None, "alpha"))
    stats.record(_make_event(2, "alpha", "beta"))
    stats.record(_make_event(3, "beta", "gamma"))

    events = stats.get_events()
    assert [event.activation_id for event in events] == [1, 2, 3]

    summary = stats.get_summary()
    assert summary["total_recorded"] == 3
    assert summary["switch_count"] == 2
    assert summary["last"]["activation_id"] == 3
    assert summary["last"]["to_project"] == "gamma"


def test_clear_resets_stats() -> None:
    stats = ProjectSwitchStats()
    stats.record(_make_event(1, None, "alpha"))
    stats.record(_make_event(2, "alpha", "beta"))
    stats.clear()

    summary = stats.get_summary()
    assert summary["total_recorded"] == 0
    assert summary["switch_count"] == 0
    assert summary["last"] is None


def test_ring_buffer_evicts_oldest_when_full() -> None:
    stats = ProjectSwitchStats(max_events=5)
    for i in range(1, 11):
        stats.record(_make_event(i, f"p{i - 1}", f"p{i}"))

    events = stats.get_events()
    assert len(events) == 5
    assert [event.activation_id for event in events] == [6, 7, 8, 9, 10]


def test_concurrent_writes_are_thread_safe() -> None:
    stats = ProjectSwitchStats(max_events=200)
    threads_count = 10
    events_per_thread = 10

    def worker(thread_id: int) -> None:
        for index in range(events_per_thread):
            activation_id = thread_id * events_per_thread + index
            stats.record(_make_event(activation_id, f"src{thread_id}", f"dst{thread_id}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = stats.get_summary()
    assert summary["total_recorded"] == threads_count * events_per_thread
    assert summary["switch_count"] == threads_count * events_per_thread
