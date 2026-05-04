"""Regression tests for the concurrency hardening pass.

These tests cover:
- Stale generated PromptFactory: ``__getattr__`` falls back to dynamic rendering
  for templates that exist on disk but are missing from the generated subclass.
- MemoriesManager under multi-threaded contention: atomic writes + per-file
  cross-process locks must not lose updates or yield partial reads.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from interprompt.prompt_factory import PromptFactoryBase
from serena.project import MemoriesManager


class TestPromptFactoryStaleGenerated:
    """Simulate the AttributeError that surfaced before the regen + fallback fix."""

    def _make_factory(self, prompts_dir: Path) -> PromptFactoryBase:
        # Bare PromptFactoryBase, with no generated subclass at all — represents the
        # worst case: a generated factory that has no create_* methods for the templates.
        return PromptFactoryBase(prompts_dir=str(prompts_dir))

    def test_dynamic_render_for_existing_template(self, tmp_path: Path) -> None:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "demo.yml").write_text(
            "prompts:\n  greeting: |\n    Hello {{ name }}!\n",
            encoding="utf-8",
        )

        factory = self._make_factory(prompts)
        # Before this fix, the next line raised AttributeError because PromptFactoryBase
        # has no create_greeting method and no fallback.
        rendered = factory.create_greeting(name="world")
        assert rendered.strip() == "Hello world!"

    def test_unknown_method_still_raises(self, tmp_path: Path) -> None:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "demo.yml").write_text(
            "prompts:\n  greeting: |\n    hi\n",
            encoding="utf-8",
        )
        factory = self._make_factory(prompts)
        with pytest.raises(AttributeError):
            factory.create_does_not_exist()  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            factory.totally_unrelated_attribute  # type: ignore[attr-defined]


class TestMemoriesManagerConcurrency:
    """Multi-thread contention; cross-process semantics are inherited from filelock."""

    def _make(self, tmp_path: Path) -> MemoriesManager:
        data = tmp_path / ".serena"
        data.mkdir()
        return MemoriesManager(serena_data_folder=str(data))

    def test_concurrent_save_never_yields_partial_read(self, tmp_path: Path) -> None:
        mm = self._make(tmp_path)
        # Two distinct payloads of differing length so a torn write would be observable.
        short = "a" * 16
        long = "b" * 4096
        # Seed the file so readers have something to observe from the start.
        mm.save_memory("racy", short, is_tool_context=False)

        observed: list[str] = []
        stop = threading.Event()

        def writer() -> None:
            while not stop.is_set():
                mm.save_memory("racy", short, is_tool_context=False)
                mm.save_memory("racy", long, is_tool_context=False)

        def reader() -> None:
            for _ in range(200):
                observed.append(mm.load_memory("racy"))

        writers = [threading.Thread(target=writer) for _ in range(2)]
        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in writers + readers:
            t.start()
        for t in readers:
            t.join()
        stop.set()
        for t in writers:
            t.join()

        # Every observation must equal one of the two complete payloads — never a torn one.
        valid = {short, long}
        assert observed
        for snapshot in observed:
            assert snapshot in valid, f"Torn read observed: len={len(snapshot)}"

    def test_concurrent_edit_on_distinct_files_does_not_corrupt(self, tmp_path: Path) -> None:
        mm = self._make(tmp_path)
        # Each thread owns its own memory file, so no read-modify-write race exists
        # across threads — but lock bookkeeping inside the manager is still exercised.
        for i in range(8):
            mm.save_memory(f"slot_{i}", "before", is_tool_context=False)

        def edit(slot: int) -> None:
            for _ in range(20):
                mm.edit_memory(
                    f"slot_{slot}",
                    "before",
                    "after",
                    mode="literal",
                    allow_multiple_occurrences=False,
                    is_tool_context=False,
                )
                mm.edit_memory(
                    f"slot_{slot}",
                    "after",
                    "before",
                    mode="literal",
                    allow_multiple_occurrences=False,
                    is_tool_context=False,
                )

        threads = [threading.Thread(target=edit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(8):
            content = mm.load_memory(f"slot_{i}")
            assert content == "before", f"slot_{i} ended in unexpected state: {content!r}"

    def test_move_memory_contention_does_not_deadlock(self, tmp_path: Path) -> None:
        mm = self._make(tmp_path)
        mm.save_memory("a", "alpha", is_tool_context=False)

        errors: list[Exception] = []

        def shuffle() -> None:
            try:
                mm.move_memory("a", "b", is_tool_context=False)
                mm.move_memory("b", "a", is_tool_context=False)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=shuffle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        for t in threads:
            assert not t.is_alive(), "deadlock: thread still running after timeout"

        # Errors are acceptable (FileExistsError when two threads both try to land on 'b'
        # at once), but no thread may have hung. The final state must contain exactly one
        # of the two memories.
        names = mm.list_memories().get_full_list()
        assert names in (["a"], ["b"]), f"unexpected final state: {names}"
