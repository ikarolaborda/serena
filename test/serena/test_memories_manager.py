"""Tests for MemoriesManager additions: frontmatter parsing, search, tagging."""
from __future__ import annotations

from pathlib import Path

import pytest

from serena.project import MemoriesManager


@pytest.fixture
def memories_manager(tmp_path: Path) -> MemoriesManager:
    serena_data_folder = tmp_path / ".serena"
    serena_data_folder.mkdir()
    return MemoriesManager(serena_data_folder=str(serena_data_folder))


class TestFrontmatterParsing:
    def test_no_frontmatter_returns_empty_metadata(self) -> None:
        metadata, body = MemoriesManager.parse_frontmatter("# Some title\n\nbody text\n")
        assert metadata == {}
        assert body == "# Some title\n\nbody text\n"

    def test_well_formed_frontmatter_extracts_supported_fields(self) -> None:
        raw = (
            "---\n"
            'tags: ["bug", "auth"]\n'
            "type: incident\n"
            "description: post-mortem of session leak\n"
            "---\n"
            "Body starts here.\n"
        )
        metadata, body = MemoriesManager.parse_frontmatter(raw)
        assert metadata == {
            "tags": ["bug", "auth"],
            "type": "incident",
            "description": "post-mortem of session leak",
        }
        assert body == "Body starts here.\n"

    def test_csv_tags_form_is_accepted(self) -> None:
        raw = "---\ntags: bug, auth, perf\n---\nbody\n"
        metadata, _ = MemoriesManager.parse_frontmatter(raw)
        assert metadata["tags"] == ["bug", "auth", "perf"]

    def test_unknown_keys_are_ignored(self) -> None:
        raw = "---\nfoo: bar\ntags: [a]\n---\nbody\n"
        metadata, _ = MemoriesManager.parse_frontmatter(raw)
        assert metadata == {"tags": ["a"]}

    def test_unterminated_frontmatter_falls_back_to_full_body(self) -> None:
        raw = "---\ntags: [a]\nno-closing-delim\nbody continues\n"
        metadata, body = MemoriesManager.parse_frontmatter(raw)
        assert metadata == {}
        assert body == raw

    def test_legacy_thematic_break_does_not_misparse(self) -> None:
        raw = "---\nNot YAML at all, this looks like a thematic break\n"
        metadata, body = MemoriesManager.parse_frontmatter(raw)
        # missing closing delimiter -> conservative fallback
        assert metadata == {}
        assert body == raw

    def test_serialize_then_parse_roundtrip(self) -> None:
        block = MemoriesManager.serialize_frontmatter({"tags": ["x", "y"], "type": "decision"})
        parsed, body = MemoriesManager.parse_frontmatter(block + "body\n")
        assert parsed == {"tags": ["x", "y"], "type": "decision"}
        assert body == "body\n"


class TestSearchMemories:
    def _write(self, mgr: MemoriesManager, name: str, body: str) -> None:
        mgr.save_memory(name, body, is_tool_context=False)

    def test_substring_search_returns_matches(self, memories_manager: MemoriesManager) -> None:
        self._write(memories_manager, "auth/login", "Handles login with JWT.\nValidates session.\n")
        self._write(memories_manager, "infra/redis", "Redis caches login tokens for 1h.\n")
        results = memories_manager.search_memories("login")
        names = sorted(r.memory_name for r in results)
        assert names == ["auth/login", "infra/redis"]

    def test_regex_mode(self, memories_manager: MemoriesManager) -> None:
        self._write(memories_manager, "perf", "p99=120ms baseline\np50=30ms\n")
        results = memories_manager.search_memories(r"p\d+=\d+ms", mode="regex")
        assert len(results) == 2
        assert all(r.memory_name == "perf" for r in results)

    def test_tags_filter_skips_untagged(self, memories_manager: MemoriesManager) -> None:
        self._write(
            memories_manager,
            "tagged",
            MemoriesManager.serialize_frontmatter({"tags": ["bug"]}) + "Has the keyword foo.\n",
        )
        self._write(memories_manager, "untagged", "Has the keyword foo too.\n")
        results = memories_manager.search_memories("foo", tags=["bug"])
        assert [r.memory_name for r in results] == ["tagged"]

    def test_caps_are_respected(self, memories_manager: MemoriesManager) -> None:
        body = "\n".join(["match here"] * 50) + "\n"
        self._write(memories_manager, "long", body)
        results = memories_manager.search_memories("match", max_matches_per_memory=3)
        assert len(results) == 3

    def test_snippet_truncation(self, memories_manager: MemoriesManager) -> None:
        long_line = "match " + ("x" * 1000)
        self._write(memories_manager, "wide", long_line + "\n")
        results = memories_manager.search_memories("match", snippet_max_chars=50)
        assert len(results[0].snippet) <= 50
        assert results[0].snippet.endswith("…")

    def test_invalid_regex_raises(self, memories_manager: MemoriesManager) -> None:
        self._write(memories_manager, "any", "content\n")
        with pytest.raises(Exception):
            memories_manager.search_memories("[unclosed", mode="regex")

    def test_frontmatter_body_not_polluted_by_match_count(self, memories_manager: MemoriesManager) -> None:
        # The frontmatter delimiter "---" must not be matched when searching for "---"
        self._write(
            memories_manager,
            "fm",
            MemoriesManager.serialize_frontmatter({"tags": ["a"]}) + "the body has the word frontier\n",
        )
        results = memories_manager.search_memories("---")
        assert results == []

    def test_write_then_search_e2e(self, memories_manager: MemoriesManager) -> None:
        """End-to-end: tags written via frontmatter are queryable via tag filter
        and surface in read_memory_metadata."""
        body = MemoriesManager.serialize_frontmatter({"tags": ["bug", "auth"], "type": "incident"}) + (
            "Login flow returned 500 on token refresh.\nFixed by rotating the JWK set.\n"
        )
        self._write(memories_manager, "auth/login-incident", body)
        # tag-filtered search hits
        hits = memories_manager.search_memories("token", tags=["bug"])
        assert [h.memory_name for h in hits] == ["auth/login-incident"]
        assert hits[0].line_number == 0  # 0-based: "Login flow returned 500 on token refresh."
        # metadata is exposed
        meta = memories_manager.read_memory_metadata("auth/login-incident")
        assert meta == {"tags": ["bug", "auth"], "type": "incident"}
        # tag-filtered search miss when filter doesn't overlap
        assert memories_manager.search_memories("token", tags=["unrelated"]) == []
