from typing import Literal

from serena.project import MemoriesManager
from serena.tools import Tool, ToolMarkerCanEdit


class WriteMemoryTool(Tool, ToolMarkerCanEdit):
    """
    Write some information (utf-8-encoded) about this project that can be useful for future tasks to a memory in md format.
    The memory name should be meaningful.
    """

    def apply(
        self,
        memory_name: str,
        content: str,
        max_chars: int = -1,
        tags: list[str] | None = None,
    ) -> str:
        """
        Write information (utf-8-encoded) about this project that can be useful for future tasks to a memory in md format.
        The memory name should be meaningful and can include "/" to organize into topics (e.g., "auth/login/logic").
        If explicitly instructed, use the "global/" prefix for writing a memory that is shared across projects
        (e.g., "global/java/style_guide")

        :param max_chars: the maximum number of characters to write. By default, determined by the config,
            change only if instructed to do so.
        :param tags: optional list of short string tags. When provided, they are serialized as YAML
            frontmatter (``tags: [...]``) at the top of the memory so future searches can filter by tag.
            Existing memories without frontmatter remain valid.
        """
        # NOTE: utf-8 encoding is configured in the MemoriesManager
        if max_chars == -1:
            max_chars = self.agent.serena_config.default_max_tool_answer_chars

        if tags:
            normalized_tags = [str(t).strip() for t in tags if str(t).strip()]
            if normalized_tags:
                # if caller already provided their own frontmatter, leave content alone
                if not content.startswith(MemoriesManager._FRONTMATTER_DELIM + "\n"):
                    content = MemoriesManager.serialize_frontmatter({"tags": normalized_tags}) + content

        if len(content) > max_chars:
            raise ValueError(
                f"Content for {memory_name} is too long. Max length is {max_chars} characters. " + "Please make the content shorter."
            )

        return self.memories_manager.save_memory(memory_name, content, is_tool_context=True)


class ReadMemoryTool(Tool):
    """
    Read the content of a memory file. This tool should only be used if the information
    is relevant to the current task. You can infer whether the information
    is relevant from the memory file name.
    You should not read the same memory file multiple times in the same conversation.
    """

    def apply(self, memory_name: str) -> str:
        """
        Reads the contents of a memory. Should only be used if the information
        is likely to be relevant to the current task, inferring relevance from the memory name.
        """
        return self.memories_manager.load_memory(memory_name)


class ListMemoriesTool(Tool):
    """
    List available memories. Any memory can be read using the `read_memory` tool.
    """

    def apply(self, topic: str = "") -> str:
        """
        Lists available memories, optionally filtered by topic.
        """
        return self._to_json(self.memories_manager.list_memories(topic).to_dict())


class DeleteMemoryTool(Tool, ToolMarkerCanEdit):
    """
    Delete a memory file. Should only happen if a user asks for it explicitly,
    for example by saying that the information retrieved from a memory file is no longer correct
    or no longer relevant for the project.
    """

    def apply(self, memory_name: str) -> str:
        """
        Delete a memory, only call if instructed explicitly or permission was granted by the user.
        """
        return self.memories_manager.delete_memory(memory_name, is_tool_context=True)


class RenameMemoryTool(Tool, ToolMarkerCanEdit):
    """
    Renames or moves a memory. Moving between project and global scope is supported
    (e.g., renaming "global/foo" to "bar" moves it from global to project scope).
    """

    def apply(self, old_name: str, new_name: str) -> str:
        """
        Rename or move a memory, use "/" in the name to organize into topics.
        The "global" topic should only be used if explicitly instructed.
        """
        return self.memories_manager.move_memory(old_name, new_name, is_tool_context=True)


class EditMemoryTool(Tool, ToolMarkerCanEdit):
    """
    Replaces content matching a regular expression in a memory.
    """

    def apply(
        self,
        memory_name: str,
        needle: str,
        repl: str,
        mode: Literal["literal", "regex"],
        allow_multiple_occurrences: bool = False,
    ) -> str:
        r"""
        Replaces content matching a regular expression in a memory.

        :param memory_name: the name of the memory
        :param needle: the string or regex pattern to search for.
            If `mode` is "literal", this string will be matched exactly.
            If `mode` is "regex", this string will be treated as a regular expression (syntax of Python's `re` module,
            with flags DOTALL and MULTILINE enabled).
        :param repl: the replacement string (verbatim).
        :param mode: either "literal" or "regex", specifying how the `needle` parameter is to be interpreted.
        :param allow_multiple_occurrences: whether to allow matching and replacing multiple occurrences.
            If false and multiple occurrences are found, an error will be returned.
        """
        return self.memories_manager.edit_memory(memory_name, needle, repl, mode, allow_multiple_occurrences, is_tool_context=True)


class SearchMemoriesTool(Tool):
    """
    Search the contents of project and global memories for a substring or regex pattern.

    Returns matched memory names with 0-based snippet line numbers, ordered by match
    frequency (descending) and then memory name. Useful for discovering which memory to
    read in full afterwards. Snippets are truncated to a configurable length.

    Tag filtering is *disjunctive*: when a tags list is passed, a memory matches if any of
    its YAML-frontmatter ``tags`` overlap with the requested tags. Memories whose
    frontmatter is absent or malformed have no tags and are skipped when the tag filter
    is non-empty.
    """

    def apply(
        self,
        pattern: str,
        mode: Literal["literal", "regex"] = "literal",
        topic: str = "",
        tags: list[str] | None = None,
        max_memories: int = 20,
        max_matches_per_memory: int = 5,
        case_sensitive: bool = False,
        max_answer_chars: int = -1,
    ) -> str:
        """
        Search across memory file contents.

        :param pattern: substring (when ``mode='literal'``) or Python regex (when ``mode='regex'``) to search for.
        :param mode: ``literal`` or ``regex``. Default ``literal``.
        :param topic: optional topic prefix (e.g. ``auth`` or ``global/java``) to scope the search.
        :param tags: optional list of tags; when provided, only memories whose YAML-frontmatter
            ``tags`` field overlaps are searched. Memories without frontmatter are skipped if tags are passed.
        :param max_memories: hard cap on number of distinct memories returned (default 20).
        :param max_matches_per_memory: hard cap on snippets per memory (default 5).
        :param case_sensitive: if true, match case-sensitively. Default false.
        :param max_answer_chars: total character cap for the response. -1 uses the agent default.
        :return: JSON describing matches and aggregated counts per memory.
        """
        if max_answer_chars == -1:
            max_answer_chars = self.agent.serena_config.default_max_tool_answer_chars

        try:
            matches = self.memories_manager.search_memories(
                pattern=pattern,
                mode=mode,
                topic=topic,
                tags=tuple(tags or ()),
                max_memories=max_memories,
                max_matches_per_memory=max_matches_per_memory,
                case_sensitive=case_sensitive,
            )
        except Exception as e:
            raise ValueError(f"search_memories failed: {e}") from e

        per_memory: dict[str, dict] = {}
        for m in matches:
            entry = per_memory.setdefault(
                m.memory_name,
                {"memory_name": m.memory_name, "is_read_only": m.is_read_only, "match_count": 0, "matches": []},
            )
            entry["match_count"] += 1
            entry["matches"].append({"line": m.line_number, "snippet": m.snippet})

        ranked = sorted(per_memory.values(), key=lambda e: (-e["match_count"], e["memory_name"]))
        result = {
            "pattern": pattern,
            "mode": mode,
            "topic": topic,
            "tags": tags or [],
            "memories_found": len(ranked),
            "results": ranked,
        }
        result_json = self._to_json(result)

        def make_summary() -> str:
            return self._to_json(
                {
                    "memories_found": len(ranked),
                    "memory_names": [e["memory_name"] for e in ranked],
                }
            )

        return self._limit_length(result_json, max_answer_chars, shortened_result_factories=[make_summary])
