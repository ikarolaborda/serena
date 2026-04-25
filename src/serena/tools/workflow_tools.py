"""
Tools supporting the general workflow of the agent
"""

import platform

from serena.tools import ReadMemoryTool, Tool, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional, WriteMemoryTool


class CheckOnboardingPerformedTool(Tool):
    """
    Checks whether project onboarding was already performed.
    """

    def apply(self) -> str:
        """
        Checks whether project onboarding was already performed.
        You should always call this tool before beginning to actually work on the project/after activating a project.
        """
        read_memory_tool_available = self.agent.tool_is_exposed(ReadMemoryTool.get_name_from_cls())
        perform_onboarding_tool_available = self.agent.tool_is_exposed(OnboardingTool.get_name_from_cls())

        if not read_memory_tool_available:
            return "Memory reading tool not activated, skipping onboarding check."
        project_memories = self.memories_manager.list_project_memories()
        if len(project_memories) == 0:
            msg = "Onboarding not performed yet (no memories available). "
            if perform_onboarding_tool_available:
                msg += "You should perform onboarding by calling the `onboarding` tool before proceeding with the task. "
        else:
            # Not reporting the list of memories here, as they were already reported at project activation
            # (with the system prompt if the project was activated at startup)
            msg = (
                f"Onboarding was already performed: {len(project_memories)} project memories are available. "
                "Consider reading memories if they appear relevant to the task at hand."
            )
        msg += " If you have not read the 'Serena Instructions Manual', do so now."
        return msg


class OnboardingTool(Tool):
    """
    Performs onboarding (identifying the project structure and essential tasks, e.g. for testing or building).
    """

    def apply(self) -> str:
        """
        Call this tool if onboarding was not performed yet.
        You will call this tool at most once per conversation.

        :return: instructions on how to create the onboarding information
        """
        write_memory_tool_available = self.agent.tool_is_exposed(WriteMemoryTool.get_name_from_cls())
        if not write_memory_tool_available:
            return "Memory writing tool not activated, skipping onboarding."
        system = platform.system()
        return self.prompt_factory.create_onboarding_prompt(system=system)


class ThinkAboutCollectedInformationTool(Tool, ToolMarkerOptional):
    """
    Thinking tool for pondering the completeness of collected information.
    """

    def apply(self) -> str:
        """
        Think about the collected information and whether it is sufficient and relevant.
        This tool should ALWAYS be called after you have completed a non-trivial sequence of searching steps like
        find_symbol, find_referencing_symbols, search_for_pattern, read_file, etc.
        """
        return self.prompt_factory.create_think_about_collected_information()


class ThinkAboutTaskAdherenceTool(Tool, ToolMarkerOptional):
    """
    Thinking tool for determining whether the agent is still on track with the current task.
    """

    def apply(self) -> str:
        """
        Think about the task at hand and whether you are still on track.
        Especially important if the conversation has been going on for a while and there
        has been a lot of back and forth.

        This tool should ALWAYS be called before you insert, replace, or delete code.
        """
        return self.prompt_factory.create_think_about_task_adherence()


class ThinkAboutWhetherYouAreDoneTool(Tool, ToolMarkerOptional):
    """
    Thinking tool for determining whether the task is truly completed.
    """

    def apply(self) -> str:
        """
        Whenever you feel that you are done with what the user has asked for, it is important to call this tool.
        """
        return self.prompt_factory.create_think_about_whether_you_are_done()


class InitialInstructionsTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Provides instructions Serena usage (i.e. the 'Serena Instructions Manual')
    for clients that do not read the initial instructions when the MCP server is connected.
    """

    # noinspection PyIncorrectDocstring
    # (session_id is injected via apply_ex)
    def apply(self, session_id: str) -> str:
        """
        Provides the 'Serena Instructions Manual', which contains essential information on how to use the Serena toolbox.
        IMPORTANT: If you have not yet read the manual, call this tool immediately after you are given your task by the user,
        as it will critically inform you!
        """
        return self.agent.create_system_prompt(session_id=session_id)


class SerenaInfoTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """
    Provides information about an advanced topic on demand, facilitating context-efficiency.
    """

    def apply(self, topic: str) -> str:
        """
        Retrieves Serena-specific information
        :param topic: the topic, which you must have been given explicitly
        """
        match topic:
            case "jet_brains_debug_repl":
                return self.agent.prompt_factory.create_info_jet_brains_debug_repl()
            case _:
                raise ValueError("Invalid topic: " + topic)
