from serena.tools import Tool, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional


class OpenDashboardTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """
    Opens the Serena web dashboard in the default web browser.
    The dashboard provides logs, session information, and tool usage statistics.
    """

    def apply(self) -> str:
        """
        Opens the Serena web dashboard in the default web browser.
        """
        if self.agent.open_dashboard():
            return f"Serena web dashboard has been opened in the user's default web browser: {self.agent.get_dashboard_url()}"
        else:
            return f"Serena web dashboard could not be opened automatically; tell the user to open it via {self.agent.get_dashboard_url()}"


class ActivateProjectTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Activates a project based on the project name or path.
    """

    # noinspection PyIncorrectDocstring
    # (session_id is injected via apply_ex)
    def apply(self, project: str, session_id: str) -> str:
        """
        Activates the project with the given name or path.

        :param project: the name of a registered project to activate or a path to a project directory
        """
        is_new_activation = self.agent.activate_project_from_path_or_name(project)
        receipt = self._render_activation_receipt(is_new_activation)
        result = self.agent.get_project_activation_message(session_id)
        result += "\nIMPORTANT: If you have not yet read the 'Serena Instructions Manual', do it now before continuing!"
        if receipt:
            return f"{receipt}\n{result}"
        return result

    def _render_activation_receipt(self, is_new_activation: bool) -> str:
        """
        Builds a concise activation/switch receipt that the agent sees before the standard
        activation message. Distinguishes first activation, project switch, and same-project
        re-activation so warnings are only shown when relative paths or symbol references from
        a previous project may genuinely be stale.
        """
        info = self.agent.get_last_activation_info()
        if info is None:
            return ""

        activation_id = info["activation_id"]
        to_name = info["to"]
        from_name = info["from"]

        if not is_new_activation:
            return f"Project re-activated: {to_name} (activation_id={activation_id})"
        if from_name is None:
            return f"Project activated: {to_name} (activation_id={activation_id})"
        return (
            f"Project switched: {from_name} -> {to_name} (activation_id={activation_id})\n"
            f"WARNING: cached relative paths or symbol references from '{from_name}' "
            "are no longer valid; re-read files before editing."
        )


class RemoveProjectTool(Tool, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional):
    """
    Removes a project from the Serena configuration.
    """

    def apply(self, project_name: str) -> str:
        """
        Removes a project from the Serena configuration.

        :param project_name: Name of the project to remove
        """
        self.agent.serena_config.remove_project(project_name)
        return f"Successfully removed project '{project_name}' from configuration."


class GetCurrentConfigTool(Tool):
    """
    Prints the current configuration of the agent, including the active and available projects, tools, contexts, and modes.
    """

    def apply(self) -> str:
        """
        Print the current configuration of the agent, including the active and available projects, tools, contexts, and modes.
        """
        return self.agent.get_current_config_overview()
