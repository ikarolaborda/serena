import json
from collections.abc import Callable
from types import SimpleNamespace

from serena.analytics import ProjectSwitchEvent, ProjectSwitchStats
from serena.dashboard import SerenaDashboardAPI
from solidlsp.ls_config import Language


class _DummyMemoryLogHandler:
    def get_log_messages(self, from_idx: int = 0):  # pragma: no cover - simple stub
        return SimpleNamespace(messages=[], max_idx=-1)

    def clear_log_messages(self) -> None:  # pragma: no cover - simple stub
        pass


class _DummyAgent:
    def __init__(self, project: SimpleNamespace | None, switch_stats: ProjectSwitchStats | None = None) -> None:
        self._project = project
        self._switch_stats = switch_stats or ProjectSwitchStats()

    def register_config_changed_callback(self, callback: Callable[[], None]) -> None:
        pass

    def execute_task(self, func, *, logged: bool | None = None, name: str | None = None):
        del logged, name
        return func()

    def get_active_project(self):
        return self._project

    def get_project_switch_stats(self) -> ProjectSwitchStats:
        return self._switch_stats


def _make_dashboard(project_languages: list[Language] | None) -> SerenaDashboardAPI:
    project = None
    if project_languages is not None:
        project = SimpleNamespace(project_config=SimpleNamespace(languages=project_languages))
    agent = _DummyAgent(project)
    return SerenaDashboardAPI(memory_log_handler=_DummyMemoryLogHandler(), tool_names=[], agent=agent, tool_usage_stats=None)


def test_available_languages_include_experimental_when_no_active_project():
    dashboard = _make_dashboard(project_languages=None)
    response = dashboard._get_available_languages()
    expected = sorted(lang.value for lang in Language.iter_all(include_experimental=True))
    assert response.languages == expected


def test_available_languages_exclude_project_languages():
    dashboard = _make_dashboard(project_languages=[Language.PYTHON, Language.MARKDOWN])
    response = dashboard._get_available_languages()
    available = set(response.languages)
    assert Language.PYTHON.value not in available
    assert Language.MARKDOWN.value not in available
    # ensure experimental languages remain available for selection
    assert Language.ANSIBLE.value in available


def test_get_project_switches_endpoint_returns_json_serializable_summary():
    switch_stats = ProjectSwitchStats()
    switch_stats.record(ProjectSwitchEvent(activation_id=1, from_project=None, to_project="alpha", shutdown_ms=0.0, ls_init_started=True))
    switch_stats.record(
        ProjectSwitchEvent(activation_id=2, from_project="alpha", to_project="beta", shutdown_ms=12.5, ls_init_started=True)
    )

    agent = _DummyAgent(project=None, switch_stats=switch_stats)
    dashboard = SerenaDashboardAPI(memory_log_handler=_DummyMemoryLogHandler(), tool_names=[], agent=agent, tool_usage_stats=None)

    client = dashboard._app.test_client()
    response = client.get("/get_project_switches")

    assert response.status_code == 200
    payload = json.loads(response.data)
    assert payload["total_recorded"] == 2
    assert payload["switch_count"] == 1
    assert payload["last"]["activation_id"] == 2
    assert payload["last"]["from_project"] == "alpha"
    assert payload["last"]["to_project"] == "beta"
