"""Unit tests for ProcessingOrchestrator TeamMode resolution."""

from unittest.mock import MagicMock

from agno.team.mode import TeamMode

from mcp_server_mas_sequential_thinking.config.constants import TeamModeMapping
from mcp_server_mas_sequential_thinking.services.processing_orchestrator import (
    ProcessingOrchestrator,
)


def _make_orchestrator() -> ProcessingOrchestrator:
    """Create a ProcessingOrchestrator with mock dependencies."""
    session = MagicMock()
    response_processor = MagicMock()
    retry_handler = MagicMock()
    return ProcessingOrchestrator(session, response_processor, retry_handler)


class TestResolveTeamMode:
    """Test _resolve_team_mode strategy mapping."""

    def test_full_exploration_maps_to_broadcast(self):
        orch = _make_orchestrator()
        assert orch._resolve_team_mode("full_exploration") == TeamMode.broadcast

    def test_single_direction_maps_to_route(self):
        orch = _make_orchestrator()
        assert orch._resolve_team_mode("single_direction") == TeamMode.route

    def test_route_maps_to_route(self):
        orch = _make_orchestrator()
        assert orch._resolve_team_mode("route") == TeamMode.route

    def test_default_maps_to_coordinate(self):
        orch = _make_orchestrator()
        assert orch._resolve_team_mode(None) == TeamMode.coordinate
        assert orch._resolve_team_mode("unknown") == TeamMode.coordinate

    def test_team_mode_mapping_constants(self):
        assert TeamModeMapping.FULL_EXPLORATION == "broadcast"
        assert TeamModeMapping.SINGLE_DIRECTION == "route"
        assert TeamModeMapping.COORDINATED_SEQUENCE == "coordinate"


class TestGetTeam:
    """Test _get_team caching behavior."""

    def test_get_team_returns_cached_team(self):
        orch = _make_orchestrator()
        cached_team = MagicMock()
        orch._session._team = cached_team
        assert orch._get_team() == cached_team
