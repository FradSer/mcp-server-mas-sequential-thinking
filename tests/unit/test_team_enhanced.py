"""Comprehensive tests for the team module."""

import pytest
from unittest.mock import patch, MagicMock

from src.mcp_server_mas_sequential_thinking.team import create_team, COORDINATOR_INSTRUCTIONS
from agno.team.team import Team
from tests.helpers.mocks import MockModelConfig


class TestCoordinatorInstructions:
    """Test coordinator instruction configuration."""

    def test_coordinator_instructions_exist(self):
        """Test that coordinator instructions are defined."""
        assert isinstance(COORDINATOR_INSTRUCTIONS, list)
        assert len(COORDINATOR_INSTRUCTIONS) > 0

    def test_coordinator_instructions_content(self):
        """Test coordinator instruction content."""
        instructions_text = " ".join(COORDINATOR_INSTRUCTIONS)

        # Check for key coordination concepts
        assert "coordinate" in instructions_text.lower()
        assert "specialists" in instructions_text.lower()
        assert "planner" in instructions_text
        assert "researcher" in instructions_text
        assert "analyzer" in instructions_text
        assert "critic" in instructions_text
        assert "synthesizer" in instructions_text

    def test_coordinator_process_flow(self):
        """Test that coordinator instructions include process flow."""
        process_instruction = None
        for instruction in COORDINATOR_INSTRUCTIONS:
            if "Process:" in instruction:
                process_instruction = instruction
                break

        assert process_instruction is not None
        assert "Analyze input" in process_instruction
        assert "Identify required specialists" in process_instruction
        assert "Delegate" in process_instruction
        assert "Synthesize" in process_instruction
        assert "guidance" in process_instruction

    def test_recommendation_format(self):
        """Test that coordinator instructions include recommendation format."""
        recommendation_instruction = None
        for instruction in COORDINATOR_INSTRUCTIONS:
            if "RECOMMENDATION" in instruction:
                recommendation_instruction = instruction
                break

        assert recommendation_instruction is not None
        assert "RECOMMENDATION:" in recommendation_instruction
        assert "SUGGESTION:" in recommendation_instruction
        assert "Revise thought" in recommendation_instruction
        assert "branching" in recommendation_instruction

    def test_efficiency_emphasis(self):
        """Test that coordinator instructions emphasize efficiency."""
        efficiency_instruction = None
        for instruction in COORDINATOR_INSTRUCTIONS:
            if "efficiency" in instruction.lower():
                efficiency_instruction = instruction
                break

        assert efficiency_instruction is not None
        assert "strictly necessary" in efficiency_instruction


class TestTeamCreation:
    """Test team creation functionality."""

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_create_team_basic(self, mock_create_agents, mock_get_config):
        """Test basic team creation."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="test-team-model",
            agent_model_id="test-agent-model",
        )
        mock_get_config.return_value = mock_config

        mock_agents = {
            "planner": MagicMock(),
            "researcher": MagicMock(),
            "analyzer": MagicMock(),
            "critic": MagicMock(),
            "synthesizer": MagicMock(),
        }
        mock_create_agents.return_value = mock_agents

        # Create team
        team = create_team()

        # Verify team configuration
        assert isinstance(team, Team)
        assert team.name == "SequentialThinkingTeam"
        assert team.mode == "coordinate"
        assert len(team.members) == 5
        assert team.description == "Coordinator for sequential thinking specialist team"
        assert team.instructions == COORDINATOR_INSTRUCTIONS

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_team_model_configuration(self, mock_create_agents, mock_get_config):
        """Test team model configuration."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="custom-team-model",
            agent_model_id="custom-agent-model",
        )
        mock_get_config.return_value = mock_config
        mock_create_agents.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify model configuration
        mock_model_class.assert_called_with(id="custom-team-model")
        assert team.model is not None

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_agent_model_configuration(self, mock_create_agents, mock_get_config):
        """Test agent model configuration."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="team-model",
            agent_model_id="agent-model",
        )
        mock_get_config.return_value = mock_config
        mock_create_agents.return_value = {"planner": MagicMock()}

        # Create team
        create_team()

        # Verify agent model configuration
        assert mock_model_class.call_count == 2  # Called for both team and agent models
        mock_model_class.assert_any_call(id="team-model")
        mock_model_class.assert_any_call(id="agent-model")
        mock_create_agents.assert_called_once()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_team_success_criteria(self, mock_create_agents, mock_get_config):
        """Test team success criteria configuration."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_create_agents.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify success criteria
        assert len(team.success_criteria) == 3
        criteria_text = " ".join(team.success_criteria)
        assert "delegate" in criteria_text.lower()
        assert "synthesize" in criteria_text.lower()
        assert "recommend" in criteria_text.lower()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_team_configuration_flags(self, mock_create_agents, mock_get_config):
        """Test team configuration flags."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_create_agents.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify configuration flags
        assert team.enable_agentic_context is False
        assert team.share_member_interactions is False
        assert team.markdown is True
        assert team.add_datetime_to_instructions is True

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    @patch("src.mcp_server_mas_sequential_thinking.team.logger")
    def test_team_creation_logging(
        self, mock_logger, mock_create_agents, mock_get_config
    ):
        """Test team creation logging."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "TestModel"
        mock_config = MockModelConfig(provider_class=mock_model_class)
        mock_get_config.return_value = mock_config
        mock_create_agents.return_value = {"planner": MagicMock()}

        # Create team
        create_team()

        # Verify logging
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "TestModel" in log_call_args
        assert "provider" in log_call_args

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_team_member_assignment(self, mock_create_agents, mock_get_config):
        """Test that all agents are assigned as team members."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config

        mock_agents = {
            "planner": MagicMock(name="Planner"),
            "researcher": MagicMock(name="Researcher"),
            "analyzer": MagicMock(name="Analyzer"),
            "critic": MagicMock(name="Critic"),
            "synthesizer": MagicMock(name="Synthesizer"),
        }
        mock_create_agents.return_value = mock_agents

        # Create team
        team = create_team()

        # Verify all agents are team members
        assert len(team.members) == 5
        member_names = [member.name for member in team.members]
        expected_names = ["Planner", "Researcher", "Analyzer", "Critic", "Synthesizer"]
        assert all(name in member_names for name in expected_names)

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_config_error_handling(self, mock_create_agents, mock_get_config):
        """Test handling of configuration errors."""
        # Setup mock to raise exception
        mock_get_config.side_effect = Exception("Config error")

        # Should propagate the exception
        with pytest.raises(Exception, match="Config error"):
            create_team()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_agent_creation_error_handling(self, mock_create_agents, mock_get_config):
        """Test handling of agent creation errors."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_create_agents.side_effect = Exception("Agent creation error")

        # Should propagate the exception
        with pytest.raises(Exception, match="Agent creation error"):
            create_team()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.create_all_agents")
    def test_different_provider_configurations(
        self, mock_create_agents, mock_get_config
    ):
        """Test team creation with different provider configurations."""
        providers = [
            ("DeepSeek", "deepseek-chat", "deepseek-chat"),
            ("Groq", "groq-team-model", "groq-agent-model"),
            ("OpenRouter", "openrouter-team", "openrouter-agent"),
        ]

        for provider_name, team_model, agent_model in providers:
            # Setup mock for this provider
            mock_model_class = MagicMock()
            mock_model_class.__name__ = provider_name
            mock_config = MockModelConfig(
                provider_class=mock_model_class,
                team_model_id=team_model,
                agent_model_id=agent_model,
            )
            mock_get_config.return_value = mock_config
            mock_create_agents.return_value = {"planner": MagicMock()}

            # Create team
            team = create_team()

            # Verify configuration
            assert isinstance(team, Team)
            assert team.name == "SequentialThinkingTeam"
