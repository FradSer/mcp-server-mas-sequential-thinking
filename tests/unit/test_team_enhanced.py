"""Comprehensive tests for the team module."""

from unittest.mock import MagicMock, patch

import pytest
from agno.team.team import Team

from mcp_server_mas_sequential_thinking.unified_team import (
    create_team,
)
from tests.helpers.mocks import MockModelConfig


class TestCoordinatorInstructions:
    """Test coordinator instruction configuration."""

    def test_coordinator_instructions_exist(self):
        """Test that coordinator instructions are defined."""
        assert isinstance(_STANDARD_INSTRUCTIONS, list)
        assert len(_STANDARD_INSTRUCTIONS) > 0
        assert isinstance(_ENHANCED_INSTRUCTIONS, list)
        assert len(_ENHANCED_INSTRUCTIONS) > 0

    def test_coordinator_instructions_content(self):
        """Test coordinator instruction content."""
        standard_text = " ".join(_STANDARD_INSTRUCTIONS)
        enhanced_text = " ".join(_ENHANCED_INSTRUCTIONS)

        # Check for key coordination concepts
        assert "coordinate" in standard_text.lower()
        assert "specialists" in standard_text.lower()
        assert "coordinate" in enhanced_text.lower()
        assert "specialists" in enhanced_text.lower()
        assert "Planner" in standard_text
        assert "Researcher" in standard_text
        assert "Analyzer" in standard_text
        assert "Critic" in standard_text
        assert "Synthesizer" in standard_text

    def test_coordinator_process_flow(self):
        """Test that coordinator instructions include process flow."""
        process_instruction = None
        for instruction in _STANDARD_INSTRUCTIONS:
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
        for instruction in _STANDARD_INSTRUCTIONS:
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
        for instruction in _STANDARD_INSTRUCTIONS:
            if "efficiency" in instruction.lower():
                efficiency_instruction = instruction
                break

        assert efficiency_instruction is not None
        assert "strictly necessary" in efficiency_instruction


class TestTeamCreation:
    """Test team creation functionality."""

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_create_team_basic(self, mock_agent_factory, mock_get_config):
        """Test basic team creation."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="test-team-model",
            agent_model_id="test-agent-model",
        )
        mock_get_config.return_value = mock_config

        # Mock factory instance and its methods
        mock_factory_instance = MagicMock()
        mock_agents = {
            "planner": MagicMock(),
            "researcher": MagicMock(),
            "analyzer": MagicMock(),
            "critic": MagicMock(),
            "synthesizer": MagicMock(),
        }
        mock_factory_instance.create_team_agents.return_value = mock_agents
        mock_agent_factory.return_value = mock_factory_instance

        # Create team
        team = create_team()

        # Verify team configuration (v2 compatible)
        assert isinstance(team, Team)
        assert team.name == "SequentialThinkingTeam"
        # v2: mode is removed, check new attributes instead
        assert team.respond_directly is False
        assert team.delegate_task_to_all_members is False
        assert team.determine_input_for_members is True
        assert len(team.members) == 5
        assert team.description == "Coordinator for sequential thinking specialist team"
        assert team.instructions == _STANDARD_INSTRUCTIONS

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_team_model_configuration(self, mock_agent_factory, mock_get_config):
        """Test team model configuration."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="custom-team-model",
            agent_model_id="custom-agent-model",
        )
        mock_get_config.return_value = mock_config
        mock_agent_factory.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify model configuration
        mock_model_class.assert_called_with(id="custom-team-model")
        assert team.model is not None

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_agent_model_configuration(self, mock_agent_factory, mock_get_config):
        """Test agent model configuration."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_config = MockModelConfig(
            provider_class=mock_model_class,
            team_model_id="team-model",
            agent_model_id="agent-model",
        )
        mock_get_config.return_value = mock_config
        mock_agent_factory.return_value = {"planner": MagicMock()}

        # Create team
        create_team()

        # Verify agent model configuration
        assert mock_model_class.call_count == 2  # Called for both team and agent models
        mock_model_class.assert_any_call(id="team-model")
        mock_model_class.assert_any_call(id="agent-model")
        mock_agent_factory.assert_called_once()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_team_success_criteria(self, mock_agent_factory, mock_get_config):
        """Test team success criteria configuration."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_agent_factory.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify success criteria
        assert len(team.success_criteria) == 3
        criteria_text = " ".join(team.success_criteria)
        assert "delegate" in criteria_text.lower()
        assert "synthesize" in criteria_text.lower()
        assert "recommend" in criteria_text.lower()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_team_configuration_flags(self, mock_agent_factory, mock_get_config):
        """Test team configuration flags."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_agent_factory.return_value = {"planner": MagicMock()}

        # Create team
        team = create_team()

        # Verify configuration flags
        assert team.enable_agentic_context is False
        assert team.share_member_interactions is False
        assert team.markdown is True
        # v2: add_datetime_to_instructions removed - teams handle datetime internally

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    @patch("src.mcp_server_mas_sequential_thinking.team.logger")
    def test_team_creation_logging(
        self, mock_logger, mock_agent_factory, mock_get_config
    ):
        """Test team creation logging."""
        # Setup mocks
        mock_model_class = MagicMock()
        mock_model_class.__name__ = "TestModel"
        mock_config = MockModelConfig(provider_class=mock_model_class)
        mock_get_config.return_value = mock_config
        mock_agent_factory.return_value = {"planner": MagicMock()}

        # Create team
        create_team()

        # Verify logging
        mock_logger.info.assert_called_once()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "TestModel" in log_call_args
        assert "provider" in log_call_args

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_team_member_assignment(self, mock_agent_factory, mock_get_config):
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
        mock_agent_factory.return_value = mock_agents

        # Create team
        team = create_team()

        # Verify all agents are team members
        assert len(team.members) == 5
        member_names = [member.name for member in team.members]
        expected_names = ["Planner", "Researcher", "Analyzer", "Critic", "Synthesizer"]
        assert all(name in member_names for name in expected_names)

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_config_error_handling(self, mock_agent_factory, mock_get_config):
        """Test handling of configuration errors."""
        # Setup mock to raise exception
        mock_get_config.side_effect = Exception("Config error")

        # Should propagate the exception
        with pytest.raises(Exception, match="Config error"):
            create_team()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_agent_creation_error_handling(self, mock_agent_factory, mock_get_config):
        """Test handling of agent creation errors."""
        # Setup mocks
        mock_config = MockModelConfig()
        mock_get_config.return_value = mock_config
        mock_agent_factory.side_effect = Exception("Agent creation error")

        # Should propagate the exception
        with pytest.raises(Exception, match="Agent creation error"):
            create_team()

    @patch("src.mcp_server_mas_sequential_thinking.team.get_model_config")
    @patch("src.mcp_server_mas_sequential_thinking.team.UnifiedAgentFactory")
    def test_different_provider_configurations(
        self, mock_agent_factory, mock_get_config
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
            mock_agent_factory.return_value = {"planner": MagicMock()}

            # Create team
            team = create_team()

            # Verify configuration
            assert isinstance(team, Team)
            assert team.name == "SequentialThinkingTeam"
