"""Comprehensive tests for the agents module."""

from unittest.mock import MagicMock

import pytest
from agno.tools.exa import ExaTools
from agno.tools.reasoning import ReasoningTools
from src.mcp_server_mas_sequential_thinking.unified_agents import (
    AgentCapability,
    create_agent,
    create_all_agents,
)


class TestAgentCapability:
    """Test AgentCapability dataclass functionality."""

    def test_capability_creation(self):
        """Test creating an agent capability."""
        capability = AgentCapability(
            role="Test Role",
            description="Test description",
            tools=[ReasoningTools],
            role_description="Test role description",
        )

        assert capability.role == "Test Role"
        assert capability.description == "Test description"
        assert capability.tools == [ReasoningTools]
        assert capability.role_description == "Test role description"

    def test_get_instructions(self):
        """Test instruction generation."""
        capability = AgentCapability(
            role="Test Role",
            description="Test description",
            tools=[ReasoningTools],
            role_description="Test role description",
        )

        instructions = capability.get_instructions()

        assert len(instructions) == 4
        assert "specialist agent" in instructions[0]
        assert "Test role description" in instructions[1]
        assert "Process:" in instructions[2]
        assert "accuracy and relevance" in instructions[3]

    def test_create_tools(self):
        """Test tool instantiation."""
        capability = AgentCapability(
            role="Test Role",
            description="Test description",
            tools=[ReasoningTools, ExaTools],
            role_description="Test role description",
        )

        tools = capability.create_tools()

        assert len(tools) == 2
        assert isinstance(tools[0], ReasoningTools)
        assert isinstance(tools[1], ExaTools)

    def test_capability_immutability(self):
        """Test that AgentCapability is immutable."""
        capability = AgentCapability(
            role="Test Role",
            description="Test description",
            tools=[ReasoningTools],
            role_description="Test role description",
        )

        with pytest.raises(
            Exception
        ):  # dataclass(frozen=True) should prevent modification
            capability.role = "Modified Role"


class TestAgentFactory:
    """Test AgentFactory functionality."""

    def test_capabilities_registry(self):
        """Test that all expected capabilities are registered."""
        expected_capabilities = {
            "planner",
            "researcher",
            "analyzer",
            "critic",
            "synthesizer",
        }
        assert set(AgentFactory.CAPABILITIES.keys()) == expected_capabilities

    @pytest.mark.parametrize(
        "agent_type", ["planner", "researcher", "analyzer", "critic", "synthesizer"]
    )
    def test_capability_configurations(self, agent_type):
        """Test that all capabilities have required configurations."""
        capability = AgentFactory.CAPABILITIES[agent_type]

        assert capability.role is not None
        assert capability.description is not None
        assert capability.tools is not None
        assert capability.role_description is not None
        assert len(capability.tools) > 0

    def test_researcher_has_research_tools(self):
        """Test that researcher has both thinking and research tools."""
        capability = AgentFactory.CAPABILITIES["researcher"]
        assert ReasoningTools in capability.tools
        assert ExaTools in capability.tools

    def test_other_agents_have_thinking_tools(self):
        """Test that non-researcher agents have thinking tools."""
        for agent_type in ["planner", "analyzer", "critic", "synthesizer"]:
            capability = AgentFactory.CAPABILITIES[agent_type]
            assert ReasoningTools in capability.tools

    def test_create_agent_valid_type(self):
        """Test creating agents with valid types."""
        mock_model = MagicMock()

        for agent_type in AgentFactory.CAPABILITIES:
            agent = AgentFactory.create_agent(agent_type, mock_model)

            assert agent.name == agent_type.title()
            assert agent.model == mock_model
            assert agent.role == AgentFactory.CAPABILITIES[agent_type].role

    def test_create_agent_invalid_type(self):
        """Test creating agent with invalid type."""
        mock_model = MagicMock()

        with pytest.raises(ValueError, match="Unknown agent type: invalid"):
            AgentFactory.create_agent("invalid", mock_model)

    def test_create_agent_with_extra_instructions(self):
        """Test creating agent with additional instructions."""
        mock_model = MagicMock()
        extra_instructions = ["Extra instruction 1", "Extra instruction 2"]

        agent = AgentFactory.create_agent(
            "planner", mock_model, extra_instructions=extra_instructions
        )

        # Instructions should include both base and extra
        base_instructions = AgentFactory.CAPABILITIES["planner"].get_instructions()
        expected_length = len(base_instructions) + len(extra_instructions)
        assert len(agent.instructions) == expected_length

    def test_create_agent_with_kwargs(self):
        """Test creating agent with additional kwargs."""
        mock_model = MagicMock()

        agent = AgentFactory.create_agent(
            "planner", mock_model, show_model=True, debug=True
        )

        # Agent should be created successfully with additional parameters
        assert agent.name == "Planner"
        assert agent.model == mock_model

    def test_create_all_agents(self):
        """Test creating all specialist agents."""
        mock_model = MagicMock()

        agents = AgentFactory.create_all_agents(mock_model)

        assert len(agents) == len(AgentFactory.CAPABILITIES)

        for agent_type in AgentFactory.CAPABILITIES:
            assert agent_type in agents
            assert agents[agent_type].name == agent_type.title()
            assert agents[agent_type].model == mock_model

    def test_agent_tool_instantiation(self):
        """Test that agent tools are properly instantiated."""
        mock_model = MagicMock()

        # Test researcher which has multiple tools
        researcher = AgentFactory.create_agent("researcher", mock_model)

        assert len(researcher.tools) == 2
        tool_types = [type(tool) for tool in researcher.tools]
        assert ReasoningTools in tool_types
        assert ExaTools in tool_types

    def test_agent_configuration_consistency(self):
        """Test that agent configuration is consistent across factory methods."""
        mock_model = MagicMock()

        # Create agent via factory
        factory_agent = AgentFactory.create_agent("planner", mock_model)

        # Create via convenience function
        convenience_agent = create_agent("planner", mock_model)

        # Should have same configuration
        assert factory_agent.name == convenience_agent.name
        assert factory_agent.role == convenience_agent.role
        assert factory_agent.description == convenience_agent.description
        assert len(factory_agent.tools) == len(convenience_agent.tools)


class TestConvenienceFunctions:
    """Test backward compatibility convenience functions."""

    def test_create_agent_function(self):
        """Test create_agent convenience function."""
        mock_model = MagicMock()

        agent = create_agent("planner", mock_model)

        assert agent.name == "Planner"
        assert agent.model == mock_model
        assert agent.role == "Strategic Planner"

    def test_create_all_agents_function(self):
        """Test create_all_agents convenience function."""
        mock_model = MagicMock()

        agents = create_all_agents(mock_model)

        assert len(agents) == 5  # All five agent types
        assert "planner" in agents
        assert "researcher" in agents
        assert "analyzer" in agents
        assert "critic" in agents
        assert "synthesizer" in agents

    def test_convenience_functions_match_factory(self):
        """Test that convenience functions match factory behavior."""
        mock_model = MagicMock()

        # Compare single agent creation
        factory_agent = AgentFactory.create_agent("analyzer", mock_model)
        convenience_agent = create_agent("analyzer", mock_model)

        assert factory_agent.name == convenience_agent.name
        assert factory_agent.role == convenience_agent.role

        # Compare all agents creation
        factory_agents = AgentFactory.create_all_agents(mock_model)
        convenience_agents = create_all_agents(mock_model)

        assert len(factory_agents) == len(convenience_agents)
        assert set(factory_agents.keys()) == set(convenience_agents.keys())


class TestAgentSpecializations:
    """Test specific agent type configurations."""

    def test_planner_specialization(self):
        """Test planner agent configuration."""
        capability = AgentFactory.CAPABILITIES["planner"]

        assert "Strategic Planner" in capability.role
        assert "strategic plans" in capability.description
        assert "roadmaps" in capability.role_description
        assert ReasoningTools in capability.tools

    def test_researcher_specialization(self):
        """Test researcher agent configuration."""
        capability = AgentFactory.CAPABILITIES["researcher"]

        assert "Information Gatherer" in capability.role
        assert "Gathers and validates" in capability.description
        assert "research tools" in capability.role_description
        assert ReasoningTools in capability.tools
        assert ExaTools in capability.tools

    def test_analyzer_specialization(self):
        """Test analyzer agent configuration."""
        capability = AgentFactory.CAPABILITIES["analyzer"]

        assert "Core Analyst" in capability.role
        assert "analysis" in capability.description
        assert "patterns" in capability.role_description
        assert ReasoningTools in capability.tools

    def test_critic_specialization(self):
        """Test critic agent configuration."""
        capability = AgentFactory.CAPABILITIES["critic"]

        assert "Quality Controller" in capability.role
        assert "evaluates" in capability.description
        assert "assumptions" in capability.role_description
        assert ReasoningTools in capability.tools

    def test_synthesizer_specialization(self):
        """Test synthesizer agent configuration."""
        capability = AgentFactory.CAPABILITIES["synthesizer"]

        assert "Integration Specialist" in capability.role
        assert "Integrates" in capability.description
        assert "synthesize" in capability.role_description
        assert ReasoningTools in capability.tools

    def test_agent_instruction_consistency(self):
        """Test that all agents have consistent instruction structure."""
        for capability in AgentFactory.CAPABILITIES.values():
            instructions = capability.get_instructions()

            # All should have 4 core instructions
            assert len(instructions) == 4

            # First instruction should mention specialist
            assert "specialist agent" in instructions[0]

            # Second should have role description
            assert capability.role_description in instructions[1]

            # Third should mention process
            assert "Process:" in instructions[2]

            # Fourth should mention focus
            assert "Focus on" in instructions[3]
