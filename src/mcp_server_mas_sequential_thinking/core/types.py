"""Type definitions for better type safety."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypedDict

from agno.agent import Agent
from agno.models.base import Model
from agno.team.team import Team

# Type aliases for better semantic meaning
ThoughtNumber = int
BranchId = str
ProviderName = str
TeamType = str
AgentType = str
ConfigDict = dict[str, Any]
InstructionsList = list[str]
SuccessCriteriaList = list[str]


class ExecutionMode(Enum):
    """Execution modes for different routing strategies."""

    SINGLE_AGENT = "single_agent"
    SELECTIVE_TEAM = "selective_team"  # Hybrid with specific specialists
    FULL_TEAM = "full_team"  # Complete multi-agent team


@dataclass
class CoordinationPlan:
    """Comprehensive plan combining routing and coordination decisions."""

    # Routing decisions - using string values for simplicity
    strategy: str  # ProcessingStrategy value
    complexity_level: str  # ComplexityLevel value
    complexity_score: float

    # Coordination decisions
    execution_mode: ExecutionMode
    specialist_roles: list[str]
    task_breakdown: list[str]
    coordination_strategy: str

    # Execution parameters
    timeout_seconds: float
    expected_interactions: int
    team_size: int

    # Reasoning and metadata
    reasoning: str
    confidence: float
    original_thought: str

    @classmethod
    def from_routing_decision(
        cls, routing_decision, thought_data
    ) -> "CoordinationPlan":
        """Create coordination plan from adaptive routing decision."""
        # Map ProcessingStrategy to ExecutionMode
        strategy_to_mode = {
            "single_agent": ExecutionMode.SINGLE_AGENT,
            "hybrid": ExecutionMode.SELECTIVE_TEAM,
            "multi_agent": ExecutionMode.FULL_TEAM,
        }

        # Map complexity to specialist roles
        complexity_to_specialists = {
            "simple": ["general"],
            "moderate": ["planner", "analyzer"],
            "complex": ["planner", "researcher", "analyzer", "critic"],
            "highly_complex": [
                "planner",
                "researcher",
                "analyzer",
                "critic",
                "synthesizer",
            ],
        }

        execution_mode = strategy_to_mode.get(
            routing_decision.strategy.value, ExecutionMode.SINGLE_AGENT
        )
        specialists = complexity_to_specialists.get(
            routing_decision.complexity_level.value, ["general"]
        )

        return cls(
            strategy=routing_decision.strategy.value,
            complexity_level=routing_decision.complexity_level.value,
            complexity_score=routing_decision.complexity_score,
            execution_mode=execution_mode,
            specialist_roles=specialists,
            team_size=len(specialists),
            coordination_strategy="adaptive_routing_based",
            task_breakdown=[
                f"Process {thought_data.thought_type.value} thought",
                "Generate guidance",
            ],
            expected_interactions=len(specialists),
            timeout_seconds=300.0,  # Default timeout
            reasoning=routing_decision.reasoning,
            confidence=0.8,  # Default confidence for rule-based routing
            original_thought=thought_data.thought,
        )


class ProcessingMetadata(TypedDict, total=False):
    """Type-safe processing metadata structure."""

    strategy: str
    complexity_score: float
    estimated_cost: float
    actual_cost: float
    token_usage: int
    processing_time: float
    specialists: list[str]
    provider: str
    routing_reasoning: str
    error_count: int
    retry_count: int


class SessionStats(TypedDict, total=False):
    """Type-safe session statistics structure."""

    total_thoughts: int
    total_cost: float
    total_tokens: int
    average_processing_time: float
    error_rate: float
    successful_thoughts: int
    failed_thoughts: int


class ComplexityMetrics(TypedDict):
    """Type-safe complexity analysis metrics."""

    word_count: int
    sentence_count: int
    question_count: int
    technical_terms: int
    has_branching: bool
    has_research_keywords: bool
    has_analysis_keywords: bool
    overall_score: float


class ModelProvider(Protocol):
    """Protocol for model provider implementations."""

    id: str
    cost_per_token: float


class AgentFactory(Protocol):
    """Protocol for agent factory implementations."""

    def create_team_agents(self, model: Model, team_type: str) -> dict[str, Agent]:
        """Create team agents with specified model and team type."""
        ...


class TeamBuilder(Protocol):
    """Protocol for team builder implementations."""

    def build_team(self, config: Any, agent_factory: Any) -> Team:
        """Build a team with specified configuration and agent factory."""
        ...


class CostEstimator(Protocol):
    """Protocol for cost estimation with type safety."""

    def estimate_cost(
        self, strategy: str, complexity_score: float, provider: str
    ) -> tuple[tuple[int, int], float]:
        """Estimate cost for processing strategy."""
        ...


class ComplexityAnalyzer(Protocol):
    """Protocol for complexity analysis with type safety."""

    def analyze(self, thought_text: str) -> ComplexityMetrics:
        """Analyze thought complexity and return metrics."""
        ...


class ThoughtProcessor(Protocol):
    """Protocol for thought processing with type safety."""

    async def process_thought(self, thought_data: Any) -> str:
        """Process a thought and return the result."""
        ...


class SessionManager(Protocol):
    """Protocol for session management with type safety."""

    def add_thought(self, thought_data: Any) -> None:
        """Add a thought to the session."""
        ...

    def find_thought_content(self, thought_number: int) -> str:
        """Find thought content by number."""
        ...

    def get_branch_summary(self) -> dict[str, int]:
        """Get summary of all branches."""
        ...


class ConfigurationProvider(Protocol):
    """Protocol for configuration management with type safety."""

    def get_model_config(self, provider_name: str | None = None) -> Any:
        """Get model configuration."""
        ...

    def check_required_api_keys(self, provider_name: str | None = None) -> list[str]:
        """Check for required API keys."""
        ...


# Custom Exception Classes
class ValidationError(ValueError):
    """Exception raised when data validation fails."""



class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""



class ThoughtProcessingError(Exception):
    """Exception raised when thought processing fails."""

    def __init__(self, message: str, metadata: ProcessingMetadata | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class TeamCreationError(Exception):
    """Exception raised when team creation fails."""



class RoutingDecisionError(ThoughtProcessingError):
    """Error in adaptive routing decision making."""



class CostOptimizationError(ThoughtProcessingError):
    """Error in cost optimization logic."""



class PersistentStorageError(ThoughtProcessingError):
    """Error in persistent memory storage."""



class ModelConfigurationError(ConfigurationError):
    """Error in model configuration."""



class ProviderError(Exception):
    """Error related to LLM providers."""



class AgentCreationError(Exception):
    """Error in agent creation."""

