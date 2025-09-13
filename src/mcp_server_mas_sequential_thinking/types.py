"""Type definitions for better type safety."""

from typing import Dict, List, Optional, Protocol, TypedDict, Any
from agno.models.base import Model
from agno.agent import Agent
from agno.team.team import Team


class ProcessingMetadata(TypedDict, total=False):
    """Type-safe processing metadata structure."""

    strategy: str
    complexity_score: float
    estimated_cost: float
    actual_cost: float
    token_usage: int
    processing_time: float
    specialists: List[str]
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
    """Protocol for model providers with type safety."""

    def create_team_model(self) -> Model:
        """Create a model for team coordination."""
        ...

    def create_agent_model(self) -> Model:
        """Create a model for individual agents."""
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


class TeamFactory(Protocol):
    """Protocol for team creation with type safety."""

    def create_team(self, team_type: str, config: Optional[Any] = None) -> Team:
        """Create a team of the specified type."""
        ...

    def get_available_team_types(self) -> List[str]:
        """Get list of available team types."""
        ...


class AgentFactory(Protocol):
    """Protocol for agent creation with type safety."""

    def create_agent(
        self, agent_type: str, model: Model, enhanced_mode: bool = False, **kwargs: Any
    ) -> Agent:
        """Create an agent of the specified type."""
        ...

    def create_team_agents(
        self, model: Model, team_type: str = "standard"
    ) -> Dict[str, Agent]:
        """Create a complete set of team agents."""
        ...


class SessionManager(Protocol):
    """Protocol for session management with type safety."""

    def add_thought(self, thought_data: Any) -> None:
        """Add a thought to the session."""
        ...

    def find_thought_content(self, thought_number: int) -> str:
        """Find thought content by number."""
        ...

    def get_branch_summary(self) -> Dict[str, int]:
        """Get summary of all branches."""
        ...


class ConfigurationProvider(Protocol):
    """Protocol for configuration management with type safety."""

    def get_model_config(self, provider_name: Optional[str] = None) -> Any:
        """Get model configuration."""
        ...

    def check_required_api_keys(self, provider_name: Optional[str] = None) -> List[str]:
        """Check for required API keys."""
        ...


# Custom exceptions for better error handling
class ThoughtProcessingError(Exception):
    """Base exception for thought processing errors."""

    def __init__(self, message: str, metadata: Optional[ProcessingMetadata] = None):
        super().__init__(message)
        self.metadata = metadata or {}


class RoutingDecisionError(ThoughtProcessingError):
    """Error in adaptive routing decision making."""

    pass


class CostOptimizationError(ThoughtProcessingError):
    """Error in cost optimization logic."""

    pass


class PersistentStorageError(ThoughtProcessingError):
    """Error in persistent memory storage."""

    pass


class ValidationError(ThoughtProcessingError):
    """Error in input validation."""

    pass


class ConfigurationError(Exception):
    """Error in configuration setup."""

    pass


class ModelConfigurationError(ConfigurationError):
    """Error in model configuration."""

    pass


class ProviderError(Exception):
    """Error related to LLM providers."""

    pass


class TeamCreationError(Exception):
    """Error in team creation."""

    pass


class AgentCreationError(Exception):
    """Error in agent creation."""

    pass


# Type aliases for commonly used types
ThoughtNumber = int
BranchId = str
ProviderName = str
TeamType = str
AgentType = str
ConfigDict = Dict[str, Any]
InstructionsList = List[str]
SuccessCriteriaList = List[str]
