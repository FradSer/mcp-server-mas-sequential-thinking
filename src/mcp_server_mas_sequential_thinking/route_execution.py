"""Route execution validation system for ensuring routing decision consistency."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, validator, Field

from .ai_routing import ProcessingStrategy, ComplexityLevel, RoutingDecision

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for different routing strategies."""

    SINGLE_AGENT = "single_agent"
    SELECTIVE_TEAM = "selective_team"  # Hybrid with specific specialists
    FULL_TEAM = "full_team"  # Complete multi-agent team


class RouteExecutionPlan(BaseModel):
    """Validated execution plan ensuring routing decision consistency."""

    strategy: ProcessingStrategy
    complexity_level: ComplexityLevel
    execution_mode: ExecutionMode
    required_specialists: List[str] = Field(min_items=1)
    team_size: int = Field(ge=1, le=10)
    timeout_seconds: float = Field(gt=0)

    # Metadata for validation and logging
    original_decision: Optional[Dict[str, Any]] = None
    validation_passed: bool = False

    @validator('required_specialists')
    def validate_specialists(cls, v, values):
        """Validate specialist requirements match strategy."""
        strategy = values.get('strategy')

        # Strategy-specific validation
        if strategy == ProcessingStrategy.SINGLE_AGENT:
            if len(v) != 1 or v[0] != "general":
                raise ValueError("Single agent strategy requires exactly ['general'] specialist")

        elif strategy == ProcessingStrategy.HYBRID:
            if len(v) < 2 or len(v) > 4:
                raise ValueError("Hybrid strategy requires 2-4 specialists")
            # Validate known specialist types
            valid_specialists = {"planner", "researcher", "analyzer", "critic", "synthesizer", "general"}
            invalid = set(v) - valid_specialists
            if invalid:
                raise ValueError(f"Unknown specialists: {invalid}")

        elif strategy == ProcessingStrategy.MULTI_AGENT:
            if len(v) < 3:
                raise ValueError("Multi-agent strategy requires at least 3 specialists")

        return v

    @validator('team_size')
    def validate_team_size(cls, v, values):
        """Ensure team size matches specialist count."""
        specialists = values.get('required_specialists', [])
        if v != len(specialists):
            raise ValueError(f"Team size {v} must match specialist count {len(specialists)}")
        return v

    @validator('execution_mode')
    def validate_execution_mode(cls, v, values):
        """Ensure execution mode matches strategy."""
        strategy = values.get('strategy')

        if strategy == ProcessingStrategy.SINGLE_AGENT and v != ExecutionMode.SINGLE_AGENT:
            raise ValueError("Single agent strategy requires SINGLE_AGENT execution mode")
        elif strategy == ProcessingStrategy.HYBRID and v != ExecutionMode.SELECTIVE_TEAM:
            raise ValueError("Hybrid strategy requires SELECTIVE_TEAM execution mode")
        elif strategy == ProcessingStrategy.MULTI_AGENT and v != ExecutionMode.FULL_TEAM:
            raise ValueError("Multi-agent strategy requires FULL_TEAM execution mode")

        return v

    def validate_plan(self) -> bool:
        """Final validation of the complete execution plan."""
        try:
            # Additional cross-field validation
            if self.strategy == ProcessingStrategy.SINGLE_AGENT:
                assert self.team_size == 1
                assert self.execution_mode == ExecutionMode.SINGLE_AGENT
                assert self.required_specialists == ["general"]

            elif self.strategy == ProcessingStrategy.HYBRID:
                assert 2 <= self.team_size <= 4
                assert self.execution_mode == ExecutionMode.SELECTIVE_TEAM
                assert "synthesizer" in self.required_specialists  # Hybrid requires synthesis

            elif self.strategy == ProcessingStrategy.MULTI_AGENT:
                assert self.team_size >= 3
                assert self.execution_mode == ExecutionMode.FULL_TEAM

            self.validation_passed = True
            logger.info(f"✅ Execution plan validated: {self.strategy.value} with {self.team_size} specialists")
            return True

        except AssertionError as e:
            logger.error(f"❌ Execution plan validation failed: {e}")
            self.validation_passed = False
            return False


class RouteExecutionValidator:
    """Validates and converts routing decisions to execution plans."""

    # Default timeout mappings
    TIMEOUT_MAPPING = {
        (ProcessingStrategy.SINGLE_AGENT, ComplexityLevel.SIMPLE): 45.0,
        (ProcessingStrategy.SINGLE_AGENT, ComplexityLevel.MODERATE): 60.0,
        (ProcessingStrategy.HYBRID, ComplexityLevel.MODERATE): 120.0,
        (ProcessingStrategy.HYBRID, ComplexityLevel.COMPLEX): 180.0,
        (ProcessingStrategy.MULTI_AGENT, ComplexityLevel.COMPLEX): 240.0,
        (ProcessingStrategy.MULTI_AGENT, ComplexityLevel.HIGHLY_COMPLEX): 300.0,
    }

    # Execution mode mapping
    EXECUTION_MODE_MAPPING = {
        ProcessingStrategy.SINGLE_AGENT: ExecutionMode.SINGLE_AGENT,
        ProcessingStrategy.HYBRID: ExecutionMode.SELECTIVE_TEAM,
        ProcessingStrategy.MULTI_AGENT: ExecutionMode.FULL_TEAM,
    }

    def create_execution_plan(self, routing_decision: RoutingDecision) -> RouteExecutionPlan:
        """Convert routing decision to validated execution plan."""

        # Determine execution mode
        execution_mode = self.EXECUTION_MODE_MAPPING[routing_decision.strategy]

        # Get specialists (use recommendations or defaults)
        specialists = self._get_validated_specialists(
            routing_decision.strategy,
            routing_decision.specialist_recommendations
        )

        # Calculate timeout
        timeout = self._calculate_timeout(routing_decision.strategy, routing_decision.complexity_level)

        # Create execution plan
        plan = RouteExecutionPlan(
            strategy=routing_decision.strategy,
            complexity_level=routing_decision.complexity_level,
            execution_mode=execution_mode,
            required_specialists=specialists,
            team_size=len(specialists),
            timeout_seconds=timeout,
            original_decision={
                "reasoning": routing_decision.reasoning,
                "complexity_score": routing_decision.complexity_score,
                "estimated_cost": routing_decision.estimated_cost
            }
        )

        # Validate the plan
        if not plan.validate_plan():
            raise ValueError(f"Failed to create valid execution plan from routing decision: {routing_decision}")

        logger.info(f"📋 Execution plan created: {plan.execution_mode.value} with {specialists}")
        return plan

    def _get_validated_specialists(self, strategy: ProcessingStrategy, recommendations: List[str]) -> List[str]:
        """Get validated specialist list based on strategy and recommendations."""

        if strategy == ProcessingStrategy.SINGLE_AGENT:
            return ["general"]

        elif strategy == ProcessingStrategy.HYBRID:
            # Use recommendations but ensure minimum requirements
            specialists = recommendations.copy() if recommendations else []

            # Ensure we have essential specialists for hybrid
            if "synthesizer" not in specialists:
                specialists.append("synthesizer")

            # Ensure we have at least one analysis specialist
            analysis_specialists = {"planner", "analyzer", "researcher"}
            if not any(s in specialists for s in analysis_specialists):
                specialists.append("planner")

            # Limit to 4 for hybrid efficiency
            return specialists[:4]

        elif strategy == ProcessingStrategy.MULTI_AGENT:
            # Use full team or recommendations for complex tasks
            if recommendations and len(recommendations) >= 3:
                return recommendations
            else:
                # Default full team
                return ["planner", "researcher", "analyzer", "critic", "synthesizer"]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _calculate_timeout(self, strategy: ProcessingStrategy, complexity: ComplexityLevel) -> float:
        """Calculate appropriate timeout for strategy and complexity."""

        key = (strategy, complexity)
        if key in self.TIMEOUT_MAPPING:
            return self.TIMEOUT_MAPPING[key]

        # Fallback calculation
        base_timeout = {
            ProcessingStrategy.SINGLE_AGENT: 60.0,
            ProcessingStrategy.HYBRID: 150.0,
            ProcessingStrategy.MULTI_AGENT: 240.0,
        }[strategy]

        complexity_multiplier = {
            ComplexityLevel.SIMPLE: 0.8,
            ComplexityLevel.MODERATE: 1.0,
            ComplexityLevel.COMPLEX: 1.5,
            ComplexityLevel.HIGHLY_COMPLEX: 2.0,
        }[complexity]

        return base_timeout * complexity_multiplier


@dataclass
class ExecutionResult:
    """Result of route execution with validation data."""

    plan: RouteExecutionPlan
    response: str
    actual_strategy: str
    processing_time: float
    consistency_verified: bool
    issues: List[str]

    def verify_consistency(self) -> bool:
        """Verify execution consistency with plan."""
        issues = []

        # Check strategy consistency
        expected_strategy = self.plan.execution_mode.value
        if self.actual_strategy != expected_strategy:
            issues.append(f"Strategy mismatch: expected {expected_strategy}, got {self.actual_strategy}")

        # Check timeout compliance
        if self.processing_time > self.plan.timeout_seconds * 1.1:  # 10% tolerance
            issues.append(f"Timeout exceeded: {self.processing_time:.1f}s > {self.plan.timeout_seconds:.1f}s")

        # Check response quality
        if not self.response or len(self.response.strip()) < 10:
            issues.append("Response too short or empty")

        self.issues = issues
        self.consistency_verified = len(issues) == 0

        if issues:
            logger.warning(f"⚠️ Execution consistency issues: {'; '.join(issues)}")
        else:
            logger.info(f"✅ Execution consistency verified")

        return self.consistency_verified


# Factory function for easy integration
def create_execution_validator() -> RouteExecutionValidator:
    """Create a route execution validator."""
    return RouteExecutionValidator()