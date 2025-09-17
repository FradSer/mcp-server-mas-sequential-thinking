"""Cost optimization framework for dynamic provider selection and budget control."""

import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod

from .ai_routing import ProcessingStrategy, ComplexityLevel, RoutingDecision
from .models import ThoughtData
from .constants import (
    TokenCosts,
    QualityThresholds,
    ProviderDefaults,
    ComplexityThresholds
)

logger = logging.getLogger(__name__)


class CostTier(Enum):
    """Cost tiers for provider classification."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class ProviderProfile:
    """Profile for LLM provider with cost and performance characteristics."""

    name: str
    cost_tier: CostTier
    cost_per_1k_tokens: float
    avg_quality_score: float = ProviderDefaults.DEFAULT_QUALITY_SCORE  # 0.0-1.0
    avg_response_time: float = ProviderDefaults.DEFAULT_RESPONSE_TIME  # seconds
    rate_limit_per_hour: Optional[int] = None
    supports_streaming: bool = True
    supports_function_calling: bool = True
    max_context_length: int = ProviderDefaults.DEFAULT_CONTEXT_LENGTH

    # Model specifications
    team_model_id: Optional[str] = None
    agent_model_id: Optional[str] = None

    # Availability and reliability
    uptime_score: float = ProviderDefaults.DEFAULT_UPTIME_SCORE  # 0.0-1.0
    error_rate: float = ProviderDefaults.DEFAULT_ERROR_RATE  # 0.0-1.0

    @property
    def cost_effectiveness(self) -> float:
        """Calculate cost effectiveness (quality per dollar)."""
        if self.cost_per_1k_tokens == 0:
            return float("inf")  # Free tier
        return self.avg_quality_score / self.cost_per_1k_tokens

    @property
    def overall_score(self) -> float:
        """Calculate overall provider score."""
        # Weighted scoring: quality 40%, cost 30%, speed 20%, reliability 10%
        speed_score = max(0, 1 - (self.avg_response_time - 1) / 10)  # Normalize to 0-1
        reliability_score = self.uptime_score * (1 - self.error_rate)

        score = (
            self.avg_quality_score * 0.4
            + (1 / (self.cost_per_1k_tokens + 0.0001)) * 0.0003  # Normalize cost impact
            + speed_score * 0.2
            + reliability_score * 0.1
        )
        return min(score, 1.0)


@dataclass
class BudgetConstraints:
    """Budget constraints and limits."""

    daily_limit: Optional[float] = None
    monthly_limit: Optional[float] = None
    per_thought_limit: Optional[float] = None
    emergency_reserve: float = 0.0

    # Usage tracking
    daily_spent: float = 0.0
    monthly_spent: float = 0.0
    last_reset_date: datetime = field(default_factory=datetime.utcnow)

    def reset_daily_if_needed(self) -> None:
        """Reset daily spending if date has changed."""
        today = datetime.utcnow().date()
        if self.last_reset_date.date() < today:
            self.daily_spent = 0.0
            self.last_reset_date = datetime.utcnow()

    def can_spend(self, amount: float) -> bool:
        """Check if spending amount is within budget."""
        self.reset_daily_if_needed()

        if self.per_thought_limit and amount > self.per_thought_limit:
            return False
        if self.daily_limit and (self.daily_spent + amount) > self.daily_limit:
            return False
        if self.monthly_limit and (self.monthly_spent + amount) > self.monthly_limit:
            return False

        # Check emergency reserve
        return amount <= (self.daily_limit or float("inf")) - self.emergency_reserve

    def record_spending(self, amount: float) -> None:
        """Record spending against budget."""
        self.reset_daily_if_needed()
        self.daily_spent += amount
        self.monthly_spent += amount


@dataclass
class CostOptimizationMetrics:
    """Metrics for cost optimization tracking."""

    total_requests: int = 0
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0
    cost_savings: float = 0.0  # vs always using most expensive
    quality_maintained: float = 1.0  # 0.0-1.0

    # Strategy effectiveness
    single_agent_usage: int = 0
    multi_agent_usage: int = 0
    hybrid_usage: int = 0

    # Provider distribution
    provider_usage: Dict[str, int] = field(default_factory=dict)
    provider_costs: Dict[str, float] = field(default_factory=dict)


class ProviderProfileFactory:
    """Factory for creating provider profiles with standardized defaults."""

    @staticmethod
    def create_groq_profile() -> ProviderProfile:
        """Create Groq provider profile."""
        return ProviderProfile(
            name="groq",
            cost_tier=CostTier.FREE,
            cost_per_1k_tokens=TokenCosts.GROQ_COST_PER_1K,
            avg_quality_score=0.75,
            avg_response_time=0.8,
            rate_limit_per_hour=14400,
            max_context_length=32768,
            team_model_id="llama3-groq-70b-8192-tool-use-preview",
            agent_model_id="llama3-groq-8b-8192-tool-use-preview",
        )

    @staticmethod
    def create_deepseek_profile() -> ProviderProfile:
        """Create Deepseek provider profile."""
        return ProviderProfile(
            name="deepseek",
            cost_tier=CostTier.LOW,
            cost_per_1k_tokens=TokenCosts.DEEPSEEK_COST_PER_1K,
            avg_quality_score=0.85,
            avg_response_time=ProviderDefaults.DEFAULT_RESPONSE_TIME,
            max_context_length=128000,
            team_model_id="deepseek-reasoning",
            agent_model_id="deepseek-chat",
        )

    @staticmethod
    def create_github_profile() -> ProviderProfile:
        """Create GitHub provider profile."""
        return ProviderProfile(
            name="github",
            cost_tier=CostTier.MEDIUM,
            cost_per_1k_tokens=TokenCosts.GITHUB_COST_PER_1K,
            avg_quality_score=ProviderDefaults.DEFAULT_QUALITY_SCORE,
            avg_response_time=ProviderDefaults.DEFAULT_RESPONSE_TIME,
            max_context_length=128000,
            team_model_id="gpt-4o",
            agent_model_id="gpt-4o-mini",
        )

    @staticmethod
    def create_openrouter_profile() -> ProviderProfile:
        """Create OpenRouter provider profile."""
        return ProviderProfile(
            name="openrouter",
            cost_tier=CostTier.MEDIUM,
            cost_per_1k_tokens=TokenCosts.OPENROUTER_COST_PER_1K,
            avg_quality_score=ProviderDefaults.DEFAULT_QUALITY_SCORE,
            avg_response_time=3.0,
            max_context_length=200000,
            team_model_id="anthropic/claude-3.5-sonnet",
            agent_model_id="anthropic/claude-3.5-haiku",
        )

    @staticmethod
    def create_ollama_profile() -> ProviderProfile:
        """Create Ollama provider profile."""
        return ProviderProfile(
            name="ollama",
            cost_tier=CostTier.FREE,
            cost_per_1k_tokens=TokenCosts.OLLAMA_COST_PER_1K,
            avg_quality_score=0.70,
            avg_response_time=5.0,
            max_context_length=8192,
            team_model_id="devstral:24b",
            agent_model_id="devstral:24b",
        )

    @classmethod
    def get_default_providers(cls) -> Dict[str, ProviderProfile]:
        """Get dictionary of all default provider profiles."""
        return {
            "groq": cls.create_groq_profile(),
            "deepseek": cls.create_deepseek_profile(),
            "github": cls.create_github_profile(),
            "openrouter": cls.create_openrouter_profile(),
            "ollama": cls.create_ollama_profile(),
        }


class CostOptimizer:
    """Main cost optimization framework."""

    def __init__(
        self,
        budget_constraints: Optional[BudgetConstraints] = None,
        provider_profiles: Optional[Dict[str, ProviderProfile]] = None,
        quality_threshold: float = QualityThresholds.DEFAULT_QUALITY_THRESHOLD,
    ):
        self.budget_constraints = budget_constraints or BudgetConstraints()
        self.provider_profiles = provider_profiles or ProviderProfileFactory.get_default_providers()
        self.quality_threshold = quality_threshold
        self.metrics = CostOptimizationMetrics()

        # Load custom configurations from environment
        self._load_provider_configs_from_env()

        logger.info(
            f"Cost optimizer initialized with {len(self.provider_profiles)} providers"
        )

    def _load_provider_configs_from_env(self) -> None:
        """Load provider configurations from environment variables."""
        # Update costs from environment if specified
        for provider_name, profile in self.provider_profiles.items():
            cost_key = f"{provider_name.upper()}_COST_PER_1K_TOKENS"
            if cost_env := os.getenv(cost_key):
                try:
                    profile.cost_per_1k_tokens = float(cost_env)
                    logger.info(
                        f"Updated {provider_name} cost from environment: ${cost_env}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid cost value for {provider_name}: {cost_env}"
                    )

            # Update model IDs if specified
            team_model_key = f"{provider_name.upper()}_TEAM_MODEL_ID"
            agent_model_key = f"{provider_name.upper()}_AGENT_MODEL_ID"

            if team_model := os.getenv(team_model_key):
                profile.team_model_id = team_model
            if agent_model := os.getenv(agent_model_key):
                profile.agent_model_id = agent_model

    def select_optimal_provider(
        self,
        routing_decision: RoutingDecision,
        current_provider: str = "deepseek",
        force_quality_threshold: bool = True,
    ) -> Tuple[str, str]:
        """Select optimal provider based on cost, quality, and constraints.

        Returns:
            Tuple of (selected_provider, reasoning)
        """
        available_providers = self._get_available_providers(
            routing_decision.estimated_cost
        )

        if not available_providers:
            logger.warning("No providers available within budget constraints")
            return current_provider, "No providers within budget - using current"

        # Filter by quality threshold if enforced
        if force_quality_threshold:
            quality_providers = [
                p
                for p in available_providers
                if self.provider_profiles[p].avg_quality_score >= self.quality_threshold
            ]
            if quality_providers:
                available_providers = quality_providers

        # Score providers based on current context
        best_provider = self._score_providers(
            available_providers,
            routing_decision.complexity_level,
            routing_decision.estimated_cost,
        )

        # Generate reasoning
        selected_profile = self.provider_profiles[best_provider]
        reasoning = (
            f"Selected {best_provider} "
            f"(cost: ${selected_profile.cost_per_1k_tokens:.4f}/1k, "
            f"quality: {selected_profile.avg_quality_score:.2f}, "
            f"score: {selected_profile.overall_score:.2f})"
        )

        logger.info(f"Provider selection: {reasoning}")
        return best_provider, reasoning

    def optimize_routing_decision(
        self, routing_decision: RoutingDecision, current_provider: str = "deepseek"
    ) -> Tuple[RoutingDecision, str]:
        """Optimize routing decision based on cost constraints and provider capabilities.

        Returns:
            Tuple of (optimized_routing_decision, selected_provider)
        """
        # Check budget constraints
        max_cost, min_cost = routing_decision.estimated_token_usage
        avg_cost = (max_cost + min_cost) / 2 * 0.0002  # Rough estimate

        if not self.budget_constraints.can_spend(avg_cost):
            logger.info("Budget constraint triggered - optimizing strategy")
            # Force cheaper strategy
            if routing_decision.strategy == ProcessingStrategy.MULTI_AGENT:
                routing_decision.strategy = ProcessingStrategy.HYBRID
            elif routing_decision.strategy == ProcessingStrategy.HYBRID:
                routing_decision.strategy = ProcessingStrategy.SINGLE_AGENT

            # Recalculate cost estimates
            from .adaptive_routing import CostEstimator

            cost_estimator = CostEstimator()
            token_range, estimated_cost = cost_estimator.estimate_cost(
                routing_decision.strategy,
                routing_decision.complexity_level,
                current_provider,
            )
            routing_decision.estimated_token_usage = token_range
            routing_decision.estimated_cost = estimated_cost
            routing_decision.reasoning += " | Budget-optimized strategy"

        # Select optimal provider
        optimal_provider, provider_reasoning = self.select_optimal_provider(
            routing_decision, current_provider
        )

        # Update cost estimates for selected provider
        provider_profile = self.provider_profiles[optimal_provider]
        avg_tokens = (
            routing_decision.estimated_token_usage[0]
            + routing_decision.estimated_token_usage[1]
        ) / 2
        optimized_cost = avg_tokens / 1000 * provider_profile.cost_per_1k_tokens
        routing_decision.estimated_cost = optimized_cost

        return routing_decision, optimal_provider

    def _get_available_providers(self, max_cost: float) -> List[str]:
        """Get providers that are within budget constraints."""
        available = []

        for provider_name, profile in self.provider_profiles.items():
            # Estimate cost for this provider
            estimated_cost = max_cost * (
                profile.cost_per_1k_tokens / 0.0002
            )  # Normalize

            if self.budget_constraints.can_spend(estimated_cost):
                available.append(provider_name)

        return available

    def _score_providers(
        self,
        providers: List[str],
        complexity_level: ComplexityLevel,
        estimated_cost: float,
    ) -> str:
        """Score and rank providers based on current context."""
        scores = {}

        for provider_name in providers:
            profile = self.provider_profiles[provider_name]

            # Base score from profile
            base_score = profile.overall_score

            # Adjust for complexity level
            complexity_bonus = 0.0
            if complexity_level in [
                ComplexityLevel.COMPLEX,
                ComplexityLevel.HIGHLY_COMPLEX,
            ]:
                # Prefer higher quality providers for complex tasks
                complexity_bonus = profile.avg_quality_score * 0.2
            else:
                # Prefer cost-effective providers for simple tasks
                complexity_bonus = profile.cost_effectiveness * 0.0001  # Small impact

            # Budget impact
            budget_score = 1.0
            if self.budget_constraints.daily_limit:
                budget_utilization = (
                    self.budget_constraints.daily_spent
                    / self.budget_constraints.daily_limit
                )
                if budget_utilization > 0.8:  # High budget utilization
                    budget_score = (
                        2.0
                        if profile.cost_tier in [CostTier.FREE, CostTier.LOW]
                        else 0.5
                    )

            final_score = (base_score + complexity_bonus) * budget_score
            scores[provider_name] = final_score

        # Return provider with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def record_usage(
        self,
        provider: str,
        strategy: ProcessingStrategy,
        complexity_level: ComplexityLevel,
        actual_cost: float,
        actual_tokens: int,
        quality_score: Optional[float] = None,
    ) -> None:
        """Record actual usage for optimization learning."""
        self.budget_constraints.record_spending(actual_cost)

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.total_cost += actual_cost
        self.metrics.avg_cost_per_request = (
            self.metrics.total_cost / self.metrics.total_requests
        )

        if quality_score:
            # Update running average
            old_quality = self.metrics.quality_maintained
            self.metrics.quality_maintained = (
                old_quality * (self.metrics.total_requests - 1) + quality_score
            ) / self.metrics.total_requests

        # Strategy tracking
        if strategy == ProcessingStrategy.SINGLE_AGENT:
            self.metrics.single_agent_usage += 1
        elif strategy == ProcessingStrategy.MULTI_AGENT:
            self.metrics.multi_agent_usage += 1
        elif strategy == ProcessingStrategy.HYBRID:
            self.metrics.hybrid_usage += 1

        # Provider tracking
        self.metrics.provider_usage[provider] = (
            self.metrics.provider_usage.get(provider, 0) + 1
        )
        self.metrics.provider_costs[provider] = (
            self.metrics.provider_costs.get(provider, 0.0) + actual_cost
        )

        # Update provider profile with actual performance
        if provider in self.provider_profiles:
            profile = self.provider_profiles[provider]
            # Simple moving average for quality score
            if quality_score:
                profile.avg_quality_score = (
                    profile.avg_quality_score * 0.9 + quality_score * 0.1
                )

        logger.debug(
            f"Usage recorded: {provider} {strategy.value} - "
            f"${actual_cost:.4f} ({actual_tokens} tokens)"
        )

    def get_cost_report(self) -> Dict:
        """Generate comprehensive cost optimization report."""
        self.budget_constraints.reset_daily_if_needed()

        # Budget status
        budget_status = {
            "daily_spent": self.budget_constraints.daily_spent,
            "daily_limit": self.budget_constraints.daily_limit,
            "daily_remaining": (self.budget_constraints.daily_limit or 0)
            - self.budget_constraints.daily_spent,
            "monthly_spent": self.budget_constraints.monthly_spent,
            "monthly_limit": self.budget_constraints.monthly_limit,
        }

        # Usage efficiency
        total_usage = (
            self.metrics.single_agent_usage
            + self.metrics.multi_agent_usage
            + self.metrics.hybrid_usage
        )

        efficiency_metrics = {
            "single_agent_ratio": self.metrics.single_agent_usage / max(total_usage, 1),
            "multi_agent_ratio": self.metrics.multi_agent_usage / max(total_usage, 1),
            "hybrid_ratio": self.metrics.hybrid_usage / max(total_usage, 1),
            "avg_cost_per_request": self.metrics.avg_cost_per_request,
            "quality_maintained": self.metrics.quality_maintained,
        }

        # Provider performance
        provider_performance = {}
        for provider, usage_count in self.metrics.provider_usage.items():
            cost = self.metrics.provider_costs.get(provider, 0.0)
            provider_performance[provider] = {
                "usage_count": usage_count,
                "total_cost": cost,
                "avg_cost_per_request": cost / max(usage_count, 1),
                "usage_percentage": usage_count / max(total_usage, 1),
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "budget_status": budget_status,
            "efficiency_metrics": efficiency_metrics,
            "provider_performance": provider_performance,
            "optimization_metrics": asdict(self.metrics),
        }

    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on usage patterns."""
        suggestions = []

        # Budget utilization
        if self.budget_constraints.daily_limit:
            utilization = (
                self.budget_constraints.daily_spent
                / self.budget_constraints.daily_limit
            )
            if utilization > 0.9:
                suggestions.append(
                    "Daily budget nearly exhausted - consider increasing limit or reducing usage"
                )
            elif utilization > 0.7:
                suggestions.append(
                    "High daily budget utilization - monitor usage closely"
                )

        # Strategy distribution
        total_usage = (
            self.metrics.single_agent_usage
            + self.metrics.multi_agent_usage
            + self.metrics.hybrid_usage
        )

        if total_usage > 10:  # Only suggest if we have enough data
            multi_agent_ratio = self.metrics.multi_agent_usage / total_usage
            if multi_agent_ratio > 0.7:
                suggestions.append(
                    "High multi-agent usage detected - consider using hybrid strategy for cost savings"
                )

            single_agent_ratio = self.metrics.single_agent_usage / total_usage
            if single_agent_ratio > 0.8:
                suggestions.append(
                    "High single-agent usage - you may benefit from multi-agent for complex thoughts"
                )

        # Provider optimization
        if self.metrics.provider_costs:
            most_expensive = max(
                self.metrics.provider_costs.items(), key=lambda x: x[1]
            )
            cheapest_available = min(
                [
                    (p, profile.cost_per_1k_tokens)
                    for p, profile in self.provider_profiles.items()
                ],
                key=lambda x: x[1],
            )

            if (
                most_expensive[1] > 0.01 and cheapest_available[1] == 0.0
            ):  # Significant cost with free alternative
                suggestions.append(
                    f"Consider using {cheapest_available[0]} (free) instead of {most_expensive[0]} for cost savings"
                )

        return suggestions


# Convenience functions
def create_cost_optimizer(
    daily_budget: Optional[float] = None,
    monthly_budget: Optional[float] = None,
    per_thought_budget: Optional[float] = None,
    quality_threshold: float = 0.7,
) -> CostOptimizer:
    """Create a cost optimizer with budget constraints."""
    constraints = None
    if any([daily_budget, monthly_budget, per_thought_budget]):
        constraints = BudgetConstraints(
            daily_limit=daily_budget,
            monthly_limit=monthly_budget,
            per_thought_limit=per_thought_budget,
        )

    return CostOptimizer(
        budget_constraints=constraints, quality_threshold=quality_threshold
    )


def get_cost_optimizer_from_env() -> CostOptimizer:
    """Create cost optimizer from environment variables."""
    daily_limit = float(os.getenv("DAILY_BUDGET_LIMIT", "0")) or None
    monthly_limit = float(os.getenv("MONTHLY_BUDGET_LIMIT", "0")) or None
    per_thought_limit = float(os.getenv("PER_THOUGHT_BUDGET_LIMIT", "0")) or None
    quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.7"))

    return create_cost_optimizer(
        daily_budget=daily_limit,
        monthly_budget=monthly_limit,
        per_thought_budget=per_thought_limit,
        quality_threshold=quality_threshold,
    )
