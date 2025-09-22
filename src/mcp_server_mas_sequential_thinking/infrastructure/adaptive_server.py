"""Adaptive server integration combining routing, memory, and cost optimization."""

import os
import uuid
from datetime import datetime

# Lazy import to avoid circular dependency
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.routing.adaptive_routing import (
        AdaptiveRouter,
        RoutingDecision,
    )
    from mcp_server_mas_sequential_thinking.services.server_core import ThoughtProcessor
from mcp_server_mas_sequential_thinking.agents.unified_team import create_team_by_type
from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.optimization.cost_optimization import (
    CostOptimizer,
    get_cost_optimizer_from_env,
)
from mcp_server_mas_sequential_thinking.routing.adaptive_routing import (
    ProcessingStrategy,
)

# Lazy import to avoid circular dependency - will import dynamically when needed
from .logging_config import get_logger
from .persistent_memory import PersistentMemoryManager, get_database_url_from_env

logger = get_logger(__name__)


class AdaptiveThoughtProcessor:
    """Enhanced thought processor with adaptive routing, cost optimization, and persistent memory."""

    def __init__(
        self,
        enable_adaptive_routing: bool = True,
        enable_cost_optimization: bool = True,
        enable_persistent_memory: bool = True,
        session_id: str | None = None,
    ) -> None:
        self.enable_adaptive_routing = enable_adaptive_routing
        self.enable_cost_optimization = enable_cost_optimization
        self.enable_persistent_memory = enable_persistent_memory

        # Generate session ID if not provided
        self.session_id = (
            session_id
            or f"session_{uuid.uuid4().hex[:8]}_{int(datetime.utcnow().timestamp())}"
        )

        # Initialize components
        self.adaptive_router: AdaptiveRouter | None = None
        self.cost_optimizer: CostOptimizer | None = None
        self.memory_manager: PersistentMemoryManager | None = None
        self.thought_processor: ThoughtProcessor | None = None

        self._initialize_components()

        logger.info(
            f"Adaptive processor initialized - "
            f"routing: {enable_adaptive_routing}, "
            f"cost_opt: {enable_cost_optimization}, "
            f"memory: {enable_persistent_memory}"
        )

    def _initialize_components(self) -> None:
        """Initialize all components based on configuration."""
        # Initialize adaptive routing
        if self.enable_adaptive_routing:
            if self.enable_cost_optimization:
                float(os.getenv("DAILY_BUDGET_LIMIT", "0")) or None

            # Dynamic import to avoid circular dependency
            from mcp_server_mas_sequential_thinking.routing.adaptive_routing import (
                AdaptiveRouter,
            )
            self.adaptive_router = AdaptiveRouter()
            logger.info("Adaptive routing initialized")

        # Initialize cost optimization
        if self.enable_cost_optimization:
            self.cost_optimizer = get_cost_optimizer_from_env()
            logger.info("Cost optimization initialized")

        # Initialize persistent memory
        if self.enable_persistent_memory:
            database_url = get_database_url_from_env()
            self.memory_manager = PersistentMemoryManager(database_url)

            # Create session
            provider = os.getenv("LLM_PROVIDER", "deepseek")
            self.memory_manager.create_session(self.session_id, provider)
            logger.info(f"Persistent memory initialized - session: {self.session_id}")

    async def process_thought_adaptive(
        self, thought_data: ThoughtData, provider: str | None = None
    ) -> str:
        """Process thought with adaptive routing, cost optimization, and memory management."""
        start_time = datetime.utcnow()
        provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
        processing_metadata: dict[str, Any] = {}

        try:
            # Step 1: Adaptive Routing Decision
            routing_decision = None
            selected_provider = provider

            if self.enable_adaptive_routing and self.adaptive_router:
                if self.cost_optimizer:
                    self._get_remaining_budget()

                routing_decision = self.adaptive_router.analyze(thought_data)

                logger.info(f"Adaptive routing: {routing_decision.reasoning}")

                # Step 2: Cost Optimization
                if self.enable_cost_optimization and self.cost_optimizer:
                    routing_decision, selected_provider = (
                        self.cost_optimizer.optimize_routing_decision(
                            routing_decision, provider or ""
                        )
                    )

                    logger.info(f"Cost optimization: selected {selected_provider}")

            # Step 3: Create appropriate team/processor
            team_mode = self._determine_team_mode(routing_decision)
            team = create_team_by_type(team_mode)

            # For now, use basic thought processor (can be enhanced with strategy-specific processing)
            from mcp_server_mas_sequential_thinking.core.session import SessionMemory
            from mcp_server_mas_sequential_thinking.services.server_core import (
                ThoughtProcessor,
            )

            session_memory = SessionMemory(team=team)
            basic_processor = ThoughtProcessor(session_memory)

            # Step 4: Process the thought
            response = await basic_processor.process_thought(thought_data)

            # Step 5: Calculate actual metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Estimate actual token usage and cost (simplified)
            if routing_decision:
                estimated_tokens = sum(routing_decision.estimated_token_usage) // 2
                actual_cost = routing_decision.estimated_cost
            else:
                estimated_tokens = int(len(response.split()) * 1.3)  # Rough estimate
                actual_cost = estimated_tokens / 1000 * 0.0002  # Default rate

            processing_metadata = {
                "strategy": (
                    routing_decision.strategy.value if routing_decision else "standard"
                ),
                "complexity_score": (
                    routing_decision.complexity_score if routing_decision else 0.0
                ),
                "estimated_cost": (
                    routing_decision.estimated_cost if routing_decision else actual_cost
                ),
                "actual_cost": actual_cost,
                "token_usage": int(estimated_tokens),
                "processing_time": processing_time,
                "specialists": (
                    routing_decision.specialist_recommendations
                    if routing_decision
                    else []
                ),
                "provider": selected_provider,
                "routing_reasoning": (
                    routing_decision.reasoning
                    if routing_decision
                    else "Standard processing"
                ),
            }

            # Step 6: Record usage and store in memory
            if (
                self.enable_cost_optimization
                and self.cost_optimizer
                and routing_decision
            ):
                self.cost_optimizer.record_usage(
                    selected_provider or "",
                    routing_decision.strategy,
                    routing_decision.complexity_level,
                    actual_cost,
                    int(estimated_tokens),
                )

            if self.enable_persistent_memory and self.memory_manager:
                self.memory_manager.store_thought(
                    self.session_id, thought_data, response, processing_metadata
                )

            logger.info(
                f"Thought processed - strategy: {processing_metadata['strategy']}, "
                f"cost: ${actual_cost:.4f}, time: {processing_time:.2f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error in adaptive thought processing: {e}", exc_info=True)

            # Store error information if possible
            if self.enable_persistent_memory and self.memory_manager:
                error_metadata = {
                    "strategy": "error",
                    "error": str(e),
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                }
                try:
                    self.memory_manager.store_thought(
                        self.session_id,
                        thought_data,
                        f"Error processing thought: {e}",
                        error_metadata,
                    )
                except:
                    logger.exception("Failed to store error information")

            raise

    def _determine_team_mode(self, routing_decision: "RoutingDecision | None") -> str:
        """Determine team mode based on routing decision."""
        if not routing_decision:
            return os.getenv("TEAM_MODE", "standard")

        # Map routing strategy to team mode
        strategy_to_mode = {
            ProcessingStrategy.SINGLE_AGENT: "standard",
            ProcessingStrategy.MULTI_AGENT: "enhanced",
            ProcessingStrategy.HYBRID: "hybrid",
        }

        return strategy_to_mode.get(routing_decision.strategy, "standard")

    def _get_remaining_budget(self) -> float | None:
        """Get remaining budget for cost optimization."""
        if not self.cost_optimizer:
            return None

        constraints = self.cost_optimizer.budget_constraints
        constraints.reset_daily_if_needed()

        if constraints.daily_limit:
            return constraints.daily_limit - constraints.daily_spent

        return None

    def get_session_summary(self) -> dict[str, Any]:
        """Get comprehensive session summary."""
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "features_enabled": {
                "adaptive_routing": self.enable_adaptive_routing,
                "cost_optimization": self.enable_cost_optimization,
                "persistent_memory": self.enable_persistent_memory,
            },
        }

        # Add memory statistics
        if self.memory_manager:
            try:
                stats = self.memory_manager.get_usage_stats(days_back=1)
                summary["memory_stats"] = stats
            except Exception as e:
                logger.warning(f"Failed to get memory stats: {e}")
                summary["memory_stats"] = {"error": str(e)}

        # Add cost optimization report
        if self.cost_optimizer:
            try:
                cost_report = self.cost_optimizer.get_cost_report()
                summary["cost_report"] = cost_report
                summary["optimization_suggestions"] = (
                    self.cost_optimizer.suggest_optimizations()
                )
            except Exception as e:
                logger.warning(f"Failed to get cost report: {e}")
                summary["cost_report"] = {"error": str(e)}

        return summary

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.memory_manager:
            self.memory_manager.close()

        logger.info(f"Adaptive processor cleaned up - session: {self.session_id}")


def create_adaptive_processor(
    session_id: str | None = None,
    enable_adaptive_routing: bool | None = None,
    enable_cost_optimization: bool | None = None,
    enable_persistent_memory: bool | None = None,
) -> AdaptiveThoughtProcessor:
    """Create adaptive processor with environment-based configuration."""
    # Use environment variables as defaults
    if enable_adaptive_routing is None:
        enable_adaptive_routing = (
            os.getenv("ENABLE_ADAPTIVE_ROUTING", "true").lower() == "true"
        )

    if enable_cost_optimization is None:
        enable_cost_optimization = bool(
            os.getenv("DAILY_BUDGET_LIMIT")
            or os.getenv("MONTHLY_BUDGET_LIMIT")
            or os.getenv("PER_THOUGHT_BUDGET_LIMIT")
        )

    if enable_persistent_memory is None:
        enable_persistent_memory = True  # Default enabled

    return AdaptiveThoughtProcessor(
        enable_adaptive_routing=enable_adaptive_routing,
        enable_cost_optimization=enable_cost_optimization,
        enable_persistent_memory=enable_persistent_memory,
        session_id=session_id,
    )
