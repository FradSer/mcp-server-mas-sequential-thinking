"""Intelligent Coordinator: Unified routing and coordination system."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from agno.agent import Agent
from agno.models.base import Model

from .models import ThoughtData
from .ai_routing import (
    AIComplexityAnalyzer, ProcessingStrategy, ComplexityLevel,
    RoutingDecision, AIRoutingAnalysis
)
from .route_execution import RouteExecutionPlan, ExecutionMode
from .modernized_config import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class CoordinationPlan:
    """Comprehensive plan combining routing and coordination decisions."""

    # Routing decisions
    strategy: ProcessingStrategy
    complexity_level: ComplexityLevel
    complexity_score: float

    # Coordination decisions
    execution_mode: ExecutionMode
    specialist_roles: List[str]
    task_breakdown: List[str]
    coordination_strategy: str

    # Execution parameters
    timeout_seconds: float
    expected_interactions: int
    team_size: int  # Added missing team_size attribute

    # Reasoning and metadata
    reasoning: str
    confidence: float
    original_thought: str


class IntelligentCoordinator:
    """
    Unified system that combines AI routing and strategic planning.

    Eliminates the logical duplication between AI router and Planner by:
    1. Single complexity analysis
    2. Unified strategy decisions
    3. Direct specialist coordination
    4. End-to-end responsibility
    """

    def __init__(self, model: Optional[Model] = None):
        """Initialize the intelligent coordinator."""
        self.model = model or self._get_coordination_model()
        self.coordination_agent = self._create_coordination_agent()

    def _get_coordination_model(self) -> Model:
        """Get optimized model for coordination decisions."""
        config = get_model_config()
        # Use team model for coordination (more powerful for complex decisions)
        return config.provider_class(id=config.team_model_id)

    def _create_coordination_agent(self) -> Agent:
        """Create the unified coordination agent."""
        return Agent(
            name="IntelligentCoordinator",
            role="Unified Routing and Coordination Manager",
            description="Analyzes thoughts and creates comprehensive execution plans with direct specialist coordination",
            model=self.model,
            instructions=[
                "You are the master coordinator responsible for the complete thought processing pipeline.",
                "Your job is to analyze thoughts and create detailed execution plans including:",
                "1. Complexity assessment and strategy selection",
                "2. Specialist team composition and role assignments",
                "3. Task breakdown and coordination strategy",
                "4. Direct execution parameters and expectations",
                "",
                "ELIMINATE REDUNDANCY: You replace both the router and planner - make all decisions in one pass.",
                "THINK STRATEGICALLY: Consider the full pipeline from analysis to synthesis.",
                "BE DECISIVE: Provide clear, actionable plans with specific coordination instructions."
            ],
            structured_outputs=True,
            markdown=False
        )

    async def create_coordination_plan(self, thought_data: ThoughtData) -> CoordinationPlan:
        """Create comprehensive coordination plan combining routing and planning."""

        coordination_prompt = self._build_coordination_prompt(thought_data)

        try:
            # Single AI call for complete analysis and planning
            response = await self.coordination_agent.arun(coordination_prompt)
            plan = self._parse_coordination_response(response, thought_data)

            logger.info(
                f"🎯 Coordination plan created: {plan.strategy.value} with {len(plan.specialist_roles)} specialists"
            )
            logger.info(f"📊 Complexity: {plan.complexity_level.value} (score: {plan.complexity_score:.1f})")
            logger.info(f"🔄 Coordination: {plan.coordination_strategy}")

            return plan

        except Exception as e:
            logger.warning(f"Coordination planning failed: {e}, using fallback")
            return self._create_fallback_plan(thought_data)

    def _build_coordination_prompt(self, thought_data: ThoughtData) -> str:
        """Build comprehensive coordination prompt."""

        context_info = ""
        if thought_data.thought_number > 1:
            context_info = f"This is thought #{thought_data.thought_number} in a {thought_data.total_thoughts}-thought sequence. "

        if thought_data.branch_from:
            context_info += f"This branches from thought #{thought_data.branch_from}. "

        return f"""As the Intelligent Coordinator, analyze this thought and create a complete execution plan:

{context_info}

THOUGHT TO COORDINATE:
"{thought_data.thought}"

Create a comprehensive coordination plan with this JSON structure:
{{
    "complexity_analysis": {{
        "complexity_level": "simple|moderate|complex|highly_complex",
        "complexity_score": <0-100>,
        "reasoning": "<why this complexity level>"
    }},
    "strategy_decision": {{
        "processing_strategy": "single_agent|hybrid|multi_agent",
        "execution_mode": "single_agent|selective_team|full_team",
        "reasoning": "<why this strategy>"
    }},
    "team_composition": {{
        "specialist_roles": ["<role1>", "<role2>", ...],
        "role_assignments": {{
            "<role>": "<specific responsibility for this thought>",
            ...
        }},
        "team_size": <number>
    }},
    "coordination_plan": {{
        "task_breakdown": ["<subtask1>", "<subtask2>", ...],
        "coordination_strategy": "<how to coordinate the specialists>",
        "interaction_flow": "<sequence of specialist interactions>",
        "synthesis_approach": "<how to combine specialist outputs>"
    }},
    "execution_parameters": {{
        "timeout_seconds": <recommended timeout>,
        "expected_interactions": <number of specialist interactions>,
        "success_criteria": ["<criterion1>", "<criterion2>", ...]
    }},
    "confidence": <0.0-1.0>
}}

SPECIALIST ROLES AVAILABLE:
- researcher: Information gathering and fact-finding
- analyzer: Deep analytical thinking and pattern recognition
- critic: Quality assessment and critical evaluation
- synthesizer: Integration and coherent response formation
- general: Single-agent processing for simple tasks

COORDINATION STRATEGIES:
- "sequential": Specialists work in sequence (researcher → analyzer → synthesizer)
- "parallel": Multiple specialists work simultaneously then synthesize
- "iterative": Specialists collaborate in multiple rounds
- "direct": Single specialist handles the entire task

COMPLEXITY GUIDELINES:
- simple (0-25): Direct questions, basic tasks → single_agent
- moderate (25-50): Some analysis needed → hybrid with 2-3 specialists
- complex (50-75): Multi-faceted analysis → multi_agent with coordination
- highly_complex (75-100): Comprehensive research and synthesis → full team

Focus on ELIMINATING REDUNDANCY by making all routing and planning decisions in this single analysis."""

    def _parse_coordination_response(self, response: str, thought_data: ThoughtData) -> CoordinationPlan:
        """Parse coordination response into structured plan."""

        try:
            # Extract JSON from response
            response_text = str(response).strip()

            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in coordination response")

            data = json.loads(json_text)

            # Extract and validate data
            complexity_info = data["complexity_analysis"]
            strategy_info = data["strategy_decision"]
            team_info = data["team_composition"]
            coordination_info = data["coordination_plan"]
            execution_info = data["execution_parameters"]

            # Convert to proper enums
            complexity_level = ComplexityLevel(complexity_info["complexity_level"])
            strategy = ProcessingStrategy(strategy_info["processing_strategy"])
            execution_mode = ExecutionMode(strategy_info["execution_mode"])

            # Build comprehensive reasoning
            reasoning = f"Coordination Analysis: {complexity_info['reasoning']} | Strategy: {strategy_info['reasoning']} | Team: {len(team_info['specialist_roles'])} specialists | Coordination: {coordination_info['coordination_strategy']}"

            return CoordinationPlan(
                strategy=strategy,
                complexity_level=complexity_level,
                complexity_score=float(complexity_info["complexity_score"]),
                execution_mode=execution_mode,
                specialist_roles=team_info["specialist_roles"],
                task_breakdown=coordination_info["task_breakdown"],
                coordination_strategy=coordination_info["coordination_strategy"],
                timeout_seconds=float(execution_info["timeout_seconds"]),
                expected_interactions=int(execution_info["expected_interactions"]),
                team_size=len(team_info["specialist_roles"]),  # Added team_size
                reasoning=reasoning,
                confidence=float(data["confidence"]),
                original_thought=thought_data.thought
            )

        except Exception as e:
            logger.warning(f"Failed to parse coordination response: {e}")
            return self._create_fallback_plan(thought_data)

    def _create_fallback_plan(self, thought_data: ThoughtData) -> CoordinationPlan:
        """Create fallback coordination plan when AI parsing fails."""

        # Simple heuristic-based planning
        text = thought_data.thought.lower()

        if len(text) > 100 and any(word in text for word in ["分析", "比较", "评估", "research", "analysis"]):
            strategy = ProcessingStrategy.HYBRID
            execution_mode = ExecutionMode.SELECTIVE_TEAM
            specialists = ["researcher", "analyzer", "synthesizer"]
            complexity = ComplexityLevel.MODERATE
            score = 40.0
            timeout = 120.0
        elif len(text) > 200 or any(word in text for word in ["哲学", "理论", "framework", "methodology"]):
            strategy = ProcessingStrategy.MULTI_AGENT
            execution_mode = ExecutionMode.FULL_TEAM
            specialists = ["researcher", "analyzer", "critic", "synthesizer"]
            complexity = ComplexityLevel.COMPLEX
            score = 65.0
            timeout = 180.0
        else:
            strategy = ProcessingStrategy.SINGLE_AGENT
            execution_mode = ExecutionMode.SINGLE_AGENT
            specialists = ["general"]
            complexity = ComplexityLevel.SIMPLE
            score = 25.0
            timeout = 60.0

        return CoordinationPlan(
            strategy=strategy,
            complexity_level=complexity,
            complexity_score=score,
            execution_mode=execution_mode,
            specialist_roles=specialists,
            task_breakdown=["Process the complete thought comprehensively"],
            coordination_strategy="direct" if len(specialists) == 1 else "sequential",
            timeout_seconds=timeout,
            expected_interactions=len(specialists),
            team_size=len(specialists),  # Added team_size
            reasoning="Fallback coordination analysis due to AI parsing failure",
            confidence=0.6,
            original_thought=thought_data.thought
        )

    def convert_to_routing_decision(self, plan: CoordinationPlan) -> RoutingDecision:
        """Convert coordination plan to routing decision for compatibility."""

        # Estimate token usage and cost
        token_estimates = {
            ProcessingStrategy.SINGLE_AGENT: (400, 800),
            ProcessingStrategy.HYBRID: (1200, 2400),
            ProcessingStrategy.MULTI_AGENT: (2000, 4000),
        }

        token_range = token_estimates.get(plan.strategy, (1000, 2000))
        estimated_cost = (token_range[0] + token_range[1]) / 2 * 0.0002 / 1000

        return RoutingDecision(
            strategy=plan.strategy,
            complexity_level=plan.complexity_level,
            complexity_score=plan.complexity_score,
            reasoning=plan.reasoning,
            estimated_token_usage=token_range,
            estimated_cost=estimated_cost,
            specialist_recommendations=plan.specialist_roles
        )


def create_intelligent_coordinator(model: Optional[Model] = None) -> IntelligentCoordinator:
    """Create intelligent coordinator instance."""
    return IntelligentCoordinator(model=model)