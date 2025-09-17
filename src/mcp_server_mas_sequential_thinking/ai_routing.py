"""AI-powered adaptive routing system for intelligent complexity analysis."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from agno.agent import Agent
from agno.models.base import Model

from .models import ThoughtData
from .adaptive_routing import ComplexityLevel, ProcessingStrategy, RoutingDecision

# Re-export types for convenience
__all__ = [
    'ComplexityLevel',
    'ProcessingStrategy',
    'RoutingDecision',
    'AIComplexityAnalyzer',
    'HybridComplexityAnalyzer',
    'create_ai_router'
]
from .modernized_config import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class AIRoutingAnalysis:
    """AI analysis result for routing decisions."""

    complexity_level: ComplexityLevel
    complexity_score: float
    reasoning: str
    requires_research: bool
    requires_deep_analysis: bool
    requires_synthesis: bool
    requires_critique: bool
    suggested_strategy: ProcessingStrategy
    confidence: float  # 0.0 - 1.0


class AIComplexityAnalyzer:
    """AI-powered complexity analyzer using LLM for intelligent routing decisions."""

    def __init__(self, model: Optional[Model] = None):
        """Initialize AI analyzer with model."""
        self.model = model or self._get_fast_model()
        self.routing_agent = self._create_routing_agent()

    def _get_fast_model(self) -> Model:
        """Get a fast, lightweight model for routing decisions."""
        config = get_model_config()
        # Use agent model (typically faster) for routing decisions
        return config.provider_class(id=config.agent_model_id)

    def _create_routing_agent(self) -> Agent:
        """Create specialized routing decision agent."""
        return Agent(
            name="RoutingAnalyzer",
            role="Intelligent Routing Decision Maker",
            description="Analyzes thought complexity and determines optimal processing strategy",
            model=self.model,
            instructions=[
                "You are an expert at analyzing the complexity of thoughts and determining the best processing strategy.",
                "Your task is to evaluate thoughts and decide between single-agent, hybrid, or multi-agent processing.",
                "Consider semantic complexity, required expertise, research needs, and analytical depth.",
                "Provide structured analysis with confidence scores and clear reasoning.",
                "Focus on accuracy and efficiency - avoid over-complicating simple thoughts or under-resourcing complex ones."
            ],
            structured_outputs=True,
            markdown=False
        )

    async def analyze_async(self, thought_data: ThoughtData) -> AIRoutingAnalysis:
        """Analyze thought complexity using AI with async processing."""

        # Prepare analysis prompt
        analysis_prompt = self._build_analysis_prompt(thought_data)

        try:
            # Get AI analysis
            response = await self.routing_agent.arun(analysis_prompt)

            # Parse response
            analysis = self._parse_ai_response(response, thought_data)

            logger.info(
                f"AI routing analysis: {analysis.complexity_level.value} "
                f"(score: {analysis.complexity_score:.1f}, "
                f"strategy: {analysis.suggested_strategy.value}, "
                f"confidence: {analysis.confidence:.2f})"
            )

            return analysis

        except Exception as e:
            logger.warning(f"AI routing analysis failed: {e}, falling back to rule-based")
            # Fallback to basic analysis
            return self._fallback_analysis(thought_data)

    def analyze(self, thought_data: ThoughtData) -> AIRoutingAnalysis:
        """Synchronous wrapper for analyze_async."""
        import asyncio
        try:
            return asyncio.run(self.analyze_async(thought_data))
        except Exception as e:
            logger.warning(f"Sync AI routing analysis failed: {e}, falling back")
            return self._fallback_analysis(thought_data)

    def _build_analysis_prompt(self, thought_data: ThoughtData) -> str:
        """Build analysis prompt for the routing agent."""

        context_info = ""
        if thought_data.thought_number > 1:
            context_info = f"This is thought #{thought_data.thought_number} in a {thought_data.total_thoughts}-thought sequence. "

        if thought_data.branch_from:
            context_info += f"This is a branch from thought #{thought_data.branch_from}. "

        return f"""Analyze this thought for complexity and determine the optimal processing strategy:

{context_info}

THOUGHT TO ANALYZE:
"{thought_data.thought}"

Please provide your analysis in this JSON format:
{{
    "complexity_level": "simple|moderate|complex|highly_complex",
    "complexity_score": <number 0-100>,
    "reasoning": "<detailed reasoning for your assessment>",
    "requires_research": <true/false>,
    "requires_deep_analysis": <true/false>,
    "requires_synthesis": <true/false>,
    "requires_critique": <true/false>,
    "suggested_strategy": "single_agent|hybrid|multi_agent",
    "confidence": <0.0-1.0>
}}

ANALYSIS CRITERIA:
- SIMPLE (0-25): Basic questions, straightforward tasks, no specialized expertise needed
- MODERATE (25-50): Some complexity, may benefit from specialist input, moderate analysis depth
- COMPLEX (50-75): Deep analysis required, multiple perspectives needed, significant expertise required
- HIGHLY_COMPLEX (75-100): Multi-faceted problems, extensive research, high-level synthesis required

STRATEGY GUIDELINES:
- single_agent: For simple, straightforward thoughts that don't require specialist collaboration
- hybrid: For moderate complexity where 1-2 specialists can enhance quality
- multi_agent: For complex thoughts requiring full team collaboration with multiple specialists

Consider factors like:
- Semantic complexity and depth of analysis required
- Need for research, comparison, or evaluation
- Requirement for creative synthesis or critical assessment
- Technical or specialized knowledge requirements
- Philosophical, theoretical, or abstract concepts
- Multi-perspective analysis needs"""

    def _parse_ai_response(self, response: str, thought_data: ThoughtData) -> AIRoutingAnalysis:
        """Parse AI response into structured analysis."""

        try:
            # Try to extract JSON from response
            response_text = str(response).strip()

            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            # Parse JSON
            data = json.loads(json_text)

            # Validate and convert
            complexity_level = ComplexityLevel(data["complexity_level"])
            strategy = ProcessingStrategy(data["suggested_strategy"])

            return AIRoutingAnalysis(
                complexity_level=complexity_level,
                complexity_score=float(data["complexity_score"]),
                reasoning=str(data["reasoning"]),
                requires_research=bool(data["requires_research"]),
                requires_deep_analysis=bool(data["requires_deep_analysis"]),
                requires_synthesis=bool(data["requires_synthesis"]),
                requires_critique=bool(data["requires_critique"]),
                suggested_strategy=strategy,
                confidence=float(data["confidence"])
            )

        except Exception as e:
            logger.warning(f"Failed to parse AI response: {e}")
            # Fallback parsing - extract what we can
            return self._fallback_analysis(thought_data)

    def _fallback_analysis(self, thought_data: ThoughtData) -> AIRoutingAnalysis:
        """Fallback analysis when AI fails."""
        # Simple heuristic-based fallback
        text = thought_data.thought.lower()

        # Basic complexity assessment
        if len(text) > 100 and any(word in text for word in ["分析", "比较", "评估", "research", "analysis"]):
            complexity_level = ComplexityLevel.MODERATE
            strategy = ProcessingStrategy.HYBRID
            score = 35.0
        elif len(text) > 200 or any(word in text for word in ["哲学", "理论", "framework", "methodology"]):
            complexity_level = ComplexityLevel.COMPLEX
            strategy = ProcessingStrategy.MULTI_AGENT
            score = 60.0
        else:
            complexity_level = ComplexityLevel.SIMPLE
            strategy = ProcessingStrategy.SINGLE_AGENT
            score = 20.0

        return AIRoutingAnalysis(
            complexity_level=complexity_level,
            complexity_score=score,
            reasoning="Fallback heuristic analysis due to AI failure",
            requires_research=False,
            requires_deep_analysis=False,
            requires_synthesis=False,
            requires_critique=False,
            suggested_strategy=strategy,
            confidence=0.5
        )


class HybridComplexityAnalyzer:
    """Hybrid analyzer combining AI and rule-based approaches for optimal accuracy."""

    def __init__(self, use_ai: bool = True, ai_confidence_threshold: float = 0.7):
        """Initialize hybrid analyzer.

        Args:
            use_ai: Whether to use AI analysis
            ai_confidence_threshold: Minimum confidence to trust AI results
        """
        self.use_ai = use_ai
        self.ai_confidence_threshold = ai_confidence_threshold

        if use_ai:
            try:
                self.ai_analyzer = AIComplexityAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize AI analyzer: {e}, using rule-based only")
                self.use_ai = False

        # Fallback to rule-based analyzer
        from .adaptive_routing import BasicComplexityAnalyzer
        self.rule_analyzer = BasicComplexityAnalyzer()

    def analyze(self, thought_data: ThoughtData) -> RoutingDecision:
        """Analyze using hybrid approach with AI + rule-based fallback."""

        ai_analysis = None

        # Try AI analysis first
        if self.use_ai:
            try:
                ai_analysis = self.ai_analyzer.analyze(thought_data)

                # Use AI result if confidence is high enough
                if ai_analysis.confidence >= self.ai_confidence_threshold:
                    return self._convert_ai_to_routing_decision(ai_analysis, thought_data)
                else:
                    logger.info(f"AI confidence too low ({ai_analysis.confidence:.2f}), using hybrid approach")

            except Exception as e:
                logger.warning(f"AI analysis failed: {e}, falling back to rule-based")

        # Use rule-based analysis
        rule_metrics = self.rule_analyzer.analyze(thought_data)

        # If we have AI analysis with low confidence, combine insights
        if ai_analysis and ai_analysis.confidence < self.ai_confidence_threshold:
            return self._combine_analyses(ai_analysis, rule_metrics, thought_data)
        else:
            # Pure rule-based decision
            from .adaptive_routing import AdaptiveRouter
            router = AdaptiveRouter()
            return router.route_thought(thought_data)

    def _convert_ai_to_routing_decision(self, ai_analysis: AIRoutingAnalysis, thought_data: ThoughtData) -> RoutingDecision:
        """Convert AI analysis to RoutingDecision format."""

        # Estimate token usage based on strategy and complexity
        token_estimates = {
            (ProcessingStrategy.SINGLE_AGENT, ComplexityLevel.SIMPLE): (400, 800),
            (ProcessingStrategy.SINGLE_AGENT, ComplexityLevel.MODERATE): (600, 1200),
            (ProcessingStrategy.HYBRID, ComplexityLevel.MODERATE): (1200, 2400),
            (ProcessingStrategy.HYBRID, ComplexityLevel.COMPLEX): (2000, 4000),
            (ProcessingStrategy.MULTI_AGENT, ComplexityLevel.COMPLEX): (4000, 8000),
            (ProcessingStrategy.MULTI_AGENT, ComplexityLevel.HIGHLY_COMPLEX): (5000, 10000),
        }

        token_range = token_estimates.get(
            (ai_analysis.suggested_strategy, ai_analysis.complexity_level),
            (2000, 4000)
        )

        # Estimate cost (approximate)
        estimated_cost = (token_range[0] + token_range[1]) / 2 * 0.0002 / 1000

        # Generate specialist recommendations based on AI analysis
        specialists = []
        if ai_analysis.requires_research:
            specialists.append("researcher")
        if ai_analysis.requires_deep_analysis:
            specialists.append("analyzer")
        if ai_analysis.requires_synthesis:
            specialists.append("synthesizer")
        if ai_analysis.requires_critique:
            specialists.append("critic")

        if not specialists and ai_analysis.suggested_strategy != ProcessingStrategy.SINGLE_AGENT:
            specialists = ["planner", "synthesizer"]  # Default
        elif ai_analysis.suggested_strategy == ProcessingStrategy.SINGLE_AGENT:
            specialists = ["general"]

        return RoutingDecision(
            strategy=ai_analysis.suggested_strategy,
            complexity_level=ai_analysis.complexity_level,
            complexity_score=ai_analysis.complexity_score,
            reasoning=f"AI Analysis (confidence: {ai_analysis.confidence:.2f}): {ai_analysis.reasoning}",
            estimated_token_usage=token_range,
            estimated_cost=estimated_cost,
            specialist_recommendations=specialists
        )

    def _combine_analyses(self, ai_analysis: AIRoutingAnalysis, rule_metrics, thought_data: ThoughtData) -> RoutingDecision:
        """Combine AI and rule-based analyses for better accuracy."""

        # Weight AI and rule-based scores
        ai_weight = ai_analysis.confidence
        rule_weight = 1.0 - ai_weight

        combined_score = (ai_analysis.complexity_score * ai_weight +
                         rule_metrics.complexity_score * rule_weight)

        # Determine final complexity level
        if combined_score < 25:
            final_complexity = ComplexityLevel.SIMPLE
        elif combined_score < 50:
            final_complexity = ComplexityLevel.MODERATE
        elif combined_score < 75:
            final_complexity = ComplexityLevel.COMPLEX
        else:
            final_complexity = ComplexityLevel.HIGHLY_COMPLEX

        # Choose strategy based on combined analysis
        if final_complexity == ComplexityLevel.SIMPLE:
            strategy = ProcessingStrategy.SINGLE_AGENT
        elif final_complexity == ComplexityLevel.MODERATE:
            strategy = ProcessingStrategy.HYBRID
        else:
            strategy = ProcessingStrategy.MULTI_AGENT

        reasoning = f"Hybrid Analysis - AI: {ai_analysis.complexity_score:.1f} (conf: {ai_analysis.confidence:.2f}), Rule: {rule_metrics.complexity_score:.1f}, Combined: {combined_score:.1f}"

        return RoutingDecision(
            strategy=strategy,
            complexity_level=final_complexity,
            complexity_score=combined_score,
            reasoning=reasoning,
            estimated_token_usage=(1000, 3000),  # Estimate
            estimated_cost=0.0004,  # Estimate
            specialist_recommendations=["planner", "analyzer"] if strategy != ProcessingStrategy.SINGLE_AGENT else ["general"]
        )


# Convenience function for easy integration
def create_ai_router(use_ai: bool = True, ai_confidence_threshold: float = 0.7) -> HybridComplexityAnalyzer:
    """Create AI-powered hybrid router."""
    return HybridComplexityAnalyzer(use_ai=use_ai, ai_confidence_threshold=ai_confidence_threshold)