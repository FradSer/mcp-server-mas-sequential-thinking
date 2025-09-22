"""AI-Powered Complexity Analyzer.

Uses an AI agent to intelligently assess thought complexity, replacing the rule-based approach
with more nuanced understanding of context, semantics, and depth.
"""

import json
import logging
from typing import Any

from agno.agent import Agent

from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config
from mcp_server_mas_sequential_thinking.core.models import ThoughtData

from .complexity_types import ComplexityAnalyzer, ComplexityMetrics

logger = logging.getLogger(__name__)


COMPLEXITY_ANALYSIS_PROMPT = """
You are an expert complexity analyzer for thought processing systems. Your task is to analyze the cognitive complexity of a given thought and return a structured assessment.

Analyze the following thought and provide complexity metrics:

**Thought to Analyze:** "{thought}"

**Instructions:**
1. Consider semantic depth, philosophical implications, conceptual complexity
2. Evaluate required cognitive resources (memory, reasoning, creativity)
3. Assess multi-dimensional thinking requirements
4. Consider cultural and linguistic nuances (especially for Chinese content)

**Response Format:** Return ONLY a valid JSON object with these exact fields:
```json
{{
    "complexity_score": <float 0-100>,
    "word_count": <int>,
    "sentence_count": <int>,
    "question_count": <int>,
    "technical_terms": <int>,
    "branching_references": <int>,
    "research_indicators": <int>,
    "analysis_depth": <int>,
    "philosophical_depth_boost": <int 0-15>,
    "reasoning": "<brief explanation of scoring>"
}}
```

**Scoring Guidelines:**
- 0-10: Simple factual questions or basic statements
- 11-25: Moderate complexity, requires some analysis
- 26-50: Complex topics requiring deep thinking
- 51-75: Highly complex, multi-faceted problems
- 76-100: Extremely complex philosophical/existential questions

**Special Considerations:**
- Philosophical questions like "Why do we live if we die?" should score 40-70+
- Short but profound questions can have high complexity
- Consider emotional and existential weight, not just length
- Chinese philosophical concepts carry cultural depth

Analyze now:
"""


class AIComplexityAnalyzer(ComplexityAnalyzer):
    """AI-powered complexity analyzer using language models."""

    def __init__(self, model_config=None):
        self.model_config = model_config or get_model_config()
        self._agent = None

    def _get_agent(self) -> Agent:
        """Lazy initialization of the analysis agent."""
        if self._agent is None:
            model = self.model_config.create_agent_model()
            self._agent = Agent(
                name="ComplexityAnalyzer",
                model=model,
                introduction="You are an expert in cognitive complexity assessment, specializing in philosophy and deep thinking analysis.",
            )
        return self._agent

    async def analyze(self, thought_data: ThoughtData) -> ComplexityMetrics:
        """Analyze thought complexity using AI agent."""
        logger.info("🤖 AI COMPLEXITY ANALYSIS:")
        logger.info(f"  📝 Analyzing: {thought_data.thought[:100]}...")

        try:
            agent = self._get_agent()
            prompt = COMPLEXITY_ANALYSIS_PROMPT.format(thought=thought_data.thought)

            # Get AI analysis
            result = await agent.arun(input=prompt)

            # Extract JSON response
            response_text = self._extract_response_content(result)
            complexity_data = self._parse_json_response(response_text)

            # Create metrics object with AI assessment
            metrics = ComplexityMetrics(
                complexity_score=complexity_data.get("complexity_score", 0.0),
                word_count=complexity_data.get("word_count", 0),
                sentence_count=complexity_data.get("sentence_count", 0),
                question_count=complexity_data.get("question_count", 0),
                technical_terms=complexity_data.get("technical_terms", 0),
                branching_references=complexity_data.get("branching_references", 0),
                research_indicators=complexity_data.get("research_indicators", 0),
                analysis_depth=complexity_data.get("analysis_depth", 0),
                philosophical_depth_boost=complexity_data.get("philosophical_depth_boost", 0),
                analyzer_type="ai",
                reasoning=complexity_data.get("reasoning", "AI analysis")
            )

            logger.info(f"  🎯 AI Complexity Score: {metrics.complexity_score:.1f}/100")
            logger.info(f"  💭 Reasoning: {complexity_data.get('reasoning', 'No reasoning provided')[:100]}...")

            return metrics

        except Exception as e:
            logger.error(f"❌ AI complexity analysis failed: {e}")
            # Fallback to basic analysis
            return self._basic_fallback_analysis(thought_data)

    def _extract_response_content(self, result: Any) -> str:
        """Extract content from agent response."""
        if hasattr(result, 'content'):
            return str(result.content)
        return str(result)

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from AI response, handling various formats."""
        # Try to find JSON in the response
        lines = response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        # Try to extract JSON from code blocks
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.find('```', start)
            if end > start:
                json_text = response_text[start:end].strip()
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    pass

        # Try parsing the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse AI response as JSON: {response_text[:200]}")
            raise ValueError("Could not parse AI complexity analysis response")

    def _basic_fallback_analysis(self, thought_data: ThoughtData) -> ComplexityMetrics:
        """Fallback to basic analysis if AI fails."""
        logger.warning("🔄 Falling back to basic complexity analysis")

        text = thought_data.thought.lower()

        # Basic metrics
        words = len(text.split())
        sentences = len([s for s in text.split('.') if s.strip()])
        questions = text.count('?') + text.count('？')

        # Simple heuristics
        philosophical_terms = ['意义', '存在', '生命', '死亡', '为什么', 'why', 'meaning', 'life', 'death']
        philosophical_count = sum(1 for term in philosophical_terms if term in text)

        # Basic scoring
        base_score = min(words * 2 + questions * 5 + philosophical_count * 10, 100)

        return ComplexityMetrics(
            complexity_score=base_score,
            word_count=words,
            sentence_count=max(sentences, 1),
            question_count=questions,
            technical_terms=philosophical_count,
            branching_references=0,
            research_indicators=0,
            analysis_depth=philosophical_count,
            philosophical_depth_boost=min(philosophical_count * 5, 15),
            analyzer_type="basic_fallback",
            reasoning="Fallback analysis due to AI failure"
        )


# No more monkey patching needed - complexity_score is now a direct field


def create_ai_complexity_analyzer() -> AIComplexityAnalyzer:
    """Create AI complexity analyzer instance."""
    return AIComplexityAnalyzer()