"""AI-Powered Complexity Analyzer.

Uses an AI agent to intelligently assess thought complexity, replacing the rule-based approach
with more nuanced understanding of context, semantics, and depth.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from agno.agent import Agent

from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config

from .complexity_types import ComplexityAnalyzer, ComplexityMetrics

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)


COMPLEXITY_ANALYSIS_PROMPT = """
You are an expert complexity analyzer for thought processing systems. Your task is to analyze the cognitive complexity of a given thought and return a structured assessment.

Analyze the following thought and provide complexity metrics:

**Thought to Analyze:** "{thought}"

**Instructions:**
1. Consider semantic depth, philosophical implications, conceptual complexity
2. Evaluate required cognitive resources (memory, reasoning, creativity)
3. Assess multi-dimensional thinking requirements
4. Consider cultural and linguistic nuances across different languages

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
    "primary_problem_type": "<FACTUAL|EMOTIONAL|CRITICAL|OPTIMISTIC|CREATIVE|SYNTHESIS|EVALUATIVE|PHILOSOPHICAL|DECISION>",
    "thinking_modes_needed": ["<list of required thinking modes>"],
    "reasoning": "<brief explanation of scoring and problem type analysis>"
}}
```

**Scoring Guidelines:**
- 0-10: Simple factual questions or basic statements
- 11-25: Moderate complexity, requires some analysis
- 26-50: Complex topics requiring deep thinking
- 51-75: Highly complex, multi-faceted problems
- 76-100: Extremely complex philosophical/existential questions

**Problem Type Analysis:**
- FACTUAL: Information seeking, definitions, statistics (what, when, where, who)
- EMOTIONAL: Feelings, intuition, personal experiences (feel, sense, worry)
- CRITICAL: Risk assessment, problems, disadvantages (issue, risk, wrong)
- OPTIMISTIC: Benefits, opportunities, positive aspects (good, benefit, advantage)
- CREATIVE: Innovation, alternatives, new ideas (creative, brainstorm, imagine)
- SYNTHESIS: Integration, summary, holistic view (combine, overall, strategy)
- EVALUATIVE: Comparison, assessment, judgment (compare, evaluate, best)
- PHILOSOPHICAL: Meaning, existence, values (purpose, meaning, ethics)
- DECISION: Choice making, selection, recommendations (decide, choose, should)

**Thinking Modes Needed:**
Select appropriate modes based on problem characteristics:
- FACTUAL thinking for information gathering
- EMOTIONAL thinking for intuitive insights
- CRITICAL thinking for risk analysis
- OPTIMISTIC thinking for opportunity identification
- CREATIVE thinking for innovation
- SYNTHESIS thinking for integration

**Special Considerations:**
- Philosophical questions like "Why do we live if we die?" should score 40-70+
- Short but profound questions can have high complexity
- Consider emotional and existential weight, not just length
- Multilingual philosophical concepts preserve cultural context

Analyze now:
"""


class AIComplexityAnalyzer(ComplexityAnalyzer):
    """AI-powered complexity analyzer using language models."""

    def __init__(self, model_config: Any | None = None) -> None:
        self.model_config = model_config or get_model_config()
        self._agent: Agent | None = None

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

    async def analyze(self, thought_data: "ThoughtData") -> ComplexityMetrics:
        """Analyze thought complexity using AI agent with secure prompt handling."""
        logger.info("🤖 AI COMPLEXITY ANALYSIS:")
        logger.info(f"  📝 Analyzing: {thought_data.thought[:100]}...")

        try:
            agent = self._get_agent()

            # Secure prompt construction - sanitize thought input
            sanitized_thought = self._sanitize_thought_for_analysis(thought_data.thought)
            prompt = COMPLEXITY_ANALYSIS_PROMPT.format(thought=sanitized_thought)

            # Get AI analysis with timeout
            result = await agent.arun(input=prompt)

            # Extract and validate JSON response
            response_text = self._extract_response_content(result)
            complexity_data = self._parse_and_validate_json_response(response_text)

            # Create metrics object with validated AI assessment
            metrics = ComplexityMetrics(
                complexity_score=self._validate_numeric_field(complexity_data.get("complexity_score"), 0.0, 100.0, 0.0),
                word_count=self._validate_numeric_field(complexity_data.get("word_count"), 0, 10000, 0),
                sentence_count=self._validate_numeric_field(complexity_data.get("sentence_count"), 0, 1000, 0),
                question_count=self._validate_numeric_field(complexity_data.get("question_count"), 0, 100, 0),
                technical_terms=self._validate_numeric_field(complexity_data.get("technical_terms"), 0, 100, 0),
                branching_references=self._validate_numeric_field(complexity_data.get("branching_references"), 0, 50, 0),
                research_indicators=self._validate_numeric_field(complexity_data.get("research_indicators"), 0, 50, 0),
                analysis_depth=self._validate_numeric_field(complexity_data.get("analysis_depth"), 0, 100, 0),
                philosophical_depth_boost=self._validate_numeric_field(
                    complexity_data.get("philosophical_depth_boost"), 0, 15, 0
                ),
                # AI Analysis Results (critical for routing) - validated
                primary_problem_type=self._validate_problem_type(
                    complexity_data.get("primary_problem_type", "GENERAL")
                ),
                thinking_modes_needed=self._validate_thinking_modes(
                    complexity_data.get("thinking_modes_needed", ["SYNTHESIS"])
                ),
                analyzer_type="ai",
                reasoning=self._sanitize_reasoning(complexity_data.get("reasoning", "AI analysis")),
            )

            logger.info(f"  🎯 AI Complexity Score: {metrics.complexity_score:.1f}/100")
            logger.info(
                f"  💭 Reasoning: {metrics.reasoning[:100]}..."
            )

            return metrics

        except Exception as e:
            logger.exception(f"❌ AI complexity analysis failed: {e}")
            # Fallback to basic analysis
            return self._basic_fallback_analysis(thought_data)

    def _extract_response_content(self, result: Any) -> str:
        """Extract content from agent response."""
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    def _sanitize_thought_for_analysis(self, thought: str) -> str:
        """Sanitize thought text for secure inclusion in AI prompts."""
        import re

        # Remove any potential prompt injection patterns
        sanitized = thought.replace('"', '\\"')  # Escape quotes
        sanitized = re.sub(r"[{}]", "", sanitized)  # Remove curly braces
        sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)  # Remove control chars

        # Limit length to prevent token exhaustion
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "..."

        return sanitized

    def _parse_and_validate_json_response(self, response_text: str) -> dict[str, Any]:
        """Parse and validate JSON from AI response with strict security checks."""
        # Size limit check
        if len(response_text) > 10000:  # Reasonable limit for complexity analysis
            raise ValueError("AI response too large, possible attack")

        # Try to find JSON in the response
        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    return self._validate_json_structure(data)
                except json.JSONDecodeError:
                    continue

        # Try to extract JSON from code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end > start:
                json_text = response_text[start:end].strip()
                try:
                    data = json.loads(json_text)
                    return self._validate_json_structure(data)
                except json.JSONDecodeError:
                    pass

        # Try parsing the entire response as JSON
        try:
            data = json.loads(response_text)
            return self._validate_json_structure(data)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse AI response as JSON: {response_text[:200]}"
            )
            raise ValueError("Could not parse AI complexity analysis response")

    def _validate_json_structure(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate JSON structure and sanitize content."""
        if not isinstance(data, dict):
            raise ValueError("AI response must be a JSON object")

        # Required fields validation
        required_fields = ["complexity_score", "primary_problem_type"]
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                # Set default values for missing fields
                if field == "complexity_score":
                    data[field] = 50.0
                elif field == "primary_problem_type":
                    data[field] = "GENERAL"

        return data

    def _validate_numeric_field(self, value: Any, min_val: float, max_val: float, default: float) -> float:
        """Validate and clamp numeric fields to safe ranges."""
        try:
            if value is None:
                return default
            num_val = float(value)
            return max(min_val, min(max_val, num_val))
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value: {value}, using default: {default}")
            return default

    def _validate_problem_type(self, problem_type: str) -> str:
        """Validate problem type against allowed values."""
        allowed_types = {
            "FACTUAL", "EMOTIONAL", "CRITICAL", "OPTIMISTIC",
            "CREATIVE", "SYNTHESIS", "EVALUATIVE", "PHILOSOPHICAL",
            "DECISION", "GENERAL"
        }

        if not isinstance(problem_type, str):
            return "GENERAL"

        clean_type = problem_type.upper().strip()
        return clean_type if clean_type in allowed_types else "GENERAL"

    def _validate_thinking_modes(self, modes: Any) -> list[str]:
        """Validate thinking modes list."""
        allowed_modes = {
            "FACTUAL", "EMOTIONAL", "CRITICAL", "OPTIMISTIC",
            "CREATIVE", "SYNTHESIS"
        }

        if not isinstance(modes, list):
            return ["SYNTHESIS"]

        validated_modes = []
        for mode in modes:
            if isinstance(mode, str):
                clean_mode = mode.upper().strip()
                if clean_mode in allowed_modes:
                    validated_modes.append(clean_mode)

        return validated_modes if validated_modes else ["SYNTHESIS"]

    def _sanitize_reasoning(self, reasoning: str) -> str:
        """Sanitize reasoning text to prevent injection."""
        if not isinstance(reasoning, str):
            return "AI analysis completed"

        # Remove potential injection patterns and limit length
        sanitized = reasoning.replace('"', '\\"')
        sanitized = sanitized.replace("\n", " ")
        sanitized = sanitized.replace("\t", " ")

        # Limit length
        if len(sanitized) > 500:
            sanitized = sanitized[:500] + "..."

        return sanitized

    def _basic_fallback_analysis(
        self, thought_data: "ThoughtData"
    ) -> ComplexityMetrics:
        """Fallback to basic analysis if AI fails."""
        logger.warning("🔄 Falling back to basic complexity analysis")

        text = thought_data.thought.lower()

        # Basic metrics
        words = len(text.split())
        sentences = len([s for s in text.split(".") if s.strip()])
        questions = text.count("?") + text.count("？")

        # Simple heuristics
        philosophical_terms = [
            "意义",
            "存在",
            "生命",
            "死亡",
            "为什么",
            "why",
            "meaning",
            "life",
            "death",
        ]
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
            # Basic AI analysis results for fallback
            primary_problem_type="PHILOSOPHICAL"
            if philosophical_count > 0
            else "GENERAL",
            thinking_modes_needed=["SYNTHESIS", "CREATIVE"]
            if philosophical_count > 2
            else ["FACTUAL"],
            analyzer_type="basic_fallback",
            reasoning="Fallback analysis due to AI failure",
        )


# No more monkey patching needed - complexity_score is now a direct field


def create_ai_complexity_analyzer() -> AIComplexityAnalyzer:
    """Create AI complexity analyzer instance."""
    return AIComplexityAnalyzer()
