"""Unit tests for AIComplexityAnalyzer utility methods (no AI calls)."""

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.routing.ai_complexity_analyzer import (
    AIComplexityAnalyzer,
    create_ai_complexity_analyzer,
)
from mcp_server_mas_sequential_thinking.routing.complexity_types import ComplexityMetrics


def make_analyzer() -> AIComplexityAnalyzer:
    """Create analyzer with a stub config that won't make real API calls."""
    from unittest.mock import MagicMock
    model_config = MagicMock()
    model_config.create_enhanced_model.return_value = MagicMock()
    return AIComplexityAnalyzer(model_config=model_config)


def make_thought(text: str = "analyze this thought") -> ThoughtData:
    return ThoughtData(
        thought=text,
        thoughtNumber=1,
        totalThoughts=5,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )


class TestExtractResponseContent:
    """Tests for _extract_response_content."""

    def test_extracts_content_attr(self):
        analyzer = make_analyzer()
        class Obj:
            content = "extracted"
        assert analyzer._extract_response_content(Obj()) == "extracted"

    def test_falls_back_to_str(self):
        analyzer = make_analyzer()
        assert analyzer._extract_response_content(42) == "42"


class TestSanitizeThought:
    """Tests for _sanitize_thought_for_analysis."""

    def test_escapes_quotes(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_thought_for_analysis('say "hello"')
        assert '\\"hello\\"' in result

    def test_removes_curly_braces(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_thought_for_analysis("{injection}")
        assert "{" not in result
        assert "}" not in result

    def test_truncates_long_thought(self):
        analyzer = make_analyzer()
        long_text = "a" * 3000
        result = analyzer._sanitize_thought_for_analysis(long_text)
        assert len(result) <= 2003  # 2000 + "..."
        assert result.endswith("...")

    def test_short_thought_unchanged(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_thought_for_analysis("short text")
        assert "short text" in result


class TestParseAndValidateJsonResponse:
    """Tests for _parse_and_validate_json_response."""

    def test_parses_direct_json(self):
        analyzer = make_analyzer()
        import json
        data = {"complexity_score": 50.0, "primary_problem_type": "FACTUAL"}
        result = analyzer._parse_and_validate_json_response(json.dumps(data))
        assert result["complexity_score"] == 50.0

    def test_parses_json_code_block(self):
        analyzer = make_analyzer()
        response = '```json\n{"complexity_score": 30.0, "primary_problem_type": "CREATIVE"}\n```'
        result = analyzer._parse_and_validate_json_response(response)
        assert result["complexity_score"] == 30.0

    def test_parses_inline_json_on_single_line(self):
        analyzer = make_analyzer()
        import json
        data = {"complexity_score": 20.0, "primary_problem_type": "GENERAL"}
        response = f"Here is the result:\n{json.dumps(data)}"
        result = analyzer._parse_and_validate_json_response(response)
        assert result["complexity_score"] == 20.0

    def test_raises_on_unparseable_response(self):
        analyzer = make_analyzer()
        with pytest.raises(ValueError, match="Could not parse"):
            analyzer._parse_and_validate_json_response("not json at all")

    def test_raises_on_too_large_response(self):
        analyzer = make_analyzer()
        with pytest.raises(ValueError, match="too large"):
            analyzer._parse_and_validate_json_response("x" * 10001)


class TestValidateJsonStructure:
    """Tests for _validate_json_structure."""

    def test_valid_structure(self):
        analyzer = make_analyzer()
        data = {"complexity_score": 50.0, "primary_problem_type": "FACTUAL"}
        result = analyzer._validate_json_structure(data)
        assert result["complexity_score"] == 50.0

    def test_missing_complexity_score_gets_default(self):
        analyzer = make_analyzer()
        data = {"primary_problem_type": "FACTUAL"}
        result = analyzer._validate_json_structure(data)
        assert result["complexity_score"] == 50.0

    def test_missing_problem_type_gets_default(self):
        analyzer = make_analyzer()
        data = {"complexity_score": 30.0}
        result = analyzer._validate_json_structure(data)
        assert result["primary_problem_type"] == "GENERAL"

    def test_raises_on_non_dict(self):
        analyzer = make_analyzer()
        with pytest.raises(ValueError, match="JSON object"):
            analyzer._validate_json_structure([1, 2, 3])


class TestValidateNumericField:
    """Tests for _validate_numeric_field."""

    def test_valid_value_in_range(self):
        analyzer = make_analyzer()
        assert analyzer._validate_numeric_field(50.0, 0.0, 100.0, 0.0) == 50.0

    def test_clamps_below_min(self):
        analyzer = make_analyzer()
        assert analyzer._validate_numeric_field(-5.0, 0.0, 100.0, 0.0) == 0.0

    def test_clamps_above_max(self):
        analyzer = make_analyzer()
        assert analyzer._validate_numeric_field(200.0, 0.0, 100.0, 0.0) == 100.0

    def test_none_returns_default(self):
        analyzer = make_analyzer()
        assert analyzer._validate_numeric_field(None, 0.0, 100.0, 42.0) == 42.0

    def test_invalid_type_returns_default(self):
        analyzer = make_analyzer()
        assert analyzer._validate_numeric_field("bad", 0.0, 100.0, 5.0) == 5.0


class TestValidateIntField:
    """Tests for _validate_int_field."""

    def test_returns_int(self):
        analyzer = make_analyzer()
        result = analyzer._validate_int_field(3.7, 0, 10, 0)
        assert isinstance(result, int)
        assert result == 3

    def test_clamps_to_range(self):
        analyzer = make_analyzer()
        assert analyzer._validate_int_field(100, 0, 10, 0) == 10


class TestValidateProblemType:
    """Tests for _validate_problem_type."""

    def test_valid_types(self):
        analyzer = make_analyzer()
        for t in ["FACTUAL", "EMOTIONAL", "CRITICAL", "OPTIMISTIC", "CREATIVE", "SYNTHESIS", "EVALUATIVE", "PHILOSOPHICAL", "DECISION"]:
            assert analyzer._validate_problem_type(t) == t

    def test_case_insensitive(self):
        analyzer = make_analyzer()
        assert analyzer._validate_problem_type("factual") == "FACTUAL"

    def test_invalid_type_returns_general(self):
        analyzer = make_analyzer()
        assert analyzer._validate_problem_type("UNKNOWN") == "GENERAL"

    def test_non_string_returns_general(self):
        analyzer = make_analyzer()
        assert analyzer._validate_problem_type(42) == "GENERAL"


class TestValidateThinkingModes:
    """Tests for _validate_thinking_modes."""

    def test_valid_modes(self):
        analyzer = make_analyzer()
        modes = ["FACTUAL", "SYNTHESIS"]
        result = analyzer._validate_thinking_modes(modes)
        assert result == ["FACTUAL", "SYNTHESIS"]

    def test_filters_invalid_modes(self):
        analyzer = make_analyzer()
        result = analyzer._validate_thinking_modes(["FACTUAL", "INVALID", "CREATIVE"])
        assert "INVALID" not in result
        assert "FACTUAL" in result

    def test_non_list_returns_synthesis(self):
        analyzer = make_analyzer()
        result = analyzer._validate_thinking_modes("not a list")
        assert result == ["SYNTHESIS"]

    def test_empty_list_returns_synthesis(self):
        analyzer = make_analyzer()
        result = analyzer._validate_thinking_modes([])
        assert result == ["SYNTHESIS"]

    def test_all_invalid_returns_synthesis(self):
        analyzer = make_analyzer()
        result = analyzer._validate_thinking_modes(["INVALID", "BAD"])
        assert result == ["SYNTHESIS"]


class TestSanitizeReasoning:
    """Tests for _sanitize_reasoning."""

    def test_escapes_quotes(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_reasoning('say "hello"')
        assert '\\"hello\\"' in result

    def test_removes_newlines(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_reasoning("line1\nline2")
        assert "\n" not in result

    def test_truncates_long_reasoning(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_reasoning("x" * 600)
        assert len(result) <= 503
        assert result.endswith("...")

    def test_non_string_returns_default(self):
        analyzer = make_analyzer()
        result = analyzer._sanitize_reasoning(None)
        assert result == "AI analysis completed"


class TestBasicFallbackAnalysis:
    """Tests for _basic_fallback_analysis."""

    def test_returns_complexity_metrics(self):
        analyzer = make_analyzer()
        thought = make_thought("What is the meaning of life?")
        result = analyzer._basic_fallback_analysis(thought)
        assert isinstance(result, ComplexityMetrics)

    def test_philosophical_thought_gets_higher_score(self):
        analyzer = make_analyzer()
        thought = make_thought("Why do we live if we must die? What is the meaning?")
        result = analyzer._basic_fallback_analysis(thought)
        assert result.primary_problem_type == "PHILOSOPHICAL"

    def test_simple_thought_is_factual(self):
        analyzer = make_analyzer()
        thought = make_thought("The cat sat on the mat")
        result = analyzer._basic_fallback_analysis(thought)
        assert result.primary_problem_type == "GENERAL"

    def test_question_marks_increase_score(self):
        analyzer = make_analyzer()
        thought_with_q = make_thought("What? Why? How?")
        thought_plain = make_thought("the cat sat on the mat")
        r1 = analyzer._basic_fallback_analysis(thought_with_q)
        r2 = analyzer._basic_fallback_analysis(thought_plain)
        assert r1.complexity_score > r2.complexity_score

    def test_analyzer_type_is_basic_fallback(self):
        analyzer = make_analyzer()
        thought = make_thought("test")
        result = analyzer._basic_fallback_analysis(thought)
        assert result.analyzer_type == "basic_fallback"


class TestCreateAiComplexityAnalyzer:
    """Tests for factory function."""

    def test_returns_analyzer_instance(self):
        analyzer = create_ai_complexity_analyzer()
        assert isinstance(analyzer, AIComplexityAnalyzer)
