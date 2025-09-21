"""TDD tests for ComplexityMetrics.complexity_score property calculation."""

import pytest
from src.mcp_server_mas_sequential_thinking.adaptive_routing import ComplexityMetrics


class TestComplexityMetricsScoreCalculation:
    """Test ComplexityMetrics.complexity_score property with weighted scoring."""

    def test_complexity_score_minimal_content(self):
        """RED: Test complexity score for minimal content (should be near 0)."""
        metrics = ComplexityMetrics(
            word_count=5,        # 5/20 = 0.25 points
            sentence_count=1,    # 1*2 = 2 points
            question_count=0,    # 0*3 = 0 points
            technical_terms=0,   # 0*2 = 0 points
            branching_references=0,  # 0*5 = 0 points
            research_indicators=0,   # 0*3 = 0 points
            analysis_depth=0     # 0*2 = 0 points
        )
        # Expected: 0.25 + 2 + 0 + 0 + 0 + 0 + 0 = 2.25
        assert metrics.complexity_score == pytest.approx(2.25, rel=1e-2)

    def test_complexity_score_simple_content(self):
        """RED: Test complexity score for simple content (should be under 10)."""
        metrics = ComplexityMetrics(
            word_count=40,       # min(40/20, 15) = 2 points
            sentence_count=3,    # min(3*2, 10) = 6 points
            question_count=1,    # min(1*3, 15) = 3 points
            technical_terms=0,   # min(0*2, 20) = 0 points
            branching_references=0,  # min(0*5, 15) = 0 points
            research_indicators=0,   # min(0*3, 15) = 0 points
            analysis_depth=1     # min(1*2, 10) = 2 points
        )
        # Expected: 2 + 6 + 3 + 0 + 0 + 0 + 2 = 13
        assert metrics.complexity_score == pytest.approx(13.0, rel=1e-2)

    def test_complexity_score_moderate_content(self):
        """RED: Test complexity score for moderate content (should be 15-30)."""
        metrics = ComplexityMetrics(
            word_count=100,      # min(100/20, 15) = 5 points
            sentence_count=5,    # min(5*2, 10) = 10 points
            question_count=2,    # min(2*3, 15) = 6 points
            technical_terms=3,   # min(3*2, 20) = 6 points
            branching_references=1,  # min(1*5, 15) = 5 points
            research_indicators=2,   # min(2*3, 15) = 6 points
            analysis_depth=2     # min(2*2, 10) = 4 points
        )
        # Expected: 5 + 10 + 6 + 6 + 5 + 6 + 4 = 42
        assert metrics.complexity_score == pytest.approx(42.0, rel=1e-2)

    def test_complexity_score_high_content(self):
        """RED: Test complexity score for high complexity content."""
        metrics = ComplexityMetrics(
            word_count=400,      # min(400/20, 15) = 15 points (capped)
            sentence_count=8,    # min(8*2, 10) = 10 points (capped)
            question_count=6,    # min(6*3, 15) = 15 points (capped)
            technical_terms=12,  # min(12*2, 20) = 20 points (capped)
            branching_references=4,  # min(4*5, 15) = 15 points (capped)
            research_indicators=6,   # min(6*3, 15) = 15 points (capped)
            analysis_depth=8     # min(8*2, 10) = 10 points (capped)
        )
        # Expected: 15 + 10 + 15 + 20 + 15 + 15 + 10 = 100
        assert metrics.complexity_score == pytest.approx(100.0, rel=1e-2)

    def test_complexity_score_capping_individual_components(self):
        """RED: Test that individual components are properly capped."""
        metrics = ComplexityMetrics(
            word_count=1000,     # Should cap at 15 points
            sentence_count=20,   # Should cap at 10 points
            question_count=10,   # Should cap at 15 points
            technical_terms=50,  # Should cap at 20 points
            branching_references=10,  # Should cap at 15 points
            research_indicators=20,   # Should cap at 15 points
            analysis_depth=20    # Should cap at 10 points
        )
        # All components at max: 15+10+15+20+15+15+10 = 100
        assert metrics.complexity_score == 100.0

    def test_complexity_score_maximum_bounds(self):
        """RED: Test that complexity score never exceeds 100."""
        # Even with extreme values, score should cap at 100
        metrics = ComplexityMetrics(
            word_count=99999,
            sentence_count=99999,
            question_count=99999,
            technical_terms=99999,
            branching_references=99999,
            research_indicators=99999,
            analysis_depth=99999
        )
        assert metrics.complexity_score == 100.0

    def test_complexity_score_zero_values(self):
        """RED: Test complexity score with all zero values."""
        metrics = ComplexityMetrics(
            word_count=0,
            sentence_count=0,
            question_count=0,
            technical_terms=0,
            branching_references=0,
            research_indicators=0,
            analysis_depth=0
        )
        assert metrics.complexity_score == 0.0

    def test_complexity_score_branching_weight(self):
        """RED: Test that branching references have high weight (5x multiplier)."""
        metrics_with_branching = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=0, branching_references=2,  # 2*5 = 10 points
            research_indicators=0, analysis_depth=0
        )

        metrics_without_branching = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=0, branching_references=0,
            research_indicators=0, analysis_depth=0
        )

        # Difference should be exactly 10 points (2 * 5)
        score_diff = metrics_with_branching.complexity_score - metrics_without_branching.complexity_score
        assert score_diff == pytest.approx(10.0, rel=1e-2)

    def test_complexity_score_technical_terms_weight(self):
        """RED: Test that technical terms have significant weight (2x multiplier)."""
        metrics_with_terms = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=5,  # 5*2 = 10 points
            branching_references=0, research_indicators=0, analysis_depth=0
        )

        metrics_without_terms = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=0,
            branching_references=0, research_indicators=0, analysis_depth=0
        )

        # Difference should be exactly 10 points (5 * 2)
        score_diff = metrics_with_terms.complexity_score - metrics_without_terms.complexity_score
        assert score_diff == pytest.approx(10.0, rel=1e-2)

    def test_complexity_score_questions_weight(self):
        """RED: Test that questions have high weight (3x multiplier)."""
        metrics_with_questions = ComplexityMetrics(
            word_count=20, sentence_count=1,
            question_count=3,  # 3*3 = 9 points
            technical_terms=0, branching_references=0,
            research_indicators=0, analysis_depth=0
        )

        metrics_without_questions = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=0, branching_references=0,
            research_indicators=0, analysis_depth=0
        )

        # Difference should be exactly 9 points (3 * 3)
        score_diff = metrics_with_questions.complexity_score - metrics_without_questions.complexity_score
        assert score_diff == pytest.approx(9.0, rel=1e-2)

    def test_complexity_score_realistic_thresholds(self):
        """RED: Test that scores align with routing decision thresholds."""
        # Simple content (should route to single_agent < 5)
        simple_metrics = ComplexityMetrics(
            word_count=20, sentence_count=1, question_count=0,
            technical_terms=0, branching_references=0,
            research_indicators=0, analysis_depth=0
        )
        assert simple_metrics.complexity_score < 5  # 1 + 2 + 0 + 0 + 0 + 0 + 0 = 3

        # Moderate content (should route to hybrid 5-15)
        moderate_metrics = ComplexityMetrics(
            word_count=40, sentence_count=2, question_count=1,
            technical_terms=1, branching_references=0,
            research_indicators=1, analysis_depth=0
        )
        assert 5 <= moderate_metrics.complexity_score < 15  # 2 + 4 + 3 + 2 + 0 + 3 + 0 = 14

        # Complex content (should route to multi_agent >= 15)
        complex_metrics = ComplexityMetrics(
            word_count=120, sentence_count=5, question_count=2,
            technical_terms=3, branching_references=1,
            research_indicators=2, analysis_depth=2
        )
        assert complex_metrics.complexity_score >= 15  # 6 + 10 + 6 + 6 + 5 + 6 + 4 = 43