"""TDD tests for constants module - validating magic number extraction."""

from mcp_server_mas_sequential_thinking.constants import (
    ComplexityScoring,
    DefaultSettings,
    ProcessingStrategy,
    TokenCosts,
    TokenEstimates,
    ValidationLimits,
)


class TestTokenCosts:
    """Test token cost constants for different providers."""

    def test_deepseek_cost_per_1k_tokens(self):
        """RED: Test DeepSeek cost constant exists and has expected value."""
        # This defines the expected behavior from refactoring
        assert TokenCosts.DEEPSEEK_COST_PER_1K == 0.0002
        assert isinstance(TokenCosts.DEEPSEEK_COST_PER_1K, float)

    def test_groq_cost_per_1k_tokens(self):
        """RED: Test Groq cost constant exists and has expected value."""
        assert TokenCosts.GROQ_COST_PER_1K == 0.0001
        assert isinstance(TokenCosts.GROQ_COST_PER_1K, float)

    def test_openrouter_cost_per_1k_tokens(self):
        """RED: Test OpenRouter cost constant exists and has expected value."""
        assert TokenCosts.OPENROUTER_COST_PER_1K == 0.001
        assert isinstance(TokenCosts.OPENROUTER_COST_PER_1K, float)

    def test_github_cost_per_1k_tokens(self):
        """RED: Test GitHub cost constant exists and has expected value."""
        assert TokenCosts.GITHUB_COST_PER_1K == 0.0005
        assert isinstance(TokenCosts.GITHUB_COST_PER_1K, float)

    def test_ollama_cost_per_1k_tokens(self):
        """RED: Test Ollama cost constant (free) exists and has expected value."""
        assert TokenCosts.OLLAMA_COST_PER_1K == 0.0000
        assert isinstance(TokenCosts.OLLAMA_COST_PER_1K, float)

    def test_default_cost_per_1k_tokens(self):
        """RED: Test default cost constant exists and has expected value."""
        assert TokenCosts.DEFAULT_COST_PER_1K == 0.0002
        assert isinstance(TokenCosts.DEFAULT_COST_PER_1K, float)

    def test_all_costs_are_positive_or_zero(self):
        """RED: Test all cost constants are non-negative."""
        costs = [
            TokenCosts.DEEPSEEK_COST_PER_1K,
            TokenCosts.GROQ_COST_PER_1K,
            TokenCosts.OPENROUTER_COST_PER_1K,
            TokenCosts.GITHUB_COST_PER_1K,
            TokenCosts.OLLAMA_COST_PER_1K,
            TokenCosts.DEFAULT_COST_PER_1K,
        ]
        for cost in costs:
            assert cost >= 0.0, f"Cost should be non-negative: {cost}"


class TestComplexityScoring:
    """Test complexity scoring constants for thought analysis."""

    def test_max_score_constant(self):
        """RED: Test maximum complexity score constant."""
        assert ComplexityScoring.MAX_SCORE == 100.0
        assert isinstance(ComplexityScoring.MAX_SCORE, float)

    def test_word_count_scoring_constants(self):
        """RED: Test word count scoring constants."""
        assert ComplexityScoring.WORD_COUNT_MAX_POINTS == 15
        assert ComplexityScoring.WORD_COUNT_DIVISOR == 20
        assert isinstance(ComplexityScoring.WORD_COUNT_MAX_POINTS, int)
        assert isinstance(ComplexityScoring.WORD_COUNT_DIVISOR, int)

    def test_sentence_scoring_constants(self):
        """RED: Test sentence scoring constants."""
        assert ComplexityScoring.SENTENCE_MULTIPLIER == 2
        assert ComplexityScoring.SENTENCE_MAX_POINTS == 10
        assert isinstance(ComplexityScoring.SENTENCE_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.SENTENCE_MAX_POINTS, int)

    def test_question_scoring_constants(self):
        """RED: Test question scoring constants."""
        assert ComplexityScoring.QUESTION_MULTIPLIER == 3
        assert ComplexityScoring.QUESTION_MAX_POINTS == 15
        assert isinstance(ComplexityScoring.QUESTION_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.QUESTION_MAX_POINTS, int)

    def test_technical_term_scoring_constants(self):
        """RED: Test technical term scoring constants."""
        assert ComplexityScoring.TECHNICAL_TERM_MULTIPLIER == 2
        assert ComplexityScoring.TECHNICAL_TERM_MAX_POINTS == 20
        assert isinstance(ComplexityScoring.TECHNICAL_TERM_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.TECHNICAL_TERM_MAX_POINTS, int)

    def test_branching_scoring_constants(self):
        """RED: Test branching scoring constants."""
        assert ComplexityScoring.BRANCHING_MULTIPLIER == 5
        assert ComplexityScoring.BRANCHING_MAX_POINTS == 15
        assert isinstance(ComplexityScoring.BRANCHING_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.BRANCHING_MAX_POINTS, int)

    def test_research_scoring_constants(self):
        """RED: Test research scoring constants."""
        assert ComplexityScoring.RESEARCH_MULTIPLIER == 3
        assert ComplexityScoring.RESEARCH_MAX_POINTS == 15
        assert isinstance(ComplexityScoring.RESEARCH_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.RESEARCH_MAX_POINTS, int)

    def test_analysis_scoring_constants(self):
        """RED: Test analysis scoring constants."""
        assert ComplexityScoring.ANALYSIS_MULTIPLIER == 2
        assert ComplexityScoring.ANALYSIS_MAX_POINTS == 10
        assert isinstance(ComplexityScoring.ANALYSIS_MULTIPLIER, int)
        assert isinstance(ComplexityScoring.ANALYSIS_MAX_POINTS, int)

    def test_max_points_sum_within_bounds(self):
        """RED: Test that sum of max points doesn't exceed MAX_SCORE."""
        total_max_points = (
            ComplexityScoring.WORD_COUNT_MAX_POINTS
            + ComplexityScoring.SENTENCE_MAX_POINTS
            + ComplexityScoring.QUESTION_MAX_POINTS
            + ComplexityScoring.TECHNICAL_TERM_MAX_POINTS
            + ComplexityScoring.BRANCHING_MAX_POINTS
            + ComplexityScoring.RESEARCH_MAX_POINTS
            + ComplexityScoring.ANALYSIS_MAX_POINTS
        )
        assert total_max_points == ComplexityScoring.MAX_SCORE


class TestValidationLimits:
    """Test validation limit constants for input constraints."""

    def test_max_problem_length(self):
        """RED: Test maximum problem statement length limit."""
        assert ValidationLimits.MAX_PROBLEM_LENGTH == 500
        assert isinstance(ValidationLimits.MAX_PROBLEM_LENGTH, int)

    def test_max_context_length(self):
        """RED: Test maximum context length limit."""
        assert ValidationLimits.MAX_CONTEXT_LENGTH == 300
        assert isinstance(ValidationLimits.MAX_CONTEXT_LENGTH, int)

    def test_max_thoughts_per_session(self):
        """RED: Test maximum thoughts per session limit."""
        assert ValidationLimits.MAX_THOUGHTS_PER_SESSION == 1000
        assert isinstance(ValidationLimits.MAX_THOUGHTS_PER_SESSION, int)

    def test_max_branches_per_session(self):
        """RED: Test maximum branches per session limit."""
        assert ValidationLimits.MAX_BRANCHES_PER_SESSION == 50
        assert isinstance(ValidationLimits.MAX_BRANCHES_PER_SESSION, int)

    def test_max_thoughts_per_branch(self):
        """RED: Test maximum thoughts per branch limit."""
        assert ValidationLimits.MAX_THOUGHTS_PER_BRANCH == 100
        assert isinstance(ValidationLimits.MAX_THOUGHTS_PER_BRANCH, int)

    def test_min_thought_number(self):
        """RED: Test minimum thought number constant."""
        assert ValidationLimits.MIN_THOUGHT_NUMBER == 1
        assert isinstance(ValidationLimits.MIN_THOUGHT_NUMBER, int)

    def test_all_limits_are_positive(self):
        """RED: Test all validation limits are positive integers."""
        limits = [
            ValidationLimits.MAX_PROBLEM_LENGTH,
            ValidationLimits.MAX_CONTEXT_LENGTH,
            ValidationLimits.MAX_THOUGHTS_PER_SESSION,
            ValidationLimits.MAX_BRANCHES_PER_SESSION,
            ValidationLimits.MAX_THOUGHTS_PER_BRANCH,
            ValidationLimits.MIN_THOUGHT_NUMBER,
        ]
        for limit in limits:
            assert limit > 0, f"Limit should be positive: {limit}"


class TestTokenEstimates:
    """Test token estimation constants for different complexity levels."""

    def test_single_agent_estimates_structure(self):
        """RED: Test single agent token estimates are tuples with min/max."""
        estimates = [
            TokenEstimates.SINGLE_AGENT_SIMPLE,
            TokenEstimates.SINGLE_AGENT_MODERATE,
            TokenEstimates.SINGLE_AGENT_COMPLEX,
            TokenEstimates.SINGLE_AGENT_HIGHLY_COMPLEX,
        ]
        for estimate in estimates:
            assert isinstance(estimate, tuple)
            assert len(estimate) == 2
            min_tokens, max_tokens = estimate
            assert isinstance(min_tokens, int)
            assert isinstance(max_tokens, int)
            assert min_tokens < max_tokens
            assert min_tokens > 0

    def test_multi_agent_estimates_structure(self):
        """RED: Test multi-agent token estimates are tuples with min/max."""
        estimates = [
            TokenEstimates.MULTI_AGENT_SIMPLE,
            TokenEstimates.MULTI_AGENT_MODERATE,
            TokenEstimates.MULTI_AGENT_COMPLEX,
            TokenEstimates.MULTI_AGENT_HIGHLY_COMPLEX,
        ]
        for estimate in estimates:
            assert isinstance(estimate, tuple)
            assert len(estimate) == 2
            min_tokens, max_tokens = estimate
            assert isinstance(min_tokens, int)
            assert isinstance(max_tokens, int)
            assert min_tokens < max_tokens
            assert min_tokens > 0

    def test_multi_agent_higher_than_single_agent(self):
        """RED: Test multi-agent estimates are higher than single-agent."""
        single_complex = TokenEstimates.SINGLE_AGENT_COMPLEX
        multi_simple = TokenEstimates.MULTI_AGENT_SIMPLE

        # Multi-agent simple should be higher than single-agent complex
        assert multi_simple[0] >= single_complex[0]
        assert multi_simple[1] >= single_complex[1]

    def test_complexity_increases_token_estimates(self):
        """RED: Test that higher complexity has higher token estimates."""
        single_estimates = [
            TokenEstimates.SINGLE_AGENT_SIMPLE,
            TokenEstimates.SINGLE_AGENT_MODERATE,
            TokenEstimates.SINGLE_AGENT_COMPLEX,
            TokenEstimates.SINGLE_AGENT_HIGHLY_COMPLEX,
        ]

        # Each level should have higher estimates than the previous
        for i in range(1, len(single_estimates)):
            prev_min, prev_max = single_estimates[i - 1]
            curr_min, curr_max = single_estimates[i]
            assert curr_min >= prev_min
            assert curr_max >= prev_max


class TestDefaultSettings:
    """Test default setting constants."""

    def test_default_provider(self):
        """RED: Test default LLM provider constant."""
        assert DefaultSettings.DEFAULT_PROVIDER == "deepseek"
        assert isinstance(DefaultSettings.DEFAULT_PROVIDER, str)

    def test_default_complexity_threshold(self):
        """RED: Test default complexity threshold for routing."""
        assert DefaultSettings.DEFAULT_COMPLEXITY_THRESHOLD == 30.0
        assert isinstance(DefaultSettings.DEFAULT_COMPLEXITY_THRESHOLD, float)
        assert 0.0 <= DefaultSettings.DEFAULT_COMPLEXITY_THRESHOLD <= 100.0

    def test_default_token_buffer(self):
        """RED: Test default token buffer for estimates."""
        assert DefaultSettings.DEFAULT_TOKEN_BUFFER == 0.2
        assert isinstance(DefaultSettings.DEFAULT_TOKEN_BUFFER, float)
        assert 0.0 <= DefaultSettings.DEFAULT_TOKEN_BUFFER <= 1.0

    def test_default_session_timeout(self):
        """RED: Test default session timeout in seconds."""
        assert DefaultSettings.DEFAULT_SESSION_TIMEOUT == 3600
        assert isinstance(DefaultSettings.DEFAULT_SESSION_TIMEOUT, int)
        assert DefaultSettings.DEFAULT_SESSION_TIMEOUT > 0


class TestProcessingStrategy:
    """Test processing strategy enum values."""

    def test_processing_strategy_enum_values(self):
        """RED: Test that ProcessingStrategy enum has expected values."""
        # These should match the enum values used in routing
        assert ProcessingStrategy.SINGLE_AGENT.value == "single_agent"
        assert ProcessingStrategy.MULTI_AGENT.value == "multi_agent"
        assert ProcessingStrategy.ADAPTIVE.value == "adaptive"

    def test_processing_strategy_enum_members(self):
        """RED: Test ProcessingStrategy has all expected members."""
        expected_strategies = {"SINGLE_AGENT", "MULTI_AGENT", "ADAPTIVE"}
        actual_strategies = {strategy.name for strategy in ProcessingStrategy}
        assert actual_strategies == expected_strategies
