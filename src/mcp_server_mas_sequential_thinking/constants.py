"""Constants for the MCP Sequential Thinking Server."""

from enum import Enum


class TokenCosts:
    """Token cost constants for different providers (cost per 1000 tokens)."""

    DEEPSEEK_COST_PER_1K = 0.0002
    GROQ_COST_PER_1K = 0.0001
    OPENROUTER_COST_PER_1K = 0.001
    GITHUB_COST_PER_1K = 0.0005
    OLLAMA_COST_PER_1K = 0.0000
    DEFAULT_COST_PER_1K = 0.0002


class ComplexityScoring:
    """Complexity analysis scoring constants."""

    MAX_SCORE = 100.0
    WORD_COUNT_MAX_POINTS = 15
    WORD_COUNT_DIVISOR = 20
    SENTENCE_MULTIPLIER = 2
    SENTENCE_MAX_POINTS = 10
    QUESTION_MULTIPLIER = 3
    QUESTION_MAX_POINTS = 15
    TECHNICAL_TERM_MULTIPLIER = 2
    TECHNICAL_TERM_MAX_POINTS = 20
    BRANCHING_MULTIPLIER = 5
    BRANCHING_MAX_POINTS = 15
    RESEARCH_MULTIPLIER = 3
    RESEARCH_MAX_POINTS = 15
    ANALYSIS_MULTIPLIER = 2
    ANALYSIS_MAX_POINTS = 10


class TokenEstimates:
    """Token estimation ranges by complexity and strategy."""

    # Single agent token estimates (min, max)
    SINGLE_AGENT_SIMPLE = (400, 800)
    SINGLE_AGENT_MODERATE = (600, 1200)
    SINGLE_AGENT_COMPLEX = (800, 1600)
    SINGLE_AGENT_HIGHLY_COMPLEX = (1000, 2000)

    # Multi-agent token estimates (min, max)
    MULTI_AGENT_SIMPLE = (2000, 4000)
    MULTI_AGENT_MODERATE = (3000, 6000)
    MULTI_AGENT_COMPLEX = (4000, 8000)
    MULTI_AGENT_HIGHLY_COMPLEX = (5000, 10000)


class ValidationLimits:
    """Input validation and system limits."""

    MAX_PROBLEM_LENGTH = 500
    MAX_CONTEXT_LENGTH = 300
    MAX_THOUGHTS_PER_SESSION = 1000
    MAX_BRANCHES_PER_SESSION = 50
    MAX_THOUGHTS_PER_BRANCH = 100
    GITHUB_TOKEN_LENGTH = 40
    MIN_TOTAL_THOUGHTS = 5
    MIN_THOUGHT_NUMBER = 1


class DefaultTimeouts:
    """Default timeout values in seconds."""

    PROCESSING_TIMEOUT = 30.0
    SESSION_CLEANUP_DAYS = 30
    RECENT_SESSION_KEEP_COUNT = 100


class LoggingLimits:
    """Logging configuration constants."""

    LOG_FILE_MAX_BYTES = 5 * 1024 * 1024  # 5MB
    LOG_BACKUP_COUNT = 3
    SENSITIVE_DATA_MIN_LENGTH = 8


class QualityThresholds:
    """Quality scoring and budget utilization thresholds."""

    DEFAULT_QUALITY_THRESHOLD = 0.7
    HIGH_BUDGET_UTILIZATION = 0.8
    VERY_HIGH_BUDGET_UTILIZATION = 0.9
    MULTI_AGENT_HIGH_USAGE = 0.7
    SINGLE_AGENT_HIGH_USAGE = 0.8
    MINIMUM_USAGE_FOR_SUGGESTIONS = 10
    SIGNIFICANT_COST_THRESHOLD = 0.01


class ProviderDefaults:
    """Default provider configurations."""

    DEFAULT_QUALITY_SCORE = 0.8
    DEFAULT_RESPONSE_TIME = 2.0
    DEFAULT_UPTIME_SCORE = 0.95
    DEFAULT_ERROR_RATE = 0.05
    DEFAULT_CONTEXT_LENGTH = 4096


class ComplexityThresholds:
    """Complexity level thresholds for scoring."""

    SIMPLE_MAX = 25.0
    MODERATE_MAX = 50.0
    COMPLEX_MAX = 75.0
    # HIGHLY_COMPLEX is anything above COMPLEX_MAX


class DefaultValues:
    """Default configuration values."""

    DEFAULT_LLM_PROVIDER = "deepseek"
    DEFAULT_TEAM_MODE = "standard"
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_TIMEOUT = 30.0


class FieldLengthLimits:
    """Field length limits for various inputs."""

    MIN_STRING_LENGTH = 1
    MAX_STANDARD_STRING = 2000
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_BRANCH_ID_LENGTH = 100


class DatabaseConstants:
    """Database configuration constants."""

    SESSION_CLEANUP_BATCH_SIZE = 100
    THOUGHT_QUERY_LIMIT = 1000
    CONNECTION_POOL_SIZE = 5
    CONNECTION_POOL_OVERFLOW = 10


class TechnicalTerms:
    """Technical terms for complexity analysis."""

    KEYWORDS = [
        "algorithm",
        "data",
        "analysis",
        "system",
        "process",
        "design",
        "implementation",
        "architecture",
        "framework",
        "model",
        "structure",
        "optimization",
        "performance",
        "scalability",
        "integration",
        "api",
        "database",
        "security",
        "authentication",
        "authorization",
        "testing",
        "deployment",
        "configuration",
        "monitoring",
        "logging",
        "debugging",
        "refactoring",
        "migration",
        "synchronization",
        "caching",
        "protocol",
        "interface",
        "inheritance",
        "polymorphism",
        "abstraction",
        "encapsulation",
    ]


class DefaultSettings:
    """Default application settings."""

    DEFAULT_PROVIDER = "deepseek"
    DEFAULT_COMPLEXITY_THRESHOLD = 30.0
    DEFAULT_TOKEN_BUFFER = 0.2
    DEFAULT_SESSION_TIMEOUT = 3600


class ProcessingStrategy(Enum):
    """Processing strategy enumeration."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    ADAPTIVE = "adaptive"
