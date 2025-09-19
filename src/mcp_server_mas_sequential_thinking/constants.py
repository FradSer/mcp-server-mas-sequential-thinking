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
    MIN_THOUGHT_NUMBER = 1


class DefaultTimeouts:
    """Default timeout values in seconds."""

    PROCESSING_TIMEOUT = 30.0
    DEEPSEEK_PROCESSING_TIMEOUT = 120.0  # Legacy timeout for Deepseek (deprecated)
    MULTI_AGENT_TIMEOUT_MULTIPLIER = 2.0  # Multiply timeout for multi-agent

    # Adaptive timeout strategy (v0.5.1+)
    ADAPTIVE_BASE_DEEPSEEK = 90.0  # Base timeout for Deepseek with adaptive scaling
    ADAPTIVE_BASE_GROQ = 20.0  # Base timeout for Groq
    ADAPTIVE_BASE_OPENAI = 45.0  # Base timeout for OpenAI
    ADAPTIVE_BASE_DEFAULT = 30.0  # Default base timeout

    # Maximum timeouts (safety ceiling)
    MAX_TIMEOUT_DEEPSEEK = 300.0  # 5 minutes absolute maximum for Deepseek
    MAX_TIMEOUT_DEFAULT = 180.0  # 3 minutes maximum for others

    # Complexity multipliers for adaptive timeouts
    COMPLEXITY_SIMPLE_MULTIPLIER = 1.0
    COMPLEXITY_MODERATE_MULTIPLIER = 1.5
    COMPLEXITY_COMPLEX_MULTIPLIER = 2.0
    COMPLEXITY_HIGHLY_COMPLEX_MULTIPLIER = 3.0

    # Retry configuration
    RETRY_EXPONENTIAL_BASE = 1.5  # Exponential backoff base
    MAX_RETRY_ATTEMPTS = 2  # Maximum retry attempts

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


class PerformanceMetrics:
    """Performance measurement constants."""

    # Efficiency calculation thresholds
    EFFICIENCY_TIME_THRESHOLD = 60.0  # seconds
    PERFECT_EXECUTION_CONSISTENCY = 1.0
    DEFAULT_EXECUTION_CONSISTENCY = 0.9
    PERFECT_EFFICIENCY_SCORE = 1.0
    MINIMUM_EFFICIENCY_SCORE = 0.5

    # Retry and sleep constants
    RETRY_SLEEP_DURATION = 1.0  # seconds

    # Logging formatting
    SEPARATOR_LENGTH = 50


class ProcessingDefaults:
    """Default values for thought processing."""

    ERROR_COUNT_INITIAL = 1
    RETRY_COUNT_INITIAL = 0
    PROCESSING_TIME_INITIAL = 0.0
    TEAM_INITIALIZER_INDEX = 1
    SINGLE_AGENT_TIMEOUT_MULTIPLIER = 0.5
    EXIT_CODE_ERROR = 1


class FieldLengthLimits:
    """Field length limits for various inputs."""

    MIN_STRING_LENGTH = 1
    MAX_STANDARD_STRING = 2000
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_BRANCH_ID_LENGTH = 100
    SEPARATOR_LENGTH = 50


class DatabaseConstants:
    """Database configuration constants."""

    SESSION_CLEANUP_BATCH_SIZE = 100
    THOUGHT_QUERY_LIMIT = 1000
    CONNECTION_POOL_SIZE = 5
    CONNECTION_POOL_OVERFLOW = 10


class ThoughtProcessingLimits:
    """Limits for thought processing workflow."""

    MIN_THOUGHT_SEQUENCE = 1
    MAX_TEAM_DELEGATION_COUNT = 2
    ANALYSIS_TIME_LIMIT_SECONDS = 5
    MIN_PROCESSING_STEPS = 1
    MAX_PROCESSING_STEPS = 6


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


class CostOptimizationConstants:
    """Constants for cost optimization calculations."""

    # Quality scoring weights
    QUALITY_WEIGHT = 0.4
    COST_WEIGHT = 0.3
    SPEED_WEIGHT = 0.2
    RELIABILITY_WEIGHT = 0.1

    # Cost calculation factors
    COST_NORMALIZATION_FACTOR = 0.0003
    COST_EPSILON = 0.0001  # Prevent division by zero
    DEFAULT_COST_ESTIMATE = 0.0002
    SPEED_NORMALIZATION_BASE = 10
    SPEED_THRESHOLD = 1

    # Quality scoring bounds
    MIN_QUALITY_SCORE = 0.0
    MAX_QUALITY_SCORE = 1.0

    # Budget utilization thresholds
    HIGH_BUDGET_UTILIZATION = 0.8
    MODERATE_BUDGET_UTILIZATION = 0.7
    CRITICAL_BUDGET_UTILIZATION = 0.9

    # Complexity bonuses
    QUALITY_COMPLEXITY_BONUS = 0.2
    COST_COMPLEXITY_BONUS = 0.0001

    # Provider optimization
    HIGH_USAGE_PENALTY = 2.0
    MODERATE_USAGE_PENALTY = 0.5
    QUALITY_UPDATE_WEIGHT = 0.1
    OLD_QUALITY_WEIGHT = 0.9

    # Usage analysis thresholds
    MIN_DATA_THRESHOLD = 10
    HIGH_MULTI_AGENT_RATIO = 0.7
    HIGH_SINGLE_AGENT_RATIO = 0.8
    MINIMUM_COST_DIFFERENCE = 0.01

    # Provider-specific configurations
    GROQ_RATE_LIMIT = 14400
    GROQ_CONTEXT_LENGTH = 32768
    GROQ_QUALITY_SCORE = 0.75
    GROQ_RESPONSE_TIME = 0.8

    DEEPSEEK_QUALITY_SCORE = 0.85
    DEEPSEEK_CONTEXT_LENGTH = 128000

    GITHUB_CONTEXT_LENGTH = 128000

    OPENROUTER_RESPONSE_TIME = 3.0
    OPENROUTER_CONTEXT_LENGTH = 200000

    OLLAMA_QUALITY_SCORE = 0.70
    OLLAMA_RESPONSE_TIME = 5.0
    OLLAMA_CONTEXT_LENGTH = 8192


class ComplexityAnalysisConstants:
    """Constants for complexity analysis calculations."""

    # Language detection
    CHINESE_WORD_RATIO = 2  # Estimate Chinese words as characters / 2
    CHINESE_DOMINANCE_THRESHOLD = 0.3  # Chinese chars > 30% of text

    # Complexity scoring weights (extracted from adaptive_routing.py)
    WORD_COUNT_WEIGHT = 0.15
    SENTENCE_WEIGHT = 0.10
    QUESTION_WEIGHT = 0.15
    TECHNICAL_TERM_WEIGHT = 0.20
    BRANCHING_WEIGHT = 0.15
    RESEARCH_WEIGHT = 0.15
    ANALYSIS_DEPTH_WEIGHT = 0.10

    # Complexity level thresholds
    SIMPLE_THRESHOLD = 25.0
    MODERATE_THRESHOLD = 50.0
    COMPLEX_THRESHOLD = 75.0

    # Branching bonus for actual branch context
    BRANCHING_CONTEXT_BONUS = 2

    # Text analysis limits
    MAX_PREVIEW_LENGTH = 200
    MIN_SENTENCE_LENGTH = 3


class ProcessingStrategy(Enum):
    """Processing strategy enumeration."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    ADAPTIVE = "adaptive"
