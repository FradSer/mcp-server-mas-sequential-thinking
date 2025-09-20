"""Text complexity analysis utilities with modular design."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Pattern
from enum import Enum

from .models import ThoughtData
from .constants import ComplexityAnalysisConstants


class LanguageType(Enum):
    """Supported language types for analysis."""

    ENGLISH = "english"
    CHINESE = "chinese"
    MIXED = "mixed"


@dataclass
class LanguageMetrics:
    """Metrics for specific language characteristics."""

    language_type: LanguageType
    word_count: int
    character_count: int
    chinese_char_count: int = 0
    estimated_words: int = 0


@dataclass
class TextFeatures:
    """Extracted text features for complexity analysis."""

    language_metrics: LanguageMetrics
    sentence_count: int
    question_count: int
    technical_terms: int
    research_indicators: int
    branching_references: int
    analysis_depth: int


class TextAnalyzer(ABC):
    """Abstract base class for text analysis strategies."""

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, int]:
        """Analyze text and return feature counts."""
        pass


class LanguageDetector:
    """Detects and analyzes language characteristics of text."""

    CHINESE_RANGE = ("\u4e00", "\u9fff")
    CHINESE_WORD_RATIO = ComplexityAnalysisConstants.CHINESE_WORD_RATIO

    def analyze_language(self, text: str) -> LanguageMetrics:
        """Analyze language characteristics and word counts."""
        # Count Chinese characters
        chinese_chars = len(
            [c for c in text if self.CHINESE_RANGE[0] <= c <= self.CHINESE_RANGE[1]]
        )

        # Count space-separated words (primarily English)
        space_words = text.split()

        # Estimate total words
        estimated_chinese_words = chinese_chars // self.CHINESE_WORD_RATIO
        total_words = len(space_words) + estimated_chinese_words

        # Determine language type
        threshold = ComplexityAnalysisConstants.CHINESE_DOMINANCE_THRESHOLD
        if chinese_chars > len(text) * threshold:
            language_type = (
                LanguageType.CHINESE
                if chinese_chars > len(space_words) * 3
                else LanguageType.MIXED
            )
        else:
            language_type = LanguageType.ENGLISH

        return LanguageMetrics(
            language_type=language_type,
            word_count=total_words,
            character_count=len(text),
            chinese_char_count=chinese_chars,
            estimated_words=estimated_chinese_words,
        )


class SentenceAnalyzer(TextAnalyzer):
    """Analyzes sentence structure and patterns."""

    SENTENCE_PATTERNS = {
        "sentence_endings": re.compile(r"[.!?。！？]+"),
        "questions": re.compile(r"[?？]"),
    }

    def analyze(self, text: str) -> Dict[str, int]:
        """Analyze sentence structure."""
        sentences = self.SENTENCE_PATTERNS["sentence_endings"].split(text)
        valid_sentences = [s for s in sentences if s.strip()]

        questions = len(self.SENTENCE_PATTERNS["questions"].findall(text))

        return {"sentence_count": len(valid_sentences), "question_count": questions}


class TechnicalTermAnalyzer(TextAnalyzer):
    """Analyzes technical terminology usage."""

    TECHNICAL_TERMS = [
        # Programming and development
        "algorithm",
        "api",
        "database",
        "framework",
        "architecture",
        "implementation",
        "optimization",
        "performance",
        "scalability",
        "integration",
        "authentication",
        "authorization",
        "encryption",
        # System and infrastructure
        "system",
        "process",
        "design",
        "model",
        "structure",
        "protocol",
        "interface",
        "deployment",
        "configuration",
        "monitoring",
        # Analysis and data
        "analysis",
        "data",
        "analytics",
        "metrics",
        "statistics",
        "correlation",
        "patterns",
        "trends",
        "insights",
        "visualization",
        # Chinese technical terms
        "算法",
        "接口",
        "数据库",
        "框架",
        "架构",
        "实现",
        "优化",
        "性能",
        "可扩展性",
        "集成",
        "认证",
        "授权",
        "加密",
        "系统",
        "流程",
        "设计",
        "模型",
        "结构",
        "协议",
        "部署",
        "配置",
        "监控",
        "分析",
        "数据",
        "指标",
    ]

    def analyze(self, text: str) -> Dict[str, int]:
        """Count technical terms in text."""
        text_lower = text.lower()
        technical_count = sum(1 for term in self.TECHNICAL_TERMS if term in text_lower)

        return {"technical_terms": technical_count}


class ResearchAnalyzer(TextAnalyzer):
    """Analyzes research-related indicators."""

    RESEARCH_INDICATORS = [
        "research",
        "study",
        "investigate",
        "explore",
        "examine",
        "analyze",
        "evaluate",
        "assess",
        "compare",
        "review",
        "survey",
        "experiment",
        "hypothesis",
        "methodology",
        "findings",
        "conclusion",
        "evidence",
        "data",
        "statistics",
        "correlation",
        # Chinese research terms
        "研究",
        "调研",
        "探索",
        "分析",
        "评估",
        "比较",
        "调查",
        "实验",
        "假设",
        "方法",
        "发现",
        "结论",
        "证据",
        "数据",
    ]

    def analyze(self, text: str) -> Dict[str, int]:
        """Count research indicators in text."""
        text_lower = text.lower()
        research_count = sum(
            1 for indicator in self.RESEARCH_INDICATORS if indicator in text_lower
        )

        return {"research_indicators": research_count}


class BranchingAnalyzer(TextAnalyzer):
    """Analyzes branching and decision-making patterns."""

    BRANCHING_INDICATORS = [
        "branch",
        "alternative",
        "option",
        "choice",
        "decision",
        "if",
        "else",
        "either",
        "or",
        "instead",
        "alternatively",
        "consider",
        "evaluate",
        "compare",
        "contrast",
        "versus",
        # Chinese branching terms
        "分支",
        "选择",
        "决策",
        "如果",
        "否则",
        "或者",
        "替代",
        "考虑",
        "评估",
        "比较",
        "对比",
    ]

    def analyze(self, text: str) -> Dict[str, int]:
        """Count branching indicators in text."""
        text_lower = text.lower()
        branching_count = sum(
            1 for indicator in self.BRANCHING_INDICATORS if indicator in text_lower
        )

        return {"branching_references": branching_count}


class AnalysisDepthAnalyzer(TextAnalyzer):
    """Analyzes logical depth and reasoning complexity."""

    LOGICAL_CONNECTORS = {
        "causal": [
            "because",
            "therefore",
            "consequently",
            "thus",
            "hence",
            "因为",
            "所以",
            "因此",
            "由于",
            "既然",
        ],
        "contrast": [
            "however",
            "nevertheless",
            "nonetheless",
            "although",
            "然而",
            "但是",
            "不过",
            "虽然",
            "尽管",
        ],
        "addition": [
            "moreover",
            "furthermore",
            "additionally",
            "besides",
            "而且",
            "并且",
            "另外",
            "此外",
        ],
        "conditional": [
            "if",
            "unless",
            "provided",
            "assuming",
            "如果",
            "假如",
            "除非",
            "假设",
        ],
    }

    def analyze(self, text: str) -> Dict[str, int]:
        """Analyze logical reasoning depth."""
        text_lower = text.lower()
        total_depth = 0

        for category, connectors in self.LOGICAL_CONNECTORS.items():
            total_depth += sum(text_lower.count(connector) for connector in connectors)

        return {"analysis_depth": total_depth}


class ComplexityAnalyzer:
    """Comprehensive text complexity analyzer using modular analyzers."""

    def __init__(self):
        """Initialize all analyzer components."""
        self.language_detector = LanguageDetector()
        self.analyzers = {
            "sentence": SentenceAnalyzer(),
            "technical": TechnicalTermAnalyzer(),
            "research": ResearchAnalyzer(),
            "branching": BranchingAnalyzer(),
            "analysis_depth": AnalysisDepthAnalyzer(),
        }

    def analyze_features(self, thought_data: ThoughtData) -> TextFeatures:
        """Perform comprehensive feature extraction."""
        text = thought_data.thought.lower()

        # Language analysis
        language_metrics = self.language_detector.analyze_language(text)

        # Feature analysis
        features = {}
        for analyzer_name, analyzer in self.analyzers.items():
            features.update(analyzer.analyze(text))

        # Handle branching context bonus
        if thought_data.branchFromThought is not None:
            bonus = ComplexityAnalysisConstants.BRANCHING_CONTEXT_BONUS
            features["branching_references"] = (
                features.get("branching_references", 0) + bonus
            )

        return TextFeatures(
            language_metrics=language_metrics,
            sentence_count=features.get("sentence_count", 0),
            question_count=features.get("question_count", 0),
            technical_terms=features.get("technical_terms", 0),
            research_indicators=features.get("research_indicators", 0),
            branching_references=features.get("branching_references", 0),
            analysis_depth=features.get("analysis_depth", 0),
        )

    def get_complexity_insights(self, features: TextFeatures) -> Dict[str, str]:
        """Generate human-readable complexity insights."""
        insights = []

        # Language insights
        lang_type = features.language_metrics.language_type
        if lang_type == LanguageType.MIXED:
            insights.append("Mixed language content detected")
        elif lang_type == LanguageType.CHINESE:
            insights.append("Primarily Chinese content")

        # Content complexity insights
        if features.technical_terms > 5:
            insights.append("High technical terminology usage")
        elif features.technical_terms > 2:
            insights.append("Moderate technical content")

        if features.research_indicators > 3:
            insights.append("Research-oriented content")

        if features.analysis_depth > 5:
            insights.append("Deep logical reasoning")

        if features.branching_references > 2:
            insights.append("Complex decision-making content")

        return {
            "language_type": lang_type.value,
            "complexity_level": self._determine_complexity_level(features),
            "insights": "; ".join(insights)
            if insights
            else "Standard content complexity",
        }

    def _determine_complexity_level(self, features: TextFeatures) -> str:
        """Determine overall complexity level based on features."""
        score = (
            features.language_metrics.word_count / 10
            + features.technical_terms * 2
            + features.research_indicators * 2
            + features.analysis_depth
            + features.branching_references
        )

        if score < 10:
            return "simple"
        elif score < 25:
            return "moderate"
        elif score < 40:
            return "complex"
        else:
            return "highly_complex"
