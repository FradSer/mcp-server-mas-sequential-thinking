"""Agent performance monitoring and response optimization module."""

import time

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class AgentPerformanceOptimizer:
    """Monitors and optimizes agent performance in real-time."""

    def __init__(self) -> None:
        self.processing_history = []
        self.performance_issues = []
        self.optimization_suggestions = []

    def record_processing(
        self,
        thought_data: ThoughtData,
        strategy_name: str,
        processing_time: float,
        complexity_score: float,
        response_length: int
    ) -> None:
        """Record processing event for optimization."""
        record = {
            "thought_id": thought_data.thoughtNumber,
            "strategy": strategy_name,
            "processing_time": processing_time,
            "complexity_score": complexity_score,
            "response_length": response_length,
            "timestamp": time.time()
        }

        self.processing_history.append(record)

        # Analyze for issues
        self._analyze_performance_issues(record)

        # Keep only recent history
        if len(self.processing_history) > 20:
            self.processing_history.pop(0)

    def _analyze_performance_issues(self, record: dict) -> None:
        """Analyze individual record for performance issues."""
        issues = []

        # Time-based issues
        if record["processing_time"] > 300:  # 5 minutes
            issues.append("CRITICAL: Processing time exceeds 5 minutes")
        elif record["processing_time"] > 120:  # 2 minutes
            issues.append("WARNING: Processing time exceeds 2 minutes")

        # Complexity vs time mismatch
        expected_time = record["complexity_score"] * 3  # Rough estimate
        if record["processing_time"] > expected_time * 2:
            issues.append("INEFFICIENT: Processing time much higher than complexity suggests")

        # Strategy appropriateness
        if "全面探索" in record["strategy"] and record["complexity_score"] < 30:
            issues.append("OVER_PROCESSING: Full exploration used for moderate complexity")

        if issues:
            self.performance_issues.extend(issues)
            logger.warning(f"⚠️ Performance issues detected for thought #{record['thought_id']}: {'; '.join(issues)}")

    def get_optimization_summary(self) -> dict:
        """Get comprehensive optimization summary."""
        if not self.processing_history:
            return {"status": "no_data"}

        recent_records = self.processing_history[-10:]
        avg_time = sum(r["processing_time"] for r in recent_records) / len(recent_records)
        avg_complexity = sum(r["complexity_score"] for r in recent_records) / len(recent_records)

        slow_count = sum(1 for r in recent_records if r["processing_time"] > 120)
        over_processing_count = sum(1 for r in recent_records
                                  if "全面探索" in r["strategy"] and r["complexity_score"] < 30)

        recommendations = []
        if slow_count > len(recent_records) * 0.3:
            recommendations.append("Consider switching to FAST mode for better performance")

        if over_processing_count > 0:
            recommendations.append("Complexity thresholds need adjustment - too many simple tasks using expensive processing")

        if avg_time > 120:
            recommendations.append("Overall processing time too high - review routing logic")

        return {
            "status": "analyzed",
            "avg_processing_time": avg_time,
            "avg_complexity": avg_complexity,
            "slow_processing_rate": slow_count / len(recent_records),
            "over_processing_rate": over_processing_count / len(recent_records),
            "total_issues": len(self.performance_issues),
            "recommendations": recommendations
        }


class SmartResponseFormatter:
    """Smart response formatter that adapts based on content and performance."""

    def __init__(self) -> None:
        self.formatting_rules = {
            "academic_complexity_threshold": 0.3,  # Ratio of academic indicators
            "max_reasonable_length": 1500,
            "optimal_length_range": (400, 800)
        }

    def format_response(self, content: str, thought_data: ThoughtData, strategy_name: str) -> str:
        """Format response with smart optimization."""
        # Analyze content characteristics
        analysis = self._analyze_content(content)

        # Apply formatting optimizations
        if analysis["is_overly_academic"]:
            content = self._simplify_academic_content(content)

        if analysis["is_too_long"]:
            content = self._optimize_length(content)

        # Add processing transparency if needed
        if "全面探索" in strategy_name:
            content = self._add_processing_note(content, "使用深度分析流程")

        return content

    def _analyze_content(self, content: str) -> dict:
        """Analyze content characteristics."""
        # Academic complexity indicators
        academic_indicators = (
            content.count("$$") + content.count("\\(") + content.count("###") +
            content.count("\\mathcal") + content.count("\\int") + content.count("\\sum")
        )

        academic_ratio = academic_indicators / max(len(content.split()), 1)

        return {
            "length": len(content),
            "academic_indicators": academic_indicators,
            "academic_ratio": academic_ratio,
            "is_overly_academic": academic_ratio > self.formatting_rules["academic_complexity_threshold"],
            "is_too_long": len(content) > self.formatting_rules["max_reasonable_length"],
            "is_optimal_length": (self.formatting_rules["optimal_length_range"][0] <=
                                len(content) <= self.formatting_rules["optimal_length_range"][1])
        }

    def _simplify_academic_content(self, content: str) -> str:
        """Simplify overly academic content."""
        # Remove excessive mathematical notation
        import re

        # Replace complex mathematical expressions with simpler explanations
        content = re.sub(r"\$\$.*?\$\$", "[数学公式]", content, flags=re.DOTALL)
        content = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "[数学符号]", content)

        # Reduce excessive sectioning
        content = re.sub(r"#{4,}", "###", content)  # Max 3 levels

        # Add simplification note
        if "[数学公式]" in content or "[数学符号]" in content:
            content += "\n\n*注：已简化数学表达以提高可读性*"

        return content

    def _optimize_length(self, content: str) -> str:
        """Optimize content length while preserving key information."""
        # Split into sections
        sections = content.split("\n\n")

        # Prioritize sections by importance
        important_sections = []
        for section in sections:
            # Keep sections with key indicators
            if any(indicator in section.lower() for indicator in
                   ["总结", "结论", "核心", "关键", "重要", "主要"]) or len(section) > 100:
                important_sections.append(section)

        # Reconstruct with optimal length
        optimized = "\n\n".join(important_sections[:5])  # Max 5 sections

        if len(optimized) > self.formatting_rules["max_reasonable_length"]:
            # Further truncation
            optimized = optimized[:self.formatting_rules["max_reasonable_length"]] + "...\n\n*内容已优化长度*"

        return optimized

    def _add_processing_note(self, content: str, processing_type: str) -> str:
        """Add transparent processing note."""
        note = f"\n\n---\n*{processing_type}处理完成*"
        return content + note


# Global instances for easy access
performance_optimizer = AgentPerformanceOptimizer()
response_formatter = SmartResponseFormatter()


def get_optimization_status() -> dict:
    """Get current optimization status."""
    return {
        "performance_summary": performance_optimizer.get_optimization_summary(),
        "optimizations_active": True
    }
