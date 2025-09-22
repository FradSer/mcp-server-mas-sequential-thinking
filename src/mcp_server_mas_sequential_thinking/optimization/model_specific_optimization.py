"""Model-specific optimization for different LLM providers."""

import re

from mcp_server_mas_sequential_thinking.infrastructure.logging_config import get_logger

logger = get_logger(__name__)


class ModelCharacteristics:
    """Characteristics and tendencies of different LLM models."""

    DEEPSEEK_V3 = {
        "academic_bias": 0.9,  # Very high tendency toward academic language
        "technical_solution_bias": 0.8,  # Converts everything to technical problems
        "structure_obsession": 0.85,  # Over-structures responses
        "concept_density": 0.9,  # Packs too many concepts
        "human_relatability": 0.3,  # Low natural human connection
        "practical_focus": 0.2,  # Poor at practical advice
    }

    CLAUDE_3_5 = {
        "academic_bias": 0.4,
        "technical_solution_bias": 0.3,
        "structure_obsession": 0.5,
        "concept_density": 0.4,
        "human_relatability": 0.8,
        "practical_focus": 0.7,
    }

    GPT_4 = {
        "academic_bias": 0.5,
        "technical_solution_bias": 0.4,
        "structure_obsession": 0.6,
        "concept_density": 0.5,
        "human_relatability": 0.7,
        "practical_focus": 0.6,
    }


class DeepSeekV3Optimizer:
    """Specific optimizer for DeepSeek V3's tendencies."""

    def __init__(self) -> None:
        self.academic_indicators = [
            "框架", "模型", "维度", "矩阵", "算法", "系统", "机制",
            "架构", "引擎", "协议", "方程", "公式", "变量", "参数",
            "阈值", "指标", "评估", "验证", "实施", "部署"
        ]

        self.tech_solution_indicators = [
            "芯片", "AI", "区块链", "量子", "基因编辑", "脑机接口",
            "云计算", "大数据", "神经网络", "深度学习", "算力",
            "数字化", "智能化", "自动化", "平台", "系统"
        ]

        self.over_structure_patterns = [
            r"#{3,}",  # Too many heading levels
            r"\|\s*.*\s*\|",  # Tables
            r"```[\s\S]*?```",  # Code blocks for non-code content
            r"^\d+\.\s+.*→.*→",  # Complex numbered flows
        ]

    def optimize_prompt_for_deepseek(self, original_prompt: str, question_type: str = "philosophical") -> str:
        """Optimize prompt specifically for DeepSeek V3 to get better responses."""
        # Base optimization for all question types
        optimized_prompt = f"""请用简单、人性化的语言回答，避免过度理论化。

重要指导原则：
- 用日常语言，不要学术术语
- 给出实用的思考，而不是抽象框架
- 保持温度感，像和朋友对话
- 不要设计技术方案或系统
- 避免表格、公式、复杂结构

{original_prompt}

请用平实的语言，从人的角度来回答这个问题。"""

        # Question-type specific optimizations
        if question_type == "philosophical":
            optimized_prompt += """

特别要求：这是一个关于人生的问题，请：
- 承认问题的复杂性，但给出可理解的思考
- 分享不同人可能的感受和想法
- 提供实际的生活视角，而不是理论分析
- 让回答能够帮助提问者思考，而不是显示知识"""

        elif question_type == "creative":
            optimized_prompt += """

特别要求：这是一个创意问题，请：
- 提供具体可行的想法
- 用例子说明，而不是抽象概念
- 保持创意的趣味性和可操作性
- 避免设计复杂的系统或流程"""

        elif question_type == "factual":
            optimized_prompt += """

特别要求：这是一个事实问题，请：
- 直接回答核心信息
- 用简单的语言解释
- 如果复杂，用比喻或例子帮助理解
- 避免不必要的背景展开"""

        return optimized_prompt

    def post_process_deepseek_response(self, response: str) -> str:
        """Post-process DeepSeek V3 response to remove problematic patterns."""
        # Remove excessive academic language
        response = self._simplify_academic_language(response)

        # Remove technical solutions for non-technical problems
        response = self._remove_tech_solutions(response)

        # Simplify over-structured content
        response = self._simplify_structure(response)

        # Add human warmth if missing
        return self._add_human_connection(response)


    def _simplify_academic_language(self, text: str) -> str:
        """Replace academic jargon with simpler language."""
        replacements = {
            "综合决策框架": "思考方式",
            "多维度分析": "从不同角度看",
            "系统性思维": "全面思考",
            "优化算法": "更好的方法",
            "实施路径": "具体做法",
            "评估机制": "如何判断",
            "监测指标": "观察要点",
            "协调机制": "配合方式",
            "反馈循环": "相互影响",
            "迭代优化": "不断改进"
        }

        for academic, simple in replacements.items():
            text = text.replace(academic, simple)

        return text

    def _remove_tech_solutions(self, text: str) -> str:
        """Remove inappropriate technical solutions."""
        # Remove sentences containing tech buzzwords
        lines = text.split("\n")
        filtered_lines = []

        for line in lines:
            has_tech_overkill = any(
                indicator in line for indicator in [
                    "区块链", "量子", "基因编辑", "脑机接口", "AI芯片",
                    "神经网络训练", "算法优化", "数据挖掘", "机器学习"
                ]
            )

            if not has_tech_overkill or "技术" in line:  # Keep if explicitly about technology
                filtered_lines.append(line)

        return "\n".join(filtered_lines)

    def _simplify_structure(self, text: str) -> str:
        """Simplify overly complex structure."""
        # Remove excessive headers (keep max 2 levels)
        text = re.sub(r"#{4,}", "###", text)

        # Remove complex tables for simple content
        lines = text.split("\n")
        filtered_lines = []
        in_table = False

        for line in lines:
            if "|" in line and "-" in line:  # Table separator
                in_table = True
                continue
            if "|" in line and in_table:  # Table content
                continue
            in_table = False
            filtered_lines.append(line)

        # Remove code blocks that aren't actually code
        text = "\n".join(filtered_lines)
        return re.sub(r"```[^`]*?```", "", text, flags=re.DOTALL)


    def _add_human_connection(self, text: str) -> str:
        """Add human warmth if the response is too cold."""
        # Check if response lacks human elements
        human_indicators = ["感受", "体验", "想法", "认为", "觉得", "可能", "或许"]

        if not any(indicator in text for indicator in human_indicators):
            # Add a more human opening
            if text.startswith("#"):
                text = re.sub(r"^#[^#]*\n", "", text)

            text = f"这是一个很有意思的问题。{text}"

        # Soften overly definitive statements
        text = re.sub(r"必须", "可以考虑", text)
        text = re.sub(r"应该建立", "也许可以", text)
        text = re.sub(r"需要实施", "可以尝试", text)

        return text

    def assess_response_quality(self, response: str, question: str) -> dict[str, float]:
        """Assess how well the response matches the question."""
        scores = {}

        # Academic overload score (lower is better)
        academic_count = sum(1 for indicator in self.academic_indicators if indicator in response)
        scores["academic_overload"] = min(academic_count / 10, 1.0)

        # Tech solution inappropriateness (lower is better)
        tech_count = sum(1 for indicator in self.tech_solution_indicators if indicator in response)
        is_tech_question = any(word in question for word in ["技术", "科技", "AI", "计算机"])
        scores["tech_inappropriateness"] = 0 if is_tech_question else min(tech_count / 5, 1.0)

        # Structure complexity (lower is better)
        structure_score = 0
        for pattern in self.over_structure_patterns:
            matches = len(re.findall(pattern, response, re.MULTILINE))
            structure_score += matches
        scores["structure_complexity"] = min(structure_score / 5, 1.0)

        # Human relatability (higher is better)
        human_words = ["感受", "体验", "想法", "认为", "觉得", "可能", "或许", "有时", "通常"]
        human_count = sum(1 for word in human_words if word in response)
        scores["human_relatability"] = min(human_count / 5, 1.0)

        # Overall quality (higher is better)
        scores["overall_quality"] = (
            (1 - scores["academic_overload"]) * 0.3 +
            (1 - scores["tech_inappropriateness"]) * 0.2 +
            (1 - scores["structure_complexity"]) * 0.2 +
            scores["human_relatability"] * 0.3
        )

        return scores


class ModelOptimizer:
    """General model optimizer that selects appropriate strategies."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name.lower()
        self.deepseek_optimizer = DeepSeekV3Optimizer()

    def optimize_for_model(self, prompt: str, question_type: str = "general") -> str:
        """Optimize prompt based on model characteristics."""
        if "deepseek" in self.model_name:
            return self.deepseek_optimizer.optimize_prompt_for_deepseek(prompt, question_type)
        if "claude" in self.model_name:
            # Claude generally produces good responses, minimal optimization needed
            return f"{prompt}\n\n请用自然、易懂的语言回答，保持人性化的表达。"
        if "gpt" in self.model_name:
            # GPT tends to be verbose, ask for conciseness
            return f"{prompt}\n\n请简洁明了地回答，避免过度展开。"
        # Generic optimization
        return f"{prompt}\n\n请用简单、实用的语言回答。"

    def post_process_response(self, response: str) -> str:
        """Post-process response based on model characteristics."""
        if "deepseek" in self.model_name:
            return self.deepseek_optimizer.post_process_deepseek_response(response)
        # Basic post-processing for other models
        return response

    def get_quality_assessment(self, response: str, question: str) -> dict[str, float]:
        """Get quality assessment for the response."""
        if "deepseek" in self.model_name:
            return self.deepseek_optimizer.assess_response_quality(response, question)
        # Basic assessment for other models
        return {"overall_quality": 0.7}  # Assume decent quality


# Factory function
def create_model_optimizer(model_name: str) -> ModelOptimizer:
    """Create appropriate model optimizer."""
    return ModelOptimizer(model_name)


# Quick test function
def test_deepseek_optimization() -> None:
    """Test the DeepSeek optimization."""
    optimizer = DeepSeekV3Optimizer()

    # Test philosophical question optimization
    original_prompt = "如果生命终将结束，我们为什么要活着？"
    optimizer.optimize_prompt_for_deepseek(original_prompt, "philosophical")


    # Test response post-processing
    problematic_response = """
# 🌐 生命意义综合求解框架

## 🧬 生物维度锚定点
**进化现实性平衡**
- 采用基因编辑应同步建立全球伦理监察联盟
- 将杏仁核激活能量导向艺术创造与科学突破

## 🧠 认知重建模型
| 客观有限性认知 | 主观永恒建构 |
|--------------|--------------|
| 端粒损耗速度监测 | 脑机接口记忆晶格存储 |

## 🚀 执行路线图
**三期阶梯战略**
- 建立全球生命档案链（GLAC）
- 量子生物芯片临床应用
"""

    optimizer.post_process_deepseek_response(problematic_response)


    # Quality assessment
    quality = optimizer.assess_response_quality(problematic_response, original_prompt)
    for _metric, _score in quality.items():
        pass


if __name__ == "__main__":
    test_deepseek_optimization()
