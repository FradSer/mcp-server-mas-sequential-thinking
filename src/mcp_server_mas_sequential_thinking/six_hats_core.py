"""Six Thinking Hats Core Agent Architecture

基于 Edward de Bono 六帽思维的 Agent 核心实现。
严格遵循"一次一顶帽子"原则，支持单帽到六帽的智能序列。
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

from agno.agent import Agent
from agno.models.base import Model
from agno.tools.reasoning import ReasoningTools

from .logging_config import get_logger

logger = get_logger(__name__)


class HatColor(Enum):
    """六顶思维帽子的颜色枚举"""
    WHITE = "white"    # 事实与数据
    RED = "red"        # 情感与直觉
    BLACK = "black"    # 批判与风险
    YELLOW = "yellow"  # 乐观与价值
    GREEN = "green"    # 创造与创新
    BLUE = "blue"      # 元认知与整合


class HatComplexity(Enum):
    """处理复杂度级别"""
    SINGLE = "single"      # 单帽模式
    DOUBLE = "double"      # 双帽序列
    TRIPLE = "triple"      # 三帽序列
    FULL = "full"          # 完整六帽


@dataclass(frozen=True)
class HatTimingConfig:
    """帽子使用时间配置"""
    color: HatColor
    default_time_seconds: int
    min_time_seconds: int
    max_time_seconds: int
    is_quick_reaction: bool = False  # 是否为快速反应模式（如红帽）


# 时间配置常量
HAT_TIMING_CONFIGS = {
    HatColor.WHITE: HatTimingConfig(HatColor.WHITE, 120, 60, 300, False),
    HatColor.RED: HatTimingConfig(HatColor.RED, 30, 15, 60, True),  # 快速直觉
    HatColor.BLACK: HatTimingConfig(HatColor.BLACK, 120, 60, 240, False),
    HatColor.YELLOW: HatTimingConfig(HatColor.YELLOW, 120, 60, 240, False),
    HatColor.GREEN: HatTimingConfig(HatColor.GREEN, 240, 120, 360, False),  # 创意需要更多时间
    HatColor.BLUE: HatTimingConfig(HatColor.BLUE, 60, 30, 120, False),
}


@dataclass(frozen=True)
class SixHatsCapability:
    """六帽思维能力定义"""

    hat_color: HatColor
    role: str
    description: str
    role_description: str

    # 认知特征
    thinking_mode: str
    cognitive_focus: str
    output_style: str

    # 时间管理
    timing_config: HatTimingConfig

    # 增强特性
    tools: list[type] = None
    reasoning_level: int = 1
    memory_enabled: bool = False

    def __post_init__(self):
        if self.tools is None:
            object.__setattr__(self, "tools", [ReasoningTools])

    def get_instructions(self, context: str = "", previous_results: dict = None) -> list[str]:
        """生成特定帽子的指令"""
        base_instructions = [
            f"You are wearing the {self.hat_color.value.upper()} HAT in the Six Thinking Hats methodology.",
            f"Role: {self.role}",
            f"Cognitive Focus: {self.cognitive_focus}",
            "",
            "CRITICAL RULES:",
            f"1. Think ONLY in {self.hat_color.value} hat mode - no other thinking styles allowed",
            f"2. Time limit: {self.timing_config.default_time_seconds} seconds for focused thinking",
            f"3. Output style: {self.output_style}",
            "",
            f"Your specific responsibility: {self.role_description}",
        ]

        # 添加特定帽子的详细指令
        specific_instructions = self._get_specific_instructions()
        base_instructions.extend(specific_instructions)

        # 添加上下文和前置结果
        if context:
            base_instructions.extend([
                "",
                f"Context: {context}",
            ])

        if previous_results:
            base_instructions.extend([
                "",
                "Previous thinking results from other hats:",
                *[f"  {hat}: {result[:100]}..." for hat, result in previous_results.items()]
            ])

        return base_instructions

    @abstractmethod
    def _get_specific_instructions(self) -> list[str]:
        """获取特定帽子的详细指令"""


class WhiteHatCapability(SixHatsCapability):
    """白帽能力：事实与数据"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.WHITE,
            role="Factual Information Processor",
            description="收集和处理客观事实与数据",
            role_description="I focus only on objective facts and data. I provide neutral information without personal interpretation.",
            thinking_mode="analytical_factual",
            cognitive_focus="纯粹信息处理，零情感或判断",
            output_style="客观事实列表，数据驱动的信息",
            timing_config=HAT_TIMING_CONFIGS[HatColor.WHITE],
            reasoning_level=2,
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "WHITE HAT SPECIFIC GUIDELINES:",
            "• Use simple statements to present facts: 'The data shows...', 'Known information is...'",
            "• Avoid technical jargon, explain data in everyday language",
            "• Present only verified facts and objective data",
            "• Avoid opinions, interpretations, or emotional reactions",
            "• Identify what information is missing and needed",
            "• Separate facts from assumptions clearly",
            "",
            "FORBIDDEN in white hat mode:",
            "- Personal opinions or judgments",
            "- Emotional responses or gut feelings",
            "- Speculation or 'what if' scenarios",
            "- Value judgments (good/bad, right/wrong)",
        ]


class RedHatCapability(SixHatsCapability):
    """红帽能力：情感与直觉"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.RED,
            role="Intuitive Emotional Processor",
            description="提供情感响应与直觉洞察",
            role_description="I express intuition and emotional reactions. No reasoning needed, just share feelings.",
            thinking_mode="intuitive_emotional",
            cognitive_focus="情感智能和直觉处理",
            output_style="直觉反应，情感表达，人性化视角",
            timing_config=HAT_TIMING_CONFIGS[HatColor.RED],
            reasoning_level=1,  # 最低理性，最高直觉
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "RED HAT SPECIFIC GUIDELINES:",
            "• Start responses with 'I feel...', 'My intuition tells me...', 'My gut reaction is...'",
            "• Keep expressions brief and powerful - 30-second emotional snapshots",
            "• Express immediate gut reactions and feelings",
            "• Share intuitive hunches without justification",
            "• Include visceral, immediate responses",
            "• NO need to explain or justify feelings",
            "",
            "ENCOURAGED in red hat mode:",
            "- First impressions and gut reactions",
            "- Emotional responses to ideas or situations",
            "- Intuitive concerns or excitement",
            "- 'Sixth sense' about what might work",
            "",
            "Remember: This is a 30-second emotional snapshot, not analysis!",
        ]


class BlackHatCapability(SixHatsCapability):
    """黑帽能力：批判与风险"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.BLACK,
            role="Critical Risk Assessor",
            description="批判性分析与风险识别",
            role_description="I identify risks and problems. Critical but not pessimistic, I point out real difficulties.",
            thinking_mode="critical_analytical",
            cognitive_focus="批判性思维和风险评估",
            output_style="尖锐质疑，风险警告，逻辑检验",
            timing_config=HAT_TIMING_CONFIGS[HatColor.BLACK],
            reasoning_level=3,  # 高度逻辑推理
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "BLACK HAT SPECIFIC GUIDELINES:",
            "• Point out specific possible problems, not general pessimism",
            "• Use phrases like 'The risk is...', 'This could fail because...', 'A problem might be...'",
            "• Identify potential problems, risks, and weaknesses",
            "• Challenge assumptions and look for logical flaws",
            "• Consider worst-case scenarios and failure modes",
            "• Provide logical reasons for all concerns raised",
            "",
            "KEY AREAS TO EXAMINE:",
            "- Logical inconsistencies in arguments",
            "- Practical obstacles and implementation challenges",
            "- Resource constraints and limitations",
            "- Potential negative consequences",
            "- Missing information or unproven assumptions",
            "",
            "Note: Be critical but constructive - identify real problems, not just pessimism.",
        ]


class YellowHatCapability(SixHatsCapability):
    """黄帽能力：乐观与价值"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.YELLOW,
            role="Optimistic Value Explorer",
            description="积极思考与价值发现",
            role_description="I find value and opportunities. Realistic optimism, I discover genuine benefits.",
            thinking_mode="optimistic_constructive",
            cognitive_focus="积极心理学和机会识别",
            output_style="积极探索，价值发现，机会识别",
            timing_config=HAT_TIMING_CONFIGS[HatColor.YELLOW],
            reasoning_level=2,
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "YELLOW HAT SPECIFIC GUIDELINES:",
            "• Point out specific feasible benefits, not empty praise",
            "• Use phrases like 'The benefit is...', 'This creates... value', 'An opportunity here is...'",
            "• Focus on benefits, values, and positive outcomes",
            "• Explore best-case scenarios and opportunities",
            "• Identify feasible positive possibilities",
            "• Provide logical reasons for optimism",
            "",
            "KEY AREAS TO EXPLORE:",
            "- Benefits and positive outcomes",
            "- Opportunities for growth or improvement",
            "- Feasible best-case scenarios",
            "- Value creation possibilities",
            "- Strengths and positive aspects",
            "- Why this could work well",
            "",
            "Note: Be realistically optimistic - find genuine value, not false hope.",
        ]


class GreenHatCapability(SixHatsCapability):
    """绿帽能力：创造与创新"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.GREEN,
            role="Creative Innovation Generator",
            description="创造性思维与创新方案",
            role_description="I generate new ideas and alternative approaches. I break conventional limits and explore possibilities.",
            thinking_mode="creative_generative",
            cognitive_focus="发散思维和创造力",
            output_style="新颖想法，创新方案，另类思考",
            timing_config=HAT_TIMING_CONFIGS[HatColor.GREEN],
            reasoning_level=2,
            memory_enabled=True,  # 创意可能需要记忆组合
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "GREEN HAT SPECIFIC GUIDELINES:",
            "• Provide 3-5 specific creative ideas that could work",
            "• Use phrases like 'What if...', 'Another approach could be...', 'An alternative is...'",
            "• Generate new ideas, alternatives, and creative solutions",
            "• Think laterally - explore unconventional approaches",
            "• Break normal thinking patterns and assumptions",
            "• Suggest modifications, improvements, or entirely new approaches",
            "",
            "CREATIVE TECHNIQUES TO USE:",
            "- Lateral thinking and analogies",
            "- Random word associations",
            "- 'What if' scenarios and thought experiments",
            "- Reversal thinking (what's the opposite?)",
            "- Combination of unrelated elements",
            "- Alternative perspectives and viewpoints",
            "",
            "Note: Quantity over quality - generate many ideas without judgment.",
        ]


class BlueHatCapability(SixHatsCapability):
    """蓝帽能力：元认知与整合"""

    def __init__(self):
        super().__init__(
            hat_color=HatColor.BLUE,
            role="Metacognitive Orchestrator",
            description="思维过程管理与综合统筹",
            role_description="I integrate all perspectives and provide the final balanced answer. My output is what users see - it must be practical and human-friendly.",
            thinking_mode="metacognitive_synthetic",
            cognitive_focus="元认知和执行控制",
            output_style="综合整合，过程管理，统一结论",
            timing_config=HAT_TIMING_CONFIGS[HatColor.BLUE],
            reasoning_level=3,  # 最高层次的元认知
            memory_enabled=True,  # 需要记住所有其他帽子的结果
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "BLUE HAT SPECIFIC GUIDELINES:",
            "• START BY RESTATING THE ORIGINAL QUESTION - this is your north star",
            "• Your primary goal: Answer the original question using insights from other hats",
            "• Avoid generic rehashing - focus specifically on the question asked",
            "• Use other hats' contributions as evidence/perspectives to build your answer",
            "• Provide practical, actionable insights users can understand",
            "",
            "CRITICAL QUESTION-FOCUSED APPROACH:",
            "1. Begin with: 'Based on our analysis of [original question]...'",
            "2. Extract ONLY the insights from other hats that directly address the question",
            "3. Ignore generic statements - focus on question-relevant content",
            "4. Build a coherent answer that uses multiple perspectives as support",
            "5. End with a clear, direct response to what was originally asked",
            "",
            "KEY RESPONSIBILITIES:",
            "- Return to the original question and answer it directly",
            "- Use other hats' insights as building blocks for your answer",
            "- Synthesize perspectives into a cohesive response to the specific question",
            "- Avoid academic summarization - focus on practical question-answering",
            "- Ensure your entire response serves the original question",
            "",
            "FINAL OUTPUT REQUIREMENTS:",
            "• This is the user's ONLY answer - it must directly address their question",
            "• Don't just summarize - synthesize into a clear answer",
            "• Remove content that doesn't directly relate to the original question",
            "• For philosophical questions: provide thoughtful answers, not just analysis",
            "• Structure: Question restatement → Key insights → Direct answer",
            "• Control length: 300-600 words, focused on answering the question",
        ]


class SixHatsAgentFactory:
    """六帽思维 Agent 工厂"""

    # 六个帽子能力映射
    HAT_CAPABILITIES = {
        HatColor.WHITE: WhiteHatCapability(),
        HatColor.RED: RedHatCapability(),
        HatColor.BLACK: BlackHatCapability(),
        HatColor.YELLOW: YellowHatCapability(),
        HatColor.GREEN: GreenHatCapability(),
        HatColor.BLUE: BlueHatCapability(),
    }

    def __init__(self):
        self._agent_cache = {}  # 缓存已创建的agents

    def create_hat_agent(
        self,
        hat_color: HatColor,
        model: Model,
        context: str = "",
        previous_results: dict = None,
        **kwargs
    ) -> Agent:
        """创建特定颜色的帽子Agent"""
        capability = self.HAT_CAPABILITIES[hat_color]

        # 生成缓存键
        cache_key = f"{hat_color.value}_{model.__class__.__name__}_{hash(context)}"

        if cache_key in self._agent_cache:
            # 返回缓存的agent但更新指令
            agent = self._agent_cache[cache_key]
            agent.instructions = capability.get_instructions(context, previous_results)
            return agent

        # 创建新的agent
        agent = Agent(
            name=f"{hat_color.value.title()}HatAgent",
            role=capability.role,
            description=capability.description,
            model=model,
            tools=capability.tools,
            instructions=capability.get_instructions(context, previous_results),
            markdown=True,
            **kwargs
        )

        # 添加特殊配置
        if capability.memory_enabled:
            agent.enable_user_memories = True

        # 缓存agent
        self._agent_cache[cache_key] = agent

        logger.info(f"Created {hat_color.value} hat agent with {capability.timing_config.default_time_seconds}s time limit")
        return agent

    def get_timing_config(self, hat_color: HatColor) -> HatTimingConfig:
        """获取特定帽子的时间配置"""
        return HAT_TIMING_CONFIGS[hat_color]

    def get_available_hats(self) -> list[HatColor]:
        """获取所有可用的帽子颜色"""
        return list(self.HAT_CAPABILITIES.keys())

    def clear_cache(self):
        """清除agent缓存"""
        self._agent_cache.clear()
        logger.info("Hat agent cache cleared")


# 全局工厂实例
_hat_factory = SixHatsAgentFactory()


# 便利函数
def create_hat_agent(hat_color: HatColor, model: Model, **kwargs) -> Agent:
    """创建帽子Agent的便利函数"""
    return _hat_factory.create_hat_agent(hat_color, model, **kwargs)


def get_hat_timing(hat_color: HatColor) -> HatTimingConfig:
    """获取帽子时间配置的便利函数"""
    return _hat_factory.get_timing_config(hat_color)


def get_all_hat_colors() -> list[HatColor]:
    """获取所有帽子颜色的便利函数"""
    return _hat_factory.get_available_hats()
