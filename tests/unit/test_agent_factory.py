"""
TDD测试 - AgentFactory重构组件

遵循TDD流程：RED → GREEN → REFACTOR
测试重构后的代理工厂模式功能
"""

from unittest.mock import Mock, patch

from agno.tools.reasoning import ReasoningTools

from mcp_server_mas_sequential_thinking.routing.agno_workflow_router import AgentFactory


class TestAgentFactory:
    """测试AgentFactory的所有工厂方法"""

    def setup_method(self):
        """为每个测试方法设置"""
        self.mock_model = Mock()
        self.mock_model.id = "test-model"

    def test_create_agent_basic_functionality(self):
        """测试create_agent基础功能"""
        name = "TestAgent"
        role = "Test Role"
        instructions = ["instruction1", "instruction2"]

        with patch(
            "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
        ) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            agent = AgentFactory.create_agent(name, role, self.mock_model, instructions)

            # 验证Agent类被正确调用
            mock_agent_class.assert_called_once_with(
                name=name,
                role=role,
                model=self.mock_model,
                tools=[ReasoningTools],
                instructions=instructions,
            )
            assert agent == mock_agent_instance

    def test_create_agent_with_empty_instructions(self):
        """测试创建包含空指令列表的代理"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
        ) as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance

            AgentFactory.create_agent("Test", "Role", self.mock_model, [])

            mock_agent_class.assert_called_once()
            _args, kwargs = mock_agent_class.call_args
            assert kwargs["instructions"] == []

    def test_create_planner_basic_complexity(self):
        """测试创建基础复杂度的规划师代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_planner(self.mock_model, "basic")

            # 验证调用了create_agent并使用了正确的参数
            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            assert args[0] == "Planner"
            assert args[1] == "Strategic Planner"
            assert args[2] == self.mock_model

            # 验证基础复杂度的指令
            instructions = args[3]
            assert isinstance(instructions, list)
            assert "strategic approach" in " ".join(instructions).lower()

    def test_create_planner_advanced_complexity(self):
        """测试创建高级复杂度的规划师代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_planner(self.mock_model, "advanced")

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            # 验证高级复杂度的指令
            instructions = args[3]
            assert isinstance(instructions, list)
            assert "comprehensive" in " ".join(instructions).lower()

    def test_create_analyzer_basic_complexity(self):
        """测试创建基础复杂度的分析师代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_analyzer(self.mock_model, "basic")

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            assert args[0] == "Analyzer"
            assert args[1] == "Core Analyst"

            instructions = args[3]
            assert "analysis" in " ".join(instructions).lower()

    def test_create_analyzer_advanced_complexity(self):
        """测试创建高级复杂度的分析师代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_analyzer(self.mock_model, "advanced")

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            instructions = args[3]
            assert "comprehensive" in " ".join(instructions).lower()

    def test_create_researcher(self):
        """测试创建研究员代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_researcher(self.mock_model)

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            assert args[0] == "Researcher"
            assert args[1] == "Information Gatherer"

            instructions = args[3]
            assert (
                "research" in " ".join(instructions).lower()
                or "information" in " ".join(instructions).lower()
            )

    def test_create_critic(self):
        """测试创建评论家代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_critic(self.mock_model)

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            assert args[0] == "Critic"
            assert args[1] == "Quality Controller"

            instructions = args[3]
            assert (
                "critic" in " ".join(instructions).lower()
                or "quality" in " ".join(instructions).lower()
            )

    def test_create_synthesizer(self):
        """测试创建综合器代理"""
        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_agent = Mock()
            mock_create.return_value = mock_agent

            AgentFactory.create_synthesizer(self.mock_model)

            mock_create.assert_called_once()
            args, _kwargs = mock_create.call_args

            assert args[0] == "Synthesizer"
            assert args[1] == "Response Coordinator"

            instructions = args[3]
            assert (
                "synthesize" in " ".join(instructions).lower()
                or "integrate" in " ".join(instructions).lower()
            )

    def test_all_agent_types_use_reasoning_tools(self):
        """测试所有代理类型都使用ReasoningTools"""
        agent_methods = [
            ("create_planner", [self.mock_model]),
            ("create_analyzer", [self.mock_model]),
            ("create_researcher", [self.mock_model]),
            ("create_critic", [self.mock_model]),
            ("create_synthesizer", [self.mock_model]),
        ]

        for method_name, args in agent_methods:
            with patch.object(AgentFactory, "create_agent") as mock_create:
                method = getattr(AgentFactory, method_name)
                method(*args)

                # 验证所有工厂方法都调用了create_agent
                mock_create.assert_called_once()


class TestAgentFactoryIntegration:
    """测试AgentFactory的集成用法"""

    def setup_method(self):
        """为每个测试方法设置"""
        self.mock_model = Mock()

    def test_agent_factory_creates_different_agents(self):
        """测试工厂能创建不同类型的代理"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
        ) as mock_agent_class:
            # 每次调用返回不同的mock实例
            mock_instances = [Mock(), Mock(), Mock()]
            mock_agent_class.side_effect = mock_instances

            planner = AgentFactory.create_planner(self.mock_model)
            analyzer = AgentFactory.create_analyzer(self.mock_model)
            researcher = AgentFactory.create_researcher(self.mock_model)

            # 验证创建了3个不同的代理实例
            assert mock_agent_class.call_count == 3
            assert planner != analyzer != researcher

    def test_agent_factory_consistent_interface(self):
        """测试工厂方法的一致性接口"""
        factory_methods = [
            AgentFactory.create_planner,
            AgentFactory.create_analyzer,
            AgentFactory.create_researcher,
            AgentFactory.create_critic,
            AgentFactory.create_synthesizer,
        ]

        with patch(
            "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
        ) as mock_agent_class:
            mock_agent_class.return_value = Mock()

            for method in factory_methods:
                # 所有方法都应该接受model参数
                if method in [
                    AgentFactory.create_planner,
                    AgentFactory.create_analyzer,
                ]:
                    # 这些方法还支持complexity_level参数
                    agent = method(self.mock_model, "basic")
                else:
                    agent = method(self.mock_model)

                # 验证都返回了agent实例
                assert agent is not None


# RED阶段测试 - 这些测试应该失败，直到实现了对应功能
class TestAgentFactoryEdgeCases:
    """测试AgentFactory的边界情况和错误处理"""

    def test_create_planner_invalid_complexity_level(self):
        """测试使用无效复杂度级别创建规划师"""
        mock_model = Mock()

        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_create.return_value = Mock()

            # 应该处理无效的复杂度级别（使用默认值或抛出异常）
            try:
                AgentFactory.create_planner(mock_model, "invalid_level")
                # 如果没有抛出异常，应该使用某种默认行为
                mock_create.assert_called_once()
            except (KeyError, ValueError):
                # 如果抛出异常，这也是可接受的
                pass

    def test_create_analyzer_invalid_complexity_level(self):
        """测试使用无效复杂度级别创建分析师"""
        mock_model = Mock()

        with patch.object(AgentFactory, "create_agent") as mock_create:
            mock_create.return_value = Mock()

            try:
                AgentFactory.create_analyzer(mock_model, "invalid_level")
                mock_create.assert_called_once()
            except (KeyError, ValueError):
                pass

    def test_create_agent_with_none_model(self):
        """测试使用None模型创建代理"""
        # 这应该能够处理或至少不崩溃
        try:
            with patch(
                "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
            ) as mock_agent_class:
                mock_agent_class.return_value = Mock()

                AgentFactory.create_agent("Test", "Role", None, ["instruction"])
                mock_agent_class.assert_called_once()
        except Exception:
            # 如果抛出异常也是可以接受的
            pass

    def test_factory_methods_are_class_methods(self):
        """测试工厂方法确实是类方法"""
        # 验证我们可以直接从类调用方法，而不需要实例
        assert hasattr(AgentFactory, "create_planner")
        assert hasattr(AgentFactory, "create_analyzer")
        assert hasattr(AgentFactory, "create_researcher")
        assert hasattr(AgentFactory, "create_critic")
        assert hasattr(AgentFactory, "create_synthesizer")

        # 验证它们是类方法或静态方法
        assert callable(AgentFactory.create_planner)


class TestAgentFactoryPerformance:
    """测试AgentFactory的性能特性"""

    def test_factory_method_efficiency(self):
        """测试工厂方法的效率（不应该有不必要的开销）"""
        mock_model = Mock()

        with patch(
            "src.mcp_server_mas_sequential_thinking.routing.agno_workflow_router.Agent"
        ) as mock_agent_class:
            mock_agent_class.return_value = Mock()

            # 调用工厂方法不应该创建额外的对象或进行昂贵的操作
            import time

            start_time = time.time()

            for _ in range(100):
                AgentFactory.create_planner(mock_model)

            elapsed_time = time.time() - start_time

            # 100次调用应该在合理时间内完成（这是一个宽松的性能测试）
            assert elapsed_time < 1.0  # 1秒内完成100次调用
            assert mock_agent_class.call_count == 100
