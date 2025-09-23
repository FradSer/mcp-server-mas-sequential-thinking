"""
TDD测试 - StepExecutorMixin重构组件

遵循TDD流程：RED → GREEN → REFACTOR
测试重构后的步骤执行器混入类功能
"""

from typing import Any

from agno.workflow.types import StepOutput

from mcp_server_mas_sequential_thinking.routing.agno_workflow_router import (
    StepExecutorMixin,
)


class TestStepExecutorMixin:
    """测试StepExecutorMixin的所有方法"""

    def setup_method(self):
        """为每个测试方法设置"""
        self.mixin = StepExecutorMixin()

    def test_create_step_output_basic_success(self):
        """测试创建基础成功的步骤输出"""
        content = "Processing completed successfully"
        strategy = "single_agent"

        result = self.mixin._create_step_output(content, strategy)

        assert isinstance(result, StepOutput)
        assert result.content["result"] == content
        assert result.success is True
        assert result.content["strategy"] == strategy
        assert result.content["complexity"] == 0
        assert result.error is None

    def test_create_step_output_with_session_state(self):
        """测试使用会话状态创建步骤输出"""
        content = "Processing with session state"
        strategy = "multi_agent"
        session_state = {"current_complexity_score": 75.5, "other_data": "test"}

        result = self.mixin._create_step_output(
            content=content, strategy=strategy, session_state=session_state
        )

        assert result.content["complexity"] == 75.5
        assert result.content["strategy"] == strategy

    def test_create_step_output_with_specialists(self):
        """测试包含专家信息的步骤输出"""
        content = "Multi-agent processing"
        strategy = "hybrid"
        specialists = ["planner", "analyzer", "critic"]

        result = self.mixin._create_step_output(
            content=content, strategy=strategy, specialists=specialists
        )

        assert result.content["specialists"] == specialists
        assert result.content["strategy"] == strategy

    def test_create_step_output_failure_case(self):
        """测试创建失败的步骤输出"""
        content = "Processing failed"
        strategy = "single_agent"
        error_msg = "Connection timeout"

        result = self.mixin._create_step_output(
            content=content, strategy=strategy, success=False, error=error_msg
        )

        assert result.success is False
        assert result.error == error_msg
        assert result.content["result"] == content

    def test_create_step_output_none_session_state(self):
        """测试None会话状态的处理"""
        content = "Processing without session"
        strategy = "test_strategy"

        result = self.mixin._create_step_output(
            content=content, strategy=strategy, session_state=None
        )

        assert result.content["complexity"] == 0
        assert result.content["strategy"] == strategy

    def test_update_session_state_success(self):
        """测试成功更新会话状态"""
        session_state = {}
        strategy = "multi_agent"
        completed_key = "multi_agent_completed"

        self.mixin._update_session_state(session_state, strategy, completed_key)

        assert session_state[completed_key] is True
        assert session_state["processing_strategy"] == strategy

    def test_update_session_state_none_session(self):
        """测试更新None会话状态（应该安全处理）"""
        # 这应该不会抛出异常
        self.mixin._update_session_state(None, "test_strategy", "test_key")
        # 没有异常就是成功

    def test_update_session_state_existing_data(self):
        """测试更新已有数据的会话状态"""
        session_state = {
            "existing_key": "existing_value",
            "processing_strategy": "old_strategy",
        }
        strategy = "hybrid"
        completed_key = "hybrid_completed"

        self.mixin._update_session_state(session_state, strategy, completed_key)

        # 应该保留现有数据
        assert session_state["existing_key"] == "existing_value"
        # 应该更新相关字段
        assert session_state[completed_key] is True
        assert session_state["processing_strategy"] == strategy

    def test_handle_execution_error_basic(self):
        """测试基础错误处理"""
        error = Exception("Test error occurred")
        strategy = "single_agent"

        result = self.mixin._handle_execution_error(error, strategy)

        assert isinstance(result, StepOutput)
        assert result.success is False
        assert "Single agent processing failed: Test error occurred" in result.content
        assert result.error == "Test error occurred"

    def test_handle_execution_error_complex_exception(self):
        """测试复杂异常的错误处理"""
        error = ValueError("Invalid input parameter: expected int, got str")
        strategy = "multi_agent"

        result = self.mixin._handle_execution_error(error, strategy)

        assert result.success is False
        assert "Multi agent processing failed" in result.content
        assert "Invalid input parameter" in result.content
        assert result.error == "Invalid input parameter: expected int, got str"

    def test_handle_execution_error_different_strategies(self):
        """测试不同策略的错误处理"""
        error = RuntimeError("Runtime error")
        strategies = ["single_agent", "hybrid", "multi_agent", "parallel_analysis"]

        for strategy in strategies:
            result = self.mixin._handle_execution_error(error, strategy)

            assert result.success is False
            assert strategy.replace("_", " ").capitalize() in result.content
            assert "processing failed" in result.content


class TestStepExecutorMixinIntegration:
    """测试StepExecutorMixin的集成用法"""

    def test_mixin_inheritance(self):
        """测试StepExecutorMixin可以被正确继承"""

        class TestExecutor(StepExecutorMixin):
            def execute_step(self, content: str, strategy: str):
                return self._create_step_output(content, strategy)

            def handle_error(self, error: Exception, strategy: str):
                return self._handle_execution_error(error, strategy)

        executor = TestExecutor()

        # 测试成功执行
        result = executor.execute_step("test content", "test_strategy")
        assert isinstance(result, StepOutput)
        assert result.success is True

        # 测试错误处理
        error_result = executor.handle_error(Exception("test error"), "test_strategy")
        assert isinstance(error_result, StepOutput)
        assert error_result.success is False

    def test_complete_workflow_simulation(self):
        """测试完整的工作流程模拟"""

        class MockWorkflowStep(StepExecutorMixin):
            def execute(
                self, content: str, session_state: dict[str, Any] | None = None
            ):
                try:
                    # 模拟处理
                    if content == "error_case":
                        raise ValueError("Simulated error")

                    # 更新会话状态
                    self._update_session_state(
                        session_state, "test_strategy", "test_completed"
                    )

                    # 返回成功结果
                    return self._create_step_output(
                        content=f"Processed: {content}",
                        strategy="test_strategy",
                        session_state=session_state,
                        specialists=["test_specialist"],
                    )

                except Exception as e:
                    return self._handle_execution_error(e, "test_strategy")

        workflow = MockWorkflowStep()
        session_state = {"initial_data": "test"}

        # 测试成功案例
        result = workflow.execute("valid_content", session_state)
        assert result.success is True
        assert "Processed: valid_content" in result.content["result"]
        assert session_state["test_completed"] is True
        assert session_state["processing_strategy"] == "test_strategy"

        # 测试错误案例
        error_result = workflow.execute("error_case", session_state)
        assert error_result.success is False
        assert "Test strategy processing failed" in error_result.content


class TestStepExecutorMixinEdgeCases:
    """测试StepExecutorMixin的边界情况和错误处理"""

    def test_create_step_output_with_all_parameters(self):
        """测试使用所有参数创建步骤输出"""
        mixin = StepExecutorMixin()

        content = "Full parameter test"
        strategy = "comprehensive"
        session_state = {"current_complexity_score": 99.9, "data": "test"}
        error_msg = "Warning message"
        specialists = ["expert1", "expert2"]

        result = mixin._create_step_output(
            content=content,
            strategy=strategy,
            success=True,
            session_state=session_state,
            error=error_msg,
            specialists=specialists,
        )

        assert result.content["result"] == content
        assert result.success is True
        assert result.error == error_msg
        assert result.content["strategy"] == strategy
        assert result.content["complexity"] == 99.9
        assert result.content["specialists"] == specialists

    def test_session_state_modification_isolation(self):
        """测试会话状态修改的隔离性"""
        mixin = StepExecutorMixin()

        original_state = {"existing": "value"}
        session_state = original_state.copy()

        mixin._update_session_state(session_state, "test", "test_key")

        # 原始状态不应该被修改（如果传入的是副本）
        assert "test_key" not in original_state
        # 修改的状态应该包含新值
        assert session_state["test_key"] is True

    def test_error_handling_with_nested_exceptions(self):
        """测试嵌套异常的错误处理"""
        mixin = StepExecutorMixin()

        # 创建嵌套异常
        inner_error = ValueError("Inner error")
        outer_error = RuntimeError("Outer error")
        outer_error.__cause__ = inner_error

        result = mixin._handle_execution_error(outer_error, "nested_test")

        assert result.success is False
        assert "Nested test processing failed" in result.content
        assert result.error == "Outer error"

    def test_step_output_data_types(self):
        """测试步骤输出的数据类型验证"""
        mixin = StepExecutorMixin()

        result = mixin._create_step_output(
            content="type test", strategy="validation", session_state={"score": 42.5}
        )

        # 验证数据类型
        assert isinstance(result.content, dict)
        assert isinstance(result.success, bool)
        assert isinstance(result.content["result"], str)
        assert isinstance(result.content["complexity"], (int, float))

    def test_empty_string_handling(self):
        """测试空字符串的处理"""
        mixin = StepExecutorMixin()

        result = mixin._create_step_output("", "empty_test")

        assert result.content["result"] == ""
        assert result.success is True
        assert result.content["strategy"] == "empty_test"
