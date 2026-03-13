"""Unit tests for services/context_builder.py ContextBuilder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.services.context_builder import ContextBuilder


def make_thought(
    n: int,
    thought_text: str = "test thought",
    is_revision: bool = False,
    branch_from: int | None = None,
    branch_id: str | None = None,
) -> ThoughtData:
    return ThoughtData(
        thought=thought_text,
        thoughtNumber=n,
        totalThoughts=10,
        nextThoughtNeeded=True,
        isRevision=is_revision,
        branchFromThought=branch_from,
        branchId=branch_id,
        needsMoreThoughts=False,
    )


def make_session_mock(thought_content: str = "original thought") -> MagicMock:
    session = MagicMock()
    session.find_thought_content = AsyncMock(return_value=thought_content)
    session.thought_history = []
    return session


class TestContextBuilderBuildContextPrompt:
    """Tests for ContextBuilder.build_context_prompt."""

    @pytest.mark.asyncio
    async def test_standard_thought(self):
        session = make_session_mock()
        builder = ContextBuilder(session)
        thought = make_thought(1)
        result = await builder.build_context_prompt(thought)
        assert "Process Thought #1:" in result
        assert "test thought" in result

    @pytest.mark.asyncio
    async def test_revision_thought(self):
        session = make_session_mock("original content")
        builder = ContextBuilder(session)
        thought = make_thought(2, is_revision=True, branch_from=1)
        result = await builder.build_context_prompt(thought)
        assert "REVISION of Thought #1" in result
        assert "original content" in result

    @pytest.mark.asyncio
    async def test_branch_thought(self):
        session = make_session_mock("branch origin")
        builder = ContextBuilder(session)
        thought = make_thought(3, branch_from=1, branch_id="feature_x")
        result = await builder.build_context_prompt(thought)
        assert "BRANCH" in result
        assert "feature_x" in result
        assert "branch origin" in result

    @pytest.mark.asyncio
    async def test_find_thought_content_safe_handles_exception(self):
        session = MagicMock()
        session.find_thought_content = AsyncMock(side_effect=RuntimeError("not found"))
        builder = ContextBuilder(session)
        thought = make_thought(2, is_revision=True, branch_from=1)
        result = await builder.build_context_prompt(thought)
        assert "[not found]" in result


class TestContextBuilderLogContextBuilding:
    """Tests for ContextBuilder.log_context_building."""

    @pytest.mark.asyncio
    async def test_log_standard_thought(self):
        session = make_session_mock()
        session.thought_history = []
        builder = ContextBuilder(session)
        thought = make_thought(1)
        # Should not raise
        await builder.log_context_building(thought, "prompt text")

    @pytest.mark.asyncio
    async def test_log_revision_thought(self):
        session = make_session_mock("original")
        session.thought_history = [make_thought(1)]
        builder = ContextBuilder(session)
        thought = make_thought(2, is_revision=True, branch_from=1)
        await builder.log_context_building(thought, "revision prompt")

    @pytest.mark.asyncio
    async def test_log_branch_thought(self):
        session = make_session_mock("origin")
        session.thought_history = [make_thought(1)]
        builder = ContextBuilder(session)
        thought = make_thought(3, branch_from=1, branch_id="branch_b")
        await builder.log_context_building(thought, "branch prompt")
