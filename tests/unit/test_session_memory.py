"""Unit tests for core/session.py SessionMemory."""

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.core.session import SessionMemory


def make_thought(n: int, branch_from: int | None = None, branch_id: str | None = None, is_revision: bool = False) -> ThoughtData:
    return ThoughtData(
        thought=f"thought {n}",
        thoughtNumber=n,
        totalThoughts=10,
        nextThoughtNeeded=True,
        isRevision=is_revision,
        branchFromThought=branch_from,
        branchId=branch_id,
        needsMoreThoughts=False,
    )


class TestSessionMemory:
    """Tests for SessionMemory."""

    @pytest.mark.asyncio
    async def test_add_thought_basic(self):
        session = SessionMemory()
        t = make_thought(1)
        await session.add_thought(t)
        assert len(session.thought_history) == 1
        assert session._thought_cache[1] is t

    @pytest.mark.asyncio
    async def test_add_thought_with_branch(self):
        session = SessionMemory()
        t1 = make_thought(1)
        t2 = make_thought(2, branch_from=1, branch_id="branch_a")
        await session.add_thought(t1)
        await session.add_thought(t2)
        assert "branch_a" in session.branches
        assert len(session.branches["branch_a"]) == 1

    @pytest.mark.asyncio
    async def test_find_thought_content_found(self):
        session = SessionMemory()
        await session.add_thought(make_thought(1))
        content = await session.find_thought_content(1)
        assert content == "thought 1"

    @pytest.mark.asyncio
    async def test_find_thought_content_not_found(self):
        session = SessionMemory()
        content = await session.find_thought_content(99)
        assert content == "Unknown thought"

    @pytest.mark.asyncio
    async def test_get_branch_summary(self):
        session = SessionMemory()
        await session.add_thought(make_thought(1))
        await session.add_thought(make_thought(2, branch_from=1, branch_id="b1"))
        await session.add_thought(make_thought(3, branch_from=1, branch_id="b1"))
        summary = await session.get_branch_summary()
        assert summary["b1"] == 2

    @pytest.mark.asyncio
    async def test_get_branch_summary_empty(self):
        session = SessionMemory()
        summary = await session.get_branch_summary()
        assert summary == {}

    def test_get_current_branch_id_with_branch(self):
        session = SessionMemory()
        t = make_thought(2, branch_from=1, branch_id="my_branch")
        assert session.get_current_branch_id(t) == "my_branch"

    def test_get_current_branch_id_no_branch(self):
        session = SessionMemory()
        t = make_thought(1)
        assert session.get_current_branch_id(t) == "main"

    @pytest.mark.asyncio
    async def test_session_limit_exceeded(self):
        session = SessionMemory()
        session.MAX_THOUGHTS_PER_SESSION = 2
        await session.add_thought(make_thought(1))
        await session.add_thought(make_thought(2))
        with pytest.raises(ValueError, match="exceeds maximum"):
            await session.add_thought(make_thought(3))

    @pytest.mark.asyncio
    async def test_branch_limit_exceeded(self):
        session = SessionMemory()
        session.MAX_BRANCHES_PER_SESSION = 1
        await session.add_thought(make_thought(1))
        await session.add_thought(make_thought(2, branch_from=1, branch_id="b1"))
        with pytest.raises(ValueError, match="exceeds maximum"):
            await session.add_thought(make_thought(3, branch_from=1, branch_id="b2"))

    @pytest.mark.asyncio
    async def test_thoughts_per_branch_limit_exceeded(self):
        session = SessionMemory()
        session.MAX_THOUGHTS_PER_BRANCH = 1
        await session.add_thought(make_thought(1))
        await session.add_thought(make_thought(2, branch_from=1, branch_id="b1"))
        with pytest.raises(ValueError, match="exceeds maximum"):
            await session.add_thought(make_thought(3, branch_from=1, branch_id="b1"))
