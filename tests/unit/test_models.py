"""Unit tests for core/models.py ThoughtData."""

import pytest
from pydantic import ValidationError

from mcp_server_mas_sequential_thinking.core.models import ThoughtData, ThoughtType


def base_thought(**kwargs) -> dict:
    defaults = {
        "thought": "hello world",
        "thoughtNumber": 1,
        "totalThoughts": 5,
        "nextThoughtNeeded": True,
        "isRevision": False,
        "branchFromThought": None,
        "branchId": None,
        "needsMoreThoughts": False,
    }
    defaults.update(kwargs)
    return defaults


class TestThoughtData:
    """Tests for ThoughtData model."""

    def test_create_standard_thought(self):
        t = ThoughtData(**base_thought())
        assert t.thought == "hello world"
        assert t.thoughtNumber == 1
        assert t.thought_type == ThoughtType.STANDARD

    def test_create_revision_thought(self):
        t = ThoughtData(**base_thought(thoughtNumber=2, isRevision=True, branchFromThought=1))
        assert t.thought_type == ThoughtType.REVISION

    def test_create_branch_thought(self):
        t = ThoughtData(**base_thought(thoughtNumber=2, branchFromThought=1, branchId="b1"))
        assert t.thought_type == ThoughtType.BRANCH

    def test_format_for_log_standard(self):
        t = ThoughtData(**base_thought())
        log = t.format_for_log()
        assert "Thought 1/5" in log
        assert "hello world" in log

    def test_format_for_log_revision(self):
        t = ThoughtData(**base_thought(thoughtNumber=2, isRevision=True, branchFromThought=1))
        log = t.format_for_log()
        assert "Revision" in log
        assert "#1" in log

    def test_format_for_log_branch(self):
        t = ThoughtData(**base_thought(thoughtNumber=3, branchFromThought=1, branchId="feature"))
        log = t.format_for_log()
        assert "Branch" in log
        assert "feature" in log

    def test_branch_id_without_branch_from_raises(self):
        with pytest.raises(ValidationError):
            ThoughtData(**base_thought(branchId="b1", branchFromThought=None))

    def test_branch_from_ge_current_raises(self):
        with pytest.raises(ValidationError):
            ThoughtData(**base_thought(thoughtNumber=1, branchFromThought=1))

    def test_branch_from_gt_current_raises(self):
        with pytest.raises(ValidationError):
            ThoughtData(**base_thought(thoughtNumber=1, branchFromThought=2))

    def test_empty_thought_raises(self):
        with pytest.raises(ValidationError):
            ThoughtData(**base_thought(thought=""))

    def test_is_frozen(self):
        t = ThoughtData(**base_thought())
        with pytest.raises(ValidationError):
            t.thought = "changed"

    def test_next_thought_needed_false(self):
        t = ThoughtData(**base_thought(nextThoughtNeeded=False))
        assert t.nextThoughtNeeded is False

    def test_needs_more_thoughts_true(self):
        t = ThoughtData(**base_thought(needsMoreThoughts=True))
        assert t.needsMoreThoughts is True
