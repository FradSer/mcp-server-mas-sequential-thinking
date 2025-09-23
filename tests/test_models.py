"""Tests for the models module."""

import pytest
from pydantic import ValidationError

from mcp_server_mas_sequential_thinking.core.models import ThoughtData


class TestThoughtData:
    """Test ThoughtData model validation."""

    def test_total_thoughts_minimum_value(self):
        """Test that total_thoughts accepts values >= 1."""
        # This should work - minimum value of 1
        thought_data = ThoughtData(
            thought="Test thought",
            thoughtNumber=1,
            totalThoughts=1,  # Minimum allowed value
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        assert thought_data.totalThoughts == 1

        # This should also work - larger value
        thought_data = ThoughtData(
            thought="Test thought",
            thoughtNumber=1,
            totalThoughts=10,  # Higher value
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        assert thought_data.totalThoughts == 10

    def test_total_thoughts_invalid_values(self):
        """Test that total_thoughts rejects values < 1."""
        # This should fail - value of 0
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Test thought",
                thoughtNumber=1,
                totalThoughts=0,  # Below minimum
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

        # This should work now - value of 3 is valid
        ThoughtData(
            thought="Test thought",
            thoughtNumber=1,
            totalThoughts=3,  # Now valid
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )

        # This should fail - negative value
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Test thought",
                thoughtNumber=1,
                totalThoughts=-1,  # Negative value
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )
