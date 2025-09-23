"""Tests for main.py validation functions using TDD approach."""

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.server_core import (
    create_validated_thought_data,
)


class TestCreateValidatedThoughtData:
    """Test _create_validated_thought_data function with proper error handling."""

    def test_valid_thought_data_creation(self):
        """Test that valid data creates ThoughtData successfully."""
        result = create_validated_thought_data(
            thought="Test thought",
            thought_number=1,
            total_thoughts=5,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=None,
            branch_id=None,
            needs_more=True,
        )

        assert isinstance(result, ThoughtData)
        assert result.thought == "Test thought"
        assert result.thought_number == 1
        assert result.total_thoughts == 5

    def test_thought_whitespace_stripping(self):
        """Test that thought whitespace is properly stripped."""
        result = create_validated_thought_data(
            thought="  Test thought with whitespace  ",
            thought_number=1,
            total_thoughts=5,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=None,
            branch_id=None,
            needs_more=True,
        )

        assert result.thought == "Test thought with whitespace"

    def test_branch_id_whitespace_stripping(self):
        """Test that branch_id whitespace is properly stripped."""
        result = create_validated_thought_data(
            thought="Test thought",
            thought_number=2,
            total_thoughts=5,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=1,  # branch_from must be less than thought_number
            branch_id="  branch123  ",
            needs_more=True,
        )

        assert result.branch_id == "branch123"

    def test_none_branch_id_preserved(self):
        """Test that None branch_id is preserved."""
        result = create_validated_thought_data(
            thought="Test thought",
            thought_number=1,
            total_thoughts=5,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=None,
            branch_id=None,
            needs_more=True,
        )

        assert result.branch_id is None

    def test_invalid_thought_number_raises_validation_error(self):
        """Test that invalid thought_number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid thought data"):
            create_validated_thought_data(
                thought="Test thought",
                thought_number=0,  # Invalid: must be >= 1
                total_thoughts=5,
                next_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from=None,
                branch_id=None,
                needs_more=True,
            )

    def test_invalid_total_thoughts_raises_validation_error(self):
        """Test that invalid total_thoughts raises ValueError."""
        with pytest.raises(ValueError, match="Invalid thought data"):
            create_validated_thought_data(
                thought="Test thought",
                thought_number=1,
                total_thoughts=4,  # Invalid: must be >= 5
                next_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from=None,
                branch_id=None,
                needs_more=True,
            )

    def test_empty_thought_after_strip_raises_validation_error(self):
        """Test that empty thought after stripping raises ValueError."""
        with pytest.raises(ValueError, match="Invalid thought data"):
            create_validated_thought_data(
                thought="   ",  # Empty after strip
                thought_number=1,
                total_thoughts=5,
                next_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from=None,
                branch_id=None,
                needs_more=True,
            )

    def test_malformed_data_raises_validation_error(self):
        """Test that malformed data raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Invalid thought data"):
            create_validated_thought_data(
                thought="Test thought",
                thought_number="not_a_number",  # Invalid type
                total_thoughts=5,
                next_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from=None,
                branch_id=None,
                needs_more=True,
            )
