"""Tests for the models module."""

import pytest
from pydantic import ValidationError
from src.mcp_server_mas_sequential_thinking.models import ThoughtData


class TestThoughtData:
    """Test ThoughtData model validation."""

    def test_total_thoughts_minimum_value(self):
        """Test that total_thoughts accepts values >= 5."""
        # This should work - minimum value of 5
        thought_data = ThoughtData(
            thought="Test thought",
            thought_number=1,
            total_thoughts=5,  # Minimum allowed value
            next_needed=True,
        )
        assert thought_data.total_thoughts == 5

        # This should also work - larger value
        thought_data = ThoughtData(
            thought="Test thought",
            thought_number=1,
            total_thoughts=10,  # Higher value
            next_needed=True,
        )
        assert thought_data.total_thoughts == 10

    def test_total_thoughts_invalid_values(self):
        """Test that total_thoughts rejects values < 1."""
        # This should fail - value of 0
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Test thought",
                thought_number=1,
                total_thoughts=0,  # Below minimum
                next_needed=True,
            )

        # This should work now - value of 3 is valid
        ThoughtData(
            thought="Test thought",
            thought_number=1,
            total_thoughts=3,  # Now valid
            next_needed=True,
        )

        # This should fail - negative value
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Test thought",
                thought_number=1,
                total_thoughts=-1,  # Negative value
                next_needed=True,
            )
