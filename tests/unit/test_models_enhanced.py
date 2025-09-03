"""Comprehensive tests for the models module."""

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.mcp_server_mas_sequential_thinking.models import ThoughtData, ValidationRule, ThoughtType
from tests.helpers.factories import ThoughtDataBuilder, ThoughtSequenceFactory


class TestThoughtDataValidation:
    """Comprehensive ThoughtData validation testing."""

    @given(
        thought=st.text(min_size=1, max_size=1000),
        thought_number=st.integers(min_value=1, max_value=100),
        total_thoughts=st.integers(min_value=5, max_value=100),
    )
    def test_valid_thought_data_creation(self, thought, thought_number, total_thoughts):
        """Property-based test for valid ThoughtData creation."""
        # Ensure thought_number doesn't exceed total_thoughts for validity
        if thought_number > total_thoughts:
            total_thoughts = thought_number + 5

        thought_data = ThoughtData(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_needed=True,
        )

        assert thought_data.thought == thought
        assert thought_data.thought_number == thought_number
        assert thought_data.total_thoughts == total_thoughts

    def test_thought_data_immutability(self):
        """Test that ThoughtData instances are immutable."""
        thought_data = ThoughtDataBuilder().build()

        with pytest.raises(ValidationError):
            thought_data.thought = "Modified thought"

    def test_revision_validation_rules(self):
        """Test revision validation logic."""
        # Valid revision
        thought_data = (
            ThoughtDataBuilder().as_revision(revises=1).with_number(2).build()
        )
        assert thought_data.thought_type == ThoughtType.REVISION

        # Invalid: revises_thought >= current number
        with pytest.raises(ValidationError):
            ThoughtDataBuilder().as_revision(revises=5).with_number(3).build()

    def test_branch_validation_rules(self):
        """Test branch validation logic."""
        # Valid branch
        thought_data = (
            ThoughtDataBuilder()
            .as_branch(from_thought=1, branch_id="branch-1")
            .with_number(2)
            .build()
        )
        assert thought_data.thought_type == ThoughtType.BRANCH

        # Invalid: branch_from >= current number
        with pytest.raises(ValidationError):
            ThoughtDataBuilder().as_branch(
                from_thought=5, branch_id="test"
            ).with_number(3).build()

    def test_empty_thought_validation(self):
        """Test validation of empty thoughts."""
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="", thought_number=1, total_thoughts=5, next_needed=True
            )

    def test_invalid_thought_number_validation(self):
        """Test validation of invalid thought numbers."""
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Valid thought",
                thought_number=0,  # Invalid: must be >= 1
                total_thoughts=5,
                next_needed=True,
            )

    def test_invalid_total_thoughts_validation(self):
        """Test validation of invalid total thoughts."""
        with pytest.raises(ValidationError):
            ThoughtData(
                thought="Valid thought",
                thought_number=1,
                total_thoughts=4,  # Invalid: must be >= 5
                next_needed=True,
            )

    @pytest.mark.parametrize(
        "thought_type,expected_prefix",
        [
            (ThoughtType.STANDARD, "Thought"),
            (ThoughtType.REVISION, "Revision"),
            (ThoughtType.BRANCH, "Branch"),
        ],
    )
    def test_format_for_log(self, thought_type, expected_prefix):
        """Test log formatting for different thought types."""
        if thought_type == ThoughtType.REVISION:
            thought_data = (
                ThoughtDataBuilder().as_revision(revises=1).with_number(2).build()
            )
        elif thought_type == ThoughtType.BRANCH:
            thought_data = (
                ThoughtDataBuilder()
                .as_branch(from_thought=1, branch_id="test-branch")
                .with_number(2)
                .build()
            )
        else:
            thought_data = ThoughtDataBuilder().build()

        formatted = thought_data.format_for_log()
        assert expected_prefix in formatted
        assert "Content:" in formatted
        assert "Next:" in formatted


class TestValidationRule:
    """Test validation rule logic in isolation."""

    def test_revision_consistency_validation(self):
        """Test revision consistency rules."""
        # Valid cases
        valid_data = {"is_revision": True, "revises_thought": 1}
        assert ValidationRule.validate_revision_consistency(valid_data) == []

        valid_data = {"is_revision": False, "revises_thought": None}
        assert ValidationRule.validate_revision_consistency(valid_data) == []

        # Invalid cases
        invalid_data = {"is_revision": False, "revises_thought": 1}
        errors = ValidationRule.validate_revision_consistency(invalid_data)
        assert len(errors) == 1
        assert "revises_thought requires is_revision=True" in errors[0]

    def test_branch_consistency_validation(self):
        """Test branch consistency rules."""
        # Valid cases
        valid_data = {"branch_from": 1, "branch_id": "test-branch"}
        assert ValidationRule.validate_branch_consistency(valid_data) == []

        valid_data = {"branch_from": None, "branch_id": None}
        assert ValidationRule.validate_branch_consistency(valid_data) == []

        # Invalid cases
        invalid_data = {"branch_from": None, "branch_id": "test-branch"}
        errors = ValidationRule.validate_branch_consistency(invalid_data)
        assert len(errors) == 1
        assert "branch_id requires branch_from to be set" in errors[0]

    def test_thought_numbers_validation(self):
        """Test thought number relationship validation."""
        # Valid cases
        valid_data = {"thought_number": 3, "revises_thought": 1, "branch_from": 2}
        assert ValidationRule.validate_thought_numbers(valid_data) == []

        # Invalid revision number
        invalid_data = {"thought_number": 3, "revises_thought": 5}
        errors = ValidationRule.validate_thought_numbers(invalid_data)
        assert len(errors) == 1
        assert "revises_thought must be less than current thought_number" in errors[0]

        # Invalid branch_from number
        invalid_data = {"thought_number": 3, "branch_from": 5}
        errors = ValidationRule.validate_thought_numbers(invalid_data)
        assert len(errors) == 1
        assert "branch_from must be less than current thought_number" in errors[0]

    def test_validate_all_aggregates_errors(self):
        """Test that validate_all aggregates all validation errors."""
        invalid_data = {
            "thought_number": 3,
            "is_revision": False,
            "revises_thought": 5,  # Invalid: greater than thought_number
            "branch_from": None,
            "branch_id": "test-branch",  # Invalid: branch_id without branch_from
        }

        with pytest.raises(ValueError) as exc_info:
            ValidationRule.validate_all(invalid_data)

        error_message = str(exc_info.value)
        assert "branch_id requires branch_from to be set" in error_message
        assert (
            "revises_thought must be less than current thought_number" in error_message
        )


class TestThoughtSequenceFactory:
    """Test the factory for creating thought sequences."""

    def test_create_linear_sequence(self):
        """Test creation of linear thought sequences."""
        sequence = ThoughtSequenceFactory.create_linear_sequence(5, "Test")

        assert len(sequence) == 5
        for i, thought in enumerate(sequence, 1):
            assert thought.thought_number == i
            assert thought.total_thoughts == 5
            assert f"Test {i}" in thought.thought

    def test_create_revision_sequence(self):
        """Test creation of revision thoughts."""
        original = ThoughtDataBuilder().with_number(1).build()
        revision = ThoughtSequenceFactory.create_revision_sequence(
            original, "Revised content"
        )

        assert revision.is_revision
        assert revision.revises_thought == 1
        assert revision.thought_number == 2
        assert revision.thought == "Revised content"

    def test_create_branch_sequence(self):
        """Test creation of branch thoughts."""
        source = ThoughtDataBuilder().with_number(1).build()
        branch = ThoughtSequenceFactory.create_branch_sequence(
            source, "Branch content", "test-branch"
        )

        assert branch.branch_from == 1
        assert branch.branch_id == "test-branch"
        assert branch.thought_number == 2
        assert branch.thought == "Branch content"


class TestThoughtType:
    """Test ThoughtType enum and related functionality."""

    def test_thought_type_property_standard(self):
        """Test thought type property for standard thoughts."""
        thought = ThoughtDataBuilder().build()
        assert thought.thought_type == ThoughtType.STANDARD

    def test_thought_type_property_revision(self):
        """Test thought type property for revisions."""
        thought = ThoughtDataBuilder().as_revision(revises=1).with_number(2).build()
        assert thought.thought_type == ThoughtType.REVISION

    def test_thought_type_property_branch(self):
        """Test thought type property for branches."""
        thought = (
            ThoughtDataBuilder()
            .as_branch(from_thought=1, branch_id="test")
            .with_number(2)
            .build()
        )
        assert thought.thought_type == ThoughtType.BRANCH

    def test_thought_type_enum_values(self):
        """Test ThoughtType enum values."""
        assert ThoughtType.STANDARD.value == "standard"
        assert ThoughtType.REVISION.value == "revision"
        assert ThoughtType.BRANCH.value == "branch"
