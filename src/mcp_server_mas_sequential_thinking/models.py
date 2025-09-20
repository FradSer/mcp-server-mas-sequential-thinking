"""Streamlined models with consolidated validation logic."""

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from .constants import ValidationLimits, FieldLengthLimits
from .types import ThoughtNumber, BranchId


class ThoughtType(Enum):
    """Types of thoughts in the sequential thinking process."""

    STANDARD = "standard"
    REVISION = "revision"
    BRANCH = "branch"


def _validate_thought_relationships(data: dict) -> None:
    """Validate thought relationships with optimized validation logic."""
    # Extract values once with modern dict methods
    is_revision = data.get("isRevision", False)
    branch_from_thought = data.get("branchFromThought")
    branch_id = data.get("branchId")
    current_number = data.get("thoughtNumber")

    # Collect validation errors efficiently
    errors = []

    # Relationship validation with guard clauses
    if branch_id is not None and branch_from_thought is None:
        errors.append("branchId requires branchFromThought to be set")

    # Numeric validation with early exit
    if current_number is None:
        if errors:
            raise ValueError("; ".join(errors))
        return

    # Validate numeric relationships
    if branch_from_thought is not None and branch_from_thought >= current_number:
        errors.append("branchFromThought must be less than current thoughtNumber")

    if errors:
        raise ValueError("; ".join(errors))


class ThoughtData(BaseModel):
    """Streamlined thought data model with consolidated validation."""

    model_config = {"validate_assignment": True, "frozen": True}

    # Core fields
    thought: str = Field(
        ...,
        min_length=FieldLengthLimits.MIN_STRING_LENGTH,
        description="Content of the thought",
    )
    thoughtNumber: ThoughtNumber = Field(
        ...,
        ge=ValidationLimits.MIN_THOUGHT_NUMBER,
        description="Sequence number starting from 1",
    )
    totalThoughts: int = Field(
        ...,
        ge=1,
        description="Estimated total thoughts",
    )
    nextThoughtNeeded: bool = Field(..., description="Whether another thought is needed")

    # Required workflow fields
    isRevision: bool = Field(..., description="Whether this revises a previous thought")
    branchFromThought: Optional[ThoughtNumber] = Field(
        ...,
        ge=ValidationLimits.MIN_THOUGHT_NUMBER,
        description="Thought number to branch from",
    )
    branchId: Optional[BranchId] = Field(..., description="Unique branch identifier")
    needsMoreThoughts: bool = Field(
        ..., description="Whether more thoughts are needed beyond estimate"
    )

    @property
    def thought_type(self) -> ThoughtType:
        """Determine the type of thought based on field values."""
        if self.isRevision:
            return ThoughtType.REVISION
        elif self.branchFromThought is not None:
            return ThoughtType.BRANCH
        return ThoughtType.STANDARD

    @model_validator(mode="before")
    @classmethod
    def validate_thought_data(cls, data):
        """Consolidated validation with simplified logic."""
        if isinstance(data, dict):
            _validate_thought_relationships(data)
        return data

    def format_for_log(self) -> str:
        """Format thought for logging with optimized type-specific formatting."""
        # Use match statement for modern Python pattern matching
        match self.thought_type:
            case ThoughtType.REVISION:
                prefix = f"Revision {self.thoughtNumber}/{self.totalThoughts} (revising #{self.branchFromThought})"
            case ThoughtType.BRANCH:
                prefix = f"Branch {self.thoughtNumber}/{self.totalThoughts} (from #{self.branchFromThought}, ID: {self.branchId})"
            case _:  # ThoughtType.STANDARD
                prefix = f"Thought {self.thoughtNumber}/{self.totalThoughts}"

        # Use multiline string formatting for better readability
        return (
            f"{prefix}\n"
            f"  Content: {self.thought}\n"
            f"  Next: {self.nextThoughtNeeded}, More: {self.needsMoreThoughts}"
        )
