"""Streamlined models with consolidated validation logic."""

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class ThoughtType(Enum):
    """Types of thoughts in the sequential thinking process."""

    STANDARD = "standard"
    REVISION = "revision"
    BRANCH = "branch"


class ValidationRule:
    """Encapsulates validation logic for thought relationships."""

    @staticmethod
    def validate_revision_consistency(data: dict) -> List[str]:
        """Validate revision field consistency."""
        errors = []
        is_revision = data.get("is_revision", False)
        revises_thought = data.get("revises_thought")

        if revises_thought is not None and not is_revision:
            errors.append("revises_thought requires is_revision=True")

        return errors

    @staticmethod
    def validate_branch_consistency(data: dict) -> List[str]:
        """Validate branch field consistency."""
        errors = []
        branch_from = data.get("branch_from")
        branch_id = data.get("branch_id")

        if branch_id is not None and branch_from is None:
            errors.append("branch_id requires branch_from to be set")

        return errors

    @staticmethod
    def validate_thought_numbers(data: dict) -> List[str]:
        """Validate thought number relationships."""
        errors = []
        current_number = data.get("thought_number")
        revises_thought = data.get("revises_thought")
        branch_from = data.get("branch_from")

        if current_number is None:
            return errors

        if revises_thought is not None and revises_thought >= current_number:
            errors.append("revises_thought must be less than current thought_number")

        if branch_from is not None and branch_from >= current_number:
            errors.append("branch_from must be less than current thought_number")

        return errors

    @classmethod
    def validate_all(cls, data: dict) -> None:
        """Run all validation rules and raise on first error."""
        all_errors = []
        all_errors.extend(cls.validate_revision_consistency(data))
        all_errors.extend(cls.validate_branch_consistency(data))
        all_errors.extend(cls.validate_thought_numbers(data))

        if all_errors:
            raise ValueError("; ".join(all_errors))


class ThoughtData(BaseModel):
    """Streamlined thought data model with consolidated validation."""

    # Core fields
    thought: str = Field(..., min_length=1, description="Content of the thought")
    thought_number: int = Field(
        ..., ge=1, description="Sequence number starting from 1"
    )
    total_thoughts: int = Field(
        ..., ge=5, description="Estimated total thoughts (minimum 5)"
    )
    next_needed: bool = Field(..., description="Whether another thought is needed")

    # Optional workflow fields
    is_revision: bool = Field(
        False, description="Whether this revises a previous thought"
    )
    revises_thought: Optional[int] = Field(
        None, ge=1, description="Thought number being revised"
    )
    branch_from: Optional[int] = Field(
        None, ge=1, description="Thought number to branch from"
    )
    branch_id: Optional[str] = Field(None, description="Unique branch identifier")
    needs_more: bool = Field(
        False, description="Whether more thoughts are needed beyond estimate"
    )

    model_config = {"validate_assignment": True, "frozen": True}

    @property
    def thought_type(self) -> ThoughtType:
        """Determine the type of thought based on field values."""
        if self.is_revision:
            return ThoughtType.REVISION
        elif self.branch_from is not None:
            return ThoughtType.BRANCH
        return ThoughtType.STANDARD

    @model_validator(mode="before")
    @classmethod
    def validate_thought_data(cls, data):
        """Consolidated validation using rule-based approach."""
        if isinstance(data, dict):
            ValidationRule.validate_all(data)
        return data

    def format_for_log(self) -> str:
        """Format thought for logging with type-specific prefix."""
        formatters = {
            ThoughtType.REVISION: lambda: f"Revision {self.thought_number}/{self.total_thoughts} (revising #{self.revises_thought})",
            ThoughtType.BRANCH: lambda: f"Branch {self.thought_number}/{self.total_thoughts} (from #{self.branch_from}, ID: {self.branch_id})",
            ThoughtType.STANDARD: lambda: f"Thought {self.thought_number}/{self.total_thoughts}",
        }

        prefix = formatters[self.thought_type]()
        return f"{prefix}\n  Content: {self.thought}\n  Next: {self.next_needed}, More: {self.needs_more}"
