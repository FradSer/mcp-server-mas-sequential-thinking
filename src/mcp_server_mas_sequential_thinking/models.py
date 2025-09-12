"""Streamlined models with consolidated validation logic."""

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class ThoughtType(Enum):
    """Types of thoughts in the sequential thinking process."""

    STANDARD = "standard"
    REVISION = "revision"
    BRANCH = "branch"


def _validate_thought_relationships(data: dict) -> None:
    """Validate thought relationships with optimized validation logic."""
    # Extract values once with modern dict methods
    is_revision = data.get("is_revision", False)
    revises_thought = data.get("revises_thought")
    branch_from = data.get("branch_from") 
    branch_id = data.get("branch_id")
    current_number = data.get("thought_number")
    
    # Collect validation errors efficiently
    errors = []
    
    # Relationship validation with guard clauses
    if revises_thought is not None and not is_revision:
        errors.append("revises_thought requires is_revision=True")
    
    if branch_id is not None and branch_from is None:
        errors.append("branch_id requires branch_from to be set")
    
    # Numeric validation with early exit
    if current_number is None:
        if errors:
            raise ValueError("; ".join(errors))
        return
        
    # Validate numeric relationships
    if revises_thought is not None and revises_thought >= current_number:
        errors.append("revises_thought must be less than current thought_number")
        
    if branch_from is not None and branch_from >= current_number:
        errors.append("branch_from must be less than current thought_number")
    
    if errors:
        raise ValueError("; ".join(errors))


class ThoughtData(BaseModel):
    """Streamlined thought data model with consolidated validation."""
    
    model_config = {"validate_assignment": True, "frozen": True}

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
        """Consolidated validation with simplified logic."""
        if isinstance(data, dict):
            _validate_thought_relationships(data)
        return data

    def format_for_log(self) -> str:
        """Format thought for logging with optimized type-specific formatting."""
        # Use match statement for modern Python pattern matching
        match self.thought_type:
            case ThoughtType.REVISION:
                prefix = f"Revision {self.thought_number}/{self.total_thoughts} (revising #{self.revises_thought})"
            case ThoughtType.BRANCH:
                prefix = f"Branch {self.thought_number}/{self.total_thoughts} (from #{self.branch_from}, ID: {self.branch_id})"
            case _:  # ThoughtType.STANDARD
                prefix = f"Thought {self.thought_number}/{self.total_thoughts}"

        # Use multiline string formatting for better readability
        return (f"{prefix}\n"
                f"  Content: {self.thought}\n"  
                f"  Next: {self.next_needed}, More: {self.needs_more}")
