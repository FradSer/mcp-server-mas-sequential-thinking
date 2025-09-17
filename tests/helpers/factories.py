"""Factory classes for creating test data using the builder pattern."""

from dataclasses import dataclass
from typing import Optional
from src.mcp_server_mas_sequential_thinking.models import ThoughtData


@dataclass
class ThoughtDataBuilder:
    """Builder pattern for ThoughtData test instances."""

    thought: str = "Default test thought"
    thought_number: int = 1
    total_thoughts: int = 5
    next_needed: bool = True
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more: bool = False

    def with_thought(self, content: str) -> "ThoughtDataBuilder":
        """Set thought content."""
        self.thought = content
        return self

    def with_number(self, number: int) -> "ThoughtDataBuilder":
        """Set thought number."""
        self.thought_number = number
        return self

    def with_total(self, total: int) -> "ThoughtDataBuilder":
        """Set total thoughts."""
        self.total_thoughts = total
        return self

    def as_revision(self, revises: int) -> "ThoughtDataBuilder":
        """Configure as revision."""
        self.is_revision = True
        self.revises_thought = revises
        return self

    def as_branch(self, from_thought: int, branch_id: str) -> "ThoughtDataBuilder":
        """Configure as branch."""
        self.branch_from = from_thought
        self.branch_id = branch_id
        return self

    def needs_more_thoughts(self, needs_more: bool = True) -> "ThoughtDataBuilder":
        """Set needs_more flag."""
        self.needs_more = needs_more
        return self

    def build(self) -> ThoughtData:
        """Build the ThoughtData instance."""
        return ThoughtData(
            thought=self.thought,
            thought_number=self.thought_number,
            total_thoughts=self.total_thoughts,
            next_needed=self.next_needed,
            is_revision=self.is_revision,
            revises_thought=self.revises_thought,
            branch_from=self.branch_from,
            branch_id=self.branch_id,
            needs_more=self.needs_more,
        )


class ThoughtSequenceFactory:
    """Factory for creating sequences of related thoughts."""

    @staticmethod
    def create_linear_sequence(
        count: int, base_content: str = "Thought"
    ) -> list[ThoughtData]:
        """Create a linear sequence of thoughts."""
        return [
            ThoughtDataBuilder()
            .with_thought(f"{base_content} {i}")
            .with_number(i)
            .with_total(count)
            .build()
            for i in range(1, count + 1)
        ]

    @staticmethod
    def create_revision_sequence(
        original_thought: ThoughtData, revision_content: str
    ) -> ThoughtData:
        """Create a revision of an existing thought."""
        return (
            ThoughtDataBuilder()
            .with_thought(revision_content)
            .with_number(original_thought.thought_number + 1)
            .with_total(original_thought.total_thoughts)
            .as_revision(original_thought.thought_number)
            .build()
        )

    @staticmethod
    def create_branch_sequence(
        source_thought: ThoughtData, branch_content: str, branch_id: str
    ) -> ThoughtData:
        """Create a branch from an existing thought."""
        return (
            ThoughtDataBuilder()
            .with_thought(branch_content)
            .with_number(source_thought.thought_number + 1)
            .with_total(source_thought.total_thoughts)
            .as_branch(source_thought.thought_number, branch_id)
            .build()
        )
