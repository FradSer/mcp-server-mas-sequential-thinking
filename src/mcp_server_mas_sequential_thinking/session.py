"""Session management for thought history and branching."""

from dataclasses import dataclass, field
from typing import Dict, List
from agno.team.team import Team
from .models import ThoughtData


@dataclass
class SessionMemory:
    """Manages thought history and branches for a session."""

    team: Team
    thought_history: List[ThoughtData] = field(default_factory=list)
    branches: Dict[str, List[ThoughtData]] = field(default_factory=dict)

    def add_thought(self, thought: ThoughtData) -> None:
        """Add a thought to history and manage branches."""
        self.thought_history.append(thought)

        # Handle branching
        if thought.branch_from is not None and thought.branch_id is not None:
            if thought.branch_id not in self.branches:
                self.branches[thought.branch_id] = []
            self.branches[thought.branch_id].append(thought)

    def find_thought_content(self, thought_number: int) -> str:
        """Find the content of a specific thought by number."""
        for thought in self.thought_history:
            if thought.thought_number == thought_number:
                return thought.thought
        return "Unknown thought"

    def get_branch_summary(self) -> Dict[str, int]:
        """Get summary of all branches."""
        return {
            branch_id: len(thoughts) for branch_id, thoughts in self.branches.items()
        }

    def get_current_branch_id(self, thought: ThoughtData) -> str:
        """Get the current branch ID for a thought."""
        return thought.branch_id if thought.branch_from is not None else "main"
