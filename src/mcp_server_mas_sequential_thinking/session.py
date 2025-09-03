"""Session management for thought history and branching."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from agno.team.team import Team
from .models import ThoughtData


@dataclass
class SessionMemory:
    """Manages thought history and branches for a session with optimized lookups."""

    team: Team
    thought_history: List[ThoughtData] = field(default_factory=list)
    branches: Dict[str, List[ThoughtData]] = field(default_factory=dict)
    # Optimization: Cache for faster thought lookups by number
    _thought_cache: Dict[int, ThoughtData] = field(default_factory=dict, init=False)

    def add_thought(self, thought: ThoughtData) -> None:
        """Add a thought to history and manage branches with optimization."""
        self.thought_history.append(thought)
        # Update cache for faster lookups
        self._thought_cache[thought.thought_number] = thought

        # Handle branching with optimized logic
        if thought.branch_from is not None and thought.branch_id is not None:
            self.branches.setdefault(thought.branch_id, []).append(thought)

    def find_thought_content(self, thought_number: int) -> str:
        """Find the content of a specific thought by number using optimized cache lookup."""
        # Use cache for O(1) lookup instead of O(n) search
        thought = self._thought_cache.get(thought_number)
        return thought.thought if thought else "Unknown thought"

    def get_branch_summary(self) -> Dict[str, int]:
        """Get summary of all branches."""
        return {
            branch_id: len(thoughts) for branch_id, thoughts in self.branches.items()
        }

    def get_current_branch_id(self, thought: ThoughtData) -> str:
        """Get the current branch ID for a thought with improved logic."""
        return thought.branch_id or "main"
