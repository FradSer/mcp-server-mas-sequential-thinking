"""Session management for thought history and branching."""

from dataclasses import dataclass, field

from agno.team.team import Team

from .constants import ValidationLimits
from .models import ThoughtData
from .types import BranchId, ThoughtNumber


@dataclass
class SessionMemory:
    """Manages thought history and branches with optimized lookups and DoS protection."""

    team: Team
    thought_history: list[ThoughtData] = field(default_factory=list)
    branches: dict[BranchId, list[ThoughtData]] = field(default_factory=dict)
    # High-performance cache for O(1) thought lookups by number
    _thought_cache: dict[ThoughtNumber, ThoughtData] = field(
        default_factory=dict, init=False, repr=False
    )

    # DoS protection constants as class attributes
    MAX_THOUGHTS_PER_SESSION: int = ValidationLimits.MAX_THOUGHTS_PER_SESSION
    MAX_BRANCHES_PER_SESSION: int = ValidationLimits.MAX_BRANCHES_PER_SESSION
    MAX_THOUGHTS_PER_BRANCH: int = ValidationLimits.MAX_THOUGHTS_PER_BRANCH

    def add_thought(self, thought: ThoughtData) -> None:
        """Add thought with efficient DoS protection and optimized branch management."""
        # Early DoS protection checks with descriptive errors
        self._validate_session_limits(thought)

        # Update data structures atomically
        self.thought_history.append(thought)
        self._thought_cache[thought.thoughtNumber] = thought

        # Handle branching with optimized setdefault pattern
        if thought.branchFromThought is not None and thought.branchId is not None:
            self.branches.setdefault(thought.branchId, []).append(thought)

    def _validate_session_limits(self, thought: ThoughtData) -> None:
        """Validate session limits with early exit optimization."""
        # Primary session limit check
        if len(self.thought_history) >= self.MAX_THOUGHTS_PER_SESSION:
            raise ValueError(
                f"Session exceeds maximum {self.MAX_THOUGHTS_PER_SESSION} thoughts"
            )

        # Branch-specific validations only if needed
        if not thought.branchId:
            return

        # Check total branch limit
        if len(self.branches) >= self.MAX_BRANCHES_PER_SESSION:
            raise ValueError(
                f"Session exceeds maximum {self.MAX_BRANCHES_PER_SESSION} branches"
            )

        # Check individual branch limit
        if (
            thought.branchId in self.branches
            and len(self.branches[thought.branchId]) >= self.MAX_THOUGHTS_PER_BRANCH
        ):
            raise ValueError(
                f"Branch '{thought.branchId}' exceeds maximum {self.MAX_THOUGHTS_PER_BRANCH} thoughts"
            )

    def find_thought_content(self, thought_number: ThoughtNumber) -> str:
        """Find the content of a specific thought by number using optimized cache lookup."""
        # Use cache for O(1) lookup instead of O(n) search
        thought = self._thought_cache.get(thought_number)
        return thought.thought if thought else "Unknown thought"

    def get_branch_summary(self) -> dict[BranchId, int]:
        """Get summary of all branches."""
        return {
            branch_id: len(thoughts) for branch_id, thoughts in self.branches.items()
        }

    def get_current_branch_id(self, thought: ThoughtData) -> str:
        """Get the current branch ID for a thought with improved logic."""
        return thought.branchId or "main"
