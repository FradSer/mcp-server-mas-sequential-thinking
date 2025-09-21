"""Comprehensive tests for the session management module."""

from unittest.mock import MagicMock

from src.mcp_server_mas_sequential_thinking.models import ThoughtData
from src.mcp_server_mas_sequential_thinking.session import SessionMemory

from tests.helpers.factories import ThoughtDataBuilder, ThoughtSequenceFactory


class TestSessionMemory:
    """Comprehensive session memory testing."""

    def test_initialization(self):
        """Test SessionMemory initialization."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        assert session.team == mock_team
        assert session.thought_history == []
        assert session.branches == {}

    def test_thought_history_management(self):
        """Test thought history tracking."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        thoughts = [
            ThoughtDataBuilder().with_number(i).with_thought(f"Thought {i}").build()
            for i in range(1, 4)
        ]

        for thought in thoughts:
            session.add_thought(thought)

        assert len(session.thought_history) == 3
        assert session.find_thought_content(2) == "Thought 2"
        assert session.find_thought_content(3) == "Thought 3"

    def test_thought_history_ordering(self):
        """Test that thoughts are stored in the order they're added."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add thoughts out of numerical order
        thoughts = [
            ThoughtDataBuilder().with_number(3).with_thought("Third").build(),
            ThoughtDataBuilder().with_number(1).with_thought("First").build(),
            ThoughtDataBuilder().with_number(2).with_thought("Second").build(),
        ]

        for thought in thoughts:
            session.add_thought(thought)

        # Should be stored in order of addition, not thought number
        assert session.thought_history[0].thought == "Third"
        assert session.thought_history[1].thought == "First"
        assert session.thought_history[2].thought == "Second"

    def test_find_thought_content_not_found(self):
        """Test finding content for non-existent thought."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add one thought
        thought = ThoughtDataBuilder().with_number(1).build()
        session.add_thought(thought)

        # Try to find non-existent thought
        assert session.find_thought_content(99) == "Unknown thought"

    def test_branch_management(self):
        """Test branch creation and tracking."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add main thoughts
        main_thought = (
            ThoughtDataBuilder().with_number(1).with_thought("Main thought").build()
        )
        session.add_thought(main_thought)

        # Add branch
        branch_thought = (
            ThoughtDataBuilder()
            .with_number(2)
            .with_thought("Branch thought")
            .as_branch(from_thought=1, branch_id="experiment-1")
            .build()
        )
        session.add_thought(branch_thought)

        assert "experiment-1" in session.branches
        assert len(session.branches["experiment-1"]) == 1
        assert session.get_current_branch_id(branch_thought) == "experiment-1"

    def test_multiple_branches(self):
        """Test handling of multiple branches."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add main thought
        main_thought = ThoughtDataBuilder().with_number(1).build()
        session.add_thought(main_thought)

        # Add multiple branches from same source
        branch1 = (
            ThoughtDataBuilder()
            .with_number(2)
            .as_branch(from_thought=1, branch_id="branch-a")
            .build()
        )
        branch2 = (
            ThoughtDataBuilder()
            .with_number(3)
            .as_branch(from_thought=1, branch_id="branch-b")
            .build()
        )

        session.add_thought(branch1)
        session.add_thought(branch2)

        assert len(session.branches) == 2
        assert "branch-a" in session.branches
        assert "branch-b" in session.branches
        assert len(session.branches["branch-a"]) == 1
        assert len(session.branches["branch-b"]) == 1

    def test_branch_extension(self):
        """Test extending an existing branch."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add main thought
        main_thought = ThoughtDataBuilder().with_number(1).build()
        session.add_thought(main_thought)

        # Add first branch thought
        branch1 = (
            ThoughtDataBuilder()
            .with_number(2)
            .as_branch(from_thought=1, branch_id="extended-branch")
            .build()
        )
        session.add_thought(branch1)

        # Add second thought to same branch
        branch2 = (
            ThoughtDataBuilder()
            .with_number(3)
            .as_branch(from_thought=2, branch_id="extended-branch")
            .build()
        )
        session.add_thought(branch2)

        assert len(session.branches["extended-branch"]) == 2

    def test_get_branch_summary(self):
        """Test branch summary generation."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add main thought
        main_thought = ThoughtDataBuilder().with_number(1).build()
        session.add_thought(main_thought)

        # Add branches
        branch1 = (
            ThoughtDataBuilder()
            .with_number(2)
            .as_branch(from_thought=1, branch_id="branch-1")
            .build()
        )
        branch2 = (
            ThoughtDataBuilder()
            .with_number(3)
            .as_branch(from_thought=1, branch_id="branch-2")
            .build()
        )
        branch3 = (
            ThoughtDataBuilder()
            .with_number(4)
            .as_branch(from_thought=2, branch_id="branch-1")  # Extend branch-1
            .build()
        )

        session.add_thought(branch1)
        session.add_thought(branch2)
        session.add_thought(branch3)

        summary = session.get_branch_summary()
        assert summary["branch-1"] == 2  # Two thoughts in branch-1
        assert summary["branch-2"] == 1  # One thought in branch-2

    def test_get_current_branch_id_main(self):
        """Test getting branch ID for main line thoughts."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        main_thought = ThoughtDataBuilder().build()
        session.add_thought(main_thought)

        assert session.get_current_branch_id(main_thought) == "main"

    def test_get_current_branch_id_revision(self):
        """Test getting branch ID for revision thoughts."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        revision_thought = (
            ThoughtDataBuilder().with_number(2).as_revision(revises=1).build()
        )
        session.add_thought(revision_thought)

        # Revisions should be on main branch
        assert session.get_current_branch_id(revision_thought) == "main"

    def test_concurrent_access_simulation(self):
        """Test concurrent access patterns."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Simulate concurrent thought additions
        thoughts = [
            ThoughtDataBuilder().with_number(i).with_thought(f"Thought {i}").build()
            for i in range(1, 11)
        ]

        for thought in thoughts:
            session.add_thought(thought)

        # Verify integrity
        assert len(session.thought_history) == 10
        for i in range(1, 11):
            assert session.find_thought_content(i) == f"Thought {i}"

    def test_large_history_performance(self):
        """Test performance with large thought history."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add many thoughts
        large_sequence = ThoughtSequenceFactory.create_linear_sequence(
            1000, "Performance test"
        )
        for thought in large_sequence:
            session.add_thought(thought)

        # Should still be able to find thoughts efficiently
        assert len(session.thought_history) == 1000
        assert session.find_thought_content(500) == "Performance test 500"
        assert session.find_thought_content(1000) == "Performance test 1000"

    def test_branch_with_None_branch_id(self):
        """Test handling of branch thoughts with None branch_id."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Add thought with branch_from but None branch_id (shouldn't happen in practice)
        thought = ThoughtData(
            thought="Test thought",
            thought_number=2,
            total_thoughts=5,
            next_needed=True,
            branch_from=1,
            branch_id=None,
        )

        session.add_thought(thought)

        # Should not create a branch entry
        assert len(session.branches) == 0

    def test_empty_session_operations(self):
        """Test operations on empty session."""
        mock_team = MagicMock()
        session = SessionMemory(team=mock_team)

        # Operations on empty session should handle gracefully
        assert session.find_thought_content(1) == "Unknown thought"
        assert session.get_branch_summary() == {}

        # Adding a thought should work normally
        thought = ThoughtDataBuilder().build()
        session.add_thought(thought)
        assert len(session.thought_history) == 1
