"""Unit tests for MCP tool orchestration contract fields."""

from mcp_server_mas_sequential_thinking.main import (
    StopReason,
    _build_call_result,
    _derive_progress_state,
)


def test_derive_progress_state_enforces_multi_step_sequence():
    """Remaining planned steps should force continuation."""
    should_continue, next_thought_number, stop_reason = _derive_progress_state(
        thought_number=1,
        total_thoughts=3,
        next_thought_needed=False,
        needs_more_thoughts=False,
    )

    assert should_continue is True
    assert next_thought_number == 2
    assert stop_reason == StopReason.NEXT_THOUGHT_REQUIRED


def test_derive_progress_state_stops_on_final_step():
    """Final step without extension flags should stop."""
    should_continue, next_thought_number, stop_reason = _derive_progress_state(
        thought_number=3,
        total_thoughts=3,
        next_thought_needed=False,
        needs_more_thoughts=False,
    )

    assert should_continue is False
    assert next_thought_number is None
    assert stop_reason == StopReason.THOUGHT_SEQUENCE_COMPLETE


def test_build_call_result_includes_structured_contract_fields():
    """Tool result should include the required structured loop fields."""
    result = _build_call_result(
        message="ok",
        thought_number=1,
        total_thoughts=3,
        should_continue=True,
        stop_reason=StopReason.NEXT_THOUGHT_REQUIRED,
        next_thought_number=2,
    )

    assert result.structured_content is not None
    assert result.structured_content["should_continue"] is True
    assert result.structured_content["next_thought_number"] == 2
    assert result.structured_content["stop_reason"] == "next_thought_required"
    assert result.structured_content["next_call_arguments"]["thoughtNumber"] == 2
