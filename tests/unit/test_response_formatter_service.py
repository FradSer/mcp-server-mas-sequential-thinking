"""Unit tests for services/response_formatter.py ResponseFormatter."""

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.services.response_formatter import (
    ResponseFormatter,
)


def make_thought(n: int = 1) -> ThoughtData:
    return ThoughtData(
        thought="test thought",
        thoughtNumber=n,
        totalThoughts=5,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )


class TestResponseFormatter:
    """Tests for services/response_formatter.py ResponseFormatter."""

    def test_format_response_returns_content(self):
        formatter = ResponseFormatter()
        thought = make_thought()
        result = formatter.format_response("hello", thought)
        assert result == "hello"

    def test_format_response_empty_content(self):
        formatter = ResponseFormatter()
        thought = make_thought()
        result = formatter.format_response("", thought)
        assert result == ""

    def test_format_response_logs_and_returns(self):
        formatter = ResponseFormatter()
        thought = make_thought()
        content = "some content here"
        result = formatter.format_response(content, thought)
        assert result == content

    def test_extract_response_content_string(self):
        formatter = ResponseFormatter()
        result = formatter.extract_response_content("plain string")
        assert result == "plain string"

    def test_extract_response_content_none(self):
        formatter = ResponseFormatter()
        result = formatter.extract_response_content(None)
        assert result == ""

    def test_extract_response_content_object_with_content(self):
        class FakeResponse:
            content = "extracted content"

        formatter = ResponseFormatter()
        result = formatter.extract_response_content(FakeResponse())
        assert result == "extracted content"

    def test_log_response_formatting_called(self):
        """Ensure _log_response_formatting is exercised via format_response."""
        formatter = ResponseFormatter()
        thought = make_thought()
        # Just calling format_response is enough to cover _log_response_formatting
        formatter.format_response("response text", thought)
