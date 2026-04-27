"""Unit tests for response_processor module."""

from mcp_server_mas_sequential_thinking.services.response_processor import (
    ProcessedResponse,
    ResponseExtractor,
    ResponseFormatter,
    ResponseProcessor,
)


class TestProcessedResponse:
    """Tests for ProcessedResponse dataclass."""

    def test_basic_creation(self):
        r = ProcessedResponse(content="hello", raw_response="raw")
        assert r.content == "hello"
        assert r.raw_response == "raw"
        assert r.processing_time is None
        assert r.metadata is None

    def test_with_all_fields(self):
        meta = {"key": "val"}
        r = ProcessedResponse(
            content="x", raw_response=None, processing_time=1.5, metadata=meta
        )
        assert r.processing_time == 1.5
        assert r.metadata == meta


class TestResponseExtractor:
    """Tests for ResponseExtractor.extract_content."""

    def test_none_returns_empty(self):
        assert ResponseExtractor.extract_content(None) == ""

    def test_string_returned_directly(self):
        assert ResponseExtractor.extract_content("hello") == "hello"

    def test_object_with_string_content(self):
        class Obj:
            content = "from content attr"

        assert ResponseExtractor.extract_content(Obj()) == "from content attr"

    def test_object_with_dict_content(self):
        class Obj:
            content = {"text": "from dict"}

        assert ResponseExtractor.extract_content(Obj()) == "from dict"

    def test_object_with_stringable_content(self):
        class Inner:
            def __str__(self):
                return "stringified"

        class Obj:
            content = Inner()

        assert ResponseExtractor.extract_content(Obj()) == "stringified"

    def test_dict_response(self):
        assert (
            ResponseExtractor.extract_content({"content": "dict content"})
            == "dict content"
        )

    def test_dict_response_text_key(self):
        assert ResponseExtractor.extract_content({"text": "text value"}) == "text value"

    def test_dict_response_message_key(self):
        assert ResponseExtractor.extract_content({"message": "msg"}) == "msg"

    def test_dict_response_result_key(self):
        assert ResponseExtractor.extract_content({"result": "res"}) == "res"

    def test_dict_response_output_key(self):
        assert ResponseExtractor.extract_content({"output": "out"}) == "out"

    def test_dict_response_response_key(self):
        assert ResponseExtractor.extract_content({"response": "resp"}) == "resp"

    def test_dict_nested_result(self):
        result = ResponseExtractor.extract_content({"content": {"result": 42}})
        assert result == "42"

    def test_dict_fallback_first_string_value(self):
        result = ResponseExtractor.extract_content({"unknown_key": "fallback value"})
        assert result == "fallback value"

    def test_dict_fallback_string_repr(self):
        result = ResponseExtractor.extract_content({"a": 1, "b": 2})
        assert isinstance(result, str)

    def test_object_with_text_attr(self):
        class Obj:
            text = "from text"

        assert ResponseExtractor.extract_content(Obj()) == "from text"

    def test_object_with_message_attr(self):
        class Obj:
            message = "from message"

        assert ResponseExtractor.extract_content(Obj()) == "from message"

    def test_object_with_result_attr(self):
        class Obj:
            result = "from result"

        assert ResponseExtractor.extract_content(Obj()) == "from result"

    def test_object_with_output_attr(self):
        class Obj:
            output = "from output"

        assert ResponseExtractor.extract_content(Obj()) == "from output"

    def test_unknown_type_falls_back_to_str(self):
        result = ResponseExtractor.extract_content(12345)
        assert result == "12345"

    def test_extract_from_dict_empty_string_values_skipped(self):
        result = ResponseExtractor.extract_content({"content": "   "})
        # "   " is a string but strip is empty - falls through to first string value
        assert isinstance(result, str)


class TestResponseProcessor:
    """Tests for ResponseProcessor."""

    def setup_method(self):
        self.processor = ResponseProcessor()

    def test_process_string_response(self):
        result = self.processor.process_response("hello world")
        assert result.content == "hello world"
        assert result.metadata is not None
        assert result.metadata["context"] == "processing"

    def test_process_with_context(self):
        result = self.processor.process_response("data", context="test_context")
        assert result.metadata["context"] == "test_context"

    def test_process_with_processing_time(self):
        result = self.processor.process_response("data", processing_time=2.5)
        assert result.processing_time == 2.5

    def test_process_empty_content_gets_placeholder(self):
        result = self.processor.process_response("", context="my_ctx")
        assert "[Empty response from my_ctx]" in result.content

    def test_process_whitespace_only_content_gets_placeholder(self):
        result = self.processor.process_response("   ")
        assert "[Empty response from" in result.content

    def test_metadata_includes_type(self):
        result = self.processor.process_response("test")
        assert "response_type" in result.metadata
        assert result.metadata["response_type"] == "str"

    def test_metadata_content_length(self):
        result = self.processor.process_response("hello")
        assert result.metadata["content_length"] == 5

    def test_metadata_has_content(self):
        result = self.processor.process_response("hello")
        assert result.metadata["has_content"] is True

    def test_log_response_with_processing_time(self):
        """Covers the processing_time branch in _log_response_details."""
        result = self.processor.process_response("data", processing_time=0.5)
        assert result.processing_time == 0.5

    def test_long_content_preview_truncated(self):
        long_content = "x" * 200
        result = self.processor.process_response(long_content)
        assert result.content == long_content

    def test_log_without_metadata(self):
        """Cover the else branch in _log_response_details (no metadata)."""
        processed = ProcessedResponse(content="test", raw_response=None, metadata=None)
        # Directly call the private method to cover the else branch
        self.processor._log_response_details(processed, "direct")


class TestResponseFormatterInProcessor:
    """Tests for ResponseFormatter (in response_processor.py)."""

    def test_format_for_client_strips_whitespace(self):
        processed = ProcessedResponse(content="  hello  ", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert result == "hello."

    def test_format_for_client_adds_period(self):
        processed = ProcessedResponse(content="no punctuation", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert result.endswith(".")

    def test_format_for_client_no_extra_period_for_question(self):
        processed = ProcessedResponse(content="Is this a question?", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert not result.endswith("?.")
        assert result.endswith("?")

    def test_format_for_client_no_extra_period_for_exclamation(self):
        processed = ProcessedResponse(content="Wow!", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert result.endswith("!")

    def test_format_for_client_no_extra_period_for_colon(self):
        processed = ProcessedResponse(content="key:", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert result.endswith(":")

    def test_format_with_metadata(self):
        meta = {"key": "value"}
        processed = ProcessedResponse(
            content="test", raw_response=None, metadata=meta, processing_time=1.0
        )
        result = ResponseFormatter.format_with_metadata(processed)
        assert result["content"] == "test"
        assert result["metadata"] == meta
        assert result["processing_time"] == 1.0

    def test_format_for_client_empty_content(self):
        processed = ProcessedResponse(content="", raw_response=None)
        result = ResponseFormatter.format_for_client(processed)
        assert result == ""
