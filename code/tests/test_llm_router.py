import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pipeline.llm_router import (
    ConfigurationError,
    RateLimitError,
    _build_image_block,
    _build_text_block,
    _clean_response_text,
    _extract_retry_after,
    _find_json_object,
    _pil_to_base64,
    _strip_think_tags,
    _try_parse_json,
    extract_json,
    get_token_usage,
    llm_complete,
    llm_complete_with_fallback,
)


class TestLLMRouterUtilities:
    def test_pil_to_base64(self):
        img = Image.new('RGB', (10, 10), color='red')
        b64 = _pil_to_base64(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_build_image_block(self):
        img = Image.new('RGB', (10, 10), color='blue')
        block = _build_image_block(img)
        assert block['type'] == 'image_url'
        assert block['image_url']['url'].startswith('data:image/jpeg;base64,')

    def test_build_text_block(self):
        block = _build_text_block('hello world')
        assert block['type'] == 'text'
        assert block['text'] == 'hello world'


class TestExtractJSON:
    def test_extract_valid_json(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value"}'
        result = extract_json(mock_response)
        assert result == {"key": "value"}

    def test_extract_json_with_markdown_fence(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"key": "value"}\n```'
        result = extract_json(mock_response)
        assert result == {"key": "value"}

    def test_extract_empty_response(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ''
        result = extract_json(mock_response)
        assert result is None

    def test_extract_invalid_json(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'not json at all'
        result = extract_json(mock_response)
        assert result is None

    def test_extract_no_choices(self):
        mock_response = MagicMock()
        mock_response.choices = []
        result = extract_json(mock_response)
        assert result is None

    def test_extract_think_tags_stripped(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '<think>Let me analyze this image...</think>{"key": "value"}'
        result = extract_json(mock_response)
        assert result == {"key": "value"}

    def test_extract_json_embedded_in_text(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Here is my analysis: {"key": "value"} Hope that helps.'
        result = extract_json(mock_response)
        assert result == {"key": "value"}

    def test_extract_json_multiple_objects(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"first": "object"} some text {"second": "object"}'
        result = extract_json(mock_response)
        # Should return the first valid object
        assert result == {"first": "object"}

    def test_extract_json_after_think_with_fence(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '<think>reasoning</think>```json\n{"key": "value"}\n```'
        result = extract_json(mock_response)
        assert result == {"key": "value"}

    def test_extract_json_truncated(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"key": "value", "nested": {"inner": "val"'
        result = extract_json(mock_response)
        assert result is None  # truncated JSON should fail gracefully


class TestResponseCleanup:
    def test_clean_response_text_think_tags(self):
        assert _clean_response_text('<think>foo</think>{"a": 1}') == '{"a": 1}'

    def test_clean_response_text_markdown_fence(self):
        assert _clean_response_text('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_clean_response_text_noop(self):
        assert _clean_response_text('{"a": 1}') == '{"a": 1}'

    def test_clean_response_text_empty(self):
        assert _clean_response_text('') == ''

    def test_find_json_object_simple(self):
        assert _find_json_object('{"a": 1}') == {"a": 1}

    def test_find_json_object_embedded(self):
        assert _find_json_object('text {"a": 1} more') == {"a": 1}

    def test_find_json_object_nested(self):
        assert _find_json_object('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_find_json_object_first_of_many(self):
        assert _find_json_object('{"a": 1} junk {"b": 2}') == {"a": 1}

    def test_find_json_object_invalid(self):
        assert _find_json_object('no json here') is None

    def test_try_parse_json_valid(self):
        assert _try_parse_json('{"a": 1}') == {"a": 1}

    def test_try_parse_json_invalid(self):
        assert _try_parse_json('not json') is None

    def test_try_parse_json_array(self):
        assert _try_parse_json('[{"a": 1}, {"b": 2}]') == {"a": 1}

    def test_try_parse_json_non_dict(self):
        assert _try_parse_json('[1, 2, 3]') is None

    def test_strip_think_tags_simple(self):
        assert _strip_think_tags('<think>foo</think>bar') == 'bar'

    def test_strip_think_tags_multiple(self):
        assert _strip_think_tags('<think>a</think><think>b</think>c') == 'c'

    def test_strip_think_tags_no_tags(self):
        assert _strip_think_tags('hello') == 'hello'


class TestGetTokenUsage:
    def test_with_usage(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        usage = get_token_usage(mock_response)
        assert usage['input_tokens'] == 100
        assert usage['output_tokens'] == 50

    def test_without_usage(self):
        mock_response = MagicMock()
        mock_response.usage = None
        usage = get_token_usage(mock_response)
        assert usage['input_tokens'] == 0
        assert usage['output_tokens'] == 0


class TestLLMCompleteWithFallback:
    @patch('pipeline.llm_router.litellm.completion')
    def test_primary_succeeds(self, mock_completion):
        mock_completion.return_value = MagicMock()
        with patch('pipeline.llm_router.FALLBACK_CHAIN', 'openai/gpt-4o'):
            result = llm_complete_with_fallback(
                messages=[{"role": "user", "content": "hello"}],
                model="gemini/gemini-2.0-flash",
            )
        assert result is not None
        mock_completion.assert_called_once()

    @patch('pipeline.llm_router.litellm.completion')
    def test_fallback_on_failure(self, mock_completion):
        mock_completion.side_effect = [
            Exception("Gemini down"),
            MagicMock(),
        ]
        with patch('pipeline.llm_router.FALLBACK_CHAIN', 'openai/gpt-4o'):
            result = llm_complete_with_fallback(
                messages=[{"role": "user", "content": "hello"}],
                model="gemini/gemini-2.0-flash",
            )
        assert result is not None
        assert mock_completion.call_count == 2

    @patch('pipeline.llm_router.litellm.completion')
    def test_all_fail_raises(self, mock_completion):
        mock_completion.side_effect = Exception("All down")
        with patch('pipeline.llm_router.FALLBACK_CHAIN', 'openai/gpt-4o'):
            with pytest.raises(Exception):
                llm_complete_with_fallback(
                    messages=[{"role": "user", "content": "hello"}],
                    model="gemini/gemini-2.0-flash",
                )

    @patch('pipeline.llm_router.litellm.completion')
    def test_empty_fallback_chain_still_raises(self, mock_completion):
        mock_completion.side_effect = Exception("Down")
        with patch('pipeline.llm_router.FALLBACK_CHAIN', ''):
            with pytest.raises(Exception):
                llm_complete_with_fallback(
                    messages=[{"role": "user", "content": "hello"}],
                    model="gemini/gemini-2.0-flash",
                )
        assert mock_completion.call_count >= 1

    def test_extract_retry_after_str_direct(self):
        e = Exception('retryDelay: "53.686s"')
        d = _extract_retry_after(e)
        assert abs(d - 53.686) < 0.01

    def test_extract_retry_after_no_match(self):
        e = Exception('no delay here')
        d = _extract_retry_after(e)
        assert d == 0.0

    def test_rate_limit_error_has_retry_after(self):
        err = RateLimitError("rate limited", retry_after=30.5)
        assert err.retry_after == 30.5
        assert "rate limited" in str(err)

    def test_rate_limit_error_default_retry_after(self):
        err = RateLimitError("rate limited")
        assert err.retry_after == 0.0

    @patch('pipeline.llm_router.litellm.completion')
    def test_no_fallback_and_all_fail(self, mock_completion):
        mock_completion.side_effect = Exception("API Error")
        with patch('pipeline.llm_router.FALLBACK_CHAIN', ''):
            with pytest.raises(Exception):
                llm_complete_with_fallback(
                    messages=[{"role": "user", "content": "hello"}],
                    model="gemini/gemini-2.0-flash",
                )

    @patch('pipeline.llm_router.litellm.completion')
    def test_multiple_fallbacks(self, mock_completion):
        mock_completion.side_effect = [
            Exception("Gemini down"),
            Exception("OpenAI down"),
            MagicMock(),
        ]
        with patch('pipeline.llm_router.FALLBACK_CHAIN', 'openai/gpt-4o,anthropic/claude-3'):
            result = llm_complete_with_fallback(
                messages=[{"role": "user", "content": "hello"}],
                model="gemini/gemini-2.0-flash",
            )
        assert result is not None
        assert mock_completion.call_count == 3


class TestLLMComplete:
    @patch('pipeline.llm_router.litellm.completion')
    def test_basic_call(self, mock_completion):
        mock_completion.return_value = MagicMock()
        with patch('pipeline.llm_router.LLM_API_KEY', 'test-key'):
            result = llm_complete(
                messages=[{"role": "user", "content": "hello"}],
                model="gemini/gemini-2.0-flash",
                api_key="test-key"
            )
        assert result is not None
        mock_completion.assert_called_once()

    @patch('pipeline.llm_router.litellm.completion')
    def test_missing_api_key(self, mock_completion):
        with patch('pipeline.llm_router.LLM_API_KEY', ''):
            with pytest.raises(ConfigurationError):
                llm_complete(
                    messages=[{"role": "user", "content": "hello"}],
                    api_key=""
                )
        mock_completion.assert_not_called()

    @patch('pipeline.llm_router.litellm.completion')
    def test_structured_output_gemini(self, mock_completion):
        mock_completion.return_value = MagicMock()
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        with patch('pipeline.llm_router.LLM_API_KEY', 'test-key'):
            llm_complete(
                messages=[{"role": "user", "content": "test"}],
                model="gemini/gemini-2.0-flash",
                response_schema=schema,
                api_key="test-key"
            )
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs['response_format'] == {"type": "json_object"}
        assert 'extra_body' in call_kwargs

    @patch('pipeline.llm_router.litellm.completion')
    def test_fallback_on_structured_output_failure(self, mock_completion):
        mock_completion.side_effect = [Exception("API Error"), MagicMock()]
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        with patch('pipeline.llm_router.LLM_API_KEY', 'test-key'):
            result = llm_complete(
                messages=[{"role": "user", "content": "test"}],
                model="gemini/gemini-2.0-flash",
                response_schema=schema,
                api_key="test-key"
            )
        assert result is not None
        assert mock_completion.call_count >= 2
