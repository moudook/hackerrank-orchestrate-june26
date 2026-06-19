import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

from pipeline.llm_router import (
    llm_complete, llm_complete_with_fallback, extract_json, _pil_to_base64,
    _build_image_block, _build_text_block,
    ConfigurationError, get_token_usage
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
