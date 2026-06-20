import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from PIL import Image

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', '')
LLM_MODEL = os.getenv('LLM_MODEL', '')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
VISION_MODEL = os.getenv('VISION_MODEL', LLM_MODEL)
FALLBACK_CHAIN = os.getenv('LLM_FALLBACK_CHAIN', '')


class ConfigurationError(Exception):
    pass


class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after
        self.message = message


def _pil_to_base64(pil_image: Image.Image, fmt: str = 'JPEG') -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_image_block(pil_image: Image.Image) -> Dict:
    b64 = _pil_to_base64(pil_image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }


def _build_text_block(text: str) -> Dict:
    return {"type": "text", "text": text}


def _build_messages(contents: List[Union[str, Image.Image]], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content_blocks: List[Dict[str, Any]] = []
    for item in contents:
        if isinstance(item, Image.Image):
            content_blocks.append(_build_image_block(item))
        elif isinstance(item, str):
            content_blocks.append(_build_text_block(item))
        else:
            logger.warning(f"Unexpected content type: {type(item)}")
    messages.append({"role": "user", "content": content_blocks})
    return messages


def _extract_retry_after(error: BaseException) -> float:
    msg = str(error)
    try:
        m = re.search(r'(?:retryAfter|retryDelay)["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', msg)
        if m:
            return float(m.group(1))
        data = json.loads(msg)
        details = data.get('error', {}).get('details', [])
        for d in details:
            if d.get('@type', '').endswith('RetryInfo'):
                rd = d.get('retryDelay', '0s')
                m2 = re.search(r'(\d+(?:\.\d+)?)s', rd)
                if m2:
                    return float(m2.group(1))
    except (json.JSONDecodeError, AttributeError, KeyError, ValueError):
        pass
    return 0.0


def _wrap_exception(e: Exception) -> Exception:
    msg = str(e)
    if '429' in msg or 'rate_limit' in msg.lower() or 'quota' in msg.lower() or 'resource_exhausted' in msg.lower():
        retry_after = _extract_retry_after(e)
        return RateLimitError(msg, retry_after)
    return e


def _build_completion_kwargs(messages: List[Dict[str, Any]], model: str, response_schema: Optional[Dict[str, Any]], temperature: float, timeout: int, api_key: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    provider = model.split('/')[0] if '/' in model else LLM_PROVIDER
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
        "timeout": timeout,
        "num_retries": 0,
    }
    if api_key:
        kwargs["api_key"] = api_key

    if response_schema:
        if provider in ('gemini', 'anthropic', 'groq', 'openrouter'):
            kwargs["response_format"] = {"type": "json_object"}
            if provider == 'gemini':
                kwargs["extra_body"] = {
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": response_schema
                    }
                }
        else:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True}
            }
    else:
        kwargs["response_format"] = {"type": "json_object"}

    return kwargs, provider


def _try_single_call(messages: List[Dict[str, Any]], model: str, response_schema: Optional[Dict[str, Any]], temperature: float, timeout: int, api_key: Optional[str] = None, rate_limiter=None) -> Any:
    kwargs, provider = _build_completion_kwargs(
        messages, model, response_schema, temperature, timeout, api_key
    )
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            if rate_limiter:
                rate_limiter.wait_for_cooldown()
            result = litellm.completion(**kwargs)
            if rate_limiter:
                rate_limiter.note_success()
            return result
        except Exception as e:
            wrapped = _wrap_exception(e)
            if isinstance(wrapped, RateLimitError):
                if rate_limiter:
                    rate_limiter.note_429(str(wrapped))
                if attempt < max_retries:
                    continue
                raise wrapped
            logger.warning(f"Provider {provider} with {model} failed ({type(wrapped).__name__}), retrying without schema")
            kwargs.pop("api_key", None)
            kwargs.pop("extra_body", None)
            kwargs.pop("response_format", None)
            break
    try:
        if rate_limiter:
            rate_limiter.wait_for_cooldown()
        result = litellm.completion(**kwargs)
        if rate_limiter:
            rate_limiter.note_success()
        return result
    except Exception as e:
        wrapped2 = _wrap_exception(e)
        if rate_limiter and isinstance(wrapped2, RateLimitError):
            rate_limiter.note_429(str(wrapped2))
        raise wrapped2


def llm_complete(messages: List[Dict[str, Any]], model: Optional[str] = None, api_key: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0, timeout: int = 180, rate_limiter=None) -> Any:
    litellm.set_verbose = False

    model = model or VISION_MODEL
    api_key = api_key or LLM_API_KEY

    if not api_key:
        raise ConfigurationError(
            "No LLM API key found. Set LLM_API_KEY (or a provider-specific key like GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY) in .env"
        )

    return _try_single_call(messages, model, response_schema, temperature, timeout, api_key, rate_limiter)


def llm_complete_with_fallback(messages: List[Dict[str, Any]], model: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0, timeout: int = 180, rate_limiter=None) -> Any:
    litellm.set_verbose = False

    model = model or VISION_MODEL

    models_to_try = [model]
    if FALLBACK_CHAIN:
        fallback_models = [m.strip() for m in FALLBACK_CHAIN.split(',') if m.strip()]
        models_to_try.extend(fallback_models)

    last_error = None
    for attempt_model in models_to_try:
        try:
            logger.info(f"Attempting LLM call with model: {attempt_model}")
            return _try_single_call(
                messages, attempt_model, response_schema, temperature,
                timeout, api_key=None, rate_limiter=rate_limiter
            )
        except ConfigurationError:
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"Model {attempt_model} failed: {e}. Trying next fallback...")
            continue

    if last_error:
        raise last_error
    raise Exception("All LLM providers in fallback chain failed")


def _strip_think_tags(text: str) -> str:
    import re
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _clean_response_text(text: str) -> str:
    text = _strip_think_tags(text)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```', '', text)
    return text.strip()


def _try_parse_json(text: str) -> Optional[Dict]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        logger.warning(f"LLM returned JSON array with {len(parsed)} items, using first item")
        return parsed[0]
    if not isinstance(parsed, dict):
        return None
    return parsed


def _find_json_object(text: str) -> Optional[Dict]:
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if start == -1:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                candidate = text[start:i + 1]
                result = _try_parse_json(candidate)
                if result is not None:
                    return result
                start = -1
    if start != -1:
        candidate = text[start:]
        result = _try_parse_json(candidate)
        if result is not None:
            return result
    return None


def extract_json(response: Any) -> Optional[Dict]:
    text = response.choices[0].message.content if response.choices else ''
    if not text:
        logger.error("Empty response from LLM")
        return None
    text = _clean_response_text(text)
    parsed = _try_parse_json(text)
    if parsed is not None:
        return parsed
    parsed = _find_json_object(text)
    if parsed is not None:
        logger.warning("Found JSON object via brace matching after direct parse failed")
        return parsed
    logger.error("Failed to parse LLM response as JSON")
    return None


def get_token_usage(response: Any) -> Dict:
    if hasattr(response, 'usage') and response.usage:
        return {
            'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'output_tokens': getattr(response.usage, 'completion_tokens', 0),
        }
    return {'input_tokens': 0, 'output_tokens': 0}
