import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini/gemini-2.0-flash')
LLM_API_KEY = os.getenv('LLM_API_KEY', os.getenv('GEMINI_API_KEY', ''))
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


def _is_retryable(exception: BaseException) -> bool:
    if isinstance(exception, ConfigurationError):
        return False
    if isinstance(exception, RateLimitError):
        return True
    try:
        status = getattr(exception, 'status_code', 0) or getattr(exception, 'code', 0)
        if status == 429:
            return True
        return status not in (401, 403)
    except Exception:
        return True


def _wrap_exception(e: Exception) -> Exception:
    msg = str(e)
    if '429' in msg or 'rate_limit' in msg.lower() or 'quota' in msg.lower() or 'resource_exhausted' in msg.lower():
        retry_after = _extract_retry_after(e)
        return RateLimitError(msg, retry_after)
    return e


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(6),
    retry=retry_if_exception(_is_retryable)
)
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

    if provider == 'gemini':
        kwargs["response_format"] = {"type": "json_object"}
        if response_schema:
            kwargs["extra_body"] = {
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": response_schema
                }
            }
    elif response_schema:
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
    try:
        if rate_limiter:
            rate_limiter.wait_for_cooldown()
        result = litellm.completion(**kwargs)
        if rate_limiter:
            rate_limiter.note_success()
        return result
    except Exception as e:
        wrapped = _wrap_exception(e)
        if rate_limiter and isinstance(wrapped, RateLimitError):
            rate_limiter.note_429(str(wrapped))
            if wrapped.retry_after > 0:
                import time
                time.sleep(min(wrapped.retry_after + 1, 120))
        logger.warning(f"Provider {provider} with {model} failed ({type(wrapped).__name__}), retrying without schema")
        kwargs.pop("api_key", None)
        kwargs.pop("extra_body", None)
        kwargs.pop("response_format", None)
        try:
            result = litellm.completion(**kwargs)
            if rate_limiter:
                rate_limiter.note_success()
            return result
        except Exception as e2:
            wrapped2 = _wrap_exception(e2)
            if rate_limiter and isinstance(wrapped2, RateLimitError):
                rate_limiter.note_429(str(wrapped2))
            raise wrapped2


def llm_complete(messages: List[Dict[str, Any]], model: Optional[str] = None, api_key: Optional[str] = None, response_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0, timeout: int = 180, rate_limiter=None) -> Any:
    litellm.set_verbose = False

    model = model or VISION_MODEL
    api_key = api_key or LLM_API_KEY

    if not api_key:
        raise ConfigurationError(
            "No LLM API key found. Set LLM_API_KEY (or GEMINI_API_KEY) in .env"
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


def extract_json(response: Any) -> Optional[Dict]:
    text = response.choices[0].message.content if response.choices else ''
    if not text:
        logger.error("Empty response from LLM")
        return None
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        return None


def get_token_usage(response: Any) -> Dict:
    if hasattr(response, 'usage') and response.usage:
        return {
            'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'output_tokens': getattr(response.usage, 'completion_tokens', 0),
        }
    return {'input_tokens': 0, 'output_tokens': 0}
