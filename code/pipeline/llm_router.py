import os
import io
import json
import base64
import logging
from PIL import Image
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
import litellm

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini/gemini-2.0-flash')
LLM_API_KEY = os.getenv('LLM_API_KEY', os.getenv('GEMINI_API_KEY', ''))
VISION_MODEL = os.getenv('VISION_MODEL', LLM_MODEL)
FALLBACK_CHAIN = os.getenv('LLM_FALLBACK_CHAIN', '')


class ConfigurationError(Exception):
    pass


def _pil_to_base64(pil_image, fmt='JPEG'):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt, quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _build_image_block(pil_image):
    b64 = _pil_to_base64(pil_image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }


def _build_text_block(text):
    return {"type": "text", "text": text}


def _build_messages(contents, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    content_blocks = []
    for item in contents:
        if isinstance(item, Image.Image):
            content_blocks.append(_build_image_block(item))
        elif isinstance(item, str):
            content_blocks.append(_build_text_block(item))
        else:
            logger.warning(f"Unexpected content type: {type(item)}")
    messages.append({"role": "user", "content": content_blocks})
    return messages


def _is_retryable(exception):
    if isinstance(exception, ConfigurationError):
        return False
    try:
        status = getattr(exception, 'status_code', 0) or getattr(exception, 'code', 0)
        return status not in (429, 401, 403)
    except Exception:
        return True


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_is_retryable)
)
def _build_completion_kwargs(messages, model, response_schema, temperature, timeout, api_key=None):
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


def _try_single_call(messages, model, response_schema, temperature, timeout, api_key=None):
    kwargs, provider = _build_completion_kwargs(
        messages, model, response_schema, temperature, timeout, api_key
    )
    try:
        return litellm.completion(**kwargs)
    except Exception:
        logger.warning(f"Provider {provider} with {model} failed, retrying without schema")
        kwargs.pop("api_key", None)
        kwargs.pop("extra_body", None)
        kwargs.pop("response_format", None)
        return litellm.completion(**kwargs)


def llm_complete(messages, model=None, api_key=None, response_schema=None, temperature=0.0, timeout=180):
    litellm.set_verbose = False

    model = model or VISION_MODEL
    api_key = api_key or LLM_API_KEY

    if not api_key:
        raise ConfigurationError(
            "No LLM API key found. Set LLM_API_KEY (or GEMINI_API_KEY) in .env"
        )

    return _try_single_call(messages, model, response_schema, temperature, timeout, api_key)


def llm_complete_with_fallback(messages, model=None, response_schema=None, temperature=0.0, timeout=180):
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
                timeout, api_key=None
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


def extract_json(response):
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


def get_token_usage(response):
    if hasattr(response, 'usage') and response.usage:
        return {
            'input_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'output_tokens': getattr(response.usage, 'completion_tokens', 0),
        }
    return {'input_tokens': 0, 'output_tokens': 0}
