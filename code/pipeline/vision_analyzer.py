import json
import os
import logging
from pathlib import Path
from google import genai
from google.genai import errors, types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

from config import GEMINI_API_KEY, MODEL_NAME, STRUCTURED_OUTPUT_SCHEMA, CACHE_ENABLED, CACHE_DIR
from utils.image_utils import resize_image
from utils.cache import ResponseCache

logger = logging.getLogger(__name__)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=180_000)
)

_cache = ResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)

TOKENS_PER_512_IMAGE = 258

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / 'prompts' / 'system_vision.txt'
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8') if SYSTEM_PROMPT_PATH.exists() else ''


def _build_prompt(claim_object, user_claim, minimum_evidence, image_ids):
    image_section = '\n'.join([f'Image {i+1}: {img_id}' for i, img_id in enumerate(image_ids)])
    return (
        f"Object type: {claim_object}\n"
        f"User claim: {user_claim}\n"
        f"Minimum evidence required: {minimum_evidence}\n\n"
        f"Submitted images:\n{image_section}\n\n"
        f"Analyze each image carefully. Determine:\n"
        f"1. What issue type is visible (if any)\n"
        f"2. Which object part is affected\n"
        f"3. The confidence level of your assessment\n"
        f"4. Which image IDs support your finding\n"
        f"5. Whether evidence standard is met\n"
        f"6. The severity of the damage\n"
        f"7. The quality of each image (blurry, dark, etc.)\n"
        f"8. Whether manipulation is suspected\n"
        f"9. Any risk flags that apply\n\n"
        f"Return ONLY valid JSON matching the provided schema."
    )


def _estimate_tokens(prompt, num_images):
    text_tokens = int(len(prompt.split()) * 1.3)
    image_tokens = num_images * TOKENS_PER_512_IMAGE
    return text_tokens + image_tokens


def _is_retryable(exception):
    if isinstance(exception, errors.APIError):
        return exception.code != 429
    return True


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(_is_retryable)
)
def analyze_with_gemini(images, prompt, image_ids, token_tracker):
    processed_images = []
    for img_path in images:
        pil_img = resize_image(img_path)
        processed_images.append(pil_img)

    cached = _cache.get(prompt, images, MODEL_NAME)
    if cached is not None:
        logger.debug(f"Cache hit for {len(images)} images")
        return cached

    input_tokens = _estimate_tokens(prompt, len(processed_images))
    token_tracker.add_input(input_tokens)

    contents = []
    for i, pil_img in enumerate(processed_images):
        img_id = image_ids[i] if i < len(image_ids) else f'img_{i+1}'
        contents.append(f'=== Image {i+1}: {img_id} ===')
        contents.append(pil_img)
    contents.append(prompt)

    generation_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=0.95,
        response_mime_type='application/json',
        response_json_schema=STRUCTURED_OUTPUT_SCHEMA,
        system_instruction=SYSTEM_PROMPT if SYSTEM_PROMPT else None,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=generation_config,
    )

    output_tokens = int(len(response.text.split()) * 1.3)
    token_tracker.add_output(output_tokens)

    try:
        parsed = json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse structured response: {e}")
        return None

    if 'confidence' in parsed and isinstance(parsed['confidence'], str):
        try:
            parsed['confidence'] = float(parsed['confidence'])
        except ValueError:
            parsed['confidence'] = 0.0

    if 'supporting_image_ids' in parsed and isinstance(parsed['supporting_image_ids'], list):
        parsed['supporting_image_ids'] = ';'.join(parsed['supporting_image_ids'])

    if 'risk_flags' in parsed and isinstance(parsed['risk_flags'], list):
        parsed['risk_flags'] = ';'.join(parsed['risk_flags']) if parsed['risk_flags'] else 'none'

    if 'image_quality_issues' in parsed and isinstance(parsed['image_quality_issues'], list):
        parsed['image_quality_issues'] = ';'.join(parsed['image_quality_issues']) if parsed['image_quality_issues'] else 'none'

    _cache.set(prompt, images, MODEL_NAME, parsed)

    return parsed


def run_vision_analysis(preprocessed, evidence_rule, token_tracker):
    if not preprocessed['valid_image']:
        return None

    prompt = _build_prompt(
        preprocessed['claim_object'],
        preprocessed['user_claim'],
        evidence_rule.get('minimum_image_evidence', 'The claimed object and relevant part should be visible clearly.'),
        preprocessed['image_ids']
    )

    logger.info(f"Sending {len(preprocessed['image_paths'])} images to Gemini for user {preprocessed['user_id']}")
    return analyze_with_gemini(preprocessed['image_paths'], prompt, preprocessed['image_ids'], token_tracker)


FALLBACK_VISION_RESULT = {
    'issue_type': 'unknown',
    'object_part': 'unknown',
    'confidence': 0.0,
    'supporting_image_ids': 'none',
    'evidence_standard_met': False,
    'visual_description': 'API call failed after retries',
    'severity': 'unknown',
    'image_quality': 'poor',
    'image_quality_issues': 'none',
    'manipulation_suspected': False,
    'risk_flags': 'manual_review_required'
}


def safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter=None):
    if not preprocessed['valid_image']:
        return None

    if rate_limiter:
        rate_limiter.wait_if_needed()

    try:
        return run_vision_analysis(preprocessed, evidence_rule, token_tracker)
    except Exception as e:
        logger.error(f"Vision analysis failed for user {preprocessed['user_id']}: {e}")
        return dict(FALLBACK_VISION_RESULT)
