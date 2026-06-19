import json
import re
import logging
from google import genai
from google.genai import errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

from config import GEMINI_API_KEY, MODEL_NAME
from utils.image_utils import resize_image

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

TOKENS_PER_512_IMAGE = 258


def _build_prompt(claim_object, user_claim, minimum_evidence):
    return (
        f"You are a damage verification system. Analyze images and user claim.\n\n"
        f"Object type: {claim_object}\n"
        f'User claim: "{user_claim}"\n'
        f"Minimum evidence required: {minimum_evidence}\n\n"
        f"Return STRICT JSON only:\n"
        f"{{\n"
        f'  "issue_type": "dent|scratch|crack|glass_shatter|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown",\n'
        f'  "object_part": "exact_part_name",\n'
        f'  "confidence": 0.0-1.0,\n'
        f'  "supporting_image_ids": ["img_1"],\n'
        f'  "evidence_standard_met": true,\n'
        f'  "visual_description": "max 15 words",\n'
        f'  "severity": "none|low|medium|high|unknown"\n'
        f"}}\n\n"
        f"Rules:\n"
        f"- Use 'none' only if part is clearly visible and undamaged\n"
        f"- Use 'unknown' if image is blurry or part not visible\n"
        f"- object_part must match allowed list for {claim_object}\n"
        f"- No extra text, no markdown"
    )


def _parse_json_response(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


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
def analyze_with_gemini(images, prompt, token_tracker):
    processed_images = []
    for img_path in images:
        pil_img = resize_image(img_path)
        processed_images.append(pil_img)

    input_tokens = _estimate_tokens(prompt, len(processed_images))
    token_tracker.add_input(input_tokens)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt] + processed_images
    )

    output_tokens = int(len(response.text.split()) * 1.3)
    token_tracker.add_output(output_tokens)

    parsed = _parse_json_response(response.text)
    if parsed is None:
        logger.error(f"Failed to parse Gemini response: {response.text[:200]}")
        return None

    if 'confidence' in parsed and isinstance(parsed['confidence'], str):
        try:
            parsed['confidence'] = float(parsed['confidence'])
        except ValueError:
            parsed['confidence'] = 0.0

    if 'supporting_image_ids' in parsed and isinstance(parsed['supporting_image_ids'], list):
        parsed['supporting_image_ids'] = ';'.join(parsed['supporting_image_ids'])

    return parsed


def run_vision_analysis(preprocessed, evidence_rule, token_tracker):
    if not preprocessed['valid_image']:
        return None

    prompt = _build_prompt(
        preprocessed['claim_object'],
        preprocessed['user_claim'],
        evidence_rule.get('minimum_image_evidence', 'The claimed object and relevant part should be visible clearly.')
    )

    logger.info(f"Sending {len(preprocessed['image_paths'])} images to Gemini for user {preprocessed['user_id']}")
    return analyze_with_gemini(preprocessed['image_paths'], prompt, token_tracker)


FALLBACK_VISION_RESULT = {
    'issue_type': 'unknown',
    'object_part': 'unknown',
    'confidence': 0.0,
    'supporting_image_ids': 'none',
    'evidence_standard_met': False,
    'visual_description': 'API call failed after retries',
    'severity': 'unknown',
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
