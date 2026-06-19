import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    ALLOWED_ISSUE_TYPES,
    ALLOWED_OBJECT_PARTS,
    CACHE_DIR,
    CACHE_ENABLED,
    STRUCTURED_OUTPUT_SCHEMA,
    VISION_MODEL,
)
from pipeline.llm_router import ConfigurationError, extract_json, get_token_usage, llm_complete_with_fallback
from utils.cache import ResponseCache
from utils.image_utils import resize_image
from utils.rate_limiter import AdaptiveRateLimiter
from utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)

_cache = ResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)

TOKENS_PER_512_IMAGE = 258

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / 'prompts' / 'system_vision.txt'
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding='utf-8') if SYSTEM_PROMPT_PATH.exists() else ''


def _build_prompt(claim_object: str, user_claim: str, minimum_evidence: str, image_ids: List[str]) -> str:
    image_section = '\n'.join([f'Image {i+1}: {img_id}' for i, img_id in enumerate(image_ids)])
    valid_parts = ALLOWED_OBJECT_PARTS.get(claim_object, ['unknown'])
    parts_str = ', '.join(valid_parts)
    issue_types_str = ', '.join(ALLOWED_ISSUE_TYPES)
    return (
        f"Object type: {claim_object}\n"
        f"User claim: {user_claim}\n"
        f"Minimum evidence required: {minimum_evidence}\n\n"
        f"Submitted images:\n{image_section}\n\n"
        f"Valid issue types for this object: {issue_types_str}\n"
        f"Valid object parts for {claim_object}: {parts_str}\n\n"
        f"Review all images together and determine:\n"
        f"1. What issue type is visible (if any) — single overall assessment\n"
        f"2. Which object part is affected (must be from the valid parts list)\n"
        f"3. The confidence level of your assessment\n"
        f"4. Which image IDs support your finding\n"
        f"5. Whether evidence standard is met\n"
        f"6. The severity of the damage\n"
        f"7. The quality of each image (blurry, dark, etc.)\n"
        f"8. Whether manipulation is suspected\n"
        f"9. Any risk flags that apply\n\n"
        f"Return a single JSON object (NOT an array), matching the provided schema exactly."
    )


def _estimate_tokens(prompt: str, num_images: int) -> int:
    text_tokens = int(len(prompt.split()) * 1.3)
    image_tokens = num_images * TOKENS_PER_512_IMAGE
    return text_tokens + image_tokens


def _parse_response(parsed: Dict) -> Dict:
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

    return parsed


def analyze_with_llm(images: List[str], prompt: str, image_ids: List[str], token_tracker: TokenTracker, rate_limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[Dict]:
    processed_images = []
    for img_path in images:
        pil_img = resize_image(img_path)
        processed_images.append(pil_img)

    cached = _cache.get(prompt, images, VISION_MODEL)
    if cached is not None:
        logger.debug(f"Cache hit for {len(images)} images")
        return cached

    input_tokens = _estimate_tokens(prompt, len(processed_images))
    token_tracker.add_input(input_tokens)

    contents: list[Any] = []
    for i, pil_img in enumerate(processed_images):
        img_id = image_ids[i] if i < len(image_ids) else f'img_{i+1}'
        contents.append(f'=== Image {i+1}: {img_id} ===')
        contents.append(pil_img)
    contents.append(prompt)

    messages: List[Dict[str, Any]] = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    content_blocks: List[Dict[str, Any]] = []
    for item in contents:
        if isinstance(item, str):
            content_blocks.append({"type": "text", "text": item})
        else:
            import base64
            import io
            buf = io.BytesIO()
            item.save(buf, format='JPEG', quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    messages.append({"role": "user", "content": content_blocks})

    response = llm_complete_with_fallback(
        messages=messages,
        response_schema=STRUCTURED_OUTPUT_SCHEMA,
        temperature=0.0,
        rate_limiter=rate_limiter,
    )

    usage = get_token_usage(response)
    token_tracker.add_output(usage['output_tokens'])

    parsed = extract_json(response)
    if parsed is None:
        return None

    parsed = _parse_response(parsed)
    _cache.set(prompt, images, VISION_MODEL, parsed)
    return parsed


def run_vision_analysis(preprocessed: Dict, evidence_rule: Dict, token_tracker: TokenTracker, rate_limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[Dict]:
    if not preprocessed['valid_image']:
        return None

    prompt = _build_prompt(
        preprocessed['claim_object'],
        preprocessed['user_claim'],
        evidence_rule.get('minimum_image_evidence', 'The claimed object and relevant part should be visible clearly.'),
        preprocessed['image_ids']
    )

    logger.info(f"Sending {len(preprocessed['image_paths'])} images for analysis for user {preprocessed['user_id']}")
    return analyze_with_llm(preprocessed['image_paths'], prompt, preprocessed['image_ids'], token_tracker, rate_limiter)


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


def safe_run_vision_analysis(preprocessed: Dict, evidence_rule: Dict, token_tracker: TokenTracker, rate_limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[Dict]:
    if not preprocessed['valid_image']:
        return None

    try:
        return run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return dict(FALLBACK_VISION_RESULT)
    except Exception as e:
        logger.error(f"Vision analysis failed for user {preprocessed['user_id']}: {e}")
        return dict(FALLBACK_VISION_RESULT)
