import os
import sys
import logging
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_env_path)

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.0-flash')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini')
LLM_MODEL = os.getenv('LLM_MODEL', f'{LLM_PROVIDER}/{MODEL_NAME}')
VISION_MODEL = os.getenv('VISION_MODEL', LLM_MODEL)
LLM_API_KEY = os.getenv('LLM_API_KEY', os.getenv('GEMINI_API_KEY', ''))
LLM_FALLBACK_CHAIN = os.getenv('LLM_FALLBACK_CHAIN', '')
MAX_IMAGES_PER_CALL = int(os.getenv('MAX_IMAGES_PER_CALL', '4'))
RATE_LIMIT_RPM = int(os.getenv('RATE_LIMIT_RPM', '2000'))
RATE_LIMIT_TPM = int(os.getenv('RATE_LIMIT_TPM', '4000000'))
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() in ('true', '1', 'yes')
CACHE_DIR = os.getenv('CACHE_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache'))
SAFEGUARD_ENABLED = os.getenv('SAFEGUARD_ENABLED', 'true').lower() in ('true', '1', 'yes')
TEMP_DIR = os.getenv('TEMP_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.tmp'))

ALLOWED_ISSUE_TYPES = ['dent', 'scratch', 'crack', 'glass_shatter', 'broken_part',
                       'missing_part', 'torn_packaging', 'crushed_packaging',
                       'water_damage', 'stain', 'none', 'unknown']

ALLOWED_CLAIM_STATUS = ['supported', 'contradicted', 'not_enough_information']

ALLOWED_OBJECT_PARTS = {
    'car': ['front_bumper', 'rear_bumper', 'door', 'hood', 'windshield',
            'side_mirror', 'headlight', 'taillight', 'fender', 'quarter_panel', 'body', 'unknown'],
    'laptop': ['screen', 'keyboard', 'trackpad', 'hinge', 'lid', 'corner', 'port', 'base', 'body', 'unknown'],
    'package': ['box', 'package_corner', 'package_side', 'seal', 'label', 'contents', 'item', 'unknown']
}

ALLOWED_RISK_FLAGS = ['none', 'blurry_image', 'cropped_or_obstructed', 'low_light_or_glare',
                      'wrong_angle', 'wrong_object', 'wrong_object_part', 'damage_not_visible',
                      'claim_mismatch', 'possible_manipulation', 'non_original_image',
                      'text_instruction_present', 'user_history_risk', 'manual_review_required']

ALLOWED_SEVERITY = ['none', 'low', 'medium', 'high', 'unknown']

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_type": {"type": "string", "enum": ALLOWED_ISSUE_TYPES},
        "object_part": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "supporting_image_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_standard_met": {"type": "boolean"},
        "visual_description": {"type": "string", "maxLength": 120},
        "severity": {"type": "string", "enum": ALLOWED_SEVERITY},
        "image_quality": {"type": "string", "enum": ["good", "fair", "poor"]},
        "image_quality_issues": {
            "type": "array",
            "items": {"type": "string", "enum": ["blurry", "dark", "glare", "cropped", "obstructed", "wrong_angle", "none"]}
        },
        "manipulation_suspected": {"type": "boolean"},
        "risk_flags": {"type": "array", "items": {"type": "string", "enum": ALLOWED_RISK_FLAGS}}
    },
    "required": ["issue_type", "object_part", "confidence", "supporting_image_ids",
                  "evidence_standard_met", "visual_description", "severity",
                  "image_quality", "image_quality_issues", "manipulation_suspected", "risk_flags"]
}


def validate_config():
    errors = []

    api_key = LLM_API_KEY or GEMINI_API_KEY
    if not api_key:
        errors.append("LLM_API_KEY (or GEMINI_API_KEY) is not set in .env file")
    elif len(api_key) < 10:
        errors.append(f"API key appears invalid (too short: {len(api_key)} chars)")

    if MAX_IMAGES_PER_CALL < 1 or MAX_IMAGES_PER_CALL > 10:
        errors.append(f"MAX_IMAGES_PER_CALL={MAX_IMAGES_PER_CALL} is out of range (1-10)")

    if RATE_LIMIT_RPM < 1:
        errors.append(f"RATE_LIMIT_RPM={RATE_LIMIT_RPM} is invalid")

    if RATE_LIMIT_TPM < 1:
        errors.append(f"RATE_LIMIT_TPM={RATE_LIMIT_TPM} is invalid")

    if errors:
        for err in errors:
            logger.error(f"Config error: {err}")
        return False
    return True
