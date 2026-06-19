import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'gemini-2.0-flash'
MAX_IMAGES_PER_CALL = 4
RATE_LIMIT_RPM = 90
RATE_LIMIT_TPM = 900000

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
