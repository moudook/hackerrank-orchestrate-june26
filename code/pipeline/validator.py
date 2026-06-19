import logging
import os
from typing import Dict, Union

from config import ALLOWED_CLAIM_STATUS, ALLOWED_ISSUE_TYPES, ALLOWED_OBJECT_PARTS, ALLOWED_RISK_FLAGS, ALLOWED_SEVERITY

logger = logging.getLogger(__name__)

RISK_FLAG_DEPENDENCIES = {
    'blurry_image': 'image_quality',
    'cropped_or_obstructed': 'image_quality',
    'low_light_or_glare': 'image_quality',
    'wrong_angle': 'image_quality',
    'wrong_object': 'claim_mismatch',
    'wrong_object_part': 'claim_mismatch',
    'damage_not_visible': 'claim_mismatch',
    'claim_mismatch': 'claim_mismatch',
    'possible_manipulation': 'authenticity',
    'non_original_image': 'authenticity',
    'text_instruction_present': 'authenticity',
    'user_history_risk': 'history',
    'manual_review_required': 'review',
}


def _clamp_confidence(value: Union[str, float, int]) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return 0.0


def _bool_or_false(value: Union[bool, str, int]) -> bool:
    if value is True or str(value).lower() in ('true', '1', 'yes'):
        return True
    return False


def validate_output(output: Dict) -> Dict:
    validated = dict(output)
    claim_object = validated.get('claim_object', 'unknown')

    issue_type = validated.get('issue_type', 'unknown')
    if issue_type not in ALLOWED_ISSUE_TYPES:
        validated['issue_type'] = 'unknown'

    valid_parts = ALLOWED_OBJECT_PARTS.get(claim_object, ['unknown'])
    object_part = validated.get('object_part', 'unknown')
    if object_part not in valid_parts:
        object_part_lower = str(object_part).lower().replace(' ', '_')
        if object_part_lower in valid_parts:
            validated['object_part'] = object_part_lower
        else:
            validated['object_part'] = 'unknown'

    claim_status = validated.get('claim_status', 'not_enough_information')
    if claim_status not in ALLOWED_CLAIM_STATUS:
        validated['claim_status'] = 'not_enough_information'

    severity = validated.get('severity', 'unknown')
    if severity not in ALLOWED_SEVERITY:
        validated['severity'] = 'unknown'

    raw_flags = validated.get('risk_flags', 'none')
    if raw_flags in (None, '', 'none', []):
        validated['risk_flags'] = 'none'
    else:
        if isinstance(raw_flags, list):
            flags = [str(f).strip() for f in raw_flags if str(f).strip()]
        else:
            flags = [f.strip() for f in str(raw_flags).split(';') if f.strip()]

        valid_flags = []
        for f in flags:
            if f in ALLOWED_RISK_FLAGS:
                valid_flags.append(f)

        if 'manual_review_required' not in valid_flags:
            for critical in ['blurry_image', 'low_light_or_glare', 'possible_manipulation',
                           'claim_mismatch', 'non_original_image', 'text_instruction_present']:
                if critical in valid_flags and 'manual_review_required' not in valid_flags:
                    valid_flags.append('manual_review_required')

        validated['risk_flags'] = ';'.join(valid_flags) if valid_flags else 'none'

    raw_ids = validated.get('supporting_image_ids', 'none')
    if raw_ids and raw_ids not in ('none', '', [], '[]'):
        if isinstance(raw_ids, list):
            ids = [str(id).strip() for id in raw_ids if str(id).strip()]
        else:
            ids = [id.strip() for id in str(raw_ids).split(';') if id.strip()]
        cleaned = [os.path.splitext(id)[0] for id in ids]
        validated['supporting_image_ids'] = ';'.join(cleaned) if cleaned else 'none'
    else:
        validated['supporting_image_ids'] = 'none'

    validated['evidence_standard_met'] = _bool_or_false(validated.get('evidence_standard_met', False))

    validated['valid_image'] = _bool_or_false(validated.get('valid_image', False))

    evidence_reason = validated.get('evidence_standard_met_reason', '')
    if not evidence_reason or str(evidence_reason).strip() == '':
        if claim_status == 'not_enough_information':
            validated['evidence_standard_met_reason'] = validated.get('claim_status_justification', 'Insufficient evidence')
        else:
            validated['evidence_standard_met_reason'] = ''

    justification = validated.get('claim_status_justification', '')
    if not justification or str(justification).strip() == '':
        validated['claim_status_justification'] = f'Claim status: {validated["claim_status"]}.'

    return validated
