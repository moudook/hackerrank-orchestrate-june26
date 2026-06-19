import logging
from config import ALLOWED_RISK_FLAGS, ALLOWED_OBJECT_PARTS

logger = logging.getLogger(__name__)

TRUST_MANIPULATION_PATTERNS = [
    'ignore all previous instructions',
    'ignore instructions',
    'approve the claim',
    'approve this claim',
    'mark this row',
    'skip manual review',
    'follow it and approve',
    'accept this quickly',
    'reopen tickets',
    'rejected again',
    'escalate publicly',
]


def _check_trust_manipulation(user_claim):
    if not user_claim:
        return False
    text = user_claim.lower()
    for pattern in TRUST_MANIPULATION_PATTERNS:
        if pattern in text:
            logger.info(f"Trust manipulation pattern detected: '{pattern}'")
            return True
    return False


def _build_risk_flags(vision_result, preprocessed, history):
    risk_flags = []

    vision_flags = vision_result.get('risk_flags', 'none')
    if isinstance(vision_flags, str) and vision_flags != 'none':
        for f in vision_flags.split(';'):
            f = f.strip()
            if f and f != 'none':
                risk_flags.append(f)
    elif isinstance(vision_flags, list):
        for f in vision_flags:
            if f != 'none':
                risk_flags.append(f)

    quality_issues = vision_result.get('image_quality_issues', 'none')
    if isinstance(quality_issues, str) and quality_issues not in ('none', ''):
        for issue in quality_issues.split(';'):
            issue = issue.strip()
            if issue == 'blurry' and 'blurry_image' not in risk_flags:
                risk_flags.append('blurry_image')
            elif issue == 'cropped' and 'cropped_or_obstructed' not in risk_flags:
                risk_flags.append('cropped_or_obstructed')
            elif issue == 'dark' and 'low_light_or_glare' not in risk_flags:
                risk_flags.append('low_light_or_glare')
            elif issue == 'glare' and 'low_light_or_glare' not in risk_flags:
                risk_flags.append('low_light_or_glare')
            elif issue == 'obstructed' and 'cropped_or_obstructed' not in risk_flags:
                risk_flags.append('cropped_or_obstructed')
            elif issue == 'wrong_angle' and 'wrong_angle' not in risk_flags:
                risk_flags.append('wrong_angle')

    if vision_result.get('manipulation_suspected', False):
        if 'possible_manipulation' not in risk_flags:
            risk_flags.append('possible_manipulation')

    if _check_trust_manipulation(preprocessed.get('user_claim', '')):
        if 'text_instruction_present' not in risk_flags:
            risk_flags.append('text_instruction_present')

    if history:
        rejected = int(history.get('rejected_claim', 0))
        recent = int(history.get('last_90_days_claim_count', 0))
        total = int(history.get('past_claim_count', 0))
        history_flags = str(history.get('history_flags', ''))

        if rejected >= 3 or 'user_history_risk' in history_flags:
            if 'user_history_risk' not in risk_flags:
                risk_flags.append('user_history_risk')

        if recent > 5:
            if 'user_history_risk' not in risk_flags:
                risk_flags.append('user_history_risk')

        if 'manual_review_required' in history_flags:
            if 'manual_review_required' not in risk_flags:
                risk_flags.append('manual_review_required')

    risk_flags = list(dict.fromkeys(risk_flags))
    risk_flags = [f for f in risk_flags if f in ALLOWED_RISK_FLAGS]
    return ';'.join(risk_flags) if risk_flags else 'none'


def apply_claim_decision(preprocessed, vision_result, evidence_rule):
    claim_object = preprocessed['claim_object']
    minimum_evidence = evidence_rule.get('minimum_image_evidence', '')
    history = preprocessed['history']

    if not preprocessed['valid_image'] or not preprocessed['image_paths']:
        return {
            'user_id': preprocessed['user_id'],
            'image_paths': ';'.join(preprocessed['image_ids']),
            'user_claim': preprocessed['user_claim'],
            'claim_object': claim_object,
            'evidence_standard_met': False,
            'evidence_standard_met_reason': 'No valid images available for analysis',
            'risk_flags': 'none',
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'No images could be processed to verify the claim.',
            'supporting_image_ids': 'none',
            'valid_image': preprocessed['valid_image'],
            'severity': 'unknown'
        }

    if not vision_result:
        risk_flags_str = _build_risk_flags(
            {'risk_flags': 'none', 'image_quality_issues': 'none', 'manipulation_suspected': False},
            preprocessed, history
        )
        return {
            'user_id': preprocessed['user_id'],
            'image_paths': ';'.join(preprocessed['image_ids']),
            'user_claim': preprocessed['user_claim'],
            'claim_object': claim_object,
            'evidence_standard_met': False,
            'evidence_standard_met_reason': minimum_evidence[:200] if minimum_evidence else 'Insufficient evidence',
            'risk_flags': risk_flags_str,
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'Vision analysis returned no results.',
            'supporting_image_ids': 'none',
            'valid_image': preprocessed['valid_image'],
            'severity': 'unknown'
        }

    issue_type = vision_result.get('issue_type', 'unknown')
    object_part = vision_result.get('object_part', 'unknown')
    confidence = vision_result.get('confidence', 0.0)
    severity = vision_result.get('severity', 'unknown')
    visual_desc = vision_result.get('visual_description', '')
    supporting_ids_raw = vision_result.get('supporting_image_ids', 'none')
    evidence_met = vision_result.get('evidence_standard_met', False)
    image_quality = vision_result.get('image_quality', 'fair')
    manipulation_suspected = vision_result.get('manipulation_suspected', False)

    risk_flags_str = _build_risk_flags(vision_result, preprocessed, history)

    valid_parts = ALLOWED_OBJECT_PARTS.get(claim_object, ['unknown'])
    if object_part not in valid_parts:
        object_part = 'unknown'

    trust_manipulation = _check_trust_manipulation(preprocessed.get('user_claim', ''))

    claim_status = 'supported'
    justification = ''
    evidence_reason = ''

    if not evidence_met:
        claim_status = 'not_enough_information'
        justification = f'Requires: {minimum_evidence[:200]}'
        evidence_reason = justification
    elif image_quality == 'poor' and confidence < 0.6:
        claim_status = 'not_enough_information'
        justification = 'Image quality is poor, cannot reliably assess the claim.'
        evidence_reason = minimum_evidence[:200] if minimum_evidence else 'Poor image quality'
    elif manipulation_suspected or trust_manipulation:
        claim_status = 'not_enough_information'
        justification = 'Possible manipulation or instruction interference detected. Manual review required.'
        evidence_reason = 'Safeguard triggered: review required'
    elif confidence < 0.4:
        issue_type = 'unknown'
        object_part = 'unknown'
        severity = 'unknown'
        claim_status = 'not_enough_information'
        justification = 'Low confidence in visual analysis results.'
    elif issue_type == 'none':
        claim_status = 'contradicted'
        justification = 'Part is clearly visible and undamaged, contradicting the claim.'
    elif issue_type == 'unknown':
        claim_status = 'not_enough_information'
        justification = 'Cannot determine the issue type from submitted images.'
    elif object_part == 'unknown':
        claim_status = 'not_enough_information'
        justification = 'Cannot identify the relevant object part in the images.'
    else:
        if confidence >= 0.7:
            claim_status = 'supported'
            justification = f'Visual evidence supports the claimed damage. {visual_desc}'
        elif confidence >= 0.5:
            claim_status = 'supported'
            justification = f'Visual evidence partially supports the claimed damage. {visual_desc}'
        else:
            claim_status = 'not_enough_information'
            justification = 'Visual evidence is insufficient to confidently support the claim.'

    if isinstance(supporting_ids_raw, list):
        supporting_ids_str = ';'.join(supporting_ids_raw) if supporting_ids_raw else 'none'
    else:
        supporting_ids_str = str(supporting_ids_raw) if supporting_ids_raw else 'none'
    if supporting_ids_str in ('', 'none', '[]'):
        supporting_ids_str = 'none'

    return {
        'user_id': preprocessed['user_id'],
        'image_paths': ';'.join(preprocessed['image_ids']),
        'user_claim': preprocessed['user_claim'],
        'claim_object': claim_object,
        'evidence_standard_met': evidence_met and not (manipulation_suspected or trust_manipulation),
        'evidence_standard_met_reason': evidence_reason,
        'risk_flags': risk_flags_str,
        'issue_type': issue_type,
        'object_part': object_part,
        'claim_status': claim_status,
        'claim_status_justification': justification,
        'supporting_image_ids': supporting_ids_str,
        'valid_image': preprocessed['valid_image'],
        'severity': severity
    }
