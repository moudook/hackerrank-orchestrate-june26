import logging
from config import ALLOWED_RISK_FLAGS

logger = logging.getLogger(__name__)


def apply_claim_decision(preprocessed, vision_result, evidence_rule):
    claim_object = preprocessed['claim_object']
    minimum_evidence = evidence_rule.get('minimum_image_evidence', '')
    history = preprocessed['history']

    if not vision_result or not preprocessed['valid_image']:
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

    issue_type = vision_result.get('issue_type', 'unknown')
    object_part = vision_result.get('object_part', 'unknown')
    confidence = vision_result.get('confidence', 0.0)
    severity = vision_result.get('severity', 'unknown')
    visual_desc = vision_result.get('visual_description', '')
    supporting_ids_raw = vision_result.get('supporting_image_ids', 'none')
    evidence_met = vision_result.get('evidence_standard_met', False)

    risk_flags = []
    if vision_result.get('risk_flags'):
        flags = vision_result['risk_flags']
        if isinstance(flags, str) and flags != 'none':
            risk_flags = [f.strip() for f in flags.split(';') if f.strip()]
        elif isinstance(flags, list):
            risk_flags = flags

    if history:
        rejected = int(history.get('rejected_claim', 0))
        recent = int(history.get('last_90_days_claim_count', 0))
        if rejected >= 3:
            risk_flags.append('user_history_risk')
        if recent > 5:
            risk_flags.append('user_history_risk')

    claim_status = 'supported'
    justification = ''

    if not evidence_met:
        claim_status = 'not_enough_information'
        justification = f"Requires: {minimum_evidence[:120]}"
    else:
        if confidence < 0.5:
            issue_type = 'unknown'
            object_part = 'unknown'
            severity = 'unknown'
            claim_status = 'not_enough_information'
            justification = 'Low confidence in visual analysis results.'
        elif issue_type == 'none':
            if 'not visible' in visual_desc.lower():
                issue_type = 'unknown'
                claim_status = 'not_enough_information'
                justification = 'Part not clearly visible in submitted images.'
            else:
                claim_status = 'contradicted'
                justification = 'Part is clearly visible and undamaged, contradicting the claim.'
        else:
            if confidence >= 0.7:
                claim_status = 'supported'
                justification = 'Visual evidence supports the claimed damage.'
            else:
                claim_status = 'supported'
                justification = 'Visual evidence partially supports the claimed damage.'

    risk_flags = list(dict.fromkeys(risk_flags))
    risk_flags = [f for f in risk_flags if f in ALLOWED_RISK_FLAGS]
    risk_flags_str = ';'.join(risk_flags) if risk_flags else 'none'

    supporting_ids_str = supporting_ids_raw
    if isinstance(supporting_ids_raw, list):
        supporting_ids_str = ';'.join(supporting_ids_raw)
    if not supporting_ids_str or supporting_ids_str == 'none':
        supporting_ids_str = 'none'

    return {
        'user_id': preprocessed['user_id'],
        'image_paths': ';'.join(preprocessed['image_ids']),
        'user_claim': preprocessed['user_claim'],
        'claim_object': claim_object,
        'evidence_standard_met': evidence_met,
        'evidence_standard_met_reason': justification if claim_status == 'not_enough_information' else '',
        'risk_flags': risk_flags_str,
        'issue_type': issue_type,
        'object_part': object_part,
        'claim_status': claim_status,
        'claim_status_justification': justification,
        'supporting_image_ids': supporting_ids_str,
        'valid_image': preprocessed['valid_image'],
        'severity': severity
    }
