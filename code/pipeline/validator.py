import os
import logging
from config import ALLOWED_ISSUE_TYPES, ALLOWED_CLAIM_STATUS, ALLOWED_OBJECT_PARTS, ALLOWED_RISK_FLAGS, ALLOWED_SEVERITY

logger = logging.getLogger(__name__)


def validate_output(output):
    validated = dict(output)
    claim_object = validated.get('claim_object', 'unknown')

    if validated.get('issue_type') not in ALLOWED_ISSUE_TYPES:
        validated['issue_type'] = 'unknown'

    valid_parts = ALLOWED_OBJECT_PARTS.get(claim_object, ['unknown'])
    if validated.get('object_part') not in valid_parts:
        validated['object_part'] = 'unknown'

    if validated.get('claim_status') not in ALLOWED_CLAIM_STATUS:
        validated['claim_status'] = 'not_enough_information'

    if validated.get('severity') not in ALLOWED_SEVERITY:
        validated['severity'] = 'unknown'

    raw_flags = validated.get('risk_flags', 'none')
    if raw_flags == 'none' or not raw_flags:
        validated['risk_flags'] = 'none'
    else:
        flags = [f.strip() for f in str(raw_flags).split(';') if f.strip()]
        valid_flags = [f for f in flags if f in ALLOWED_RISK_FLAGS]
        validated['risk_flags'] = ';'.join(valid_flags) if valid_flags else 'none'

    raw_ids = validated.get('supporting_image_ids', 'none')
    if raw_ids and raw_ids != 'none':
        ids = [id.strip() for id in str(raw_ids).split(';') if id.strip()]
        cleaned = [os.path.splitext(id)[0] for id in ids]
        validated['supporting_image_ids'] = ';'.join(cleaned) if cleaned else 'none'
    else:
        validated['supporting_image_ids'] = 'none'

    if validated.get('evidence_standard_met') is None:
        validated['evidence_standard_met'] = False

    if validated.get('valid_image') is None:
        validated['valid_image'] = False

    return validated
