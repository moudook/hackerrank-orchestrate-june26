import logging
import random
from typing import Dict, List, Optional

random.seed(42)

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    'system prompt', 'you are an ai', 'you are a language model',
    'forget all previous', 'new instructions', 'override',
    'you must ignore', 'disregard', 'pretend',
    'ignore all previous', 'ignore instructions',
    'you are a security', 'you are now',
    'act as', 'role play', 'roleplay',
    'jailbreak', 'bypass', 'hack',
    'do anything now', 'dan mode',
    'developer mode', 'debug mode',
]

HIGH_RISK_KEYWORDS = [
    'refund', 'escalate', 'complaint', 'legal', 'lawsuit',
    'attorney', 'lawyer', 'compensation', 'damages',
    'sue', 'court', 'regulator', 'fraud accusation',
]

TRUST_MANIPULATION_PATTERNS = [
    'ignore all previous instructions', 'ignore instructions',
    'approve the claim', 'approve this claim', 'mark this row',
    'skip manual review', 'follow it and approve',
    'accept this quickly', 'reopen tickets', 'rejected again',
    'escalate publicly',
    'pay the full', 'pay out', 'total loss',
    'set severity', 'mark as approved', 'change verdict',
    'override system', 'bypass verification',
    'you should approve', 'you must approve',
    'this is pre-approved', 'pre-approved claim',
    'authorized payout', 'payment authorized',
]

COERCION_KEYWORDS = [
    'if this gets rejected', 'i will keep reopening',
    'i will escalate', 'i will sue', 'i will report',
    'complaint to', 'report to', 'lawsuit',
    'unacceptable', 'terrible service',
    'i want to speak to', 'manager', 'supervisor',
]

RUBRIC_KEYWORDS = [
    'scoring rubric', 'evaluation criteria', 'assessment framework',
    'quality control', 'pass rate', 'progress bar',
    'score 5', 'score 4', 'score 3', 'score 2', 'score 1',
    'evaluation dashboard', 'review dashboard',
    'model quality', 'claim quality',
]


def _check_text_risk(user_claim: str) -> List[str]:
    if not user_claim:
        return []
    text = user_claim.lower()
    flags = []

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in text:
            logger.info(f"Safety gate: prompt injection pattern '{pattern}'")
            flags.append('possible_manipulation')
            break

    for pattern in TRUST_MANIPULATION_PATTERNS:
        if pattern in text:
            logger.info(f"Safety gate: trust manipulation pattern '{pattern}'")
            flags.append('text_instruction_present')
            break

    for kw in HIGH_RISK_KEYWORDS:
        if kw in text:
            logger.info(f"Safety gate: high-risk keyword '{kw}'")
            flags.append('manual_review_required')
            break

    coercion_count = sum(1 for kw in COERCION_KEYWORDS if kw in text)
    if coercion_count >= 2:
        if 'manual_review_required' not in flags:
            flags.append('manual_review_required')
        if 'possible_manipulation' not in flags:
            flags.append('possible_manipulation')
        logger.info(f"Safety gate: coercion detected ({coercion_count} keywords)")

    rubric_count = sum(1 for kw in RUBRIC_KEYWORDS if kw in text)
    if rubric_count >= 2:
        if 'possible_manipulation' not in flags:
            flags.append('possible_manipulation')
        logger.info(f"Safety gate: rubric steering detected ({rubric_count} keywords)")

    return flags


def _check_history_risk(history: Optional[Dict]) -> List[str]:
    if not history:
        return []
    flags = []
    rejected = int(history.get('rejected_claim', 0))
    recent = int(history.get('last_90_days_claim_count', 0))
    history_flags = str(history.get('history_flags', ''))

    if rejected >= 3 or recent > 5 or 'user_history_risk' in history_flags:
        flags.append('user_history_risk')
    if 'manual_review_required' in history_flags:
        flags.append('manual_review_required')
    return flags


def _check_image_forensics_flags(forensics_result: Optional[Dict]) -> List[str]:
    if not forensics_result:
        return []

    flags = []
    anomalies = forensics_result.get('anomalies', [])

    if 'text_injection_detected' in anomalies:
        flags.append('possible_manipulation')
        flags.append('text_instruction_present')

    if 'instruction_layout' in anomalies:
        flags.append('possible_manipulation')

    if 'ui_elements_in_image' in anomalies:
        flags.append('possible_manipulation')

    if 'screenshot_detected' in anomalies:
        flags.append('non_original_image')

    if any(a in anomalies for a in ['editing_software:photoshop', 'editing_software:gimp']):
        flags.append('possible_manipulation')

    if 'metadata_stripped' in anomalies:
        if 'manual_review_required' not in flags:
            flags.append('manual_review_required')

    return list(dict.fromkeys(flags))


def evaluate_safety_gate(preprocessed: Dict, forensics_result: Optional[Dict] = None) -> Optional[Dict]:
    user_id = preprocessed.get('user_id', '')
    user_claim = preprocessed.get('user_claim', '')
    history = preprocessed.get('history')

    risk_flags = []
    risk_flags.extend(_check_text_risk(user_claim))
    risk_flags.extend(_check_history_risk(history))
    risk_flags.extend(_check_image_forensics_flags(forensics_result))

    if not risk_flags:
        return None

    risk_flags = list(dict.fromkeys(risk_flags))
    blocked = 'text_instruction_present' in risk_flags
    needs_review = 'manual_review_required' in risk_flags

    if blocked or needs_review or risk_flags:
        msg = 'blocked' if blocked else ('flagged' if needs_review else 'flagged')
        logger.info(f"Safety gate {msg} user={user_id}: {risk_flags}")
        return {
            'blocked': blocked,
            'risk_flags': ';'.join(risk_flags),
            'reason': '; '.join(risk_flags),
        }

    return None
