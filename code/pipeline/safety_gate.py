import logging
import random
from typing import Dict, List, Optional

random.seed(42)

logger = logging.getLogger(__name__)

PROMPT_INJECTION_PATTERNS = [
    'system prompt', 'you are an ai', 'you are a language model',
    'forget all previous', 'new instructions', 'override',
    'you must ignore', 'disregard', 'pretend',
]

HIGH_RISK_KEYWORDS = [
    'refund', 'escalate', 'complaint', 'legal', 'lawsuit',
    'attorney', 'lawyer', 'compensation', 'damages',
]

TRUST_MANIPULATION_PATTERNS = [
    'ignore all previous instructions', 'ignore instructions',
    'approve the claim', 'approve this claim', 'mark this row',
    'skip manual review', 'follow it and approve',
    'accept this quickly', 'reopen tickets', 'rejected again',
    'escalate publicly',
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


def evaluate_safety_gate(preprocessed: Dict) -> Optional[Dict]:
    user_id = preprocessed.get('user_id', '')
    user_claim = preprocessed.get('user_claim', '')
    history = preprocessed.get('history')

    risk_flags = []
    risk_flags.extend(_check_text_risk(user_claim))
    risk_flags.extend(_check_history_risk(history))

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
