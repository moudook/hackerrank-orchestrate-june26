import logging
import random
from typing import Dict, Optional

import pandas as pd

random.seed(42)

logger = logging.getLogger(__name__)

ISSUE_KEYWORD_MAP = {
    'dent': ['dent', 'scratch', 'scrape', 'scuff', 'ding', 'dented', 'denting'],
    'crack': ['crack', 'shatter', 'broken', 'fracture', 'cracked', 'shattered'],
    'water': ['water', 'wet', 'liquid', 'spill', 'moisture', 'damp', 'leak', 'rain', 'water_damage'],
    'torn': ['torn', 'rip', 'tear', 'ripped', 'tore', 'tearing', 'open'],
    'crush': ['crush', 'squash', 'squeeze', 'crumple', 'crushed', 'squashed', 'dab', 'dab gaya'],
    'stain': ['stain', 'mark', 'spot', 'oil', 'grease', 'stained', 'oily', 'dirty'],
    'missing': ['missing', 'lost', 'gone', 'absent', 'falt', 'gaya', 'missing keys'],
    'glass_shatter': ['glass', 'shatter', 'shattered', 'windshield', 'broken glass'],
}

MULTILANG_KEYWORDS = {
    'hindi': {
        'dent': ['dent', 'dent lag', 'thokar', 'thokar se'],
        'crack': ['crack', 'darar', 'phoot', 'phat'],
        'water': ['pani', 'geela', 'bheega', 'nam'],
        'torn': ['phat', 'phat gaya', 'torn'],
        'crush': ['dab', 'dab gaya', 'daba', 'kuchal', 'crush'],
        'stain': ['daag', 'dhabba', 'oil'],
        'missing': ['falt', 'gaya', 'kho', 'missing'],
        'glass_shatter': ['sheesha', 'khanch', 'toot'],
    },
    'spanish': {
        'dent': ['abolladura', 'abollado', 'golpe'],
        'crack': ['grieta', 'rajadura', 'roto', 'quebrado'],
        'water': ['agua', 'humedo', 'mojado'],
        'torn': ['rasgado', 'roto', 'desgarrado'],
        'crush': ['aplastado', 'aplastar'],
        'stain': ['mancha', 'sucio'],
        'missing': ['falta', 'perdido', 'desaparecido'],
        'glass_shatter': ['vidrio', 'cristal', 'parabrisas'],
    },
    'chinese': {
        'crack': ['lie', 'po', 'sui', 'broken'],
        'water': ['shui', 'shi'],
        'screen': ['ping mu', 'ping'],
        'keyboard': ['jian pan'],
    }
}

FALLBACK_APPLIES_TO = 'general claim review'


def _detect_issue_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    text_lower = text.lower()

    for keyword, keywords_list in ISSUE_KEYWORD_MAP.items():
        if any(k in text_lower for k in keywords_list):
            return keyword

    for lang, lang_map in MULTILANG_KEYWORDS.items():
        for issue, keywords in lang_map.items():
            if any(k in text_lower for k in keywords):
                return issue

    return None


def get_relevant_rule(claim_object: str, user_claim: str, evidence_df: pd.DataFrame) -> Dict:
    text = user_claim.lower() if user_claim else ''

    matched_keyword = _detect_issue_from_text(text)

    object_rows = evidence_df[
        evidence_df['claim_object'].isin([claim_object, 'all'])
    ]

    if matched_keyword:
        matched = object_rows[
            object_rows['applies_to'].str.contains(matched_keyword, case=False, na=False)
        ]
        if not matched.empty:
            return matched.iloc[0].to_dict()

    general = object_rows[
        object_rows['applies_to'] == FALLBACK_APPLIES_TO
    ]
    if not general.empty:
        return general.iloc[0].to_dict()

    if not object_rows.empty:
        return object_rows.iloc[0].to_dict()

    return evidence_df.iloc[0].to_dict()
