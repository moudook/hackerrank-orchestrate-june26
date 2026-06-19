import logging
import pandas as pd

logger = logging.getLogger(__name__)

ISSUE_KEYWORD_MAP = {
    'dent': ['dent', 'scratch', 'scrape', 'scuff'],
    'crack': ['crack', 'shatter', 'broken'],
    'water': ['water', 'wet', 'liquid', 'spill', 'moisture', 'damp'],
    'torn': ['torn', 'rip', 'tear', 'ripped', 'tore'],
    'crush': ['crush', 'squash', 'squeeze', 'crumple'],
    'stain': ['stain', 'mark', 'spot', 'oil'],
}

FALLBACK_APPLIES_TO = 'general claim review'


def get_relevant_rule(claim_object, user_claim, evidence_df):
    text = user_claim.lower() if user_claim else ''

    matched_keyword = None
    for keyword, keywords_list in ISSUE_KEYWORD_MAP.items():
        if any(k in text for k in keywords_list):
            matched_keyword = keyword
            break

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
