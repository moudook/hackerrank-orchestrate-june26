import logging
import pandas as pd

logger = logging.getLogger(__name__)

ISSUE_KEYWORD_MAP = {
    'dent or scratch': ['dent', 'scratch', 'scrape', 'scuff'],
    'crack': ['crack', 'shatter', 'broken glass'],
    'broken or missing part': ['broken', 'missing', 'fell off', 'came off', 'snapped'],
    'water_damage': ['water', 'wet', 'liquid', 'spill', 'moisture', 'damp'],
    'torn_packaging': ['torn', 'rip', 'tear', 'ripped', 'tore'],
    'crushed_packaging': ['crush', 'dent', 'squash', 'squeeze', 'crumple'],
    'stain': ['stain', 'mark', 'spot', 'discoloration', 'oil'],
    'glass_shatter': ['shatter', 'glass broken', 'glass crack'],
}


def get_relevant_rule(claim_object, user_claim, evidence_df):
    text = user_claim.lower() if user_claim else ''

    matched_family = 'all'
    for family, keywords in ISSUE_KEYWORD_MAP.items():
        if any(k in text for k in keywords):
            matched_family = family
            break

    filtered = evidence_df[
        (evidence_df['claim_object'].isin([claim_object, 'all'])) &
        (evidence_df['applies_to'].isin([matched_family, 'all']))
    ]

    if filtered.empty:
        logger.warning(f"No matching rule for object={claim_object}, family={matched_family}; using 'all'/'all' fallback")
        filtered = evidence_df[
            (evidence_df['claim_object'] == 'all') &
            (evidence_df['applies_to'] == 'all')
        ]

    return filtered.iloc[0].to_dict()
