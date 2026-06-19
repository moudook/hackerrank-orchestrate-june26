import logging
import os
import sys
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_CLAIMS_COLUMNS = ['user_id', 'image_paths', 'user_claim', 'claim_object']
EXPECTED_HISTORY_COLUMNS = ['user_id', 'past_claim_count', 'accept_claim', 'manual_review_claim',
                            'rejected_claim', 'last_90_days_claim_count', 'history_flags', 'history_summary']
EXPECTED_EVIDENCE_COLUMNS = ['requirement_id', 'claim_object', 'applies_to', 'minimum_image_evidence']

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(MODULE_DIR)), 'dataset')

VALID_OBJECTS = {'car', 'laptop', 'package'}


def _validate_columns(df: pd.DataFrame, expected: list[str], name: str) -> bool:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return True


def _try_read_csv(path: str, encodings: Tuple[str, ...] = ('utf-8-sig', 'utf-8', 'latin-1')) -> pd.DataFrame:
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            if not df.empty:
                logger.debug(f"Read {path} with encoding {enc}")
                return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            logger.debug(f"Failed {enc} for {path}: {e}")
            continue
    raise ValueError(f"Could not read {path} with any encoding in {encodings}")


def load_claims(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, 'claims.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"claims.csv not found at {path}")
    df = _try_read_csv(path)
    _validate_columns(df, EXPECTED_CLAIMS_COLUMNS, 'claims.csv')

    for col in EXPECTED_CLAIMS_COLUMNS:
        df[col] = df[col].astype(str)

    invalid_objects = df[~df['claim_object'].str.strip().str.lower().isin(VALID_OBJECTS)]
    if not invalid_objects.empty:
        logger.warning(f"Invalid claim_objects found: {invalid_objects[['user_id', 'claim_object']].to_dict('records')}")

    if df.empty:
        print("WARNING: claims.csv is empty. Exiting.")
        sys.exit(0)
    return df


def load_sample_claims(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, 'sample_claims.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"sample_claims.csv not found at {path}")
    df = _try_read_csv(path)
    return df


def load_user_history(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, 'user_history.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"user_history.csv not found at {path}")
    df = _try_read_csv(path)
    _validate_columns(df, EXPECTED_HISTORY_COLUMNS, 'user_history.csv')

    for col in ['past_claim_count', 'accept_claim', 'manual_review_claim', 'rejected_claim', 'last_90_days_claim_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


def load_evidence_requirements(path: Optional[str] = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, 'evidence_requirements.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"evidence_requirements.csv not found at {path}")
    df = _try_read_csv(path)
    _validate_columns(df, EXPECTED_EVIDENCE_COLUMNS, 'evidence_requirements.csv')
    return df


def load_all() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f"Loading data from {DATA_DIR}")
    claims = load_claims()
    user_history = load_user_history()
    evidence = load_evidence_requirements()
    logger.info(f"Loaded: {len(claims)} claims, {len(user_history)} history rows, {len(evidence)} evidence rules")
    return claims, user_history, evidence
