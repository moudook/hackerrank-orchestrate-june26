import os
import sys
import pandas as pd

EXPECTED_CLAIMS_COLUMNS = ['user_id', 'image_paths', 'user_claim', 'claim_object']
EXPECTED_HISTORY_COLUMNS = ['user_id', 'past_claim_count', 'accept_claim', 'manual_review_claim',
                            'rejected_claim', 'last_90_days_claim_count', 'history_flags', 'history_summary']
EXPECTED_EVIDENCE_COLUMNS = ['requirement_id', 'claim_object', 'applies_to', 'minimum_image_evidence']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset')


def _validate_columns(df, expected, name):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_claims(path=None):
    path = path or os.path.join(DATA_DIR, 'claims.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"claims.csv not found at {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    _validate_columns(df, EXPECTED_CLAIMS_COLUMNS, 'claims.csv')
    if df.empty:
        print("WARNING: claims.csv is empty. Exiting.")
        sys.exit(0)
    return df


def load_sample_claims(path=None):
    path = path or os.path.join(DATA_DIR, 'sample_claims.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"sample_claims.csv not found at {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df


def load_user_history(path=None):
    path = path or os.path.join(DATA_DIR, 'user_history.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"user_history.csv not found at {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    _validate_columns(df, EXPECTED_HISTORY_COLUMNS, 'user_history.csv')
    return df


def load_evidence_requirements(path=None):
    path = path or os.path.join(DATA_DIR, 'evidence_requirements.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"evidence_requirements.csv not found at {path}")
    df = pd.read_csv(path, encoding='utf-8-sig')
    _validate_columns(df, EXPECTED_EVIDENCE_COLUMNS, 'evidence_requirements.csv')
    return df


def load_all():
    claims = load_claims()
    user_history = load_user_history()
    evidence = load_evidence_requirements()
    return claims, user_history, evidence
