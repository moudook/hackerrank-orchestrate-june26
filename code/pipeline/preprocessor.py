import os
import logging
from urllib.parse import unquote
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

MIN_IMAGE_DIMENSION = 100
MAX_IMAGE_SIZE_MB = 20


def _normalize_path(raw_path):
    raw_path = raw_path.strip()
    raw_path = unquote(raw_path)
    normalized = raw_path.replace('\\', '/')
    normalized = os.path.normpath(normalized).replace('\\', '/')
    return normalized


def _resolve_image_path(normalized_path):
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset')
    path_attempts = [
        os.path.join(base_dir, normalized_path),
        os.path.join(base_dir, normalized_path.lstrip('dataset/')),
    ]
    for attempt in path_attempts:
        if os.path.exists(attempt):
            return attempt
    if normalized_path.startswith('dataset/'):
        alt = normalized_path[8:]
        for sub in ['sample', 'test']:
            candidate = os.path.join(base_dir, sub, alt)
            if os.path.exists(candidate):
                return candidate
    return None


def _is_valid_image(filepath):
    try:
        if os.path.getsize(filepath) == 0:
            logger.warning(f"Zero-byte image: {filepath}")
            return False

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            logger.warning(f"Image too large ({size_mb:.1f}MB): {filepath}")
            return False

        with Image.open(filepath) as img:
            img.verify()

        with Image.open(filepath) as img:
            img.load()
            w, h = img.size
            if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                logger.warning(f"Image too small ({w}x{h}): {filepath}")
                return False

        return True
    except Exception:
        logger.warning(f"Corrupted image: {filepath}")
        return False


def _extract_image_id(path):
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    return name.lower()


def preprocess_claim(row, user_history_df):
    raw_paths = row.get('image_paths')
    user_claim = row.get('user_claim', '')
    if pd.isna(user_claim):
        user_claim = ''

    if pd.isna(raw_paths) or str(raw_paths).strip() == '':
        return {
            'error': 'no_images',
            'user_id': row.get('user_id', ''),
            'claim_object': str(row.get('claim_object', '')).strip().lower(),
            'user_claim': user_claim,
            'image_paths': [],
            'image_ids': [],
            'history': None,
            'valid_image': False
        }

    raw = str(raw_paths).strip()
    parts = raw.split(';')
    parts = [p.strip() for p in parts if p.strip()]

    parts = list(dict.fromkeys(parts))

    if len(parts) > 4:
        logger.warning(f"Truncating {len(parts)} images to 4 for user {row.get('user_id')}")
        parts = parts[:4]

    image_ids = []
    valid_paths = []
    for p in parts:
        normalized = _normalize_path(p)
        full_path = _resolve_image_path(normalized)

        if full_path is None or not os.path.exists(full_path):
            logger.warning(f"Image not found: {p}")
            continue

        if not _is_valid_image(full_path):
            logger.warning(f"Invalid image (corrupt/empty/tiny): {p}")
            continue

        image_id = _extract_image_id(full_path)
        image_ids.append(image_id)
        valid_paths.append(full_path)

    user_id = row.get('user_id', '')
    history = user_history_df[user_history_df['user_id'] == user_id]
    history_dict = history.iloc[0].to_dict() if not history.empty else None

    return {
        'user_id': user_id,
        'claim_object': str(row.get('claim_object', '')).strip().lower(),
        'user_claim': user_claim,
        'image_paths': valid_paths,
        'image_ids': image_ids,
        'history': history_dict,
        'valid_image': len(valid_paths) > 0
    }
