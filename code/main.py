import sys
import os
import logging
import concurrent.futures

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

import pandas as pd

from pipeline.loader import load_all
from pipeline.preprocessor import preprocess_claim
from pipeline.evidence_filter import get_relevant_rule
from pipeline.vision_analyzer import safe_run_vision_analysis
from pipeline.postprocessor import apply_claim_decision
from pipeline.validator import validate_output
from utils.token_tracker import TokenTracker
from utils.rate_limiter import TokenBucketRateLimiter
from utils.checkpoint import CheckpointManager
from config import CACHE_ENABLED, CACHE_DIR, validate_config

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(REPO_ROOT, 'output.csv')
CHECKPOINT_PATH = os.path.join(REPO_ROOT, '.checkpoint.json')


def build_output_row(row, preprocessed, decision=None):
    if decision:
        return decision
    return {
        'user_id': row.get('user_id', ''),
        'image_paths': str(row.get('image_paths', '')),
        'user_claim': str(row.get('user_claim', '')),
        'claim_object': str(row.get('claim_object', '')),
        'evidence_standard_met': False,
        'evidence_standard_met_reason': 'Not processed',
        'risk_flags': 'manual_review_required',
        'issue_type': 'unknown',
        'object_part': 'unknown',
        'claim_status': 'not_enough_information',
        'claim_status_justification': 'Claim not processed.',
        'supporting_image_ids': 'none',
        'valid_image': False,
        'severity': 'unknown',
    }


def process_single_claim(idx, row, user_history, evidence, token_tracker, rate_limiter):
    user_id = row['user_id']
    logger.info(f"[{idx+1}] Processing user={user_id}, object={row['claim_object']}")

    preprocessed = preprocess_claim(row, user_history)
    evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)

    estimated_tokens = 1000 + len(preprocessed['image_paths']) * 258
    rate_limiter.acquire(estimated_tokens)

    vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
    decision = apply_claim_decision(preprocessed, vision_result, evidence_rule)
    validated = validate_output(decision)

    return validated


def main():
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)

    claims, user_history, evidence = load_all()
    logger.info(f"Loaded {len(claims)} claims, {len(user_history)} history rows, {len(evidence)} evidence rows")

    token_tracker = TokenTracker()
    rate_limiter = TokenBucketRateLimiter()
    checkpoint = CheckpointManager(CHECKPOINT_PATH)

    if CACHE_ENABLED and CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)

    all_results = []
    remaining_indices = []

    for idx, row in claims.iterrows():
        uid = row['user_id']
        if checkpoint.is_processed(uid):
            chk = checkpoint._data[uid]
            all_results.append((idx, chk))
        else:
            remaining_indices.append((idx, row))

    logger.info(f"Checkpoint: {len(all_results)} cached, {len(remaining_indices)} remaining")

    if remaining_indices:
        def worker(item):
            idx, row = item
            result = process_single_claim(idx + 1, row, user_history, evidence, token_tracker, rate_limiter)
            checkpoint.mark_processed(row['user_id'], result)
            return idx, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            new_results = list(executor.map(worker, remaining_indices))

        all_results.extend(new_results)

        if len(new_results) % 5 == 0 or len(new_results) == len(remaining_indices):
            rl_stats = rate_limiter.stats()
            logger.info(f"Progress: {checkpoint.get_completed_count()}/{len(claims)} done, "
                        f"cost: ${token_tracker.get_cost():.6f}, "
                        f"RPM: {rl_stats['current_rpm']}, TPM: {rl_stats['current_tpm']}")

    checkpoint.save()
    all_results.sort(key=lambda x: x[0])
    final_results = [r for _, r in all_results]

    output_df = pd.DataFrame(final_results, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_PATH, index=False, quoting=1)
    logger.info(f"Output written to {OUTPUT_PATH} ({len(final_results)} rows)")

    summary = token_tracker.summary()
    rl_stats = rate_limiter.stats()
    print("\n=== Pipeline Summary ===")
    print(f"  Claims processed: {len(final_results)}")
    print(f"  New API calls: {summary['total_calls']}")
    print(f"  From checkpoint: {checkpoint.get_completed_count() - len(remaining_indices)}")
    print(f"  Input tokens: {summary['input_tokens']}")
    print(f"  Output tokens: {summary['output_tokens']}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.6f}")
    print(f"  Elapsed time: {summary['elapsed_seconds']:.1f}s")
    print(f"  Peak RPM: {rl_stats['current_rpm']}")
    print(f"  Peak TPM: {rl_stats['current_tpm']}")


if __name__ == '__main__':
    main()
