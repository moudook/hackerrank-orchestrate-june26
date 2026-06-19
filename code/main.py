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
from utils.rate_limiter import RateLimiter
from config import CACHE_ENABLED, CACHE_DIR

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(REPO_ROOT, 'output.csv')


def main():
    claims, user_history, evidence = load_all()
    logger.info(f"Loaded {len(claims)} claims, {len(user_history)} history rows, {len(evidence)} evidence rows")

    token_tracker = TokenTracker()
    rate_limiter = RateLimiter()

    if CACHE_ENABLED and not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def process_single_claim(item):
        idx, row = item
        logger.info(f"[{idx+1}/{len(claims)}] Processing user={row['user_id']}, object={row['claim_object']}")

        preprocessed = preprocess_claim(row, user_history)
        evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)
        vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
        decision = apply_claim_decision(preprocessed, vision_result, evidence_rule)
        validated = validate_output(decision)

        if (idx + 1) % 5 == 0:
            logger.info(f"Progress: {idx+1}/{len(claims)} processed, cost so far: ${token_tracker.get_cost():.4f}")

        return validated

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_single_claim, claims.iterrows()))

    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_PATH, index=False, quoting=1)
    logger.info(f"Output written to {OUTPUT_PATH} ({len(results)} rows)")

    summary = token_tracker.summary()
    print("\n=== Pipeline Summary ===")
    print(f"  Claims processed: {len(results)}")
    print(f"  Total model calls: {summary['total_calls']}")
    print(f"  Input tokens: {summary['input_tokens']}")
    print(f"  Output tokens: {summary['output_tokens']}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.6f}")
    print(f"  Elapsed time: {summary['elapsed_seconds']:.1f}s")


if __name__ == '__main__':
    main()
