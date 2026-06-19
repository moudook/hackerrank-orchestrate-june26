import argparse
import concurrent.futures
import logging
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

import pandas as pd  # noqa: E402
from config import CACHE_DIR, CACHE_ENABLED, validate_config  # noqa: E402
from pipeline.evidence_filter import get_relevant_rule  # noqa: E402
from pipeline.loader import load_all  # noqa: E402
from pipeline.postprocessor import apply_claim_decision  # noqa: E402
from pipeline.preprocessor import preprocess_claim  # noqa: E402
from pipeline.safety_gate import evaluate_safety_gate  # noqa: E402
from pipeline.validator import validate_output  # noqa: E402
from pipeline.vision_analyzer import safe_run_vision_analysis  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402
from utils.rate_limiter import TokenBucketRateLimiter  # noqa: E402
from utils.token_tracker import TokenTracker  # noqa: E402

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(REPO_ROOT, 'output.csv')
CHECKPOINT_PATH = os.path.join(REPO_ROOT, '.checkpoint.json')
DRY_RUN = False


def build_output_row(row: pd.Series, preprocessed: Dict[str, Any], decision: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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


def process_single_claim(idx: int, row: pd.Series, user_history: pd.DataFrame, evidence: pd.DataFrame, token_tracker: TokenTracker, rate_limiter: TokenBucketRateLimiter) -> Dict[str, Any]:
    user_id = row['user_id']
    logger.info(f"[{idx+1}] Processing user={user_id}, object={row['claim_object']}")

    preprocessed = preprocess_claim(row, user_history)
    evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)

    gate_result = evaluate_safety_gate(preprocessed)
    if gate_result and gate_result.get('blocked'):
        decision = apply_claim_decision(
            preprocessed, None, evidence_rule,
            override_risk_flags=gate_result['risk_flags'],
            override_justification=gate_result['reason']
        )
        validated = validate_output(decision)
        return validated

    estimated_tokens = 1000 + len(preprocessed['image_paths']) * 258
    rate_limiter.acquire(estimated_tokens)

    vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
    if gate_result:
        if vision_result:
            existing = vision_result.get('risk_flags', 'none')
            combined = f"{existing};{gate_result['risk_flags']}" if existing != 'none' else gate_result['risk_flags']
            vision_result['risk_flags'] = combined
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
        if DRY_RUN:
            logger.info("Dry-run: processing claims through preprocessor + safety gate only")
            for idx, row in remaining_indices:
                pre = preprocess_claim(row, user_history)
                gate = evaluate_safety_gate(pre)
                flagged = 'BLOCKED' if gate and gate.get('blocked') else 'FLAGGED' if gate else 'PASS'
                logger.info(f"  [{idx+1}] {row['user_id']}: {flagged}")
            print(f"\nDry-run complete: {len(remaining_indices)} claims checked, "
                  f"{sum(1 for _, r in remaining_indices if evaluate_safety_gate(preprocess_claim(r, user_history)) and evaluate_safety_gate(preprocess_claim(r, user_history)).get('blocked'))} blocked by safety gate")
            return

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
    print(f"  From checkpoint: {checkpoint.get_completed_count()}")
    print(f"  Input tokens: {summary['input_tokens']}")
    print(f"  Output tokens: {summary['output_tokens']}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.6f}")
    print(f"  Elapsed time: {summary['elapsed_seconds']:.1f}s")
    print(f"  Peak RPM: {rl_stats['current_rpm']}")
    print(f"  Peak TPM: {rl_stats['current_tpm']}")


def cli():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Evidence Review Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                       # Run production pipeline on claims.csv
  python main.py --reset-checkpoint    # Clear checkpoint and re-run all claims
  python main.py --verbose             # Enable debug logging
  python main.py --model gemini-2.5-flash  # Override model from env/config
        """
    )
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Clear checkpoint and re-process all claims')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug-level logging')
    parser.add_argument('--model', type=str, default=None,
                        help='Override Gemini model name')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output CSV path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load and validate data, run preprocessor + safety gate, skip VLM calls')
    args = parser.parse_args()

    if args.verbose:
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    if args.model:
        import config as cfg
        cfg.MODEL_NAME = args.model
        logger.info(f"Model overridden to: {args.model}")

    if args.output:
        global OUTPUT_PATH
        OUTPUT_PATH = args.output

    if args.dry_run:
        global DRY_RUN
        DRY_RUN = True
        logger.info("Dry-run mode: data will be loaded and preprocessed, VLM calls skipped")

    if args.reset_checkpoint:
        from utils.checkpoint import CheckpointManager
        cm = CheckpointManager(CHECKPOINT_PATH)
        cm.reset()
        logger.info("Checkpoint reset. All claims will be re-processed.")

    main()


if __name__ == '__main__':
    cli()
