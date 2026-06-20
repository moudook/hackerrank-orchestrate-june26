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

import pandas as pd
from config import CACHE_DIR, CACHE_ENABLED, MAX_WORKERS, validate_config
from pipeline.evidence_filter import get_relevant_rule
from pipeline.loader import load_all
from pipeline.postprocessor import apply_claim_decision
from pipeline.preprocessor import preprocess_claim
from pipeline.safety_gate import evaluate_safety_gate
from pipeline.validator import validate_output
from pipeline.vision_analyzer import safe_run_vision_analysis, pre_vlm_forensics_scan
from utils.checkpoint import CheckpointManager
from utils.rate_limiter import AdaptiveRateLimiter
from utils.token_tracker import TokenTracker

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


def process_single_claim(idx: int, row: pd.Series, user_history: pd.DataFrame, evidence: pd.DataFrame, token_tracker: TokenTracker, rate_limiter: AdaptiveRateLimiter) -> Dict[str, Any]:
    user_id = row['user_id']
    logger.info(f"[{idx+1}] Processing user={user_id}, object={row['claim_object']}")

    preprocessed = preprocess_claim(row, user_history)
    evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)

    forensics_result = pre_vlm_forensics_scan(preprocessed.get('image_paths', []))

    gate_result = evaluate_safety_gate(preprocessed, forensics_result=forensics_result)

    rate_limiter.acquire()
    try:
        vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
    finally:
        rate_limiter.release()

    if gate_result:
        if vision_result:
            existing = vision_result.get('risk_flags', 'none')
            combined = f"{existing};{gate_result['risk_flags']}" if existing != 'none' else gate_result['risk_flags']
            vision_result['risk_flags'] = combined
    decision = apply_claim_decision(preprocessed, vision_result, evidence_rule)
    return validate_output(decision)


def main(eval_mode=False):
    if not validate_config():
        logger.error("Configuration validation failed. Exiting.")
        sys.exit(1)

    if eval_mode:
        from pipeline.loader import load_evidence_requirements, load_sample_claims
        claims = load_sample_claims()
        evidence = load_evidence_requirements()
        user_history = pd.DataFrame({'user_id': ['none'], 'rejected_claim': [0], 'last_90_days_claim_count': [0], 'history_flags': ['']})
        logger.info(f"Evaluation mode: loaded {len(claims)} sample claims")
    else:
        claims, user_history, evidence = load_all()
    logger.info(f"Loaded {len(claims)} claims, {len(user_history)} history rows, {len(evidence)} evidence rows")

    token_tracker = TokenTracker()
    rate_limiter = AdaptiveRateLimiter()
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    safety_blocked = sum(1 for r in final_results if 'safety gate blocked' in r.get('claim_status_justification', '').lower())
    safety_flagged = sum(1 for r in final_results if 'manual review suggested' in r.get('claim_status_justification', '').lower())
    vlm_results = sum(1 for r in final_results if r.get('claim_status') in ('supported', 'contradicted'))
    print("\n=== Pipeline Summary ===")
    print(f"  Claims processed: {len(final_results)}")
    print(f"  From checkpoint: {checkpoint.get_completed_count()}")
    print(f"  New API calls: {summary['total_calls']}")
    print(f"  Input tokens: {summary['input_tokens']}")
    print(f"  Output tokens: {summary['output_tokens']}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.6f}")
    print(f"  Elapsed time: {summary['elapsed_seconds']:.1f}s")
    print(f"  Safety gate blocked: {safety_blocked}")
    print(f"  Safety gate flagged: {safety_flagged}")
    print(f"  VLM decisions made: {vlm_results}")
    print(f"  Peak RPM: {rl_stats['current_rpm']}")
    print(f"  Peak TPM: {rl_stats['current_tpm']}")


def cli():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Evidence Review Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                            # Run production pipeline on claims.csv
  python main.py --verbose                  # Enable debug logging
  python main.py --dry-run                  # Data validation + safety gate only
  python main.py --eval                     # Run evaluation on sample_claims.csv
  python main.py --reset-checkpoint         # Clear checkpoint and re-run all claims
  python main.py --skip-checkpoint          # Skip checkpoint (re-process all)
  python main.py --model groq/qwen/qwen3.6-27b   # Override model from env/config
        """
    )
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Clear checkpoint and re-process all claims')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug-level logging')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model (LiteLLM format, e.g. groq/qwen/qwen3.6-27b)')
    parser.add_argument('--output', type=str, default=None,
                        help='Override output CSV path')
    parser.add_argument('--dry-run', action='store_true',
                        help='Load and validate data, run preprocessor + safety gate, skip VLM calls')
    parser.add_argument('--eval', action='store_true',
                        help='Run evaluation on sample_claims.csv instead of claims.csv')
    parser.add_argument('--skip-checkpoint', action='store_true',
                        help='Ignore existing checkpoint and re-process all claims')
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

    if args.skip_checkpoint:
        import os
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            logger.info("Checkpoint file removed. All claims will be re-processed.")

    main(eval_mode=args.eval)


if __name__ == '__main__':
    cli()
