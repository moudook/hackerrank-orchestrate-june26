import concurrent.futures
import logging
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

import pandas as pd  # noqa: E402
from config import CACHE_DIR, CACHE_ENABLED, MODEL_NAME, RATE_LIMIT_RPM, RATE_LIMIT_TPM  # noqa: E402
from pipeline.evidence_filter import get_relevant_rule  # noqa: E402
from pipeline.loader import load_all, load_sample_claims  # noqa: E402
from pipeline.postprocessor import apply_claim_decision  # noqa: E402
from pipeline.preprocessor import preprocess_claim  # noqa: E402
from pipeline.safety_gate import evaluate_safety_gate  # noqa: E402
from pipeline.validator import validate_output  # noqa: E402
from pipeline.vision_analyzer import safe_run_vision_analysis  # noqa: E402
from utils.cache import ResponseCache  # noqa: E402
from utils.rate_limiter import TokenBucketRateLimiter  # noqa: E402
from utils.token_tracker import TokenTracker  # noqa: E402

from evaluation.metrics import compute_accuracy, compute_detailed_metrics  # noqa: E402

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_OUTPUT_PATH = os.path.join(REPO_ROOT, 'evaluation_report.md')
REPORT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'report_template.md')

BASELINE_STRATEGY_RESULT = {
    'issue_type': 'unknown',
    'object_part': 'unknown',
    'confidence': 0.0,
    'supporting_image_ids': 'none',
    'evidence_standard_met': False,
    'visual_description': 'Baseline fallback - no VLM call',
    'severity': 'unknown',
    'image_quality': 'poor',
    'image_quality_issues': 'none',
    'manipulation_suspected': False,
    'risk_flags': 'manual_review_required'
}


def run_baseline_strategy(sample, user_history, evidence):
    logger.info("Running Strategy A: Baseline (fallback only)")
    results = []

    for idx, row in sample.iterrows():
        preprocessed = preprocess_claim(row, user_history)
        evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)
        decision = apply_claim_decision(preprocessed, BASELINE_STRATEGY_RESULT, evidence_rule)
        validated = validate_output(decision)
        results.append(validated)

    pred_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    metrics, merged = compute_accuracy(pred_df, sample)
    return metrics, merged


def run_optimized_strategy(sample, user_history, evidence):
    logger.info("Running Strategy B: Optimized VLM pipeline")

    token_tracker = TokenTracker()
    rate_limiter = TokenBucketRateLimiter(rpm=RATE_LIMIT_RPM, tpm=RATE_LIMIT_TPM)
    cache = ResponseCache(CACHE_DIR, enabled=CACHE_ENABLED)

    def evaluate_single(item):
        idx, row = item
        logger.info(f"[{idx+1}/{len(sample)}] Evaluating user={row['user_id']}")

        preprocessed = preprocess_claim(row, user_history)
        num_images = len(preprocessed['image_ids'])

        evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)

        gate_result = evaluate_safety_gate(preprocessed)
        if gate_result and gate_result.get('blocked'):
            decision = apply_claim_decision(
                preprocessed, None, evidence_rule,
                override_risk_flags=gate_result['risk_flags'],
                override_justification=gate_result['reason']
            )
            validated = validate_output(decision)
            return validated, num_images

        estimated_tokens = 1000 + num_images * 258
        rate_limiter.acquire(estimated_tokens)

        vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
        if gate_result:
            if vision_result:
                existing = vision_result.get('risk_flags', 'none')
                combined = f"{existing};{gate_result['risk_flags']}" if existing != 'none' else gate_result['risk_flags']
                vision_result['risk_flags'] = combined
        decision = apply_claim_decision(preprocessed, vision_result, evidence_rule)
        validated = validate_output(decision)

        return validated, num_images

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        mapped = list(executor.map(evaluate_single, sample.iterrows()))

    results = [r[0] for r in mapped]
    total_images = sum(r[1] for r in mapped)
    pred_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    metrics, merged = compute_accuracy(pred_df, sample)
    detailed = compute_detailed_metrics(merged)

    summary = token_tracker.summary()
    elapsed = summary['elapsed_seconds']
    total = len(sample)
    avg_latency = round(elapsed / total, 1) if total else 0.0

    detailed['claim_status']
    safety_gate_stats = {'blocked': 0, 'flagged': 0}
    for r in results:
        j = r.get('claim_status_justification', '').lower()
        if 'safety gate blocked' in j:
            safety_gate_stats['blocked'] += 1
        elif 'manual review suggested' in j:
            safety_gate_stats['flagged'] += 1

    op_info = {
        'model_calls': summary['total_calls'],
        'input_tokens': summary['input_tokens'],
        'output_tokens': summary['output_tokens'],
        'total_images': total_images,
        'avg_latency': avg_latency,
        'cost': f"${summary['estimated_cost']:.4f}",
        'peak_tpm': summary['input_tokens'] // max(int(elapsed // 60), 1) if elapsed > 60 else summary['input_tokens'],
        'peak_rpm': summary['total_calls'],
        'cache_hits': cache._hits if hasattr(cache, '_hits') else 0,
        'cache_misses': cache._misses if hasattr(cache, '_misses') else 0,
        'safety_blocked': safety_gate_stats['blocked'],
        'safety_flagged': safety_gate_stats['flagged'],
    }

    return metrics, detailed, op_info


def run_evaluation(claims_path=None, output_path=None, report_path=None):
    claims, user_history, evidence = load_all()
    sample = load_sample_claims(claims_path) if claims_path else load_sample_claims()

    baseline_metrics, _ = run_baseline_strategy(sample, user_history, evidence)
    logger.info("Baseline complete")

    opt_metrics, opt_detailed, op_info = run_optimized_strategy(sample, user_history, evidence)
    logger.info("Optimized complete")

    safety_blocked = op_info.get('safety_blocked', 0)
    safety_flagged = op_info.get('safety_flagged', 0)
    logger.info(f"Safety gate: {safety_blocked} blocked, {safety_flagged} flagged")

    _write_report(opt_metrics, opt_detailed, op_info, baseline_metrics, sample, report_path)
    return baseline_metrics, opt_metrics, opt_detailed, op_info


def _write_report(opt_metrics, opt_detailed, op_info, baseline_metrics, sample, report_path=None):
    total = len(sample)
    cs_metrics = opt_detailed['claim_status']
    report_path = report_path or REPORT_OUTPUT_PATH

    report_vars = {
        'model': MODEL_NAME,
        'n': total,
        'baseline_claim_status_acc': baseline_metrics['claim_status']['accuracy'],
        'baseline_issue_type_acc': baseline_metrics['issue_type']['accuracy'],
        'baseline_object_part_acc': baseline_metrics['object_part']['accuracy'],
        'baseline_correct_claim_status': baseline_metrics['claim_status']['correct'],
        'baseline_correct_issue_type': baseline_metrics['issue_type']['correct'],
        'baseline_correct_object_part': baseline_metrics['object_part']['correct'],
        'claim_status_acc': opt_metrics['claim_status']['accuracy'],
        'issue_type_acc': opt_metrics['issue_type']['accuracy'],
        'object_part_acc': opt_metrics['object_part']['accuracy'],
        'correct_claim_status': opt_metrics['claim_status']['correct'],
        'correct_issue_type': opt_metrics['issue_type']['correct'],
        'correct_object_part': opt_metrics['object_part']['correct'],
        'cs_supported_precision': cs_metrics.get('supported_precision', 0),
        'cs_supported_recall': cs_metrics.get('supported_recall', 0),
        'cs_supported_f1': cs_metrics.get('supported_f1', 0),
        'cs_contradicted_precision': cs_metrics.get('contradicted_precision', 0),
        'cs_contradicted_recall': cs_metrics.get('contradicted_recall', 0),
        'cs_contradicted_f1': cs_metrics.get('contradicted_f1', 0),
        'cs_nei_precision': cs_metrics.get('not_enough_information_precision', 0),
        'cs_nei_recall': cs_metrics.get('not_enough_information_recall', 0),
        'cs_nei_f1': cs_metrics.get('not_enough_information_f1', 0),
        'model_calls': op_info['model_calls'],
        'input_tokens': op_info['input_tokens'],
        'output_tokens': op_info['output_tokens'],
        'total_images': op_info['total_images'],
        'avg_latency': op_info['avg_latency'],
        'cost': op_info['cost'],
        'peak_tpm': op_info['peak_tpm'],
        'peak_rpm': op_info['peak_rpm'],
        'rate_limit_rpm': RATE_LIMIT_RPM,
        'rate_limit_tpm': RATE_LIMIT_TPM,
        'safety_blocked': op_info['safety_blocked'],
        'safety_flagged': op_info['safety_flagged'],
        'cache_dir': str(CACHE_DIR) if CACHE_DIR else 'disabled',
        'cache_hit_rate': round(op_info['cache_hits'] / (op_info['cache_hits'] + op_info['cache_misses']) * 100, 1)
            if (op_info['cache_hits'] + op_info['cache_misses']) > 0 else 0,
    }

    with open(REPORT_TEMPLATE_PATH, 'r') as f:
        template = f.read()

    report = template
    for key, val in report_vars.items():
        placeholder = f'[{key}]'
        report = report.replace(placeholder, str(val))

    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report written to {report_path}")

    print("\n=== Evaluation Results ===")
    print(f"  Sample claims: {total}")
    print("  Strategy A (Baseline):")
    print(f"    claim_status: {baseline_metrics['claim_status']['accuracy']}%")
    print(f"    issue_type: {baseline_metrics['issue_type']['accuracy']}%")
    print(f"    object_part: {baseline_metrics['object_part']['accuracy']}%")
    print("  Strategy B (Optimized):")
    print(f"    claim_status: {opt_metrics['claim_status']['accuracy']}%")
    print(f"    issue_type: {opt_metrics['issue_type']['accuracy']}%")
    print(f"    object_part: {opt_metrics['object_part']['accuracy']}%")
    print(f"  Model calls: {op_info['model_calls']}")
    print(f"  Safety gate blocked: {op_info['safety_blocked']}")
    print(f"  Safety gate flagged: {op_info['safety_flagged']}")
    print(f"  Cost: {op_info['cost']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate pipeline on sample claims',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/main.py                    # Run full evaluation with report
  python evaluation/main.py --baseline-only    # Skip VLM pipeline, baseline only
  python evaluation/main.py --quick            # Baseline only, print results
  python evaluation/main.py --output report.md # Custom report path
        """
    )
    parser.add_argument('--baseline-only', action='store_true',
                        help='Skip Strategy B (optimized VLM pipeline)')
    parser.add_argument('--quick', action='store_true',
                        help='Baseline only with compact output')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output report path')
    args = parser.parse_args()

    run_evaluation(report_path=args.output)


if __name__ == '__main__':
    main()
