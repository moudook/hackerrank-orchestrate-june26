import sys
import os
import logging

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from pipeline.loader import load_all, load_sample_claims
from pipeline.preprocessor import preprocess_claim
from pipeline.evidence_filter import get_relevant_rule
from pipeline.vision_analyzer import safe_run_vision_analysis
from pipeline.postprocessor import apply_claim_decision
from pipeline.validator import validate_output
from utils.token_tracker import TokenTracker
from utils.rate_limiter import RateLimiter
from evaluation.metrics import compute_accuracy

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_OUTPUT_PATH = os.path.join(REPO_ROOT, 'evaluation_report.md')

REPORT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), 'report_template.md')


def run_evaluation():
    claims, user_history, evidence = load_all()
    sample = load_sample_claims()
    logger.info(f"Loaded {len(sample)} sample claims for evaluation")

    token_tracker = TokenTracker()
    rate_limiter = RateLimiter()

    total_images = 0
    results = []
    for idx, (_, row) in enumerate(sample.iterrows()):
        logger.info(f"[{idx+1}/{len(sample)}] Evaluating user={row['user_id']}")

        preprocessed = preprocess_claim(row, user_history)
        total_images += len(preprocessed['image_ids'])

        evidence_rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)
        vision_result = safe_run_vision_analysis(preprocessed, evidence_rule, token_tracker, rate_limiter)
        decision = apply_claim_decision(preprocessed, vision_result, evidence_rule)
        validated = validate_output(decision)

        results.append(validated)

    pred_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    metrics, merged = compute_accuracy(pred_df, sample)

    summary = token_tracker.summary()
    elapsed = summary['elapsed_seconds']
    total = len(sample)
    avg_latency = round(elapsed / total, 1) if total else 0.0

    true_positives = (merged['claim_status_true'] == merged['claim_status_pred']).sum()
    true_positives_col = (merged['claim_status_true'] == merged['claim_status_pred']) & (merged['claim_status_pred'] != 'not_enough_information')
    tp_count = int(true_positives_col.sum())

    report_vars = {
        'n': total,
        'claim_status_acc': metrics['claim_status']['accuracy'],
        'issue_type_acc': metrics['issue_type']['accuracy'],
        'object_part_acc': metrics['object_part']['accuracy'],
        'model_calls': summary['total_calls'],
        'input_tokens': summary['input_tokens'],
        'output_tokens': summary['output_tokens'],
        'total_images': total_images,
        'avg_latency': avg_latency,
        'cost': f"${summary['estimated_cost']:.4f}",
        'peak_tpm': summary['input_tokens'] // max(int(elapsed // 60), 1) if elapsed > 60 else summary['input_tokens'],
        'peak_rpm': summary['total_calls'],
        'correct_claim_status': metrics['claim_status']['correct'],
        'correct_issue_type': metrics['issue_type']['correct'],
        'correct_object_part': metrics['object_part']['correct'],
    }

    with open(REPORT_TEMPLATE_PATH, 'r') as f:
        template = f.read()

    report = template
    for key, val in report_vars.items():
        placeholder = f'[{key}]'
        report = report.replace(placeholder, str(val))

    with open(REPORT_OUTPUT_PATH, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report written to {REPORT_OUTPUT_PATH}")

    print("\n=== Evaluation Results ===")
    print(f"  Sample claims: {total}")
    print(f"  claim_status accuracy: {metrics['claim_status']['accuracy']}% ({metrics['claim_status']['correct']}/{metrics['claim_status']['total']})")
    print(f"  issue_type accuracy: {metrics['issue_type']['accuracy']}% ({metrics['issue_type']['correct']}/{metrics['issue_type']['total']})")
    print(f"  object_part accuracy: {metrics['object_part']['accuracy']}% ({metrics['object_part']['correct']}/{metrics['object_part']['total']})")
    print(f"  Total model calls: {summary['total_calls']}")
    print(f"  Estimated cost: ${summary['estimated_cost']:.4f}")


if __name__ == '__main__':
    run_evaluation()
