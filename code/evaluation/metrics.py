import pandas as pd

GROUND_TRUTH_COLUMNS = ['issue_type', 'object_part', 'claim_status']


def compute_accuracy(predictions_df, ground_truth_df):
    merged = ground_truth_df[['user_id'] + GROUND_TRUTH_COLUMNS].copy()
    merged = merged.merge(
        predictions_df[['user_id', 'issue_type', 'object_part', 'claim_status']],
        on='user_id', suffixes=('_true', '_pred')
    )

    metrics = {}
    total = len(merged)
    for col in GROUND_TRUTH_COLUMNS:
        correct = (merged[f'{col}_true'] == merged[f'{col}_pred']).sum()
        metrics[col] = {
            'correct': int(correct),
            'total': total,
            'accuracy': round(correct / total * 100, 1) if total else 0.0
        }

    return metrics, merged
