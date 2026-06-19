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


def compute_confusion_matrix(merged, col):
    true_vals = merged[f'{col}_true']
    pred_vals = merged[f'{col}_pred']
    labels = sorted(set(true_vals) | set(pred_vals))
    matrix = pd.crosstab(true_vals, pred_vals, rownames=['True'], colnames=['Pred'], margins=True)
    return matrix, labels


def compute_detailed_metrics(merged):
    results = {}

    for col in GROUND_TRUTH_COLUMNS:
        true_col = f'{col}_true'
        pred_col = f'{col}_pred'
        y_true = merged[true_col]
        y_pred = merged[pred_col]

        total = len(merged)
        correct = (y_true == y_pred).sum()
        accuracy = round(correct / total * 100, 1) if total else 0.0

        metrics = {
            'accuracy': accuracy,
            'correct': int(correct),
            'total': total
        }

        if col == 'claim_status':
            labels = ['supported', 'contradicted', 'not_enough_information']
            for label in labels:
                tp = ((y_true == label) & (y_pred == label)).sum()
                fp = ((y_true != label) & (y_pred == label)).sum()
                fn = ((y_true == label) & (y_pred != label)).sum()
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                metrics[f'{label}_precision'] = round(precision, 1)
                metrics[f'{label}_recall'] = round(recall, 1)
                metrics[f'{label}_f1'] = round(f1, 1)

        results[col] = metrics

    return results
