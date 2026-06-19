# Evaluation Report

## Model Configuration
- Model: gemini-2.0-flash
- Temperature: 0.1
- Max images per call: 4

## Sample Claims Performance (n=[n])
- Accuracy claim_status: [claim_status_acc]% ([correct_claim_status]/[n])
- Accuracy issue_type: [issue_type_acc]% ([correct_issue_type]/[n])
- Accuracy object_part: [object_part_acc]% ([correct_object_part]/[n])

## Operational Analysis
- Total model calls: [model_calls]
- Total input tokens: [input_tokens]
- Total output tokens: [output_tokens]
- Total images processed: [total_images]
- Average latency per claim: [avg_latency] seconds
- Estimated cost: [cost]
- Peak TPM: [peak_tpm]
- Peak RPM: [peak_rpm]

## Strategies Compared
1. Baseline: Single call with full evidence_requirements
2. Optimized: Dynamic filtering (chosen)
   - Token reduction: N/A
   - Accuracy delta: N/A

## Rate Limiting Strategy
- Exponential backoff with max 5 retries
- 0.6s sleep between calls
- Batch size: 5 claims, then 10s pause
