# Evaluation Report

## Model Configuration
- Model: gemini-2.0-flash
- Temperature: 0.1
- Max images per call: 4

## Sample Claims Performance (n=50)
- Accuracy claim_status: X%
- Accuracy issue_type: X%
- Accuracy object_part: X%

## Operational Analysis
- Total model calls: [from token_tracker]
- Total input tokens: [sum]
- Total output tokens: [sum]
- Total images processed: [count]
- Average latency per claim: [total_time/n] seconds
- Estimated cost: $[ (input/1e6*0.10) + (output/1e6*0.40) ]
- Peak TPM: [max in any minute]
- Peak RPM: [max in any minute]

## Strategies Compared
1. Baseline: Single call with full evidence_requirements
2. Optimized: Dynamic filtering (chosen)
   - Token reduction: X%
   - Accuracy delta: +X%

## Rate Limiting Strategy
- Exponential backoff with max 5 retries
- 0.6s sleep between calls
- Batch size: 5 claims, then 10s pause
