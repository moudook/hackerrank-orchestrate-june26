# Evaluation Report

## Model Configuration
- Model: gemini-2.0-flash
- Temperature: 0.1
- Max images per call: 4

## Sample Claims Performance (n=20)
- Accuracy claim_status: 15.0% (3/20)
- Accuracy issue_type: 15.0% (3/20)
- Accuracy object_part: 5.0% (1/20)

## Operational Analysis
- Total model calls: 0
- Total input tokens: 11620
- Total output tokens: 0
- Total images processed: 29
- Average latency per claim: 1.8 seconds
- Estimated cost: $0.0012
- Peak TPM: 11620
- Peak RPM: 0

## Strategies Compared
1. Baseline: Single call with full evidence_requirements
2. Optimized: Dynamic filtering (chosen)
   - Token reduction: N/A
   - Accuracy delta: N/A

## Rate Limiting Strategy
- Exponential backoff with max 5 retries
- 0.6s sleep between calls
- Batch size: 5 claims, then 10s pause
