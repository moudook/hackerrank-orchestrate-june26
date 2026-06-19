# Evaluation Report

## Model Configuration
- Model: [model]
- Temperature: 0.0
- Max images per call: 4
- Response format: Structured JSON (response_json_schema)
- System prompt: code/prompts/system_vision.txt

## Strategies Compared

### Strategy A — Baseline (Fallback)
- Description: Rule-based fallback when VLM is unavailable (quota exhausted)
- All claims default to `not_enough_information` with `manual_review_required`
- Relies entirely on evidence_requirements.csv text for justifications
- No image analysis performed

### Strategy B — Optimized (VLM + Pipeline)
- Description: Full pipeline with Gemini 2.5 Flash VLM analysis
- Features: structured JSON output, trust manipulation detection, multi-language evidence filter
- Image quality pre-filtering, risk flag enrichment from VLM output
- User history risk integration with past claim patterns

## Sample Claims Performance (n=[n])

### Strategy A — Baseline (Fallback Only)
- Accuracy claim_status: [baseline_claim_status_acc]% ([baseline_correct_claim_status]/[n])
- Accuracy issue_type: [baseline_issue_type_acc]% ([baseline_correct_issue_type]/[n])
- Accuracy object_part: [baseline_object_part_acc]% ([baseline_correct_object_part]/[n])

### Strategy B — Optimized VLM Pipeline
- Accuracy claim_status: [claim_status_acc]% ([correct_claim_status]/[n])
- Accuracy issue_type: [issue_type_acc]% ([correct_issue_type]/[n])
- Accuracy object_part: [object_part_acc]% ([correct_object_part]/[n])

### claim_status Detailed Metrics (Strategy B)
- supported — precision: [cs_supported_precision]%, recall: [cs_supported_recall]%, F1: [cs_supported_f1]%
- contradicted — precision: [cs_contradicted_precision]%, recall: [cs_contradicted_recall]%, F1: [cs_contradicted_f1]%
- not_enough_information — precision: [cs_nei_precision]%, recall: [cs_nei_recall]%, F1: [cs_nei_f1]%

## Operational Analysis
- Total model calls: [model_calls]
- Total input tokens: [input_tokens]
- Total output tokens: [output_tokens]
- Total images processed: [total_images]
- Average latency per claim: [avg_latency] seconds
- Estimated cost: [cost]
- Peak TPM: [peak_tpm]
- Peak RPM: [peak_rpm]

## Rate Limiting Strategy
- Token bucket algorithm with RPM=[rate_limit_rpm] and TPM=[rate_limit_tpm]
- Exponential backoff with max 5 retries (excluding 429 ResourceExhausted)
- Concurrent processing with 5 worker threads
- Response cache (memory + disk) avoiding redundant API calls
- Checkpoint/resume system for crash recovery

## Caching
- Type: Hybrid memory + disk (SHA256 keyed on prompt + image bytes + model)
- Location: [cache_dir]
- Hit rate: [cache_hit_rate]%

## Limitations & Risks
- Free-tier API quota causes fallback Strategy A for all claims when exhausted
- Token estimation uses heuristic (words * 1.3) rather than actual API counting
- Multi-language support covers Hindi/Spanish/Chinese but not all languages
- Manipulation detection uses VLM judgment, not dedicated forensics tools
