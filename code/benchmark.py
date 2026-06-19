"""Performance benchmark for the evidence review pipeline."""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from utils.logger import setup_logging

setup_logging()
logging.getLogger().setLevel(logging.WARNING)

from config import validate_config

OUTPUT_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]


def benchmark_dry_run(count=44):
    from pipeline.loader import load_all
    from pipeline.preprocessor import preprocess_claim
    from pipeline.safety_gate import evaluate_safety_gate
    from pipeline.evidence_filter import get_relevant_rule

    claims, user_history, evidence = load_all()
    times = []
    samples = claims.head(min(count, len(claims)))
    for idx, row in samples.iterrows():
        t0 = time.perf_counter()
        pre = preprocess_claim(row, user_history)
        evaluate_safety_gate(pre)
        get_relevant_rule(pre['claim_object'], pre['user_claim'], evidence)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    total = sum(times)
    avg = total / len(times) if times else 0
    max_t = max(times) if times else 0
    min_t = min(times) if times else 0
    throughput = len(times) / total if total > 0 else 0
    return {
        'claims': len(times),
        'total_seconds': round(total, 3),
        'avg_ms': round(avg * 1000, 1),
        'min_ms': round(min_t * 1000, 1),
        'max_ms': round(max_t * 1000, 1),
        'throughput_sec': round(throughput, 1),
    }


def benchmark_full_pipeline(count=10):
    from pipeline.loader import load_all
    from pipeline.preprocessor import preprocess_claim
    from pipeline.postprocessor import apply_claim_decision
    from pipeline.validator import validate_output
    from pipeline.evidence_filter import get_relevant_rule
    from pipeline.safety_gate import evaluate_safety_gate
    from pipeline.vision_analyzer import safe_run_vision_analysis
    from utils.token_tracker import TokenTracker
    from utils.rate_limiter import AdaptiveRateLimiter

    claims, user_history, evidence = load_all()
    token_tracker = TokenTracker()
    rate_limiter = AdaptiveRateLimiter()
    times = []
    samples = claims.head(min(count, len(claims)))
    for idx, row in samples.iterrows():
        t0 = time.perf_counter()
        pre = preprocess_claim(row, user_history)
        gate = evaluate_safety_gate(pre)
        rule = get_relevant_rule(pre['claim_object'], pre['user_claim'], evidence)
        if gate and gate.get('blocked'):
            decision = apply_claim_decision(pre, None, rule, override_risk_flags=gate['risk_flags'], override_justification=gate['reason'])
            validated = validate_output(decision)
        else:
            rate_limiter.acquire(1000)
            vision = safe_run_vision_analysis(pre, rule, token_tracker, rate_limiter)
            if gate and vision:
                existing = vision.get('risk_flags', 'none')
                combined = f"{existing};{gate['risk_flags']}" if existing != 'none' else gate['risk_flags']
                vision['risk_flags'] = combined
            decision = apply_claim_decision(pre, vision, rule)
            validated = validate_output(decision)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    total = sum(times)
    avg = total / len(times) if times else 0
    return {
        'claims': len(times),
        'total_seconds': round(total, 3),
        'avg_ms': round(avg * 1000, 1),
        'output_claim_status': validated['claim_status'],
    }


def main():
    if not validate_config():
        print("Config check failed — running benchmark anyway")

    print("=" * 50)
    print("Pipeline Performance Benchmark")
    print("=" * 50)

    print("\n--- Dry-Run Benchmark (preprocessor + safety gate + evidence filter) ---")
    dry = benchmark_dry_run(count=44)
    print(f"  Claims: {dry['claims']}")
    print(f"  Total: {dry['total_seconds']}s")
    print(f"  Avg: {dry['avg_ms']}ms/claim")
    print(f"  Min: {dry['min_ms']}ms | Max: {dry['max_ms']}ms")
    print(f"  Throughput: {dry['throughput_sec']} claims/sec")

    print("\n--- Full Pipeline Benchmark (first 10 claims) ---")
    full = benchmark_full_pipeline(count=10)
    print(f"  Claims: {full['claims']}")
    print(f"  Total: {full['total_seconds']}s")
    print(f"  Avg: {full['avg_ms']}ms/claim")
    print(f"  Last output status: {full['output_claim_status']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
