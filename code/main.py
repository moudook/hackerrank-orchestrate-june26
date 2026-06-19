import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

from pipeline.loader import load_all, load_sample_claims
from pipeline.preprocessor import preprocess_claim
from pipeline.evidence_filter import get_relevant_rule
from pipeline.vision_analyzer import run_vision_analysis
from utils.token_tracker import TokenTracker


def main():
    claims, user_history, evidence = load_all()
    sample = load_sample_claims()

    row = sample.iloc[0]
    print(f"=== Testing 1 claim: user={row['user_id']}, object={row['claim_object']} ===\n")

    preprocessed = preprocess_claim(row, user_history)
    print(f"Preprocessor: {preprocessed['image_ids']}, valid={preprocessed['valid_image']}")

    rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)
    print(f"Evidence rule: {rule['requirement_id']} -> {rule['minimum_image_evidence'][:60]}...\n")

    tracker = TokenTracker()
    result = run_vision_analysis(preprocessed, rule, tracker)

    if result:
        print("=== Gemini Response ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
        print(f"\nToken cost: ${tracker.get_cost():.6f}")
        print(f"Calls: {tracker.calls}")
    else:
        print("Vision analysis returned None (no valid images or parsing failed)")


if __name__ == '__main__':
    main()
