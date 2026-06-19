import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

from pipeline.loader import load_all, load_sample_claims
from pipeline.preprocessor import preprocess_claim
from pipeline.evidence_filter import get_relevant_rule
from pipeline.postprocessor import apply_claim_decision
from pipeline.validator import validate_output

EXPECTED_COLUMNS = [
    'user_id', 'image_paths', 'user_claim', 'claim_object',
    'evidence_standard_met', 'evidence_standard_met_reason', 'risk_flags',
    'issue_type', 'object_part', 'claim_status', 'claim_status_justification',
    'supporting_image_ids', 'valid_image', 'severity'
]


def _mock_vision(overrides=None):
    base = {
        'issue_type': 'dent',
        'object_part': 'rear_bumper',
        'confidence': 0.92,
        'supporting_image_ids': 'img_1',
        'evidence_standard_met': True,
        'visual_description': 'clear dent on rear bumper',
        'severity': 'medium',
        'risk_flags': ''
    }
    if overrides:
        base.update(overrides)
    return base


NO_VISION = object()


def test_case(label, sample_row, user_history, evidence, vision_overrides=None, vision_src=None):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print(f"{'=' * 60}")

    preprocessed = preprocess_claim(sample_row, user_history)
    rule = get_relevant_rule(preprocessed['claim_object'], preprocessed['user_claim'], evidence)

    if vision_src is NO_VISION:
        vision_result = None
    elif vision_overrides:
        vision_result = _mock_vision(vision_overrides)
    else:
        vision_result = _mock_vision()

    decision = apply_claim_decision(preprocessed, vision_result, rule)
    validated = validate_output(decision)

    for col in EXPECTED_COLUMNS:
        val = validated.get(col, 'MISSING')
        print(f"  {col}: {val}")

    missing = [c for c in EXPECTED_COLUMNS if c not in validated]
    if missing:
        print(f"  ** MISSING COLUMNS: {missing}")
    else:
        print(f"  All {len(EXPECTED_COLUMNS)} columns present.")

    return validated


def main():
    claims, user_history, evidence = load_all()
    sample = load_sample_claims()

    print(f"Loaded {len(sample)} sample claims")

    test_case("Supported claim (dent, high confidence)", sample.iloc[0], user_history, evidence)

    test_case("No valid images", sample.iloc[0], user_history, evidence,
              vision_src=NO_VISION)

    test_case("Evidence standard not met", sample.iloc[0], user_history, evidence,
              vision_overrides={'evidence_standard_met': False})

    test_case("Low confidence (< 0.5)", sample.iloc[4], user_history, evidence,
              vision_overrides={'confidence': 0.3, 'issue_type': 'scratch',
                                'object_part': 'rear_bumper', 'severity': 'low'})

    test_case("None issue type (undamaged)", sample.iloc[4], user_history, evidence,
              vision_overrides={'issue_type': 'none', 'object_part': 'rear_bumper',
                                'confidence': 0.85, 'visual_description': 'rear bumper undamaged'})

    test_case("None with not visible description", sample.iloc[4], user_history, evidence,
              vision_overrides={'issue_type': 'none', 'confidence': 0.85,
                                'visual_description': 'part not visible in image'})

    test_case("Invalid enum correction", sample.iloc[0], user_history, evidence,
              vision_overrides={'issue_type': 'scratched', 'object_part': 'back_bumper',
                                'severity': 'critical'})

    test_case("History risk (rejected >= 3)", sample.iloc[4], user_history, evidence,
              vision_overrides={'issue_type': 'scratch', 'object_part': 'rear_bumper',
                                'confidence': 0.85})


if __name__ == '__main__':
    main()
