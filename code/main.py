import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.loader import load_all, load_sample_claims
from pipeline.preprocessor import preprocess_claim

TEST_ROWS = 5


def main():
    claims, user_history, evidence = load_all()
    print(f"Loaded {len(claims)} claims, {len(user_history)} history rows, {len(evidence)} evidence rows")

    sample = load_sample_claims()
    print(f"Loaded {len(sample)} sample claims")

    print(f"\n--- Testing preprocessor on first {TEST_ROWS} sample rows ---\n")
    for i, (_, row) in enumerate(sample.head(TEST_ROWS).iterrows()):
        result = preprocess_claim(row, user_history)
        print(f"Row {i+1}: user={result['user_id']}, object={result['claim_object']}, "
              f"valid_image={result['valid_image']}, image_ids={result['image_ids']}, "
              f"history={'yes' if result['history'] else 'no'}")
        if result['valid_image']:
            for p in result['image_paths']:
                print(f"  Path: {p}")
        print()


if __name__ == '__main__':
    main()
