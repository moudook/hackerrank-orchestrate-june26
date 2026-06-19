"""
End-to-end demo for the Multi-Modal Evidence Review Pipeline.

Run from repo root:
    python demo/run_demo.py

This will:
1. Process sample_claims.csv through the pipeline (with fallback if no API key)
2. Run the evaluation comparing Strategy A (rule-based) vs Strategy B (VLM)
3. Print a formatted summary table to stdout
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

import logging
from pathlib import Path

from utils.logger import setup_logging, set_request_id
from evaluation.main import run_evaluation

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    set_request_id('demo-run')
    setup_logging(level=logging.INFO, json_output=False)

    print("=" * 70)
    print("  MULTI-MODAL EVIDENCE REVIEW PIPELINE — DEMO")
    print("=" * 70)

    sample_path = REPO_ROOT / 'dataset' / 'sample_claims.csv'
    output_path = REPO_ROOT / 'output_demo.csv'
    report_path = REPO_ROOT / 'evaluation_report_demo.md'

    if not sample_path.exists():
        print(f"\n[ERROR] Sample claims not found at: {sample_path}")
        print("Make sure you're running from the repo root.")
        sys.exit(1)

    print(f"\n[1/3] Loading sample claims from: {sample_path.name}")
    print(f"[2/3] Running pipeline (44 claims)...")
    print(f"      (VLM calls will use fallback if API keys are not configured)")
    print(f"[3/3] Running evaluation...\n")

    results = run_evaluation(
        claims_path=str(sample_path),
        output_path=str(output_path),
        report_path=str(report_path),
    )

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print(f"  Output CSV:     {output_path}")
    print(f"  Report:         {report_path}")
    print(f"  Sample claims:  {sample_path}")
    print(f"  Images in:      {REPO_ROOT / 'dataset' / 'images' / 'sample'}")
    print()
    print("  To process the full test set (claims.csv):")
    print("    cd code && python main.py --output ../output.csv")
    print()
    print("  To run with a different model:")
    print("    cd code && python main.py --model openai/gpt-4o")
    print()
    print("  To reset checkpoint and re-process everything:")
    print("    cd code && python main.py --reset-checkpoint")


if __name__ == '__main__':
    main()
