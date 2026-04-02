#!/usr/bin/env python3
"""Check upstream Triton test results against baseline.

Compares the latest test run results to the baseline and fails if
the pass count has regressed.

Usage:
    python scripts/check_upstream_regression.py --report-dir reports/
"""

import argparse
import json
import os
import re
import sys


def parse_results(report_dir):
    """Parse test results from the report directory."""
    # Try JSON report first
    json_path = os.path.join(report_dir, "upstream_test_core.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            return data.get("passed", 0), data.get("failed", 0)

    # Fall back to raw text output
    txt_path = os.path.join(report_dir, "upstream_test_core.txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            text = f.read()
        # Parse "X passed, Y failed, Z skipped" from pytest output
        match = re.search(r"(\d+) passed", text)
        passed = int(match.group(1)) if match else 0
        match = re.search(r"(\d+) failed", text)
        failed = int(match.group(1)) if match else 0
        return passed, failed

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Check upstream test regression")
    parser.add_argument("--report-dir", default="reports/", help="Directory with test reports")
    parser.add_argument("--baseline", default=None, help="Path to baseline.json")
    args = parser.parse_args()

    # Load baseline
    baseline_path = args.baseline or os.path.join(args.report_dir, "baseline.json")
    if not os.path.exists(baseline_path):
        print(f"No baseline found at {baseline_path}, skipping regression check")
        sys.exit(0)

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_pass = baseline["pass_count"]
    baseline_fail = baseline.get("fail_count", 0)

    # Parse current results
    passed, failed = parse_results(args.report_dir)
    if passed is None:
        print("No test results found, skipping regression check")
        sys.exit(0)

    print(f"Baseline: {baseline_pass} passed, {baseline_fail} failed")
    print(f"Current:  {passed} passed, {failed} failed")

    # Check for regressions
    regressed = False

    if passed < baseline_pass:
        print(f"REGRESSION: Pass count dropped from {baseline_pass} to {passed} (-{baseline_pass - passed})")
        regressed = True

    if failed > baseline_fail:
        print(f"REGRESSION: Fail count increased from {baseline_fail} to {failed} (+{failed - baseline_fail})")
        regressed = True

    if regressed:
        sys.exit(1)

    if passed > baseline_pass:
        print(f"IMPROVEMENT: Pass count increased from {baseline_pass} to {passed} (+{passed - baseline_pass})")
        print(f"Consider updating baseline.json")

    print("No regression detected")


if __name__ == "__main__":
    main()
