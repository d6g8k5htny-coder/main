#!/usr/bin/env python3
"""Run every equation, package, and regression check in the V3.4 checkpoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKS = [
    "verify_checkpoint_structure.py",
    "verify_constrained_triple_contact.py",
    "verify_corrected_det_hs.py",
    "verify_value_pinned_sign_impact.py",
    "audit_original_v2_manifest.py",
    "verify_reconciliation_evidence.py",
    "verify_fixed_distance_power_count.py",
    "verify_capture_escape_budgets.py",
    "audit_density_vs_cumulative.py",
    "audit_v2_elder_mark.py",
    "verify_palm_transfer_repair.py",
    "verify_joint_collar_regression.py",
    "verify_facewise_collar_axis_integration.py",
    "verify_quartic_endpoint_counterexample.py",
    "verify_hybrid_mixed_curvature_and_axis_absorption.py",
    "audit_v5_forward_branch.py",
]

# This audit internally executes every frozen V5 target in both modes and
# compares its committed transcript.  Running the wrapper itself twice would
# duplicate the same expensive audit without adding coverage.
SELF_OPTIMIZING = {"audit_v5_forward_branch.py"}


def run(check: str, optimized: bool) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.append(str(ROOT / check))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )


for check in CHECKS:
    print(f"===== {check} =====", flush=True)
    normal = run(check, optimized=False)
    if normal.returncode != 0:
        sys.stdout.write(normal.stdout)
        sys.stderr.write(normal.stderr)
        raise SystemExit(f"CHECK_FAILED: {check} returned {normal.returncode}")
    if normal.stderr:
        sys.stderr.write(normal.stderr)
        raise SystemExit(f"CHECK_FAILED: {check} produced stderr")
    sys.stdout.write(normal.stdout)

    if check in SELF_OPTIMIZING:
        print("OPTIMIZED_COVERAGE = INTERNAL")
        continue

    optimized = run(check, optimized=True)
    if optimized.returncode != 0:
        sys.stdout.write(optimized.stdout)
        sys.stderr.write(optimized.stderr)
        raise SystemExit(
            f"CHECK_FAILED: {check} -O returned {optimized.returncode}"
        )
    if optimized.stderr:
        sys.stderr.write(optimized.stderr)
        raise SystemExit(f"CHECK_FAILED: {check} -O produced stderr")
    if optimized.stdout != normal.stdout:
        raise SystemExit(f"CHECK_FAILED: {check} optimized output differs")
    print("OPTIMIZED_OUTPUT = BYTE_IDENTICAL")

print("===== aggregate =====")
print(f"CHECK_COUNT = {len(CHECKS)}")
print(f"DUAL_MODE_CHECK_COUNT = {len(CHECKS) - len(SELF_OPTIMIZING)}")
print(f"SELF_OPTIMIZING_CHECK_COUNT = {len(SELF_OPTIMIZING)}")
print("ALL_V3_CHECKS_PASS")
