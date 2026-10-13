"""Reviewer replay for Math- PR #444 (C6 residual child-result contract), session 017Mi3hxjaxV45x6zo6o1ee3.

Runs the pinned checker reviews/c6_residual_closure_20260930/residual_check.py with no mutant and with all
fifteen mutants, in both Python modes, and feeds every child result through the PR's helper
tools/c6_residual_child_contract.py exactly as the amended workflow does.  With --simulate-installed it first
writes the checker's own final build of the proposal into frontiers/downstream_gate_20260925/GRAPH.json (the
installed register state; the zero-mutant output then equals RESULTS_INSTALLED.json byte for byte) and restores
the original bytes afterwards.  Standard library only.  Nothing is pushed; the simulation is local only.

Usage, from a checkout of the PR head (ad6667124c71483796ce2176f72475199c8890b1 or its successor):
    python3 -B -S replay_c6_residual_contract.py                      # baseline register state
    python3 -B -S replay_c6_residual_contract.py --simulate-installed # installed register state, simulated
Exit status 0 when every child result is accepted by the helper and 1 otherwise.
"""
import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(".").resolve()
RECORD = ROOT / "reviews/c6_residual_closure_20260930"
GRAPH = ROOT / "frontiers/downstream_gate_20260925/GRAPH.json"
MUTANTS = ["partial-install", "allow-symlink", "no-hash", "stale-fingerprint", "drop-required-edge", "executed-flag",
           "controlling-true", "tail-reversed", "pair-identity-broken", "route-c-broken", "drop-review-needle",
           "installed-drift", "snapshot-omitted", "record-unreviewed", "reviewed-text-drift"]


def load_checker_module():
    spec = importlib.util.spec_from_file_location("rc", RECORD / "residual_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(label):
    sys.path.insert(0, str(ROOT / "tools"))
    from c6_residual_child_contract import EXPECTED_FAILURES, validate_baseline, validate_mutant
    assert set(MUTANTS) == set(EXPECTED_FAILURES), "workflow mutant list differs from the helper map"
    script = str(RECORD / "residual_check.py")
    all_ok = True
    for flags in (["-B", "-S"], ["-B", "-O", "-S"]):
        base = subprocess.run([sys.executable, *flags, script], capture_output=True, timeout=300)
        report = json.loads(base.stdout) if base.stdout else {}
        state = report.get("register_state")
        pinned_name = "RESULTS.json" if state == "baseline" else "RESULTS_INSTALLED.json"
        pinned = (RECORD / pinned_name).read_bytes()
        print(f"[{label}] {' '.join(flags)} zero-mutant: rc={base.returncode} register_state={state} "
              f"stdout=={pinned_name}: {base.stdout == pinned} stderr_empty={base.stderr == b''}")
        try:
            baseline = validate_baseline(base, pinned)
            print("    validate_baseline: accepted")
        except Exception as exc:  # noqa: BLE001 - report the helper's refusal verbatim
            print("    validate_baseline REFUSED:", exc)
            all_ok = False
            baseline = report
        for mutant in MUTANTS:
            proc = subprocess.run([sys.executable, *flags, script, "--mutant", mutant], capture_output=True, timeout=300)
            rep = json.loads(proc.stdout) if proc.stdout else {}
            checks = rep.get("checks") if isinstance(rep.get("checks"), dict) else {}
            failed = sorted(k for k, v in checks.items() if v is False)
            try:
                validate_mutant(mutant, proc, baseline)
                outcome = "accepted"
            except Exception as exc:  # noqa: BLE001
                outcome = "REFUSED: " + str(exc)
                all_ok = False
            mark = "" if failed == sorted(EXPECTED_FAILURES[mutant]) else "   <-- differs from the frozen map"
            print(f"    {mutant}: rc={proc.returncode} stderr={len(proc.stderr)}B passed={rep.get('passed')} "
                  f"register_state={rep.get('register_state')} failed={failed} frozen={sorted(EXPECTED_FAILURES[mutant])} "
                  f"-> {outcome}{mark}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-installed", action="store_true",
                        help="write the checker's final build of the proposal into GRAPH.json for the run, then restore")
    args = parser.parse_args()
    started = time.time()
    if not args.simulate_installed:
        ok = replay("baseline")
    else:
        rc = load_checker_module()
        spec = rc.load_spec(ROOT)
        graph, _sel = rc.load_live(ROOT)
        print("live register state before simulation:", rc.installed_state(graph, spec)["state"])
        _old, new = rc.build(spec, graph, "final")
        print("state of the final build:", rc.installed_state(new, spec)["state"])
        original = GRAPH.read_bytes()
        GRAPH.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        try:
            ok = replay("installed-simulated")
        finally:
            GRAPH.write_bytes(original)
            print("GRAPH.json restored byte for byte:", GRAPH.read_bytes() == original)
    print("RESULT:", "all child results accepted" if ok else "at least one child result refused",
          f"({time.time() - started:.1f} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
