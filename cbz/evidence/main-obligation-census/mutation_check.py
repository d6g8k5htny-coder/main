#!/usr/bin/env python3
"""Adverse mutation check for build_census.py.

Each mutant disables one validation. A mutant counts as killed only when the test
suite ran at least one test and reported FAILED. A syntax-broken copy must be
reported invalid, never killed. Exit 0 only if every mutant is killed.
Run: python3 -B -I mutation_check.py
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
MUTANTS = [
    ("require never raises", "    if not condition:\n        raise CensusError(message)", "    return None"),
    ("blob check removed", 'require(blob.startswith(entry["blob"]),', 'require(True or blob.startswith(entry["blob"]),'),
    ("range check removed", "require(1 <= first <= last <= len(lines),", "require(True or 1 <= first <= last <= len(lines),"),
    ("anchor check removed", "require(anchor in window,", "require(True or anchor in window,"),
    ("target count removed", "require(len(targets) == EXPECTED_LEAN_TARGETS,",
     "require(True or len(targets) == EXPECTED_LEAN_TARGETS,"),
    ("axiom check removed", 'require(manifest.get("allowed_axioms") == APPROVED_AXIOMS,',
     'require(True or manifest.get("allowed_axioms") == APPROVED_AXIOMS,'),
    ("missing path swallowed", "raise CensusError(f\"{entry['id']}: {entry['path']} does not exist at {rev}\") from None",
     "return"),
    ("output written after refusal", "        return 1\n    args.out.mkdir", "        pass\n    args.out.mkdir"),
]


def run_suite(directory):
    proc = subprocess.run([sys.executable, "-B", "-I", str(directory / "test_build_census.py")],
                          cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ran = RAN.search(proc.stdout)
    if not ran or int(ran.group(1)) == 0:
        return "invalid"
    if proc.returncode == 0 and "\nOK" in proc.stdout:
        return "pass"
    return "fail" if "FAILED (" in proc.stdout else "invalid"


def copy_with(source, directory):
    (directory / "build_census.py").write_text(source)
    shutil.copy(HERE / "test_build_census.py", directory / "test_build_census.py")


def main():
    source = (HERE / "build_census.py").read_text()
    if run_suite(HERE) != "pass":
        print("BASELINE FAIL: the unmutated suite does not pass")
        return 2
    # The copies must live inside the repository: the in-process tests read its Git objects.
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        copy_with(source + "\ndef (:\n", Path(tmp))
        if run_suite(Path(tmp)) != "invalid":
            print("CONTROL FAIL: a syntax-broken copy was not reported invalid")
            return 2
    killed = 0
    for name, old, new in MUTANTS:
        if source.count(old) != 1:
            print(f"NOT APPLICABLE {name} ({source.count(old)} matches)")
            continue
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            copy_with(source.replace(old, new), Path(tmp))
            outcome = run_suite(Path(tmp))
        killed += outcome == "fail"
        print(f"{'KILLED  ' if outcome == 'fail' else 'INVALID ' if outcome == 'invalid' else 'SURVIVED'} {name}")
    print(f"{killed}/{len(MUTANTS)} mutants killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
