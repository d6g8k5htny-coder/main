"""Negative controls and single-point mutants for scripts/cross_check_zeros_arb.py at 90fa3bd7.

Usage: venv/bin/python -I audit_theirs.py TREE OUT.json
TREE is `git archive 90fa3bd78edcb6cfadfc583bcf21d87069d89b90` extracted. Each case runs in a
fresh copy of TREE.
A. Data controls: plant one defect in data/riemann_zeros/first_100_zeros.json (text edit of
   the stored digit strings only) and run the unmodified script on it. A control is caught
   when the summary reports a failure.
B. Script mutants: weaken one check in the script, keep the data and the committed record,
   and run the two TestZeroCrossCheck tests. A mutant is killed when a test fails.
"""
import decimal
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
PY = sys.executable
DATA = "data/riemann_zeros/first_100_zeros.json"
SCRIPT = "scripts/cross_check_zeros_arb.py"
text = (TREE / DATA).read_text()
m = re.search(r'("zeros":\s*\[)(.*?)(\])', text, re.S)
items = [s.strip() for s in m.group(2).split(",")]
assert len(items) == 100


def with_zeros(zs):
    return text[:m.start(2)] + "\n    " + ",\n    ".join(zs) + "\n  " + text[m.end(2):]


def bump(s, k):
    d = decimal.Decimal(s)
    return str(d + k * decimal.Decimal(10) ** (d.adjusted() - 19))


Z = list(items)
DATA_CASES = {
    "identity": Z,
    "n53_last_digit_plus1": Z[:52] + [bump(Z[52], 1)] + Z[53:],
    "n79_last_digit_plus1": Z[:78] + [bump(Z[78], 1)] + Z[79:],
    "swap_17_18": Z[:16] + [Z[17], Z[16]] + Z[18:],
    "dup_40": Z[:40] + [Z[39]] + Z[41:],
    "drop_77_append_101": Z[:76] + Z[77:] + ["237.76982048092520400"],
    "n99_to_12_decimals": Z[:98] + [str(round(float(Z[98]), 12))] + Z[99:],
}

SCRIPT_TEXT = (TREE / SCRIPT).read_text()
MUTANTS = {
    "M1_within_half_unit_always_true": ('"within_half_unit": bool(diff <= half_unit)', '"within_half_unit": True'),
    "M2_real_part_always_half": ('"real_part_is_half": bool(zero.real == flint.arb(0.5))', '"real_part_is_half": True'),
    "M3_no_bracket_mismatches": ('if r["nzeros"] != r["k"]]', 'if False]'),
    "M4_tolerance_x10": ("- SIG) / 2", "- SIG + 1) / 2"),
    "M5_compare_with_wrong_index": ("zip(strings, stored, enclosures)", "zip(strings, stored, enclosures[1:] + enclosures[:1])"),
}


def tree_copy(tmp):
    root = Path(tmp) / "t"
    shutil.copytree(TREE, root)
    return root


def run_data(name, zs):
    with tempfile.TemporaryDirectory() as tmp:
        root = tree_copy(tmp)
        (root / DATA).write_text(with_zeros(zs))
        out = Path(tmp) / "o.json"
        r = subprocess.run([PY, "-I", str(root / SCRIPT), str(out)], capture_output=True, text=True, timeout=600)
        if r.returncode:
            return {"case": name, "rc": r.returncode, "caught": True, "detail": r.stderr.strip().splitlines()[-1]}
        s = json.loads(out.read_text())["summary"]
        failed = [k for k, ok in (("within_half_unit", s["all_within_half_unit"]),
                                  ("real_part", s["all_real_parts_half"]),
                                  ("bracket_counts", not s["bracket_count_mismatches"])) if not ok]
        bad = [r["n"] for r in json.loads(out.read_text())["zeros"] if not r["within_half_unit"]]
        return {"case": name, "rc": 0, "caught": bool(failed), "failed_checks": failed,
                "not_within_half_unit": bad, "bracket_mismatches": s["bracket_count_mismatches"]}


def run_mutant(name, old, new):
    assert SCRIPT_TEXT.count(old) == 1, (name, old)
    with tempfile.TemporaryDirectory() as tmp:
        root = tree_copy(tmp)
        (root / SCRIPT).write_text(SCRIPT_TEXT.replace(old, new))
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_all.py",
                            "-k", "TestZeroCrossCheck"], cwd=root, capture_output=True, text=True, timeout=900)
        tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l][-1:]
        return {"mutant": name, "rc": r.returncode, "killed": r.returncode != 0, "pytest": " ".join(tail)}


res = {"data_controls": [run_data(k, v) for k, v in DATA_CASES.items()],
       "script_mutants": [run_mutant("M0_unmutated", "PREC = 128", "PREC = 128")]
       + [run_mutant(k, *v) for k, v in MUTANTS.items()]}
OUT.write_text(json.dumps(res, indent=1) + "\n")
for row in res["data_controls"] + res["script_mutants"]:
    print(json.dumps(row))
