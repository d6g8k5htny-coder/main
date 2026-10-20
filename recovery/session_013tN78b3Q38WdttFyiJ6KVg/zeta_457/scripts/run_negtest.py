import json, shutil, subprocess, sys, tempfile
from pathlib import Path
TREE = Path(sys.argv[1]).resolve(); PY = sys.argv[2]; SNIP = Path(sys.argv[3]).read_text()
SCRIPT = "scripts/cross_check_zeros_arb.py"; TEST = "tests/test_all.py"
src = (TREE / SCRIPT).read_text()
M = {
 "M0": [],
 "M1": [('"within_half_unit": bool(diff <= half_unit)', '"within_half_unit": True')],
 "M2": [('"real_part_is_half": bool(zero.real == flint.arb(0.5))', '"real_part_is_half": True')],
 "M3": [('if r["nzeros"] != r["k"]]', 'if False]')],
 "M4": [("- SIG) / 2", "- SIG + 1) / 2")],
 "M5": [("zip(strings, stored, enclosures)", "zip(strings, stored, enclosures[1:] + enclosures[:1])")],
 "M3b_no_nzeros": [('counts = [b.zeta_nzeros() for b in brackets]', 'counts = [flint.arb(k) for k, b in enumerate(brackets)]')],
 "M1b_mid": [('bool(diff <= half_unit)', 'bool(diff.mid() <= half_unit.mid() * 10)')],
}
for name, reps in M.items():
    t = src
    for old, new in reps:
        assert t.count(old) == 1; t = t.replace(old, new)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "t"; shutil.copytree(TREE, root)
        (root / SCRIPT).write_text(t)
        (root / TEST).write_text((root / TEST).read_text() + SNIP)
        r = subprocess.run([PY, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", TEST, "-k", "TestZeroCrossCheck"],
                           cwd=root, capture_output=True, text=True, timeout=900)
        per = [l.split(" - ")[0] for l in r.stdout.splitlines() if l.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED"))]
        datasame = (root/"data/riemann_zeros/first_100_zeros.json").read_bytes() == (TREE/"data/riemann_zeros/first_100_zeros.json").read_bytes()
    print(name, "rc", r.returncode, r.stdout.strip().splitlines()[-1], "| data untouched:", datasame)
    for p in per: print("   ", p)
