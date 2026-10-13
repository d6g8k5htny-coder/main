"""Verifier 1 for Q2-F1, CI side. Independent of probes/pointers/ci_sim.py.

Copies the cd32303 tree (without __pycache__) to sim/<case>, applies one generator edit to
tools/museum_data.py, regenerates docs/site/museum.json WITH THE GENERATOR, then runs the exact
three commands of the "Museum source projection and refusal controls" step of
.github/workflows/public-shop.yml at cd32303. Only deviation: museum_check.py gets --math-root
pointing at the independently byte-verified fixture (CI would fetch the same pinned raw URLs).
Network is never used. The extracted cd32303 bytes are never modified.
"""
import json, os, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CD = (HERE / "../../cd32303").resolve()
MATH = HERE / "fixture" / "Math-"
ANCHOR = 'else descriptor(review_path))\n        counterexample'
CASES = {
    "control": None,
    # Cross-claim: an immutable, pinned, real review URL -- of a DIFFERENT claim (D6 P15).
    "cross_claim_review_url": ('else descriptor(review_path, **({"review_url": descriptor("reviews/p15_full_price_nonauthor_20260926/REVIEW.md")["html_url"]} '
                               'if claim_id == "d5-all-height-annulus" else {})))\n        counterexample'),
    # Mutable issue comment on a non-pointer ("Byte-verified review") descriptor.
    "mutable_comment_review_url": ('else descriptor(review_path, **({"review_url": "https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172"} '
                                   'if claim_id == "d5-all-height-annulus" else {})))\n        counterexample'),
}
case = sys.argv[1]
sim = HERE / "sim" / case
if sim.exists():
    shutil.rmtree(sim)
shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
if CASES[case]:
    p = sim / "tools/museum_data.py"; s = p.read_text(encoding="utf-8")
    assert s.count(ANCHOR) == 1
    p.write_text(s.replace(ANCHOR, CASES[case]), encoding="utf-8")
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"], PYTHONDONTWRITEBYTECODE="1")
env.pop("MUSEUM_MATH_ROOT", None); env.pop("MUSEUM_FIXTURE", None)
print(f"[{case}] sim is inside a git work tree? ",
      subprocess.run(["git", "-C", str(sim), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True).stdout.strip() or "no")
steps = [
    ("regenerate (generator)", ["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)]),
    ("workflow 1", ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"]),
    ("workflow 2", ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)]),
    ("workflow 3", ["node", "--test", "tests/test_museum_geometry.mjs"]),
]
for name, cmd in steps:
    r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).splitlines()
    keep = [l.strip() for l in lines if re.match(r"\s*(# (tests|pass|fail|skipped)|OK|FAILED|Ran |Museum|not ok)", l)
            or "actual pinned museum" in l or "Error" in l]
    print(f"[{case}] {name} exit={r.returncode} :: " + " | ".join(keep))
m = json.loads((sim / "docs/site/museum.json").read_text(encoding="utf-8"))
r3 = m["claims"][3]["review"]
print(f"[{case}] regenerated claims[3].review: pointer_only={r3.get('pointer_only')!r} path={r3['path']} review_url={r3.get('review_url')!r}")
