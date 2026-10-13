"""Q2-F2 verifier 2: how far does project()'s '](' + review_path + ')' in bullet test reach through CI?
Case A: D2 review slot <- the R1-R4 comment (another genuine review link in the same bullet).
Case B: D5 height-window review slot <- its own PROOF path (a link target in the bullet that is the proof, not a review).
Each case: patch CLAIM_SPECS in a sim copy, regenerate, run the CI museum step, and render-check with the browser guard."""
import os, shutil, subprocess, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; CD = (HERE / "../../../cd32303").resolve(); MATH = HERE / "fixture" / "Math-"
B = "https://github.com/d6g8k5htny-coder/main/issues/"
CASES = {
 "A_D2_R1R4": ('"' + B + '67#issuecomment-5841782206",', '"' + B + '67#issuecomment-5841270276",'),
 "B_D5hw_proof_as_review": ('"reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md",\n    null,', '"frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md",\n    null,'),
}
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
for name, (old, new) in CASES.items():
    sim = HERE / "sim" / name
    if sim.exists(): shutil.rmtree(sim)
    shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
    p = sim / "tools/museum_data.py"; s = p.read_text(); assert s.count(old) == 1, old; p.write_text(s.replace(old, new))
    res = []
    for cmd in (["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)],
                ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
                ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)],
                ["node", "--test", "tests/test_museum_geometry.mjs"]):
        r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True); res.append(str(r.returncode))
    m = json.loads((sim / "docs/site/museum.json").read_text())
    c = m["claims"][0 if name.startswith("A") else 4]
    print(f"[{name}] exits(regen,unittest,museum_check,geometry)={','.join(res)} ; {c['id']} review="
          f"{c['review'].get('review_url') or c['review']['path']} pointer_only={c['review'].get('pointer_only')} ; proof={c['proof']['path']}")
