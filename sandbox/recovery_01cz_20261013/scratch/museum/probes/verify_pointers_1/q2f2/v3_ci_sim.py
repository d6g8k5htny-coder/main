"""Q2-F2 verifier 2: generator-side reachability. Copy the cd32303 tree (minus __pycache__) to sim/<case>, apply one
edit, regenerate museum.json with the cd32303 generator, then run the exact 'Museum source projection and refusal
controls' step of .github/workflows/public-shop.yml (museum_check.py given --math-root = verified fixture instead of
network). Then validate the regenerated museum.json with the cd32303 browser guard. The extracted bytes are untouched."""
import os, shutil, subprocess, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; CD = (HERE / "../../../cd32303").resolve(); MATH = HERE / "fixture" / "Math-"
B = "https://github.com/d6g8k5htny-coder/main/issues/"
D1 = '"' + B + '63#issuecomment-5841830743"),'
D2 = '"' + B + '67#issuecomment-5841782206",'
CASES = {
  "control":        ("gen", None, None),
  "D1_issue6":      ("gen", D1, '"' + B + '6"),'),
  "D1_trunc":       ("gen", D1, '"' + B + '63#issuecomment-5841830"),'),
  "D1_issue99_ctl": ("gen", D1, '"' + B + '99"),'),
  "D2_bare67":      ("gen", D2, '"' + B + '67",'),
  "D2_trunc":       ("gen", D2, '"' + B + '67#issuecomment-584178220",'),
  "D2_issue6":      ("gen", D2, '"' + B + '6",'),
  "handedit_json_D1_issue6": ("json", None, None),
}
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).strip().splitlines()
    keep = [l for l in lines if l.startswith(("# pass", "# fail", "OK", "FAILED", "Museum", "ValueError", "not ok")) or "Error:" in l][-3:]
    return r.returncode, " | ".join(keep)
for name, (kind, old, new) in CASES.items():
    sim = HERE / "sim" / name
    if sim.exists(): shutil.rmtree(sim)
    shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
    out = []
    if kind == "gen":
        if old:
            p = sim / "tools/museum_data.py"; s = p.read_text(); assert s.count(old) == 1, old; p.write_text(s.replace(old, new))
        rc, msg = run(["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)], sim); out.append(f"regenerate exit={rc} {msg}")
        if rc: print(f"[{name}] " + " ;; ".join(out)); continue
    else:
        p = sim / "docs/site/museum.json"; m = json.loads(p.read_text()); m["claims"][11]["review"]["review_url"] = B + "6"
        p.write_text(json.dumps(m, indent=2) + "\n")
    for cmd in (["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
                ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)],
                ["node", "--test", "tests/test_museum_geometry.mjs"]):
        rc, msg = run(cmd, sim); out.append(f"{cmd[2] if cmd[2] != '-m' else 'unittest'} exit={rc} {msg}")
    m = json.loads((sim / "docs/site/museum.json").read_text())
    out.append("museum.json D1 review_url=" + m["claims"][11]["review"]["review_url"] + " D2 review_url=" + m["claims"][0]["review"]["review_url"])
    print(f"[{name}] " + " ;; ".join(out))
