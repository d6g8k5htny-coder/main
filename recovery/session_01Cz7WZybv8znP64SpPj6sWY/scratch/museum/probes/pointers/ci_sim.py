"""Copy the cd32303 tree into sim/<name>, apply one textual generator edit, regenerate museum.json
with the generator itself, then run the exact museum CI commands from .github/workflows/public-shop.yml
(museum_check.py with --math-root pointing at the byte-verified fixture instead of the network).
Prints each command's exit status. Never touches the extracted cd32303 bytes."""
import os, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../cd32303").resolve()
MATH = HERE / "fixture" / "Math-"
PATCHES = {
  "control": [],
  "F1_review_url_on_byte_verified": [(
    'else descriptor(review_path))\n        counterexample',
    'else descriptor(review_path, **({"review_url": "https://example.invalid/forged-review"} if claim_id == "d5-all-height-annulus" else {})))\n        counterexample')],
  "F3_notice_claims_byte_frozen": [(
    'review_notice="The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor."',
    'review_notice="Byte-frozen review: the linked comment is sealed by the SHA-256 shown below."')],
  "F2_d1_pointer_prefix": [(
    '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),',
    '"https://github.com/d6g8k5htny-coder/main/issues/6"),')],
}
name = sys.argv[1]
sim = HERE / "sim" / name
if sim.exists(): shutil.rmtree(sim)
shutil.copytree(CD, sim)
src = (sim / "tools/museum_data.py").read_text()
for old, new in PATCHES[name]:
    assert src.count(old) == 1, ("patch anchor not unique", old); src = src.replace(old, new)
(sim / "tools/museum_data.py").write_text(src)
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
steps = [
  ["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)],                 # regenerate (generator itself)
  ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
  ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)],
  ["node", "--test", "tests/test_museum_geometry.mjs"],
]
for cmd in steps:
    r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    summary = [l for l in tail if l.startswith(("# pass", "# fail", "OK", "FAILED", "Ran ", "Museum")) or "Error" in l][-4:]
    print(f"[{name}] exit={r.returncode} :: {' '.join(cmd[2:5]) if cmd[0]=='python3' else ' '.join(cmd)} :: {' | '.join(summary)}")
