"""Verifier 1 probe for Q2-F2 (generator + CI half). Copies the cd32303 tree into ./sim/<variant>, applies one textual
edit to tools/museum_data.py, regenerates docs/site/museum.json WITH THE GENERATOR, then runs the three museum CI
commands of .github/workflows/public-shop.yml at cd32303 (museum_check gets --math-root = byte-verified local Math-
fixture; read_source() re-verifies size, git blob and SHA-256, so no network is used). Then renders the regenerated
manifest through the cd32303 museum.mjs (v3_render.mjs). The extracted cd32303 bytes are never modified."""
import os, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../cd32303").resolve()
MATH = (HERE / "../pointers/fixture/Math-").resolve()   # read-only use; bytes re-verified by read_source/verify_bytes
D1 = '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),'
D2 = '"https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206",'
VARIANTS = {
  "control": [],
  "d1_issue6": [(D1, '"https://github.com/d6g8k5htny-coder/main/issues/6"),')],
  "d1_truncated_comment": [(D1, '"https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-584183074"),')],
  "d1_arbitrary_control": [(D1, '"https://github.com/d6g8k5htny-coder/main/issues/99#issuecomment-1"),')],
  "d2_issue67_open_link": [(D2, '"https://github.com/d6g8k5htny-coder/main/issues/67",')],
  "d2_truncated_control": [(D2, '"https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-584178220",')],
}
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
for name in sys.argv[1:] or VARIANTS:
    sim = HERE / "sim" / name
    if sim.exists(): shutil.rmtree(sim)
    shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
    p = sim / "tools/museum_data.py"; src = p.read_text()
    for old, new in VARIANTS[name]:
        assert src.count(old) == 1, old; src = src.replace(old, new)
    p.write_text(src)
    steps = [["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)],
             ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
             ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)],
             ["node", "--test", "tests/test_museum_geometry.mjs"]]
    codes = []
    for cmd in steps:
        r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).splitlines()
        keep = [l.strip() for l in lines if l.startswith(("# pass", "# fail", "OK", "FAILED", "Ran ", "Museum", "ValueError")) or "not ok" in l][-4:]
        codes.append(r.returncode)
        print(f"[{name}] exit={r.returncode} :: {' '.join(cmd[1:4])} :: {' | '.join(keep)}")
    print(f"[{name}] ALL-CI-GREEN={all(c == 0 for c in codes)}")
