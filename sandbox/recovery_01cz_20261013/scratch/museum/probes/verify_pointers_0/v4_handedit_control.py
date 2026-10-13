"""Verifier 1 for Q2-F1: control. A hand edit of docs/site/museum.json alone (generator unchanged)
adding review_url to claims[3].review IS refused by tools/museum_check.py (export differs).
Copies cd32303 (no __pycache__) to sim/handedit; never touches the extracted bytes; no network."""
import json, os, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; CD = (HERE / "../../cd32303").resolve(); sim = HERE / "sim" / "handedit"
if sim.exists(): shutil.rmtree(sim)
shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
p = sim / "docs/site/museum.json"; d = json.loads(p.read_text(encoding="utf-8"))
d["claims"][3]["review"]["review_url"] = "https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172"
p.write_bytes((json.dumps(d, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
r = subprocess.run(["python3", "-B", "tools/museum_check.py", "--math-root", str(HERE / "fixture/Math-")], cwd=sim,
                   capture_output=True, text=True, env=dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"], PYTHONDONTWRITEBYTECODE="1"))
print("hand-edit control: museum_check exit=%d | %s" % (r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1].replace(str(HERE), "<v>")))
