"""Control: a hand edit of museum.json that the generator does not produce IS caught by museum_check (export differs).
So F1-F3 matter on the generator path (CI-green, sim/ runs) and for the browser guard as runtime defence."""
import json, os, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; CD = (HERE / "../../cd32303").resolve(); sim = HERE / "sim" / "handedit_control"
if sim.exists(): shutil.rmtree(sim)
shutil.copytree(CD, sim)
p = sim / "docs/site/museum.json"; d = json.loads(p.read_text())
d["claims"][3]["review"]["review_url"] = "https://example.invalid/forged-review"
p.write_text(json.dumps(d, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
r = subprocess.run(["python3", "-B", "tools/museum_check.py", "--math-root", str(HERE / "fixture/Math-")], cwd=sim, capture_output=True, text=True,
                   env=dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"]))
print("museum_check exit:", r.returncode, "|", (r.stdout + r.stderr).strip().splitlines()[-1])
