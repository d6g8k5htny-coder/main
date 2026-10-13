"""End-to-end path for the unbound display fields: an edit of the generator (tools/museum_data.py) that drops the
open-obligation availability label and rewrites the AMEND replay notice is regenerated into museum.json, every
museum CI step stays green (museum_check compares museum.json with that same generator), and the page renders the
edit as verified. Scratch copy only; offline (Math- sources from the byte-verified fixture).
usage: python3 -I p6_generator_path_ci_green.py"""
import os, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; SRC = (HERE / "../../cd32303").resolve(); W = HERE / "work/p6"
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"]); env.pop("MUSEUM_TREE", None)
if W.exists(): shutil.rmtree(W)
shutil.copytree(SRC, W, ignore=shutil.ignore_patterns("__pycache__"))
f = W / "tools/museum_data.py"; s = f.read_text(encoding="utf-8")
edits = [('            proof["availability"] = "NO COMPLETE PROOF YET"\n', '            pass\n'),
         ('"notice": "OPEN / NOT LANDED. No replay is presented as acceptance."', '"notice": "Replay reproduces the closed result."')]
for old, new in edits: assert s.count(old) == 1, old; s = s.replace(old, new)
f.write_text(s, encoding="utf-8")
r = subprocess.run(["python3", "-B", "tools/museum_data.py", "--math-root", str(HERE / "fixture/Math-")], cwd=W, env=env, capture_output=True, text=True)
print("regenerate museum.json:", r.returncode, (r.stdout + r.stderr).strip())
for name, cmd in [("S1 test_museum_data", ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"]),
                  ("S2 museum_check", ["python3", "-B", "tools/museum_check.py", "--math-root", str(HERE / "fixture/Math-")]),
                  ("S3 test_museum_geometry", ["node", "--test", "tests/test_museum_geometry.mjs"])]:
    p = subprocess.run(cmd, cwd=W, env=env, capture_output=True, text=True)
    print(f"{name}: exit {p.returncode}", ((p.stdout + p.stderr).strip().splitlines() or [""])[-1])
js = r"""
import {render, card, links, realManifest} from './lib.mjs';
const d = await render(realManifest());
console.log('museum-state:', d.getElementById('museum-state').textContent);
for (const id of ['d5-pin-neighborhoods-open','d1-parent-selection-open','sard-g-a1-a6-open']) { const c = card(d, id);
  const rep = c.walk().find(n => n.className === 'claim-column' && n.children[0]?.textContent === 'Replay');
  console.log(id, '| class', c.attributes['data-object-class'], '|', JSON.stringify(links(c)[0]), '| NO COMPLETE PROOF YET:', c.textContent.includes('NO COMPLETE PROOF YET'), '| Replay:', JSON.stringify(rep.textContent)); }
"""
(HERE / "work/p6_render.mjs").write_text(js.replace("./lib.mjs", "../lib.mjs"))
p = subprocess.run(["node", str(HERE / "work/p6_render.mjs")], cwd=HERE, env=dict(env, MUSEUM_TREE=str(W)), capture_output=True, text=True)
print(p.stdout.strip() or p.stderr.strip())
