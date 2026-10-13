"""For mutants that survived every museum CI step (out_mutants.txt), show the page they ship: a scratch copy of
docs/site with exactly that guard removed renders a tampered manifest as verified. usage: python3 -I p3_mutant_impact.py"""
import os, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent; SRC = (HERE / "../../cd32303").resolve()
import importlib.util
spec = importlib.util.spec_from_file_location("mutants", HERE / "mutants.py"); mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
by_id = {m[0]: m for m in mt.MUTANTS}
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
def tree_for(mid):
    t = HERE / "work" / ("impact_" + mid)
    if t.exists(): shutil.rmtree(t)
    shutil.copytree(SRC / "docs/site", t / "docs/site")
    _, rel, old, new, _ = by_id[mid]; f = t / rel; s = f.read_text(encoding="utf-8"); assert s.count(old) == 1; f.write_text(s.replace(old, new), encoding="utf-8")
    return t
for tamper, mid in [("promote_amend", "M31"), ("swap_review", "M02"), ("bad_proof_digest", "M33"), ("bad_proof_digest", "M21")]:
    print(f"### tamper={tamper}  mutant={mid}: {by_id[mid][4]}")
    for tree in (None, tree_for(mid)):
        e = dict(env); e.pop("MUSEUM_TREE", None)
        if tree: e["MUSEUM_TREE"] = str(tree)
        p = subprocess.run(["node", "p3_render_tampered.mjs", tamper], cwd=HERE, env=e, capture_output=True, text=True)
        print(p.stdout.rstrip() or p.stderr.rstrip())
    shutil.rmtree(HERE / "work" / ("impact_" + mid))
