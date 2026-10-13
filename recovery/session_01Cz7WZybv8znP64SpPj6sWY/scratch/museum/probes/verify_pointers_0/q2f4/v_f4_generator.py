"""Q2-F4 verifier 1: does the cd32303 GENERATOR (tools/museum_data.py, whose output museum_check.py byte-compares
with docs/site/museum.json) refuse the same review-identity substitutions? Imports the cd32303 module read-only from
its extracted path; for the open_specs edit, compiles a modified COPY in memory (the extracted file is not touched).
Feeds project() the hashlib-verified fixture bytes. No network, no git writes."""
import base64, copy, json, sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../../cd32303").resolve()
sys.path.insert(0, str(CD / "tools"))
import museum_data as md
FX = json.loads((HERE / "fixture.json").read_text())
def raw(path): return base64.b64decode(FX["sources"][md.descriptor(path)["url"]])
IDX, ST = raw("PROOF_INDEX.md"), raw("STATUS.md")
committed = json.loads((CD / "docs/site/museum.json").read_text())
def attempt(label, fn):
    try: print(label, "->", fn())
    except Exception as e: print(label, "-> REFUSED:", type(e).__name__, e)
# baseline
claims = md.project(IDX, ST); print("[baseline] project() == committed claims:", claims == committed["claims"])
# (a) open_specs d5-pin review target 'PROOF_INDEX.md' (non-https -> descriptor(), no row check for open reviews)
src = (CD / "tools/museum_data.py").read_text()
old = '"https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md"),'
assert src.count(old) == 1
mod = types.ModuleType("museum_data_a"); exec(compile(src.replace(old, '"PROOF_INDEX.md"),'), "museum_data_a", "exec"), mod.__dict__)
def a():
    c = mod.project(IDX, ST)[12]; r = c["review"]
    expected = {k: v for k, v in committed["claims"][12]["review"].items() if k not in ("pointer_only", "review_url", "review_notice")}
    return f"ACCEPTED; d5-pin review pointer_only={r.get('pointer_only')}, path={r['path']}, equals JS probe (a1) descriptor: {r == expected}"
attempt("(a) generator edit: d5-pin open_specs review_target := 'PROOF_INDEX.md'", a)
# (b) SOURCES review entry at another Math- commit (real commit bebb77ae; same blob there)
def b():
    s = copy.deepcopy(md.SOURCES); p = "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"; s[p]["commit"] = "bebb77ae9fe3b22c425a979b780fc99a72c8e761"
    orig = md.SOURCES; md.SOURCES = s
    try:
        md.verify_bytes(raw(p) if False else base64.b64decode(FX["sources"]["https://raw.githubusercontent.com/d6g8k5htny-coder/Math-/d6628da09384728992dcbe6e921cc28ba85aebb0/" + p]), s[p])
        r = md.project(IDX, ST)[3]["review"]; return f"ACCEPTED; review commit={r['commit'][:8]} while INDEX commit={md.INDEX['commit'][:8]}"
    finally: md.SOURCES = orig
attempt("(b1) generator edit: SOURCES[pr28 REVIEW.md].commit := bebb77ae", b)
def b2():
    s = copy.deepcopy(md.SOURCES["reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"]); s["repository"] = "d6g8k5htny-coder/query-"
    return md.verify_bytes(b"x", s)
attempt("(b2/control) generator: repository query- in a SOURCES identity (verify_bytes, run for every source by build())", b2)
# (c) bare REVIEW.md
def c():
    specs = copy.deepcopy(md.CLAIM_SPECS); specs[3][2] = "REVIEW.md"; orig = md.CLAIM_SPECS; md.CLAIM_SPECS = specs
    try: return md.project(IDX, ST)
    finally: md.CLAIM_SPECS = orig
attempt("(c/control) generator: CLAIM_SPECS d5-all-height review := 'REVIEW.md'", c)
# (d) proof as review
def d():
    specs = copy.deepcopy(md.CLAIM_SPECS); specs[0][2] = specs[0][1]; orig = md.CLAIM_SPECS; md.CLAIM_SPECS = specs
    try: r = md.project(IDX, ST)[0]["review"]; return f"ACCEPTED; d2 review path={r['path']} pointer_only={r.get('pointer_only')}"
    finally: md.CLAIM_SPECS = orig
attempt("(d) generator edit: CLAIM_SPECS d2 review := d2 proof path", d)
