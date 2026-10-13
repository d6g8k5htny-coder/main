"""Verifier 1 / Q1-3, builder side. Read-only against cd32303 bytes; no network (urlopen stubbed).

Loads a COPY of cd32303 tools/museum_data.py with one review-target constant changed, and runs
project() on the pinned PROOF_INDEX/STATUS bytes. Shows (a) the builder's delimited check refuses a
prefix URL on a reviewed bullet (finding's comparison holds), and (b) the builder runs NO check on
the AMEND D1 review target, so in CI (museum_check.py: build -> check_export -> actual-museum
frontend test) the browser's substring test is the only guard on that pointer.
"""
import hashlib, importlib.util, json, sys, types
from pathlib import Path

S = Path("<SCRATCH>/museum")
T, Q = S / "cd32303", S / "probes/verify_substitution_0/q1_3"
src = (T / "tools/museum_data.py").read_text(encoding="utf-8")
print("cd32303 tools/museum_data.py sha256", hashlib.sha256(src.encode()).hexdigest()[:16])
index = (S / "probes/verify_substitution_0/math/PROOF_INDEX.md").read_bytes()
status = (T / "STATUS.md").read_bytes()
ISS = "https://github.com/d6g8k5htny-coder/main/issues/"


def load(text, name):
    p = Q / f"builder_{name}.py"
    p.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"md_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
    return mod


def run(label, text, name, claim_id):
    mod = load(text, name)
    try:
        claims = mod.project(index, status)
        c = next(c for c in claims if c["id"] == claim_id)
        print(f"{label}: builder ACCEPTS, {claim_id} review_url = {c['review']['review_url']}")
        return claims
    except Exception as e:  # noqa: BLE001
        print(f"{label}: builder REFUSES ({e})")


run("unmodified builder", src, "orig", "d1-parent-selection-open")
d3 = ISS + "65#issuecomment-5841269490"
assert src.count(d3) == 1
run("builder copy, D3 target -> issues/6", src.replace(d3, ISS + "6"), "d3", "d3-side24-coefficient")
d1 = ISS + "63#issuecomment-5841830743"
assert src.count(d1) == 1
claims = run("builder copy, AMEND D1 target -> issues/6", src.replace(d1, ISS + "6"), "d1", "d1-parent-selection-open")
claims2 = run("builder copy, AMEND D1 target -> issues/9999 (not in row at all)", src.replace(d1, ISS + "9999"), "d1x", "d1-parent-selection-open")
# Hand the builder's own D1 output to the frontend guard (cd32303 museum.mjs) inside the shipped manifest.
m = json.loads((T / "docs/site/museum.json").read_text(encoding="utf-8"))
for tag, cl in (("issues6", claims), ("issues9999", claims2)):
    m2 = json.loads(json.dumps(m))
    m2["claims"][11] = next(c for c in cl if c["id"] == "d1-parent-selection-open")
    (Q / f"manifest_d1_{tag}.json").write_text(json.dumps(m2), encoding="utf-8")
print("wrote manifest_d1_issues6.json and manifest_d1_issues9999.json for v_builder_frontend.mjs")
