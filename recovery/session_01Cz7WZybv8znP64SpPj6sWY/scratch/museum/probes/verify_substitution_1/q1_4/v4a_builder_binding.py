# Verifier probe for Q1-4 (read-only; no network: urlopen is stubbed).
# Question: does the Python side (museum_data.build + museum_check byte-exact regeneration)
# already bind exhibits.{remote,annulus,p15} to the D4/D5/D6 claim proofs, so that the browser
# gap in validateBoundManifest has no realistic producer?
import json, sys, pathlib
TREE = pathlib.Path("<SCRATCH>/museum/cd32303")
sys.path.insert(0, str(TREE / "tools"))
import urllib.request
def _blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED in probe")
urllib.request.urlopen = _blocked
import museum_data
museum_data.urlopen = _blocked

raw = (TREE / "docs/site/museum.json").read_bytes()
m = json.loads(raw)
KEYS = ["repository", "path", "commit", "blob", "bytes", "sha256", "url", "html_url"]
same = lambda a, b: all(a.get(k) == b.get(k) for k in KEYS)
pairs = {"remote": 2, "annulus": 3, "p15": 10, "lifetime": 0}
for key, i in pairs.items():
    print(f"shipped exhibits.{key} == claims[{i}] ({m['claims'][i]['id']}).proof identity:", same(m["exhibits"][key], m["claims"][i]["proof"]))
print("shipped exhibits.annulus == claims[4] (d5-height-window-annulus).proof identity:", same(m["exhibits"]["annulus"], m["claims"][4]["proof"]),
      "| height-window proof path:", m["claims"][4]["proof"]["path"])
# The builder takes every identity from SOURCES, a dict keyed by path: one identity per path.
src = (TREE / "tools/museum_data.py").read_text()
print("SOURCES keyed by path (one identity per path):", all(k == v["path"] for k, v in museum_data.SOURCES.items()))
for key, path in [("remote", "frontiers/remote_window_20260924/PROOF.md"), ("annulus", "frontiers/rn_annulus_bridge_20260925/PROOF.md"), ("p15", "frontiers/full_price_20260924/PROOF.md")]:
    d = museum_data.descriptor(path)
    print(f"builder descriptor({key} path) == shipped exhibit identity:", same(d, m["exhibits"][key]))
# Canonical export: check_export compares bytes with dump(build(...)). A hand edit of the exhibit commit
# can only pass CI if build() itself emits it.
print("shipped museum.json is canonical dump():", museum_data.dump(m) == raw)
t = json.loads(raw); t["exhibits"]["annulus"]["commit"] = "4" * 40
print("hand-edited exhibit commit equals canonical export:", museum_data.dump(t) == raw)
print("CI step runs museum_check.py:", "python3 -B tools/museum_check.py" in (TREE / ".github/workflows/public-shop.yml").read_text())
# Is there any invariant (Python or JS test) asserting exhibit == claim proof for remote/annulus/p15?
for f in ["tests/test_museum_data.py", "tests/test_museum_frontend.mjs", "tests/test_museum_geometry.mjs", "tools/museum_check.py", "tools/museum_data.py"]:
    text = (TREE / f).read_text()
    hits = [ln.strip()[:90] for ln in text.splitlines() if "exhibits" in ln and ("remote" in ln or "annulus" in ln or "p15" in ln) and ("claims" in ln or "proof" in ln)]
    print(f"{f}: lines tying a remote/annulus/p15 exhibit to a claim proof: {len(hits)}", hits[:2])
