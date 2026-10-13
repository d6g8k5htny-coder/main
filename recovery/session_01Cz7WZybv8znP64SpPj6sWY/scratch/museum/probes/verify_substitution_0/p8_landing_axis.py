# Q1/P8: claims/LANDING_CLAIMS.json is pinned in museum_data.SOURCES (so build() fetches and
# byte-verifies it) but nothing consumes it. For the same proof blobs it records a different
# disposition axis than the museum's constant "Source label: ACCEPT".
import json, pathlib, re, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/substitution")
from builder import museum, HERE, TREE
src = (TREE / "tools/museum_data.py").read_text()
uses = [n for n, line in enumerate(src.splitlines(), 1) if "LANDING_CLAIMS" in line]
print("museum_data.py lines mentioning LANDING_CLAIMS:", uses, "(all inside the SOURCES literal)" if all(u < 259 for u in uses) else "")
landing = json.loads((HERE / "fixture/Math-/claims/LANDING_CLAIMS.json").read_bytes())
by_blob = {c["source"]["blob"]: c for c in landing["claims"]}
shipped = json.loads((TREE / "docs/site/museum.json").read_text())
for c in shipped["claims"]:
    l = by_blob.get(c["proof"]["blob"])
    if l: print(f"{c['id']:30} museum: class={c['class']:15} label={c['source_label']:20} | LANDING_CLAIMS {l['claim_id']}: {l['disposition']} — {l['disposition_reason'][:90]}")
print("LANDING_CLAIMS meaning:", landing["meaning"])
