# Q1/P6 (builder half): the three AMEND rows are zipped POSITIONALLY with hard-coded open_specs
# (museum_data.py:430-450). Unlike the reviewed bullets, no row is checked to contain its proof path or
# review pointer. Simulate the routine maintenance path: a successor STATUS.md in which the D5
# pin-neighborhood row has been replaced by a different open obligation, re-pinned in SOURCES.
import copy, hashlib, json, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_0/q1_6")
from builder import museum, INDEX, STATUS, HERE
old = STATUS.decode()
d5_row = next(l for l in old.splitlines() if l.startswith("| **D5 — pin neighborhoods"))
new_row = ("| **D5 — shrinking multiple-witness collision** | HYPOTHETICAL successor row: the shrinking-separation "
           "factorial-moment/collision estimate is not supplied. | [Math proof index](https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md) |")
new = old.replace(d5_row, new_row).encode()
(HERE / "p6_STATUS_successor.md").write_bytes(new)
NEW_COMMIT = "5" * 40
for path, ident in museum.SOURCES.items():          # re-pin every main source at the successor commit
    if ident["repository"] == "d6g8k5htny-coder/main":
        ident["commit"] = NEW_COMMIT
museum.SOURCES["STATUS.md"].update(bytes=len(new), sha256=hashlib.sha256(new).hexdigest(),
    blob=hashlib.sha1(b"blob %d\0" % len(new) + new).hexdigest())
museum.STATUS = museum.SOURCES["STATUS.md"]
claims = museum.project(INDEX, new)
c = claims[12]
print("builder ACCEPTED successor STATUS:")
print("  id          :", c["id"])
print("  title       :", c["title"])
print("  status_quote:", c["status_quote"][:80], "...")
print("  proof       :", c["proof"]["path"], "| availability:", c["proof"].get("availability"))
print("  review      :", c["review"]["review_url"])
manifest = json.loads((HERE.parent.parent.parent / "cd32303/docs/site/museum.json").read_text())
manifest.update(claims=claims, status_source=museum.descriptor("STATUS.md"), packets=museum.packet_descriptors())
(HERE / "p6_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
