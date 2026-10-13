# usage: add_rev.py <manifest> <json entry>   (appends to "revisions", preserving the manifest's JSON format)
import json, sys, pathlib
mp = pathlib.Path(sys.argv[1]); raw = mp.read_text(); man = json.loads(raw)
fmt = None
for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2, ensure_ascii=False, sort_keys=True), dict(indent=2), dict(indent=2, sort_keys=True)):
    if json.dumps(man, **kw) + "\n" == raw: fmt = kw
assert fmt, "manifest format"
entry = json.loads(sys.argv[2])
revs = man.setdefault("revisions", [])
assert all(r.get("version") != entry["version"] for r in revs), "duplicate version"
revs.append(entry)
mp.write_text(json.dumps(man, **fmt) + "\n")
print("ok", len(revs))
