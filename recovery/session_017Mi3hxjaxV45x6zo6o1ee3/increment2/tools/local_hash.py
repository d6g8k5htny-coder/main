"""Git blob ids + sha256 + size for every regular file in the indexed scratch set (same exclusions as checkpoint 1)."""
import os, hashlib, json, sys
ROOT = os.path.abspath(".")
EXCL_TOP = {"d365","mathf","mfrepo",".venv-audit","wtop","gate-run","gate-run2","pylib","mf15","dl_r17","ev331",
            "leanbuild","mut","pinout","iw_out","rec2","recovery","__pycache__",".git"}
rows = []
for dp, dns, fns in os.walk(ROOT):
    rel = os.path.relpath(dp, ROOT)
    top = rel.split(os.sep)[0] if rel != "." else ""
    if top in EXCL_TOP:
        dns[:] = []; continue
    dns[:] = [d for d in dns if not (rel == "." and d in EXCL_TOP) and d != "__pycache__" and d != ".git"]
    for fn in fns:
        p = os.path.join(dp, fn)
        if os.path.islink(p) or not os.path.isfile(p): continue
        data = open(p, "rb").read()
        h = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        rows.append({"path": os.path.relpath(p, ROOT), "size": len(data), "blob": h,
                     "sha256": hashlib.sha256(data).hexdigest(), "mtime": int(os.path.getmtime(p))})
json.dump(rows, open("rec2/local_files.json", "w"))
print("local files hashed:", len(rows), "bytes:", sum(r["size"] for r in rows))
