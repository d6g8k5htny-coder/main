import json, sys, subprocess, zipfile, io, hashlib
work, repo, ref = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(work + "/objects.json"))
oid = "a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b"
o = d["objects"][oid]
base = "drive/deltas/2026-09-19/DG-MIGRATION-20260919/"
cache = {}
buf = bytearray()
for c in o["chunks"]:
    p = d["chunks"][c]["pack"]
    if p not in cache:
        raw = subprocess.run(["git", "-C", repo, "show", ref + ":" + base + p], capture_output=True, check=True).stdout
        cache[p] = zipfile.ZipFile(io.BytesIO(raw))
    b = cache[p].read(c)
    assert hashlib.sha256(b).hexdigest() == c, "chunk hash"
    buf += b
assert len(buf) == o["bytes"] and hashlib.sha256(buf).hexdigest() == oid, "object hash"
print("rebuilt", len(buf), "sha256 ok")
z = zipfile.ZipFile(io.BytesIO(bytes(buf)))
names = [n for n in z.namelist() if n.endswith("H3_closure/h3_rung_floor.py") or n.endswith("H3_closure/h3_band_ceil.py") or n.endswith("H3_closure/MANIFEST") or n.endswith("H3_closure/MANIFEST.txt")]
for n in names:
    b = z.read(n); print(n, len(b), hashlib.sha256(b).hexdigest())
    if n.endswith(".py"):
        open(work + "/" + n.split("/")[-1] + ".txt", "wb").write(b)
