import sys, zipfile, tarfile, hashlib, io, json, subprocess
# Builds C8's local evidence record for the main#384 push-run qualification (run 38489017622).
# Usage: python3 -I m384_evidence.py ARTDIR SRCDIR OUT.json
art, src, out = sys.argv[1:4]
def sha(b): return hashlib.sha256(b).hexdigest()
def gitblob(b): return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
rec = {"scope": "main#384 successor qualification of push run 38489017622 (C8, activation 6158794792)",
       "scientific_effect": "NONE", "fresh_acquisition_utc": "2026-10-14T01:26:14Z", "artifacts": {}, "members": {}}
for name in ("state.zip", "metadata.zip", "browser.zip", "github-pages.zip"):
    raw = open(f"{art}/{name}", "rb").read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    rec["artifacts"][name] = {"bytes": len(raw), "sha256": sha(raw), "crc_bad": z.testzip()}
    if name != "github-pages.zip":
        rec["members"][name] = {i.filename: {"bytes": i.file_size, "sha256": sha(z.read(i))} for i in z.infolist()}
z = zipfile.ZipFile(f"{art}/github-pages.zip")
tb = z.read("artifact.tar")
t = tarfile.open(fileobj=io.BytesIO(tb))
files = sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: m.name[2:] if m.name.startswith("./") else m.name)
tree = hashlib.sha256(b"current-site-generated-browser-input-v1\0"); total = 0; site = {}
for m in files:
    rel = m.name[2:] if m.name.startswith("./") else m.name
    data = t.extractfile(m).read(); n = rel.encode()
    tree.update(len(n).to_bytes(8, "big")); tree.update(n); tree.update(m.size.to_bytes(8, "big")); tree.update(data); total += m.size
    if rel in ("site/snapshots/index.json", "site/current-meta.json", "site/current.json"):
        site[rel] = {"bytes": len(data), "sha256": sha(data)}
rec["pages_tar"] = {"bytes": len(tb), "sha256": sha(tb), "regular_files": len(files), "file_bytes": total,
                    "assembled_docs_sha256_recomputed": tree.hexdigest(),
                    "unsafe_or_links": sum(1 for m in t.getmembers() if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in m.name.split("/")),
                    "site_files": site}
st = json.loads(zipfile.ZipFile(f"{art}/state.zip").read("current-site-prepared.json"))
prep = open(f"{src}/latest_7e90e748.json", "rb").read(); obs = open(f"{src}/latest_1736e449.json", "rb").read()
p, o = json.loads(prep), json.loads(obs)
rec["receipts"] = {"prepared": {"commit": "7e90e748885f1214d7746085751919914dea66ef", "latest_json_blob": gitblob(prep), "bytes": len(prep), "receipt": p},
                   "observed": {"commit": "1736e44905ae4b6aadf8df5c678d8fd29a94fc11", "latest_json_blob": gitblob(obs), "bytes": len(obs),
                                "differs_from_prepared_in": sorted(k for k in set(p) | set(o) if p.get(k) != o.get(k))},
                   "state_artifact_head": st["head"], "state_receipt_equals_prepared": st["receipt"] == p}
for f in ("current-site.yml", "current_site_publication.py", "current_site_history.py"):
    b = open(f"{src}/{f}", "rb").read(); rec.setdefault("source_blobs_at_8bc089f8", {})[f] = {"bytes": len(b), "git_blob": gitblob(b)}
json.dump(rec, open(out, "w"), indent=1, sort_keys=True); open(out, "a").write("\n")
print(json.dumps({k: rec[k] for k in ("artifacts", "pages_tar")}, indent=1)[:2500])
