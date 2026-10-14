import sys, zipfile, tarfile, hashlib, io, json
# Generic publication-binding check for one current-site.yml run (read-only, no extraction to disk).
# Usage: python3 -I run_bind.py ARTDIR PREPARED_LATEST.json OBSERVED_LATEST.json OUT.json
art, prep_path, obs_path, out = sys.argv[1:5]
def sha(b): return hashlib.sha256(b).hexdigest()
rec = {"scientific_effect": "NONE", "artifacts": {}, "members": {}}
for name in ("state.zip", "metadata.zip", "browser.zip", "github-pages.zip"):
    raw = open(f"{art}/{name}", "rb").read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    rec["artifacts"][name] = {"bytes": len(raw), "sha256": sha(raw), "crc_bad": z.testzip(),
                              "unsafe": [i.filename for i in z.infolist() if i.filename.startswith("/") or ".." in i.filename.split("/")]}
    if name != "github-pages.zip":
        rec["members"][name] = {i.filename: {"bytes": i.file_size, "sha256": sha(z.read(i))} for i in z.infolist()}
z = zipfile.ZipFile(f"{art}/github-pages.zip")
names = z.namelist()
tb = z.read("artifact.tar")
t = tarfile.open(fileobj=io.BytesIO(tb))
rel = lambda n: n[2:] if n.startswith("./") else n
files = sorted((m for m in t.getmembers() if m.isfile()), key=lambda m: rel(m.name))
tree = hashlib.sha256(b"current-site-generated-browser-input-v1\0"); total = 0; site = {}
for m in files:
    data = t.extractfile(m).read(); n = rel(m.name).encode()
    tree.update(len(n).to_bytes(8, "big")); tree.update(n); tree.update(m.size.to_bytes(8, "big")); tree.update(data); total += m.size
    if rel(m.name) in ("site/snapshots/index.json", "site/current-meta.json", "site/current.json"):
        site[rel(m.name)] = {"bytes": len(data), "sha256": sha(data)}
rec["pages_tar"] = {"zip_members": names, "bytes": len(tb), "sha256": sha(tb), "regular_files": len(files), "file_bytes": total,
                    "assembled_docs_sha256_recomputed": tree.hexdigest(),
                    "unsafe_or_links": sum(1 for m in t.getmembers() if not (m.isfile() or m.isdir()) or m.name.startswith("/") or ".." in m.name.split("/")),
                    "site_files": site}
st = json.loads(zipfile.ZipFile(f"{art}/state.zip").read("current-site-prepared.json"))
cand = json.loads(zipfile.ZipFile(f"{art}/metadata.zip").read("candidate.json"))
ai = json.loads(zipfile.ZipFile(f"{art}/browser.zip").read("artifact-input.json"))
p, o = json.load(open(prep_path)), json.load(open(obs_path))
key = {"site/snapshots/index.json": "manifest", "site/current-meta.json": "metadata", "site/current.json": "payload"}
rec["checks"] = {
    "state_head": st["head"],
    "state_receipt_equals_prepared_latest": st["receipt"] == p,
    "observed_differs_only_in": sorted(k for k in set(p) | set(o) if p.get(k) != o.get(k)),
    "receipt_artifact_equals_recomputed": p["artifact"] == {"bytes": total, "files": len(files), "sha256": tree.hexdigest()},
    "receipt_site_digests_equal_tar": all(p[key[k]] == v for k, v in site.items()) and len(site) == 3,
    "metadata_members_equal_receipt": all(rec["members"]["metadata.zip"].get(k.replace("site/", ""), {}) == {"bytes": p[key[k]]["bytes"], "sha256": p[key[k]]["sha256"]} for k in key),
    "candidate_equals_receipt_core": all(cand.get(f) == p.get(f) for f in ("artifact", "manifest", "metadata", "payload", "run", "source", "snapshot_ids")),
    "artifact_input_assembled": [ai.get("assembled_docs_sha256"), ai.get("assembled_docs_files"), ai.get("assembled_docs_bytes")],
    "artifact_input_checked_commit": ai.get("checked_commit"),
    "artifact_input_payload_sha256": ai.get("source_payload_sha256"),
    "source_repositories": ai.get("source_repositories"),
}
json.dump(rec, open(out, "w"), indent=1, sort_keys=True); open(out, "a").write("\n")
print(json.dumps(rec["checks"], indent=1)); print(json.dumps({k: rec["pages_tar"][k] for k in ("bytes", "sha256", "regular_files", "file_bytes", "assembled_docs_sha256_recomputed", "unsafe_or_links")}))
print(json.dumps(rec["artifacts"]))
