import sys, zipfile, tarfile, hashlib, io
# Recompute the workflow's assembled-docs digest (current-site.yml @ 8bc089f8, step
# "Record the assembled browser input before optional setup") over the regular files
# of the deployed github-pages artifact.tar, without extracting anything to disk.
z = zipfile.ZipFile(sys.argv[1])
tarname = [n for n in z.namelist() if n.endswith(".tar")][0]
t = tarfile.open(fileobj=io.BytesIO(z.read(tarname)))
members = [m for m in t.getmembers() if m.isfile()]
def rel(name):
    return name[2:] if name.startswith("./") else name
members.sort(key=lambda m: rel(m.name))
tree = hashlib.sha256(b"current-site-generated-browser-input-v1\0")
total = 0
for m in members:
    name = rel(m.name).encode("utf-8")
    data = t.extractfile(m).read()
    assert len(data) == m.size
    tree.update(len(name).to_bytes(8, "big")); tree.update(name)
    tree.update(m.size.to_bytes(8, "big")); total += m.size
    tree.update(data)
print("files", len(members), "bytes", total, "assembled_docs_sha256", tree.hexdigest())
print("first", rel(members[0].name), "last", rel(members[-1].name))
other = [m.name for m in t.getmembers() if not (m.isfile() or m.isdir())]
print("non-file non-dir members", len(other))
