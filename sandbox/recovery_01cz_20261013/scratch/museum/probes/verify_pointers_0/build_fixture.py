"""Verifier 1 (Q2-F1): independent byte-verified fixture of every museum SOURCES entry.

Reads SOURCES from the cd32303 tools/museum_data.py by parsing (not importing) the literal,
fetches each blob with read-only `git cat-file blob` (main from /home/user/main, Math- from
/home/user/Math-), and checks byte count, git blob id and SHA-256 with hashlib here, not with
the code under review. No network. Writes fixture/<repo>/<path> and fixture/verified-inputs.json.
"""
import base64, hashlib, json, re, subprocess, sys
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
CD = (HERE / "../../cd32303").resolve()
src = (CD / "tools/museum_data.py").read_text(encoding="utf-8")
SOURCES = json.loads(re.search(r"SOURCES = json\.loads\(r'''(.*?)'''\)", src, re.S).group(1))
SOURCES["README.md"] = json.loads(re.search(r'^SOURCES\["README\.md"\] = (\{.*\})$', src, re.M).group(1))  # line 259
REPOS = {"d6g8k5htny-coder/main": ["/home/user/main"], "d6g8k5htny-coder/Math-": ["<SCRATCH>/repos/math", "/home/user/Math-", "/home/user/math-", "/home/user/main", "/home/user/query-"]}
def cat(repo_list, blob):
    for r in repo_list:
        res = subprocess.run(["git", "-C", r, "cat-file", "blob", blob], capture_output=True)
        if res.returncode == 0:
            return res.stdout, r
    raise SystemExit("blob not found locally: " + blob)
found = {}
out = HERE / "fixture"
mapping = {}
for path, ident in SOURCES.items():
    raw, where = cat(REPOS[ident["repository"]], ident["blob"]); found[where] = found.get(where, 0) + 1
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert len(raw) == ident["bytes"] and blob == ident["blob"] and hashlib.sha256(raw).hexdigest() == ident["sha256"], path
    target = out / ident["repository"].split("/")[-1] / ident["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    url = "https://raw.githubusercontent.com/%s/%s/%s" % (ident["repository"], ident["commit"], quote(path, safe="/"))
    mapping[url] = base64.b64encode(raw).decode("ascii")
(out / "verified-inputs.json").write_text(json.dumps(mapping), encoding="utf-8")
print("blobs read from:", found)
print("independent fixture verified (size, git blob, sha256):", len(mapping), "sources")
