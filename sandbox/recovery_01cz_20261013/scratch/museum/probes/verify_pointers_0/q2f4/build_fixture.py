"""Q2-F4 verifier 1: independent fixture. Parses (does not import) SOURCES from cd32303 tools/museum_data.py,
reads blobs with read-only `git cat-file blob`, checks size/blob/sha256 with hashlib here. Also extracts two REAL
other revisions of Math- PROOF_INDEX.md (before and after the d6628da pin) for scenario (a2). No network."""
import base64, hashlib, json, re, subprocess
from pathlib import Path
from urllib.parse import quote
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../../cd32303").resolve()
MATH = "<SCRATCH>/repos/math"
MAIN = "/home/user/main"
src = (CD / "tools/museum_data.py").read_text(encoding="utf-8")
SOURCES = json.loads(re.search(r"SOURCES = json\.loads\(r'''(.*?)'''\)", src, re.S).group(1))
SOURCES["README.md"] = json.loads(re.search(r'^SOURCES\["README\.md"\] = (\{.*\})$', src, re.M).group(1))
def git(repo, *a): return subprocess.run(["git", "-C", repo, *a], capture_output=True, check=True).stdout
def url(repo, commit, path): return f"https://raw.githubusercontent.com/{repo}/{commit}/{quote(path, safe='/')}"
mapping = {}
for path, ident in SOURCES.items():
    raw = git(MAIN if ident["repository"].endswith("/main") else MATH, "cat-file", "blob", ident["blob"])
    assert len(raw) == ident["bytes"] and hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == ident["blob"] \
        and hashlib.sha256(raw).hexdigest() == ident["sha256"], path
    mapping[url(ident["repository"], ident["commit"], path)] = base64.b64encode(raw).decode()
extra = {}
for commit in ["965011dbb301fb541f28f7ff9a8d2aea36c9db1d", "7f487c469e60c00dacff5666585e41af5aedf021"]:
    raw = git(MATH, "show", f"{commit}:PROOF_INDEX.md")
    extra[commit] = {"repository": "d6g8k5htny-coder/Math-", "commit": commit, "path": "PROOF_INDEX.md",
                     "blob": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(), "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest(), "b64": base64.b64encode(raw).decode()}
(HERE / "fixture.json").write_text(json.dumps({"sources": mapping, "proof_index_revisions": extra}))
print("fixture verified:", len(mapping), "pinned sources;", "extra PROOF_INDEX revisions:",
      {c[:8]: (e["blob"][:8], e["bytes"]) for c, e in extra.items()})
