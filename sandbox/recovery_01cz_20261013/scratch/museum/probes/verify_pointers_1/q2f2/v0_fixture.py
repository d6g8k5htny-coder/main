"""Q2-F2 verifier 2: build an independently verified fixture. Every byte string is checked against the
cd32303 museum_data.SOURCES identity (size, git blob sha1, sha256) with cd32303's own verify_bytes, and
the pinned PROOF_INDEX bytes are cross-checked against the INDEX_BYTES literal in cd32303's own tests.
Inputs: the byte map produced by an earlier lens (untrusted until verified here). No network."""
import base64, hashlib, json, sys, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../../cd32303").resolve()
SRC = (HERE / "../../pointers/fixture/verified-inputs.json").resolve()
sys.path.insert(0, str(CD / "tools")); import museum_data as md
raw_map = json.loads(SRC.read_text())
out = {}; fx = HERE / "fixture"
for path, ident in md.SOURCES.items():
    url = md.descriptor(path)["url"]; raw = base64.b64decode(raw_map[url])
    md.verify_bytes(raw, ident)                      # raises on any size/blob/sha256 mismatch
    out[url] = raw_map[url]
    t = fx / ident["repository"].split("/")[-1] / ident["path"]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(raw)
(fx / "verified-inputs.json").write_text(json.dumps(out))
# Cross-check against the test literal at cd32303
tsrc = (CD / "tests/test_museum_data.py").read_text()
node = next(n for n in ast.walk(ast.parse(tsrc)) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "INDEX_BYTES")
lit = ast.literal_eval(node.value.func.value).encode("utf-8")
idx = (fx / "Math-/PROOF_INDEX.md").read_bytes()
print(f"verified {len(out)} sources against cd32303 SOURCES; PROOF_INDEX sha256={hashlib.sha256(idx).hexdigest()} "
      f"equals cd32303 tests INDEX_BYTES: {lit == idx}")
