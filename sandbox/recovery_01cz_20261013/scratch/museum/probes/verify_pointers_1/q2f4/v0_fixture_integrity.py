"""Independently re-verify the probe fixture: every entry must equal a SOURCES identity of the cd32303
museum_data.py by byte count, git blob sha1 and sha256. Read-only; no network."""
import base64, hashlib, json, sys
from pathlib import Path
CD = Path(sys.argv[1]); FIX = Path(sys.argv[2])
sys.path.insert(0, str(CD / "tools"))
import museum_data as md
fx = json.loads(FIX.read_text())
ok = 0
for path, ident in md.SOURCES.items():
    url = md.descriptor(path)["url"]
    raw = base64.b64decode(fx[url])
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert len(raw) == ident["bytes"] and blob == ident["blob"] and hashlib.sha256(raw).hexdigest() == ident["sha256"], path
    ok += 1
extra = set(fx) - {md.descriptor(p)["url"] for p in md.SOURCES}
print(f"fixture entries verified against cd32303 SOURCES: {ok}/{len(md.SOURCES)}; extra entries: {len(extra)}")
# museum.json at cd32303 equals the generator output on these bytes (what museum_check.py asserts in CI)
import tempfile
with tempfile.TemporaryDirectory() as d:
    for path, ident in md.SOURCES.items():
        t = Path(d) / ident["repository"].split("/")[-1] / ident["path"]; t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(base64.b64decode(fx[md.descriptor(path)["url"]]))
    md.check_export(CD / "docs/site/museum.json", md.build(Path(d) / "main", Path(d) / "Math-"))
print("cd32303 docs/site/museum.json == museum_data.build(fixture): True")
