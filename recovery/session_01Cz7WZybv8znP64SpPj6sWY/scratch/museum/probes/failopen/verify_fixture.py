"""Independently verify a verified-inputs.json (url -> base64 bytes) against the cd32303 SOURCES pins
(size, git blob, sha256) and write a private copy plus a plain-file tree. No network.
usage: python3 -I verify_fixture.py <cd32303 tree> <input verified-inputs.json> <out dir>"""
import base64, importlib.util, json, sys
from pathlib import Path
CD, SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
given = json.loads(SRC.read_text())
mapping = {}
for path, ident in md.SOURCES.items():
    url = md.descriptor(path)["url"]
    raw = base64.b64decode(given[url])
    md.verify_bytes(raw, ident)            # raises on any size/blob/sha256 mismatch
    t = OUT / ident["repository"].split("/")[-1] / ident["path"]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(raw)
    mapping[url] = given[url]
extra = sorted(set(given) - set(mapping))
(OUT / "verified-inputs.json").write_text(json.dumps(mapping))
print("fixture verified:", len(mapping), "of", len(md.SOURCES), "SOURCES pins; extra keys ignored:", extra)
