"""Byte-verified local fixture of every museum SOURCES entry, read from read-only git objects.
No network. Each blob is checked with museum_data.verify_bytes (size, git blob, sha256) from the cd32303 bytes.
usage: python3 -I build_fixture.py <cd32303 tree> <Math- git dir> <main git dir> <out dir>"""
import base64, importlib.util, json, subprocess, sys
from pathlib import Path
CD, MATH_GIT, MAIN_GIT, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
mapping = {}
for path, ident in md.SOURCES.items():
    git = MAIN_GIT if ident["repository"].endswith("/main") else MATH_GIT
    raw = subprocess.run(["git", "-C", git, "cat-file", "blob", ident["blob"]], capture_output=True, check=True).stdout
    md.verify_bytes(raw, ident)
    t = OUT / ident["repository"].split("/")[-1] / ident["path"]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(raw)
    mapping[md.descriptor(path)["url"]] = base64.b64encode(raw).decode()
(OUT / "verified-inputs.json").write_text(json.dumps(mapping))
print("fixture verified:", len(mapping), "of", len(md.SOURCES), "sources")
