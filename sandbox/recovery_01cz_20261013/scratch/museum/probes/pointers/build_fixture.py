"""Build a byte-verified local fixture of every museum SOURCES entry from read-only git objects.
No network. Every file is checked with museum_data.verify_bytes (size, git blob, sha256) from the cd32303 bytes."""
import base64, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = Path(sys.argv[1]); MATH_GIT = sys.argv[2]; MAIN_GIT = sys.argv[3]
sys.path.insert(0, str(CD / "tools"))
import museum_data as md
out = HERE / "fixture"; mapping = {}
for path, ident in md.SOURCES.items():
    git = MAIN_GIT if ident["repository"].endswith("/main") else MATH_GIT
    raw = subprocess.run(["git", "-C", git, "cat-file", "blob", ident["blob"]], capture_output=True, check=True).stdout
    md.verify_bytes(raw, ident)
    t = out / ident["repository"].split("/")[-1] / ident["path"]; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(raw)
    mapping[md.descriptor(path)["url"]] = base64.b64encode(raw).decode()
(out / "verified-inputs.json").write_text(json.dumps(mapping))
print("fixture verified:", len(mapping), "sources")
