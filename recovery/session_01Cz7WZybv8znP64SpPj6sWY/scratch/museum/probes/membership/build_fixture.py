"""Build a byte-verified plain-file fixture of every museum SOURCES entry (cd32303 museum_data.py)
from read-only blob lookups (git cat-file blob <id>) in local clones, plus the PROOF_INDEX.md bytes
embedded in cd32303 tests/test_museum_data.py (extracted with ast, not executed). No network, no
state change. Each file passes museum_data.verify_bytes (size, git blob id, sha256) before it is written."""
import ast, importlib.util, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = Path(sys.argv[1]); GITS = sys.argv[2:]
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
def index_bytes():
    for node in ast.parse((CD / "tests/test_museum_data.py").read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INDEX_BYTES":
            return node.value.func.value.value.encode("utf-8")
out = HERE / "fixture"; missing = []
for path, ident in md.SOURCES.items():
    raw = None
    for git in GITS:
        r = subprocess.run(["git", "-C", git, "cat-file", "blob", ident["blob"]], capture_output=True)
        if r.returncode == 0: raw = r.stdout; break
    if raw is None and path == "PROOF_INDEX.md": raw = index_bytes()
    if raw is None and path == "README.md":
        # README.md@d6628da is not among local blob objects; reuse the copy another lens recovered,
        # accepted only if it passes verify_bytes against the cd32303 pin below.
        alt = HERE.parent / "pointers/fixture/Math-/README.md"
        raw = alt.read_bytes() if alt.exists() else None
    if raw is None: missing.append(path); continue
    md.verify_bytes(raw, ident)
    t = out / ident["repository"].split("/")[-1] / ident["path"]
    t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(raw)
print("fixture verified:", len(md.SOURCES) - len(missing), "of", len(md.SOURCES), "sources; missing:", missing)
