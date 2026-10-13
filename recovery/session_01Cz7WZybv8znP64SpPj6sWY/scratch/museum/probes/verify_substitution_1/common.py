# Verifier-2 harness for Q1-1. Loads the cd32303 builder by path; blocks network; never writes
# into the cd32303 tree. Every fixture file is re-verified against the builder's own SOURCES pins.
import ast, hashlib, importlib.util, pathlib, sys
sys.dont_write_bytecode = True
SCRATCH = pathlib.Path("<SCRATCH>/museum")
TREE = SCRATCH / "cd32303"
HERE = SCRATCH / "probes/verify_substitution_1"
FIX = HERE / "fixture/Math-"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

museum = load("museum_data_cd32303_v2", TREE / "tools/museum_data.py")
def _blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED")
museum.urlopen = _blocked

# PROOF_INDEX bytes: taken independently from the INDEX_BYTES literal in cd32303 tests/test_museum_data.py
src = (TREE / "tests/test_museum_data.py").read_text()
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "INDEX_BYTES":
        INDEX = node.value.func.value.value.encode("utf-8"); break
STATUS = (TREE / "STATUS.md").read_bytes()
museum.verify_bytes(INDEX, museum.INDEX); museum.verify_bytes(STATUS, museum.STATUS)

def verify_fixture():
    ok, miss = 0, []
    for path, ident in museum.SOURCES.items():
        if ident["repository"].endswith("/Math-"):
            f = FIX / path
        else:
            f = TREE / path
        if not f.exists(): miss.append(path); continue
        museum.verify_bytes(f.read_bytes(), ident); ok += 1
    return ok, miss
