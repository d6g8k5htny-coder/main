# Load cd32303 tools/museum_data.py by path (no test import, no network).
import importlib.util, pathlib, sys
sys.dont_write_bytecode = True  # never write .pyc into the read-only cd32303 tree
SCRATCH = pathlib.Path("<SCRATCH>/museum")
TREE = SCRATCH / "cd32303"; HERE = SCRATCH / "probes/verify_substitution_0/q1_2"
spec = importlib.util.spec_from_file_location("museum_data_cd32303", TREE / "tools/museum_data.py")
museum = importlib.util.module_from_spec(spec); spec.loader.exec_module(museum)
def _blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED in probe")
museum.urlopen = _blocked
INDEX = (HERE / "PROOF_INDEX.md").read_bytes()      # == pinned PROOF_INDEX identity (checked by extract_index.py)
STATUS = (TREE / "STATUS.md").read_bytes()          # == pinned STATUS identity
