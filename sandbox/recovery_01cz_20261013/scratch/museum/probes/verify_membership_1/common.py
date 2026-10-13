"""Shared loader: imports the cd32303 museum_data.py by path (no execution from the extracted dir)."""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = HERE.parents[1] / "cd32303"
MAIN_GIT = Path("/home/user/main")          # read-only object store holding 71400b94 and its history
FIX = HERE / "fixture"                       # byte-verified plain files (build_fixture.py)
def load():
    spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
    md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
    return md
