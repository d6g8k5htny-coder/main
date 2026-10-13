"""Emit the url->base64 map exactly as cd32303 tools/museum_check.py builds MUSEUM_FIXTURE (bytes from read_source)."""
import base64, importlib.util, json, os, sys
from pathlib import Path
wt, math_root = Path(sys.argv[1]), Path(sys.argv[2]); os.environ["GIT_OPTIONAL_LOCKS"] = "0"
spec = importlib.util.spec_from_file_location("md", wt / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
print(json.dumps({md.descriptor(s["path"])["url"]: base64.b64encode(md.read_source(s, wt, math_root)).decode() for s in md.SOURCES.values()}))
