import importlib.util, sys
from pathlib import Path
CD, FIX = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
n = 0
for path, ident in md.SOURCES.items():
    md.verify_bytes((FIX / ident["repository"].split("/")[-1] / ident["path"]).read_bytes(), ident); n += 1
extra = sorted(str(p.relative_to(FIX)) for p in FIX.rglob("*") if p.is_file()
               and not any(p == FIX / i["repository"].split("/")[-1] / i["path"] for i in md.SOURCES.values()))
print(f"fixture re-verified with cd32303 verify_bytes: {n} of {len(md.SOURCES)} SOURCES; extra files: {extra}")
