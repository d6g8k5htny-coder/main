# The query- leg of Q1-2: project() alone accepts a query- review pin, but build()/museum_check read
# every SOURCES entry through read_source -> verify_bytes, which only admits main|Math-.
import importlib.util, pathlib, sys, hashlib
sys.dont_write_bytecode = True
TREE = pathlib.Path("<SCRATCH>/museum/cd32303")
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("md", TREE / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
md.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
p = "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md"; raw = (HERE / "fixture/Math-" / p).read_bytes()
ident = dict(md.SOURCES[p], repository="d6g8k5htny-coder/query-", commit="2" * 40)
try: md.verify_bytes(raw, ident); print("verify_bytes: accepted query- pin")
except ValueError as e: print("verify_bytes on query- pin:", e)
