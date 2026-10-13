# Emit the builder's museum.json for the offline fixture (README swapped as in v_q1_9.py) to argv[1]; math root argv[2].
import hashlib, importlib.util, pathlib, shutil, sys
sys.dont_write_bytecode = True
M = pathlib.Path("<SCRATCH>/museum")
spec = importlib.util.spec_from_file_location("md", M / "cd32303/tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
md.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
math = pathlib.Path(sys.argv[2]); shutil.copytree(M / "probes/substitution/fixture/Math-", math)
raw = (M / "probes/substitution/readme_10e1f191.md").read_bytes(); (math / "README.md").write_bytes(raw)
md.SOURCES["README.md"] = dict(md.SOURCES["README.md"], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                               blob=hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest())
pathlib.Path(sys.argv[1]).write_bytes(md.dump(md.build(pathlib.Path("/home/user/main"), math)))
