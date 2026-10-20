"""Where is #448's whitespace-only oracle sentinel refused, with and without mutant R03?

Usage: python3 -I -B -S r03_diag.py SRC
SRC is a checkout of main efc2fe90 (tests, tools, experiments/semantic_memory). For each
variant (original benchmark.py, and R03 = its third non-def `_require(` call, in `_text`,
replaced by `_require(True or `) and each sentinel (" " as in #448's subtest, and "　"),
the fixture is changed exactly as the subtest does (load, set, re-encode, rebind the
case-file descriptor) and admit_benchmark() is called. The refusing benchmark.py line is
taken from the innermost traceback frame in that file.
"""
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
BENCH = SRC / "experiments/semantic_memory/benchmark.py"
FIXTURES = SRC / "experiments/semantic_memory/fixtures"
text = BENCH.read_text()
sites = [m.start() for m in re.finditer(r"_require\(", text) if not text[max(0, m.start() - 4):m.start()].endswith("def ")]
pos = sites[2]
assert text[pos:text.find("\n", pos)].startswith("_require(type(value) is str and bool(value.strip()))")
VARIANTS = {"original": text, "R03": text[:pos] + "_require(True or " + text[pos + len("_require("):]}


def encoded(value):  # same as the test file's encoded()
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def load(source, name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(BENCH)
    exec(compile(source, str(BENCH), "exec"), module.__dict__)
    return module


rows = []
for vname, source in VARIANTS.items():
    bench = load(source, "bench_" + vname)
    for label, sentinel in (("ascii_space", " "), ("ideographic_space", "　")):
        with tempfile.TemporaryDirectory(prefix="r03-") as t:
            path = Path(t) / "fixtures"
            shutil.copytree(FIXTURES, path, copy_function=shutil.copyfile)
            oracle = json.loads((path / "heldout-oracle.json").read_bytes())
            oracle["oracle_only_sentinel"] = sentinel
            raw = encoded(oracle)
            (path / "heldout-oracle.json").write_bytes(raw)
            manifest = json.loads((path / "corpus-manifest.json").read_bytes())
            manifest["case_files"]["heldout-oracle.json"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            (path / "corpus-manifest.json").write_bytes(encoded(manifest))
            try:
                out = bench.admit_benchmark(path)
                outcome = "ADMITTED admitted=%s" % out.get("benchmark_admission", out).get("admitted") if isinstance(out, dict) else "ADMITTED"
            except Exception as exc:
                frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == str(BENCH)]
                frames = [f for f in frames if f.name != "_require"]  # report the caller of _require
                f = frames[-1] if frames else None
                outcome = "REFUSED %s at benchmark.py:%s %s" % (type(exc).__name__, f.lineno if f else "?", (f.line or "").strip() if f else "")
        rows.append({"variant": vname, "sentinel": label, "outcome": outcome})
        print(json.dumps(rows[-1]), flush=True)
