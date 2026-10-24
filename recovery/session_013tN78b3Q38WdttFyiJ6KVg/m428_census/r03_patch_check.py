"""Does a one-character change to #448's whitespace subtest kill mutant R03?

Usage: python3 -I r03_patch_check.py SRC OUT.tsv
SRC is a checkout of main efc2fe90. Four runs of the main test file, each with the workflow's
flags (-I -B -S) in normal and -O mode, on real copies of tests/, tools/ and
experiments/semantic_memory/:
  test as landed, benchmark original  -> expected OK
  test as landed, benchmark R03       -> OK (R03 survives, as the census shows)
  test patched,   benchmark original  -> must stay OK
  test patched,   benchmark R03       -> should FAIL (R03 killed)
The patch replaces the subtest's sentinel " " by "\\u3000", a blank that str.strip() removes and
that occurs in no projected string, so the leak check at benchmark.py:442 cannot refuse it.
"""
import concurrent.futures
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
BENCH = "experiments/semantic_memory/benchmark.py"
TEST = "tests/test_semantic_memory_benchmark.py"
DIRS = ("tests", "tools", "experiments/semantic_memory")
CMD = ["-m", "unittest", "discover", "-s", "tests", "-p", "test_semantic_memory_benchmark.py"]

bench = (SRC / BENCH).read_text()
sites = [m.start() for m in re.finditer(r"_require\(", bench) if not bench[max(0, m.start() - 4):m.start()].endswith("def ")]
pos = sites[2]
assert bench[pos:bench.find("\n", pos)].startswith("_require(type(value) is str and bool(value.strip()))")
r03 = bench[:pos] + "_require(True or " + bench[pos + len("_require("):]
test = (SRC / TEST).read_text()
old = 'lambda v: v.update(oracle_only_sentinel=" ")))'
assert test.count(old) == 1
patched = test.replace(old, 'lambda v: v.update(oracle_only_sentinel="\\u3000")))')

CASES = [("landed", "original", test, bench), ("landed", "R03", test, r03),
         ("patched", "original", patched, bench), ("patched", "R03", patched, r03)]


def run(root, flags):
    r = subprocess.run([sys.executable, "-I", "-B", "-S"] + flags + CMD, cwd=root, capture_output=True, text=True, timeout=900)
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
    sub = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "mutation=" in l][:3]
    return r.returncode, " ".join(tail[-2:]), " | ".join(sub)


def one(case):
    tname, bname, ttext, btext = case
    with tempfile.TemporaryDirectory(prefix="r03p-") as t:
        root = Path(t)
        for d in DIRS:
            shutil.copytree(SRC / d, root / d, symlinks=True)
        (root / TEST).write_text(ttext)
        (root / BENCH).write_text(btext)
        n = run(root, [])
        o = run(root, ["-O"])
    return (tname, bname, "rc=%d %s" % (n[0], n[1]), "rc=%d %s" % (o[0], o[1]), n[2] or o[2])


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(one, CASES))
for r in rows:
    print("\t".join(r), flush=True)
OUT.write_text("".join("\t".join(r) + "\n" for r in rows))
