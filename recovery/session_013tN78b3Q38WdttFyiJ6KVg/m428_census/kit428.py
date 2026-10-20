"""Single-point mutants of main#428's validator and CLI, run against the PR's own tests.

Usage: python3 -I kit428.py SRC OUT.tsv [ID ...]
SRC is a checkout of the PR head. Each mutant gets a real copy (no hard links: the
code under test rejects st_nlink != 1) of the three directories the tests use.
"""
import concurrent.futures, re, shutil, subprocess, sys, tempfile
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
ONLY = set(sys.argv[3:])
BENCH = "experiments/semantic_memory/benchmark.py"
CLI = "tools/semantic_memory_experiment.py"
DIRS = ("tests", "tools", "experiments/semantic_memory")
TEST = ["-m", "unittest", "discover", "-s", "tests", "-p", "test_semantic_memory_benchmark.py"]

bench = (SRC / BENCH).read_text()
cli = (SRC / CLI).read_text()

def funcs(text):
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^def (\w+)", text, re.M)]
    def at(pos):
        name = "<module>"
        for s, n in starts:
            if s <= pos:
                name = n
        return name
    return at

M = []
where = funcs(bench)
call_sites = [m.start() for m in re.finditer(r"_require\(", bench)
              if not bench[max(0, m.start() - 4):m.start()].endswith("def ")]
for i, pos in enumerate(call_sites, 1):
    line = bench.count("\n", 0, pos) + 1
    snippet = bench[pos:bench.find("\n", pos)].strip()[:90]
    M.append(("R%02d" % i, "%s:%d %s" % (where(pos), line, snippet), BENCH,
              bench[:pos] + "_require(True or " + bench[pos + len("_require("):]))

def single(mid, desc, path, text, old, new):
    n = text.count(old)
    M.append((mid, desc, path, text.replace(old, new) if n == 1 else "BAD-PATTERN n=%d" % n))

single("ALL", "_require disabled entirely", BENCH, bench,
       "def _require(condition):\n    if not condition:", "def _require(condition):\n    if False:")
single("NF", "nonfinite JSON raise removed", BENCH, bench,
       '        raise ValueError("nonfinite JSON")', '        pass')
single("A1", "admitted_corpus bypasses admission", BENCH, bench,
       'return admit_benchmark(fixture_dir)["corpus_projection"]',
       'return validate_fixtures(fixture_dir)["corpus_projection"]')
single("C1", "CLI: more than one --fixtures accepted", CLI, cli,
       "if len(arguments.fixtures) != 1:", "if False:")
single("C2", "CLI: --report accepted for prepare/admit", CLI, cli,
       "if arguments.report is not None:", "if False:")
single("C3", "CLI: verify-report without exactly one --report", CLI, cli,
       "elif arguments.report is None or len(arguments.report) != 1:", "elif False:")
single("C4", "CLI: product-source type/size/nlink check disabled", CLI, cli,
       "if not stat.S_ISREG(before.st_mode) or before.st_size > 131072 or before.st_nlink != 1:",
       "if False:")
single("C5", "CLI: product-source read-stability check disabled", CLI, cli,
       "if len(raw) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):",
       "if False:")
single("C6", "CLI: short-write check disabled", CLI, cli, "if written <= 0:", "if False:")
single("C7", "CLI: error path exits 0", CLI, cli, "            pass\n        return 1", "            pass\n        return 0")
single("C8", "CLI: admit-benchmark runs structural validation only", CLI, cli,
       "raw = product.canonical_report(product.admit_benchmark(arguments.fixtures[0]))",
       "raw = product.canonical_report(product.validate_fixtures(arguments.fixtures[0]))")

def run(root, flags):
    r = subprocess.run([sys.executable, "-I", "-B", "-S"] + flags + TEST, cwd=root,
                       capture_output=True, text=True, timeout=600)
    tail = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
    return r.returncode, " ".join(tail[-2:])[:120]

def one(item):
    mid, desc, path, text = item
    if text is not None and text.startswith("BAD-PATTERN"):
        return (mid, desc, text, "", "")
    with tempfile.TemporaryDirectory(prefix="m428-") as t:
        root = Path(t)
        for d in DIRS:
            shutil.copytree(SRC / d, root / d, symlinks=True)
        if text is not None:
            (root / path).write_text(text)
        n = run(root, [])
        o = run(root, ["-O"])
    verdict = "KILLED" if n[0] and o[0] else ("SURVIVED" if not n[0] and not o[0] else "MIXED")
    return (mid, desc, verdict, n[1], o[1])

items = [("BASE", "unmodified", None, None)] + [m for m in M if not ONLY or m[0] in ONLY]
rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(one, items):
        rows.append(row)
        print("\t".join(row), flush=True)
OUT.write_text("".join("\t".join(r) + "\n" for r in rows))
