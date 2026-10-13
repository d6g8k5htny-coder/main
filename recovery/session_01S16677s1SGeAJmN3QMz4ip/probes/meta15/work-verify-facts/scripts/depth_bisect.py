"""CLI bisection of the lowest wrapper count n that exits 1 (RecursionError) for a given innermost object."""
import json, subprocess, sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import make_ws, SSA

SRC = Path(sys.argv[1]); PY = sys.argv[2]; INNER = sys.argv[3]
SCRIPT = SRC / 'tools/architecture_conformance.py'
import probe; probe.SCRIPT = SCRIPT

def code(n, opt=False):
    root = make_ws({'x.json': '{"a":' * n + INNER + '}' * n})
    env = dict(os.environ); env.pop('PYTHONOPTIMIZE', None)
    p = subprocess.run([PY, '-B', '-S', *(['-O'] if opt else []), str(SCRIPT), '--workspace', str(root)], capture_output=True, text=True, env=env)
    v = json.loads(p.stdout)['violations'] if p.stdout else None
    return p.returncode, (v[0][:30] if v else v)

lo, hi = 1, 1200  # code(lo) not 1, code(hi) is 1 (assumed)
assert code(lo)[0] != 1
while hi - lo > 1:
    mid = (lo + hi) // 2
    if code(mid)[0] == 1: hi = mid
    else: lo = mid
print(PY, INNER, 'last non-crash n =', lo, code(lo), code(lo, True), '| first crash n =', hi, code(hi)[0], code(hi, True)[0])
