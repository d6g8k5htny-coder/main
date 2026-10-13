import json, subprocess, sys, shutil
from pathlib import Path
W = Path(__file__).resolve().parent
sys.path.insert(0, str(W))
import driver
ws = W / 'bisect_ws'
if ws.exists(): shutil.rmtree(ws)
driver.build_fixture(ws)
g = ws / 'meta-framework/generated'; g.mkdir()
M = driver.M
def run(depth, kind, inner, opt=False):
    (g / 'x.json').write_text(driver.nest(depth, inner, kind))
    p = subprocess.run([driver.PY] + (['-O'] if opt else []) + [str(driver.SCRIPT), '--workspace', str(ws)], capture_output=True, text=True)
    tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''
    v = json.loads(p.stdout)['violations'] if p.returncode in (0, 2) else None
    return p.returncode, tail, v
def bisect(kind, inner, lo, hi, pred):
    # pred(rc, tail, v) True means 'still fine'
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(*run(mid, kind, inner)): lo = mid
        else: hi = mid
    return lo, hi
out = {}
for kind in ('dict', 'list'):
    for label, inner in (('true', '{' + M + ':true}'), ('benign', '{"k":"v"}')):
        lo, hi = bisect(kind, inner, 1, 600, lambda rc, t, v: rc != 1)
        out[f'{kind}_{label}_checker_recursion'] = {'last_ok_depth': lo, 'first_crash_depth': hi,
            'at_last_ok': run(lo, kind, inner), 'at_first_crash': run(hi, kind, inner),
            'at_first_crash_O': run(hi, kind, inner, True)}
    lo, hi = bisect(kind, '{"k":"v"}', 5000, 200000, lambda rc, t, v: not (v and 'parse error' in v[0]))
    out[f'{kind}_json_loads_parse_limit'] = {'last_nonparse_depth': lo, 'first_parse_error_depth': hi,
        'at_first_parse_error': run(hi, kind, '{"k":"v"}')}
# marker true at root object short-circuits before a deep sibling branch
(g / 'x.json').write_text('{' + M + ':true,"deep":' + driver.nest(1000, '{"k":"v"}', 'dict') + '}')
p = subprocess.run([driver.PY, str(driver.SCRIPT), '--workspace', str(ws)], capture_output=True, text=True)
out['root_true_then_deep1000_sibling'] = {'rc': p.returncode, 'violations': json.loads(p.stdout)['violations'] if p.returncode in (0,2) else p.stderr.strip().splitlines()[-1]}
(g / 'x.json').write_text('{"a":' + driver.nest(1000, '{"k":"v"}', 'dict') + ',"b":{' + M + ':true}}')
p = subprocess.run([driver.PY, str(driver.SCRIPT), '--workspace', str(ws)], capture_output=True, text=True)
out['deep1000_branch_then_nested_true'] = {'rc': p.returncode, 'violations': json.loads(p.stdout)['violations'] if p.returncode in (0,2) else p.stderr.strip().splitlines()[-1]}
(g / 'x.json').write_text('{"b":{' + M + ':true},"a":' + driver.nest(1000, '{"k":"v"}', 'dict') + '}')
p = subprocess.run([driver.PY, str(driver.SCRIPT), '--workspace', str(ws)], capture_output=True, text=True)
out['nested_true_then_deep1000_branch'] = {'rc': p.returncode, 'violations': json.loads(p.stdout)['violations'] if p.returncode in (0,2) else p.stderr.strip().splitlines()[-1]}
print(json.dumps(out, indent=1))
(W / 'raw/depth_bisect.json').write_text(json.dumps(out, indent=1))
