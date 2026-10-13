"""Nested statuses after withdrawal in select_h0 and in the driver."""
import sys, time, tempfile, json, collections
from pathlib import Path
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf
import global_h0_selector as g
import run_global_h0_selector as drv

def nested(obj, path=''):
    out = []
    if isinstance(obj, dict):
        if obj.get('status') in ('CERTIFIED', 'PASS'): out.append(path + ':' + obj['status'])
        for k, v in obj.items(): out += nested(v, path + '/' + k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): out += nested(v, path + '[%d]' % i)
    return out

u, v = drv.frozen_fixtures()[0]['u'], drv.frozen_fixtures()[0]['v']
# find a polynomial_limit that lets critical roots certify but fails later
for limit in (300, 600, 1200, 2500, 5000, 20000):
    b = sf.Budget(time.monotonic(), polynomial_limit=limit); b.begin_fixture('f')
    cert = g.select_h0(u, v, budget=b)
    print('limit', limit, 'status', cert['status'], 'failure', cert.get('failure'), 'critical_roots', cert['critical_roots'].get('status'),
          'levels', [l['root_certificate'].get('status') for l in cert['levels']])
    if cert['status'] != 'CERTIFIED':
        print('  nested CERTIFIED/PASS left by select_h0._withdraw:', nested(cert)[:8], '... total', len(nested(cert)))
        break

# Driver: inject a reduced Budget factory and check the written certificate for nested CERTIFIED/PASS
base = Path('.').resolve()
def importer():
    handles = drv._scientific_imports()
    real = handles['Budget']
    handles['Budget'] = lambda start: real(start, polynomial_limit=limit)
    return handles
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / 'out'
    # Base must contain all 13 consumed sources: use the worktree (read-only read of bytes) as base
    wt = Path('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/wt350')
    res = drv.run_attempt(base=wt, output_dir=out, importer=importer)
    c = json.loads((out / 'CERTIFICATE.json').read_bytes())
    print('driver status', c['status'], 'failure', c.get('failure'))
    left = nested(c)
    print('nested CERTIFIED/PASS surviving in driver output:', len(left), left[:5])
    print('withdrawals', c.get('withdrawals'))
