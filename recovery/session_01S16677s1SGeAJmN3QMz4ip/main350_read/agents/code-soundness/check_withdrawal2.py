import sys, time, tempfile, json
from pathlib import Path
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf, global_h0_selector as g, run_global_h0_selector as drv
def nested(obj, path=''):
    out = []
    if isinstance(obj, dict):
        if obj.get('status') in ('CERTIFIED', 'PASS'): out.append(path)
        for k, v in obj.items(): out += nested(v, path + '/' + k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj): out += nested(v, path + '[%d]' % i)
    return out
u, v = Q(7, 2**36), Q(-6, 2**54)
for limit in range(400, 20001, 400):
    b = sf.Budget(time.monotonic(), polynomial_limit=limit); b.begin_fixture('f')
    cert = g.select_h0(u, v, budget=b)
    if cert['status'] != 'CERTIFIED' and cert['critical_roots'].get('status') == 'CERTIFIED':
        print('limit', limit, 'selection status', cert['status'], cert['failure'])
        print('  select_h0 nested CERTIFIED left:', nested(cert))
        break
    if cert['status'] == 'CERTIFIED':
        print('fully certified at limit', limit, 'evals', cert['work']['polynomial_evaluations']); break
# driver with a whole-attempt field ceiling that fixture_01 passes and fixture_02 breaks
wt = Path('/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/wt350')
def importer():
    h = drv._scientific_imports(); real = h['Budget']
    h['Budget'] = lambda start: real(start, field_limit=40)
    return h
with tempfile.TemporaryDirectory() as tmp:
    res = drv.run_attempt(base=wt, output_dir=Path(tmp)/'out', importer=importer)
    c = json.loads((Path(tmp)/'out'/'CERTIFICATE.json').read_bytes())
    print('driver status', c['status'], c['failure']['phase'], c['failure']['detail'])
    print('fixture statuses', [(f['fixture_id'], f['status'], f.get('computed_status_before_withdrawal')) for f in c['fixtures']])
    print('nested CERTIFIED/PASS surviving:', nested(c))
    print('guard.accepted_review after withdrawal:', c['guard']['accepted_review'])
    print('grid_obstruction status after withdrawal:', c['grid_obstruction']['status'], c['grid_obstruction'].get('computed_status_before_withdrawal'))
    fr = c['work']['field_records'][0]
    print('field record model.accepted_review:', fr['model']['accepted_review'])
