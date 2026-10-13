import sys, time, copy
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
import surgery_field as sf, global_h0_selector as g
def B():
    b = sf.Budget(time.monotonic()); b.begin_fixture('t'); return b
u, v = Q(7, 2**36), Q(-6, 2**54)
cert = g.select_h0(u, v, budget=B()); assert cert['status'] == 'CERTIFIED'
assert g.verify_h0(u, v, cert, budget=B())['status'] == 'PASS'
def tamper(name, fn):
    bad = copy.deepcopy(cert); fn(bad)
    r = g.verify_h0(u, v, bad, budget=B())
    print(f'{name:55s} -> {r["status"]:22s}', [k for k, val in r['checks'].items() if val is False][:2])
tamper('swap younger/essential labels', lambda c: (c['actual_barcode']['finite'][0].__setitem__('birth_root', c['actual_barcode']['essential'][0]['birth_root'])))
tamper('lifetime nudged by 2^-200', lambda c: c['actual_barcode']['finite'][0].__setitem__('lifetime_exact', c['actual_barcode']['finite'][0]['lifetime_exact'] + Q(1, 2**200)))
tamper('drop a level root', lambda c: c['levels'][2]['root_certificate']['roots'].pop())
tamper('widen a critical isolator by 2^-170', lambda c: c['critical_roots']['roots'][0].__setitem__('isolator', (c['critical_roots']['roots'][0]['isolator'][0]-Q(1,2**170), c['critical_roots']['roots'][0]['isolator'][1])))
tamper('flip axial inertia of saddle', lambda c: next(h for h in c['critical_heights'] if h['axial_inertia']=='positive').__setitem__('axial_inertia','negative'))
tamper('shift a regular level height', lambda c: c['levels'][2].__setitem__('centered_height', c['levels'][2]['centered_height'] + Q(1, 2**120)))
tamper('merge two height groups', lambda c: (c['height_groups'][0]['root_ordinals'].extend(c['height_groups'].pop(1)['root_ordinals'])))
tamper('extra finite bar duplicated', lambda c: c['actual_barcode']['finite'].append(copy.deepcopy(c['actual_barcode']['finite'][0])))
tamper('projection count 0 on included', lambda c: c['population_projection'].__setitem__('count', 0))
tamper('wrong tie_policy string', lambda c: c['actual_barcode'].__setitem__('tie_policy', 'largest_ordinal_survives'))
tamper('remove incidence edge', lambda c: c['incidence'].pop())
tamper('alter old critical value', lambda c: c['old_criticals'][0].__setitem__('value', Q(1, 5)))
tamper('component witness moved outside', lambda c: c['levels'][2]['components'][0].__setitem__('interior_witness', Q(1, 100)))
# nested status label games: does a non-CERTIFIED nested root record still verify?
tamper('nested root status relabeled (no data change)', lambda c: c['critical_roots'].__setitem__('status', 'INCONCLUSIVE_BUDGET'))
