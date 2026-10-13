"""Tamper NOT covered by the suite: omit the two quartic roots bounding the younger
component at the two-component regular level, drop that component, rewrite events
so the younger bar disappears. Pristine verifier should reject at verify/level*/complete*."""
import copy, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
from surgery_field import Budget
import global_h0_selector as selector

def budget():
    b = Budget(time.monotonic()); b.begin_fixture('t'+str(time.monotonic())); return b

u, v = Q(7, 2**36), Q(-6, 2**54)
cert = selector.select_h0(u, v, budget=budget())
assert cert['status'] == 'CERTIFIED'
bad = copy.deepcopy(cert)
li, level = next((i, l) for i, l in enumerate(bad['levels']) if len(l['components']) == 2)
younger = level['components'][1]          # component around +2A (ordinals 2,3)
rs = level['root_certificate']
keep = [r for r in rs['roots'] if r['ordinal'] not in younger['boundary_root_ordinals']]
for k, r in enumerate(keep): r['ordinal'] = k
rs['roots'] = keep
level['components'] = [level['components'][0]]
# recompute sign_intervals the way the verifier will
signs = []
for index, lo, hi in selector._component_intervals(rs):
    w = (lo+hi)/2
    val = selector._eval(rs['polynomial'], w, budget(), 'x')
    signs.append(dict(boundary_root_ordinals=[index-1 if index else None, index if index < len(rs['roots']) else None], witness=w, value=val))
level['sign_intervals'] = signs
# incidence: drop edges touching the removed component; rewrite events: group g1 (younger max) becomes a continuation
bad['incidence'] = [e for e in bad['incidence'] if younger['id'] not in (e['upper_component'], e['lower_component'])]
groups = bad['height_groups']
# rebuild events by a straightforward replay with the tampered levels (mirrors _verify_graph's logic)
heights = {h['root_ordinal']: h for h in bad['critical_heights']}
rank = {g['id']: i for i, g in enumerate(groups)}
active, events, finite = {}, [], []
for i, group in enumerate(groups):
    upper, lower = bad['levels'][i], bad['levels'][i+1]
    edges = [e for e in bad['incidence'] if e['upper_level'] == upper['id']]
    ev = dict(group_id=group['id'], births=[], continuations=[], merges=[]); state = {}
    for c in lower['components']:
        preds = [e['upper_component'] for e in edges if e['lower_component'] == c['id']]
        inc = sorted(active[x] for x in preds)
        if not inc:
            elig = [o for o in group['root_ordinals'] if selector._root_inside(bad['critical_roots']['roots'][o], c, lower) and heights[o]['axial_inertia'] in ('negative','zero')]
            s = elig[0]; ev['births'].append(dict(component=c['id'], birth_root=s))
        elif len(inc) == 1:
            s = inc[0]; ev['continuations'].append(dict(upper_component=preds[0], lower_component=c['id'], survivor_birth=s))
        else:
            raise SystemExit('unexpected merge in tamper')
        state[c['id']] = s
    active = state; events.append(ev)
bad['events'] = events
bad['actual_barcode']['finite'] = []
bad['population_projection'] = selector._projection(u, v, [])
r = selector.verify_h0(u, v, bad, budget=budget())
print('verify status:', r['status'])
print('false checks :', [k for k, val in r['checks'].items() if val is False])
print('finite bars claimed:', bad['actual_barcode']['finite'], 'projection:', bad['population_projection'])
