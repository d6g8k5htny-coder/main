"""Which verifier check actually rejects each tampered certificate in test_global_h0_selector."""
import copy, sys, time
from fractions import Fraction as Q
sys.path.insert(0, 'experiments/universality/two_parameter_cusp_population')
from surgery_field import Budget
import global_h0_selector as selector

def budget():
    b = Budget(time.monotonic()); b.begin_fixture('instr'+str(time.monotonic())); return b

def first_false(result):
    return [k for k, v in result['checks'].items() if v is False]

def sound(u, v):
    c = selector.select_h0(u, v, budget=budget())
    assert c['status'] == 'CERTIFIED'
    return c

u, v = Q(7, 2**36), Q(-6, 2**54)
H = sound(u, v)
T = sound(Q(3, 2**33), Q(0))
cases = {}
b = copy.deepcopy(H); b['critical_roots']['roots'][0]['multiplicity'] += 1; cases['changed_root_multiplicity'] = (u, v, b)
b = copy.deepcopy(H); b['levels'][0]['root_certificate']['scalar_unit'] *= -1; cases['changed_quartic_scalar'] = (u, v, b)
b = copy.deepcopy(H); next(l for l in b['levels'] if len(l['components']) == 2)['components'].pop(); cases['omitted_regular_component'] = (u, v, b)
b = copy.deepcopy(H); comp = next(c for l in b['levels'] for c in l['components']); comp['witness_polynomial_value'] = -comp['witness_polynomial_value']; cases['changed_sign_witness'] = (u, v, b)
b = copy.deepcopy(H); b['incidence'][0]['lower_polynomial_value'] *= -1; cases['changed_incidence_sign'] = (u, v, b)
b = copy.deepcopy(H); b['actual_barcode']['finite'][0]['birth_root'] = 0; cases['swapped_elder_birth'] = (u, v, b)
b = copy.deepcopy(H); m = next(m for e in b['events'] for m in e['merges']); s, d = m['survivor_birth'], m['dying_births'][0]; m['survivor_birth'], m['dying_births'] = d, [s]; cases['swapped_elder_graph_decision'] = (u, v, b)
b = copy.deepcopy(H); bar = b['actual_barcode']['finite'][0]; bar['death_group'] = next(g['id'] for g in b['height_groups'] if g['id'] != bar['death_group']); cases['changed_death_group'] = (u, v, b)
b = copy.deepcopy(H); next(e for e in b['events'] if e['merges'])['merges'] = []; cases['omitted_merge_batch'] = (u, v, b)
b = copy.deepcopy(T); next(g for g in b['height_groups'] if len(g['root_ordinals']) == 2)['equality_witness'] = None; cases['missing_equality_witness'] = (Q(3, 2**33), Q(0), b)
b = copy.deepcopy(H); del b['checks']; cases['missing_selection_checks'] = (u, v, b)
b = copy.deepcopy(H); del b['critical_roots']['work']; cases['missing_roots_work'] = (u, v, b)
b = copy.deepcopy(T); b['critical_roots']['sturm_chains'] = []; cases['erased_sturm_chains'] = (Q(3, 2**33), Q(0), b)
b = copy.deepcopy(H); f = list(b['critical_roots']['roots'][0]['factor']); f[-1] = 1; b['critical_roots']['roots'][0]['factor'] = tuple(f); cases['integer_coefficient'] = (u, v, b)
# Additional tampers NOT in the suite: omit a critical root entirely (self-consistently), widen an isolator.
b = copy.deepcopy(H)
for lvl in b['levels']:
    pass
cases_extra = {}
for name, (uu, vv, cert) in cases.items():
    r = selector.verify_h0(uu, vv, cert, budget=budget())
    print(f'{name:35s} status={r["status"]:20s} first_false={first_false(r)}')
