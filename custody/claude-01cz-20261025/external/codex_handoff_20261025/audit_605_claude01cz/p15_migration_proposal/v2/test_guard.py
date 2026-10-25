# Negative/positive regression tests of migration_guard (my code only). Includes the
# reviewer's counterexamples (605/6404282041 items 1-4).
import json, copy
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from migration_guard import check_link_migration
class E(ValueError): pass
O = json.load(open('../work/dataset_original.json')); P = json.load(open('../work/dataset_successor_PROPOSAL.json'))
def obs(d): return {o['conclusion']: o for o in d['obligations']}
def run(name, d, expect_ok):
    try: check_link_migration(d, obs(d), E); ok = True; msg = 'accepted'
    except E as e: ok = False; msg = str(e)
    print(f"{'PASS' if ok == expect_ok else 'FAIL'}  {name}: {msg}"); return ok == expect_ok
res = []
res.append(run('original dataset (no migration)', O, True))
res.append(run('successor proposal', P, True))
d = copy.deepcopy(P); ob = obs(d); ob['P15-T3']['premises'] = sorted(ob['P15-T3']['premises'] + ['P15-E1'])
m2 = dict(d['metadata']['documented_link_migrations'][0]); m2['id'] = 'MIG-DUP'; d['metadata']['documented_link_migrations'].append(m2)
res.append(run('CE1: extra undocumented edge + duplicate-credit migration', d, False))
d = copy.deepcopy(P); ob = obs(d); ob['P15-T1']['premises'] = sorted(p for p in ob['P15-T1']['premises'] if p != 'P15-E2'); ob['P15-T3']['premises'] = sorted(ob['P15-T3']['premises'] + ['P15-E1'])
res.append(run('CE2: remove old edge, add other edge, keep count 109', d, False))
d = copy.deepcopy(P); d['metadata']['documented_link_migrations'][0]['link_count_before'] = 900; d['metadata']['documented_link_migrations'][0]['link_count_after'] = 901
res.append(run('CE3: 900->901 counts', d, False))
d = copy.deepcopy(P); d['metadata']['documented_link_migrations'][0]['status'] = 'WHATEVER'
res.append(run('CE4: arbitrary migration status', d, False))
d = copy.deepcopy(O); ob = obs(d); ob['P15-T1']['premises'] = sorted(ob['P15-T1']['premises'] + ['P15-T2'])
res.append(run('undocumented 109th edge (no migration entry)', d, False))
d = copy.deepcopy(P); ob = obs(d); ob['P15-T1']['premises'] = sorted(p for p in ob['P15-T1']['premises'] if p != 'P15-T2')
res.append(run('migration entry present but edge absent', d, False))
print('ALL EXPECTED' if all(res) else 'SOME UNEXPECTED', f'({sum(res)}/{len(res)})')
