#!/usr/bin/env python3
"""Negative controls for validate_candidate_v2.py: each mutant must make it exit 1 on the named check.
Usage: STAGING CANDIDATE RECOVERY_DIR CLONES_JSON"""
import copy, json, os, subprocess, sys, tempfile
staging, cand_path, rec, clones = sys.argv[1:5]
base = json.load(open(cand_path))
HERE = os.path.dirname(os.path.abspath(__file__))
def first(c, key, pred=lambda r: True): return next(r for r in c[key] if pred(r))
def adopted_loc(c):
    for r in c['quarantines']:
        for l in r['location'] if isinstance(r['location'], list) else [r['location']]:
            if isinstance(l, str) and l.startswith('d6g8k5htny-coder/Math-@') and '/frontiers/' not in l and ':frontiers/' in l:
                return l.split('#')[0]
M = {
 'staging value changed': ('additive-only', lambda c: c['quarantines'][0].__setitem__('title', 'x')),
 'proposal edited': ('fresh build', lambda c: c['proposed_quarantines'][100].__setitem__('exact_boundary', 'x')),
 'accounting row dropped': ('fresh build', lambda c: c['accounting_dispositions'].pop()),
 'count wrong': ('fresh build', lambda c: c['proposal_provenance']['counts'].__setitem__('proposed_byte_pins', 40)),
 'required field emptied (T3)': ('required keys', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T3')).__setitem__('blocking_reason', ' ')),
 'owner set (T6)': ('required keys', lambda c: c['proposed_quarantines_federation'][0].__setitem__('owner', 'x')),
 'fp unclassified (T5)': ('required keys', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T5')).__setitem__('floating_point', 'maybe')),
 'duplicate id': ('ids unique', lambda c: c['proposed_quarantines_federation'][1].__setitem__('id', 'Q-1001')),
 'federation binding sha': ('file bindings', lambda c: c['proposed_quarantines_federation'][2]['source_binding']['files'][0].__setitem__('sha256', '0'*64)),
 'lean binding bytes (T5)': ('file bindings', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T5'))['source_binding']['lean_files'][0].__setitem__('bytes', 1)),
 'accounting binding blob': ('file bindings', lambda c: first(c, 'accounting_dispositions', lambda r: (r.get('source_binding') or {}).get('files'))['source_binding']['files'][0].__setitem__('git_blob', '0'*40)),
 'T3 statement altered': ('verbatim', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T3'))['source_statements'][0].__setitem__('text', 'x')),
 'T5 statement shifted': ('verbatim', lambda c: (lambda st: st.__setitem__('line_start', st['line_start'] + 1))(first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T5'))['source_statements'][0])),
 'location missing (T6)': ('location', lambda c: c['proposed_quarantines_federation'][0].__setitem__('location', [c['proposed_quarantines_federation'][0]['location'][0] + '_nope'])),
 'pin sha changed': ('byte pins', lambda c: c['proposed_byte_pins'][5].__setitem__('sha256', '0'*64)),
 'pin target wrong': ('byte pins', lambda c: c['proposed_byte_pins'][0].__setitem__('record', 'Q-1001')),
 'P15 line shifted': ('P15', lambda c: first(c, 'proposed_closed_candidates', lambda r: r.get('candidate_kind') == 'HOSTED_GATE_DECLARATION').__setitem__('line', 1)),
 'proposal under adopted location': ('no new proposal', lambda c: c['proposed_quarantines'][3].__setitem__('location', [adopted_loc(c)])),
 'T4 statement altered': ('verbatim', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4'))['source_statements'][0].__setitem__('text', 'x')),
 'T4 binding bytes': ('file bindings', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4'))['source_binding']['files'][0].__setitem__('bytes', 1)),
 'T4 amendment target unadopted': ('byte pins', lambda c: first(c, 'proposed_pinning_amendments', lambda r: 'target_records' in r).__setitem__('target_records', ['Q-NOPE'])),
 'reconciled blob changed': ('reconciled federation', lambda c: first(c, 'proposed_quarantines_federation', lambda r: 'reconciliation' in r)['reconciliation']['shared_files'][0].__setitem__('git_blob', '0'*40)),
}
ok = 0
with tempfile.TemporaryDirectory() as d:
    for name, (check, fn) in M.items():
        c = copy.deepcopy(base); fn(c)
        path = os.path.join(d, 'm.json')
        open(path, 'w', encoding='utf-8').write(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
        r = subprocess.run([sys.executable, '-I', '-B', os.path.join(HERE, 'validate_candidate_v2.py'), staging, path, rec, clones], capture_output=True, text=True)
        fails = [l for l in r.stdout.split('\n') if l.startswith('FAIL ')]
        killed = r.returncode == 1 and any(check in l for l in fails)
        ok += killed
        print(('KILLED ' if killed else 'SURVIVED ') + name, '|', [l[5:45] for l in fails])
print(f'{ok}/{len(M)} mutants killed on the intended check')
sys.exit(0 if ok == len(M) else 1)
