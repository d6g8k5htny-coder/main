#!/usr/bin/env python3
"""Negative controls: each mutant of the candidate must make validate_candidate.py exit 1 on the named check."""
import copy, json, subprocess, sys, tempfile, os
staging, cand_path, t1, t2, math = sys.argv[1:6]
base = json.load(open(cand_path))
def adopted_math_dir(c):
    for r in c['quarantines']:
        for loc in r['location'] if isinstance(r['location'], list) else [r['location']]:
            if isinstance(loc, str) and loc.startswith('d6g8k5htny-coder/Math-@') and 'frontiers/' in loc:
                return loc
M = {
 'staging value changed': ('additive-only', lambda c: c['quarantines'][0].__setitem__('title', c['quarantines'][0]['title'] + ' ')),
 'staging key removed': ('additive-only', lambda c: c.pop('reference_aliases')),
 'proposal record edited': ('proposals equal', lambda c: c['proposed_quarantines'][5].__setitem__('exact_boundary', 'x')),
 'non-obligation dropped': ('proposals equal', lambda c: c['proposed_non_obligations'].pop()),
 'count wrong': ('proposals equal', lambda c: c['proposal_provenance']['counts'].__setitem__('proposed_quarantines', 95)),
 'required field emptied': ('required fields', lambda c: c['proposed_quarantines'][3].__setitem__('blocking_reason', '  ')),
 'owner set': ('status QUARANTINED', lambda c: c['proposed_quarantines'][7].__setitem__('owner', 'claude54')),
 'fp unclassified': ('status QUARANTINED', lambda c: c['proposed_quarantines'][9].__setitem__('floating_point', 'maybe')),
 'duplicate id': ('ids unique', lambda c: c['proposed_quarantines'][1].__setitem__('id', 'Q-1001')),
 'binding sha changed': ('source bindings', lambda c: c['proposed_quarantines'][0]['source_binding']['files'][0].__setitem__('sha256', '0'*64)),
 'binding bytes changed': ('source bindings', lambda c: c['proposed_quarantines'][60]['source_binding']['files'][0].__setitem__('bytes', 1)),
 'statement altered': ('quoted statements', lambda c: c['proposed_quarantines'][0]['source_statements'][0].__setitem__('text', c['proposed_quarantines'][0]['source_statements'][0]['text'].replace('1', '2', 1))),
 'statement range shifted': ('quoted statements', lambda c: c['proposed_quarantines'][70]['source_statements'][0].__setitem__('line_start', c['proposed_quarantines'][70]['source_statements'][0]['line_start'] + 1)),
 'location missing': ('every proposal location', lambda c: c['proposed_quarantines'][2].__setitem__('location', [c['proposed_quarantines'][2]['location'][0] + '_nope'])),
 'pinning target wrong': ('pinning amendments', lambda c: c['proposed_pinning_amendments'][0].__setitem__('target_record', 'Q-1001')),
 'proposal under adopted location': ('no new proposal', lambda c: c['proposed_quarantines'][4].__setitem__('location', [adopted_math_dir(c).split('#')[0]])),
}
ok = 0
with tempfile.TemporaryDirectory() as d:
    for name, (check, fn) in M.items():
        c = copy.deepcopy(base)
        fn(c)
        path = os.path.join(d, 'mutant.json')
        with open(path, 'w', encoding='utf-8') as s:
            s.write(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
        r = subprocess.run([sys.executable, '-I', '-B', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_candidate.py'), staging, path, t1, t2, math], capture_output=True, text=True)
        fails = [l for l in r.stdout.split('\n') if l.startswith('FAIL ')]
        killed = r.returncode == 1 and any(check in l for l in fails)
        ok += killed
        print(('KILLED ' if killed else 'SURVIVED ') + name + f'  rc={r.returncode} fails={[l[5:40] for l in fails]}')
print(f'{ok}/{len(M)} mutants killed on the intended check')
sys.exit(0 if ok == len(M) else 1)
