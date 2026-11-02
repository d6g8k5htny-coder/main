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
 'proposal under adopted location': ('no proposal sits', lambda c: c['proposed_quarantines'][3].__setitem__('location', [adopted_loc(c)])),
 'T4 statement altered': ('verbatim', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4'))['source_statements'][0].__setitem__('text', 'x')),
 'T4 binding bytes': ('file bindings', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4'))['source_binding']['files'][0].__setitem__('bytes', 1)),
 'T4 amendment target unadopted': ('byte pins', lambda c: first(c, 'proposed_pinning_amendments', lambda r: 'target_records' in r).__setitem__('target_records', ['Q-NOPE'])),
 'reconciled blob changed': ('reconciled federation', lambda c: first(c, 'proposed_quarantines_federation', lambda r: 'reconciliation' in r)['reconciliation']['shared_files'][0].__setitem__('git_blob', '0'*40)),
 'archive member sha changed (T7)': ('archive-member bindings', lambda c: first(c, 'accounting_dispositions', lambda r: (r.get('source_binding') or {}).get('members'))['source_binding']['members'][0].__setitem__('sha256', '0'*64)),
 'identical_copies sha changed (T6-001)': ('file bindings', lambda c: first(c, 'proposed_quarantines_federation', lambda r: r.get('identical_copies'))['identical_copies'][0]['source_binding']['files'][0].__setitem__('sha256', '0'*64)),
 'pin to a record naming only the parts': ('byte pins', lambda c: c['proposed_byte_pins'][0].update({'record': 'Q-3006', 'path': 'frontiers/README.md', 'git_blob': '855ef2db8d095342e715e943420613dbc365922b', 'bytes': 2269, 'sha256': '3a931a7ec4e9283b6a636d5cf18225afa59923bfa5b8f9490c43c35e1edea05b'})),
 'P15 namespace wrong': ('P15', lambda c: (lambda r: r.__setitem__('declaration', 'Bogus.Elsewhere.' + r['declaration'].rsplit('.', 1)[-1]))(first(c, 'proposed_closed_candidates', lambda r: r.get('candidate_kind') == 'HOSTED_GATE_DECLARATION'))),
 'proposal under a prose-located adopted path': ('no proposal sits', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4')).__setitem__('location', ['d6g8k5htny-coder/main@d2b0216e15d545c47f25b6a9265dd3298971e9fe:experiments/universality/quartic_cusp_field/PROOF.md'])),
 'federation row in a region without declaring it': ('no proposal sits', lambda c: first(c, 'proposed_quarantines_federation', lambda r: r['id'] == 'QP-C54-T6-007').pop('claim_overlaps')),
 'reconciliation lists overlap': ('reconciled federation', lambda c: (lambda rec: rec['federation_only_files'].append(dict(rec['shared_files'][0])))(first(c, 'proposed_quarantines_federation', lambda r: 'reconciliation' in r)['reconciliation'])),
 'erratum value altered': ('errata items', lambda c: first(c, 'proposed_quarantines', lambda r: 'E-T2-N4' in (r.get('errata_applied') or [])).__setitem__('floating_point', 'NONE altered')),
 'erratum tag dropped': ('errata items', lambda c: first(c, 'proposed_pinning_amendments', lambda r: 'E-R-T5-003-KIND' in (r.get('errata_applied') or [])).__setitem__('errata_applied', [])),
 'section erratum moved to a sibling row': ('errata items', lambda c: (lambda src, dst: (dst.__setitem__('errata_applied', ['E-T2-N2']), src.__setitem__('errata_applied', [])))(
     first(c, 'accounting_dispositions', lambda r: 'E-T2-N2' in (r.get('errata_applied') or [])),
     first(c, 'accounting_dispositions', lambda r: r.get('tranche') == 't2' and 'E-T2-N2' not in (r.get('errata_applied') or [])))),
 'bogus errata tag': ('errata items', lambda c: c['proposed_quarantines'][0].__setitem__('errata_applied', (c['proposed_quarantines'][0].get('errata_applied') or []) + ['E-NOT-AN-ITEM'])),
 'amendment with both target forms': ('byte pins', lambda c: first(c, 'proposed_pinning_amendments', lambda r: 'target_records' in r).update({'target_record': 'Q-1161', 'target_records': ['Q-NOPE']})),
 'P15 line negative': ('P15', lambda c: first(c, 'proposed_closed_candidates', lambda r: r.get('candidate_kind') == 'HOSTED_GATE_DECLARATION').__setitem__('line', -30)),
 'main row self-declares an engineering region': ('no proposal sits', lambda c: first(c, 'proposed_quarantines', lambda r: r['id'].startswith('QP-C54-T4')).update({'location': ['d6g8k5htny-coder/main@d2b0216e15d545c47f25b6a9265dd3298971e9fe:tests/test_living_project_validation.py'], 'claim_overlaps': [{'record': 'Q-9002', 'relation': 'INSIDE_ADOPTED_ENGINEERING_REGION', 'note': 'x'}]})),
 'pin to a negated mention': ('byte pins', lambda c: c['proposed_byte_pins'][1].update({'record': 'Q-1143', 'path': 'formal/manifest.json', 'git_blob': '894b030e383eaf1f6fbf608b69a9176396848589', 'bytes': 14855, 'sha256': 'ba7d11e43574e39da3c759e0501399a6dcd1b7da32b157a4a6b01f5c133b3062', 'commit': '8151f592918a845bf19eb8eafaf8444540a32734', 'repository': 'd6g8k5htny-coder/Math-'})),
 'gz member renamed': ('archive-member bindings', lambda c: first(c, 'accounting_dispositions', lambda r: (r.get('source_binding') or {}).get('files') and r['source_binding']['files'][0]['path'].endswith('.json.gz') and r['source_binding'].get('members'))['source_binding']['members'][0].__setitem__('member', 'no/such/member.json')),
 'pin subsumed_by wrong': ('byte pins', lambda c: first(c, 'proposed_byte_pins', lambda r: r.get('subsumed_by')).__setitem__('subsumed_by', ['QP-C54-T4-027'])),
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
