#!/usr/bin/env python3
"""Candidate CLOSED-index rows for the P15 targets landed by main#740. Usage: MAIN_CLONE STAGING_CLOSED_JSON OUT"""
import hashlib, json, re, subprocess, sys
MERGE = 'ec6962ff5fafe04364e094a954896dc1e62941f5'   # main merge of #740; tree b482cbca = tested head 52cea4c5
TESTED_HEAD = '52cea4c59bb6a906862b2685645da64934f21141'
main_c, closed_p, out = sys.argv[1:4]
git = lambda *a: subprocess.run(['git', '-C', main_c, *a], capture_output=True, check=True).stdout
assert git('rev-parse', f'{MERGE}^{{tree}}') == git('rev-parse', f'{TESTED_HEAD}^{{tree}}')
manifest = json.loads(git('show', f'{MERGE}:formal/manifest.json'))
closed = json.load(open(closed_p))['entries']
template = next(e for e in closed if e['id'] == 'K-MAIN-001')
indexed = {(e['repository'], e['declaration']) for e in closed}
rows = []
for t in manifest['targets']:
    if not t['module'].startswith('UniversalLaw/P15/'):
        continue
    path = 'formal/' + t['module']
    data = git('show', f'{MERGE}:{path}')
    short = t['name'].rsplit('.', 1)[-1]
    lines = data.decode().split('\n')
    hits = [i + 1 for i, l in enumerate(lines) if re.match(rf'\s*(theorem|lemma)\s+{re.escape(short)}\b', l)]
    assert len(hits) == 1, (t['name'], hits)
    rows.append({
        'id': f'K-MAIN-P15-{len(rows) + 1:03d}', 'declaration': t['name'], 'status': 'CANDIDATE_CLOSED_PENDING_INDEX_RECEIPT',
        'scope': template['scope'], 'repository': 'main', 'source_commit': MERGE, 'tested_head': TESTED_HEAD,
        'path': path, 'line': hits[0], 'source_sha256': hashlib.sha256(data).hexdigest(),
        'axioms': None,
        'axiom_gate': ('The hosted formal / formal-evidence job 124889085814 (run 41320377025) on the tested head succeeded; its '
                       'gate fails closed unless every manifest target passes the transitive #print axioms audit against '
                       f'allowed_axioms {manifest["allowed_axioms"]}. The exact per-declaration set is in the run artifact '
                       '(axioms.log), which is not read here, so axioms is null until it is.'),
        'receipt_hosted': {'run_id': 41320377025, 'job_id': 124889085814, 'artifact': 'formal-verification-41320377025',
                           'artifact_id': 13088624355,
                           'artifact_sha256': 'd04efcb121a60389cd5bc8ae8c3396f9b23af3420c52c52eaa63ee96536bcd7d'},
        'vacuity_boundary': template['vacuity_boundary'], 'title': t['title'],
        'informal_anchor': t.get('informal_anchor'), 'does_not_claim': t.get('does_not_claim'),
        'already_indexed': ('main', t['name']) in indexed})
json.dump({'schema': 'claude54.p15-closed-candidates.v1', 'merge_commit': MERGE, 'tested_head': TESTED_HEAD,
           'lean_toolchain': manifest['lean_toolchain'], 'alignment_status': manifest['alignment_status'],
           'rows': rows}, open(out, 'x'), indent=1, ensure_ascii=False)
print(len(rows), 'rows;', sum(r['already_indexed'] for r in rows), 'already indexed;', manifest['lean_toolchain'], manifest['alignment_status'])
