"""Check the three new statement entries: anchors vs registered def_prop entries; does_not_claim vs
the theorem docstring's 'Does not establish:' field; compare structure with the round-5 entries."""
import json, re, sys
REPO = sys.argv[1]
reg = json.load(open(REPO + '/specifications/lean_statements_20261010/REGISTER.json'))
row = next(c for c in reg['claims'] if c['claim_id'] == 'side24-coefficient')
src = open(REPO + '/specifications/lean_statement_proofs_20261011/SpecificationProofs/Side24Images.lean').read()
src_cov = open(REPO + '/specifications/lean_statement_proofs_20261011/SpecificationProofs/Side24Covariance.lean').read()

def docfield(text, thm):
    # docstring immediately preceding `theorem thm :`
    m = re.search(r'/--((?:(?!/--).)*?)-/\s*theorem ' + re.escape(thm) + r'\b', text, re.S)
    doc = m.group(1)
    i = doc.index('Does not establish:')
    body = doc[i + len('Does not establish:'):]
    return ' '.join(body.split()), ' '.join(doc.split())

props = {'imageBound': 'ImageBound', 'entrywiseCovarianceBound': 'EntrywiseCovarianceBound',
         'covarianceOrdering': 'CovarianceOrdering'}
for st in row['statements']:
    print(st.get('declaration'), st.get('kind'), st.get('formal_progress'), st.get('module'))
print()
for thm, prop in props.items():
    decl = 'UniversalLaw.Proofs.Side24Images.' + thm
    e = [s for s in row['statements'] if s.get('declaration') == decl]
    assert len(e) == 1, (decl, len(e))
    e = e[0]
    dp = [s for s in row['statements'] if s.get('declaration') == 'UniversalLaw.Spec.Side24.' + prop]
    print(thm, 'keys', list(e.keys()))
    print('  def_prop entries', len(dp), [ (s.get('kind'), s.get('formal_progress')) for s in dp])
    print('  anchor equal:', e['anchor'] == dp[0]['anchor'], repr(e['anchor']), repr(dp[0]['anchor']))
    f, whole = docfield(src, thm)
    print('  does_not_claim == docstring field:', e['does_not_claim'] == f)
    if e['does_not_claim'] != f:
        import difflib
        for op in difflib.SequenceMatcher(None, e['does_not_claim'], f, autojunk=False).get_opcodes():
            if op[0] != 'equal':
                print('   ', op, repr(e['does_not_claim'][op[1]-20:op[2]+20]), repr(f[op[3]-20:op[4]+20]))
# round-5 comparison
r5 = [s for s in row['statements'] if s.get('declaration') == 'UniversalLaw.Proofs.Side24Covariance.referenceCovarianceLowerBound'][0]
f5, _ = docfield(src_cov, 'referenceCovarianceLowerBound')
print('\nround5 keys', list(r5.keys()), 'dnc==docfield', r5['does_not_claim'] == f5)
dp5 = [s for s in row['statements'] if s.get('declaration') == 'UniversalLaw.Spec.Side24.ReferenceCovarianceLowerBound'][0]
print('round5 anchor equal', r5['anchor'] == dp5['anchor'])
print('round5 package', r5['package'])
