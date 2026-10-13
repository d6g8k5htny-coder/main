import json, re, sys
root = sys.argv[1]
new = json.load(open(root + '/specifications/lean_statements_20261010/REGISTER.json'))
st = new['claims'][0]['statements']
for idx, mod, decl in [(72, 'Side24Covariance', 'referenceCovarianceLowerBound'), (73, 'Side24ConeMoment', 'coneMomentD2Identity')]:
    src = open(f'{root}/specifications/lean_statement_proofs_20261011/SpecificationProofs/{mod}.lean').read()
    # find docstring preceding theorem decl
    m = re.search(r'/--((?:(?!/--).)*?)-/\s*theorem ' + decl + r'\b', src, re.S)
    doc = m.group(1)
    dm = re.search(r'Does not establish:(.*)$', doc, re.S)
    dne = ' '.join(dm.group(1).split())
    reg = st[idx]['does_not_claim']
    print(idx, 'verbatim-equal:', dne == reg, len(dne), len(reg))
    if dne != reg:
        import difflib
        for l in difflib.ndiff([dne], [reg]): print(l[:400])
