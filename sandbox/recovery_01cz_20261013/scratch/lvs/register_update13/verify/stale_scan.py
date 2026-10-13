"""Scan every string leaf of the worktree REGISTER for possibly stale counts / phrases after round 6."""
import json, re, sys
REPO = sys.argv[1]
new = json.load(open(REPO + '/specifications/lean_statements_20261010/REGISTER.json'))
def leaves(o, path=''):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, f'{path}/{k}')
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, f'{path}[{i}]')
    else: yield path, o
pats = [r'SOURCE_CHECK_PASS: \d+', r'\b\d+ distinct registered', r'\b(75|74|67|13|64|141|78|77) (inventory|theorems|distinct|registered|modules|Props)',
        r'every theorem of this row', r'all \d+ theorems', r'no theorem', r'ImageBound', r'CovarianceOrdering', r'EntrywiseCovarianceBound',
        r'imageBound', r'covarianceOrdering', r'entrywiseCovarianceBound', r'Side24Images', r'round 6', r'kernel-checked', r'iteratedFDeriv_comp_perm']
for p, v in leaves(new):
    if not isinstance(v, str): continue
    for pat in pats:
        for m in re.finditer(pat, v):
            s = max(0, m.start() - 90); e = min(len(v), m.end() + 90)
            print(f'{p} [{pat}] ...{v[s:e]}...')
