#!/usr/bin/env python3
"""verify_docs: adversarial doc checks (table shape, run-id binding, stale counts, vocabulary, dagger set).
Scientific effect: NONE. Read-only."""
import re, sys, json
from pathlib import Path
M = Path('<SCRATCH>/repos/math')
MAIN = Path('<SCRATCH>/lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md')
DOCS = {
 'proofs/README': M/'specifications/lean_statement_proofs_20261011/README.md',
 'proofs/SCOPE': M/'specifications/lean_statement_proofs_20261011/SCOPE.md',
 'stmts/README': M/'specifications/lean_statements_20261010/README.md',
 'stmts/SCOPE': M/'specifications/lean_statements_20261010/SCOPE.md',
 'main/SPEC': MAIN,
}
MERGE = {'38116277597':'c4c66bb8','38121713840':'5b198699','38122386459':'866136cc','38124262242':'318ff194',
         '38149888699':'4363b692','38149524759':'3306233b','38150363063':'86a6e8c8'}
HEADS = {'38116277597':'00d2ac6e','38121713840':'910809d6','38122386459':'d9b8dd33','38124262242':'79731b96',
         '38149888699':'0dcb0016','38149524759':'88c21058','38150363063':'12e072d6'}
NEW13 = {'thetaSummable','derivativePairingBound','derivativeBoundAtLeastOne','derivativeBoundAtZero','referenceOddAndPinBlocks',
         'referencePinDensityProduct','integrandRatioBounds','latticeImageSumBound','densityComparison',
         'coneMomentD1Identity','traceMoments','rayleighSquareLaw','rareClusterPoissonObstructionO'}
STALE = [r'\b59 of the 141', r'\b60 theorems', r'\b59 registered Props', r'\bother 82\b', r'\b82 fall', r'\b70 entries', r'\b142 `?proved`? entries', r'\(339\)', r'\b142\b.*proved']
report = {}
def cells(row):
    # split a markdown table row on unescaped pipes
    parts = re.split(r'(?<!\\)\|', row.strip())
    if parts and parts[0]=='' : parts = parts[1:]
    if parts and parts[-1]=='': parts = parts[:-1]
    return parts
for name, p in DOCS.items():
    txt = p.read_text(encoding='utf-8')
    r = {'bytes': len(txt.encode()), 'table_problems': [], 'unbound_runs': [], 'stale': [], 'vocab': [], 'dagger_names': set(), 'no_head_runs': []}
    lines = txt.split('\n')
    # tables
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith('|') and i+1 < len(lines) and re.match(r'^\s*\|?\s*:?-{3,}', lines[i+1]):
            n = len(cells(lines[i])); j = i+2
            if len(cells(lines[i+1])) != n: r['table_problems'].append((i+1, 'separator width', len(cells(lines[i+1])), n))
            while j < len(lines) and lines[j].lstrip().startswith('|'):
                if len(cells(lines[j])) != n: r['table_problems'].append((j+1, 'row width', len(cells(lines[j])), n, lines[j][:80]))
                j += 1
            i = j
        else:
            i += 1
    # paragraphs & cells for run binding
    units = []
    for para in re.split(r'\n\s*\n', txt):
        if para.lstrip().startswith('|'):
            for row in para.split('\n'):
                for c in cells(row): units.append(('cell', c))
        else:
            units.append(('para', para))
    for kind, u in units:
        for run in set(re.findall(r'\b(38\d{9})\b', u)):
            if run in MERGE:
                if MERGE[run] not in u: r['unbound_runs'].append((run, kind, u[:160].replace('\n',' ')))
                if HEADS[run] not in u: r['no_head_runs'].append((run, kind, u[:120].replace('\n',' ')))
            else:
                r['unbound_runs'].append((run, kind, 'UNKNOWN RUN ID: ' + u[:160].replace('\n',' ')))
    for pat in STALE:
        for m in re.finditer(pat, txt):
            s = max(0, m.start()-80); r['stale'].append((pat, txt[s:m.end()+60].replace('\n',' ')))
    # vocabulary: words in context
    for pat in [r'\bclosed\b', r'\baccepted\b', r'\bindependent\b', r'kernel-checked', r'\bproved\b']:
        for m in re.finditer(pat, txt):
            s = max(0, m.start()-70); r['vocab'].append((pat, txt[s:m.end()+70].replace('\n',' ')))
    for m in re.finditer(r'`?([A-Za-z0-9]+)\.([a-zA-Z0-9]+)`?†|`([a-zA-Z0-9]+) \(proved\)†|proof:`([A-Za-z0-9]+)\.([a-zA-Z0-9]+)` \(proved\)†', txt):
        nm = m.group(2) or m.group(3) or m.group(5)
        r['dagger_names'].add(nm)
    r['dagger_count'] = txt.count('†')
    r['dagger_names'] = sorted(r['dagger_names'])
    r['dagger_not_in_new13'] = sorted(set(r['dagger_names']) - NEW13)
    r['new13_without_dagger'] = sorted(NEW13 - set(r['dagger_names'])) if r['dagger_count'] else 'n/a'
    report[name] = r
for name, r in report.items():
    print('=====', name, r['bytes'], 'bytes; dagger marks', r['dagger_count'])
    print(' table_problems:', r['table_problems'])
    print(' unbound_runs:', r['unbound_runs'])
    print(' runs without PR head in same unit:', len(r['no_head_runs']), [x[:2] for x in r['no_head_runs']][:10])
    print(' stale:', r['stale'])
    print(' dagger names:', r['dagger_names'], 'not-in-13:', r['dagger_not_in_new13'], 'missing:', r['new13_without_dagger'])
    vc = {}
    for pat, ctx in r['vocab']:
        vc.setdefault(pat, []).append(ctx)
    for pat in vc: print(' vocab', pat, len(vc[pat]))
json.dump({k:{kk:(vv if kk!='vocab' else None) for kk,vv in v.items()} for k,v in report.items()}, open(sys.argv[1],'w'), indent=1, default=str)
json.dump({k:v['vocab'] for k,v in report.items()}, open(sys.argv[2],'w'), indent=1)
