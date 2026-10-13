"""Independent per-paragraph / per-table-cell scan of the four docs (read-only)."""
import re, sys, json, subprocess
R = '<SCRATCH>/repos/math_r5'
DOCS = ['specifications/lean_statements_20261010/README.md', 'specifications/lean_statements_20261010/SCOPE.md',
        'specifications/lean_statement_proofs_20261011/README.md', 'specifications/lean_statement_proofs_20261011/SCOPE.md']
RUN11 = '38277458954'
def units(text):
    out = []
    para = []
    for ln, line in enumerate(text.split('\n'), 1):
        if line.startswith('|'):
            if para:
                out.append(('para', para[0][0], ' '.join(l for _, l in para))); para = []
            cells = re.split(r'(?<!\\)\|', line)[1:-1]
            for i, c in enumerate(cells):
                out.append((f'cell{i}', ln, c))
        elif line.strip() == '':
            if para:
                out.append(('para', para[0][0], ' '.join(l for _, l in para))); para = []
        else:
            para.append((ln, line))
    if para:
        out.append(('para', para[0][0], ' '.join(l for _, l in para)))
    return out
RUNS = {  # run -> (head, merge prefix)
 '38116277597': ('00d2ac6e', 'c4c66bb8'), '38121713840': ('910809d6', '5b198699'), '38122386459': ('d9b8dd33', '866136cc'),
 '38124262242': ('79731b96', '318ff194'), '38149888699': ('0dcb0016', '4363b692'), '38149524759': ('88c21058', '3306233b'),
 '38150363063': ('12e072d6', '86a6e8c8'), '38180705952': ('48bfcbae', '9833d6d9'), '38187538744': ('be014794', '79d3964c'),
 '38195810463': ('c6ce0256', '1320c5b2'), '38277458954': ('9e305ea5', '1376175c')}
for d in DOCS:
    text = open(f'{R}/{d}', encoding='utf-8').read()
    print('==', d, 'run11 total', text.count(RUN11), 'dagger', text.count('†'))
    for kind, ln, u in units(text):
        for run, (h, m) in RUNS.items():
            if run in u and not (h in u and m in u):
                print('  C422 FAIL', kind, ln, run)
        if RUN11 in u:
            forms = {'short(tested merge differing)': u.count('the tested merge differing from the head only outside both packets'),
                     'hdr(tested merge differing from PR head)': u.count('the tested merge differing from PR head 9e305ea5 only outside both packets'),
                     "long(the merge differs from its head)": u.count('the merge differs from its head only outside both packets'),
                     "proofREADME(tested merge's tree)": u.count("the tested merge's tree differs from the head's only outside both packets"),
                     'a0323b94': u.count('a0323b94'), 'outside': u.count('only outside both packets')}
            print(f'  {kind} L{ln}: run11 x{u.count(RUN11)}', {k: v for k, v in forms.items() if v})
            if forms['outside'] == 0:
                print('   !!! NO CAVEAT')
        for bad in ('the merge differing', 'whose tree differs', 'PR #440, whose', 'every theorem of this row receipted on PR head 9e305ea5',
                    'no hosted receipt', 'next hosted', 'their first receipt)', 'whose receipt the register binds', 'marked †'):
            if bad in u:
                print('  BAD', repr(bad), kind, ln)
