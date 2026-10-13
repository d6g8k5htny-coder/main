import re, subprocess, sys, os
W = '<SCRATCH>/repos/math_r5'
RUNS = {'38116277597': ('00d2ac6e', 'c4c66bb8'), '38121713840': ('910809d6', '5b198699'), '38122386459': ('d9b8dd33', '866136cc'),
        '38124262242': ('79731b96', '318ff194'), '38149888699': ('0dcb0016', '4363b692'), '38149524759': ('88c21058', '3306233b'),
        '38150363063': ('12e072d6', '86a6e8c8'), '38180705952': ('48bfcbae', '9833d6d9'), '38187538744': ('be014794', '79d3964c'),
        '38195810463': ('c6ce0256', '1320c5b2')}
def units(text):
    out = []
    for para in re.split(r'\n\s*\n', text):
        if para.lstrip().startswith('|'):
            for line in para.split('\n'):
                for c in re.split(r'(?<!\\)\|', line):
                    out.append(c)
        else:
            out.append(para)
    return out
files = ['specifications/lean_statements_20261010/README.md', 'specifications/lean_statements_20261010/SCOPE.md',
         'specifications/lean_statement_proofs_20261011/README.md', 'specifications/lean_statement_proofs_20261011/SCOPE.md']
for f in files:
    new = open(os.path.join(W, f), encoding='utf-8').read()
    old = subprocess.run(['git', '-C', W, 'show', '4afcca6a:' + f], capture_output=True, text=True, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'}).stdout
    oldu = set(units(old))
    issues = []; changed_with_run = 0; allruns = set()
    for u in units(new):
        for r in set(re.findall(r'\b38\d{9}\b', u)):
            allruns.add(r)
            if r not in RUNS:
                issues.append(('unlisted run', r, u in oldu)); continue
            h, m = RUNS[r]
            if u not in oldu:
                changed_with_run += 1
            if h not in u or m not in u:
                # 'that run'-style references are checked by hand
                issues.append((r, 'head' if h not in u else '', 'merge' if m not in u else '', 'unchanged' if u in oldu else 'CHANGED', u[:160].replace('\n', ' ')))
    print(f, 'runs:', sorted(allruns), 'changed units with a run:', changed_with_run)
    for i in issues: print('   ', i)
