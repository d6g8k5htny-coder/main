import re, subprocess
REPO = '<SCRATCH>/repos/math_r5'
DOCS = ['specifications/lean_statements_20261010/README.md','specifications/lean_statements_20261010/SCOPE.md',
        'specifications/lean_statement_proofs_20261011/README.md','specifications/lean_statement_proofs_20261011/SCOPE.md']
def show(rev, path):
    return subprocess.run(['git','-C',REPO,'show',f'{rev}:{path}'],capture_output=True,check=True).stdout.decode('utf-8')
def check(text):
    lines=text.split('\n'); probs=[]; ntab=0; i=0
    while i < len(lines):
        if lines[i].startswith('|') and (i==0 or not lines[i-1].startswith('|')):
            ntab+=1
            hdr=len(re.findall(r'(?<!\\)\|',lines[i]))
            sep=lines[i+1] if i+1<len(lines) else ''
            if not re.fullmatch(r'\|(\s*:?-+:?\s*\|)+', sep): probs.append((i+1,'bad separator'))
            elif len(re.findall(r'\|',sep))!=hdr: probs.append((i+1,'sep cols'))
            j=i+2
            while j<len(lines) and lines[j].startswith('|'):
                n=len(re.findall(r'(?<!\\)\|',lines[j]))
                if n!=hdr: probs.append((j+1,f'cols {n} vs {hdr}'))
                if not lines[j].rstrip().endswith('|'): probs.append((j+1,'no trailing pipe'))
                j+=1
            i=j
        else: i+=1
    return ntab, probs
for d in DOCS:
    nt, pr = check(open(f'{REPO}/{d}',encoding='utf-8').read())
    ot, opr = check(show('9e305ea5', d))
    print(d, 'tables', ot, '->', nt, 'problems old', opr[:5], 'new', pr[:5])
    t=open(f'{REPO}/{d}',encoding='utf-8').read()
    print('   trailing ws lines:', sum(1 for l in t.split('\n') if l!=l.rstrip()), 'old:', sum(1 for l in show('9e305ea5',d).split('\n') if l!=l.rstrip()), 'ends-with-newline', t.endswith('\n'), 'nonascii-new-chars:', sorted(set(c for c in t if ord(c)>127)-set(c for c in show('9e305ea5',d) if ord(c)>127)))
