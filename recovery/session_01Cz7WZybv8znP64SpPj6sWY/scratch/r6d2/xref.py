import re,sys
src=open(sys.argv[1],encoding='utf-8').read()
# strip comments: block comments /- ... -/ (non-nested enough here) and line comments
def strip(s):
    out=[];i=0;depth=0
    while i<len(s):
        if s.startswith('/-',i):
            depth+=1;i+=2;continue
        if depth and s.startswith('-/',i):
            depth-=1;i+=2;continue
        if depth: i+=1;continue
        if s.startswith('--',i):
            j=s.find('\n',i); i=j if j>=0 else len(s); continue
        out.append(s[i]);i+=1
    return ''.join(out)
code=strip(src)
docs=src  # full
pat=re.compile(r'(?:UniversalLaw\.Proofs\.)?(Side24(?:Analysis|Covariance|Elementary|Interfaces|ConeMoment)(?:\.Aux)?\.[A-Za-z_][A-Za-z0-9_\']*)')
used=sorted(set(pat.findall(code)))
print('USED in code:')
for u in used: print('  ',u)
# header block
hdr=src[src.index('/-!'):src.index('-/',src.index('/-!'))]
prov=hdr[hdr.index('Provenance:'):hdr.index('Remark recorded')]
print('PROVENANCE text names:')
names=re.findall(r'`([^`]+)`',prov)
print('  ',names)
