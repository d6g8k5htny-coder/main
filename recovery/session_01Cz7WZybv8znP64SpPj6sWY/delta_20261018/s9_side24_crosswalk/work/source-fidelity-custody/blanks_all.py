import sys,re,glob,os,hashlib
D=sys.argv[1]; summ={}
for p in sorted(glob.glob(D+'/*.txt')):
    n=os.path.basename(p)[:-4]
    t=open(p,'rb').read().decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
    if not re.search(r'^BODY_SHA256:',t,re.M): continue
    L=t.split('\n'); bi=[i for i,l in enumerate(L) if re.fullmatch(r'BEGIN_[A-Z0-9_]+_FROZEN_BODY',l)][0]
    ei=L.index('END_'+L[bi][6:]); body='\n'.join(L[bi+1:ei]).rstrip('\n')+'\n'
    runs=set(len(m.group(1)) for m in re.finditer(r'\n(\n+)',body))
    ds=re.search(r'^BODY_SHA256:\s*([0-9a-f]{64})',t,re.M).group(1)
    st = hashlib.sha256(body.encode()).hexdigest()==ds
    co = hashlib.sha256(re.sub(r'\n\n\n+','\n\n',body).encode()).hexdigest()==ds
    k=('STRICT' if st else 'COLLAPSE' if co else 'NEITHER'); summ.setdefault(k,[]).append(n)
    if runs!={2}: print('NONUNIFORM', n, runs)
for k,v in summ.items(): print(k, len(v), ' '.join(x.replace('-v1.0','') for x in v))
