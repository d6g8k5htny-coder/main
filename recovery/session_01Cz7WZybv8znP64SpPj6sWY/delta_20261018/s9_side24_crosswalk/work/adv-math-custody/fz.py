import sys, re, hashlib, os, glob
D=sys.argv[1]
for p in sorted(glob.glob(D+'/*.txt')):
    raw=open(p,'rb').read()
    n=os.path.basename(p)[:-4]
    bom = raw.startswith(b'\xef\xbb\xbf')
    crlf = raw.count(b'\r\n')
    t=raw.decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
    L=t.split('\n')
    begs=[i for i,l in enumerate(L) if re.fullmatch(r'BEGIN_[A-Z0-9_]+_FROZEN_BODY',l.strip())]
    bb=re.findall(r'(?m)^\s*BODY_BYTES:\s*([\d,]+)',t); bs=re.findall(r'(?m)^\s*BODY_SHA256:\s*([0-9a-f]{64})',t)
    pol=[i+1 for i,l in enumerate(L) if 'FREEZE POLICY' in l.upper() or 'FREEZE_POLICY' in l.upper()]
    if not begs:
        print(n,'NOBEGIN','bb',bb,'bs',[x[:8] for x in bs],'policy_lines',pol[:3]); continue
    b=begs[0]; tag=L[b].strip()[6:]; e=[i for i,l in enumerate(L) if l.strip()=='END_'+tag]
    body='\n'.join(L[b+1:e[0]]).rstrip('\n')+'\n'
    by=body.encode(); h=hashlib.sha256(by).hexdigest()
    c=re.sub(r'\n\n\n+','\n\n',body).encode(); hc=hashlib.sha256(c).hexdigest()
    print(n,'beg',b+1,'end',e[0]+1,'nbeg',len(begs),'nend',len(e),'bytes',len(by),'decl',bb,'sha',h[:8],'decl',[x[:8] for x in bs],'strict',h in bs,'collapse',len(c),hc in bs,'policy',pol[:3],'bom',bom,'crlf',crlf)
