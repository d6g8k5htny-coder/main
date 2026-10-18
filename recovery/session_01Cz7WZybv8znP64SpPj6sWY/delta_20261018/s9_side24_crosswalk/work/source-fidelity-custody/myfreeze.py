import sys, re, hashlib, os, glob
D = sys.argv[1]
for p in sorted(glob.glob(os.path.join(D,'*.txt'))):
    n = os.path.basename(p)[:-4]
    raw = open(p,'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    lone_cr = len(re.findall(rb'\r(?!\n)', raw))
    t = raw.decode('utf-8')
    if t.startswith('﻿'): t = t[1:]
    t = t.replace('\r\n','\n').replace('\r','\n')
    L = t.split('\n')
    begins = [i for i,l in enumerate(L) if re.fullmatch(r'BEGIN_[A-Z0-9_]+_FROZEN_BODY', l)]
    bb = [ (i+1,l) for i,l in enumerate(L) if re.match(r'^BODY_BYTES:', l)]
    bs = [ (i+1,l) for i,l in enumerate(L) if re.match(r'^BODY_SHA256:', l)]
    fp = [ i+1 for i,l in enumerate(L) if l.strip()=='FREEZE POLICY']
    if not begins:
        print(f"{n}\tbom={bom}\tloneCR={lone_cr}\tNO_BEGIN\tBODY_BYTES={bb}\tBODY_SHA={bs}\tFREEZE_POLICY_lines={fp}")
        continue
    out=[]
    for bi in begins:
        tag = L[bi][len('BEGIN_'):]
        endl = 'END_'+tag
        ei = [i for i,l in enumerate(L) if l==endl]
        nb = sum(1 for l in L if l==L[bi])
        body = '\n'.join(L[bi+1:ei[0]]).rstrip('\n')+'\n'
        b = body.encode('utf-8'); h = hashlib.sha256(b).hexdigest()
        c = re.sub(r'\n\n\n+','\n\n',body).encode('utf-8'); hc = hashlib.sha256(c).hexdigest()
        decl_b = [int(x[1].split(':')[1]) for x in bb]; decl_s=[x[1].split(':')[1].strip() for x in bs]
        out.append(f"{n}\tbom={bom}\tloneCR={lone_cr}\tB=L{bi+1}(x{nb})\tE=L{ei[0]+1 if ei else None}(x{len(ei)})\tstrict={len(b)}/{h[:8]}..{h[-3:]}\tcollapse={len(c)}/{hc[:8]}\tdecl={bb}{[s[:8] for s in decl_s]}\tSTRICT={'MATCH' if h in decl_s else 'MISMATCH'}\tCOLLAPSE={'MATCH' if hc in decl_s else 'no'}\tFP={fp}")
    print('\n'.join(out))
