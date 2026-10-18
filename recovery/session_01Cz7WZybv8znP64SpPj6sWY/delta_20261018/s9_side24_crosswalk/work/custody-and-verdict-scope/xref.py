import re,sys,os,glob
D=sys.argv[1]
decl={}
for p in glob.glob(D+'/*.txt'):
    t=open(p,'rb').read().decode('utf-8-sig').replace('\r\n','\n')
    n=os.path.basename(p)[:-4]
    m=re.search(r'^BODY_SHA256:\s*([0-9a-f]{64})',t,re.M); b=re.search(r'^BODY_BYTES:\s*(\d+)',t,re.M)
    if m: decl[m.group(1)]=(n,b.group(1) if b else None)
for p in sorted(glob.glob(D+'/*.txt')):
    t=open(p,'rb').read().decode('utf-8-sig').replace('\r\n','\n')
    n=os.path.basename(p)[:-4]
    for mm in re.finditer(r'(?:([\d,]+)\s*bytes\s*/\s*)?SHA-256[^0-9a-f]{0,30}([0-9a-f]{64})',t):
        h=mm.group(2); by=mm.group(1)
        tgt=decl.get(h)
        if tgt is None: continue
        if tgt[0]==n: continue
        ok = (by is None) or (by.replace(',','')==tgt[1])
        print(f"{n} cites {tgt[0]} {by or '-'} bytes / {h[:12]}... -> declared {tgt[1]} : {'CONSISTENT' if ok else 'BYTES-DIFFER'}")
