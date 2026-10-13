import subprocess, hashlib, sys
wt=sys.argv[1]
T='001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6'
C='d4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213'
out=subprocess.run(['git','-C',wt,'cat-file','--batch-all-objects','--unordered','--batch-check=%(objectname) %(objecttype) %(objectsize)'],capture_output=True,text=True,check=True).stdout
rows=[l.split() for l in out.splitlines()]
blobs=[r for r in rows if r[1]=='blob']
print("total objects", len(rows), "blobs", len(blobs), flush=True)
cand=[r[0] for r in blobs if int(r[2]) in (32809, 28073)]
print("blobs sized 32809/28073:", cand, flush=True)
for b in cand:
    body=subprocess.run(['git','-C',wt,'cat-file','blob',b],capture_output=True,check=True).stdout
    h=hashlib.sha256(body).hexdigest()
    print(b, len(body), h, "EXTERNAL-MATCH" if h==T else ("CORE-MATCH" if h==C else "no"), flush=True)
