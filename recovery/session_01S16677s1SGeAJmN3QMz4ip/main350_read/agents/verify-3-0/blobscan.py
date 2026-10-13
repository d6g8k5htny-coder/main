import subprocess, hashlib, sys
T='001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6'
C='d4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213'
wt=sys.argv[1]
# all objects: packed + loose, regardless of reachability
out=subprocess.run(['git','-C',wt,'cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)'],capture_output=True,text=True,check=True).stdout
blobs=[l.split()[0] for l in out.splitlines() if l.split()[1]=='blob']
sizes={l.split()[0]:int(l.split()[2]) for l in out.splitlines() if l.split()[1]=='blob'}
print("total objects", len(out.splitlines()), "blobs", len(blobs))
cand=[b for b in blobs if sizes[b] in (32809,28073)]
print("blobs with size 32809 or 28073:", [(b,sizes[b]) for b in cand])
p=subprocess.Popen(['git','-C',wt,'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
p.stdin.write(('\n'.join(blobs)+'\n').encode()); p.stdin.close()
hitT=[];hitC=[]
import re
for b in blobs:
    hdr=p.stdout.readline().decode().split()
    size=int(hdr[2]); body=p.stdout.read(size); p.stdout.read(1)
    h=hashlib.sha256(body).hexdigest()
    if h==T: hitT.append(b)
    if h==C: hitC.append(b)
print("blobs with SHA256==001c4f (external 696-line):", hitT)
print("blobs with SHA256==d4c2f0 (core):", hitC)
