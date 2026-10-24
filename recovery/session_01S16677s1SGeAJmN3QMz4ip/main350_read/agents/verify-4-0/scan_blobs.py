import subprocess, hashlib, sys
TARGET = 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08'
out = subprocess.run(['git','cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)'],capture_output=True,text=True,check=True).stdout
blobs = [l.split() for l in out.splitlines() if l.split()[1]=='blob']
print('total blobs', len(blobs))
size_hits = [b for b in blobs if int(b[2])==22303]
print('blobs of size 22303:', size_hits)
# full sha256 scan of all blobs
proc = subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
hits=[]
for oid,_,size in blobs:
    proc.stdin.write((oid+'\n').encode()); proc.stdin.flush()
    hdr = proc.stdout.readline()
    n = int(hdr.split()[2])
    data = proc.stdout.read(n); proc.stdout.read(1)
    h = hashlib.sha256(data).hexdigest()
    if h==TARGET or h.startswith('3ea8c38d'):
        hits.append((oid,n,h))
proc.stdin.close()
print('sha256 hits:', hits)
