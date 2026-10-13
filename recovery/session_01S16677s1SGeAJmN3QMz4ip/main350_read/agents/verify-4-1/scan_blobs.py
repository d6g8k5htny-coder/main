import subprocess, hashlib, sys
TARGET_SHA = 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08'
TARGET_BYTES = 22303
# enumerate all objects (packed + loose, reachable or not)
out = subprocess.run(['git','cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)'],capture_output=True,text=True,check=True).stdout
blobs = [l.split() for l in out.splitlines() if l.split()[1]=='blob']
print('total blobs', len(blobs))
size_hits = [b for b in blobs if int(b[2])==TARGET_BYTES]
print('blobs of size 22303:', size_hits)
for oid,_,sz in size_hits:
    raw = subprocess.run(['git','cat-file','blob',oid],capture_output=True,check=True).stdout
    print(oid, len(raw), hashlib.sha256(raw).hexdigest())
# full scan for sha
found = []
p = subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for oid,_,sz in blobs:
    p.stdin.write((oid+'\n').encode()); p.stdin.flush()
    hdr = p.stdout.readline()
    n = int(hdr.split()[2])
    raw = p.stdout.read(n); p.stdout.read(1)
    if hashlib.sha256(raw).hexdigest()==TARGET_SHA:
        found.append((oid,n))
p.stdin.close()
print('blobs with target sha256:', found)
