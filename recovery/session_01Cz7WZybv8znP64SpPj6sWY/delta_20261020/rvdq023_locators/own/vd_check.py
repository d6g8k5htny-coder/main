import re, subprocess, hashlib, sys
R='/home/user/main'
A='a01c72f1937875b94bfd78e183a6dbbdb2b04da4'; M='7e95ddcc69cd0a35732b85443d6600c6007a5f8d'
d=open(sys.argv[1],encoding='utf-8').read()
def g(*a): return subprocess.run(['git','-C',R,*a],capture_output=True).stdout
known={}
# blob ids in draft: check existence, size, sha
blobs=set(re.findall(r'`([0-9a-f]{40})`',d))
for b in sorted(blobs):
    t=g('cat-file','-t',b).strip().decode()
    if t=='blob':
        data=g('cat-file','blob',b); known[hashlib.sha256(data).hexdigest()]=(b,len(data))
        print('blob',b,len(data),hashlib.sha256(data).hexdigest())
    else: print('NONBLOB',b,t)
# derived
csv=g('show',A+':registers/csv/review_queue.csv').split(b'\r\n')
l13=csv[12]; print('row13',len(l13),hashlib.sha256(l13).hexdigest()); known[hashlib.sha256(l13).hexdigest()]='row13'
cand=open('<SCRATCH>/rvdq023/LS-DER-021_candidate_body.txt','rb').read()
print('cand',len(cand),hashlib.sha256(cand).hexdigest()); known[hashlib.sha256(cand).hexdigest()]='cand'
cov=g('show',A+':drive/deltas/2026-09-19/DG-MIGRATION-20260919/coverage.jsonl').split(b'\n')[87]
print('cov88',cov[:400])
for h in re.findall(r'[0-9a-f]{64}',cov.decode()): known[h]='cov88'
for h in re.findall(r'[0-9a-f]{64}',l13.decode()+csv[16].decode()): known[h]='rqrow'
for h in sorted(set(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',d))):
    print('OK ' if h in known else 'MISSING', h, known.get(h))
