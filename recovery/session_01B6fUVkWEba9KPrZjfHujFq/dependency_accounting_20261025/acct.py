import csv,subprocess,hashlib,re,json,sys
S="/tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad"
REPO={'d6g8k5htny-coder/main':'/home/user/main','d6g8k5htny-coder/Math-':S+'/repos/Math-full'}
topics=sys.argv[1].split(',')
rows=[r for r in csv.DictReader(open(S+'/all104.tsv'),delimiter='\t') if r['topic'] in topics and r['605_source_accounting']=='NOT YET']
def git(repo,*a): return subprocess.run(['git','-C',repo,*a],capture_output=True)
def ensure(repo,c):
    if git(repo,'cat-file','-e',c).returncode: git(repo,'fetch','-q','--depth=1','origin',c)
    return git(repo,'cat-file','-e',c).returncode==0
out=[]
for r in rows:
    repo=REPO[r['source_repo']]; c=r['source_commit']; p=r['source_path']
    rec={k:r[k] for k in('record_id','topic','title','source_repo','source_path','proof_ref','necessary_premises','external_imports')}; rec['commit']=c
    if not ensure(repo,c): rec['resolved']=False; out.append(rec); continue
    b=git(repo,'show',f'{c}:{p}')
    if b.returncode: rec['resolved']=False; out.append(rec); continue
    data=b.stdout
    rec.update(resolved=True,bytes_ok=len(data)==int(r['source_bytes']),sha_ok=hashlib.sha256(data).hexdigest()==r['source_sha256'],blob_ok=git(repo,'rev-parse',f'{c}:{p}').stdout.decode().strip()==r['source_git_blob'])
    txt=data.decode('utf8','replace').split('\n'); rec['lines']=len(txt)
    rec['headings']=[(i,l.strip()[:90]) for i,l in enumerate(txt,1) if l.startswith('#')]
    d=p.rsplit('/',1)[0]
    ls=git(repo,'ls-tree','-r','--name-only',c,d+'/').stdout.decode().split('\n')
    rec['dir_files']=[x for x in ls if x]
    rec['impl']=[x for x in rec['dir_files'] if re.search(r'\.(py|lean|json|sage|nb|jl|cpp|c|rs|ts|js|sh)$',x) and not x.endswith(('SOURCES.json',))]
    out.append(rec)
json.dump(out,open(S+'/acct_'+re.sub(r'[^A-Za-z0-9]+','_',sys.argv[1])+'.json','w'),indent=1)
for r in out:
    print(r['record_id'],'|',r['source_repo'].split('/')[1],r['commit'][:8],'|',r['source_path'][-60:],'| ok:',r.get('bytes_ok'),r.get('sha_ok'),r.get('blob_ok'),'| ref:',r['proof_ref'][:50],'| impl:',len(r.get('impl',[])),'/',len(r.get('dir_files',[])))
