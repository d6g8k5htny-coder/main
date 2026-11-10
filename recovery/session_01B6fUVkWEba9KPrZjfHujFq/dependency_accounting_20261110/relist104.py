import csv, subprocess, os, json, collections
rows=list(csv.DictReader(open('all104.tsv'),delimiter='\t'))
repos={'d6g8k5htny-coder/Math-':'repos/Math-full','d6g8k5htny-coder/main':'/home/user/main'}
CODE=('.py','.lean','.sage','.ipynb','.m','.jl','.c','.cpp','.rs','.sh','.js','.nb','.wl','.gp','.mpl','.sql')
out=[]
cache={}
for r in rows:
    repo=r['source_repo']; c=r['source_commit']; p=r['source_path']
    d=os.path.dirname(p)
    g=repos.get(repo)
    rec={'id':r['record_id'],'repo':repo,'commit':c,'path':p,'dir':d}
    if not g:
        rec['status']='REPO_UNREADABLE'; out.append(rec); continue
    key=(repo,c,d)
    if key not in cache:
        res=subprocess.run(['git','-C',g,'ls-tree','-r','--long',c,'--',d+'/' if d else '.'],capture_output=True,text=True)
        if res.returncode!=0: cache[key]=None
        else:
            files=[]
            for ln in res.stdout.splitlines():
                meta,name=ln.split('\t',1)
                parts=meta.split()
                files.append((name,parts[2],parts[3]))
            cache[key]=files
    files=cache[key]
    if files is None: rec['status']='COMMIT_MISSING'
    else:
        code=[f for f in files if f[0].endswith(CODE)]
        rec['n_files']=len(files); rec['code']=[(f[0],f[1][:8],f[2]) for f in code]
        rec['status']='CODE_IN_FOLDER' if code else ('FOLDER_EMPTY' if not files else 'NO_CODE_IN_FOLDER')
    out.append(rec)
json.dump(out,open('relist104.json','w'),indent=1)
cnt=collections.Counter(r['status'] for r in out); print(cnt)
for r in out:
    if r['status']!='CODE_IN_FOLDER': print(r['id'],r['status'],r.get('n_files'),r['dir'][:90])
