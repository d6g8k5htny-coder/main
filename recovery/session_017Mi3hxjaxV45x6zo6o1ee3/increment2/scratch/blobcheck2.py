import json,subprocess,sys,re
def git(*a): return subprocess.run(['git',*a],capture_output=True,text=True,cwd='/home/user/Math-').stdout.strip()
def walk(o,ctx=None):
    if isinstance(o,dict):
        yield o,ctx
        for k,v in o.items(): yield from walk(v,k if isinstance(v,(list,dict)) else ctx)
    elif isinstance(o,list):
        for v in o: yield from walk(v,ctx)
main=git('rev-parse','origin/main')
for pr in sys.argv[1:]:
    head=git('rev-parse',f'refs/remotes/pr/{pr}')
    files=git('diff','--name-only',f'origin/main...{head}').split()
    mans=[f for f in files if f.endswith('.json')]
    res={'ok':0,'bad':[]}
    for m in mans:
        try: j=json.loads(subprocess.run(['git','show',f'{head}:{m}'],capture_output=True,text=True,cwd='/home/user/Math-').stdout)
        except Exception: continue
        d=m.rsplit('/',1)[0]
        for o,ctx in walk(j):
            gb=o.get('git_blob'); p=o.get('path')
            if not (isinstance(gb,str) and isinstance(p,str)): continue
            if 'commit' in o:
                where=o['commit']; label=f'commit {where}'
            elif ctx=='pins_on_main':
                where=main; label='main'
            else:
                where=head; label='head'
            act=git('rev-parse',f'{where}:{p}')
            if not re.fullmatch(r'[0-9a-f]{40}',act): act=git('rev-parse',f'{where}:{d}/{p}')
            if re.fullmatch(r'[0-9a-f]{40}',act) and act.startswith(gb): res['ok']+=1
            else: res['bad'].append(f'{m} [{ctx}] {p} @{label}: declared {gb[:12]} actual {act[:12] or "MISSING"}')
    print(f'#{pr} head={head[:7]} ok={res["ok"]} bad={len(res["bad"])}')
    for b in res['bad']: print('   ',b)
