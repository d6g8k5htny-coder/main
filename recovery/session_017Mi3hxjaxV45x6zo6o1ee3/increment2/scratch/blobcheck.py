import json,subprocess,sys,re
def git(*a): return subprocess.run(['git',*a],capture_output=True,text=True,cwd='/home/user/Math-').stdout
def walk(o,path=()):
    if isinstance(o,dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o,list):
        for v in o: yield from walk(v)
for pr in sys.argv[1:]:
    head=git('rev-parse',f'refs/remotes/pr/{pr}').strip()
    files=git('diff','--name-only',f'origin/main...{head}').split()
    mans=[f for f in files if f.endswith('.json')]
    bad=0;n=0
    for m in mans:
        try: j=json.loads(git('show',f'{head}:{m}'))
        except Exception: continue
        d=m.rsplit('/',1)[0] if '/' in m else ''
        for o in walk(j):
            gb=o.get('git_blob') or o.get('blob') or o.get('git_blob_sha1')
            p=o.get('path') or o.get('file')
            if not (isinstance(gb,str) and re.fullmatch(r'[0-9a-f]{7,40}',gb) and isinstance(p,str)): continue
            cands=[p, f'{d}/{p}']
            act=None
            for c in cands:
                r=git('rev-parse',f'{head}:{c}').strip()
                if re.fullmatch(r'[0-9a-f]{40}',r): act=r;break
            n+=1
            if act is None: print(f'  #{pr} {m}: {p} not found at head (declared {gb[:12]})'); bad+=1
            elif not act.startswith(gb): print(f'  #{pr} {m}: {p} declared {gb[:12]} actual {act[:12]}'); bad+=1
    print(f'#{pr} head={head[:7]} manifests={len(mans)} blob-entries={n} mismatches={bad}')
