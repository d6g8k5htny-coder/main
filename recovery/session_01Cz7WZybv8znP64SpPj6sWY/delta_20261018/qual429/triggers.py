import subprocess, re, sys, fnmatch, yaml
main, tree = sys.argv[1], sys.argv[2]
ls = subprocess.run(['git','ls-tree','--name-only',main,'.github/workflows/'],capture_output=True,text=True).stdout.split()
pkt = subprocess.run(['git','diff','--name-only',main,tree],capture_output=True,text=True).stdout.split()
def m(p,f):
    rx = re.escape(p).replace(r'\*\*','§§').replace(r'\*','[^/]*').replace('§§','.*')
    return re.fullmatch(rx,f) is not None
def anym(pats,files): return any(m(p,f) for p in pats for f in files)
hits=set()
for w in ls:
    y=yaml.safe_load(subprocess.run(['git','show',main+':'+w],capture_output=True,text=True).stdout)
    on=y.get(True, y.get('on'))
    if isinstance(on,str): on={on:None}
    if isinstance(on,list): on={k:None for k in on}
    for ev in ('pull_request','pull_request_target','push'):
        if ev not in on: continue
        cfg=on[ev] or {}
        br=cfg.get('branches'); bi=cfg.get('branches-ignore')
        if ev=='push':
            b='claude/qual429-current-main-20261018'
            if br is not None and not any(fnmatch.fnmatch(b,x) for x in br): continue
            if bi is not None and any(fnmatch.fnmatch(b,x) for x in bi): continue
        else:
            if br is not None and 'main' not in br: continue
        paths=cfg.get('paths'); pign=cfg.get('paths-ignore')
        if paths is not None and not anym(paths,pkt): continue
        if pign is not None and all(anym(pign,[f]) for f in pkt): continue
        hits.add((w.split('/')[-1],ev))
print(' '.join(sorted(f'{a}:{b}' for a,b in hits)))
