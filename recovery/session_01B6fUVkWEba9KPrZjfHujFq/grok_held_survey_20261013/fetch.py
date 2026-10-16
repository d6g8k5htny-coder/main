import subprocess, json, sys, os
from concurrent.futures import ThreadPoolExecutor
jobs=[]
for r in ['main','Math-']:
    for line in open(f'open_{r}.tsv'):
        n,pr=line.split('\t')[:2]
        jobs.append((r,n,'comments',f'repos/d6g8k5htny-coder/{r}/issues/{n}/comments?per_page=100'))
        if pr=='true':
            jobs.append((r,n,'reviews',f'repos/d6g8k5htny-coder/{r}/pulls/{n}/reviews?per_page=100'))
            jobs.append((r,n,'pr',f'repos/d6g8k5htny-coder/{r}/pulls/{n}'))
def run(j):
    r,n,k,url=j
    args=['gh','api',url]+(['--paginate'] if k!='pr' else [])
    p=subprocess.run(args,capture_output=True,text=True,stdin=subprocess.DEVNULL)
    if p.returncode: return (j,None,p.stderr[:300])
    out=p.stdout
    if k!='pr':
        # paginate concatenates arrays: ][ join
        out='['+out.strip()[1:-1].replace('][',',').replace(']\n[',',')+']' if out.strip() else '[]'
    try: data=json.loads(out)
    except Exception as e: return (j,None,'parse '+str(e))
    return (j,data,None)
res={}
with ThreadPoolExecutor(6) as ex:
    for j,data,err in ex.map(run,jobs):
        r,n,k,_=j
        if err: print('ERR',j[:3],err,file=sys.stderr); continue
        res.setdefault(f'{r}#{n}',{})[k]=data
json.dump(res,open('all.json','w'))
print(len(res), sum(len(v.get('comments',[])) for v in res.values()), sum(len(v.get('reviews',[])) for v in res.values()))
