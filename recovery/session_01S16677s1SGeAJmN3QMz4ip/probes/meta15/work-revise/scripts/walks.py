import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import *
PY=sys.argv[1]
inners={'true':'{'+M+':true}','mal1':'{'+M+':1}','none':'{"k":"v"}'}
for src in ('mut_rec1','ref_iter'):
    for n in (900,990,1500,5000,9000):
        for name,inner in inners.items():
            a=run(src,PY,deep(n,inner),False); b=run(src,PY,deep(n,inner),True)
            print(PY,src,n,name,short(a),'' if short(a)==short(b) else '/O:'+short(b),flush=True)
