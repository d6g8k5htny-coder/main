import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import *
SRC=sys.argv[1]; PY=sys.argv[2]; opt=len(sys.argv)>3 and sys.argv[3]=='O'
shapes={'true':'{'+M+':true}','ssa1':'{'+M+':1}','kv':'{"k":"v"}','empty':'{}','ssa_obj':'{'+M+':{}}'}
def rc(n,inner): return run(SRC,PY,deep(n,inner),opt)[0]
for name,inner in shapes.items():
    lo,hi=100,1200  # lo: rc!=1 ; hi: rc==1 expected
    assert rc(lo,inner)!=1
    if rc(hi,inner)!=1: hi=900
    assert rc(hi,inner)==1,(name,hi)
    while hi-lo>1:
        mid=(lo+hi)//2
        if rc(mid,inner)==1: hi=mid
        else: lo=mid
    print(PY,'O' if opt else 'N',name,'last_ok',lo,short(run(SRC,PY,deep(lo,inner),opt)),'| first_exit1',hi,short(run(SRC,PY,deep(hi,inner),opt)),flush=True)
