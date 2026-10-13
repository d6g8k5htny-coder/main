import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import *
SRC=sys.argv[1]; PY=sys.argv[2]; lo=int(sys.argv[3]); hi=int(sys.argv[4]); target=int(sys.argv[5])
inner='{'+M+':true}'
def rc(n): return run(SRC,PY,deep(n,inner))[0]
assert rc(lo)!=target and rc(hi)==target
while hi-lo>1:
    m=(lo+hi)//2
    if rc(m)==target: hi=m
    else: lo=m
print(PY,SRC,'last',lo,short(run(SRC,PY,deep(lo,inner))),'| first rc',target,'at',hi,short(run(SRC,PY,deep(hi,inner))))
