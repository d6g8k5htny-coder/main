import sys; sys.path.insert(0,sys.argv[0].rsplit('/',1)[0])
from common import *
SRC=sys.argv[1]; PY=sys.argv[2]
T='{'+M+':true}'
for n in (600,1000):
    for opt in (False,True):
        a=run(SRC,PY,'{"a":'+deep(n,'{}')+',"b":'+T+'}',opt)
        b=run(SRC,PY,'{"b":'+T+',"a":'+deep(n,'{}')+'}',opt)
        print(PY,'O' if opt else 'N','order n=',n,'deep-then-true:',short(a),'|| true-then-deep:',short(b),flush=True)
for n in (333,500,900,991,992,499,9997,9998):
    for opt in (False,True):
        print(PY,'O' if opt else 'N','depth',n,short(run(SRC,PY,deep(n,T),opt)),flush=True)
