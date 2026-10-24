import sys,runpy
from pathlib import Path
ns={}
src=Path(__file__).with_name('matrix.py').read_text().split("M='\"scientific_status_authority\"'")[0]
exec(src,ns)
run=ns['run']
M='"scientific_status_authority"'
def deep(n): return '{"a":'*n+'{'+M+':true}'+'}'*n
def kind(n):
    rc,v,t=run(deep(n),False)
    return 'crash' if v is None else ('parse' if v and v[0].startswith('generated JSON parse') else 'auth')
def first(lo,hi,pred):
    while lo<hi:
        mid=(lo+hi)//2
        if pred(mid): hi=mid
        else: lo=mid+1
    return lo
c=first(1,2000,lambda n:kind(n)!='auth')
p=first(c,20000,lambda n:kind(n)=='parse')
print(sys.argv[2],'first_crash_depth',c,kind(c),'first_parse_error_depth',p,kind(p), 'kind(p-1)',kind(p-1))
