from fractions import Fraction as F
from itertools import product
def ck(p, label):
    if not p:
        raise RuntimeError(label)
def law_check(weights):
    total=sum(weights)
    if not total: return False
    pk=[F(x,total) for x in weights]
    m=sum(k*x for k,x in enumerate(pk)); p=1-pk[0]
    if not p: return False
    D=sum(k*pk[k] for k in range(2,len(pk)))
    delta=sum((k-1)*pk[k] for k in range(1,len(pk)))
    ck(delta==m-p,'excess identity')
    ck(D/2<=delta<=D,'excess sandwich')
    ck(delta==pk[2]+sum((k-1)*pk[k] for k in range(3,len(pk))),'two-bar decomposition')
    I=[k*pk[k]/m for k in range(1,len(pk))]
    J=[pk[k]/p for k in range(1,len(pk))]
    TV=sum(abs(x-y) for x,y in zip(I,J))/2
    ck(TV<=delta/m,'TV bound')
    # This exact formula is the sharper field-aware count-mark consequence.
    ck(TV==sum(pk[k]*abs(k*p-m) for k in range(1,len(pk)))/(2*m*p),'exact count TV')
    if m<2*p: ck(TV==pk[1]*delta/(m*p),'singleton maximizes discrepancy')
    if delta:
        excess=[(k-1)*pk[k]/delta for k in range(1,len(pk))]
        ck(all(x==p/m*y+delta/m*z for x,y,z in zip(I,J,excess)),'mixture equality')
        ck(TV==delta/m*sum(abs(x-y) for x,y in zip(excess,J))/2,'TV equality')
    # A lifetime-only mark identically 1/2 has equal selection laws even for delta>0.
    ck(sum(I)==sum(J)==1,'constant lifetime mark cancellation')
    return True

n=sum(law_check(w) for w in product(range(4),repeat=6))
ck(n==4092,'enumerated count laws')

print(n)
