from fractions import Fraction as Q
p=q=Q(1,200); u=Q(1,2); v=Q(3,2)
P=p+q; w=1-P; a=p*u+q*v; c=a/P
b=P/w; g0=p*q*(u-v)**2/P
points=[a,Q(11,1000),Q(3,20),Q(4,25),Q(999,1000),c]
expected=[-1,1,1,-1,-1,1]
for s,sign in zip(points,expected):
    h=g0+b*(s-c)**2
    G=s*h*h-b*(c-s)*(h*h+2*h*s*s+2*s**4)
    if (G>0)-(G<0)!=sign:
        raise RuntimeError('rational sign mismatch')
    print(s,G)
