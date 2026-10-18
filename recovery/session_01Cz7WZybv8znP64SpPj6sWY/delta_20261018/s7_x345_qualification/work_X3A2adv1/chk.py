import sympy as sp
r=sp.symbols('r',positive=True)
def T(m):
    n=4+2*m; M=sp.zeros(n,n)
    # P column order: f(a), fx(a), f(c), fx(c), (fy(a), fy(c))*m ; rows U0..U3, (Vj0,Vj1)
    M[0,0]=M[0,2]=sp.Rational(1,2)
    M[1,0]=-1/r; M[1,2]=1/r
    M[2,1]=-1/r; M[2,3]=1/r
    M[3,1]=M[3,3]=6/r**2; M[3,2]=-12/r**3; M[3,0]=12/r**3
    for j in range(m):
        M[4+2*j,4+2*j]=M[4+2*j,5+2*j]=sp.Rational(1,2)
        M[5+2*j,4+2*j]=-1/r; M[5+2*j,5+2*j]=1/r
    return M
for m in range(1,4):
    print(m, sp.simplify(T(m).det()*r**(m+4)))

# Schur complement two-sided bound, symbolic-rational random PD joints via mpmath
import mpmath as mp, random
random.seed(1)
worst=mp.mpf(0)
for _ in range(200):
    n,p=4,2
    X=mp.matrix([[random.gauss(0,1) for _ in range(n+p)] for _ in range(n+p)])
    J=X*X.T+mp.mpf('0.05')*mp.eye(n+p)
    A=J[0:n,0:n]; B=J[0:n,n:n+p]; C=J[n:n+p,n:n+p]
    S=C-B.T*mp.inverse(A)*B
    lmin=min(mp.eig(J)[0], key=lambda z: mp.re(z))
    es=min(mp.re(z) for z in mp.eig(S)[0]); ea=min(mp.re(z) for z in mp.eig(C-S)[0])
    worst=min(worst, es-mp.re(lmin), ea)
print('schur worst (>= -1e-9 ok):', worst)
