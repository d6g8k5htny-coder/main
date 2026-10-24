import sympy as sp, random, cmath, math
w,r=sp.symbols('omega r',positive=True); x=sp.symbols('x',real=True)
f=sp.exp(sp.I*w*x); fx=sp.diff(f,x); a,c=-r/2,r/2
rows={'U1':(f.subs(x,c)-f.subs(x,a))/r,
      'U2':(fx.subs(x,c)-fx.subs(x,a))/r,
      'U3':6/r**2*(fx.subs(x,a)+fx.subs(x,c)-2*(f.subs(x,c)-f.subs(x,a))/r)}
random.seed(0)
for name,e in rows.items():
    k=int(name[1]); F=sp.lambdify((w,r),e,'cmath'); mx=0
    for _ in range(100000):
        ww=10**random.uniform(-2,3); rr=10**random.uniform(-3,0)
        mx=max(mx,abs(F(ww,rr))/ww**k)
    print(name,'max |symbol|/omega^%d ='%k, round(mx,6))
for k in range(4):
    g=x**k; gx=sp.diff(g,x)
    print('U3 on x^%d:'%k, sp.simplify(6/r**2*(gx.subs(x,a)+gx.subs(x,c)-2*(g.subs(x,c)-g.subs(x,a))/r)))
# conditional mean/variance and C-S bound for conditional cross-covariance, random PD instances (pure python)
def chol_solve(S,b):
    n=len(S); L=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=S[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            L[i][j]=math.sqrt(s) if i==j else s/L[j][j]
    y=[0.0]*n
    for i in range(n): y[i]=(b[i]-sum(L[i][k]*y[k] for k in range(i)))/L[i][i]
    z=[0.0]*n
    for i in reversed(range(n)): z[i]=(y[i]-sum(L[k][i]*z[k] for k in range(i+1,n)))/L[i][i]
    return z
worst=0; worstc=0
for _ in range(3000):
    n=6; N=n+2
    M=[[random.gauss(0,1) for _ in range(N+3)] for _ in range(N)]
    C=[[sum(M[i][k]*M[j][k] for k in range(N+3)) for j in range(N)] for i in range(N)]
    d=[math.sqrt(C[i][i]) for i in range(N)]
    C=[[C[i][j]/(d[i]*d[j]) for j in range(N)] for i in range(N)]
    S=[row[1:n+1] for row in C[1:n+1]]; cxu=C[0][1:n+1]; cyu=C[n+1][1:n+1]
    v=[3*random.gauss(0,1) for _ in range(n)]
    Siv=chol_solve(S,v); E=sum(v[i]*Siv[i] for i in range(n))
    mean=sum(cxu[i]*Siv[i] for i in range(n))
    Sic=chol_solve(S,cxu); var=1-sum(cxu[i]*Sic[i] for i in range(n))
    Siy=chol_solve(S,cyu); covQ=C[0][n+1]-sum(cxu[i]*Siy[i] for i in range(n))
    assert var>-1e-12
    worst=max(worst,abs(mean)/math.sqrt(E)); worstc=max(worstc,abs(covQ))
print('max |E_Q xi|/sqrt(energy) =',round(worst,6),' (<=1 expected)')
print('max |Cov_Q(X,Y)| with unit unconditional variances =',round(worstc,6),' (<=1 expected)')
