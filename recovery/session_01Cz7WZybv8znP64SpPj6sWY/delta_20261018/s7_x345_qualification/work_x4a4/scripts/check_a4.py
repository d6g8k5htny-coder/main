import sympy as sp, random
w,r,t=sp.symbols('omega r t',positive=True)
x=sp.symbols('x',real=True)
f=sp.exp(sp.I*w*x); fx=sp.diff(f,x)
a,c=-r/2,r/2
U1=(f.subs(x,c)-f.subs(x,a))/r
U2=(fx.subs(x,c)-fx.subs(x,a))/r
U3=6/r**2*(fx.subs(x,a)+fx.subs(x,c)-2*(f.subs(x,c)-f.subs(x,a))/r)
# U3 symbol equals (24 i/r^3)(t cos t - sin t), t = w r/2
print('U3 form:', sp.simplify(sp.expand_complex(U3 - 24*sp.I/r**3*((w*r/2)*sp.cos(w*r/2)-sp.sin(w*r/2)))) == 0)
# |t cos t - sin t| <= t^3/3 since derivative is -t sin t
print('deriv:', sp.simplify(sp.diff(t*sp.cos(t)-sp.sin(t),t)+t*sp.sin(t))==0)
# hence |U3| <= 8 t^3 / r^3 = w^3 ; check numerically on a grid
import numpy as np
F=sp.lambdify((w,r),U3,'numpy')
ww=10**np.random.default_rng(0).uniform(-2,3,200000); rr=10**np.random.default_rng(2).uniform(-3,0,200000)
mx=float(np.max(np.abs(F(ww,rr))/ww**3))
print('max |U3|/w^3 =',mx)
# U3 annihilates 1,x,x^2 and maps x^3/6 -> 1
for k in range(4):
    g=x**k; gx=sp.diff(g,x)
    print(k, sp.simplify(6/r**2*(gx.subs(x,a)+gx.subs(x,c)-2*(g.subs(x,c)-g.subs(x,a))/r)))
# conditional mean / variance bound, random Gaussian instance (Schur): |Cov(xi,U)S^-1 v| <= sqrt(v'S^-1 v)
import numpy as np
rng=np.random.default_rng(1); worst=0
for _ in range(2000):
    n=6; M=rng.normal(size=(n+2,n+2)); C=M@M.T; C/=np.sqrt(np.outer(np.diag(C),np.diag(C)))  # unit variances
    S=C[1:n+1,1:n+1]; cxu=C[0,1:n+1]; v=rng.normal(size=n)*3
    Si=np.linalg.inv(S); mean=cxu@Si@v; E=v@Si@v; var=1-cxu@Si@cxu
    worst=max(worst, abs(mean)/np.sqrt(E)); assert var>=-1e-12
    # C-S route for B_r: |Cov_Q(X,Y)| <= sqrt(Var X Var Y) unconditional
    y=C[n+1,1:n+1]; covQ=C[0,n+1]-cxu@Si@y
    assert abs(covQ)<=1+1e-9
print('max |E_Q xi|/sqrt(energy) =',worst)
