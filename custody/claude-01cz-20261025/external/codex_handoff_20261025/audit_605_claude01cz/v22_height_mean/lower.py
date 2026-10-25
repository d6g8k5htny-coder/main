import sympy as sp
q,eta,s,k=sp.symbols('q eta s k',positive=True); c,t,a,d=sp.symbols('c t a d',real=True)
D=c*d-a*t
X=2*q*t+eta*(2*D+d*q); Y=2*q*t-eta*(2*D-3*d*q)
Pi=-k*s**3*(2*D+d*q)**2*X*Y/(32*d*D*(t+d*eta)**2)
etaX=sp.solve(X,eta)[0]; etaY=sp.solve(Y,eta)[0]
print('X root:',sp.simplify(etaX),' Y root:',sp.simplify(etaY))
print('X == (2D+dq)(eta-etaX):',sp.simplify(X-(2*D+d*q)*(eta-etaX))==0,' Y == -(2D-3dq)(eta-etaY):',sp.simplify(Y+(2*D-3*d*q)*(eta-etaY))==0)
print('product of roots sign: etaX*etaY =',sp.factor(sp.simplify(etaX*etaY)),'(negative for small q => opposite signs)')
# numeric spot check of the lower bound ratio |Pi| / (k s^3 (q+eta)|eta-eta*|) on a sample sector point
vals={c:1,t:1,a:0,d:1,k:1,s:sp.Rational(1,100)}
for qq in [sp.Rational(1,100),sp.Rational(1,20)]:
    es=etaY.subs(vals).subs(q,qq) if etaY.subs(vals).subs(q,qq)>0 else etaX.subs(vals).subs(q,qq)
    rat=[sp.N(sp.Abs(Pi.subs(vals).subs({q:qq,eta:e}))/(sp.Rational(1,100)**3*(qq+e)*sp.Abs(e-es))) for e in [sp.Rational(1,1000),es/2,es*sp.Rational(99,100),es*sp.Rational(101,100),2*es,sp.Rational(1,10)]]
    print('q=',qq,' eta*=',sp.N(es,6),' ratios',[round(float(x),4) for x in rat])
