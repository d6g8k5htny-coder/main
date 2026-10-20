import sympy as sp, mpmath as mp
mp.mp.dps=30
mu=mp.mpf('-1.2'); sd=mp.sqrt(2)
pdf=lambda x: mp.exp(-(x-mu)**2/(2*sd**2))/(sd*mp.sqrt(2*mp.pi))
z0=mp.quad(lambda x: x**2*pdf(x),[-mp.inf,0])
E3=mp.quad(lambda x: abs(x)**3*pdf(x),[-mp.inf,0,mp.inf])
E3neg=mp.quad(lambda x: abs(x)**3*pdf(x),[-mp.inf,0])
print('z0(6/5)=E[S^2;S<0] =',z0)
print('E|S|^3 =',E3,' E[|S|^3;S<0] =',E3neg, ' P(S<0)=',mp.quad(pdf,[-mp.inf,0]))
for p in [mp.mpf('0.15'),mp.mpf('-0.15')]:
    f=8*(1-p)**2/(1-2*p)**3
    print('p=',p,' untyped const 8(1-p)^2/(1-2p)^3 E|S|^3 =',f*E3,'  restricted to S<0:',f*E3neg)
# A4 forcing in collinear quartic model (P2 coords: M at 0, S at r, X at rp)
x,r,p,k,A4,b=sp.symbols('x r p k A4 b',real=True)
H=b+k*(2*x**3-3*r*x**2)
g=H+A4/24*x**2*(x-r)**2
sol=sp.solve(sp.Eq(sp.diff(g,x).subs(x,r*p),0),A4)
print('forced A4 =',sp.factor(sol[0]))
gs=g.subs(A4,sol[0])
gpp=sp.diff(gs,x,2)
print('g_xx(M),g_xx(S),g_xx(X):',[sp.factor(gpp.subs(x,v)) for v in (0,r,r*p)])
prod=sp.factor(gpp.subs(x,0)*gpp.subs(x,r)*gpp.subs(x,r*p))
print('axial product of g_xx =',prod)
# axis cost c(X): station kappa(X^2-1/4)+a X Y + (w/2) Y^2=0, a,w iid N(0,2)
X,Y,kap=sp.symbols('X Y kappa',positive=True)
varL=2*(X*Y)**2+2*(Y**2/2)**2
cost=sp.simplify((kap*(X**2-sp.Rational(1,4)))**2/(2*varL))
print('min cost =',sp.factor(cost),'; at X=0:',sp.simplify(cost.subs(X,0)),'; Y->0 leading c(X)/Y^2 with c(X)=',sp.factor(sp.limit(cost*Y**2,Y,0)))
# C100 density exponent at kappa=1 equals this cost:
print('C100 exponent - cost(kappa=1):',sp.simplify((X**2-sp.Rational(1,4))**2/(Y**2*(4*X**2+Y**2))-cost.subs(kap,1)))
# FlatSaddle tail-set: f_tt(S)=kap r(1+eps r), f_ss=-mu, f_ts=r nu
eps,mu_,nu,rr,m=sp.symbols('epsilon mu nu r m',real=True)
ftt=kap*rr*(1+eps*rr); fss=-mu_; fts=rr*nu
detHS=sp.expand(ftt*fss-fts**2)
mut=mu_+rr*nu**2/kap
err=sp.factor(sp.expand(detHS-(-kap*rr*mut*(1+eps*rr))))
print('det H_S - [-kap r mu~ (1+eps r)] =',err)
schur=sp.simplify(-detHS/ftt)
print('Schur -detH_S/f_tt =',sp.simplify(schur),'; det H_S = -f_tt*Schur exactly:',sp.simplify(detHS+ftt*schur)==0)
# on tail set mu~ = m r^2: order of main term vs error
print('main -kap r mu~ on tail set:',sp.expand(-kap*rr*m*rr**2),' error:',err)
# FlatSaddle ledger
print('FlatSaddle ledger 4-5+8-2 =',4-5+8-2,'; dets 3+3+2 =',3+3+2,'; intensity ledger -3+6-2 =',-3+6-2)
