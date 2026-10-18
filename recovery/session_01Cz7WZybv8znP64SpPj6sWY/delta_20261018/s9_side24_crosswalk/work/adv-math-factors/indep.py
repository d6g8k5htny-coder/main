import sympy as sp, mpmath as mp
mp.mp.dps = 22
r,k,kap,b = sp.symbols('r k kappa b', positive=True)
# ---- A. full d=3 pin transforms (8x8), columns: f(a),fx(a),fy1(a),fy2(a), f(c),fx(c),fy1(c),fy2(c)
fa,fxa,f1a,f2a,fc,fxc,f1c,f2c = sp.symbols('fa fxa f1a f2a fc fxc f1c f2c')
O = sp.Matrix([fa,fxa,f1a,f2a,fc,fxc,f1c,f2c])
P_rows = [(fa+fc)/2,(fc-fa)/r,(fxc-fxa)/r,6/r**2*(fxa+fxc-2*(fc-fa)/r),
          (f1a+f1c)/2,(f1c-f1a)/r,(f2a+f2c)/2,(f2c-f2a)/r]
C_rows = [(fc+fa)/2,(fc-fa-r*(fxc+fxa)/2)/r**3,(fxc+fxa)/2,(fxc-fxa)/r,
          (f1c+f1a)/2,(f1c-f1a)/r,(f2c+f2a)/2,(f2c-f2a)/r]
TP = sp.Matrix([[sp.diff(e,v) for v in O] for e in P_rows])
TC = sp.Matrix([[sp.diff(e,v) for v in O] for e in C_rows])
print('A1 |det T| P (3.2), chain (4):', sp.simplify(abs(TP.det())), sp.simplify(abs(TC.det())))
print('A2 V-corr + U3/12 =', sp.simplify(C_rows[1]+P_rows[3]/12))
# targets: P f(c)=b-k r^3 ; chain f(c)=b-kappa r^3/6 ; zero gradients
subsP={fa:b,fc:b-k*r**3,fxa:0,fxc:0,f1a:0,f1c:0,f2a:0,f2c:0}
subsC={fa:b,fc:b-kap*r**3/6,fxa:0,fxc:0,f1a:0,f1c:0,f2a:0,f2c:0}
print('A3 P target', [sp.simplify(e.subs(subsP)) for e in P_rows[:4]], ' chain target', [sp.simplify(e.subs(subsC)) for e in C_rows[:4]])
# ---- B. reference covariance exp(-|x|^2/2), d=3: derivative covariances via sympy
x1,x2,x3 = sp.symbols('x1 x2 x3'); X=(x1,x2,x3)
K = sp.exp(-(x1**2+x2**2+x3**2)/2)
def cov(a,bb):  # a,bb multi-index tuples
    e=K
    for i,n in enumerate(a): e=sp.diff(e,X[i],n)
    for i,n in enumerate(bb): e=sp.diff(e,X[i],n)
    return sp.simplify((-1)**sum(bb)*e.subs({x1:0,x2:0,x3:0}))
# P limit vector U0* = (f, fx, fxx, fxxx, fy1, fxy1, fy2, fxy2); chain Y0 = (f, -fttt/12, ft, ftt, fn1, ftn1, fn2, ftn2), t=e1
idx = [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1)]
S = sp.Matrix(8,8,lambda i,j: cov(idx[i],idx[j]))
Lc = sp.zeros(8,8); Lc[0,0]=1; Lc[1,3]=sp.Rational(-1,12); Lc[2,1]=1; Lc[3,2]=1
for i in range(4,8): Lc[i,i]=1
SC = Lc*S*Lc.T
def dens(Sig,v):
    n=Sig.shape[0]; return (2*sp.pi)**sp.Rational(-n,2)/sp.sqrt(Sig.det())*sp.exp(-(v.T*Sig.inv()*v)[0]/2)
vP = sp.Matrix([b,0,0,12*k,0,0,0,0]); vC = sp.Matrix([b,-kap/6,0,0,0,0,0,0])
pi0 = sp.simplify(dens(S,vP)); p0 = sp.simplify(dens(SC,vC))
print('B1 det Cov U0* =', S.det(), '  det Cov Y0 =', SC.det(), ' (LS-DER-047 (5) says 1/12)')
print('B2 chain p0 =', p0)
print('B3 LS-DER-047 (7) =', sp.simplify(2*sp.sqrt(3)/(2*sp.pi)**4*sp.exp(-3*b**2/4-kap**2/3)), ' ratio p0/(047 eq7) =', sp.simplify(p0/(2*sp.sqrt(3)/(2*sp.pi)**4*sp.exp(-3*b**2/4-kap**2/3))))
print('B4 p0(kappa=6k)/pi0(k) =', sp.simplify(p0.subs(kap,6*k)/pi0))
# ---- C. end-to-end at reference, PRE-elimination (b-dependent law, no parity collapse)
# E_b = E[det(G2-bI)^2 1{lmax(G2)<b}], G2 diag var 2 offdiag var 1 (LS-DER-047 line 43, (10)-(11))
# s=(a+d)/2~N(0,1), R^2 ~ chi2_2 (density e^{-z/2}/2)  => evaluate as 2D integral
zz,aa=sp.symbols('zz aa',positive=True)
gsym=sp.simplify(sp.integrate((aa-zz)**2*sp.exp(-zz/2)/2,(zz,0,aa)))
print('C0 int_0^a (a-z)^2 e^{-z/2}/2 dz =', sp.expand(gsym), ' minus PROOF line 65 form:', sp.simplify(gsym-(aa**2-4*aa+8-8*sp.exp(-aa/2))))
gf=sp.lambdify(aa,gsym,'mpmath')
def Eb(bv):
    return mp.quad(lambda s: mp.npdf(s)*gf((bv-s)**2), [-mp.inf, bv])
pi0f = sp.lambdify((b,k), pi0, 'mpmath'); p0f = sp.lambdify((b,kap), p0, 'mpmath')
Ebv = {}
def Ebc(bv):
    key=mp.nstr(bv,25)
    if key not in Ebv: Ebv[key]=Eb(bv)
    return Ebv[key]
area = 4*mp.pi
cP = 4*area*mp.quad(lambda bv: Ebc(bv)*mp.quad(lambda kv: kv**(-mp.mpf(2)/3)*pi0f(bv,kv)*(6*kv)**2, [0,mp.inf]), [-mp.inf,0,mp.inf])
cC = mp.mpf(1)/6*mp.mpf(6)**(mp.mpf(2)/3)/3*area*mp.quad(lambda bv: Ebc(bv)*mp.quad(lambda kv: p0f(bv,kv)*kv**(mp.mpf(4)/3), [0,mp.inf]), [-mp.inf,0,mp.inf])
cref = mp.mpf(2)**(-mp.mpf(23)/6)*mp.mpf(3)**(-mp.mpf(8)/3)*(29*mp.sqrt(6)-36)*mp.gamma(mp.mpf(1)/6)*mp.pi**(-mp.mpf(5)/2)
cPROOF = mp.gamma(mp.mpf(7)/6)*mp.cbrt(mp.mpf(3)/2)*(mp.mpf(29)/6-mp.sqrt(6))/(2*mp.sqrt(3)*mp.pi**2*mp.sqrt(mp.pi))
c056_38 = mp.mpf(2)**(mp.mpf(1)/6)*mp.mpf(3)**(mp.mpf(1)/3)*(29*mp.sqrt(6)-36)*mp.gamma(mp.mpf(1)/6)/(432*mp.pi**(mp.mpf(5)/2))
print('C1 P (13.6) pre-elim  =', mp.nstr(cP,22))
print('C2 chain 046(19)+047(11) pre-elim =', mp.nstr(cC,22))
print('C3 LS-CLS-077 closed  =', mp.nstr(cref,22), ' PROOF(1) d=3 =', mp.nstr(cPROOF,22), ' 056(38) =', mp.nstr(c056_38,22))
print('C4 ratios cP/cref, cC/cref:', mp.nstr(cP/cref,18), mp.nstr(cC/cref,18))
# ---- D. b-integrated even block at reference: int exp(-3b^2/4) E_b db vs (2pi)... check P line 463 / 056 (17)
# p_(f,V)(b,0) for V=(fxx,fxy1,fxy2): even block (f,fxx) cov [[1,-1],[-1,3]], fxy indep var 1
lhs = mp.quad(lambda bv: mp.exp(-mp.mpf(3)/4*bv**2)/(2*mp.pi*mp.sqrt(2))/(2*mp.pi)*Ebc(bv), [-mp.inf,0,mp.inf])
rhs = (2*mp.pi)**(-mp.mpf(3)/2)/mp.sqrt(3)*(mp.mpf(29)/6-mp.sqrt(6))
print('D1 int p_(f,V)(b,0)E[F|f=b,V=0]db / [p_V(0) D_2] =', mp.nstr(lhs/rhs,18))
