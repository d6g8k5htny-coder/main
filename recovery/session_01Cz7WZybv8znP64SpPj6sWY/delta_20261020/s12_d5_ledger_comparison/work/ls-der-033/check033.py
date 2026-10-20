import sympy as sp
c,s,eta,kap,al,a,Q,w,z,X,Y = sp.symbols('c s eta kappa alpha a Q w z X Y', real=True)
P = kap*(X**3/3-eta**2*X/4-eta**3/12)+(a/2)*(X**2-eta**2/4)*Y+(Q/2)*Y**2+(w/2)*X*Y**2+(z/6)*Y**3
def at(e,x,y): return e.subs({X:x,Y:y})
PX,PY = sp.diff(P,X), sp.diff(P,Y)
out={}
# pin checks
out['pins_grad_zero'] = all(sp.simplify(at(g,x0,0))==0 for g in (PX,PY) for x0 in (-eta/2,eta/2))
out['pin_value_drop'] = sp.simplify(at(P,eta/2,0)-at(P,-eta/2,0))  # expect -kappa eta^3/6
# station elimination (generic)
sol = sp.solve([at(P,c,s)+al*kap*eta**3/6, at(PX,c,s), at(PY,c,s)],[a,Q,z],dict=True)[0]
a_g = (-c**2*kap+eta**2*kap/4-s**2*w/2)/(c*s)
Q_g = eta**2*(-al*c*eta*kap+c**2*kap/2+c*eta*kap/2+eta**2*kap/8-s**2*w/4)/(c*s**2)
z_g = (32*al*c*eta**3*kap+16*c**4*kap-24*c**2*eta**2*kap-24*c**2*s**2*w-16*c*eta**3*kap-3*eta**4*kap+6*eta**2*s**2*w)/(16*c*s**3)
out['a_matches'] = sp.simplify(sol[a]-a_g)==0
out['Q_matches'] = sp.simplify(sol[Q]-Q_g)==0
out['z_matches'] = sp.simplify(sol[z]-z_g)==0
H = sp.hessian(P,(X,Y))
dM = sp.simplify(H.subs({X:-eta/2,Y:0}).det().subs(sol))
dS = sp.simplify(H.subs({X:eta/2,Y:0}).det().subs(sol))
dX = sp.simplify(H.subs({X:c,Y:s}).det().subs(sol))
W = w/kap; K = 2*W*s**2-4*c**2-4*c*eta-eta**2
pref = kap**2*eta**2/(64*c**2*s**2)
BM = 64*al*c**2*eta**2-K**2; BS = 64*(1-al)*c**2*eta**2-(K+8*c*eta)**2; BX=(K+8*al*c*eta)**2+64*al*(1-al)*c**2*eta**2
on_circle = {s: sp.sqrt(1-c**2)}
for nm,d,claim in [('3.1',dM,pref*BM),('3.2',dS,pref*BS),('3.3',dX,-pref*BX)]:
    diff = sp.simplify(d-claim)
    out[nm+'_identity_free_cs'] = diff==0
    out[nm+'_identity_on_circle'] = sp.simplify(diff.subs(on_circle))==0
out['3.4_product'] = sp.simplify(dM*dS*dX + kap**6*eta**6/(64**3*c**6*s**6)*BM*BS*BX)==0
# longitudinal second derivatives at pins
out['PXX_M'] = sp.simplify(at(sp.diff(P,X,2),-eta/2,0)); out['PXX_S']=sp.simplify(at(sp.diff(P,X,2),eta/2,0))
# transverse chart c=0, s=sg
for sg in (1,-1):
    Ab = sp.symbols('Abar',real=True)
    eqs=[at(P,0,sg)+al*kap*eta**3/6, at(PX,0,sg), at(PY,0,sg)]
    st = sp.solve(eqs,[w,Q,z],dict=True)[0]
    sub = dict(st); sub[a]=sg*Ab*kap   # Abar = s a / kappa, s=+-1
    tM = sp.simplify(H.subs({X:-eta/2,Y:0}).det().subs(st).subs(a,sg*Ab*kap))
    tS = sp.simplify(H.subs({X:eta/2,Y:0}).det().subs(st).subs(a,sg*Ab*kap))
    tX = sp.simplify(H.subs({X:0,Y:sg}).det().subs(st).subs(a,sg*Ab*kap))
    out[f'4.1_s={sg}'] = sp.simplify(tM - kap**2*eta**2/4*(4*al*eta**2-(Ab+eta)**2))==0
    out[f'4.2_s={sg}'] = sp.simplify(tS - kap**2*eta**2/4*(4*(1-al)*eta**2-(Ab-eta)**2))==0
    out[f'4.3_s={sg}'] = sp.simplify(tX + kap**2*eta**2/4*((Ab+(1-2*al)*eta)**2+4*al*(1-al)*eta**2))==0
    out[f'transverse_w_s={sg}'] = sp.simplify(st[w])
# covariance Sigma_0
S0 = sp.Matrix([[s**2*(3*c**4+3*c**2*s**2+s**4)/24, -c*s**2*(2*c**2+s**2)/4, 0],
                [-c*s**2*(2*c**2+s**2)/4, s**2*(4*c**2+s**2)/2, 0],[0,0,2*s**2]])
out['6.2'] = sp.simplify(S0.det() - s**8*(c**2+s**2)*(3*c**2+s**2)/24)==0
cv = sp.simplify(S0[0,0]-S0[0,1]**2/S0[1,1])
out['6.3'] = sp.simplify(cv - s**4*(c**2+s**2)*(3*c**2+s**2)/(24*(4*c**2+s**2)))==0
out['6.2_at_c0_s1'] = S0.det().subs({c:0,s:1})
out['VarY2_le_2s2'] = sp.simplify(2*s**2 - S0[1,1]).subs(on_circle)  # = s^2*(... )>=0 ?
# 7.1
out['7.1_c2_min'] = sp.Rational(3,4)-sp.Rational(4,15)/4
out['7.1_m2_min'] = sp.Rational(3,4)*(sp.Rational(3,4)-sp.Rational(4,15)/4)
# collar exclusion eta^2 bound
t_,r_ = sp.symbols('t r',positive=True)
out['eta2_at_tmin'] = sp.simplify((r_/(sp.sqrt(15)*r_/2))**2)
# axis chart
Ae = 1-sp.Rational(4,15)/4
out['A_eta_min'] = Ae
Aeta = sp.symbols('A_eta',positive=True)
out['axis_detcov'] = sp.simplify(Aeta**6/1920*2*Aeta**2/3*Aeta**2/2)
# (8.1)-(8.3) ledger
out['t6eta6'] = sp.simplify(t_**6*(r_/t_)**6)
out['8.2_power'] = sp.simplify(r_**-2 * t_**-6 * r_**6)
I = sp.integrate(r_**4*t_**-6*sp.Rational(5,24)*r_**3*t_, (t_, sp.sqrt(15)*r_/2, sp.Symbol('delta',positive=True)))
out['8.3_integral'] = sp.simplify(I)
out['8.3_leading_coeff_of_r3'] = sp.limit(I/r_**3, r_, 0)
for k,v in out.items(): print(k, '=>', v)
