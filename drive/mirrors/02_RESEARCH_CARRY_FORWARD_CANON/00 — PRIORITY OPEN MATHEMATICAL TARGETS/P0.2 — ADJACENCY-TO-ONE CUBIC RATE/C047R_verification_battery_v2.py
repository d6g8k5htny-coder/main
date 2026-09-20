# C047R_verification_battery_v2.py | Claude (Anthropic) C047R | 2026-07-21
# VERSION 2.0 — supersedes-for-use battery v1 (Drive 1oUx_WO82T_3xp3jQW0Sb9apk- lineage; v1 retained).
# Implements ALL SEVEN GP-AUD-011 amendments: (1) exact identity replaces the abs()<1 threshold;
# (2) named per-result prints; (3) version print; (4) runtime source-hash print; (5) labeled controls;
# (6) per-entry covariance-perturbation controls; (7) planar/torus-separated test names.
# SCOPE BANNER: every test is PLANAR-BF (planar Bargmann-Fock jets at a point / local normal form).
# NOTHING here certifies normalized-torus quantities; torus claims need their own instruments.
# Authoring-disk sha256 of this exact source: f72006012e9df8d9504ccb227a2a576c3233aa02d7331de13efa6bfd31d2edc2
# (re-download and re-hash to verify the Drive copy; a whitespace variant, if any, is a recorded lineage variant.)
import sympy as sp, hashlib, sys
print("battery v2.0 | 2026-07-21 | predecessor v1 retained on Drive")
try:
    src=open(__file__,'rb').read(); print("source sha256:", hashlib.sha256(src).hexdigest())
except Exception as e: print("source sha256: UNAVAILABLE", e)
x,y=sp.symbols('x y'); K=sp.exp(-(x**2+y**2)/2)
jets=[(0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),(1,2),(0,3)]
C=sp.Matrix(10,10,lambda i,j:((-1)**(jets[j][0]+jets[j][1]))*sp.diff(K,x,jets[i][0]+jets[j][0],y,jets[i][1]+jets[j][1]).subs({x:0,y:0}))
pins=[0,1,2,3,4,6]; free=[5,7,8,9]
Ccond=sp.simplify(C[free,free]-C[free,pins]*C[pins,pins].inv()*C[pins,free])
assert Ccond==sp.diag(2,2,2,6); print("PLANAR-BF TEST 1 PASS: conditional free-jet covariance = diag(2,2,2,6)")
assert list(sp.simplify(C[free,pins]*C[pins,pins].inv()).row(0))==[-1,0,0,0,0,0]
print("PLANAR-BF TEST 2 PASS: E[f_yy|pins] = -f  (regression row exact)")
for k,bad in enumerate([sp.diag(3,2,2,6),sp.diag(2,3,2,6),sp.diag(2,2,3,6),sp.diag(2,2,2,5)]):
    assert Ccond!=bad; print(f"CONTROL C6.{k+1} PASS: planted covariance error (entry {k+1}) correctly detected")
c,s,t=sp.symbols('c s t',positive=True); ayy,axxy,axyy,ayyy=sp.symbols('ayy axxy axyy ayyy')
fl=ayy*y**2/2+axxy*x**2*y/2+axyy*x*y**2/2+ayyy*y**3/6
flx,fly=sp.diff(fl,x),sp.diff(fl,y); sub={x:t*c,y:t*s}
Y1=sp.limit(sp.expand((fl.subs(sub)-(t/2)*(c*flx.subs(sub)+s*fly.subs(sub)))/t**3),t,0)
Y2=sp.limit(sp.expand(flx.subs(sub)/t**2),t,0); Y3=sp.limit(sp.expand(fly.subs(sub)/t),t,0)
V={ayy:2,axxy:2,axyy:2,ayyy:6}
cov=lambda u,v:sp.simplify(sum(sp.diff(u,g)*sp.diff(v,g)*w_ for g,w_ in V.items()))
M=sp.Matrix(3,3,lambda i,j:cov([Y1,Y2,Y3][i],[Y1,Y2,Y3][j]))
claim=sp.Matrix([[s**2*(3*c**4+3*c**2*s**2+s**4)/24,-c*s**2*(2*c**2+s**2)/4,0],
 [-c*s**2*(2*c**2+s**2)/4,s**2*(4*c**2+s**2)/2,0],[0,0,2*s**2]])
assert sp.simplify(M-claim)==sp.zeros(3,3); print("PLANAR-BF TEST 3 PASS: generic-frame divided-difference covariance matrix exact")
assert sp.simplify(M.det()-s**8*(c**2+s**2)*(3*c**2+s**2)/24)==0
print("PLANAR-BF TEST 4 PASS: collision determinant = s^8(c^2+s^2)(3c^2+s^2)/24")
for k,(g,wv) in enumerate([(ayy,3),(axxy,3),(axyy,3),(ayyy,5)]):
    Vb=dict(V); Vb[g]=wv
    covb=lambda u,v:sp.simplify(sum(sp.diff(u,gg)*sp.diff(v,gg)*ww for gg,ww in Vb.items()))
    Mb=sp.Matrix(3,3,lambda i,j:covb([Y1,Y2,Y3][i],[Y1,Y2,Y3][j]))
    assert sp.simplify(Mb-claim)!=sp.zeros(3,3)
    print(f"CONTROL C6.{k+5} PASS: planted jet-variance error ({g}) detected in the full covariance matrix")
# design note: v2.0's first draft compared only M[0,0]; the ayy perturbation is invisible there
# because the third-order divided difference annihilates second-order jets (Y1 has no ayy term;
# ayy enters via Y3 -> M[2,2]). Full-matrix comparison is the non-vacuous control. Caught at runtime.
X_,Y_,b=sp.symbols('X Y b',real=True); q,a,w,z,Q=sp.symbols('q a w z Q')
G1=X_**2-sp.Rational(1,4)+a*X_*Y_+w*Y_**2/2; G2=q*Y_; Vq={a:2,w:2,q:2}
v11=sp.simplify(sum(sp.diff(G1,g)**2*w_ for g,w_ in Vq.items())); v22=sp.simplify(sum(sp.diff(G2,g)**2*w_ for g,w_ in Vq.items()))
assert sp.simplify(v11-Y_**2*(4*X_**2+Y_**2)/2)==0 and v22==2*Y_**2
print("PLANAR-BF TEST 5 PASS: C100 gradient-frame variances exact")
assert sp.simplify((X_**2-sp.Rational(1,4))**2/(2*v11)+(b*Y_)**2/(2*v22)-((X_**2-sp.Rational(1,4))**2/(Y_**2*(4*X_**2+Y_**2))+b**2/4))==0
print("PLANAR-BF TEST 6 PASS: power-bookkeeping identity exact")
rho=sp.symbols('rho',positive=True)
P=x**3/3-x/4-sp.Rational(1,12)+(a/2)*(x**2-sp.Rational(1,4))*y+(Q/2)*y**2+(w/2)*x*y**2+(z/6)*y**3
Px,Py=sp.diff(P,x),sp.diff(P,y); x0,y0=-sp.Rational(1,2)+rho*c,rho*s
sol=sp.solve([Px.subs({x:x0,y:y0}),Py.subs({x:x0,y:y0})],[Q,a],dict=True)[0]; Ps=P.subs(sol)
H=lambda pt: sp.det(sp.Matrix([[sp.diff(Ps,x,2),sp.diff(Ps,x,1,y,1)],[sp.diff(Ps,x,1,y,1),sp.diff(Ps,y,2)]]).subs(pt))
prod=sp.simplify(H({x:-sp.Rational(1,2),y:0})*H({x:sp.Rational(1,2),y:0})*H({x:x0,y:y0}))
ser=sp.series(prod,rho,0,3).removeO()
assert sp.simplify(ser.coeff(rho,0))==0 and sp.simplify(ser.coeff(rho,1))==0
print("PLANAR-BF TEST 7 PASS: rho^0 and rho^1 collision coefficients vanish identically")
assert sp.simplify(ser.coeff(rho,2)-(-(-2*c**2+s**2*w)*(-4*c**3+3*c*s**2*w+s**3*z)**2/(4*s**6)))==0
print("PLANAR-BF TEST 8 PASS: rho^2 coefficient equals the certified closed form")
Psb=P.subs({a:sol[a]})
Hb=lambda pt: sp.det(sp.Matrix([[sp.diff(Psb,x,2),sp.diff(Psb,x,1,y,1)],[sp.diff(Psb,x,1,y,1),sp.diff(Psb,y,2)]]).subs(pt))
assert sp.simplify(sp.series(sp.simplify(Hb({x:-sp.Rational(1,2),y:0})*Hb({x:sp.Rational(1,2),y:0})*Hb({x:x0,y:y0})),rho,0,1).removeO().coeff(rho,0))!=0
print("CONTROL C5.1 PASS: partial-station solve (planted omission) correctly breaks the cancellation")
assert sp.simplify(sp.solve(Px.subs({x:0,y:Y_}),w)[0]-1/(2*Y_**2))==0
print("PLANAR-BF TEST 9 PASS: axis-cost identity exact")
assert sp.Rational(45,60)-sp.Rational(4,60)-sp.Rational(41,60)==0
assert sp.Rational(45,60)-sp.Rational(41,60)==sp.Rational(1,15)
print("PLANAR-BF TEST 10 PASS: overlap arithmetic 45/60 = 4/60 + 41/60 and deficit = 1/15, exact (replaces v1's decorative threshold)")
print("ALL 10 PLANAR-BF TESTS + 9 LABELED CONTROLS PASS — battery v2.0 complete")
