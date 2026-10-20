# Independent re-derivation of LS-DER-033 leading-order identities (written fresh, not copied).
import sympy as sp
c,s,e,k,al,a,Q,w,z,X,Y = sp.symbols('c s eta kappa alpha a Q w z X Y', real=True)
P = k*(X**3/3 - e**2*X/4 - e**3/12) + a/2*(X**2-e**2/4)*Y + Q/2*Y**2 + w/2*X*Y**2 + z/6*Y**3
gX, gY = P.diff(X), P.diff(Y)
ev = lambda f,x,y: f.subs({X:x, Y:y}, simultaneous=True)
print('pins grad:', [sp.simplify(ev(g,x0,0)) for g in (gX,gY) for x0 in (-e/2,e/2)])
print('pin values M,S:', sp.simplify(ev(P,-e/2,0)), sp.simplify(ev(P,e/2,0)))
# (P4)-(P5) bridge: x=r/2+tX from M, r=t*eta, kappa=6k
t,r,kk,b,x,T3 = sp.symbols('t r k b x T3', real=True)
H = b + kk*(2*x**3 - 3*r*x**2)
lhs = (b + t**3*P.subs({Y:0})).subs({k:6*kk})
print('cubic == P2 H(x):', sp.simplify(lhs - H.subs({x: r/2+t*X}).subs({r:t*e})) == 0)
p = sp.symbols('p', real=True)
hh = (t**2*(a/2)*(X**2-e**2/4)).subs({X:(r*p - r/2)/t, e:r/t})
print('transverse term == (r^2/2)p(p-1)a:', sp.simplify(hh - r**2/2*p*(p-1)*a) == 0)
# station elimination
sol = sp.solve([ev(P,c,s)+al*k*e**3/6, ev(gX,c,s), ev(gY,c,s)], [a,Q,z], dict=True)
assert len(sol)==1; sol=sol[0]
A_ = (-c**2*k + e**2*k/4 - s**2*w/2)/(c*s)
Q_ = e**2*(-al*c*e*k + c**2*k/2 + c*e*k/2 + e**2*k/8 - s**2*w/4)/(c*s**2)
Z_ = (32*al*c*e**3*k + 16*c**4*k - 24*c**2*e**2*k - 24*c**2*s**2*w - 16*c*e**3*k - 3*e**4*k + 6*e**2*s**2*w)/(16*c*s**3)
print('a,Q,z match:', [sp.simplify(sol[v]-f)==0 for v,f in ((a,A_),(Q,Q_),(z,Z_))])
Hs = sp.hessian(P,(X,Y))
d = {n: sp.simplify(Hs.subs({X:x0,Y:y0}).det().subs(sol)) for n,(x0,y0) in {'M':(-e/2,0),'S':(e/2,0),'y':(c,s)}.items()}
W = w/k; K = 2*W*s**2 - 4*c**2 - 4*c*e - e**2
pre = k**2*e**2/(64*c**2*s**2)
BM = 64*al*c**2*e**2 - K**2; BS = 64*(1-al)*c**2*e**2 - (K+8*c*e)**2; BX = (K+8*al*c*e)**2 + 64*al*(1-al)*c**2*e**2
print('(3.1),(3.2),(3.3) without c^2+s^2=1:', [sp.simplify(d['M']-pre*BM)==0, sp.simplify(d['S']-pre*BS)==0, sp.simplify(d['y']+pre*BX)==0])
print('(3.4):', sp.simplify(d['M']*d['S']*d['y'] + k**6*e**6/(64**3*c**6*s**6)*BM*BS*BX)==0)
print('Pxx at M,S:', sp.simplify(ev(P.diff(X,2),-e/2,0)), sp.simplify(ev(P.diff(X,2),e/2,0)))
# transverse c=0
Ab = sp.symbols('Abar', real=True)
for sg in (1,-1):
    st = sp.solve([ev(P,0,sg)+al*k*e**3/6, ev(gX,0,sg), ev(gY,0,sg)], [w,Q,z], dict=True)[0]
    sub = lambda f: sp.simplify(f.subs(st).subs(a, sg*k*Ab))
    tM = sub(Hs.subs({X:-e/2,Y:0}).det()); tS = sub(Hs.subs({X:e/2,Y:0}).det()); tY = sub(Hs.subs({X:0,Y:sg}).det())
    print('s=',sg,'w=',st[w], '(4.1-3):', [sp.simplify(tM - k**2*e**2/4*(4*al*e**2-(Ab+e)**2))==0,
          sp.simplify(tS - k**2*e**2/4*(4*(1-al)*e**2-(Ab-e)**2))==0,
          sp.simplify(tY + k**2*e**2/4*((Ab+(1-2*al)*e)**2+4*al*(1-al)*e**2))==0])
# Sigma_0
S0 = sp.Matrix([[s**2*(3*c**4+3*c**2*s**2+s**4)/24, -c*s**2*(2*c**2+s**2)/4, 0],
                [-c*s**2*(2*c**2+s**2)/4, s**2*(4*c**2+s**2)/2, 0],[0,0,2*s**2]])
print('(6.2):', sp.simplify(S0.det() - s**8*(c**2+s**2)*(3*c**2+s**2)/24)==0, ' at (0,1):', S0.det().subs({c:0,s:1}))
print('(6.3):', sp.simplify(S0[0,0]-S0[0,1]**2/S0[1,1] - s**4*(c**2+s**2)*(3*c**2+s**2)/(24*(4*c**2+s**2)))==0)
print('2s^2-Var(Y2) on circle:', sp.factor(sp.expand((2*s**2 - S0[1,1]).subs(c**2, 1-s**2))))
# (7.1), axis, eta bound
print('c^2-eta^2/4 min:', sp.Rational(3,4)-sp.Rational(4,15)/4, ' m2 min:', sp.Rational(3,4)*(sp.Rational(3,4)-sp.Rational(1,15)))
print('eta^2 at t=sqrt15/2 r:', sp.nsimplify((2/sp.sqrt(15))**2), ' A_eta min:', 1-sp.Rational(1,15))
Ae = sp.symbols('A', positive=True); print('axis det:', sp.simplify(Ae**6/1920*2*Ae**2/3*Ae**2/2))
# P2 drift identity
print("P2 H'(r/2+tc) == kappa t^2 (c^2-eta^2/4):", sp.simplify(sp.diff(H,x).subs({x:r/2+t*c}).subs({r:t*e, kk:k/6}) - k*t**2*(c**2-e**2/4))==0)
