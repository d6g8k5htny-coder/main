import sympy as sp
c,s,e,k,al,a,Q,w,z,X,Y,t = sp.symbols('c s eta kappa alpha a Q w z X Y t', real=True)
q = sp.symbols('q0:5', real=True)
P3 = k*(X**3/3 - e**2*X/4 - e**3/12) + a/2*(X**2-e**2/4)*Y + Q/2*Y**2 + w/2*X*Y**2 + z/6*Y**3
# general pin-preserving quartic layer: (X^2-e^2/4)^2, X(X^2-e^2/4)Y, and Y^2*(quadratic)
P4 = q[0]*(X**2-e**2/4)**2 + q[1]*X*(X**2-e**2/4)*Y + Y**2*(q[2]*X**2 + q[3]*X*Y + q[4]*Y**2)
P = P3 + t*P4
ev = lambda f,x0,y0: f.subs({X:x0,Y:y0}, simultaneous=True)
for g in (P4, P4.diff(X), P4.diff(Y)):
    assert all(sp.simplify(ev(g,x0,0))==0 for x0 in (-e/2,e/2))
sol = sp.solve([ev(P,c,s)+al*k*e**3/6, ev(P.diff(X),c,s), ev(P.diff(Y),c,s)], [a,Q,z], dict=True)[0]
P4h = ev(P4.subs(e,0), c, s)
dQ = sp.simplify(sp.diff(sol[Q], t).subs({t:0, e:0}))
print('dQ/dt at t=0,eta=0  ==  2 P4h/s^2 :', sp.simplify(dQ - 2*P4h/s**2)==0)
Hs = sp.hessian(P,(X,Y))
dets = {n: Hs.subs({X:x0,Y:y0}).det().subs(sol) for n,(x0,y0) in {'M':(-e/2,0),'S':(e/2,0),'y':(c,s)}.items()}
for n,sg in (('M',-1),('S',1)):
    d1 = sp.diff(dets[n], t).subs(t,0)
    coef_eta1 = sp.simplify(sp.diff(d1, e).subs(e,0))
    print(f'd(det_{n})/dt at t=0: value at eta=0 ->', sp.simplify(d1.subs(e,0)), '; eta^1 coeff == %+d*2*kappa*P4h/s^2 :'%sg, sp.simplify(coef_eta1 - sg*2*k*P4h/s**2)==0)
dy1 = sp.simplify(sp.diff(dets['y'], t).subs({t:0, e:0}))
print('d(det_y)/dt at t=0,eta=0 == (2c^2 kappa - s^2 w) P4h/(c s^2):', sp.simplify(dy1 - (2*c**2*k - s**2*w)*P4h/(c*s**2))==0)
print('det_y at t=0,eta=0:', sp.simplify(dets['y'].subs({t:0,e:0})))
