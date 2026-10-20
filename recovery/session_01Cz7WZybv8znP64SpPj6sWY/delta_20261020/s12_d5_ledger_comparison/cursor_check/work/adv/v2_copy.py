# Independent finite-t test of LS-DER-033 (8.1), in PHYSICAL coordinates (not the lens's scaled P4 layer).
# f = general quartic Taylor polynomial about the midpoint x0=0 (15 coefficients).
# Impose: six pins at M=(-r/2,0), S=(r/2,0) and the witness value+gradient at y=t(c,s) (9 linear constraints).
# Free: c_{12} (= w-type cubic jet) and the five quartic coefficients, held at fixed generic O(1) values.
import sympy as sp, mpmath as mp
x,y = sp.symbols('x y', real=True)
mons = [(i,j) for n in range(5) for i in range(n+1) for j in [n-i]]
C = {m: sp.Symbol(f'c{m[0]}{m[1]}') for m in mons}
f = sum(C[m]*x**m[0]*y**m[1] for m in mons)
r,t,cc,ss,b,kap,al = sp.symbols('r t c s b kappa alpha', real=True)
fx,fy = f.diff(x), f.diff(y)
ev = lambda g,px,py: g.subs({x:px,y:py}, simultaneous=True)
eqs = [ev(f,-r/2,0)-b, ev(fx,-r/2,0), ev(fy,-r/2,0),
       ev(f, r/2,0)-(b-kap*r**3/6), ev(fx,r/2,0), ev(fy,r/2,0),
       ev(f,t*cc,t*ss)-(b-al*kap*r**3/6), ev(fx,t*cc,t*ss), ev(fy,t*cc,t*ss)]
free = [C[(1,2)], C[(4,0)], C[(3,1)], C[(2,2)], C[(1,3)], C[(0,4)]]
unk = [C[m] for m in mons if C[m] not in free]
sol = sp.solve(eqs, unk, dict=True)[0]
fs = f.subs(sol)
Hs = sp.hessian(fs,(x,y))
dM = Hs.subs({x:-r/2,y:0}).det(); dS = Hs.subs({x:r/2,y:0}).det(); dY = Hs.subs({x:t*cc,y:t*ss}).det()
# generic jets (taylor coeff c_ij = D^{i,j}f/(i! j!)); kappa via c30 = kappa/3 is forced by pins (check)
print('forced c30 =', sp.simplify(sol[C[(3,0)]].subs(t*0,0)).series(r,0,1).removeO() if False else sp.limit(sol[C[(3,0)]], r, 0))
vals = {b:sp.Rational(6,5), kap:1, al:sp.Rational(1,2), cc:sp.Rational(3,5), ss:sp.Rational(4,5),
        C[(1,2)]:sp.Rational(3,20), C[(4,0)]:sp.Rational(7,240), C[(3,1)]:sp.Rational(-11,60), C[(2,2)]:sp.Rational(9,40),
        C[(1,3)]:sp.Rational(1,18), C[(0,4)]:sp.Rational(-1,48)}
fM = sp.lambdify((r,t), dM.subs(vals), 'mpmath'); fS = sp.lambdify((r,t), dS.subs(vals), 'mpmath'); fY = sp.lambdify((r,t), dY.subs(vals), 'mpmath')
mp.mp.dps = 60
print('\nFixed t=1/10: dets and ratios as r->0')
for R in ['1e-3','1e-4','1e-5','1e-6']:
    R = mp.mpf(R); T = mp.mpf(1)/10
    m,s_,yy = fM(R,T), fS(R,T), fY(R,T)
    print(f' r={mp.nstr(R,2)}: detM/r={mp.nstr(m/R,6)} detS/r={mp.nstr(s_/R,6)} detY={mp.nstr(yy,6)} |prod|/r^2={mp.nstr(abs(m*s_*yy)/R**2,6)} |prod|/r^6={mp.nstr(abs(m*s_*yy)/R**6,4)}')
print('\nt=r^beta: |prod|/r^6')
for beta in ['1/2','3/5','2/3','3/4','9/10']:
    bb = mp.mpf(sp.Rational(beta)); row=[]
    for R in ['1e-2','1e-4','1e-6','1e-8','1e-10']:
        R = mp.mpf(R); T = R**bb
        row.append(mp.nstr(abs(fM(R,T)*fS(R,T)*fY(R,T))/R**6,4))
    print(f' beta={beta}:', row)
# also the physical S_0=f_yy(M) vs the Euler prediction 2 t^2 P4h(omega)/s^2 - w r/2 (P4h = quartic at omega)
S0 = sp.lambdify((r,t), Hs[1,1].subs({x:-r/2,y:0}).subs(vals), 'mpmath')
P4h = sum(vals[C[m]]*vals[cc]**m[0]*vals[ss]**m[1] for m in [(4,0),(3,1),(2,2),(1,3),(0,4)])
print('\nS0=f_yy(M) vs 2t^2 P4h/s^2 (t=1/10, r=1e-8):', mp.nstr(S0(mp.mpf('1e-8'), mp.mpf(1)/10),8), mp.nstr(mp.mpf(2)*mp.mpf(1)/100*mp.mpf(P4h)/mp.mpf(vals[ss])**2,8))
print('S0 vs 2t^2 P4h/s^2 (t=1/1000, r=1e-12):', mp.nstr(S0(mp.mpf('1e-12'), mp.mpf(1)/1000),8), mp.nstr(mp.mpf(2)*mp.mpf(1)/10**6*mp.mpf(P4h)/mp.mpf(vals[ss])**2,8))
