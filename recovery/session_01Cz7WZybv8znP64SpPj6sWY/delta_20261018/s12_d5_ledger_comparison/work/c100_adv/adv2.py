# Check C100's exact cubic-model three-determinant product against P2's deterministic (P16), (P17)
# (these are theorems for ANY C^3 function with three critical points, so the cubic model must obey them).
import sympy as sp, random, math
x, y, X, Y = sp.symbols('x y X Y', real=True)
a, Q, w, z = sp.symbols('a Q w z', real=True)
P = (x**3/3 - x/4 - sp.Rational(1,12) + a/2*(x**2-sp.Rational(1,4))*y + Q/2*y**2 + w/2*x*y**2 + z/6*y**3)
Px, Py = sp.diff(P,x), sp.diff(P,y)
sol = sp.solve([Px.subs({x:X,y:Y}), Py.subs({x:X,y:Y})],[a,Q],dict=True)[0]
H = sp.hessian(P,(x,y))
fa = sp.lambdify((X,Y,w,z), sol[a], modules="math"); fQ = sp.lambdify((X,Y,w,z), sol[Q], modules="math")
fH = sp.lambdify((x,y,a,Q,w,z), H.tolist(), modules="math")
def T3(av, wv, zv, th):  # D^3P[u,u,u], u=(cos th, sin th); P_xxx=2, P_xxy=a, P_xyy=w, P_yyy=z
    c, s = math.cos(th), math.sin(th)
    return 2*c**3 + 3*av*c**2*s + 3*wv*c*s**2 + zv*s**3
def Lnorm(av,wv,zv):
    return max(abs(T3(av,wv,zv, 2*math.pi*i/4000)) for i in range(4000))
def det2(m):
    return m[0][0]*m[1][1]-m[0][1]*m[1][0]
def opn(m):
    a_,b_,d_ = m[0][0], m[0][1], m[1][1]
    tr = (a_+d_)/2; disc = math.sqrt(((a_-d_)/2)**2 + b_**2)
    return max(abs(tr+disc), abs(tr-disc))
random.seed(1)
worst16 = 0; worst17 = 0; worstC = 0
for trial in range(4000):
    rho = 10**random.uniform(-4, math.log10(0.25))
    th = random.uniform(0, 2*math.pi)
    c, s = math.cos(th), math.sin(th)
    if abs(s) < 1e-6: continue
    wv = random.gauss(0, 2**0.5); zv = random.gauss(0, 6**0.5)
    Xv, Yv = -0.5 + rho*c, rho*s
    av = fa(Xv,Yv,wv,zv); Qv = fQ(Xv,Yv,wv,zv)
    dM = det2(fH(-0.5,0,av,Qv,wv,zv))
    dS = det2(fH(0.5,0,av,Qv,wv,zv))
    dY = det2(fH(Xv,Yv,av,Qv,wv,zv))
    prod = abs(dM*dS*dY)          # = |prod| / r^6 in physical units
    L = Lnorm(av,wv,zv)
    # (P16): |prod| <= L^6 q^2 (1+(p/q)^2)^3 with p=rho c, q=rho s   (r^6 cancels)
    rhs16 = L**6 * (rho*s)**2 * (1 + (c/s)**2)**3
    worst16 = max(worst16, prod/rhs16)
    # (P17): |prod_phys| <= (1/8) L^3 K2^3 r d^2. In the cubic model the physical Hessian is r D2P, K2 = r*sup||D2P|| on triangle
    grid = [i/8 for i in range(9)]
    pts = [(-0.5 + t*(Xv+0.5) + u*(1.0), t*Yv) for t in grid for u in grid if t+u<=1+1e-12]
    K2s = max(opn(fH(px,py,av,Qv,wv,zv)) for px,py in pts)
    rhs17 = 0.125 * L**3 * K2s**3 * rho**2   # physical: r^6 cancels (L^3 r^3K2s^3 r (r rho)^2 = r^6 ...)
    worst17 = max(worst17, prod/rhs17)
    # C100 leading factor check vs exact/rho^2
    lead = -((-2*c**2 + s**2*wv)*(-4*c**3+3*c*s**2*wv+s**3*zv)**2)/(4*s**6)
    if rho < 1e-3:
        worstC = max(worstC, abs(dM*dS*dY/rho**2 - lead)/max(1,abs(lead)))
print('max ratio exact/P16 bound:', worst16)
print('max ratio exact/P17 bound:', worst17)
print('max rel dev of exact/rho^2 from C100 leading factor (rho<1e-3):', worstC)
