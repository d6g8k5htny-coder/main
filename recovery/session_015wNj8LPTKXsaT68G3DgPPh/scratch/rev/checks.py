from fractions import Fraction as Fr
import math, random

# ---------- polynomial helpers in v with Fraction coefficients (dict power->coeff), parameters as Fractions
def b0(u): return 3 - 12*u*u
def Q_poly_v(u):
    # Q(u,v) = (4v^2 - b0^2)(b0 - 4u v) as polynomial in v
    b = b0(u)
    # (4v^2 - b^2)(b - 4u v) = 4b v^2 - 16u v^3 - b^3 + 4u b^2 v
    return {3: -16*u, 2: 4*b, 1: 4*u*b*b, 0: -b**3}
def integ_v(poly, lo, hi):
    s = Fr(0)
    for p, c in poly.items():
        s += c * (hi**(p+1) - lo**(p+1)) / (p+1)
    return s
def mul_poly(pa, pb):
    out = {}
    for i, a in pa.items():
        for j, b in pb.items():
            out[i+j] = out.get(i+j, 0) + a*b
    return out

# integrate f(u) * Q(u,v) over S exactly where f(u) is polynomial in u: do v-integration symbolically in u?
# Simpler: Gaussian-free exact route: integrate over v gives a polynomial in u (piecewise), then integrate in u exactly.
# Build polynomials in u using Fraction-coefficient polynomial arithmetic.
class P:  # univariate polynomial in u
    def __init__(self, c): self.c = {k: Fr(v) for k, v in c.items() if v != 0}
    def __add__(self, o): 
        c = dict(self.c); 
        for k, v in o.c.items(): c[k] = c.get(k, 0) + v
        return P(c)
    def __sub__(self, o): return self + o*(-1)
    def __mul__(self, o):
        if not isinstance(o, P): return P({k: v*o for k, v in self.c.items()})
        c = {}
        for i, a in self.c.items():
            for j, b in o.c.items(): c[i+j] = c.get(i+j, 0) + a*b
        return P(c)
    def __pow__(self, n):
        r = P({0: 1})
        for _ in range(n): r = r*self
        return r
    def integ(self, lo, hi):
        return sum(v*(Fr(hi)**(k+1) - Fr(lo)**(k+1))/(k+1) for k, v in self.c.items())
U = P({1: 1}); ONE = P({0: 1})
B0 = ONE*3 - U*U*12
def vint(F, lo, hi):
    """integral_{lo}^{hi} F(u,v) dv where F given as dict power_of_v -> P(u), and lo, hi are P(u)."""
    tot = P({})
    for p, cu in F.items():
        tot = tot + cu * (hi**(p+1) - lo**(p+1)) * Fr(1, p+1)
    return tot
Qv = {3: U*(-16), 2: B0*4, 1: U*B0*B0*4, 0: B0**3*(-1)}
def shape_integral(weight):  # weight: P(u)
    Fw = {p: cu*weight for p, cu in Qv.items()}
    # strip 1: u in (-3/2,-1/2): lower |b0|/2 = (12u^2-3)/2, upper 3-6u
    lo1 = (U*U*12 - ONE*3)*Fr(1,2); hi = ONE*3 - U*6
    s1 = vint(Fw, lo1, hi).integ(Fr(-3,2), Fr(-1,2))
    lo2 = B0*Fr(1,2)
    s2 = vint(Fw, lo2, hi).integ(Fr(-1,2), Fr(1,2))
    return s1, s2, s1+s2
I = shape_integral(ONE)[2]; U2 = shape_integral(U*U)[2]; Bint = shape_integral(B0)[2]
# |u|: split at 0: strip (-3/2,-1/2) has u<0 -> -u ; (-1/2,0): -u ; (0,1/2): u
Fw = {p: cu*U for p, cu in Qv.items()}
lo1 = (U*U*12 - ONE*3)*Fr(1,2); hi = ONE*3 - U*6; lo2 = B0*Fr(1,2)
Uabs = -vint(Fw, lo1, hi).integ(Fr(-3,2), Fr(-1,2)) - vint(Fw, lo2, hi).integ(Fr(-1,2), 0) + vint(Fw, lo2, hi).integ(0, Fr(1,2))
print('I  =', I, '=?', Fr(246528,35), I == Fr(246528,35))
print('U2 =', U2, '=?', Fr(240192,35), U2 == Fr(240192,35))
print('B  =', Bint, '=?', Fr(-428544,7), Bint == Fr(-428544,7), ' 3I-12U2 =', 3*I-12*U2 == Bint)
print('Uabs=', Uabs, '=?', Fr(5898627,880), Uabs == Fr(5898627,880))
# height moments under Q/I: E eta = 5771/7062, E eta^2 = 2440/3531 with eta = u+1/2+v/6
def eta_moment(j):
    # (u+1/2+v/6)^j expanded as poly in v with P(u) coeffs
    base = {0: U + ONE*Fr(1,2), 1: ONE*Fr(1,6)}
    cur = {0: ONE}
    for _ in range(j):
        new = {}
        for p, cu in cur.items():
            for q, cv in base.items(): new[p+q] = new.get(p+q, P({})) + cu*cv
        cur = new
    Fw = {}
    for p, cu in cur.items():
        for q, cq in Qv.items(): Fw[p+q] = Fw.get(p+q, P({})) + cu*cq
    return vint(Fw, lo1, hi).integ(Fr(-3,2), Fr(-1,2)) + vint(Fw, lo2, hi).integ(Fr(-1,2), Fr(1,2))
print('E eta =', eta_moment(1)/I, '=?', Fr(5771,7062), eta_moment(1)/I == Fr(5771,7062))
print('E eta^2 =', eta_moment(2)/I, '=?', Fr(2440,3531), eta_moment(2)/I == Fr(2440,3531))

# ---------- #169: companion map, t_c sign, E16, E17, sectors
def companion(u, v):
    b = b0(u); L = 4*v*v - 8*b*u*v + b*b
    tc = (4*v*v - b*b)/L
    uc = (2*b*v - u*(4*v*v + b*b))/L
    vc = v*tc*tc
    return uc, vc, tc, L
def inS(u, v):
    return Fr(-3,2) < u < Fr(1,2) and abs(b0(u))/2 < v < 3 - 6*u
def eta(u, v): return u + Fr(1,2) + v/6
def cub_n2(u, v):
    # CUB: n=2 iff v < -b and 12(v-bu)^2 < (v+b)^2 (2v-b)
    b = b0(u)
    return v < -b and 12*(v-b*u)**2 < (v+b)**2*(2*v-b)
random.seed(7)
bad = 0; n2 = 0; checked = 0
for _ in range(20000):
    u = Fr(random.randint(-1500, 499), 1000); v = Fr(random.randint(1, 9000), 1000)
    if not inS(u, v): continue
    checked += 1
    uc, vc, tc, L = companion(u, v)
    if L == 0: continue
    is2 = inS(uc, vc)
    if is2 != cub_n2(u, v): bad += 1; print('classifier mismatch', u, v)
    if is2:
        n2 += 1
        p = -u; A = 12*p*p - 3
        # E16, t_c<0, involution, E13
        if not (tc < 0): print('tc not negative', u, v, tc); bad += 1
        if (abs(tc) > 1) != (v > A*p): print('E16 fail', u, v); bad += 1
        ucc, vcc, tcc, _ = companion(uc, vc)
        if not (ucc == u and vcc == v and tc*tcc == 1): print('involution fail', u, v); bad += 1
        if not (b0(uc) == b0(u)*tc*tc and vc - b0(uc)*uc == (v - b0(u)*u)*tc**3): print('E13 fail'); bad += 1
        ell = v/A
        e17 = 2*(ell - p)*(4*p*ell - 1)**3/(4*ell*ell - 8*p*ell + 1)**2
        if eta(uc, vc) - eta(u, v) != e17: print('E17 fail', u, v); bad += 1
        # outer anchor (|tc|<1, v<Ap): eta_c < eta
        if abs(tc) < 1 and not (eta(uc, vc) < eta(u, v)): print('ordering fail'); bad += 1
        # sector membership E14
        Vp = None
print('S samples', checked, 'doublets', n2, 'failures', bad)

# ---------- D via the primitive: D = int_{1/2}^1 [P0(p,V_+(p)) - P0(p,A p)] dp (float GL) vs closed form
def P0(p, v):
    A = 12*p*p - 3
    return 4*p*v**4 - (4*A/3)*v**3 - 2*p*A*A*v*v + A**3*v
def Vplus(p):
    return (36*p*p - 12*p - 3 + 3*math.sqrt((2*p-1)**3*(18*p+7)))/4
def gl(n):
    xs, ws = [], []
    for i in range(1, n+1):
        x = math.cos(math.pi*(i-0.25)/(n+0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for m in range(2, n+1): p0, p1 = p1, ((2*m-1)*x*p1 - (m-1)*p0)/m
            dp = n*(x*p1-p0)/(x*x-1); dx = p1/dp; x -= dx
            if abs(dx) < 1e-15: break
        p0, p1 = 1.0, x
        for m in range(2, n+1): p0, p1 = p1, ((2*m-1)*x*p1 - (m-1)*p0)/m
        dp = n*(x*p1-p0)/(x*x-1)
        xs.append(x); ws.append(2/((1-x*x)*dp*dp))
    return xs, ws
xs, ws = gl(200)
# integrand has sqrt((2p-1)^3) endpoint behaviour: substitute p = 1/2 + s^2 to smooth
Dnum = 0.0
for x, w in zip(xs, ws):
    s = (x+1)/2*math.sqrt(0.5); p = 0.5 + s*s
    Dnum += w*(math.sqrt(0.5)/2)*2*s*(P0(p, Vplus(p)) - P0(p, 12*p*p*p - 3*p))
Dclosed = 27066286003/223205220 - (79298560/4782969)*math.log(2)
Jnum = 0.0
for x, w in zip(xs, ws):
    p = 0.5 + (x+1)/4; Jnum += w*0.25*(P0(p, (12*p*p-3)*p) - P0(p, (12*p*p-3)/2))
    p = 1 + (x+1)/4; Jnum += w*0.25*(P0(p, 3+6*p) - P0(p, (12*p*p-3)/2))
print('D numeric', Dnum, 'closed', Dclosed, 'diff', Dnum - Dclosed)
print('J numeric', Jnum, 'closed', 1083417/280, 'diff', Jnum - 1083417/280)
If = float(I); Jf = 1083417/280
print('I-D-J =', If - Dclosed - Jf, '>0', If - Dclosed - Jf > 0)
print('theta* =', (If - Dclosed)/If, ' J/(I-D) =', Jf/(If-Dclosed), ' D/(I-D) =', Dclosed/(If-Dclosed), ' I/(I-D) =', If/(If-Dclosed))
# E33: E eta_max = [I*5771/7062 - D1]/(I-D)
D1 = 1259212131941593/15982386572880 - (3186360320/387420489)*math.log(2)
D2 = 72706616008642669/1315122095139840 - (15686696960/3486784401)*math.log(2)
print('E eta_max =', (If*5771/7062 - D1)/(If - Dclosed), ' E eta_max^2 =', (If*2440/3531 - D2)/(If - Dclosed))
# E19 numeric: int_{S2_outer} Q |t_c|^11 = D ?
def Qf(u, v): b = 3-12*u*u; return (4*v*v-b*b)*(b-4*u*v)
def tcf(u, v):
    b = 3-12*u*u; L = 4*v*v-8*b*u*v+b*b; return (4*v*v-b*b)/L
xs2, ws2 = gl(120)
E19 = 0.0
for x, w in zip(xs2, ws2):
    p = 0.5 + (x+1)/4; A = 12*p*p-3; lo, hi = A/2, A*p
    for y, wy in zip(xs2, ws2):
        v = lo + (y+1)/2*(hi-lo); E19 += w*0.25*wy*(hi-lo)/2*Qf(-p, v)*abs(tcf(-p, v))**11
    p = 1 + (x+1)/4; A = 12*p*p-3; lo, hi = A/2, 3+6*p
    for y, wy in zip(xs2, ws2):
        v = lo + (y+1)/2*(hi-lo); E19 += w*0.25*wy*(hi-lo)/2*Qf(-p, v)*abs(tcf(-p, v))**11
print('E19: int_outer Q|tc|^11 =', E19, ' D =', Dclosed, ' diff', E19 - Dclosed)
# E32 bracket nonnegativity and mass: numerically integrate the inner-sector subtraction in e
