"""Slice A checks for Math-#181 at head 9dc87121 (PROOF blob 91f88e39). Stdlib only."""
import sys, math, random, importlib.util, time
from fractions import Fraction as Fr
sys.path.insert(0, '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/rev181')
spec = importlib.util.spec_from_file_location('elder', '/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/rev181/elder.py')
elder = importlib.util.module_from_spec(spec); spec.loader.exec_module(elder)
import tail_constants as tc

def b0(u): return 3 - 12*u*u
def Q(u, v):
    b = b0(u); return (4*v*v - b*b)*(b - 4*u*v)
def eta(u, v): return u + Fr(1,2) + v/6
def in_S(u, v): return Fr(-3,2) < u < Fr(1,2) and abs(b0(u))/2 < v < 3 - 6*u
def in_O2(u, v):
    p = -u; A = 12*p*p - 3
    return (Fr(1,2) < p < 1 and A/2 < v < A*p) or (1 <= p < Fr(3,2) and A/2 < v < 3 + 6*p)
def companion(u, v):
    b = b0(u); L = 4*v*v - 8*b*u*v + b*b
    if L == 0: return None
    t = (4*v*v - b*b)/L
    return (2*b*v - u*(4*v*v + b*b))/L, v*t*t, t

# ---- 1. exact shape masses: I, J = int_O2 Q, E0 = I - J, ratio E0/I
def vint(u, lo, hi):   # int_lo^hi Q(u,v) dv exactly, polynomial primitive
    b = b0(u)
    prim = lambda v: (4*b/3)*v**3 - 4*u*v**4 - b**3*v + 2*u*b*b*v*v
    return prim(hi) - prim(lo)
def poly_u_integral(f, lo, hi, deg=12):
    # exact integration of a polynomial in u given as callable: interpolate via deg+1 rational nodes (f polynomial of degree <= deg)
    nodes = [lo + (hi-lo)*Fr(i, deg) for i in range(deg+1)]
    vals = [f(x) for x in nodes]
    # Lagrange integrate: sum vals_i * int L_i
    total = Fr(0)
    for i, xi in enumerate(nodes):
        # coefficients of L_i
        coeffs = [Fr(1)]
        denom = Fr(1)
        for j, xj in enumerate(nodes):
            if j == i: continue
            denom *= (xi - xj)
            new = [Fr(0)]*(len(coeffs)+1)
            for k, c in enumerate(coeffs):
                new[k+1] += c; new[k] -= c*xj
            coeffs = new
        integ = sum(c*(hi**(k+1) - lo**(k+1))/(k+1) for k, c in enumerate(coeffs))
        total += vals[i]*integ/denom
    return total
I = poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, 3-6*u), Fr(-3,2), Fr(-1,2)) + poly_u_integral(lambda u: vint(u, (3-12*u*u)/2, 3-6*u), Fr(-1,2), Fr(1,2))
J = poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, (12*u*u-3)*(-u)), Fr(-1), Fr(-1,2)) + poly_u_integral(lambda u: vint(u, (12*u*u-3)/2, 3-6*u), Fr(-3,2), Fr(-1))
E0 = I - J
print('I  =', I, I == Fr(246528,35))
print('J  =', J, J == Fr(1083417,280))
print('E0 =', E0, E0 == Fr(888807,280))
print('E0/I =', E0/I, E0/I == Fr(296269,657408), float(E0/I))
# elder.py agreement
print('elder.py point_moment(0)=I, outer_moment(0)=J, moment(0)=E0:', elder.point_moment(0)==I, elder.outer_moment(0)==J, elder.moment(0)==E0)

# ---- 2. D = int_{S2_inner} Q numerically (v exact, p by GL) vs closed form
Dcf = 27066286003/223205220 - (79298560/4782969)*math.log(2)
def Vplus(p): return (36*p*p - 12*p - 3 + 3*math.sqrt((2*p-1)**3*(18*p+7)))/4
def vintf(u, lo, hi):
    b = 3-12*u*u
    prim = lambda v: (4*b/3)*v**3 - 4*u*v**4 - b**3*v + 2*u*b*b*v*v
    return prim(hi) - prim(lo)
rule = tc.gauss_legendre(400)
# integrand has sqrt((2p-1)^3) behaviour at p=1/2: substitute p = 1/2 + w^2 to smooth
Dq = tc.gl(lambda w: 2*w*vintf(-(0.5+w*w), (12*(0.5+w*w)**2-3)*(0.5+w*w), Vplus(0.5+w*w)), 0.0, math.sqrt(0.5), rule)
print('D closed form', Dcf, ' quadrature', Dq, ' rel diff', abs(Dq-Dcf)/Dcf)
print('ratios: E0/I', float(E0)/float(I), ' E0/(I-D)', float(E0)/(float(I)-Dcf), ' D/E0', Dcf/float(E0))
print('I-D-J (singleton mass) =', float(I)-Dcf-float(J), ' D/E0 vs (I-D-J)/E0 sum', (Dcf + float(I)-Dcf-float(J))/float(E0))

# ---- 3. E17 exact and the sector/selector logic on random rational points
random.seed(181)
def rnd(lo, hi, den=997):
    return lo + (hi-lo)*Fr(random.randint(1, den-1), den)
stats = dict(pts=0, doublet=0, outer=0, inner=0, single=0, e17=0, bad=[])
for _ in range(4000):
    u = rnd(Fr(-3,2), Fr(1,2)); lo = abs(b0(u))/2; hi = 3-6*u
    if lo >= hi: continue
    v = rnd(lo, hi)
    if not in_S(u, v): continue
    stats['pts'] += 1
    c = companion(u, v)
    n2 = c is not None and in_S(c[0], c[1])
    sel = elder.selection(u, v)
    if not n2:
        stats['single'] += 1
        if in_O2(u, v): stats['bad'].append(('singleton in O2', u, v))
        if sel != 1: stats['bad'].append(('singleton weight', u, v, sel))
        continue
    stats['doublet'] += 1
    uc, vc, t = c
    # E17
    p = -u; A = 12*p*p - 3
    if A <= 0: stats['bad'].append(('doublet with b0>=0', u, v)); continue
    ell = v/A
    e17 = 2*(ell-p)*(4*p*ell-1)**3/(4*ell*ell - 8*p*ell + 1)**2
    if eta(uc, vc) - eta(u, v) == e17: stats['e17'] += 1
    else: stats['bad'].append(('E17', u, v))
    if t >= 0: stats['bad'].append(('t_c not negative', u, v, t))
    # involution
    cc = companion(uc, vc)
    if cc is None or cc[0] != u or cc[1] != v or cc[2]*t != 1: stats['bad'].append(('involution', u, v))
    if in_O2(u, v):
        stats['outer'] += 1
        ok = (abs(t) < 1) and (eta(uc, vc) < eta(u, v)) and (not in_O2(uc, vc)) and in_S(uc, vc) and sel == 0
        if not ok: stats['bad'].append(('outer logic', u, v, abs(t) < 1, eta(uc, vc) < eta(u, v), in_O2(uc, vc), sel))
    else:
        stats['inner'] += 1
        ok = (abs(t) > 1) and (eta(uc, vc) > eta(u, v)) and in_O2(uc, vc) and sel == 1
        if not ok: stats['bad'].append(('inner logic', u, v, abs(t) > 1, eta(uc, vc) > eta(u, v), in_O2(uc, vc), sel))
    if sel + elder.selection(uc, vc) != 1: stats['bad'].append(('weights do not sum to 1', u, v))
print('sector/selector stats:', {k: v for k, v in stats.items() if k != 'bad'}, 'violations:', len(stats['bad']), stats['bad'][:5])

# ---- 4. no-atom quadratic: leading coefficient 1 + a^2/(144k^2) > 0 (trivial); report
# ---- 5. mass check of Lemma 1 (7) and the doublet split against Math-#168 alpha values (planar, b = 0)
def mass(k, region, nu=32, nv=32, ns=160, na=32):
    """108 k^7 p_b(0)/z0 int_region Q int |Z|^-12 int p_odd da dZ du dv  (R11 without radius indicator, shape restricted)."""
    ru, rv, rs, ra = tc.gauss_legendre(nu), tc.gauss_legendre(nv), tc.gauss_legendre(ns), tc.gauss_legendre(na)
    pieces = []   # (ulo, uhi, vlo(u), vhi(u))
    Ap = lambda u: (12*u*u-3)*(-u)
    if region == 'S':
        pieces = [(-1.5, -0.5, lambda u: (12*u*u-3)/2, lambda u: 3-6*u), (-0.5, 0.5, lambda u: (3-12*u*u)/2, lambda u: 3-6*u)]
    elif region == 'O2':
        pieces = [(-1.5, -1.0, lambda u: (12*u*u-3)/2, lambda u: 3-6*u), (-1.0, -0.5, lambda u: (12*u*u-3)/2, Ap)]
    elif region == 'E':
        pieces = [(-1.0, -0.5, Ap, lambda u: 3-6*u), (-0.5, 0.5, lambda u: (3-12*u*u)/2, lambda u: 3-6*u)]
    elif region == 'S2_inner':
        pieces = [(-1.0, -0.5, Ap, lambda u: Vplus(-u))]
    tot = 0.0
    for (ulo, uhi, vlo, vhi) in pieces:
        def fu(u):
            b = 3-12*u*u
            def fv(v):
                q = (4*v*v-b*b)*(b-4*u*v)
                c3 = 2*k*(v-b*u)
                cands = []
                if c3 != 0: cands.append(math.log(abs(c3)/6.0)/3.0)
                if b != 0: cands.append(0.5*math.log(k*abs(b)/4.0))
                s_star = max(cands) if cands else 0.0
                def fs(s):
                    Zi2 = math.exp(-2*s); Zi3 = math.exp(-3*s); acc = 0.0
                    for sign in (1.0, -1.0):
                        acc += tc.gl(lambda a: tc.p_odd(a, a*a/(12*k) + k*b*Zi2, a**3/(144*k*k) + (a*b/4)*Zi2 + sign*c3*Zi3), -14.0, 14.0, ra)
                    return math.exp(-11*s)*acc
                return q*tc.gl(fs, s_star-3.0, s_star+4.5, rs)
            lo, hi = vlo(u), vhi(u)
            return tc.gl(fv, lo, hi, rv) if hi > lo else 0.0
        tot += tc.gl(fu, ulo, uhi, ru)
    return 108*k**7*tc.prefactor(k, 0.0)*tot

for k in (0.5, 1.0, 2.0):
    a1, a2 = tc.ALPHA_GH60[k]; m1, m2 = tc.ALPHA_MC[k]
    t0 = time.time()
    mS = mass(k, 'S'); mE = mass(k, 'E'); mO = mass(k, 'O2'); mI = mass(k, 'S2_inner')
    print(f'k={k}: S={mS:.5f} (a1+2a2 GH60 {a1+2*a2:.5f}, rel {abs(mS-(a1+2*a2))/(a1+2*a2):.1e}) | E={mE:.5f} (a1+a2 {a1+a2:.5f}, rel {abs(mE-(a1+a2))/(a1+a2):.1e}) | O2={mO:.5f} inner={mI:.5f} (a2 GH60 {a2:.5f}, MC {m2:.5f}; rel O2 {abs(mO-a2)/a2:.1e}, inner {abs(mI-a2)/a2:.1e}) | E-inner={mE-mI:.5f} (a1 {a1:.5f}, rel {abs(mE-mI-a1)/a1:.1e}) | S-E-O2={mS-mE-mO:.1e}  [{time.time()-t0:.0f}s]')
