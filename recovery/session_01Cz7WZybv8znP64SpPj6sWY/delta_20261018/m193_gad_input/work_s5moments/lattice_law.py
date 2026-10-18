# High-precision check on the ACTUAL full positive Gaussian Fourier law of Section 3:
# w_m = exp(-|p_m|^2/2)/Z_L over the ENTIRE lattice p_m = 2 pi m / L (constant mode m=0 retained).
#  (i) product/sign-symmetric structure: d-dim lattice moments (Z_L as a d-dim sum, not z^d) vs 1D a,b,c;
# (ii) sign conventions: finite-difference derivatives of K_L(h)=sum_m w_m cos(p_m.h) vs (8),(9),E6;
#(iii) role of the m=0 mode; (iv) V(e) of (10) vs direct Schur complement and > 0.
import itertools, random
import mpmath as mp
mp.mp.dps = 40
random.seed(7)
tol = mp.mpf(10)**-25

def one_d(L, M):
    h = 2*mp.pi/L; al = h*h/2
    ks = range(-M, M+1)
    z = mp.fsum(mp.e**(-al*k*k) for k in ks)
    mom = lambda n: mp.fsum((h*k)**n*mp.e**(-al*k*k) for k in ks)/z
    return h, al, z, mom(2), mom(4), mom(6)

def cutoff(L, digits=60):
    h = 2*mp.pi/L; al = h*h/2
    return int(mp.ceil(mp.sqrt(digits*mp.log(10)/al))) + 1

def rand_unit(d):
    v = [mp.mpf(random.gauss(0, 1)) for _ in range(d)]; n = mp.sqrt(mp.fsum(x*x for x in v))
    return [x/n for x in v]

allok = True
for (L, dims) in [(1, (2, 3, 4)), (3, (2, 3, 4)), (2*mp.pi, (2, 3)), (10, (2,)), (25, (2,))]:
    M = cutoff(L, 45)
    h, al, z, a, b, c = one_d(L, M)
    k4 = b - 3*a*a; k6 = c - 15*a*b + 30*a**3
    for d in dims:
        pts = list(itertools.product(range(-M, M+1), repeat=d))
        P = [[h*k for k in m] for m in pts]
        W = [mp.e**(-mp.fsum(x*x for x in p)/2) for p in P]
        Z = mp.fsum(W); W = [w/Z for w in W]
        zok = abs(Z - z**d) < tol*Z
        E = lambda f: mp.fsum(w*f(p) for w, p in zip(W, P))
        # (i) tensor (8) and Cov(D)
        err8 = mp.mpf(0)
        dl = lambda p, q: 1 if p == q else 0
        for (i, j, k, l) in itertools.product(range(d), repeat=4):
            lhs = E(lambda p: p[i]*p[j]*p[k]*p[l])
            rhs = a*a*(dl(i,j)*dl(k,l) + dl(i,k)*dl(j,l) + dl(i,l)*dl(j,k)) + k4*sum(dl(i,s)*dl(j,s)*dl(k,s)*dl(l,s) for s in range(d))
            err8 = max(err8, abs(lhs - rhs))
        errD = max(abs(E(lambda p: p[i]*p[j]) - (a if i == j else 0)) for i in range(d) for j in range(d))
        e = rand_unit(d)
        dot = lambda p: mp.fsum(x*y for x, y in zip(p, e))
        v = [3*a*a*e[i] + k4*e[i]**3 for i in range(d)]
        err9 = max(abs(E(lambda p: p[i]*dot(p)**3) - v[i]) for i in range(d))
        p4 = mp.fsum(x**4 for x in e); p6 = mp.fsum(x**6 for x in e)
        E6 = E(lambda p: dot(p)**6)
        err6 = abs(E6 - (15*a**3 + 15*a*k4*p4 + k6*p6))
        Vreg = E6 - mp.fsum(x*x for x in v)/a   # Cov(D)=aI verified above
        V10 = 6*a**3 + 9*a*k4*p4 + (k6 - k4*k4/a)*p6
        err10 = abs(Vreg - V10)
        scale = max(abs(c), abs(a)**3, mp.mpf(1))
        ok = zok and err8 < tol*max(b,1) and errD < tol and err9 < tol*scale and err6 < tol*scale and err10 < tol*scale and V10 > 0
        allok &= ok
        print(f'L={mp.nstr(L,6)} d={d} M={M}: Z_L=z^d {zok}; max|(8) err|={mp.nstr(err8,3)} |CovD err|={mp.nstr(errD,3)} '
              f'|(9) err|={mp.nstr(err9,3)} |E6 err|={mp.nstr(err6,3)} |(10) err|={mp.nstr(err10,3)}; a={mp.nstr(a,8)} V(e)={mp.nstr(V10,8)} '
              f'c-b^2/a={mp.nstr(c-b*b/a,8)} OK={ok}')

# (ii) sign conventions from finite differences of the actual covariance K_L, d=2 and d=3
for (L, d) in [(3, 2), (2*mp.pi, 2), (3, 3)]:
    M = cutoff(L, 45)
    h, al, z, a, b, c = one_d(L, M)
    k4 = b - 3*a*a; k6 = c - 15*a*b + 30*a**3
    pts = list(itertools.product(range(-M, M+1), repeat=d))
    P = [[h*k for k in m] for m in pts]
    W = [mp.e**(-mp.fsum(x*x for x in p)/2) for p in P]; Z = mp.fsum(W); W = [w/Z for w in W]
    K = lambda x: mp.fsum(w*mp.cos(mp.fsum(pi*xi for pi, xi in zip(p, x))) for w, p in zip(W, P))
    e = rand_unit(d)
    # Cov(d_e^3 F(x), d_i F(y)) = d_t^3 d_s K(t e - s u) at 0
    res = []
    for i in range(d):
        u = [1 if j == i else 0 for j in range(d)]
        g = lambda t, s: K([t*ej - s*uj for ej, uj in zip(e, u)])
        cov = mp.diff(g, (0, 0), (3, 1))
        res.append((cov, -(3*a*a*e[i] + k4*e[i]**3)))
    gT = lambda t, s: K([(t - s)*ej for ej in e])
    varT = mp.diff(gT, (0, 0), (3, 3))
    p4 = mp.fsum(x**4 for x in e); p6 = mp.fsum(x**6 for x in e)
    # Cov(H_11, H_22) and Var(H_11): d_t^2 d_s^2 K(t u1 - s u2)
    gH = lambda t, s: K([t if j == 0 else 0 for j in range(d)][j] - (s if j == 1 else 0) for j in range(d)) if False else None
    u1 = [1 if j == 0 else 0 for j in range(d)]; u2 = [1 if j == 1 else 0 for j in range(d)]
    cH12 = mp.diff(lambda t, s: K([t*x - s*y for x, y in zip(u1, u2)]), (0, 0), (2, 2))
    cH11 = mp.diff(lambda t, s: K([(t - s)*x for x in u1]), (0, 0), (2, 2))
    cDH = mp.diff(lambda t, s: K([t*x - s*y for x, y in zip(u1, e)]), (0, 0), (1, 2))
    cTH = mp.diff(lambda t, s: K([t*x - s*y for x, y in zip(e, u1)]), (0, 0), (3, 2))
    cFH = mp.diff(lambda t, s: K([(t - s)*x for x in u1]), (0, 0), (0, 2))
    errs = [abs(x - y) for x, y in res] + [abs(varT - (15*a**3 + 15*a*k4*p4 + k6*p6)), abs(cH12 - a*a), abs(cH11 - b), abs(cDH), abs(cTH), abs(cFH + a)]
    ftol = mp.mpf(10)**-12
    ok = all(x < ftol*max(1, abs(c)) for x in errs)
    allok &= ok
    print(f'FD-sign L={mp.nstr(L,6)} d={d}: Cov(T_e,D_i) vs -v_i: ' + ', '.join(f'{mp.nstr(x,10)}/{mp.nstr(y,10)}' for x, y in res)
          + f'; Var T vs E6 err {mp.nstr(errs[d],3)}; Cov(H11,H22)-a^2 {mp.nstr(errs[d+1],3)}; Var H11-b {mp.nstr(errs[d+2],3)}; Cov(D_1,H_ee) {mp.nstr(cDH,3)}; Cov(T_e,H_11) {mp.nstr(cTH,3)}; Cov(F,H11)+a {mp.nstr(errs[-1],3)} OK={ok}')

# (iii) the m=0 constant mode: its atom at Q=0 contributes 0 to all positive-order moments; it enters only Z_L.
L = 3; d = 2; M = cutoff(L, 45)
h, al, z, a, b, c = one_d(L, M)
pts = list(itertools.product(range(-M, M+1), repeat=d))
P = [[h*k for k in m] for m in pts]
Wr = [mp.e**(-mp.fsum(x*x for x in p)/2) for p in P]; Z = mp.fsum(Wr)
w0 = 1/Z
num22 = mp.fsum(w*p[0]**2*p[1]**2 for w, p in zip(Wr, P))
num2 = mp.fsum(w*p[0]**2 for w, p in zip(Wr, P))
num22_nz = mp.fsum(w*p[0]**2*p[1]**2 for w, p, m in zip(Wr, P, pts) if any(m))
print(f'm=0 mode: w_0={mp.nstr(w0,10)}; unnormalized E[Q1^2Q2^2] with/without m=0 differ by {mp.nstr(abs(num22-num22_nz),3)} (exactly 0 term)')
# If the zero mode were REMOVED and the field renormalized to unit variance, moments scale by 1/(1-w0) and product law breaks:
a_r = (num2/Z)/(1 - w0); m22_r = (num22/Z)/(1 - w0)
print(f'  retained: E[Q1^2Q2^2]-a^2 = {mp.nstr(num22/Z - a*a,3)}; removed+renormalized: E[Q1^2Q2^2]-a_r^2 = {mp.nstr(m22_r - a_r*a_r,8)} (nonzero => (8) would fail)')
print('ALL LATTICE-LAW CHECKS PASS' if allok else 'SOME LATTICE-LAW CHECK FAILED')
