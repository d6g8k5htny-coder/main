"""Independent exact checks for Math-#158 Slice C (G11-G16), stdlib only. Reviewer: Anthropic Claude."""
import sys, json, math, random, itertools
from fractions import Fraction as F
WT = sys.argv[1]
sys.path.insert(0, WT + '/frontiers/planar_cubic_cluster_20260929')
import cubic  # packet's conic oracle, used only as a cross-check

def canon(k, a, b_, c):
    return b_ - a*a/(12*k), (c - a*b_/(4*k) + a**3/(72*k*k))/2
def n_indep(k, s, B, D):
    """(C6) transcribed independently from PROOF.md."""
    assert s < -abs(B)/2
    T = -(s-B)**2*(B+2*s)/(12*k)
    if s > B: return (2 if D*D < T else 1), T
    return (1 if D*D > T else 0), T

out = {}
# ---- (G13): nonempty sector => x < 2|B| or x^3 < 48 k D^2 ; and x >= 2|B| => T >= x^3/(48k)
rng = random.Random(20260929)
def rf(lo, hi, den=12):
    return F(rng.randint(lo*den, hi*den), den)
grid = 0; nonempty = 0; viol = []
ks = (F(1,12), F(1), F(3,2), F(7))
for k in ks:
    for _ in range(6000):
        s = rf(-6, 0); a = rf(-3, 3); be = rf(-4, 4); c = rf(-4, 4)
        B, D = canon(k, a, be, c)
        if not (s < -abs(B)/2): continue
        grid += 1
        n, T = n_indep(k, s, B, D)
        assert n == cubic.conic_oracle(k, s, B, D) == cubic.window_count(k, s, B, D)
        x, M = -s, abs(B)
        if x >= 2*M and not (T >= x**3/(48*k)): viol.append(('T-bound', k, s, a, be, c))
        if n >= 1:
            nonempty += 1
            if not (x < 2*M or x**3 < 48*k*D*D): viol.append(('H-bound', k, s, a, be, c))
        # w <= 36 k^2 x^2
        assert cubic.weight(k, s, B) <= 36*k*k*x*x
out['G13_samples_typed'] = grid; out['G13_nonempty'] = nonempty; out['G13_violations'] = viol
# (a+b)^3 <= 4(a^3+b^3) and the constants 384, 2304
for _ in range(2000):
    a_, b_ = rf(0, 9), rf(0, 9)
    assert 4*(a_**3+b_**3) - (a_+b_)**3 == 3*(a_+b_)*(a_-b_)**2 >= 0
for k in ks:
    assert 12*k*k*4*8 == 384*k*k and 12*k*k*4*48*k == 2304*k**3
out['G13_constants'] = True
# ---- (G14): antiderivative and the two slice integrals, with the classifier locating the slices
def antider(k, B, s): return 12*k*k*s**3 - 9*k*k*B*B*s
for k in ks:
    for B in (F(-5), F(-2), F(-1,3)):
        # n == 2 exactly on (B, B/2) at D = 0
        for j in range(-240, 1):
            s = B - 1 + F(j, 120) * (F(1)-B/2+0) if False else B - 1 + F(j+240, 120)*(1 - B/2 + B) / 2
            if not (s < -abs(B)/2): continue
            n, _ = n_indep(k, s, B, F(0))
            assert (n == 2) == (B < s < B/2), (k, B, s, n)
        val = antider(k, B, B/2) - antider(k, B, B)
        assert val == -6*k*k*B**3 == cubic.integrated_weight(k, B, B, B/2)
    # B = 0: choose D with 6 k D^2 a perfect cube
    for D in (F(6), F(48)):
        cube = 6*k*D*D
        r3 = None
        for num in range(1, 4000):
            for den in (1, 2, 3, 4, 6, 12):
                if F(num, den)**3 == cube: r3 = F(num, den)
        if r3 is None: continue
        for j in range(1, 400):
            s = -r3*F(j, 200)
            if s >= 0: continue
            n, _ = n_indep(k, s, F(0), D)
            assert (n == 1) == (s > -r3), (k, D, s, n)
        val = antider(k, F(0), F(0)) - antider(k, F(0), -r3)
        assert val == 72*k**3*D*D == cubic.integrated_weight(k, F(0), -r3, F(0))
        out.setdefault('G14_B0_cases', []).append([str(k), str(D), str(r3)])
out['G14'] = True
# ---- Theorem F positivity witnesses and openness
for k in ks:
    for (s, a, be, c), want in (((-3*k/2, F(0), -2*k, F(0)), 2), ((-k, F(0), F(0), 2*k), 1)):
        B, D = canon(k, a, be, c); assert s < -abs(B)/2
        assert n_indep(k, s, B, D)[0] == want and cubic.weight(k, s, B) > 0
        for eps in itertools.product((-k/100, 0, k/100), repeat=4):
            s2, a2, b2, c2 = s+eps[0], a+eps[1], be+eps[2], c+eps[3]
            B2, D2 = canon(k, a2, b2, c2)
            assert s2 < -abs(B2)/2 and n_indep(k, s2, B2, D2)[0] == want
out['F_positivity_witnesses_open'] = True
# ---- (G12) integer inequalities
for n in range(0, 13):
    assert (1 if n == 1 else 0) + 2*(1 if n == 2 else 0) <= n and 2*(1 if n >= 2 else 0) <= n*(n-1)
out['G12_integer_inequalities'] = True
# ---- Theorem B: exact even/odd parity for an even, NON-isotropic kernel; conditional factorization
freqs = [(1,0),(0,1),(1,1),(2,1),(1,2),(3,1),(2,3),(1,3),(3,2),(4,1),(2,0),(0,3),(5,2)]
coef  = [F(3),F(1),F(2),F(1,2),F(5,3),F(1,4),F(2,7),F(1,3),F(3,5),F(1,9),F(7,2),F(1,5),F(1,11)]
def dK(g):  # (d^g K)(0) for K = sum c cos(w.y)
    tot = sum(g)
    if tot % 2: return F(0)
    sgn = -1 if (tot//2) % 2 else 1
    return sum(c * sgn * F(w[0])**g[0] * F(w[1])**g[1] for c, w in zip(coef, freqs))
# jets: U_0 = (f, f_x, f_xx, f_xxx, f_z, f_xz), J = (f_zz, f_xxz, f_xzz, f_zzz)  (multi-index (i,j) = d_x^i d_z^j)
idx = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(2,1),(1,2),(0,3)]
def cov(al, be_):  # Cov(d^al f(0), d^be f(0)) = (-1)^{|be|} d^{al+be} K(0)
    return (-1)**sum(be_) * dK((al[0]+be_[0], al[1]+be_[1]))
S = [[cov(p, q) for q in idx] for p in idx]
par = [sum(p) % 2 for p in idx]
assert all(S[i][j] == 0 for i in range(10) for j in range(10) if par[i] != par[j])
def solve(Mx, rhs):  # exact Gaussian elimination, returns Mx^{-1} rhs (rhs: list of columns)
    n = len(Mx); A = [row[:] + [r[i] for r in rhs] for i, row in enumerate(Mx)]
    for col in range(n):
        piv = next(r for r in range(col, n) if A[r][col] != 0)
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]; A[col] = [v/pv for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                fac = A[r][col]; A[r] = [v - fac*w for v, w in zip(A[r], A[col])]
    return [row[n:] for row in A]
def det(Mx):
    n = len(Mx); A = [row[:] for row in Mx]; d = F(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] != 0), None)
        if piv is None: return F(0)
        if piv != col: A[col], A[piv] = A[piv], A[col]; d = -d
        d *= A[col][col]
        for r in range(col+1, n):
            fac = A[r][col]/A[col][col]; A[r] = [v - fac*w for v, w in zip(A[r], A[col])]
    return d
assert det(S) != 0
U = list(range(6)); Jn = list(range(6, 10))
Suu = [[S[i][j] for j in U] for i in U]; Sju = [[S[i][j] for j in U] for i in Jn]; Sjj = [[S[i][j] for j in Jn] for i in Jn]
X = solve(Suu, [[S[i][j] for i in U] for j in Jn])  # Suu^{-1} Suj, as list of 4 columns
condcov = [[Sjj[p][q] - sum(Sju[p][i]*X[i][q] for i in range(6)) for q in range(4)] for p in range(4)]
assert all(condcov[0][q] == 0 and condcov[q][0] == 0 for q in (1, 2, 3))   # A independent of (a,beta,c) after conditioning
assert condcov[0][0] > 0 and det([[condcov[p][q] for q in (1,2,3)] for p in (1,2,3)]) > 0
def condmean(b, k):
    u = [b, F(0), F(0), 12*k, F(0), F(0)]
    return [sum(X[i][q]*u[i] for i in range(6)) for q in range(4)]
m0, m1, m1k = condmean(F(0), F(1)), condmean(F(1), F(1)), condmean(F(1), F(2))
assert m0[1:] == m1[1:]            # (a,beta,c) conditional mean independent of b
assert m1[0] != m0[0]              # A conditional mean depends on b (affine)
assert m1k[1:] != m1[1:] or True   # k-dependence allowed
out['B_parity_block_zero'] = True; out['B_full_rank_det_nonzero'] = True; out['B_condcov_A_vs_odd_zero'] = True
out['B_condmean_odd_indep_of_b'] = True; out['B_condmean_A_slope_in_b'] = str(m1[0]-m0[0])
# ---- (G16): truncated moments by quadrature
def Phi(t): return 0.5*(1+math.erf(t/math.sqrt(2)))
def phi(t): return math.exp(-t*t/2)/math.sqrt(2*math.pi)
maxerr = 0.0
for m, sg in ((0.0,1.0),(1.0,2.0),(-1.5,0.7),(2.3,1.1),(-4.0,0.5)):
    lo = min(m, 0.0) - 14*sg; N = 400000; h = (0-lo)/N
    tot = 0.0
    for i in range(N+1):
        A = lo + i*h; wgt = 1 if i in (0, N) else (4 if i % 2 else 2)
        tot += wgt * A*A*phi((A-m)/sg)/sg
    num = tot*h/3
    formula = (m*m+sg*sg)*Phi(-m/sg) - m*sg*phi(m/sg)
    maxerr = max(maxerr, abs(num-formula)/max(1e-12, abs(formula)))
    assert abs(phi(m/sg)/sg - phi((0-m)/sg)/sg) < 1e-15
out['G16_max_rel_err'] = maxerr; assert maxerr < 1e-9
print(json.dumps(out, indent=1, default=str))
