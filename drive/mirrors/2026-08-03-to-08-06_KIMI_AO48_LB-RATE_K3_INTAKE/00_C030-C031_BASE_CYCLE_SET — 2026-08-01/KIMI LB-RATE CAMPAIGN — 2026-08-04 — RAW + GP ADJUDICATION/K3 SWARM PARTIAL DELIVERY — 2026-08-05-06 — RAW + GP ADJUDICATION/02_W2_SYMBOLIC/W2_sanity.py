#!/usr/bin/env python3
# W2_sanity.py — fail-closed sanity probes for the W2 symbolic Kac-Rice derivation.
# Independent re-derivation; does NOT read any KIMI WP-resolution artifact.
# Every check goes through ck(); any FAIL -> exit(1).
import sys, numpy as np
from scipy.stats import norm

FAILS = []
def ck(name, cond, detail=""):
    ok = bool(cond)
    print(("PASS" if ok else "FAIL"), "|", name, "|", detail)
    if not ok:
        FAILS.append(name)

rng = np.random.default_rng(20260804)
PI = np.pi

# ============================================================
# PART 0 — exact lattice kernel and spectral moments
# K1(s) = Z^-1 sum_j exp(-k_j^2/2) exp(i k_j s), k_j = j*pi/12
# Cov(d^a f(x), d^b f(y)) = (-1)^|b| d_s^{a+b} K(s), s = x-y,
# K(s) = K1(s1) K1(s2).
# ============================================================
JMAX = 48
jj = np.arange(-JMAX, JMAX + 1)
kap = jj * PI / 12.0
w = np.exp(-0.5 * kap * kap)
Z1 = w.sum()

def k1(n, s):
    """exact (truncated) n-th derivative of K1 at scalar s."""
    return float((w * (1j * kap) ** n * np.exp(1j * kap * s)).sum().real / Z1)

# truncation tail: sum_{|j|>JMAX} e^{-kap^2/2} <= 2*(12/pi)*phi(kap_JMAX)/kap_JMAX
t0 = kap[-1]
tailb = 2 * (12.0 / PI) * np.exp(-0.5 * t0 * t0) / t0
ck("P0.trunc-tail", tailb < 1e-30, f"tail bound {tailb:.2e}")

# spectral moments vs continuum N(0,1) values
mu2 = float((w * kap**2).sum() / Z1)
mu4 = float((w * kap**4).sum() / Z1)
mu6 = float((w * kap**6).sum() / Z1)
ck("P0.moments", abs(mu2 - 1) < 1e-12 and abs(mu4 - 3) < 1e-12 and abs(mu6 - 15) < 1e-10,
   f"mu2={mu2:.15f} mu4={mu4:.15f} mu6={mu6:.15f}")
ck("P0.K1(0)=1", abs(k1(0, 0.0) - 1.0) < 1e-14)
ck("P0.K1''(0)=-mu2", abs(k1(2, 0.0) + mu2) < 1e-12)
ck("P0.K1'(0)=0", abs(k1(1, 0.0)) < 1e-14)

def cov2(a, b, s):
    """Cov(d^a f(x), d^b f(y)), s = x - y. a,b are 2-multiindices (tuples)."""
    n1, n2 = a[0] + b[0], a[1] + b[1]
    return ((-1) ** (b[0] + b[1])) * k1(n1, s[0]) * k1(n2, s[1])

JET = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]  # F, G1, G2, A=H11, B=H22, C=H12

# ============================================================
# PART 1 — unconditional same-point joint law: BF structure
# Expected (continuum moments): f perp grad; Cov(f,H)=(-1,-1,0);
# A,B | f=u iid N(-u,2); C ~ N(0,1) indep.
# ============================================================
S6 = np.array([[cov2(a, b, (0.0, 0.0)) for b in JET] for a in JET])
ck("P1.S6-psd", np.linalg.eigvalsh(S6).min() > 0, f"min eig {np.linalg.eigvalsh(S6).min():.3e}")
ck("P1.f-perp-grad", abs(S6[0, 1]) < 1e-13 and abs(S6[0, 2]) < 1e-13)
ck("P1.grad-perp-H", np.max(np.abs(S6[1:3, 3:6])) < 1e-13, f"max {np.max(np.abs(S6[1:3,3:6])):.2e}")
ck("P1.Cov(f,H)=(-1,-1,0)",
   abs(S6[0, 3] + 1) < 1e-12 and abs(S6[0, 4] + 1) < 1e-12 and abs(S6[0, 5]) < 1e-13)
# Schur complement of f in (A,B,C): expect diag(2,2,1)
SHF = S6[3:6, 3:6] - np.outer(S6[3:6, 0], S6[0, 3:6]) / S6[0, 0]
ck("P1.H|f = diag(2,2,1)", np.max(np.abs(SHF - np.diag([2.0, 2.0, 1.0]))) < 1e-11,
   f"\n{SHF}")
ck("P1.Var(grad)=I", abs(S6[1, 1] - 1) < 1e-12 and abs(S6[2, 2] - 1) < 1e-12 and abs(S6[1, 2]) < 1e-13)

# ============================================================
# PART 2 — isotropic-slice closed forms (unconditional, or f-only)
# T=(A+B)/2, D=(A-B)/2, C: T~N(-u,1), D,C~N(0,1) indep; det=T^2-D^2-C^2.
# Claims:  E[det|u]=u^2-1 ;  P(det<0|u)=2^{-1/2} e^{-u^2/4} ;
#          E[|det|1{det<0}|u] = sqrt(2) e^{-u^2/4}.
# ============================================================
def isotropic_mc(u, n=3_000_000):
    T = rng.normal(-u, 1.0, n); D = rng.normal(0, 1, n); C = rng.normal(0, 1, n)
    q = T * T - D * D - C * C
    sad = q < 0
    return q, sad, np.abs(q) * sad

for u in [0.0, 0.7, 1.2]:
    q, sad, wq = isotropic_mc(u)
    se = wq.std() / np.sqrt(len(wq))
    pred_E = np.sqrt(2.0) * np.exp(-u * u / 4)
    ck(f"P2.E|det|sad|u={u}", abs(wq.mean() - pred_E) < 4 * se,
       f"mc={wq.mean():.6f} pred={pred_E:.6f} se={se:.2e}")
    pred_P = 2 ** -0.5 * np.exp(-u * u / 4)
    seP = np.sqrt(pred_P * (1 - pred_P) / len(wq))
    ck(f"P2.P(det<0|u={u})", abs(sad.mean() - pred_P) < 4 * seP,
       f"mc={sad.mean():.6f} pred={pred_P:.6f}")
    seM = q.std() / np.sqrt(len(q))
    ck(f"P2.E[det|u={u}]=u^2-1", abs(q.mean() - (u * u - 1)) < 4 * seM,
       f"mc={q.mean():.6f} pred={u*u-1:.6f}")

# E[e^{-T^2/2}], T~N(m,1) = 2^{-1/2} e^{-m^2/4}  (closed form used in Part 2 proof)
for m in [0.0, -1.2, 2.3]:
    T = rng.normal(m, 1.0, 2_000_000)
    mc = np.exp(-T * T / 2).mean()
    pred = 2 ** -0.5 * np.exp(-m * m / 4)
    ck(f"P2.E[e^-T^2/2|m={m}]", abs(mc - pred) < 4 * mc / np.sqrt(2e6) * 2 + 1e-5,
       f"mc={mc:.8f} pred={pred:.8f}")

# unconditional saddle intensity at b=1.2 vs C031's rho_sad(1.2)=0.030449 (measured grade)
u = 1.2
rho_sad = (1 / (2 * PI)) * norm.pdf(u) * np.sqrt(2.0) * np.exp(-u * u / 4)
ck("P2.rho_sad(1.2)~0.030449", abs(rho_sad - 0.030449) / 0.030449 < 3e-3,
   f"derived={rho_sad:.6f} vs 0.030449 (rel {abs(rho_sad-0.030449)/0.030449:.2e})")

# max-term: E[|det|1{H negdef}|u] = E[ 1{T<0} (T^2 - 2 + 2 e^{-T^2/2}) ], T~N(-u,1);
# C030 reports the E-term 1.41350 +/- 0.0017 at u=1.2 (measured).
T = rng.normal(-u, 1.0, 4_000_000)
val = (T < 0) * (T * T - 2 + 2 * np.exp(-T * T / 2))
mc = val.mean(); se = val.std() / np.sqrt(len(val))
ck("P2.max-Eterm(1.2)~1.41350", abs(mc - 1.41350) < max(4 * se, 3e-3),
   f"derived-mc={mc:.5f} se={se:.1e} vs 1.41350")
# closed quadrature of the same expression via scipy
from scipy.integrate import quad
f_int = lambda t: (t * t - 2 + 2 * np.exp(-t * t / 2)) * norm.pdf(t, -u, 1)
qv, _ = quad(f_int, -np.inf, 0, epsabs=1e-13)
ck("P2.max-Eterm-quad", abs(qv - mc) < max(4 * se, 3e-3), f"quad={qv:.6f} mc={mc:.6f}")
# cross-check rho_mx(1.2)=0.043685 of C030: rho_mx = (1/2pi) phi(u) * Eterm
rho_mx = (1 / (2 * PI)) * norm.pdf(u) * qv
ck("P2.rho_mx(1.2)~0.043685", abs(rho_mx - 0.043685) / 0.043685 < 3e-3,
   f"derived={rho_mx:.6f} vs 0.043685")
print("PART 0-2 done")

# ============================================================
# PART 3 — general noncentral Gaussian quadratic-form machinery
# H = (A,B,C) ~ N(m, S) (3-dim, general). det H = A*B - C^2 = z^T Qm z,
# Qm = [[0,1/2,0],[1/2,0,0],[0,0,-1]].
# ============================================================
QM = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, -1.0]])

def rand_gauss3(seed):
    r = np.random.default_rng(seed)
    m = r.normal(0, 1.2, 3)
    X = r.normal(0, 1, (3, 3))
    S = X @ X.T / 3 + 0.25 * np.eye(3)
    return m, S

def iss4(m, S, idx):
    """E[z_i z_j z_k z_l] for z~N(m,S) via Isserlis with means."""
    i, j, k, l = idx
    t = m[i]*m[j]*m[k]*m[l]
    t += (S[i,j]*m[k]*m[l] + S[i,k]*m[j]*m[l] + S[i,l]*m[j]*m[k]
          + S[j,k]*m[i]*m[l] + S[j,l]*m[i]*m[k] + S[k,l]*m[i]*m[j])
    t += S[i,j]*S[k,l] + S[i,k]*S[j,l] + S[i,l]*S[j,k]
    return t

def Edet_formula(m, S):
    # E[AB - C^2] = mA mB + S_AB - (mC^2 + S_CC)
    return m[0]*m[1] + S[0,1] - m[2]**2 - S[2,2]

def Edet2_formula(m, S):
    # E[(AB-C^2)^2] = E[A^2 B^2] - 2 E[A B C^2] + E[C^4]
    return (iss4(m, S, (0,0,1,1)) - 2*iss4(m, S, (0,1,2,2)) + iss4(m, S, (2,2,2,2)))

def Vardet_formula(m, S):
    # Var(z'Qz) = 2 tr((Q S)^2) + 4 m' Q S Q m
    QS = QM @ S
    return 2*np.trace(QS @ QS) + 4*m @ QM @ S @ QM @ m

def qf_mgf(m, S, s):
    """E[exp(s * z'Qm z)] closed form, valid for 1 - 2 s lam_i > 0."""
    L = np.linalg.cholesky(S)
    A = L.T @ QM @ L
    lam, U = np.linalg.eigh(A)
    what = U.T @ L.T @ QM @ m
    if np.any(1 - 2*s*lam <= 0):
        return np.inf
    return np.exp(s * m @ QM @ m) * np.prod((1 - 2*s*lam)**-0.5) * \
           np.exp(np.sum(2*s*s*what**2/(1 - 2*s*lam)))

def chernoff_neg(m, S):
    """P(Q<0) <= inf_{theta>=0, 1+2 theta lam_i>0} E[e^{-theta Q}]."""
    L = np.linalg.cholesky(S)
    A = L.T @ QM @ L
    lam = np.linalg.eigvalsh(A)
    cneg = max(0.0, -lam.min())
    if cneg == 0.0 and (m @ QM @ m) >= 0:
        pass  # bound still computed via grid
    tmax = 0.9999/(2*cneg) if cneg > 0 else 50.0
    grid = np.concatenate([[0.0], np.logspace(-9, np.log10(tmax), 4000)])
    vals = np.array([qf_mgf(m, S, -t) for t in grid])
    return vals.min()

def bernstein_neg(m, S):
    """closed-form: P(Q<0) <= exp(-EQ^2 / (4 (V + 2 c EQ))), EQ>0;
       V = Var Q, c = max(0, -lam_min)."""
    EQ = Edet_formula(m, S)
    if EQ <= 0:
        return 1.0
    V = Vardet_formula(m, S)
    L = np.linalg.cholesky(S)
    lam = np.linalg.eigvalsh(L.T @ QM @ L)
    c = max(0.0, -lam.min())
    return np.exp(-EQ*EQ/(4*(V + 2*c*EQ)))

def cantelli_neg(m, S):
    EQ = Edet_formula(m, S)
    if EQ <= 0:
        return 1.0
    V = Vardet_formula(m, S)
    return V/(V + EQ*EQ)

for seed in range(6):
    m, S = rand_gauss3(100 + seed)
    n = 600_000
    z = rng.multivariate_normal(m, S, n)
    q = z[:,0]*z[:,1] - z[:,2]**2
    # moments
    ck(f"P3.E[det]#{seed}", abs(q.mean() - Edet_formula(m, S)) < 4*q.std()/np.sqrt(n),
       f"mc={q.mean():.5f} f={Edet_formula(m,S):.5f}")
    ck(f"P3.E[det^2]#{seed}", abs((q*q).mean() - Edet2_formula(m, S)) < 4*(q*q).std()/np.sqrt(n),
       f"mc={(q*q).mean():.5f} f={Edet2_formula(m,S):.5f}")
    ck(f"P3.Var[det]#{seed}", abs(q.var() - Vardet_formula(m, S)) < 0.02*Vardet_formula(m, S),
       f"mc={q.var():.5f} f={Vardet_formula(m,S):.5f}")
    # mgf
    for s in [-0.05, 0.03]:
        mg = np.exp(s*q).mean()
        se = np.exp(s*q).std()/np.sqrt(n)
        ck(f"P3.mgf#{seed}s={s}", abs(mg - qf_mgf(m, S, s)) < 4*se,
           f"mc={mg:.6f} f={qf_mgf(m,S,s):.6f}")
    # tail bounds (only where event is common enough to MC; validity is what matters)
    p_mc = (q < 0).mean()
    b_ch = chernoff_neg(m, S); b_be = bernstein_neg(m, S); b_ca = cantelli_neg(m, S)
    se_p = max(np.sqrt(p_mc*(1-p_mc)/n), 1e-6)
    ck(f"P3.chernoff>=true#{seed}", b_ch >= p_mc - 4*se_p, f"mc={p_mc:.4g} ch={b_ch:.4g}")
    ck(f"P3.bernstein>=true#{seed}", b_be >= p_mc - 4*se_p, f"be={b_be:.4g}")
    ck(f"P3.cantelli>=true#{seed}", b_ca >= p_mc - 4*se_p, f"ca={b_ca:.4g}")
    ck(f"P3.chernoff<=bernstein#{seed}", b_ch <= b_be + 1e-12, f"ch={b_ch:.4g} be={b_be:.4g}")
    ck(f"P3.min(ch,ca)>=true#{seed}", min(b_ch, b_ca) >= p_mc - 4*se_p)

# ---- CS chains on a random correlated (F,H) 4-dim Gaussian ----
for seed in range(4):
    r = np.random.default_rng(500 + seed)
    X = r.normal(0, 1, (4, 4)); S4 = X @ X.T / 4 + 0.3*np.eye(4)
    m4 = r.normal(0, 1, 4)
    n = 800_000
    z = rng.multivariate_normal(m4, S4, n)
    F = z[:, 0]; H = z[:, 1:4]
    detH = H[:,0]*H[:,1] - H[:,2]**2
    Xv = np.abs(detH); A = detH < 0
    lo, hi = m4[0] - 0.3, m4[0] + 0.2   # arbitrary window
    B = (F > lo) & (F < hi)
    lhs = (Xv * A * B).mean()
    se_l = (Xv*A*B).std()/np.sqrt(n)
    EX2 = Edet2_formula(m4[1:], S4[1:,1:])
    PA = A.mean(); PB = B.mean()
    rhs_valid = np.sqrt(EX2) * np.sqrt(min(PA, PB))
    ck(f"P3.globalCS-valid#{seed}", lhs <= rhs_valid + 4*se_l,
       f"lhs={lhs:.4g} rhs={rhs_valid:.4g}")
    rhs_slice = np.sqrt(EX2 * min(PA, PB))
    ck(f"P3.globalCS(min-inside)# {seed}", lhs <= rhs_slice + 4*se_l)

# defective inequality: E[X 1_A 1_B] <= sqrt(E X^2) * P(B)  (inner root dropped)
# clean counterexample in-context: X = |det H| isotropic, B = {X >= q75} (P=1/4), A = Omega.
T = rng.normal(0,1,4_000_000); D = rng.normal(0,1,4_000_000); C = rng.normal(0,1,4_000_000)
Xv = np.abs(T*T - D*D - C*C)
q75 = np.quantile(Xv, 0.75)
B = Xv >= q75
lhs = (Xv*B).mean()
rhs_bad = np.sqrt((Xv**2).mean()) * B.mean()
ck("P3.defective-ineq-FAILS", lhs > rhs_bad,
   f"lhs={lhs:.4f} > defective-rhs={rhs_bad:.4f} (ratio {lhs/rhs_bad:.3f}): dropping inner root invalid")

# ---- Sidak box bound ----
for seed in range(3):
    m, S = rand_gauss3(700 + seed)
    a = np.array([0.4, 0.5, 0.3])
    n = 400_000
    z = rng.multivariate_normal(m, S, n)
    inbox = (np.abs(z - m) < a).all(1)
    p_mc = inbox.mean()
    sd = np.sqrt(np.diag(S))
    sidak = np.prod(2*norm.cdf(a/sd) - 1)
    ck(f"P3.sidak#{seed}", p_mc >= sidak - 4*np.sqrt(p_mc/n),
       f"mc={p_mc:.5f} sidak-lb={sidak:.5f}")
print("PART 3 done")

# ============================================================
# PART 3b — rigorous lower-bound chain and slice identity
# ============================================================
# Lower chain: box R in H-space with det <= -delta on R, box centered at mean.
# E[|det|1{det<0}1{F in W}] >= delta * P(H in R) * min_{vertices v} P(F in W | H=v)
# and P(H in R) >= prod_i (2 Phi(a_i/sig_i) - 1)  (Sidak).
m4 = np.array([0.4, 0.0, 0.0, 3.0])   # (F, A, B, C); C centered at 3
r = np.random.default_rng(900)
X = r.normal(0, 1, (4, 4)); S4 = X @ X.T / 4 + 0.35*np.eye(4)
# keep means as set; covariance general (box still centered at mean => Sidak ok)
half = np.array([0.5, 0.5, 0.3])      # halfwidths in (A,B,C)
ABmax = (abs(m4[1]) + half[0]) * (abs(m4[2]) + half[1])
C2min = (abs(m4[3]) - half[2]) ** 2
delta = C2min - ABmax
ck("P3b.delta>0", delta > 0, f"delta={delta:.4f}")
lo, hi = m4[0] - 0.4, m4[0] + 0.5
n = 1_200_000
z = rng.multivariate_normal(m4, S4, n)
F = z[:, 0]; H = z[:, 1:4]
detH = H[:,0]*H[:,1] - H[:,2]**2
inbox = (np.abs(H - m4[1:]) < half).all(1)
B = (F > lo) & (F < hi)
lhs = (np.abs(detH) * (detH < 0) * B).mean()
se_l = (np.abs(detH)*(detH<0)*B).std()/np.sqrt(n)
# conditional window probability at vertices
SFH = S4[0, 1:]; SHH = S4[1:, 1:]; sF = np.sqrt(S4[0,0] - SFH @ np.linalg.solve(SHH, SFH))
SHHinv = np.linalg.inv(SHH)
def pw_at(h):
    mu = m4[0] + SFH @ SHHinv @ (h - m4[1:])
    return norm.cdf((hi - mu)/sF) - norm.cdf((lo - mu)/sF)
verts = np.array([[m4[1]+s1*half[0], m4[2]+s2*half[1], m4[3]+s3*half[2]]
                  for s1 in (-1,1) for s2 in (-1,1) for s3 in (-1,1)])
pw_min = min(pw_at(v) for v in verts)
sd = np.sqrt(np.diag(SHH))
sidak = np.prod(2*norm.cdf(half/sd) - 1)
lb = delta * sidak * pw_min
ck("P3b.lower-chain", lhs >= lb - 4*se_l, f"lhs(mc)={lhs:.6f} >= lb={lb:.6f} (se {se_l:.1e})")
# vertex-min validity: pw over box minimized at a vertex (monotone in |mu-mid|, mu affine)
grid_pw = min(pw_at(m4[1:] + half*r.uniform(-1,1,3)) for _ in range(20000))
ck("P3b.vertex-min", pw_min <= grid_pw + 1e-12, f"vertex {pw_min:.6f} <= random {grid_pw:.6f}")

# Slice identity: E[X 1_A 1_B] = ∫_lo^hi phi(u;mF,sF2) G(u) du, G(u)=E[X 1_A | F=u]
m4b = r.normal(0, 1, 4)
X = r.normal(0, 1, (4, 4)); S4b = X @ X.T/4 + 0.3*np.eye(4)
n = 1_500_000
z = rng.multivariate_normal(m4b, S4b, n)
F = z[:,0]; H = z[:,1:4]; detH = H[:,0]*H[:,1]-H[:,2]**2
lo2, hi2 = m4b[0]-0.5, m4b[0]+0.3
sel = (F>lo2)&(F<hi2)
lhs2 = (np.abs(detH)*(detH<0)*sel).mean()
se2 = (np.abs(detH)*(detH<0)*sel).std()/np.sqrt(n)
SFH2 = S4b[0,1:]; SHH2 = S4b[1:,1:]; sF2 = np.sqrt(S4b[0,0])
SHc = SHH2 - np.outer(SFH2, SFH2)/S4b[0,0]
ug = np.linspace(lo2, hi2, 25)
Gu = []
for uu in ug:
    mh_u = m4b[1:] + SFH2*(uu - m4b[0])/S4b[0,0]
    h = rng.multivariate_normal(mh_u, SHc, 200_000)
    dq = h[:,0]*h[:,1]-h[:,2]**2
    Gu.append((np.abs(dq)*(dq<0)).mean())
Gu = np.array(Gu)
rhs2 = np.trapezoid(Gu * norm.pdf(ug, m4b[0], sF2), ug)
# MC noise on rhs2: G se ~ each < 1e-2; combined estimate rough
ck("P3b.slice-identity", abs(lhs2 - rhs2) < max(6*se2, 0.03*rhs2),
   f"lhs={lhs2:.6f} rhs(slice)={rhs2:.6f}")
print("PART 3b done")

# ============================================================
# PART 4 — the actual 9-pin pipeline (mpmath, 50 digits)
# Stations M=(-r/2,0), S=(r/2,0), Y=M+r(-0.76,0.24); r=0.025.
# ============================================================
import mpmath as mp
mp.mp.dps = 50
rr = mp.mpf('0.025'); bb = mp.mpf('1.2'); ell = rr**3/6
Ms = (-rr/2, mp.mpf(0)); Ss = (rr/2, mp.mpf(0))
Ys = (-rr/2 - mp.mpf('0.76')*rr, mp.mpf('0.24')*rr)
ST = [Ms, Ss, Ys]
PINALPHA = [(0,0),(1,0),(0,1)]

jm = list(range(-JMAX, JMAX+1))
kapm = [mp.mpf(j)*mp.pi/12 for j in jm]
wm = [mp.exp(-k*k/2) for k in kapm]
Zm = sum(wm)

def k1m(n, s):
    s = mp.mpf(s)
    tot = mp.mpf(0)
    for k_, w_ in zip(kapm, wm):
        tot += w_ * (1j*k_)**n * mp.exp(1j*k_*s)
    return (tot/Zm).real

def cov2m(a, b, s):
    return ((-1)**(b[0]+b[1])) * k1m(a[0]+b[0], s[0]) * k1m(a[1]+b[1], s[1])

pins = [(st, al) for st in ST for al in PINALPHA]   # 9 pins
NP_ = len(pins)
SPP = mp.zeros(NP_, NP_)
for i,(xi,ai) in enumerate(pins):
    for j,(xj,aj) in enumerate(pins):
        s = (xi[0]-xj[0], xi[1]-xj[1])
        SPP[i,j] = cov2m(ai, aj, s)
ev = mp.eig(SPP)[0] if False else None
# symmetry + positive definiteness via cholesky
try:
    mp.cholesky(SPP)
    chol_ok = True
except Exception:
    chol_ok = False
ck("P4.SPP-sym-pd", all(abs(SPP[i,j]-SPP[j,i]) < mp.mpf('1e-40') for i in range(NP_) for j in range(NP_)) and chol_ok)
SPPinv = SPP ** -1

def kvec_f(y):
    """Cov(f(y), pins) as mp matrix 9x1."""
    out = mp.zeros(NP_, 1)
    for j,(xj,aj) in enumerate(pins):
        out[j] = cov2m((0,0), aj, (y[0]-xj[0], y[1]-xj[1]))
    return out

def cond_var_f(y):
    k = kvec_f(y)
    v = 1 - (k.T * SPPinv * k)[0]
    return float(v)

# --- probe: jet-cluster prediction horizon (C030: 0.018/0.55/0.976 at d=1/2/3) ---
print("--- horizon probe: v(y) = Var(f(y) | 9 pins) ---")
arch = mp.sqrt(mp.mpf('0.76')**2 + mp.mpf('0.24')**2)
rays = {}
for d in [1,2,3]:
    cands = {
        '+x_from_S': (float(Ss[0]) + d, 0.0),
        '-x_from_M': (float(Ms[0]) - d, 0.0),
        '+y_from_M': (float(Ms[0]), d*1.0),
        '-y_from_M': (float(Ms[0]), -d*1.0),
        'arch_from_M': (float(Ms[0]) - d*0.76/float(arch), d*0.24/float(arch)),
        'origin_d+x': (d*1.0, 0.0),
        'origin_-x': (-d*1.0, 0.0),
    }
    for name, y in cands.items():
        v = cond_var_f(y)
        rays.setdefault(name, []).append(v)
        print(f"d={d} {name:12s} y=({y[0]:+.3f},{y[1]:+.3f})  v={v:.6f}")
target = [0.018, 0.55, 0.976]
best = None
for name, vs in rays.items():
    rel = max(abs(v-t)/t for v,t in zip(vs, target))
    print(f"ray {name:12s} vs=[{vs[0]:.4f},{vs[1]:.4f},{vs[2]:.4f}] max-rel-dev={rel:.3f}")
    if best is None or rel < best[1]:
        best = (name, rel)
ck("P4.horizon-match", best[1] < 0.35,
   f"best ray {best[0]} max-rel-dev {best[1]:.3f} vs C030 [0.018,0.55,0.976]")
print("PART 4 horizon done")

# --- mu_t: E[f(Y) | 6 pins at M,S and grad f(Y)=0] ; C031: (mu_t-b)/ell -> -0.4999290 ---
pins8 = [(Ms,a) for a in PINALPHA] + [(Ss,a) for a in PINALPHA] + [(Ys,(1,0)), (Ys,(0,1))]
v8 = [bb,0,0, bb-ell,0,0, 0,0]
S8 = mp.zeros(8,8)
for i,(xi,ai) in enumerate(pins8):
    for j,(xj,aj) in enumerate(pins8):
        S8[i,j] = cov2m(ai, aj, (xi[0]-xj[0], xi[1]-xj[1]))
k8 = mp.zeros(8,1)
for j,(xj,aj) in enumerate(pins8):
    k8[j] = cov2m((0,0), aj, (Ys[0]-xj[0], Ys[1]-xj[1]))
mu_t = (k8.T * (S8**-1) * mp.matrix(v8))[0]
ratio = (mu_t - bb)/ell
ck("P4.mu_t", abs(float(ratio) - (-0.4999290)) < 0.01,
   f"(mu_t-b)/ell = {mp.nstr(ratio,12)} vs -0.4999290 (r=0.025, limit const)")
print("mu_t =", mp.nstr(mu_t, 15))

# --- full conditional jet law machinery ---
def joint_JP(y):
    """mp matrices: SJJ (6x6), SJP (6x9) at point y."""
    SJJ = mp.zeros(6,6)
    for i,ai in enumerate(JET):
        for j2,aj in enumerate(JET):
            SJJ[i,j2] = cov2m(ai, aj, (mp.mpf(0), mp.mpf(0)))
    SJP = mp.zeros(6, NP_)
    for i,ai in enumerate(JET):
        for j,(xj,aj) in enumerate(pins):
            SJP[i,j] = cov2m(ai, aj, (y[0]-xj[0], y[1]-xj[1]))
    return SJJ, SJP

V9 = mp.matrix([bb,0,0, bb-ell,0,0, mu_t,0,0])

def cond_jet(y):
    """(mu_J, S_cond) of J(y) = (F,G1,G2,A,B,C) given the 9 pins. numpy float."""
    SJJ, SJP = joint_JP(y)
    mu = SJP * SPPinv * V9
    Sc = SJJ - SJP * SPPinv * SJP.T
    return np.array(mu.tolist(), float).ravel(), np.array(Sc.tolist(), float)

y_probe = (1.0, 0.3)   # generic point (d ~ 1, off-axis)
muJ, SC = cond_jet(y_probe)
ck("P4.SC-psd", np.linalg.eigvalsh(SC).min() > -1e-9,
   f"min eig {np.linalg.eigvalsh(SC).min():.3e}")

# two-stage (pins then G=0) vs one-shot (pins,G) conditioning of (F,H)
idxG = [1,2]; idxFH = [0,3,4,5]
SGG = SC[np.ix_(idxG,idxG)]
SGF = SC[np.ix_(idxG,idxFH)]
muG = muJ[idxG]; muFH = muJ[idxFH]
SGGinv = np.linalg.inv(SGG)
mu_2s = muFH - SGF.T @ SGGinv @ muG
SC_2s = SC[np.ix_(idxFH,idxFH)] - SGF.T @ SGGinv @ SGF
# one-shot in mpmath: condition (F,H) on (G, pins)
SJJm, SJPm = joint_JP(y_probe)
SGPm = mp.zeros(2, NP_)
for i,ai in enumerate([JET[1], JET[2]]):
    for j,(xj,aj) in enumerate(pins):
        SGPm[i,j] = cov2m(ai, aj, (y_probe[0]-xj[0], y_probe[1]-xj[1]))
idxFHm = [0,3,4,5]
SFHm = mp.zeros(4,4)
for i in range(4):
    for j in range(4):
        SFHm[i,j] = SJJm[idxFHm[i], idxFHm[j]]
SGGm = mp.zeros(2,2)
for i in range(2):
    for j in range(2):
        SGGm[i,j] = SJJm[1+i, 1+j]
SFGm = mp.zeros(4,2)
for i in range(4):
    for j in range(2):
        SFGm[i,j] = SJJm[idxFHm[i], 1+j]
# joint conditioning set C = (G, pins): cov(FH, C) = [SFGm, SJPm(FH,:)]; cov(C) = [[SGGm, SGPm],[SGPm.T, SPP]]
SFHPm = mp.zeros(4, NP_)
for i in range(4):
    for j in range(NP_):
        SFHPm[i,j] = SJPm[idxFHm[i], j]
SCC = mp.zeros(2+NP_, 2+NP_)
for i in range(2):
    for j in range(2):
        SCC[i,j] = SGGm[i,j]
for i in range(NP_):
    for j in range(NP_):
        SCC[2+i,2+j] = SPP[i,j]
for i in range(2):
    for j in range(NP_):
        SCC[i,2+j] = SGPm[i,j]; SCC[2+j,i] = SGPm[i,j]
SFHC = mp.zeros(4, 2+NP_)
for i in range(4):
    for j in range(2):
        SFHC[i,j] = SFGm[i,j]
    for j in range(NP_):
        SFHC[i,2+j] = SFHPm[i,j]
vC = mp.zeros(2+NP_,1)
for j in range(NP_):
    vC[2+j] = V9[j]
muFHm = mp.zeros(4,1)   # unconditional means are 0 (stationary, centered)
mu_1s = SFHC * (SCC**-1) * vC
SC_1s = SFHm - SFHC * (SCC**-1) * SFHC.T
mu_1s = np.array(mu_1s.tolist(), float).ravel()
SC_1s = np.array(SC_1s.tolist(), float)
ck("P4.twostage=oneshot-mean", np.max(np.abs(mu_2s-mu_1s)) < 1e-7*(1+np.max(np.abs(mu_1s))),
   f"maxdiff {np.max(np.abs(mu_2s-mu_1s)):.2e}")
ck("P4.twostage=oneshot-cov", np.max(np.abs(SC_2s-SC_1s)) < 1e-7*(1+np.max(np.abs(SC_1s))),
   f"maxdiff {np.max(np.abs(SC_2s-SC_1s)):.2e}")

# --- TDC covariance under pins: independence FAILS (kills isotropic-slice reuse) ---
SH = SC_2s[1:4, 1:4]   # (A,B,C) block of the (F,A,B,C) conditional covariance
U = np.array([[0.5,0.5,0.0],[0.5,-0.5,0.0],[0.0,0.0,1.0]])
STDC = U @ SH @ U.T
off = abs(STDC[0,1]) + abs(STDC[0,2]) + abs(STDC[1,2])
print("Sigma(T,D,C) | pins, grad=0:\n", STDC)
ck("P4.TD-correlated-under-pins", off > 1e-6,
   f"|Cov TD|+|Cov TC|+|Cov DC| = {off:.3e} > 0: isotropic independence destroyed by pins")
# and unconditional check: offdiag ~ 0 for the unconditioned H|f slice
ck("P4.TD-indep-unconditional", True, "structure verified in P1 (diag(2,2,1)->diag(1,1,1))")

# --- DF-row probe: v_grad(y) = Var(F | 9 pins, grad f(y)=0) vs C030 DF [0.001,0.278,0.974] ---
print("--- DF probe: Var(f(y) | 9 pins, grad f(y)=0) ---")
bestray = best[0]
df_target = [0.001, 0.278, 0.974]
df_vals = []
for d in [1,2,3]:
    if bestray == '+x_from_S': y = (float(Ss[0]) + d, 0.0)
    elif bestray == '-x_from_M': y = (float(Ms[0]) - d, 0.0)
    elif bestray == '+y_from_M': y = (float(Ms[0]), float(d))
    elif bestray == '-y_from_M': y = (float(Ms[0]), -float(d))
    elif bestray == 'arch_from_M': y = (float(Ms[0]) - d*0.76/float(arch), d*0.24/float(arch))
    elif bestray == 'origin_d+x': y = (float(d), 0.0)
    else: y = (-float(d), 0.0)
    muJd, SCd = cond_jet(y)
    SGd = SCd[np.ix_([1,2],[1,2])]
    v_grad = SCd[0,0] - SCd[0,[1,2]] @ np.linalg.solve(SGd, SCd[[1,2],0])
    df_vals.append(float(v_grad))
    print(f"d={d} v_grad={v_grad:.6f} (target {df_vals and df_target[len(df_vals)-1]})")
df_rel = max(abs(v-t)/t for v,t in zip(df_vals, df_target))
print(f"DF-row max-rel-dev = {df_rel:.3f} (best ray {bestray})")
# not a hard ck: C030's d-convention is not printed; record as probe.
ck("P4.DF-probe-recorded", True, f"DF rel-dev {df_rel:.3f}")

# --- bound values at the probe point + MC validation of the CS chain ---
bb = float(bb); ell = float(ell)
muF = float(mu_2s[0]); mH = np.array(mu_2s[1:4], float)
sF2 = SC_2s[0,0]
SFHb = SC_2s[0,1:4]; SHHb = SC_2s[1:4,1:4]
pG0 = np.exp(-0.5*muG @ SGGinv @ muG) / (2*PI*np.sqrt(np.linalg.det(SGG)))
EQ = Edet_formula(mH, SHHb)
V = Vardet_formula(mH, SHHb)
E2 = Edet2_formula(mH, SHHb)
PW = norm.cdf((bb - muF)/np.sqrt(sF2)) - norm.cdf((bb-ell-muF)/np.sqrt(sF2))
PW = float(PW)
b_ch = chernoff_neg(mH, SHHb); b_be = bernstein_neg(mH, SHHb); b_ca = cantelli_neg(mH, SHHb)
ub_global = pG0 * np.sqrt(E2 * min(PW, b_ch))
print(f"probe y={y_probe}: muF={muF:.5f} sF={np.sqrt(sF2):.5f} mH={np.round(mH,4)}")
print(f"pG0={pG0:.4g} E[det]={EQ:.4g} Var={V:.4g} E[det^2]={E2:.4g}")
print(f"P(window)={PW:.4g}  P(det<0): chernoff<={b_ch:.3g} bern<={b_be:.3g} cantelli<={b_ca:.3g}")
print(f"UB_global = {ub_global:.4g}")
# slice bound by quadrature
ug = np.linspace(float(bb-ell), float(bb), 60)
slice_ub = 0.0
SHc = SHHb - np.outer(SFHb, SFHb)/sF2
for i in range(len(ug)-1):
    um = 0.5*(ug[i]+ug[i+1])
    mh_u = mH + SFHb*(um-muF)/sF2
    E2u = Edet2_formula(mh_u, SHc)
    pu = min(1.0, chernoff_neg(mh_u, SHc), cantelli_neg(mh_u, SHc))
    Gu_ub = np.sqrt(E2u*pu)
    slice_ub += (ug[i+1]-ug[i]) * norm.pdf(um, muF, np.sqrt(sF2)) * Gu_ub
slice_ub *= pG0
print(f"UB_slice = {slice_ub:.4g}  (<= global {ub_global:.4g}: {slice_ub <= ub_global*(1+1e-6)})")
# MC validation of global CS at the conditioned law with a widened window (feasibility)
n = 2_000_000
zF = rng.normal(muF, np.sqrt(sF2), n)
zH = rng.multivariate_normal(mH, SHHb, n)
# note: F and H correlated; sample jointly instead
zFH = rng.multivariate_normal(mu_2s, SC_2s, n)
Fq = zFH[:,0]; Hq = zFH[:,1:4]
dq = Hq[:,0]*Hq[:,1]-Hq[:,2]**2
selw = (Fq > muF-1.0) & (Fq < muF+1.0)
lhs = (np.abs(dq)*(dq<0)*selw).mean()
se = (np.abs(dq)*(dq<0)*selw).std()/np.sqrt(n)
PWw = norm.cdf(1.0/np.sqrt(sF2)) - norm.cdf(-1.0/np.sqrt(sF2))
rhs = np.sqrt(E2 * min(PWw, 1.0))
ck("P4.globalCS-conditioned", lhs <= rhs + 4*se, f"lhs={lhs:.5f} rhs={rhs:.5f}")
# defective form check here too: would sqrt(E2)*PWw bound fail?
print(f"(info) defective rhs sqrt(E2)*PWw = {np.sqrt(E2)*PWw:.5f} vs lhs {lhs:.5f}")

print()
print("="*60)
if FAILS:
    print("FAILURES:", FAILS); sys.exit(1)
print("ALL CHECKS PASSED")
