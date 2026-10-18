"""(i) a,b,c of the actual lattice law vs L (physical wave numbers 2 pi m / L, Gaussian weights
exp(-p^2/2)), with a Poisson-summation cross-check of z_L; (ii) numerical units check of
formula (1): spectral dilation Q -> lam Q must give C_alg -> lam^d C_alg; amplitude
f -> sig f must give C_alg -> sig^(-2/3) C_alg; (iii) positivity/finiteness of the
integrand of (1) on a dense sphere grid (d=2, exact F = Gamma_e/2; d=3 F by CRN-MC),
all directly from the moments with high precision (mpmath, 40 digits).
Run: venv/bin/python -I scale_L.py
"""
import mpmath as mp
mp.mp.dps = 40


def lattice_coord(L):
    h = 2 * mp.pi / L
    K = 1
    while mp.exp(-(h * K) ** 2 / 2) > mp.mpf(10) ** -45:
        K += 1
    pts = [(h * k, mp.exp(-(h * k) ** 2 / 2)) for k in range(-K, K + 1)]
    z = mp.fsum(p for _, p in pts)
    return [(v, p / z) for v, p in pts], z


print('(i) lattice moments a=mu2, b=mu4, c=mu6 vs L (continuum 1, 3, 15)')
for L in [mp.mpf(x) for x in ('0.5', '1', '2', '3', '5', '8', '12', '20')]:
    co, z = lattice_coord(L)
    a = mp.fsum(p * v ** 2 for v, p in co); b = mp.fsum(p * v ** 4 for v, p in co); c = mp.fsum(p * v ** 6 for v, p in co)
    zP = L / mp.sqrt(2 * mp.pi) * mp.fsum(mp.exp(-n * n * L * L / 2) for n in range(-60, 61))
    k4 = b - 3 * a * a
    print('  L=%5s a=%s b=%s c=%s kappa4=%s  |z-Poisson|/z=%s  c-b^2/a=%s' % (
        mp.nstr(L, 3), mp.nstr(a, 12), mp.nstr(b, 12), mp.nstr(c, 12), mp.nstr(k4, 6), mp.nstr(abs(z - zP) / z, 3),
        mp.nstr(c - b * b / a, 8)))


def calg2(co, sig=1, nth=64):
    """C_alg in d=2 from formula (1), all objects from moments of the product law with coordinate law co."""
    m = lambda k: mp.fsum(p * v ** k for v, p in co) * sig ** 2   # field variance sig^2 scales all spectral moments
    a, b, c = m(2), m(4), m(6)
    k4 = b - 3 * a * a / sig ** 2   # moments of the product law with total mass sig^2: E Q1^2 Q2^2 = a^2/sig^2
    # direct (not via (10)-(13)): build moment tensors of Q for the product law with mass sig^2
    def EQ(pows):
        out = sig ** 2
        for pw in pows:
            out *= mp.fsum(p * v ** pw for v, p in co)
        return out
    def mom(idx):  # E[prod Q_idx] for idx a tuple of coordinates in {0,1}
        return EQ([idx.count(0), idx.count(1)])
    tot = mp.mpf(0); mins = [mp.inf] * 3
    for j in range(nth):
        th = 2 * mp.pi * j / nth
        e = [mp.cos(th), mp.sin(th)]; u = [-mp.sin(th), mp.cos(th)]
        # Cov(H_ij,H_kl)=E[Qi Qj Qk Ql]; He_i = sum_j H_ij e_j ; A = u^T H u
        def cH(i, j, k, l):
            return mom((i, j, k, l))
        B = mp.matrix(2, 2)
        for i in range(2):
            for k in range(2):
                B[i, k] = mp.fsum(cH(i, j, k, l) * e[j] * e[l] for j in range(2) for l in range(2))
        RA = [mp.fsum(cH(p, q, i, j) * u[p] * u[q] * e[j] for p in range(2) for q in range(2) for j in range(2)) for i in range(2)]
        CA = mp.fsum(cH(p, q, s, t) * u[p] * u[q] * u[s] * u[t] for p in range(2) for q in range(2) for s in range(2) for t in range(2))
        RAv = mp.matrix(RA)
        gam = CA - (RAv.T * mp.inverse(B) * RAv)[0]
        F = gam / 2
        # T_e = d^3 f along e ; Var(T_e)=E(Q.e)^6 ; Cov(T_e,D_i)=-E[(Q.e)^3 Q_i] ; Cov(D)=E[QQ^T]
        def EQe(powe, extra):
            # E[(Q.e)^powe * prod Q_extra] by multinomial expansion
            from math import comb
            s = mp.mpf(0)
            for k in range(powe + 1):
                idx = (0,) * k + (1,) * (powe - k) + tuple(extra)
                s += comb(powe, k) * e[0] ** k * e[1] ** (powe - k) * mom(idx)
            return s
        VT = EQe(6, ())
        v = mp.matrix([EQe(3, (i,)) for i in range(2)])
        CD = mp.matrix([[mom((i, k)) for k in range(2)] for i in range(2)])
        V = VT - (v.T * mp.inverse(CD) * v)[0]
        detB = mp.det(B)
        mins = [min(mins[0], V), min(mins[1], F), min(mins[2], detB)]
        tot += F * V ** (mp.mpf(2) / 3) / mp.sqrt(detB) * 2 * mp.pi / nth
    aa = CD[0, 0]
    pref = mp.gamma(mp.mpf(7) / 6) / (mp.cbrt(24) * mp.sqrt(mp.pi) * (2 * mp.pi) ** 2 * aa)
    return pref * tot, mins


print('(ii) units/scaling of formula (1), d=2')
co3, _ = lattice_coord(mp.mpf(3))
C0, mins0 = calg2(co3)
lam = mp.mpf(3) / 2
C1, _ = calg2([(lam * v, p) for v, p in co3])
sig = mp.mpf(7) / 5
C2, _ = calg2(co3, sig=sig)
print('  C_alg(L=3) = %s ; spectral dilation ratio/lam^2 = %s ; amplitude ratio/sig^(-2/3) = %s' % (
    mp.nstr(C0, 15), mp.nstr(C1 / C0 / lam ** 2, 15), mp.nstr(C2 / C0 / sig ** (-mp.mpf(2) / 3), 15)))
for nth in (16, 32, 64, 128):
    Cn, _ = calg2(co3, nth=nth)
    print('  trapezoid nodes %3d : C_alg(L=3) = %s' % (nth, mp.nstr(Cn, 20)))
print('(iii) positivity on the sphere grid (min V, min F, min det B):')
for Ls in ('0.8', '1.5', '3', '6', '15'):
    co, _ = lattice_coord(mp.mpf(Ls))
    C, mins = calg2(co, nth=96)
    print('  L=%4s C_alg=%s  minV=%s minF=%s mindetB=%s' % (Ls, mp.nstr(C, 12), mp.nstr(mins[0], 6), mp.nstr(mins[1], 6), mp.nstr(mins[2], 6)))
# continuum reference (L -> infinity: Q ~ N(0,I)) for d=2 in closed form: a=1, B=I+2ee^T (det 3), V=6, Gamma=? computed
