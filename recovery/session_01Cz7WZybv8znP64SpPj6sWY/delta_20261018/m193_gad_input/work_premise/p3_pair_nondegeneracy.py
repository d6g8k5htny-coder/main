"""Premise audit, part 3: the pair vectors used in Section 7.

(i)  O=(F(M),grad F(M),F(S),grad F(S)) has a strictly positive definite covariance for
     every M!=S on the torus (needed for GAD's canonical regression Q_O and p_O at every
     literal target).  Grid over the fundamental domain incl. half-lattice points.
(ii) Rescaled pin vector Y_r=(F(M),P_r,D_r,W_r) for M=x-re/2, S=x+re/2: its covariance
     converges as r->0 to Cov(F,D,He,T_e) (uniformly in e on the sampled grid) and stays
     uniformly positive definite; the contact limit drops nothing.
(iii) Jacobian of (D_M,D_S,Delta)->(P_r,D_r,W_r) equals 12 r^-(d+3) (exact, sympy).
Symbols: lambda(e^{ip.}) ; Cov(lambda F, mu F) = sum_m w_m Re[phi_lambda(p_m) conj(phi_mu(p_m))].
"""
import itertools
import numpy as np
import mpmath as mp
import sympy as sp

def lattice(L, d, tol_exp=40):
    h = 2*np.pi/L
    K = int(np.sqrt(2*(tol_exp*np.log(10)+30))/h) + 2
    rng = np.arange(-K, K+1)
    m = np.array(list(itertools.product(rng, repeat=d)), dtype=float)
    p = h*m
    w = np.exp(-np.sum(p**2, axis=1)/2)
    return p, w/np.sum(w)

def cov_from_symbols(phis, w):
    Phi = np.array(phis)                    # (k, nm) complex
    G = (Phi*w) @ Phi.conj().T
    return G.real

def corr_mineig(C):
    s = np.sqrt(np.diag(C))
    R = C/np.outer(s, s)
    return np.linalg.eigvalsh(R)[0]

print('(i) Cov(O) positive definiteness for M!=S')
for d, L, n in [(2, 0.7, 41), (2, 2.0, 41), (2, 6.0, 41), (3, 1.0, 13), (3, 4.0, 13)]:
    p, w = lattice(L, d)
    worst = (np.inf, None)
    grid = np.linspace(-L/2, L/2, n)
    for y in itertools.product(grid, repeat=d):
        y = np.array(y)
        if np.linalg.norm(y) < 0.04*L:
            continue
        ph = np.exp(1j*(p @ y))
        phis = [np.ones_like(ph)] + [1j*p[:, j] for j in range(d)] + [ph] + [1j*p[:, j]*ph for j in range(d)]
        me = corr_mineig(cov_from_symbols(phis, w))
        if me < worst[0]:
            worst = (me, y)
    print('   d=%d L=%.2f: min eig(corr O) over |y|>=0.04L grid = %.3e at y=%s' % (d, L, worst[0], np.round(worst[1], 3)))

print('(ii) rescaled contact vector Y_r -> (F,D,He,T_e)')
mp.mp.dps = 50
def mp_lattice(L, d):
    h = 2*mp.pi/L
    K = int(mp.sqrt(2*(50*mp.log(10)+30))/h) + 2
    pts = []
    tot = mp.mpf(0)
    for m in itertools.product(range(-K, K+1), repeat=d):
        p = [h*k for k in m]
        wv = mp.e**(-mp.fsum(x*x for x in p)/2)
        pts.append((p, wv)); tot += wv
    return [(p, wv/tot) for p, wv in pts]

def mp_cov(phis_fun, pts, k):
    C = mp.matrix(k, k)
    for p, wv in pts:
        ph = phis_fun(p)
        for i in range(k):
            for j in range(i, k):
                C[i, j] += wv*mp.re(ph[i]*mp.conj(ph[j]))
    for i in range(k):
        for j in range(i):
            C[i, j] = C[j, i]
    return C

def thc(th):  # theta cos theta - sin theta, cancellation-free series for small theta
    if abs(th) > mp.mpf('0.1'):
        return th*mp.cos(th) - mp.sin(th)
    s = mp.mpf(0); n = 1
    while True:
        t = (-1)**n*th**(2*n+1)*(2*n)/mp.factorial(2*n+1)
        s += t
        if abs(t) < mp.mpf(10)**(-60)*max(abs(s), mp.mpf(10)**(-300)):
            break
        n += 1
    return s

for d, L in [(2, 1.0), (2, 5.0), (3, 2.0)]:
    pts = mp_lattice(L, d)
    k = 1 + d + d + 1
    for ang in ([1, 0, 0], [1, 1, 0], [3, 1, 2], [1, 2, 0]):
        e = [mp.mpf(x) for x in ang[:d]]
        nrm = mp.sqrt(mp.fsum(x*x for x in e)); e = [x/nrm for x in e]
        def lim_syms(p):
            pe = mp.fsum(a*b for a, b in zip(p, e))
            return [mp.mpc(1)] + [mp.mpc(0, x) for x in p] + [-pe*x for x in p] + [mp.mpc(0, -pe**3)]
        C0 = mp_cov(lim_syms, pts, k)
        lam0 = min(mp.eigsy(C0, eigvals_only=True))
        out = []
        for r in [mp.mpf('0.3'), mp.mpf('0.03'), mp.mpf('0.003'), mp.mpf('0.0003')]:
            r = r*L
            def syms(p):
                pe = mp.fsum(a*b for a, b in zip(p, e))
                th = r*pe/2
                FM = mp.e**(mp.mpc(0, -th))
                P = [mp.mpc(0, x)*mp.cos(th) for x in p]
                D = [-2*x*mp.sin(th)/r for x in p]
                W = mp.mpc(0, 24)/r**3*thc(th)
                return [FM] + P + D + [W]
            Cr = mp_cov(syms, pts, k)
            diff = mp.mnorm(Cr - C0, 1)/mp.mnorm(C0, 1)
            lam = min(mp.eigsy(Cr, eigvals_only=True))
            out.append('r/L=%s: rel|C_r-C_0|=%s lmin=%s' % (mp.nstr(r/L, 2), mp.nstr(diff, 3), mp.nstr(lam, 5)))
        print('   d=%d L=%s e=%s: lmin(Cov(F,D,He,T_e))=%s ; %s' % (d, L, [mp.nstr(x, 3) for x in e], mp.nstr(lam0, 5), ' | '.join(out)))

print('(iii) Jacobian of the Section-7 pin transform')
for d in range(2, 6):
    r, ee = sp.symbols('r', positive=True), sp.symbols('f0:%d' % d)
    DM = sp.symbols('m0:%d' % d); DS = sp.symbols('s0:%d' % d); Dl = sp.Symbol('Delta')
    P = [(DM[i]+DS[i])/2 for i in range(d)]
    Dr = [(DS[i]-DM[i])/r for i in range(d)]
    W = 12/r**3*(Dl + r*sum(ee[i]*P[i] for i in range(d)))
    J = sp.Matrix(P + Dr + [W]).jacobian(list(DM) + list(DS) + [Dl])
    print('   d=%d: |det| = %s' % (d, sp.simplify(sp.Abs(J.det()))))
# Taylor identity Delta + r e.P_r = r^3 T_e/12 + O(r^5) along the line (W_r -> T_e)
t, r = sp.symbols('t r')
f = sp.Function('f')
expr = f(-r/2) - f(r/2) + r*(sp.diff(f(t), t).subs(t, -r/2) + sp.diff(f(t), t).subs(t, r/2))/2
ser = sp.series(expr, r, 0, 6).removeO()
print('   series of F(M)-F(S)+r e.P_r :', sp.simplify(ser.doit()))
