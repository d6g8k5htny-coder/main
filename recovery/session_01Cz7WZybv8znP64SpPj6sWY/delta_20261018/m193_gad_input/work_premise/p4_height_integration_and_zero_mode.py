"""Premise audit, part 4.

(A) Identity (16) at a literal zero-gradient/gap target, for the ACTUAL torus law (d=2):
    p_L(0,0,l) * Law(W | L=(0,0,l))  ==  int_R p_O(b,0,b-l,0) Law(W | O=(b,0,b-l,0)) db,
    L=(grad F(M), grad F(S), F(M)-F(S)), W = (Hessian entries at M and S, a far value F(z)).
    Checked on total mass, first and second moments (mixture of Gaussians with mean
    linear in b under Gaussian b-weight is Gaussian, so these determine the measure).
(B) Retained zero mode: for an explicit field sample on a periodic grid, finite superlevel
    H0 bar lengths are identical with the zero mode, without it, and after samplewise
    spatial-mean removal (the spatial mean of (5) is exactly sqrt(w0) Z0).
"""
import itertools
import numpy as np
import mpmath as mp

mp.mp.dps = 40
d, L = 2, 3.0
h = 2*np.pi/L
K = int(np.sqrt(2*(40*np.log(10)+30))/h) + 2
m = np.array(list(itertools.product(range(-K, K+1), repeat=d)), dtype=float)
p = h*m
w = np.exp(-np.sum(p**2, axis=1)/2); w /= w.sum()

M = np.array([0.0, 0.0]); S = np.array([0.41, 0.17]); Z = np.array([1.3, -0.7])
eM = np.exp(1j*(p @ M)); eS = np.exp(1j*(p @ S)); eZ = np.exp(1j*(p @ Z))
def hess(ex):
    return [-p[:, 0]**2*ex, -p[:, 0]*p[:, 1]*ex, -p[:, 1]**2*ex]
syms = ([eM] + [1j*p[:, j]*eM for j in range(d)] + [eS] + [1j*p[:, j]*eS for j in range(d)]
        + hess(eM) + hess(eS) + [eZ])
Phi = np.array(syms)
C = ((Phi*w) @ Phi.conj().T).real
C = (C + C.T)/2  # exact symmetry before mp conversion
C = mp.matrix(C.tolist())
nO = 2 + 2*d
iO = list(range(nO)); iW = list(range(nO, C.rows))

def sub(A, I, J):
    B = mp.matrix(len(I), len(J))
    for a_, i in enumerate(I):
        for b_, j in enumerate(J):
            B[a_, b_] = A[i, j]
    return B

# linear map T: X -> L=(gM, gS, F(M)-F(S)) ; T rows over all coords
iFM, igM, iFS, igS = 0, [1, 2], 3, [4, 5]
nX = C.rows
T = mp.matrix(2*d+1, nX)
for k, i in enumerate(igM):
    T[k, i] = 1
for k, i in enumerate(igS):
    T[d+k, i] = 1
T[2*d, iFM] = 1; T[2*d, iFS] = -1
CL = T*C*T.T
CWL = sub(C, iW, range(nX))*T.T
CW = sub(C, iW, iW)
ell = mp.mpf('0.05')
lv = mp.matrix([0]*(2*d) + [ell])
def gauss_pdf(x, Sig):
    n = Sig.rows
    return mp.e**(-(x.T*mp.inverse(Sig)*x)[0]/2)/mp.sqrt((2*mp.pi)**n*mp.det(Sig))
pL = gauss_pdf(lv, CL)
muL = CWL*mp.inverse(CL)*lv
SigL = CW - CWL*mp.inverse(CL)*CWL.T

CO = sub(C, iO, iO); CWO = sub(C, iW, iO); COi = mp.inverse(CO)
SigO = CW - CWO*COi*CWO.T
def target(bv):
    return mp.matrix([bv, 0, 0, bv-ell, 0, 0])
def pO(bv):
    return gauss_pdf(target(bv), CO)
mass = mp.quad(pO, [-mp.inf, 0, mp.inf])
nW = len(iW)
mean = mp.matrix(nW, 1); second = mp.matrix(nW, nW)
for i in range(nW):
    mean[i] = mp.quad(lambda bv: pO(bv)*(CWO*COi*target(bv))[i], [-mp.inf, 0, mp.inf])/mass
for i in range(nW):
    for j in range(i, nW):
        val = mp.quad(lambda bv: pO(bv)*((CWO*COi*target(bv))[i]*(CWO*COi*target(bv))[j]), [-mp.inf, 0, mp.inf])/mass
        second[i, j] = second[j, i] = val + SigO[i, j]
print('(A) mass: int p_O db = %s ; p_L = %s ; rel diff %s' % (mp.nstr(mass, 15), mp.nstr(pL, 15), mp.nstr(abs(mass-pL)/pL, 3)))
print('    mean rel diff  %s' % mp.nstr(mp.mnorm(mean - muL, 1)/max(mp.mnorm(muL, 1), mp.mpf(1e-30)), 3))
print('    2nd-moment rel diff %s' % mp.nstr(mp.mnorm(second - (SigL + muL*muL.T), 1)/mp.mnorm(SigL + muL*muL.T, 1), 3))
print('    Jacobian |det| of O -> (b, L_MS):', abs(mp.det(mp.matrix([[1 if (r == c) else 0 for c in range(6)] for r in range(6)]))))

# ---------------- (B) zero mode and finite H0 bars ----------------
rng = np.random.default_rng(20261018)
N, L2 = 96, 14.0
h2 = 2*np.pi/L2
K2 = 14
ms = [mm for mm in itertools.product(range(-K2, K2+1), repeat=2)]
H = [mm for mm in ms if mm > (0, 0)]   # one of each +-pair
Zs = sum(np.exp(-(h2**2)*(a_*a_+b_*b_)/2) for a_, b_ in ms)
wf = {mm: np.exp(-(h2**2)*(mm[0]**2+mm[1]**2)/2)/Zs for mm in ms}
xs = np.arange(N)*L2/N
X1, X2 = np.meshgrid(xs, xs, indexing='ij')
f_nz = np.zeros((N, N))
for mm in H:
    ph = h2*(mm[0]*X1 + mm[1]*X2)
    f_nz += np.sqrt(2*wf[mm])*(rng.standard_normal()*np.cos(ph) + rng.standard_normal()*np.sin(ph))
Z0 = rng.standard_normal()
f_full = np.sqrt(wf[(0, 0)])*Z0 + f_nz

def h0_bars(f):
    n0, n1 = f.shape
    order = np.argsort(-f, axis=None, kind='stable')
    parent = -np.ones(f.size, dtype=int); birth = np.zeros(f.size)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    bars = []
    for idx in order:
        i, j = divmod(idx, n1)
        parent[idx] = idx; birth[idx] = f[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = ((i+di) % n0)*n1 + (j+dj) % n1
            if parent[nb] < 0:
                continue
            ra, rb = find(idx), find(nb)
            if ra == rb:
                continue
            # elder rule: younger (lower birth) dies at current level
            if birth[ra] < birth[rb]:
                ra, rb = rb, ra
            bars.append(birth[rb] - f[i, j])
            parent[rb] = ra
    return np.sort(np.array([x for x in bars if x > 0]))

b_full = h0_bars(f_full); b_nz = h0_bars(f_nz); b_mr = h0_bars(f_full - f_full.mean())
print('(B) grid mean of field - sqrt(w0)Z0 = %.2e' % (f_full.mean() - np.sqrt(wf[(0, 0)])*Z0))
print('    #finite bars: full=%d, no-zero-mode=%d, mean-removed=%d' % (len(b_full), len(b_nz), len(b_mr)))
print('    max |bar diff| full vs no-zero-mode = %.2e ; full vs mean-removed = %.2e'
      % (np.max(np.abs(b_full - b_nz)), np.max(np.abs(b_full - b_mr))))
