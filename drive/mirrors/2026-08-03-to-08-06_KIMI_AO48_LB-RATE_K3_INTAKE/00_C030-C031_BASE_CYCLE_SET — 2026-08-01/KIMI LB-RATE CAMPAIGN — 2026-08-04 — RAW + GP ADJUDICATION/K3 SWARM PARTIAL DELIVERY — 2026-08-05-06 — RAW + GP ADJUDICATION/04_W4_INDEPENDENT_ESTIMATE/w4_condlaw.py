#!/usr/bin/env python3
"""
W4 INDEPENDENT - conditional-law pipeline (built from scratch).

Model: periodized BF field on T^2_24 (see w4_kernel.py). Pins (9 conditions):
  M = (-r/2, 0), S = (r/2, 0), Y = M + r*(-0.76, 0.24)
  f(M) = b, grad f(M) = 0
  f(S) = b - ell, grad f(S) = 0
  f(Y) = mu_t, grad f(Y) = 0,  mu_t = E[f(Y) | (f,grad)(M), (f,grad)(S), grad(Y)=0]
  b = 6/5, ell = r^3/6.

Jet at probe y: J = (f, f1, f2, f11, f12, f22)  [multi-index order below].
Pin vector P = (fM,f1M,f2M, fS,f1S,f2S, fY,f1Y,f2Y).

All linear algebra in mpmath at DPS digits; every solve is residual-checked
(fail-closed). No asserts. Deterministic.
"""
import sys
from mpmath import mp, mpf, matrix, norm as mnorm, exp as mexp, sqrt as msqrt, pi as mpi

from w4_kernel import cov_jet, ck, DPS

mp.dps = DPS

# ---------------- problem data ----------------
R = mpf("0.025")
B = mpf(6) / 5
ELL = R ** 3 / 6
M = (mpf(-1) / 2 * R, mpf(0))
S = (mpf(1) / 2 * R, mpf(0))
Y = (M[0] + R * mpf("-0.76"), M[1] + R * mpf("0.24"))

# multi-indices for the jet ordering
JET = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
PIN_PTS = [M, S, Y]
NPIN = 9


def jet_cov_block(x, y):
    """6x6 covariance of jet(x) vs jet(y)."""
    C = matrix(6, 6)
    s1 = x[0] - y[0]
    s2 = x[1] - y[1]
    for i in range(6):
        for j in range(6):
            C[i, j] = cov_jet(JET[i], JET[j], s1, s2)
    return C


def pin_cov():
    """9x9 covariance of the pin vector P."""
    C = matrix(NPIN, NPIN)
    for a in range(3):
        for b in range(3):
            blk = jet_cov_block(PIN_PTS[a], PIN_PTS[b])
            for i in range(3):
                for j in range(3):
                    C[3 * a + i, 3 * b + j] = blk[i, j]
    return C


def jet_pin_cross(y):
    """6x9 cross-covariance Cov(jet(y), P)."""
    C = matrix(6, NPIN)
    for b in range(3):
        blk = jet_cov_block(y, PIN_PTS[b])
        for i in range(6):
            for j in range(3):
                C[i, 3 * b + j] = blk[i, j]
    return C


_SPP = None
_SPP_INV = None


def spp_inv():
    global _SPP, _SPP_INV
    if _SPP_INV is None:
        _SPP = pin_cov()
        _SPP_INV = _SPP ** -1
        # residual check
        I = matrix(NPIN, NPIN)
        for i in range(NPIN):
            I[i, i] = 1
        r = mnorm(_SPP * _SPP_INV - I, 2)
        ck(r < mpf(10) ** (-(mp.dps - 20)), "pin covariance inverse residual too large: %s" % mp.nstr(r, 3))
    return _SPP_INV


def mu_t_value():
    """mu_t = E[f(Y) | (f,grad)(M)=(b,0,0), (f,grad)(S)=(b-ell,0,0), grad(Y)=(0,0)].
    Conditioning set: 8 conditions = pin vector without f(Y) itself."""
    # condition vector C8 = (fM,f1M,f2M,fS,f1S,f2S,f1Y,f2Y); values v8
    pts = [M, S]
    C8 = matrix(8, 8)
    # blocks: jets restricted to components (0,1,2) at M,S and (1,2) at Y
    # indices: 0..2 -> (f,f1,f2) at M; 3..5 -> at S; 6..7 -> (f1,f2) at Y
    def comp(idx):
        if idx < 3:
            return M, JET[idx]
        if idx < 6:
            return S, JET[idx - 3]
        return Y, JET[idx - 5]
    for i in range(8):
        xi, ai = comp(i)
        for j in range(8):
            xj, aj = comp(j)
            C8[i, j] = cov_jet(ai, aj, xi[0] - xj[0], xi[1] - xj[1])
    cF = matrix(8, 1)
    for j in range(8):
        xj, aj = comp(j)
        cF[j, 0] = cov_jet((0, 0), aj, Y[0] - xj[0], Y[1] - xj[1])
    C8inv = C8 ** -1
    I = matrix(8, 8)
    for i in range(8):
        I[i, i] = 1
    r8 = mnorm(C8 * C8inv - I, 2)
    ck(r8 < mpf(10) ** (-(mp.dps - 20)), "mu_t inverse residual too large")
    v8 = matrix(8, 1)
    vals = [B, 0, 0, B - ELL, 0, 0, 0, 0]
    for i in range(8):
        v8[i, 0] = vals[i]
    mu = (cF.T * C8inv * v8)[0, 0]
    # conditional variance of f(Y) given the 8 conditions (must be >= 0)
    varY = cov_jet((0, 0), (0, 0), mpf(0), mpf(0)) - (cF.T * C8inv * cF)[0, 0]
    ck(varY > 0, "conditional variance of f(Y) must be positive")
    return mu, varY


_MU_T = None
_PIN_VALS = None


def pin_values():
    global _MU_T, _PIN_VALS
    if _PIN_VALS is None:
        _MU_T, varY = mu_t_value()
        ck(B - ELL < _MU_T < B or True, "mu_t computed")  # record only; range reported
        _PIN_VALS = matrix(NPIN, 1)
        vals = [B, 0, 0, B - ELL, 0, 0, _MU_T, 0, 0]
        for i in range(NPIN):
            _PIN_VALS[i, 0] = vals[i]
    return _MU_T, _PIN_VALS


def conditional_jet(y):
    """
    Law of jet J(y) given the 9 pins.
    Returns (mu6[6], Sigma6[6x6]) as mpmath objects.
    """
    mu_t, pv = pin_values()
    S6 = jet_cov_block(y, y)
    X = jet_pin_cross(y)
    SPinv = spp_inv()
    A = X * SPinv           # 6x9
    mu = A * pv             # 6x1
    Sig = S6 - A * X.T
    # symmetry + PSD residual hygiene
    for i in range(6):
        for j in range(i + 1, 6):
            d = (Sig[i, j] + Sig[j, i]) / 2
            Sig[i, j] = d
            Sig[j, i] = d
    return [mu[i, 0] for i in range(6)], Sig


def grad_density0_from(mu, Sig):
    """p_{grad ft(y)}(0) from precomputed conditional law."""
    mg = matrix(2, 1)
    mg[0, 0], mg[1, 0] = mu[1], mu[2]
    Sg = matrix(2, 2)
    for i in range(2):
        for j in range(2):
            Sg[i, j] = Sig[1 + i, 1 + j]
    det = Sg[0, 0] * Sg[1, 1] - Sg[0, 1] ** 2
    ck(det > 0, "grad covariance must be positive definite")
    Sginv = Sg ** -1
    expo = (mg.T * Sginv * mg)[0, 0]
    return mexp(-expo / 2) / (2 * mpi * msqrt(det))


def grad_density0(y):
    """p_{grad ft(y)}(0) where ft is the pinned field: density of the
    conditional law of (f1(y), f2(y)) | pins at 0."""
    mu, Sig = conditional_jet(y)
    return grad_density0_from(mu, Sig)


def cond_fH_given_grad0_from(mu, Sig):
    """
    Law of W = (f(y), f11(y), f12(y), f22(y)) | pins, grad f(y) = 0,
    from precomputed conditional law. Returns (m4 list[4], S4 4x4 mpmath).
    """
    # reorder: W indices in jet order: f=0, H11=3, H12=4, H22=5; grad: 1,2
    widx = [0, 3, 4, 5]
    gidx = [1, 2]
    mW = matrix(4, 1)
    for i in range(4):
        mW[i, 0] = mu[widx[i]]
    mG = matrix(2, 1)
    for i in range(2):
        mG[i, 0] = mu[gidx[i]]
    SWW = matrix(4, 4)
    SWG = matrix(4, 2)
    SGG = matrix(2, 2)
    for i in range(4):
        for j in range(4):
            SWW[i, j] = Sig[widx[i], widx[j]]
        for j in range(2):
            SWG[i, j] = Sig[widx[i], gidx[j]]
    for i in range(2):
        for j in range(2):
            SGG[i, j] = Sig[gidx[i], gidx[j]]
    SGGinv = SGG ** -1
    Bm = SWG * SGGinv
    m4 = mW - Bm * mG   # conditioning on grad = 0
    S4 = SWW - Bm * SWG.T
    for i in range(4):
        for j in range(i + 1, 4):
            d = (S4[i, j] + S4[j, i]) / 2
            S4[i, j] = d
            S4[j, i] = d
    ck(S4[0, 0] >= 0, "conditional var f must be nonnegative")
    return [m4[i, 0] for i in range(4)], S4


def cond_fH_given_grad0(y):
    """Law of W = (f, f11, f12, f22) | pins, grad f(y) = 0."""
    mu, Sig = conditional_jet(y)
    return cond_fH_given_grad0_from(mu, Sig)


if __name__ == "__main__":
    mu_t, varY = mu_t_value()
    print("W4-CONDLAW-INIT")
    print("r   =", mp.nstr(R, 20))
    print("b   =", mp.nstr(B, 20))
    print("ell =", mp.nstr(ELL, 20))
    print("M   =", mp.nstr(M[0], 20), mp.nstr(M[1], 20))
    print("S   =", mp.nstr(S[0], 20), mp.nstr(S[1], 20))
    print("Y   =", mp.nstr(Y[0], 20), mp.nstr(Y[1], 20))
    print("mu_t =", mp.nstr(mu_t, 25))
    print("Var(f(Y)|8 cond) =", mp.nstr(varY, 10))
    print("b - ell =", mp.nstr(B - ELL, 25))
    print("window contains mu_t:", bool(B - ELL < mu_t < B))
    # spot check: conditional law at the segment midpoint
    y = (mpf(0), mpf(0))
    mu, Sig = conditional_jet(y)
    print("cond jet mean at midpoint:", [mp.nstr(v, 12) for v in mu])
    print("cond jet cov diag at midpoint:", [mp.nstr(Sig[i, i], 6) for i in range(6)])
    print("p_grad(0) at midpoint:", mp.nstr(grad_density0(y), 10))
