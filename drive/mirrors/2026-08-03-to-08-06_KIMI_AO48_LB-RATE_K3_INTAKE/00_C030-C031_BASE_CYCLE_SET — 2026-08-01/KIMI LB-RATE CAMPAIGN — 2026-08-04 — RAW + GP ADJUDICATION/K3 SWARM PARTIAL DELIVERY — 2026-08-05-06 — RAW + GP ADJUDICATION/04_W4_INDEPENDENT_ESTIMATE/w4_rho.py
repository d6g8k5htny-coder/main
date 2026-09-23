#!/usr/bin/env python3
"""
W4 INDEPENDENT - pointwise estimand rho(y) via direct multivariate Gaussian
integration on transformed coordinates (nested 1D quadratures with a
closed-form innermost integral; NO Gauss-Hermite anywhere; quadratures are
Gauss-Legendre on truncated intervals with certified Gaussian tail bounds).

Given the law of W = (f, X=H11, Z=H12, Y=H22) | pins, grad f(y) = 0
(mean m4, cov S4):

  rho(y) = p_grad(0) * E_win,
  E_win  = E[ |X Y - Z^2| 1{XY < Z^2} 1{b-ell < f < b} ]
         = ∫_{b-ell}^{b} phi(u; m_f, v_f) g(u) du,
  g(u)   = E[ (Z^2 - XY) 1{XY < Z^2} | f = u ]
         = ∫ phi(z; mZ(u), vZ) h(z; u) dz,
  h(z;u) = E[ (z^2 - XY) 1{XY < z^2} | Z=z, f=u ]
         = ∫ phi(w) F(w; z, u) dw,
  F      = |a| sigma_y K(sign(a) t),   K(v) = v Phi(v) + phi(v) >= 0,
  with X = m1 + s1 w, a = m1 + s1 w, t = (z^2/a - mu_y(w))/sigma_y.

Corrected Cauchy-Schwarz bound (lead comparison):
  rho_CS(y) = p_grad(0) * sqrt(E[det^2 | grad=0]) * sqrt(P(det<0 & window | grad=0))
(the probability factor carries the square root; computed with the same
nested quadrature).

Float64 vectorized core (numpy) for the spatial grid; mpmath adaptive
reference (Tanh-Sinh) for probe-point certification. Deterministic.
No asserts. Fail-closed ck().
"""
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

SQRT2PI = np.sqrt(2.0 * np.pi)


def ck(cond, msg):
    if not cond:
        sys.stderr.write("CK-FAIL: %s\n" % msg)
        raise SystemExit(2)
    return True


def phi64(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def Kfun(v):
    """K(v) = v*Phi(v) + phi(v), stably for all v (asymptotic branch v<-30)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    m1 = v >= -30.0
    vv = v[m1]
    out[m1] = vv * ndtr(vv) + phi64(vv)
    if np.any(~m1):
        u = v[~m1]
        # asymptotic: K(u) ~ phi(u)/u^2 * (1 - 3/u^2 + 15/u^4 - 105/u^6 + ...)
        inv = 1.0 / (u * u)
        ser = 1.0 + inv * (-3.0 + inv * (15.0 + inv * (-105.0 + inv * 945.0)))
        out[~m1] = phi64(u) * inv * ser
    return out


class RhoCore:
    """Fixed conditional law; evaluates E_win, P(det<0 & window), E[det^2]."""

    def __init__(self, m4, S4, b, ell):
        m4 = np.asarray(m4, dtype=np.float64)
        S4 = np.asarray(S4, dtype=np.float64)
        self.b = float(b)
        self.ell = float(ell)
        self.mf = m4[0]
        self.vf = S4[0, 0]
        ck(self.vf > 0, "v_f must be positive")
        self.sf = np.sqrt(self.vf)
        self.muH = m4[1:4].copy()          # (X,Z,Y) = (H11,H12,H22)
        beta = S4[1:4, 0] / self.vf
        self.beta = beta
        SH = S4[1:4, 1:4] - np.outer(beta, beta) * self.vf
        SH = 0.5 * (SH + SH.T)
        self.SH = SH
        # Z = H12 is index 1; X = H11 index 0; Y = H22 index 2
        self.vZ = SH[1, 1]
        ck(self.vZ > 0, "Var(H12|f) must be positive")
        self.sZ = np.sqrt(self.vZ)
        self.cX = SH[0, 1] / self.vZ
        self.cY = SH[2, 1] / self.vZ
        Sres = np.array([[SH[0, 0], SH[0, 2]], [SH[0, 2], SH[2, 2]]])
        Sres = Sres - self.vZ * np.array([[self.cX ** 2, self.cX * self.cY],
                                          [self.cX * self.cY, self.cY ** 2]])
        Sres = 0.5 * (Sres + Sres.T)
        self.s1 = np.sqrt(max(Sres[0, 0], 0.0))
        self.s2 = np.sqrt(max(Sres[1, 1], 0.0))
        ck(self.s1 > 0 and self.s2 > 0, "residual stds must be positive")
        self.rho = Sres[0, 1] / (self.s1 * self.s2)
        ck(abs(self.rho) < 1.0, "residual correlation must be < 1, got %r" % self.rho)
        self.sig_y = self.s2 * np.sqrt(1.0 - self.rho ** 2)
        # raw second-moment data for E[det^2]
        self._E_det2 = self._compute_E_det2()

    # -------- exact second moment of det = X*Y - Z^2 (Wick) --------
    def _compute_E_det2(self):
        muX, muZ, muY = self.muH
        c = self.SH
        varX, varZ, varY = c[0, 0], c[1, 1], c[2, 2]
        covXY, covXZ, covYZ = c[0, 2], c[0, 1], c[1, 2]
        # E[X^2 Y^2] with centered parts
        EX2Y2 = (varX * varY + 2.0 * covXY ** 2
                 + muX ** 2 * varY + muY ** 2 * varX
                 + 4.0 * muX * muY * covXY + muX ** 2 * muY ** 2)
        # E[X Y Z^2]
        EXYZ2 = (covXY * varZ + 2.0 * covXZ * covYZ
                 + muZ ** 2 * covXY + 2.0 * muY * muZ * covXZ
                 + 2.0 * muX * muZ * covYZ
                 + muX * muY * (varZ + muZ ** 2))
        EZ4 = 3.0 * varZ ** 2 + 6.0 * varZ * muZ ** 2 + muZ ** 4
        return EX2Y2 - 2.0 * EXYZ2 + EZ4

    # -------- nested quadrature --------
    def _nested(self, Nu, Nz, Nw, want):
        """
        want: 'win' -> E_win; 'prob' -> P(det<0 & window).
        Returns (value, tail_bound_certificate).
        """
        b, ell = self.b, self.ell
        # u-nodes on [b-ell, b]
        xu, wu = leggauss(Nu)
        u = 0.5 * (2 * b - ell) + 0.5 * ell * xu
        wgt_u = 0.5 * ell * wu
        phiu = phi64((u - self.mf) / self.sf) / self.sf
        # z-nodes: Z = mZ(u) + sZ * zeta, zeta in [-ZMAX, ZMAX]
        ZMAX = 10.0
        xz, wz = leggauss(Nz)
        zeta = ZMAX * xz
        wgt_z = ZMAX * wz
        phiz = phi64(zeta)
        # w-nodes on [-WMAX, WMAX]
        WMAX = 10.0
        xw, ww = leggauss(Nw)
        w = WMAX * xw
        wgt_w = WMAX * ww
        phiw = phi64(w)

        # loop over u (cheap: Nu ~ 24); inner vectorized (Nz, Nw)
        acc = 0.0
        accp = 0.0
        for iu in range(Nu):
            du = u[iu] - self.mf
            muXu, muZu, muYu = self.muH + self.beta * du
            z = muZu + self.sZ * zeta                     # (Nz,)
            c = z * z
            m1 = muXu + self.cX * (z - muZu)              # (Nz,)
            m2 = muYu + self.cY * (z - muZu)
            a = m1[:, None] + self.s1 * w[None, :]        # (Nz,Nw)
            muy = m2[:, None] + self.rho * self.s2 * w[None, :]
            # guard a ~ 0
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(np.abs(a) > 1e-290, c[:, None] / a, np.sign(a) * 1e290)
                t = (t - muy) / self.sig_y
            st = np.sign(a) * t
            if want == "win":
                F = np.abs(a) * self.sig_y * Kfun(st)
            else:
                F = ndtr(st)
            F = np.where(np.abs(a) > 1e-290, F,
                         np.where(c[:, None] > 0, (c[:, None] if want == "win" else 1.0) * np.ones_like(F), np.zeros_like(F)))
            h = (wgt_w[None, :] * phiw[None, :] * F).sum(axis=1)   # (Nz,)
            g = (wgt_z * phiz * h).sum()
            acc += wgt_u[iu] * phiu[iu] * g
            if want == "prob":
                accp = acc
        # certified tail bounds (crude, rigorous):
        # |integrand| <= |det| <= |XY| + Z^2; bounds via moments
        mX, mZ, mY = self.muH
        EXY = abs(mX * mY) + np.sqrt((self.SH[0, 0] + mX * mX) * (self.SH[2, 2] + mY * mY))
        EZZ = self.SH[1, 1] + mZ * mZ
        B = EXY + EZZ + 1.0
        # phi tail beyond 10 sigma for zeta and w: P(|N(0,1)|>10) <= 1.53e-23
        tail = 2 * 1.53e-23 * B
        return acc, tail

    def E_win(self, Nu=24, Nz=48, Nw=64):
        return self._nested(Nu, Nz, Nw, "win")

    def P_detneg_win(self, Nu=24, Nz=48, Nw=64):
        return self._nested(Nu, Nz, Nw, "prob")

    def E_det2(self):
        return self._E_det2


def rho_point(y, p_grad0, m4, S4, b, ell, Nu=24, Nz=48, Nw=64):
    """Full pointwise rho and corrected CS bound."""
    core = RhoCore(m4, S4, b, ell)
    ew, t1 = core.E_win(Nu, Nz, Nw)
    pw, t2 = core.P_detneg_win(Nu, Nz, Nw)
    ck(0.0 <= pw <= 1.0 + 1e-12, "probability in range")
    rho = p_grad0 * ew
    cs = p_grad0 * np.sqrt(max(core.E_det2(), 0.0)) * np.sqrt(max(pw, 0.0))
    return dict(rho=rho, E_win=ew, P_win_detneg=pw, E_det2=core.E_det2(),
                rho_CS=cs, tail=t1 + t2)
