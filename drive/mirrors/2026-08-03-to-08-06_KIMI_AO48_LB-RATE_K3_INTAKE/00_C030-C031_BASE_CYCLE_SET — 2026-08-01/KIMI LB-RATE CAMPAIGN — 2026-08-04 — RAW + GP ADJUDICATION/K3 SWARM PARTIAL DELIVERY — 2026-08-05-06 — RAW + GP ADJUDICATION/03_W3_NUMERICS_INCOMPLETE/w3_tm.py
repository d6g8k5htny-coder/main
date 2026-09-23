# Taylor-model arithmetic layer: rho_box on polynomial models of (muJ, SigJ)
# so that Schur-complement cancellations (vt = det3/det2 etc.) cancel at the
# COEFFICIENT level, not via entry-wise intervals.
import math
from mpmath import iv, mpi
import mpmath as mp

import importlib.util
_spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3c)

ivm = iv.mpf
lo, hi, mid, width = w3c.lo, w3c.hi, w3c.mid, w3c.width
ck = w3c.ck
JET6 = w3c.JET6


def midx(order):
    out = []
    for tot in range(order + 1):
        for a in range(tot, -1, -1):
            out.append((a, tot - a))
    return out


class P:
    """truncated 2-var polynomial with interval coefficients: list over Gs"""
    __slots__ = ('co', 'Gs')

    def __init__(self, co, Gs):
        self.co = list(co)
        self.Gs = Gs

    @staticmethod
    def const(c, Gs):
        co = [ivm(0)] * len(Gs)
        co[0] = c
        return P(co, Gs)

    def __add__(self, o):
        return P([a + b for a, b in zip(self.co, o.co)], self.Gs)

    def __sub__(self, o):
        return P([a - b for a, b in zip(self.co, o.co)], self.Gs)

    def __mul__(self, o):
        if isinstance(o, P):
            N = max(g[0] + g[1] for g in self.Gs)
            idx = {g: k for k, g in enumerate(self.Gs)}
            co = [ivm(0)] * len(self.Gs)
            for i, gi in enumerate(self.Gs):
                ai = self.co[i]
                if float(hi(iv.fabs(ai))) == 0.0:
                    continue
                for j, gj in enumerate(self.Gs):
                    g = (gi[0] + gj[0], gi[1] + gj[1])
                    if g[0] + g[1] > N:
                        continue
                    co[idx[g]] = co[idx[g]] + ai * o.co[j]
            return P(co, self.Gs)
        return P([a * o for a in self.co], self.Gs)

    __rmul__ = __mul__

    def __neg__(self):
        return P([-a for a in self.co], self.Gs)

    def eval(self, dx, dy):
        v = ivm(0)
        for g, c in zip(self.Gs, self.co):
            v += c * dx ** g[0] * dy ** g[1]
        return v

    def eval_der(self, dx, dy, which):
        """derivative d/dx (which=0) or d/dy (which=1) evaluated"""
        v = ivm(0)
        for g, c in zip(self.Gs, self.co):
            if g[which] > 0:
                gg = list(g)
                e = gg[which]
                gg[which] -= 1
                v += c * ivm(e) * dx ** gg[0] * dy ** gg[1]
        return v


def poly_eval_bound(p, r1, r2):
    """sup of |p| over the symmetric box |dx|<=r1, |dy|<=r2"""
    s = ivm(0)
    for g, c in zip(p.Gs, p.co):
        s += iv.fabs(c) * ivm(r1) ** g[0] * ivm(r2) ** g[1]
    return s


class RhoTM:
    """all rho factors over a box, from TaylorLaw mu/sig coefficient lists."""
    def __init__(self, TL, ybox):
        Gs = TL.Gs
        self.Gs = Gs
        (x0, x1), (y0, y1) = (lo(ybox[0]), hi(ybox[0])), (lo(ybox[1]), hi(ybox[1]))
        xc, yc = float(mid(TL.y_c[0])), float(mid(TL.y_c[1]))
        self.dx = mpi(x0 - xc, x1 - xc)
        self.dy = mpi(y0 - yc, y1 - yc)
        self.r1 = max(abs(x0 - xc), abs(x1 - xc))
        self.r2 = max(abs(y0 - yc), abs(y1 - yc))
        self.N = TL.N
        Ns = [g for g in midx(self.N + 1) if g[0] + g[1] == self.N + 1]
        # mu/sig polys + remainders
        self.mu = [P(TL.mu_co[i], Gs) for i in range(6)]
        self.sig = [[P(TL.sig_co[i * 6 + j], Gs) for j in range(6)] for i in range(6)]
        self.Rmu = ivm(0)
        for g in Ns:
            self.Rmu += max((TL.mu_rem(i, g) for i in range(6)),
                            key=lambda q: float(hi(q))) * \
                ivm(self.r1) ** g[0] * ivm(self.r2) ** g[1] / ivm(math.factorial(g[0]) * math.factorial(g[1]))
        self.Rsig = ivm(0)
        for g in Ns:
            mg = ivm(0)
            for i in range(6):
                for j in range(6):
                    s = TL.sig_rem(i, j, g)
                    if lo(mg) < lo(s):
                        mg = s
            self.Rsig += mg * ivm(self.r1) ** g[0] * ivm(self.r2) ** g[1] / \
                ivm(math.factorial(g[0]) * math.factorial(g[1]))
        # sup bounds for remainder propagation
        self.Smax = ivm(0)
        for i in range(6):
            for j in range(6):
                b = poly_eval_bound(self.sig[i][j], self.r1, self.r2)
                if lo(self.Smax) < lo(b):
                    self.Smax = b
        self.Mmax = ivm(0)
        for i in range(6):
            b = poly_eval_bound(self.mu[i], self.r1, self.r2)
            if lo(self.Mmax) < lo(b):
                self.Mmax = b

    def ev(self, p):
        return p.eval(self.dx, self.dy)

    def build(self):
        """returns dict of interval factors over the box"""
        Gs = self.Gs
        Sg11, Sg12, Sg22 = self.sig[1][1], self.sig[1][2], self.sig[2][2]
        det2 = Sg11 * Sg22 - Sg12 * Sg12
        Sf00, Sf01, Sf02 = self.sig[0][0], self.sig[0][1], self.sig[0][2]
        det3 = (Sf00 * det2 - Sf01 * (Sf01 * Sg22 - Sf02 * Sg12)
                + Sf02 * (Sf01 * Sg12 - Sf02 * Sg11))
        # remainder bounds: |det2_true - det2_poly| <= 2 Smax Rsig + Rsig^2
        Rs = self.Rsig
        Rdet2 = 2 * self.Smax * Rs + Rs * Rs
        Adj2max = self.Smax  # each adj entry is a sig entry
        # det3 remainder: 3*adj*Rs + 3*Smax*Rs^2 + Rs^3, adj entries = 2x2 minors <= 2 Smax^2
        adjmax = 2 * self.Smax * self.Smax
        Rdet3 = 3 * adjmax * Rs + 3 * self.Smax * Rs * Rs + Rs ** 3
        det2v = self.ev(det2) + mpi(-hi(Rdet2), hi(Rdet2))
        det3v = self.ev(det3) + mpi(-hi(Rdet3), hi(Rdet3))
        ck(lo(det2v) > 0, "TM det2>0")
        ck(lo(det3v) > 0, "TM det3>0")
        vt = det3v / det2v
        ck(lo(vt) > 0, "TM vt>0")
        # mu remainder
        Rm = self.Rmu
        mu1v = self.ev(self.mu[1]) + mpi(-hi(Rm), hi(Rm))
        mu2v = self.ev(self.mu[2]) + mpi(-hi(Rm), hi(Rm))
        mu0v = self.ev(self.mu[0]) + mpi(-hi(Rm), hi(Rm))
        # chi2 = mug' adj2 mug / det2; adj2 poly entries: (Sg22, -Sg12; -Sg12, Sg11)
        a1 = Sg22 * self.mu[1] - Sg12 * self.mu[2]
        a2 = -Sg12 * self.mu[1] + Sg11 * self.mu[2]
        chin = self.mu[1] * a1 + self.mu[2] * a2
        # remainder: chin_true error <= |d(mug' adj2 mug)|: propagate via sup bounds:
        # terms: 2*|adj2*mug|*Rm + 2*|mug*mug|*Rs + higher; bound with sup polys:
        a1max = poly_eval_bound(a1, self.r1, self.r2)
        a2max = poly_eval_bound(a2, self.r1, self.r2)
        Rchin = 2 * (a1max + a2max) * Rm + 2 * self.Mmax * self.Mmax * Rs + \
            4 * self.Mmax * Rm * Rs + 2 * Rm * Rm * Rs
        chiv = (self.ev(chin) + mpi(-hi(Rchin), hi(Rchin))) / det2v
        if lo(chiv) < 0:
            chiv = mpi(0, hi(chiv))
        pgrad = iv.exp(-chiv / 2) / (2 * iv.pi * iv.sqrt(det2v))
        # mt = mu0 - Sfg adj2 mug / det2 = (mu0 det2 - Sfg adj2 mug)/det2
        sf = Sf01 * a1 + Sf02 * a2
        # remainder for sf ~ 2 terms like chin
        sfmax = poly_eval_bound(sf, self.r1, self.r2)
        Rsf = (poly_eval_bound(Sf01, self.r1, self.r2) + poly_eval_bound(Sf02, self.r1, self.r2)) * \
            (2 * self.Mmax * Rs + Rm * self.Smax + Rm * Rs) + sfmax * 0 + \
            (a1max + a2max) * Rs * 2
        mt = (self.ev(self.mu[0] * det2 - sf) + mpi(-hi(Rsf + Rm * (self.Smax * self.Smax) + Rm), hi(Rsf + Rm * self.Smax * self.Smax + Rm))) / det2v
        out = dict(vt=vt, mt=mt, pgrad=pgrad, det2=det2v, det3=det3v, chi2=chiv,
                   mu=(mu0v, mu1v, mu2v), sig_poly=self.sig, TL=None)
        # Hessian conditional law: V = SHH - SHfg adj3 SHfg^T / det3
        Sfgm = [[self.sig[3 + i][0], self.sig[3 + i][1], self.sig[3 + i][2]] for i in range(3)]
        SHH = [[self.sig[3 + i][3 + j] for j in range(3)] for i in range(3)]
        # adj3 poly entries (cofactors of the (f,g) block)
        A00 = det2
        A01 = -(Sf01 * Sg22 - Sf02 * Sg12)
        A02 = Sf01 * Sg12 - Sf02 * Sg11
        A11 = Sf00 * Sg22 - Sf02 * Sf02
        A12 = -(Sf00 * Sg12 - Sf02 * Sf01)
        A22 = Sf00 * Sg11 - Sf01 * Sf01
        ADJ = [[A00, A01, A02], [A01, A11, A12], [A02, A12, A22]]
        # W3num[i][k] = sum_l Sfgm[i][l] ADJ[l][k]; V = SHH - W3num*Sfgm^T/det3
        W3num = [[sum((Sfgm[i][l] * ADJ[l][k] for l in range(3)), P.const(ivm(0), self.Gs)) for k in range(3)] for i in range(3)]
        out['ADJ'] = ADJ
        out['Sfgm'] = Sfgm
        out['W3num'] = W3num
        out['SHH'] = SHH
        out['det3p'] = det3
        out['Rdet3'] = Rdet3
        out['det2p'] = det2
        out['Rdet2'] = Rdet2
        out['Rsf'] = Rsf
        return out

    def evp(self, p):
        return p.eval(self.dx, self.dy)
