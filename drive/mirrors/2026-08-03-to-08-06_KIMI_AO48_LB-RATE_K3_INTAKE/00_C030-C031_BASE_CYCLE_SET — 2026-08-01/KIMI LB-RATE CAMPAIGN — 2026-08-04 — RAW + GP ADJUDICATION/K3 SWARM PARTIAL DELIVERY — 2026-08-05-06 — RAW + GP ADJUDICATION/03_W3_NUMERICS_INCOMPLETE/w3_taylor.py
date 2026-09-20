# Taylor-law layer: certified conditional-law Taylor models at a center y_c.
# All quantities are exact Gaussian-regression outputs; y-dependence enters via
#   Vt_{ij}(y) = Cov(d^{a_i} f(y), T_j),   T = C L (whitened pins, Var T = I)
# whose Taylor coefficients at y_c are point-exact intervals, with global
# Cauchy-Schwarz remainder bounds |d^g Vt_{ij}| <= sqrt(K^{(2a_i+2g)}(0)) * 1.
import math
from mpmath import iv, mpi
import mpmath as mp

import importlib.util
_spec = importlib.util.spec_from_file_location("w3c", "/mnt/agents/output/K3_SIDE24_LB/W3_numerics/w3_cert.py")
w3c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w3c)

def share_cache(w3c_ext):
    """share kernel cache with an already-initialized w3_cert instance"""
    global w3c
    w3c._CACHE.update(w3c_ext._CACHE)

ivm = iv.mpf
lo, hi, mid, width = w3c.lo, w3c.hi, w3c.mid, w3c.width
ck = w3c.ck

# multi-index helpers
def midx(order):
    out = []
    for tot in range(order + 1):
        for a in range(tot, -1, -1):
            out.append((a, tot - a))
    return out

def mfac(g):
    return math.factorial(g[0]) * math.factorial(g[1])

JET6 = w3c.JET6

def deriv_sup_bound(alpha, g):
    """sup_y |d^g Vt_{ij}| <= sqrt(K1^{(2(a1+g1))}(0) K1^{(2(a2+g2))}(0)) (CS, Var T_j = 1)"""
    return iv.sqrt(iv.fabs(w3c.K1w(2 * (alpha[0] + g[0]), ivm(0))) *
                   iv.fabs(w3c.K1w(2 * (alpha[1] + g[1]), ivm(0))))

class TaylorLaw:
    """certified Taylor models of muJ(y), SigJ(y) around y_c to order N."""
    def __init__(self, y_c, N, law, tv):
        self.y_c = y_c
        self.N = N
        self.law = law
        # gamma lattice for derivatives of Vt (order N) and products (order N)
        Gs = midx(N)
        self.Gs = Gs
        # cross-covariance derivatives at y_c: Ader[i][g][j] = Cov(d^{a_i+g}f(y_c), L_j)
        # then VTder[i][g] = (Ader C^T) row -> Cov(d^{a_i+g}f(y_c), T_j)  (6-vector? no: 9)
        C, CT, Ginv = law['C'], law['CT'], law['Ginv']
        lam = w3c.matvec_iv(Ginv, tv)          # 9
        self.lam = lam
        # certified ||Ginv||_inf (row-sum of absolute values)
        self.ginv_norm = max(float(hi(sum((iv.fabs(Ginv[k][l]) for l in range(9)), ivm(0)))) for k in range(9))
        # extended alpha list: a_i + g for i in 6-jet, |g| <= N
        # compute A_ext rows lazily per (i, g)
        # VTder[i][g] = sum_k C[j][k] * Cov(d^{a_i+g} f(y_c), L_k)   (over j: 9 values)
        VTder = []
        for i in range(6):
            ai = JET6[i]
            row_g = []
            for g in Gs:
                ag = (ai[0] + g[0], ai[1] + g[1])
                avec = [w3c.cov2c(ag, w3c.LFUNS[k][1], y_c,
                                  (w3c.civ(w3c.LFUNS[k][0][0]), w3c.civ(w3c.LFUNS[k][0][1])))
                        for k in range(9)]
                row_g.append(w3c.matvec_iv(C, avec))
            VTder.append(row_g)
        self.VTder = VTder
        # B constants: B_ij = Cov(d^{a_i}f, d^{a_j}f)(any point) = (-1)^{|a_j|} d^{a_i+a_j} K(0)
        zero = (ivm(0), ivm(0))
        self.B = [[w3c.cov2c(JET6[i], JET6[j], zero, zero) for j in range(6)] for i in range(6)]
        # muJ Taylor coefficients: mu_i^(g)(y_c) = VTder[i][g] . lam
        self.mu_co = [[sum((VTder[i][gi][k] * lam[k] for k in range(9)), ivm(0)) for gi in range(len(Gs))]
                      for i in range(6)]
        # SigJ Taylor coefficients at center via Leibniz:
        # d^g SigJ_ij = - sum_{d<=g} C(g,d) (VT_i^(d) Ginv VT_j^(g-d)^T)   (B const)
        self.sig_co = [[None] * len(Gs) for _ in range(36)]
        for i in range(6):
            for j in range(6):
                co = []
                for gi, g in enumerate(Gs):
                    # sum over d <= g
                    s = ivm(0)
                    for di, d in enumerate(Gs):
                        if d[0] > g[0] or d[1] > g[1]:
                            continue
                        e = (g[0] - d[0], g[1] - d[1])
                        ei = Gs.index(e)
                        binom = math.comb(g[0], d[0]) * math.comb(g[1], d[1])
                        # VT_i^(d) Ginv VT_j^(e)^T
                        vd = VTder[i][di]
                        ve = VTder[j][ei]
                        Mv = w3c.matvec_iv(Ginv, ve)
                        s += ivm(binom) * sum((vd[k] * Mv[k] for k in range(9)), ivm(0))
                    co.append(-s)
                self.sig_co[i * 6 + j] = co
        # set g=0 coefficient to B_ij - product (same formula gives -(product); add B)
        for i in range(6):
            for j in range(6):
                self.sig_co[i * 6 + j][0] = self.B[i][j] + self.sig_co[i * 6 + j][0]
        # global remainder bounds for SigJ coefficients of order N+1:
        # |d^g SigJ_ij| <= sum_{d<=g} C(g,d)*2*|Ginv| * S(i,d) S(j,g-d), S(i,d)=sqrt(K^{2(a_i+d)}(0))
        self._sig_rem_const = {}
        self._mu_rem_const = {}

    def sig_rem(self, i, j, g):
        key = (i, j, g)
        if key in self._sig_rem_const:
            return self._sig_rem_const[key]
        s = ivm(0)
        for d in midx(g[0] + g[1]):
            if d[0] > g[0] or d[1] > g[1]:
                continue
            e = (g[0] - d[0], g[1] - d[1])
            binom = math.comb(g[0], d[0]) * math.comb(g[1], d[1])
            s += ivm(binom) * ivm(self.ginv_norm) * deriv_sup_bound(JET6[i], d) * deriv_sup_bound(JET6[j], e)
        self._sig_rem_const[key] = s
        return s

    def mu_rem(self, i, g):
        key = (i, g)
        if key not in self._mu_rem_const:
            self._mu_rem_const[key] = deriv_sup_bound(JET6[i], g) * self.lam_norm
        return self._mu_rem_const[key]

    @property
    def lam_norm(self):
        if not hasattr(self, '_lam_norm'):
            self._lam_norm = iv.sqrt(sum((self.lam[k] ** 2 for k in range(9)), ivm(0)))
        return self._lam_norm

    def eval_box(self, ybox):
        """returns muJ[6], SigJ[6][6] intervals valid over ybox = ((x0,x1),(y0,y1))"""
        N = self.N
        (x0, x1), (y0, y1) = (lo(ybox[0]), hi(ybox[0])), (lo(ybox[1]), hi(ybox[1]))
        xc, yc = float(mid(self.y_c[0])), float(mid(self.y_c[1]))
        dx = mpi(x0 - xc, x1 - xc)
        dy = mpi(y0 - yc, y1 - yc)
        r1 = max(abs(x0 - xc), abs(x1 - xc))
        r2 = max(abs(y0 - yc), abs(y1 - yc))
        Gs = self.Gs
        Ns = [g for g in midx(N + 1) if g[0] + g[1] == N + 1]
        muJ = []
        for i in range(6):
            v = ivm(0)
            for gi, g in enumerate(Gs):
                v += self.mu_co[i][gi] * dx ** g[0] * dy ** g[1] / ivm(mfac(g))
            rem = ivm(0)
            for g in Ns:
                rem += self.mu_rem(i, g) * ivm(r1) ** g[0] * ivm(r2) ** g[1] / ivm(mfac(g))
            muJ.append(v + mpi(-hi(rem), hi(rem)))
        SigJ = [[None] * 6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                co = self.sig_co[i * 6 + j]
                v = ivm(0)
                for gi, g in enumerate(Gs):
                    v += co[gi] * dx ** g[0] * dy ** g[1] / ivm(mfac(g))
                rem = ivm(0)
                for g in Ns:
                    rem += self.sig_rem(i, j, g) * ivm(r1) ** g[0] * ivm(r2) ** g[1] / ivm(mfac(g))
                SigJ[i][j] = v + mpi(-hi(rem), hi(rem))
        return muJ, SigJ
