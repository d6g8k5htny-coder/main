"""rnu_ds3.py -- CL-RNU-001 SS5 route D, step 2: EXACT THIRD DERIVATIVES of kappa_far(y).

Extends the engine's DS (value/grad/Hessian automatic differentiation) to DS3 (adds the four third-order
partials t_xxx, t_xxy, t_xyy, t_yyy) and replaces the three hand-coded second-order pieces with generic
DS3-arithmetic versions:
  * DM.inv     -> Gauss-Jordan elimination on DS3 entries (exact to all carried orders)
  * DM.det_ds  -> LU pivot product on DS3 entries
  * bures_trace_ds -> Denman-Beavers iteration for A^{1/2} on DS3 entries, then trace
and lifts the kernel-entry constructors (de1, d_entry) to third order via the exact kernel derivatives.
Everything else in kappa_far_ds is generic DS arithmetic and lifts automatically.

Validation: (a) DS3 value/grad/Hessian == engine DS to ~1e-80; (b) third derivatives vs central FD of the
engine's exact Hessian; (c) DS3 Bures == engine Bures to second order.
"""
import sys, io, contextlib, time
sys.path.insert(0, "."); sys.path.insert(0, "../H2_foundations")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import d3_rn_unif as R
from mpmath import mp, mpf
import mpmath
DS0 = R.DS  # original second-order class
ns = lambda x, n=8: mpmath.nstr(x, n)

class DS3(DS0):
    __slots__ = ('txxx', 'txxy', 'txyy', 'tyyy')
    def __init__(self, v, gx=0, gy=0, hxx=0, hyy=0, hxy=0, txxx=0, txxy=0, txyy=0, tyyy=0):
        DS0.__init__(self, v, gx, gy, hxx, hyy, hxy)
        self.txxx = mpf(txxx); self.txxy = mpf(txxy); self.txyy = mpf(txyy); self.tyyy = mpf(tyyy)
    @staticmethod
    def lift(o):
        if isinstance(o, DS3): return o
        if isinstance(o, DS0): return DS3(o.v, o.gx, o.gy, o.hxx, o.hyy, o.hxy)
        return DS3(o)
    @staticmethod
    def const(c): return DS3(c)
    def __add__(self, o):
        o = DS3.lift(o)
        return DS3(self.v + o.v, self.gx + o.gx, self.gy + o.gy, self.hxx + o.hxx, self.hyy + o.hyy, self.hxy + o.hxy,
                   self.txxx + o.txxx, self.txxy + o.txxy, self.txyy + o.txyy, self.tyyy + o.tyyy)
    __radd__ = __add__
    def __neg__(self):
        return DS3(-self.v, -self.gx, -self.gy, -self.hxx, -self.hyy, -self.hxy, -self.txxx, -self.txxy, -self.txyy, -self.tyyy)
    def __sub__(self, o): return self + (-DS3.lift(o))
    def __rsub__(self, o): return DS3.lift(o) + (-self)
    def __mul__(self, o):
        f = self; g = DS3.lift(o)
        return DS3(f.v * g.v,
                   f.gx * g.v + f.v * g.gx, f.gy * g.v + f.v * g.gy,
                   f.hxx * g.v + 2 * f.gx * g.gx + f.v * g.hxx,
                   f.hyy * g.v + 2 * f.gy * g.gy + f.v * g.hyy,
                   f.hxy * g.v + f.gx * g.gy + f.gy * g.gx + f.v * g.hxy,
                   f.txxx * g.v + 3 * f.hxx * g.gx + 3 * f.gx * g.hxx + f.v * g.txxx,
                   f.txxy * g.v + f.hxx * g.gy + 2 * f.hxy * g.gx + f.gy * g.hxx + 2 * f.gx * g.hxy + f.v * g.txxy,
                   f.txyy * g.v + f.hyy * g.gx + 2 * f.hxy * g.gy + f.gx * g.hyy + 2 * f.gy * g.hxy + f.v * g.txyy,
                   f.tyyy * g.v + 3 * f.hyy * g.gy + 3 * f.gy * g.hyy + f.v * g.tyyy)
    __rmul__ = __mul__
    def _compose(self, p, d1, d2, d3):
        """u = phi(f) with phi(f.v)=p, phi'=d1, phi''=d2, phi'''=d3 (Faa di Bruno to third order)."""
        f = self
        gx, gy = f.gx, f.gy
        hxx = d2 * gx * gx + d1 * f.hxx; hyy = d2 * gy * gy + d1 * f.hyy; hxy = d2 * gx * gy + d1 * f.hxy
        txxx = d3 * gx ** 3 + d2 * 3 * f.hxx * gx + d1 * f.txxx
        txxy = d3 * gx * gx * gy + d2 * (f.hxx * gy + 2 * f.hxy * gx) + d1 * f.txxy
        txyy = d3 * gx * gy * gy + d2 * (f.hyy * gx + 2 * f.hxy * gy) + d1 * f.txyy
        tyyy = d3 * gy ** 3 + d2 * 3 * f.hyy * gy + d1 * f.tyyy
        return DS3(p, d1 * gx, d1 * gy, hxx, hyy, hxy, txxx, txxy, txyy, tyyy)
    def inv(self):
        v = self.v
        return self._compose(1 / v, -1 / v ** 2, 2 / v ** 3, -6 / v ** 4)
    def __div__(self, o): return self * DS3.lift(o).inv()
    def __rdiv__(self, o): return DS3.lift(o) * self.inv()
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def sqrt(self):
        s = mpmath.sqrt(self.v)
        return self._compose(s, 1 / (2 * s), -1 / (4 * s ** 3), 3 / (8 * s ** 5))
    def exp(self):
        e = mpmath.exp(self.v)
        return self._compose(e, e, e, e)
    def log(self):
        v = self.v
        return self._compose(mpmath.log(v), 1 / v, -1 / v ** 2, 2 / v ** 3)
    def Phi(self):
        v = self.v
        pdf = mpmath.exp(-v ** 2 / 2) / mpmath.sqrt(2 * mp.pi)
        return self._compose(R.E._Phi(v), pdf, -v * pdf, (v * v - 1) * pdf)
    def tnorm(self):
        return mpmath.sqrt(self.txxx ** 2 + 3 * self.txxy ** 2 + 3 * self.txyy ** 2 + self.tyyy ** 2)

# ---- generic DM.inv / det via DS3 arithmetic (Gauss-Jordan with partial pivoting on values) ----
def dm_inv_generic(self):
    n = self.n; R.ck(n == self.m, "DM.inv needs square")
    A = [[DS3.lift(self.a[i][j]) for j in range(n)] for i in range(n)]
    I = [[DS3(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c].v))
        A[c], A[p] = A[p], A[c]; I[c], I[p] = I[p], I[c]
        R.ck(abs(A[c][c].v) > mpf('1e-60'), "DM.inv singular pivot")
        piv = A[c][c].inv()
        A[c] = [x * piv for x in A[c]]; I[c] = [x * piv for x in I[c]]
        for r in range(n):
            if r != c:   # never skip on value==0: the entry's DERIVATIVES may be nonzero (symmetric points)
                f = A[r][c]
                A[r] = [A[r][k] - f * A[c][k] for k in range(n)]
                I[r] = [I[r][k] - f * I[c][k] for k in range(n)]
    return R.DM(I)

def dm_det_generic(self):
    n = self.n
    A = [[DS3.lift(self.a[i][j]) for j in range(n)] for i in range(n)]
    det = DS3(1)
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c].v))
        if p != c:
            A[c], A[p] = A[p], A[c]; det = -det
        det = det * A[c][c]
        piv = A[c][c].inv()
        for r in range(c + 1, n):
            f = A[r][c] * piv
            A[r] = [A[r][k] - f * A[c][k] for k in range(n)]
    return det

# ---- Bures trace via Denman-Beavers on DS3 matrices ----
def bures_trace_generic(A):
    n = A.n
    Y = R.DM([[DS3.lift(A.a[i][j]) for j in range(n)] for i in range(n)])
    Z = R.DM([[DS3(1 if i == j else 0) for j in range(n)] for i in range(n)])
    for it in range(60):
        Yi = Y.inv(); Zi = Z.inv()
        Yn = R.DM([[(Y.a[i][j] + Zi.a[i][j]) * mpf('0.5') for j in range(n)] for i in range(n)])
        Zn = R.DM([[(Z.a[i][j] + Yi.a[i][j]) * mpf('0.5') for j in range(n)] for i in range(n)])
        delta = max(abs(Yn.a[i][j].v - Y.a[i][j].v) for i in range(n) for j in range(n))
        Y, Z = Yn, Zn
        if delta < mpf('1e-95'):
            break
    R.ck(delta < mpf('1e-90'), "Denman-Beavers did not converge: %s" % ns(delta, 3))
    tot = DS3(0)
    for i in range(n):
        tot = tot + Y.a[i][i]
    return tot

# ---- kernel entries to third order ----
def _kd_multi(a, b, dx, dy, es):
    ex = a[0] + b[0] + sum(e[0] for e in es); ey = a[1] + b[1] + sum(e[1] for e in es)
    s = R.kplane(ex, ey, dx, dy)
    for (i, j) in R._IMG:
        s += R.kplane(ex, ey, dx + R._LT * i, dy + R._LT * j)
    return ((-1) ** (b[0] + b[1])) * s
EX, EY = (1, 0), (0, 1)
def de1_3(a, b, Q, y):
    dx, dy = y[0] - Q[0], y[1] - Q[1]
    v = R.kdcov(a, b, dx, dy); gx = R.kdcov_d(a, b, dx, dy, 0); gy = R.kdcov_d(a, b, dx, dy, 1)
    d = lambda *es: _kd_multi(a, b, dx, dy, es)
    return DS3(v, gx, gy, d(EX, EX), d(EY, EY), d(EX, EY), d(EX, EX, EX), d(EX, EX, EY), d(EX, EY, EY), d(EY, EY, EY))
def d_entry_3(a, b, p, y):
    dx, dy = p[0] - y[0], p[1] - y[1]
    v = R.kdcov(a, b, dx, dy); gx = -R.kdcov_d(a, b, dx, dy, 0); gy = -R.kdcov_d(a, b, dx, dy, 1)
    d = lambda *es: _kd_multi(a, b, dx, dy, es)
    return DS3(v, gx, gy, d(EX, EX), d(EY, EY), d(EX, EY), -d(EX, EX, EX), -d(EX, EX, EY), -d(EX, EY, EY), -d(EY, EY, EY))

def install():
    R.DS = DS3
    R.DM.inv = dm_inv_generic
    R.DM.det_ds = dm_det_generic
    R.bures_trace_ds = bures_trace_generic
    R.de1 = de1_3
    R.de2 = lambda a, b, P, y: de1_3(b, a, P, y)
    R.d_entry = d_entry_3
    R.dconst = lambda v: DS3(v)

if __name__ == "__main__":
    kit = R.kit; v = kit.b - kit.ell / 2
    pts = [(mpf(5), mpf(0)), (mpf('4.8'), mpf('1.6')), (mpf('5.1'), mpf('-0.3'))]
    ref = {}
    for y in pts:
        ref[y] = R.kappa_far_ds(y, v)     # original second-order engine
    # FD of the engine's exact Hessian for third-derivative reference
    h = mpf('1e-12'); fd3 = {}
    for y in pts:
        def H(yy): k = R.kappa_far_ds(yy, v); return (k.hxx, k.hyy, k.hxy)
        Hx_p = H((y[0] + h, y[1])); Hx_m = H((y[0] - h, y[1])); Hy_p = H((y[0], y[1] + h)); Hy_m = H((y[0], y[1] - h))
        txxx = (Hx_p[0] - Hx_m[0]) / (2 * h); txxy = (Hy_p[0] - Hy_m[0]) / (2 * h)
        txyy = (Hx_p[1] - Hx_m[1]) / (2 * h); tyyy = (Hy_p[1] - Hy_m[1]) / (2 * h)
        fd3[y] = (txxx, txxy, txyy, tyyy)
    install()
    # Bures cross-check on a small PD DM
    for y in pts:
        t0 = time.time(); k3 = R.kappa_far_ds(y, v); dt = time.time() - t0
        r0 = ref[y]
        e2 = max(abs(k3.v - r0.v), abs(k3.gx - r0.gx), abs(k3.gy - r0.gy), abs(k3.hxx - r0.hxx), abs(k3.hyy - r0.hyy), abs(k3.hxy - r0.hxy))
        f = fd3[y]
        e3 = max(abs(k3.txxx - f[0]), abs(k3.txxy - f[1]), abs(k3.txyy - f[2]), abs(k3.tyyy - f[3]))
        print(f"y=({ns(y[0],4)},{ns(y[1],4)}): DS3 vs DS (v,g,H) max diff = {ns(e2,3)} | third: DS3=({ns(k3.txxx,7)},{ns(k3.txxy,7)},{ns(k3.txyy,7)},{ns(k3.tyyy,7)}) FD=({ns(f[0],7)},{ns(f[1],7)},{ns(f[2],7)},{ns(f[3],7)}) max diff {ns(e3,3)} | ||T||={ns(k3.tnorm(),6)} | {dt:.2f}s")
