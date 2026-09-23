"""Scalar classes for the generic station DAG.
Jet2: forward-mode value+gradient+Hessian jet over mpf (order-2 Taylor jet in y).
Jsup: global sup-bound arithmetic carrying (M0,M1,M2,M3) = sup bounds of
|f|, |grad f|, |D2 f| (spectral), |D3 f| (as trilinear form) over the domain
seeded by the inputs. All rules are mechanical over-estimates: sound by
construction (no cancellation hazards, bounds only grow).
"""
from mpmath import mp, mpf
from fractions import Fraction as _Fr

def tomp(x):
    if type(x) is mpf:
        return x
    if isinstance(x, _Fr):
        return mpf(x.numerator) / mpf(x.denominator)
    return mpf(x)

# ---------------- Jet2 (mpf base) ----------------
class J2:
    __slots__ = ('v', 'gx', 'gy', 'hxx', 'hxy', 'hyy')
    def __init__(self, v, gx=0, gy=0, hxx=0, hxy=0, hyy=0):
        self.v = tomp(v); self.gx = tomp(gx); self.gy = tomp(gy)
        self.hxx = tomp(hxx); self.hxy = tomp(hxy); self.hyy = tomp(hyy)
    @classmethod
    def _raw(cls, v, gx, gy, hxx, hxy, hyy):
        o = cls.__new__(cls)
        o.v = v; o.gx = gx; o.gy = gy; o.hxx = hxx; o.hxy = hxy; o.hyy = hyy
        return o
    def __add__(a, b):
        if type(b) is J2:
            return J2._raw(a.v + b.v, a.gx + b.gx, a.gy + b.gy,
                           a.hxx + b.hxx, a.hxy + b.hxy, a.hyy + b.hyy)
        b = _j2(b)
        return J2._raw(a.v + b.v, a.gx + b.gx, a.gy + b.gy,
                       a.hxx + b.hxx, a.hxy + b.hxy, a.hyy + b.hyy)
    __radd__ = __add__
    def __neg__(a):
        return J2(-a.v, -a.gx, -a.gy, -a.hxx, -a.hxy, -a.hyy)
    def __sub__(a, b):
        return a + (-_j2(b))
    def __rsub__(a, b):
        return _j2(b) + (-a)
    def __mul__(a, b):
        if type(b) is not J2:
            b = _j2(b)
        av, bv = a.v, b.v
        agx, bgx = a.gx, b.gx
        agy, bgy = a.gy, b.gy
        return J2._raw(av * bv,
                       agx * bv + av * bgx,
                       agy * bv + av * bgy,
                       a.hxx * bv + 2 * agx * bgx + av * b.hxx,
                       a.hxy * bv + agx * bgy + agy * bgx + av * b.hxy,
                       a.hyy * bv + 2 * agy * bgy + av * b.hyy)
    __rmul__ = __mul__
    def inv(a):
        q = 1 / a.v
        gx = -a.gx * q * q; gy = -a.gy * q * q
        hxx = 2 * a.gx * a.gx * q ** 3 - a.hxx * q * q
        hxy = 2 * a.gx * a.gy * q ** 3 - a.hxy * q * q
        hyy = 2 * a.gy * a.gy * q ** 3 - a.hyy * q * q
        return J2._raw(q, gx, gy, hxx, hxy, hyy)
    def __truediv__(a, b):
        return a * _j2(b).inv()
    def __rtruediv__(a, b):
        return _j2(b) * a.inv()
    def __pow__(a, k):
        r = J2(1)
        for _ in range(int(k)):
            r = r * a
        return r

def _j2(x):
    return x if isinstance(x, J2) else J2(x)

def j2_const(x):
    return J2(x)

# ---------------- Jsup (global sup bounds) ----------------
class JS:
    """Carries (M0,M1,M2,M3): sup bounds of value, grad (euclidean),
    Hessian (spectral), third derivative (symmetric trilinear form sup).
    Domain is whatever the seeds range over (zone or box)."""
    __slots__ = ('m0', 'm1', 'm2', 'm3')
    def __init__(self, m0, m1=0, m2=0, m3=0):
        self.m0 = tomp(m0); self.m1 = tomp(m1); self.m2 = tomp(m2); self.m3 = tomp(m3)
    def __add__(a, b):
        b = _js(b)
        return JS(a.m0 + b.m0, a.m1 + b.m1, a.m2 + b.m2, a.m3 + b.m3)
    __radd__ = __add__
    def __neg__(a):
        return JS(a.m0, a.m1, a.m2, a.m3)
    def __sub__(a, b):
        return a + (-_js(b))
    def __rsub__(a, b):
        return _js(b) + (-a)
    def __mul__(a, b):
        b = _js(b)
        return JS(a.m0 * b.m0,
                  a.m1 * b.m0 + a.m0 * b.m1,
                  a.m2 * b.m0 + 2 * a.m1 * b.m1 + a.m0 * b.m2,
                  a.m3 * b.m0 + 3 * a.m2 * b.m1 + 3 * a.m1 * b.m2 + a.m0 * b.m3)
    __rmul__ = __mul__
    def inv(a, m_min):
        """m_min: certified lower bound of |value| on the domain."""
        q = 1 / m_min
        return JS(q,
                  a.m1 * q ** 2,
                  2 * a.m1 ** 2 * q ** 3 + a.m2 * q ** 2,
                  6 * a.m1 ** 3 * q ** 4 + 6 * a.m1 * a.m2 * q ** 3 + a.m3 * q ** 2)
    def div(a, b, m_min):
        return a * _js(b).inv(m_min)
    def __truediv__(a, b):
        if isinstance(b, JS):
            raise TypeError('JS division by JS requires certified min: use .div(b, m_min)')
        return JS(a.m0 / b, a.m1 / b, a.m2 / b, a.m3 / b)
    def __pow__(a, k):
        k = int(k)
        if k == 0:
            return JS(1, 0, 0, 0)
        # |f^k|' <= k|f|^{k-1}|f'|; iterate product rule for safety
        r = JS(1, 0, 0, 0)
        for _ in range(k):
            r = r * a
        return r

def _js(x):
    return x if isinstance(x, JS) else JS(x, 0, 0, 0)

# transcendental sup rules (assembly stage)
def js_exp_neg_half(q):
    """f = exp(-q/2) given JS q with q >= 0: M0<=1, chain rules."""
    return JS(1, q.m1 / 2, (q.m1 ** 2 + 2 * q.m2) / 4,
              (q.m1 ** 3 + 6 * q.m1 * q.m2 + 8 * q.m3) / 8)

def js_sqrt(t, t_min):
    """sqrt(t), t >= t_min > 0."""
    import math
    s0 = 1 / mp.sqrt(t_min)          # sup of 1/sqrt(t) <= 1/sqrt(t_min)... sqrt itself <= sqrt(t.m0)
    return JS(mp.sqrt(t.m0),
              t.m1 * s0 / 2,
              (t.m1 ** 2 * s0 ** 3 / 2 + t.m2 * s0) / 2,
              (3 * t.m1 ** 3 * s0 ** 5 / 4 + 3 * t.m1 * t.m2 * s0 ** 3 + t.m3 * s0) / 4)

PHI0 = 1 / mp.sqrt(2 * mp.pi)

def js_Phi_diff(z1, z2):
    """Pw = Phi(z1)-Phi(z2) with z1 >= z2: value in [0,1]; derivatives via
    phi=Phi' bounded by PHI0, |phi'|=|z phi| bounded by PHI0*max(1,|z|)... use
    global sup of |phi'| <= PHI0*e^{-1/2}*1 = PHI0/sqrt(e) (max of x e^{-x^2/2}).
    second: |phi''|=|(x^2-1)phi| <= PHI0*max|(x^2-1)e^{-x^2/2}| <= PHI0*1
    (max at x=0 gives 1; at x=sqrt3: 2e^{-1.5}=0.446) -> PHI0."""
    M1 = PHI0 * (z1.m1 + z2.m1)
    M2 = PHI0 * (z1.m2 + z2.m2) + (PHI0 / mp.e ** mpf('0.5')) * (z1.m1 ** 2 + z2.m1 ** 2)
    M3 = PHI0 * (z1.m3 + z2.m3) + PHI0 * (z1.m1 * z1.m2 + z2.m1 * z2.m2) \
         + PHI0 * (z1.m1 ** 3 + z2.m1 ** 3)
    return JS(1, M1, M2, M3)
