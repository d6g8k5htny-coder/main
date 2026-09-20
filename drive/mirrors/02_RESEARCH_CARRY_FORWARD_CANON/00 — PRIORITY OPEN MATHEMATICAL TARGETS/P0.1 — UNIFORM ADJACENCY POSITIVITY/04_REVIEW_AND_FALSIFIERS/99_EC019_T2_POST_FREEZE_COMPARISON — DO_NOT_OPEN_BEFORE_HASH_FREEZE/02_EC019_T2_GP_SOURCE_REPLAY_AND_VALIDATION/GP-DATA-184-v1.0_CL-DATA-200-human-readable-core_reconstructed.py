from fractions import Fraction as Fr
from math import isqrt




class FullDimError(Exception):
    """Raised by the full-dimensionality validator before any dynamical margin."""
    pass




def _fr(x):
    if isinstance(x, IV):
        return x
    if isinstance(x, int):
        return IV(Fr(x), Fr(x))
    if isinstance(x, Fr):
        return IV(x, x)
    raise TypeError("cannot coerce %r to IV" % type(x))




class IV:
    """Closed rational interval [lo, hi] with lo <= hi (both fractions.Fraction)."""
    __slots__ = ("lo", "hi")


    def __init__(self, lo, hi):
        lo = Fr(lo)
        hi = Fr(hi)
        if lo > hi:
            raise ValueError("lo>hi: %s > %s" % (lo, hi))
        self.lo = lo
        self.hi = hi


    @staticmethod
    def cr(center, radius):
        c = Fr(center)
        rad = Fr(radius)
        return IV(c - rad, c + rad)


    # ---- arithmetic (exact) ----
    def __add__(self, o):
        o = _fr(o)
        return IV(self.lo + o.lo, self.hi + o.hi)


    __radd__ = __add__


    def __sub__(self, o):
        o = _fr(o)
        return IV(self.lo - o.hi, self.hi - o.lo)


    def __rsub__(self, o):
        return _fr(o).__sub__(self)


    def __neg__(self):
        return IV(-self.hi, -self.lo)


    def __mul__(self, o):
        o = _fr(o)
        ps = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return IV(min(ps), max(ps))


    __rmul__ = __mul__


    def recip(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("reciprocal of interval spanning zero")
        return IV(Fr(1) / self.hi, Fr(1) / self.lo)


    def __truediv__(self, o):
        return self * _fr(o).recip()


    def __rtruediv__(self, o):
        return _fr(o).__truediv__(self)


    # ---- powers (tight for monomials) ----
    def sq(self):
        a = self.lo * self.lo
        b = self.hi * self.hi
        if self.lo <= 0 <= self.hi:
            lo = Fr(0)
        else:
            lo = min(a, b)
        return IV(lo, max(a, b))


    def powi(self, n):
        if n < 0:
            raise ValueError("negative power")
        if n == 0:
            return IV(Fr(1), Fr(1))
        a = self.lo ** n
        b = self.hi ** n
        if n % 2 == 1:
            # x -> x^n monotonic increasing
            return IV(a, b)
        # even power: tight
        if self.lo <= 0 <= self.hi:
            lo = Fr(0)
        else:
            lo = min(a, b)
        return IV(lo, max(a, b))


    # ---- rigorous absolute value ----
    def absI(self):
        if self.lo >= 0:
            return IV(self.lo, self.hi)
        if self.hi <= 0:
            return IV(-self.hi, -self.lo)
        return IV(Fr(0), max(-self.lo, self.hi))


    def supabs(self):
        return max(abs(self.lo), abs(self.hi))  # returns Fraction


    def __repr__(self):
        return "IV(%s, %s)" % (self.lo, self.hi)




def cube(x):
    return x.powi(3)




# ---------- directed rational square root ----------
def sqrt_lo(x, S):
    """Rational L with L^2 <= x  (x = p/q >= 0). L = isqrt(p*q*S^2)/(q*S)."""
    if x < 0:
        raise ValueError("sqrt of negative")
    p = x.numerator
    q = x.denominator
    root = isqrt(p * q * S * S)
    return Fr(root, q * S)




def sqrt_hi(x, S):
    """Rational U with U^2 >  x  (x = p/q >= 0). U = (isqrt(p*q*S^2)+1)/(q*S)."""
    if x < 0:
        raise ValueError("sqrt of negative")
    p = x.numerator
    q = x.denominator
    root = isqrt(p * q * S * S) + 1
    return Fr(root, q * S)




def sqrtIV(x, S):
    """Interval sqrt over [x.lo, x.hi], x.lo >= 0."""
    return IV(sqrt_lo(x.lo, S), sqrt_hi(x.hi, S))




# ---------- parameter box ----------
# BASE = {name: (center, radius)} as exact Fractions (from GP-SPEC-166 Section 3).
BASE = {
    "a":   (Fr(25, 10000), Fr(1, 10000)),   # 0.0025 +/- 0.0001
    "w":   (Fr(0),          Fr(1, 1000)),    # 0     +/- 0.001
    "z":   (Fr(1),          Fr(1, 100)),     # 1     +/- 0.01
    "s":   (Fr(-1),         Fr(2, 100)),     # -1    +/- 0.02
    "c40": (Fr(0),          Fr(1, 100)),     # 0     +/- 0.01
    "c31": (Fr(0),          Fr(1, 100)),
    "c22": (Fr(0),          Fr(1, 100)),
    "c13": (Fr(0),          Fr(1, 100)),
    "c04": (Fr(0),          Fr(1, 100)),
}
PARAM_ORDER = ["a", "w", "z", "s", "c40", "c31", "c22", "c13", "c04"]




def validate_full_dimensional(spec):
    """Reject any missing or nonpositive radius BEFORE any dynamical margin."""
    for name in PARAM_ORDER:
        if name not in spec:
            raise FullDimError("missing parameter %s" % name)
        center, radius = spec[name]
        if not isinstance(radius, Fr):
            radius = Fr(radius)
        if radius <= 0:
            raise FullDimError("FULL_DIMENSIONAL_BOX_FAIL: radius(%s)=%s <= 0" % (name, radius))
    return True




def build_ivs(spec):
    return {name: IV.cr(spec[name][0], spec[name][1]) for name in PARAM_ORDER}




# ---------- degree-four flow, Jacobian/Hessian ----------
def F_x(X, Y, P):
    a, w = P["a"], P["w"]
    c40, c31, c22, c13 = P["c40"], P["c31"], P["c22"], P["c13"]
    r = Fr(1, 20)
    base = X.sq() - Fr(1, 4) + a * X * Y + (w / 2) * Y.sq()
    poly = (c40 * (cube(X) - X / 4) / 6
            + c31 * (3 * X.sq() - Fr(1, 4)) * Y / 6
            + c22 * X * Y.sq() / 2
            + c13 * cube(Y) / 6)
    return base + r * poly




def F_y(X, Y, P):
    a, w, z, s = P["a"], P["w"], P["z"], P["s"]
    c31, c22, c13, c04 = P["c31"], P["c22"], P["c13"], P["c04"]
    r = Fr(1, 20)
    base = a * (X.sq() - Fr(1, 4)) / 2 + s * Y + w * X * Y + (z / 2) * Y.sq()
    poly = (c31 * (cube(X) - X / 4) / 6
            + c22 * X.sq() * Y / 2
            + c13 * X * Y.sq() / 2
            + c04 * cube(Y) / 6)
    return base + r * poly




def J11(X, Y, P):
    a = P["a"]
    c40, c31, c22 = P["c40"], P["c31"], P["c22"]
    r = Fr(1, 20)
    return 2 * X + a * Y + r * (c40 * (3 * X.sq() - Fr(1, 4)) / 6 + c31 * X * Y + c22 * Y.sq() / 2)




def J12(X, Y, P):
    a, w = P["a"], P["w"]
    c31, c22, c13 = P["c31"], P["c22"], P["c13"]
    r = Fr(1, 20)
    return a * X + w * Y + r * (c31 * (3 * X.sq() - Fr(1, 4)) / 6 + c22 * X * Y + c13 * Y.sq() / 2)




def J22(X, Y, P):
    w, z, s = P["w"], P["z"], P["s"]
    c22, c13, c04 = P["c22"], P["c13"], P["c04"]
    r = Fr(1, 20)
    return s + w * X + z * Y + r * (c22 * X.sq() / 2 + c13 * X * Y + c04 * Y.sq() / 2)




# ---------- Route-B S-side cone/handoff polynomials ----------
def Q_x(xi, t, P):
    a, w = P["a"], P["w"]
    c40, c31, c22, c13 = P["c40"], P["c31"], P["c22"], P["c13"]
    r = Fr(1, 20)
    num = (12 * a * t * xi - 6 * a * t
           - 2 * c13 * r * cube(t) * xi.sq()
           + 6 * c22 * r * t.sq() * xi.sq() - 3 * c22 * r * t.sq() * xi
           - 6 * c31 * r * t * xi.sq() + 6 * c31 * r * t * xi - c31 * r * t
           + 2 * c40 * r * xi.sq() - 3 * c40 * r * xi + c40 * r
           - 6 * t.sq() * w * xi - 12 * xi + 12)
    return num / 12




def Q_y(xi, t, P):
    a, w, z, s = P["a"], P["w"], P["z"], P["s"]
    c31, c22, c13, c04 = P["c31"], P["c22"], P["c13"], P["c04"]
    r = Fr(1, 20)
    num = (12 * a * xi - 12 * a
           + 4 * c04 * r * cube(t) * xi.sq()
           - 12 * c13 * r * t.sq() * xi.sq() + 6 * c13 * r * t.sq() * xi
           + 12 * c22 * r * t * xi.sq() - 12 * c22 * r * t * xi + 3 * c22 * r * t
           - 4 * c31 * r * xi.sq() + 6 * c31 * r * xi - 2 * c31 * r
           + 24 * s * t + 12 * t.sq() * xi * z - 24 * t * w * xi + 12 * t * w)
    return num / 24




# ---------- full margin evaluation ----------
CONE_SLOPE_GATE_FAIL = "CONE_SLOPE_GATE_FAIL"
MARGIN_ORDER = [
    "m_threshold", "m_saddle_det", "m_max_A", "m_max_D", "m_max_det",
    "m_axial", "m_cone_slope", "m_cone_axis", "m_cone_upper", "m_cone_lower",
    "m_handoff", "m_strip_top", "m_strip_bottom", "m_central_drift",
    "m_transverse_contraction", "m_chart_X", "m_chart_Y", "m_gersh_1", "m_gersh_2",
]




def compute(spec, kappa=Fr(1, 100), S=10 ** 80):
    """Validate full-dimensionality, then compute all 19 margins + derived quantities."""
    validate_full_dimensional(spec)
    P = build_ivs(spec)
    a, w, z, s = P["a"], P["w"], P["z"], P["s"]
    c40, c31, c22, c13, c04 = P["c40"], P["c31"], P["c22"], P["c13"], P["c04"]
    r = Fr(1, 20)
    sigma = Fr(1, 10)


    # constants requiring sqrt
    sqrt2 = sqrtIV(IV(Fr(2), Fr(2)), S)
    mu = (sqrt2 * 40).recip()                     # 1/(40 sqrt2), interval enclosure


    # Section 5 — witness/strip quantities
    F = a.absI() / 8 + (r * c31.absI()) / 48
    A = (z.absI() + (r * c13.absI()) / 2) * F
    B = (r * c04.absI()) * F.sq() / 2
    C = w.absI() / 2 + mu + (r * c22.absI()) / 8 + A / mu + B / mu.sq()
    W = F / mu
    m_threshold = (-s - C).lo


    # Section 6 — endpoint typing / unstable-slope
    A_S = 1 + (r * c40) / 12
    B_S = a / 2 + (r * c31) / 12
    D_S = s + w / 2 + (r * c22) / 8
    det_S = A_S * D_S - B_S.sq()
    A_M = -1 + (r * c40) / 12
    B_M = -a / 2 + (r * c31) / 12
    D_M = s - w / 2 + (r * c22) / 8
    det_M = A_M * D_M - B_M.sq()
    g = A_S - D_S
    m_saddle_det = -det_S.hi
    m_max_A = -A_M.hi
    m_max_D = -D_M.hi
    m_max_det = det_M.lo
    m_axial = g.lo
    if g.lo > 0:
        tau_upper = B_S.supabs() / g.lo
        m_cone_slope = kappa - 2 * tau_upper
    else:
        tau_upper = None
        m_cone_slope = CONE_SLOPE_GATE_FAIL


    # Section 7 — S-side cone / handoff
    xi = IV(Fr(0), sigma)
    t_full = IV(-kappa, kappa)
    t_pos = IV(kappa, kappa)
    t_neg = IV(-kappa, -kappa)
    m_cone_axis = Q_x(xi, t_full, P).lo
    m_cone_upper = (kappa * Q_x(xi, t_pos, P) - Q_y(xi, t_pos, P)).lo
    m_cone_lower = (kappa * Q_x(xi, t_neg, P) + Q_y(xi, t_neg, P)).lo
    m_handoff = W.lo / 2 - kappa * sigma


    # Section 8 — central strip
    Xc = IV(-Fr(1, 2) + sigma, Fr(1, 2) - sigma)
    Ytop = W
    Ybot = -W
    Ystrip = IV(-W.hi, W.hi)
    m_strip_top = (-F_y(Xc, Ytop, P)).lo
    m_strip_bottom = (F_y(Xc, Ybot, P)).lo
    m_central_drift = (-F_x(Xc, Ystrip, P)).lo
    m_transverse_contraction = -(J22(Xc, Ystrip, P)).hi


    # Section 9 — M-side capture ball / sink
    rho = 2 * sqrtIV(IV(sigma * sigma, sigma * sigma) + W.sq(), S)
    rho_u = rho.hi
    Xcap = IV(-Fr(1, 2) - rho_u, -Fr(1, 2) + rho_u)
    Ycap = IV(-rho_u, rho_u)
    offdiag = J12(Xcap, Ycap, P).supabs()
    m_chart_X = Fr(3, 4) - (Fr(1, 2) + rho_u)
    m_chart_Y = Fr(1) - rho_u
    m_gersh_1 = (-(J11(Xcap, Ycap, P)).hi) - offdiag
    m_gersh_2 = (-(J22(Xcap, Ycap, P)).hi) - offdiag


    margins = {
        "m_threshold": m_threshold, "m_saddle_det": m_saddle_det, "m_max_A": m_max_A,
        "m_max_D": m_max_D, "m_max_det": m_max_det, "m_axial": m_axial,
        "m_cone_slope": m_cone_slope, "m_cone_axis": m_cone_axis, "m_cone_upper": m_cone_upper,
        "m_cone_lower": m_cone_lower, "m_handoff": m_handoff, "m_strip_top": m_strip_top,
        "m_strip_bottom": m_strip_bottom, "m_central_drift": m_central_drift,
        "m_transverse_contraction": m_transverse_contraction, "m_chart_X": m_chart_X,
        "m_chart_Y": m_chart_Y, "m_gersh_1": m_gersh_1, "m_gersh_2": m_gersh_2,
    }
    derived = {"mu": mu, "F": F, "A": A, "B": B, "C": C, "W": W, "rho": rho,
               "tau_upper": tau_upper, "kappa": kappa, "S": S}
    return {"margins": margins, "derived": derived}




def all_positive(margins):
    for name in MARGIN_ORDER:
        v = margins[name]
        if not isinstance(v, Fr) or v <= 0:
            return False
    return True
