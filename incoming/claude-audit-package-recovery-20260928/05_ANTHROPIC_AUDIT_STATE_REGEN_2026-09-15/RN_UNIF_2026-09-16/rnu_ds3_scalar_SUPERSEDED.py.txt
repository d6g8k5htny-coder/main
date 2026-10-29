"""DS3: exact third-order arithmetic on scalars of y=(y1,y2).

Extends the engine DS (value + gradient + Hessian) by the four third
jets txxx, txxy, txyy, tyyy. Product/quotient/exp/log/sqrt/pow rules
are the standard Faà-di-Bruno specializations in two variables.

This module is standalone arithmetic. It does not close D3-LEMMA-RN-UNIF.
"""
from mpmath import mp, mpf
import mpmath


def _m(x):
    return x if isinstance(x, mpf) else mpf(x)


class DS3:
    __slots__ = ('v', 'gx', 'gy', 'hxx', 'hyy', 'hxy',
                 'txxx', 'txxy', 'txyy', 'tyyy')

    def __init__(self, v, gx=0, gy=0, hxx=0, hyy=0, hxy=0,
                 txxx=0, txxy=0, txyy=0, tyyy=0):
        self.v = _m(v); self.gx = _m(gx); self.gy = _m(gy)
        self.hxx = _m(hxx); self.hyy = _m(hyy); self.hxy = _m(hxy)
        self.txxx = _m(txxx); self.txxy = _m(txxy)
        self.txyy = _m(txyy); self.tyyy = _m(tyyy)

    @staticmethod
    def const(c):
        return DS3(c)

    def gnorm(self):
        return mpmath.sqrt(self.gx ** 2 + self.gy ** 2)

    def hnorm(self):
        return mpmath.sqrt(self.hxx ** 2 + self.hyy ** 2 + 2 * self.hxy ** 2)

    def tnorm(self):
        # Frobenius of the 2x2x2 symmetric 3-tensor
        return mpmath.sqrt(
            self.txxx ** 2 + self.tyyy ** 2
            + 3 * self.txxy ** 2 + 3 * self.txyy ** 2
        )

    def __add__(self, o):
        o = o if isinstance(o, DS3) else DS3(o)
        return DS3(self.v + o.v, self.gx + o.gx, self.gy + o.gy,
                   self.hxx + o.hxx, self.hyy + o.hyy, self.hxy + o.hxy,
                   self.txxx + o.txxx, self.txxy + o.txxy,
                   self.txyy + o.txyy, self.tyyy + o.tyyy)

    __radd__ = __add__

    def __neg__(self):
        return DS3(-self.v, -self.gx, -self.gy, -self.hxx, -self.hyy, -self.hxy,
                   -self.txxx, -self.txxy, -self.txyy, -self.tyyy)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, DS3) else DS3(o)))

    def __rsub__(self, o):
        return (o if isinstance(o, DS3) else DS3(o)) + (-self)

    def __mul__(self, o):
        o = o if isinstance(o, DS3) else DS3(o)
        return DS3(
            self.v * o.v,
            self.gx * o.v + self.v * o.gx,
            self.gy * o.v + self.v * o.gy,
            self.hxx * o.v + 2 * self.gx * o.gx + self.v * o.hxx,
            self.hyy * o.v + 2 * self.gy * o.gy + self.v * o.hyy,
            self.hxy * o.v + self.gx * o.gy + self.gy * o.gx + self.v * o.hxy,
            self.txxx * o.v + 3 * self.hxx * o.gx + 3 * self.gx * o.hxx + self.v * o.txxx,
            self.txxy * o.v + self.hxx * o.gy + 2 * self.hxy * o.gx
            + 2 * self.gx * o.hxy + self.gy * o.hxx + self.v * o.txxy,
            self.txyy * o.v + self.hyy * o.gx + 2 * self.hxy * o.gy
            + 2 * self.gy * o.hxy + self.gx * o.hyy + self.v * o.txyy,
            self.tyyy * o.v + 3 * self.hyy * o.gy + 3 * self.gy * o.hyy + self.v * o.tyyy,
        )

    __rmul__ = __mul__

    def inv(self):
        v = self.v
        iv = 1 / v
        iv2 = iv * iv
        iv3 = iv2 * iv
        iv4 = iv2 * iv2
        gx, gy = self.gx, self.gy
        hxx, hyy, hxy = self.hxx, self.hyy, self.hxy
        return DS3(
            iv,
            -gx * iv2, -gy * iv2,
            2 * gx * gx * iv3 - hxx * iv2,
            2 * gy * gy * iv3 - hyy * iv2,
            2 * gx * gy * iv3 - hxy * iv2,
            -6 * gx ** 3 * iv4 + 6 * gx * hxx * iv3 - self.txxx * iv2,
            -6 * gx * gx * gy * iv4 + 2 * gx * hxy * iv3 + 2 * gy * hxx * iv3
            + 2 * gx * hxy * iv3 - self.txxy * iv2,
            -6 * gx * gy * gy * iv4 + 2 * gy * hxy * iv3 + 2 * gx * hyy * iv3
            + 2 * gy * hxy * iv3 - self.txyy * iv2,
            -6 * gy ** 3 * iv4 + 6 * gy * hyy * iv3 - self.tyyy * iv2,
        )

    def __truediv__(self, o):
        o = o if isinstance(o, DS3) else DS3(o)
        return self * o.inv()

    def __rtruediv__(self, o):
        return (o if isinstance(o, DS3) else DS3(o)) * self.inv()

    def exp(self):
        e = mpmath.exp(self.v)
        gx, gy = self.gx, self.gy
        return DS3(
            e,
            e * gx, e * gy,
            e * (self.hxx + gx * gx),
            e * (self.hyy + gy * gy),
            e * (self.hxy + gx * gy),
            e * (self.txxx + 3 * gx * self.hxx + gx ** 3),
            e * (self.txxy + 2 * gx * self.hxy + gy * self.hxx + gx * gx * gy),
            e * (self.txyy + 2 * gy * self.hxy + gx * self.hyy + gx * gy * gy),
            e * (self.tyyy + 3 * gy * self.hyy + gy ** 3),
        )

    def log(self):
        v = self.v
        gx, gy = self.gx, self.gy
        return DS3(
            mpmath.log(v),
            gx / v, gy / v,
            self.hxx / v - gx * gx / v ** 2,
            self.hyy / v - gy * gy / v ** 2,
            self.hxy / v - gx * gy / v ** 2,
            self.txxx / v - 3 * gx * self.hxx / v ** 2 + 2 * gx ** 3 / v ** 3,
            self.txxy / v - (2 * gx * self.hxy + gy * self.hxx) / v ** 2
            + 2 * gx * gx * gy / v ** 3,
            self.txyy / v - (2 * gy * self.hxy + gx * self.hyy) / v ** 2
            + 2 * gx * gy * gy / v ** 3,
            self.tyyy / v - 3 * gy * self.hyy / v ** 2 + 2 * gy ** 3 / v ** 3,
        )

    def sqrt(self):
        s = mpmath.sqrt(self.v)
        return DS3(s) * (DS3(self.v, self.gx, self.gy, self.hxx, self.hyy,
                             self.hxy, self.txxx, self.txxy, self.txyy,
                             self.tyyy) / DS3(self.v)).sqrt_from_log()

    def sqrt_from_log(self):
        # used internally; prefer .sqrt()
        return (self.log() * DS3(mpf('0.5'))).exp()

    def pow(self, p):
        p = _m(p)
        if p == 0:
            return DS3(1)
        # integer powers by multiplication so y=0 is legal
        try:
            pi = int(p)
            if p == pi and 0 < pi <= 8:
                acc = DS3(1)
                for _ in range(pi):
                    acc = acc * self
                return acc
        except Exception:
            pass
        return (self.log() * DS3(p)).exp()


def ds3_selftest():
    """f(x,y)=(x^2+y^3)*exp(x/5) at (5,0): compare jets to closed form."""
    x, y = mpf(5), mpf(0)
    # seed coordinates as DS3
    X = DS3(x, gx=1)
    Y = DS3(y, gy=1)
    f = (X * X + Y.pow(3)) * (X / DS3(5)).exp()
    # closed form
    e = mpmath.exp(x / 5)
    v = (x ** 2 + y ** 3) * e
    # first
    fx = (2 * x + (x ** 2 + y ** 3) / 5) * e
    fy = (3 * y ** 2) * e
    # second
    fxx = (2 + 2 * x / 5 + 2 * x / 5 + (x ** 2 + y ** 3) / 25) * e
    fyy = (6 * y) * e
    fxy = (3 * y ** 2 / 5) * e
    # third
    fxxx = (2 / 5 + 2 / 5 + 2 * x / 25 + 2 / 5 + 2 * x / 25 + (x ** 2 + y ** 3) / 125) * e
    # simpler: differentiate fxx
    # fxx = (2 + 4x/5 + (x^2+y^3)/25) e
    fxxx_cf = (mpf(6) / 5 + mpf(6) * x / 25 + (x ** 2 + y ** 3) / 125) * e
    fxxy_cf = (3 * y ** 2 / 25) * e
    fxyy_cf = (6 * y / 5) * e
    fyyy_cf = 6 * e
    errs = [
        abs(f.v - v), abs(f.gx - fx), abs(f.gy - fy),
        abs(f.hxx - fxx), abs(f.hyy - fyy), abs(f.hxy - fxy),
        abs(f.txxx - fxxx_cf), abs(f.txxy - fxxy_cf),
        abs(f.txyy - fxyy_cf), abs(f.tyyy - fyyy_cf),
    ]
    return max(errs), errs
