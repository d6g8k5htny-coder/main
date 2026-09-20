"""Connect source-bound RN cell proofs to the exact cover ledger.

The supplied rectangle is a local pilot domain, never the entire RN annulus.
Every RN bound remains conditional on the imported H3 floor. No scientific
status, global RN result, all-small-r claim or independence credit follows.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from research.interval import Interval as I
from research.cover.ledger import Box
from research.cover.regions import INSIDE
from research.rn.conditioning import ConditioningInconclusive


@dataclass(frozen=True)
class RNRectangleRegion:
    """An exact Cartesian rectangle lying wholly in the RN5 near annulus."""

    rectangle: Box
    name: str = 'RN5_LOCAL_RECTANGLE'
    coords: str = 'cartesian(x,y)'

    def __post_init__(self):
        if type(self.rectangle) is not Box or self.rectangle.is_degenerate():
            raise ValueError('nondegenerate exact Box required')
        radius = self.radius2_enclosure(self.rectangle)
        if radius.lo < F(1, 100) or radius.hi > 25:
            raise ValueError('the entire pilot rectangle must lie in the near annulus')

    def domain(self):
        return self.rectangle

    def roots(self):
        return [self.rectangle]

    def classify(self, box):
        if not self.rectangle.contains_box(box) or box.is_degenerate():
            raise ValueError('cell escapes the pilot rectangle or is degenerate')
        return INSIDE

    def area(self, box, prec):
        self.classify(box)
        return I.exact(box.param_area())

    def area_rational_upper(self, box):
        self.classify(box)
        return box.param_area()

    def diameter_bound(self, box):
        self.classify(box)
        return box.du() + box.dv()

    def subdivide(self, box):
        self.classify(box)
        return list(box.quad())

    def cartesian_enclosure(self, box, prec):
        self.classify(box)
        return I(box.u0, box.u1), I(box.v0, box.v1)

    def radius2_enclosure(self, box):
        return I(box.u0, box.u1)**2 + I(box.v0, box.v1)**2


class RNSide24Integrand:
    """Return [0,U] for the actual typed integrand, with explicit H3 premise.

    The region supplies containing Cartesian coordinates and multiplies by its
    own geometric area. No Jacobian or area factor enters this adapter. Cell
    admission failures are recoverable; invalid inputs and source failures
    propagate as errors. Completed and recoverable bound attempts remain recorded for audit.
    """

    name = 'RN5:SIDE24:typed_integrand:H3_CONDITIONAL'
    label = ('Source-derived RN5 typed integrand bound, fixed r=1/20; '
             'conditional on the pinned imported H3 floor; no full-annulus '
             'or all-small-r theorem, status promotion or independence credit.')
    certifying = True

    def __init__(self, *, bits=192):
        if type(bits) is not int or not 128 <= bits <= 1024:
            raise ValueError('bits must be an integer in 128..1024')
        self.bits = bits
        self.attempts = []

    def range_enclosure(self, region, box, prec):
        from research.cover.driver import RecoverableEnclosureError
        from research.rn.side24_cell import spatial_cell
        x, y = region.cartesian_enclosure(box, prec)
        if not isinstance(x, I) or not isinstance(y, I):
            raise TypeError('region must supply exact Cartesian intervals')
        cartesian = Box(x.lo, x.hi, y.lo, y.hi)
        attempt = {'parameter_box': box.as_json(),
                   'cartesian_box': cartesian.as_json(), 'bits': self.bits}
        try:
            proof = spatial_cell(cartesian, bits=self.bits)
        except ConditioningInconclusive as error:
            self.attempts.append({**attempt, 'status': 'INCONCLUSIVE',
                                  'reason': str(error)})
            raise RecoverableEnclosureError(str(error)) from error
        upper = proof['integrand_upper']
        if type(upper) is not F or upper < 0:
            raise ValueError('cell proof must return a nonnegative exact upper')
        self.attempts.append({**attempt, 'status': 'BOUNDED', 'proof': proof})
        return I(F(0), upper)
