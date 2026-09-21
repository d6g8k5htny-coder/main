"""Bounded three-jet determinant/Mahalanobis certificate for one RN5 wedge.

The caller must freshly authenticate the injected authored N6 adapter and its
source dependencies. No source loader, cache, archive execution or claim/status
write occurs here. See PROOF.md for the covariance-polynomial argument.
"""
from fractions import Fraction as F
from itertools import permutations
from math import factorial

from research.cover import Box
from research.interval import Interval as I, sqrt
from research.rn import side24
from research.rn.conditioning import ConditioningInconclusive

ZERO = I.exact(0)
OUTER_BOX = Box(F(9999, 100000), F(11, 100), -F(7, 10000), F(7, 10000))
RADII = (F(1, 10), F(11, 100))
TURNS = (-F(1, 1024), F(1, 1024))
MAX_PIECES = 256
PI_UPPER = F(22, 7)


def _settings(order, bits):
    if type(order) is not int or not 1 <= order <= 6:
        raise ValueError('Taylor order must be an integer in 1..6')
    if type(bits) is not int or not 128 <= bits <= 512:
        raise ValueError('rounding bits must be an integer in 128..512')


def _rectangle(box):
    if type(box) is not Box:
        raise TypeError('an exact Box is required')
    for x in (box.u0, box.u1, box.v0, box.v1):
        if type(x) is not F or max(abs(x.numerator).bit_length(), x.denominator.bit_length()) > 4096:
            raise ValueError('rectangle endpoints must be bounded exact Fractions')
    if not (OUTER_BOX.u0 <= box.u0 < box.u1 <= OUTER_BOX.u1 and
            OUTER_BOX.v0 <= box.v0 < box.v1 <= OUTER_BOX.v1):
        raise ValueError('rectangle must be a positive-area subset of the bounded pilot box')
    return ((box.u0+box.u1)/2, (box.v0+box.v1)/2), ((box.u1-box.u0)/2, (box.v1-box.v0)/2)


def _polyadd(a, b, scale=1, *, bits):
    out = dict(a)
    for g, value in b.items():
        out[g] = (out.get(g, ZERO)+value*scale).round_out(bits)
    return {g: value for g, value in out.items() if value != ZERO}


def _polymul(a, b, *, bits):
    out = {}
    for g, x in a.items():
        if x == ZERO:
            continue
        for h, y in b.items():
            if y == ZERO:
                continue
            power = (g[0]+h[0], g[1]+h[1])
            out[power] = (out.get(power, ZERO)+(x*y).round_out(bits)).round_out(bits)
    return {g: value for g, value in out.items() if value != ZERO}


def _evaluate(coefficients, halfwidths, *, bits):
    """Outward interval evaluation, after equal monomials are aggregated.

    Both-even nonconstant monomials are nonnegative on a centered rectangle.
    Every other nonconstant monomial may have either sign.
    """
    result = ZERO
    for (a, b), coefficient in coefficients.items():
        radius = halfwidths[0]**a*halfwidths[1]**b
        monomial = (I.exact(1) if a+b == 0 else
                    I(0, radius) if a % 2 == b % 2 == 0 else I(-radius, radius))
        result = (result+(coefficient*monomial).round_out(bits)).round_out(bits)
    return result


def _determinant_polynomial(covariance, *, bits):
    entries = [[{g: m[i][j] for g, m in covariance.items() if m[i][j] != ZERO}
                for j in range(3)] for i in range(3)]
    a, b, c = entries[0]
    _, d, e = entries[1]
    _, _, f = entries[2]
    out = _polymul(_polymul(a, d, bits=bits), f, bits=bits)
    for x, y, z, factor in ((b, c, e, 2), (a, e, e, -1),
                             (d, c, c, -1), (f, b, b, -1)):
        term = _polymul(_polymul(x, y, bits=bits), z, bits=bits)
        out = _polyadd(out, term, factor, bits=bits)
    return out


def _determinant_error(covariance, error):
    """Absolute determinant perturbation bound for every symmetric P+E.

    Sum all nonempty error products in each of the six determinant products;
    entrywise magnitudes bound their absolute values. No PSD of the interval
    hull or surrogate lower variance is assumed.
    """
    bound = F(0)
    for permutation in permutations(range(3)):
        outer = inner = F(1)
        for i, j in enumerate(permutation):
            magnitude = covariance[i][j].mag()
            inner *= magnitude
            outer *= magnitude+error[i][j]
        bound += outer-inner
    return bound


def _polynomials(n6, center, order, bits):
    lift = n6.center_jet_lift(center, order=order, bits=bits)
    lookup = {alpha: 6+i for i, alpha in enumerate(lift['jets'])}
    ids = tuple(lookup[alpha] for alpha in side24.JETS)
    central_covariance = tuple(tuple(lift['covariance'][i][j] for j in ids) for i in ids)
    transform = n6.inverse_lower(n6.approximate_cholesky(central_covariance, bits))
    support = n6.indices(order)
    rows = tuple(tuple((alpha, lookup[(beta[0]+alpha[0], beta[1]+alpha[1])],
                        F(1, factorial(alpha[0])*factorial(alpha[1]))) for alpha in support)
                 for beta in side24.JETS)
    mean = {alpha: [ZERO]*3 for alpha in support}
    for i, row in enumerate(rows):
        for alpha, j, weight in row:
            mean[alpha][i] = (lift['mean'][j]*weight).round_out(bits)
    mean = {alpha: tuple(values) for alpha, values in mean.items()}
    covariance = {g: [[ZERO]*3 for _ in range(3)] for g in n6.indices(2*order)}
    for i, row in enumerate(rows):
        for j in range(i+1):
            for alpha, u, x in row:
                for beta, v, y in rows[j]:
                    power = (alpha[0]+beta[0], alpha[1]+beta[1])
                    term = (lift['covariance'][u][v]*(x*y)).round_out(bits)
                    covariance[power][i][j] = (covariance[power][i][j]+term).round_out(bits)
            for g in covariance:
                covariance[g][j][i] = covariance[g][i][j]
    covariance = {g: tuple(tuple(row) for row in matrix) for g, matrix in covariance.items()}
    transformed = {g: n6.congruence(transform, matrix, bits) for g, matrix in covariance.items()}
    return lift, transform, mean, covariance, transformed


def certify_rectangle(n6, box, *, order=6, bits=256):
    """Prove det G>0 and q>=Q on every point of an exact rectangle.

    G is the actual conditional three-jet covariance of the authenticated
    normalized SIDE24 law. q is the full three-jet Mahalanobis square at
    (height,0,0), for every height (thus the complete requested mark window).
    A refusal raises ConditioningInconclusive and supplies no bound.
    """
    _settings(order, bits)
    center, h = _rectangle(box)
    lift, transform, mean, covariance, transformed = _polynomials(n6, center, order, bits)
    det_coefficients = _determinant_polynomial(transformed, bits=bits)
    det_polynomial = _evaluate(det_coefficients, h, bits=bits)
    eps = tuple(n6.remainder_bound(beta, h, order) for beta in side24.JETS)
    eta = tuple(sum((abs(x)*e for x, e in zip(row, eps)), F(0)) for row in transform)
    evaluated = tuple(tuple(_evaluate({g: m[i][j] for g, m in transformed.items()}, h, bits=bits)
                            for j in range(3)) for i in range(3))
    if any(evaluated[i][i].hi < 0 for i in range(3)):
        raise ConditioningInconclusive('Taylor covariance has negative upper variance')
    sigma = tuple(sqrt(I(0, evaluated[i][i].hi), n6.options(bits)).hi for i in range(3))
    error = tuple(tuple(sigma[i]*eta[j]+sigma[j]*eta[i]+eta[i]*eta[j]
                        for j in range(3)) for i in range(3))
    det_error = _determinant_error(evaluated, error)
    jacobian = transform[0][0]*transform[1][1]*transform[2][2]
    if jacobian == 0:
        raise ConditioningInconclusive('singular exact preconditioner')
    determinant = ((det_polynomial+I(-det_error, det_error))/(jacobian**2)).round_out(bits)
    if determinant.lo <= 0:
        raise ConditioningInconclusive('three-jet determinant lower bound is not positive')
    raw_mean = _evaluate({g: values[1] for g, values in mean.items()}, h, bits=bits)
    raw_variance = _evaluate({g: matrix[1][1] for g, matrix in covariance.items()}, h, bits=bits)
    if raw_variance.hi < 0 or lift['conditioning_energy'].lo < 0:
        raise ConditioningInconclusive('negative upper variance or conditioning energy')
    mean_error = sqrt(lift['conditioning_energy'], n6.options(bits)).hi*eps[1]
    fx_mean = (raw_mean+I(-mean_error, mean_error)).round_out(bits)
    fx_sigma = sqrt(I(0, raw_variance.hi), n6.options(bits)).hi
    variance_error = 2*fx_sigma*eps[1]+eps[1]**2
    fx_variance = (raw_variance+I(-variance_error, variance_error)).round_out(bits)
    if fx_variance.hi <= 0:
        raise ConditioningInconclusive('fx variance upper bound is not positive')
    q_lower = (fx_mean**2).lo/fx_variance.hi
    return dict(box=box, center=center, halfwidths=h, order=order, bits=bits,
        determinant=determinant, determinant_lower=determinant.lo,
        q_lower=q_lower, fx_mean=fx_mean, fx_variance=fx_variance,
        conditioning_energy=lift['conditioning_energy'], six_pin_pivots=lift['six_pin_pivots'],
        source_binding=lift['source_binding'], transform=transform, determinant_transform=jacobian,
        determinant_polynomial_interval=det_polynomial, determinant_error_upper=det_error,
        determinant_monomial_count=len(det_coefficients), l2_remainders=eps,
        transformed_l2_remainders=eta, covariance_error_upper=error,
        mark_domain=side24.MARK_DOMAIN,
        statement='ACTUAL_SOURCE_LAW_THREE_JET_DETERMINANT_AND_MAHALANOBIS_LOWER',
        authority='NONE', independence_credit=0, field_certified=False,
        h3_reproved=False, original_prize_closed=False, scientific_status_changed=False)


def containment():
    """Exact rational inequalities using pi<22/7, |sin t|<=|t|, cos t>=1-t²/2."""
    theta_upper = PI_UPPER/512
    x_lower = RADII[0]*(1-theta_upper**2/2)
    y_upper = RADII[1]*theta_upper
    if not (0 < theta_upper < 1 and x_lower >= OUTER_BOX.u0 and
            RADII[1] <= OUTER_BOX.u1 and y_upper <= OUTER_BOX.v1 and
            -y_upper >= OUTER_BOX.v0):
        raise ValueError('polar wedge is not contained in the auxiliary rectangle')
    return dict(radii=RADII, turns=TURNS, containing_rectangle=OUTER_BOX,
        pi_strict_upper=PI_UPPER, absolute_angle_upper=theta_upper,
        x_lower=x_lower, absolute_y_upper=y_upper, area_pi_coefficient=F(21, 5120000),
        area_uses='one polar wedge; auxiliary Cartesian areas are not integrated')


def certify_wedge(n6, *, pieces=12, order=6, bits=256):
    """A finite complete auxiliary cover; one exact polar wedge is the target."""
    _settings(order, bits)
    if type(pieces) is not int or not 1 <= pieces <= MAX_PIECES:
        raise ValueError('piece count must be an integer in 1..256')
    geometry = containment()
    width = (OUTER_BOX.u1-OUTER_BOX.u0)/pieces
    boxes = tuple(Box(OUTER_BOX.u0+i*width, OUTER_BOX.u0+(i+1)*width,
                      OUTER_BOX.v0, OUTER_BOX.v1) for i in range(pieces))
    if boxes[0].u0 != OUTER_BOX.u0 or boxes[-1].u1 != OUTER_BOX.u1 or any(
            left.u1 != right.u0 for left, right in zip(boxes, boxes[1:])):
        raise ValueError('auxiliary cover has a gap or wrong boundary')
    records = tuple(certify_rectangle(n6, box, order=order, bits=bits) for box in boxes)
    return dict(geometry=geometry, pieces=records, piece_count=pieces,
        pending_pieces=0, determinant_lower=min(r['determinant_lower'] for r in records),
        q_lower=min(r['q_lower'] for r in records),
        conditioning_energy_upper=max(r['conditioning_energy'].hi for r in records),
        mark_domain=side24.MARK_DOMAIN,
        scope='fixed r=1/20,b=6/5 and specified positive-x wedge only',
        authority='NONE', independence_credit=0, field_certified=False,
        full_annulus_certified=False, all_radii_certified=False,
        h3_reproved=False, original_prize_closed=False, scientific_status_changed=False)
