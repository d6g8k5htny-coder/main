"""Certified fixed-point density/window factor under the RN5 six-pin law.

The H3 floor is imported as a pinned source premise, never re-proved here.
This author candidate supplies no spatial integral/cover, all-r result,
scientific status change, field-wide certificate or independence credit.
"""
from fractions import Fraction as F
import hashlib
import re
from zipfile import ZipFile

from research.interval import Interval as I, exp, pi, sqrt
from research.rn import side24
from research.rn.conditioning import ConditioningInconclusive, condition_gaussian, interval_ldlt

H3_MEMBER = 'intake/rn_source/K3_SIDE24_LB/UPPER2D/H3_closure/H3_RUNG_FLOOR.md'
H3_SHA = '6347275d86c56842b719b36180e535bc1793d6995ddb68464db2960820440dfa'
H3_BYTES = 7003
Z_LO = F('0.0077592917375327855')


def _exact(value):
    if type(value) not in (int, F):
        raise TypeError('exact int/Fraction required; float and bool refused')
    result = F(value)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError('rational input exceeds 4096-bit resource limit')
    return result


def _interval(value):
    if isinstance(value, I):
        return I(_exact(value.lo), _exact(value.hi))
    return I.exact(_exact(value))


def _precision(bits):
    if type(bits) is not int or not 128 <= bits <= 1024:
        raise ValueError('rounding must use an integer in 128..1024 bits')
    return min(400, (bits+2)//3+24)


def _dot(left, right, bits):
    if len(left) != len(right):
        raise ValueError('dot product dimensions disagree')
    total = I.exact(0)
    for x, y in zip(left, right):
        total = (total+(x*y).round_out(bits)).round_out(bits)
    return total


def gradient_density(mean, covariance, *, bits=192):
    """Enclose a nonsingular two-dimensional Gaussian density at zero.

    With G=L diag(d) L^T and w=L^-1 mean, q=sum(w_i^2/d_i).
    This sum of interval squares retains q>=0 without a cancellation-prone
    dot product with G^-1. Positive pivots are mandatory.
    """
    prec = _precision(bits)
    if len(mean) != 2 or len(covariance) != 2:
        raise ValueError('two gradient coordinates required')
    mean = tuple(_interval(x) for x in mean)
    factor = interval_ldlt(covariance, round_bits=bits)
    whitened = []
    quadratic = I.exact(0)
    determinant = I.exact(1)
    for i in range(2):
        w = (mean[i]-_dot(factor.lower[i][:i], whitened, bits)).round_out(bits)
        whitened.append(w)
        quadratic = (quadratic+(w**2/factor.pivots[i]).round_out(bits)).round_out(bits)
        determinant = (determinant*factor.pivots[i]).round_out(bits)
    if quadratic.lo < 0 or determinant.lo <= 0:
        raise ConditioningInconclusive('density positivity not established')
    density = (exp(-quadratic/2, prec)/(2*pi(prec)*sqrt(determinant, prec))).round_out(bits)
    return {'gradient_mean': mean, 'gradient_covariance': tuple(tuple(_interval(x) for x in row) for row in covariance),
            'gradient_pivots': factor.pivots, 'gradient_mahalanobis': quadratic,
            'gradient_determinant': determinant, 'gradient_density': density}


def height_window_cap(mean, variance, window, *, bits=192):
    """Bound the mass of the entire closed window by length times density sup.

    Every admitted mean lies in its interval and every admitted variance is
    positive. The nearest possible mean/window distance gives a lower bound
    on |v-mu| for all v in the window. Variance is retained as an interval in
    exp(-distance^2/(2 variance))/sqrt(2 pi variance); this is conservative
    even though variance occurs twice. Clipping the mass upper to one is valid.
    """
    prec = _precision(bits)
    mean, variance = _interval(mean), _interval(variance)
    if not isinstance(window, (tuple, list)) or len(window) != 2:
        raise ValueError('two exact window endpoints required')
    lo, hi = map(_exact, window)
    if lo > hi:
        raise ValueError('window endpoints are reversed')
    if variance.lo <= 0:
        raise ConditioningInconclusive('conditional height variance not certainly positive')
    distance = max(F(0), mean.lo-hi, lo-mean.hi)
    density = (exp(-I.exact(distance**2)/(2*variance), prec)/sqrt(2*pi(prec)*variance, prec)).round_out(bits)
    upper = min(F(1), (hi-lo)*density.hi)
    return {'height_mean': mean, 'height_variance': variance,
            'height_window': (lo, hi), 'height_window_length': hi-lo,
            'height_distance_lower': distance, 'height_density_upper': density.hi,
            'window_mass_upper': upper, 'window_method': 'MIN_ONE_LENGTH_TIMES_UNIFORM_DENSITY_UPPER'}


def imported_h3_floor():
    """Authenticate the source bytes and exact floor; retain imported status."""
    binding = side24.source_binding()
    with ZipFile(side24.ARCHIVE) as archive:
        info = archive.getinfo(H3_MEMBER)
        if info.file_size != H3_BYTES:
            raise ValueError('H3 floor source size mismatch')
        data = archive.read(info)
    if hashlib.sha256(data).hexdigest() != H3_SHA:
        raise ValueError('H3 floor source identity mismatch')
    matches = re.findall(r'Z_\{0\.05\} ∈ \[\s*([\d.e+-]+)\s*,', data.decode('utf-8'))
    if len(matches) != 1 or F(matches[0]) != Z_LO or Z_LO <= 0:
        raise ValueError('exact imported H3 floor mismatch')
    return {'value': Z_LO, 'member': H3_MEMBER, 'member_sha256': H3_SHA,
            'member_bytes': H3_BYTES, 'archive_sha256': binding['archive_sha256'],
            'status': 'IMPORTED_SOURCE_PREMISE_NOT_REPROVED', 'r': side24.R,
            'height': side24.HEIGHT, 'authority': 'NONE', 'independence_credit': 0}


def density_window(point=(F(1), F(1)), *, bits=192):
    """Return the exact density/window upper under six fixed pins at one y.

    The output bounds p(grad f(y)=0 | six pins) times the conditional height
    window probability. It does not yet multiply the determinant moment cap
    or divide by the imported H3 floor; these remain explicit composition steps.
    """
    prec = _precision(bits)
    block = side24.nine_pin_blocks(point, prec=prec, bits=bits)
    raw = block['conditioning_covariance']
    law = block['conditioning_law'].replace(':nine-pins:centered-mark', ':six-pins')
    jet = condition_gaussian(tuple(tuple(row[:6]) for row in raw[:6]),
        tuple(tuple(row[:6]) for row in raw[6:]), tuple(tuple(row[6:]) for row in raw[6:]),
        block['conditioning_intercept'][:6], (F(0),)*6,
        conditioning_order=block['conditioning_order'][:6], target_order=block['conditioning_order'][6:],
        covariance_law=law, mean_law=law, conditioning_law=law,
        mark_domain=(F(0), F(0)), round_bits=bits)
    jet_factor = interval_ldlt(jet.covariance, round_bits=bits)
    gradient_covariance = tuple(tuple(row[1:]) for row in jet.covariance[1:])
    gradient = gradient_density(jet.intercept[1:], gradient_covariance, bits=bits)
    factor = interval_ldlt(gradient_covariance, round_bits=bits)
    cross = jet.covariance[0][1:]
    height_mean = (jet.intercept[0]-_dot(cross, factor.solve(jet.intercept[1:]), bits)).round_out(bits)
    height_variance = (jet.covariance[0][0]-_dot(cross, factor.solve(cross), bits)).round_out(bits)
    window = height_window_cap(height_mean, height_variance,
                               (side24.HEIGHT-side24.ELL, side24.HEIGHT), bits=bits)
    mass_upper = gradient['gradient_density'].hi*window['window_mass_upper']
    return {**gradient, **window, 'point': block['point'], 'round_bits': bits,
            'source_binding': block['source_binding'], 'conditioning_law': law,
            'six_pin_order': block['conditioning_order'][:6],
            'six_pin_values': block['conditioning_intercept'][:6],
            'jet_order': block['conditioning_order'][6:],
            'jet_mean': jet.intercept, 'jet_covariance': jet.covariance,
            'six_pin_pivots': jet.pivots, 'jet_pivots': jet_factor.pivots,
            'density_mass_upper': mass_upper, 'imported_h3_floor': imported_h3_floor(),
            'authority': 'NONE', 'independence_credit': 0, 'field_certified': False,
            'spatial_cover_certified': False, 'scientific_status_changed': False,
            'all_small_r_certified': False, 'h3_reproved': False,
            'original_prize_closed': False}
