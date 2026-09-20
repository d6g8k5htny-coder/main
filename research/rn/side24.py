"""Exact interval SIDE24 kernel and RN5 fixed-r nine-pin Gaussian inputs.

This author-side adapter binds the mathematical field and coordinate law. It
does not certify density/window/H3 factors, a spatial cover, all-small-r scope,
review independence or any scientific status. No frozen code is executed.
"""
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from pathlib import Path
from zipfile import ZipFile

from engine.operations.rn_applicability import Context, RN5_ID, RN5_SHA
from engine.operations.rn_family_applicability import FamilyLaw
from research.bands.hermite_gaussian import hermite_coefficients
from research.interval import Interval as I, exp

ROOT = Path(__file__).resolve().parents[2]
MIRROR = ROOT/'drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM'
ARCHIVE = MIRROR/'RN5_REPAIR_AND_ERRATUM_BUNDLE.zip'
ARCHIVE_SHA = '28c1c385406452a706f11aea1646a87585b002841005d57febdf32222c368c6e'
MEMBERS = {
    'closure_round2/rn_field.py': 'd9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8',
    'round5/near_moments.py': '03c6e35c426eee8ed91139b93f0ada4e96c7293e5afb4176677850596fbfca6e',
    'round5/verify_round5.py': '614daf280f5037757d261281b73ea66edb350da4619066ee56ea5a4216b7737e',
}
R, HEIGHT = F(1, 20), F(6, 5)
ELL = R**3/6
MID = HEIGHT-ELL/2
MARK_DOMAIN = (-ELL/2, ELL/2)
JETS = ((0, 0), (1, 0), (0, 1))
HJETS = ((2, 0), (0, 2), (1, 1))


def _rational(x):
    if type(x) not in (int, F):
        raise TypeError('exact int/Fraction required; float and bool refused')
    result = F(x)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError('rational input exceeds 4096-bit resource limit')
    return result


def _parameters(prec, bits):
    if type(prec) is not int or not 20 <= prec <= 400:
        raise ValueError('precision must be an integer in 20..400 decimal digits')
    if type(bits) is not int or not 128 <= bits <= 1024:
        raise ValueError('rounding must use an integer in 128..1024 bits')


def _displacement(value):
    value = value if isinstance(value, I) else I.exact(_rational(value))
    _rational(value.lo)
    _rational(value.hi)
    if value.hi-value.lo > 12:
        raise ValueError('displacement interval width must be at most 12')
    # Periodicity permits one exact common translation of the whole interval.
    center = (value.lo+value.hi)/2
    shift = 24*((center+12)//24)
    reduced = value-shift
    if max(abs(reduced.lo), abs(reduced.hi)) > 18:
        raise ValueError('reduced image-tail radius exceeds 18')
    return reduced


def image_tail(order, radius, *, prec=80):
    """Upper bound for all omitted images |j|>=2, uniform for |s|<=radius.

    For u>=1, |He_n(u)| <= C_n*u^n. The jth pair is bounded by
    2*C_n*(24j+B)^n*exp(-(24j-B)^2/2). Its consecutive ratios decrease,
    so the ratio at j=2 bounds the entire geometric tail. See companion proof.
    """
    if type(order) is not int or not 0 <= order <= 4:
        raise ValueError('kernel derivative order must be 0..4')
    radius = _rational(radius)
    if not 0 <= radius <= 18:
        raise ValueError('image-tail radius must be in 0..18')
    if type(prec) is not int or not 20 <= prec <= 400:
        raise ValueError('precision must be an integer in 20..400')
    coefficient = sum(abs(x) for x in hermite_coefficients(order))
    first = coefficient*(48+radius)**order*exp(I.exact(-(48-radius)**2/2), prec)
    ratio = F(72+radius, 48+radius)**order*exp(I.exact(-((72-radius)**2-(48-radius)**2)/2), prec)
    if ratio.hi >= 1:
        raise ValueError('image-tail contraction not proved')
    return (2*first/(1-ratio)).hi


@lru_cache(maxsize=32)
def normalizer(*, prec=80, bits=192):
    """Positive full one-axis sum, with the omitted positive tail retained."""
    _parameters(prec, bits)
    finite = I.exact(1)+2*exp(I.exact(-288), prec)
    result = (finite+I(0, image_tail(0, F(0), prec=prec))).round_out(bits)
    if result.lo <= 0:
        raise ValueError('normalization denominator not certainly positive')
    return result


def kernel_derivative(order, displacement, *, prec=80, bits=192):
    """Enclose normalized k^(order) over the whole displacement interval."""
    _parameters(prec, bits)
    if type(order) is not int or not 0 <= order <= 4:
        raise ValueError('kernel derivative order must be 0..4')
    s = _displacement(displacement)
    if s.lo == s.hi == 0:
        if order == 0:
            return I.exact(1)  # The identical full sum occurs in numerator/denominator.
        if order % 2:
            return I.exact(0)  # Exact evenness, not a discarded small number.
    coefficients = hermite_coefficients(order)
    finite = I.exact(0)
    for j in (-1, 0, 1):
        x = s+24*j
        polynomial = I.exact(0)
        for c in reversed(coefficients):
            polynomial = polynomial*x+c
        finite += (-1)**order*polynomial*exp(-x**2/2, prec)
    tail = image_tail(order, max(abs(s.lo), abs(s.hi)), prec=prec)
    return ((finite+I(-tail, tail))/normalizer(prec=prec, bits=bits)).round_out(bits)


@lru_cache(maxsize=1024)
def _point_derivative(order, s, prec, bits):
    return kernel_derivative(order, s, prec=prec, bits=bits)


def covariance(p, alpha, q, beta, *, prec=80, bits=192):
    """Cov(D^alpha f(p),D^beta f(q)); second-argument derivative sign retained."""
    if len(p) != 2 or len(q) != 2 or len(alpha) != 2 or len(beta) != 2:
        raise ValueError('two-axis points and derivative multi-indices required')
    if any(type(n) is not int or n < 0 for n in (*alpha, *beta)):
        raise ValueError('nonnegative integer derivative indices required')
    orders = tuple(alpha[i]+beta[i] for i in range(2))
    if any(n > 4 for n in orders):
        raise ValueError('each kernel derivative order must be at most four')
    displacement = tuple(_rational(p[i])-_rational(q[i]) for i in range(2))
    value = I.exact((-1)**sum(beta))
    for n, s in zip(orders, displacement):
        value *= _point_derivative(n, s, prec, bits)
    return value.round_out(bits)


def source_binding():
    """Read and hash fixed source bytes only; no archive member is executed."""
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError('RN5 archive source identity mismatch')
    document = MIRROR/'RN5_NEAR_MOMENT_REPAIR.md'
    if hashlib.sha256(document.read_bytes()).hexdigest() != RN5_SHA:
        raise ValueError('RN5 mathematical source identity mismatch')
    with ZipFile(ARCHIVE) as archive:
        for name, digest in MEMBERS.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('RN5 archive member identity mismatch: '+name)
    return {'archive_sha256': ARCHIVE_SHA, 'member_sha256': dict(MEMBERS),
            'source_id': RN5_ID, 'source_sha256': RN5_SHA,
            'normalization': 'K(x)=k(x1)k(x2); k=sum_j exp(-(s+24j)^2/2)/sum_j exp(-(24j)^2/2)',
            'scope': 'fixed r=1/20,b=6/5,axis; one rational station, full centered mark interval'}


def nine_pin_blocks(point=(F(1), F(1)), *, prec=80, bits=192):
    """Assemble all 18 raw coordinates directly from the normalized kernel."""
    _parameters(prec, bits)
    if len(point) != 2:
        raise ValueError('two-dimensional rational point required')
    point = tuple(map(_rational, point))
    if any(abs(x) > 17 for x in point):
        raise ValueError('point coordinates must lie in [-17,17]')
    binding = source_binding()
    points = ((-R/2, F(0)), (R/2, F(0)), point)
    if point in points[:2]:
        raise ValueError('moving station collides with a fixed pin')
    labels = ('M', 'S', 'y')
    observations = [(p, jet) for p in points for jet in JETS]
    targets = [(p, jet) for p in points for jet in HJETS]
    coordinates = observations+targets
    matrix = [[None]*18 for _ in range(18)]
    for i, (p, a) in enumerate(coordinates):
        for j in range(i+1):
            q, b = coordinates[j]
            matrix[i][j] = matrix[j][i] = covariance(p, a, q, b, prec=prec, bits=bits)
    raw = tuple(tuple(row) for row in matrix)
    law = f'RN5:SIDE24:r=1/20:b=6/5:y=({point[0]},{point[1]}):nine-pins:centered-mark'
    return {'conditioning_covariance': tuple(tuple(row[:9]) for row in raw[:9]),
            'target_conditioning_covariance': tuple(tuple(row[:9]) for row in raw[9:]),
            'target_covariance': tuple(tuple(row[9:]) for row in raw[9:]),
            'conditioning_intercept': (HEIGHT, F(0), F(0), HEIGHT-ELL, F(0), F(0), MID, F(0), F(0)),
            'conditioning_slope': (F(0),)*6+(F(1), F(0), F(0)),
            'conditioning_order': tuple(f'{p}:{j}' for p in labels for j in ('f', 'fx', 'fy')),
            'target_order': tuple(f'{p}:{j}' for p in labels for j in ('xx', 'yy', 'xy')),
            'covariance_law': law, 'mean_law': law, 'conditioning_law': law,
            'mark_domain': MARK_DOMAIN, 'point': point, 'raw_covariance': raw,
            'source_binding': binding, 'round_bits': bits, 'precision_decimal_digits': prec}


def point_laws(point=(F(1), F(1)), *, bits=192):
    """Condition the actual source-law blocks and expose three marginal laws.

    Only observation and marginal Hessian pivots are required. Mutually
    correlated Hessian blocks are retained; no joint-Hessian SPD is imposed.
    """
    from research.rn.conditioning import condition_gaussian, interval_ldlt
    prec = min(400, (bits+2)//3+24)
    block = nine_pin_blocks(point, prec=prec, bits=bits)
    result = condition_gaussian(block['conditioning_covariance'],
        block['target_conditioning_covariance'], block['target_covariance'],
        block['conditioning_intercept'], block['conditioning_slope'],
        conditioning_order=block['conditioning_order'], target_order=block['target_order'],
        covariance_law=block['covariance_law'], mean_law=block['mean_law'],
        conditioning_law=block['conditioning_law'], mark_domain=MARK_DOMAIN, round_bits=bits)
    context = Context(RN5_ID, RN5_SHA, 2, ('xx', 'yy', 'xy'),
                      'SIDE24 normalized K(0)=1; original Hessian units',
                      block['conditioning_law'], MARK_DOMAIN)
    laws, marginal_pivots = {}, {}
    for label, offset in (('M', 0), ('S', 3), ('y', 6)):
        cov = tuple(tuple(result.covariance[i][j] for j in range(offset, offset+3))
                    for i in range(offset, offset+3))
        marginal_pivots[label] = interval_ldlt(cov, round_bits=bits).pivots
        laws[label] = FamilyLaw(context, result.intercept[offset:offset+3],
                               result.slope[offset:offset+3], cov,
                               context.conditioned_law, context.conditioned_law)
    return {**block, 'laws': laws, 'pivot_intervals': result.pivots,
            'marginal_pivot_intervals': marginal_pivots,
            'joint_covariance': result.covariance, 'joint_intercept': result.intercept,
            'joint_slope': result.slope, 'conditioning_gain': result.gain,
            'authority': 'NONE', 'independence_credit': 0,
            'field_certified': False, 'original_prize_closed': False,
            'spatial_cover_certified': False, 'scientific_status_changed': False}
