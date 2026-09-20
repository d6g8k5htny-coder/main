"""Uniform RN5 rectangle laws by six-pin conditioning and L2 transport.

This author-side cell enclosure imports H3 but re-proves no H3 assertion.
It supplies no complete annulus cover, remote budget, all-r theorem,
scientific promotion or organizational independence. See RN_SIDE24_CELL.md.
"""
from fractions import Fraction as F
from functools import lru_cache
from dataclasses import asdict

from engine.operations.rn_applicability import Context, RN5_ID, RN5_SHA
from engine.operations.rn_family_applicability import FamilyLaw
from research.cover import Box
from research.interval import Interval as I, exp, sqrt
from research.rn import side24
from research.rn.conditioning import (ConditioningInconclusive, ConditioningResourceLimit,
                                      condition_gaussian, interval_ldlt)
from research.rn.side24_density import gradient_density, height_window_cap, imported_h3_floor


def _options(bits):
    if type(bits) is not int or not 128 <= bits <= 1024:
        raise ValueError('rounding must use an integer in 128..1024 bits')
    return min(400, (bits+2)//3+24)


def _box(box):
    if not isinstance(box, Box):
        raise TypeError('an exact research.cover.Box is required')
    values = (box.u0, box.u1, box.v0, box.v1)
    for x in values:
        if type(x) is not F or max(abs(x.numerator).bit_length(), x.denominator.bit_length()) > 4096:
            raise ValueError('box endpoint must be an exact Fraction within 4096 bits')
        if abs(x) > 17:
            raise ValueError('box coordinates must lie in [-17,17]')
    if box.du() > 2 or box.dv() > 2:
        raise ValueError('cell side length exceeds bounded pilot width two')
    for x in (-side24.R/2, side24.R/2):
        if box.u0 <= x <= box.u1 and box.v0 <= 0 <= box.v1:
            raise ConditioningInconclusive('cell contains a fixed pin')
    return ((box.u0+box.u1)/2, (box.v0+box.v1)/2), (box.du()/2, box.dv()/2)


def _dot(a, b, bits):
    result = I.exact(0)
    for x, y in zip(a, b):
        result = (result+(x*y).round_out(bits)).round_out(bits)
    return result


def sixth_kernel_derivative_zero(*, bits=192):
    """Normalized k^(6)(0), including all images, not the planar value -15.

    He_6(x)=x^6-15x^4+45x^2-15; coefficient absolute sum is 76.
    For |j|>=2, use 2*76*(24j)^6*exp(-(24j)^2/2) and its decreasing
    consecutive ratio, bounded by (3/2)^6 exp(-1440) at j=2.
    """
    prec = _options(bits)
    finite = I.exact(-15)+2*(24**6-15*24**4+45*24**2-15)*exp(I.exact(-288), prec)
    ratio = F(3, 2)**6*exp(I.exact(-1440), prec)
    if ratio.hi >= 1:
        raise ConditioningInconclusive('sixth-derivative image tail does not contract')
    tail = (2*76*48**6*exp(I.exact(-1152), prec)/(1-ratio)).hi
    return ((finite+I(-tail, tail))/side24.normalizer(prec=prec, bits=bits)).round_out(bits)


def derivative_variance(alpha, *, bits=192):
    """Stationary variance of derivative alpha, total order at most three."""
    _options(bits)
    if (not isinstance(alpha, tuple) or len(alpha) != 2 or
        any(type(n) is not int or n < 0 for n in alpha) or sum(alpha) > 3):
        raise ValueError('nonnegative two-axis derivative of total order at most three required')
    return _derivative_variance(alpha, bits)


@lru_cache(maxsize=128)
def _derivative_variance(alpha, bits):
    prec = _options(bits)
    value = I.exact((-1)**sum(alpha))
    for n in alpha:
        derivative = (sixth_kernel_derivative_zero(bits=bits) if n == 3
                      else side24.kernel_derivative(2*n, 0, prec=prec, bits=bits))
        value = (value*derivative).round_out(bits)
    if value.lo <= 0:
        raise ConditioningInconclusive('stationary derivative variance not positive')
    return value


def _center_law(center, bits):
    block = side24.nine_pin_blocks(center, prec=_options(bits), bits=bits)
    raw = block['raw_covariance']
    # Target order: fixed pair Hessians, moving three-jet, moving Hessian.
    indices = tuple(range(9, 15))+tuple(range(6, 9))+tuple(range(15, 18))
    all_names = block['conditioning_order']+block['target_order']
    target_order = tuple(all_names[i] for i in indices)
    label = block['conditioning_law'].replace(':nine-pins:centered-mark', ':six-pins')
    result = condition_gaussian(tuple(tuple(row[:6]) for row in raw[:6]),
        tuple(tuple(raw[i][j] for j in range(6)) for i in indices),
        tuple(tuple(raw[i][j] for j in indices) for i in indices),
        block['conditioning_intercept'][:6], (F(0),)*6,
        conditioning_order=block['conditioning_order'][:6], target_order=target_order,
        covariance_law=label, mean_law=label, conditioning_law=label,
        mark_domain=(F(0), F(0)), round_bits=bits)
    # Q=c^T A^-1 c as a positive LDL square sum, no cancellation.
    whitened, energy = [], I.exact(0)
    for i, pin in enumerate(block['conditioning_intercept'][:6]):
        w = (I.exact(pin)-_dot(result.factorization.lower[i][:i], whitened, bits)).round_out(bits)
        whitened.append(w)
        energy = (energy+w**2/result.pivots[i]).round_out(bits)
    return block, result, energy


def transported_law(box, *, bits=192):
    """Enclose the actual 12-dimensional six-pin law for every y in box.

    For moving derivative a, its centered conditional L2 increment is at most
    eps_a=sum h_j sqrt(Var D^(a+e_j)f). Projection onto the six fixed pins
    contracts variance. Its conditional mean increment is at most sqrt(Q) eps_a.
    The covariance error follows from Cauchy-Schwarz on the two increments.
    """
    prec = _options(bits)
    center, radii = _box(box)
    block, central, energy = _center_law(center, bits)
    eps = [F(0)]*6
    derivative_bounds = {}
    for alpha in side24.JETS+side24.HJETS:
        error = I.exact(0)
        for axis in (0, 1):
            beta = tuple(n+int(k == axis) for k, n in enumerate(alpha))
            variance = derivative_variance(beta, bits=bits)
            derivative_bounds[str(beta)] = variance
            error = (error+radii[axis]*sqrt(variance, prec)).round_out(bits)
        eps.append(error.hi)
    energy_root = sqrt(energy, prec).round_out(bits).hi
    standard_deviations = []
    for i in range(12):
        diagonal = central.covariance[i][i]
        if diagonal.hi < 0:
            raise ConditioningInconclusive('central covariance has negative upper variance')
        standard_deviations.append(sqrt(I(0, diagonal.hi), prec).round_out(bits).hi)
    mean = tuple((central.intercept[i]+I(-energy_root*eps[i], energy_root*eps[i])).round_out(bits)
                 for i in range(12))
    covariance = []
    errors = []
    for i in range(12):
        row, error_row = [], []
        for j in range(12):
            error = standard_deviations[i]*eps[j]+standard_deviations[j]*eps[i]+eps[i]*eps[j]
            error_row.append(error)
            row.append((central.covariance[i][j]+I(-error, error)).round_out(bits))
        covariance.append(tuple(row))
        errors.append(tuple(error_row))
    return {'box': box, 'center': center, 'halfwidths': radii, 'round_bits': bits,
            'source_binding': {**block['source_binding'],
                               'scope': 'fixed r=1/20,b=6/5,axis; every spatial point in the declared rectangle'},
            'six_pin_order': block['conditioning_order'][:6],
            'six_pin_values': block['conditioning_intercept'][:6], 'six_pin_pivots': central.pivots,
            'target_order': central.target_order, 'mean': mean, 'covariance': tuple(covariance),
            'center_mean': central.intercept, 'center_covariance': central.covariance,
            'conditioning_energy': energy, 'l2_increment_upper': tuple(eps),
            'stationary_derivative_variances': derivative_bounds,
            'center_standard_deviation_upper': tuple(standard_deviations),
            'covariance_error_upper': tuple(errors), 'imported_h3_floor': imported_h3_floor()}


def cell_laws(box, *, bits=192):
    """Three full-mark Hessian families and density cap, uniform on one cell."""
    transport = transported_law(box, bits=bits)
    mean, cov = transport['mean'], transport['covariance']
    yi, hi = (6, 7, 8), (0, 1, 2, 3, 4, 5, 9, 10, 11)
    ymean = tuple(mean[i] for i in yi)
    ycov = tuple(tuple(cov[i][j] for j in yi) for i in yi)
    cross = tuple(tuple(cov[i][j] for j in yi) for i in hi)
    hcov = tuple(tuple(cov[i][j] for j in hi) for i in hi)
    names = transport['target_order']
    law = f'RN5:SIDE24:r=1/20:b=6/5:cell=({box.u0},{box.u1},{box.v0},{box.v1}):nine-pins:centered-mark'
    conditional = condition_gaussian(ycov, cross, hcov, (side24.MID, F(0), F(0)), (F(1), F(0), F(0)),
        conditioning_order=tuple(names[i] for i in yi), target_order=tuple(names[i] for i in hi),
        covariance_law=law, mean_law=law, conditioning_law=law, mark_domain=side24.MARK_DOMAIN,
        prior_conditioning_intercept=ymean, prior_target_intercept=tuple(mean[i] for i in hi), round_bits=bits)
    context = Context(RN5_ID, RN5_SHA, 2, ('xx', 'yy', 'xy'),
                      'SIDE24 normalized K(0)=1; original Hessian units', law, side24.MARK_DOMAIN)
    laws, pivots = {}, {}
    for name, offset in (('M', 0), ('S', 3), ('y', 6)):
        marginal = tuple(tuple(row[offset:offset+3]) for row in conditional.covariance[offset:offset+3])
        pivots[name] = interval_ldlt(marginal, round_bits=bits).pivots
        laws[name] = FamilyLaw(context, conditional.intercept[offset:offset+3],
                               conditional.slope[offset:offset+3], marginal, law, law)
    gcov = tuple(tuple(row[1:]) for row in ycov[1:])
    gradient = gradient_density(ymean[1:], gcov, bits=bits)
    factor = interval_ldlt(gcov, round_bits=bits)
    cg = ycov[0][1:]
    height_mean = (ymean[0]-_dot(cg, factor.solve(ymean[1:]), bits)).round_out(bits)
    height_variance = (ycov[0][0]-_dot(cg, factor.solve(cg), bits)).round_out(bits)
    window = height_window_cap(height_mean, height_variance,
                               (side24.HEIGHT-side24.ELL, side24.HEIGHT), bits=bits)
    density = {**gradient, **window, 'density_mass_upper': gradient['gradient_density'].hi*window['window_mass_upper']}
    return {**transport, 'laws': laws, 'conditioning_law': law,
            'jet_mean': ymean, 'jet_covariance': ycov, 'jet_pivots': conditional.pivots,
            'marginal_pivots': pivots, 'joint_hessian_intercept': conditional.intercept,
            'joint_hessian_slope': conditional.slope, 'joint_hessian_covariance': conditional.covariance,
            'density': density, 'authority': 'NONE', 'independence_credit': 0,
            'field_certified': False, 'spatial_cover_certified': False,
            'scientific_status_changed': False, 'all_small_r_certified': False,
            'h3_reproved': False, 'original_prize_closed': False}


def spatial_cell(box, *, bits=192):
    """Replay three moment witnesses and return one uniform typed-integrand cap.

    This proves a local conditional-on-H3 upper only. The cover controller must
    separately prove a complete partition and account for every boundary cell.
    """
    from research.rn.certificate import ResourceLimit, canonical_bytes, produce, verify_bytes
    result = cell_laws(box, bits=bits)
    source = (side24.MIRROR/'RN5_NEAR_MOMENT_REPAIR.md').read_bytes()
    certificates, replays, caps = {}, {}, {}
    for name, degree in (('M', 4), ('S', 4), ('y', 2)):
        law = result['laws'][name]
        context = asdict(law.context)
        context['order'] = list(context['order'])
        context['domain'] = list(map(str, context['domain']))
        try:
            certificate = produce(law, degree=degree, depth=2)
        except ResourceLimit as error:
            raise ConditioningResourceLimit('cell moment resource limit: '+str(error)) from error
        replay = verify_bytes(canonical_bytes(certificate), source_bytes=source, expected_context=context)
        if not replay['certificate_valid'] or not replay['requested_checks_passed']:
            if replay['outcome'] == 'INCONCLUSIVE':
                raise ConditioningResourceLimit('cell moment resource limit: '+replay['reason'])
            raise ValueError('cell moment witness replay failed: '+name+': '+replay['reason'])
        certificates[name], replays[name] = certificate, replay
        caps[name] = F(certificate['proof']['upper'])
    fourth = caps['M']*caps['S']*caps['y']**2
    holder = sqrt(sqrt(I.exact(fourth), _options(bits)), _options(bits)).round_out(bits)
    if holder.lo < 0 or holder.hi**4 < fourth:
        raise ValueError('cell Holder root enclosure failed')
    upper = (I.exact(holder.hi*result['density']['density_mass_upper'])
             /result['imported_h3_floor']['value']).round_out(bits).hi
    return {**result, 'moment_certificates': certificates, 'moment_replays': replays,
            'moment_caps': caps, 'holder_fourth_power_upper': fourth, 'holder_upper': holder.hi,
            'integrand_upper': upper, 'integrand_range': I(0, upper),
            'spatial_method': 'FIXED_SIX_PIN_CENTERED_L2_INCREMENT_TRANSPORT',
            'quantifier': 'FOR_EVERY_Y_IN_CLOSED_RECTANGLE_AND_EVERY_MARK_IN_FULL_WINDOW'}
