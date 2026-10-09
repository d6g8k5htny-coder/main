"""Fraction certificates for the frozen two-control cusp population.

The critical-height selector uses the consumed global-selection proof. This
module does not construct the surgery field or run an independent H0 selector.
"""
from fractions import Fraction as Q
from pathlib import Path
import sys
import time

_INTERVAL_PATH = Path(__file__).resolve().parents[2] / 'periodic_h0'
if str(_INTERVAL_PATH) not in sys.path:
    sys.path.insert(0, str(_INTERVAL_PATH))
from finite_certificate import sqrt_bounds
from gaussian_tail import exp_neg_bounds
from mean_adaptive_confidence import log_bounds

BITS = 128
SCALE = 1 << BITS
CORE_PANELS = 1 << 16
TAIL_PANELS = 512
MAX_EVALUATIONS = 80000
A = Q(1, 1 << 16)
U0, V0 = 3 * A ** 2, 2 * A ** 3
WIDTH_LIMIT = Q(1, 1 << 16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value, nonnegative=False, positive=False):
    require(type(value) is Q, 'Mathematical values must be exact Fractions')
    require(not nonnegative or value >= 0, 'Nonnegative Fraction required')
    require(not positive or value > 0, 'Positive Fraction required')
    return value


def interval(value, nonnegative=False, unit=False):
    require(type(value) in (tuple, list) and len(value) == 2,
            'Two ordered Fraction endpoints required')
    lo, hi = (rational(x) for x in value)
    require(lo <= hi, 'Reversed interval')
    require(not nonnegative or lo >= 0, 'Nonnegative interval required')
    require(not unit or 0 <= lo <= hi <= 1, 'Probability interval required')
    return lo, hi


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _subtract(a, b):
    return a[0] - b[1], a[1] - b[0]


def _scale(a, weight):
    require(weight >= 0, 'Positive interval weight required')
    return a[0] * weight, a[1] * weight


def _multiply(a, b):
    require(a[0] >= 0 and b[0] >= 0, 'Nonnegative multiplication required')
    return a[0] * b[0], a[1] * b[1]


def _reciprocal(a):
    require(a[0] > 0, 'Strictly positive reciprocal required')
    return 1 / a[1], 1 / a[0]


def _sqrt(a):
    require(a[0] >= 0, 'Nonnegative square root required')
    return sqrt_bounds(a[0], BITS)[0], sqrt_bounds(a[1], BITS)[1]


def _outward(a):
    lo, hi = a
    lower, upper = lo * SCALE, hi * SCALE
    return (Q(lower.numerator // lower.denominator, SCALE),
            Q(-((-upper.numerator) // upper.denominator), SCALE))


def quarter_power_bounds(value):
    rational(value, nonnegative=True)
    return _sqrt(_sqrt((value, value)))


def cube_root_bounds(value):
    rational(value, nonnegative=True)
    if value == 0:
        return Q(0), Q(0)
    # Fixed 128 bisections on a enclosing dyadic initial bracket.
    lo, hi = Q(0), Q(1)
    while hi ** 3 < value:
        hi *= 2
    if hi ** 3 == value:
        return hi, hi
    for _ in range(BITS):
        mid = (lo + hi) / 2
        if mid ** 3 == value:
            return mid, mid
        if mid ** 3 < value:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _polynomial(s, u, v):
    return -s ** 4 / 4 + u * s ** 2 / 2 + v * s


def critical_fixture(a, h, sign=1):
    rational(a, positive=True)
    rational(h, nonnegative=True)
    require(type(sign) is int and sign in (-1, 1), 'Reflection sign must be +/-1')
    u, v = 3 * a ** 2 + h ** 2, sign * 2 * a * (a ** 2 - h ** 2)
    roots = tuple(sorted(sign * s for s in (-a - h, -a + h, 2 * a)))
    heights = tuple(_polynomial(s, u, v) for s in roots)
    curvatures = tuple(u - 3 * s ** 2 for s in roots)
    discriminant = 4 * u ** 3 - 27 * v ** 2
    stationary = all(-s ** 3 + u * s + v == 0 for s in roots)
    require(stationary, 'Root-control witness is not stationary')
    status, count, lifetime, side = 'FINITE_BAR', 1, Q(0), None
    if discriminant == 0:
        status, count = 'EXCLUDED_DISCRIMINANT', 0
    elif heights[0] == heights[2]:
        status, count = 'EXCLUDED_TIE', 0
    else:
        require(discriminant > 0 and curvatures[0] < 0 < curvatures[1]
                and curvatures[2] < 0, 'Invalid three-critical-point witness')
        index = 0 if heights[0] < heights[2] else 2
        side = 'left' if index == 0 else 'right'
        lifetime = heights[index] - heights[1]
        require(lifetime > 0, 'Younger maximum must exceed saddle height')
    return {'a': a, 'h': h, 'sign': sign, 'u': u, 'v': v,
            'roots': roots, 'critical_heights': heights, 'curvatures': curvatures,
            'hessian_signs': tuple((x > 0) - (x < 0) for x in curvatures),
            'discriminant': discriminant, 'stationary': stationary,
            'rectangle': -U0 < u < U0 and -V0 < v < V0,
            'status': status, 'count': count, 'younger_side': side,
            'lifetime': lifetime}


def guard_witnesses():
    r, r1 = Q(1, 8), Q(1, 32)
    a_star, delta, g_star = r1 / 4, r1 ** 4 / 64, r1 ** 3 / 64
    b0, d0, b1, d1 = r1 ** 2 / 2, r1, 33 * r1, Q(65)
    gradient, height = b1 * U0 + d1 * V0, b0 * U0 + d0 * V0
    gates = {'chart': r ** 2 < Q(1, 9),
             'root_u': U0 < a_star ** 2 / 4,
             'root_v': V0 < a_star ** 3 / 4,
             'wedge_coverage': 27 * V0 ** 2 >= 4 * U0 ** 3,
             'gradient': gradient == Q(101441, 1 << 47)
             and gradient < Q(1, 1 << 22) <= g_star / 2,
             'height': height == Q(1537, 1 << 52)
             and height < Q(1, 1 << 28) == delta / 4}
    return {'status': 'PASS_RATIONAL_GUARDS' if all(gates.values()) else 'FAIL_IMPLEMENTATION',
            'R': r, 'r1': r1, 'a_star': a_star, 'delta': delta,
            'g_star_lower': g_star, 'B0_upper': b0, 'D0_upper': d0,
            'B1_upper': b1, 'D1_upper': d1, 'u0': U0, 'v0': V0,
            'gradient_perturbation': gradient, 'height_perturbation': height,
            'gates': gates, 'analytic_cutoff_derivative_premise': 'REVIEW_REQUIRED',
            'complete_surgery_field': 'NOT_RUN', 'independent_global_selector': 'NOT_RUN'}


def coefficient_envelopes(b, e):
    b, e = interval(b), interval(e, nonnegative=True)
    return {'inner': (b[1] - e[0], b[0] + e[0]),
            'outer': (b[0] - e[1], b[1] + e[1])}


def interval_distance(a, b):
    a, b = interval(a), interval(b)
    return max(Q(0), a[0] - b[1], b[0] - a[1])


def classify_coefficient(observed, b, e):
    observed = interval(observed)
    envelopes = coefficient_envelopes(b, e)
    inner, outer = envelopes['inner'], envelopes['outer']
    if observed[1] < outer[0] or observed[0] > outer[1]:
        return 'FALSIFIED'
    if observed[0] >= inner[0] and observed[1] <= inner[1]:
        return 'PASS'
    return 'INCONCLUSIVE_PRECISION'


def classify_void(observed, target):
    observed, target = interval(observed, unit=True), interval(target, unit=True)
    if observed[1] - observed[0] > WIDTH_LIMIT or target[1] - target[0] > WIDTH_LIMIT:
        return 'INCONCLUSIVE_PRECISION'
    if max(abs(observed[0] - target[1]), abs(observed[1] - target[0])) < Q(1, 8192):
        return 'PASS'
    if interval_distance(observed, target) > Q(1, 8192):
        return 'FALSIFIED'
    return 'INCONCLUSIVE_PRECISION'


def rectangle_mass_control(observed):
    observed = interval(observed, unit=True)
    return observed == (Q(1, 5), Q(1, 5))


def root_bracket(tau):
    rational(tau, positive=True)
    require(tau < Q(9, 4), 'Split root requires tau below support endpoint')
    lo, hi = Q(0), Q(1)
    for _ in range(96):
        mid = (lo + hi) / 2
        # Cross multiplication retains exact sign, including an exact dyadic root.
        if 36 * mid ** 3 <= tau * (3 + mid ** 2) ** 2:
            lo = mid
        else:
            hi = mid
    require(hi - lo <= Q(1, 1 << 96), 'Split bracket width failed')
    require(36 * lo ** 3 <= tau * (3 + lo ** 2) ** 2
            and 36 * hi ** 3 >= tau * (3 + hi ** 2) ** 2,
            'Split root is not enclosed')
    return lo, hi


def cdf_mass(tau):
    rational(tau, nonnegative=True)
    if tau == 0:
        return Q(0), Q(0)
    if tau >= Q(9, 4):
        return Q(1, 5), Q(1, 5)
    s = root_bracket(tau)
    require(s[0] > 0, 'Split root positivity failed')
    square = s[0] ** 2, s[1] ** 2
    base = _add((Q(1), Q(1)), _scale(square, Q(1, 3)))
    denominator = _multiply(base, _sqrt(base))
    first = _subtract((Q(3), Q(3)), _scale(
        _multiply(_subtract((Q(1), Q(1)), square), _reciprocal(denominator)), Q(3)))
    quarter = _sqrt(_sqrt(s))
    inverse_seventh = _reciprocal((quarter[0] ** 7, quarter[1] ** 7))
    second = _subtract(_scale(_subtract(inverse_seventh, (Q(1), Q(1))), Q(36, 7)),
                       _scale(_subtract((Q(1), Q(1)), quarter), Q(4)))
    coefficient = _scale(quarter_power_bounds(tau / 4), tau / 4)
    result = _outward(_scale(_add(first, _multiply(coefficient, second)), Q(1, 15)))
    require(0 <= result[0] <= result[1] <= Q(1, 5), 'CDF probability enclosure failed')
    return result


def coefficient_interval():
    return _scale(_reciprocal(cube_root_bounds(Q(2))), Q(9, 28))


def _schedule(j):
    require(type(j) is int and 1 <= j <= 8, 'Declared schedule requires integer j in [1,8]')
    return Q(1, 1 << (3 * j)), 4 ** j


def error_interval(j):
    _, n = _schedule(j)
    return _add((Q(9, 8 * n), Q(9, 8 * n)),
                quarter_power_bounds(Q(1, 1 << (7 * j))))


def void_interval(mass, n):
    mass = interval(mass, unit=True)
    require(type(n) is int and n > 0, 'Positive strict integer copy count required')
    # Monotone log/exp avoids multi-million-bit exact rational powers.
    def endpoint(m):
        if m == 0:
            return Q(1), Q(1)
        if m == 1:
            return Q(0), Q(0)
        lo, hi = log_bounds(1 - m, bits=BITS)
        # Round the exponent outward before feeding the shared series evaluator;
        # this bounds denominator size and includes the rounding in final width.
        xlo, xhi = _outward((-n * hi, -n * lo))
        require(0 <= xlo <= xhi <= 4096, 'Void exponential domain exhausted')
        return exp_neg_bounds(xhi)[0], exp_neg_bounds(xlo)[1]
    return endpoint(mass[1])[0], endpoint(mass[0])[1]


def poisson_target(b=None):
    b = interval(coefficient_interval() if b is None else b, nonnegative=True)
    require(b[1] <= 4096, 'Poisson exponential domain exhausted')
    return exp_neg_bounds(b[1])[0], exp_neg_bounds(b[0])[1]


class _BudgetExhausted(Exception):
    pass


def direct_mass(j, max_evaluations=MAX_EVALUATIONS, deadline=None):
    """Independent control-measure quadrature with fixed panels and exact widths.

    Cached unique nodes count once. Positive lower and upper uses both count
    toward arithmetic inflation. No partial quadrature is a mass enclosure.
    """
    tau, n = _schedule(j)
    require(type(max_evaluations) is int and 0 <= max_evaluations <= MAX_EVALUATIONS,
            'Evaluation control may only reduce the frozen 80000 budget')
    require(deadline is None or type(deadline) in (int, float), 'Monotonic deadline required')
    root = root_bracket(tau)
    rlo, rhi = root
    gates = {'root_width': rhi - rlo <= Q(1, 1 << 96),
             'root_positive': rlo > 0, 'root_upper': rhi < Q(1, 2),
             'root_auxiliary_lower': rhi > Q(1, 1 << 9),
             'core_scale': n * rlo ** 2 <= Q(1, 2),
             # Derivative numerator on theta^2 in [0,rlo^2].
             'core_monotonicity': 27 - 45 * rlo ** 2 + 2 * rlo ** 4 > 0,
             'core_range': Q(3, 5) * n * rlo ** 2 <= Q(3, 10)}
    result = {'j': j, 'tau': tau, 'n': n, 'root_bracket': root,
              'status': 'INCONCLUSIVE_BUDGET', 'evaluations': 0,
              'interval': None, 'normalized_interval': None,
              'arithmetic_width': Q(0), 'discretization_width': None,
              'sliver_width': Q(3, 5) * n * (rhi - rlo), 'gates': gates,
              'partial': {'core_nodes': 0, 'tail_nodes': 0, 'completed_octaves': 0}}
    if not all(gates.values()):
        result['status'] = 'INCONCLUSIVE_PRECISION'
        return result
    if max_evaluations == 0 or (deadline is not None and time.monotonic() >= deadline):
        return result
    tau_power = _scale(quarter_power_bounds(tau), tau)
    four_power = _scale(quarter_power_bounds(Q(4)), Q(4))
    rquarter = quarter_power_bounds(rhi)
    rinverse = _reciprocal((rquarter[0] ** 7, rquarter[1] ** 7))
    coefficient = _outward(_scale(_multiply(_multiply(tau_power, _reciprocal(four_power)),
                                            rinverse), Q(n, 15)))
    gates['tail_coefficient'] = coefficient[1] <= Q(1, 30)
    # f'' = C*(1485/16*t^-19/4 - 21/16*rhi^2*t^-11/4).
    gates['tail_convexity'] = 1485 - 21 > 0
    gates['tail_second_derivative'] = coefficient[1] * Q(1485, 16) <= Q(99, 32)
    if not all(gates.values()):
        result['status'] = 'INCONCLUSIVE_PRECISION'
        return result
    caches = {'core': {}, 'tail': {}}

    def evaluate(route, node):
        cache = caches[route]
        if node in cache:
            return cache[node]
        if result['evaluations'] >= max_evaluations or (
                deadline is not None and time.monotonic() >= deadline):
            raise _BudgetExhausted
        if route == 'core':
            theta2 = (rlo * node) ** 2
            base = Q(3) / (3 + theta2)
            value = _scale(sqrt_bounds(base, BITS),
                           Q(n, 15) * rlo ** 2 * node * (9 - theta2) * base ** 2)
        else:
            quarter = quarter_power_bounds(node)
            inverse = _reciprocal((quarter[0] ** 11, quarter[1] ** 11))
            value = _scale(_multiply(coefficient, inverse), 9 - (rhi * node) ** 2)
        value = _outward(value)
        require(0 <= value[0] <= value[1], 'Direct point enclosure positivity failed')
        cache[node] = tuple(int(x * SCALE) for x in value)
        result['evaluations'] += 1
        result['partial'][route + '_nodes'] += 1
        return cache[node]

    core_low = core_upper_floor = core_upper_ceiling = 0
    core_lower_width = core_upper_width = 0
    tail_low = tail_upper_floor = tail_upper_ceiling = Q(0)
    tail_lower_width = tail_upper_width = Q(0)
    try:
        # Darboux lower and upper use all 65537 nodes, cached once each.
        for index in range(CORE_PANELS + 1):
            lo, hi = evaluate('core', Q(index, CORE_PANELS))
            if index < CORE_PANELS:
                core_low += lo
                core_lower_width += hi - lo
            if index > 0:
                core_upper_floor += lo
                core_upper_ceiling += hi
                core_upper_width += hi - lo
        left, finish = Q(1), 1 / rhi
        while left < finish:
            right = min(2 * left, finish)
            step = (right - left) / TAIL_PANELS
            midpoint_lower = midpoint_width = 0
            endpoint_floor = endpoint_ceiling = endpoint_width = 0
            for index in range(TAIL_PANELS + 1):
                lo, hi = evaluate('tail', left + step * index)
                weight = 1 if index in (0, TAIL_PANELS) else 2
                endpoint_floor += weight * lo
                endpoint_ceiling += weight * hi
                endpoint_width += weight * (hi - lo)
                if index < TAIL_PANELS:
                    lo, hi = evaluate('tail', left + step * Q(2 * index + 1, 2))
                    midpoint_lower += lo
                    midpoint_width += hi - lo
            tail_low += step * Q(midpoint_lower, SCALE)
            tail_upper_floor += step * Q(endpoint_floor, 2 * SCALE)
            tail_upper_ceiling += step * Q(endpoint_ceiling, 2 * SCALE)
            tail_lower_width += step * Q(midpoint_width, SCALE)
            tail_upper_width += step * Q(endpoint_width, 2 * SCALE)
            result['partial']['completed_octaves'] += 1
            left = right
    except _BudgetExhausted:
        result['partial']['core_lower_sum'] = Q(core_low, CORE_PANELS * SCALE)
        result['partial']['tail_lower_sum_completed_octaves'] = tail_low
        result['arithmetic_width'] = (Q(core_lower_width + core_upper_width,
                                       CORE_PANELS * SCALE)
                                      + tail_lower_width + tail_upper_width)
        return result
    core_lower = Q(core_low, CORE_PANELS * SCALE)
    core_upper = Q(core_upper_ceiling, CORE_PANELS * SCALE)
    core_floor_upper = Q(core_upper_floor, CORE_PANELS * SCALE)
    arithmetic = Q(core_lower_width + core_upper_width, CORE_PANELS * SCALE)
    arithmetic += tail_lower_width + tail_upper_width
    core_discretization = core_floor_upper - core_lower
    tail_discretization = tail_upper_floor - tail_low
    sliver = result['sliver_width']
    normalized = core_lower + tail_low, core_upper + tail_upper_ceiling + sliver
    gates.update({'point_widths': all(Q(hi - lo, SCALE) <= Q(1, 1 << 40)
                                     for cache in caches.values() for lo, hi in cache.values()),
                  'core_discretization': Q(0) <= core_discretization <= Q(3, 10 * CORE_PANELS),
                  'tail_discretization': Q(0) <= tail_discretization <= Q(297, 512 * 512 ** 2),
                  'arithmetic_width': arithmetic <= Q(1, 1 << 22),
                  'normalized_width': normalized[1] - normalized[0] < Q(1, 1 << 17),
                  'evaluation_budget': result['evaluations'] <= max_evaluations,
                  'runtime_budget': deadline is None or time.monotonic() < deadline})
    result.update({'status': 'CERTIFIED' if all(gates.values()) else 'INCONCLUSIVE_PRECISION',
                   'interval': _scale(normalized, Q(1, n)), 'normalized_interval': normalized,
                   'arithmetic_width': arithmetic,
                   'discretization_width': core_discretization + tail_discretization,
                   'arithmetic_components': {
                       'core_lower': Q(core_lower_width, CORE_PANELS * SCALE),
                       'core_upper': Q(core_upper_width, CORE_PANELS * SCALE),
                       'tail_lower': tail_lower_width, 'tail_upper': tail_upper_width},
                   'discretization_components': {'core': core_discretization,
                                                 'tail': tail_discretization},
                   'core_interval': (core_lower, core_upper),
                   'tail_interval': (tail_low, tail_upper_ceiling),
                   'tail_coefficient_interval': coefficient,
                   'normalized_width': normalized[1] - normalized[0]})
    require(sum(result['arithmetic_components'].values(), Q(0)) == arithmetic,
            'Arithmetic inflation replay failed')
    return result


def negative_controls():
    a = A / 4
    controls = [critical_fixture(a, a * theta, sign)
                for theta in (Q(1, 4), Q(1, 2), Q(3, 4), Q(0), Q(1), Q(3), Q(2))
                for sign in (1, -1)]
    positive = [c for c in controls if c['h'] / a in (Q(1, 4), Q(1, 2), Q(3, 4))]
    gaps = all(c['lifetime'] == 4 * a * c['h'] ** 3 for c in positive)
    elder_heights = all(c['critical_heights'][2] - c['critical_heights'][0]
                       == c['sign'] * a ** 4 * (1 - c['h'] / a) * (3 + c['h'] / a) ** 3 / 4
                       for c in positive)
    adversarial = [c for c in controls if c['h'] == 2 * a]
    saturation = cdf_mass(Q(9, 4))
    gates = {'all_stationary_and_inside_rectangle': all(c['stationary'] and c['rectangle'] for c in controls),
             'chart_gaps': gaps, 'outer_height_identity': elder_heights,
             'actual_theta2_gap': all(c['lifetime'] == 3 * a ** 4 / 4 for c in adversarial),
             'reject_fixed_left': all(c['lifetime'] != 32 * a ** 4 for c in adversarial),
             'boundary_exclusion': all(c['count'] == 0 for c in controls if c['h'] / a in (0, 1, 3)),
             'whole_rectangle_mass': rectangle_mass_control(saturation),
             'missing_sign_detected': not rectangle_mass_control((Q(1, 10), Q(1, 10))),
             'conditioning_detected': not rectangle_mass_control((Q(1), Q(1)))}
    return {'status': 'PASS' if all(gates.values()) else 'FAIL_IMPLEMENTATION',
            'controls': controls, 'gates': gates, 'whole_rectangle_mass': saturation,
            'missing_sign_mass': (Q(1, 10), Q(1, 10)), 'conditioned_mass': (Q(1), Q(1))}
