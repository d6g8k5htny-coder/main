"""Exact complete-field definition and physical-coordinate enclosures.

Source implementation after Root's D02 semantic RED. This module does no file,
network, process, sampler, H0, prediction, or acceptance work. The seven public
APIs instantiate the frozen mathematical recipe; runtime qualification is an
external gate. Coordinates s,t below are sine-chart coordinates, not physical y.
"""
from fractions import Fraction as F

A = F(1, 65536)
R = F(1, 8)
R1 = F(1, 32)
B = F(4, 9)
U0 = 3 * A**2
V0 = 2 * A**3
ARITHMETIC_ALLOWANCE = F(1, 2**120)
_METRIC_CONSTANTS = None


def _rational(value, name):
    if type(value) is int:
        return F(value)
    if type(value) is not F:
        raise ValueError(name + ' must be an exact rational')
    return value


def _sign(epsilon):
    if type(epsilon) is not int or epsilon not in (-1, 1):
        raise ValueError('epsilon must be the integer -1 or +1')
    return epsilon


def _bits(bits):
    if type(bits) is not int or bits < 1:
        raise ValueError('bits must be a positive integer')
    return bits


def control(Q, Z, epsilon):
    """Exact control map, including its null boundary controls."""
    Q, Z = _rational(Q, 'Q'), _rational(Z, 'Z')
    epsilon = _sign(epsilon)
    if not (0 <= Q <= 1 and 0 <= Z <= 1):
        raise ValueError('Q and Z must be in [0,1]')
    return {'Q': Q, 'Z': Z, 'epsilon': epsilon,
            'u': 3 * A**2 * Q**2,
            'v': epsilon * 2 * A**3 * Q**3 * (1 - Z**8)}


def _control_values(record):
    if not isinstance(record, dict):
        raise ValueError('control must be an exact control record')
    try:
        expected = control(record['Q'], record['Z'], record['epsilon'])
        u, v = _rational(record['u'], 'u'), _rational(record['v'], 'v')
    except KeyError as error:
        raise ValueError('incomplete control record') from error
    if (u, v) != (expected['u'], expected['v']):
        raise ValueError('control record does not satisfy the frozen map')
    return u, v


def cell_mass(cell, k):
    """Unconditional mass of one dyadic rectangle and one sign."""
    if type(k) is not int or k not in (8, 12, 15, 16):
        raise ValueError('unsupported law exponent')
    if not isinstance(cell, dict):
        raise ValueError('cell must be a dictionary')
    try:
        qa, qb, za, zb = (_rational(cell[key], key)
                          for key in ('Qa', 'Qb', 'Za', 'Zb'))
        _sign(cell['epsilon'])
    except KeyError as error:
        raise ValueError('incomplete control cell') from error
    for endpoint in (qa, qb, za, zb):
        denominator = endpoint.denominator
        if denominator & (denominator - 1):
            raise ValueError('cell endpoints must be dyadic')
    if not (0 <= qa < qb <= 1 and 0 <= za < zb <= 1):
        raise ValueError('cell must have positive ordered widths in [0,1]')
    return (qb**5 - qa**5) * (zb**k - za**k) / 2


def step_bounds(z, bits=128):
    """Enclose the flat step S(z), with exact endpoint and midpoint values.

    For z<1/2, S=e^-q/(1+e^-q), q=1/z-1/(1-z). Range
    reduction makes the alternating exponential series decreasing. Adjacent
    partial sums enclose exp(-r); squaring and the logistic map are monotone.
    The interval width is at most 2^-bits, without floating arithmetic.
    """
    bits = _bits(bits)
    z = _rational(z, 'z')
    if z <= 0:
        return F(0), F(0)
    if z >= 1:
        return F(1), F(1)
    if z == F(1, 2):
        return z, z
    if z > F(1, 2):
        lo, hi = step_bounds(1 - z, bits)
        return 1 - hi, 1 - lo
    q = 1 / z - 1 / (1 - z)
    # e>2 implies exp(-q)<2^-M whenever q>=M.
    M = bits + 4
    if q >= M:
        return F(0), F(1, 2**M)
    r, halvings = q, 0
    while r > F(1, 2):
        r /= 2
        halvings += 1
    target = F(1, 2**(bits + halvings + 4))
    lo, hi, partial, term, n = F(0), F(1), F(1), F(1), 0
    while hi - lo > target:
        n += 1
        term *= r / n
        partial += -term if n % 2 else term
        if n % 2:
            lo = partial
        else:
            hi = partial
    for _ in range(halvings):
        lo, hi = lo * lo, hi * hi
    return lo / (1 + lo), hi / (1 + hi)


def scaled_core_value(x, y, control):
    """Exact A^-4(F-b), only after checking the complete-field core guard."""
    x, y = _rational(x, 'x'), _rational(y, 'y')
    u, v = _control_values(control)
    s, t = A * x, A**2 * y
    if s*s + t*t > R1**2 / 4:
        raise ValueError('scaled node lies outside the exact core')
    return (-x**4 / 4 + (u / A**2) * x*x / 2 + (v / A**3) * x
            - (1 + A**2 * x*x / 4) * y*y)


def _scale_interval(coefficient, interval):
    lo, hi = interval
    return ((coefficient * lo, coefficient * hi) if coefficient >= 0
            else (coefficient * hi, coefficient * lo))


def full_residual_bounds(s, t, control, bits=128):
    """Enclose the complete physical field's centered residual F-b.

    s=sqrt(2/3)sin(pi*y1), t=sqrt(2/9)sin(pi*y2). Their squared
    values cover the principal physical torus; outside the support collar the
    exact seed is -s^2-t^2. No subtraction of nearly equal physical heights
    is used. Inside the support, c and chi are evaluated on every branch.
    """
    bits = _bits(bits)
    s, t = _rational(s, 's'), _rational(t, 't')
    u, v = _control_values(control)
    if s*s > F(2, 3) or t*t > F(2, 9):
        raise ValueError('s,t lie outside the physical sine-coordinate range')
    q = s*s + t*t
    if q >= R**2 / 2:
        return -q, -q
    if q <= R1**2 / 4:
        value = -s**4 / 4 - (1 + s*s / 4) * t*t + u*s*s / 2 + v*s
        return value, value
    if q <= R1**2:
        c = (q / 4, q / 4)
    else:
        step = step_bounds((q - R1**2) / (R**2 / 2 - R1**2), bits)
        c = tuple(q / 4 + (1 - q / 4) * endpoint for endpoint in step)
    if q >= R1**2:
        chi = (F(0), F(0))
    else:
        slo, shi = step_bounds((q - R1**2 / 4) / (3 * R1**2 / 4), bits)
        chi = 1 - shi, 1 - slo
    cterm = _scale_interval(-s*s, c)
    perturbation = _scale_interval(u*s*s / 2 + v*s, chi)
    return (cterm[0] - t*t + perturbation[0],
            cterm[1] - t*t + perturbation[1])


def _atan_bounds(z):
    """96 alternating terms: even partial sum is lower, next term upper."""
    partial, power = F(0), z
    for n in range(96):
        term = power / (2*n + 1)
        partial += term if n % 2 == 0 else -term
        power *= z*z
    return partial, partial + power / 193


def _metric_constants():
    global _METRIC_CONSTANTS
    if _METRIC_CONSTANTS is None:
        lo, hi = F(1), F(2)
        for _ in range(128):
            mid = (lo + hi) / 2
            if mid*mid <= F(3, 2):
                lo = mid
            else:
                hi = mid
        a, b = _atan_bounds(F(1, 5)), _atan_bounds(F(1, 239))
        pi = 16*a[0] - 4*b[1], 16*a[1] - 4*b[0]
        _METRIC_CONSTANTS = ((lo, hi), pi)
    return _METRIC_CONSTANTS


def _asin_positive_bounds(z):
    # 17 positive terms. The omitted coefficient of every z^(2n+1) is <=1.
    partial, central, power = F(0), F(1), z
    for n in range(17):
        partial += central * power / (2*n + 1)
        central *= F(2*n + 1, 2*n + 2)
        power *= z*z
    return partial, partial + power / (1 - z*z)


def _rho_point(x):
    if x < 0:
        lo, hi = _rho_point(-x)
        return -hi, -lo
    if x == 0:
        return F(0), F(0)
    root, pi = _metric_constants()
    zlo, zhi = A*x*root[0], A*x*root[1]
    alo = _asin_positive_bounds(zlo)[0]
    ahi = _asin_positive_bounds(zhi)[1]
    return alo / pi[1], ahi / pi[0]


def _physical_interval(interval, name):
    if not isinstance(interval, (tuple, list)) or len(interval) != 2:
        raise ValueError(name + ' must be an ordered critical bracket')
    lo, hi = (_rational(endpoint, name) for endpoint in interval)
    if not (-3 <= lo <= hi <= 3):
        raise ValueError(name + ' must be ordered inside [-3,3]')
    return (_rho_point(lo)[0] - ARITHMETIC_ALLOWANCE,
            _rho_point(hi)[1] + ARITHMETIC_ALLOWANCE)


def physical_mark(birth, saddle):
    """Map whole critical brackets; separation is in physical torus units.

    Direction is saddle minus birth. The frozen diameter bound (<6A<1/2)
    makes this absolute displacement the shortest torus metric. An overlap
    never manufactures a direction or a resolved physical mark.
    """
    birth = _physical_interval(birth, 'birth')
    saddle = _physical_interval(saddle, 'saddle')
    if birth[1] < saddle[0]:
        direction = 1
        separation = saddle[0] - birth[1], saddle[1] - birth[0]
    elif saddle[1] < birth[0]:
        direction = -1
        separation = birth[0] - saddle[1], birth[1] - saddle[0]
    else:
        direction = 0
        separation = F(0), max(abs(saddle[1] - birth[0]),
                               abs(birth[1] - saddle[0]))
    return {'birth': birth, 'saddle': saddle, 'separation': separation,
            'direction': direction, 'resolved': direction != 0,
            'arithmetic_allowance_per_endpoint': ARITHMETIC_ALLOWANCE}


def realization_descriptor():
    """Return definition/provenance, not an implementation or science verdict."""
    b0, d0 = R1**2 / 2, R1
    b1, d1, g = 35*R1/3, F(67, 3), R1**3 / 16
    return {
        'constants': {'dimension': 2, 'period': F(1),
                      'weights': (F(1, 3), F(1, 9)), 'b': B,
                      'R': R, 'r1': R1, 'A': A, 'u0': U0, 'v0': V0,
                      'a_star': R1/4, 'delta': R1**4/64,
                      'floor_scaled': F(-16), 'integer_scale': 2**80},
        'region_coverage': ('core', 'annulus', 'collar', 'torus_exterior'),
        'whole_field_defined': True,
        'definition': {
            'seed': 'H=cos(2*pi*y1)/3+cos(2*pi*y2)/9; period 1',
            'chart': 's=sqrt(2/3)*sin(pi*y1); t=sqrt(2/9)*sin(pi*y2)',
            'flat_step': 'S=0 at z<=0; S=1 at z>=1; otherwise e^(-1/z)/(e^(-1/z)+e^(-1/(1-z)))',
            'c': 'q/4+(1-q/4)*S((q-r1^2)/(R^2/2-r1^2)); q=s^2+t^2',
            'chi': '1-S((q-r1^2/4)/(3*r1^2/4))',
            'inside_chart': 'F=b-s^2*c(q)-t^2+chi(q)*(u*s^2/2+v*s)',
            'collar_and_exterior': 'F=H when q>=R^2/2; extend by H outside chart',
            'fiber_H0': 'physical vertical projection increases F at every level; natural component bijection, no higher-homotopy assertion',
            'axis_inventory': 'three controlled core roots plus the deep old seed minimum of physical value -2/9, centered value -2/3',
            'core_box': 's=A*x,t=A^2*y; [-8,8]x[-16,16] is inside q<=r1^2/4',
            'exterior_floor': 'complement of core box lies below b-16*A^4; at/below floor the full superlevel has one component'},
        'quantitative_guards': {
            'gradient_coordinates': 's,t (sine chart), not untransformed physical y',
            'step_derivative_upper': F(8), 'B0_upper': b0, 'D0_upper': d0,
            'B1_upper': b1, 'D1_upper': d1, 'g_lower': g,
            'gradient_perturbation_upper': b1*U0 + d1*V0,
            'height_perturbation_upper': b0*U0 + d0*V0,
            'critical_floor_scaled': F(-3, 4), 'endpoint_taylor_coefficient': F(15),
            'global_Q_lipschitz': F(45), 'global_Z_lipschitz': F(48),
            'arithmetic_allowance_per_endpoint': ARITHMETIC_ALLOWANCE},
        'source_sha256': {
            'math-design/DESIGN_PROOF.md': '5a20d81b57a216212f88940b79317f1e3593456855914f206a031e50dfb7767d',
            'math-design/REQUIREMENTS_INTERFACES.md': '4393d967cabb3ff67852e4b2b3888478030cabad88bb6a37d59e338b481fcb1e',
            'DESIGN.md': '6d853b109f09a446181cf0433c2ab8fdb0f4cc8b9a335c04a5bae4fc5b2d01ba',
            'PLAN.md': '553a1c11dd8ecf32698ce521c5c120867158117a3185afc070f2fcaf7c54fd50',
            'ROOT_SOURCE_IMPLEMENTATION_GO.json': '860e69ed193566a2f2fa97d198872f4be9a711b4513b58d8b0f77a0753493815',
            '.github/workflows/universality-controls.yml': '260e6921861d97f01d47f8dbec4fc5fd4f67d74d5cbe6c5f4cf3ddd73e7a3826',
            'experiments/periodic_h0/exact_barcode_certificate.py': '986da88dd703b4598ebf7782f874854124272ef6dbf67d7deb130972a86a217b',
            'experiments/periodic_h0/exact_h0.py': '48ee16f2316cb1a233594ed10466340aa321d553c2b888198fe86b4bd9052970',
            'experiments/periodic_h0/finite_certificate.py': '0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b',
            'experiments/periodic_h0/gaussian_coupling.py': '07e4d38720b778fb2a7487996a1beeafe29ffae34273491f5d6e1245e4dddffe',
            'experiments/periodic_h0/gaussian_coupling_certificate.py': '75023f4ab771ffd86a434ef3fc3bc9e9744ca6906c225ee986d85aec58a803b6',
            'experiments/periodic_h0/mean_adaptive_confidence.py': 'd83b4d530d00aafab1ebf5aa262869911db94d3b088309df3d878089285922a5',
            'experiments/universality/models.py': '7ba088158b5e97ba6e7d715e66e6f9d14268ac3a82a6d187a776821936cc6686',
            'experiments/universality/quartic_cusp_field/PROOF.md': '33ba8362069346058ba33bbd8c1bd1feed62b749a96b6cbbc8836ddcbc6e4f2e',
            'experiments/universality/run_pilot.py': '1224e556eaa83f43b97b68396cfb5962247f0d67afdb74bec6289760044aec9c',
            'experiments/universality/two_parameter_cusp_population/CONTROLLED_FALSIFICATION.md': 'a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e',
            'experiments/universality/two_parameter_cusp_population/PROOF.md': '9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326',
            'experiments/universality/two_parameter_cusp_population/population_certificate.py': 'be1d135b838733e4549b2519bae30d02377c97561a8876a16d9b7ea198305edf',
            'experiments/universality/two_parameter_cusp_population/run_population_certificate.py': '45b1a1d66564ac99799cc13d467b5948f685488de88cec90648c93401be22c67',
            'experiments/universality/weighted_cusp_sampling/PROOF.md': 'd97abfac1af80cf2c8c2705049bcc4c12f4e04be34ff67a8cee00e753fa9866b',
            'experiments/universality/weighted_cusp_sampling/README.md': 'fc8a8a40f885ac61d5066aa64e091558fc1ea9df95af48dc7c85b2c749f660fd',
            'experiments/universality/weighted_cusp_sampling/REVIEW.md': 'c7307648cd49af6544c6c742a6cb6f799124f95c7211a63a3e2ca6f2766fd564'},
        'source_refs': {'reconnaissance': 'c283035b471fab6ea3efafb305e24247d2ec2577',
                        'weighted': 'b79ca1f6ea5f3cfab95a29401ebd16d9b5ee565b'},
        'qualification': 'definition only; external source, review and runtime gates required'}
