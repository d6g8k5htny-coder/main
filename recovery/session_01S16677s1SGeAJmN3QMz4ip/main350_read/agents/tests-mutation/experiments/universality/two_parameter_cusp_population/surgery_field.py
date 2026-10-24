"""Outward enclosures of the reviewed explicit periodic cusp field.

Analytic smoothness/completeness guards are accepted source-bound premises,
not consequences of finite evaluations. Actual author: OpenAI/Codex, source
exposed; organizational and blind independence credit zero.
"""
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
import math
from pathlib import Path
import sys
import time

_BASE = Path(__file__).resolve().parent
_PRIMITIVE_BASE = _BASE.parents[1] / 'periodic_h0'
if str(_PRIMITIVE_BASE) not in sys.path:
    sys.path.insert(0, str(_PRIMITIVE_BASE))
import finite_certificate as _fc
import gaussian_tail as _gt

Interval = tuple[Q, Q]
MODEL_ID = 'explicit_periodic_cusp_v1'
BITS = 128
SINE_TERMS = 64
SINE_REMAINDER = Q(2**129, math.factorial(129))
FIELD_WIDTH = Q(1, 2**104)
D, L, R, R1 = 2, Q(1), Q(1, 8), Q(1, 32)
A, U0, V0 = Q(1, 2**16), Q(3, 2**32), Q(1, 2**47)
A_STAR, DELTA, B = Q(1, 128), Q(1, 2**26), Q(4, 9)
_ZERO, _ONE = (Q(0), Q(0)), (Q(1), Q(1))
_SOURCE_BINDINGS = {
    'ANALYTIC_GUARDS.md': {'bytes': 10494, 'sha256':
        '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9'},
    'finite_certificate.py': {'bytes': 14843, 'sha256':
        '0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b'},
    'gaussian_tail.py': {'bytes': 4688, 'sha256':
        '07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8'}}
_ACCEPTED_REVIEW = {'bytes': 22303, 'sha256':
    'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08',
    'status': 'PASS_BOUNDED_ANALYTIC_GUARDS'}
_IMPLEMENTATION_BYTES = Path(__file__).read_bytes()
_MODEL_BINDING = {
    'model_id': MODEL_ID,
    'analytic_source': dict(_SOURCE_BINDINGS['ANALYTIC_GUARDS.md']),
    'accepted_review': dict(_ACCEPTED_REVIEW),
    'dependencies': {name: dict(_SOURCE_BINDINGS[name]) for name in
                     ('finite_certificate.py', 'gaussian_tail.py')},
    'implementation_source': {'filename': 'surgery_field.py',
        'bytes': len(_IMPLEMENTATION_BYTES),
        'sha256': hashlib.sha256(_IMPLEMENTATION_BYTES).hexdigest()}}


class Inconclusive(Exception):
    """Retain a bounded failure without replacing its evidence by a guess."""
    def __init__(self, status: str, phase: str, detail: str):
        if not all(type(value) is str and value for value in (status, phase, detail)):
            raise TypeError('Nonempty status, phase and detail strings required')
        self.status, self.phase, self.detail = status, phase, detail
        super().__init__(f'{status}: {phase}: {detail}')


class Budget:
    """One deadline, whole-attempt requests and unique fixture algebra.

    Admission precedes execution; a refused next operation does not inflate
    executed-work counts. Gcd/division counts have no separate ceiling.
    """
    def __init__(self, start: float, *, seconds: int = 900,
                 polynomial_limit: int = 100000, field_limit: int = 4096,
                 primitive_limit: int = 32768, root_splits: int = 256):
        if type(start) is not float or not math.isfinite(start) or start < 0:
            raise ValueError('Finite nonnegative monotonic float start required')
        if start > time.monotonic():
            raise ValueError('Budget start cannot be in the future')
        limits = dict(seconds=seconds, polynomial_limit=polynomial_limit,
                      field_limit=field_limit, primitive_limit=primitive_limit,
                      root_splits=root_splits)
        ceilings = dict(seconds=900, polynomial_limit=100000, field_limit=4096,
                        primitive_limit=32768, root_splits=256)
        for name, value in limits.items():
            if type(value) is not int or not 0 <= value <= ceilings[name]:
                raise ValueError('Strict integer reduced ceiling required: ' + name)
        self._start, self._deadline = start, start + seconds
        self._limits = limits
        self._fixture_id = None
        self._fixtures = {}
        self._attempt_algebra = self._algebra_zero()
        self._field_calls = self._primitive_requests = 0
        self._primitive_kinds = dict.fromkeys(('sqrt', 'pi', 'sin', 'exp'), 0)
        self._field_records = []
        self._root_refinements = []

    @staticmethod
    def _algebra_zero():
        return dict(polynomial_evaluations=0, gcd_steps=0,
                    polynomial_divisions=0, max_branch_splits=0)

    def checkpoint(self, phase: str = 'work'):
        if type(phase) is not str or not phase:
            raise TypeError('Nonempty phase string required')
        if time.monotonic() >= self._deadline:
            raise Inconclusive('INCONCLUSIVE_BUDGET', phase, 'Shared deadline exhausted')

    def begin_fixture(self, fixture_id: str):
        if type(fixture_id) is not str:
            raise TypeError('Fixture ID must be a string')
        if not fixture_id or fixture_id in self._fixtures:
            raise ValueError('Nonempty unique fixture ID required')
        self.checkpoint('begin_fixture')
        self._fixture_id = fixture_id
        self._fixtures[fixture_id] = self._algebra_zero()

    def _charge_algebra(self, key: str, count: int, phase: str):
        if type(count) is not int or count <= 0:
            raise ValueError('Positive strict integer count required')
        if self._fixture_id is None:
            raise ValueError('Open a fixture before algebra work')
        self.checkpoint(phase)
        local = self._fixtures[self._fixture_id]
        if key == 'polynomial_evaluations' and local[key] + count > self._limits['polynomial_limit']:
            raise Inconclusive('INCONCLUSIVE_BUDGET', phase, 'Per-fixture polynomial ceiling')
        local[key] += count
        self._attempt_algebra[key] += count

    def charge_polynomial(self, count: int = 1, *, phase: str = 'polynomial'):
        self._charge_algebra('polynomial_evaluations', count, phase)

    def charge_gcd(self, count: int = 1, *, phase: str = 'gcd'):
        self._charge_algebra('gcd_steps', count, phase)

    def charge_division(self, count: int = 1, *, phase: str = 'division'):
        self._charge_algebra('polynomial_divisions', count, phase)

    def branch_splits(self, depth: int, *, phase: str = 'roots'):
        if type(depth) is not int or depth < 0:
            raise ValueError('Nonnegative strict integer branch depth required')
        if self._fixture_id is None:
            raise ValueError('Open a fixture before root work')
        self.checkpoint(phase)
        if depth > self._limits['root_splits']:
            raise Inconclusive('INCONCLUSIVE_BUDGET', phase, 'Root branch split ceiling')
        for work in (self._fixtures[self._fixture_id], self._attempt_algebra):
            work['max_branch_splits'] = max(work['max_branch_splits'], depth)
        self._root_refinements.append({'fixture_id': self._fixture_id, 'phase': phase, 'depth': depth})

    def charge_field(self, *, phase: str = 'field'):
        self.checkpoint(phase)
        if self._field_calls >= self._limits['field_limit']:
            raise Inconclusive('INCONCLUSIVE_BUDGET', phase, 'Whole-attempt field ceiling')
        self._field_calls += 1

    def charge_primitive(self, kind: str, *, phase: str = 'primitive'):
        if type(kind) is not str or kind not in self._primitive_kinds:
            raise ValueError('Only sqrt/pi/sin/exp requests charge primitive scope')
        self.checkpoint(phase)
        if self._primitive_requests >= self._limits['primitive_limit']:
            raise Inconclusive('INCONCLUSIVE_BUDGET', phase, 'Whole-attempt primitive ceiling')
        self._primitive_requests += 1
        self._primitive_kinds[kind] += 1

    def snapshot(self) -> dict:
        local = (self._fixtures[self._fixture_id] if self._fixture_id is not None
                 else self._algebra_zero())
        return deepcopy({**local, 'fixture_id': self._fixture_id,
            'fixtures': self._fixtures, 'attempt_algebra': self._attempt_algebra,
            'field_calls': self._field_calls, 'primitive_requests': self._primitive_requests,
            'primitive_requests_by_kind': self._primitive_kinds,
            'field_records': self._field_records, 'limits': self._limits,
            'root_refinements': self._root_refinements,
            'scopes': {'polynomial_evaluations': 'per_fixture', 'gcd_steps': 'per_fixture',
                'polynomial_divisions': 'per_fixture', 'max_branch_splits': 'per_fixture',
                'attempt_algebra': 'whole_attempt', 'field_calls': 'whole_attempt',
                'primitive_requests': 'whole_attempt', 'root_refinements': 'whole_attempt',
                'deadline': 'whole_attempt'},
            'monotonic_start': str(self._start), 'monotonic_deadline': str(self._deadline)})


def _fraction(value, name: str):
    if type(value) is not Q:
        raise TypeError(name + ' must have exact Fraction type')


def _interval(value, name: str = 'interval'):
    if type(value) is not tuple or len(value) != 2:
        raise TypeError(name + ' must be a two-endpoint tuple')
    for endpoint in value:
        _fraction(endpoint, name)
    if value[0] > value[1]:
        raise ValueError('Reversed ' + name)
    return value


def _outward(lo: Q, hi: Q) -> Interval:
    _interval((lo, hi))
    scale = 1 << BITS
    a, b = lo * scale, hi * scale
    return Q(a.numerator // a.denominator, scale), Q(-((-b.numerator) // b.denominator), scale)


def _add(a: Interval, b: Interval) -> Interval:
    _interval(a); _interval(b)
    return _outward(a[0] + b[0], a[1] + b[1])


def _neg(a: Interval) -> Interval:
    _interval(a)
    return -a[1], -a[0]


def _sub(a: Interval, b: Interval) -> Interval:
    return _add(a, _neg(b))


def _mul(a: Interval, b: Interval) -> Interval:
    _interval(a); _interval(b)
    values = [x * y for x in a for y in b]
    return _outward(min(values), max(values))


def _square(a: Interval) -> Interval:
    _interval(a)
    lo = Q(0) if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2)
    return _outward(lo, max(a[0]**2, a[1]**2))


def _div(a: Interval, b: Interval) -> Interval:
    _interval(a); _interval(b)
    if b[0] <= 0:
        raise ValueError('Certified positive denominator required')
    values = [x / y for x in a for y in b]
    return _outward(min(values), max(values))


def _sqrt_bounds(value: Q, *, budget: Budget) -> Interval:
    _fraction(value, 'radicand')
    if value < 0:
        raise ValueError('Nonnegative radicand required')
    budget.charge_primitive('sqrt')
    result = _fc.sqrt_bounds(value, bits=BITS)
    budget.checkpoint('sqrt')
    return _outward(*result)


def _pi_bounds(*, budget: Budget) -> Interval:
    budget.charge_primitive('pi')
    result = _fc.pi_bounds()
    if not Q(0) < result[0] <= result[1] < Q(4):
        raise Inconclusive('FAIL_IMPLEMENTATION', 'pi', 'Consumed pi bounds do not certify 0<pi<4')
    budget.checkpoint('pi')
    return _outward(*result)


def _sin_bounds(value: Interval, *, budget: Budget) -> Interval:
    _interval(value, 'sine argument')
    if value[0] < -2 or value[1] > 2:
        raise ValueError('Sine argument must lie wholly in [-2,2]')
    budget.charge_primitive('sin')
    square = _square(value)
    term = _outward(*value)
    total = term
    for k in range(SINE_TERMS - 1):
        divisor = Q((2*k+2)*(2*k+3))
        term = _div(_neg(_mul(term, square)), (divisor, divisor))
        total = _add(total, term)
        budget.checkpoint('sine_recurrence')
    lo, hi = _outward(total[0] - SINE_REMAINDER, total[1] + SINE_REMAINDER)
    # The analytic sine range is [-1,1] on the whole real line.
    return max(Q(-1), lo), min(Q(1), hi)


def _q_point(t: Q, *, budget: Budget) -> Interval:
    if t <= 0:
        return _ZERO
    reciprocal = 1 / t
    if reciprocal > 4096:
        # exp(1)>1+1=2, hence exp(-1/t)<2^-4096. This analytic dyadic
        # bound needs no finer numerical precision or primitive request.
        return Q(0), Q(1, 2**4096)
    budget.charge_primitive('exp')
    result = _gt.exp_neg_bounds(reciprocal)
    budget.checkpoint('exp')
    return _outward(*result)


def _q_bounds(value: Interval, *, budget: Budget) -> Interval:
    _interval(value, 'q argument')
    budget.checkpoint('q')
    # Monotonicity gives endpoint bounds, including the exact zero extension.
    return _q_point(value[0], budget=budget)[0], _q_point(value[1], budget=budget)[1]


def _psi_point(t: Q, *, budget: Budget) -> Interval:
    if t <= 0:
        return _ONE
    if t >= 1:
        return _ZERO
    first = _q_bounds((t, t), budget=budget)
    second = _q_bounds((1-t, 1-t), budget=budget)
    denominator = _add(first, second)
    if denominator[0] <= 0:
        raise Inconclusive('INCONCLUSIVE_PRECISION', 'psi', 'Step denominator lower enclosure is not positive')
    result = _div(second, denominator)
    return max(Q(0), result[0]), min(Q(1), result[1])


def _psi_bounds(value: Interval, *, budget: Budget) -> Interval:
    _interval(value, 'psi argument')
    budget.checkpoint('psi')
    # Accepted derivative sign: psi is descending.
    return _psi_point(value[1], budget=budget)[0], _psi_point(value[0], budget=budget)[1]


def _read_source(path: Path) -> bytes:
    return path.read_bytes()


def guard_certificate() -> dict:
    """Bind accepted analytic premises and replay their rational consequents."""
    for name, expected in _SOURCE_BINDINGS.items():
        path = (_BASE if name.endswith('.md') else _PRIMITIVE_BASE) / name
        try:
            content = _read_source(path)
        except OSError as exc:
            raise Inconclusive('INCONCLUSIVE_SOURCE_DRIFT', 'guard_source', 'Unavailable ' + name) from exc
        if {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()} != expected:
            raise Inconclusive('INCONCLUSIVE_SOURCE_DRIFT', 'guard_source', 'Changed ' + name)
    if (Path(_fc.__file__).resolve() != _PRIMITIVE_BASE / 'finite_certificate.py' or
            Path(_gt.__file__).resolve() != _PRIMITIVE_BASE / 'gaussian_tail.py' or
            _fc.BITS != BITS or _gt.EXP_BITS != 256 or _gt.fc is not _fc):
        raise Inconclusive('INCONCLUSIVE_SOURCE_DRIFT', 'guard_dependencies', 'Unexpected imported primitive identity')
    bounds = dict(psi_derivative=Q(4), chi_gradient=32/(3*R1),
                  B0=R1**2/2, D0=R1, B1=33*R1, D1=Q(65),
                  annular_gradient=R1**3/16, g_star=R1**3/64)
    bounds['gradient_perturbation'] = bounds['B1']*U0 + bounds['D1']*V0
    bounds['height_perturbation'] = bounds['B0']*U0 + bounds['D0']*V0
    bounds['critical_centered_height'] = A_STAR**4/4 + U0*A_STAR**2/2 + V0*A_STAR
    bounds['exterior_height'] = B-3*DELTA/4
    bounds['new_height_lower'] = B-DELTA/4
    bounds['separator'] = B-DELTA/2
    checks = {
        'parameters': U0 == 3*A**2 and V0 == 2*A**3 and A_STAR == R1/4 and DELTA == R1**4/64,
        'surgery_transition_positive': R**2/2-R1**2 > 0,
        'support_inside_chart': R1 < R and R**2 < Q(1, 9),
        'psi_frozen_bound': bounds['psi_derivative'] <= 8,
        'chi_frozen_bound': bounds['chi_gradient'] < 64/R1,
        'annular_frozen_bound': bounds['annular_gradient'] > bounds['g_star'],
        'gradient_exact': bounds['gradient_perturbation'] == Q(101441, 2**47),
        'gradient_guard': bounds['gradient_perturbation'] < Q(1, 2**22) <= bounds['g_star']/2,
        'height_exact': bounds['height_perturbation'] == Q(1537, 2**52),
        'height_guard': bounds['height_perturbation'] < DELTA/4,
        'root_u_containment': U0 < A_STAR**2/4,
        'root_v_containment': V0 < A_STAR**3/4,
        'wedge_squared_identity': V0**2 == 4*U0**3/27,
        'core_critical_height': bounds['critical_centered_height'] < DELTA/4,
        'barrier_loss': (R1**2/16)*(R1**2/4) == DELTA and R**2 > DELTA,
        'height_separation': bounds['exterior_height'] < bounds['separator'] < bounds['new_height_lower']}
    if not all(checks.values()):
        raise Inconclusive('FAIL_IMPLEMENTATION', 'guard_checks', 'Rational guard consequent failed')
    return {'status': 'ACCEPTED_ANALYTIC_PREMISE', 'model_id': MODEL_ID,
        'analytic_source': dict(_SOURCE_BINDINGS['ANALYTIC_GUARDS.md']),
        'accepted_review': dict(_ACCEPTED_REVIEW),
        'dependencies': deepcopy(_MODEL_BINDING['dependencies']),
        'parameters': dict(d=D, L=L, R=R, r1=R1, A=A, u0=U0, v0=V0,
                           a_star=A_STAR, delta=DELTA, b=B),
        'bounds': bounds, 'checks': checks,
        'analytic_premises': ('q_smooth_zero_extension', 'psi_descending_and_flat',
            'chi_smooth_support', 'c_nondecreasing_and_matching', 'annular_gradient',
            'critical_completeness', 'global_height_barrier'),
        'finite_sampling_proves_smoothness': False}


def _controls(u: Q, v: Q):
    _fraction(u, 'u'); _fraction(v, 'v')
    if not -U0 < u < U0 or not -V0 < v < V0:
        raise ValueError('Controls must lie in the strict original rectangle')


def _step_branches(argument: Interval, prefix: str) -> list[str]:
    result = []
    if argument[0] <= 0:
        result.append(prefix + '_one')
    if argument[0] < 1 and argument[1] > 0:
        result.append(prefix + '_transition')
    if argument[1] >= 1:
        result.append(prefix + '_zero')
    return result


def _full_field(u: Q, v: Q, s: Interval, z: Interval, *, budget: Budget, record: dict) -> Interval:
    s2, z2 = _square(s), _square(z)
    r = _add(s2, z2)
    chi_argument = _div(_sub(r, (R1**2/4, R1**2/4)), (3*R1**2/4, 3*R1**2/4))
    surgery_argument = _div(_sub(r, (R1**2, R1**2)), (R**2/2-R1**2, R**2/2-R1**2))
    chi_branches = _step_branches(chi_argument, 'chi')
    surgery_branches = _step_branches(surgery_argument, 'surgery')
    names = {'chi_one': 'chi_plateau_one', 'chi_zero': 'chi_plateau_zero',
             'surgery_one': 'surgery_core', 'surgery_zero': 'seed_collar'}
    record['branches'] = [names.get(name, name) for name in chi_branches + surgery_branches]
    record['chart_enclosure'] = {'s': s, 'z': z, 'r': r}
    chi = _psi_bounds(chi_argument, budget=budget)
    step = _psi_bounds(surgery_argument, budget=budget)
    c = _sub(_ONE, _mul(_sub(_ONE, _div(r, (Q(4), Q(4)))), step))
    # Physical r<=8/9 and chart r<=R^2 imply the proved convex range.
    if r[0] < 0 or r[1] > 1:
        raise Inconclusive('INCONCLUSIVE_PRECISION', 'radial_range', 'Unable to certify 0<=r<=1')
    c = max(Q(0), c[0]), min(Q(1), c[1])
    perturbation = _add(_mul((u/2, u/2), s2), _mul((v, v), s))
    return _add(_sub(_sub((B, B), _mul(s2, c)), z2), _mul(chi, perturbation))


def _new_record(u: Q, v: Q, route: str, *, budget: Budget, **inputs) -> dict:
    if not isinstance(budget, Budget):
        raise TypeError('Shared Budget required')
    budget.charge_field()
    record = {'route': route, 'u': u, 'v': v, 'fixture_id': budget._fixture_id,
        **inputs, 'branches': [], 'returned_interval': None, 'status': 'COMPUTING',
        'model': deepcopy(_MODEL_BINDING), 'gates': {'identity': False, 'ordered': None, 'width': None}}
    budget._field_records.append(record)
    return record


def _bind_record(record: dict, *, budget: Budget):
    budget.checkpoint('field_source')
    guard_certificate()
    try:
        content = _read_source(Path(__file__))
    except OSError as exc:
        raise Inconclusive('INCONCLUSIVE_SOURCE_DRIFT', 'field_source', 'Unavailable evaluator source') from exc
    if hashlib.sha256(content).hexdigest() != _MODEL_BINDING['implementation_source']['sha256']:
        raise Inconclusive('INCONCLUSIVE_SOURCE_DRIFT', 'field_source', 'Changed evaluator source since import')
    record['gates']['identity'] = True
    budget.checkpoint('field_source')


def _finish_record(record: dict, result: Interval, *, budget: Budget) -> Interval:
    record['returned_interval'] = result
    record['gates']['ordered'] = result[0] <= result[1]
    record['gates']['width'] = result[1]-result[0] <= FIELD_WIDTH
    if not record['gates']['ordered']:
        raise Inconclusive('FAIL_IMPLEMENTATION', 'field_interval', 'Reversed result enclosure')
    if not record['gates']['width']:
        raise Inconclusive('INCONCLUSIVE_PRECISION', 'field_width', 'Full field width exceeds 2^-104')
    budget.checkpoint('field_finish')
    record['status'] = 'CERTIFIED'
    return result


def _retain_failure(record: dict, exc: Inconclusive):
    record['status'] = exc.status
    record['failure'] = {'status': exc.status, 'phase': exc.phase, 'detail': exc.detail}


def chart_field_bounds(u: Q, v: Q, s: Interval, z: Interval, *, budget: Budget) -> Interval:
    _controls(u, v)
    _interval(s, 's box'); _interval(z, 'z box')
    # Check the whole domain exactly without widening a boundary first.
    if max(s[0]**2, s[1]**2) + max(z[0]**2, z[1]**2) > R**2:
        raise ValueError('Chart box must lie wholly inside s^2+z^2<=R^2')
    record = _new_record(u, v, 'chart', budget=budget, input_box={'s': s, 'z': z})
    try:
        _bind_record(record, budget=budget)
        return _finish_record(record, _full_field(u, v, s, z, budget=budget, record=record), budget=budget)
    except Inconclusive as exc:
        _retain_failure(record, exc)
        raise


def torus_field_bounds(u: Q, v: Q, y: tuple[Q, Q], *, budget: Budget) -> Interval:
    _controls(u, v)
    if type(y) is not tuple or len(y) != 2:
        raise TypeError('Torus point must be a two-coordinate tuple')
    for coordinate in y:
        _fraction(coordinate, 'torus coordinate')
    canonical = tuple(value - ((value+Q(1, 2)).numerator // (value+Q(1, 2)).denominator) for value in y)
    record = _new_record(u, v, 'torus', budget=budget, input_point=y, canonical_point=canonical)
    try:
        _bind_record(record, budget=budget)
        pi = _pi_bounds(budget=budget)
        s = _mul(_sqrt_bounds(Q(2, 3), budget=budget),
                 _sin_bounds(_mul(pi, (canonical[0], canonical[0])), budget=budget))
        z = _mul(_sqrt_bounds(Q(2, 9), budget=budget),
                 _sin_bounds(_mul(pi, (canonical[1], canonical[1])), budget=budget))
        return _finish_record(record, _full_field(u, v, s, z, budget=budget, record=record), budget=budget)
    except Inconclusive as exc:
        _retain_failure(record, exc)
        raise
