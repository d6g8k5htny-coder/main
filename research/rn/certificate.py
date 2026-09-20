"""Bounded, data-only replay certificates for generic Gaussian moment bounds.

Replay verifies the stated rational law family, not its identity with an RN
field. No scientific status, spatial coverage or independence is established.
The checker uses a separate iterative moment recurrence; it shares Interval,
covariance feasibility and the Gaussian identity with the producer. This is
not an independently reviewed proof kernel. See docs/RN_CERTIFICATES.md.
"""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import json
from math import comb
import re

from engine.operations.rn_applicability import Context
from research.interval import Interval
from research.rn.gaussian_families import FamilyMomentEngine, interval
from research.rn.gaussian_moments import Cost

SCHEMA = "RN_GAUSSIAN_MOMENT_CERTIFICATE_V1"
MAX_BYTES = 524288
MAX_DEPTH = 16
MAX_LEAVES = 64
MAX_DIGITS = 2048
INPUT_DIGITS = 96
MAX_BITS = 8192
MAX_OPERATIONS = 200000
FLAGS = {"authority": "NONE", "field_certified": False,
         "spatial_cover_certified": False, "independence_credit": 0,
         "scientific_status_changed": False, "original_prize_closed": False}
CONTEXT_KEYS = {"source_id", "source_sha256", "dimension", "order",
                "normalization", "conditioned_law", "domain", "evidence_tier"}
LAW_KEYS = {"context", "intercept", "slope", "covariance", "mean_law",
            "covariance_law", "degree"}


class ResourceLimit(ValueError):
    """A bounded verifier stopped without a mathematical conclusion."""


class Budget(Cost):
    def __init__(self, limit=MAX_OPERATIONS):
        if type(limit) is not int or not 0 <= limit <= MAX_OPERATIONS:
            raise ValueError("operation limit must be an integer in 0..200000")
        super().__init__(budget=limit)

    def charge(self, n=1):
        self.used += n
        if self.used > self.budget:
            raise ResourceLimit("arithmetic operation budget exhausted")

    def checked(self, value):
        endpoints = (value.lo, value.hi) if isinstance(value, Interval) else (F(value),)
        if any(max(abs(x.numerator).bit_length(), x.denominator.bit_length()) > MAX_BITS
               for x in endpoints):
            raise ResourceLimit("intermediate rational bit budget exhausted")
        return value

    def add(self, a, b):
        return self.checked(super().add(a, b))

    def mul(self, a, b):
        return self.checked(super().mul(a, b))

    def div(self, a, b):
        return self.checked(super().div(a, b))


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("missing or unknown object fields")


def _list(value, length):
    if type(value) is not list or len(value) != length:
        raise ValueError("incorrect array length")
    return value


def _rational(value, digits=MAX_DIGITS):
    if type(value) is not str:
        raise ValueError("rational endpoints must be canonical strings")
    if len(value) > 2*digits+2:
        raise ResourceLimit("rational digit budget exhausted")
    if not re.fullmatch(r"-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?", value):
        raise ValueError("invalid rational encoding")
    if any(len(part.lstrip('-')) > digits for part in value.split('/')):
        raise ResourceLimit("rational digit budget exhausted")
    out = F(value)
    if str(out) != value:
        raise ValueError("rational encoding is not reduced and canonical")
    return out


def _interval(value, digits=MAX_DIGITS):
    lo, hi = _list(value, 2)
    return Interval(_rational(lo, digits), _rational(hi, digits))


def _encode(value):
    if isinstance(value, Interval):
        return [str(value.lo), str(value.hi)]
    if isinstance(value, F):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_encode(x) for x in value]
    if isinstance(value, dict):
        return {key: _encode(x) for key, x in value.items()}
    return value


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _reject_number(value):
    raise ValueError("floating point and non-finite JSON numbers are forbidden")


def load_bytes(data):
    """Strict bounded JSON reader; neither code nor paths are evaluated."""
    if type(data) is not bytes:
        raise ValueError("certificate input must be bytes")
    if len(data) > MAX_BYTES:
        raise ResourceLimit("certificate byte budget exhausted")
    text = data.decode('utf-8')
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > MAX_DEPTH:
                raise ResourceLimit("JSON nesting budget exhausted")
        elif char in ']}':
            depth -= 1

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    def integer(value):
        if len(value) > 10:
            raise ResourceLimit("JSON integer budget exhausted")
        return int(value)

    return json.loads(text, object_pairs_hook=pairs, parse_int=integer,
                      parse_float=_reject_number, parse_constant=_reject_number)


def _context(value):
    _keys(value, CONTEXT_KEYS)
    for name in ('source_id', 'source_sha256', 'normalization',
                 'conditioned_law', 'evidence_tier'):
        if type(value[name]) is not str or not 1 <= len(value[name]) <= 512:
            raise ValueError("context text must contain 1..512 characters")
    order = tuple(_list(value['order'], 3))
    domain = _interval(value['domain'], INPUT_DIGITS)
    c = Context(**{**value, 'order': order, 'domain': (domain.lo, domain.hi)})
    c.validate()
    return c


def _replay_moments(engine, degree, budget):
    """Iterative recurrence, independent of MomentEngine.moment/determinant.

    Both implementations use the same Gaussian integration-by-parts identity.
    This reduces implementation coupling, not the shared mathematical trust.
    """
    zero = Interval.exact(0)

    def trim(p):
        while len(p) > 1 and p[-1].lo == p[-1].hi == 0:
            p.pop()
        return p

    def plus(a, b):
        return trim([budget.add(a[i] if i < len(a) else zero,
                                b[i] if i < len(b) else zero)
                     for i in range(max(len(a), len(b)))])

    def times(a, b):
        out = [zero]*(len(a)+len(b)-1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                out[i+j] = budget.add(out[i+j], budget.mul(x, y))
        return trim(out)

    moments = {(0, 0, 0): [Interval.exact(1)]}
    for total in range(1, 2*degree+1):
        for a in range(total+1):
            for b in range(total-a+1):
                alpha = (a, b, total-a-b)
                i = next(k for k in range(3) if alpha[k])
                beta = list(alpha)
                beta[i] -= 1
                p = times(list(engine.mean[i]), moments[tuple(beta)])
                for j in range(3):
                    if beta[j]:
                        gamma = beta.copy()
                        gamma[j] -= 1
                        weight = budget.mul(F(beta[j]), engine.cov[i][j])
                        p = plus(p, times([weight], moments[tuple(gamma)]))
                moments[alpha] = p
    result = [zero]
    for j in range(degree+1):
        result = plus(result, times([F((-1)**j*comb(degree, j))],
                                    moments[(degree-j, degree-j, 2*j)]))
    return tuple(result)


def _law(value, budget, *, replay=False):
    _keys(value, LAW_KEYS)
    context = _context(value['context'])
    if type(value['degree']) is not int or value['degree'] not in (2, 4):
        raise ValueError("determinant degree must be 2 or 4")
    if any(value[k] != context.conditioned_law for k in ('mean_law', 'covariance_law')):
        raise ValueError("mean and covariance must identify the same conditioned law")
    a = tuple(_interval(x, INPUT_DIGITS) for x in _list(value['intercept'], 3))
    b = tuple(_interval(x, INPUT_DIGITS) for x in _list(value['slope'], 3))
    s = tuple(tuple(_interval(x, INPUT_DIGITS) for x in _list(row, 3))
              for row in _list(value['covariance'], 3))
    engine = FamilyMomentEngine(a, b, s, order=context.order, cost=budget)
    polynomial = (_replay_moments(engine, value['degree'], budget) if replay
                  else engine.determinant(value['degree']))
    return context, engine, polynomial


def _bernstein(polynomial, domain, budget):
    """Exact coefficient conversion, not samples or optimization over marks."""
    n = len(polynomial)-1
    lo, width = domain.lo, domain.hi-domain.lo
    power = []
    for j in range(n+1):
        total = Interval.exact(0)
        for k in range(j, n+1):
            factor = budget.checked(comb(k, j)*lo**(k-j)*width**j)
            total = budget.add(total, budget.mul(polynomial[k], factor))
        power.append(total)
    result = []
    for k in range(n+1):
        total = Interval.exact(0)
        for j in range(k+1):
            total = budget.add(total, budget.mul(power[j], F(comb(k, j), comb(n, j))))
        result.append(total)
    return tuple(result)


def produce(law, *, degree, depth=2, claimed_upper=None, max_operations=MAX_OPERATIONS):
    """Create a deterministic certificate for a Law or FamilyLaw object."""
    if type(depth) is not int or not 0 <= depth <= 6:
        raise ValueError("certificate subdivision depth must be 0..6")
    law.context.validate()
    context_data = _encode(asdict(law.context))
    context_data['domain'] = [str(F(x)) for x in law.context.domain]
    claim = {'context': context_data,
             'intercept': _encode(tuple(map(interval, law.intercept))),
             'slope': _encode(tuple(map(interval, law.slope))),
             'covariance': _encode(tuple(tuple(map(interval, row)) for row in law.covariance)),
             'mean_law': law.mean_law, 'covariance_law': law.covariance_law,
             'degree': degree}
    budget = Budget(max_operations)
    context, engine, polynomial = _law(claim, budget)
    lo, hi = context.domain
    count = 2**depth if lo != hi else 1
    leaves = []
    upper = None
    for i in range(count):
        domain = Interval(lo+(hi-lo)*F(i, count), lo+(hi-lo)*F(i+1, count))
        bernstein = _bernstein(polynomial, domain, budget)
        bound = max(x.hi for x in bernstein)
        upper = bound if upper is None else max(upper, bound)
        leaves.append({'domain': _encode(domain), 'bernstein': _encode(bernstein)})
    if upper < 0:
        raise ValueError("negative even-moment cap is not admitted")
    if claimed_upper is not None:
        if type(claimed_upper) not in (int, F) or claimed_upper < upper:
            raise ValueError("claimed upper must be exact and at least the replay cap")
        upper = F(claimed_upper)
    certificate = {'schema': SCHEMA, 'claim': claim, 'scope': dict(FLAGS),
                   'proof': {'coefficients': _encode(polynomial),
                             'feasibility': _encode(engine.feasibility),
                             'leaves': leaves, 'upper': str(upper)}}
    encoded = canonical_bytes(certificate)
    # The producer obeys the same size/encoding admission contract as replay.
    result = verify_bytes(encoded, max_operations=max_operations)
    if not result['certificate_valid']:
        if result['outcome'] == 'INCONCLUSIVE':
            raise ResourceLimit(result['reason'])
        raise ValueError(result['reason'])
    return certificate


def verify_bytes(data, *, source_bytes=None, expected_context=None,
                 max_operations=MAX_OPERATIONS):
    """Replay a certificate. Requested provenance mismatches are separate.

    certificate_valid concerns the declared generic law only. Source hashing
    checks supplied bytes, and context matching checks caller declarations;
    neither authenticates the law as the physical/research field.
    """
    result = {**FLAGS, 'certificate_valid': False, 'outcome': 'REJECTED',
              'source_bytes_match': None, 'expected_context_match': None,
              'requested_checks_passed': False, 'reason': ''}
    try:
        certificate = load_bytes(data)
        _keys(certificate, {'schema', 'claim', 'proof', 'scope'})
        if certificate['schema'] != SCHEMA:
            raise ValueError("unsupported certificate schema/version")
        if canonical_bytes(certificate['scope']) != canonical_bytes(FLAGS):
            raise ValueError("scope flags must exactly preserve the no-authority boundary")
        proof = certificate['proof']
        _keys(proof, {'coefficients', 'feasibility', 'leaves', 'upper'})
        budget = Budget(max_operations)
        context, engine, polynomial = _law(certificate['claim'], budget, replay=True)
        coefficients = tuple(_interval(x) for x in _list(proof['coefficients'], len(polynomial)))
        if coefficients != polynomial:
            raise ValueError("moment coefficient derivation mismatch")
        if canonical_bytes(proof['feasibility']) != canonical_bytes(_encode(engine.feasibility)):
            raise ValueError("covariance feasibility witness or quantifier mismatch")
        leaves = proof['leaves']
        if type(leaves) is not list or not leaves:
            raise ValueError("nonempty subdivision partition required")
        if len(leaves) > MAX_LEAVES:
            raise ResourceLimit("subdivision leaf budget exhausted")
        lo, hi = context.domain
        cursor = lo
        upper = None
        for leaf in leaves:
            _keys(leaf, {'domain', 'bernstein'})
            domain = _interval(leaf['domain'], INPUT_DIGITS)
            if domain.lo != cursor or domain.hi > hi:
                raise ValueError("subdivision gap, overlap, ordering or domain mismatch")
            if domain.lo == domain.hi and (lo != hi or len(leaves) != 1):
                raise ValueError("degenerate subdivision leaf")
            cursor = domain.hi
            supplied = tuple(_interval(x) for x in _list(leaf['bernstein'], len(polynomial)))
            expected = _bernstein(polynomial, domain, budget)
            if supplied != expected:
                raise ValueError("Bernstein coefficient derivation mismatch")
            bound = max(x.hi for x in expected)
            upper = bound if upper is None else max(upper, bound)
        if cursor != hi:
            raise ValueError("subdivision does not cover full mark domain")
        claimed = _rational(proof['upper'])
        result['replay_upper'] = str(upper)
        result['certified_lower_bound_on_slack'] = str(claimed-upper)
        result['diagnostic_kind'] = 'SUFFICIENT_BOUND_SLACK_NOT_ACTUAL_OPTIMUM'
        if claimed < upper or upper < 0:
            raise ValueError("claimed cap fails the replayed sufficient bound; no actual counterexample asserted")
        if source_bytes is not None:
            if type(source_bytes) is not bytes:
                raise ValueError("supplied source must be bytes")
            result['source_bytes_match'] = hashlib.sha256(source_bytes).hexdigest() == context.source_sha256
        if expected_context is not None:
            expected = _context(expected_context)
            result['expected_context_match'] = asdict(expected) == asdict(context)
        result.update(certificate_valid=True, outcome='VALID',
                      requested_checks_passed=all(result[k] is not False for k in
                                                  ('source_bytes_match', 'expected_context_match')),
                      claim_sha256=hashlib.sha256(canonical_bytes(certificate['claim'])).hexdigest(),
                      certificate_sha256=hashlib.sha256(canonical_bytes(certificate)).hexdigest(),
                      upper=str(claimed), operations=budget.used,
                      reason='Exact replay for declared Gaussian family; field applicability unestablished.')
    except ResourceLimit as error:
        result.update(outcome='INCONCLUSIVE', reason=str(error))
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError) as error:
        result['reason'] = str(error)
    return result
