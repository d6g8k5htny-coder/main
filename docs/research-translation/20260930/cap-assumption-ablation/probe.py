#!/usr/bin/env python3
"""Stdlib exact controls for PROOF.md, not a continuum/formal proof."""
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def add(*polys):
    out = [F(0)] * max(map(len, polys))
    for p in polys:
        for i, x in enumerate(p):
            out[i] += x
    while len(out) > 1 and not out[-1]:
        out.pop()
    return out


def scale(p, c):
    return [c * x for x in p]


def mul(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i+j] += x*y
    return add(out)


def exp_bounds(x, n=80):
    """Exact rational bounds on exp(x), x>=0, via a geometric tail."""
    require(x >= 0, 'exponential argument must be nonnegative')
    term = total = F(1)
    for k in range(1, n+1):
        term *= x / k
        total += term
    first_omitted = term * x / (n+1)
    ratio = x / (n+2)
    require(ratio < 1, 'Taylor remainder ratio is not contractive')
    return total, total + first_omitted / (1-ratio)


def eta_bounds(t):
    if abs(t) >= 1:
        return F(0), F(0)
    lo, hi = exp_bounds(t*t/(1-t*t))
    return 1/hi, 1/lo


def derivative_bounds(t):
    # Only the increasing side of eta is needed for both stationary roots.
    require(F(1, 2) <= t <= 1, 'root bracket outside the proved interval')
    if t == F(1, 2):
        return F(-128), F(-128)
    u = 2-2*t
    lo, hi = eta_bounds(u)
    multiplier = 1024*u/(1-u*u)**2
    return -256*t+multiplier*lo, -256*t+multiplier*hi


def isolate_root(a, b, rising):
    # Uniqueness/nondegeneracy is analytic in PROOF.md, not inferred by sampling.
    for _ in range(36):
        c = (a+b)/2
        lo, hi = derivative_bounds(c)
        require(lo > 0 or hi < 0, 'uncertain derivative sign; refine exponential')
        positive = lo > 0
        if positive == rising:
            b = c
        else:
            a = c
    al, ah = derivative_bounds(a)
    bl, bh = derivative_bounds(b)
    require((ah < 0 and bl > 0) if rising else (al > 0 and bh < 0),
            'bracket does not carry opposite certified signs')
    return a, b


def decimal_bound(x, rounding):
    with localcontext() as ctx:
        ctx.prec = 24
        ctx.rounding = rounding
        return str(Decimal(x.numerator)/Decimal(x.denominator))


def interval_report(lo, hi):
    return {
        'lower_exact': str(lo), 'upper_exact': str(hi),
        'lower_decimal': decimal_bound(lo, ROUND_FLOOR),
        'upper_decimal': decimal_bound(hi, ROUND_CEILING),
    }


def main():
    # Check the complete numerator after clearing the positive denominator in B5.
    u = [F(0), F(1)]
    two_minus_u = [F(2), F(-1)]
    one_minus_u2 = [F(1), F(0), F(-1)]
    square = mul(one_minus_u2, one_minus_u2)
    numerator = add(
        scale(mul(u, square), -1),
        scale(mul(mul(mul(u, u), two_minus_u), one_minus_u2), -4),
        scale(mul(two_minus_u, square), -1),
        scale(mul(mul(u, u), two_minus_u), 2),
    )
    expected = list(map(F, [-2, 0, 0, 2, 6, -4]))
    require(numerator == expected, 'B5 logarithmic derivative identity failed')
    mutated = list(map(F, [-2, 0, 0, 2, 6, 4]))
    require(numerator != mutated, 'wrong final sign was not rejected')
    # exp(-t^2-t^4+O(t^6)) has coefficients 1,-1,-1/2 through degree 4.
    exponent = [F(0), F(0), F(-1), F(0), F(-1)]
    expansion = add([F(1)], exponent, scale(mul(exponent, exponent), F(1, 2)))[:5]
    require(expansion == [1, 0, -1, 0, F(-1, 2)], 'eta fourth-jet identity failed')
    eta_fourth = 24*expansion[4]
    require(eta_fourth == -12, 'eta fourth derivative differs')
    require(256*16*abs(eta_fourth) == 49152, 'M4 scaling differs')
    exp_lo, exp_hi = exp_bounds(F(1, 3))
    require(exp_hi < F(3, 2), 'needed exp(1/3)<3/2 bound failed')
    require(54*exp_hi < 256, 'B4 two-root level inequality failed')
    tmin = isolate_root(F(1, 2), F(3, 4), rising=True)
    tmax = isolate_root(F(3, 4), F(1), rising=False)
    # eta(2t-2) is increasing on the minimum's bracket. Interval dependency
    # is deliberately not cancelled: these are rigorous enclosing endpoints.
    eta_lo = eta_bounds(2*tmin[0]-2)[0]
    eta_hi = eta_bounds(2*tmin[1]-2)[1]
    mlo = -128*tmin[1]**2+256*eta_lo
    mhi = -128*tmin[0]**2+256*eta_hi
    require(-128 < mlo < mhi < 0, 'minimum enclosure is inconsistent')
    samples = []
    for delta in [F(1, 64), F(1, 128), F(1, 1024)]:
        norm = 256*delta**2
        floor = -128*delta**2
        require(F(-1, 2) < floor < 0, 'path floor does not exclude old saddle')
        require(49152/delta**2 > F(3, 10), 'M4 failure was lost')
        samples.append({'delta': str(delta), 'C0_norm_exact': str(norm),
                        'path_floor_exact': str(floor),
                        'older_endpoint_exact': str(128*delta**2),
                        'death_decimal_enclosure': [
                            decimal_bound(delta**2*mlo, ROUND_FLOOR),
                            decimal_bound(delta**2*mhi, ROUND_CEILING)],
                        'M4_lower_exact': str(49152/delta**2)})
    sources = {
        'AUDIT.md': (19468, '98356c5625aa07bcf62b9e6f1a3869ec1fea7dd789e8ea495a97b5c7abb473cf',
                     'fdfccaf5356f4cc0cf8764bf323f94b6ff071768'),
        'MARKED_CYLINDER_CAP_PROOF.md':
            (15160, '0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc',
             '0633aca3c2a2882b0de4399da0a75d64c2e6b2e1'),
    }
    for filename, (size, digest, blob) in sources.items():
        raw = (Path(__file__).parent/'sources'/filename).read_bytes()
        require(len(raw) == size, 'source size mismatch: '+filename)
        require(hashlib.sha256(raw).hexdigest() == digest, 'source digest mismatch: '+filename)
        require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == blob,
                'source Git-blob mismatch: '+filename)
    print(json.dumps({
        'result': 'PASS',
        'scientific_effect': 'NONE', 'formal_or_external_review': False,
        'method': 'exact Fraction polynomial identities and Taylor-remainder intervals',
        'root_min_t': interval_report(*tmin), 'root_max_t': interval_report(*tmax),
        'normalized_death_m_decimal_enclosure': [
            decimal_bound(mlo, ROUND_FLOOR), decimal_bound(mhi, ROUND_CEILING)],
        'samples': samples, 'local_exact_source_bindings_checked': list(sources),
        'scope': 'finite algebra/interval controls, not a continuum proof or a D1 verdict',
    }, indent=2))


if __name__ == '__main__':
    main()
