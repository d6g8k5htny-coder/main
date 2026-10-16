"""Exact finite algebra controls for the local isolation proposal; no field sampler."""
from fractions import Fraction as Q
from math import factorial
import json


def clean(p):
    return {ij: c for ij, c in p.items() if c}


def add(*ps):
    out = {}
    for p in ps:
        for ij, c in p.items():
            out[ij] = out.get(ij, Q(0)) + c
    return clean(out)


def scale(c, p):
    return clean({ij: c * x for ij, x in p.items()})


def mul(p, q):
    out = {}
    for (i, j), a in p.items():
        for (k, l), b in q.items():
            ij = (i + k, j + l)
            out[ij] = out.get(ij, Q(0)) + a * b
    return clean(out)


def power(p, n):
    out = {(0, 0): Q(1)}
    for _ in range(n):
        out = mul(out, p)
    return out


def derivative(p, axis, order=1):
    for _ in range(order):
        out = {}
        for ij, c in p.items():
            d = list(ij)
            if d[axis]:
                out[tuple(d[:axis] + [d[axis] - 1] + d[axis + 1:])] = c * d[axis]
        p = out
    return p


def restrict(p, h):
    return add(*(scale(c, mul({(i, 0): Q(1)}, power(h, j)))
                 for (i, j), c in p.items()))


def evaluate(p, x, y=Q(0)):
    return sum(c * x**i * y**j for (i, j), c in p.items())


def run():
    x = {(1, 0): Q(1)}
    y = {(0, 1): Q(1)}
    one = {(0, 0): Q(1)}
    # Prescribe a nontrivial exact stationary ridge with all cubic mixed terms present.
    h = add(scale(Q(1, 7), power(x, 2)), scale(Q(1, 11), power(x, 3)))
    f = add(scale(-2, power(x, 2)), scale(Q(3, 7), power(x, 3)),
            scale(Q(-1, 11), power(x, 4)), scale(Q(1, 13), power(x, 5)))
    a = add(scale(2, one), scale(Q(1, 5), x), scale(Q(1, 17), power(x, 2)))
    b = add(scale(Q(1, 3), one), scale(Q(-1, 13), x))
    q = add(y, scale(-1, h))
    field = add(f, scale(Q(-1, 2), mul(a, power(q, 2))),
                scale(Q(1, 6), mul(b, power(q, 3))))
    hp = derivative(h, 0)
    assert restrict(derivative(field, 1), h) == {}
    assert restrict(field, h) == f
    rhs = add(restrict(derivative(field, 0, 3), h),
              scale(3, mul(restrict(derivative(derivative(field, 0, 2), 1), h), hp)),
              scale(3, mul(restrict(derivative(derivative(field, 0), 1, 2), h), power(hp, 2))),
              mul(restrict(derivative(field, 1, 3), h), power(hp, 3)))
    assert rhs == derivative(f, 0, 3)

    # Endpoint-distance normalization: r=2delta, k=1/12, not a half-distance mark.
    delta, av = Q(2, 7), Q(3)
    cubic = add(scale(Q(1, 6), power(x, 3)),
                scale(-delta**2 / 2, x), scale(-av / 2, power(y, 2)))
    ell = evaluate(cubic, -delta) - evaluate(cubic, delta)
    r = 2 * delta
    k = ell / r**3
    assert k == Q(1, 12)
    assert evaluate(derivative(cubic, 0, 3), -delta) == 12 * k

    # Positive k does not imply a nonzero endpoint cubic without an r*C4 condition.
    rr, kk = Q(2, 11), Q(5, 7)
    quartic = add(scale(kk / rr, power(x, 4)),
                  scale(-2 * kk * rr, power(x, 2)), scale(-av / 2, power(y, 2)))
    assert evaluate(derivative(quartic, 0), 0) == 0
    assert evaluate(derivative(quartic, 0), rr) == 0
    assert evaluate(derivative(quartic, 0, 2), 0) < 0
    assert evaluate(derivative(quartic, 0, 2), rr) > 0
    assert evaluate(derivative(quartic, 0, 3), 0) == 0
    qgap = evaluate(quartic, 0) - evaluate(quartic, rr)
    assert qgap == kk * rr**3
    assert rr * evaluate(derivative(quartic, 0, 4), 0) / 2 == 12 * kk

    # The reduced cubic changes at the B^2/a scale even when D4F=0.
    aa, tau, mixed, mu = Q(3), Q(2), Q(7), Q(1, 10)
    cross = add(scale(-mu / 2, power(x, 2)), scale(tau / 6, power(x, 3)),
                scale(mixed / 2, mul(power(x, 2), y)), scale(-aa / 2, power(y, 2)))
    hh = scale(mixed / (2 * aa), power(x, 2))
    reduced = restrict(cross, hh)
    expected = add(scale(tau, one), scale(3 * mixed**2 / aa, x))
    assert derivative(reduced, 0, 3) == expected
    assert all(derivative(derivative(cross, 0, j), 1, 4-j) == {} for j in range(5))
    xloss = -aa * tau / (3 * mixed**2)
    assert evaluate(expected, xloss) == 0

    # A cubic fold has no guaranteed full enclosing height barrier.
    radius, mm, tt = Q(1, 10), Q(1, 100), Q(1)
    escape = add(scale(-mm / 2, power(x, 2)), scale(tt / 6, power(x, 3)),
                 scale(-av / 2, power(y, 2)))
    assert evaluate(escape, radius) > evaluate(escape, 0)

    # Two maxima can approach only as the derivative bounds deteriorate in this control.
    sep, target = Q(1, 17), Q(3, 5)
    amp = target / (12 * sep)
    doublewell = add(scale(-amp, mul(power(x, 2), power(add(x, scale(-sep, one)), 2))),
                     scale(-av / 2, power(y, 2)))
    for point in (Q(0), sep):
        assert evaluate(derivative(doublewell, 0), point) == 0
        assert evaluate(derivative(doublewell, 0, 2), point) < 0
        assert abs(evaluate(derivative(doublewell, 0, 3), point)) == target
    assert abs(evaluate(derivative(doublewell, 0, 4), 0)) == 2 * target / sep

    # Rational checks of every numerical margin used by the chart argument.
    p = Q(1, 8)
    assert 3*p + 3*p*p + p**3 <= 4*p
    assert Q(31, 32)**2 / 4 >= Q(1, 8)
    assert Q(1, 2) - Q(1, 64) == Q(31, 64)
    assert Q(16, 32) == Q(1, 2)
    return {"classification": "EXACT_FINITE_ALGEBRA_CONTROLS_ONLY",
            "arithmetic": "Python Fraction; no floating arithmetic",
            "sampling": "NONE", "process_claim": "NONE", "scientific_effect": "NONE",
            "checks": {"all_mixed_reduced_third_derivative_identity": "PASS",
                       "endpoint_distance_and_factor12": "PASS",
                       "quartic_positive_mark_zero_endpoint_cubic_counterexample": "PASS",
                       "mixed_cubic_Bsquared_over_a_counterexample": "PASS",
                       "false_full_height_barrier_counterexample": "PASS",
                       "two_maxima_derivative_scale_control": "PASS",
                       "chart_margin_constants": "PASS"},
            "exact_values": {"cubic_k": str(k), "mixed_cubic_sign_loss_x": str(xloss),
                             "quartic_gap": str(qgap),
                             "horizontal_escape_height": str(evaluate(escape, radius)),
                             "two_maxima_separation": str(sep)}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
