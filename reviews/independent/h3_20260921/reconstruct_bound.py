"""Independent reconstruction of the H3 scalar sufficient bound. NON-CERTIFYING.

Rebuilds the numeric chain of `engine/solver_pilots/h3_20260921/PROOF.md`
(sections 1-4) from the prose alone, in mpmath floating point, importing
nothing from `engine/solver_pilots/`. Its purpose is to answer one question:
does an independently written implementation land inside the enclosure the
author-side rational certificate reports?

WHAT THIS DOES NOT ESTABLISH. This is mpmath arithmetic at finite precision,
so under CLAUDE.md rule 3 it is NON-CERTIFYING and is not a certificate, not a
replacement for the rational certificate in `h3_scalar_certificate.py`, and not
a bound of any kind. Agreement here is evidence that two implementations read
the same prose the same way; it is not evidence that the prose is correct.
It verifies no register, moves no gate, discharges no premise and awards zero
organizational independence credit. The pilot it cross-checks is NOT DEPLOYED
and no scientific status depends on this file.

Run: python3 reviews/independent/h3_20260921/reconstruct_bound.py
"""
from __future__ import annotations

import sys

try:
    from mpmath import mp, mpf, sqrt, exp, pi, erfc, nsum, inf
except ImportError:  # pragma: no cover - mpmath is not a guaranteed dependency
    print("reconstruct_bound: mpmath absent; NON-CERTIFYING cross-check skipped.")
    sys.exit(0)

mp.dps = 120

R = mpf(1) / 20
EPSILON = mpf(103) / 500
GAP_HALF = mpf(9) / 100
TARGET = mpf(1747) / 1000


def he(n, x):
    """Probabilists' Hermite He_n(x)."""
    p, q = mpf(1), x
    if n == 0:
        return p
    for k in range(1, n):
        p, q = q, x * q - k * p
    return q


def moments():
    """m_{2n} = (-1)^n k^{(2n)}(0) for the periodized SIDE24 kernel."""
    norm = nsum(lambda j: exp(-((24 * j) ** 2) / 2), [-inf, inf])
    out = {}
    for n2 in (2, 4, 6, 8):
        num = nsum(lambda j: he(n2, 24 * j) * exp(-((24 * j) ** 2) / 2), [-inf, inf])
        out[n2] = (-1) ** (n2 // 2) * num / norm
    return out


def normal():
    return (lambda x: erfc(-x / sqrt(2)) / 2,
            lambda x: erfc(x / sqrt(2)) / 2,
            lambda x: exp(-x ** 2 / 2) / sqrt(2 * pi))


def reconstruct():
    cdf, sf, pdf = normal()
    m = moments()
    m2, m4, m6, m8 = m[2], m[4], m[6], m[8]
    sigma_e2 = m4 - m2 ** 2
    beta = m4 * m2 / 4
    a = 1 - EPSILON

    checks = []

    # Section 1: transport budget and the uniform pin-energy bound Q*.
    transport2 = (m2 / 4 + m4 / 4 + m2 ** 2 / 4
                  + m6 / 16 + m4 * m2 / 16 + m8 / 4096)
    checks.append(("transport^2 < (31/20)^2", transport2 < (mpf(31) / 20) ** 2, transport2))
    q_zero = mpf(36) / 25 * m4 / (m4 - m2 ** 2) + 4 * m2 / (m2 * m6 - m4 ** 2)
    checks.append(("Q(0) < 17/6", q_zero < mpf(17) / 6, q_zero))
    checks.append(("(193/1000)^2 < 3/80", (mpf(193) / 1000) ** 2 < mpf(3) / 80,
                   (mpf(193) / 1000) ** 2))
    gap = 1 - (mpf(31) / 20) * R / (mpf(193) / 1000)
    q_star = (mpf(17) / 6) / gap ** 2
    checks.append(("Q* < 8", q_star < 8, q_star))

    # Section 2: axial event.
    axial = EPSILON / (R * sqrt(m8) / 12) - sqrt(q_star)
    p_axial = 1 - 2 * sf(axial)

    # Section 3: transverse event and positive-part moments over the whole band.
    delta = (2 * GAP_HALF - m2 * R ** 3 / 6) / (R * sqrt(sigma_e2 * m2))
    p_delta = 1 - 2 * sf(delta)

    def positive_part(mu, var):
        sd = sqrt(var)
        t = mu / sd
        return sd * pdf(t) + mu * cdf(t), (var + mu ** 2) * cdf(t) + mu * sd * pdf(t)

    corners = [(mu, var)
               for mu in (m2 * (mpf(6) / 5 - R ** 3 / 12) - GAP_HALF,
                          m2 * mpf(6) / 5 - GAP_HALF)
               for var in (sigma_e2 * (1 - m2 * R ** 2 / 4), sigma_e2)]
    m1_hi = max(positive_part(*c)[0] for c in corners)
    m2_lo = min(positive_part(*c)[1] for c in corners)

    # Section 4: typed determinant bracket, negative term at r = R.
    bracket = a * a * m2_lo - a * R * beta * m1_hi
    lower = p_axial * p_delta * bracket
    return checks, {"Q*": q_star, "p_axial": p_axial, "p_delta": p_delta,
                    "M1_hi": m1_hi, "M2_lo": m2_lo, "bracket": bracket,
                    "lower": lower}


def main():
    checks, v = reconstruct()
    problems = 0
    for name, ok, value in checks:
        if not ok:
            problems += 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name:24s} value={mp.nstr(value, 12)}")
    for key in ("Q*", "p_axial", "p_delta", "M1_hi", "M2_lo", "bracket", "lower"):
        print(f"  {key:9s} = {mp.nstr(v[key], 14)}")
    # The author-side rational enclosure this reconstruction is compared against.
    enclosure = (mpf("1.747106129274"), mpf("1.747994749902"))
    inside = enclosure[0] <= v["lower"] <= enclosure[1]
    above = v["lower"] > TARGET
    if not inside:
        problems += 1
    if not above:
        problems += 1
    print(f"  inside author-side enclosure {enclosure}: {inside}")
    print(f"  exceeds 1747/1000: {above} (margin {mp.nstr(v['lower'] - TARGET, 6)})")
    print(f"reconstruct_bound: problems={problems}")
    print("NON-CERTIFYING mpmath cross-check. No certificate, no status change, "
          "no independence credit.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
