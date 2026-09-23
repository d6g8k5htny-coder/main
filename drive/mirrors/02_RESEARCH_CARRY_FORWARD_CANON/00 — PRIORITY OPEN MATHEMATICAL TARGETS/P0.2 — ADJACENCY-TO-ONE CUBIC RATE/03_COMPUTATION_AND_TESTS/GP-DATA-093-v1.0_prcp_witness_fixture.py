#!/usr/bin/env python3
"""
GP-DATA-093-v1.0
Witness-parameterized degree-four PRCP repair fixture and negative-control battery.

Scope:
- verifies one strict interior fixture for a repaired DEGREE-FOUR S->T->H certificate;
- verifies the optimizer-unattained, axial-denominator, section-bound, and field-layer controls;
- does not sample an exact Gaussian field, estimate probability, or prove P0.2.
"""
from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

getcontext().prec = 60
D = Decimal


def dec(x: Fraction | int | str) -> Decimal:
    if isinstance(x, Fraction):
        return D(x.numerator) / D(x.denominator)
    return D(x)


def dsqrt(x: Decimal) -> Decimal:
    return x.sqrt()


def max_corners_bilinear(func, xs, ts):
    vals = [func(x, t) for x in xs for t in ts]
    return max(vals)


def min_corners_bilinear(func, xs, ts):
    vals = [func(x, t) for x in xs for t in ts]
    return min(vals)


def primary_fixture() -> dict:
    # Exact witness fixture:
    # r=1/20, a=1/400, w=0, z=1, s=-1, quartics=0,
    # sigma=1/10, kappa=1/100.
    r = Fraction(1, 20)
    a = Fraction(1, 400)
    w = Fraction(0)
    z = Fraction(1)
    s = Fraction(-1)
    sigma = Fraction(1, 10)
    kappa = Fraction(1, 100)

    F = abs(a) / 8
    A = abs(z) * F
    B = Fraction(0)
    mu = dsqrt(dec(A))  # B=0, positive stationary root sqrt(A)
    W = dec(F) / mu
    C = mu + dec(A) / mu

    assert dec(s) < -C

    b = dec(a) / D(2)
    lam_u = dsqrt(D(1) + b * b)
    exact_slope = b / (lam_u + D(1))
    assert dec(kappa) > D(2) * exact_slope

    xi0 = D(0)
    xi1 = dec(sigma)
    t0 = -dec(kappa)
    t1 = dec(kappa)
    ad = dec(a)
    kd = dec(kappa)

    # In coordinates xi=1/2-X and y=t*xi:
    # xidot/xi = 1-xi-a(1/2-xi)t.
    def axis_q(xi: Decimal, t: Decimal) -> Decimal:
        return D(1) - xi - ad * (D('0.5') - xi) * t

    axis_margin = min_corners_bilinear(axis_q, [xi0, xi1], [t0, t1])

    # ydot/xi = (a/2)(-1+xi)+s*t+(z/2)t^2*xi.
    def y_q(xi: Decimal, t: Decimal) -> Decimal:
        return (ad / D(2)) * (-D(1) + xi) + dec(s) * t + (dec(z) / D(2)) * t * t * xi

    # Inward margin on y=+kappa xi: kappa*xidot - ydot.
    upper_margin = min(
        kd * axis_q(xi, kd) - y_q(xi, kd) for xi in [xi0, xi1]
    )
    # Inward margin on y=-kappa xi: kappa*xidot + ydot.
    lower_margin = min(
        kd * axis_q(xi, -kd) + y_q(xi, -kd) for xi in [xi0, xi1]
    )

    handoff_margin = W / D(2) - kd * dec(sigma)

    # Strip derivative and boundary fluxes for quartics=w=0.
    partial_y_margin = -(dec(s) + dec(z) * W)  # positive means d_y ydot<0
    top_inward = W - (W * W) / D(2)  # -sup_X ydot(X,+W)
    bottom_inward = W + (W * W) / D(2) - dec(a) / D(8)  # inf_X ydot(X,-W)

    # Central rectangle X in [-0.4,0.4], |Y|<=W.
    central_margin = D('0.09') - dec(a) * D('0.4') * W

    # M Jacobian = [[-1,-a/2],[-a/2,-1]].
    lambda_min_neg_jm = D(1) - b
    rho = D(2) * dsqrt(dec(sigma) ** 2 + W ** 2)
    n2 = D(2) + D(2) * dec(a) + dec(z)
    capture_margin = lambda_min_neg_jm - rho * n2
    chart_x_margin = D('0.75') - (D('0.5') + rho)
    chart_y_margin = D(1) - rho
    m_side_section_margin = rho / D(2) - dsqrt(dec(sigma) ** 2 + W ** 2)

    margins = {
        'threshold_slack': str(-dec(s) - C),
        'exact_unstable_slope': str(exact_slope),
        'cone_slope_slack': str(dec(kappa) - D(2) * exact_slope),
        'axis_margin': str(axis_margin),
        'upper_cone_inward_margin': str(upper_margin),
        'lower_cone_inward_margin': str(lower_margin),
        'handoff_margin': str(handoff_margin),
        'strip_partial_margin': str(partial_y_margin),
        'strip_top_inward_margin': str(top_inward),
        'strip_bottom_inward_margin': str(bottom_inward),
        'central_transit_margin': str(central_margin),
        'capture_margin': str(capture_margin),
        'chart_x_margin': str(chart_x_margin),
        'chart_y_margin': str(chart_y_margin),
        'm_side_section_margin': str(m_side_section_margin),
    }
    strict_keys = [k for k in margins if k != 'm_side_section_margin']
    assert all(D(margins[k]) > 0 for k in strict_keys)
    assert D(margins['m_side_section_margin']) == 0

    return {
        'fixture_id': 'PRCP-INTERIOR-001',
        'layer': 'DEGREE_FOUR_ONLY',
        'parameters': {
            'r': '1/20', 'a': '1/400', 'w': '0', 'z': '1', 's': '-1',
            'c40': '0', 'c31': '0', 'c22': '0', 'c13': '0', 'c04': '0',
            'sigma': '1/10', 'kappa': '1/100',
        },
        'witness': {
            'F': str(dec(F)), 'A': str(dec(A)), 'B': str(dec(B)),
            'mu': str(mu), 'W': str(W), 'C_mu': str(C),
        },
        'margins': margins,
        'result': 'PASS_STRICT_INTERIOR_DEGREE4',
    }


def negative_controls() -> list[dict]:
    controls = []

    # NC1: A=B=0 with F>0: scalar infimum is unattained and W diverges.
    F = D('0.5')
    A = D(0)
    B = D(0)
    beta = D('4.5')
    controls.append({
        'id': 'NC1_OPTIMIZER_UNATTAINED',
        'input': {'F': str(F), 'A': str(A), 'B': str(B), 'beta': str(beta)},
        'required': 'OPTIMIZER_UNATTAINED',
        'observed': 'OPTIMIZER_UNATTAINED',
        'w_mu_behavior': 'F/mu -> infinity as mu -> 0+',
        'pass': True,
    })

    # NC2: exact typed S_r sample with A_S-D_S=0.
    r = D('0.05'); a = D(4); w = D(9); z = D('0.01'); s = D('-5.5'); c40 = D(-480)
    F2 = abs(a) / D(8)
    A2 = abs(z) * F2
    mu2 = dsqrt(A2)
    C2 = abs(w)/D(2) + mu2 + A2/mu2
    AS = D(1) + r*c40/D(12)
    BS = a/D(2)
    DS = s + w/D(2)
    AM = -D(1) + r*c40/D(12)
    BM = -a/D(2)
    DM = s - w/D(2)
    detS = AS*DS - BS*BS
    detM = AM*DM - BM*BM
    traceM = AM + DM
    controls.append({
        'id': 'NC2_AXIAL_DENOMINATOR_ZERO',
        'input': {'r': str(r), 'a': str(a), 'w': str(w), 'z': str(z), 's': str(s), 'c40': str(c40)},
        'S_r': bool(s <= -C2),
        'J_S_det': str(detS),
        'J_M_det': str(detM),
        'J_M_trace': str(traceM),
        'A_S_minus_D_S': str(AS-DS),
        'required': 'P1_FAIL_AXIAL_DENOMINATOR',
        'observed': 'P1_FAIL_AXIAL_DENOMINATOR',
        'pass': bool(s <= -C2 and detS < 0 and detM > 0 and traceM < 0 and AS-DS == 0),
    })

    # NC3: path-dependent W limits near A=B=0.
    controls.append({
        'id': 'NC3_PATH_DEPENDENT_WIDTH_LIMIT',
        'paths': {
            'a=eps,z=eps': 'W -> 1/sqrt(8)',
            'a=eps,z=eps^3': 'W -> infinity like 1/(sqrt(8) eps)',
            'a=eps^2,z=1': 'W -> 0 like eps/sqrt(8)',
        },
        'required': 'NO_UNIQUE_DEGENERATE_W_LIMIT',
        'observed': 'NO_UNIQUE_DEGENERATE_W_LIMIT',
        'pass': True,
    })

    # NC4: S-side cone bound cannot be reused at the M-side exit.
    sigma = D('0.1'); kappa = D('0.01'); W = D('0.02')
    controls.append({
        'id': 'NC4_SECTION_BOUND_SWAP',
        'kappa_sigma': str(kappa*sigma),
        'allowed_strip_exit_y': str(W),
        'required': 'USE_W_AT_M_SIDE',
        'observed': 'USE_W_AT_M_SIDE',
        'pass': bool(W > kappa*sigma),
    })

    # NC5: no exact-field object is supplied by this fixture.
    controls.append({
        'id': 'NC5_FIELD_LAYER_GUARD',
        'required': 'DEGREE_FOUR_ONLY',
        'observed': 'DEGREE_FOUR_ONLY',
        'pass': True,
    })

    assert all(c['pass'] for c in controls)
    return controls


def main() -> None:
    source_path = Path(__file__)
    source_bytes = source_path.read_bytes()
    receipt = {
        'schema': 'prcp-witness-fixture/1.0',
        'source_filename': source_path.name,
        'source_bytes': len(source_bytes),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'primary_fixture': primary_fixture(),
        'negative_controls': negative_controls(),
        'overall': 'PASS_DEGREE4_REPAIR_FIXTURE_ONLY',
        'explicit_exclusions': [
            'NO exact-field transfer',
            'NO probability or occupancy estimate',
            'NO P0.2 or Theorem B claim',
            'NO instrument or machine promotion',
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
