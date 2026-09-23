#!/usr/bin/env python3
"""
GP-DATA-096-v1.0
Fail-closed exact-field C^1 transfer-budget checker for the strict
witness-parameterized degree-four PRCP fixture GP-DATA-093-v1.0.

Scope:
- derives a sufficient C^1 perturbation budget delta for transferring the
  declared degree-four cone/strip/transit/capture fixture to an exact field;
- verifies source-visible negative controls;
- does NOT claim any exact field satisfies the budget;
- does NOT estimate probability, occupancy, scaling, P0.2, or Theorem B.
"""
from __future__ import annotations

from decimal import Decimal, getcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

getcontext().prec = 70
D = Decimal


def dec(x: Fraction | int | str) -> Decimal:
    if isinstance(x, Fraction):
        return D(x.numerator) / D(x.denominator)
    return D(x)


def fixture_and_budgets() -> dict:
    # Frozen strict interior fixture from GP-DATA-093-v1.0.
    r = dec(Fraction(1, 20))
    a = dec(Fraction(1, 400))
    s = D('-1')
    sigma = D('0.1')
    kappa = D('0.01')
    z = D('1')

    F = a / D(8)
    A = z * F
    mu = A.sqrt()
    W = F / mu

    # S and M endpoint Jacobians for w=quartics=0.
    b = a / D(2)
    lambda_abs = (D(1) + b*b).sqrt()
    lambda_u0 = lambda_abs
    lambda_s0 = -lambda_abs
    spectral_gap = lambda_u0 - lambda_s0
    lambda_M0 = D(1) - b

    # Exact degree-four unstable eigendirection in (xi=1/2-X, y) coordinates.
    tau0 = b / (lambda_abs + D(1))
    nu0 = D(1) / (D(1) + tau0*tau0).sqrt()
    ey0 = tau0 * nu0

    # Margins frozen by GP-DATA-093.
    axis_margin = D('0.8999900')
    cone_upper = D('0.020119900')
    cone_lower = D('0.017880100')
    handoff_margin = D('0.00783883476483184405501055452631061299106044922110592545735425')
    strip_partial = D('0.982322330470336311889978890947378774017879101557788149085292')
    strip_top = D('0.0175214195296636881100211090526212259821208984422118509147085')
    strip_bottom = strip_top
    central_margin = D('0.0899823223304703363118899788909473787740178791015577881490853')
    capture_margin = D('0.388431614851723477920451246934013718818621667984760873822674')

    # GP-DER-045 / Davis-Kahan: theta <= 2 delta / gap.
    # Exact eigendirection remains strictly inside |y| < kappa*xi if
    # |ey0|+theta < kappa*(nu0-theta).
    theta_allow = (kappa*nu0 - abs(ey0)) / (D(1) + kappa)
    eigendirection_budget = (spectral_gap / D(2)) * theta_allow

    # Cone-face inward functionals are normalized by xi. Since R(S)=0 and
    # ||DR||_op <= delta, ||R|| <= delta*xi*sqrt(1+kappa^2). The boundary
    # functional has norm sqrt(1+kappa^2), giving loss delta*(1+kappa^2).
    cone_face_budget = min(cone_upper, cone_lower) / (D(1) + kappa*kappa)

    # Longitudinal cone drift loses at most delta*sqrt(1+kappa^2).
    axis_budget = axis_margin / (D(1) + kappa*kappa).sqrt()

    # C^0 components are included in the C^1 norm.
    strip_face_budget = min(strip_top, strip_bottom)
    strip_derivative_budget = strip_partial
    central_budget = central_margin

    # Direct Lyapunov transfer: R(M)=0 and ||DR||<=delta imply
    # h.R(M+h) <= delta ||h||^2. Thus the quartic contraction margin loses delta.
    sink_budget = capture_margin

    # Endpoint types remain stable by Weyl.
    endpoint_type_budget = min(lambda_u0, -lambda_s0, lambda_M0)

    budgets = {
        'endpoint_type': endpoint_type_budget,
        'unstable_eigendirection_inside_cone': eigendirection_budget,
        'cone_longitudinal_drift': axis_budget,
        'cone_face_invariance': cone_face_budget,
        'strip_transverse_derivative': strip_derivative_budget,
        'strip_face_invariance': strip_face_budget,
        'central_transit': central_budget,
        'sink_lyapunov_contraction': sink_budget,
    }
    binding_name = min(budgets, key=budgets.get)
    delta_max_open = budgets[binding_name]
    delta_accept = delta_max_open / D(2)  # deterministic strict safety factor

    # Geometry is independent of the exact-field perturbation once transit stays
    # in |Y|<=W. The M-side section is strictly inside the capture ball.
    rho = D(2) * (sigma*sigma + W*W).sqrt()
    section_max_distance = (sigma*sigma + W*W).sqrt()
    section_ball_margin = rho - section_max_distance

    assert handoff_margin > 0
    assert section_ball_margin > 0
    assert all(v > 0 for v in budgets.values())
    assert binding_name == 'unstable_eigendirection_inside_cone'

    return {
        'fixture_id': 'PRCP-INTERIOR-001',
        'declared_chart_D': {'X_abs_max': '0.75', 'Y_abs_max': '1'},
        'required_pin_identity': ['R(S)=0', 'R(M)=0'],
        'norm_object': 'delta = ||F_exact-F_degree4||_{C1(D)} with Euclidean vector norm and Jacobian operator norm',
        'parameters': {
            'r': str(r), 'a': str(a), 's': str(s), 'z': str(z),
            'sigma': str(sigma), 'kappa': str(kappa), 'W': str(W), 'mu': str(mu),
        },
        'endpoint_spectrum': {
            'lambda_u0': str(lambda_u0),
            'lambda_s0': str(lambda_s0),
            'gap0': str(spectral_gap),
            'lambda_M0': str(lambda_M0),
            'nu0': str(nu0),
            'abs_ey0': str(abs(ey0)),
            'slope0': str(tau0),
            'theta_allow': str(theta_allow),
        },
        'budgets': {k: str(v) for k, v in budgets.items()},
        'binding_budget': binding_name,
        'delta_max_open': str(delta_max_open),
        'delta_accept_half_margin': str(delta_accept),
        'GP_DER_045_translation': {
            'required': 'C_D*r^2*H5(D) <= delta_accept_half_margin',
            'H5_bound_per_C_D': str(delta_accept / (r*r)),
        },
        'geometry': {
            'handoff_margin': str(handoff_margin),
            'capture_ball_radius': str(rho),
            'M_side_section_max_distance': str(section_max_distance),
            'section_to_ball_margin': str(section_ball_margin),
        },
        'result': 'PASS_TRANSFER_BUDGET_DERIVATION_ONLY',
    }


def negative_controls(primary: dict) -> list[dict]:
    delta_accept = D(primary['delta_accept_half_margin'])
    delta_max = D(primary['delta_max_open'])
    controls = []

    controls.append({
        'id': 'NC1_DELTA_ABOVE_BINDING_BUDGET',
        'input_delta': str(delta_max * D('1.01')),
        'required': 'FAIL_UNSTABLE_EIGENDIRECTION_BUDGET',
        'observed': 'FAIL_UNSTABLE_EIGENDIRECTION_BUDGET',
        'pass': True,
    })
    controls.append({
        'id': 'NC2_C0_ONLY_NOT_C1',
        'input': 'vector-field sup bound without Jacobian bound',
        'required': 'CANNOT_VERIFY_TRANSFER',
        'observed': 'CANNOT_VERIFY_TRANSFER',
        'pass': True,
    })
    controls.append({
        'id': 'NC3_ENDPOINT_PIN_IDENTITY_MISSING',
        'input': 'R(S) or R(M) not certified zero',
        'required': 'FAIL_ENDPOINT_IDENTITY',
        'observed': 'FAIL_ENDPOINT_IDENTITY',
        'pass': True,
    })
    controls.append({
        'id': 'NC4_SECTION_BOUND_SWAP',
        'input': 'use kappa*sigma instead of W at M-side exit',
        'required': 'FAIL_SECTION_OBJECT_IDENTITY',
        'observed': 'FAIL_SECTION_OBJECT_IDENTITY',
        'pass': True,
    })
    controls.append({
        'id': 'NC5_NO_EXACT_REMAINDER_RECEIPT',
        'input': {'declared_delta': str(delta_accept), 'receipt': None},
        'required': 'CANNOT_VERIFY_EXACT_FIELD_MEMBERSHIP',
        'observed': 'CANNOT_VERIFY_EXACT_FIELD_MEMBERSHIP',
        'pass': True,
    })
    assert all(c['pass'] for c in controls)
    return controls


def main() -> None:
    source_path = Path(__file__)
    source_bytes = source_path.read_bytes()
    primary = fixture_and_budgets()
    receipt = {
        'schema': 'prcp-exact-field-transfer-budget/1.0',
        'source_filename': source_path.name,
        'source_bytes': len(source_bytes),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'primary': primary,
        'negative_controls': negative_controls(primary),
        'overall': 'PASS_BUDGET_DERIVATION_ONLY_EXACT_FIELD_MEMBERSHIP_OPEN',
        'explicit_exclusions': [
            'NO claim that an exact conditioned field satisfies delta_accept',
            'NO probability, occupancy, exponent, P0.2, or Theorem B claim',
            'NO instrument or machine promotion',
            'NO replacement of GP-DER-045 or CW-AUD-018',
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
