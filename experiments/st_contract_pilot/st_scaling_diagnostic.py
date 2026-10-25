"""Report-only ST algebra diagnostics for the frozen R2 pilot.

A matching source declaration is not artifact authentication. Algebra is not
field admission. This module evaluates no expression strings and accepts no
scientific evidence. Callers must authenticate the unchanged dependencies.
"""
from fractions import Fraction
import re
from fold_controls import cumulative_exponent, density_exponent
from st_object_overlay import apply_overlay, OverlayError

SOURCE = {
    'repository': 'd6g8k5htny-coder/main',
    'commit': '09def6f380215b77d2303f79b6263c0bf08056d0',
    'path': 'experiments/universality/fold_structural_universality/PROOF.md',
    'git_blob': '483d779cfdfdcea0bd8d0ec47c9fdaeff485fd4b',
}
ARITHMETIC_BLOB = '9d84ee8c1b06a2be23484222e76a0cb5e709d54f'
REQUIRED = ('COUNT', 'CONTACT', 'SELECT', 'ENVELOPE', 'FAR', 'positive_coefficient')
KINDS = {'joint_before_lifetime_pushforward': 'representation',
         'fixed_lifetime_near_kernel': 'fixed_lifetime_alternative'}
REQUIRED_KIND = 'joint_before_lifetime_pushforward'
LEDGER_KEYS = ('d', 'gamma', 'height_jacobian_power', 'a', 'observation_power', 'w', 'm')
JOINT_SHAPE = {
    'independent_variables': ['r', 'b', 'k', 'x', 'u'],
    'differential': ['dr', 'db', 'dk', 'dvol(x)', 'd_sigma_x(u)'],
    'held_fixed_during_lifetime_pushforward': ['k'],
    'passive_parameters_not_transformed': ['b', 'x', 'u'],
    'lifetime_map': 'ell=k*r^m',
    'positive_parameters': ['r', 'k', 'm'],
    'source_near_domain': '0<r<=r_0',
}


def _refuse(result: dict, *codes: str) -> dict:
    result['status'] = 'REFUSED'
    result['diagnostics'].extend(codes)
    return result


def _evidence(doc: dict, result: dict) -> None:
    """Report supplied/missing records; neither category is verification."""
    obligations = doc.get('analytic_obligations')
    supplied = obligations.get('evidence_supplied_by_this_proposal') if type(obligations) is dict else None
    if type(supplied) is not dict:
        supplied = {}
    missing = [key for key in REQUIRED if supplied.get(key) is None]
    present = [key for key in REQUIRED if supplied.get(key) is not None]
    result['missing_evidence'] = missing
    result['supplied_unverified_evidence'] = present
    result['diagnostics'].extend(key.upper() + '_EVIDENCE_NOT_SUPPLIED' for key in missing)
    if present:
        result['diagnostics'].append('SUPPLIED_EVIDENCE_NOT_VERIFIED')
    boundary = doc.get('consumer_boundary')
    node = boundary.get('reviewed_graph_node') if type(boundary) is dict else None
    if node is None:
        result['diagnostics'].append('REVIEWED_NODE_ASSOCIATION_NOT_SUPPLIED')
    else:
        result['diagnostics'].append('SUPPLIED_NODE_ASSOCIATION_NOT_VERIFIED')
    result['mapping_status'] = 'UNKNOWN' if node is None else 'UNVERIFIED_SUPPLIED'


def diagnose(document: dict, operation: str = 'ordinary_joint_pushforward') -> dict:
    """Diagnose the finite R2 baseline/contact-counterfactual template.

    Only bounded plain-JSON inputs and exact integer ledgers are supported.
    Dimension substitutions are algebraic, not dimension-specific admissions.
    m=3 and w=2 delimit this pilot, not other possible scaling theories.
    The result never grants acceptance, even with non-null evidence records.
    """
    result = {
        'status': 'REFUSED', 'diagnostics': [],
        'scientific_acceptance': False, 'field_admission': False,
        'complete_downstream_coverage': False,
        'requested_reporting_uses_allowed': False,
        'remote_source_authenticated': False, 'source_declaration_matches': False,
        'operation_allowed': False, 'ordinary_pushforward_called': False,
        'algebra_result_retained': False,
        'old_analytic_evidence_carries_automatically': False,
    }
    try:
        doc = apply_overlay(document, {}, [])
    except OverlayError as exc:
        return _refuse(result, 'INVALID_INPUT', exc.code)
    if doc.get('format') != 'ulhi.st.inert-proposal/2':
        return _refuse(result, 'UNSUPPORTED_CONTRACT_FORMAT')
    source = doc.get('source')
    arithmetic = doc.get('arithmetic_to_reuse')
    if (type(source) is not dict or any(source.get(k) != v for k, v in SOURCE.items())
            or type(arithmetic) is not dict or arithmetic.get('git_blob') != ARITHMETIC_BLOB):
        return _refuse(result, 'SOURCE_IDENTITY_MISMATCH')
    result['source_declaration_matches'] = True
    if type(operation) is not str or operation != 'ordinary_joint_pushforward':
        return _refuse(result, 'UNSUPPORTED_OPERATION')
    selector = doc.get('selected_representation')
    if type(selector) is not str or selector not in KINDS:
        return _refuse(result, 'INVALID_REPRESENTATION_SELECTOR')
    rep = doc.get(KINDS[selector])
    if type(rep) is not dict or rep.get('kind') != selector:
        return _refuse(result, 'REPRESENTATION_SELECTOR_MISMATCH')
    # Operation dispatch precedes scalar/integrability arithmetic.
    if selector != REQUIRED_KIND:
        codes = ['WRONG_REPRESENTATION']
        differential = rep.get('differential')
        if type(differential) is not list or 'dk' not in differential:
            codes.append('NO_INDEPENDENT_DK')
        if rep.get('radial_power') != '-2':
            codes.append('PINNED_RADIAL_POWER_MISMATCH')
        return _refuse(result, *codes)
    if any(rep.get(k) != v for k, v in JOINT_SHAPE.items()):
        return _refuse(result, 'REPRESENTATION_CONTRACT_MISMATCH')
    scope = doc.get('scope')
    if type(scope) is not dict or scope.get('lifetime_domain') != 'ell>0':
        return _refuse(result, 'LIFETIME_DOMAIN_MISMATCH')
    ledger = doc.get('ledger')
    if type(ledger) is not dict:
        return _refuse(result, 'BASELINE_LEDGER_INCOMPLETE')
    if 'height_jacobian_power' not in ledger:
        return _refuse(result, 'MISSING_HEIGHT_JACOBIAN', 'BASELINE_LEDGER_INCOMPLETE')
    if any(key not in ledger for key in LEDGER_KEYS):
        return _refuse(result, 'BASELINE_LEDGER_INCOMPLETE')
    if any(type(ledger[key]) is not int for key in LEDGER_KEYS):
        return _refuse(result, 'NONINTEGER_LEDGER')
    d, gamma, jac, a, observation, w, m = (ledger[key] for key in LEDGER_KEYS)
    # Defensive software bounds, not hypotheses of the source theorem.
    if not 1 <= d <= 1024 or m <= 0 or any(abs(ledger[key]) > 4096 for key in LEDGER_KEYS):
        return _refuse(result, 'LEDGER_DOMAIN')
    if gamma != d - 1 or jac != m or observation != -a:
        return _refuse(result, 'INCOHERENT_LEDGER')
    if m != 3 or w != 2:
        return _refuse(result, 'UNSUPPORTED_TEMPLATE')
    role = doc.get('instance_role')
    relation = doc.get('counterfactual_relation')
    if role is None:
        if relation is not None or a != d + 3:
            return _refuse(result, 'UNDECLARED_COUNTERFACTUAL')
    elif (role != 'counterfactual_template' or relation != 'a=d+2' or a != d + 2):
        return _refuse(result, 'UNSUPPORTED_COUNTERFACTUAL')
    radial = gamma + jac - a + w
    result['ordinary_pushforward_called'] = True
    try:
        cumulative = cumulative_exponent(radial, m)
        density = density_exponent(radial, m)
    except ValueError:
        return _refuse(result, 'NONINTEGRABLE_OR_INVALID_PUSHFORWARD')
    result.update(status='ALGEBRA_ONLY', operation_allowed=True, algebra_result_retained=True,
                  joint_radial_power=str(radial), cumulative_power=str(cumulative),
                  density_power=str(density), mark_power=str(-cumulative),
                  pushforward_prefactor=str(Fraction(1, m)),
                  post_transform_mark_domain='k>=ell/r_0^3',
                  new_envelope_weight='k^(' + str(-cumulative) + ')',
                  new_far_condition=('nu_far(ell)->0' if density == 0 else
                                     'ell^(' + str(-density) + ')*nu_far(ell)->0'))
    expected = doc.get('expected_algebra')
    keys = ('joint_radial_power', 'cumulative_power', 'density_power', 'mark_power',
            'pushforward_prefactor', 'post_transform_mark_domain')
    result['stale_expected_algebra'] = (type(expected) is dict and
        any(k in expected and expected[k] != result[k] for k in keys))
    if result['stale_expected_algebra']:
        result['diagnostics'].append('STALE_EXPECTED_ALGEBRA')
    if 'requested_extra_density_conversion' in doc:
        result['diagnostics'].append('REPEATED_DENSITY_CONVERSION')
        result['submitted_density_power_accepted'] = False
    elif 'submitted_density_power' in doc:
        text = doc['submitted_density_power']
        valid = type(text) is str and len(text) <= 64 and re.fullmatch(r'-?[0-9]+(?:/[0-9]+)?', text)
        try:
            agrees = bool(valid) and Fraction(text) == density
        except (ValueError, ZeroDivisionError):
            agrees = False
        result['submitted_density_power_accepted'] = agrees
        if not agrees:
            result['diagnostics'].append('SUBMITTED_DENSITY_MISMATCH')
    _evidence(doc, result)
    return result
