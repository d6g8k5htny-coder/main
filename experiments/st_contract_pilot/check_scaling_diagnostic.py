"""Bounded algebra/representation diagnostics; never theorem acceptance."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PINS = {
 'deps/st_object_overlay.py': '343af2feddf9e49798da09c222a19ae2e9f548e89fcf9882e2ed2ff6b146fdd2',
 'fixtures/st_measure_contract.json': '5a521b035d257349530cef33721e1aaa45eb2280e009f4cea641dcef66461e6d',
 'fixtures/st_diagnostic_cases.json': 'a24b17ff803cb0cf72531f460c757f7c4c65534792a98912ccb0bde131ac322e',
}
for name, sha in PINS.items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
        raise RuntimeError('Frozen input mismatch: ' + name)
raw = (ROOT / 'deps/fold_controls.py').read_bytes()
if hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != '9d84ee8c1b06a2be23484222e76a0cb5e709d54f':
    raise RuntimeError('Exact arithmetic source mismatch')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

FC = load('fold_controls', ROOT / 'deps/fold_controls.py')
OV = load('st_object_overlay', ROOT / 'deps/st_object_overlay.py')
D = load('st_scaling_diagnostic', ROOT / 'st_scaling_diagnostic.py')
BASE = OV.loads_object((ROOT / 'fixtures/st_measure_contract.json').read_text())
CASES = OV.loads_object((ROOT / 'fixtures/st_diagnostic_cases.json').read_text())['cases']

class DiagnosticTests(unittest.TestCase):
    def fields(self, result, **expected):
        self.assertIsInstance(result, dict)
        for key, value in expected.items():
            self.assertIn(key, result, key)
            self.assertEqual(result[key], value, key)

    def codes(self, result, *codes):
        self.assertIn('diagnostics', result)
        for code in codes:
            self.assertIn(code, result['diagnostics'])

    def case(self, n):
        c = CASES[n - 1]
        doc = OV.apply_overlay(BASE, c['set'], c['remove'])
        before = copy.deepcopy(doc)
        result = D.diagnose(doc, c['operation'])
        self.assertEqual(doc, before)
        self.fields(result, scientific_acceptance=False, field_admission=False,
                    complete_downstream_coverage=False)
        return result

    def test_st01_complete_ledger(self):
        self.fields(self.case(1), joint_radial_power='1', density_power='-1/3',
                    cumulative_power='2/3', mark_power='-2/3',
                    pushforward_prefactor='1/3', operation_allowed=True)

    def test_st02_contact_counterfactual(self):
        r = self.case(2)
        self.fields(r, joint_radial_power='2', density_power='0', cumulative_power='1',
                    mark_power='-1', new_envelope_weight='k^(-1)',
                    new_far_condition='nu_far(ell)->0', stale_expected_algebra=True,
                    old_analytic_evidence_carries_automatically=False)

    def test_st03_missing_jacobian(self):
        r = self.case(3)
        self.codes(r, 'MISSING_HEIGHT_JACOBIAN', 'BASELINE_LEDGER_INCOMPLETE')
        self.fields(r, operation_allowed=False, ordinary_pushforward_called=False)
        self.assertNotIn('density_power', r)

    def test_st04_wrong_measure(self):
        r = self.case(4)
        self.codes(r, 'WRONG_REPRESENTATION', 'NO_INDEPENDENT_DK')
        self.fields(r, operation_allowed=False, ordinary_pushforward_called=False)

    def test_st05_scalar_cannot_bypass_type(self):
        r = self.case(5)
        self.codes(r, 'WRONG_REPRESENTATION', 'NO_INDEPENDENT_DK', 'PINNED_RADIAL_POWER_MISMATCH')
        self.fields(r, operation_allowed=False, ordinary_pushforward_called=False)

    def test_st06_double_density_conversion(self):
        r = self.case(6)
        self.codes(r, 'REPEATED_DENSITY_CONVERSION')
        self.fields(r, density_power='-1/3', submitted_density_power_accepted=False)

    def test_st07_missing_evidence_and_mapping(self):
        r = self.case(7)
        self.codes(r, 'SELECT_EVIDENCE_NOT_SUPPLIED', 'REVIEWED_NODE_ASSOCIATION_NOT_SUPPLIED')
        self.fields(r, density_power='-1/3', requested_reporting_uses_allowed=False)
        self.assertEqual(set(r.get('missing_evidence', [])),
                         {'COUNT','CONTACT','SELECT','ENVELOPE','FAR','positive_coefficient'})

    def test_st08_source_declaration_mismatch(self):
        r = self.case(8)
        self.codes(r, 'SOURCE_IDENTITY_MISMATCH')
        self.fields(r, operation_allowed=False, ordinary_pushforward_called=False)

    def test_expected_answers_do_not_drive_computation(self):
        doc = copy.deepcopy(BASE)
        doc['expected_algebra'] = {'density_power': '123456', 'joint_radial_power': '0'}
        doc['id'] = 'ST-08'
        self.fields(D.diagnose(doc), density_power='-1/3', stale_expected_algebra=True)

    def test_no_expected_answers_required(self):
        doc = copy.deepcopy(BASE); del doc['expected_algebra']
        self.fields(D.diagnose(doc), density_power='-1/3')

    def test_dimension_cancellation_is_algebra_only(self):
        for dimension in (1, 2, 3, 7, 16, 31):
            with self.subTest(d=dimension):
                doc = copy.deepcopy(BASE)
                doc['ledger'].update(d=dimension, gamma=dimension-1, a=dimension+3,
                                     observation_power=-(dimension+3))
                self.fields(D.diagnose(doc), density_power='-1/3', joint_radial_power='1',
                            field_admission=False, scientific_acceptance=False)

    def test_counterfactual_dimension_cancellation(self):
        for dimension in (1, 2, 5, 20):
            with self.subTest(d=dimension):
                doc = OV.apply_overlay(BASE, CASES[1]['set'], [])
                doc['ledger'].update(d=dimension, gamma=dimension-1, a=dimension+2,
                                     observation_power=-(dimension+2))
                self.fields(D.diagnose(doc), density_power='0', field_admission=False)

    def test_incoherent_contact_change_refused(self):
        doc = copy.deepcopy(BASE); doc['ledger']['a'] = 4
        r = D.diagnose(doc)
        self.codes(r, 'INCOHERENT_LEDGER')
        self.fields(r, ordinary_pushforward_called=False)

    def test_unmarked_counterfactual_refused(self):
        doc = copy.deepcopy(BASE); doc['ledger'].update(a=4, observation_power=-4)
        self.codes(D.diagnose(doc), 'UNDECLARED_COUNTERFACTUAL')

    def test_unsupported_counterfactual_not_guessed(self):
        doc = copy.deepcopy(BASE); doc.update(instance_role='counterfactual_template', counterfactual_relation='exec(unsafe)')
        self.codes(D.diagnose(doc), 'UNSUPPORTED_COUNTERFACTUAL')

    def test_float_bool_string_ledger_refused(self):
        for value in (True, 3.0, '3', None):
            with self.subTest(value=value):
                doc = copy.deepcopy(BASE); doc['ledger']['m'] = value
                self.codes(D.diagnose(doc), 'NONINTEGER_LEDGER')

    def test_invalid_domains_refused(self):
        for key,value in [('m',0),('d',0),('d',1025)]:
            with self.subTest(key=key):
                doc = copy.deepcopy(BASE); doc['ledger'][key] = value
                self.codes(D.diagnose(doc), 'LEDGER_DOMAIN')

    def test_missing_selector_refused(self):
        doc = copy.deepcopy(BASE); del doc['selected_representation']
        self.codes(D.diagnose(doc), 'INVALID_REPRESENTATION_SELECTOR')

    def test_selector_member_kind_mismatch_refused(self):
        doc = copy.deepcopy(BASE); doc['representation']['kind'] = 'fixed_lifetime_near_kernel'
        self.codes(D.diagnose(doc), 'REPRESENTATION_SELECTOR_MISMATCH')

    def test_unknown_operation_refused(self):
        r = D.diagnose(BASE, 'accept_theorem')
        self.codes(r, 'UNSUPPORTED_OPERATION')
        self.fields(r, scientific_acceptance=False, ordinary_pushforward_called=False)

    def test_joint_without_dk_refused(self):
        doc = copy.deepcopy(BASE); doc['representation']['differential'].remove('dk')
        self.codes(D.diagnose(doc), 'REPRESENTATION_CONTRACT_MISMATCH')

    def test_positive_parameter_contract_required(self):
        doc = copy.deepcopy(BASE); doc['representation']['positive_parameters'] = []
        self.codes(D.diagnose(doc), 'REPRESENTATION_CONTRACT_MISMATCH')

    def test_near_domain_contract_required(self):
        doc = copy.deepcopy(BASE); doc['representation']['source_near_domain'] = 'r arbitrary'
        self.codes(D.diagnose(doc), 'REPRESENTATION_CONTRACT_MISMATCH')

    def test_positive_lifetime_domain_required(self):
        doc = copy.deepcopy(BASE); doc['scope']['lifetime_domain'] = 'ell arbitrary'
        self.codes(D.diagnose(doc), 'LIFETIME_DOMAIN_MISMATCH')

    def test_altered_lifetime_map_refused(self):
        doc = copy.deepcopy(BASE); doc['representation']['lifetime_map'] = 'ell=k*r^2'
        self.codes(D.diagnose(doc), 'REPRESENTATION_CONTRACT_MISMATCH')

    def test_declared_pin_does_not_authenticate_remote_source(self):
        self.fields(D.diagnose(BASE), source_declaration_matches=True, remote_source_authenticated=False)

    def test_different_commit_refused(self):
        doc = copy.deepcopy(BASE); doc['source']['commit'] = 'f'*40
        self.codes(D.diagnose(doc), 'SOURCE_IDENTITY_MISMATCH')

    def test_missing_obligation_list_cannot_hide_missing_evidence(self):
        doc = copy.deepcopy(BASE); doc['analytic_obligations']['required'] = []
        r = D.diagnose(doc)
        self.assertEqual(len(r.get('missing_evidence', [])), 6)
        self.fields(r, scientific_acceptance=False)

    def test_supplied_evidence_is_not_verified_evidence(self):
        doc = copy.deepcopy(BASE)
        for key in doc['analytic_obligations']['evidence_supplied_by_this_proposal']:
            doc['analytic_obligations']['evidence_supplied_by_this_proposal'][key] = {'verdict':'PASS'}
        doc['consumer_boundary']['reviewed_graph_node'] = 'invented'
        r = D.diagnose(doc)
        self.fields(r, scientific_acceptance=False, field_admission=False,
                    complete_downstream_coverage=False)
        self.assertEqual(len(r.get('supplied_unverified_evidence', [])), 6)

    def test_non_json_and_nonfinite_refused(self):
        for doc in ([], {'x': float('nan')}, {'x': object()}):
            with self.subTest(kind=type(doc).__name__):
                r = D.diagnose(doc)
                self.fields(r, status='REFUSED', ordinary_pushforward_called=False,
                            scientific_acceptance=False)

    def test_caller_data_remains_unchanged(self):
        doc = copy.deepcopy(BASE); before = copy.deepcopy(doc)
        r = D.diagnose(doc)
        self.assertEqual(doc,before)
        self.fields(r, density_power='-1/3')

    def test_wrong_representation_does_not_call_arithmetic(self):
        doc = OV.apply_overlay(BASE, CASES[3]['set'], [])
        with patch.object(D, 'cumulative_exponent', side_effect=AssertionError('arithmetic reached')):
            r = D.diagnose(doc)
        self.codes(r, 'WRONG_REPRESENTATION')

if __name__ == '__main__':
    unittest.main(verbosity=2)
