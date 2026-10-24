"""Literal fixtures for the controlled exact-population certificate.

These tests exercise actual selectors, interval routes and filesystem custody.
No full eight-row integration campaign is performed by this test file.
"""
from fractions import Fraction as Q
from pathlib import Path
import importlib.util
import tempfile
import time
import unittest

import population_certificate as pc
import run_population_certificate as runner


class CriticalHeightTests(unittest.TestCase):
    # A fixed-left selector would fail the theta=2 fixture.
    def test_actual_younger_height_and_reflection(self):
        a = Q(1, 1 << 18)
        for theta, lifetime in ((Q(1, 4), Q(1, 1 << 76)),
                                (Q(1, 2), Q(1, 1 << 73)),
                                (Q(3, 4), Q(27, 1 << 76)),
                                (Q(2), Q(3, 1 << 74))):
            for sign in (1, -1):
                with self.subTest(theta=theta, sign=sign):
                    row = pc.critical_fixture(a, a * theta, sign)
                    self.assertEqual(row['status'], 'FINITE_BAR')
                    self.assertEqual(row['count'], 1)
                    self.assertEqual(row['lifetime'], lifetime)
                    self.assertTrue(row['stationary'])
                    self.assertTrue(row['rectangle'])
                    self.assertEqual(row['hessian_signs'], (-1, 1, -1))
                    self.assertEqual(row['younger_side'],
                                     'left' if (theta < 1) == (sign == 1) else 'right')
        attack = pc.critical_fixture(a, 2 * a)
        self.assertEqual(attack['roots'], (-3 * a, a, 2 * a))
        self.assertEqual(attack['v'], -6 * a ** 3)
        self.assertNotEqual(attack['lifetime'], Q(1, 1 << 67))

    def test_null_discriminants_and_tie_are_excluded(self):
        a = Q(1, 1 << 18)
        for theta, status in ((Q(0), 'EXCLUDED_DISCRIMINANT'),
                              (Q(1), 'EXCLUDED_TIE'),
                              (Q(3), 'EXCLUDED_DISCRIMINANT')):
            for sign in (1, -1):
                row = pc.critical_fixture(a, theta * a, sign)
                self.assertEqual(row['status'], status)
                self.assertEqual(row['count'], 0)
                self.assertEqual(row['lifetime'], Q(0))

    def test_selector_rejects_inexact_and_invalid_inputs(self):
        for a, h, sign in ((0.25, Q(1), 1), (True, Q(1), 1),
                           (Q(1), 0.5, 1), (Q(0), Q(0), 1),
                           (Q(1), Q(-1), 1), (Q(1), Q(1), True),
                           (Q(1), Q(1), 0)):
            with self.assertRaises(ValueError):
                pc.critical_fixture(a, h, sign)

    def test_guard_witnesses_check_rational_consequents(self):
        guards = pc.guard_witnesses()
        self.assertEqual(guards['status'], 'PASS_RATIONAL_GUARDS')
        self.assertEqual(guards['gradient_perturbation'], Q(101441, 1 << 47))
        self.assertEqual(guards['height_perturbation'], Q(1537, 1 << 52))
        self.assertTrue(all(guards['gates'].values()))
        self.assertEqual(guards['complete_surgery_field'], 'NOT_RUN')
        self.assertEqual(guards['independent_global_selector'], 'NOT_RUN')


class PredicateTests(unittest.TestCase):
    def test_common_inner_and_outer_coefficient_polarity(self):
        b, e = (Q(1, 5), Q(3, 10)), (Q(1, 10), Q(1, 10))
        for observed, want in (((Q(21, 100), Q(29, 100)), 'PASS'),
                               ((Q(3, 20), Q(1, 5)), 'INCONCLUSIVE_PRECISION'),
                               ((Q(41, 100), Q(1, 2)), 'FALSIFIED')):
            self.assertEqual(pc.classify_coefficient(observed, b, e), want)

    def test_void_endpoint_distance_strict_boundary(self):
        target = (Q(1, 2), Q(1, 2))
        self.assertEqual(pc.classify_void(target, target), 'PASS')
        self.assertEqual(pc.classify_void((Q(1, 2) + Q(1, 8192),) * 2, target),
                         'INCONCLUSIVE_PRECISION')
        self.assertEqual(pc.classify_void((Q(1, 2) + Q(1, 4096),) * 2, target),
                         'FALSIFIED')
        self.assertEqual(pc.classify_void((Q(0), Q(1, 4)), target),
                         'INCONCLUSIVE_PRECISION')

    def test_interval_predicates_reject_bad_endpoints(self):
        good = (Q(1, 5), Q(1, 5))
        for bad in ((Q(1), Q(0)), (0.2, Q(1)), (True, Q(1)),
                    (Q(0), 1), (Q(0),), '0/1'):
            with self.assertRaises(ValueError):
                pc.classify_coefficient(bad, good, good)
            with self.assertRaises(ValueError):
                pc.classify_void(good, bad)
        with self.assertRaises(ValueError):
            pc.classify_coefficient(good, good, (Q(-1), Q(0)))


class IntervalRouteTests(unittest.TestCase):
    def test_cdf_boundaries_keep_whole_rectangle_probability(self):
        self.assertEqual(pc.cdf_mass(Q(0)), (Q(0), Q(0)))
        self.assertEqual(pc.cdf_mass(Q(9, 4)), (Q(1, 5), Q(1, 5)))
        self.assertEqual(pc.cdf_mass(Q(3)), (Q(1, 5), Q(1, 5)))
        self.assertFalse(pc.rectangle_mass_control((Q(1, 10), Q(1, 10))))
        self.assertFalse(pc.rectangle_mass_control((Q(1), Q(1))))
        self.assertTrue(pc.rectangle_mass_control((Q(1, 5), Q(1, 5))))

    def test_split_root_encloses_hand_checked_half(self):
        lo, hi = pc.root_bracket(Q(72, 169))
        self.assertLessEqual(lo, Q(1, 2))
        self.assertGreaterEqual(hi, Q(1, 2))
        self.assertLessEqual(hi - lo, Q(1, 1 << 96))

    def test_power_enclosures_have_correct_polarity(self):
        self.assertEqual(pc.cube_root_bounds(Q(8)), (Q(2), Q(2)))
        self.assertEqual(pc.quarter_power_bounds(Q(16)), (Q(2), Q(2)))
        lo, hi = pc.coefficient_interval()
        self.assertGreater(lo, Q(1, 4))
        self.assertLess(hi, Q(13, 50))
        self.assertLessEqual(pc.error_interval(8)[1], Q(41, 524288))

    def test_budget_zero_preserves_no_execution_credit(self):
        row = pc.direct_mass(8, max_evaluations=0)
        self.assertEqual(row['status'], 'INCONCLUSIVE_BUDGET')
        self.assertEqual(row['evaluations'], 0)
        self.assertIsNone(row['normalized_interval'])

    def test_partial_budget_counts_actual_evaluations(self):
        row = pc.direct_mass(1, max_evaluations=3)
        self.assertEqual(row['status'], 'INCONCLUSIVE_BUDGET')
        self.assertEqual(row['evaluations'], 3)
        self.assertIsNone(row['interval'])
        self.assertEqual(row['partial']['core_nodes'], 3)

    def test_one_complete_direct_row_has_sound_widths_and_cdf_overlap(self):
        # Removing either sign, arithmetic-width accounting, or tail integration
        # breaks the separate-route overlap or the literal work/width controls.
        row = pc.direct_mass(1)
        cdf = pc.cdf_mass(Q(1, 8))
        self.assertEqual(row['status'], 'CERTIFIED')
        self.assertEqual(row['evaluations'], 67586)
        self.assertTrue(all(row['gates'].values()))
        self.assertLess(row['normalized_width'], Q(1, 1 << 17))
        self.assertEqual(sum(row['arithmetic_components'].values(), Q(0)),
                         row['arithmetic_width'])
        self.assertLessEqual(row['arithmetic_width'], Q(1, 1 << 22))
        self.assertGreater(row['interval'][0], Q(0))
        self.assertLess(row['interval'][1], Q(1, 5))
        self.assertLessEqual(max(row['interval'][0], cdf[0]),
                             min(row['interval'][1], cdf[1]))

    def test_void_encloses_literal_four_copy_probability(self):
        lo, hi = pc.void_interval((Q(1, 5), Q(1, 5)), 4)
        self.assertLessEqual(lo, Q(256, 625))
        self.assertGreaterEqual(hi, Q(256, 625))
        self.assertLess(hi - lo, Q(1, 1 << 16))
        self.assertEqual(pc.void_interval((Q(0), Q(0)), 4), (Q(1), Q(1)))
        self.assertEqual(pc.void_interval((Q(1), Q(1)), 4), (Q(0), Q(0)))

    def test_routes_reject_invalid_math_and_schedule_inputs(self):
        for bad in (0.5, True, 0, Q(-1)):
            with self.assertRaises(ValueError):
                pc.cdf_mass(bad)
        for bad in (Q(0), Q(9, 4), False):
            with self.assertRaises(ValueError):
                pc.root_bracket(bad)
        for bad in (0, 9, True, Q(1)):
            with self.assertRaises(ValueError):
                pc.direct_mass(bad, max_evaluations=0)
        with self.assertRaises(ValueError):
            pc.direct_mass(1, max_evaluations=80001)
        with self.assertRaises(ValueError):
            pc.void_interval((Q(0), Q(1)), True)
        with self.assertRaises(ValueError):
            pc.void_interval((Q(-1), Q(0)), 1)


class RunnerCustodyTests(unittest.TestCase):
    def test_source_drift_withdraws_nested_void_acceptance(self):
        # Literal custody payload; no scientific execution is represented here.
        certificate = {'status': 'PASS', 'rows': [
            {'status': 'PASS', 'coefficient_status': 'PASS',
             'finite_void': {'status': 'PASS'}}],
            'scope_statuses': {'exact_non_gaussian_population': 'PASS'}}
        runner.apply_source_drift(certificate, ['PROOF.md'])
        row = certificate['rows'][0]
        self.assertEqual(certificate['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertEqual(row['finite_void']['status'], 'NOT_ACCEPTED_SOURCE_DRIFT')
        self.assertEqual(row['finite_void'].get('computed_status_before_source_drift'), 'PASS')

    def test_source_drift_disqualifies_full_campaign_receipt(self):
        certificate = {'rows': [{'direct': {'status': 'CERTIFIED'}} for _ in range(8)]}
        self.assertTrue(runner.full_declared_campaign(certificate, 80000, 900, []))
        self.assertFalse(runner.full_declared_campaign(certificate, 80000, 900, ['PROOF.md']))

    def test_final_timer_accounts_for_posthash_and_serialization_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'attempt'
            run = runner.run(output_dir=output, max_evaluations=0, time_budget=0)
            certificate = runner.load_json(output / 'CERTIFICATE.json')
            self.assertEqual(run.get('final_runtime_budget_gate'), 'INCONCLUSIVE_BUDGET')
            self.assertEqual(certificate.get('computed_status_before_runtime_budget'),
                             'INCONCLUSIVE_BUDGET')
            self.assertEqual(certificate['status'], 'INCONCLUSIVE_BUDGET')
            self.assertEqual(len(certificate['rows']), 8)
            self.assertIn('serialization', run['timer_scope'])
            self.assertIn('excluded', run['timer_scope'])

    def test_final_timer_withdraws_literal_accepted_statuses_on_expired_clock(self):
        # This is a custody payload fixture, not an executed scientific result.
        certificate = {'status': 'PASS', 'rows': [
            {'status': 'PASS', 'coefficient_status': 'PASS',
             'finite_void': {'status': 'PASS'}}],
            'scope_statuses': {'exact_non_gaussian_population': 'PASS'}}
        expired = time.monotonic() - 1
        self.assertTrue(runner.apply_final_runtime_budget(certificate, expired, 0))
        self.assertEqual(certificate['computed_status_before_runtime_budget'], 'PASS')
        self.assertEqual(certificate['status'], 'INCONCLUSIVE_BUDGET')
        row = certificate['rows'][0]
        self.assertEqual(row['computed_status_before_runtime_budget'], 'PASS')
        self.assertEqual(row['computed_coefficient_status_before_runtime_budget'], 'PASS')
        self.assertEqual(row['coefficient_status'], 'INCONCLUSIVE_BUDGET')
        self.assertEqual(row['finite_void']['computed_status_before_runtime_budget'], 'PASS')
        self.assertEqual(row['finite_void']['status'], 'INCONCLUSIVE_BUDGET')
        self.assertEqual(certificate['scope_statuses']['exact_non_gaussian_population'],
                         'INCONCLUSIVE_BUDGET')

    def test_preexisting_frozen_source_drift_prevents_population_evaluation(self):
        # Use an actual copied nine-file source cut, not a mocked source reader.
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for source in runner.SOURCE_PATHS:
                target = base / source.relative_to(runner.REPOSITORY)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            contract = base / 'experiments/universality/two_parameter_cusp_population/CONTROLLED_FALSIFICATION.md'
            contract.write_bytes(contract.read_bytes() + b'\n')
            copy_path = base / runner.POPULATION.relative_to(runner.REPOSITORY) / 'run_population_certificate.py'
            spec = importlib.util.spec_from_file_location('copied_runner_control', copy_path)
            copied = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copied)
            copied.run(output_dir=base / 'attempt', max_evaluations=0)
            certificate = copied.load_json(base / 'attempt/CERTIFICATE.json')
            self.assertEqual(certificate['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
            self.assertEqual(certificate['rows'], [])
            self.assertEqual(certificate['scope_statuses']['exact_non_gaussian_population'],
                             'INCONCLUSIVE_SOURCE_DRIFT')

    def test_frozen_manifest_detects_real_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.py'
            path.write_text('first\n')
            before = runner.freeze_sources([path], Path(temporary))
            self.assertEqual(runner.source_drift(before, Path(temporary)), [])
            path.write_text('second\n')
            self.assertEqual(runner.source_drift(before, Path(temporary)), ['source.py'])

    def test_zero_budget_attempt_serializes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'attempt'
            result = runner.run(output_dir=output, max_evaluations=0)
            self.assertEqual(result['exit_code'], 2)
            certificate = runner.load_json(output / 'CERTIFICATE.json')
            run = runner.load_json(output / 'RUN.json')
            self.assertEqual(certificate['status'], 'INCONCLUSIVE_BUDGET')
            self.assertEqual(len(certificate['rows']), 8)
            self.assertEqual(run['actual_route_evaluations'], 0)
            self.assertEqual(len(run['sources_before']), 9)
            self.assertFalse(run['full_declared_campaign'])
            original = (output / 'RUN.json').read_bytes()
            with self.assertRaises(FileExistsError):
                runner.run(output_dir=output, max_evaluations=0)
            self.assertEqual((output / 'RUN.json').read_bytes(), original)

    def test_json_rejects_duplicate_nonfinite_and_float_math(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.json'
            for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":0.5}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    runner.load_json(path)


if __name__ == '__main__':
    unittest.main()
