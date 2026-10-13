"""Literal fixtures for the controlled exact-population certificate.

These tests exercise actual selectors, interval routes and filesystem custody.
No full eight-row integration campaign is performed by this test file.
"""
from fractions import Fraction as Q
from pathlib import Path
import importlib.util
import tempfile
import time
import types
import unittest
from unittest import mock

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


def stub_pc(b_shift=Q(0), cdf_shift=Q(0), override=None, void=None):
    """Module stand-in for the campaign wiring. The direct route replays
    n * cdf_mass(tau) widened by 2^-20 per side as CERTIFIED; optional literal
    overrides for a row (both routes), a coefficient shift, a CDF shift or a
    fixed void interval inject the fault under test. No scientific execution is
    represented by any certificate built from it."""
    override = override or {}
    stub = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})

    def cdf_mass(tau):
        for j, (lo, hi) in override.items():
            t, n = pc._schedule(j)
            if tau == t:
                return lo / n, hi / n
        return tuple(x + cdf_shift for x in pc.cdf_mass(tau))

    def direct_mass(j, max_evaluations=80000, deadline=None):
        tau, n = pc._schedule(j)
        if j in override:
            lo, hi = override[j]
        else:
            m = pc.cdf_mass(tau)
            lo, hi = n * m[0] - Q(1, 1 << 20), n * m[1] + Q(1, 1 << 20)
        return {'j': j, 'tau': tau, 'n': n, 'status': 'CERTIFIED', 'evaluations': 1,
                'normalized_interval': (lo, hi), 'interval': (lo / n, hi / n)}

    base_b = pc.coefficient_interval()
    stub.cdf_mass, stub.direct_mass = cdf_mass, direct_mass
    stub.coefficient_interval = lambda: (base_b[0] + b_shift, base_b[1] + b_shift)
    stub.poisson_target = lambda b=None: pc.poisson_target(stub.coefficient_interval() if b is None else b)
    if void is not None:
        stub.void_interval = lambda mass, n: void
    return stub


def campaign(**kw):
    return runner._campaign(stub_pc(**kw), 80000, time.monotonic() + 900)


class EnclosureTests(unittest.TestCase):
    # A4: enclosure soundness (sqrt lower endpoint, root sliver, tail trapezoid
    # upper bound, CERTIFIED only when every direct gate holds).
    def test_power_enclosures_bracket_irrational_radicands(self):
        """Kills M02 (sqrt lower endpoint rounded up). Exact radicands (8, 16) cannot see polarity."""
        for v in (Q(2), Q(3), Q(1, 3), Q(5, 7)):
            lo, hi = pc._sqrt((v, v))
            self.assertTrue(lo < hi and lo * lo <= v <= hi * hi, (v, lo, hi))
            lo, hi = pc.quarter_power_bounds(v)
            self.assertTrue(lo < hi and lo ** 4 <= v <= hi ** 4, (v, lo, hi))
            lo, hi = pc.cube_root_bounds(v)
            self.assertTrue(lo < hi and lo ** 3 <= v <= hi ** 3, (v, lo, hi))

    def test_direct_row_replays_its_component_composition(self):
        """Kills M04 (root sliver dropped) and M06 (tail trapezoid replaced by midpoint sum)."""
        row = pc.direct_mass(1)
        core, tail = row['core_interval'], row['tail_interval']
        self.assertEqual(row['normalized_interval'],
                         (core[0] + tail[0], core[1] + tail[1] + row['sliver_width']))
        self.assertGreater(row['sliver_width'], 0)
        self.assertGreater(row['discretization_components']['core'], 0)
        self.assertGreater(row['discretization_components']['tail'], 0)
        self.assertEqual(row['interval'], tuple(x / row['n'] for x in row['normalized_interval']))
        self.assertEqual((row['tau'], row['n']), (Q(1, 8), 4))

    def test_failed_direct_gate_is_not_certified(self):
        """Kills M16 (status CERTIFIED regardless of gates)."""
        with mock.patch.object(pc, 'CORE_PANELS', 64):
            row = pc.direct_mass(1)
        self.assertFalse(row['gates']['normalized_width'])
        self.assertEqual(row['status'], 'INCONCLUSIVE_PRECISION')
        self.assertGreaterEqual(row['normalized_width'], Q(1, 1 << 17))

    def test_saturation_is_reached_by_the_closed_form_not_only_the_literal(self):
        """The literal branch cdf_mass(9/4) == 1/5 cannot fail; this checks the computed route."""
        lo, hi = pc.cdf_mass(Q(9, 4) - Q(1, 1 << 40))
        self.assertLess(Q(1, 5) - lo, Q(1, 1 << 80))
        self.assertLessEqual(hi, Q(1, 5))

    def test_void_interval_replays_its_outward_log_exp_composition(self):
        """Kills M03a (void endpoints taken inward); containment of (1-m)^n alone cannot see it."""
        def exponent(m, n):
            lo, hi = pc.log_bounds(1 - m, bits=pc.BITS)
            return pc._outward((-n * hi, -n * lo))
        # Normalized mass n*m in (1/7, 1/5) at n = 4 and at the row-8 count, plus a point mass.
        for mass, n in (((Q(1, 7), Q(1, 5)), 4), ((Q(1, 7 * 4 ** 8), Q(1, 5 * 4 ** 8)), 4 ** 8),
                        ((Q(1, 3), Q(1, 3)), 4)):
            lo, hi = pc.void_interval(mass, n)
            self.assertEqual((lo, hi), (pc.exp_neg_bounds(exponent(mass[1], n)[1])[0],
                                        pc.exp_neg_bounds(exponent(mass[0], n)[0])[1]))
            self.assertTrue(lo < (1 - mass[1]) ** n <= (1 - mass[0]) ** n < hi, (mass, n))

    def test_poisson_target_replays_its_outward_exp_composition(self):
        """Kills M03b (Poisson target endpoints taken inward); exp(-B) is contained either way."""
        for b in (pc.coefficient_interval(), (Q(1, 3), Q(1, 2))):
            lo, hi = pc.poisson_target(b)
            self.assertEqual((lo, hi), (pc.exp_neg_bounds(b[1])[0], pc.exp_neg_bounds(b[0])[1]))
            self.assertLess(lo, hi)
        self.assertEqual(pc.poisson_target(), pc.poisson_target(pc.coefficient_interval()))


class PredicateFixtureTests(unittest.TestCase):
    # A5: predicate fixtures strictly between the outer and inner envelopes,
    # the void width gate at its declared limit, and the declared overall priority.
    def test_coefficient_inconclusive_strictly_between_outer_and_inner(self):
        """Kills M08/M08b (FALSIFIED threshold moved from the outer to the inner envelope)."""
        b, e = (Q(1, 5), Q(3, 10)), (Q(1, 10), Q(1, 10))   # inner=[1/5,3/10], outer=[1/10,2/5]
        self.assertEqual(pc.classify_coefficient((Q(12, 100), Q(15, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(pc.classify_coefficient((Q(35, 100), Q(38, 100)), b, e), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(pc.classify_coefficient((Q(9, 100), Q(99, 1000)), b, e), 'FALSIFIED')

    def test_void_width_gate_at_its_declared_limit(self):
        """Kills M15 (WIDTH_LIMIT relaxed 256x); the existing (0,1/4) fixture is 16384x over the limit."""
        target = (Q(1, 2), Q(1, 2))
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16)), target), 'PASS')
        self.assertEqual(pc.classify_void((Q(1, 2), Q(1, 2) + Q(1, 1 << 16) + Q(1, 1 << 40)), target),
                         'INCONCLUSIVE_PRECISION')

    def test_overall_priority_is_the_declared_one(self):
        """Kills M20 (FALSIFIED dropped from the priority list)."""
        self.assertEqual(runner._overall(['PASS'] * 8), 'PASS')
        self.assertEqual(runner._overall(['PASS'] * 7 + ['FALSIFIED']), 'FALSIFIED')
        self.assertEqual(runner._overall(['FALSIFIED', 'INCONCLUSIVE_PRECISION']), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(runner._overall(['FALSIFIED', 'FAIL_IMPLEMENTATION']), 'FAIL_IMPLEMENTATION')
        self.assertEqual(runner._overall([]), 'INCONCLUSIVE_PRECISION')

    def test_overall_requires_every_status_to_pass(self):
        """Kills M21 (PASS if any status passes): an undeclared status beside PASS is no pass."""
        self.assertEqual(runner._overall(['PASS', 'NOT_A_DECLARED_STATUS']), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(runner._overall(['NOT_RUN', 'PASS', 'PASS']), 'INCONCLUSIVE_PRECISION')
        self.assertEqual(runner._overall(['PASS', 'PASS']), 'PASS')
        self.assertEqual(runner._overall([]), 'INCONCLUSIVE_PRECISION')


class NegativeControlTests(unittest.TestCase):
    # A1: the mass negative controls must be able to fail; an injected fixed-left
    # selector has to trip the declared gates instead of comparing literals.
    DECLARED = {'all_stationary_and_inside_rectangle', 'chart_gaps', 'outer_height_identity',
                'actual_theta2_gap', 'reject_fixed_left', 'boundary_exclusion',
                'whole_rectangle_mass', 'missing_sign_detected', 'conditioning_detected'}

    def test_negative_controls_evaluate_every_declared_gate(self):
        """Kills M10 (a gate silently deleted)."""
        result = pc.negative_controls()
        self.assertEqual(set(result['gates']), self.DECLARED)
        self.assertTrue(all(result['gates'].values()))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(result['controls']), 14)

    def test_negative_controls_fail_when_the_selector_is_fixed_left(self):
        """Kills M11 (status PASS regardless of gates): an injected fixed-left lifetime must fail."""
        real = pc.critical_fixture

        def fixed_left(a, h, sign=1):
            row = real(a, h, sign)
            if h == 2 * a:
                row['lifetime'] = 32 * a ** 4
            return row
        with mock.patch.object(pc, 'critical_fixture', fixed_left):
            result = pc.negative_controls()
        self.assertFalse(result['gates']['actual_theta2_gap'])
        self.assertFalse(result['gates']['reject_fixed_left'])
        self.assertEqual(result['status'], 'FAIL_IMPLEMENTATION')


class CampaignWiringTests(unittest.TestCase):
    # A2: the acceptance wiring of runner._campaign (route intersection, coefficient
    # and void verdicts, row-8 gates, exit code, full-campaign receipt) is executed
    # on literal stubs. A3: the end-of-run source-drift check is exercised.
    def test_consistent_routes_and_correct_coefficient_pass(self):
        """Executes runner lines 193-235 on a known-good stub; kills M24 (runner schedule diverging from pc)."""
        cert = campaign()
        self.assertEqual(cert['status'], 'PASS')
        self.assertEqual([r['status'] for r in cert['rows']], ['PASS'] * 8)
        self.assertEqual(cert['rows'][7]['finite_void']['status'], 'PASS')
        for r in cert['rows']:
            self.assertEqual(r['intersection_normalized'],
                             (max(r['direct']['normalized_interval'][0], r['cdf_normalized'][0]),
                              min(r['direct']['normalized_interval'][1], r['cdf_normalized'][1])))

    def test_wrong_coefficient_is_falsified_at_row_eight_only(self):
        """Kills M19 (row status PASS regardless) and M20; B shifted by 1.2e-4 > E(tau_8) = 7.8e-5."""
        cert = campaign(b_shift=Q(3, 25000))
        row8 = cert['rows'][7]
        self.assertEqual(row8['coefficient_status'], 'FALSIFIED')
        self.assertEqual(row8['status'], 'FALSIFIED')
        self.assertEqual(row8['finite_void']['status'], 'PASS')
        self.assertEqual(cert['status'], 'FALSIFIED')
        self.assertEqual([r['status'] for r in cert['rows'][:7]], ['PASS'] * 7)

    def test_disjoint_routes_are_route_consistency_failures(self):
        """Kills M17 (overlap predicate disabled) and M18 (CDF route ignored in the intersection)."""
        cert = campaign(cdf_shift=Q(1, 100))
        self.assertEqual([r['status'] for r in cert['rows']], ['FAIL_ROUTE_CONSISTENCY'] * 8)
        self.assertEqual(cert['status'], 'FAIL_ROUTE_CONSISTENCY')
        self.assertTrue(all(r['coefficient_status'] == 'NOT_RUN' for r in cert['rows']))

    def test_row_eight_mass_gate_withholds_the_void_test(self):
        """Kills M25 (mass gate trivially true)."""
        cert = campaign(override={8: (Q(34, 100), Q(34, 100) + Q(1, 1 << 20))})
        row8 = cert['rows'][7]
        self.assertFalse(row8['finite_void']['gates']['mass_gate'])
        self.assertEqual(row8['finite_void']['status'], 'INCONCLUSIVE_PRECISION')
        self.assertEqual(row8['status'], 'INCONCLUSIVE_PRECISION')

    def test_wide_void_interval_is_inconclusive_even_with_a_passing_coefficient(self):
        """Kills M28 (void/target width gates ignored)."""
        cert = campaign(void=(Q(0), Q(1, 4)))
        row1 = cert['rows'][0]
        self.assertEqual(row1['coefficient_status'], 'PASS')
        self.assertFalse(row1['void_width_gate'])
        self.assertEqual(row1['status'], 'INCONCLUSIVE_PRECISION')
        self.assertEqual(cert['status'], 'INCONCLUSIVE_PRECISION')

    def test_falsified_attempt_exits_one_and_is_no_full_campaign_under_reduced_budget(self):
        """Kills M26 (exit 0 on FALSIFIED) and M27 (full campaign receipt ignores budget arguments)."""
        falsified = campaign(b_shift=Q(3, 25000))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, '_campaign', lambda *a: falsified):
            rec = runner.run(output_dir=Path(tmp) / 'attempt', max_evaluations=80000, time_budget=900)
        self.assertEqual(rec['status'], 'FALSIFIED')
        self.assertEqual(rec['exit_code'], 1)
        certified = {'rows': [{'direct': {'status': 'CERTIFIED'}} for _ in range(8)]}
        self.assertTrue(runner.full_declared_campaign(certified, 80000, 900, []))
        self.assertFalse(runner.full_declared_campaign(certified, 79999, 900, []))
        self.assertFalse(runner.full_declared_campaign(certified, 80000, 899, []))

    def test_source_edited_during_evaluation_is_source_drift(self):
        """Kills M22 (post-run hash comparison dropped). Contract: 'check them again at the end'."""
        # Same copied nine-file source cut as the pre-run drift test above.
        population = runner.POPULATION.relative_to(runner.REPOSITORY)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for source in runner.SOURCE_PATHS:
                target = base / source.relative_to(runner.REPOSITORY)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            copy_path = base / population / 'run_population_certificate.py'
            spec = importlib.util.spec_from_file_location('copied_runner_drift', copy_path)
            copied = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copied)
            proof = base / population / 'PROOF.md'

            def campaign_that_edits_a_source(pc_module, max_evaluations, deadline):
                proof.write_bytes(proof.read_bytes() + b'\n')
                return {'schema_version': 1, 'status': 'PASS', 'rows': [], 'scope_statuses':
                        {'exact_non_gaussian_population': 'PASS'}}
            copied._campaign = campaign_that_edits_a_source
            rec = copied.run(output_dir=base / 'attempt', max_evaluations=80000, time_budget=900)
            self.assertEqual(rec['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
            self.assertEqual(rec['exit_code'], 2)
            self.assertEqual(rec['source_drift'], [str(population / 'PROOF.md')])
            self.assertFalse(rec['full_declared_campaign'])


if __name__ == '__main__':
    unittest.main()
