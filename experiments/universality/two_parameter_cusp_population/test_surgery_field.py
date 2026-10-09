"""Behavioral tests for the complete field, independent of any H0 pairing."""
from fractions import Fraction as Q
import math
import time
import unittest
from unittest.mock import patch

import surgery_field as sf


def budget(**limits):
    return sf.Budget(time.monotonic(), **limits)


class ArithmeticTests(unittest.TestCase):
    def test_outward_rounding_contains_nondyadic_and_negative_values(self):
        lo, hi = sf._outward(Q(-1, 3), Q(2, 7))
        self.assertLessEqual(lo, Q(-1, 3))
        self.assertGreaterEqual(hi, Q(2, 7))
        self.assertLess(Q(-1, 3) - lo, Q(1, 2**128))
        self.assertLess(hi - Q(2, 7), Q(1, 2**128))
        self.assertEqual((lo * 2**128).denominator, 1)
        self.assertEqual((hi * 2**128).denominator, 1)

    def test_multiplication_uses_all_signed_endpoint_products(self):
        self.assertEqual(sf._mul((Q(-2), Q(3)), (Q(-5), Q(7))),
                         (Q(-15), Q(21)))
        self.assertEqual(sf._mul((Q(-4), Q(-2)), (Q(-3), Q(-1))),
                         (Q(2), Q(12)))

    def test_square_crossing_zero_keeps_zero_lower_endpoint(self):
        self.assertEqual(sf._square((Q(-2), Q(3))), (Q(0), Q(9)))
        self.assertEqual(sf._square((Q(-3), Q(-2))), (Q(4), Q(9)))

    def test_division_handles_signed_numerator_and_positive_denominator(self):
        self.assertEqual(sf._div((Q(-6), Q(3)), (Q(2), Q(4))),
                         (Q(-3), Q(3, 2)))

    def test_division_refuses_uncertified_positive_denominator(self):
        for denominator in ((Q(0), Q(1)), (Q(-1), Q(2)), (Q(-2), Q(-1))):
            with self.subTest(denominator=denominator):
                with self.assertRaises(ValueError):
                    sf._div((Q(1), Q(2)), denominator)


class PrimitiveTests(unittest.TestCase):
    def assert_contains(self, interval, exact):
        self.assertLessEqual(interval[0], exact)
        self.assertGreaterEqual(interval[1], exact)

    def test_sine_zero_and_odd_reflection(self):
        b = budget()
        self.assert_contains(sf._sin_bounds((Q(0), Q(0)), budget=b), Q(0))
        positive = sf._sin_bounds((Q(1), Q(1)), budget=b)
        negative = sf._sin_bounds((Q(-1), Q(-1)), budget=b)
        self.assertEqual(negative, (-positive[1], -positive[0]))
        # Independent alternating Taylor bounds: P19 <= sin(1) <= P17.
        p17 = sum((Q((-1)**k, math.factorial(2*k+1)) for k in range(9)), Q(0))
        p19 = p17 - Q(1, math.factorial(19))
        self.assertGreaterEqual(positive[0], p19)
        self.assertLessEqual(positive[1], p17)

    def test_sine_remainder_domain_and_pi_half_identity(self):
        self.assertEqual(sf.SINE_REMAINDER, Q(2**129, math.factorial(129)))
        b = budget()
        pi = sf._pi_bounds(budget=b)
        result = sf._sin_bounds((pi[0]/2, pi[1]/2), budget=b)
        self.assert_contains(result, Q(1))
        self.assertLess(result[1]-result[0], Q(1, 2**110))
        with self.assertRaises(ValueError):
            sf._sin_bounds((Q(-2), Q(2)+Q(1, 2**160)), budget=b)

    def test_q_zero_extension_and_tiny_tail_without_exponential_request(self):
        b = budget()
        self.assertEqual(sf._q_bounds((Q(-2), Q(0)), budget=b), (Q(0), Q(0)))
        self.assertEqual(sf._q_bounds((Q(1, 8192), Q(1, 8192)), budget=b),
                         (Q(0), Q(1, 2**4096)))
        self.assertEqual(b.snapshot()['primitive_requests'], 0)

    def test_q_monotone_endpoint_enclosure_and_boundary_argument(self):
        b = budget()
        wider = sf._q_bounds((Q(-1), Q(1)), budget=b)
        self.assertEqual(wider[0], Q(0))
        self.assertTrue(Q(1, 3) < wider[1] < Q(1, 2))
        edge = sf._q_bounds((Q(1, 4096), Q(1, 4096)), budget=b)
        self.assertTrue(Q(0) <= edge[0] <= edge[1] <= Q(1, 2**128))
        self.assertEqual(b.snapshot()['primitive_requests_by_kind']['exp'], 3)

    def test_psi_plateaus_midpoint_and_descending_interval(self):
        b = budget()
        self.assertEqual(sf._psi_bounds((Q(-2), Q(0)), budget=b), (Q(1), Q(1)))
        self.assertEqual(sf._psi_bounds((Q(1), Q(3)), budget=b), (Q(0), Q(0)))
        self.assert_contains(sf._psi_bounds((Q(1, 2), Q(1, 2)), budget=b), Q(1, 2))
        transition = sf._psi_bounds((Q(1, 4), Q(3, 4)), budget=b)
        self.assertTrue(Q(0) < transition[0] < Q(1, 2) < transition[1] < Q(1))
        self.assertEqual(sf._psi_bounds((Q(-1), Q(2)), budget=b), (Q(0), Q(1)))

    def test_psi_does_not_invent_a_positive_denominator(self):
        with patch.object(sf, '_q_bounds', return_value=(Q(0), Q(0))):
            with self.assertRaises(sf.Inconclusive) as caught:
                sf._psi_bounds((Q(1, 3), Q(1, 3)), budget=budget())
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_PRECISION')

    def test_psi_extreme_transition_endpoints_keep_certified_denominator(self):
        b = budget()
        left = sf._psi_bounds((Q(1, 8192), Q(1, 8192)), budget=b)
        right = sf._psi_bounds((Q(8191, 8192), Q(8191, 8192)), budget=b)
        self.assertTrue(Q(1)-Q(1, 2**110) < left[0] <= left[1] <= Q(1))
        self.assertTrue(Q(0) <= right[0] <= right[1] < Q(1, 2**110))

    def test_primitive_requests_include_repeated_cached_exponential_calls(self):
        b = budget()
        sf._q_bounds((Q(1), Q(1)), budget=b)
        sf._q_bounds((Q(1), Q(1)), budget=b)
        self.assertEqual(b.snapshot()['primitive_requests_by_kind']['exp'], 4)
        sf._pi_bounds(budget=b)
        sf._sqrt_bounds(Q(2, 3), budget=b)
        sf._sin_bounds((Q(0), Q(0)), budget=b)
        self.assertEqual(b.snapshot()['primitive_requests'], 7)


class FieldTests(unittest.TestCase):
    def assert_contains(self, interval, exact):
        self.assertLessEqual(interval[0], exact)
        self.assertGreaterEqual(interval[1], exact)
        self.assertLessEqual(interval[1]-interval[0], Q(1, 2**104))

    def test_core_value_contains_exact_full_polynomial(self):
        # Literal exact centered value at s=1/128,z=1/256,u=2^-34,v=-2^-49.
        result = sf.chart_field_bounds(Q(1, 2**34), Q(-1, 2**49),
                                       (Q(1, 128), Q(1, 128)),
                                       (Q(1, 256), Q(1, 256)), budget=budget())
        expected = Q(4, 9)-Q(1, 2**30)-Q(1, 2**16)-Q(1, 2**32)+Q(127, 2**56)
        self.assert_contains(result, expected)

    def test_origin_and_seed_collar(self):
        b = budget()
        self.assert_contains(sf.chart_field_bounds(Q(0), Q(0), (Q(0), Q(0)),
                                                   (Q(0), Q(0)), budget=b), Q(4, 9))
        self.assert_contains(sf.chart_field_bounds(Q(1, 2**34), Q(1, 2**49),
                                                   (Q(3, 32), Q(3, 32)),
                                                   (Q(0), Q(0)), budget=b),
                             Q(4, 9)-Q(9, 1024))
        self.assertIn('seed_collar', b.snapshot()['field_records'][-1]['branches'])

    def test_chi_and_surgery_transitions_change_the_actual_field(self):
        b = budget()
        # At r=9*r1^2/16 chi is strictly between zero and one, c=r/4.
        lo, hi = sf.chart_field_bounds(Q(1, 2**34), Q(0),
                                       (Q(3, 128), Q(3, 128)),
                                       (Q(0), Q(0)), budget=b)
        base = Q(4, 9)-Q(81, 2**30)
        perturbation = Q(9, 2**49)
        self.assertTrue(base < lo < hi < base+perturbation)
        self.assertIn('chi_transition', b.snapshot()['field_records'][-1]['branches'])
        # At s=1/16 c is strictly between r/4 and one; chi=0.
        lo, hi = sf.chart_field_bounds(Q(0), Q(0), (Q(1, 16), Q(1, 16)),
                                       (Q(0), Q(0)), budget=b)
        self.assertTrue(Q(4, 9)-Q(1, 256) < lo < hi < Q(4, 9)-Q(1, 2**18))
        self.assertIn('surgery_transition', b.snapshot()['field_records'][-1]['branches'])

    def test_plateau_boundaries_replay_core_and_seed(self):
        b = budget()
        for s in (Q(1, 64), Q(1, 32)):
            result = sf.chart_field_bounds(Q(0), Q(0), (s, s), (Q(0), Q(0)), budget=b)
            self.assert_contains(result, Q(4, 9)-s**4/4)
        # Surgery endpoint r=R^2/2, realized by two exact chart coordinates.
        result = sf.chart_field_bounds(Q(0), Q(0), (Q(1, 16), Q(1, 16)),
                                       (Q(1, 16), Q(1, 16)), budget=b)
        self.assert_contains(result, Q(4, 9)-Q(1, 128))

    def test_torus_old_critical_values_and_integer_seam_translations(self):
        b = budget()
        for point, value in (((Q(0), Q(1, 2)), Q(2, 9)),
                             ((Q(1, 2), Q(0)), Q(-2, 9)),
                             ((Q(1, 2), Q(1, 2)), Q(-4, 9))):
            original = sf.torus_field_bounds(Q(0), Q(0), point, budget=b)
            translated = sf.torus_field_bounds(Q(0), Q(0),
                                               (point[0]+1, point[1]-1), budget=b)
            self.assertEqual(original, translated)
            self.assert_contains(original, value)
            records = b.snapshot()['field_records'][-2:]
            self.assertEqual(records[0]['canonical_point'], records[1]['canonical_point'])
            self.assertIn('seed_collar', records[0]['branches'])
        for point in ((Q(-1, 2), Q(1, 8)), (Q(1, 8), Q(-1, 2)),
                      (Q(1, 4), Q(0)), (Q(0), Q(1, 4))):
            original = sf.torus_field_bounds(Q(1, 2**34), Q(1, 2**49), point, budget=b)
            translated = sf.torus_field_bounds(Q(1, 2**34), Q(1, 2**49),
                                               (point[0]+1, point[1]-1), budget=b)
            self.assertEqual(original, translated)

    def test_records_expose_input_model_branches_output_and_gates(self):
        b = budget()
        output = sf.torus_field_bounds(Q(0), Q(0), (Q(1), Q(-1)), budget=b)
        row = b.snapshot()['field_records'][0]
        self.assertEqual(row['input_point'], (Q(1), Q(-1)))
        self.assertEqual(row['canonical_point'], (Q(0), Q(0)))
        self.assertEqual(row['returned_interval'], output)
        self.assertEqual(row['model']['model_id'], 'explicit_periodic_cusp_v1')
        self.assertEqual(row['model']['analytic_source']['sha256'],
                         '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9')
        self.assertTrue(row['gates']['width'] and row['gates']['identity'])
        row['model']['model_id'] = 'tampered_snapshot'
        self.assertEqual(b.snapshot()['field_records'][0]['model']['model_id'],
                         'explicit_periodic_cusp_v1')

    def test_chart_rejects_box_not_wholly_in_domain(self):
        with self.assertRaises(ValueError):
            sf.chart_field_bounds(Q(0), Q(0), (Q(-1, 8), Q(1, 8)),
                                  (Q(1, 128), Q(1, 128)), budget=budget())

    def test_missing_evaluator_source_withholds_field_and_retains_call(self):
        b = budget()
        read = sf._read_source
        def missing_evaluator(path):
            if path.name == 'surgery_field.py':
                raise FileNotFoundError('controlled missing evaluator source')
            return read(path)
        with patch.object(sf, '_read_source', side_effect=missing_evaluator):
            with self.assertRaises(sf.Inconclusive) as caught:
                sf.chart_field_bounds(Q(0), Q(0), (Q(0), Q(0)),
                                      (Q(0), Q(0)), budget=b)
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_SOURCE_DRIFT')
        row = b.snapshot()['field_records'][0]
        self.assertEqual(row['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertFalse(row['gates']['identity'])

    def test_changed_evaluator_source_withholds_field_and_retains_call(self):
        b = budget()
        read = sf._read_source
        def changed_evaluator(path):
            return b'changed evaluator' if path.name == 'surgery_field.py' else read(path)
        with patch.object(sf, '_read_source', side_effect=changed_evaluator):
            with self.assertRaises(sf.Inconclusive) as caught:
                sf.chart_field_bounds(Q(0), Q(0), (Q(0), Q(0)),
                                      (Q(0), Q(0)), budget=b)
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertEqual(b.snapshot()['field_records'][0]['status'], 'INCONCLUSIVE_SOURCE_DRIFT')

    def test_wide_valid_box_retains_record_and_precision_disposition(self):
        b = budget()
        with self.assertRaises(sf.Inconclusive) as caught:
            sf.chart_field_bounds(Q(0), Q(0), (Q(0), Q(1, 64)),
                                  (Q(0), Q(0)), budget=b)
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_PRECISION')
        row = b.snapshot()['field_records'][-1]
        self.assertFalse(row['gates']['width'])
        self.assertIsNotNone(row['returned_interval'])

    def test_invalid_mathematical_types_controls_and_reversed_intervals(self):
        for u, v in ((0.0, Q(0)), (True, Q(0)), (0, Q(0))):
            with self.subTest(u=u, v=v):
                with self.assertRaises(TypeError):
                    sf.torus_field_bounds(u, v, (Q(0), Q(0)), budget=budget())
        for u, v in ((Q(3, 2**32), Q(0)), (Q(0), Q(-1, 2**47))):
            with self.assertRaises(ValueError):
                sf.torus_field_bounds(u, v, (Q(0), Q(0)), budget=budget())
        for point in ((Q(0), 0.0), [Q(0), Q(0)], (Q(0), True)):
            with self.assertRaises(TypeError):
                sf.torus_field_bounds(Q(0), Q(0), point, budget=budget())
        with self.assertRaises(ValueError):
            sf.chart_field_bounds(Q(0), Q(0), (Q(1), Q(0)), (Q(0), Q(0)), budget=budget())
        for interval in ((Q(0), 0), [Q(0), Q(0)]):
            with self.assertRaises(TypeError):
                sf.chart_field_bounds(Q(0), Q(0), interval, (Q(0), Q(0)), budget=budget())


class GuardTests(unittest.TestCase):
    def test_exact_premises_and_rational_guard_consequents(self):
        guard = sf.guard_certificate()
        self.assertEqual(guard['status'], 'ACCEPTED_ANALYTIC_PREMISE')
        self.assertEqual(guard['analytic_source']['sha256'],
                         '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9')
        self.assertEqual(guard['accepted_review'], {
            'bytes': 22303,
            'sha256': 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08',
            'status': 'PASS_BOUNDED_ANALYTIC_GUARDS'})
        self.assertEqual(guard['bounds']['gradient_perturbation'], Q(101441, 2**47))
        self.assertEqual(guard['bounds']['height_perturbation'], Q(1537, 2**52))
        self.assertEqual(guard['bounds']['chi_gradient'], Q(1024, 3))
        self.assertTrue(all(guard['checks'].values()))
        self.assertFalse(guard['finite_sampling_proves_smoothness'])

    def test_changed_analytic_bytes_withhold_guard_acceptance(self):
        with patch.object(sf, '_read_source', return_value=b'changed'):
            with self.assertRaises(sf.Inconclusive) as caught:
                sf.guard_certificate()
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_SOURCE_DRIFT')


class BudgetTests(unittest.TestCase):
    def test_refinements_retain_every_admitted_depth_across_fixtures(self):
        b = budget(root_splits=2)
        b.begin_fixture('one')
        b.branch_splits(1)
        b.branch_splits(2)
        b.begin_fixture('two')
        b.branch_splits(1, phase='verify_roots')
        with self.assertRaises(sf.Inconclusive):
            b.branch_splits(3)
        self.assertEqual(b.snapshot().get('root_refinements', []), [
            {'fixture_id': 'one', 'phase': 'roots', 'depth': 1},
            {'fixture_id': 'one', 'phase': 'roots', 'depth': 2},
            {'fixture_id': 'two', 'phase': 'verify_roots', 'depth': 1}])

    def test_future_start_cannot_extend_shared_deadline(self):
        with patch.object(sf.time, 'monotonic', return_value=100.0):
            with self.assertRaises(ValueError):
                sf.Budget(100.1)

    def test_zero_reduced_ceilings_are_enforced(self):
        for limits, kind in (({'field_limit': 0}, 'field'),
                             ({'primitive_limit': 0}, 'primitive'),
                             ({'polynomial_limit': 0}, 'polynomial'),
                             ({'root_splits': 0}, 'root')):
            b = budget(**limits)
            b.begin_fixture('zero')
            call = {'field': b.charge_field, 'primitive': lambda: b.charge_primitive('pi'),
                    'polynomial': b.charge_polynomial, 'root': lambda: b.branch_splits(1)}[kind]
            with self.assertRaises(sf.Inconclusive):
                call()
            self.assertEqual(b.snapshot()['field_calls'], 0)
            self.assertEqual(b.snapshot()['primitive_requests'], 0)
            self.assertEqual(b.snapshot()['polynomial_evaluations'], 0)
        with patch.object(sf.time, 'monotonic', return_value=100.0):
            b = sf.Budget(100.0, seconds=0)
            with self.assertRaises(sf.Inconclusive):
                b.checkpoint()

    def test_unique_fixture_resets_only_polynomial_and_local_algebra_counts(self):
        b = budget(polynomial_limit=2)
        b.begin_fixture('one')
        b.charge_polynomial(2)
        b.charge_gcd()
        b.charge_division(2)
        b.branch_splits(7)
        b.charge_field()
        b.charge_primitive('pi')
        with self.assertRaises(ValueError):
            b.begin_fixture('one')
        self.assertEqual(b.snapshot()['polynomial_evaluations'], 2)
        b.begin_fixture('two')
        work = b.snapshot()
        self.assertEqual(work['polynomial_evaluations'], 0)
        self.assertEqual(work['gcd_steps'], 0)
        self.assertEqual(work['polynomial_divisions'], 0)
        self.assertEqual(work['field_calls'], 1)
        self.assertEqual(work['primitive_requests'], 1)
        self.assertEqual(work['fixtures']['one']['max_branch_splits'], 7)
        self.assertEqual(work['attempt_algebra']['polynomial_divisions'], 2)
        self.assertEqual(work['scopes']['polynomial_evaluations'], 'per_fixture')
        self.assertEqual(work['scopes']['field_calls'], 'whole_attempt')

    def test_reduced_counters_refuse_next_request_without_resetting_evidence(self):
        for kind in ('polynomial', 'field', 'primitive', 'root'):
            b = budget(polynomial_limit=1, field_limit=1, primitive_limit=1, root_splits=1)
            b.begin_fixture('x')
            call = {'polynomial': b.charge_polynomial, 'field': b.charge_field,
                    'primitive': lambda: b.charge_primitive('sqrt'),
                    'root': lambda: b.branch_splits(2)}[kind]
            if kind != 'root':
                call()
            with self.assertRaises(sf.Inconclusive) as caught:
                call()
            self.assertEqual(caught.exception.status, 'INCONCLUSIVE_BUDGET')
            self.assertIn('x', b.snapshot()['fixtures'])

    def test_deadline_is_shared_across_new_fixtures(self):
        with patch.object(sf.time, 'monotonic', return_value=100.0):
            b = sf.Budget(100.0, seconds=2)
            b.begin_fixture('one')
        with patch.object(sf.time, 'monotonic', return_value=101.0):
            b.begin_fixture('two')
        with patch.object(sf.time, 'monotonic', return_value=102.0):
            with self.assertRaises(sf.Inconclusive) as caught:
                b.checkpoint('after_write')
        self.assertEqual(caught.exception.phase, 'after_write')
        self.assertEqual(b.snapshot()['fixture_id'], 'two')

    def test_limits_reject_increases_bool_counts_and_invalid_clock(self):
        for limits in ({'seconds': 901}, {'polynomial_limit': 100001},
                       {'field_limit': 4097}, {'primitive_limit': 32769},
                       {'root_splits': 257}, {'field_limit': True}, {'seconds': -1}):
            with self.assertRaises(ValueError):
                budget(**limits)
        for start in (True, 1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                sf.Budget(start)
        b = budget()
        b.begin_fixture('x')
        for call in (lambda: b.charge_polynomial(True), lambda: b.charge_gcd(True),
                     lambda: b.charge_division(True), lambda: b.branch_splits(True)):
            with self.assertRaises(ValueError):
                call()

    def test_field_primitive_exhaustion_retains_partial_call(self):
        b = budget(primitive_limit=1)
        with self.assertRaises(sf.Inconclusive) as caught:
            sf.torus_field_bounds(Q(0), Q(0), (Q(1, 4), Q(0)), budget=b)
        self.assertEqual(caught.exception.status, 'INCONCLUSIVE_BUDGET')
        self.assertEqual(b.snapshot()['field_calls'], 1)
        self.assertEqual(len(b.snapshot()['field_records']), 1)
        self.assertEqual(b.snapshot()['field_records'][0]['status'], 'INCONCLUSIVE_BUDGET')


if __name__ == '__main__':
    unittest.main()
