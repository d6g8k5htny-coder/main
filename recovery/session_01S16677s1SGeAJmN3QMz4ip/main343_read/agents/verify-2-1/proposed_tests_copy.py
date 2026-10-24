"""Proposed amendment tests for test_population_certificate.py.
Each is verified to PASS on the pristine copy and to FAIL on the survivor(s) named in its docstring.
Run from any copy root: python3.12 -B -E -S proposed_tests.py <copy-root> [-v]"""
import importlib.util, sys, tempfile, time, types, unittest
from unittest import mock
from fractions import Fraction as Q
from pathlib import Path
TARGET = Path(sys.argv.pop(1)).resolve()
POP_REL = 'experiments/universality/two_parameter_cusp_population'
sys.path.insert(0, str(TARGET / POP_REL))
import population_certificate as pc
import run_population_certificate as runner


def stub_pc(b_shift=Q(0), cdf_shift=Q(0), override=None, void=None):
    """Module stand-in: direct route = n*cdf_mass(tau) widened by 2^-20 per side (CERTIFIED),
    optional literal overrides for a row (both routes), B shift, CDF shift or a fixed void interval."""
    override = override or {}
    p = types.SimpleNamespace(**{k: getattr(pc, k) for k in dir(pc) if not k.startswith('__')})
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
    p.cdf_mass, p.direct_mass = cdf_mass, direct_mass
    p.coefficient_interval = lambda: (base_b[0] + b_shift, base_b[1] + b_shift)
    p.poisson_target = lambda b=None: pc.poisson_target(p.coefficient_interval() if b is None else b)
    if void is not None:
        p.void_interval = lambda mass, n: void
    return p


def campaign(**kw):
    return runner._campaign(stub_pc(**kw), 80000, time.monotonic() + 900)


class EnclosureTests(unittest.TestCase):
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


class PredicateFixtureTests(unittest.TestCase):
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


class NegativeControlTests(unittest.TestCase):
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
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for source in runner.SOURCE_PATHS:
                target = base / source.relative_to(runner.REPOSITORY)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            copy_path = base / POP_REL / 'run_population_certificate.py'
            spec = importlib.util.spec_from_file_location('copied_runner_drift', copy_path)
            copied = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copied)
            proof = base / POP_REL / 'PROOF.md'
            def campaign_that_edits_a_source(pc_module, max_evaluations, deadline):
                proof.write_bytes(proof.read_bytes() + b'\n')
                return {'schema_version': 1, 'status': 'PASS', 'rows': [], 'scope_statuses':
                        {'exact_non_gaussian_population': 'PASS'}}
            copied._campaign = campaign_that_edits_a_source
            rec = copied.run(output_dir=base / 'attempt', max_evaluations=80000, time_budget=900)
            self.assertEqual(rec['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
            self.assertEqual(rec['exit_code'], 2)
            self.assertEqual(rec['source_drift'], [POP_REL + '/PROOF.md'])
            self.assertFalse(rec['full_declared_campaign'])


if __name__ == '__main__':
    unittest.main()
