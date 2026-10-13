"""TEST_ONLY Task 3 handoff; synthetic records, never a scientific campaign.

Later adoption: copy to test_global_h0_runner.py after Task 2 review. Run with
Python 3.12 -B -S, normally and under -O, retaining genuine RED/GREEN evidence.
Do not execute this staged file as a scientific result. All writes are temporary.
"""
from __future__ import annotations

import builtins
import copy
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TEST_DIRECTORY = Path(__file__).resolve().parent
POP = 'experiments/universality/two_parameter_cusp_population/'
DEFAULT_RUNNER = TEST_DIRECTORY / 'run_global_h0_selector.py'
if Path(__file__).name == 'global-selector-task-3-tests.py':
    # Outside staged candidate; a tracked test defaults to its sibling driver.
    DEFAULT_RUNNER = TEST_DIRECTORY / 'two-parameter-cusp-worktree' / POP / 'run_global_h0_selector.py'
RUNNER_PATH = Path(os.environ.get('GLOBAL_SELECTOR_RUNNER_PATH', str(DEFAULT_RUNNER)))
OLD_PINS = {
    POP + 'PROOF.md': (38350, '9e8caa52b6448e23e465e646ca98d6e42e0cd71442ad4118b59145ee849e1326'),
    POP + 'CONTROLLED_FALSIFICATION.md': (15296, 'a23f88801e678dd1af484cac8f5564801d9d410d835b560a280c0ea6dfc8996e'),
    POP + 'ANALYTIC_GUARDS.md': (10494, '2f87cafba115a314e44d04d784701aebcebdc123feced7f22540063fa7570db9'),
    'experiments/periodic_h0/finite_certificate.py': (14843, '0674a2621dff6c040cbb24cc0b6e6df7847541ebc171ebdb28070afa05e04e6b'),
    'experiments/periodic_h0/gaussian_tail.py': (4688, '07e55a62ed0bb6da6a9395762e2e0c033126f83b46abd48db4c24df710779db8'),
}
NEW_PATHS = {
    POP + 'GLOBAL_SELECTOR_CONTRACT.md',
    'docs/plans/2026-10-09-global-h0-selector.md',
    *(POP + name for name in ('surgery_field.py', 'test_surgery_field.py',
                            'global_h0_selector.py', 'test_global_h0_selector.py',
                            'run_global_h0_selector.py', 'test_global_h0_runner.py')),
}
SOUNDNESS_GATES = {
    'accepted_guard_binding', 'critical_completeness', 'field_width',
    'field_source_binding', 'core_identity_overlap', 'strict_height_order',
    'positive_signs', 'component_incidence',
}
INCONCLUSIVES = ('INCONCLUSIVE_SOURCE_DRIFT', 'INCONCLUSIVE_BUDGET')


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def synthetic_records(*, tied=False):
    """TEST_ONLY: deliberately not a scientifically valid selector certificate."""
    bar = {'birth_root': 2, 'death_group': 'TEST_ONLY_death',
           'birth_centered_interval': (Q(2), Q(2)),
           'death_centered_interval': (Q(1), Q(1)),
           'lifetime_interval': (Q(1), Q(1)), 'lifetime_exact': Q(1),
           'tied_birth': tied}
    selection = {
        'schema_version': 1, 'model_id': 'explicit_periodic_cusp_v1',
        'u': Q(1, 100), 'v': Q(0), 'status': 'CERTIFIED',
        'guard': {'status': 'PASS', 'test_only': True},
        'actual_barcode': {'status': 'CERTIFIED', 'essential': [{'root': 0}],
                           'finite': [bar],
                           'tie_policy': 'smallest_axial_root_ordinal_survives'},
        'population_projection': {'status': 'EXCLUDED_TIE' if tied else 'CERTIFIED',
                                  'count': 0 if tied else 1,
                                  'actual_finite_count': 1,
                                  'reason': 'exact tie' if tied else 'positive bar'},
        'checks': {'synthetic_check': {'status': 'PASS'}},
        'work': {'polynomial_evaluations': 7}, 'test_only': True,
    }
    verification = {'status': 'PASS', 'checks': {'synthetic_replay': {'status': 'PASS'}},
                    'work': {'polynomial_evaluations': 11}, 'test_only': True}
    soundness = {name: 'PASS' for name in SOUNDNESS_GATES}
    return selection, verification, soundness


def nested_statuses(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'status':
                yield item
            elif key not in ('computed_status_before_withdrawal', 'accepted_premise'):
                yield from nested_statuses(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from nested_statuses(item)


class RunnerHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import only the driver's stdlib helper layer. Scientific imports must
        # be deferred until source freezing and accepted-premise validation.
        original_import = builtins.__import__
        blocked = {'surgery_field', 'global_h0_selector', 'finite_certificate',
                   'gaussian_tail', 'population_certificate', 'critical_fixture'}

        def guard(name, *args, **kwargs):
            if any(part in blocked for part in name.split('.')):
                raise AssertionError('Scientific import before bootstrap: ' + name)
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location('TEST_ONLY_global_runner', RUNNER_PATH)
        cls.driver = importlib.util.module_from_spec(spec)
        with mock.patch('builtins.__import__', side_effect=guard):
            spec.loader.exec_module(cls.driver)

    def test_fraction_encoding_is_existing_str_Q_convention(self):
        raw = self.driver.canonical_bytes({'zero': Q(0), 'one': Q(1),
                                           'half': Q(2, 4), 'negative': Q(-6, 3)})
        parsed = self.driver.load_exact_json(raw)
        self.assertEqual(parsed, {'zero': '0', 'one': '1', 'half': '1/2', 'negative': '-2'})
        self.assertEqual(raw, self.driver.canonical_bytes(parsed))
        self.assertTrue(raw.endswith(b'\n'))
        self.assertEqual(raw.count(b'\n'), 1)

    def test_encoding_is_sorted_compact_utf8_and_preserves_metadata_strings(self):
        value = {'z': ('TEST_ONLY', Q(1)), 'a': {'duration_seconds': '0.125000000',
                 'utc': '2026-10-09T12:00:00.000000+00:00', 'count': 18,
                 'tied': True, 'unknown': None}}
        raw = self.driver.canonical_bytes(value)
        expected = json.dumps({'z': ['TEST_ONLY', '1'], 'a': value['a']},
                              sort_keys=True, separators=(',', ':'),
                              allow_nan=False, ensure_ascii=False).encode('utf-8') + b'\n'
        self.assertEqual(raw, expected)
        self.assertEqual(type(self.driver.load_exact_json(raw)['a']['duration_seconds']), str)

    def test_encoder_rejects_nested_floats_nonfinite_and_nonstring_keys(self):
        for value in (0.0, 1.5, float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                self.driver.canonical_bytes({'nested': [{'bad': value}]})
        with self.assertRaises(ValueError):
            self.driver.canonical_bytes({1: 'TEST_ONLY'})
        with self.assertRaises(ValueError):
            self.driver.canonical_bytes({'bad': object()})

    def test_loader_rejects_duplicate_keys_at_every_depth(self):
        for raw in (b'{"status":"PASS","status":"FAIL"}',
                    b'{"outer":[{"n":1,"n":2}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.driver.load_exact_json(raw)

    def test_loader_rejects_floats_exponents_and_nonfinite_literals(self):
        for literal in (b'0.0', b'1.25', b'1e0', b'NaN', b'Infinity', b'-Infinity'):
            with self.subTest(literal=literal), self.assertRaises(ValueError):
                self.driver.load_exact_json(b'{"nested":[' + literal + b']}')
        self.assertEqual(self.driver.load_exact_json(b'{"count":18,"q":"1/2"}'),
                         {'count': 18, 'q': '1/2'})

    def test_exactly_eighteen_ordered_fraction_controls(self):
        fixtures = self.driver.frozen_fixtures()
        self.assertEqual(len(fixtures), 18)
        self.assertEqual([f['fixture_id'] for f in fixtures],
                         [f'fixture_{i:02d}' for i in range(1, 19)])
        a, u0, v0 = Q(1, 2**18), Q(3, 2**32), Q(1, 2**47)
        expected = []
        for theta in (Q(1, 4), Q(1, 2), Q(3, 4), Q(0), Q(1), Q(3), Q(2)):
            for sign in (1, -1):
                expected.append(((3 + theta**2) * a**2,
                                 sign * 2 * (1 - theta**2) * a**3))
        expected += [(-u0 / 2, v0 / 2), (Q(0), v0 / 2),
                     (u0 / 2, v0 / 2), (u0 / 2, Q(0))]
        self.assertEqual([(f['u'], f['v']) for f in fixtures], expected)
        for f in fixtures:
            self.assertEqual(set(f), {'fixture_id', 'u', 'v'})
            self.assertIs(type(f['u']), Q)
            self.assertIs(type(f['v']), Q)
            self.assertTrue(-u0 < f['u'] < u0)
            self.assertTrue(-v0 < f['v'] < v0)
        # Two theta=1 sign labels share the same control; preserve both rows.
        self.assertEqual((fixtures[8]['u'], fixtures[8]['v']),
                         (fixtures[9]['u'], fixtures[9]['v']))
        self.assertNotEqual(fixtures[8]['fixture_id'], fixtures[9]['fixture_id'])

    def test_literal_expectations_match_all_eighteen_contract_rows(self):
        lifetimes = [Q(1, 2**76), Q(1, 2**76), Q(1, 2**73), Q(1, 2**73),
                     Q(27, 2**76), Q(27, 2**76), None, None,
                     Q(1, 2**70), Q(1, 2**70), None, None,
                     Q(3, 2**74), Q(3, 2**74), None, None, None, Q(9, 2**68)]
        counts = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        # Consumer ordinals are zero-based over sorted distinct axial roots.
        younger = [0, 2, 0, 2, 0, 2, None, None, None, None,
                   None, None, 2, 0, None, None, None, None]
        for index, (lifetime, count, root) in enumerate(zip(lifetimes, counts, younger), 1):
            with self.subTest(index=index):
                expected = self.driver.literal_expectation(f'fixture_{index:02d}')
                self.assertEqual(expected, {'lifetimes': () if lifetime is None else (lifetime,),
                                            'population_count': count, 'younger_root_ordinal': root})
        self.assertNotEqual(self.driver.literal_expectation('fixture_13')['lifetimes'], (Q(1, 2**67),))
        with self.assertRaises(ValueError):
            self.driver.literal_expectation('fixture_19')

    def test_no_expected_lookup_on_any_unsound_gate(self):
        selection, verification, gates = synthetic_records()
        for gate in SOUNDNESS_GATES:
            broken = dict(gates, **{gate: 'INCONCLUSIVE_PRECISION'})
            lookup = mock.Mock(side_effect=AssertionError('Expected oracle reached before soundness'))
            with self.subTest(gate=gate):
                result = self.driver.compare_expectations('TEST_ONLY', selection, verification,
                                                         broken, expected_lookup=lookup)
                self.assertEqual(result['status'], 'NOT_COMPARED')
                lookup.assert_not_called()

    def test_missing_gate_incomplete_selection_or_failed_verifier_never_reads_expectation(self):
        selection, verification, gates = synthetic_records()
        missing = dict(gates)
        missing.pop('component_incidence')
        cases = [(selection, verification, missing)]
        for where in ('selection', 'barcode', 'verification'):
            s, v = copy.deepcopy(selection), copy.deepcopy(verification)
            if where == 'selection':
                s['status'] = 'INCONCLUSIVE_PRECISION'
            elif where == 'barcode':
                s['actual_barcode']['status'] = 'INCONCLUSIVE_PRECISION'
            else:
                v['status'] = 'FAIL_IMPLEMENTATION'
            cases.append((s, v, gates))
        for s, v, g in cases:
            lookup = mock.Mock(side_effect=AssertionError('Forbidden expected lookup'))
            result = self.driver.compare_expectations('TEST_ONLY', s, v, g, expected_lookup=lookup)
            self.assertEqual(result['status'], 'NOT_COMPARED')
            lookup.assert_not_called()

    def test_sound_comparison_reads_literal_expectation_once_without_mutating_numeric_evidence(self):
        selection, verification, gates = synthetic_records()
        before = copy.deepcopy(selection)
        lookup = mock.Mock(return_value={'lifetimes': (Q(1),), 'population_count': 1,
                                         'younger_root_ordinal': 2})
        result = self.driver.compare_expectations('TEST_ONLY', selection, verification,
                                                 gates, expected_lookup=lookup)
        self.assertEqual(result['status'], 'PASS')
        lookup.assert_called_once_with('TEST_ONLY')
        self.assertEqual(selection, before)

    def test_sound_numeric_mismatch_is_retained_comparison_falsifier(self):
        selection, verification, gates = synthetic_records()
        before = copy.deepcopy(selection)
        lookup = mock.Mock(return_value={'lifetimes': (Q(2),), 'population_count': 1,
                                         'younger_root_ordinal': 2})
        result = self.driver.compare_expectations('TEST_ONLY', selection, verification,
                                                 gates, expected_lookup=lookup)
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(selection, before)
        self.assertEqual(selection['actual_barcode']['finite'][0]['lifetime_exact'], Q(1))
        self.assertTrue(result['mismatches'])

    def test_actual_tie_bar_compares_separately_from_excluded_population_count(self):
        selection, verification, gates = synthetic_records(tied=True)
        lookup = mock.Mock(return_value={'lifetimes': (Q(1),), 'population_count': 0,
                                         'younger_root_ordinal': None})
        result = self.driver.compare_expectations('TEST_ONLY', selection, verification,
                                                 gates, expected_lookup=lookup)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(selection['actual_barcode']['finite']), 1)
        self.assertEqual(selection['population_projection']['status'], 'EXCLUDED_TIE')

    def test_source_paths_and_consumed_pins_have_contract_scope(self):
        paths = {str(p) for p in self.driver.SOURCE_PATHS}
        self.assertTrue(set(OLD_PINS) | NEW_PATHS <= paths)
        for path, (size, sha) in OLD_PINS.items():
            self.assertEqual(self.driver.CONSUMED_SOURCE_PINS[path], {'bytes': size, 'sha256': sha})
        # The driver cannot carry hardcoded hashes of its own new test/module files.
        self.assertFalse(set(self.driver.CONSUMED_SOURCE_PINS) & (NEW_PATHS - {POP + 'GLOBAL_SELECTOR_CONTRACT.md'}))

    def test_manifest_captures_bytes_and_new_own_source_hash_without_selfhash_cycle(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            old, own = base / 'old.md', base / 'run_global_h0_selector.py'
            old.write_bytes(b'TEST_ONLY old')
            own.write_bytes(b'TEST_ONLY own source')
            paths = ('old.md', 'run_global_h0_selector.py')
            frozen = self.driver.freeze_sources(paths, base=base, expected={'old.md': identity(old.read_bytes())})
            self.assertEqual(frozen, {p: identity((base / p).read_bytes()) for p in paths})
            self.assertEqual(self.driver.source_drift(frozen, base=base), [])
            own.write_bytes(b'TEST_ONLY changed own source')
            self.assertEqual(self.driver.source_drift(frozen, base=base), ['run_global_h0_selector.py'])

    def test_manifest_rejects_old_consumed_mismatch_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            (base / 'old.md').write_bytes(b'TEST_ONLY changed')
            expected = {'old.md': identity(b'TEST_ONLY frozen')}
            with self.assertRaises(ValueError):
                self.driver.freeze_sources(('old.md',), base=base, expected=expected)
            with self.assertRaises(ValueError):
                self.driver.freeze_sources(('old.md', 'old.md'), base=base, expected={})

    def test_source_drift_detects_changed_same_size_bytes_and_missing_sources(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            for name, raw in (('a.py', b'AAAA'), ('b.py', b'BBBB')):
                (base / name).write_bytes(raw)
            frozen = self.driver.freeze_sources(('a.py', 'b.py'), base=base, expected={})
            (base / 'a.py').write_bytes(b'CCCC')
            (base / 'b.py').unlink()
            self.assertEqual(self.driver.source_drift(frozen, base=base), ['a.py', 'b.py'])

    def test_bootstrap_freezes_every_source_before_injected_scientific_import(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            paths = ('old.md', 'new.py', 'test_new.py', 'driver.py', 'contract.md', 'plan.md')
            for path in paths:
                (base / path).write_bytes(('TEST_ONLY ' + path).encode())
            expected = {'old.md': identity((base / 'old.md').read_bytes())}
            events = []
            real_freeze = self.driver.freeze_sources

            def freeze(*args, **kwargs):
                result = real_freeze(*args, **kwargs)
                events.append(('freeze', set(result)))
                return result

            def importer():
                self.assertEqual(events, [('freeze', set(paths))])
                events.append(('import', None))
                return 'TEST_ONLY_scientific_handles'

            with mock.patch.object(self.driver, 'SOURCE_PATHS', paths), \
                 mock.patch.object(self.driver, 'CONSUMED_SOURCE_PINS', expected), \
                 mock.patch.object(self.driver, 'freeze_sources', side_effect=freeze):
                manifest, handles = self.driver.prepare_sources(base=base, importer=importer)
            self.assertEqual(set(manifest), set(paths))
            self.assertEqual(handles, 'TEST_ONLY_scientific_handles')

    def test_bootstrap_consumed_source_mismatch_never_imports(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            (base / 'old.md').write_bytes(b'TEST_ONLY mismatched')
            importer = mock.Mock(side_effect=AssertionError('Import reached after source rejection'))
            with mock.patch.object(self.driver, 'SOURCE_PATHS', ('old.md',)), \
                 mock.patch.object(self.driver, 'CONSUMED_SOURCE_PINS', {'old.md': identity(b'TEST_ONLY frozen')}):
                with self.assertRaises(ValueError):
                    self.driver.prepare_sources(base=base, importer=importer)
            importer.assert_not_called()

    def test_accepted_premise_is_portable_exact_metadata_not_host_path_dependency(self):
        source = {'bytes': OLD_PINS[POP + 'ANALYTIC_GUARDS.md'][0],
                  'sha256': OLD_PINS[POP + 'ANALYTIC_GUARDS.md'][1]}
        premise = {'status': 'ACCEPTED', 'record_role': 'historical_accepted_analytic_premise',
                   'analytic_source': source,
                   'review': {'bytes': 22303,
                              'sha256': 'ef831e2778e805f277f0e1ec1185554e812f75adde815d05c60016d6e4761b08',
                              'status': 'PASS_BOUNDED_ANALYTIC_GUARDS',
                              'nonauthor': True, 'source_exposure': 'source-exposed'},
                   'accepted_by': 'OpenAI/Codex root',
                   'scope': 'bounded analytic guards for explicit_periodic_cusp_v1',
                   'organization_independence_credit': 0, 'blind_independence_credit': 0}
        manifest = {POP + 'ANALYTIC_GUARDS.md': source}
        with mock.patch.object(Path, 'read_bytes', side_effect=AssertionError('Review host path consulted')):
            accepted = self.driver.bind_accepted_premise(premise, manifest)
        self.assertEqual(accepted, premise)
        changed = copy.deepcopy(premise)
        changed['analytic_source']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.driver.bind_accepted_premise(changed, manifest)
        with self.assertRaises(ValueError):
            self.driver.bind_accepted_premise(None, manifest)
        for field, bad in (('sha256', '0' * 64), ('bytes', 22304),
                           ('status', 'TEST_ONLY_unaccepted'), ('nonauthor', False)):
            changed = copy.deepcopy(premise)
            changed['review'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.driver.bind_accepted_premise(changed, manifest)

    def test_exclusive_artifact_creation_and_exact_run_bindings(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            output = Path(tmp) / 'TEST_ONLY_attempt'
            certificate = {'status': 'TEST_ONLY', 'bar': Q(1, 2), 'duration_seconds': '0.000000001'}
            report = '# TEST_ONLY synthetic driver report\n'
            receipt = self.driver.write_exclusive_artifacts(output, certificate=certificate,
                results_md=report, run_metadata={'status': 'TEST_ONLY', 'utc': '2026-10-09T00:00:00.000000+00:00'})
            self.assertEqual({p.name for p in output.iterdir()}, {'RUN.json', 'CERTIFICATE.json', 'RESULTS.md'})
            self.assertEqual((output / 'CERTIFICATE.json').read_bytes(), self.driver.canonical_bytes(certificate))
            self.assertEqual((output / 'RESULTS.md').read_bytes(), report.encode())
            run = self.driver.load_exact_json((output / 'RUN.json').read_bytes())
            self.assertEqual(run, receipt)
            self.assertEqual(set(run['outputs']), {'CERTIFICATE.json', 'RESULTS.md'})
            for name, metadata in run['outputs'].items():
                self.assertEqual(metadata, identity((output / name).read_bytes()))
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                self.driver.write_exclusive_artifacts(output, certificate={'status': 'TEST_ONLY_second'},
                                                     results_md='replacement', run_metadata={})
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_existing_empty_directory_or_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            base = Path(tmp)
            for name, is_dir in (('empty', True), ('file', False)):
                output = base / name
                if is_dir:
                    output.mkdir()
                else:
                    output.write_bytes(b'TEST_ONLY preserve me')
                with self.assertRaises(FileExistsError):
                    self.driver.write_exclusive_artifacts(output, certificate={}, results_md='', run_metadata={})
                if is_dir:
                    self.assertEqual(list(output.iterdir()), [])
                else:
                    self.assertEqual(output.read_bytes(), b'TEST_ONLY preserve me')

    def test_invalid_exact_record_cannot_create_partial_result_directory(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_global_runner_') as tmp:
            output = Path(tmp) / 'TEST_ONLY_invalid'
            with self.assertRaises(ValueError):
                self.driver.write_exclusive_artifacts(output, certificate={'bad': 0.5},
                                                     results_md='TEST_ONLY', run_metadata={})
            self.assertFalse(output.exists())

    def test_global_withdrawal_replaces_every_nested_acceptance_retaining_original_statuses(self):
        for status in INCONCLUSIVES:
            selection, verification, _ = synthetic_records(tied=True)
            original = {'status': 'CERTIFIED', 'fixtures': [
                {'status': 'CERTIFIED', 'selection': selection, 'verification': verification}],
                'field_execution': {'status': 'CERTIFIED', 'records': [{'interval': (Q(1), Q(2))}]},
                'selector_execution': {'status': 'CERTIFIED'},
                'accepted_premise': {'status': 'ACCEPTED',
                    'record_role': 'historical_accepted_analytic_premise', 'review': {
                    'status': 'PASS_BOUNDED_ANALYTIC_GUARDS', 'sha256': 'TEST_ONLY review identity'}},
                'test_only': True}
            before = copy.deepcopy(original)
            withdrawn = self.driver.withdraw_acceptance(original, status, phase='TEST_ONLY_final',
                                                        detail='TEST_ONLY drift/deadline')
            self.assertEqual(original, before)
            self.assertEqual(withdrawn['accepted_premise'], original['accepted_premise'])
            self.assertEqual(withdrawn['status'], status)
            self.assertEqual(withdrawn['computed_status_before_withdrawal'], 'CERTIFIED')
            self.assertTrue(all(s == status for s in nested_statuses(withdrawn)))
            row = withdrawn['fixtures'][0]
            self.assertEqual(row['selection']['computed_status_before_withdrawal'], 'CERTIFIED')
            self.assertEqual(row['verification']['computed_status_before_withdrawal'], 'PASS')
            barcode = row['selection']['actual_barcode']
            projection = row['selection']['population_projection']
            self.assertEqual(barcode['computed_status_before_withdrawal'], 'CERTIFIED')
            self.assertEqual(projection['computed_status_before_withdrawal'], 'EXCLUDED_TIE')
            self.assertEqual(barcode['finite'], selection['actual_barcode']['finite'])
            self.assertEqual(barcode['essential'], selection['actual_barcode']['essential'])
            self.assertEqual(projection['reason'], 'exact tie')
            self.assertEqual(projection['count'], 0)
            self.assertEqual(row['selection']['work'], selection['work'])
            self.assertEqual(withdrawn['field_execution']['records'], original['field_execution']['records'])

    def test_task1_raw_boolean_predicates_preserve_truth_while_current_acceptance_withdraws(self):
        # TEST_ONLY records use Task 1's actual checks/gates/field_records keys.
        # They do not assert any real field computation or accepted guard proof.
        guard = {'status': 'ACCEPTED_ANALYTIC_PREMISE',
                 'model_id': 'explicit_periodic_cusp_v1',
                 'checks': {'parameters': True, 'gradient_exact': True},
                 'finite_sampling_proves_smoothness': False}
        field_records = []
        for status, gates, interval in (
            ('CERTIFIED', {'identity': True, 'ordered': True, 'width': True}, (Q(1), Q(1))),
            ('INCONCLUSIVE_PRECISION', {'identity': True, 'ordered': True, 'width': False}, (Q(0), Q(1))),
            ('INCONCLUSIVE_BUDGET', {'identity': True, 'ordered': None, 'width': None}, None),
        ):
            field_records.append({'route': 'chart', 'u': Q(0), 'v': Q(0),
                'fixture_id': 'TEST_ONLY_boolean_gates',
                'input_box': {'s': (Q(0), Q(0)), 'z': (Q(0), Q(0))},
                'branches': [], 'returned_interval': interval, 'status': status,
                'model': {'test_only': True}, 'gates': gates})
        original = {'status': 'CERTIFIED', 'guard': guard,
                    'work': {'field_records': field_records},
                    'accepted_premise': {
                        'record_role': 'historical_accepted_analytic_premise',
                        'status': 'ACCEPTED', 'guard': copy.deepcopy(guard),
                        'review': {'status': 'PASS_BOUNDED_ANALYTIC_GUARDS',
                                   'sha256': 'TEST_ONLY immutable review identity'}},
                    'test_only': True}
        before = copy.deepcopy(original)
        normalized = self.driver.normalize_computed_predicates(original)
        self.assertEqual(original, before)
        self.assertEqual(normalized['accepted_premise'], original['accepted_premise'])
        self.assertFalse(normalized['guard']['finite_sampling_proves_smoothness'])
        for name, value in guard['checks'].items():
            self.assertEqual(normalized['guard']['checks'][name],
                             {'status': 'PASS', 'computed_value': value})
        for raw, current in zip(field_records, normalized['work']['field_records']):
            self.assertEqual(current['returned_interval'], raw['returned_interval'])
            for name, value in raw['gates'].items():
                expected_status = 'PASS' if value is True else 'FAIL' if value is False else 'NOT_EVALUATED'
                self.assertEqual(current['gates'][name],
                                 {'status': expected_status, 'computed_value': value})
        self.assertEqual(self.driver.normalize_computed_predicates(normalized), normalized)
        for disposition in INCONCLUSIVES:
            withdrawn = self.driver.withdraw_acceptance(normalized, disposition,
                phase='TEST_ONLY_final_boolean_withdrawal', detail='TEST_ONLY drift/deadline')
            self.assertTrue(all(status == disposition for status in nested_statuses(withdrawn)))
            self.assertEqual(withdrawn['accepted_premise'], original['accepted_premise'])
            self.assertFalse(withdrawn['guard']['finite_sampling_proves_smoothness'])
            for name, value in guard['checks'].items():
                check = withdrawn['guard']['checks'][name]
                self.assertIs(check['computed_value'], value)
                self.assertEqual(check['status'], disposition)
                self.assertEqual(check['computed_status_before_withdrawal'], 'PASS')
            for raw, current in zip(field_records, withdrawn['work']['field_records']):
                self.assertEqual(current['computed_status_before_withdrawal'], raw['status'])
                self.assertEqual(current['returned_interval'], raw['returned_interval'])
                for name, value in raw['gates'].items():
                    gate = current['gates'][name]
                    self.assertIs(gate['computed_value'], value)
                    expected_status = 'PASS' if value is True else 'FAIL' if value is False else 'NOT_EVALUATED'
                    self.assertEqual(gate['computed_status_before_withdrawal'], expected_status)
                    self.assertEqual(gate['status'], disposition)

    def test_repeated_withdrawal_does_not_erase_first_computed_status(self):
        selection, verification, _ = synthetic_records()
        record = {'status': 'CERTIFIED', 'selection': selection, 'verification': verification}
        first = self.driver.withdraw_acceptance(record, INCONCLUSIVES[0], phase='TEST_ONLY_hash', detail='drift')
        second = self.driver.withdraw_acceptance(first, INCONCLUSIVES[1], phase='TEST_ONLY_final', detail='deadline')
        self.assertEqual(second['selection']['computed_status_before_withdrawal'], 'CERTIFIED')
        self.assertEqual(second['verification']['computed_status_before_withdrawal'], 'PASS')
        self.assertNotIn('CERTIFIED', tuple(nested_statuses(second)))
        self.assertNotIn('PASS', tuple(nested_statuses(second)))

    def test_final_gate_applies_post_computation_drift_and_exact_deadline(self):
        selection, verification, _ = synthetic_records()
        original = {'status': 'CERTIFIED', 'selection': selection, 'verification': verification}
        for now in (Q(900), Q(901)):
            # The live clock API is separately a float; these are converted
            # outside the certificate solely to exercise the internal gate.
            withdrawn = self.driver.finalize_acceptance(original, source_drift=[], start=0.0,
                                                       now=float(now), seconds=900)
            self.assertEqual(withdrawn['status'], 'INCONCLUSIVE_BUDGET')
            self.assertNotIn('PASS', tuple(nested_statuses(withdrawn)))
            self.assertEqual(withdrawn['selection']['actual_barcode']['finite'],
                             selection['actual_barcode']['finite'])
        drifted = self.driver.finalize_acceptance(original, source_drift=['TEST_ONLY.py'],
                                                 start=0.0, now=1.0, seconds=900)
        self.assertEqual(drifted['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        accepted = self.driver.finalize_acceptance(original, source_drift=[], start=0.0,
                                                  now=899.0, seconds=900)
        self.assertEqual(accepted, original)
        self.assertEqual(original['status'], 'CERTIFIED')



class MainLifecycle(unittest.TestCase):
    """TEST_ONLY orchestration doubles; no valid scientific selection is claimed."""
    @classmethod
    def setUpClass(cls):
        RunnerHandoff.setUpClass()
        cls.driver = RunnerHandoff.driver

    def run_synthetic(self, *, fault=None, premise='DEFAULT'):
        driver = self.driver
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_runner_main_') as tmp:
            base = Path(tmp)
            for name in driver.SOURCE_PATHS:
                path = base/name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = ('TEST_ONLY source '+name).encode()
                if name == POP+'ANALYTIC_GUARDS.md':
                    raw = (RUNNER_PATH.parent/'ANALYTIC_GUARDS.md').read_bytes()
                path.write_bytes(raw)
            pins = {POP+'ANALYTIC_GUARDS.md': driver.CONSUMED_SOURCE_PINS[POP+'ANALYTIC_GUARDS.md']}
            manifest = {name: identity((base/name).read_bytes()) for name in driver.SOURCE_PATHS}
            events, clocks, budgets = [], [1.0], []
            if fault == 'pinned_source':
                (base/(POP+'ANALYTIC_GUARDS.md')).write_bytes(b'TEST_ONLY changed pinned analytic source')
            selection, verification, _ = synthetic_records()
            guard = {'status': 'ACCEPTED_ANALYTIC_PREMISE', 'model_id': 'explicit_periodic_cusp_v1',
                'analytic_source': manifest[POP+'ANALYTIC_GUARDS.md'], 'accepted_review': dict(driver.REVIEW_PIN),
                'checks': {'parameters': True, 'core_critical_height': True},
                'finite_sampling_proves_smoothness': False}
            selection.update(u=Q(0), v=Q(0), guard=guard,
                critical_roots={'status': 'CERTIFIED', 'roots': [
                    {'ordinal': 0, 'isolator': (Q(0), Q(0)), 'exact_value': Q(0)}]},
                critical_heights=[{'root_ordinal': 0, 'centered_interval': (Q(0), Q(0))}],
                height_groups=[{'id': 'TEST_ONLY_group', 'centered_interval': (Q(0), Q(0))}],
                levels=[{'id': 'TEST_ONLY_level', 'components': [], 'sign_intervals': []}],
                old_criticals=[], incidence=[], events=[])
            verification['checks'] = {name: True for name in (
                'guard/source_bound_rational_replay', 'guard/old_criticals', 'verify/critical/complete0',
                'heights/values_and_inertia', 'groups/partition', 'groups/barrier_band',
                'levels/regular_sign0/0', 'levels/components0', 'graph/incidence_witnesses', 'graph/barcode')}
            model = {'model_id': 'explicit_periodic_cusp_v1', 'analytic_source': guard['analytic_source'],
                'accepted_review': guard['accepted_review'],
                'implementation_source': {'filename': 'surgery_field.py', **manifest[POP+'surgery_field.py']},
                'dependencies': {name: manifest['experiments/periodic_h0/'+name]
                                 for name in ('finite_certificate.py', 'gaussian_tail.py')}}

            class TestBudget:
                def __init__(self, start, **kwargs):
                    self.start, self.records, self.fixture = start, [], None
                    self.polynomials = 0
                    budgets.append(self)
                def checkpoint(self, phase='work'):
                    events.append(('checkpoint', phase))
                def begin_fixture(self, fixture):
                    self.fixture = fixture
                def charge_polynomial(self, count=1, **kwargs):
                    if fault == 'core_budget' and kwargs.get('phase') == 'driver/core_anchor':
                        failure = RuntimeError('TEST_ONLY core comparison admission exhausted')
                        failure.status = 'INCONCLUSIVE_BUDGET'
                        failure.phase = 'driver/core_anchor'
                        raise failure
                    self.polynomials += count
                def snapshot(self):
                    return copy.deepcopy({'field_records': self.records, 'root_refinements': [
                        {'fixture_id': self.fixture, 'phase': 'TEST_ONLY_refinement', 'depth': 1}],
                        'polynomial_evaluations': self.polynomials, 'field_calls': len(self.records),
                        'primitive_requests': 0, 'fixture_id': self.fixture,
                        'scopes': {'field_records': 'whole_attempt', 'root_refinements': 'whole_attempt'}})

            def field_record(budget, route, interval, **inputs):
                record = {'route': route, 'u': Q(0), 'v': Q(0), 'fixture_id': budget.fixture,
                    'status': 'CERTIFIED', 'returned_interval': interval, 'branches': [],
                    'model': copy.deepcopy(model), 'gates': {'identity': True, 'ordered': True, 'width': True}, **inputs}
                if fault == 'field_source':
                    record['model']['implementation_source']['sha256'] = '0'*64
                budget.records.append(record)
                return interval

            def chart(u, v, s, z, *, budget):
                x, y = s[0], z[0]
                # TEST_ONLY exact values for core/collar checks; transition
                # double values are not enclosures of the physical model.
                value = Q(4, 9)-x**4/4-(1+x*x/4)*y*y
                if x*x+y*y >= Q(1, 128):
                    value = Q(4, 9)-x*x-y*y
                return field_record(budget, 'chart', (value, value), input_box={'s': s, 'z': z})

            def torus(u, v, point, *, budget):
                canonical = tuple(x-((x+Q(1, 2)).numerator//(x+Q(1, 2)).denominator) for x in point)
                values = {(Q(0), Q(-1, 2)): Q(2, 9), (Q(-1, 2), Q(0)): Q(-2, 9),
                          (Q(-1, 2), Q(-1, 2)): Q(-4, 9)}
                value = values.get(canonical, Q(0))
                return field_record(budget, 'torus', (value, value), input_point=point, canonical_point=canonical)

            def select(u, v, *, budget):
                events.append(('select', None))
                budget.charge_polynomial()
                return copy.deepcopy(selection)

            def verify(u, v, raw, *, budget):
                events.append(('verify_raw', type(raw['guard']['checks']['parameters'])))
                self.assertIs(raw['guard']['checks']['parameters'], True)
                budget.charge_polynomial()
                result = copy.deepcopy(verification)
                if fault == 'verification':
                    result['status'] = 'FAIL_IMPLEMENTATION'
                return result

            def importer():
                events.append(('import', None))
                self.assertEqual(events[0], ('freeze', set(driver.SOURCE_PATHS)))
                return {'Budget': TestBudget, 'guard_certificate': lambda: copy.deepcopy(guard),
                        'select_h0': select, 'verify_h0': verify,
                        'chart_field_bounds': chart, 'torus_field_bounds': torus}

            def lookup(fixture_id):
                events.append(('expected', fixture_id))
                self.assertIn(('verify_raw', bool), events)
                if fault == 'drift':
                    (base/(POP+'run_global_h0_selector.py')).write_bytes(b'TEST_ONLY post-computation drift')
                if fault == 'deadline':
                    clocks[0] = 901.0
                return {'lifetimes': (Q(1),), 'population_count': 1, 'younger_root_ordinal': 2}

            real_freeze, real_normalize = driver.freeze_sources, driver.normalize_computed_predicates
            def freeze(*args, **kwargs):
                result = real_freeze(*args, **kwargs)
                events.append(('freeze', set(result)))
                return result
            def normalize(value):
                events.append(('normalize', None))
                return real_normalize(value)
            output = base/'TEST_ONLY_artifacts'
            kwargs = {'base': base, 'output_dir': output, 'importer': importer}
            if premise != 'DEFAULT':
                kwargs['premise'] = premise
            with mock.patch.object(driver, 'CONSUMED_SOURCE_PINS', pins), \
                 mock.patch.object(driver, 'freeze_sources', side_effect=freeze), \
                 mock.patch.object(driver, 'normalize_computed_predicates', side_effect=normalize), \
                 mock.patch.object(driver, 'frozen_fixtures', return_value=(
                     {'fixture_id': 'TEST_ONLY', 'u': Q(0), 'v': Q(0)},)), \
                 mock.patch.object(driver, 'literal_expectation', side_effect=lookup), \
                 mock.patch.object(driver.time, 'monotonic', side_effect=lambda: clocks[0]):
                result = driver.run_attempt(**kwargs)
            emitted = None if not output.exists() else {
                p.name: p.read_bytes() for p in output.iterdir()}
            return result, emitted, events, budgets

    def test_main_orders_freeze_raw_verification_lookup_normalization_and_exclusive_writes(self):
        result, emitted, events, budgets = self.run_synthetic()
        self.assertEqual(result['certificate']['status'], 'CERTIFIED')
        self.assertEqual(set(emitted), {'RUN.json', 'CERTIFICATE.json', 'RESULTS.md'})
        self.assertLess(events.index(('verify_raw', bool)), events.index(('expected', 'TEST_ONLY')))
        self.assertLess(events.index(('expected', 'TEST_ONLY')), events.index(('normalize', None)))
        self.assertEqual(len(budgets), 1)
        self.assertEqual(result['certificate']['work']['field_calls'], 33)
        self.assertEqual(result['certificate']['work']['polynomial_evaluations'], budgets[0].polynomials)
        # Two injected producer/replay operations, six real core-identity
        # evaluations and two real seed-collar polynomial evaluations.
        self.assertEqual(result['certificate']['work']['polynomial_evaluations'], 10)
        self.assertEqual(len(result['certificate']['work']['root_refinements']), 1)
        self.assertEqual(emitted['CERTIFICATE.json'], self.driver.canonical_bytes(result['certificate']))
        self.assertEqual(result['certificate']['fixtures'][0]['comparison']['status'], 'PASS')

    def test_main_source_drift_withdraws_live_acceptance_and_keeps_all_numeric_evidence(self):
        result, emitted, _, _ = self.run_synthetic(fault='drift')
        certificate = result['certificate']
        self.assertEqual(certificate['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertNotIn('PASS', tuple(nested_statuses(certificate)))
        self.assertNotIn('CERTIFIED', tuple(nested_statuses(certificate)))
        self.assertEqual(certificate['fixtures'][0]['selection']['actual_barcode']['finite'][0]['lifetime_exact'], Q(1))
        self.assertEqual(certificate['work']['field_calls'], 33)
        self.assertEqual(certificate['accepted_premise']['status'], 'ACCEPTED')
        self.assertIsNotNone(emitted)

    def test_main_exact_deadline_withdraws_before_artifact_creation(self):
        result, emitted, _, _ = self.run_synthetic(fault='deadline')
        self.assertEqual(result['certificate']['status'], 'INCONCLUSIVE_BUDGET')
        self.assertNotIn('PASS', tuple(nested_statuses(result['certificate'])))
        self.assertEqual(self.driver.load_exact_json(emitted['RUN.json'])['status'], 'INCONCLUSIVE_BUDGET')

    def test_main_failed_verifier_and_bad_field_source_never_read_expectations(self):
        for fault in ('verification', 'field_source'):
            with self.subTest(fault=fault):
                result, _, events, _ = self.run_synthetic(fault=fault)
                self.assertNotIn(('expected', 'TEST_ONLY'), events)
                self.assertEqual(result['certificate']['status'], 'FAIL_IMPLEMENTATION')

    def test_main_missing_premise_leaves_scientific_operations_not_run(self):
        result, _, events, budgets = self.run_synthetic(premise=None)
        self.assertEqual(result['certificate']['status'], 'NOT_RUN')
        self.assertFalse(budgets)
        self.assertNotIn(('select', None), events)

    def test_main_derived_gate_acceptance_is_explicit_and_whole_ledger_occurs_once(self):
        result, _, _, _ = self.run_synthetic(fault='drift')
        certificate = result['certificate']
        row = certificate['fixtures'][0]
        for gate in row['soundness'].values():
            self.assertIsInstance(gate, dict)
            self.assertEqual(gate['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
            self.assertEqual(gate['computed_status_before_withdrawal'], 'PASS')
        self.assertIn('checks', row['field_checks'])
        self.assertEqual(row['field_checks']['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        for gate in row['field_checks']['checks'].values():
            self.assertEqual(gate['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
            self.assertIs(gate['computed_value'], True)
        def count_key(value, key):
            if isinstance(value, dict):
                return int(key in value and isinstance(value[key], list))+sum(count_key(v, key) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(count_key(v, key) for v in value)
            return 0
        self.assertEqual(count_key(certificate, 'field_records'), 1)
        self.assertEqual(count_key(certificate, 'root_refinements'), 1)

    def test_main_malformed_premise_blocks_budget_and_selector_before_calls(self):
        premise = copy.deepcopy(self.driver.ACCEPTED_PREMISE)
        premise['review']['bytes'] = Q(22303)
        result, _, events, budgets = self.run_synthetic(premise=premise)
        self.assertEqual(result['certificate']['status'], 'FAIL_IMPLEMENTATION')
        self.assertFalse(budgets)
        self.assertNotIn(('select', None), events)

    def test_main_existing_output_is_refused_before_scientific_import(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_existing_output_') as tmp:
            output = Path(tmp)/'existing'
            output.mkdir()
            def forbidden():
                raise AssertionError('Scientific import reached existing output')
            with self.assertRaises(FileExistsError):
                self.driver.run_attempt(base=Path(tmp), output_dir=output, importer=forbidden)
            self.assertEqual(list(output.iterdir()), [])

    def test_main_pinned_source_failure_blocks_import_and_scientific_operations(self):
        result, emitted, events, budgets = self.run_synthetic(fault='pinned_source')
        self.assertEqual(result['certificate']['status'], 'INCONCLUSIVE_SOURCE_DRIFT')
        self.assertNotIn(('import', None), events)
        self.assertNotIn(('select', None), events)
        self.assertFalse(budgets)
        self.assertIsNotNone(emitted)

    def test_core_comparison_admission_failure_keeps_field_and_bar_without_computing_identity(self):
        result, _, events, _ = self.run_synthetic(fault='core_budget')
        certificate = result['certificate']
        row = certificate['fixtures'][0]
        self.assertEqual(certificate['status'], 'INCONCLUSIVE_BUDGET')
        self.assertEqual(certificate['work']['polynomial_evaluations'], 2)
        self.assertEqual(certificate['work']['field_calls'], 1)
        self.assertEqual(len(row['field_evaluations']), 1)
        self.assertEqual(row['field_evaluations'][0]['returned_interval'], (Q(4, 9), Q(4, 9)))
        self.assertNotIn('exact_identity_value', row['field_evaluations'][0])
        self.assertEqual(row['selection']['actual_barcode']['finite'][0]['lifetime_exact'], Q(1))
        self.assertNotIn(('expected', 'TEST_ONLY'), events)

    def test_io_failure_preserves_partial_file_and_refuses_second_write(self):
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY_partial_io_') as tmp:
            output = Path(tmp)/'new'
            with mock.patch.object(self.driver.os, 'fsync', side_effect=OSError('TEST_ONLY fsync failure')):
                with self.assertRaises(OSError):
                    self.driver.write_exclusive_artifacts(output, certificate={'q': Q(1, 2)},
                        results_md='TEST_ONLY', run_metadata={'status': 'TEST_ONLY'})
            self.assertEqual({p.name for p in output.iterdir()}, {'CERTIFICATE.json'})
            before = (output/'CERTIFICATE.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.driver.write_exclusive_artifacts(output, certificate={}, results_md='', run_metadata={})
            self.assertEqual((output/'CERTIFICATE.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
