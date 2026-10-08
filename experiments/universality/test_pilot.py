"""Retained-row and exact quantized-grid observation controls."""
import copy
import contextlib
import importlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as Q
from unittest.mock import patch


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.universality.run_pilot'),
                             'The exploratory retained-row runner is missing')
        self.pilot = importlib.import_module('experiments.universality.run_pilot')

    def test_all_fifty_models_have_two_retained_checked_fields(self):
        result = self.pilot.build_observations(n=8, cutoff=2)
        self.assertEqual(result['purpose'], 'exploratory_software_pilot')
        self.assertEqual(len(result['rows']), 100)
        self.assertEqual(len({x['model_id'] for x in result['rows']}), 50)
        self.assertEqual(len({x['seed'] for x in result['rows']}), 100)
        self.assertTrue(all(x['failure'] is None and x['connectivity_verified'] for x in result['rows']))
        self.assertTrue(self.pilot.verify_observations(result))

    def test_half_open_bins_use_integer_lifetimes_and_exact_area_mass(self):
        self.assertEqual(self.pilot.bin_counts([[1, 0], [2, 0], [3, 0], [4, 0]],
                                               2, [Q(1, 2), Q(1), Q(2)]), [1, 2])
        summary = self.pilot.summarize_rows([{'counts': [1, 2], 'failure': None},
                                            {'counts': [0, 0], 'failure': None}],
                                           2, side=2, vertex_count=16)
        self.assertEqual(summary['mean_mass_bounds'], [['1/8', '1/8'], ['1/4', '1/4']])

    def test_bool_and_fraction_cannot_impersonate_integer_plan_parameters(self):
        for kwargs in [{'fields_per_model': True}, {'scale': Q(65536)},
                       {'n': True}, {'cutoff': Q(2)}, {'side': True},
                       {'edges': [.1, .2]}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pilot.build_observations(**kwargs)
        for intervals, scale, edges in [([[True, 0]], 1, [Q(1), Q(2)]),
                                        ([[2, 0]], True, [Q(1), Q(2)]),
                                        ([[2, 0]], 1, [1, 3])]:
            with self.assertRaises(ValueError):
                self.pilot.bin_counts(intervals, scale, edges)

    def test_boolean_observation_schema_is_rejected(self):
        result = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        result['schema_version'] = True
        with self.assertRaises(ValueError):
            self.pilot.verify_observations(result)

    def test_failure_is_retained_in_denominator_with_unresolved_sample_count(self):
        def fail_one(model, seed, **kwargs):
            if model['id'] == 'gaussian__gaussian':
                raise ArithmeticError('deliberate sampler failure')
            return self.pilot.models.sample_grid(model, seed, **kwargs)
        result = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1,
                                              sampler=fail_one)
        self.assertEqual(len(result['rows']), 50)
        row = next(x for x in result['rows'] if x['failure'] is not None)
        self.assertEqual(row['failure']['type'], 'ArithmeticError')
        self.assertIsNone(row['counts'])
        summary = result['summaries'][row['model_id']]
        self.assertEqual(summary['planned_fields'], 1)
        self.assertEqual(summary['failed_fields'], 1)
        self.assertEqual(summary['mean_mass_bounds'][0], ['0', '7/64'])
        self.assertTrue(self.pilot.verify_observations(result))
        self.assertEqual(result['environment'].get('sampler_mode'), 'injected_uncertified_sampler')
        self.assertNotIn('MT19937', result['environment']['sampler'])

    def test_verify_rejects_relabelled_bin_or_rounding_convention(self):
        result = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        for field, text in [('bin_convention', 'Final bin closed'),
                            ('quantization', 'Floor each value')]:
            other = copy.deepcopy(result)
            other['config'][field] = text
            with self.assertRaises(ValueError):
                self.pilot.verify_observations(other)

    def test_cli_failure_exit_retains_all_failed_rows_before_returning(self):
        def fail(model, seed, **kwargs):
            raise ArithmeticError('deliberate complete sampler failure')
        failed = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1, sampler=fail)
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'failed-run'
            # Replace only generation; exercise real CLI custody/write/exit behavior.
            with patch.object(self.pilot, '_generate_observations', return_value=failed), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.pilot.main(['--output', str(output)]), 1)
            saved = json.loads((output/'observations.json').read_text())
            self.assertEqual(len(saved['rows']), 50)
            self.assertTrue(all(row['failure'] is not None for row in saved['rows']))
            validation = json.loads((output/'validation.json').read_text())
            self.assertEqual(validation['final_record_verification'], 'PASS')
            self.assertEqual(validation['failed_fields'], 50)
            self.assertEqual(validation['exit_code'], 1)

    def test_omitted_duplicated_or_rebound_row_is_rejected(self):
        result = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        variants = []
        other = copy.deepcopy(result); other['rows'].pop(); variants.append(other)
        other = copy.deepcopy(result); other['rows'][1] = copy.deepcopy(other['rows'][0]); variants.append(other)
        other = copy.deepcopy(result); other['rows'][0]['model_sha256'] = '0'*64; variants.append(other)
        other = copy.deepcopy(result); other['rows'][0]['counts'][0] += 1; variants.append(other)
        for other in variants:
            with self.assertRaises(ValueError):
                self.pilot.verify_observations(other)

    def test_cli_writes_replayable_output_and_refuses_overwrite(self):
        script = Path(self.pilot.__file__)
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'run'
            args = [sys.executable, '-B', str(script), '--output', str(output),
                    '--grid', '8', '--cutoff', '2', '--fields-per-model', '1']
            success = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            observations = output/'observations.json'
            original = observations.read_bytes()
            self.assertTrue(self.pilot.verify_observations(json.loads(original)))
            validation_path = output/'validation.json'
            original_validation = validation_path.read_bytes()
            validation = json.loads(original_validation)
            self.assertEqual(validation['final_record_verification'], 'PASS')
            self.assertEqual(validation['exit_code'], 0)
            self.assertEqual(validation['observations_sha256'], hashlib.sha256(original).hexdigest())
            refusal = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(refusal.returncode, 0)
            self.assertEqual(observations.read_bytes(), original)
            self.assertEqual(validation_path.read_bytes(), original_validation)

    def test_cli_final_verifier_failure_retains_complete_observations_and_error(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'failed-final-verification'
            verified = []

            def reject_final(document):
                verified.append(document)
                self.assertEqual(len(document['rows']), 50)
                self.assertTrue((output/'observations.json').is_file(),
                                'Generated bytes must be saved before the final verifier runs')
                raise RuntimeError('deliberate final verifier failure after all fields')

            escaped = None
            with patch.object(self.pilot, 'verify_observations', side_effect=reject_final), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    result = self.pilot.main(['--output', str(output), '--grid', '8',
                                              '--cutoff', '2', '--fields-per-model', '1'])
                except Exception as error:
                    escaped = error
                    result = None
            observations = output/'observations.json'
            self.assertTrue(observations.is_file(), 'All generated fields must survive final verifier failure')
            self.assertIsNone(escaped)
            self.assertEqual(result, 2)
            self.assertEqual(len(verified), 1)
            original = observations.read_bytes()
            retained = json.loads(original)
            self.assertEqual(len(retained['rows']), 50)
            self.assertEqual(sum(row['failure'] is not None for row in retained['rows']), 0)
            validation = json.loads((output/'validation.json').read_text())
            self.assertEqual(validation['final_record_verification'], 'FAIL')
            self.assertEqual(validation['observations_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(validation['error'], {'type': 'RuntimeError',
                                                   'message': 'deliberate final verifier failure after all fields'})
            with self.assertRaises(FileExistsError):
                self.pilot.main(['--output', str(output)])
            self.assertEqual(observations.read_bytes(), original)

    def test_public_builder_still_propagates_final_verification_errors(self):
        with patch.object(self.pilot, 'verify_observations',
                          side_effect=RuntimeError('public builder verifier failure')):
            with self.assertRaisesRegex(RuntimeError, 'public builder verifier failure'):
                self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)

    def test_cli_invalid_cutoff_is_refused_before_output_reservation(self):
        script = Path(self.pilot.__file__)
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'invalid-plan'
            result = subprocess.run([sys.executable, '-B', str(script), '--output', str(output),
                                     '--cutoff', '1'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists(), 'An invalid cutoff must not reserve execution custody')

    def test_cli_invalid_plan_leaves_destination_available_for_corrected_retry(self):
        script = Path(self.pilot.__file__)
        invalid_plans = [('--grid', '2'), ('--grid', '4', '--cutoff', '2'),
                         ('--grid', '6', '--cutoff', '3'),
                         ('--fields-per-model', '0'), ('--fields-per-model', '-1')]
        for plan in invalid_plans:
            with self.subTest(plan=plan), tempfile.TemporaryDirectory() as scratch:
                output = Path(scratch)/'same-destination'
                refused = subprocess.run([sys.executable, '-B', str(script), '--output', str(output), *plan],
                                          capture_output=True, text=True)
                self.assertNotEqual(refused.returncode, 0)
                self.assertFalse(output.exists(), 'An invalid plan must not reserve the destination')
                corrected = subprocess.run([sys.executable, '-B', str(script), '--output', str(output),
                                            '--grid', '8', '--cutoff', '2', '--fields-per-model', '1'],
                                           capture_output=True, text=True)
                self.assertEqual(corrected.returncode, 0, corrected.stderr)
                validation = json.loads((output/'validation.json').read_text())
                self.assertEqual(validation['exit_code'], 0)
                self.assertEqual(validation['planned_fields'], 50)

    def test_generation_failure_after_valid_plan_keeps_reserved_custody(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'generation-failed'
            with patch.object(self.pilot, '_generate_observations',
                              side_effect=RuntimeError('deliberate generation runtime failure')):
                with self.assertRaisesRegex(RuntimeError, 'deliberate generation runtime failure'):
                    self.pilot.main(['--output', str(output), '--grid', '8', '--cutoff', '2',
                                     '--fields-per-model', '1'])
            self.assertTrue(output.is_dir())

    def test_cli_verify_reports_retained_all_and_mixed_failures_as_non_success(self):
        def fail_all(model, seed, **kwargs):
            raise ArithmeticError('deliberate all-field failure')

        def fail_one(model, seed, **kwargs):
            if model['id'] == 'gaussian__gaussian':
                raise ArithmeticError('deliberate mixed-field failure')
            return self.pilot.models.sample_grid(model, seed, **kwargs)

        script = Path(self.pilot.__file__)
        for sampler, expected_failures in [(fail_all, 50), (fail_one, 1)]:
            with self.subTest(failures=expected_failures), tempfile.TemporaryDirectory() as scratch:
                document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1,
                                                        sampler=sampler)
                self.assertTrue(self.pilot.verify_observations(document))
                observations = Path(scratch)/'observations.json'
                original = json.dumps(document, sort_keys=True).encode('utf-8')
                observations.write_bytes(original)
                result = subprocess.run([sys.executable, '-B', str(script), '--verify', str(observations)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1, result.stdout+result.stderr)
                self.assertIn(f'retained failed fields: {expected_failures}', result.stdout)
                self.assertNotIn('VERIFICATION_PASS', result.stdout)
                self.assertEqual(observations.read_bytes(), original)

    def test_cli_verify_success_zero_and_malformed_two(self):
        script = Path(self.pilot.__file__)
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        with tempfile.TemporaryDirectory() as scratch:
            observations = Path(scratch)/'observations.json'
            observations.write_text(json.dumps(document))
            success = subprocess.run([sys.executable, '-B', str(script), '--verify', str(observations)],
                                     capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertIn('VERIFICATION_PASS', success.stdout)
            missing_config = copy.deepcopy(document)
            del missing_config['config']
            invalid_environment = copy.deepcopy(document)
            invalid_environment['environment'] = []
            malformed = Path(scratch)/'malformed.json'
            for contents in ('{}', '[]', json.dumps(missing_config), json.dumps(invalid_environment)):
                with self.subTest(contents=contents[:40]):
                    malformed.write_text(contents)
                    rejected = subprocess.run([sys.executable, '-B', str(script), '--verify', str(malformed)],
                                              capture_output=True, text=True)
                    self.assertEqual(rejected.returncode, 2)
                    self.assertNotIn('VERIFICATION_PASS', rejected.stdout)

    def _stage_failure(self, stage):
        def sampler(model, seed, **kwargs):
            if stage == 'sampling':
                raise ArithmeticError('deliberate sampling failure')
            if stage == 'quantization':
                # A complete finite float grid whose multiplication overflows.
                return [float.fromhex('0x1.fffffffffffffp+1023')]*64
            return [float((x*7+y*3) % 17) for x in range(8) for y in range(8)]

        original_bins = self.pilot.bin_counts

        def reject_completed_bins(intervals, scale, edges):
            if intervals:
                raise ArithmeticError('deliberate bin-count failure')
            return original_bins(intervals, scale, edges)

        with contextlib.ExitStack() as stack:
            if stage == 'barcode':
                stack.enter_context(patch.object(self.pilot.exact_h0, 'compute',
                                                 side_effect=ArithmeticError('deliberate barcode failure')))
            elif stage == 'connectivity':
                stack.enter_context(patch.object(self.pilot.exact_h0, 'verify_by_connectivity',
                                                 side_effect=ArithmeticError('deliberate connectivity failure')))
            elif stage == 'bin_counts':
                stack.enter_context(patch.object(self.pilot, 'bin_counts', side_effect=reject_completed_bins))
            return self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1, sampler=sampler)

    def test_legitimate_stage_failures_retain_only_completed_intermediates(self):
        expected = [('sampling', False, False, False),
                    ('quantization', True, False, False),
                    ('barcode', True, True, False),
                    ('connectivity', True, True, True),
                    ('bin_counts', True, True, True)]
        for stage, floats, quantized, barcode in expected:
            with self.subTest(stage=stage):
                document = self._stage_failure(stage)
                self.assertEqual(len(document['rows']), 50)
                for row in document['rows']:
                    self.assertEqual(row['failure']['stage'], stage)
                    self.assertEqual(row['float_samples_hex'] is not None, floats)
                    self.assertEqual(row['quantized_samples'] is not None, quantized)
                    self.assertEqual(row['barcode'] is not None, barcode)
                    self.assertIsNone(row['counts'])
                    self.assertIs(row['connectivity_verified'], False)
                self.assertTrue(self.pilot.verify_observations(document))

    def test_failed_stage_rejects_malformed_or_incompatible_intermediates(self):
        documents = {stage: self._stage_failure(stage) for stage in
                     ('sampling', 'quantization', 'barcode', 'connectivity', 'bin_counts')}
        variants = []

        def mutated(stage, field, value):
            document = copy.deepcopy(documents[stage])
            document['rows'][0][field] = value
            variants.append((stage+'/'+field, document))

        for field in ('float_samples_hex', 'quantized_samples', 'barcode'):
            mutated('sampling', field, [])
        for stage in ('quantization', 'barcode', 'connectivity', 'bin_counts'):
            for value in (None, ['0x1.0p+0'], ['inf']*64, [1]*64,
                          [text.upper() for text in documents[stage]['rows'][0]['float_samples_hex']]):
                mutated(stage, 'float_samples_hex', value)
        mutated('quantization', 'quantized_samples', [])
        mutated('quantization', 'barcode', {})
        for stage in ('barcode', 'connectivity', 'bin_counts'):
            values = documents[stage]['rows'][0]['quantized_samples']
            for value in (None, values[:-1], [True]+values[1:], [values[0]+1]+values[1:]):
                mutated(stage, 'quantized_samples', value)
        mutated('barcode', 'barcode', {})
        for stage in ('connectivity', 'bin_counts'):
            original = documents[stage]['rows'][0]['barcode']
            malformed = copy.deepcopy(original); malformed['zero_count'] += 1
            mutated(stage, 'barcode', malformed)
            malformed = copy.deepcopy(original); malformed['essential'][0] += 1
            mutated(stage, 'barcode', malformed)
            malformed = copy.deepcopy(original); malformed['extra'] = 'not emitted'
            mutated(stage, 'barcode', malformed)
            malformed = copy.deepcopy(original); malformed['zero_count'] = True
            mutated(stage, 'barcode', malformed)
            mutated(stage, 'barcode', None)
        for original, relabelled in [('sampling', 'quantization'), ('quantization', 'barcode'),
                                      ('barcode', 'connectivity'), ('connectivity', 'barcode'),
                                      ('bin_counts', 'sampling')]:
            failure = dict(documents[original]['rows'][0]['failure'], stage=relabelled)
            mutated(original, 'failure', failure)
        for label, document in variants:
            with self.subTest(mutation=label), self.assertRaises(ValueError):
                self.pilot.verify_observations(document)

    def test_connectivity_failure_replays_source_without_claiming_independent_check(self):
        document = self._stage_failure('connectivity')
        with patch.object(self.pilot.exact_h0, 'verify_by_connectivity',
                          side_effect=ArithmeticError('independent verifier still unavailable')):
            self.assertTrue(self.pilot.verify_observations(document))
            relabelled = copy.deepcopy(document)
            relabelled['rows'][0]['failure']['stage'] = 'bin_counts'
            with self.assertRaisesRegex(ArithmeticError, 'independent verifier still unavailable'):
                self.pilot.verify_observations(relabelled)

    def test_successful_float_serialization_requires_canonical_hex(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1,
                                                sampler=lambda model, seed, **kwargs: [1.0]*64)
        document['rows'][0]['float_samples_hex'][0] = '1.0'
        with self.assertRaises(ValueError):
            self.pilot.verify_observations(document)

    def test_structural_verifier_rejects_extra_metadata_at_each_declared_boundary(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        for location in ('observation', 'environment', 'config', 'row'):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as scratch:
                changed = copy.deepcopy(document)
                target = {'observation': changed, 'environment': changed['environment'],
                          'config': changed['config'], 'row': changed['rows'][0]}[location]
                target['scientific_status'] = 'CONFIRMED'
                with self.assertRaises(ValueError):
                    self.pilot.verify_observations(changed)
                observations = Path(scratch)/'observations.json'
                original = json.dumps(changed).encode('utf-8')
                observations.write_bytes(original)
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(self.pilot.main(['--verify', str(observations)]), 2)
                self.assertEqual(observations.read_bytes(), original)

    def test_environment_fields_reject_non_scalar_empty_or_noncanonical_strings(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        invalid = [None, {}, {'claims': 'CONFIRMED'}, True, False, 0, 1, 1.0,
                   [], ['Linux'], '', ' ', ' Linux', 'Linux ', '\n',
                   'Lin\nux', 'Lin\tux', 'Lin\x00ux', 'Lin\x7fux',
                   'Lin\u0085ux', 'Lin\u200bux']
        for field in ('python', 'implementation', 'machine', 'system'):
            for value in invalid:
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(document)
                    changed['environment'][field] = value
                    with self.assertRaises(ValueError):
                        self.pilot.verify_observations(changed)

    def test_python_environment_version_requires_portable_canonical_format(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        malformed = ['3', '3.12', '3.12.1.0', '3.12.x', 'v3.12.1', '03.12.1',
                     '3.012.1', '3.12.01', '3.13.0RC1', '3.13.0rc', '3.13.0rc01',
                     '3.13.0++', '3.13.0+local', '3.13.0.dev1', '3.13.0-final',
                     '\u0663.12.1']
        for version in malformed:
            with self.subTest(version=version):
                changed = copy.deepcopy(document)
                changed['environment']['python'] = version
                with self.assertRaises(ValueError):
                    self.pilot.verify_observations(changed)

    def test_environment_accepts_actual_runtime_and_portable_release_declarations(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        self.assertTrue(self.pilot.verify_observations(document))
        declarations = [('2.7.18', 'Jython', 'i686', 'Java'),
                        ('3.9.6', 'CPython', 'x86_64', 'Darwin'),
                        ('3.12.10', 'PyPy', 'aarch64', 'Linux'),
                        ('3.13.0a0', 'CPython', 'AMD64', 'Windows'),
                        ('3.13.0b1', 'CPython', 'arm64', 'Darwin'),
                        ('3.13.0rc2', 'CPython', 'ppc64le', 'FreeBSD'),
                        ('3.14.0a7+', 'CPython', 'Power Macintosh', 'Darwin'),
                        ('3.13.1+', 'IronPython', 'AMD64', 'Windows CE')]
        for values in declarations:
            with self.subTest(environment=values):
                changed = copy.deepcopy(document)
                changed['environment'].update(zip(('python', 'implementation', 'machine', 'system'), values))
                self.assertTrue(self.pilot.verify_observations(changed))

    def test_cli_rejects_malformed_environment_without_modifying_retained_bytes(self):
        document = self.pilot.build_observations(n=8, cutoff=2, fields_per_model=1)
        document['environment']['system'] = {'claims': 'CONFIRMED'}
        with tempfile.TemporaryDirectory() as scratch:
            observations = Path(scratch)/'observations.json'
            original = json.dumps(document).encode('utf-8')
            observations.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.pilot.main(['--verify', str(observations)]), 2)
            self.assertNotIn('VERIFICATION_PASS', stdout.getvalue())
            self.assertEqual(observations.read_bytes(), original)

    def test_cli_retains_invalid_returned_barcode_before_final_audit_rejection(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/'invalid-returned-barcode'
            with patch.object(self.pilot.exact_h0, 'compute', return_value={}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = self.pilot.main(['--output', str(output), '--grid', '8',
                                          '--cutoff', '2', '--fields-per-model', '1'])
            self.assertEqual(result, 2)
            observations = output/'observations.json'
            original = observations.read_bytes()
            document = json.loads(original)
            self.assertEqual(len(document['rows']), 50)
            self.assertTrue(all(row['failure']['stage'] == 'connectivity'
                                and row['barcode'] == {} and row['counts'] is None
                                and row['connectivity_verified'] is False for row in document['rows']))
            validation = json.loads((output/'validation.json').read_text())
            self.assertEqual(validation['final_record_verification'], 'FAIL')
            self.assertEqual(validation['exit_code'], 2)
            self.assertEqual(validation['observations_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(validation['error']['type'], 'ValueError')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.pilot.main(['--verify', str(observations)]), 2)
            with self.assertRaises(FileExistsError):
                self.pilot.main(['--output', str(output)])
            self.assertEqual(observations.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
