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


if __name__ == '__main__':
    unittest.main()
