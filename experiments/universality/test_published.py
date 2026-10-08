"""Published observation/validation binding and refusal controls."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.universality import run_pilot


class PublishedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = run_pilot.build_observations(n=8, cutoff=2, fields_per_model=1)

    def setUp(self):
        self.verifier = importlib.import_module('experiments.universality.verify_published')
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.directory = Path(self.scratch.name)

    def save(self, record=None):
        record = copy.deepcopy(self.record if record is None else record)
        payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
        (self.directory/'observations.json').write_bytes(payload)
        failed = sum(row['failure'] is not None for row in record['rows'])
        validation = {'schema_version': 1, 'scientific_status_authority': False,
                      'observations_file': 'observations.json',
                      'observations_sha256': hashlib.sha256(payload).hexdigest(),
                      'final_record_verification': 'PASS', 'planned_fields': len(record['rows']),
                      'failed_fields': failed, 'error': None, 'exit_code': 0 if not failed else 1,
                      'scope': 'Retained-record consistency only; no generator-authenticity or continuum claim'}
        self.save_validation(validation)
        return validation

    def save_validation(self, validation):
        (self.directory/'validation.json').write_text(json.dumps(validation)+'\n')

    def test_source_bound_successful_record_and_validation_pass(self):
        self.save()
        self.assertTrue(self.verifier.verify_published_artifact(self.directory))

    def test_changed_observation_bytes_refuse_stale_validation_digest(self):
        self.save()
        path = self.directory/'observations.json'
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(ValueError):
            self.verifier.verify_published_artifact(self.directory)

    def test_corrupt_grid_count_refused_even_with_updated_validation_digest(self):
        record = copy.deepcopy(self.record)
        record['rows'][0]['counts'][0] += 1
        self.save(record)
        with self.assertRaises(ValueError):
            self.verifier.verify_published_artifact(self.directory)

    def test_stale_source_refused_even_with_updated_validation_digest(self):
        record = copy.deepcopy(self.record)
        record['source_sha256']['experiments/universality/models.py'] = '0'*64
        self.save(record)
        with self.assertRaises(ValueError):
            self.verifier.verify_published_artifact(self.directory)

    def test_false_or_noncanonical_validation_refused(self):
        for key, value in [('schema_version', True), ('scientific_status_authority', 0),
                           ('planned_fields', True), ('failed_fields', False),
                           ('exit_code', 1), ('final_record_verification', 'FAIL'),
                           ('error', {'type': 'RuntimeError', 'message': 'failed'}),
                           ('scientific_status', 'CONFIRMED')]:
            with self.subTest(key=key):
                validation = self.save()
                validation[key] = value
                self.save_validation(validation)
                with self.assertRaises(ValueError):
                    self.verifier.verify_published_artifact(self.directory)

    def test_consistent_retained_field_failures_cannot_pass_publication_check(self):
        def fail(*args, **kwargs):
            raise ArithmeticError('deliberate failure')
        record = run_pilot.build_observations(n=8, cutoff=2, fields_per_model=1, sampler=fail)
        self.assertTrue(run_pilot.verify_observations(record))
        validation = self.save(record)
        validation['exit_code'] = 0
        self.save_validation(validation)
        with self.assertRaises(ValueError):
            self.verifier.verify_published_artifact(self.directory)

    def test_duplicate_validation_key_refused(self):
        self.save()
        path = self.directory/'validation.json'
        path.write_text(path.read_text().rstrip()[:-1]+', "exit_code": 0}\n')
        with self.assertRaises(ValueError):
            self.verifier.verify_published_artifact(self.directory)


if __name__ == '__main__':
    unittest.main()
