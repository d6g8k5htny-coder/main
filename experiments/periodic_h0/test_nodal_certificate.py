"""Behavior checks for new sample-bound nodal certificates."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc


def fixture():
    data = {'schema_version': 1, 'model': fc.MODEL, 'side': 24, 'cutoff': 1,
            'grids': [4], 'records': [{'seed': 0, 'dc': ['0', '0'],
            'modes': [[0, 1, '0', '0'], [1, -1, '0', '0'],
                      [1, 0, '1', '0'], [1, 1, '0', '0']]}]}
    samples = {'schema_version': 1, 'coefficient_object_sha256': fc.digest(data),
               'seed': 0, 'grid': 4, 'layout': 'row-major-x-then-y',
               'values': [float(v).hex() for v in [2]*4+[0]*4+[-2]*4+[0]*4]}
    return data, samples


class NodalCertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('nodal_certificate'),
                             'Missing sample-bound nodal certificate implementation')
        self.nc = importlib.import_module('nodal_certificate')

    def test_exact_cosine_and_every_node_including_last(self):
        data, samples = fixture()
        cert = self.nc.certify(data, samples)
        self.assertLess(Q(cert['nodal_error']), Q(1, 10**20))
        self.assertEqual(Q(cert['abstract_diagram_bound']), Q(cert['spatial_bound'])+Q(cert['nodal_error']))
        altered = copy.deepcopy(samples)
        altered['values'][-1] = '0x1.0000000000000p+0'
        other = self.nc.certify(data, altered)
        self.assertGreaterEqual(Q(other['nodal_error']), 1)
        with self.assertRaises(ValueError): self.nc.verify(data, altered, cert)

    def test_reject_wrong_object_grid_layout_and_bad_float(self):
        data, samples = fixture()
        for key, value in [('coefficient_object_sha256', '0'*64), ('grid', True),
                           ('grid', 8), ('seed', 42), ('layout', 'y-first'),
                           ('values', samples['values'][:-1])]:
            changed = copy.deepcopy(samples); changed[key] = value
            with self.subTest(key=key, value=str(value)[:50]):
                with self.assertRaises(ValueError): self.nc.certify(data, changed)
        for value in ['nan', 'inf', '0x1p+0', '0x1p+99999', 1, True]:
            changed = copy.deepcopy(samples); changed['values'][0] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError): self.nc.certify(data, changed)

    def test_reject_rebound_certificate_scope_and_lowered_bound(self):
        data, samples = fixture(); cert = self.nc.certify(data, samples)
        for key, value in [('nodal_error', '0'), ('abstract_diagram_bound', '0'),
                           ('historical_fft_certified', True),
                           ('persistence_implementation_certified', True),
                           ('continuum_field_certified', True)]:
            changed = copy.deepcopy(cert); changed[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): self.nc.verify(data, samples, changed)

    def test_sample_generator_preserves_exact_coefficient_and_coordinates(self):
        producer = importlib.import_module('run_nodal_certificate')
        data, _ = fixture(); samples = producer.sample(data, 0, 4)
        self.assertEqual(samples['values'], fixture()[1]['values'])
        # A non-binary64 coefficient is not silently rounded before evaluation.
        data['records'][0]['modes'][2][2] = str(Q(1)+Q(1, 2**60))
        with self.assertRaises(ValueError): producer.sample(data, 0, 4)

    def test_directory_custody_and_receipt_scope(self):
        producer = importlib.import_module('run_nodal_certificate')
        base = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'new'
            producer.produce(out, seed=34000, grid=128)
            self.assertTrue(self.nc.verify_directory(out))
            receipt = fc.load_json(out/'RUN.json')
            receipt['scientific_effect'] = 'Human-reviewed continuum theorem'
            raw = fc.canonical_bytes(receipt)
            (out/'RUN.json').write_bytes(raw)
            import hashlib
            (out/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
            with self.assertRaises(ValueError): self.nc.verify_directory(out)
            with self.assertRaises(ValueError): producer.produce(out, seed=34000, grid=128)


if __name__ == '__main__': unittest.main()
