"""Custody and promotion controls for the C38 exact-grid barcode successor."""
import copy
import hashlib
import importlib
import importlib.util
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc


class ExactBarcodeCertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exact_barcode_certificate'),
                             'Exact barcode certificate is not implemented')
        self.mod = importlib.import_module('exact_barcode_certificate')

    def test_upstream_certificate_is_a_fixed_reviewed_object(self):
        # Rebinding an edited bound to a new digest cannot turn it into C38.
        upstream = self.mod.load_upstream()
        self.assertEqual(upstream['seed'], 34000)
        raw = (self.mod.BASE/self.mod.UPSTREAM_CERT).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/self.mod.UPSTREAM_CERT
            path.parent.mkdir(parents=True)
            edited = fc.load_json(self.mod.BASE/self.mod.UPSTREAM_CERT)
            edited['abstract_diagram_bound'] = '0'
            path.write_bytes(fc.canonical_bytes(edited))
            with self.assertRaises(ValueError): self.mod.load_upstream(Path(tmp))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), self.mod.UPSTREAM_SHA256)

    def test_tiny_exact_report_keeps_unbounded_short_bin_and_large_integers(self):
        # Hand-derived 3x3 torus: every row/column wraps; two maxima merge at 0.
        values = [2**96, 0, 0, 0, 2**95, 0, 0, 0, 0]
        result = self.mod.analyze(values, 3, 2**96, Q(1, 1000), [Q(1,1000),Q(1,2),Q(1)])
        self.assertEqual(result['barcode']['intervals'], [[str(2**95), '0']])
        self.assertEqual(result['barcode']['essential'], [str(2**96)])
        self.assertIsNone(result['bins'][0]['target_upper_count'])
        self.assertEqual(result['bins'][1]['sample_count'], 1)
        self.assertEqual(result['bins'][1]['target_upper_count'], 1)
        self.assertEqual(result['bins'][1]['target_lower_count'], 0)

    def test_stored_successor_rejects_rebound_counts_and_claims(self):
        # Exercise real publication bytes without rerunning million-node work
        # for each mutation: identity validation must reject the output first.
        original = self.mod.BASE/'results/exact_h0_1'
        if not original.exists(): self.skipTest('Production snapshot generated after implementation')
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'copy'; target.mkdir()
            for name in ('CERTIFICATE.json','RUN.json','RUN.sha256','RESULTS.md'):
                (target/name).write_bytes((original/name).read_bytes())
            cert = fc.load_json(target/'CERTIFICATE.json')
            for key, value in [('gaussian_law_certified', True), ('upstream_sha256','0'*64),
                               ('computed_barcode_error','1/100'), ('infinite_field_certified',True)]:
                altered = copy.deepcopy(cert); altered[key] = value
                (target/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(altered))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.mod.validate_directory(target)
            (target/'CERTIFICATE.json').write_bytes((original/'CERTIFICATE.json').read_bytes())
            receipt = fc.load_json(target/'RUN.json')
            role = next(iter(receipt['sources']))
            receipt['sources'][role]['bytes'] = float(receipt['sources'][role]['bytes'])
            raw = fc.canonical_bytes(receipt); (target/'RUN.json').write_bytes(raw)
            (target/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
            with self.assertRaises(ValueError): self.mod.validate_directory(target)
        with self.assertRaises(ValueError): self.mod.produce(original)


if __name__ == '__main__': unittest.main()
