"""Rebound metadata cannot certify a different polynomial or scientific scope."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc


class WordPolynomialCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('word_polynomial_certificate') is None:
            raise AssertionError('Word-polynomial certificate is missing')
        cls.m=importlib.import_module('word_polynomial_certificate')
        cls.c=cls.m.certify()

    def test_exact_fixture_and_bin_count(self):
        c=self.c; r=c['result']
        self.assertTrue(self.m.verify(c))
        self.assertEqual(c['input']['words'],[str(j*((1<<128)-1)//17) for j in [1,13,4,16,7,11,2,15,9]])
        self.assertEqual(int(r['lift']),1<<97)
        self.assertEqual(len(r['barcode']['intervals']),3)
        target=next(x for x in r['bins'] if Q(x['lower'])==Q(1,125))
        self.assertEqual(Q(target['upper']),Q(2,125))
        self.assertEqual((target['target_lower_count'],target['target_upper_count']),(1,1))
        for row in r['bins']:
            if row is not target:self.assertEqual((row['target_lower_count'],row['target_upper_count']),(0,0))

    def test_semantic_mutants_fail_even_after_rebinding(self):
        mutations=[lambda c:c['input']['words'].__setitem__(0,'0'),
                   lambda c:c['real_polynomial'].update(dc='0'),
                   lambda c:c['result'].update(sample_scale=str(1<<96)),
                   lambda c:c['result'].update(nodal_error='0'),
                   lambda c:c['result'].update(spatial_error='0'),
                   lambda c:c['result'].update(sample_sha256='0'*64),
                   lambda c:c['result']['barcode']['intervals'].pop(),
                   lambda c:c['result']['bins'][3].update(target_upper_count=0),
                   lambda c:c.update(iid_input_law_certified=True),
                   lambda c:c.update(new_gaussian_barcodes=1),
                   lambda c:c['result'].update(infinite_field_diagram_bound='0'),
                   lambda c:c['result'].update(historical_coupling_error='0')]
        for mutate in mutations:
            bad=copy.deepcopy(self.c);mutate(bad)
            bad['input_sha256']=fc.digest(bad['input'])
            bad['real_polynomial_sha256']=fc.digest(bad['real_polynomial'])
            with self.assertRaises(ValueError):self.m.verify(bad)

    def test_strict_receipt_and_report_custody(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'run';m.produce(d);self.assertTrue(m.verify_directory(d))
            original=fc.load_json(d/'RUN.json')
            mutations=[lambda r:r.update(schema_version=True),
                       lambda r:r.update(utc='2026-10-01'),
                       lambda r:r.update(extra='ignored'),
                       lambda r:r['sources'].pop(next(iter(r['sources']))),
                       lambda r:r['outputs']['CERTIFICATE.json'].update(bytes=True)]
            for mutate in mutations:
                r=copy.deepcopy(original);mutate(r)
                (d/'RUN.json').write_bytes(fc.canonical_bytes(r))
                (d/'RUN.sha256').write_text(m.identity(d/'RUN.json')['sha256']+'\n')
                with self.assertRaises(ValueError):m.verify_directory(d)
            (d/'RUN.json').write_bytes(fc.canonical_bytes(original))
            (d/'RUN.sha256').write_text(m.identity(d/'RUN.json')['sha256']+'\n')
            (d/'RESULTS.md').write_text('Promoted to a Gaussian theorem.\n')
            with self.assertRaises(ValueError):m.verify_directory(d)


if __name__=='__main__':unittest.main()
