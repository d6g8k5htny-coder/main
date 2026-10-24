"""Receipt and scientific-boundary controls for a probabilistic tail result."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc


class GaussianTailCertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_tail_certificate'),
                             'Tail certificate wrapper is not implemented')
        self.mod=importlib.import_module('gaussian_tail_certificate')

    def test_frozen_finite_object_is_not_probabilistically_promoted(self):
        c=self.mod.certify()
        self.assertEqual(len(c['tail_bounds']),4)
        self.assertEqual(c['finite_object']['cutoff'],24)
        self.assertEqual(c['finite_object']['seed_label'],34000)
        self.assertIsNone(c['finite_object']['coupling_error'])
        self.assertIsNone(c['finite_object']['infinite_field_diagram_bound'])
        self.assertFalse(c['historical_gaussian_draw_certified'])
        self.assertFalse(c['lifetime_law_certified'])
        self.assertLess(Q(c['tail_bounds'][0]['normalization_loss_upper']),Q(3,10**64))
        for key,value in [('historical_gaussian_draw_certified',True),('lifetime_law_certified',True)]:
            altered=copy.deepcopy(c);altered[key]=value
            with self.assertRaises(ValueError):self.mod.verify(altered)
        altered=copy.deepcopy(c);altered['finite_object']['coupling_error']='0'
        with self.assertRaises(ValueError):self.mod.verify(altered)

    def test_new_directory_and_receipt_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'new';self.mod.produce(d)
            self.assertTrue(self.mod.verify_directory(d))
            with self.assertRaises(ValueError):self.mod.produce(d)
            c=fc.load_json(d/'CERTIFICATE.json');c['tail_bounds'][0]['tail_supremum_upper']='0'
            (d/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(c))
            # Rebinding a falsified output in the outer receipt is still rejected.
            r=fc.load_json(d/'RUN.json');r['outputs']['CERTIFICATE.json']=self.mod.identity(d/'CERTIFICATE.json')
            (d/'RUN.json').write_bytes(fc.canonical_bytes(r))
            (d/'RUN.sha256').write_text(self.mod.identity(d/'RUN.json')['sha256']+'\n')
            with self.assertRaises(ValueError):self.mod.verify_directory(d)


if __name__=='__main__':unittest.main()
