"""Boundary and custody controls for the compact finite-grid certificate."""
import copy
import hashlib
import importlib
import importlib.util
import tempfile
import unittest
from fractions import Fraction as Q
from pathlib import Path
import finite_certificate as fc


def fixture():
    return {'schema_version': 1, 'model': fc.MODEL, 'side': 24, 'cutoff': 1,
            'grids': [4, 8], 'records': [{'seed': 0, 'dc': ['0', '0'],
            'modes': [[0, 1, '0', '0'], [1, -1, '0', '0'],
                      [1, 0, '1/2', '0'], [1, 1, '0', '0']]}]}


class HessianGridCertificateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('hessian_grid_certificate'),
                             'Compact exact-grid certificate is not implemented')
        self.hg = importlib.import_module('hessian_grid_certificate')

    def test_exact_dyadic_sample_digest_and_enclosure(self):
        # Catches wrong axis, normalization, or substituting NumPy samples.
        cert = self.hg.certify(fixture(), 0, 4, 4)
        scale = 2**96
        expected = [str(v) for v in [scale]*4+[0]*4+[-scale]*4+[0]*4]
        self.assertEqual(cert['samples']['real_integer_sha256'], fc.digest(expected))
        self.assertEqual(cert['samples']['count'], 16)
        self.assertGreater(Q(cert['samples']['nodal_error']), 0)
        self.assertLess(Q(cert['samples']['nodal_error']), Q(1, 10**20))
        self.assertEqual(Q(cert['abstract_diagram_bound']),
                         Q(cert['samples']['nodal_error'])+Q(cert['spatial_bound']))
        self.assertIsNone(cert['computed_barcode_error'])

    def test_digest_stream_preserves_sign_order_and_last_node(self):
        centers = [(7, 0), (-11, 8), (0, -2), (13, 0)]
        self.assertEqual(self.hg.real_center_digest(centers), fc.digest(['7','-11','0','13']))
        self.assertNotEqual(self.hg.real_center_digest(centers[:-1]+[(14,0)]),
                            self.hg.real_center_digest(centers))

    def test_reject_invalid_selection_and_coefficient_sources(self):
        for seed,dgrid,sgrid in [(True,4,4),(1,4,4),(0,True,4),(0,4,True),
                                (0,3,4),(0,4,16),(0,16,4)]:
            with self.subTest(selection=(seed,dgrid,sgrid)), self.assertRaises(ValueError):
                self.hg.certify(fixture(), seed, dgrid, sgrid)
        data=fixture();data['records'][0]['modes'][0][2]='1/3'
        with self.assertRaises(ValueError):self.hg.certify(data,0,4,4)

    def test_replay_rejects_lowered_bounds_digest_and_scope_promotions(self):
        data=fixture();cert=self.hg.certify(data,0,4,8)
        for key,value in [('abstract_diagram_bound','0'),('spatial_bound','0'),
                          ('gaussian_law_certified',True),('computed_barcode_error','0'),
                          ('persistence_implementation_certified',True)]:
            other=copy.deepcopy(cert);other[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.hg.verify(data,other)
        other=copy.deepcopy(cert);other['samples']['real_integer_sha256']='0'*64
        with self.assertRaises(ValueError):self.hg.verify(data,other)

    def test_bin_conditions_keep_strict_diagonal_and_empty_contraction(self):
        # Equality a=2epsilon cannot support the clean upper inequality;
        # equality b-a=4epsilon leaves an empty contracted interval.
        rows=self.hg.bin_conditions(Q(1,1000))
        self.assertFalse(rows[1]['clean_upper'])
        self.assertTrue(rows[2]['clean_upper'])
        self.assertFalse(rows[2]['nonempty_contraction'])
        self.assertTrue(rows[3]['nonempty_contraction'])

    def test_receipt_types_scope_and_regenerated_report(self):
        # Full real production and verification, followed by rebound mutations.
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new'
            self.hg.produce(path,seed=34000,derivative_grid=128,sample_grid=128)
            self.assertTrue(self.hg.verify_directory(path))
            original=(path/'RUN.json').read_bytes()
            receipt=fc.load_json(path/'RUN.json')
            role=next(iter(receipt['sources']))
            receipt['sources'][role]['bytes']=float(receipt['sources'][role]['bytes'])
            raw=fc.canonical_bytes(receipt);(path/'RUN.json').write_bytes(raw)
            (path/'RUN.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
            with self.assertRaises(ValueError):self.hg.verify_directory(path)
            (path/'RUN.json').write_bytes(original)
            (path/'RUN.sha256').write_text(hashlib.sha256(original).hexdigest()+'\n')
            (path/'RESULTS.md').write_text('Unqualified continuum confirmation')
            with self.assertRaises(ValueError):self.hg.verify_directory(path)
            with self.assertRaises(ValueError):self.hg.produce(path)


if __name__=='__main__':unittest.main()
