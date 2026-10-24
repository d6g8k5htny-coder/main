"""Exact replay and false-scope controls for finite-cutoff expectation losses."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc


class FiniteCountCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('finite_count_loss_certificate') is None:
            raise AssertionError('Finite count-loss certificate is missing')
        cls.m=importlib.import_module('finite_count_loss_certificate')
        cls.c=cls.m.certify()

    def test_count_weighted_losses_enclose_known_scales(self):
        for row,cap,upper in zip(self.c['budgets'],[16,9216,16384],
                                 [Q(182,10**15),Q(28,10**9),Q(88,10**9)]):
            self.assertEqual(row['finite_bar_cap'],cap)
            self.assertEqual(Q(row['expected_count_loss_upper']),cap*Q(row['clipping_failure_upper']))
            self.assertGreater(Q(row['expected_count_loss_upper']),Q(row['clipping_failure_upper']))
            self.assertLess(Q(row['expected_count_loss_upper']),upper)
            self.assertGreaterEqual(Q(row['expected_count_loss_decimal_up']),Q(row['expected_count_loss_upper']))
        self.assertTrue(self.m.verify(self.c))

    def test_rejects_false_caps_probabilities_and_promotions(self):
        mutants=[lambda c:c['budgets'][0].update(finite_bar_cap=8),
                 lambda c:c['budgets'][1].update(expected_count_loss_upper=c['budgets'][1]['clipping_failure_upper']),
                 lambda c:c['budgets'][2].update(coefficient_error_upper='0'),
                 lambda c:c['budgets'][0].update(cutoff=24),
                 lambda c:c.update(iid_input_law_certified=True),
                 lambda c:c.update(new_gaussian_barcodes=1),
                 lambda c:c.update(infinite_field_expected_count_certified=True),
                 lambda c:c.update(expected_bin_counts_computed=True)]
        for mutate in mutants:
            c=copy.deepcopy(self.c);mutate(c)
            with self.assertRaises(ValueError):self.m.verify(c)

    def test_rebound_receipt_cannot_hide_semantic_or_report_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'run';self.m.produce(d)
            self.assertTrue(self.m.verify_directory(d))
            cert=fc.load_json(d/'CERTIFICATE.json')
            cert['budgets'][0]['expected_count_loss_upper']='0'
            (d/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(cert))
            (d/'RESULTS.md').write_text(self.m.render_report(cert))
            r=fc.load_json(d/'RUN.json')
            r['outputs']['CERTIFICATE.json']=self.m.identity(d/'CERTIFICATE.json')
            (d/'RUN.json').write_bytes(fc.canonical_bytes(r))
            (d/'RUN.sha256').write_text(self.m.identity(d/'RUN.json')['sha256']+'\n')
            with self.assertRaises(ValueError):self.m.verify_directory(d)


if __name__=='__main__':unittest.main()
