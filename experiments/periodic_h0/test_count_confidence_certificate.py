"""Exact finite-law controls and rejection of rebound false confidence claims."""
import copy
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction as Q
import finite_certificate as fc


class ConfidenceCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('count_confidence_certificate') is None:
            raise AssertionError('Confidence certificate is missing')
        cls.m=importlib.import_module('count_confidence_certificate')
        cls.c=cls.m.certify()

    def test_planning_scales_and_parent_loss(self):
        for row,cap,n in zip(self.c['plans'],[16,9216,16384],[1024,339738624,1073741824]):
            self.assertEqual(row['finite_bar_cap'],cap)
            self.assertEqual(Q(row['radius']),Q(cap,100))
            self.assertEqual(row['sample_count_for_radius_one'],n)
            self.assertEqual(Q(row['risk_decimal_up']),Q('0.005367402046440190'))
            self.assertLessEqual(Q(row['risk_upper']),Q(row['risk_decimal_up']))
            self.assertEqual(Q(row['expected_count_loss_upper']),cap*Q(row['clipping_failure_upper']))
        self.assertTrue(self.m.verify(self.c))

    def test_exhaustive_toy_laws_and_invalid_shortcuts(self):
        c=self.c['toy_controls']
        self.assertEqual(c['enumerated_sequences'],256)
        self.assertEqual(Q(c['exact_two_sided_failure']),Q(9,128))
        self.assertEqual(Q(c['retained_unresolved_upper_failure']),Q(9,256))
        self.assertEqual(Q(c['discarding_failure']),Q(255,256))
        self.assertEqual(c['adaptive_widening_violations'],0)
        self.assertEqual(Q(c['family_failure']),1)
        self.assertLess(Q(c['wrong_single_bin_risk']),1)
        self.assertEqual(Q(c['correct_family_risk']),1)
        self.assertEqual(Q(c['dependent_row_failure']),1)
        self.assertLess(Q(c['wrong_independent_risk']),1)
        self.assertLess(Q(c['wrong_capless_risk']),Q(9,128))
        self.assertGreater(Q(c['correct_two_sided_risk']),Q(9,128))

    def test_semantic_changes_rejected_even_with_rebound_receipt(self):
        mutations=[lambda c:c['plans'][0].update(sample_count=39999),
            lambda c:c['plans'][0].update(bins=1),
            lambda c:c['plans'][0].update(cutoff=24),
            lambda c:c['plans'][0].update(clipping_failure_upper='0'),
            lambda c:c['plans'][0].update(risk_upper='0'),
            lambda c:c['toy_controls'].update(exact_two_sided_failure='0'),
            lambda c:c.update(iid_input_law_certified=True),
            lambda c:c.update(observed_ensemble_means_computed=True),
            lambda c:c.update(infinite_field_expected_count_certified=True),
            lambda c:c.update(scope='Unconditional observed Gaussian confidence')]
        for change in mutations:
            c=copy.deepcopy(self.c);change(c)
            with self.assertRaises(ValueError):self.m.verify(c)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'run';self.m.produce(d)
            self.assertTrue(self.m.verify_directory(d))
            c=fc.load_json(d/'CERTIFICATE.json');c['plans'][0]['risk_upper']='0'
            (d/'CERTIFICATE.json').write_bytes(fc.canonical_bytes(c))
            (d/'RESULTS.md').write_text(self.m.render_report(c))
            r=fc.load_json(d/'RUN.json');r['outputs']['CERTIFICATE.json']=self.m.identity(d/'CERTIFICATE.json')
            (d/'RUN.json').write_bytes(fc.canonical_bytes(r))
            (d/'RUN.sha256').write_text(self.m.identity(d/'RUN.json')['sha256']+'\n')
            with self.assertRaises(ValueError):self.m.verify_directory(d)


if __name__=='__main__':unittest.main()
