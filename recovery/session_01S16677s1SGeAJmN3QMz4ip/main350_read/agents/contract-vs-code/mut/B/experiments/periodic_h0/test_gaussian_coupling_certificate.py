"""A changed claim cannot be made valid merely by rebinding its hashes."""
import copy
import importlib
import importlib.util
import unittest


class CouplingCertificateTests(unittest.TestCase):
    def test_exact_claim_and_semantic_negative_controls(self):
        self.assertIsNotNone(importlib.util.find_spec('gaussian_coupling_certificate'),
                             'Prospective coupling certificate is not implemented')
        m=importlib.import_module('gaussian_coupling_certificate')
        c=m.certify();self.assertTrue(m.verify(c))
        mutations=[lambda x:x.update(historical_gaussian_draw_certified=True),
                   lambda x:x.update(iid_input_law_certified=True),
                   lambda x:x.update(new_gaussian_barcodes=1),
                   lambda x:x['historical_object'].update(coupling_error='0'),
                   lambda x:x['budgets'][0].update(clipping_failure_upper='0'),
                   lambda x:x['budgets'][0].update(coefficient_error_upper='0'),
                   lambda x:x['budgets'][0].update(real_gaussians=1200)]
        for mutate in mutations:
            bad=copy.deepcopy(c);mutate(bad)
            with self.assertRaises(ValueError):m.verify(bad)


if __name__=='__main__':unittest.main()
