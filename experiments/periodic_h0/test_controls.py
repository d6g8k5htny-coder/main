import unittest
import importlib.util
import numpy as np
spec=importlib.util.find_spec('controls')
if spec:import controls
else:controls=None

class ControlTests(unittest.TestCase):
    def test_real_generator_basis_controls_reject_both_wrong_ensembles(self):
        self.assertIsNotNone(controls,'Independent controls are not yet implemented')
        report=controls.run_controls()
        self.assertTrue(report['all_controls_pass'])
        self.assertLess(report['covariance_basis_max_error'],1e-12)
        for name in ('wrong_spectrum','missing_conjugate_variance','missing_periodic_gluing','wrong_elder','missing_volume'):
            self.assertTrue(report['mutants_rejected'][name],name)

if __name__=='__main__':unittest.main()
