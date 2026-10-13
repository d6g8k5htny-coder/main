"""Run the finder's F4 probes against the origin/main checker copy, reusing every
non-test helper of the current house fixture (the finder copied a fixed list that
predates install_source)."""
import os, sys, unittest
os.environ.setdefault('INTAKE_MODE', 'current')
sys.path.insert(0, os.path.dirname(__file__))
import probe_content_rules as pcr
for name in dir(pcr.house.IntakeTests):
    if not name.startswith(('test_', '_')) and callable(getattr(pcr.house.IntakeTests, name)) and not hasattr(pcr.Probes, name):
        setattr(pcr.Probes, name, getattr(pcr.house.IntakeTests, name))
suite = unittest.TestSuite(pcr.Probes(n) for n in ['test_F4a_out_of_range_float_literal_parses_to_inf',
                                                    'test_F4_literal_infinity_is_refused_in_both_modes',
                                                    'test_F4_ordinary_floats_control'])
raise SystemExit(0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1)
