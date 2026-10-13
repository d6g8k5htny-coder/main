import sys, unittest
# Mirrors: python -m unittest discover -s <dir> -p 'test_population_certificate.py'
sys.argv = ['unittest', 'discover', '-s', 'experiments/universality/two_parameter_cusp_population', '-p', 'test_population_certificate.py']
unittest.main(module=None)
