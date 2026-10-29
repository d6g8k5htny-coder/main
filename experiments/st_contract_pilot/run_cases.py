"""Verification runner, not a production CLI or field-admission service."""
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostic_test_bootstrap', root / 'check_scaling_diagnostic.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)  # Pins dependencies/fixtures; defines tests, does not run them.
results = []
for case in t.CASES:
    document = t.OV.apply_overlay(t.BASE, case['set'], case['remove'])
    results.append({'case_id': case['id'], 'observed': t.D.diagnose(document, case['operation'])})
print(json.dumps({'scope': 'BOUNDED_ST_ALGEBRA_DIAGNOSTIC_ONLY',
                  'optimized': sys.flags.optimize, 'cases_executed': len(results),
                  'scientific_acceptance': False, 'results': results}, sort_keys=True, indent=2))
