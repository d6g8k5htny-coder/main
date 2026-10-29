#!/usr/bin/env python3
"""Structured, fail-closed unittest runner used by G0 and mutations."""
import argparse
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test', action='append', default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'tests'))
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromNames(args.test)
    else:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    value = {'tests_run': result.testsRun, 'assertion_failures': len(result.failures),
             'errors': len(result.errors), 'skipped': len(result.skipped),
             'expected_failures': len(result.expectedFailures),
             'unexpected_successes': len(result.unexpectedSuccesses),
             'successful': result.wasSuccessful(), 'log': stream.getvalue()}
    Path(args.output).write_text(json.dumps(value, indent=2) + '\n')
    print(value['log'], end='')
    return 0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
