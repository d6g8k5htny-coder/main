#!/usr/bin/env python3
"""Run exactly two Atlas browser tests; skips are failure, never optional green.

Use without -S after installing tests/browser-requirements.txt. The output
folder must be new. This is trusted unittest execution, not hostile-code
containment, mathematical certification or a signature of remote execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = (
    'tests.test_atlas_focus_geometry.AtlasListFocusGeometry.test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary',
    'tests.test_atlas_reader_scroll.AtlasReaderScroll.test_new_selection_after_a_long_read_keeps_its_heading_visible',
)


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


class ExactResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_ids = []

    def addSuccess(self, test):
        self.successful_ids.append(test.id())
        super().addSuccess(test)


def run_suite(suite, stream):
    if tuple(test_ids(suite)) != EXPECTED_IDS:
        raise ValueError('test inventory must equal the two fixed Atlas method IDs')
    result = unittest.TextTestRunner(stream=stream, verbosity=2,
                                    resultclass=ExactResult).run(suite)
    passed = (result.wasSuccessful() and result.testsRun == 2
              and tuple(result.successful_ids) == EXPECTED_IDS
              and not result.skipped and not result.expectedFailures
              and not result.unexpectedSuccesses)
    return {'passed': passed, 'tests_run': result.testsRun,
            'successful_ids': result.successful_ids,
            'skips': [[test.id(), reason] for test, reason in result.skipped],
            'failures': len(result.failures), 'errors': len(result.errors),
            'expected_failures': len(result.expectedFailures),
            'unexpected_successes': len(result.unexpectedSuccesses)}


def git_identity():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('tracked source is dirty')
    return {'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}')}


def runtime_identity():
    browsers = []
    for module_name, attribute in (
            ('tests.test_atlas_focus_geometry', 'CHROMIUM'),
            ('tests.test_atlas_reader_scroll', 'BROWSER')):
        executable = getattr(sys.modules[module_name], attribute)
        if not executable:
            raise ValueError('required packaged browser executable is unavailable')
        path = Path(executable).resolve(strict=True)
        version = subprocess.check_output([str(path), '--version'], text=True,
                                          stderr=subprocess.STDOUT, timeout=15).strip()
        browsers.append({'module': module_name, 'path': str(path), 'version': version,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'python': sys.version, 'optimize': sys.flags.optimize,
            'playwright': importlib.metadata.version('playwright'), 'browsers': browsers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args(argv).output.resolve()
    # Refuse reuse before any browser/test work; retain earlier attempts intact.
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'atlas-browser-regressions/1', 'passed': False,
              'expected_ids': list(EXPECTED_IDS), 'scientific_effect': 'NONE'}
    try:
        report['source_before'] = git_identity()
        sys.path.insert(0, str(ROOT))
        os.environ['ATLAS_FOCUS_EVIDENCE'] = str(output / 'focus')
        os.environ['ATLAS_READER_EVIDENCE'] = str(output / 'reader')
        suite = unittest.defaultTestLoader.loadTestsFromNames(EXPECTED_IDS)
        # Loader errors and absent prerequisites must not become a skip-green.
        if tuple(test_ids(suite)) != EXPECTED_IDS:
            raise ValueError('test inventory must equal the two fixed Atlas method IDs')
        report['runtime'] = runtime_identity()
        report.update(run_suite(suite, sys.stderr))
        report['source_after'] = git_identity()
        if report['source_before'] != report['source_after']:
            raise ValueError('source identity changed during browser tests')
    except Exception as error:
        report['passed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        (output / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
