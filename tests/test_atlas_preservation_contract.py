"""Exercise the workflow's report validator, without emitting or decoding frames."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


def validator():
    lines = (ROOT / '.github/workflows/public-shop.yml').read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "<<'UL_ATLAS_PRESERVER_PY'" in line) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == 'UL_ATLAS_PRESERVER_PY')
    parsed = ast.parse(textwrap.dedent('\n'.join(lines[start:end])))
    names = {'Hold', 'strict_json', 'report_observation'}
    constants = {'EXPECTED_CASES', 'HEX64'}
    selected = []
    for node in parsed.body:
        if isinstance(node, ast.Import) and all(item.name in {'json', 'math', 're'} for item in node.names):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<workflow-report-validator>', 'exec'), namespace)
    return namespace


class PreservationContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = validator()
        self.report = json.loads((ROOT / 'docs/reports/atlas-hosted-admission-20261029/integrated45.json').read_bytes())
        self.checkout = self.report['checked_commit']

    def observe(self, report):
        return self.ns['report_observation'](json.dumps(report).encode(), self.checkout)

    def test_actual_complete_45_case_report_is_preserved(self):
        result = self.observe(self.report)
        self.assertEqual(result, {'checked_commit': self.checkout, 'cases': 45,
                                 'passed_cases': 45, 'failed_cases': 0})

    def test_old_39_case_report_is_incomplete(self):
        report = deepcopy(self.report)
        report['cases'] = [c for c in report['cases'] if not c.get('case', '').startswith('hosted-spatial-')]
        self.assertEqual(len(report['cases']), 39)
        with self.assertRaisesRegex(self.ns['Hold'], '^INCOMPLETE_REPORT$'):
            self.observe(report)

    def test_complete_failed_cases_remain_evidence_without_becoming_passes(self):
        report = deepcopy(self.report)
        for case in report['cases']:
            if case.get('case', '').startswith('hosted-spatial-render-pick-recovery-'):
                case['passed'] = False
                case['error'] = 'Controlled rendering failure'
        result = self.observe(report)
        self.assertEqual((result['cases'], result['passed_cases'], result['failed_cases']), (45, 43, 2))

    def test_missing_case_nonboolean_or_top_error_cannot_be_preserved_as_complete(self):
        for kind in ['missing', 'nonboolean', 'error']:
            with self.subTest(kind=kind):
                report = deepcopy(self.report)
                if kind == 'missing':
                    report['cases'].pop()
                elif kind == 'nonboolean':
                    report['cases'][0]['passed'] = 1
                else:
                    report['error'] = 'Source or browser initialization failed'
                with self.assertRaisesRegex(self.ns['Hold'], '^INCOMPLETE_REPORT$'):
                    self.observe(report)

    def test_checkout_digest_duplicate_json_and_nonfinite_refusals_stay_enforced(self):
        report = deepcopy(self.report)
        report['checked_commit'] = 'f' * 40
        with self.assertRaisesRegex(self.ns['Hold'], '^REPORT_CHECKOUT_MISMATCH$'):
            self.observe(report)
        report = deepcopy(self.report)
        report['source_payload_sha256'] = 'invalid'
        with self.assertRaisesRegex(self.ns['Hold'], '^INCOMPLETE_REPORT$'):
            self.observe(report)
        for raw in [b'{"cases":[],"cases":[]}', b'{"value":NaN}', b'{"value":1e999}']:
            with self.subTest(raw=raw), self.assertRaisesRegex(self.ns['Hold'], '^INVALID_OR_PARTIAL_REPORT_JSON$'):
                self.ns['report_observation'](raw, self.checkout)


if __name__ == '__main__':
    unittest.main()
