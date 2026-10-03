"""Stdlib contract checks for the optional browser-QA runner, not browser evidence."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class BrowserCheckContract(unittest.TestCase):
    def test_browser_job_is_read_only_and_retains_failure_artifacts(self):
        workflow=(ROOT/'.github/workflows/public-shop.yml').read_text()
        self.assertIn('  browser-smoke:',workflow)
        self.assertNotIn('pull_request_target:',workflow)
        self.assertIn('contents: read',workflow)
        block=workflow.split('  browser-smoke:',1)[1]
        self.assertIn('persist-credentials: false',block)
        self.assertIn('if: always()',block)
        self.assertIn('path: browser-evidence',block)
        self.assertIn('timeout-minutes: 10',block)
        self.assertNotIn('playwright install',block, 'Use the runner’s supported packaged browser sandbox')
        self.assertIn('runs-on: ubuntu-24.04',block)
        self.assertNotIn('secrets.',block)
        self.assertNotIn('deploy',block)
        for test in ('tests/test_measurement_model.mjs','tests/test_dependency_viewer.mjs'):
            self.assertIn(test,workflow)

    def test_runner_is_loopback_only_with_a_four_case_matrix(self):
        path=ROOT/'tools/public_shop_browser_check.py'
        self.assertTrue(path.is_file(),'Browser runner is missing')
        source=path.read_text();ast.parse(source)
        self.assertIn('("127.0.0.1", 0)',source)
        self.assertNotIn('0.0.0.0',source)
        for item in ['1200','900','390','844','"light"','"dark"','go_back','go_forward','to_be_focused','scrollWidth','screenshot','checked_commit','browser.version']:
            self.assertIn(item,source)
        self.assertIn('channel="chrome",chromium_sandbox=True',source)
        self.assertIn('browser_executable_sha256',source)
        self.assertNotIn('assert ',source, 'CLI checks must not disappear under Python -O')
        self.assertIn('Access-Control-Allow-Origin',source)
        self.assertIn('Source unavailable (503)',source)
        self.assertIn('len(route_hits)==1',source)
        self.assertNotIn('ignore_https_errors',source)
        self.assertNotIn('--no-sandbox',source)

    def test_browser_dependencies_are_separate_and_version_pinned(self):
        path=ROOT/'tests/browser-requirements.txt'
        self.assertTrue(path.is_file(),'Optional browser dependencies are missing')
        requirements=[line for line in path.read_text().splitlines() if line and not line.startswith('#')]
        self.assertEqual(len(requirements),4)
        self.assertTrue(all('==' in line and '://' not in line for line in requirements))
        self.assertIn('playwright==1.62.0',requirements)

class ReaderSourceBrowserContract(unittest.TestCase):
    def test_existing_runner_covers_source_quote_and_keyboard_disclosure(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in ['check_source_card_flow','source-quote-readable','source-quote-original','Exact source quote','issuecomment-5841270276','len(report["cases"])==11']:
            self.assertIn(token,source)

    def test_reader_tools_have_desktop_and_mobile_browser_evidence(self):
        source=(ROOT/'tools/public_shop_browser_check.py').read_text()
        for token in [
            'check_reader_tools_flow', 'measure.html', 'dependencies.html',
            'Pinned Math commit 7858329974e2', 'math.rn-fixed-annulus-window',
            'Saved selection unavailable', 'to_be_focused', 'post-layout saved-link focus',
            'documentElement.scrollWidth',
            'len(report["cases"])==11',
        ]:
            self.assertIn(token,source)

if __name__=='__main__':unittest.main()
