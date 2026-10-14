"""Real scanner regressions for the committed formal-guide page declaration.

Only the committed page list is reused. Disposable synthetic documents and an
empty public-target list keep these controls local; no repository file is edited.
"""
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUIDE = 'docs/FORMAL_VERIFICATION.md'
SCANNER = ROOT / 'tools/navigation_check.py'
spec = importlib.util.spec_from_file_location('navigation_manifest_scanner', SCANNER)
nav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nav)


class FormalGuideNavigationTests(unittest.TestCase):
    def setUp(self):
        self.pages = nav.load(ROOT)['pages']
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for page in self.pages:
            rel = PurePosixPath(page)
            self.assertFalse(rel.is_absolute() or '..' in rel.parts)
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('# Fixture\n', encoding='utf-8')
        (self.root / 'docs').mkdir(exist_ok=True)
        (self.root / 'docs/NAVIGATION.json').write_text(
            json.dumps({'version': 1, 'pages': self.pages, 'public_targets': []}),
            encoding='utf-8',
        )
        (self.root / 'README.md').write_text(
            f'# Home\n[Guide]({GUIDE})\n', encoding='utf-8',
        )
        self.guide = self.root / GUIDE
        self.guide.write_text(
            '# Guide\n[Target](navigation-fixture-target.md#result)\n',
            encoding='utf-8',
        )
        (self.root / 'docs/navigation-fixture-target.md').write_text(
            '# Result\n', encoding='utf-8',
        )

    def cli(self):
        flags = ['-O'] if sys.flags.optimize else []
        process = subprocess.run(
            [sys.executable, '-B', '-S', *flags, str(SCANNER),
             '--root', str(self.root)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding='utf-8', timeout=15, cwd=self.root,
        )
        self.assertEqual(process.stderr, '')
        return process.returncode, json.loads(process.stdout)

    def test_manifest_lists_guide_once(self):
        self.assertEqual(self.pages.count(GUIDE), 1)

    def test_valid_guide_links(self):
        code, result = self.cli()
        self.assertEqual(code, 0)
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['public_sources_verified'], [])

    def test_guide_outgoing_link_is_scanned(self):
        result = nav.check(self.root)
        self.assertEqual(result['local_links'], 2)
        self.assertEqual(result['fragment_links'], 1)

    def test_missing_guide_target_is_rejected(self):
        self.guide.write_text(
            '# Guide\n[Missing](navigation-fixture-missing.md)\n',
            encoding='utf-8',
        )
        code, result = self.cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(result['problems']), 1)
        self.assertIn(GUIDE + ': navigation-fixture-missing.md:',
                      result['problems'][0])

    def test_missing_guide_anchor_is_rejected(self):
        self.guide.write_text(
            '# Guide\n[Missing](navigation-fixture-target.md#absent)\n',
            encoding='utf-8',
        )
        code, result = self.cli()
        self.assertEqual(code, 1)
        self.assertEqual(len(result['problems']), 1)
        self.assertIn(GUIDE + ':', result['problems'][0])
        self.assertIn('missing fragment absent', result['problems'][0])


if __name__ == '__main__':
    unittest.main()
