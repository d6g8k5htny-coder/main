"""Static contract for the navigation workflow's page and test path filters.

This reads the workflow's existing single-line, quoted flow-list syntax. It
is not a general YAML parser, a GitHub event replay, or a whole-workflow audit.
The basename matcher is limited to direct children of tests/: unlike a raw
fnmatch over the whole path, its '*' cannot consume a directory separator.
Declared-page coverage recognizes literal paths and the existing docs/**
pattern only; it deliberately does not implement general GitHub glob matching.
"""
from __future__ import annotations

import ast
import fnmatch
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github' / 'workflows' / 'navigation.yml'


def selected_test_path(path: str, patterns: list[str]) -> bool:
    """Evaluate only the workflow's single-directory tests/ filter subset."""
    directory, separator, name = path.partition('/')
    if directory != 'tests' or not separator or not name or '/' in name:
        return False
    return any(
        pattern.startswith('tests/')
        and '/' not in pattern[len('tests/'):]
        and fnmatch.fnmatchcase(name, pattern[len('tests/'):])
        for pattern in patterns
    )


class NavigationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding='utf-8')
        matches = re.findall(
            r'^  pull_request:\n    paths: (\[[^\n]*\])$',
            cls.workflow,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise AssertionError('expected one pull_request paths flow list')
        cls.paths = ast.literal_eval(matches[0])
        if not isinstance(cls.paths, list) or not all(
            isinstance(path, str) for path in cls.paths
        ):
            raise AssertionError('expected a list of quoted path strings')

    def test_filter_is_narrow_and_retains_existing_triggers(self):
        self.assertEqual(self.paths, [
            'README.md',
            'docs/**',
            'formal/README.md',
            'formal/GLOSSARY.md',
            'formal/REVIEW_LANE.md',
            'tools/navigation_check.py',
            'tests/test_navigation*.py',
            '.github/workflows/navigation.yml',
        ])

    def test_every_declared_navigation_page_has_a_trigger(self):
        manifest = json.loads((ROOT / 'docs' / 'NAVIGATION.json').read_text(
            encoding='utf-8'))
        pages = manifest['pages']
        self.assertIsInstance(pages, list)
        self.assertTrue(pages, 'declared-page coverage must not be empty')
        for page in pages:
            with self.subTest(page=page):
                self.assertIsInstance(page, str)
                self.assertTrue(
                    page in self.paths
                    or ('docs/**' in self.paths and page.startswith('docs/')),
                    f'declared navigation page lacks a trigger: {page}',
                )

    def test_original_navigation_test_is_selected(self):
        self.assertTrue(selected_test_path('tests/test_navigation.py', self.paths))

    def test_manifest_regression_file_is_selected(self):
        self.assertTrue(selected_test_path(
            'tests/test_navigation_manifest.py', self.paths))

    def test_this_trigger_regression_file_is_selected(self):
        self.assertTrue(selected_test_path(
            'tests/test_navigation_workflow.py', self.paths))

    def test_other_top_level_navigation_tests_are_selected(self):
        self.assertTrue(selected_test_path(
            'tests/test_navigation_links.py', self.paths))

    def test_nearby_unrelated_and_nested_paths_are_not_selected(self):
        for path in (
            'tests/test_unrelated.py',
            'tests/navigation.py',
            'tests/test_navigation.py.txt',
            'tests/nested/test_navigation_manifest.py',
            'other/tests/test_navigation_manifest.py',
            'tests/test_Navigation_manifest.py',
        ):
            with self.subTest(path=path):
                self.assertFalse(selected_test_path(path, self.paths))

    def test_both_existing_discovery_commands_remain(self):
        for flags, log_name in (
            ('-B -S', 'tests-normal.log'),
            ('-B -O -S', 'tests-optimized.log'),
        ):
            command = (
                f"python {flags} -m unittest discover -s tests -p 'test_*.py' -v "
                f"> navigation-evidence/{log_name} 2>&1"
            )
            with self.subTest(flags=flags):
                self.assertEqual(self.workflow.count(command), 1)


if __name__ == '__main__':
    unittest.main()
