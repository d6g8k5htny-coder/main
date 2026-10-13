"""Skeptic probe for CR-F8 (CSV spreadsheet-formula cells). No network.

INTAKE_MODE=current  -> checker copy from origin/main (dbb9dcf); the F8a probe asserts
                        the green result the finder reports.
INTAKE_MODE=patched  -> the same checker with only the F8 part of the finder's patch
                        applied; F8a asserts refusal.
Control probes (suffix _control) must be green in BOTH modes.
Wrong-rejection probes (prefix test_WR_) are green on current and demonstrate that the
patch refuses legitimate CSV content (fail-closed of a legitimate package = MINOR).
Bypass probes (prefix test_BY_) are green in BOTH modes: content a spreadsheet would
still evaluate that the patch's heuristic does not catch.
The fixture is reused verbatim from tests/test_public_intake.py.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ.get('INTAKE_MODE', 'current')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()

DDE = b'=cmd|\' /C calc\'!A0'


class F8Probes(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    install_source = house.IntakeTests.install_source
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def wrong_green_now_refused(self, message):
        if MODE == 'current':
            return self.green()
        self.reject(message)

    def green_now_wrongly_refused(self, message):
        """A legitimate package: green on current, refused by the patch."""
        if MODE == 'current':
            return self.green()
        self.reject(message)

    # --- reproduction of the finder's F8a and control ---------------------------------
    def test_F8a_csv_formula_injection(self):
        self.raw['table.csv'] = b'name,value\n"' + DDE + b'",1\n@SUM(1+1),2\n-2+3+cmd|\' /C calc\'!A0,3\n'
        self.save()
        self.wrong_green_now_refused('formula')

    def test_F8_numeric_csv_control(self):
        self.raw['table.csv'] = b'n,bound,err\n1,-0.5,+1e-3\n2,-inf,"-1,5"\n3,-(1/2),.5\n'
        self.save()
        self.green()

    # --- the same bytes in a .txt or .md artifact are not covered by the patch at all ---
    def test_F8_same_cells_in_txt_and_md_pass_both_modes(self):
        self.raw['table.txt'] = b'name,value\n' + DDE + b',1\n'
        self.raw['notes.md'] = b'| name | value |\n|---|---|\n| ' + DDE + b' | 1 |\n'
        self.save()
        self.green()

    # --- wrong rejections the patch introduces ------------------------------------------
    def test_WR_github_handle_column(self):
        # A claim/reviewer table listing GitHub handles, the coordination convention CONTRIBUTING uses.
        self.raw['claims.csv'.replace('claims', 'reviewers')] = b'task,claimant,reviewer\n141,@visitor,@d6g8k5htny-coder\n'
        self.save()
        self.green_now_wrongly_refused('formula')

    def test_WR_relation_symbol_column(self):
        # An inequality table whose relation column holds '=' as a symbol.
        self.raw['bounds.csv'] = b'lhs,relation,rhs\n0.5,<=,0.51\n0.5,=,0.5\n'
        self.save()
        self.green_now_wrongly_refused('formula')

    def test_WR_leading_apostrophe_text_cell(self):
        # A leading apostrophe is the spreadsheet convention that FORCES text; the patch strips it and refuses.
        self.raw['labels.csv'] = b"id,label\n1,'=x\n"
        self.save()
        self.green_now_wrongly_refused('formula')

    def test_WR_classic_mac_cr_line_endings(self):
        # csv.reader raises csv.Error on a bare CR in an unquoted field; the patch maps it to a refusal.
        self.raw['old.csv'] = b'a,b\r1,2\r'
        self.save()
        self.green_now_wrongly_refused('unparseable CSV')

    def test_WR_single_field_over_csv_field_size_limit(self):
        # One quoted cell of 140000 bytes (< 256 KiB file limit, > csv.field_size_limit() default 131072).
        self.raw['long.csv'] = b'id,text\n1,"' + b'x' * 140000 + b'"\n'
        self.save()
        self.green_now_wrongly_refused('unparseable CSV')

    # --- content a spreadsheet still evaluates that the patch does not catch -------------
    def test_BY_plus_hyperlink_without_pipe(self):
        self.raw['t.csv'] = b'name,value\n+HYPERLINK("http://example.invalid";"click"),1\n'
        self.save()
        self.green()

    def test_BY_semicolon_delimited_locale(self):
        self.raw['t.csv'] = b'name;value\nx;' + DDE + b'\n'
        self.save()
        self.green()

    def test_BY_bom_prefixed_first_cell(self):
        self.raw['t.csv'] = '﻿'.encode() + DDE + b',1\n'
        self.save()
        self.green()


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
