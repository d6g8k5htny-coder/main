"""Skeptic re-run of CR-F8 (.csv spreadsheet-formula cells). No network.

INTAKE_TREE=<dir with tools/public_intake_check.py and tests/test_public_intake.py>
INTAKE_MODE=current  -> asserts the checker returns green for formula-like cells
INTAKE_MODE=patched  -> asserts the F8-only patched checker refuses them
Tests suffixed _control must pass in both modes; tests suffixed _bypass and
_wrong_rejection characterize the finder's proposed heuristic.
The FakeAPI fixture is reused verbatim from the tree's tests/test_public_intake.py.
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

MODE = os.environ['INTAKE_MODE']
TREE = Path(os.environ['INTAKE_TREE'])
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec)
spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve(), intake.__file__

FINDER_CSV = b'name,value\n"=cmd|\' /C calc\'!A0",1\n@SUM(1+1),2\n-2+3+cmd|\' /C calc\'!A0,3\n'


class Probe(unittest.TestCase):
    setUp = house.IntakeTests.setUp
    save = house.IntakeTests.save
    install = house.IntakeTests.install
    run_check = house.IntakeTests.run_check
    reject = house.IntakeTests.reject
    if hasattr(house.IntakeTests, 'install_source'):
        install_source = house.IntakeTests.install_source

    def green(self):
        result = self.run_check()
        self.assertEqual(result['lane'], 'incoming')
        self.assertEqual(result['scientific_effect'], 'NONE')
        return result

    def green_now_refused(self, message):
        if MODE == 'current':
            return self.green()
        self.reject(message)

    # --- the finder's claim, re-run on a fresh copy ---
    def test_F8a_finder_payload(self):
        self.raw['table.csv'] = FINDER_CSV
        self.save()
        self.green_now_refused('formula')

    def test_F8_numeric_control(self):
        self.raw['table.csv'] = b'n,bound,err\n1,-0.5,+1e-3\n2,-inf,"-1,5"\n3,-(1/2),.5\n'
        self.save()
        self.green()

    # --- the proposed heuristic is bypassable: green in BOTH modes ---
    def test_bom_prefixed_formula_bypass(self):
        # Excel/LibreOffice consume a UTF-8 BOM as an encoding mark; the patch's lstrip does not.
        self.raw['table.csv'] = '﻿=cmd|\' /C calc\'!A0,1\n'.encode('utf-8')
        self.save()
        self.green()

    def test_semicolon_locale_formula_bypass(self):
        # EU-locale Excel splits on ';'. csv.reader (comma) sees one field starting with 'x'.
        self.raw['table.csv'] = b'x;=cmd|\' /C calc\'!A0\n'
        self.save()
        self.green()

    def test_formula_in_txt_and_md_not_covered(self):
        # A reviewer may open a .txt as delimited text in a spreadsheet; the patch scans only .csv.
        self.raw['table.txt'] = FINDER_CSV
        self.raw['notes.md'] = b'| a | b |\n|---|---|\n| =cmd|\' /C calc\'!A0 | 1 |\n'
        self.save()
        self.green()

    # --- the proposed heuristic wrongly refuses legitimate mathematical CSV ---
    def test_relation_column_wrong_rejection(self):
        self.raw['table.csv'] = b'lhs,relation,rhs\n1,=,1\n2,<,3\n'
        self.save()
        if MODE == 'current':
            self.green()
        else:
            self.reject('formula')

    def test_handle_cell_wrong_rejection(self):
        self.raw['table.csv'] = b'reviewer,verdict\n@alice,ok\n'
        self.save()
        if MODE == 'current':
            self.green()
        else:
            self.reject('formula')


if __name__ == '__main__':
    print(f'[probe] mode={MODE} checker={intake.__file__}', file=sys.stderr)
    unittest.main()
