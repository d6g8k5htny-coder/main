"""Skeptic #3 probes for CR-F8. INTAKE_MODE selects a tree dir beside this file:
e495 (fresh checker at e495837), main (fresh origin/main), patched_e495 (finder's
bundle applied), minpatch_e495 (csv branch only). No network."""
import importlib.util, os, sys, unittest
from pathlib import Path
MODE = os.environ.get('INTAKE_MODE', 'e495')
TREE = Path(__file__).resolve().parent / MODE
spec = importlib.util.spec_from_file_location('house_tests', TREE / 'tests/test_public_intake.py')
house = importlib.util.module_from_spec(spec); spec.loader.exec_module(house)
intake = house.intake
assert Path(intake.__file__).resolve() == (TREE / 'tools/public_intake_check.py').resolve()
PATCHED = 'patch' in MODE

class Probes(house.IntakeTests):
    def green(self):
        r = self.run_check(); self.assertEqual(r['lane'], 'incoming'); self.assertEqual(r['scientific_effect'], 'NONE'); return r
    def test_F8a_finder_bytes(self):
        self.raw['table.csv'] = b'name,value\n"=cmd|\' /C calc\'!A0",1\n@SUM(1+1),2\n-2+3+cmd|\' /C calc\'!A0,3\n'
        self.save()
        self.reject('formula') if PATCHED else self.green()
    def test_F8_numeric_control(self):
        self.raw['table.csv'] = b'n,bound,err\n1,-0.5,+1e-3\n2,-inf,"-1,5"\n3,-(1/2),.5\n'
        self.save(); self.green()
    def test_bypass_bom_prefixed_formula(self):
        # Excel strips a leading UTF-8 BOM and evaluates; finder's lstrip set does not include U+FEFF.
        self.raw['table.csv'] = '﻿=cmd|\' /C calc\'!A0,1\n'.encode()
        self.save(); self.green()   # green in EVERY mode = heuristic bypassed when patched
    def test_bypass_semicolon_delimited(self):
        # ;-delimited CSV (European Excel list separator): csv.reader default sees one field.
        self.raw['table.csv'] = b'n;v\n1;=cmd|\' /C calc\'!A0\n'
        self.save(); self.green()
    def test_bypass_tab_delimited(self):
        self.raw['table.csv'] = b'n\tv\n1\t=cmd|\' /C calc\'!A0\n'
        self.save(); self.green()
    def test_false_rejection_handle_column(self):
        # Legitimate reviewer/author handle column; patched checker refuses it.
        self.raw['reviewers.csv'] = b'role,handle\nauthor,@d6g8k5htny-coder\nreviewer,@someone\n'
        self.save()
        self.reject('formula') if PATCHED else self.green()

if __name__ == '__main__':
    print(f'[refute-probe] mode={MODE} checker={intake.__file__}', file=sys.stderr); unittest.main(defaultTest='Probes')
