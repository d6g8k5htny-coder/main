#!/usr/bin/env python3
"""Discoverable source-bound checks for the quantitative D2 note, not Lean verification."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKET = "reviews/d2_two_atom_floors_20261006"
WORKFLOW = ".github/workflows/d2-two-atom-floors.yml"
FILES = {
    PACKET + "/NOTE.md", PACKET + "/two_atom_bounds.py",
    PACKET + "/test_two_atom_bounds.py", "tests/test_d2_two_atom_floors.py", WORKFLOW,
}
CORE = "formal/ResearchFormalCoreR1/D2Schur.lean"
CORE_BLOB = "b95460c0d263a32ea274b347079cca6aaab3d2e9"
EXPECTED_CASES = {
    "delta_attainment": 16, "directional_floor": 384, "excluded_atom_case": 3,
    "gap_attainment": 3, "invalid_domain": 7, "mass_lower_bound": 2,
    "nonunit_mass": 1, "probability_law": 64, "scale_identity": 18,
    "subtractive_schur": 8, "two_square": 216,
    "wrong_extremum_witness": 1, "wrong_formula_witness": 2,
}
MUTANTS = {
    "M1": ("delta = two_square_floor(p*u, q*v, u, v)",
           "delta = two_square_floor(p, q, u, v)",
           {"test_delta_floor_attains_with_zero_remainder",
            "test_finite_probability_laws_and_all_d2_floors",
            "test_scaling_has_correct_degrees",
            "test_wrong_missing_factor_and_normalizer_fail_on_exact_witnesses"}),
    "M2": ("determinant = min(m*m*(gap+m*m), gap*(gap+4*m*m)/4)",
           "determinant = max(m*m*(gap+m*m), gap*(gap+4*m*m)/4)",
           {"test_finite_probability_laws_and_all_d2_floors"}),
    "M3": ("schur = min(gap, 2*m*m)", "schur = max(gap, 2*m*m)",
           {"test_finite_probability_laws_and_all_d2_floors"}),
    "M4": ("tau = min(delta, (delta+9*m*gap)/4)",
           "tau = max(delta, (delta+9*m*gap)/4)",
           {"test_endpoint_max_cannot_replace_lower_min",
            "test_finite_probability_laws_and_all_d2_floors"}),
}

def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()}

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result

class D2TwoAtomPacketTests(unittest.TestCase):
    def setUp(self):
        self.side = ROOT / PACKET
        manifest = self.side / "SOURCE_FILES.json"
        self.assertTrue(manifest.is_file(), "quantitative packet manifest absent")
        self.meta = json.loads(manifest.read_text(), object_pairs_hook=unique)
        self.assertEqual(set(self.meta["files"]), FILES)
        self.assertEqual(self.meta["scientific_effect"], "NONE")
        self.assertEqual({x.name for x in self.side.iterdir()},
                         {"NOTE.md", "two_atom_bounds.py", "test_two_atom_bounds.py",
                          "SOURCE_FILES.json"})
        for name in sorted(FILES):
            path = ROOT / name
            self.assertFalse(any((ROOT / Path(*Path(name).parts[:i])).is_symlink()
                                 for i in range(1, len(Path(name).parts)+1)))
            self.assertTrue(path.is_file(), "missing source: " + name)
            self.assertEqual(identity(path.read_bytes()), self.meta["files"][name], name)
        self.assertEqual(identity((ROOT/CORE).read_bytes())["git_blob"], CORE_BLOB)

    def run_suite(self, directory):
        flags = ["-B", "-S"] + (["-O"] if sys.flags.optimize else [])
        run = subprocess.run([sys.executable, *flags, str(directory/"test_two_atom_bounds.py")],
                             capture_output=True, text=True, timeout=60)
        payload = json.loads(run.stdout, object_pairs_hook=unique)
        self.assertEqual(set(payload), {"tests", "errors", "failures", "exact_cases"})
        self.assertEqual(payload["tests"], 12)
        self.assertEqual(payload["errors"], 0)
        self.assertIn("Ran 12 tests", run.stderr)
        self.assertNotIn("skipped", run.stderr.lower())
        self.assertNotIn("ERROR:", run.stderr)
        return run, payload

    def test_source_scope(self):
        self.assertIn("NOT a Lean proof", (self.side/"NOTE.md").read_text())
        self.assertEqual(self.meta["schema"], 1)
        self.assertEqual(self.meta["consumed_sources"][0]["git_blob"], CORE_BLOB)

    def test_baseline_exact_cases(self):
        run, payload = self.run_suite(self.side)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(payload["failures"], 0)
        self.assertEqual(payload["exact_cases"], EXPECTED_CASES)
        self.assertRegex(run.stderr, r"\nOK\s*$")

    def mutation(self, label):
        old, new, names = MUTANTS[label]
        with tempfile.TemporaryDirectory(prefix="d2-atom-mutant-") as temp:
            dest = Path(temp)
            raw = (self.side/"two_atom_bounds.py").read_text()
            self.assertEqual(raw.count(old), 1)
            (dest/"two_atom_bounds.py").write_text(raw.replace(old, new))
            shutil.copyfile(self.side/"test_two_atom_bounds.py", dest/"test_two_atom_bounds.py")
            run, payload = self.run_suite(dest)
            self.assertEqual(run.returncode, 1, run.stderr)
            found = re.findall(r"^FAIL: ([A-Za-z0-9_]+) \(", run.stderr, re.M)
            self.assertEqual(set(found), names)
            self.assertEqual(len(found), len(names))
            self.assertEqual(payload["failures"], len(names))

    def test_M1_atom_square_weights_required(self): self.mutation("M1")
    def test_M2_determinant_min_required(self): self.mutation("M2")
    def test_M3_schur_min_required(self): self.mutation("M3")
    def test_M4_tau_min_required(self): self.mutation("M4")

if __name__ == "__main__":
    unittest.main(verbosity=2)
