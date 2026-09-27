"""Positive and negative controls for the formalization gate.

Synthetic controls use a disposable mini project so that each mutation is
isolated. Repository controls run the real registry statically, and — when a
Lean toolchain is on PATH or in ~/.elan — build the real library, run the
axiom audit, and confirm that tightened inequalities are rejected by Lean.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate", ROOT / "tools/formal_gate_check.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

TOOLCHAIN = "leanprover/lean4:v4.34.1"
LEAN = """namespace Demo

/-- Source: "two and two make four". A comment mentioning sorry is harmless. -/
theorem two_add : 2 + 2 = 4 := by decide

theorem tight : 77 < 78 := by decide

end Demo
"""
SOURCE = "Section 1. In the source, two and two make four; also 77 < 78 here.\n"
AUDIT = ("'Demo.two_add' does not depend on any axioms\n"
         "'Demo.tight' does not depend on any axioms\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SyntheticControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "formal/Demo").mkdir(parents=True)
        (self.root / "formal/sources").mkdir()
        (self.root / "formal/reviews").mkdir()
        self.write("formal/lean-toolchain", TOOLCHAIN + "\n")
        self.write("formal/lakefile.toml", 'name = "Demo"\n[[lean_lib]]\nname = "Demo"\n')
        self.write("formal/Demo.lean", "import Demo.Basic\n")
        self.write("formal/Demo/Basic.lean", LEAN)
        self.write("formal/Demo/Audit.lean", "import Demo\n#print axioms Demo.two_add\n#print axioms Demo.tight\n")
        self.write("formal/sources/SOURCE.md", SOURCE)
        self.audit = self.root / "audit.txt"
        self.audit.write_text(AUDIT)
        self.registry = {
            "schema": 1, "scope": "FORMALIZATION_STATUS_ONLY", "scientific_status_authority": False,
            "meaning": "test", "lean_toolchain": TOOLCHAIN, "project_root": "formal",
            "audit_module": "formal/Demo/Audit.lean",
            "allowed_axioms": ["propext", "Classical.choice", "Quot.sound"],
            "forbidden_axioms": ["sorryAx", "Lean.ofReduceBool"],
            "status_levels": {"none": "", "specified": "", "proved": "", "kernel-checked": ""},
            "alignment_review_levels": {"open": "", "reviewed": "", "not-applicable": ""},
            "project_axioms": [],
            "lean_files": [],
            "sources": [{"id": "src", "repository": "d6g8k5htny-coder/Math-", "commit": "a" * 40,
                         "path": "SOURCE.md", "bytes": len(SOURCE.encode()), "sha256": sha(SOURCE.encode()),
                         "local_copy": "formal/sources/SOURCE.md"}],
            "cross_checks": [],
            "claims": [
                {"id": "two-add", "layer0_object": "Demo", "title": "two add", "status": "kernel-checked",
                 "lean_declaration": "Demo.two_add", "lean_file": "formal/Demo/Basic.lean", "source": "src",
                 "informal_anchor": "two and two make four", "does_not_claim": "anything else",
                 "alignment_review": {"status": "open", "author": "author", "reviewer": None, "record": None}},
                {"id": "tight", "layer0_object": "Demo", "title": "tight", "status": "kernel-checked",
                 "lean_declaration": "Demo.tight", "lean_file": "formal/Demo/Basic.lean", "source": "src",
                 "informal_anchor": "77 < 78", "does_not_claim": "anything else",
                 "alignment_review": {"status": "open", "author": "author", "reviewer": None, "record": None}},
                {"id": "big-theorem", "layer0_object": "Demo", "title": "big", "status": "none",
                 "lean_declaration": None, "lean_file": None, "source": None, "informal_anchor": None,
                 "does_not_claim": "any formal verification",
                 "alignment_review": {"status": "not-applicable", "author": None, "reviewer": None, "record": None}},
            ],
        }
        self.refresh_files()
        (self.root / "formal/ALIGNMENT.md").write_text(gate.render_alignment(self.registry))

    def write(self, relative, text):
        (self.root / relative).write_text(text)

    def refresh_files(self):
        self.registry["lean_files"] = [
            {"path": p, "sha256": sha((self.root / p).read_bytes())}
            for p in ["formal/lean-toolchain", "formal/lakefile.toml", "formal/Demo.lean",
                      "formal/Demo/Basic.lean", "formal/Demo/Audit.lean"]]
        self.save()

    def save(self):
        (self.root / "formal/registry.json").write_text(json.dumps(self.registry))

    def run_check(self, **kwargs):
        kwargs.setdefault("axioms_output", self.audit)
        return gate.check(self.root, **kwargs)

    def assertProblem(self, fragment, **kwargs):
        problems = self.run_check(**kwargs)["problems"]
        self.assertTrue(any(fragment in p for p in problems), problems)

    def assertInvalid(self, fragment, **kwargs):
        with self.assertRaisesRegex((ValueError, OSError), fragment):
            self.run_check(**kwargs)

    # positive control

    def test_valid_and_no_writes(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.run_check(alignment="formal/ALIGNMENT.md")
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["status_counts"]["kernel-checked"], 2)
        self.assertEqual(result["status_counts"]["none"], 1)
        self.assertEqual(result["axiom_audit"]["declarations_required"], 2)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_static_only_is_reported_not_silent(self):
        result = self.run_check(axioms_output=None, static_only=True)
        self.assertEqual(result["problems"], [])
        self.assertIn("NOT RUN", result["axiom_audit"])

    def test_axiom_lane_required_by_default(self):
        self.assertInvalid("axiom lane required", axioms_output=None)

    # Lean file identity and content

    def test_lean_hash_mismatch(self):
        self.write("formal/Demo/Basic.lean", LEAN.replace("77 < 78", "77 < 79"))
        self.assertProblem("sha256 differs")

    def test_unregistered_lean_file(self):
        self.write("formal/Demo/Extra.lean", "theorem extra : True := trivial\n")
        self.assertProblem("present but not registered")

    def test_sorry_in_code(self):
        self.write("formal/Demo/Basic.lean", LEAN.replace("by decide\n\ntheorem tight", "by sorry\n\ntheorem tight"))
        self.refresh_files()
        self.assertProblem("contains sorry")

    def test_sorry_only_in_comment_is_ignored(self):
        self.write("formal/Demo/Basic.lean", LEAN + "-- sorry, native_decide and axiom words in a comment\n/- nested /- sorry -/ still comment -/\n")
        self.refresh_files()
        self.assertEqual(self.run_check()["problems"], [])

    def test_native_decide_refused(self):
        self.write("formal/Demo/Basic.lean", LEAN.replace(":= by decide\n\ntheorem tight", ":= by native_decide\n\ntheorem tight"))
        self.refresh_files()
        self.assertProblem("native_decide")

    def test_unregistered_axiom_declaration(self):
        self.write("formal/Demo/Basic.lean", LEAN + "axiom parent_theorem : 1 = 1\n")
        self.refresh_files()
        self.assertProblem("axiom parent_theorem is not registered")

    def test_registered_axiom_with_scope_note_accepted(self):
        self.write("formal/Demo/Basic.lean", LEAN + "axiom parent_theorem : 1 = 1\n")
        self.registry["project_axioms"] = [{"name": "parent_theorem", "scope_note": "assumed, not verified"}]
        self.refresh_files()
        self.assertEqual(self.run_check()["problems"], [])

    def test_project_axiom_needs_scope_note(self):
        self.registry["project_axioms"] = [{"name": "parent_theorem", "scope_note": ""}]
        self.save()
        self.assertInvalid("scope_note")

    def test_toolchain_mismatch(self):
        self.write("formal/lean-toolchain", "leanprover/lean4:v4.0.0\n")
        self.refresh_files()
        self.assertProblem("lean-toolchain file differs")

    def test_toolchain_must_be_listed(self):
        self.registry["lean_files"] = [f for f in self.registry["lean_files"] if not f["path"].endswith("lean-toolchain")]
        self.save()
        self.assertInvalid("lean-toolchain must be a listed")

    # sources and anchors

    def test_source_bytes_changed(self):
        self.write("formal/sources/SOURCE.md", SOURCE + "x")
        self.assertProblem("differs from pinned bytes")

    def test_anchor_not_verbatim(self):
        self.registry["claims"][0]["informal_anchor"] = "two and two make five"
        self.save()
        self.assertProblem("not a verbatim substring")

    def test_nonpublic_source_repository_refused(self):
        self.registry["sources"][0]["repository"] = "someone/private"
        self.save()
        self.assertInvalid("nonpublic source repository")

    def test_bad_commit_refused(self):
        self.registry["sources"][0]["commit"] = "abc"
        self.save()
        self.assertInvalid("invalid commit")

    # claims

    def test_declaration_missing_from_file(self):
        self.registry["claims"][0]["lean_declaration"] = "Demo.three_add"
        self.save()
        self.assertProblem("three_add not found")

    def test_declaration_in_wrong_namespace(self):
        self.registry["claims"][0]["lean_declaration"] = "Other.two_add"
        self.save()
        self.assertProblem("not found")

    def test_none_status_cannot_carry_declaration(self):
        self.registry["claims"][2]["lean_declaration"] = "Demo.two_add"
        self.save()
        self.assertInvalid("status none cannot carry")

    def test_unknown_status(self):
        self.registry["claims"][0]["status"] = "verified"
        self.save()
        self.assertInvalid("unknown status")

    def test_duplicate_claim_id(self):
        self.registry["claims"][1]["id"] = "two-add"
        self.save()
        self.assertInvalid("duplicate claim id")

    def test_duplicate_json_key(self):
        (self.root / "formal/registry.json").write_text('{"schema":1,"schema":1}')
        self.assertInvalid("duplicate JSON key")

    def test_status_authority_must_be_false(self):
        self.registry["scientific_status_authority"] = True
        self.save()
        self.assertInvalid("scientific_status_authority")

    def test_reviewed_alignment_needs_record_file(self):
        self.registry["claims"][0]["alignment_review"] = {"status": "reviewed", "author": "a", "reviewer": "r",
                                                          "record": "formal/reviews/missing.md"}
        self.save()
        self.assertInvalid("missing or outside-root")

    def test_reviewed_alignment_with_record_accepted(self):
        self.write("formal/reviews/two_add.md", "reviewed\n")
        self.registry["claims"][0]["alignment_review"] = {"status": "reviewed", "author": "a", "reviewer": "r",
                                                          "record": "formal/reviews/two_add.md"}
        self.save()
        self.assertEqual(self.run_check()["problems"], [])

    def test_does_not_claim_required(self):
        self.registry["claims"][0]["does_not_claim"] = " "
        self.save()
        self.assertInvalid("does_not_claim")

    # axiom audit

    def test_missing_from_audit(self):
        self.audit.write_text("'Demo.two_add' does not depend on any axioms\n")
        self.assertProblem("Demo.tight missing from axiom audit")

    def test_sorry_axiom_in_audit(self):
        self.audit.write_text(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [sorryAx]"))
        problems = self.run_check()["problems"]
        self.assertTrue(any("forbidden axioms ['sorryAx']" in p for p in problems), problems)

    def test_native_decide_axiom_in_audit(self):
        self.audit.write_text(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [Lean.ofReduceBool]"))
        self.assertProblem("Lean.ofReduceBool")

    def test_allowed_axioms_pass_for_kernel_checked(self):
        self.audit.write_text(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [propext, Classical.choice, Quot.sound]"))
        self.assertEqual(self.run_check()["problems"], [])

    def test_unlisted_axiom_fails_kernel_checked_but_not_proved(self):
        self.audit.write_text(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [Demo.parent]"))
        self.assertProblem("non-allowed axioms ['Demo.parent']")
        self.registry["claims"][1]["status"] = "proved"
        self.save()
        self.assertEqual(self.run_check()["problems"], [])

    def test_audit_error_line_fails(self):
        self.audit.write_text(AUDIT + "Basic.lean:4:0: error: something\n")
        self.assertProblem("reports an error")

    # alignment table

    def test_alignment_table_drift(self):
        (self.root / "formal/ALIGNMENT.md").write_text("stale\n")
        self.assertProblem("differs from the registry rendering", alignment="formal/ALIGNMENT.md")


class RepositoryControls(unittest.TestCase):
    def test_real_registry_static(self):
        result = gate.check(ROOT, static_only=True, alignment="formal/ALIGNMENT.md")
        self.assertEqual(result["problems"], [])
        self.assertGreaterEqual(result["status_counts"]["kernel-checked"], 1)
        self.assertEqual(result["status_counts"]["proved"], 0)

    def test_real_registry_has_no_status_authority(self):
        data = json.loads((ROOT / "formal/registry.json").read_text())
        self.assertIs(data["scientific_status_authority"], False)
        for claim in data["claims"]:
            self.assertTrue(claim["does_not_claim"].strip())

    @unittest.skipUnless(gate.lake_binary(), "Lean toolchain not installed")
    def test_real_library_builds_and_audit_passes(self):
        result = gate.check(ROOT, run=True, alignment="formal/ALIGNMENT.md")
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["axiom_audit"]["declarations_reported"], result["axiom_audit"]["declarations_required"])

    @unittest.skipUnless(gate.lake_binary(), "Lean toolchain not installed")
    def test_lean_rejects_tightened_bounds(self):
        lake = gate.lake_binary()
        project = ROOT / "formal"
        env = dict(os.environ, PATH=f"{Path(lake).parent}{os.pathsep}{os.environ.get('PATH', '')}")
        ledger = (project / "UniversalLaw/Side24/Ledger.lean").read_text()
        mutants = {
            "exponent d=3 tight": ("sixB d < 78", "sixB d < 77"),
            "covariance bound": ("60 * imageConstant < 10 ^ 17", "60 * imageConstant < 10 ^ 15"),
            "exp partial sum": ("10 * (125 ^ 20 * fact 20) < expTaylorNumerator", "11 * (125 ^ 20 * fact 20) < expTaylorNumerator"),
            "image constant": ("= 21175738586478", "= 21175738586479"),
            "eigenvalue third": ("58 * 9 < 23 ^ 2", "58 * 9 < 22 ^ 2"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (old, new) in mutants.items():
                self.assertIn(old, ledger, name)
                mutant = Path(tmp) / "Mutant.lean"
                mutant.write_text(ledger.replace(old, new))
                run = subprocess.run([lake, "env", "lean", str(mutant)], cwd=project, capture_output=True, text=True, env=env)
                self.assertNotEqual(run.returncode, 0, f"{name}: Lean accepted a false statement")
                self.assertIn("is false", run.stdout + run.stderr, name)
            control = Path(tmp) / "Control.lean"
            control.write_text(ledger)
            run = subprocess.run([lake, "env", "lean", str(control)], cwd=project, capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
