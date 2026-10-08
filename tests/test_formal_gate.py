"""Positive and negative controls for the main-side formal gate.

Synthetic controls build a disposable mini package so each mutation is
isolated. Repository controls run the real manifest in source-only mode, and —
when a Lean toolchain is on PATH or in ~/.elan — execute the real package
(build, leanchecker, axiom audit, negative controls, receipt).
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate", ROOT / "tools/formal_gate_check.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

TOOLCHAIN = "leanprover/lean4:v4.34.1"
LEAN = """namespace Demo

/-- Source: "two and two make four". A docstring mentioning sorry is harmless. -/
theorem two_add : 2 + 2 = 4 := by decide

theorem tight : 77 < 78 := by decide

end Demo
"""
SOURCE = "Section 1. In the source, two and two make four; also 77 < 78 here.\n"
AUDIT = "'Demo.two_add' does not depend on any axioms\n'Demo.tight' does not depend on any axioms\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SyntheticControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for d in ("pkg/Demo", "pkg/sources", "pkg/reviews", "tools", "tests"):
            (self.root / d).mkdir(parents=True)
        self.write("pkg/lean-toolchain", TOOLCHAIN + "\n")
        self.write("pkg/lakefile.toml", 'name = "Demo"\n[[lean_lib]]\nname = "Demo"\n')
        self.write("pkg/lake-manifest.json", '{"packages": []}\n')
        self.write("pkg/Demo.lean", "import Demo.Basic\n\n/-! root docstring -/\n")
        self.write("pkg/Demo/Basic.lean", LEAN)
        self.write("pkg/sources/SOURCE.md", SOURCE)
        for name in ("SCOPE.md", "GLOSSARY.md", "README.md"):
            self.write("pkg/" + name, name + "\n")
        self.write("tools/formal_gate_check.py", "# bound control copy\n")
        self.write("tests/test_formal_gate.py", "# bound control copy\n")
        self.bound = ["pkg/lean-toolchain", "pkg/lakefile.toml", "pkg/lake-manifest.json", "pkg/Demo.lean",
                      "pkg/Demo/Basic.lean", "pkg/sources/SOURCE.md", "pkg/SCOPE.md", "pkg/GLOSSARY.md",
                      "pkg/README.md", "tools/formal_gate_check.py", "tests/test_formal_gate.py"]
        self.manifest = {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "package": "demo", "package_root": "pkg", "root_module": "Demo",
            "formalization_status": "proved", "alignment_status": "PENDING_INDEPENDENT_REVIEW",
            "lean_toolchain": TOOLCHAIN, "dependency_revisions": {},
            "allowed_axioms": ["propext", "Classical.choice", "Quot.sound"],
            "files": {}, "unbound_files": ["pkg/manifest.json", "pkg/ALIGNMENT.md"],
            "sources": [{"id": "src", "repository": "d6g8k5htny-coder/Math-", "commit": "a" * 40, "path": "SOURCE.md",
                         "bytes": len(SOURCE.encode()), "sha256": sha(SOURCE.encode()), "local_copy": "pkg/sources/SOURCE.md"}],
            "source_modules": ["Demo/Basic.lean"],
            "targets": [
                {"name": "Demo.two_add", "module": "Demo/Basic.lean", "title": "two add", "source": "src",
                 "informal_anchor": "two and two make four", "does_not_claim": "anything else"},
                {"name": "Demo.tight", "module": "Demo/Basic.lean", "title": "tight", "source": "src",
                 "informal_anchor": "77 < 78", "does_not_claim": "anything else"},
            ],
            "negative_controls": {"tight": {"module": "Demo/Basic.lean", "replace": "77 < 78", "with": "77 < 77"}},
        }
        self.refresh()
        self.audit = self.root / "audit.txt"
        self.audit.write_text(AUDIT)

    def write(self, relative, text):
        (self.root / relative).write_text(text)

    def refresh(self):
        self.manifest["files"] = {p: sha((self.root / p).read_bytes()) for p in self.bound}
        self.save()

    def save(self):
        (self.root / "pkg/manifest.json").write_text(json.dumps(self.manifest))
        (self.root / "pkg/ALIGNMENT.md").write_text(gate.render_alignment(self.manifest))

    def check(self):
        return gate.source_check(self.root, "pkg/manifest.json")

    def assertInvalid(self, fragment):
        with self.assertRaisesRegex((ValueError, KeyError, TypeError), fragment):
            self.check()

    # positive controls

    def test_valid_source_check_and_no_writes(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        m, digest, pkg = self.check()
        self.assertEqual(digest, sha((self.root / "pkg/manifest.json").read_bytes()))
        self.assertEqual([t["name"] for t in m["targets"]], ["Demo.two_add", "Demo.tight"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_cli_source_only_reports_not_a_build(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = gate.main(["--root", str(self.root), "--manifest", "pkg/manifest.json",
                              "--alignment-table", "pkg/ALIGNMENT.md"])
        self.assertEqual(code, 0)
        self.assertIn("SOURCE_IDENTITY_PASS (not a Lean build", buf.getvalue())

    def test_audit_from_file_passes(self):
        gate.audit_axioms(self.audit.read_text(), ["Demo.two_add", "Demo.tight"])

    # manifest cannot self-award

    def test_manifest_cannot_claim_kernel_checked(self):
        self.manifest["formalization_status"] = "kernel-checked"
        self.save()
        self.assertInvalid("cannot self-award")

    def test_manifest_cannot_claim_alignment(self):
        self.manifest["alignment_status"] = "ACCEPTED"
        self.save()
        self.assertInvalid("cannot self-award")

    def test_status_authority_must_be_false(self):
        self.manifest["scientific_status_authority"] = True
        self.save()
        self.assertInvalid("scientific_status_authority")

    def test_duplicate_json_key(self):
        (self.root / "pkg/manifest.json").write_text('{"schema_version":1,"schema_version":1}')
        self.assertInvalid("duplicate JSON key")

    # file identity

    def test_hash_mismatch(self):
        self.write("pkg/Demo/Basic.lean", LEAN.replace("77 < 78", "77 < 79"))
        self.assertInvalid("source hash mismatch")

    def test_unregistered_lean_module(self):
        self.write("pkg/Demo/Extra.lean", "theorem extra : True := trivial\n")
        self.manifest["files"]["pkg/Demo/Extra.lean"] = sha(b"theorem extra : True := trivial\n")
        self.save()
        self.assertInvalid("unregistered Lean module")

    def test_unbound_package_file(self):
        self.write("pkg/notes.txt", "x\n")
        self.assertInvalid("not bound or declared unbound")

    def test_control_files_must_be_bound(self):
        del self.manifest["files"]["tools/formal_gate_check.py"]
        self.save()
        self.assertInvalid("unbound control or scope file")

    def test_scope_must_be_bound(self):
        del self.manifest["files"]["pkg/SCOPE.md"]
        self.save()
        self.assertInvalid("unbound control or scope file")

    def test_toolchain_mismatch(self):
        self.write("pkg/lean-toolchain", "leanprover/lean4:v4.0.0\n")
        self.refresh()
        self.assertInvalid("wrong Lean toolchain")

    def test_dependency_lock_mismatch(self):
        self.write("pkg/lake-manifest.json", '{"packages": [{"name": "mathlib", "rev": "' + "b" * 40 + '"}]}\n')
        self.refresh()
        self.assertInvalid("dependency lock mismatch")

    def test_root_must_import_exactly_registered_modules(self):
        self.write("pkg/Demo.lean", "import Demo.Basic\nimport Demo.Extra\n")
        self.refresh()
        self.assertInvalid("root must only import")

    # Lean text scan

    def test_sorry_in_code(self):
        self.write("pkg/Demo/Basic.lean", LEAN.replace("by decide\n\ntheorem tight", "by sorry\n\ntheorem tight"))
        self.refresh()
        self.assertInvalid("sorry in")

    def test_sorry_only_in_comments_ignored(self):
        self.write("pkg/Demo/Basic.lean", LEAN + "-- sorry native_decide axiom in a line comment\n/- nested /- sorry -/ still comment -/\n")
        self.refresh()
        self.check()

    def test_native_decide_refused(self):
        self.write("pkg/Demo/Basic.lean", LEAN.replace(":= by decide\n\ntheorem tight", ":= by native_decide\n\ntheorem tight"))
        self.refresh()
        self.assertInvalid("native_decide in")

    def test_custom_axiom_refused(self):
        self.write("pkg/Demo/Basic.lean", LEAN.replace("end Demo", "axiom parent : 1 = 1\n\nend Demo"))
        self.refresh()
        self.assertInvalid("custom axiom declaration")

    # target inventory and anchors

    def test_unregistered_theorem_refused(self):
        self.write("pkg/Demo/Basic.lean", LEAN.replace("end Demo", "theorem extra : 1 = 1 := rfl\n\nend Demo"))
        self.refresh()
        self.assertInvalid("target inventory differs")

    def test_target_order_must_match_source(self):
        self.manifest["targets"].reverse()
        self.save()
        self.assertInvalid("target inventory differs")

    def test_missing_theorem_refused(self):
        self.manifest["targets"].append({"name": "Demo.three_add", "module": "Demo/Basic.lean", "title": "t", "source": "src",
                                         "informal_anchor": "two and two", "does_not_claim": "x"})
        self.save()
        self.assertInvalid("target inventory differs")

    def test_anchor_not_verbatim(self):
        self.manifest["targets"][0]["informal_anchor"] = "two and two make five"
        self.save()
        self.assertInvalid("not a verbatim substring")

    def test_source_bytes_changed(self):
        self.write("pkg/sources/SOURCE.md", SOURCE + "x")
        self.manifest["files"]["pkg/sources/SOURCE.md"] = sha((SOURCE + "x").encode())
        self.save()
        self.assertInvalid("local copy differs from pinned bytes")

    def test_nonpublic_source_repository_refused(self):
        self.manifest["sources"][0]["repository"] = "someone/private"
        self.save()
        self.assertInvalid("nonpublic source repository")

    def test_does_not_claim_required(self):
        self.manifest["targets"][0]["does_not_claim"] = " "
        self.save()
        self.assertInvalid("does_not_claim must be nonempty")

    # axiom audit

    def test_audit_missing_target(self):
        with self.assertRaisesRegex(ValueError, "missing target axiom report"):
            gate.audit_axioms("'Demo.two_add' does not depend on any axioms\n", ["Demo.two_add", "Demo.tight"])

    def test_audit_sorry_axiom(self):
        with self.assertRaisesRegex(ValueError, "forbidden transitive axiom"):
            gate.audit_axioms(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [sorryAx]"), ["Demo.two_add", "Demo.tight"])

    def test_audit_native_axiom(self):
        with self.assertRaisesRegex(ValueError, "Lean.ofReduceBool"):
            gate.audit_axioms(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                            "'Demo.tight' depends on axioms: [Lean.ofReduceBool]"), ["Demo.two_add", "Demo.tight"])

    def test_audit_allowed_axioms_pass(self):
        records = gate.audit_axioms(AUDIT.replace("'Demo.tight' does not depend on any axioms",
                                                  "'Demo.tight' depends on axioms: [propext, Classical.choice, Quot.sound]"),
                                    ["Demo.two_add", "Demo.tight"])
        self.assertEqual(records["Demo.tight"], ["Classical.choice", "Quot.sound", "propext"])

    def test_audit_unrecognized_output_fails(self):
        with self.assertRaisesRegex(ValueError, "unrecognized audit output"):
            gate.audit_axioms(AUDIT + "Basic.lean:4:0: error: something\n", ["Demo.two_add", "Demo.tight"])

    def test_audit_unexpected_target_fails(self):
        with self.assertRaisesRegex(ValueError, "unexpected or duplicate"):
            gate.audit_axioms(AUDIT + "'Demo.extra' does not depend on any axioms\n", ["Demo.two_add", "Demo.tight"])

    # alignment record

    def review(self, **overrides):
        m, digest, _ = self.check()
        record = {"disposition": "ACCEPTED", "manifest_sha256": digest, "scope_sha256": m["files"]["pkg/SCOPE.md"],
                  "targets": ["Demo.two_add", "Demo.tight"],
                  "author": {"provider": "Anthropic", "family": "Claude", "agent": "cursor"},
                  "reviewer": {"provider": "OpenAI", "family": "GPT", "agent": "codex"},
                  "evidence": {"repository": "d6g8k5htny-coder/main", "commit": "c" * 40,
                               "path": "formal/reviews/x.md", "sha256": "d" * 64}}
        record.update(overrides)
        return record, digest, m

    def test_alignment_valid(self):
        record, digest, m = self.review()
        gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    def test_alignment_same_provider_refused(self):
        record, digest, m = self.review(reviewer={"provider": "anthropic", "family": "Other", "agent": "x"})
        with self.assertRaisesRegex(ValueError, "lineage not independent: provider"):
            gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    def test_alignment_stale_manifest_refused(self):
        record, digest, m = self.review(manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "stale alignment manifest"):
            gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    def test_alignment_partial_coverage_refused(self):
        record, digest, m = self.review(targets=["Demo.two_add"])
        with self.assertRaisesRegex(ValueError, "partial or ambiguous"):
            gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    def test_alignment_not_accepted_refused(self):
        record, digest, m = self.review(disposition="ACCEPTED_AT_NARROWER_SCOPE")
        with self.assertRaisesRegex(ValueError, "alignment not accepted"):
            gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    def test_alignment_mutable_ref_refused(self):
        record, digest, m = self.review(evidence={"repository": "d6g8k5htny-coder/main", "commit": "main",
                                                  "path": "formal/reviews/x.md", "sha256": "d" * 64})
        with self.assertRaisesRegex(ValueError, "mutable review ref"):
            gate.check_alignment(record, digest, ["Demo.two_add", "Demo.tight"], m["files"]["pkg/SCOPE.md"])

    # alignment table drift

    def test_alignment_table_drift_fails_cli(self):
        (self.root / "pkg/ALIGNMENT.md").write_text("stale\n")
        with contextlib.redirect_stdout(io.StringIO()):
            code = gate.main(["--root", str(self.root), "--manifest", "pkg/manifest.json", "--alignment-table", "pkg/ALIGNMENT.md"])
        self.assertEqual(code, 1)


    # Ordering cases derive from the published, source-bound diagnostic:
    # https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6066278094
    # Only subprocess outputs and Git identity are synthetic. The real execute,
    # run/log writer, axiom parser and temporary-package source checker run here.
    RUNTIME_RELEASE = "Lean (version 4.34.1, x86_64-unknown-linux-gnu, commit abc, Release)\n"
    WRONG_RUNTIMES = (
        "Lean (version 4.34.10, x86_64, Release)",
        "Lean (version 4.34.100, x86_64, Release)",
        "Lean (version 4.34.1-rc1, x86_64, Release)",
        "Lean (version 4.34.1-nightly, x86_64, Release)",
        "unrelated diagnostic: version 4.34.1",
        "Lean (version 4.34.2, x86_64, Release)",
        "Lean (version 4.35.1, x86_64, Release)",
        "Lean (version 4.34.0, x86_64, Release)",
        "",
    )

    def runtime_probe(self, versions, *, version_returncode=0, version_timeout=False,
                      mutate_source=False):
        m, digest, pkg = self.check()
        shutil.rmtree(pkg / ".lake", ignore_errors=True)
        sentinel = pkg / ".lake/build/existing-build-sentinel"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("temporary build must survive preflight refusal\n")
        out = pkg / ".lake/formal-evidence"
        events = []
        self.runtime_state = {"events": events, "sentinel": sentinel, "out": out}
        version_outputs = iter(versions)
        source_check = gate.source_check

        def check_source(root):
            events.append("source-check")
            return source_check(root, "pkg/manifest.json")

        def run_process(command, **kwargs):
            if command == ["git", "rev-parse", "HEAD"]:
                events.append("git-head")
                return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n")
            self.assertEqual(command[0], "/synthetic/lake")
            self.assertEqual(kwargs["cwd"], pkg)
            if command[1:] == ["env", "lean", "--version"]:
                events.append("version")
                if version_timeout:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                return subprocess.CompletedProcess(command, version_returncode, next(version_outputs))
            if command[1:] == ["build"]:
                events.append("build")
                if mutate_source:
                    with (pkg / "sources/SOURCE.md").open("a") as f:
                        f.write("synthetic build changed a bound source\n")
                return subprocess.CompletedProcess(command, 0, "synthetic build\n")
            if command[1:] == ["env", "leanchecker", "Demo"]:
                events.append("leanchecker")
                return subprocess.CompletedProcess(command, 0, "synthetic replay\n")
            self.assertEqual(command[1:3], ["env", "lean"])
            name = Path(command[3]).name
            events.append(name)
            if name == "Audit.lean":
                return subprocess.CompletedProcess(command, 0, AUDIT)
            if name == "Types.lean":
                return subprocess.CompletedProcess(command, 0, "synthetic elaborated types\n")
            if name == "tight.lean":
                return subprocess.CompletedProcess(command, 1, "is false\n")
            if name in {"sorry.lean", "custom_imported.lean", "native.lean"}:
                return subprocess.CompletedProcess(command, 0, "'injected' depends on axioms: [sorryAx]\n")
            self.fail("unexpected synthetic process: " + repr(command))

        with mock.patch.object(gate, "lake_binary", return_value="/synthetic/lake"), \
                mock.patch.object(gate.subprocess, "run", side_effect=run_process), \
                mock.patch.object(gate, "source_check", side_effect=check_source), \
                contextlib.redirect_stdout(io.StringIO()):
            return gate.execute(m, digest, pkg, self.root)

    def test_wrong_runtime_preserves_build_and_refuses_before_work(self):
        for version in self.WRONG_RUNTIMES:
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "unexpected running Lean version"):
                    self.runtime_probe([version])
                self.assertTrue(self.runtime_state["sentinel"].is_file())
                self.assertEqual(self.runtime_state["events"], ["version"])
                self.assertFalse((self.runtime_state["out"] / "receipt.json").exists())
                self.assertEqual((self.runtime_state["out"] / "preflight-version.log").read_text(), version)

    def test_valid_runtime_preserves_both_version_observations_and_logs(self):
        final_version = "Lean (version 4.34.1, aarch64-unknown-linux-gnu, commit def, Release)\n"
        receipt = self.runtime_probe([self.RUNTIME_RELEASE, final_version])
        self.assertEqual(self.runtime_state["events"], [
            "version", "build", "leanchecker", "Audit.lean", "Types.lean", "tight.lean",
            "sorry.lean", "custom_imported.lean", "native.lean", "source-check", "version", "git-head"])
        self.assertFalse(self.runtime_state["sentinel"].exists())
        self.assertEqual(receipt["lean_version"], final_version.strip())
        out = self.runtime_state["out"]
        self.assertEqual((out / "preflight-version.log").read_text(), self.RUNTIME_RELEASE)
        self.assertEqual((out / "version.log").read_text(), final_version)
        self.assertEqual(set(receipt["logs"]), {
            "preflight-version.log", "version.log", "build.log", "leanchecker.log",
            "axioms.log", "elaborated-types.log", "tight.log", "sorry.log",
            "custom_imported.log", "native.log"})
        for name, digest in receipt["logs"].items():
            self.assertEqual(digest, sha((out / name).read_bytes()), name)

    def test_wrong_postflight_runtime_refuses_receipt_after_source_recheck(self):
        for version in self.WRONG_RUNTIMES:
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "unexpected running Lean version"):
                    self.runtime_probe([self.RUNTIME_RELEASE, version])
                self.assertEqual(self.runtime_state["events"][0], "version")
                self.assertEqual(self.runtime_state["events"][-2:], ["source-check", "version"])
                self.assertFalse((self.runtime_state["out"] / "receipt.json").exists())

    def test_postflight_source_change_still_refuses_receipt(self):
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.runtime_probe([self.RUNTIME_RELEASE], mutate_source=True)
        self.assertEqual(self.runtime_state["events"][0], "version")
        self.assertEqual(self.runtime_state["events"][-1], "source-check")
        self.assertEqual(self.runtime_state["events"].count("version"), 1)
        self.assertFalse((self.runtime_state["out"] / "receipt.json").exists())

    def test_preflight_process_failure_preserves_build(self):
        with self.assertRaisesRegex(ValueError, "unexpected process outcome"):
            self.runtime_probe([self.RUNTIME_RELEASE], version_returncode=17)
        self.assertTrue(self.runtime_state["sentinel"].is_file())
        self.assertEqual(self.runtime_state["events"], ["version"])
        self.assertFalse((self.runtime_state["out"] / "receipt.json").exists())

    def test_preflight_timeout_preserves_build(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.runtime_probe([], version_timeout=True)
        self.assertTrue(self.runtime_state["sentinel"].is_file())
        self.assertEqual(self.runtime_state["events"], ["version"])
        self.assertFalse((self.runtime_state["out"] / "receipt.json").exists())


class RepositoryControls(unittest.TestCase):
    def test_real_manifest_source_check(self):
        m, digest, pkg = gate.source_check(ROOT)
        self.assertEqual(m["formalization_status"], "proved")
        self.assertEqual(m["alignment_status"], "PENDING_INDEPENDENT_REVIEW")
        self.assertIs(m["scientific_status_authority"], False)
        self.assertEqual(len(m["targets"]), 31)
        self.assertEqual(m["dependency_revisions"], {})
        self.assertEqual((ROOT / "formal/ALIGNMENT.md").read_text(), gate.render_alignment(m))

    def test_real_manifest_cli_source_only(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(gate.main(["--root", str(ROOT)]), 0)
        self.assertIn("SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance)", buf.getvalue())

    def test_real_template_record_is_refused(self):
        m, digest, _ = gate.source_check(ROOT)
        template = json.loads((ROOT / "formal/reviews/TEMPLATE_alignment_review.json").read_text())
        with self.assertRaises(ValueError):
            gate.check_alignment(template, digest, [t["name"] for t in m["targets"]], m["files"]["formal/SCOPE.md"])

    @unittest.skipUnless(gate.lake_binary(), "Lean toolchain not installed")
    def test_real_package_executes_with_receipt(self):
        m, digest, pkg = gate.source_check(ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            receipt = gate.execute(m, digest, pkg, ROOT)
        self.assertEqual(receipt["formalization_status"], "kernel-checked")
        self.assertEqual(receipt["alignment_status"], "PENDING_INDEPENDENT_REVIEW")
        self.assertEqual(set(receipt["axioms"]), {t["name"] for t in m["targets"]})
        self.assertTrue(all(v == [] for v in receipt["axioms"].values()))
        controls = receipt["negative_controls"]
        self.assertEqual(controls["sorry"], "REJECTED_BY_AXIOM_GATE")
        self.assertEqual(controls["custom_imported"], "REJECTED_BY_AXIOM_GATE")
        self.assertEqual(controls["native"], "REJECTED_BY_AXIOM_GATE")
        for label in m["negative_controls"]:
            self.assertEqual(controls[label], "REJECTED_BY_LEAN", label)
        self.assertTrue((pkg / ".lake/formal-evidence/receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
