"""External stdin contract controls for a conservative evidence adapter.

Every graph, source, receipt, Lean-looking text and log below is a synthetic TEST
fixture. Nothing was elaborated by Lean or authenticated by Git/GitHub. Hash and
byte agreement is packet integrity only; custody is always unknown in v1.
Execute this module only in the hosted ephemeral runner, not on the laptop.
"""

import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools/architecture_evidence_adapter.py"
REPO = "d6g8k5htny-coder/main"
HEAD = "a" * 40
BASE = "b" * 40
AXES = {"source", "review", "kernel", "computation", "alignment"}
STATES = {"recorded", "not_recorded", "unknown", "not_applicable"}
APPLICABILITY = {"current", "stale", "unknown", "not_applicable"}
ALPHA = "TEST alpha statement: λ\n".encode("utf-8")
PROOF = b"TEST proof old\n"
BETA = b"TEST beta statement\n"
SOURCE = ALPHA + PROOF + BETA
TARGETS = ["TEST.Formal.alpha", "TEST.Formal.beta"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(document):
    return (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def raw_bytes(capture):
    return base64.b64decode(capture["raw_base64"], validate=True)


def document(capture):
    return json.loads(raw_bytes(capture))


def git_capture(raw, path, commit=HEAD, repository=REPO):
    return {"ref": {
        "repository": repository, "commit": commit, "path": path,
        "git_blob": hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(),
        "sha256": sha(raw), "bytes": len(raw),
    }, "raw_base64": base64.b64encode(raw).decode("ascii")}


def replace_git(capture, raw):
    ref = capture["ref"]
    return git_capture(raw, ref["path"], ref["commit"], ref["repository"])


def run_capture(raw, member):
    return {"ref": {
        "repository": REPO, "checked_commit": HEAD, "run_id": "123", "run_attempt": "1",
        "artifact_id": "456", "member_path": "formal-evidence/" + member,
        "sha256": sha(raw), "bytes": len(raw),
    }, "raw_base64": base64.b64encode(raw).decode("ascii")}


def replace_run(capture, raw):
    updated = copy.deepcopy(capture)
    updated["ref"].update({"sha256": sha(raw), "bytes": len(raw)})
    updated["raw_base64"] = base64.b64encode(raw).decode("ascii")
    return updated


def slice_of(raw, start, end):
    return {"start_byte": start, "end_byte": end, "sha256": sha(raw[start:end])}


class ArchitectureEvidenceAdapterContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cwd = Path(temporary.name)
        self.expected = {"repository": REPO, "commit": HEAD, "run-id": "123", "run-attempt": "1"}
        graph = {
            "schema_version": 1, "object": "TEST-ONLY-EVIDENCE-GRAPH",
            "nodes": {
                "TEST.A": {"classification": "PROVED_REVIEWED", "controlling": False,
                           "source": "TEST/source.md", "review_issue": 65, "review_disposition": "ACCEPTED"},
                "TEST.B": {"classification": "PROVED_REVIEWED", "controlling": False,
                           "notes": "TEST.Formal.alpha resembles this title; this is no target mapping",
                           "review_disposition": "ACCEPTED"},
                "TEST.C": {"classification": "AUTHOR_SIDE_CANDIDATE", "controlling": False},
            },
            "edges": [
                {"from": "TEST.B", "to": "TEST.A", "required": True, "relation": "TEST premise"},
                {"from": "TEST.C", "to": "TEST.B", "required": False, "relation": "TEST context"},
            ],
        }
        snapshots = {}
        for label, commit in (("old", BASE), ("new", HEAD)):
            source = git_capture(SOURCE, "TEST/source.md", commit)
            snapshots[label] = {
                "graph": git_capture(encoded(graph), "TEST/graph.json", commit),
                "bindings": [{"node": "TEST.A", "source": source,
                              "statement_slice": slice_of(SOURCE, 0, len(ALPHA)),
                              "proof_slice": slice_of(SOURCE, len(ALPHA), len(ALPHA) + len(PROOF))}],
            }
        self.packet = {"schema_version": 1, **snapshots, "formal_records": [], "retrofit_records": []}

    def change_graph(self, packet, label, change):
        capture = packet[label]["graph"]
        graph = document(capture)
        change(graph)
        packet[label]["graph"] = replace_git(capture, encoded(graph))

    def change_source(self, packet, label, raw, *, sliced=True):
        binding = packet[label]["bindings"][0]
        binding["source"] = replace_git(binding["source"], raw)
        if sliced:
            binding["statement_slice"] = slice_of(raw, 0, len(ALPHA))
            binding["proof_slice"] = slice_of(raw, len(ALPHA), len(ALPHA) + len(PROOF))
        else:
            binding["statement_slice"] = binding["proof_slice"] = None

    def formal(self, packet=None, *, source_snapshot="new"):
        packet = self.packet if packet is None else packet
        source = copy.deepcopy(packet[source_snapshot]["bindings"][0]["source"])
        local_copy = git_capture(raw_bytes(source), "TEST/formal/sources/source.md")
        scope = git_capture(b"TEST scope: two declared True targets; no elaboration or scientific claim\n", "TEST/formal/SCOPE.md")
        module = git_capture(b"namespace TEST.Formal\ntheorem alpha : True := by trivial\ntheorem beta : True := by trivial\n", "TEST/formal/Demo.lean")
        files = [local_copy, scope, module]
        # Synthetic source-check floor; these bytes are never executed by the adapter.
        files.extend(git_capture(raw, path) for path, raw in (
            ("TEST/formal/lean-toolchain", b"leanprover/lean4:v4.34.1\n"),
            ("TEST/formal/lakefile.toml", b'name = "TESTOnly"\nversion = "0.0.0"\n'),
            ("TEST/formal/lake-manifest.json", b'{"packages": []}\n'),
            ("TEST/formal/TEST.Formal.lean", b"import Demo\n"),
            ("TEST/formal/GLOSSARY.md", b"TEST synthetic glossary; no semantic acceptance\n"),
            ("TEST/formal/README.md", b"TEST synthetic package; no executed Lean evidence\n"),
            ("tools/formal_gate_check.py", b"# TEST synthetic bound control; not executed\n"),
            ("tests/test_formal_gate.py", b"# TEST synthetic bound tests; not executed\n"),
        ))
        source_ref = source["ref"]
        manifest = {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "package": "TEST-synthetic-package", "package_root": "TEST/formal", "root_module": "TEST.Formal",
            "formalization_status": "proved", "alignment_status": "PENDING_INDEPENDENT_REVIEW",
            "meaning": "TEST synthetic source metadata; no executed or authenticated evidence",
            "coordination": {"guide": "TEST/guide.md", "work_item": "TEST fixture only"},
            "author": {"provider": "TEST-author", "family": "TEST-author-family", "agent": "TEST-author-agent"},
            "lean_toolchain": "leanprover/lean4:v4.34.1", "dependency_revisions": {}, "allowed_axioms": [],
            "unbound_files": ["TEST/formal/manifest.json", "TEST/formal/ALIGNMENT.md"],
            "negative_controls": {"TEST_false_claim": {"module": "Demo.lean", "replace": "True", "with": "False"}},
            "source_modules": ["Demo.lean"], "files": {item["ref"]["path"]: item["ref"]["sha256"] for item in files},
            "sources": [{"id": "TEST-source", "repository": source_ref["repository"], "commit": source_ref["commit"],
                         "path": source_ref["path"], "bytes": source_ref["bytes"], "sha256": source_ref["sha256"],
                         "local_copy": local_copy["ref"]["path"]}],
            "targets": [{"name": target, "module": "Demo.lean", "title": "TEST declared target",
                         "source": "TEST-source", "informal_anchor": anchor.decode("utf-8"),
                         "does_not_claim": "actual Lean declaration existence or scientific acceptance"}
                        for target, anchor in zip(TARGETS, (ALPHA, BETA))],
        }
        manifest_capture = git_capture(encoded(manifest), "TEST/formal/manifest.json")
        logs = [run_capture(raw, name) for name, raw in (
            ("build.log", b"TEST synthetic build text; not an executed build\n"),
            ("leanchecker.log", b"TEST synthetic checker text; not an executed checker\n"),
            ("axioms.log", b"'TEST.Formal.alpha' does not depend on any axioms\n'TEST.Formal.beta' does not depend on any axioms\n"),
            ("elaborated-types.log", b"TEST.Formal.alpha : True\nTEST.Formal.beta : True\n"),
            ("TEST_false_claim.log", b"TEST/formal/Demo.lean:2:22: error: unsolved goals\n\xe2\x8a\xa2 False\n"),
            ("sorry.log", b"TEST/formal/sorry.lean:2:8: warning: declaration uses 'sorry'\n'injected' depends on axioms: [sorryAx]\n"),
            ("custom_imported.log", b"'injected' depends on axioms: [hiddenPremise]\n"),
            ("native.log", b"'injected' depends on axioms: [Lean.ofReduceBool]\n"),
            ("version.log", b"Lean (version 4.34.1, TEST fixture only)\n"),
        )]
        receipt = {
            "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "package": "TEST-synthetic-package", "formalization_status": "kernel-checked",
            "alignment_status": "PENDING_INDEPENDENT_REVIEW", "manifest_sha256": sha(raw_bytes(manifest_capture)),
            "checked_commit": HEAD, "repository": REPO, "workflow_run_id": "123",
            "lean_version": "Lean (version 4.34.1, TEST fixture only)", "dependency_revisions": {},
            "axioms": {target: [] for target in TARGETS},
            "negative_controls": {"TEST_false_claim": "REJECTED_BY_LEAN", "sorry": "REJECTED_BY_AXIOM_GATE",
                                  "custom_imported": "REJECTED_BY_AXIOM_GATE", "native": "REJECTED_BY_AXIOM_GATE"},
            "logs": {item["ref"]["member_path"].rsplit("/", 1)[1]: item["ref"]["sha256"] for item in logs},
        }
        alignment = {
            "disposition": "ACCEPTED", "manifest_sha256": sha(raw_bytes(manifest_capture)),
            "scope_sha256": scope["ref"]["sha256"], "targets": TARGETS,
            "author": {"provider": "TEST-author", "family": "TEST-author-family", "agent": "TEST-author-agent"},
            "reviewer": {"provider": "TEST-reviewer", "family": "TEST-reviewer-family", "agent": "TEST-reviewer-agent"},
            "evidence": {"repository": REPO, "commit": HEAD, "path": "TEST/review-substance.md", "sha256": "e" * 64},
        }
        return {
            "id": "TEST-formal-1", "format": "main-formal-gate/v1", "manifest": manifest_capture,
            "scope": scope, "source_files": files, "receipt": run_capture(encoded(receipt), "receipt.json"),
            "logs": logs, "alignment": git_capture(encoded(alignment), "TEST/alignment.json"),
            "native_run": {"repository": REPO, "checked_commit": HEAD, "run_head_sha": "d" * 40,
                           "run_id": "123", "run_attempt": "1", "purpose": "check",
                           "conclusion": "success", "expected_conclusion": "success"},
            "node_targets": [{"node": "TEST.A", "target": "TEST.Formal.alpha"}],
        }

    def with_formal(self):
        packet = copy.deepcopy(self.packet)
        packet["formal_records"] = [self.formal(packet)]
        return packet

    def math_formal(self, packet):
        formal = self.formal(packet)
        formal["format"] = "math-formal-gate/v1"
        manifest = document(formal["manifest"])
        manifest["targets"] = TARGETS
        # Math native file keys are formal-root-relative; Git refs remain repo-relative.
        prefix = formal["manifest"]["ref"]["path"].rsplit("/", 1)[0] + "/"
        # Preserve the prior Math fixture rather than borrowing Main-only control captures.
        main_only = {"TEST/formal/lean-toolchain", "TEST/formal/lakefile.toml",
                     "TEST/formal/lake-manifest.json", "TEST/formal/TEST.Formal.lean",
                     "TEST/formal/GLOSSARY.md", "TEST/formal/README.md",
                     "tools/formal_gate_check.py", "tests/test_formal_gate.py"}
        formal["source_files"] = [capture for capture in formal["source_files"]
                                  if capture["ref"]["path"] not in main_only]
        manifest["files"] = {path: digest for path, digest in manifest["files"].items() if path not in main_only}
        formal["logs"] = [capture for capture in formal["logs"]
                          if not capture["ref"]["member_path"].endswith("/version.log")]
        self.update_receipt(formal, lambda receipt: receipt["logs"].pop("version.log"))
        manifest["files"] = {path[len(prefix):]: digest for path, digest in manifest["files"].items()}
        for field in ("package", "package_root", "root_module", "scientific_status_authority",
                      "sources", "allowed_axioms", "lean_toolchain", "meaning", "coordination",
                      "author", "unbound_files", "negative_controls"):
            del manifest[field]
        declared_archive = b"TEST archive metadata only; no retained archive or custody claim"
        manifest["source_archive"] = {"name": "TEST-declared-only.zip", "sha256": sha(declared_archive),
                                      "size_bytes": len(declared_archive)}
        formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
        self.update_receipt(formal, lambda receipt: (receipt.pop("package"), receipt.pop("scientific_status_authority"),
                            receipt.update({"manifest_sha256": formal["manifest"]["ref"]["sha256"]})))
        self.update_alignment(formal, lambda review: review.update({"manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
        return formal

    def update_receipt(self, formal, change):
        receipt = document(formal["receipt"])
        change(receipt)
        formal["receipt"] = replace_run(formal["receipt"], encoded(receipt))

    def update_alignment(self, formal, change):
        alignment = document(formal["alignment"])
        change(alignment)
        formal["alignment"] = replace_git(formal["alignment"], encoded(alignment))

    def replace_log(self, formal, name, raw):
        for index, capture in enumerate(formal["logs"]):
            if capture["ref"]["member_path"].endswith("/" + name):
                formal["logs"][index] = replace_run(capture, raw)
                self.update_receipt(formal, lambda receipt: receipt["logs"].update({name: sha(raw)}))
                return
        self.fail("TEST fixture log not found: " + name)

    def run_adapter(self, packet=None, *, raw=None, expected=None, arguments=None):
        self.assertTrue(CLI.is_file(), "required evidence adapter CLI tools/architecture_evidence_adapter.py is absent")
        if raw is None:
            raw = encoded(self.packet if packet is None else packet).decode("utf-8")
        identity = self.expected if expected is None else expected
        if arguments is None:
            arguments = []
            for name in ("repository", "commit", "run-id", "run-attempt"):
                arguments.extend(["--expected-" + name, identity[name]])
        flags = ["-B", "-S"]
        if sys.flags.optimize:
            flags.append("-" + "O" * sys.flags.optimize)
        env = os.environ.copy()
        env.pop("PYTHONOPTIMIZE", None)
        try:
            return subprocess.run([sys.executable, *flags, str(CLI), *arguments], input=raw,
                                  cwd=self.cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                                  timeout=20, check=False)
        except subprocess.TimeoutExpired:
            self.fail("evidence adapter did not finish within 20 seconds")

    def accepted(self, packet=None, **kwargs):
        packet = self.packet if packet is None else packet
        result = self.run_adapter(packet, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        report = json.loads(result.stdout)
        self.assertEqual(set(report), {"schema_version", "scientific_effect", "scientific_status_authority",
                                      "custody", "original_packet", "dimensions", "formal_summary", "regression"})
        self.assertIs(type(report["schema_version"]), int)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["scientific_status_authority"], False)
        self.assertEqual(report["custody"], "unknown")
        self.assertEqual(json.dumps(report["original_packet"], sort_keys=True, allow_nan=False),
                         json.dumps(packet, sort_keys=True, allow_nan=False), "original typed records changed")
        self.assertEqual(set(report["dimensions"]), set(document(packet["new"]["graph"])["nodes"]))
        for node, dimensions in report["dimensions"].items():
            self.assertEqual(set(dimensions), AXES, node)
            for axis in dimensions.values():
                self.assertEqual(set(axis), {"record_state", "applicability", "reasons", "evidence_ids", "custody"})
                self.assertIn(axis["record_state"], STATES)
                self.assertIn(axis["applicability"], APPLICABILITY)
                self.assertEqual(axis["custody"], "unknown")
                self.assertIsInstance(axis["reasons"], list)
                self.assertIsInstance(axis["evidence_ids"], list)
        self.assertEqual(set(report["regression"]), {"changed_nodes", "impacted_nodes", "statement_changed",
                        "proof_changed", "unscoped_source_changed", "traversal_edges", "revalidation_required",
                        "affected_alignment_records"})
        for proposal in report["regression"]["revalidation_required"]:
            self.assertEqual(set(proposal), {"node", "reasons"})
        for summary, formal in zip(report["formal_summary"], packet["formal_records"]):
            self.assertEqual(set(summary), {"id", "format", "manifest_targets", "node_targets",
                                          "retained_manifest", "retained_receipt", "retained_alignment"})
            self.assertEqual(summary["id"], formal["id"])
            self.assertEqual(summary["format"], formal["format"])
            retained = {"retained_manifest": document(formal["manifest"]),
                        "retained_receipt": None if formal["receipt"] is None else document(formal["receipt"]),
                        "retained_alignment": None if formal["alignment"] is None else document(formal["alignment"]),
                        "node_targets": formal["node_targets"]}
            for key, original in retained.items():
                self.assertEqual(json.dumps(summary[key], sort_keys=True, allow_nan=False),
                                 json.dumps(original, sort_keys=True, allow_nan=False), key)
        self.assertEqual(len(report["formal_summary"]), len(packet["formal_records"]))
        return report, result

    def refused(self, packet=None, **kwargs):
        result = self.run_adapter(packet, **kwargs)
        self.assertNotEqual(result.returncode, 0, "invalid packet was accepted")
        self.assertNotIn('"original_packet"', result.stdout, "refusal emitted a valid report")

    def test_minimal_unchanged_packet_is_deterministic_unknown_and_inert(self):
        before = tuple(self.cwd.iterdir())
        first, first_result = self.accepted()
        _, second_result = self.accepted()
        self.assertEqual(first_result.stdout, second_result.stdout)
        self.assertEqual(tuple(self.cwd.iterdir()), before)
        self.assertEqual(first["regression"]["changed_nodes"], [])
        self.assertEqual(first["regression"]["impacted_nodes"], [])
        self.assertEqual(first["regression"]["affected_alignment_records"], [])
        self.assertIn("declared_byte_slice_only", first["dimensions"]["TEST.A"]["source"]["reasons"])
        for axis in ("kernel", "computation", "alignment"):
            self.assertEqual(first["dimensions"]["TEST.A"][axis]["applicability"], "unknown")

    def test_complete_formal_scope_is_retained_without_authenticated_or_fuzzy_joins(self):
        packet = self.with_formal()
        report, _ = self.accepted(packet)
        self.assertEqual(report["formal_summary"][0]["manifest_targets"], TARGETS)
        self.assertEqual(set(report["formal_summary"][0]["retained_receipt"]["axioms"]), set(TARGETS))
        for axis in ("kernel", "alignment"):
            self.assertEqual(report["dimensions"]["TEST.A"][axis]["applicability"], "current")
            self.assertEqual(report["dimensions"]["TEST.A"][axis]["custody"], "unknown")
        self.assertEqual(report["dimensions"]["TEST.B"]["kernel"]["applicability"], "unknown")
        self.assertNotIn("lemma_exists", report)
        self.assertNotIn("kernel_verified", report)

    def test_math_native_receipt_does_not_acquire_main_only_native_fields(self):
        packet = copy.deepcopy(self.packet)
        formal = self.math_formal(packet)
        formal["node_targets"] = []
        packet["formal_records"] = [formal]
        report, _ = self.accepted(packet)
        retained = report["formal_summary"][0]["retained_receipt"]
        self.assertNotIn("package", retained)
        self.assertNotIn("scientific_status_authority", retained)
        self.assertIn("SCOPE.md", report["formal_summary"][0]["retained_manifest"]["files"])
        self.assertIn("TEST/formal/SCOPE.md", {capture["ref"]["path"] for capture in formal["source_files"]})
        self.assertEqual(report["formal_summary"][0]["manifest_targets"], TARGETS)
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")

    def test_math_alignment_checks_every_proposer_normalized_lineage_and_target_scope(self):
        packet = copy.deepcopy(self.packet)
        formal = self.math_formal(packet)
        packet["formal_records"] = [formal]
        proposers = [
            {"provider": "TEST-proposer-one", "family": "TEST-family-one", "agent": "TEST-agent-one", "targets": [TARGETS[0]]},
            {"provider": "TEST-proposer-two", "family": "TEST-family-two", "agent": "TEST-agent-two", "targets": [TARGETS[1]]},
        ]
        self.update_alignment(formal, lambda review: review.update({"proposal_authors": proposers}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
        for mutation in ("nonfirst_provider", "normalized_family", "placeholder_agent", "foreign_scope"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(packet)
                def alter(review):
                    if mutation == "nonfirst_provider":
                        review["proposal_authors"][1]["provider"] = review["reviewer"]["provider"]
                    elif mutation == "normalized_family":
                        review["reviewer"]["family"] = "TEST Reviewer Family"
                        review["proposal_authors"][1]["family"] = "  test   reviewer\tFAMILY  "
                    elif mutation == "placeholder_agent":
                        review["proposal_authors"][1]["agent"] = "  NOT\tKNOWN  "
                    else:
                        review["proposal_authors"][1]["targets"] = ["TEST.Formal.absent"]
                self.update_alignment(changed["formal_records"][0], alter)
                report, _ = self.accepted(changed)
                self.assertNotEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")

    def test_failed_or_skipped_incomplete_run_and_expected_failure_never_become_positive_kernel(self):
        for conclusion, purpose in (("failure", "check"), ("skipped", "check"), ("cancelled", "check"),
                                    ("failure", "negative_control")):
            with self.subTest(conclusion=conclusion, purpose=purpose):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                formal["receipt"] = None
                formal["logs"] = formal["logs"][:1]
                formal["native_run"].update({"conclusion": conclusion, "purpose": purpose, "expected_conclusion": "failure"})
                report, _ = self.accepted(packet)
                self.assertNotEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                self.assertEqual(report["formal_summary"][0]["retained_receipt"], None)
        # Complete positive-looking original logs cannot override the native purpose/conclusion.
        for conclusion, purpose, expected in (("failure", "check", "failure"),
                                               ("success", "negative_control", "success")):
            with self.subTest(complete_receipt=True, conclusion=conclusion, purpose=purpose):
                packet = self.with_formal()
                packet["formal_records"][0]["native_run"].update(
                    {"conclusion": conclusion, "purpose": purpose, "expected_conclusion": expected})
                report, _ = self.accepted(packet)
                self.assertNotEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")

    def test_expected_native_context_mismatch_is_valid_stale_not_authentication(self):
        packet = self.with_formal()
        for field, stale in (("repository", "other-owner/research"), ("commit", "c" * 40), ("run-id", "124"), ("run-attempt", "2")):
            with self.subTest(field=field):
                report, _ = self.accepted(packet, expected={**self.expected, field: stale})
                kernel = report["dimensions"]["TEST.A"]["kernel"]
                self.assertNotEqual(kernel["applicability"], "current")
                self.assertIn("native_context_changed", kernel["reasons"])

    def test_partial_successful_target_inventory_is_retained_but_not_current_kernel(self):
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        self.replace_log(formal, "axioms.log", b"'TEST.Formal.alpha' does not depend on any axioms\n")
        self.update_receipt(formal, lambda receipt: receipt.update({"axioms": {TARGETS[0]: []}}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["formal_summary"][0]["manifest_targets"], TARGETS)
        self.assertEqual(report["formal_summary"][0]["retained_receipt"]["axioms"], {TARGETS[0]: []})
        kernel = report["dimensions"]["TEST.A"]["kernel"]
        self.assertEqual(kernel["applicability"], "unknown")
        self.assertIn("target_inventory_incomplete", kernel["reasons"])

    def test_original_axioms_dump_cannot_contradict_receipt_target_scope_or_values(self):
        for raw in (b"'TEST.Formal.alpha' does not depend on any axioms\n'TEST.Formal.foreign' does not depend on any axioms\n",
                    b"'TEST.Formal.alpha' depends on axioms: [Classical.choice]\n'TEST.Formal.beta' does not depend on any axioms\n",
                    b"'TEST.Formal.alpha' does not depend on any axioms\n'TEST.Formal.alpha' does not depend on any axioms\n"):
            with self.subTest(raw=raw):
                packet = self.with_formal()
                self.replace_log(packet["formal_records"][0], "axioms.log", raw)
                self.refused(packet)

    def test_main_target_requires_exact_source_identity_and_explicit_statement_slice(self):
        for mutation in ("missing_statement", "partial_statement", "foreign_source_path", "foreign_source_commit"):
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                binding = packet["new"]["bindings"][0]
                if mutation == "missing_statement":
                    binding["statement_slice"] = None
                elif mutation == "partial_statement":
                    binding["statement_slice"] = slice_of(SOURCE, 0, len(ALPHA) - 1)
                elif mutation == "foreign_source_path":
                    binding["source"]["ref"]["path"] = "TEST/different-source.md"
                else:
                    binding["source"]["ref"]["commit"] = "c" * 40
                report, _ = self.accepted(packet)
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")

    def test_main_origin_source_is_distinct_from_its_bound_local_copy(self):
        packet = copy.deepcopy(self.packet)
        for label in ("old", "new"):
            packet[label]["bindings"][0]["source"] = git_capture(
                SOURCE, "TEST/origin-source.md", "c" * 40, "d6g8k5htny-coder/Math-")
        formal = self.formal(packet)
        packet["formal_records"] = [formal]
        report, _ = self.accepted(packet)
        origin = report["formal_summary"][0]["retained_manifest"]["sources"][0]
        self.assertEqual((origin["repository"], origin["commit"], origin["path"]),
                         ("d6g8k5htny-coder/Math-", "c" * 40, "TEST/origin-source.md"))
        self.assertEqual(origin["local_copy"], "TEST/formal/sources/source.md")
        local = formal["source_files"][0]
        self.assertEqual((local["ref"]["repository"], local["ref"]["commit"], local["ref"]["path"]),
                         (REPO, HEAD, origin["local_copy"]))
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        mismatch = copy.deepcopy(packet)
        mismatch["new"]["bindings"][0]["source"]["ref"]["commit"] = "e" * 40
        report, _ = self.accepted(mismatch)
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")

    def test_unknown_or_duplicate_explicit_node_target_binding_is_a_contradiction(self):
        for binding in ([{"node": "TEST.A", "target": "TEST.Formal.absent"}],
                        [{"node": "TEST.absent", "target": TARGETS[0]}],
                        [{"node": "TEST.A", "target": TARGETS[0]}] * 2):
            with self.subTest(binding=binding):
                packet = self.with_formal()
                packet["formal_records"][0]["node_targets"] = binding
                self.refused(packet)

    def test_statement_change_uses_removed_and_context_edges_and_names_exact_review(self):
        packet = copy.deepcopy(self.packet)
        # Retain the historical source at BASE; changed bytes belong to HEAD.
        packet["formal_records"] = [self.formal(packet, source_snapshot="old")]
        raw = SOURCE.replace(b"alpha", b"ALPHA", 1)
        self.change_source(packet, "new", raw)
        self.change_graph(packet, "new", lambda graph: graph.update({"edges": graph["edges"][1:]}))
        report, _ = self.accepted(packet)
        regression = report["regression"]
        self.assertEqual(regression["statement_changed"], ["TEST.A"])
        self.assertEqual(regression["impacted_nodes"], ["TEST.A", "TEST.B", "TEST.C"])
        self.assertEqual(regression["traversal_edges"], [["TEST.B", "TEST.A"], ["TEST.C", "TEST.B"]])
        self.assertEqual([item["node"] for item in regression["revalidation_required"]], ["TEST.A", "TEST.B", "TEST.C"])
        self.assertEqual(regression["affected_alignment_records"], [packet["formal_records"][0]["alignment"]["ref"]])
        self.assertEqual(document(report["original_packet"]["new"]["graph"])["nodes"]["TEST.B"]["review_disposition"], "ACCEPTED")

    def test_proof_only_change_stales_whole_manifest_alignment_without_statement_change(self):
        packet = self.with_formal()
        original_alignment = copy.deepcopy(packet["formal_records"][0]["alignment"])
        self.change_source(packet, "new", SOURCE.replace(b"proof old", b"proof new"))
        packet["formal_records"] = [self.formal(packet)]
        packet["formal_records"][0]["alignment"] = original_alignment
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["statement_changed"], [])
        self.assertEqual(report["regression"]["proof_changed"], ["TEST.A"])
        alignment = report["dimensions"]["TEST.A"]["alignment"]
        self.assertEqual(alignment["applicability"], "stale")
        self.assertIn("manifest_binding_changed", alignment["reasons"])
        self.assertNotIn("statement_changed", alignment["reasons"])
        self.assertEqual(report["regression"]["affected_alignment_records"], [original_alignment["ref"]])

    def test_alignment_partial_coverage_stale_scope_or_lineage_remains_original_stale_record(self):
        for change in (lambda review: review.update({"targets": [TARGETS[0]]}),
                       lambda review: review.update({"scope_sha256": "f" * 64}),
                       lambda review: review.update({"disposition": "PENDING"}),
                       lambda review: review.update({"reviewer": copy.deepcopy(review["author"])})):
            packet = self.with_formal()
            self.update_alignment(packet["formal_records"][0], change)
            report, _ = self.accepted(packet)
            self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "stale")
            self.assertEqual(report["regression"]["affected_alignment_records"], [packet["formal_records"][0]["alignment"]["ref"]])

    def test_native_SCOPE_file_cannot_be_substituted_by_another_manifest_bound_file(self):
        for native_format in ("main-formal-gate/v1", "math-formal-gate/v1"):
            with self.subTest(native_format=native_format):
                packet = copy.deepcopy(self.packet)
                formal = self.formal(packet) if native_format == "main-formal-gate/v1" else self.math_formal(packet)
                packet["formal_records"] = [formal]
                module = next(capture for capture in formal["source_files"]
                              if capture["ref"]["path"] == "TEST/formal/Demo.lean")
                formal["scope"] = copy.deepcopy(module)
                self.update_alignment(formal, lambda review: review.update({"scope_sha256": module["ref"]["sha256"]}))
                # The real SCOPE.md, complete file inventory, receipt and logs remain intact.
                result = self.run_adapter(packet)
                self.assertNotEqual(result.returncode, 0, "a module replaced the source-native SCOPE.md identity")
                self.assertEqual(result.stdout, "", "scope substitution emitted a JSON report")

    def test_native_negative_control_outcomes_cannot_swap_Lean_and_axiom_gate_rejections(self):
        for name, wrong_outcome in (("TEST_false_claim", "REJECTED_BY_AXIOM_GATE"),
                                    ("sorry", "REJECTED_BY_LEAN"),
                                    ("custom_imported", "REJECTED_BY_LEAN"),
                                    ("native", "REJECTED_BY_LEAN")):
            with self.subTest(name=name):
                packet = self.with_formal()
                self.update_receipt(packet["formal_records"][0],
                                    lambda receipt: receipt["negative_controls"].update({name: wrong_outcome}))
                # Only this original receipt outcome and its capture byte hashes change.
                result = self.run_adapter(packet)
                self.assertNotEqual(result.returncode, 0, "native rejection mechanism was substituted for " + name)
                self.assertEqual(result.stdout, "", "contradictory native rejection emitted a JSON report")

    def test_main_native_producer_requires_its_fixed_toolchain_and_Lean_version(self):
        for mutation in ("receipt_version_only", "manifest_and_receipt_version"):
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                if mutation == "manifest_and_receipt_version":
                    manifest = document(formal["manifest"])
                    manifest["lean_toolchain"] = "leanprover/lean4:v5.0.0"
                    formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
                    self.update_alignment(formal, lambda review: review.update(
                        {"manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
                self.update_receipt(formal, lambda receipt: receipt.update(
                    {"lean_version": "Lean (version 5.0.0, TEST fixture only)",
                     "manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
                result = self.run_adapter(packet)
                self.assertNotEqual(result.returncode, 0, "unsupported native producer toolchain/version was accepted")
                self.assertEqual(result.stdout, "", "unsupported native producer version emitted a JSON report")

    def test_affected_alignment_references_deduplicate_full_original_identity(self):
        packet = self.with_formal()
        second = copy.deepcopy(packet["formal_records"][0])
        second["id"] = "TEST-formal-2"
        packet["formal_records"].append(second)
        self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({"notes": "TEST changed record"}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["affected_alignment_records"], [second["alignment"]["ref"]])

    def test_distinct_original_alignment_identities_are_retained_and_canonically_sorted(self):
        packet = self.with_formal()
        second = copy.deepcopy(packet["formal_records"][0])
        second["id"] = "TEST-formal-2"
        second["alignment"]["ref"]["path"] = "TEST/another-original-alignment.json"
        second["alignment"]["ref"]["commit"] = BASE
        packet["formal_records"].append(second)
        self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({"notes": "TEST changed record"}))
        report, _ = self.accepted(packet)
        original_refs = [formal["alignment"]["ref"] for formal in packet["formal_records"]]
        expected = sorted(original_refs, key=lambda ref: json.dumps(ref, sort_keys=True, separators=(",", ":")))
        self.assertEqual(report["regression"]["affected_alignment_records"], expected)
        self.assertEqual(len(report["regression"]["affected_alignment_records"]), 2)

    def test_unmapped_similar_node_does_not_invent_an_affected_alignment_record(self):
        packet = self.with_formal()
        self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.B"].update({"notes": "TEST.Formal.alpha changed title"}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["impacted_nodes"], ["TEST.B", "TEST.C"])
        self.assertEqual(report["regression"]["affected_alignment_records"], [])

    def test_unscoped_source_change_is_unknown_not_a_guessed_statement_or_proof_split(self):
        packet = copy.deepcopy(self.packet)
        self.change_source(packet, "old", SOURCE, sliced=False)
        self.change_source(packet, "new", SOURCE + b"TEST changed bytes\n", sliced=False)
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["unscoped_source_changed"], ["TEST.A"])
        self.assertEqual(report["regression"]["statement_changed"], [])
        self.assertEqual(report["regression"]["proof_changed"], [])

    def test_canonical_record_numeric_types_are_changes_even_when_python_equality_agrees(self):
        packet = copy.deepcopy(self.packet)
        self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({"review_issue": 65.0}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["changed_nodes"], ["TEST.A"])
        self.assertEqual(report["regression"]["impacted_nodes"], ["TEST.A", "TEST.B", "TEST.C"])
        self.assertIs(type(document(report["original_packet"]["new"]["graph"])["nodes"]["TEST.A"]["review_issue"]), float)

    def test_graph_context_changes_revalidate_all_nodes_and_preserve_context(self):
        packet = copy.deepcopy(self.packet)
        self.change_graph(packet, "new", lambda graph: graph.update({"object": "TEST-CHANGED-CONTEXT"}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["changed_nodes"], ["TEST.A", "TEST.B", "TEST.C"])

    def test_deleted_and_added_nodes_use_union_edges_but_only_survivors_get_proposals(self):
        packet = copy.deepcopy(self.packet)
        def delete(graph):
            del graph["nodes"]["TEST.B"]
            graph["edges"] = []
        self.change_graph(packet, "new", delete)
        report, _ = self.accepted(packet)
        regression = report["regression"]
        self.assertEqual(regression["changed_nodes"], ["TEST.B", "TEST.C"])
        self.assertEqual(regression["impacted_nodes"], ["TEST.C"])
        self.assertEqual(regression["traversal_edges"], [["TEST.B", "TEST.A"], ["TEST.C", "TEST.B"]])
        self.assertEqual([item["node"] for item in regression["revalidation_required"]], ["TEST.C"])
        self.assertIn("TEST.B", document(report["original_packet"]["old"]["graph"])["nodes"])
        packet = copy.deepcopy(self.packet)
        def add(graph):
            graph["nodes"]["TEST.D"] = {"classification": "AUTHOR_SIDE_CANDIDATE", "controlling": False}
            graph["edges"].append({"from": "TEST.D", "to": "TEST.A", "required": False, "relation": "TEST added context"})
        self.change_graph(packet, "new", add)
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["changed_nodes"], ["TEST.D"])
        self.assertEqual(report["regression"]["impacted_nodes"], ["TEST.D"])
        self.assertIn(["TEST.D", "TEST.A"], report["regression"]["traversal_edges"])

    def test_source_availability_loss_is_unresolved_and_propagates_without_guessed_slices(self):
        packet = copy.deepcopy(self.packet)
        binding = packet["new"]["bindings"][0]
        binding["source"] = binding["statement_slice"] = binding["proof_slice"] = None
        report, _ = self.accepted(packet)
        self.assertEqual(report["regression"]["impacted_nodes"], ["TEST.A", "TEST.B", "TEST.C"])
        self.assertEqual(report["regression"]["statement_changed"], [])
        self.assertEqual(report["regression"]["proof_changed"], [])
        self.assertNotEqual(report["dimensions"]["TEST.A"]["source"]["applicability"], "current")

    def test_required_cycle_and_boolean_coercion_are_refused_but_context_cycle_is_valid(self):
        for mutation in ("required_cycle", "numeric_controlling", "numeric_required"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.packet)
                def change(graph):
                    if mutation == "required_cycle":
                        graph["edges"].append({"from": "TEST.A", "to": "TEST.B", "required": True, "relation": "TEST cycle"})
                    elif mutation == "numeric_controlling":
                        graph["nodes"]["TEST.A"]["controlling"] = 0
                    else:
                        graph["edges"][0]["required"] = 1
                self.change_graph(packet, "new", change)
                self.refused(packet)
        packet = copy.deepcopy(self.packet)
        self.change_graph(packet, "new", lambda graph: graph["edges"].append(
            {"from": "TEST.A", "to": "TEST.C", "required": False, "relation": "TEST context cycle"}))
        self.accepted(packet)

    def test_closed_packet_capture_binding_and_formal_wrappers_refuse_extra_or_missing_keys(self):
        for mutation in ("top_extra", "top_missing", "schema_bool", "ref_extra", "binding_extra", "formal_extra"):
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                if mutation == "top_extra": packet["authenticated_capture"] = True
                elif mutation == "top_missing": del packet["old"]
                elif mutation == "schema_bool": packet["schema_version"] = True
                elif mutation == "ref_extra": packet["new"]["graph"]["ref"]["authenticated"] = True
                elif mutation == "binding_extra": packet["new"]["bindings"][0]["target_guess"] = TARGETS[0]
                else: packet["formal_records"][0]["capture_verified"] = True
                self.refused(packet)

    def test_git_capture_checks_exact_raw_byte_count_sha_blob_and_canonical_base64(self):
        for field, value in (("bytes", True), ("bytes", len(SOURCE) + 1), ("sha256", "f" * 64),
                             ("git_blob", "e" * 40), ("commit", "A" * 40), ("path", "../TEST/source.md")):
            with self.subTest(field=field):
                packet = copy.deepcopy(self.packet)
                packet["new"]["bindings"][0]["source"]["ref"][field] = value
                self.refused(packet)
        packet = copy.deepcopy(self.packet)
        packet["new"]["bindings"][0]["source"]["raw_base64"] = "not base64!!"
        self.refused(packet)

    def test_conflicting_bytes_for_one_immutable_git_identity_anywhere_are_refused(self):
        for location in ("snapshots", "bindings", "graph_and_source", "formal_and_binding"):
            with self.subTest(location=location):
                packet = self.with_formal() if location == "formal_and_binding" else copy.deepcopy(self.packet)
                if location == "snapshots":
                    packet["old"]["bindings"][0]["source"]["ref"]["commit"] = HEAD
                    self.change_source(packet, "new", SOURCE.replace(b"proof old", b"proof new"))
                else:
                    if location == "bindings":
                        ref = packet["new"]["bindings"][0]["source"]["ref"]
                    elif location == "graph_and_source":
                        ref = packet["new"]["graph"]["ref"]
                    else:
                        ref = packet["formal_records"][0]["scope"]["ref"]
                    conflicting = git_capture(b"TEST conflicting bytes", ref["path"], ref["commit"], ref["repository"])
                    packet["new"]["bindings"].append({"node": "TEST.B", "source": conflicting,
                                                     "statement_slice": None, "proof_slice": None})
                self.refused(packet)

    def test_run_artifact_is_not_a_git_file_and_raw_hashes_and_native_bindings_must_match(self):
        for mutation in ("raw_hash", "git_field", "attempt_contradiction"):
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                ref = packet["formal_records"][0]["receipt"]["ref"]
                if mutation == "raw_hash": ref["sha256"] = "f" * 64
                elif mutation == "git_field": ref["git_blob"] = "e" * 40
                else: ref["run_attempt"] = "2"
                self.refused(packet)

    def test_native_manifest_file_coverage_and_receipt_log_hashes_cannot_contradict_captures(self):
        for mutation in ("missing_source", "changed_source", "changed_receipt_log"):
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                if mutation == "missing_source":
                    del formal["source_files"][0]
                elif mutation == "changed_source":
                    formal["source_files"][0] = replace_git(formal["source_files"][0], SOURCE + b"TEST changed capture\n")
                else:
                    self.update_receipt(formal, lambda receipt: receipt["logs"].update({"axioms.log": "f" * 64}))
                self.refused(packet)

    def test_statement_and_proof_slice_bounds_types_and_hashes_are_exact_bytes(self):
        for field, value in (("start_byte", True), ("start_byte", -1), ("end_byte", len(SOURCE) + 1),
                             ("end_byte", 0), ("sha256", "f" * 64)):
            with self.subTest(field=field, value=value):
                packet = copy.deepcopy(self.packet)
                packet["new"]["bindings"][0]["statement_slice"][field] = value
                self.refused(packet)
        packet = copy.deepcopy(self.packet)
        packet["new"]["bindings"][0]["source"] = None
        self.refused(packet)

    def test_duplicate_and_nonfinite_json_and_oversized_packets_are_refused(self):
        valid = encoded(self.packet).decode("utf-8")
        for raw in ('{"schema_version":1,' + valid[1:],
                    valid.replace('"schema_version": 1', '"schema_version": NaN'),
                    valid.replace('"schema_version": 1', '"schema_version": Infinity'),
                    valid.replace('"schema_version": 1', '"schema_version": 1e999'),
                    valid + "{}", valid[:-1] + " TEST",
                    valid + " " * (16 * 1024 * 1024 + 1 - len(valid.encode("utf-8")))):
            with self.subTest(raw=raw[:40]):
                self.refused(raw=raw)

    def test_decoded_capture_limit_and_embedded_json_depth_and_duplicate_keys_are_bounded(self):
        packet = copy.deepcopy(self.packet)
        self.change_source(packet, "new", b"T" * (4 * 1024 * 1024 + 1), sliced=False)
        self.refused(packet)
        packet = copy.deepcopy(self.packet)
        nested = "TEST deepest metadata"
        for _ in range(65):
            nested = {"TEST": nested}
        self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({"metadata": nested}))
        self.refused(packet)
        for raw in (raw_bytes(self.packet["new"]["graph"]).replace(b'"review_issue": 65', b'"review_issue": 65, "review_issue": 65'),
                    raw_bytes(self.packet["new"]["graph"]).replace(b'"review_issue": 65', b'"review_issue": NaN'),
                    raw_bytes(self.packet["new"]["graph"]).replace(b'"review_issue": 65', b'"review_issue": 1e999')):
            packet = copy.deepcopy(self.packet)
            packet["new"]["graph"] = replace_git(packet["new"]["graph"], raw)
            self.refused(packet)

    def test_all_expected_context_flags_are_required_unique_and_syntax_checked(self):
        for omitted in self.expected:
            with self.subTest(omitted=omitted):
                arguments = [part for name, value in self.expected.items() if name != omitted for part in ("--expected-" + name, value)]
                self.refused(arguments=arguments)
        arguments = [part for name, value in self.expected.items() for part in ("--expected-" + name, value)]
        for name, value in self.expected.items():
            with self.subTest(duplicate=name):
                self.refused(arguments=arguments + ["--expected-" + name, value])
        for field, value in (("repository", "owner/repo/extra"), ("commit", "A" * 40), ("run-id", "0123"), ("run-attempt", "0")):
            with self.subTest(field=field):
                self.refused(expected={**self.expected, field: value})

    def test_unknown_native_format_or_packet_authentication_claim_is_refused(self):
        packet = self.with_formal()
        packet["formal_records"][0]["format"] = "unified-formal-schema/guessed"
        self.refused(packet)
        self.refused(arguments=["--capture-authenticated", "true"])

    def test_retrofit_failed_evidence_remains_recorded_without_positive_kernel_or_computation(self):
        packet = copy.deepcopy(self.packet)
        ref = packet["new"]["bindings"][0]["source"]["ref"]
        run = {"kind": "workflow_run", "repository": "main", "ref": "123", "attempt": 1, "job": "789",
               "run_head_sha": "d" * 40, "checked_commit": HEAD, "purpose": "negative_control",
               "conclusion": "failure", "expected_conclusion": "failure"}
        person = {"provider": "TEST", "model_or_agent": "TEST synthetic", "session": "TEST not an actual review"}
        records = []
        for record_id, axis in (("TEST-formal", "formal_evidence"), ("TEST-numeric", "numerical_reproduction")):
            records.append({"id": record_id, "subject": {"repository": "main", "commit": HEAD,
                            "path": ref["path"], "blob": ref["git_blob"]}, "delta": True,
                            "axis": axis, "state": "recorded", "evidence": [run], "performer": person,
                            "exposure": "TEST synthetic failed control", "independence_credit": 0,
                            "alias_of": None, "notes": "TEST no success selection"})
        data = {"schema": "retrofit-records/v0.1", "shard": "TEST", "baseline": {
                    "Math-": "08f86862f859ac4804b4fd6c6477a7ed0421e37f", "main": "cddbb7f6cf3f57f3b495277f7da148287e027b19"},
                "author": person, "scientific_effect": "NONE", "status_authority": False, "records": records}
        packet["retrofit_records"] = [{"id": "TEST-retrofit", "record": git_capture(encoded(data), "TEST/RECORDS.json"),
                                      "node_records": [{"node": "TEST.A", "record_id": record["id"]} for record in records]}]
        report, _ = self.accepted(packet)
        for axis in ("kernel", "computation"):
            self.assertEqual(report["dimensions"]["TEST.A"][axis]["record_state"], "recorded")
            self.assertEqual(report["dimensions"]["TEST.A"][axis]["applicability"], "unknown")
        self.assertEqual(document(report["original_packet"]["retrofit_records"][0]["record"]), data)

    def test_captured_code_and_paths_remain_inert_packet_data(self):
        packet = copy.deepcopy(self.packet)
        canary = self.cwd / "TEST-untrusted-code-executed"
        raw = ("from pathlib import Path\nPath(" + repr(str(canary)) + ").write_text('executed')\n").encode("utf-8")
        binding = packet["new"]["bindings"][0]
        binding["source"] = git_capture(raw, "TEST/untrusted.py")
        binding["statement_slice"] = binding["proof_slice"] = None
        self.accepted(packet)
        self.assertFalse(canary.exists())

    def test_main_alignment_author_binds_retained_manifest_lineage_without_relabeling(self):
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        author = document(formal["manifest"])["author"]
        self.update_alignment(formal, lambda review: review.update({
            "author": {field: " \t" + value.swapcase() + "\t " for field, value in author.items()}}))
        report, _ = self.accepted(packet)
        self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        self.assertEqual(report["regression"]["affected_alignment_records"], [])

        mutations = ("mismatched_provider", "mismatched_family", "mismatched_agent",
                     "relabeled_author_real_reviewer", "missing_manifest_author", "invalid_manifest_author",
                     "missing_manifest_provider", "missing_manifest_family", "missing_manifest_agent",
                     "invalid_manifest_provider", "invalid_manifest_family", "invalid_manifest_agent")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                manifest = document(formal["manifest"])
                original_author = copy.deepcopy(manifest["author"])
                if mutation.startswith("mismatched_"):
                    field = mutation.removeprefix("mismatched_")
                    self.update_alignment(formal, lambda review: review["author"].update(
                        {field: "TEST-relabeled-" + field}))
                elif mutation == "relabeled_author_real_reviewer":
                    self.update_alignment(formal, lambda review: review.update({
                        "author": {field: "TEST-relabeled-" + field for field in original_author},
                        "reviewer": original_author}))
                else:
                    if mutation == "missing_manifest_author":
                        del manifest["author"]
                    elif mutation == "invalid_manifest_author":
                        manifest["author"] = True
                    elif mutation.startswith("missing_manifest_"):
                        del manifest["author"][mutation.removeprefix("missing_manifest_")]
                    else:
                        manifest["author"][mutation.removeprefix("invalid_manifest_")] = 0
                    formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
                    manifest_hash = formal["manifest"]["ref"]["sha256"]
                    self.update_receipt(formal, lambda receipt: receipt.update({"manifest_sha256": manifest_hash}))
                    self.update_alignment(formal, lambda review: review.update({"manifest_sha256": manifest_hash}))
                # Every native/source/log binding stays consistent; only declared authorship is unresolved.
                report, _ = self.accepted(packet)
                alignment = report["dimensions"]["TEST.A"]["alignment"]
                self.assertEqual(alignment["record_state"], "recorded")
                self.assertIn(alignment["applicability"], {"stale", "unknown"})
                self.assertTrue(alignment["reasons"], "authorship inconsistency had no derived reason")
                self.assertEqual(alignment["custody"], "unknown")
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [formal["alignment"]["ref"]])

    def test_math_optional_proposer_scopes_preserve_historical_schema_and_every_lineage(self):
        proposers = [
            {"provider": "TEST-unscoped-one", "family": "TEST-unscoped-family-one", "agent": "TEST-unscoped-agent-one"},
            {"provider": "TEST-unscoped-two", "family": "TEST-unscoped-family-two", "agent": "TEST-unscoped-agent-two"},
        ]
        for schema in ("historical_single_author", "unscoped_proposers"):
            with self.subTest(schema=schema):
                packet = copy.deepcopy(self.packet)
                formal = self.math_formal(packet)
                packet["formal_records"] = [formal]
                if schema == "unscoped_proposers":
                    self.update_alignment(formal, lambda review: review.update({"proposal_authors": proposers}))
                report, _ = self.accepted(packet)
                retained = report["formal_summary"][0]
                self.assertNotIn("author", retained["retained_manifest"])
                if schema == "historical_single_author":
                    self.assertNotIn("proposal_authors", retained["retained_alignment"])
                else:
                    self.assertEqual(retained["retained_alignment"]["proposal_authors"], proposers)
                    for proposer in retained["retained_alignment"]["proposal_authors"]:
                        self.assertNotIn("targets", proposer)
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])

        for mutation in ("unscoped_proposer_reviewer_conflict", "nonfirst_normalized_conflict"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.packet)
                formal = self.math_formal(packet)
                packet["formal_records"] = [formal]
                def alter(review):
                    review["proposal_authors"] = copy.deepcopy(proposers)
                    if mutation == "unscoped_proposer_reviewer_conflict":
                        review["proposal_authors"][0]["agent"] = review["reviewer"]["agent"]
                    else:
                        review["reviewer"]["family"] = "TEST Reviewer Family"
                        review["proposal_authors"][1]["family"] = "  test   reviewer\tFAMILY  "
                self.update_alignment(formal, alter)
                report, _ = self.accepted(packet)
                alignment = report["dimensions"]["TEST.A"]["alignment"]
                self.assertEqual(alignment["record_state"], "recorded")
                self.assertIn(alignment["applicability"], {"stale", "unknown"})
                self.assertIn("lineage_not_distinct", alignment["reasons"])
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                self.assertEqual(report["regression"]["affected_alignment_records"], [formal["alignment"]["ref"]])


    def test_main_placeholder_lineage_cannot_be_current_even_with_coherent_author_bindings(self):
        placeholders = ("unknown", "unverified", "not known", "not available", "not provided",
                        "unavailable", "unspecified", "pending", "tbd", "todo", "none", "null",
                        "n/a", "na", "?", "-")
        for party in ("reviewer", "bound_author"):
            for token in placeholders:
                for field in ("all", "provider", "family", "agent") if token == "unknown" else ("all",):
                    with self.subTest(party=party, token=token, field=field):
                        packet = self.with_formal()
                        formal = packet["formal_records"][0]
                        value = " \t" + " \t ".join(token.upper().split()) + "\t "
                        fields = ("provider", "family", "agent") if field == "all" else (field,)
                        replacement = {name: value for name in fields}
                        if party == "reviewer":
                            self.update_alignment(formal, lambda review: review["reviewer"].update(replacement))
                        else:
                            manifest = document(formal["manifest"])
                            manifest["author"].update(replacement)
                            formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
                            manifest_hash = formal["manifest"]["ref"]["sha256"]
                            self.update_receipt(formal, lambda receipt: receipt.update({"manifest_sha256": manifest_hash}))
                            def alter(review):
                                review["author"].update(replacement)
                                review["manifest_sha256"] = manifest_hash
                            self.update_alignment(formal, alter)
                        report, _ = self.accepted(packet)
                        alignment = report["dimensions"]["TEST.A"]["alignment"]
                        self.assertEqual(alignment["record_state"], "recorded")
                        self.assertIn(alignment["applicability"], {"stale", "unknown"})
                        self.assertIn("lineage_not_distinct", alignment["reasons"])
                        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                        self.assertEqual(report["regression"]["affected_alignment_records"], [formal["alignment"]["ref"]])

    def test_dependency_regression_stales_shared_whole_alignment_without_erasing_kernel(self):
        statement = b"TEST upstream statement old\n"
        proof = b"TEST upstream proof old\n"
        upstream = statement + proof
        base = copy.deepcopy(self.packet)
        for label, commit in (("old", BASE), ("new", HEAD)):
            binding = base[label]["bindings"][0]
            binding["node"] = "TEST.B"
            binding["statement_slice"] = slice_of(SOURCE, len(ALPHA) + len(PROOF), len(SOURCE))
            binding["proof_slice"] = None
            for node in ("TEST.D", "TEST.E"):
                self.change_graph(base, label, lambda graph, node=node: graph["nodes"].update({
                    node: {"classification": "PROVED_REVIEWED", "controlling": False}}))
                extra = copy.deepcopy(binding)
                extra["node"] = node
                extra["statement_slice"] = slice_of(SOURCE, 0, len(ALPHA))
                base[label]["bindings"].append(extra)
            base[label]["bindings"].append({
                "node": "TEST.A", "source": git_capture(upstream, "TEST/upstream.md", commit),
                "statement_slice": slice_of(upstream, 0, len(statement)),
                "proof_slice": slice_of(upstream, len(statement), len(upstream))})
        formal = self.formal(base)
        formal["node_targets"] = [{"node": "TEST.B", "target": TARGETS[1]}, {"node": "TEST.D", "target": TARGETS[0]}]
        separate = copy.deepcopy(formal)
        separate["id"] = "TEST-unaffected-formal"
        separate["node_targets"] = [{"node": "TEST.E", "target": TARGETS[0]}]
        separate["alignment"] = git_capture(raw_bytes(separate["alignment"]), "TEST/separate-alignment.json")
        base["formal_records"] = [formal, separate]
        unchanged, _ = self.accepted(base)
        for node in ("TEST.B", "TEST.D", "TEST.E"):
            self.assertEqual(unchanged["dimensions"][node]["alignment"]["applicability"], "current")
            self.assertEqual(unchanged["dimensions"][node]["kernel"]["applicability"], "current")
        self.assertEqual(unchanged["regression"]["affected_alignment_records"], [])
        for mutation in ("statement", "proof", "source_loss", "upstream_record",
                         "removed_dependency", "dependency_relation", "graph_context"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(base)
                if mutation in ("statement", "proof"):
                    raw = upstream.replace((mutation + " old").encode("ascii"), (mutation + " new").encode("ascii"))
                    binding = packet["new"]["bindings"][-1]
                    binding["source"] = replace_git(binding["source"], raw)
                    binding["statement_slice"] = slice_of(raw, 0, len(statement))
                    binding["proof_slice"] = slice_of(raw, len(statement), len(raw))
                elif mutation == "source_loss":
                    binding = packet["new"]["bindings"][-1]
                    binding["source"] = binding["statement_slice"] = binding["proof_slice"] = None
                elif mutation == "upstream_record":
                    self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({"notes": "TEST changed"}))
                elif mutation == "removed_dependency":
                    self.change_graph(packet, "new", lambda graph: graph.update({"edges": graph["edges"][1:]}))
                elif mutation == "dependency_relation":
                    self.change_graph(packet, "new", lambda graph: graph["edges"][0].update({"relation": "TEST changed"}))
                else:
                    self.change_graph(packet, "new", lambda graph: graph.update({"object": "TEST changed context"}))
                report, _ = self.accepted(packet)
                regression = report["regression"]
                self.assertIn("TEST.B", regression["impacted_nodes"])
                self.assertIn("TEST.C", regression["impacted_nodes"])
                self.assertIn(formal["alignment"]["ref"], regression["affected_alignment_records"])
                for node in ("TEST.B", "TEST.D"):
                    alignment = report["dimensions"][node]["alignment"]
                    self.assertEqual(alignment["record_state"], "recorded")
                    self.assertEqual(alignment["applicability"], "stale")
                    self.assertTrue(alignment["reasons"])
                    self.assertEqual(report["dimensions"][node]["kernel"]["applicability"], "current")
                if mutation != "graph_context":
                    self.assertNotIn("TEST.D", regression["impacted_nodes"])
                    self.assertNotIn("TEST.E", regression["impacted_nodes"])
                    self.assertEqual(regression["affected_alignment_records"], [formal["alignment"]["ref"]])
                    self.assertEqual(report["dimensions"]["TEST.E"]["alignment"]["applicability"], "current")
                else:
                    self.assertIn("TEST.E", regression["impacted_nodes"])
                    self.assertEqual(report["dimensions"]["TEST.E"]["alignment"]["applicability"], "stale")
                    self.assertIn(separate["alignment"]["ref"], regression["affected_alignment_records"])
                self.assertEqual(report["dimensions"]["TEST.C"]["alignment"]["record_state"], "unknown")
                self.assertEqual(report["dimensions"]["TEST.C"]["alignment"]["applicability"], "unknown")


    def test_explicit_successor_graph_review_clears_only_its_exact_regression_context(self):
        for mutation in ("proof_body", "added_dependency", "graph_context"):
            with self.subTest(mutation=mutation):
                packet = copy.deepcopy(self.packet)
                if mutation == "proof_body":
                    self.change_source(packet, "new", ALPHA + PROOF.replace(b"old", b"new") + BETA)
                elif mutation == "added_dependency":
                    self.change_graph(packet, "new", lambda graph: graph["edges"].append({"from": "TEST.A", "to": "TEST.C", "required": False, "relation": "TEST successor context"}))
                else:
                    self.change_graph(packet, "new", lambda graph: graph.update({"object": "TEST successor context"}))
                formal = self.formal(packet)
                # Binding a new source/manifest alone does not record review of graph dependencies.
                packet["formal_records"] = [formal]
                legacy, _ = self.accepted(packet)
                self.assertEqual(legacy["dimensions"]["TEST.A"]["alignment"]["applicability"], "stale")
                self.assertEqual(legacy["regression"]["affected_alignment_records"], [formal["alignment"]["ref"]])
                self.update_alignment(formal, lambda review: review.update({"reviewed_graph": copy.deepcopy(packet["new"]["graph"]["ref"])}))
                current, _ = self.accepted(packet)
                self.assertEqual(current["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(current["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                self.assertEqual(current["regression"]["affected_alignment_records"], [])
                self.assertTrue(current["regression"]["changed_nodes"], "successor erased the original comparison")
                self.assertEqual(current["formal_summary"][0]["retained_alignment"]["reviewed_graph"], packet["new"]["graph"]["ref"])
                for context in (packet["old"]["graph"]["ref"], {**packet["new"]["graph"]["ref"], "sha256": "f" * 64}):
                    historical = copy.deepcopy(packet)
                    self.update_alignment(historical["formal_records"][0], lambda review, context=context: review.update({"reviewed_graph": context}))
                    held, _ = self.accepted(historical)
                    self.assertEqual(held["dimensions"]["TEST.A"]["alignment"]["applicability"], "stale")
                    self.assertEqual(held["regression"]["affected_alignment_records"], [historical["formal_records"][0]["alignment"]["ref"]])
                for malformed in ({**packet["new"]["graph"]["ref"], "bytes": True},
                                  {**packet["new"]["graph"]["ref"], "extra": "TEST"},
                                  {key: value for key, value in packet["new"]["graph"]["ref"].items() if key != "git_blob"}):
                    broken = copy.deepcopy(packet)
                    self.update_alignment(broken["formal_records"][0], lambda review, malformed=malformed: review.update({"reviewed_graph": malformed}))
                    self.refused(broken)

    def test_native_expected_context_changes_only_kernel_not_statement_alignment(self):
        packet = self.with_formal()
        for field, value in (("repository", "TEST-other/repository"), ("commit", "c" * 40),
                             ("run-id", "124"), ("run-attempt", "2")):
            with self.subTest(field=field):
                report, _ = self.accepted(packet, expected={**self.expected, field: value})
                kernel = report["dimensions"]["TEST.A"]["kernel"]
                alignment = report["dimensions"]["TEST.A"]["alignment"]
                self.assertEqual(kernel["applicability"], "stale")
                self.assertIn("native_context_changed", kernel["reasons"])
                self.assertEqual(alignment["applicability"], "current")
                self.assertNotIn("native_context_changed", alignment["reasons"])
                self.assertEqual(report["regression"]["affected_alignment_records"], [])

    def test_deleted_formal_mapping_retains_original_review_and_holds_surviving_shared_mapping(self):
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        formal["node_targets"].append({"node": "TEST.B", "target": TARGETS[1]})
        def remove(graph):
            del graph["nodes"]["TEST.A"]
            graph["edges"] = []
        self.change_graph(packet, "new", remove)
        packet["new"]["bindings"] = [{"node": "TEST.B", "source": git_capture(SOURCE, "TEST/source.md"),
                                      "statement_slice": slice_of(SOURCE, len(ALPHA) + len(PROOF), len(SOURCE)),
                                      "proof_slice": None}]
        report, _ = self.accepted(packet)
        self.assertNotIn("TEST.A", report["dimensions"])
        self.assertIn("TEST.A", report["regression"]["changed_nodes"])
        self.assertNotIn("TEST.A", [item["node"] for item in report["regression"]["revalidation_required"]])
        self.assertEqual(report["formal_summary"][0]["node_targets"], formal["node_targets"])
        self.assertEqual(report["regression"]["affected_alignment_records"], [formal["alignment"]["ref"]])
        self.assertEqual(report["dimensions"]["TEST.B"]["alignment"]["applicability"], "stale")
        self.assertEqual(report["dimensions"]["TEST.B"]["kernel"]["applicability"], "current")
        self.assertEqual(report["dimensions"]["TEST.C"]["alignment"]["record_state"], "unknown")
        formal["node_targets"][0]["node"] = "TEST.NEVER_IN_EITHER_GRAPH"
        self.refused(packet)

    def test_elaborated_target_coverage_is_required_and_silent_checker_is_valid(self):
        packet = self.with_formal()
        self.replace_log(packet["formal_records"][0], "leanchecker.log", b"")
        positive, _ = self.accepted(packet)
        self.assertEqual(positive["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        complete = b"TEST.Formal.alpha : True\nTEST.Formal.beta : True\n"
        for raw in (b"", b" \n\t", b"TEST.Formal.alpha : True\n", complete + b"TEST.Formal.alpha : True\n",
                    complete + b"TEST.Formal.foreign : True\n", b"TEST.Formal.alphaSuffix : True\nTEST.Formal.beta : True\n",
                    b"TEST.Formal.alpha\nTEST.Formal.beta\n"):
            with self.subTest(raw=raw):
                missing = copy.deepcopy(packet)
                self.replace_log(missing["formal_records"][0], "elaborated-types.log", raw)
                report, _ = self.accepted(missing)
                self.assertNotEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        self.replace_log(packet["formal_records"][0], "elaborated-types.log",
                         b"TEST.Formal.alpha (TEST_parameter : Nat)\n  : True\nTEST.Formal.beta :\n  True\n")
        multiline, _ = self.accepted(packet)
        self.assertEqual(multiline["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")


    def test_elaborated_comment_only_types_are_incomplete_but_real_types_remain_current(self):
        positive_logs = (
            b"TEST.Formal.alpha : True -- TEST trailing comment\nTEST.Formal.beta : True -- TEST trailing comment\n",
            b"TEST.Formal.alpha : True /- TEST outer /- TEST nested -/ TEST tail -/\nTEST.Formal.beta : True /- TEST comment -/\n",
            b'TEST.Formal.alpha : ("TEST -- /- literal -/" = "TEST -- /- literal -/")\nTEST.Formal.beta : True\n',
            b"TEST.Formal.alpha (TEST_parameter : Nat)\n  : True /- TEST outer\n    /- TEST nested -/ TEST tail -/\nTEST.Formal.beta :\n  True -- TEST multiline type\n",
        )
        for raw in positive_logs:
            with self.subTest(kind="substantive_type", raw=raw):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                self.replace_log(formal, "leanchecker.log", b"")
                self.replace_log(formal, "elaborated-types.log", raw)
                report, _ = self.accepted(packet)
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        comment_bodies = (
            "-- TEST a comment is not a type",
            "/- TEST outer /- TEST nested -/ TEST tail -/",
            "\n  /- TEST outer\n    /- TEST nested -/\n    TEST tail -/",
        )
        for comment in comment_bodies:
            for incomplete in (set(TARGETS), {TARGETS[0]}, {TARGETS[1]}):
                with self.subTest(kind="comment_only_type", comment=comment, incomplete=sorted(incomplete)):
                    raw = "".join(target + " : " + (comment if target in incomplete else "True") + "\n"
                                  for target in TARGETS).encode("utf-8")
                    packet = self.with_formal()
                    formal = packet["formal_records"][0]
                    self.replace_log(formal, "leanchecker.log", b"")
                    self.replace_log(formal, "elaborated-types.log", raw)
                    report, _ = self.accepted(packet)
                    kernel = report["dimensions"]["TEST.A"]["kernel"]
                    self.assertEqual(kernel["applicability"], "unknown")
                    self.assertIn("target_inventory_incomplete", kernel["reasons"])
                    self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                    self.assertEqual(report["regression"]["affected_alignment_records"], [])

    def test_exact_successor_review_cannot_clear_shared_alignment_with_one_unresolved_main_join(self):
        base = copy.deepcopy(self.packet)
        for label, commit in (("old", BASE), ("new", HEAD)):
            for node, start, end in (("TEST.D", len(ALPHA) + len(PROOF), len(SOURCE)),
                                     ("TEST.E", 0, len(ALPHA))):
                self.change_graph(base, label, lambda graph, node=node: graph["nodes"].update({
                    node: {"classification": "PROVED_REVIEWED", "controlling": False}}))
                base[label]["bindings"].append({"node": node, "source": git_capture(SOURCE, "TEST/source.md", commit),
                                                "statement_slice": slice_of(SOURCE, start, end), "proof_slice": None})
        for graph_changed in (False, True):
            packet = copy.deepcopy(base)
            if graph_changed:
                self.change_graph(packet, "new", lambda graph: graph["nodes"]["TEST.A"].update({
                    "notes": "TEST successor node record requires explicit graph review"}))
            formal = self.formal(packet)
            self.update_alignment(formal, lambda review: review.update({
                "reviewed_graph": copy.deepcopy(packet["new"]["graph"]["ref"])}))
            shared = copy.deepcopy(formal)
            shared["id"] = "TEST-shared-survivor-formal"
            shared["node_targets"] = [{"node": "TEST.D", "target": TARGETS[1]}]
            separate = copy.deepcopy(formal)
            separate["id"] = "TEST-disjoint-formal"
            separate["node_targets"] = [{"node": "TEST.E", "target": TARGETS[0]}]
            separate["alignment"] = git_capture(raw_bytes(separate["alignment"]), "TEST/disjoint-alignment.json")
            packet["formal_records"] = [formal, shared, separate]
            with self.subTest(kind="all_joins_present", graph_changed=graph_changed):
                current, _ = self.accepted(packet)
                for node in ("TEST.A", "TEST.D", "TEST.E"):
                    self.assertEqual(current["dimensions"][node]["kernel"]["applicability"], "current")
                    self.assertEqual(current["dimensions"][node]["alignment"]["applicability"], "current")
                self.assertEqual(current["regression"]["affected_alignment_records"], [])
                self.assertEqual(bool(current["regression"]["impacted_nodes"]), graph_changed)
            if not graph_changed:
                continue
            for mutation in ("missing_source", "missing_statement", "different_origin"):
                changed = copy.deepcopy(packet)
                binding = changed["new"]["bindings"][0]
                if mutation == "missing_source":
                    binding["source"] = binding["statement_slice"] = binding["proof_slice"] = None
                elif mutation == "missing_statement":
                    binding["statement_slice"] = None
                else:
                    binding["source"] = git_capture(SOURCE, "TEST/unjoined-origin.md", "c" * 40, "TEST-other/origin")
                report, _ = self.accepted(changed)
                regression = report["regression"]
                with self.subTest(mutation=mutation, check="independent_kernel_and_disjoint_review"):
                    self.assertIn("TEST.A", regression["impacted_nodes"])
                    self.assertNotIn("TEST.D", regression["impacted_nodes"])
                    self.assertNotIn("TEST.E", regression["impacted_nodes"])
                    self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                    self.assertEqual(report["dimensions"]["TEST.D"]["kernel"]["applicability"], "current")
                    self.assertEqual(report["dimensions"]["TEST.E"]["kernel"]["applicability"], "current")
                    self.assertEqual(report["dimensions"]["TEST.E"]["alignment"]["applicability"], "current")
                    self.assertEqual(report["dimensions"]["TEST.C"]["alignment"]["applicability"], "unknown")
                for node in ("TEST.A", "TEST.D"):
                    with self.subTest(mutation=mutation, check="shared_whole_record_hold", node=node):
                        alignment = report["dimensions"][node]["alignment"]
                        self.assertEqual(alignment["record_state"], "recorded")
                        self.assertEqual(alignment["applicability"], "stale")
                        self.assertIn("alignment_record_revalidation_required", alignment["reasons"])
                with self.subTest(mutation=mutation, check="original_identity_affected_once"):
                    self.assertEqual(regression["affected_alignment_records"], [formal["alignment"]["ref"]])
                    self.assertEqual(changed["formal_records"][0]["alignment"], changed["formal_records"][1]["alignment"])
                    self.assertNotEqual(formal["alignment"]["ref"], separate["alignment"]["ref"])

    def test_graph_node_and_edge_admission_limits_cover_both_snapshots_and_alias_captures(self):
        def fixture(node_count, edge_count, *, label=None, aliases=False, invalid_classification=False):
            packet = copy.deepcopy(self.packet)
            graph = document(packet["new"]["graph"])
            for index in range(3, node_count):
                graph["nodes"]["TEST.LIMIT.%04d" % index] = {
                    "classification": "AUTHOR_SIDE_CANDIDATE", "controlling": False}
            graph["edges"] = [{"from": "TEST.A", "to": "TEST.B", "required": False,
                               "relation": "TEST limit context %04d" % index} for index in range(edge_count)]
            if invalid_classification:
                # Only graph data changes: resource refusal must precede gate classification.
                graph["nodes"]["TEST.A"]["classification"] = "TEST_UNSUPPORTED_CLASSIFICATION"
            raw = (json.dumps(graph, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            self.assertEqual(len(graph["nodes"]), node_count)
            self.assertEqual(len(graph["edges"]), edge_count)
            self.assertEqual(len({(edge["from"], edge["to"], edge["relation"]) for edge in graph["edges"]}), edge_count)
            self.assertLess(len(raw), 512 * 1024, "TEST limit fixture became a large timing probe")
            for snapshot, commit in (("old", BASE), ("new", HEAD)):
                if label is None or snapshot == label:
                    packet[snapshot]["graph"] = git_capture(raw, "TEST/graph.json", commit)
            if aliases:
                packet["old"]["graph"] = copy.deepcopy(packet["new"]["graph"])
                packet["old"]["bindings"][0]["source"] = copy.deepcopy(packet["new"]["bindings"][0]["source"])
            self.assertLess(len(encoded(packet)), 2 * 1024 * 1024, "TEST packet exceeded its small resource-control scope")
            return packet

        for nodes, edges, aliases in ((256, 0, False), (3, 2048, False), (256, 2048, False), (256, 2048, True)):
            with self.subTest(kind="exact_admitted_boundary", nodes=nodes, edges=edges, aliases=aliases):
                packet = fixture(nodes, edges, aliases=aliases)
                report, _ = self.accepted(packet)
                self.assertEqual(len(report["dimensions"]), nodes)
                self.assertEqual(report["regression"]["changed_nodes"], [])
                self.assertEqual(report["regression"]["impacted_nodes"], [])
        for nodes, edges, resource in ((257, 0, "node"), (3, 2049, "edge")):
            for label, aliases in (("old", False), ("new", False), (None, True)):
                for invalid_classification in (False, True):
                    with self.subTest(kind="resource_refusal", nodes=nodes, edges=edges, label=label,
                                      aliases=aliases, invalid_classification=invalid_classification):
                        packet = fixture(nodes, edges, label=label, aliases=aliases,
                                         invalid_classification=invalid_classification)
                        first = self.run_adapter(packet)
                        second = self.run_adapter(packet)
                        self.assertEqual((first.returncode, first.stdout, first.stderr),
                                         (second.returncode, second.stdout, second.stderr), "resource refusal was nondeterministic")
                        self.assertEqual(first.returncode, 1, "graph beyond the declared resource ABI was admitted")
                        self.assertEqual(first.stdout, "", "resource refusal emitted a valid report")
                        self.assertTrue(first.stderr.startswith("architecture evidence adapter refused: "))
                        self.assertRegex(first.stderr, r"graph.*" + resource + r".*limit",
                                         "resource admission did not precede the pinned graph gate")


    def test_main_negative_control_logs_must_be_present_under_exact_receipt_bound_names(self):
        names = ("TEST_false_claim", "TEST_second_claim", "sorry", "custom_imported", "native")
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        manifest = document(formal["manifest"])
        manifest["negative_controls"]["TEST_second_claim"] = copy.deepcopy(manifest["negative_controls"]["TEST_false_claim"])
        formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
        extra = run_capture(b"TEST/formal/TEST_second_claim.lean:2:22: error: proposition is false\n",
                            "TEST_second_claim.log")
        formal["logs"].append(extra)
        def additional_registered_control(receipt):
            receipt["manifest_sha256"] = formal["manifest"]["ref"]["sha256"]
            receipt["negative_controls"]["TEST_second_claim"] = "REJECTED_BY_LEAN"
            receipt["logs"]["TEST_second_claim.log"] = extra["ref"]["sha256"]
        self.update_receipt(formal, additional_registered_control)
        self.update_alignment(formal, lambda review: review.update({"manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
        self.replace_log(formal, "leanchecker.log", b"")
        positive, _ = self.accepted(packet)
        self.assertEqual(positive["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        self.assertEqual(positive["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
        for missing in tuple((name,) for name in names) + (names,):
            with self.subTest(kind="coherent_missing_control_logs", missing=missing):
                changed = copy.deepcopy(packet)
                formal = changed["formal_records"][0]
                filenames = {name + ".log" for name in missing}
                formal["logs"] = [capture for capture in formal["logs"]
                                  if capture["ref"]["member_path"].rsplit("/", 1)[1] not in filenames]
                self.update_receipt(formal, lambda receipt: [receipt["logs"].pop(name) for name in filenames])
                # Both inventories agree exactly: this is missing retained execution, not a bad hash.
                self.assertEqual(set(document(formal["receipt"])["negative_controls"]), set(names))
                report, _ = self.accepted(changed)
                kernel = report["dimensions"]["TEST.A"]["kernel"]
                self.assertEqual(kernel["record_state"], "recorded")
                self.assertEqual(kernel["applicability"], "unknown")
                self.assertIn("target_inventory_incomplete", kernel["reasons"])
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        for name in names:
            with self.subTest(kind="coherent_wrong_control_basename", name=name):
                changed = copy.deepcopy(packet)
                formal = changed["formal_records"][0]
                old_name, new_name = name + ".log", "TEST_other_" + name + ".log"
                capture = next(capture for capture in formal["logs"]
                               if capture["ref"]["member_path"].endswith("/" + old_name))
                capture["ref"]["member_path"] = "formal-evidence/" + new_name
                self.update_receipt(formal, lambda receipt: receipt["logs"].update(
                    {new_name: receipt["logs"].pop(old_name)}))
                report, _ = self.accepted(changed)
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        for name in names:
            with self.subTest(kind="negative_control_receipt_hash_contradiction", name=name):
                changed = copy.deepcopy(packet)
                self.update_receipt(changed["formal_records"][0], lambda receipt: receipt["logs"].update(
                    {name + ".log": "f" * 64}))
                self.refused(changed)

    def test_main_negative_control_logs_need_the_native_rejection_evidence(self):
        names = ("TEST_false_claim", "sorry", "custom_imported", "native")
        for raw in (b"", b" \n\t", b"TEST diagnostic without a proof rejection or axiom report\n"):
            for name in names:
                with self.subTest(kind="absent_rejection_body", name=name, raw=raw):
                    packet = self.with_formal()
                    formal = packet["formal_records"][0]
                    self.replace_log(formal, name + ".log", raw)
                    report, _ = self.accepted(packet)
                    self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                    self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                    self.assertEqual(report["regression"]["affected_alignment_records"], [])
        for name, raw in (("TEST_false_claim", b"'injected' depends on axioms: [sorryAx]\n"),
                          ("sorry", b"'injected' does not depend on any axioms\n"),
                          ("custom_imported", b"'injected' depends on axioms: [Classical.choice]\n"),
                          ("native", b"'injected' depends on axioms: [propext, Quot.sound]\n")):
            with self.subTest(kind="body_does_not_establish_declared_rejection_phase", name=name):
                packet = self.with_formal()
                self.replace_log(packet["formal_records"][0], name + ".log", raw)
                report, _ = self.accepted(packet)
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        # Main accepts each of these actual producer diagnostics; a successful checker may be silent.
        for raw in (b"TEST/formal/Demo.lean:2:22: error: unsolved goals\n\xe2\x8a\xa2 False\n",
                    b"TEST/formal/Demo.lean:2:22: error: proposition is false\n",
                    b"TEST/formal/Demo.lean:2:22: error: tactic 'trivial' failed\n"):
            with self.subTest(kind="native_proof_failure_marker", raw=raw):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                self.replace_log(formal, "TEST_false_claim.log", raw)
                self.replace_log(formal, "leanchecker.log", b"")
                report, _ = self.accepted(packet)
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        # Built-ins overwrite same-named registered mutations in the native producer.
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        manifest = document(formal["manifest"])
        manifest["negative_controls"]["sorry"] = copy.deepcopy(manifest["negative_controls"]["TEST_false_claim"])
        formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
        self.update_receipt(formal, lambda receipt: receipt.update({"manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
        self.update_alignment(formal, lambda review: review.update({"manifest_sha256": formal["manifest"]["ref"]["sha256"]}))
        overwritten, _ = self.accepted(packet)
        self.assertEqual(overwritten["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")

    def test_math_negative_control_log_inventory_keeps_its_distinct_native_names(self):
        packet = copy.deepcopy(self.packet)
        formal = self.math_formal(packet)
        packet["formal_records"] = [formal]
        capture = next(capture for capture in formal["logs"]
                       if capture["ref"]["member_path"].endswith("/TEST_false_claim.log"))
        capture["ref"]["member_path"] = "formal-evidence/false_fold.log"
        formal["logs"].append(run_capture(b"TEST/formal/false_power.lean:2:3: error: tactic 'norm_num' failed\n",
                                          "false_power.log"))
        def native_math_receipt(receipt):
            receipt["logs"]["false_fold.log"] = receipt["logs"].pop("TEST_false_claim.log")
            receipt["logs"]["false_power.log"] = formal["logs"][-1]["ref"]["sha256"]
            receipt["negative_controls"] = {"false_fold": "REJECTED_BY_LEAN", "false_power": "REJECTED_BY_LEAN",
                "sorry": "REJECTED_BY_AXIOM_GATE", "custom_imported": "REJECTED_BY_AXIOM_GATE",
                "native": "REJECTED_BY_AXIOM_GATE"}
        self.update_receipt(formal, native_math_receipt)
        self.replace_log(formal, "leanchecker.log", b"")
        report, _ = self.accepted(packet)
        retained = report["formal_summary"][0]
        self.assertNotIn("negative_controls", retained["retained_manifest"])
        self.assertNotIn("TEST_false_claim", retained["retained_receipt"]["negative_controls"])
        self.assertIn("false_fold.log", retained["retained_receipt"]["logs"])
        self.assertIn("false_power.log", retained["retained_receipt"]["logs"])
        self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
        self.assertEqual(report["regression"]["affected_alignment_records"], [])


    def _parity_rebind_manifest(self, formal, manifest):
        formal["manifest"] = replace_git(formal["manifest"], encoded(manifest))
        digest = formal["manifest"]["ref"]["sha256"]
        self.update_receipt(formal, lambda receipt: receipt.update({"manifest_sha256": digest}))
        self.update_alignment(formal, lambda review: review.update({"manifest_sha256": digest}))

    def _parity_replace_file(self, formal, path, raw):
        manifest = document(formal["manifest"])
        for index, capture in enumerate(formal["source_files"]):
            if capture["ref"]["path"] == path:
                formal["source_files"][index] = replace_git(capture, raw)
                key = path if formal["format"] == "main-formal-gate/v1" else path.removeprefix("TEST/formal/")
                manifest["files"][key] = sha(raw)
                self._parity_rebind_manifest(formal, manifest)
                return
        self.fail("TEST parity source file not found: " + path)

    def _parity_add_file(self, formal, path, raw):
        self.assertNotIn(path, {capture["ref"]["path"] for capture in formal["source_files"]})
        formal["source_files"].append(git_capture(raw, path))
        manifest = document(formal["manifest"])
        manifest["files"][path] = sha(raw)
        self._parity_rebind_manifest(formal, manifest)

    def _parity_omit_file(self, formal, path):
        formal["source_files"] = [capture for capture in formal["source_files"] if capture["ref"]["path"] != path]
        manifest = document(formal["manifest"])
        manifest["files"].pop(path)
        self._parity_rebind_manifest(formal, manifest)

    def _parity_main_current(self, packet):
        report, _ = self.accepted(packet)
        self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "current")
        self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
        self.assertEqual(report["regression"]["affected_alignment_records"], [])
        return report

    def _parity_main_refused(self, packet):
        result = self.run_adapter(packet)
        self.assertEqual(result.returncode, 1, "native source contradiction escaped: " + result.stdout + result.stderr)
        self.assertEqual(result.stdout, "", "native source contradiction emitted a valid report")
        self.assertTrue(result.stderr.startswith("architecture evidence adapter refused: "), result.stderr)

    def _parity_two_modules(self):
        packet = self.with_formal()
        formal = packet["formal_records"][0]
        self._parity_add_file(formal, "TEST/formal/Nested/Beta.lean",
                             b"namespace TEST.Other\ntheorem gamma : True := by trivial\n")
        self._parity_replace_file(formal, "TEST/formal/TEST.Formal.lean", b"import Demo\nimport Nested.Beta\n")
        manifest = document(formal["manifest"])
        manifest["source_modules"].append("Nested/Beta.lean")
        row = copy.deepcopy(manifest["targets"][-1])
        row.update({"name": "TEST.Other.gamma", "module": "Nested/Beta.lean"})
        manifest["targets"].append(row)
        self._parity_rebind_manifest(formal, manifest)
        self.update_receipt(formal, lambda receipt: receipt["axioms"].update({"TEST.Other.gamma": []}))
        self.replace_log(formal, "axioms.log",
                         b"'TEST.Formal.alpha' does not depend on any axioms\n"
                         b"'TEST.Formal.beta' does not depend on any axioms\n"
                         b"'TEST.Other.gamma' does not depend on any axioms\n")
        self.replace_log(formal, "elaborated-types.log",
                         b"TEST.Formal.alpha : True\nTEST.Formal.beta : True\nTEST.Other.gamma : True\n")
        self.update_alignment(formal, lambda review: review.update({"targets": TARGETS + ["TEST.Other.gamma"]}))
        return packet

    def test_main_original_version_log_is_required_and_matches_the_entire_trimmed_receipt_value(self):
        version = "Lean (version 4.34.1, TEST fixture only)"
        for case, raw, receipt_version in (
                ("V01_exact", (version + "\n").encode("utf-8"), version),
                ("V02_outer_whitespace", (" \t" + version + "\n\n ").encode("utf-8"), version),
                ("V03_whole_multiline", ("TEST diagnostic\n" + version + "\n").encode("utf-8"),
                 "TEST diagnostic\n" + version)):
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                self.replace_log(formal, "version.log", raw)
                self.replace_log(formal, "leanchecker.log", b"")
                self.update_receipt(formal, lambda receipt: receipt.update({"lean_version": receipt_version}))
                self._parity_main_current(packet)
        for case, raw in (("V04_missing", None), ("V05_empty", b""), ("V06_whitespace", b" \n\t")):
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                if raw is None:
                    formal["logs"] = [capture for capture in formal["logs"]
                                      if not capture["ref"]["member_path"].endswith("/version.log")]
                    self.update_receipt(formal, lambda receipt: receipt["logs"].pop("version.log"))
                else:
                    self.replace_log(formal, "version.log", raw)
                report, _ = self.accepted(packet)
                kernel = report["dimensions"]["TEST.A"]["kernel"]
                self.assertEqual(kernel["record_state"], "recorded")
                self.assertEqual(kernel["applicability"], "unknown")
                self.assertIn("target_inventory_incomplete", kernel["reasons"])
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])
        for case, raw in (("V07_nonempty_different_vendor", b"Lean (version 4.34.1, TEST different vendor)\n"),
                          ("V08_nonempty_different_version", b"Lean (version 4.34.2, TEST fixture only)\n")):
            with self.subTest(case=case):
                packet = self.with_formal()
                self.replace_log(packet["formal_records"][0], "version.log", raw)
                self._parity_main_refused(packet)

    def test_main_bound_dependency_lock_matches_native_package_names_and_revisions(self):
        valid = [{"name": "TEST_dep_a", "rev": "c" * 40}, {"name": "TEST_dep_b", "rev": "d" * 40}]
        for case, packages in (("L01_empty", []), ("L02_native_ignored_metadata", [{**valid[0], "TEST_extra": True}]),
                               ("L03_order_independent", list(reversed(valid)))):
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                self._parity_replace_file(formal, "TEST/formal/lake-manifest.json",
                                          encoded({"packages": packages, "TEST_lock_metadata": "inert"}))
                revisions = {package["name"]: package["rev"] for package in packages}
                manifest = document(formal["manifest"])
                manifest["dependency_revisions"] = revisions
                self._parity_rebind_manifest(formal, manifest)
                self.update_receipt(formal, lambda receipt: receipt.update({"dependency_revisions": revisions}))
                self._parity_main_current(packet)
        cases = (
            ("L04_changed_locked_revision", [{**valid[0], "rev": "e" * 40}], {"TEST_dep_a": "c" * 40}),
            ("L05_extra_locked_package", valid, {"TEST_dep_a": "c" * 40}),
            ("L06_missing_locked_package", [valid[0]], {"TEST_dep_a": "c" * 40, "TEST_dep_b": "d" * 40}),
            ("L07_duplicate_same_revision", [valid[0], valid[0]], {"TEST_dep_a": "c" * 40}),
            ("L08_duplicate_different_revision", [valid[0], {**valid[0], "rev": "e" * 40}], {"TEST_dep_a": "e" * 40}),
            ("L09_missing_capture", None, {}),
            ("L10_nonlist_packages", {}, {}),
            ("L11_nonobject_package", ["TEST malformed package"], {}),
            ("L12_missing_revision_key", [{"name": "TEST_dep_a"}], {}),
            ("L13_malformed_locked_revision", [{"name": "TEST_dep_a", "rev": "c" * 39}], {"TEST_dep_a": "c" * 40}),
        )
        for case, packages, revisions in cases:
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                if packages is None:
                    self._parity_omit_file(formal, "TEST/formal/lake-manifest.json")
                else:
                    self._parity_replace_file(formal, "TEST/formal/lake-manifest.json", encoded({"packages": packages}))
                manifest = document(formal["manifest"])
                manifest["dependency_revisions"] = revisions
                self._parity_rebind_manifest(formal, manifest)
                self.update_receipt(formal, lambda receipt: receipt.update({"dependency_revisions": revisions}))
                self._parity_main_refused(packet)

    def test_main_prohibited_proof_constructs_follow_native_comment_and_word_boundary_rules(self):
        base = b"namespace TEST.Formal\ntheorem alpha : True := by trivial\ntheorem beta : True := by trivial\n"
        positives = (
            ("P01_line_comments", b"-- sorry native_decide axiom TEST comment\n-- theorem ghost : False := by sorry\n" + base),
            ("P02_nested_comments", b"/- sorry /- native_decide\nnamespace TEST.Ghost\ntheorem ghost : False -/ axiom ignored -/\n" + base),
            ("P03_word_boundaries", base.replace(b"by trivial", b"by\n  let sorrySuffix := True\n  let native_decideSuffix := True\n  let axiomSuffix := True\n  trivial", 1)),
        )
        for case, raw in positives:
            with self.subTest(case=case):
                packet = self.with_formal()
                self._parity_replace_file(packet["formal_records"][0], "TEST/formal/Demo.lean", raw)
                self._parity_main_current(packet)
        for case, raw in (
                ("P04_sorry", base.replace(b"by trivial", b"by sorry", 1)),
                ("P05_native_decide", base.replace(b"by trivial", b"by native_decide", 1)),
                ("P06_custom_axiom", base + b"axiom hiddenPremise : False\n"),
                ("P07_indented_custom_axiom", base + b"  axiom hiddenPremise : False\n")):
            with self.subTest(case=case):
                packet = self.with_formal()
                self._parity_replace_file(packet["formal_records"][0], "TEST/formal/Demo.lean", raw)
                # Original declared axioms/types still look positive; actual retained source contradicts the gate.
                self._parity_main_refused(packet)

    def test_main_ordered_declarations_namespaces_and_root_imports_match_the_native_inventory(self):
        base = b"namespace TEST.Formal\ntheorem alpha : True := by trivial\ntheorem beta : True := by trivial\n"
        for case in ("D01_original", "D02_lemma", "D03_two_modules", "D04_comment_blank_root"):
            with self.subTest(case=case):
                packet = self._parity_two_modules() if case == "D03_two_modules" else self.with_formal()
                formal = packet["formal_records"][0]
                if case == "D02_lemma":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base.replace(b"theorem alpha", b"lemma alpha"))
                elif case == "D04_comment_blank_root":
                    self._parity_replace_file(formal, "TEST/formal/TEST.Formal.lean",
                        b"-- TEST ignored root comment\n\nimport Demo\n/- TEST outer /- TEST inner -/ tail -/\n\n")
                self._parity_main_current(packet)
        cases = ("D05_missing_declaration", "D06_renamed_declaration", "D07_reordered_declarations",
                 "D08_reordered_manifest_targets", "D09_missing_namespace", "D10_multiple_namespaces",
                 "D11_different_namespace", "D12_missing_root", "D13_extra_import", "D14_duplicate_import",
                 "D15_reordered_imports", "D16_root_declaration", "D17_indented_import",
                 "D18_trailing_import_whitespace", "D19_reordered_modules")
        for case in cases:
            with self.subTest(case=case):
                packet = self._parity_two_modules() if case in ("D15_reordered_imports", "D19_reordered_modules") else self.with_formal()
                formal = packet["formal_records"][0]
                if case == "D05_missing_declaration":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base.replace(b"theorem alpha : True := by trivial\n", b""))
                elif case == "D06_renamed_declaration":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base.replace(b"theorem alpha", b"theorem absent"))
                elif case == "D07_reordered_declarations":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean",
                        b"namespace TEST.Formal\ntheorem beta : True := by trivial\ntheorem alpha : True := by trivial\n")
                elif case == "D08_reordered_manifest_targets":
                    manifest = document(formal["manifest"])
                    manifest["targets"].reverse()
                    self._parity_rebind_manifest(formal, manifest)
                elif case == "D09_missing_namespace":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base.replace(b"namespace TEST.Formal\n", b""))
                elif case == "D10_multiple_namespaces":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base + b"namespace TEST.Extra\n")
                elif case == "D11_different_namespace":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean", base.replace(b"namespace TEST.Formal", b"namespace TEST.Other"))
                elif case == "D12_missing_root":
                    self._parity_omit_file(formal, "TEST/formal/TEST.Formal.lean")
                elif case in ("D13_extra_import", "D14_duplicate_import", "D15_reordered_imports",
                              "D16_root_declaration", "D17_indented_import", "D18_trailing_import_whitespace"):
                    roots = {
                        "D13_extra_import": b"import Demo\nimport TEST.Unregistered\n",
                        "D14_duplicate_import": b"import Demo\nimport Demo\n",
                        "D15_reordered_imports": b"import Nested.Beta\nimport Demo\n",
                        "D16_root_declaration": b"import Demo\ntheorem rootExtra : True := by trivial\n",
                        "D17_indented_import": b" import Demo\n",
                        "D18_trailing_import_whitespace": b"import Demo \n",
                    }
                    self._parity_replace_file(formal, "TEST/formal/TEST.Formal.lean", roots[case])
                else:
                    manifest = document(formal["manifest"])
                    manifest["source_modules"].reverse()
                    self._parity_rebind_manifest(formal, manifest)
                    self._parity_replace_file(formal, "TEST/formal/TEST.Formal.lean", b"import Nested.Beta\nimport Demo\n")
                self._parity_main_refused(packet)

    def test_math_source_and_version_records_keep_their_separate_historical_boundary(self):
        for case in ("M01_original_schema", "M02_no_main_namespace"):
            with self.subTest(case=case):
                packet = copy.deepcopy(self.packet)
                formal = self.math_formal(packet)
                packet["formal_records"] = [formal]
                if case == "M02_no_main_namespace":
                    self._parity_replace_file(formal, "TEST/formal/Demo.lean",
                        b"theorem alpha : True := by trivial\ntheorem beta : True := by trivial\n")
                report, _ = self.accepted(packet)
                retained = report["formal_summary"][0]
                self.assertNotIn("root_module", retained["retained_manifest"])
                self.assertNotIn("negative_controls", retained["retained_manifest"])
                self.assertNotIn("tools/formal_gate_check.py", retained["retained_manifest"]["files"])
                self.assertNotIn("version.log", retained["retained_receipt"]["logs"])
                self.assertEqual(report["dimensions"]["TEST.A"]["kernel"]["applicability"], "unknown")
                self.assertEqual(report["dimensions"]["TEST.A"]["alignment"]["applicability"], "current")
                self.assertEqual(report["regression"]["affected_alignment_records"], [])


    def test_main_required_control_file_floor_and_captured_toolchain_bytes_match_native_source_check(self):
        required = ("TEST/formal/lean-toolchain", "TEST/formal/lakefile.toml", "TEST/formal/lake-manifest.json",
                    "TEST/formal/TEST.Formal.lean", "TEST/formal/SCOPE.md", "TEST/formal/GLOSSARY.md",
                    "TEST/formal/README.md", "tools/formal_gate_check.py", "tests/test_formal_gate.py")
        packet = self.with_formal()
        self._parity_main_current(packet)
        for path in required:
            with self.subTest(case="F_missing_native_required_file", path=path):
                changed = copy.deepcopy(packet)
                self._parity_omit_file(changed["formal_records"][0], path)
                self._parity_main_refused(changed)
        for case, raw in (("F_wrong_toolchain", b"leanprover/lean4:v4.34.2\n"),
                          ("F_blank_toolchain", b" \n\t")):
            with self.subTest(case=case):
                changed = copy.deepcopy(packet)
                self._parity_replace_file(changed["formal_records"][0], "TEST/formal/lean-toolchain", raw)
                self._parity_main_refused(changed)
        with self.subTest(case="F_native_trimmed_toolchain_positive"):
            self._parity_replace_file(packet["formal_records"][0], "TEST/formal/lean-toolchain",
                                      b" \tleanprover/lean4:v4.34.1\n\n ")
            self._parity_main_current(packet)

    def test_main_registered_lean_suffix_and_visible_captured_lean_inventory_match_native_scope(self):
        packet = self.with_formal()
        self._parity_main_current(packet)
        for case, path in (("R_native_build_tree_exclusion", "TEST/formal/.lake/Hidden.lean"),
                           ("R_outside_package", "TEST/outside-package/Other.lean")):
            with self.subTest(case=case):
                changed = copy.deepcopy(packet)
                self._parity_add_file(changed["formal_records"][0], path,
                                      b"namespace TEST.Unrelated\ntheorem untouched : True := by trivial\n")
                self._parity_main_current(changed)
        for case in ("R_registered_nonlean_suffix", "R_bound_unregistered_lean", "R_unbound_declared_lean"):
            with self.subTest(case=case):
                changed = copy.deepcopy(packet)
                formal = changed["formal_records"][0]
                if case == "R_registered_nonlean_suffix":
                    capture = next(capture for capture in formal["source_files"]
                                   if capture["ref"]["path"] == "TEST/formal/Demo.lean")
                    raw = raw_bytes(capture)
                    self._parity_omit_file(formal, "TEST/formal/Demo.lean")
                    self._parity_add_file(formal, "TEST/formal/Demo.txt", raw)
                    manifest = document(formal["manifest"])
                    manifest["source_modules"] = ["Demo.txt"]
                    for target in manifest["targets"]:
                        target["module"] = "Demo.txt"
                    manifest["negative_controls"]["TEST_false_claim"]["module"] = "Demo.txt"
                    self._parity_rebind_manifest(formal, manifest)
                    self._parity_replace_file(formal, "TEST/formal/TEST.Formal.lean", b"import Dem\n")
                elif case == "R_bound_unregistered_lean":
                    self._parity_add_file(formal, "TEST/formal/Unregistered.lean",
                                          b"namespace TEST.Unregistered\ntheorem extra : True := by trivial\n")
                else:
                    path = "TEST/formal/DeclaredUnbound.lean"
                    formal["source_files"].append(git_capture(b"namespace TEST.Unregistered\n", path))
                    manifest = document(formal["manifest"])
                    manifest["unbound_files"].append(path)
                    self._parity_rebind_manifest(formal, manifest)
                # Captured contradictions are visible; this makes no assertion about uncaptured repository files.
                self._parity_main_refused(changed)

    def test_main_origin_repository_policy_and_selected_utf8_sources_match_native_reads(self):
        def origin_packet(repository):
            packet = copy.deepcopy(self.packet)
            for label in ("old", "new"):
                packet[label]["bindings"][0]["source"] = git_capture(SOURCE, "TEST/source.md", "c" * 40, repository)
            packet["formal_records"] = [self.formal(packet)]
            return packet
        for repository in (REPO, "d6g8k5htny-coder/Math-", "d6g8k5htny-coder/query-",
                           "d6g8k5htny-coder/meta-framework", "d6g8k5htny-coder/Universal-Law-Workspace"):
            with self.subTest(case="O_native_public_origin", repository=repository):
                self._parity_main_current(origin_packet(repository))
        with self.subTest(case="O_syntactically_valid_nonpublic_origin"):
            self._parity_main_refused(origin_packet("TEST-other/nonpublic-source"))
        with self.subTest(case="O_invalid_utf8_pinned_source"):
            packet = copy.deepcopy(self.packet)
            for label, commit in (("old", BASE), ("new", HEAD)):
                packet[label]["bindings"][0]["source"] = git_capture(SOURCE + b"\xff", "TEST/source.md", commit)
            packet["formal_records"] = [self.formal(packet)]
            self._parity_main_refused(packet)
        for case, path in (("O_invalid_utf8_registered_module", "TEST/formal/Demo.lean"),
                           ("O_invalid_utf8_root_module", "TEST/formal/TEST.Formal.lean")):
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                capture = next(capture for capture in formal["source_files"] if capture["ref"]["path"] == path)
                self._parity_replace_file(formal, path, raw_bytes(capture) + b"\xff")
                self._parity_main_refused(packet)
        with self.subTest(case="O_hash_only_binary_file_is_inert"):
            packet = self.with_formal()
            self._parity_add_file(packet["formal_records"][0], "TEST/formal/opaque.bin", b"\x00\xff\xfeTEST")
            self._parity_main_current(packet)

    def test_main_registered_mutation_source_preconditions_hold_before_builtin_overrides(self):
        for case in ("N_original_multiple_occurrences", "N_native_ignored_metadata", "N_valid_builtin_override"):
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                manifest = document(formal["manifest"])
                spec = manifest["negative_controls"]["TEST_false_claim"]
                if case == "N_native_ignored_metadata":
                    spec["TEST_metadata"] = "native producer ignores this inert annotation"
                elif case == "N_valid_builtin_override":
                    manifest["negative_controls"]["native"] = copy.deepcopy(spec)
                self._parity_rebind_manifest(formal, manifest)
                self._parity_main_current(packet)
        cases = ("N_missing_module_capture", "N_missing_module_key", "N_missing_replace_key",
                 "N_missing_with_key", "N_absent_replace_text", "N_nonstring_replace",
                 "N_nonstring_with", "N_nonobject_spec", "N_overridden_builtin_bad_precondition",
                 "N_nonstring_module")
        for case in cases:
            with self.subTest(case=case):
                packet = self.with_formal()
                formal = packet["formal_records"][0]
                manifest = document(formal["manifest"])
                spec = manifest["negative_controls"]["TEST_false_claim"]
                if case == "N_missing_module_capture":
                    spec["module"] = "TEST.Missing.lean"
                elif case == "N_missing_module_key":
                    del spec["module"]
                elif case == "N_missing_replace_key":
                    del spec["replace"]
                elif case == "N_missing_with_key":
                    del spec["with"]
                elif case == "N_absent_replace_text":
                    spec["replace"] = "TEST_NEEDLE_ABSENT_FROM_CAPTURED_MODULE"
                elif case == "N_nonstring_replace":
                    spec["replace"] = 7
                elif case == "N_nonstring_with":
                    spec["with"] = False
                elif case == "N_nonobject_spec":
                    manifest["negative_controls"]["TEST_false_claim"] = "TEST malformed mutation"
                elif case == "N_overridden_builtin_bad_precondition":
                    manifest["negative_controls"]["native"] = {**spec, "replace": "TEST_NEEDLE_ABSENT_FROM_CAPTURED_MODULE"}
                else:
                    spec["module"] = ["Demo.lean"]
                self._parity_rebind_manifest(formal, manifest)
                # Plausible bound outcome/log labels do not bypass the actual generator's raw-source preconditions.
                self._parity_main_refused(packet)


if __name__ == "__main__":
    unittest.main()
