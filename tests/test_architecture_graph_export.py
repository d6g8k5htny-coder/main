"""Contract controls for the read-only, source-bound graph exporter.

The CLI is the subject of these tests. Fixtures use copies of the dated graph
and gate; mutations are explicitly TEST provenance, never revised historical
records. This module imports no project code. Run it in the hosted ephemeral
runner, not on the owner's laptop.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "tools/proof_graph_export.py"
PINNED_SOURCE = ROOT / "docs/site/dependency-source"
SOURCE_FILES = ("GRAPH.json", "hard_gate.py", "PROVENANCE.json")
PINNED_GRAPH_SHA256 = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"
PINNED_GATE_SHA256 = "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8"
PINNED_PROVENANCE_SHA256 = "c54cd92d93b2cbc650e024f30a8762fbc277d847580245053bf0c415265872c5"
DIMENSION_KEYS = {"source", "review", "kernel", "computation", "alignment"}
REVIEW_FIELDS = (
    "review_source", "review_issue", "review_basis", "review_provider",
    "review_providers", "review_disposition",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# Observe module execution and the pinned validator without substituting a
# validator or changing either source file. The wrapper still invokes the CLI's
# __main__ entry, with exactly the same source and output arguments.
TRACE_CLI = r"""
import json
from pathlib import Path
import runpy
import sys

marker = Path(sys.argv[1])
gate = Path(sys.argv[2]).resolve()
tool = sys.argv[3]
sys.argv = sys.argv[3:]

def observe(frame, event, arg):
    if event != 'call' or frame.f_code.co_name not in ('<module>', 'validate_graph_fail_closed'):
        return
    if Path(frame.f_code.co_filename).resolve() == gate:
        with marker.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'function': frame.f_code.co_name, 'file': str(gate)}) + '\n')

sys.setprofile(observe)
runpy.run_path(tool, run_name='__main__')
"""


class ArchitectureGraphExportContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name in SOURCE_FILES:
            shutil.copyfile(PINNED_SOURCE / name, self.source / name)
        self.output = self.root / "export"
        self.marker = self.root / "gate-trace.jsonl"
        self.test_provenance_digest = None

    def graph(self):
        return json.loads((self.source / "GRAPH.json").read_text(encoding="utf-8"))

    def provenance(self):
        return json.loads((self.source / "PROVENANCE.json").read_text(encoding="utf-8"))

    def source_bytes(self):
        return {name: (self.source / name).read_bytes() for name in SOURCE_FILES}

    def save_test_graph(self, graph):
        data = (json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self.save_test_graph_bytes(data)

    def save_test_graph_bytes(self, data):
        """Bind changed test bytes without making a pinned-commit claim."""
        (self.source / "GRAPH.json").write_bytes(data)
        provenance = self.provenance()
        provenance.update({
            "object": "TEST-ARCHITECTURE-GRAPH-EXPORT-FIXTURE",
            "repository": "TEST/fixture-only",
            "commit": "0" * 40,
            "captured_at": "2026-10-08T00:00:00Z",
            "scope": "TEST ONLY: mutated graph fixture; no historical or scientific status claim.",
        })
        record = provenance["files"]["GRAPH.json"]
        record["bytes"] = len(data)
        record["sha256"] = sha256(data)
        record["git_blob"] = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        self.save_test_provenance(provenance)

    def save_test_provenance(self, provenance):
        self.save_test_provenance_bytes((json.dumps(provenance, indent=2) + "\n").encode("utf-8"))

    def save_test_provenance_bytes(self, data):
        (self.source / "PROVENANCE.json").write_bytes(data)
        # The caller pins alternate provenance out of band. This pin cannot
        # authorize a different executable gate from the reviewed gate pin.
        self.test_provenance_digest = sha256(data)

    def run_export(self, *, source=None, output=None, trace=False, pin_test_source=True):
        # An absent CLI must be an intentional RED assertion, not an import or
        # file-not-found error which could masquerade as a refusal test passing.
        self.assertTrue(EXPORTER.is_file(), "required exporter CLI tools/proof_graph_export.py is absent")
        source = self.source if source is None else source
        output = self.output if output is None else output
        arguments = [str(EXPORTER), "--source-dir", str(source), "--output", str(output)]
        if pin_test_source and self.test_provenance_digest is not None:
            arguments.extend(["--expected-provenance-sha256", self.test_provenance_digest])
        if trace:
            command = [sys.executable, "-B", "-c", TRACE_CLI, str(self.marker),
                       str(source / "hard_gate.py"), *arguments]
        else:
            command = [sys.executable, "-B", *arguments]
        env = os.environ.copy()
        env.update({
            "PATH": "", "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        })
        try:
            return subprocess.run(command, cwd=self.root, env=env, capture_output=True,
                                  text=True, timeout=20, check=False)
        except subprocess.TimeoutExpired:
            self.fail("graph export did not finish within 20 seconds")

    def exported_graph(self, result, *, output=None):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        graph_path = (self.output if output is None else output) / "graph.json"
        self.assertTrue(graph_path.is_file(), "successful export must create graph.json")
        return json.loads(graph_path.read_text(encoding="utf-8"))

    def assert_rejected(self, fragment, *, source=None, output=None, trace=False, pin_test_source=True):
        result = self.run_export(source=source, output=output, trace=trace, pin_test_source=pin_test_source)
        self.assertNotEqual(result.returncode, 0, "invalid source was accepted")
        self.assertRegex(result.stdout + result.stderr, fragment)
        graph_path = (self.output if output is None else output) / "graph.json"
        self.assertFalse(graph_path.exists(), "failed export left a success graph.json")
        return result

    def trace_events(self):
        if not self.marker.exists():
            return []
        return [json.loads(line) for line in self.marker.read_text(encoding="utf-8").splitlines()]

    def test_pinned_49_node_55_edge_export_is_deterministic_and_preserves_records(self):
        original = self.graph()
        provenance = self.provenance()
        before = self.source_bytes()
        self.assertEqual(len(original["nodes"]), 49)
        self.assertEqual(len(original["edges"]), 55)
        self.assertEqual(sha256(before["GRAPH.json"]), PINNED_GRAPH_SHA256)
        self.assertEqual(sha256(before["hard_gate.py"]), PINNED_GATE_SHA256)
        self.assertEqual(sha256(before["PROVENANCE.json"]), PINNED_PROVENANCE_SHA256)
        self.assertEqual(len(before["GRAPH.json"]), provenance["files"]["GRAPH.json"]["bytes"])
        self.assertEqual(len(before["hard_gate.py"]), provenance["files"]["hard_gate.py"]["bytes"])

        exported = self.exported_graph(self.run_export())
        self.assertEqual(set(exported), {"schema_version", "source", "nodes", "edges", "dimensions",
                                         "scientific_effect", "scientific_status_authority"})
        self.assertIs(type(exported["schema_version"]), int)
        self.assertEqual(exported["schema_version"], 1)
        self.assertEqual(exported["nodes"], original["nodes"])
        self.assertEqual(exported["edges"], original["edges"])
        self.assertEqual(exported["source"], {
            "repository": provenance["repository"], "commit": provenance["commit"],
            "captured_at": provenance["captured_at"],
            "graph_sha256": PINNED_GRAPH_SHA256, "gate_sha256": PINNED_GATE_SHA256,
        })
        self.assertEqual(exported["scientific_effect"], "NONE")
        self.assertIs(exported["scientific_status_authority"], False)
        first_bytes = (self.output / "graph.json").read_bytes()
        second_output = self.root / "second-export"
        self.exported_graph(self.run_export(output=second_output), output=second_output)
        self.assertEqual(first_bytes, (second_output / "graph.json").read_bytes())
        self.assertEqual(self.source_bytes(), before, "export mutated dated source records")

    def test_exact_source_bound_hard_gate_validator_is_called(self):
        self.exported_graph(self.run_export(trace=True))
        events = self.trace_events()
        validator_calls = [event for event in events if event["function"] == "validate_graph_fail_closed"]
        self.assertTrue(validator_calls, "export must reuse the pinned validate_graph_fail_closed")
        self.assertEqual({event["file"] for event in events}, {str((self.source / "hard_gate.py").resolve())})

    def test_dimensions_describe_only_explicit_recorded_metadata(self):
        original = self.graph()
        exported = self.exported_graph(self.run_export())
        self.assertEqual(set(exported["dimensions"]), set(original["nodes"]))
        for node_id, node in original["nodes"].items():
            with self.subTest(node=node_id):
                self.assertEqual(exported["dimensions"][node_id], {
                    "source": "recorded" if node.get("source") else "not_recorded",
                    "review": "recorded" if any(node.get(field) for field in REVIEW_FIELDS) else "not_recorded",
                    "kernel": "not_recorded", "computation": "not_recorded", "alignment": "not_recorded",
                })

    def test_classification_and_proof_review_text_cannot_manufacture_evidence(self):
        graph = self.graph()
        graph["nodes"]["TEST.metadata-only-lemma"] = {
            "classification": "PROVED_REVIEWED", "controlling": False,
            "kind": "lemma", "fingerprint": "TEST metadata, no evidence adapter",
            "notes": "Kernel checked; computation reproduced; alignment accepted; lemma exists.",
            "scope": "Metadata mentions Lean and review; no bound record has been consumed.",
        }
        graph["nodes"]["TEST.review-text"] = {
            "classification": "ENGINEERING_CONTROL", "controlling": False,
            "source": "TEST/proof/PROOF.md", "review_source": "TEST/reviews/REVIEW.md",
            "review_disposition": "KERNEL_CHECKED REPRODUCED ALIGNMENT_ACCEPTED",
            "review_basis": [{"verdict": "ACCEPT", "notes": "formal lemma exists"}],
            "kernel_status": "checked", "computation_status": "reproduced",
            "alignment_status": "accepted", "formal_target": "TEST.NonexistentLemma",
        }
        self.save_test_graph(graph)
        exported = self.exported_graph(self.run_export())
        self.assertEqual(exported["nodes"], graph["nodes"], "metadata must remain literal original records")
        self.assertEqual(exported["dimensions"]["TEST.metadata-only-lemma"], {
            "source": "not_recorded", "review": "not_recorded", "kernel": "not_recorded",
            "computation": "not_recorded", "alignment": "not_recorded",
        })
        self.assertEqual(exported["dimensions"]["TEST.review-text"], {
            "source": "recorded", "review": "recorded", "kernel": "not_recorded",
            "computation": "not_recorded", "alignment": "not_recorded",
        })
        self.assertEqual(set(exported["dimensions"]["TEST.metadata-only-lemma"]), DIMENSION_KEYS)
        self.assertNotIn("lemmas", exported, "graph metadata must not assert formal lemma existence")
        self.assertIs(exported["scientific_status_authority"], False)

    def test_graph_identity_is_checked_before_gate_import(self):
        data = (self.source / "GRAPH.json").read_bytes()
        # Same byte count forces an actual SHA check, rather than size-only validation.
        self.assertIn(b"Historical D3 carrier", data)
        (self.source / "GRAPH.json").write_bytes(data.replace(b"Historical D3 carrier", b"historical D3 carrier", 1))
        self.assert_rejected(r"(?i)(hash|sha256|identity|provenance)", trace=True)
        self.assertEqual(self.trace_events(), [], "gate was imported before graph identity validation")

    def test_gate_identity_is_checked_before_untrusted_gate_executes(self):
        data = (self.source / "hard_gate.py").read_bytes()
        canary = self.root / "untrusted-gate-ran"
        injection = ("\nfrom pathlib import Path\nPath(" + repr(str(canary)) + ").write_text('executed')\n").encode("utf-8")
        (self.source / "hard_gate.py").write_bytes(data + injection)
        self.assert_rejected(r"(?i)(hash|sha256|bytes|identity|provenance)", trace=True)
        self.assertFalse(canary.exists(), "unvalidated hard_gate.py executed")
        self.assertEqual(self.trace_events(), [], "gate was imported before its own identity validation")

    def test_same_size_gate_change_is_rejected_by_sha_before_import(self):
        data = (self.source / "hard_gate.py").read_bytes()
        self.assertIn(b"Fail-closed downstream", data)
        (self.source / "hard_gate.py").write_bytes(data.replace(b"Fail-closed downstream", b"fail-closed downstream", 1))
        self.assert_rejected(r"(?i)(hash|sha256|identity|provenance)", trace=True)
        self.assertEqual(self.trace_events(), [])

    def test_source_byte_counts_are_validated_even_when_sha_matches(self):
        original = self.provenance()
        for name in ("GRAPH.json", "hard_gate.py"):
            with self.subTest(file=name):
                provenance = json.loads(json.dumps(original))
                provenance["files"][name]["bytes"] += 1
                self.save_test_provenance(provenance)
                self.assert_rejected(r"(?i)(bytes|size|length|identity|provenance)", trace=True)
                self.assertEqual(self.trace_events(), [])

    def test_missing_required_dependency_is_rejected_by_pinned_hard_gate(self):
        graph = self.graph()
        edge = next(edge for edge in graph["edges"] if edge["required"])
        del graph["nodes"][edge["to"]]
        self.save_test_graph(graph)
        self.assert_rejected("edge references missing node", trace=True)
        self.assertIn("validate_graph_fail_closed", [event["function"] for event in self.trace_events()])

    def test_required_dependency_cycle_is_rejected_by_pinned_hard_gate(self):
        graph = self.graph()
        edge = next(edge for edge in graph["edges"] if edge["required"])
        graph["edges"].append({"from": edge["to"], "to": edge["from"],
                               "required": True, "relation": "TEST_required_cycle"})
        self.save_test_graph(graph)
        self.assert_rejected("required dependency cycle", trace=True)
        self.assertIn("validate_graph_fail_closed", [event["function"] for event in self.trace_events()])

    def test_context_only_cycle_is_allowed_without_changing_gate_semantics(self):
        graph = self.graph()
        left = "hist.rnu_env.py"
        right = "hist.CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip"
        graph["edges"].extend([
            {"from": left, "to": right, "required": False, "relation": "TEST_context_only"},
            {"from": right, "to": left, "required": False, "relation": "TEST_context_only"},
        ])
        self.save_test_graph(graph)
        exported = self.exported_graph(self.run_export(trace=True))
        self.assertEqual(exported["edges"], graph["edges"])
        self.assertEqual(exported["source"]["repository"], "TEST/fixture-only")
        self.assertEqual(exported["source"]["graph_sha256"], sha256((self.source / "GRAPH.json").read_bytes()))
        self.assertEqual(exported["source"]["gate_sha256"], PINNED_GATE_SHA256)
        self.assertIn("validate_graph_fail_closed", [event["function"] for event in self.trace_events()])

    def test_default_trust_rejects_self_consistent_changed_graph_and_provenance(self):
        graph = self.graph()
        graph["nodes"]["hist.CH-LIFT"]["notes"] = "TEST ONLY: changed metadata, no historical source claim"
        self.save_test_graph(graph)
        # Deliberately omit the caller's alternate pin: the default expected
        # provenance identity must be independently fixed by reviewed code.
        self.assert_rejected(r"(?i)provenance.*(hash|sha256|identity|mismatch)|expected.*provenance",
                             trace=True, pin_test_source=False)
        self.assertEqual(self.trace_events(), [])

    def test_caller_pinned_provenance_cannot_authorize_different_gate_code(self):
        canary = self.root / "self-authorized-gate-ran"
        data = (self.source / "hard_gate.py").read_bytes()
        injection = ("\nfrom pathlib import Path\nPath(" + repr(str(canary)) + ").write_text('executed')\n").encode("utf-8")
        data += injection
        (self.source / "hard_gate.py").write_bytes(data)
        provenance = self.provenance()
        provenance["object"] = "TEST-SELF-CONSISTENT-UNTRUSTED-GATE"
        record = provenance["files"]["hard_gate.py"]
        record["bytes"] = len(data)
        record["sha256"] = sha256(data)
        record["git_blob"] = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
        self.save_test_provenance(provenance)
        self.assert_rejected(r"(?i)(gate|hard_gate).*(pin|trust|hash|sha256|identity|mismatch)", trace=True)
        self.assertFalse(canary.exists(), "self-declared gate hash was treated as executable authority")
        self.assertEqual(self.trace_events(), [])

    def test_duplicate_graph_json_keys_are_rejected_after_identity_validation(self):
        data = (self.source / "GRAPH.json").read_bytes()
        data = data.replace(b'  "schema_version": 1,', b'  "schema_version": 1,\n  "schema_version": 1,', 1)
        self.save_test_graph_bytes(data)
        self.assert_rejected("duplicate JSON key")

    def test_duplicate_provenance_json_keys_are_rejected_before_gate_import(self):
        data = (self.source / "PROVENANCE.json").read_bytes()
        data = data.replace(b'  "schema_version": 1,', b'  "schema_version": 1,\n  "schema_version": 1,', 1)
        self.save_test_provenance_bytes(data)
        self.assert_rejected("duplicate JSON key", trace=True)
        self.assertEqual(self.trace_events(), [])

    def test_symlink_source_files_are_rejected_before_gate_import(self):
        for name in SOURCE_FILES:
            with self.subTest(file=name):
                path = self.source / name
                data = path.read_bytes()
                path.unlink()
                path.symlink_to(PINNED_SOURCE / name)
                try:
                    self.assert_rejected(r"(?i)symlink|symbolic link", trace=True)
                    self.assertEqual(self.trace_events(), [])
                finally:
                    path.unlink()
                    path.write_bytes(data)

    def test_symlink_source_directory_is_rejected(self):
        source_link = self.root / "linked-source"
        source_link.symlink_to(self.source, target_is_directory=True)
        self.assert_rejected(r"(?i)symlink|symbolic link", source=source_link, trace=True)
        self.assertEqual(self.trace_events(), [])

    def test_failed_export_preserves_an_earlier_valid_export(self):
        self.exported_graph(self.run_export())
        valid_bytes = (self.output / "graph.json").read_bytes()
        graph = self.graph()
        edge = next(edge for edge in graph["edges"] if edge["required"])
        graph["edges"].append({"from": edge["to"], "to": edge["from"],
                               "required": True, "relation": "TEST_required_cycle"})
        self.save_test_graph(graph)
        result = self.run_export()
        self.assertNotEqual(result.returncode, 0, "required cycle was accepted")
        self.assertRegex(result.stdout + result.stderr, "required dependency cycle")
        self.assertEqual((self.output / "graph.json").read_bytes(), valid_bytes,
                         "a failed replacement overwrote the earlier valid export")


if __name__ == "__main__":
    unittest.main()
