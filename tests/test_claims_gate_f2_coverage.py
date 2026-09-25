"""F2 source-coverage: controlling nodes must be byte-monitorable.

OpenAI combined-tip re-review AMEND on 6e3f774. Scientific effect: NONE.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "claims_gate_adapter", ROOT / "tools" / "claims_gate_adapter.py"
)
assert SPEC and SPEC.loader
CGA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGA)


def _fixture():
    return {
        "as_of": "2026-09-25",
        "premises": {
            "P": {"status_register_note": "OPEN", "source": {"path": "proof.md"}}
        },
        "claims": {
            "T": {
                "grade": "OPEN",
                "controlling": False,
                "depends_on": ["P"],
                "source": {"path": "theorem.md"},
            },
            "U": {"grade": "OPEN", "source": {"path": "unrelated.md"}},
        },
    }


class F2CoverageTests(unittest.TestCase):
    def _repo(self):
        raw = tempfile.TemporaryDirectory(prefix="f2-cov-")
        self.addCleanup(raw.cleanup)
        repo = Path(raw.name) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@example.invalid"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "t"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        arch = repo / "architecture" / "scientific_state" / "v1"
        arch.mkdir(parents=True)
        (arch / "ID_CROSSWALK.json").write_text('{"rows":[]}\n', encoding="utf-8")
        (arch / "AUTHORITY_MAP.json").write_text(
            '{"authorities":{}}\n', encoding="utf-8"
        )
        for name in ("proof.md", "theorem.md", "unrelated.md", "second.md"):
            (repo / name).write_text(f"Synthetic {name} v1\n", encoding="utf-8")
        return repo

    def _commit(self, repo: Path, claims: dict, label: str = "snap") -> str:
        (repo / "claims").mkdir(exist_ok=True)
        (repo / "claims" / "graph.json").write_text(
            json.dumps(claims, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", label],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _event(self, repo: Path, before: str, after: str):
        env = os.environ.copy()
        env["CLAIMS_GATE_BEFORE_REF"] = before
        env["CLAIMS_GATE_AFTER_REF"] = after
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                "-S",
                str(ROOT / "tools" / "claims_gate_adapter.py"),
                "event-compare",
                "--repo-root",
                str(repo),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(p.stdout)
        return p.returncode, report

    def test_tip_controlling_nodes_are_byte_monitorable(self):
        claims = json.loads((ROOT / "claims" / "graph.json").read_text(encoding="utf-8"))
        for nid in ("Q0-C101-QUALITATIVE-RATE", "D1-v2.2(1)"):
            rec = claims["claims"][nid]
            self.assertTrue(rec.get("source_bindings"), nid)
            bound = CGA.bind_source_at_revision(ROOT, "HEAD", rec)
            self.assertTrue(
                CGA._source_binding_monitorable(bound),
                f"{nid} kind={bound.get('kind')}",
            )

    def test_controlling_with_unresolved_prose_refused_at_event_cli(self):
        repo = self._repo()
        g = _fixture()
        g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            source="Some Prose Carrier.md with spaces",
        )
        before = self._commit(repo, g, "before")
        after = self._commit(repo, g, "after-same")
        rc, report = self._event(repo, before, after)
        self.assertNotEqual(rc, 0, report)
        self.assertFalse(report["transition_ok"])
        self.assertIn("T", report["unresolved_controlling_sources"])
        self.assertTrue(
            any(
                "UNRESOLVED_CONTROLLING_SOURCE" in (e.get("reasons") or [])
                for e in report["transition_errors"]
                if e.get("node") == "T"
            ),
            report["transition_errors"],
        )

    def test_controlling_bound_source_byte_change_refuses_until_demoted(self):
        repo = self._repo()
        g = _fixture()
        g["claims"]["T"].update(grade="LIVE_ROOT_THEOREM", controlling=True)
        before = self._commit(repo, g, "before")
        (repo / "proof.md").write_text("Changed load-bearing lemma\n", encoding="utf-8")
        after = self._commit(repo, copy.deepcopy(g), "byte-change")
        rc, report = self._event(repo, before, after)
        self.assertNotEqual(rc, 0, report)
        self.assertIn("T", report["controlling_impacted"])
        self.assertIn("P", report["reverse_impact"]["source_byte_seeds"])

        demoted = copy.deepcopy(g)
        demoted["claims"]["T"].update(grade="RETRACTED_TO_CANDIDATE", controlling=False)
        (repo / "proof.md").write_text("Changed load-bearing lemma\n", encoding="utf-8")
        # rebuild: before controlling, after demoted with same byte change
        repo2 = self._repo()
        before2 = self._commit(repo2, g, "before")
        (repo2 / "proof.md").write_text("Changed load-bearing lemma\n", encoding="utf-8")
        after2 = self._commit(repo2, demoted, "demoted")
        rc2, report2 = self._event(repo2, before2, after2)
        self.assertEqual(rc2, 0, report2)
        self.assertTrue(report2["transition_ok"])

    def test_multi_binding_second_source_change_seeds_impact(self):
        repo = self._repo()
        g = _fixture()
        g["premises"]["P"].pop("source", None)
        g["premises"]["P"]["source_bindings"] = [
            {"repo": CGA.CURRENT_REPO, "path": "proof.md"},
            {"repo": CGA.CURRENT_REPO, "path": "second.md"},
        ]
        g["claims"]["T"].update(grade="LIVE_ROOT_THEOREM", controlling=True)
        before = self._commit(repo, g, "before")
        (repo / "second.md").write_text("Only second carrier changed\n", encoding="utf-8")
        after = self._commit(repo, copy.deepcopy(g), "second-only")
        rc, report = self._event(repo, before, after)
        self.assertNotEqual(rc, 0, report)
        self.assertIn("P", report["reverse_impact"]["source_byte_seeds"])
        self.assertIn("T", report["controlling_impacted"])
        kinds = {b.get("kind") for b in report["new_sources"]["P"]["bindings"]}
        self.assertIn("blob", kinds)

    def test_cross_repo_binding_not_silently_bound_locally(self):
        bound = CGA.bind_source_at_revision(
            ROOT,
            "HEAD",
            {
                "source_bindings": [
                    {
                        "repo": "other-org/other-repo",
                        "path": "claims/graph.json",
                        "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    }
                ]
            },
        )
        self.assertEqual(bound["kind"], "unsupported_cross_repo")
        self.assertFalse(CGA._source_binding_monitorable(bound))
        cross = bound["bindings"][0]
        self.assertEqual(cross["declared_repo"], "other-org/other-repo")
        self.assertEqual(cross["declared_path"], "claims/graph.json")
        self.assertEqual(
            cross["declared_commit"], "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )
        self.assertEqual(
            cross["declared_hash"],
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        self.assertNotIn("sha256", cross)  # no local byte invention

    def test_tip_scientific_object_hashes_match_expected(self):
        claims = json.loads((ROOT / "claims" / "graph.json").read_text(encoding="utf-8"))
        q0 = CGA.bind_source_at_revision(
            ROOT, "HEAD", claims["claims"]["Q0-C101-QUALITATIVE-RATE"]
        )
        self.assertTrue(CGA._source_binding_monitorable(q0), q0)
        sci = [
            b
            for b in q0["bindings"]
            if b.get("role") == "scientific_object" and b.get("kind") == "blob"
        ]
        self.assertEqual(len(sci), 1)
        self.assertEqual(
            sci[0]["object_sha256"],
            "8c2ded652973f0e6232e76854e2505d8ed532afef1815e0ebc88966dab0706bc",
        )
        self.assertEqual(sci[0]["object_bytes"], 3957)
        self.assertTrue(sci[0]["object_hash_ok"])
        info = [b for b in q0["bindings"] if b.get("role") == "informational_carrier"]
        self.assertEqual(len(info), 1)
        self.assertFalse(CGA._binding_is_scientific_monitorable(info[0]))

        d1 = CGA.bind_source_at_revision(
            ROOT, "HEAD", claims["claims"]["D1-v2.2(1)"]
        )
        self.assertTrue(CGA._source_binding_monitorable(d1), d1)
        d1b = d1["bindings"][0]
        self.assertEqual(d1b["extraction_rule"], "frozen_body")
        self.assertEqual(
            d1b["object_sha256"],
            "490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6",
        )
        self.assertEqual(d1b["object_bytes"], 18311)
        self.assertTrue(d1b["object_hash_ok"])
        self.assertEqual(
            d1b["carrier_sha256"],
            "7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174",
        )

    def _d1_wrapper(self, frozen_body: str, wrapper_note: str = "wrapper v1") -> str:
        return (
            f"# D1 wrapper\n{wrapper_note}\n"
            f"BEGIN_FROZEN_BODY\n{frozen_body.rstrip(chr(10))}\nEND_FROZEN_BODY\n"
            f"trailer {wrapper_note}\n"
        )

    def test_e_altered_extracted_body_refuses_controlling(self):
        repo = self._repo()
        body_v1 = "theorem statement v1\n"
        body_v2 = "theorem statement CHANGED\n"
        (repo / "d1.md").write_text(self._d1_wrapper(body_v1), encoding="utf-8")
        expected = CGA._sha256_bytes(
            CGA.extract_scientific_bytes(
                (repo / "d1.md").read_bytes(), "frozen_body"
            )
        )
        g = _fixture()
        g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            source_bindings=[
                {
                    "repo": CGA.CURRENT_REPO,
                    "path": "d1.md",
                    "role": "scientific_object",
                    "extraction_rule": "frozen_body",
                    "expected_sha256": expected,
                    "mirror_freshness": "external_sync_obligation",
                }
            ],
        )
        g["claims"]["T"].pop("source", None)
        before = self._commit(repo, g, "before")
        (repo / "d1.md").write_text(self._d1_wrapper(body_v2), encoding="utf-8")
        after = self._commit(repo, copy.deepcopy(g), "body-change")
        rc, report = self._event(repo, before, after)
        self.assertNotEqual(rc, 0, report)
        self.assertIn("T", report["controlling_impacted"])
        # After state also has object_hash_mismatch vs declared expected.
        after_bound = report["new_sources"]["T"]
        self.assertFalse(CGA._source_binding_monitorable(after_bound), after_bound)

    def test_e_wrapper_only_change_does_not_seed_scientific_impact(self):
        repo = self._repo()
        body = "stable frozen theorem\n"
        (repo / "d1.md").write_text(
            self._d1_wrapper(body, "wrapper v1"), encoding="utf-8"
        )
        expected = CGA._sha256_bytes(
            CGA.extract_scientific_bytes(
                (repo / "d1.md").read_bytes(), "frozen_body"
            )
        )
        g = _fixture()
        g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            source_bindings=[
                {
                    "repo": CGA.CURRENT_REPO,
                    "path": "d1.md",
                    "role": "scientific_object",
                    "extraction_rule": "frozen_body",
                    "expected_sha256": expected,
                    "mirror_freshness": "external_sync_obligation",
                }
            ],
        )
        g["claims"]["T"].pop("source", None)
        before = self._commit(repo, g, "before")
        (repo / "d1.md").write_text(
            self._d1_wrapper(body, "wrapper ONLY changed"), encoding="utf-8"
        )
        after = self._commit(repo, copy.deepcopy(g), "wrapper-only")
        rc, report = self._event(repo, before, after)
        self.assertEqual(rc, 0, report)
        self.assertTrue(report["transition_ok"], report)
        self.assertNotIn("T", report["reverse_impact"].get("source_byte_seeds", []))
        self.assertEqual(
            report["old_sources"]["T"]["coverage_sha256"],
            report["new_sources"]["T"]["coverage_sha256"],
        )

    def test_e_informational_master_edit_does_not_seed_impact(self):
        repo = self._repo()
        (repo / "theorem.md").write_text("exact theorem object\n", encoding="utf-8")
        (repo / "master.md").write_text("master carrier v1\n", encoding="utf-8")
        expected = CGA._sha256_bytes(b"exact theorem object\n")
        g = _fixture()
        g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            source_bindings=[
                {
                    "repo": CGA.CURRENT_REPO,
                    "path": "theorem.md",
                    "role": "scientific_object",
                    "extraction_rule": "whole_file",
                    "expected_sha256": expected,
                    "mirror_freshness": "external_sync_obligation",
                },
                {
                    "repo": CGA.CURRENT_REPO,
                    "path": "master.md",
                    "role": "informational_carrier",
                    "mirror_freshness": "external_sync_obligation",
                },
            ],
        )
        g["claims"]["T"].pop("source", None)
        before = self._commit(repo, g, "before")
        (repo / "master.md").write_text("master carrier UNRELATED edit\n", encoding="utf-8")
        after = self._commit(repo, copy.deepcopy(g), "master-only")
        rc, report = self._event(repo, before, after)
        self.assertEqual(rc, 0, report)
        self.assertTrue(report["transition_ok"], report)
        self.assertNotIn("T", report["reverse_impact"].get("source_byte_seeds", []))

    def test_e_stale_or_unverified_freshness_refuses_controlling(self):
        repo = self._repo()
        (repo / "theorem.md").write_text("exact theorem object\n", encoding="utf-8")
        expected = CGA._sha256_bytes(b"exact theorem object\n")
        for freshness in ("unverified", "stale", "absent"):
            with self.subTest(freshness=freshness):
                g = _fixture()
                g["claims"]["T"].update(
                    grade="LIVE_ROOT_THEOREM",
                    controlling=True,
                    source_bindings=[
                        {
                            "repo": CGA.CURRENT_REPO,
                            "path": "theorem.md",
                            "role": "scientific_object",
                            "extraction_rule": "whole_file",
                            "expected_sha256": expected,
                            "mirror_freshness": freshness,
                        }
                    ],
                )
                g["claims"]["T"].pop("source", None)
                before = self._commit(repo, g, f"before-{freshness}")
                after = self._commit(repo, copy.deepcopy(g), f"after-{freshness}")
                rc, report = self._event(repo, before, after)
                self.assertNotEqual(rc, 0, report)
                self.assertFalse(report["transition_ok"])
                self.assertIn("T", report["unresolved_controlling_sources"])

    def test_e_same_carrier_precision_upgrade_is_coverage_repair(self):
        """Legacy path binding → frozen_body+expected on unchanged carrier is OK."""
        repo = self._repo()
        body = "stable frozen theorem\n"
        (repo / "d1.md").write_text(self._d1_wrapper(body), encoding="utf-8")
        expected = CGA._sha256_bytes(
            CGA.extract_scientific_bytes(
                (repo / "d1.md").read_bytes(), "frozen_body"
            )
        )
        before_g = _fixture()
        before_g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            source_bindings=[
                {"repo": CGA.CURRENT_REPO, "path": "d1.md", "role": "assembly_mirror"}
            ],
        )
        before_g["claims"]["T"].pop("source", None)
        before = self._commit(repo, before_g, "legacy-bind")
        after_g = copy.deepcopy(before_g)
        after_g["claims"]["T"]["source_bindings"] = [
            {
                "repo": CGA.CURRENT_REPO,
                "path": "d1.md",
                "role": "scientific_object",
                "extraction_rule": "frozen_body",
                "expected_sha256": expected,
                "mirror_freshness": "external_sync_obligation",
            }
        ]
        after = self._commit(repo, after_g, "precision-upgrade")
        rc, report = self._event(repo, before, after)
        self.assertEqual(rc, 0, report)
        self.assertTrue(report["transition_ok"], report)
        self.assertIn("T", report.get("coverage_repairs") or [])
        # Actual body drift after precision still refuses.
        (repo / "d1.md").write_text(
            self._d1_wrapper("CHANGED frozen theorem\n"), encoding="utf-8"
        )
        after2 = self._commit(repo, copy.deepcopy(after_g), "body-drift")
        rc2, report2 = self._event(repo, after, after2)
        self.assertNotEqual(rc2, 0, report2)
        self.assertIn("T", report2["controlling_impacted"])

    def test_e6_precision_upgrade_with_semantic_change_still_refuses(self):
        """E6: same-carrier precision upgrade must not mask statement/semantic drift."""
        repo = self._repo()
        body = "stable frozen theorem\n"
        (repo / "d1.md").write_text(self._d1_wrapper(body), encoding="utf-8")
        expected = CGA._sha256_bytes(
            CGA.extract_scientific_bytes(
                (repo / "d1.md").read_bytes(), "frozen_body"
            )
        )
        before_g = _fixture()
        before_g["claims"]["T"].update(
            grade="LIVE_ROOT_THEOREM",
            controlling=True,
            statement="original controlling statement",
            source_bindings=[
                {"repo": CGA.CURRENT_REPO, "path": "d1.md", "role": "assembly_mirror"}
            ],
        )
        before_g["claims"]["T"].pop("source", None)
        before = self._commit(repo, before_g, "legacy-bind")
        after_g = copy.deepcopy(before_g)
        after_g["claims"]["T"]["statement"] = "CHANGED controlling statement"
        after_g["claims"]["T"]["source_bindings"] = [
            {
                "repo": CGA.CURRENT_REPO,
                "path": "d1.md",
                "role": "scientific_object",
                "extraction_rule": "frozen_body",
                "expected_sha256": expected,
                "mirror_freshness": "external_sync_obligation",
            }
        ]
        after = self._commit(repo, after_g, "precision-plus-statement")
        rc, report = self._event(repo, before, after)
        self.assertNotEqual(rc, 0, report)
        self.assertFalse(report["transition_ok"], report)
        self.assertIn("T", report["controlling_impacted"])
        self.assertNotIn("T", report.get("coverage_repairs") or [])


if __name__ == "__main__":
    unittest.main()
