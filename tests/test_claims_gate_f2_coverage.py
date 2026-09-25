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
                    }
                ]
            },
        )
        self.assertEqual(bound["kind"], "unsupported_cross_repo")
        self.assertFalse(CGA._source_binding_monitorable(bound))


if __name__ == "__main__":
    unittest.main()
