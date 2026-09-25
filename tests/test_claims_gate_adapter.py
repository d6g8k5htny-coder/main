"""Negative controls for the claims→gate #90 adapter (D7 / main #95 follow-on).

Scientific effect: NONE. Proposals are HOLD/REVALIDATION only; never promotion.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "claims_gate_adapter", ROOT / "tools" / "claims_gate_adapter.py"
)
assert SPEC and SPEC.loader
CGA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGA)


def _load_tip_claims() -> dict:
    return json.loads((ROOT / "claims" / "graph.json").read_text(encoding="utf-8"))


def _demote_controlling_hints(claims: dict) -> dict:
    """Sparse git fixtures omit drive/ mirrors; demote controlling hints so F2
    unresolved-controlling refusal does not mask edge-impact negative controls.
    """
    out = copy.deepcopy(claims)
    for bucket in ("claims", "premises"):
        for _nid, rec in (out.get(bucket) or {}).items():
            if not isinstance(rec, dict):
                continue
            grade = str(rec.get("grade") or "")
            if grade.strip().upper() in CGA.SOURCE_CONTROLLING_HINTS:
                rec["grade"] = "OPEN"
            for key in (
                "status_frozen_v2_2",
                "status_register_note",
                "scientific_status",
            ):
                val = rec.get(key)
                if isinstance(val, str) and val.strip().upper() in CGA.SOURCE_CONTROLLING_HINTS:
                    rec[key] = "OPEN"
            if rec.get("controlling") is True:
                rec["controlling"] = False
    return out


def _crosswalk() -> dict:
    return json.loads(
        (ROOT / "architecture" / "scientific_state" / "v1" / "ID_CROSSWALK.json").read_text(
            encoding="utf-8"
        )
    )


def _authority() -> dict:
    return json.loads(
        (ROOT / "architecture" / "scientific_state" / "v1" / "AUTHORITY_MAP.json").read_text(
            encoding="utf-8"
        )
    )


class ClaimsGateAdapterTests(unittest.TestCase):
    def test_tip_projection_includes_sub_obligations(self):
        report = CGA.audit_tip(ROOT)
        self.assertEqual(report["problems"], [])
        self.assertGreaterEqual(report["sub_obligation_edges"], 3)
        self.assertFalse(report["promotion_permission"])
        self.assertEqual(report["identity_impacted"], [])
        self.assertGreater(report["hold_node_count"], 0)
        self.assertEqual(report["hold_node_count"], len(report["hold_nodes"]))

    def test_deleted_depends_on_edge_still_impacts_via_union(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new_claims = copy.deepcopy(claims)
        # Delete D1-v2.2(2) → OBL-D1-PROMOTE edge.
        deps = list(new_claims["claims"]["D1-v2.2(2)"]["depends_on"])
        deps.remove("OBL-D1-PROMOTE")
        new_claims["claims"]["D1-v2.2(2)"]["depends_on"] = deps
        new = CGA.claims_to_gate_graph(new_claims)
        # Change the premise fingerprint so it is a changed node.
        new["nodes"]["OBL-D1-PROMOTE"]["fingerprint"] = "mutated"
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("D1-v2.2(2)", impact["impacted"])
        self.assertFalse(impact["promotion_permission"])
        self.assertTrue(
            any(p["proposal"] == "REVALIDATION_REQUIRED" for p in impact["proposals"])
        )

    def test_deleted_sub_obligation_edge_still_impacts_via_union(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new_claims = copy.deepcopy(claims)
        subs = list(new_claims["premises"]["OBL-D1-PROMOTE"]["sub_obligations"])
        self.assertIn("OBL-H5-JETMOD", subs)
        subs.remove("OBL-H5-JETMOD")
        new_claims["premises"]["OBL-D1-PROMOTE"]["sub_obligations"] = subs
        new = CGA.claims_to_gate_graph(new_claims)
        new["nodes"]["OBL-H5-JETMOD"]["fingerprint"] = "mutated-jetmod"
        impact = CGA.reverse_impact_between(old, new)
        # UNION edges retain OBL-D1-PROMOTE ← OBL-H5-JETMOD from old graph.
        self.assertIn("OBL-D1-PROMOTE", impact["impacted"])
        self.assertFalse(impact["promotion_permission"])

    def test_refuted_required_premise_forces_hold(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        # K3-THM-001 is REFUTED; synthesize a dependent that still requires it.
        graph["nodes"]["synthetic.dependent"] = {
            "bucket": "claims",
            "classification": "AUTHOR_SIDE_CANDIDATE",
            "controlling": False,
            "fingerprint": "syn",
            "version": claims["as_of"],
        }
        graph["edges"].append(
            {
                "from": "synthetic.dependent",
                "to": "K3-THM-001",
                "required": True,
                "relation": "depends_on",
            }
        )
        hold = CGA.required_holds(graph, "synthetic.dependent")
        self.assertIn("K3-THM-001", hold["refuted_required"])
        self.assertIn("HOLD", hold["proposals"])
        self.assertFalse(hold["promotion_permission"])

    def test_superseded_nonblocking_does_not_satisfy_required_premise(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        graph["nodes"]["prem.superseded"] = {
            "bucket": "premises",
            "classification": "SUPERSEDED_NONBLOCKING",
            "controlling": False,
            "fingerprint": "sup",
            "version": claims["as_of"],
        }
        graph["nodes"]["synthetic.dependent"] = {
            "bucket": "claims",
            "classification": "AUTHOR_SIDE_CANDIDATE",
            "controlling": False,
            "fingerprint": "syn",
            "version": claims["as_of"],
        }
        graph["edges"].append(
            {
                "from": "synthetic.dependent",
                "to": "prem.superseded",
                "required": True,
                "relation": "depends_on",
            }
        )
        hold = CGA.required_holds(graph, "synthetic.dependent")
        self.assertTrue(
            any(u["id"] == "prem.superseded" for u in hold["unsatisfied_required"]),
            hold,
        )
        self.assertIn("HOLD", hold["proposals"])
        self.assertFalse(hold["promotion_permission"])

    def test_unrelated_node_stable_when_lower_lemma_changes(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        # Mutate a leaf with no dependents in the projected graph where possible.
        # Self-hold includes the changed node; unrelated claims must stay clean.
        new["nodes"]["Q0-C101-QUALITATIVE-RATE"]["semantic_digest"] = "unrelated-mutation"
        new["nodes"]["Q0-C101-QUALITATIVE-RATE"]["source_snapshot"] = "unrelated-mutation"
        new["nodes"]["Q0-C101-QUALITATIVE-RATE"]["fingerprint"] = "unrelated-mutation"
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("Q0-C101-QUALITATIVE-RATE", impact["changed_nodes"])
        self.assertIn("Q0-C101-QUALITATIVE-RATE", impact["impacted"])  # self-hold
        # LIVE_ROOT with empty depends_on should not drag unrelated claims.
        self.assertNotIn("SIDE24-3D-AO48-OPR-045", impact["impacted"])
        self.assertNotIn("P15-A..D", impact["impacted"])

    def test_edge_only_change_is_impact_seed_without_node_fingerprint_change(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        # Delete one depends_on edge; leave every node fingerprint/snapshot alone.
        target = ("D1-v2.2(2)", "OBL-D1-PROMOTE", "depends_on", True)
        new["edges"] = [e for e in new["edges"] if CGA._edge_key(e) != target]
        self.assertEqual(len(new["edges"]), len(old["edges"]) - 1)
        # Node snapshots unchanged.
        for nid in ("D1-v2.2(2)", "OBL-D1-PROMOTE"):
            self.assertEqual(
                old["nodes"][nid]["source_snapshot"],
                new["nodes"][nid]["source_snapshot"],
            )
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("D1-v2.2(2)", impact["edge_only_seeds"])
        self.assertIn("OBL-D1-PROMOTE", impact["edge_only_seeds"])
        self.assertIn("D1-v2.2(2)", impact["impacted"])
        self.assertFalse(impact["promotion_permission"])

    def test_required_true_to_false_flip_is_edge_only_impact_seed(self):
        """Math- PR13 loss-only: required True→False seeds child without FP drift."""
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        flipped = False
        for edge in new["edges"]:
            if (
                edge["from"] == "D1-v2.2(2)"
                and edge["to"] == "OBL-D1-PROMOTE"
                and edge["relation"] == "depends_on"
            ):
                edge["required"] = False
                flipped = True
                break
        self.assertTrue(flipped)
        for nid in ("D1-v2.2(2)", "OBL-D1-PROMOTE"):
            self.assertEqual(
                old["nodes"][nid]["semantic_digest"],
                new["nodes"][nid]["semantic_digest"],
            )
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("D1-v2.2(2)", impact["edge_only_seeds"])
        self.assertIn("D1-v2.2(2)", impact["changed_nodes"])
        self.assertIn("D1-v2.2(2)", impact["impacted"])
        self.assertFalse(impact["promotion_permission"])

    def test_statement_change_detected_via_canonical_source_snapshot(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new_claims = copy.deepcopy(claims)
        # Mutate statement only; do not touch a manually curated fingerprint field.
        record = new_claims["premises"]["OBL-H5-JETMOD"]
        record["statement"] = (record.get("statement") or "") + " [loss-only statement drift]"
        new = CGA.claims_to_gate_graph(new_claims)
        self.assertNotEqual(
            old["nodes"]["OBL-H5-JETMOD"]["source_snapshot"],
            new["nodes"]["OBL-H5-JETMOD"]["source_snapshot"],
        )
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("OBL-H5-JETMOD", impact["changed_nodes"])
        self.assertIn("OBL-H5-JETMOD", impact["impacted"])  # self-hold
        self.assertIn("OBL-D1-PROMOTE", impact["impacted"])  # union parent

    def test_changed_controlling_node_included_in_revalidation_proposal(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        new["nodes"]["D1-v2.2(2)"]["semantic_digest"] = "self-hold-mutation"
        new["nodes"]["D1-v2.2(2)"]["source_snapshot"] = "self-hold-mutation"
        new["nodes"]["D1-v2.2(2)"]["fingerprint"] = "self-hold-mutation"
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("D1-v2.2(2)", impact["impacted"])
        self.assertTrue(
            any(
                p["node"] == "D1-v2.2(2)" and p["proposal"] == "REVALIDATION_REQUIRED"
                for p in impact["proposals"]
            ),
            impact["proposals"],
        )
        self.assertFalse(impact["promotion_permission"])

    def test_required_flag_rejects_non_boolean_fail_closed(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        for bad in (0, 1, "true", None):
            bad_graph = copy.deepcopy(graph)
            bad_graph["edges"][0]["required"] = bad
            with self.assertRaises(CGA.AdapterError) as ctx:
                CGA.validate_graph_fail_closed(bad_graph)
            self.assertIn("strict boolean", str(ctx.exception))

    def test_required_zero_cannot_silently_skip_premise(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        graph["nodes"]["prem.refuted"] = {
            "bucket": "premises",
            "classification": "REFUTED",
            "controlling": False,
            "source_snapshot": "r",
            "fingerprint": "r",
            "version": claims["as_of"],
        }
        graph["nodes"]["synthetic.dependent"] = {
            "bucket": "claims",
            "classification": "AUTHOR_SIDE_CANDIDATE",
            "controlling": False,
            "source_snapshot": "s",
            "fingerprint": "s",
            "version": claims["as_of"],
        }
        graph["edges"].append(
            {
                "from": "synthetic.dependent",
                "to": "prem.refuted",
                "required": 0,  # must not silently skip
                "relation": "depends_on",
            }
        )
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.required_holds(graph, "synthetic.dependent")
        self.assertIn("strict boolean", str(ctx.exception))

    def test_duplicate_edge_records_fail_closed(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        dup = {
            "from": "D1-v2.2(2)",
            "to": "OBL-D1-PROMOTE",
            "required": True,
            "relation": "depends_on",
        }
        graph["edges"].append(dup)
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.validate_graph_fail_closed(graph)
        self.assertIn("duplicate edge", str(ctx.exception))

    def test_unknown_dependency_id_fails_closed(self):
        claims = _load_tip_claims()
        claims = copy.deepcopy(claims)
        claims["claims"]["D1-v2.2(1)"]["depends_on"].append("NOT-A-REAL-ID")
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.claims_to_gate_graph(claims)
        self.assertIn("unknown dependency", str(ctx.exception))

    def test_ambiguous_canonical_owner_fails_closed(self):
        claims = _load_tip_claims()
        crosswalk = _crosswalk()
        authority = _authority()
        # Duplicate OBL-H5-JETMOD under a second authority.
        crosswalk = copy.deepcopy(crosswalk)
        crosswalk["rows"].append(
            {
                "row_id": "xwalk.obl-h5-jetmod-dup",
                "main_claim_or_premise_id": "OBL-H5-JETMOD",
                "main_bucket": "premises",
                "math_gate_id": "hist.OBL-H5-JETMOD",
                "authority": "math_downstream_gate",
                "notes": "ambiguous duplicate",
            }
        )
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.claims_to_gate_graph(
                claims, crosswalk=crosswalk, authority_map=authority
            )
        self.assertIn("ambiguous canonical owner", str(ctx.exception))

    def test_missing_as_of_fails_closed_as_stale(self):
        claims = _load_tip_claims()
        claims = copy.deepcopy(claims)
        del claims["as_of"]
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.claims_to_gate_graph(claims)
        self.assertIn("as_of", str(ctx.exception))

    def test_required_cycle_fails_closed(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        # Introduce a cycle among synthetic nodes.
        graph["nodes"]["cyc.a"] = {
            "bucket": "premises",
            "classification": "OPEN_ACTIVE",
            "controlling": False,
            "fingerprint": "a",
            "version": "x",
        }
        graph["nodes"]["cyc.b"] = {
            "bucket": "premises",
            "classification": "OPEN_ACTIVE",
            "controlling": False,
            "fingerprint": "b",
            "version": "x",
        }
        graph["edges"].extend(
            [
                {"from": "cyc.a", "to": "cyc.b", "required": True, "relation": "depends_on"},
                {"from": "cyc.b", "to": "cyc.a", "required": True, "relation": "depends_on"},
            ]
        )
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.validate_graph_fail_closed(graph)
        self.assertIn("cycle", str(ctx.exception))

    def test_adapter_never_exposes_promotion_permission(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        hold = CGA.required_holds(graph, "D1-v2.2(2)")
        impact = CGA.reverse_impact_between(graph, copy.deepcopy(graph))
        self.assertFalse(hold["promotion_permission"])
        self.assertFalse(impact["promotion_permission"])
        self.assertFalse(graph.get("promotion_permission", True) is True and False)
        self.assertIs(graph["promotion_permission"], False)

    def test_compare_claims_files_surfaces_unsatisfied_in_hold_proposals(self):
        """Regression: aggregate must not drop OPEN_ACTIVE / AUTHOR_SIDE holds."""
        claims = _load_tip_claims()
        report = CGA.compare_claims_files(claims, claims)
        # D1-v2.2(2) requires OBL-D1-PROMOTE which maps to OPEN_ACTIVE — not REFUTED.
        self.assertIn("D1-v2.2(2)", report["hold_proposals"])
        hold = report["hold_proposals"]["D1-v2.2(2)"]
        self.assertTrue(hold["unsatisfied_required"])
        self.assertFalse(hold["refuted_required"])
        self.assertIn("HOLD", hold["proposals"])
        self.assertFalse(report["promotion_permission"])

    def test_aggregate_hold_includes_superseded_nonblocking_required(self):
        claims = _load_tip_claims()
        graph = CGA.claims_to_gate_graph(claims)
        graph["nodes"]["prem.superseded"] = {
            "bucket": "premises",
            "classification": "SUPERSEDED_NONBLOCKING",
            "controlling": False,
            "semantic_digest": "s",
            "source_snapshot": "s",
            "fingerprint": "s",
            "version": claims["as_of"],
        }
        graph["nodes"]["synthetic.dependent"] = {
            "bucket": "claims",
            "classification": "AUTHOR_SIDE_CANDIDATE",
            "controlling": False,
            "semantic_digest": "d",
            "source_snapshot": "d",
            "fingerprint": "d",
            "version": claims["as_of"],
        }
        graph["edges"].append(
            {
                "from": "synthetic.dependent",
                "to": "prem.superseded",
                "required": True,
                "relation": "depends_on",
            }
        )
        holds = CGA.aggregate_hold_proposals(graph)
        self.assertIn("synthetic.dependent", holds)
        self.assertTrue(
            any(u["id"] == "prem.superseded" for u in holds["synthetic.dependent"]["unsatisfied_required"]),
            holds["synthetic.dependent"],
        )
        self.assertIn("HOLD", holds["synthetic.dependent"]["proposals"])

    def test_audit_tip_reports_hold_inventory(self):
        report = CGA.audit_tip(ROOT)
        self.assertIn("hold_node_count", report)
        self.assertIn("hold_nodes", report)
        self.assertEqual(report["hold_node_count"], len(report["hold_nodes"]))
        self.assertIsInstance(report["hold_proposals"], dict)
        # Tip claims have no PROVED_REVIEWED premises; dependents with required
        # edges must show up as HOLD (fail-closed inventory visible to CI).
        self.assertGreater(report["hold_node_count"], 0)
        self.assertFalse(report["promotion_permission"])
        self.assertEqual(report["mode"], "tip_health")
        self.assertIn("NOT evidence", report["meaning"])

    def test_malformed_depends_on_containers_fail_closed(self):
        claims = _load_tip_claims()
        for bad in (False, 0, "", {}, None):
            mutated = copy.deepcopy(claims)
            mutated["claims"]["D1-v2.2(2)"]["depends_on"] = bad
            with self.assertRaises(CGA.AdapterError) as ctx:
                CGA.claims_to_gate_graph(mutated)
            self.assertIn("must be a list", str(ctx.exception))

    def test_malformed_sub_obligations_containers_fail_closed(self):
        claims = _load_tip_claims()
        for bad in (False, 0, "", {}, None):
            mutated = copy.deepcopy(claims)
            mutated["premises"]["OBL-D1-PROMOTE"]["sub_obligations"] = bad
            with self.assertRaises(CGA.AdapterError) as ctx:
                CGA.claims_to_gate_graph(mutated)
            self.assertIn("must be a list", str(ctx.exception))

    def test_malformed_as_of_values_fail_closed(self):
        claims = _load_tip_claims()
        for bad in (None, "", False, 0):
            mutated = copy.deepcopy(claims)
            mutated["as_of"] = bad
            with self.assertRaises(CGA.AdapterError) as ctx:
                CGA.claims_to_gate_graph(mutated)
            self.assertIn("as_of", str(ctx.exception))

    def test_cli_rejects_unknown_arguments(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "claims_gate_adapter.py"), "--before", "x"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_cli_compare_reports_deleted_depends_on_edge(self):
        """Entry-point sentinel: same compare command path CI uses for file mode."""
        import subprocess
        import sys
        import tempfile

        claims = _load_tip_claims()
        before = claims
        after = copy.deepcopy(claims)
        deps = list(after["claims"]["D1-v2.2(2)"]["depends_on"])
        deps.remove("OBL-D1-PROMOTE")
        after["claims"]["D1-v2.2(2)"]["depends_on"] = deps
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            before_path = tmp / "before.json"
            after_path = tmp / "after.json"
            report_path = tmp / "report.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "claims_gate_adapter.py"),
                    "compare",
                    "--before",
                    str(before_path),
                    "--after",
                    str(after_path),
                    "--write-report",
                    str(report_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["mode"], "path_compare")
            self.assertIn("D1-v2.2(2)", report["reverse_impact"]["impacted"])
            self.assertFalse(report["promotion_permission"])
            self.assertIn("blob_sha256", report["before_identity"])
            self.assertIn("blob_sha256", report["after_identity"])

    def test_cli_compare_reports_deleted_sub_obligation_edge(self):
        import subprocess
        import sys
        import tempfile

        claims = _load_tip_claims()
        after = copy.deepcopy(claims)
        subs = list(after["premises"]["OBL-D1-PROMOTE"]["sub_obligations"])
        subs.remove("OBL-H5-JETMOD")
        after["premises"]["OBL-D1-PROMOTE"]["sub_obligations"] = subs
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            before_path = tmp / "before.json"
            after_path = tmp / "after.json"
            before_path.write_text(json.dumps(claims), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "claims_gate_adapter.py"),
                    "compare",
                    "--before",
                    str(before_path),
                    "--after",
                    str(after_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertIn("OBL-D1-PROMOTE", report["reverse_impact"]["impacted"])

    def test_event_compare_fails_closed_on_missing_base(self):
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.resolve_event_refs(
                environ={"GITHUB_EVENT_NAME": "push"},
                event={"before": CGA.ZERO_SHA, "after": "abc" * 10 + "abcdefab"},
            )
        self.assertIn("all-zero", str(ctx.exception))

    def test_event_compare_uses_pull_request_base_head(self):
        before = "a" * 40
        after = "b" * 40
        b, a, name = CGA.resolve_event_refs(
            environ={"GITHUB_EVENT_NAME": "pull_request"},
            event={
                "pull_request": {
                    "base": {"sha": before},
                    "head": {"sha": after},
                }
            },
        )
        self.assertEqual((b, a, name), (before, after, "pull_request"))

    def test_cli_compare_refs_with_temporary_commits_reports_edge_delete(self):
        """Negative control at git-ref entry point (CI event-compare uses refs)."""
        import subprocess
        import sys
        import tempfile

        claims = _demote_controlling_hints(_load_tip_claims())
        after_claims = copy.deepcopy(claims)
        deps = list(after_claims["claims"]["D1-v2.2(2)"]["depends_on"])
        deps.remove("OBL-D1-PROMOTE")
        after_claims["claims"]["D1-v2.2(2)"]["depends_on"] = deps

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "test"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            # Minimal tree with claims + architecture files required by compare-refs.
            (repo / "claims").mkdir()
            arch = repo / "architecture" / "scientific_state" / "v1"
            arch.mkdir(parents=True)
            (repo / "claims" / "graph.json").write_text(
                json.dumps(claims), encoding="utf-8"
            )
            for name in ("ID_CROSSWALK.json", "AUTHORITY_MAP.json"):
                src = ROOT / "architecture" / "scientific_state" / "v1" / name
                (arch / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "before"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            before_ref = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (repo / "claims" / "graph.json").write_text(
                json.dumps(after_claims), encoding="utf-8"
            )
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "after-edge-delete"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            after_ref = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "claims_gate_adapter.py"),
                    "compare-refs",
                    "--before-ref",
                    before_ref,
                    "--after-ref",
                    after_ref,
                    "--repo-root",
                    str(repo),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertIn("D1-v2.2(2)", report["reverse_impact"]["impacted"])
            self.assertEqual(report["before_ref"], before_ref)
            self.assertIn("blob_sha256", report["before_identity"])
            self.assertFalse(report["promotion_permission"])

            result2 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "claims_gate_adapter.py"),
                    "event-compare",
                    "--repo-root",
                    str(repo),
                ],
                capture_output=True,
                text=True,
                env={
                    **{k: v for k, v in __import__("os").environ.items()},
                    "CLAIMS_GATE_BEFORE_REF": before_ref,
                    "CLAIMS_GATE_AFTER_REF": after_ref,
                },
            )
            self.assertEqual(result2.returncode, 0, result2.stdout + result2.stderr)
            report2 = json.loads(result2.stdout)
            self.assertEqual(report2["mode"], "event_compare")
            self.assertIn("D1-v2.2(2)", report2["reverse_impact"]["impacted"])


    def test_refuted_classification_preserved_on_revalidation(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        # K3-THM-001 is REFUTED in tip projection.
        self.assertEqual(new["nodes"]["K3-THM-001"]["classification"], "REFUTED")
        new["nodes"]["K3-THM-001"]["semantic_digest"] = "mutated-refuted"
        new["nodes"]["K3-THM-001"]["source_snapshot"] = "mutated-refuted"
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("K3-THM-001", impact["impacted"])
        preserved = impact["graph"]["nodes"]["K3-THM-001"]["classification"]
        self.assertEqual(preserved, "REFUTED")
        self.assertEqual(
            impact["graph"]["nodes"]["K3-THM-001"]["revalidation_proposal"],
            "REVALIDATION_REQUIRED",
        )

    def test_source_file_byte_drift_seeds_impact_without_claims_record_change(self):
        claims = _load_tip_claims()
        old = CGA.claims_to_gate_graph(claims)
        new = copy.deepcopy(old)
        old_sources = {
            "D1-v2.2(2)": {
                "kind": "blob",
                "reference": "docs/fake.md",
                "path": "docs/fake.md",
                "bytes": 3,
                "sha256": "aaa",
            }
        }
        new_sources = {
            "D1-v2.2(2)": {
                "kind": "blob",
                "reference": "docs/fake.md",
                "path": "docs/fake.md",
                "bytes": 4,
                "sha256": "bbb",
            }
        }
        impact = CGA.reverse_impact_between(
            old, new, old_sources=old_sources, new_sources=new_sources
        )
        self.assertIn("D1-v2.2(2)", impact["source_byte_seeds"])
        self.assertIn("D1-v2.2(2)", impact["impacted"])

    def test_external_source_marked_unresolved_not_invented(self):
        bound = CGA.bind_source_at_revision(
            ROOT,
            "HEAD",
            {"source": "https://example.com/theorem.pdf"},
        )
        self.assertEqual(bound["kind"], "external_unresolved")
        self.assertNotIn("sha256", bound)


    def test_write_report_refuses_in_repo_path(self):
        """Regression: in-tree artifacts/ must not pollute REPOSITORY_TOP_LEVEL."""
        import subprocess
        import sys
        import tempfile

        claims = _load_tip_claims()
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            before_path = tmp / "before.json"
            after_path = tmp / "after.json"
            before_path.write_text(json.dumps(claims), encoding="utf-8")
            after_path.write_text(json.dumps(claims), encoding="utf-8")
            in_repo = ROOT / "artifacts" / "should-not-be-created.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "claims_gate_adapter.py"),
                    "compare",
                    "--before",
                    str(before_path),
                    "--after",
                    str(after_path),
                    "--write-report",
                    str(in_repo),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("outside the repository", result.stdout)
            self.assertFalse(in_repo.exists())
            self.assertFalse((ROOT / "artifacts").exists())

    def test_cli_tip_health_labeled_not_transition_evidence(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "claims_gate_adapter.py"), "tip-health"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "tip_health")
        self.assertEqual(report["identity_impacted"], [])
        self.assertIn("NOT evidence", report["meaning"])


class FiveBoundaryEnforcementTests(unittest.TestCase):
    """OpenAI trial PR124 probe families — repaired behavior, not defect replay."""

    def _mini_graph(self):
        return {
            "as_of": "2026-09-25",
            "premises": {
                "P": {
                    "status_register_note": "OPEN",
                    "source": {"path": "proof.md"},
                }
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

    def _init_fixture(self, repo: Path, graph: dict, *, crosswalk=None, authority=None):
        import subprocess

        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "test"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        (repo / "claims").mkdir()
        arch = repo / "architecture" / "scientific_state" / "v1"
        arch.mkdir(parents=True)
        (repo / "claims" / "graph.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )
        cw = crosswalk or {
            "rows": [{"main_claim_or_premise_id": "P", "authority": "a"}]
        }
        au = authority or {
            "authorities": {"a": {}, "b": {}},
            "this_package": {"id": "adapter"},
        }
        (arch / "ID_CROSSWALK.json").write_text(
            json.dumps(cw, indent=2) + "\n", encoding="utf-8"
        )
        (arch / "AUTHORITY_MAP.json").write_text(
            json.dumps(au, indent=2) + "\n", encoding="utf-8"
        )
        for name in ("proof.md", "theorem.md", "unrelated.md", "mirror.md"):
            (repo / name).write_text(f"Synthetic {name} version 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "before"],
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

    def _commit_after(self, repo: Path, label: str = "after") -> str:
        import subprocess

        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", label, "--allow-empty"],
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

    def _event_compare(self, repo: Path, before: str, after: str):
        import os
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "claims_gate_adapter.py"),
                "event-compare",
                "--repo-root",
                str(repo),
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CLAIMS_GATE_BEFORE_REF": before,
                "CLAIMS_GATE_AFTER_REF": after,
            },
        )
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            report = None
        return result.returncode, report

    def test_strict_json_rejects_duplicate_keys_and_nonfinite(self):
        with self.assertRaises(CGA.AdapterError) as ctx:
            CGA.load_json_strict(
                '{"depends_on": ["P"], "depends_on": []}', where="dup"
            )
        self.assertIn("duplicate JSON key", str(ctx.exception))
        with self.assertRaises(CGA.AdapterError) as ctx2:
            CGA.load_json_strict('{"n": NaN}', where="nan")
        self.assertIn("nonfinite", str(ctx2.exception))

    def test_bind_all_source_bindings_and_mirror_does_not_shadow(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            g["premises"]["P"].pop("source")
            g["premises"]["P"]["source_bindings"] = [
                {"repo": "d6g8k5htny-coder/main", "path": "proof.md"}
            ]
            before = self._init_fixture(repo, g)
            (repo / "proof.md").write_text("Changed canonical source\n", encoding="utf-8")
            after = self._commit_after(repo, "bindings-edit")
            rc, report = self._event_compare(repo, before, after)
            self.assertEqual(rc, 0, report)
            self.assertTrue(report["transition_ok"])
            self.assertIn("P", report["reverse_impact"]["impacted"])
            self.assertIn("T", report["reverse_impact"]["impacted"])
            self.assertIn(
                report["new_sources"]["P"]["kind"], {"blob", "tree", "multi"}
            )
            fields = {b.get("field") for b in report["new_sources"]["P"]["bindings"]}
            self.assertTrue(any("source_bindings" in f for f in fields))

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            g["premises"]["P"]["mirror_path"] = "mirror.md"
            before = self._init_fixture(repo, g)
            (repo / "proof.md").write_text(
                "Changed second declared source\n", encoding="utf-8"
            )
            after = self._commit_after(repo, "shadow-edit")
            rc, report = self._event_compare(repo, before, after)
            self.assertEqual(rc, 0, report)
            self.assertIn("P", report["reverse_impact"]["impacted"])
            paths = {
                b.get("path")
                for b in report["new_sources"]["P"]["bindings"]
                if b.get("kind") in {"blob", "tree"}
            }
            self.assertIn("proof.md", paths)
            self.assertIn("mirror.md", paths)

    def test_illegal_controlling_over_refuted_fails_transition(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            g["premises"]["P"]["status_register_note"] = "REFUTED"
            before = self._init_fixture(repo, g)
            g2 = json.loads((repo / "claims" / "graph.json").read_text(encoding="utf-8"))
            g2["claims"]["T"].update(grade="LIVE_ROOT_THEOREM", controlling=True)
            (repo / "claims" / "graph.json").write_text(
                json.dumps(g2, indent=2) + "\n", encoding="utf-8"
            )
            after = self._commit_after(repo, "illegal-promote")
            rc, report = self._event_compare(repo, before, after)
            self.assertNotEqual(rc, 0, report)
            self.assertIsInstance(report, dict)
            self.assertFalse(report["transition_ok"])
            self.assertFalse(report["promotion_permission"])
            self.assertIn("T", report["controlling_impacted"])
            self.assertEqual(
                report["hold_proposals"]["T"]["refuted_required"], ["P"]
            )
            self.assertTrue(report["illegal_controlling_transitions"])

    def test_retained_controlling_with_changed_premise_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            g["claims"]["T"].update(grade="LIVE_ROOT_THEOREM", controlling=True)
            before = self._init_fixture(repo, g)
            (repo / "proof.md").write_text(
                "Changed load-bearing lemma\n", encoding="utf-8"
            )
            after = self._commit_after(repo, "premise-under-controlling")
            rc, report = self._event_compare(repo, before, after)
            self.assertNotEqual(rc, 0, report)
            self.assertFalse(report["transition_ok"])
            self.assertIn("T", report["controlling_impacted"])
            # Safe corrective counterpart: same premise edit under OPEN stays OK.
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            before = self._init_fixture(repo, g)
            (repo / "proof.md").write_text("Synthetic correction\n", encoding="utf-8")
            after = self._commit_after(repo, "safe-edit")
            rc, report = self._event_compare(repo, before, after)
            self.assertEqual(rc, 0, report)
            self.assertTrue(report["transition_ok"])
            self.assertEqual(set(report["reverse_impact"]["impacted"]), {"P", "T"})

    def test_crosswalk_owner_drift_seeds_impact(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            before = self._init_fixture(repo, g)
            arch = repo / "architecture" / "scientific_state" / "v1"
            cw = json.loads((arch / "ID_CROSSWALK.json").read_text(encoding="utf-8"))
            cw["rows"][0]["authority"] = "b"
            (arch / "ID_CROSSWALK.json").write_text(
                json.dumps(cw, indent=2) + "\n", encoding="utf-8"
            )
            after = self._commit_after(repo, "owner-drift")
            rc, report = self._event_compare(repo, before, after)
            self.assertEqual(rc, 0, report)
            self.assertIn("P", report["reverse_impact"]["impacted"])
            self.assertIn("P", report["authority_owner_seeds"])
            self.assertIn("old_crosswalk_identity", report)
            self.assertIn("blob_sha256", report["old_crosswalk_identity"])

    def test_mutable_refs_resolve_to_full_commit_ids(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            before = self._init_fixture(repo, g)
            after = self._commit_after(repo, "noop-after")
            subprocess.run(
                ["git", "branch", "review-before", before],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "branch", "review-after", after],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            rc, report = self._event_compare(
                repo, "review-before", "review-after"
            )
            self.assertEqual(rc, 0, report)
            self.assertEqual(report["base_commit"], before)
            self.assertEqual(report["head_commit"], after)
            self.assertRegex(report["base_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(report["head_commit"], r"^[0-9a-f]{40}$")

    def test_duplicate_json_dependency_key_rejected_at_event_boundary(self):
        import tempfile

        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw) / "repo"
            repo.mkdir()
            g = self._mini_graph()
            before = self._init_fixture(repo, g)
            path = repo / "claims" / "graph.json"
            text = json.dumps(g)
            mutated = text.replace(
                '"depends_on": ["P"]', '"depends_on": ["P"], "depends_on": []'
            )
            self.assertNotEqual(mutated, text)
            path.write_text(mutated + "\n", encoding="utf-8")
            after = self._commit_after(repo, "dup-key")
            rc, report = self._event_compare(repo, before, after)
            self.assertNotEqual(rc, 0, report)
            self.assertIsInstance(report, dict)
            self.assertIn("error", report)
            self.assertIn("duplicate JSON key", report["error"])


if __name__ == "__main__":
    unittest.main()
