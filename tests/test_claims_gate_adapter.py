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
        # H-B3 is depended on by K3-THM-001 only among tip claims.
        new["nodes"]["Q0-C101-QUALITATIVE-RATE"]["fingerprint"] = "unrelated-mutation"
        impact = CGA.reverse_impact_between(old, new)
        self.assertIn("Q0-C101-QUALITATIVE-RATE", impact["changed_nodes"])
        # LIVE_ROOT with empty depends_on should not drag unrelated claims.
        self.assertNotIn("SIDE24-3D-AO48-OPR-045", impact["impacted"])
        self.assertNotIn("P15-A..D", impact["impacted"])

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


if __name__ == "__main__":
    unittest.main()
