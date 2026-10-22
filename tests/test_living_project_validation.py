"""LANES-20261022 Lane 5: validation of living project records (node-status v0).

Scope (main#475 comment 6346276856, "Shared interface v0"):
records/node_status/<node_id>.json with fields
  schema_version, node_id, owner{id, evidence_url}|null, next_action,
  missing_input[], stops[], sources[{url, rev}], source_rev, observed_at,
  status in {fresh, stale, unknown}, scientific_effect == "NONE".

This module checks five fault classes:
  1. missing artifacts      - a record, or a repo path it references, is absent
  2. broken links           - malformed source/evidence URLs, sources without rev
  3. invalid dependencies   - depends_on names an unknown node, or forms a cycle
  4. conflicting owners     - two records claim one node_id with different owners,
                              or an owner without a PICKUP/assignment evidence_url
  5. stale projections      - a rendered projection (e.g. the atlas panel) whose
                              source_rev/observed_at disagrees with its record, or a
                              record marked "fresh" whose source_rev is not current

IMPORTANT: dependency validation is not proof verification. A record that passes
every check here says nothing about scientific validity; "status" is freshness
only. Nothing here grants credit, discharges an obligation or flips a flag.

`depends_on` is a Lane 5 proposal, NOT part of interface v0; it is validated only
when present. Until Lane 2 lands architecture/node-status.schema.json and
records/node_status/, every fixture below is self-contained and adoption against
the real merged records is PENDING (test_adoption_on_merged_records skips).

Run: python -B -S -m pytest tests/test_living_project_validation.py
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RECORDS_DIR = REPO / "records" / "node_status"

REQUIRED = {
    "schema_version", "node_id", "owner", "next_action", "missing_input",
    "stops", "sources", "source_rev", "observed_at", "status", "scientific_effect",
}
STATUSES = {"fresh", "stale", "unknown"}
GH_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/(issues|pull)/\d+(#issuecomment-\d+)?$|"
                    r"^https://github\.com/[\w.-]+/[\w.-]+/(blob|tree)/[0-9a-f]{7,40}/\S+$")
PICKUP_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/(issues|pull)/\d+#issuecomment-\d+$")
REV = re.compile(r"^[0-9a-f]{7,40}$")
UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate(records, *, repo_paths=frozenset(), current_revs=None, projections=None,
             expected_nodes=()):
    """Return a sorted list of (fault_class, node_id, detail). Empty == clean.

    Dependency validation is not proof verification.
    """
    current_revs = current_revs or {}
    projections = projections or {}
    errs = []
    by_node = {}
    for r in records:
        nid = r.get("node_id", "<missing node_id>")
        # schema shape
        missing = REQUIRED - set(r)
        if missing:
            errs.append(("schema", nid, "missing fields " + ",".join(sorted(missing))))
            continue
        if r["schema_version"] != 0:
            errs.append(("schema", nid, "schema_version != 0"))
        if r["status"] not in STATUSES:
            errs.append(("schema", nid, "bad status"))
        if r["scientific_effect"] != "NONE":
            errs.append(("schema", nid, "scientific_effect must be NONE"))
        if not UTC_Z.match(str(r["observed_at"])):
            errs.append(("schema", nid, "observed_at not UTC Z"))
        # owners
        own = r["owner"]
        if own is not None:
            if not own.get("evidence_url") or not PICKUP_URL.match(own["evidence_url"]):
                errs.append(("owner", nid, "owner without PICKUP/assignment evidence_url"))
        prev = by_node.get(nid)
        if prev is not None:
            a = (prev["owner"] or {}).get("id")
            b = (own or {}).get("id")
            errs.append(("owner", nid, f"conflicting records for one node ({a!r} vs {b!r})")
                        if a != b else ("owner", nid, "duplicate record for one node"))
        else:
            by_node[nid] = r
        # links
        if not r["sources"]:
            errs.append(("link", nid, "no sources"))
        for s in r["sources"]:
            if not GH_URL.match(str(s.get("url", ""))):
                errs.append(("link", nid, f"malformed source url {s.get('url')!r}"))
            if not REV.match(str(s.get("rev", ""))):
                errs.append(("link", nid, f"source without pinned rev {s.get('url')!r}"))
        # artifacts (repo-relative paths a record cites)
        for p in r.get("artifacts", []):
            if p not in repo_paths:
                errs.append(("artifact", nid, f"missing artifact {p}"))
        # staleness of the record itself
        cur = current_revs.get(nid)
        if r["status"] == "fresh" and cur is not None and cur != r["source_rev"]:
            errs.append(("stale", nid, "marked fresh but source_rev is not current"))
    for nid in expected_nodes:
        if nid not in by_node:
            errs.append(("artifact", nid, "expected record missing"))
    # dependencies
    for nid, r in by_node.items():
        for d in r.get("depends_on", []):
            if d not in by_node:
                errs.append(("dependency", nid, f"unknown dependency {d}"))
    state = {}

    def visit(n, stack):
        state[n] = 1
        for d in by_node[n].get("depends_on", []):
            if d not in by_node:
                continue
            if state.get(d) == 1:
                errs.append(("dependency", n, "cycle " + " -> ".join(stack + [n, d])))
            elif d not in state:
                visit(d, stack + [n])
        state[n] = 2

    for n in sorted(by_node):
        if n not in state:
            visit(n, [])
    # projections
    for nid, proj in projections.items():
        r = by_node.get(nid)
        if r is None:
            errs.append(("stale", nid, "projection for node with no record"))
            continue
        if proj.get("source_rev") != r["source_rev"]:
            errs.append(("stale", nid, "projection source_rev differs from record"))
        try:
            if datetime.fromisoformat(proj["observed_at"].replace("Z", "+00:00")) < \
               datetime.fromisoformat(r["observed_at"].replace("Z", "+00:00")):
                errs.append(("stale", nid, "projection older than record"))
        except (KeyError, ValueError):
            errs.append(("stale", nid, "projection without valid observed_at"))
    return sorted(errs)


# ---------------------------------------------------------------- fixtures
REV_A = "f3b27d3cae5c56c65c47187244073dd280b8a2cd"  # main @ read time (for fixture shape only)
REV_B = "a0d41af27789a0d9c1d2f1a7734e6df136e35772"
PICKUP = "https://github.com/d6g8k5htny-coder/main/issues/475#issuecomment-6346314047"


def _rec(nid, **kw):
    r = {
        "schema_version": 0, "node_id": nid, "owner": None,
        "next_action": "fixture", "missing_input": [], "stops": [],
        "sources": [{"url": "https://github.com/d6g8k5htny-coder/main/issues/475", "rev": REV_A}],
        "source_rev": REV_A, "observed_at": "2026-10-22T17:00:00Z",
        "status": "fresh", "scientific_effect": "NONE",
    }
    r.update(kw)
    return r


def clean():
    recs = [
        _rec("fx.a", owner={"id": "agent16", "evidence_url": PICKUP},
             artifacts=["docs/site/atlas.mjs"], depends_on=["fx.b"]),
        _rec("fx.b", depends_on=["fx.c"]),
        _rec("fx.c", status="unknown", missing_input=["source unreachable"]),
    ]
    return dict(records=recs, repo_paths=frozenset({"docs/site/atlas.mjs"}),
                current_revs={"fx.a": REV_A, "fx.b": REV_A},
                projections={"fx.a": {"source_rev": REV_A, "observed_at": "2026-10-22T17:00:00Z"}},
                expected_nodes=("fx.a", "fx.b", "fx.c"))


def _mut(f):
    def build():
        c = clean()
        c["records"] = copy.deepcopy(c["records"])
        c["projections"] = copy.deepcopy(c["projections"])
        f(c)
        return c
    return build


R = lambda c, i: c["records"][i]  # noqa: E731

FAULTS = {
    # missing artifacts
    "missing_artifact_path": ("artifact", _mut(lambda c: R(c, 0)["artifacts"].append("tools/validate_node_status.py"))),
    "missing_expected_record": ("artifact", _mut(lambda c: c["records"].pop(2) and c["records"][1].pop("depends_on"))),
    # broken links
    "malformed_source_url": ("link", _mut(lambda c: R(c, 1)["sources"][0].update(url="http://example.com/x"))),
    "source_without_rev": ("link", _mut(lambda c: R(c, 1)["sources"][0].update(rev="main"))),
    "no_sources": ("link", _mut(lambda c: R(c, 2).update(sources=[]))),
    # invalid dependencies
    "unknown_dependency": ("dependency", _mut(lambda c: R(c, 2).update(depends_on=["fx.nonexistent"]))),
    "dependency_cycle": ("dependency", _mut(lambda c: R(c, 2).update(depends_on=["fx.a"]))),
    "self_dependency": ("dependency", _mut(lambda c: R(c, 2).update(depends_on=["fx.c"]))),
    # conflicting owners
    "two_records_one_node": ("owner", _mut(lambda c: c["records"].append(
        _rec("fx.a", owner={"id": "agent9", "evidence_url": PICKUP})))),
    "owner_without_evidence": ("owner", _mut(lambda c: R(c, 1).update(owner={"id": "agent3", "evidence_url": None}))),
    "owner_evidence_not_pickup": ("owner", _mut(lambda c: R(c, 1).update(
        owner={"id": "agent3", "evidence_url": "https://github.com/d6g8k5htny-coder/main/issues/475"}))),
    # stale projections
    "projection_rev_mismatch": ("stale", _mut(lambda c: c["projections"]["fx.a"].update(source_rev=REV_B))),
    "projection_older_than_record": ("stale", _mut(lambda c: c["projections"]["fx.a"].update(observed_at="2026-10-21T00:00:00Z"))),
    "projection_without_record": ("stale", _mut(lambda c: c["projections"].update(
        {"fx.zz": {"source_rev": REV_A, "observed_at": "2026-10-22T17:00:00Z"}}))),
    "fresh_but_rev_moved": ("stale", _mut(lambda c: c["current_revs"].update({"fx.b": REV_B}))),
    # schema guards (freshness is never scientific validity)
    "scientific_effect_not_none": ("schema", _mut(lambda c: R(c, 0).update(scientific_effect="PROVED"))),
    "observed_at_not_utc": ("schema", _mut(lambda c: R(c, 0).update(observed_at="2026-10-22 12:00 CT"))),
    "missing_required_field": ("schema", _mut(lambda c: R(c, 1).pop("source_rev"))),
}


def test_clean_fixture_passes():
    assert validate(**clean()) == []


@pytest.mark.parametrize("name", sorted(FAULTS))
def test_planted_fault_is_rejected(name):
    cls, build = FAULTS[name]
    errs = validate(**build())
    assert errs, f"planted fault {name} was accepted"
    assert any(e[0] == cls for e in errs), (name, errs)


def test_every_fault_class_is_seeded():
    assert {c for c, _ in FAULTS.values()} >= {"artifact", "link", "dependency", "owner", "stale"}


def test_docstring_disclaims_proof_verification():
    assert "dependency validation is not proof verification" in __doc__.lower()


def test_adoption_on_merged_records():
    """Adoption against real records (Lane 2). Skips until records/node_status exists."""
    if not RECORDS_DIR.is_dir():
        pytest.skip("records/node_status/ not merged yet; adoption PENDING (Lane 2)")
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RECORDS_DIR.glob("*.json"))]
    assert recs, "records/node_status/ exists but is empty"
    for p, r in zip(sorted(RECORDS_DIR.glob("*.json")), recs):
        assert p.stem == r.get("node_id"), f"{p.name} does not match node_id"
    paths = frozenset(str(p.relative_to(REPO)) for p in REPO.rglob("*") if p.is_file())
    assert validate(recs, repo_paths=paths) == []
