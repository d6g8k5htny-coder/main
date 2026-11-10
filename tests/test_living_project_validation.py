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
when present. Real records under records/node_status/ are checked with
tools/validate_node_status.py and architecture/node-status.schema.json.
Required fields and the source_rev = sources[0].rev rule are asserted on those
records. The evidence id pattern and owner/released_url consistency are asserted
only when a real record carries that data; otherwise the test skips with a
reason. A missing records/node_status directory, schema, or validator fails,
as an empty records directory does. Optional unexercised fields stay an
explicit skip that names the missing data. The module is unittest-only and
-O safe: checks use _require(), which raises AssertionError explicitly instead
of a bare assert, so `python -O` cannot erase them.

Run: python -B -S -m unittest tests.test_living_project_validation
     python -B -O -S -m unittest tests.test_living_project_validation
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RECORDS_DIR = REPO / "records" / "node_status"
SCHEMA_PATH = REPO / "architecture" / "node-status.schema.json"
VALIDATOR_PATH = REPO / "tools" / "validate_node_status.py"
PROJECTION_PATH = REPO / "docs" / "site" / "node-status.json"
# The three records landed by main#495.
LANE2_RECORD_NAMES = (
    "eng.d0-packet-allowlist-fix.json",
    "eng.hard-gate.json",
    "math.side24-coefficient.json",
)



class MandatoryInputMissing(Exception):
    """A mandatory input is absent. Raised, not asserted, so -O keeps it and it
    is never mistaken for an ordinary assertion failure."""


def _require(condition, *detail):
    """-O-safe check: raise AssertionError explicitly (a bare assert is erased by -O)."""
    if not condition:
        raise AssertionError(detail if detail else "check failed")


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


def _load_real_records(records_dir):
    """Load JSON records from records_dir.

    A missing directory and a directory missing a required record both raise
    MandatoryInputMissing. That is a raised outcome, not an assert, so python -O
    does not erase it. Optional fields are not decided here.
    """
    records_dir = Path(records_dir)
    if not records_dir.is_dir():
        raise MandatoryInputMissing("mandatory records directory is missing: " + str(records_dir))
    missing = [name for name in LANE2_RECORD_NAMES if not (records_dir / name).is_file()]
    if missing:
        raise MandatoryInputMissing("Lane 2 records missing from " + str(records_dir) + ": " + ", ".join(missing))
    paths = sorted(path for path in records_dir.glob("*.json") if path.is_file() and not path.is_symlink())
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def _require_real_records():
    return _load_real_records(RECORDS_DIR)


def _load_schema(path):
    path = Path(path)
    if not path.is_file():
        raise MandatoryInputMissing("mandatory node-status schema is missing: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _schema():
    return _load_schema(SCHEMA_PATH)


def _load_validator(path):
    path = Path(path)
    if not path.is_file():
        raise MandatoryInputMissing("mandatory node-status validator is missing: " + str(path))
    spec = importlib.util.spec_from_file_location("lane2_validate_node_status", path)
    if spec is None or spec.loader is None:
        raise MandatoryInputMissing("mandatory node-status validator is missing: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _main_validator():
    return _load_validator(VALIDATOR_PATH)


def _record_evidence_urls(record):
    found = []
    owner = record.get("owner")
    if isinstance(owner, dict):
        for field in ("evidence_url", "released_url"):
            if field in owner:
                found.append((field, owner[field]))
    carried = record.get("carried_forward")
    if isinstance(carried, list):
        for index, item in enumerate(carried):
            if isinstance(item, dict) and "resolution_url" in item:
                found.append((f"carried_forward[{index}].resolution_url", item["resolution_url"]))
    return found


def check_real_records_match_published_projection(records_dir=None, projection_path=None):
    """Shared by the real-record test and the synthetic boundary tests."""
    records = _load_real_records(RECORDS_DIR if records_dir is None else records_dir)
    projection_path = PROJECTION_PATH if projection_path is None else Path(projection_path)
    if not projection_path.is_file():
        raise MandatoryInputMissing("mandatory published projection is missing: docs/site/node-status.json")
    published = json.loads(projection_path.read_text(encoding="utf-8"))
    by_id = {record["node_id"]: record for _, record in records}
    _require(set(published) == set(by_id), sorted(set(published) ^ set(by_id)))
    for node_id, record in by_id.items():
        _require(published[node_id] == record, node_id)
        _require(published[node_id]["source_rev"] == record["source_rev"], node_id)
        _require(published[node_id]["observed_at"] == record["observed_at"], node_id)


class SyntheticValidatorTests(unittest.TestCase):
    def test_clean_fixture_passes(self):
        self.assertEqual(validate(**clean()), [])

    def test_every_fault_class_is_seeded(self):
        self.assertTrue({c for c, _ in FAULTS.values()} >= {"artifact", "link", "dependency", "owner", "stale"})

    def test_docstring_disclaims_proof_verification(self):
        self.assertIn("dependency validation is not proof verification", __doc__.lower())


class PlantedFaultTests(unittest.TestCase):
    """One generated test method per planted fault (was pytest.mark.parametrize)."""

    def _check_fault(self, name):
        cls, build = FAULTS[name]
        errs = validate(**build())
        self.assertTrue(errs, f"planted fault {name} was accepted")
        self.assertTrue(any(e[0] == cls for e in errs), (name, errs))


def _add_fault_test(name):
    def test(self):
        self._check_fault(name)
    test.__name__ = "test_planted_fault_is_rejected__" + name
    setattr(PlantedFaultTests, test.__name__, test)


for _name in sorted(FAULTS):
    _add_fault_test(_name)
del _name



# Independent Q9006 current-input expectations. These helpers use only stdlib,
# captured native originals and reviewed fixed paths; they perform no HTTP,
# invoke no production checker and create no source or owner data.
_Q9006_CODE = (
    "tools/node_status_live_check.py", "tools/node_status_refresh.py",
    "tools/node_status_v1.py", "tools/validate_node_status.py",
)
_Q9006_SCHEMAS = (
    "architecture/node-status.schema.json", "architecture/node-status-v1.schema.json",
)
_Q9006_MANIFEST = "architecture/node-status-inputs.json"
_Q9006_SNAPSHOT = "docs/site/current.json"
_Q9006_GRAPH = "docs/site/dependency-source/GRAPH.json"
_Q9006_BOOTSTRAP = (*_Q9006_CODE, *_Q9006_SCHEMAS, _Q9006_MANIFEST,
                    _Q9006_SNAPSHOT, _Q9006_GRAPH)
_Q9006_SEEDS = (
    "eng.d0-packet-allowlist-fix", "eng.hard-gate",
    "math.lifetime-remainder", "math.side24-coefficient",
)
_Q9006_ARTIFACTS = tuple("cbz/evidence/hard-gate-source-refresh-20261110/" + name
                       for name in ("README.md", "current-SOURCE_FILES.json", "observation.json"))
_Q9006_GRAPH_ID = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde",
    "path": "frontiers/downstream_gate_20260925/GRAPH.json",
    "blob": "220b56867f24c5ef6f55a3cce70490e9777cf3af", "bytes": 38753,
    "sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
}
_Q9006_PINNED = re.compile(
    r"https://github\.com/d6g8k5htny-coder/(main|Math-)/(blob|tree)/"
    r"([0-9a-f]{40})/([A-Za-z0-9_./-]+)\Z")
_Q9006_DISCUSSION = re.compile(
    r"https://github\.com/d6g8k5htny-coder/(main|Math-)/(issues|pull)/"
    r"[1-9][0-9]*(?:#issuecomment-[1-9][0-9]*)?\Z")
_Q9006_READBACK_LIMIT = 2 * 1024 * 1024


def _q9006_same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _q9006_same(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _q9006_same(a, b) for a, b in zip(left, right))
    return left == right


def _q9006_json(raw, limit):
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded original JSON required")
    text = raw.decode("utf-8")
    depth, quoted, escaped = 0, False, False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            _require(depth <= 64, "current JSON depth exceeds 64")
        elif character in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate current JSON key")
            result[key] = value
        return result

    def constant(_value):
        raise AssertionError("nonfinite current JSON constant")

    def floating(value):
        result = float(value)
        _require(result not in (float("inf"), float("-inf")), "nonfinite current JSON number")
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)


def _q9006_layout(raw):
    """Derive the finite current set independently of any production receipt."""
    manifest = _q9006_json(raw[_Q9006_MANIFEST], 65536)
    _require(type(manifest) is dict and set(manifest) == {
        "schema", "profile", "records", "schema_paths", "code_paths", "graph", "artifact_paths"},
        "closed current manifest required")
    _require(type(manifest["schema"]) is str and manifest["schema"] == "node-status-inputs.v2"
             and type(manifest["profile"]) is str
             and manifest["profile"] in ("legacy-four-v0", "graph-closure-v1"),
             "unsupported current manifest")
    expected_schemas = [{"record_version": version, "path": path}
                        for version, path in enumerate(_Q9006_SCHEMAS)]
    _require(_q9006_same(manifest["code_paths"], list(_Q9006_CODE))
             and _q9006_same(manifest["schema_paths"], expected_schemas)
             and _q9006_same(manifest["graph"], {
                 "snapshot_path": _Q9006_SNAPSHOT, "raw_path": _Q9006_GRAPH}),
             "manifest cannot select project imports, schemas or Graph paths")
    artifacts = [] if manifest["profile"] == "legacy-four-v0" else list(_Q9006_ARTIFACTS)
    _require(_q9006_same(manifest["artifact_paths"], artifacts), "artifact policy differs")
    graph_raw = raw[_Q9006_GRAPH]
    _require(len(graph_raw) == _Q9006_GRAPH_ID["bytes"]
             and hashlib.sha256(graph_raw).hexdigest() == _Q9006_GRAPH_ID["sha256"]
             and hashlib.sha1(b"blob " + str(len(graph_raw)).encode() + b"\0" + graph_raw).hexdigest()
             == _Q9006_GRAPH_ID["blob"], "reviewed pinned Graph bytes differ")
    graph = _q9006_json(graph_raw, 2_000_000)
    snapshot = _q9006_json(raw[_Q9006_SNAPSHOT], 2_000_000)
    _require(type(snapshot) is dict and type(snapshot.get("graph")) is dict
             and set(snapshot["graph"]) == {"identity", "data"}
             and _q9006_same(snapshot["graph"]["identity"], _Q9006_GRAPH_ID)
             and _q9006_same(snapshot["graph"]["data"], graph), "full snapshot Graph join differs")
    _require(type(graph) is dict and type(graph.get("nodes")) is dict
             and type(graph.get("edges")) is list, "Graph structure differs")
    nodes, edges = graph["nodes"], graph["edges"]
    _require(set(_Q9006_SEEDS) <= nodes.keys()
             and all(type(node) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", node)
                     and type(value) is dict for node, value in nodes.items()),
             "Graph seed/node scope differs")
    outgoing, seen = {node: [] for node in nodes}, set()
    for edge in edges:
        _require(type(edge) is dict and set(edge) == {"from", "to", "required", "relation"}
                 and type(edge["from"]) is str and edge["from"] in nodes
                 and type(edge["to"]) is str and edge["to"] in nodes
                 and type(edge["required"]) is bool
                 and type(edge["relation"]) is str and bool(edge["relation"]), "invalid Graph edge")
        key = (edge["from"], edge["to"], edge["required"], edge["relation"])
        _require(key not in seen, "duplicate Graph edge")
        seen.add(key)
        if edge["required"]:
            outgoing[edge["from"]].append(edge["to"])
    cohort, pending = set(), list(_Q9006_SEEDS)
    while pending:
        node = pending.pop()
        if node not in cohort:
            cohort.add(node)
            if manifest["profile"] == "graph-closure-v1":
                pending.extend(outgoing[node])
    version = 0 if manifest["profile"] == "legacy-four-v0" else 1
    rows = [{"node_id": node, "path": "records/node_status/" + node + ".json",
             "record_version": version} for node in sorted(cohort)]
    _require(type(manifest["records"]) is list
             and all(type(row) is dict and set(row) == {"node_id", "path", "record_version"}
                     and type(row["record_version"]) is int and row["record_version"] in (0, 1)
                     for row in manifest["records"])
             and _q9006_same(manifest["records"], rows), "manifest cohort/version differs from Graph")
    return manifest, rows, tuple(sorted((*_Q9006_BOOTSTRAP, *(row["path"] for row in rows), *artifacts)))


def _q9006_capture_current(root, identity):
    """Native local custody and independent expectations; no project imports or HTTP."""
    root = Path(root)
    _require(root.is_absolute() and root.resolve(strict=True) == root
             and all(not path.is_symlink() and path.is_dir() for path in (root, *root.parents)),
             "canonical current source root required")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0",
                       GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1")

    def git(*arguments):
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-c", "core.fsmonitor=false",
             "-c", "core.hooksPath=/dev/null", "-C", str(root), *arguments],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
        _require(result.returncode == 0, "native expectation source Git read failed")
        return result.stdout

    commit = git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    tree = git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    _require(identity["checked_commit"] == commit and re.fullmatch(r"[0-9a-f]{40}", commit)
             and re.fullmatch(r"[0-9a-f]{40}", tree)
             and git("rev-parse", "--show-toplevel").decode().strip() == str(root),
             "native expectation source cut differs")
    raw, files = {}, {}

    def capture(relative):
        _require(type(relative) is str and not relative.startswith("/")
                 and all(part not in ("", ".", "..") for part in relative.split("/")),
                 "noncanonical expectation path")
        limit = 65536 if relative == _Q9006_MANIFEST or relative.startswith("records/") else 2_000_000
        entry = git("ls-tree", "--full-tree", "-z", commit, "--", relative)
        _require(entry.count(b"\0") == 1 and entry.endswith(b"\0"), "one native path required")
        metadata, name = entry[:-1].split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split(" ")
        _require(name == relative.encode("ascii") and kind == "blob"
                 and mode in ("100644", "100755") and re.fullmatch(r"[0-9a-f]{40}", blob),
                 "native expectation path/mode/blob differs")
        size = git("cat-file", "-s", blob).decode("ascii").strip()
        _require(size.isdecimal() and int(size) <= limit, "native expectation object exceeds bound")
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptor = None
        try:
            parts = relative.split("/")
            for part in parts[:-1]:
                following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                os.close(directory)
                directory = following
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                     and before.st_size <= limit
                     and bool(before.st_mode & 0o111) == (mode == "100755"),
                     "physical expectation input mode/type/size differs")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = None
                body = stream.read(limit + 1)
                after = os.fstat(stream.fileno())
            current = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
            _require(all(getattr(before, field) == getattr(observed, field)
                         for observed in (after, current) for field in fields)
                     and len(body) == before.st_size <= limit, "expectation input changed during capture")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)
        _require(body == git("cat-file", "blob", blob)
                 and hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() == blob,
                 "expectation working/native bytes differ")
        raw[relative] = body
        files[relative] = {"git_blob": blob, "git_mode": mode,
                           "sha256": hashlib.sha256(body).hexdigest()}

    for path in _Q9006_BOOTSTRAP:
        capture(path)
    for path, blob in zip(_Q9006_SCHEMAS, (
            "2bad95d30bcb89ea686d1ac5a50d52113dd730be", "05f5dbf110497bbc89fd6030df2c21c1d8ae52ac")):
        _require(files[path]["git_blob"] == blob, "reviewed record schema changed")
    manifest, rows, paths = _q9006_layout(raw)
    expected_names = {row["path"].rsplit("/", 1)[1] for row in rows}
    directory = root / "records/node_status"
    _require(directory.is_dir() and not directory.is_symlink()
             and {path.name for path in directory.iterdir()} == expected_names,
             "complete physical current record cohort differs")
    for path in paths:
        if path not in raw:
            capture(path)
    records = {}
    for row in rows:
        record = _q9006_json(raw[row["path"]], 65536)
        _require(type(record) is dict and record.get("node_id") == row["node_id"]
                 and type(record.get("schema_version")) is int
                 and record["schema_version"] == row["record_version"]
                 and type(record.get("status")) is str and record["status"] in STATUSES
                 and record.get("scientific_effect") == "NONE"
                 and type(record.get("sources")) is list and bool(record["sources"]),
                 "captured current record/version/status differs")
        for source in record["sources"]:
            _require(type(source) is dict and set(source) == {"url", "rev"}
                     and type(source["url"]) is str and type(source["rev"]) is str,
                     "captured current source declaration differs")
        records[row["node_id"]] = record
    _require(set(files) == set(paths)
             and git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() == commit
             and git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip() == tree,
             "complete expectation set or native cut changed")
    return {"manifest": manifest, "records": records, "raw": raw, "files": files,
            "checked_commit": commit, "checked_tree": tree}


def _q9006_check_freshness(receipt, capture, identity):
    """Compare a v2 receipt with independent captured declarations, including multiplicity."""
    _require(type(receipt) is dict and set(receipt) == {
        "schema", "scope", "records_total", "record_versions", "fresh_candidates", "fresh_checked",
        "fresh_nodes", "nonfresh_records", "source_bindings", "manifest_binding", "input_sha256",
        "identity", "observed_heads", "source_observation_count", "source_observations",
        "pinned_graph_binding", "scientific_effect", "cbz_green"}, "closed v2 freshness receipt required")
    records, files = capture["records"], capture["files"]
    fresh = [record for record in records.values() if record["status"] == "fresh"]
    nonfresh = sorted(({"node_id": r["node_id"], "declared_status": r["status"]}
                       for r in records.values() if r["status"] != "fresh"), key=lambda row: row["node_id"])
    _require(receipt["schema"] == "cbz.node-status-live-check.v2"
             and receipt["scope"] == "NODE_RECORD_FRESHNESS_ONLY"
             and receipt["scientific_effect"] == "NONE" and receipt["cbz_green"] is False
             and _q9006_same(receipt["identity"], identity), "v2 freshness identity/scope differs")
    expected_versions = {node: record["schema_version"] for node, record in records.items()}
    _require(type(receipt["record_versions"]) is dict
             and all(type(version) is int and version in (0, 1)
                     for version in receipt["record_versions"].values())
             and _q9006_same(receipt["record_versions"], expected_versions)
             and type(receipt["records_total"]) is int and receipt["records_total"] == len(records)
             and bool(fresh) and type(receipt["fresh_candidates"]) is int
             and type(receipt["fresh_checked"]) is int
             and receipt["fresh_candidates"] == receipt["fresh_checked"] == len(fresh)
             and _q9006_same(receipt["fresh_nodes"], sorted(r["node_id"] for r in fresh))
             and _q9006_same(receipt["nonfresh_records"], nonfresh), "current versions/status partition differs")
    _require(_q9006_same(receipt["source_bindings"], files), "complete current binding set/values differ")
    expected_manifest = dict(path=_Q9006_MANIFEST, profile=capture["manifest"]["profile"],
                             bytes=len(capture["raw"][_Q9006_MANIFEST]), **files[_Q9006_MANIFEST])
    _require(_q9006_same(receipt["manifest_binding"], expected_manifest), "actual manifest binding differs")
    _require(_q9006_same(receipt["input_sha256"], {
        node + ".json": files["records/node_status/" + node + ".json"]["sha256"] for node in records}),
        "complete current record hashes differ")
    graph = _Q9006_GRAPH_ID
    expected_pinned = dict(scope="PINNED_MATH_GRAPH_OBJECT_IDENTITY_ONLY",
        repository=graph["repository"], commit=graph["commit"], path=graph["path"],
        git_kind="blob", git_mode="100644", git_blob=graph["blob"], bytes=graph["bytes"],
        sha256=graph["sha256"], snapshot_path=_Q9006_SNAPSHOT, raw_path=_Q9006_GRAPH,
        snapshot_git_blob=files[_Q9006_SNAPSHOT]["git_blob"],
        raw_git_blob=files[_Q9006_GRAPH]["git_blob"], full_data_equal=True, complete=True)
    _require(_q9006_same(receipt["pinned_graph_binding"], expected_pinned),
             "separate native pinned Graph join differs")
    requested = [(source["url"], source["rev"]) for record in fresh for source in record["sources"]]
    observations, heads = receipt["source_observations"], receipt["observed_heads"]
    _require(type(observations) is list and type(heads) is dict
             and all(type(row) is dict and type(row.get("url")) is str for row in observations),
             "retained native source observations required")
    _require(type(receipt["source_observation_count"]) is int
             and receipt["source_observation_count"] == len(observations) == len(requested)
             and Counter(row["url"] for row in observations) == Counter(url for url, _ in requested),
             "fresh source URL multiplicities differ")
    repositories = set()
    for row in observations:
        match = _Q9006_PINNED.fullmatch(row["url"])
        if match:
            repository, kind = match.group(1), match.group(2)
            repositories.add(repository)
            _require(type(heads.get(repository)) is str
                     and re.fullmatch(r"[0-9a-f]{40}", heads[repository])
                     and row.get("head") == heads[repository], "captured default-head binding differs")
            before, after = row.get("pinned_object"), row.get("current_object")
            _require(type(before) is list and type(after) is list and len(before) == len(after) == 3
                     and all(type(value) is str for value in before + after)
                     and before[0] == kind and before[1] in (
                         ("040000",) if kind == "tree" else ("100644", "100755"))
                     and re.fullmatch(r"[0-9a-f]{40}", before[2])
                     and before == after, "native cited object tuple differs")
            _require({revision for url, revision in requested if url == row["url"]} == {match.group(3)},
                     "pinned source revision differs from original URL")
        else:
            _require(_Q9006_DISCUSSION.fullmatch(row["url"])
                     and type(row.get("updated_at")) is str and UTC_Z.fullmatch(row["updated_at"])
                     and {revision for url, revision in requested if url == row["url"]}
                     == {row["updated_at"]}, "retained discussion revision differs")
    _require(set(heads) == repositories, "default-head repository coverage differs")


def _q9006_readback_names():
    names = []
    for mode in ("normal", "optimized"):
        names.extend(("node-status-live-" + mode + suffix, mode)
                     for suffix in (".json", ".log", "-exit.json"))
        names.extend(("q9006-real-record-" + mode + suffix, mode)
                     for suffix in (".json", ".stdout.log", ".stderr.log", ".exit.json"))
        names.extend(("q9006-original-" + mode + suffix, mode)
                     for suffix in ("-source-before.json", "-source-after.json", "-parent-exit.json",
                                    "-unittest.log", "-strict-guard.json"))
    return tuple(names)


def _q9006_readback_payload(originals, context, marker):
    """Preflight every original and its complete framing against ONE 2 MiB budget."""
    names = _q9006_readback_names()
    _require(type(originals) is dict and set(originals) == {name for name, _ in names},
             "complete original evidence inventory required")
    _require(type(marker) is str and re.fullmatch(r"[0-9a-f]{48}", marker),
             "workflow command marker required")
    parts = [("::stop-commands::" + marker + "\n").encode("ascii")]
    total = len(parts[0])
    for name, mode in names:
        raw = originals[name]
        _require(type(raw) is bytes, "original evidence must be bytes")
        raw.decode("utf-8")
        separator = b"" if raw.endswith(b"\n") else b"\n"
        metadata = {"file": name, "mode": mode, "context": context, "encoding": "utf-8",
                    "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "boundary_separator_bytes": len(separator)}
        body = json.dumps(metadata, sort_keys=True)
        pieces = [("Q9006_ORIGINAL_BYTES_BEGIN " + body + "\n").encode("utf-8"), raw, separator,
                  ("Q9006_ORIGINAL_BYTES_END " + body + "\n").encode("utf-8")]
        total += sum(map(len, pieces))
        _require(total <= _Q9006_READBACK_LIMIT, "aggregate original readback exceeds 2 MiB")
        parts.extend(pieces)
    parts.append(("::" + marker + "::\n").encode("ascii"))
    total += len(parts[-1])
    _require(total <= _Q9006_READBACK_LIMIT, "aggregate framed readback exceeds 2 MiB")
    return b"".join(parts)


class RealRecordTests(unittest.TestCase):
    def test_adoption_on_merged_records(self):
        """Real records/node_status files, checked by main's validator and schema."""
        records = _require_real_records()
        schema = _schema()
        mod = _main_validator()
        self.assertEqual(schema["required"], list(mod.REQUIRED))
        self.assertEqual(mod.check_schema_file(REPO), [])
        self.assertEqual(mod.validate_directory(REPO, RECORDS_DIR), [])
        names = {path.name for path, _ in records}
        self.assertEqual([name for name in LANE2_RECORD_NAMES if name not in names], [])
        ids = [record["node_id"] for _, record in records]
        self.assertEqual(len(ids), len(set(ids)))
        for path, record in records:
            self.assertEqual(path.stem, record.get("node_id"))

    def test_real_records_required_fields(self):
        records = _require_real_records()
        required = _schema()["required"]
        self.assertTrue(required)
        for path, record in records:
            missing = [name for name in required if name not in record]
            self.assertEqual(missing, [], path.name)

    def test_real_records_source_rev_equals_first_source(self):
        records = _require_real_records()
        mod = _main_validator()
        for path, record in records:
            sources = record.get("sources")
            self.assertTrue(isinstance(sources, list) and sources, path.name)
            self.assertEqual(record.get("source_rev"), sources[0].get("rev"), path.name)
            self.assertNotIn(",", str(record.get("source_rev")), path.name)
            self.assertEqual(mod._source_rev_problems(record, sources), [], path.name)

    def test_real_records_evidence_id_pattern(self):
        records = _require_real_records()
        found = [(path.name, label, url) for path, record in records for label, url in _record_evidence_urls(record)]
        if not found:
            self.skipTest(
                "no real record has evidence_url, released_url, or resolution_url; "
                "the evidence id pattern is not exercised on records/node_status"
            )
        mod = _main_validator()
        for name, label, url in found:
            self.assertTrue(mod.EVIDENCE_URL.fullmatch(url), (name, label, url))

    def test_real_records_owner_released_url_consistency(self):
        records = _require_real_records()
        exercised = [(path, record, record["owner"]) for path, record in records
                     if isinstance(record.get("owner"), dict)]
        if not exercised:
            self.skipTest(
                "every real record has owner null; owner/released_url consistency is not exercised "
                "(released_url must stay on the owner object and must not clear it)"
            )
        mod = _main_validator()
        for path, record, owner in exercised:
            problems = []
            mod.check_owner(record, problems)
            self.assertEqual(problems, [], path.name)
            if "released_url" in owner:
                self.assertIsInstance(record["owner"], dict)
                self.assertTrue("id" in owner and "evidence_url" in owner)
                self.assertTrue(mod.EVIDENCE_URL.fullmatch(owner["evidence_url"]))
                self.assertTrue(mod.EVIDENCE_URL.fullmatch(owner["released_url"]))

    def test_real_records_match_published_projection(self):
        check_real_records_match_published_projection()

    def test_real_records_dependencies_when_present(self):
        payloads = [record for _, record in _require_real_records()]
        self.assertTrue(any(record.get('depends_on') for record in payloads),
                        'current adoption requires actual dependency declarations')
        self.assertEqual([err for err in validate(payloads) if err[0] == "dependency"], [])

    def test_real_records_cited_artifacts_when_present(self):
        records = _require_real_records()
        cited = [artifact for _, record in records for artifact in record.get("artifacts", [])]
        self.assertTrue(cited, 'current adoption requires actual hard-gate artifact declarations')
        present = frozenset(str(path.relative_to(REPO)) for path in REPO.rglob("*") if path.is_file())
        self.assertEqual(sorted(artifact for artifact in cited if artifact not in present), [])

    def test_real_records_fresh_against_live_source_head(self):
        records = _require_real_records()

        def prepared(environment):
            flag = environment.get("NODE_STATUS_LIVE_REQUIRED")
            if flag is None:
                self.skipTest("native authenticated live shard prerequisites are absent; "
                              "set NODE_STATUS_LIVE_REQUIRED=1 with native context, token "
                              "and a new external NODE_STATUS_LIVE_OUTPUT")
            self.assertEqual(flag, "1", "invalid mandatory live-shard flag")
            self.assertEqual(environment.get("GITHUB_ACTIONS"), "true",
                             "native GitHub Actions context required")
            self.assertEqual(environment.get("GITHUB_REPOSITORY"),
                             "d6g8k5htny-coder/main")
            identity = {"repository": environment["GITHUB_REPOSITORY"],
                        "checked_commit": environment.get("GITHUB_SHA"),
                        "run_id": environment.get("GITHUB_RUN_ID"),
                        "run_attempt": environment.get("GITHUB_RUN_ATTEMPT"),
                        "python_optimize": sys.flags.optimize}
            self.assertIn(sys.flags.optimize, (0, 1))
            for field, pattern in (("checked_commit", r"[0-9a-f]{40}"),
                                   ("run_id", r"[1-9][0-9]*"),
                                   ("run_attempt", r"[1-9][0-9]*")):
                self.assertTrue(isinstance(identity[field], str)
                                and re.fullmatch(pattern, identity[field]) is not None,
                                "invalid native " + field)
            self.assertTrue(isinstance(environment.get("GITHUB_TOKEN"), str)
                            and bool(environment["GITHUB_TOKEN"].strip()),
                            "inherited authentication prerequisite missing")
            selected = environment.get("NODE_STATUS_LIVE_OUTPUT")
            self.assertTrue(isinstance(selected, str) and bool(selected),
                            "new external live output required")
            output = Path(selected)
            self.assertTrue(output.is_absolute(), "absolute live output required")
            self.assertTrue(output.parent.is_dir(), "existing output parent required")
            self.assertEqual(output.parent.resolve(), output.parent,
                             "symlink output parent refused")
            self.assertFalse(output.is_relative_to(REPO), "live output must be external")
            for path in (output, output.with_suffix(".stdout.log"),
                         output.with_suffix(".stderr.log"), output.with_suffix(".exit.json")):
                self.assertFalse(os.path.lexists(path), "new evidence files required")
            return identity, output

        identity, output = prepared(os.environ)
        fresh = [record for _, record in records if record.get("status") == "fresh"]
        self.assertTrue(fresh, "mandatory live shard has no declared-fresh candidates")
        captured = _q9006_capture_current(REPO, identity)
        self.assertEqual({path.stem: record for path, record in records}, captured["records"])
        self.assertEqual(captured["checked_commit"], identity["checked_commit"])

        def source_bindings():
            latest = _q9006_capture_current(REPO, identity)
            self.assertEqual(latest["checked_tree"], captured["checked_tree"])
            self.assertEqual(latest["manifest"], captured["manifest"])
            self.assertEqual(latest["records"], captured["records"])
            return latest["files"]

        before = source_bindings()
        command = [sys.executable, "-B", *(["-O"] if sys.flags.optimize else []), "-S",
                   str(REPO / "tools/node_status_live_check.py"),
                   "--root", str(REPO), "--output", str(output)]
        stdout_path, stderr_path = output.with_suffix(".stdout.log"), output.with_suffix(".stderr.log")
        exit_path = output.with_suffix(".exit.json")
        outcome = {"command": command, "identity": identity, "executed": False,
                   "returncode": None, "timed_out": False, "exception_type": None,
                   "source_bindings_before": before, "source_bindings_after": None,
                   "source_recheck_error": None, "prerequisite_controls": [],
                   "prerequisite_controls_scope": "ADMISSION_ONLY_NO_LIVE_CREDIT"}
        stdout_raw = stderr_raw = None
        try:
            controls = [("offline_absent_flag", {}, unittest.SkipTest)]
            for label, key in (("missing_native_context", "GITHUB_ACTIONS"),
                               ("missing_token", "GITHUB_TOKEN"),
                               ("missing_output", "NODE_STATUS_LIVE_OUTPUT")):
                environment = dict(os.environ)
                environment.pop(key, None)
                controls.append((label, environment, AssertionError))
            invalid = dict(os.environ)
            invalid["NODE_STATUS_LIVE_REQUIRED"] = "invalid"
            controls.append(("invalid_mandatory_flag", invalid, AssertionError))
            for label, environment, expected in controls:
                observed = None
                try:
                    prepared(environment)
                except Exception as error:
                    observed = type(error)
                outcome["prerequisite_controls"].append({
                    "id": label, "expected_exception": expected.__name__,
                    "observed_exception": observed.__name__ if observed else None,
                    "status": "PASS" if observed is expected else "FAIL"})
            if any(row["status"] != "PASS" for row in outcome["prerequisite_controls"]):
                raise AssertionError("live-shard prerequisite control failed")
            with os.fdopen(os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                   os.O_NOFOLLOW, 0o600), "wb") as stdout, \
                    os.fdopen(os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                      os.O_NOFOLLOW, 0o600), "wb") as stderr:
                outcome["executed"] = True
                try:
                    child = subprocess.run(command, cwd=REPO, stdout=stdout, stderr=stderr,
                                           timeout=420, check=False)
                    outcome["returncode"] = child.returncode
                except subprocess.TimeoutExpired:
                    outcome["timed_out"] = True
        except Exception as error:
            outcome["exception_type"] = type(error).__name__
        finally:
            try:
                outcome["source_bindings_after"] = source_bindings()
            except Exception as error:
                outcome["source_recheck_error"] = type(error).__name__
            for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
                if path.is_file():
                    raw = path.read_bytes()
                    outcome[label + "_bytes"] = len(raw)
                    outcome[label + "_sha256"] = hashlib.sha256(raw).hexdigest()
                    if label == "stdout":
                        stdout_raw = raw
                    else:
                        stderr_raw = raw
            with os.fdopen(os.open(exit_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                   os.O_NOFOLLOW, 0o600), "w", encoding="utf-8",
                           newline="\n") as stream:
                stream.write(json.dumps(outcome, sort_keys=True, indent=2) + "\n")

        self.assertIsNone(outcome["exception_type"], outcome["exception_type"])
        self.assertTrue(outcome["executed"])
        self.assertFalse(outcome["timed_out"], "actual live CLI timed out")
        self.assertEqual(outcome["returncode"], 0, "actual live CLI refused; see retained originals")
        self.assertIsNone(outcome["source_recheck_error"])
        self.assertEqual(outcome["source_bindings_after"], before)
        self.assertEqual(stderr_raw, b"")
        self.assertIsInstance(stdout_raw, bytes)
        self.assertLessEqual(len(stdout_raw), 4 * 1024 * 1024)
        self.assertFalse(output.is_symlink())
        self.assertTrue(output.is_file(), "actual CLI report missing")
        self.assertLessEqual(output.stat().st_size, 4 * 1024 * 1024)
        self.assertEqual(output.read_bytes(), stdout_raw, "CLI file/stdout bytes differ")
        report = _q9006_json(stdout_raw, 4 * 1024 * 1024)
        self.assertEqual(set(report), {
            "schema", "status", "scope", "scientific_effect", "cbz_green", "identity",
            "checked_commit_before", "checked_commit_after", "checked_tree_before",
            "checked_tree_after", "source_bindings", "source_bindings_after",
            "started_at", "source_bound_at", "observation_started_at",
            "observation_finished_at", "source_rechecked_at", "finished_at",
            "elapsed_seconds", "freshness"})
        self.assertEqual(report["schema"], "cbz.node-status-live-cli.v2")
        self.assertEqual(report["status"], "FRESH")
        self.assertEqual(report["scope"], "NODE_RECORD_FRESHNESS_ONLY")
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["cbz_green"], False)
        self.assertEqual(report["identity"], identity)
        self.assertEqual(report["checked_commit_before"], identity["checked_commit"])
        self.assertEqual(report["checked_commit_after"], identity["checked_commit"])
        self.assertTrue(isinstance(report["checked_tree_before"], str)
                        and re.fullmatch(r"[0-9a-f]{40}", report["checked_tree_before"]) is not None)
        self.assertEqual(report["checked_tree_after"], report["checked_tree_before"])
        self.assertEqual(report["checked_tree_before"], captured["checked_tree"])
        self.assertEqual(report["source_bindings"], before)
        self.assertEqual(report["source_bindings_after"], before)
        freshness = report["freshness"]
        _q9006_check_freshness(freshness, captured, identity)


class Q9006V2OracleTests(unittest.TestCase):
    """Supplied-byte oracle controls only; no authentication, HTTP or native credit."""

    def fixture(self):
        nodes = ("eng.d0-packet-allowlist-fix", "eng.hard-gate",
                 "math.lifetime-remainder", "math.side24-coefficient")
        url = ("https://github.com/d6g8k5htny-coder/Math-/blob/"
               "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde/"
               "frontiers/downstream_gate_20260925/GRAPH.json")
        manifest = {
            "schema": "node-status-inputs.v2", "profile": "legacy-four-v0",
            "records": [{"node_id": node, "path": "records/node_status/" + node + ".json",
                         "record_version": 0} for node in nodes],
            "schema_paths": [{"record_version": 0, "path": "architecture/node-status.schema.json"},
                             {"record_version": 1, "path": "architecture/node-status-v1.schema.json"}],
            "code_paths": ["tools/node_status_live_check.py", "tools/node_status_refresh.py",
                           "tools/node_status_v1.py", "tools/validate_node_status.py"],
            "graph": {"snapshot_path": "docs/site/current.json",
                      "raw_path": "docs/site/dependency-source/GRAPH.json"}, "artifact_paths": []}
        raw = {"architecture/node-status-inputs.json":
               (json.dumps(manifest, sort_keys=True) + "\n").encode("ascii")}
        files = {path: {"git_blob": "1" * 40, "git_mode": "100644", "sha256": "2" * 64}
                 for path in (*_Q9006_BOOTSTRAP,
                              *("records/node_status/" + node + ".json" for node in nodes))}
        files["docs/site/dependency-source/GRAPH.json"] = {
            "git_blob": "220b56867f24c5ef6f55a3cce70490e9777cf3af", "git_mode": "100644",
            "sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"}
        records = {node: _rec(node, sources=[{"url": url, "rev":
                    "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"}],
                    source_rev="dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde",
                    status="fresh" if index < 2 else "unknown")
                   for index, node in enumerate(nodes)}
        identity = {"repository": "d6g8k5htny-coder/main", "checked_commit": "3" * 40,
                    "run_id": "123", "run_attempt": "1", "python_optimize": 0}
        capture = {"manifest": manifest, "records": records, "raw": raw, "files": files}
        observation = {"url": url, "head": "4" * 40,
                       "pinned_object": ["blob", "100644", "220b56867f24c5ef6f55a3cce70490e9777cf3af"],
                       "current_object": ["blob", "100644", "220b56867f24c5ef6f55a3cce70490e9777cf3af"]}
        receipt = {
            "schema": "cbz.node-status-live-check.v2", "scope": "NODE_RECORD_FRESHNESS_ONLY",
            "records_total": 4, "record_versions": {node: 0 for node in nodes},
            "fresh_candidates": 2, "fresh_checked": 2, "fresh_nodes": list(nodes[:2]),
            "nonfresh_records": [{"node_id": node, "declared_status": "unknown"} for node in nodes[2:]],
            "source_bindings": copy.deepcopy(files),
            "manifest_binding": dict(path="architecture/node-status-inputs.json",
                profile="legacy-four-v0", bytes=len(raw["architecture/node-status-inputs.json"]),
                **files["architecture/node-status-inputs.json"]),
            "input_sha256": {node + ".json": "2" * 64 for node in nodes},
            "identity": identity, "observed_heads": {"Math-": "4" * 40},
            "source_observation_count": 2, "source_observations": [copy.deepcopy(observation),
                                                                  copy.deepcopy(observation)],
            "pinned_graph_binding": {
                "scope": "PINNED_MATH_GRAPH_OBJECT_IDENTITY_ONLY",
                "repository": "d6g8k5htny-coder/Math-",
                "commit": "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde",
                "path": "frontiers/downstream_gate_20260925/GRAPH.json", "git_kind": "blob",
                "git_mode": "100644", "git_blob": "220b56867f24c5ef6f55a3cce70490e9777cf3af",
                "bytes": 38753, "sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
                "snapshot_path": "docs/site/current.json", "raw_path": "docs/site/dependency-source/GRAPH.json",
                "snapshot_git_blob": "1" * 40, "raw_git_blob": "220b56867f24c5ef6f55a3cce70490e9777cf3af",
                "full_data_equal": True, "complete": True},
            "scientific_effect": "NONE", "cbz_green": False}
        return receipt, capture, identity

    def test_supplied_consistent_v2_receipt_has_only_oracle_credit(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)

    def test_same_count_source_binding_substitution_refuses(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)
        old = receipt["source_bindings"].pop("tools/node_status_v1.py")
        receipt["source_bindings"]["tools/unrelated.py"] = old
        with self.assertRaisesRegex(AssertionError, "binding set"):
            _q9006_check_freshness(receipt, capture, identity)

    def test_primitive_record_version_substitution_refuses(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)
        for value in (False, 1):
            changed = copy.deepcopy(receipt)
            changed["record_versions"]["eng.hard-gate"] = value
            with self.assertRaisesRegex(AssertionError, "versions/status"):
                _q9006_check_freshness(changed, capture, identity)

    def test_same_count_status_partition_substitution_refuses(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)
        receipt["nonfresh_records"][0]["declared_status"] = "stale"
        with self.assertRaisesRegex(AssertionError, "versions/status"):
            _q9006_check_freshness(receipt, capture, identity)

    def test_same_count_url_multiplicity_substitution_refuses(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)
        receipt["source_observations"][1]["url"] = receipt["source_observations"][1]["url"].replace(
            "GRAPH.json", "unrelated.json")
        with self.assertRaisesRegex(AssertionError, "multiplicities"):
            _q9006_check_freshness(receipt, capture, identity)

    def test_equal_wrong_object_tuple_and_pinned_join_refuse(self):
        receipt, capture, identity = self.fixture()
        _q9006_check_freshness(receipt, capture, identity)
        changed = copy.deepcopy(receipt)
        changed["source_observations"][0]["pinned_object"][1] = "040000"
        changed["source_observations"][0]["current_object"][1] = "040000"
        with self.assertRaisesRegex(AssertionError, "object tuple"):
            _q9006_check_freshness(changed, capture, identity)
        receipt["pinned_graph_binding"]["complete"] = 1
        with self.assertRaisesRegex(AssertionError, "pinned Graph join"):
            _q9006_check_freshness(receipt, capture, identity)

    def test_complete_framed_readback_enforces_one_aggregate_budget(self):
        names = (
            "node-status-live-normal.json", "node-status-live-normal.log", "node-status-live-normal-exit.json",
            "q9006-real-record-normal.json", "q9006-real-record-normal.stdout.log",
            "q9006-real-record-normal.stderr.log", "q9006-real-record-normal.exit.json",
            "q9006-original-normal-source-before.json", "q9006-original-normal-source-after.json",
            "q9006-original-normal-parent-exit.json", "q9006-original-normal-unittest.log",
            "q9006-original-normal-strict-guard.json",
            "node-status-live-optimized.json", "node-status-live-optimized.log", "node-status-live-optimized-exit.json",
            "q9006-real-record-optimized.json", "q9006-real-record-optimized.stdout.log",
            "q9006-real-record-optimized.stderr.log", "q9006-real-record-optimized.exit.json",
            "q9006-original-optimized-source-before.json", "q9006-original-optimized-source-after.json",
            "q9006-original-optimized-parent-exit.json", "q9006-original-optimized-unittest.log",
            "q9006-original-optimized-strict-guard.json",
        )
        originals = {name: b"" for name in names}
        context = {"repository": "d6g8k5htny-coder/main", "checked_commit": "3" * 40,
                   "checked_tree": "4" * 40, "run_id": "123", "run_attempt": "1"}
        marker = "a" * 48
        payload = _q9006_readback_payload(originals, context, marker)
        first = next(iter(originals))
        padding = _Q9006_READBACK_LIMIT - len(payload) - 32
        originals[first] = b"x" * padding
        payload = _q9006_readback_payload(originals, context, marker)
        originals[first] += b"x" * (_Q9006_READBACK_LIMIT - len(payload))
        self.assertEqual(len(_q9006_readback_payload(originals, context, marker)), 2 * 1024 * 1024)
        originals[first] += b"x"
        with self.assertRaisesRegex(AssertionError, "aggregate"):
            _q9006_readback_payload(originals, context, marker)
        originals = {name: b"" for name in names}
        originals.pop(first)
        with self.assertRaisesRegex(AssertionError, "inventory"):
            _q9006_readback_payload(originals, context, marker)


class MandatoryInputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_records_directory_is_a_failure(self):
        with self.assertRaisesRegex(MandatoryInputMissing, "mandatory records directory is missing"):
            _load_real_records(self.tmp_path / "no-such-node-status")

    def test_empty_records_directory_is_a_failure(self):
        empty = self.tmp_path / "node_status"
        empty.mkdir()
        with self.assertRaisesRegex(MandatoryInputMissing, "Lane 2 records missing"):
            _load_real_records(empty)

    def test_missing_schema_is_a_failure(self):
        with self.assertRaisesRegex(MandatoryInputMissing, "mandatory node-status schema is missing"):
            _load_schema(self.tmp_path / "node-status.schema.json")

    def test_missing_validator_is_a_failure(self):
        with self.assertRaisesRegex(MandatoryInputMissing, "mandatory node-status validator is missing"):
            _load_validator(self.tmp_path / "validate_node_status.py")


class PublishedProjectionBoundaryTests(unittest.TestCase):
    """Synthetic files exercise real loading; no project ownership is invented.

    One method per state (was pytest.mark.parametrize). A skip here is itself a
    failure: the projection is mandatory.
    """

    def _run_state(self, state):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            directory = tmp_path / "records"
            directory.mkdir()
            records = {}
            for name in LANE2_RECORD_NAMES:
                record = _rec(name[:-5])
                records[record["node_id"]] = record
                (directory / name).write_text(json.dumps(record), encoding="utf-8")
            projection = tmp_path / "node-status.json"
            if state == "directory":
                projection.mkdir()
            elif state not in ("missing", "directory"):
                published = copy.deepcopy(records)
                if state == "missing-node":
                    published.pop(next(iter(published)))
                elif state == "changed-value":
                    published[next(iter(published))]["source_rev"] = REV_B
                projection.write_text(json.dumps(published), encoding="utf-8")
            try:
                if state in ("missing", "directory"):
                    with self.assertRaisesRegex(MandatoryInputMissing, "mandatory published projection"):
                        check_real_records_match_published_projection(directory, projection)
                elif state in ("missing-node", "changed-value"):
                    with self.assertRaises(AssertionError):
                        check_real_records_match_published_projection(directory, projection)
                else:
                    check_real_records_match_published_projection(directory, projection)
            except unittest.SkipTest as error:
                self.fail(f"mandatory projection was skipped instead of checked: {error}")

    def test_state_missing(self):
        self._run_state("missing")

    def test_state_directory(self):
        self._run_state("directory")

    def test_state_matching(self):
        self._run_state("matching")

    def test_state_missing_node(self):
        self._run_state("missing-node")

    def test_state_changed_value(self):
        self._run_state("changed-value")


if __name__ == "__main__":
    unittest.main()
