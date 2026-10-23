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
import importlib.util
import json
import re
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
        if not any(record.get("depends_on") for record in payloads):
            self.skipTest(
                "no real record carries depends_on; dependency checks are not exercised. "
                "depends_on is not interface v0. Dependency validation is not proof verification."
            )
        self.assertEqual([err for err in validate(payloads) if err[0] == "dependency"], [])

    def test_real_records_cited_artifacts_when_present(self):
        records = _require_real_records()
        cited = [artifact for _, record in records for artifact in record.get("artifacts", [])]
        if not cited:
            self.skipTest("no real record cites repo artifact paths; missing-artifact checks are not exercised")
        present = frozenset(str(path.relative_to(REPO)) for path in REPO.rglob("*") if path.is_file())
        self.assertEqual(sorted(artifact for artifact in cited if artifact not in present), [])

    def test_real_records_fresh_against_live_source_head(self):
        records = _require_real_records()
        fresh = [record["node_id"] for _, record in records if record.get("status") == "fresh"]
        if not fresh:
            self.skipTest("no real record is marked fresh; live-head freshness is not exercised")
        self.skipTest(
            "fresh records "
            + ", ".join(fresh)
            + " are not compared to a live source head; tools/validate_node_status.py performs "
            "no network fetch, so this check is skipped instead of passed"
        )


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
