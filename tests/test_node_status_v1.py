"""TEST-only v1 node-status contracts; production helper and schema are absent.

The original Graph and three Main source-refresh files are read from captured
Main. All nine v1 records and mutated source frames below are TEST specimens.
They establish neither an assignment nor live freshness, Git/HTTP authority,
independent mathematical review, custody, full Main discovery, G5 or CBZ.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Initial native RED must be the real missing module, before fixture/schema work.
import tools.node_status_v1 as subject
from tools.node_status_v1 import (
    check_v1_record, check_v1_population, project_v1_population,
)

if Path(subject.__file__).resolve() != ROOT / "tools/node_status_v1.py":
    raise AssertionError("subject escaped the captured source checkout")

GRAPH_REV = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"
MAIN_ARTIFACT_REV = "02973f3975a1c4cc91b7c51d372f65717c9b3c5c"
GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
GRAPH_BLOB = "220b56867f24c5ef6f55a3cce70490e9777cf3af"
GRAPH_SHA = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"
SCHEMA_V0_BLOB = "2bad95d30bcb89ea686d1ac5a50d52113dd730be"
PARENT = "math.uniform-matrix-cap-lifetime"
E1 = "math.d1-component.congruence-erratum"
E2 = "math.d1-component.section9-replacement-v1_1"
CAP = "math.d1-component.marked-cylinder-cap"
REC = "math.d1-component.reconciliation-record"
D0 = "eng.d0-packet-allowlist-fix"
HARD = "eng.hard-gate"
LIFE = "math.lifetime-remainder"
SIDE = "math.side24-coefficient"
NODES = tuple(sorted((D0, HARD, LIFE, SIDE, PARENT, E1, E2, CAP, REC)))
COMPONENTS = (E1, E2, CAP, REC)
EDGES = (
    (LIFE, PARENT, "imports_named_interfaces"),
    (SIDE, PARENT, "evaluates_eq_15_2"),
    (PARENT, E1, "reads_with_congruence_erratum"),
    (PARENT, E2, "reads_section9_as_replaced_v1_1"),
    (PARENT, CAP, "consumes_cap_sections_1_5"),
    (PARENT, REC, "reads_with_reconciled_rule_W1_radius"),
)
SUPPORT = {
    PARENT: ("blob", "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md"),
    E1: ("blob", "imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md"),
    E2: ("blob", "reviews/d1_section9_borel_repair_20260925/REPAIR.md"),
    CAP: ("blob", "imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md"),
    REC: ("tree", "reviews/d1_chain_reconciliation_20260928"),
}
ARTIFACT_BLOBS = {
    "cbz/evidence/hard-gate-source-refresh-20261110/README.md":
        "97f1406ffb21aefd9b8d31a0f6d64b4222f40133",
    "cbz/evidence/hard-gate-source-refresh-20261110/observation.json":
        "300eca7ad096c5b7f44c3307c27da3d6cd8ac1e3",
    "cbz/evidence/hard-gate-source-refresh-20261110/current-SOURCE_FILES.json":
        "1fc4ed2bdd1ad4efbeb19dd358a4e8f3e3138f4f",
}
ARTIFACT_PATHS = tuple(sorted(ARTIFACT_BLOBS))
ROLE = "engineering-source-identity-evidence"
SCOPE = "ENGINEERING_OBJECT_IDENTITY_ONLY"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def original(relative, expected_blob):
    path = ROOT
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise AssertionError("captured original traverses symlink: " + relative)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise AssertionError("captured original is not regular: " + relative)
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, k) for k in fields) != tuple(getattr(after, k) for k in fields):
        raise AssertionError("captured original changed: " + relative)
    if blob(raw) != expected_blob:
        raise AssertionError("captured original blob differs: " + relative)
    return raw, "100755" if before.st_mode & stat.S_IXUSR else "100644"


def graph_identity(raw, revision=GRAPH_REV):
    return {"repository": "d6g8k5htny-coder/Math-", "commit": revision,
            "path": GRAPH_PATH, "git_blob": blob(raw), "bytes": len(raw),
            "sha256": sha(raw)}


def fixture_record(node_id, identity, graph):
    if node_id in SUPPORT:
        kind, relative = SUPPORT[node_id]
        primary = {"url": "https://github.com/d6g8k5htny-coder/Math-/" + kind + "/"
                   + identity["commit"] + "/" + relative, "rev": identity["commit"]}
        value = {"schema_version": 1, "node_id": node_id, "owner": None,
                 "next_action": "TEST v1 support specimen only; no actual adoption.",
                 "missing_input": ["TEST: current-default equality and independent proof review unobserved."],
                 "stops": ["TEST fixture only; scientific effect NONE."],
                 "sources": [primary], "source_rev": primary["rev"],
                 "observed_at": "2000-01-01T00:00:00Z", "status": "unknown",
                 "scientific_effect": "NONE"}
    else:
        value = json.loads((ROOT / "records/node_status" / (node_id + ".json")).read_bytes())
        value["schema_version"] = 1
        value["owner"] = None
        value["status"] = "unknown"
        value["observed_at"] = "2000-01-01T00:00:00Z"
        value["next_action"] = "TEST v1 specimen only; production record remains unchanged."
    value["graph_source"] = copy.deepcopy(identity)
    edges = [edge for edge in graph["edges"]
             if edge["from"] == node_id and edge["required"] is True]
    value.pop("depends_on", None)
    value.pop("dependency_evidence", None)
    if edges:
        value["depends_on"] = [edge["to"] for edge in edges]
        value["dependency_evidence"] = [
            {"node_id": edge["to"], "relation": edge["relation"], "required": True}
            for edge in edges]
    return value


class NodeStatusV1Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1_schema = json.loads((ROOT / "architecture/node-status-v1.schema.json").read_bytes())
        schema_raw, _schema_mode = original("architecture/node-status.schema.json", SCHEMA_V0_BLOB)
        cls.v0_schema = json.loads(schema_raw)
        raw, _mode = original("docs/site/dependency-source/GRAPH.json", GRAPH_BLOB)
        if len(raw) != 38753 or sha(raw) != GRAPH_SHA:
            raise AssertionError("captured actual Graph identity differs")
        cls.graph = json.loads(raw)
        actual_edges = [(e["from"], e["to"], e["relation"])
                        for e in cls.graph["edges"][:6] if e["required"] is True]
        if actual_edges != list(EDGES):
            raise AssertionError("captured Graph source-order assumptions changed")
        if cls.graph["nodes"][PARENT]["reading_rule"] != list(COMPONENTS):
            raise AssertionError("captured parent reading rule changed")
        identity = graph_identity(raw)
        artifacts = []
        for path in ARTIFACT_PATHS:
            body, mode = original(path, ARTIFACT_BLOBS[path])
            artifacts.append({"identity": {
                "path": path, "repository": "d6g8k5htny-coder/main",
                "commit": MAIN_ARTIFACT_REV, "git_mode": mode, "git_blob": blob(body),
                "bytes": len(body), "sha256": sha(body)}, "raw": body})
        cls.inputs = {"schema": "node-status-v1.source-inputs.v1",
                      "graph": {"identity": identity, "raw": raw}, "artifacts": artifacts}
        cls.record_files = []
        for node_id in NODES:
            value = fixture_record(node_id, identity, cls.graph)
            if node_id == HARD:
                value["artifacts"] = list(ARTIFACT_PATHS)
                value["artifact_evidence"] = [
                    dict(frame["identity"], role=ROLE, scope=SCOPE) for frame in artifacts]
            cls.record_files.append((node_id + ".json", encoded(value)))
        spec = importlib.util.spec_from_file_location(
            "TEST_original_v0_validator", ROOT / "tools/validate_node_status.py")
        cls.legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.legacy)

    def setUp(self):
        self.records = copy.deepcopy(self.record_files)
        self.frame = copy.deepcopy(self.inputs)

    def expected(self, records=None):
        records = self.records if records is None else records
        return {json.loads(raw)["node_id"]: json.loads(raw) for _name, raw in records}

    def one(self, node_id, records=None):
        records = self.records if records is None else records
        return next((name, raw) for name, raw in records if name == node_id + ".json")

    def changed(self, node_id, mutate, records=None):
        result = copy.deepcopy(self.records if records is None else records)
        for index, (name, raw) in enumerate(result):
            if name == node_id + ".json":
                value = json.loads(raw)
                mutate(value)
                result[index] = (name, encoded(value))
                return result
        raise AssertionError("TEST mutation target missing")

    def refused(self, code, function, *arguments):
        with self.assertRaises(ValueError) as refusal:
            function(*arguments)
        self.assertEqual(str(refusal.exception), code)

    def positive(self, records=None, frame=None):
        records = self.records if records is None else records
        frame = self.frame if frame is None else frame
        expected = self.expected(records)
        self.assertEqual(check_v1_population(records, frame), expected)
        return expected

    def rebound_graph(self, graph):
        frame = copy.deepcopy(self.frame)
        raw = encoded(graph)
        identity = graph_identity(raw, "a" * 40)  # TEST identity, never a native commit claim.
        frame["graph"] = {"identity": identity, "raw": raw}
        result = []
        for name, body in self.records:
            value = json.loads(body)
            node_id = value["node_id"]
            value["graph_source"] = copy.deepcopy(identity)
            if node_id in SUPPORT:
                kind, relative = SUPPORT[node_id]
                value["sources"][0] = {
                    "url": "https://github.com/d6g8k5htny-coder/Math-/" + kind + "/"
                           + identity["commit"] + "/" + relative,
                    "rev": identity["commit"]}
                value["source_rev"] = identity["commit"]
            edges = [edge for edge in graph["edges"]
                     if edge["from"] == node_id and edge["required"] is True]
            value.pop("depends_on", None)
            value.pop("dependency_evidence", None)
            if edges:
                value["depends_on"] = [edge["to"] for edge in edges]
                value["dependency_evidence"] = [
                    {"node_id": edge["to"], "relation": edge["relation"], "required": True}
                    for edge in edges]
            result.append((name, encoded(value)))
        return result, frame

    def test_actual_graph_and_artifact_originals_support_the_complete_current_fixture(self):
        schema = self.v1_schema
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "architecture/node-status-v1.schema.json")
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["required"], self.v0_schema["required"] + ["graph_source"])
        self.assertEqual(schema["properties"]["schema_version"], {"type": "integer", "const": 1})
        self.assertEqual(set(schema["properties"]), set(self.v0_schema["properties"])
                         | {"graph_source", "depends_on", "dependency_evidence",
                            "artifacts", "artifact_evidence"})
        self.assertEqual(schema["dependentRequired"], {
            "depends_on": ["dependency_evidence"], "dependency_evidence": ["depends_on"],
            "artifacts": ["artifact_evidence"], "artifact_evidence": ["artifacts"]})
        for name, required in (
            ("graph_identity", ["repository", "commit", "path", "git_blob", "bytes", "sha256"]),
            ("dependency", ["node_id", "relation", "required"]),
            ("artifact", ["path", "repository", "commit", "git_mode", "git_blob",
                          "bytes", "sha256", "role", "scope"]),
        ):
            self.assertIs(schema["$defs"][name]["additionalProperties"], False)
            self.assertEqual(schema["$defs"][name]["required"], required)
        expected = self.expected()
        self.assertEqual(tuple(sorted(expected)), NODES)
        self.assertEqual(len(expected), 9)
        self.assertEqual(sum(len(r.get("depends_on", [])) for r in expected.values()), 6)
        before = copy.deepcopy((self.records, self.frame))
        for name, raw in self.records:
            self.assertEqual(check_v1_record(name, raw, self.frame), json.loads(raw))
        self.assertEqual(check_v1_population(self.records, self.frame), expected)
        projected = project_v1_population(self.records, self.frame)
        self.assertIs(type(projected), bytes)
        self.assertEqual(projected, encoded(expected))
        self.assertEqual((self.records, self.frame), before)
        for node_id, record in expected.items():
            self.assertIsNone(record["owner"])
            self.assertEqual(record["status"], "unknown")
            self.assertEqual(record["scientific_effect"], "NONE")
            if node_id not in (LIFE, SIDE, PARENT):
                self.assertNotIn("depends_on", record)
                self.assertNotIn("dependency_evidence", record)
            if node_id != HARD:
                self.assertNotIn("artifacts", record)
                self.assertNotIn("artifact_evidence", record)

    def test_artifact_relevance_scope_and_original_bytes_are_required(self):
        self.positive()
        hard = self.expected()[HARD]
        for node_id in (LIFE, D0):
            with self.subTest(wrong_node=node_id):
                records = self.changed(node_id, lambda r: r.update(
                    artifacts=hard["artifacts"], artifact_evidence=hard["artifact_evidence"]))
                self.refused("ARTIFACT_SCOPE_MISMATCH", check_v1_record,
                             *self.one(node_id, records), self.frame)
        for field, wrong in (("role", "TEST-unrelated-role"), ("scope", "TEST-proof-acceptance")):
            with self.subTest(field=field):
                records = self.changed(HARD, lambda r: r["artifact_evidence"][0].update({field: wrong}))
                self.refused("ARTIFACT_SCOPE_MISMATCH", check_v1_record,
                             *self.one(HARD, records), self.frame)
        for mutate in (
            lambda r: (r["artifacts"].pop(), r["artifact_evidence"].pop()),
            lambda r: (r.pop("artifacts"), r.pop("artifact_evidence")),
        ):
            records = self.changed(HARD, mutate)
            self.refused("ARTIFACT_SET_MISMATCH", check_v1_record,
                         *self.one(HARD, records), self.frame)
        frame = copy.deepcopy(self.frame)
        frame["artifacts"][0]["raw"] += b"TEST changed original"
        self.refused("ARTIFACT_IDENTITY_MISMATCH", check_v1_population, self.records, frame)
        for key, wrong in (("git_mode", "100755"), ("git_blob", "0" * 40),
                           ("sha256", "0" * 64), ("bytes", 0)):
            with self.subTest(identity=key):
                records = self.changed(HARD, lambda r: r["artifact_evidence"][0].update({key: wrong}))
                self.refused("ARTIFACT_IDENTITY_MISMATCH", check_v1_record,
                             *self.one(HARD, records), self.frame)

    def test_closed_record_source_and_json_primitives_are_enforced(self):
        self.positive()
        name, raw = self.one(HARD)
        malformed = (
            raw[:-2] + b',"schema_version":1}\n',
            raw + b"{}\n", b"\xff", b'{"x":NaN}', b" " * 65537,
            b'{"x":' + b"[" * 70 + b"0" + b"]" * 70 + b"}",
        )
        for body in malformed:
            with self.subTest(raw_bytes=len(body)):
                self.refused("INVALID_JSON", check_v1_record, name, body, self.frame)
        for version in (0, True, 1.0, 2):
            records = self.changed(HARD, lambda r: r.update(schema_version=version))
            self.refused("UNSUPPORTED_VERSION", check_v1_record,
                         *self.one(HARD, records), self.frame)
        changes = (
            lambda r: r.update(extra=True),
            lambda r: r.update(scientific_effect="PROVED"),
            lambda r: r.update(lemma_closed=False),
            lambda r: r.update(owner={"id": "TEST", "evidence_url": 7}),
            lambda r: r["graph_source"].update(extra=False),
            lambda r: r["graph_source"].update(bytes=True),
            lambda r: r["artifact_evidence"][0].update(extra=False),
            lambda r: r["artifact_evidence"][0].update(role=7),
            lambda r: r["artifact_evidence"][0].update(scope=False),
            lambda r: r["artifact_evidence"][0].update(sha256="not-a-hash"),
            lambda r: r["artifact_evidence"][0].update(path="../TEST"),
            lambda r: r.update(artifacts=[], artifact_evidence=[]),
            lambda r: r.pop("artifact_evidence"),
        )
        for mutate in changes:
            with self.subTest(mutation=mutate):
                records = self.changed(HARD, mutate)
                self.refused("INVALID_RECORD", check_v1_record,
                             *self.one(HARD, records), self.frame)
        records = self.changed(PARENT, lambda r: r.update(depends_on=[], dependency_evidence=[]))
        self.refused("INVALID_RECORD", check_v1_record, *self.one(PARENT, records), self.frame)

    def test_dependency_order_and_all_six_declared_edges_are_preserved(self):
        expected = self.positive()
        actual = [(node_id, evidence["node_id"], evidence["relation"])
                  for node_id in (LIFE, SIDE, PARENT)
                  for evidence in expected[node_id]["dependency_evidence"]]
        self.assertEqual(actual, list(EDGES))
        self.assertEqual(expected[PARENT]["depends_on"], list(COMPONENTS))
        records = self.changed(PARENT, lambda r: (
            r["depends_on"].reverse(), r["dependency_evidence"].reverse()))
        self.refused("DEPENDENCY_MISMATCH", check_v1_record,
                     *self.one(PARENT, records), self.frame)
        for node_id in (LIFE, SIDE, PARENT):
            records = self.changed(node_id, lambda r: (
                r.pop("depends_on"), r.pop("dependency_evidence")))
            self.refused("DEPENDENCY_MISMATCH", check_v1_record,
                         *self.one(node_id, records), self.frame)

    def test_directory_scope_and_full_reading_rule_components_are_not_reduced(self):
        self.positive()
        for node_id in SUPPORT:
            kind, relative = SUPPORT[node_id]
            self.assertEqual(self.expected()[node_id]["sources"][0]["url"],
                             "https://github.com/d6g8k5htny-coder/Math-/" + kind + "/"
                             + GRAPH_REV + "/" + relative)
        for node_id, wrong in (
            (REC, "https://github.com/d6g8k5htny-coder/Math-/blob/" + GRAPH_REV
             + "/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md"),
            (E2, "https://github.com/d6g8k5htny-coder/Math-/blob/" + GRAPH_REV
             + "/reviews/d1_section9_borel_repair_20260925/RECONNAISSANCE.md"),
            (CAP, "https://github.com/d6g8k5htny-coder/Math-/blob/" + "b" * 40
             + "/imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md"),
        ):
            records = self.changed(node_id, lambda r: (
                r["sources"][0].update(url=wrong, rev=wrong.split("/")[6]),
                r.update(source_rev=wrong.split("/")[6])))
            self.refused("READING_RULE_MISMATCH", check_v1_record,
                         *self.one(node_id, records), self.frame)
        for node_id in COMPONENTS:
            records = [(name, raw) for name, raw in self.records if name != node_id + ".json"]
            self.refused("POPULATION_MISMATCH", check_v1_population, records, self.frame)
        graph = copy.deepcopy(self.graph)
        graph["nodes"][PARENT]["reading_rule"].reverse()
        records, frame = self.rebound_graph(graph)
        self.refused("READING_RULE_MISMATCH", check_v1_population, records, frame)

    def test_missing_extra_and_same_count_population_substitutions_are_refused(self):
        self.positive()
        for name, _raw in self.records:
            records = [(n, raw) for n, raw in self.records if n != name]
            self.refused("POPULATION_MISMATCH", check_v1_population, records, self.frame)
            self.refused("POPULATION_MISMATCH", project_v1_population, records, self.frame)
        extra = "eng.main-pr87-crosswalk"
        value = copy.deepcopy(self.expected()[D0])
        value["node_id"] = extra
        for records in (self.records + [(extra + ".json", encoded(value))],
                        [(n, raw) for n, raw in self.records if n != PARENT + ".json"]
                        + [(extra + ".json", encoded(value))]):
            self.refused("NODE_SCOPE_MISMATCH", check_v1_population, records, self.frame)
        self.refused("DUPLICATE_FILENAME", check_v1_population,
                     self.records + [self.records[0]], self.frame)
        records = copy.deepcopy(self.records)
        records[1] = (records[1][0], records[0][1])
        self.refused("DUPLICATE_NODE_ID", check_v1_population, records, self.frame)
        name, raw = self.one(D0)
        self.refused("NODE_SCOPE_MISMATCH", check_v1_record, HARD + ".json", raw, self.frame)

    def test_native_identity_claims_cannot_be_inferred_from_inert_inputs(self):
        expected = self.positive()
        for key in ("authenticated", "current_default", "proof", "custody", "provider"):
            frame = copy.deepcopy(self.frame)
            frame[key] = True
            self.refused("INVALID_INPUTS", check_v1_population, self.records, frame)
        result = check_v1_population(self.records, self.frame)
        self.assertEqual(result, expected)
        for record in result.values():
            self.assertEqual(record["status"], "unknown")
            self.assertIsNone(record["owner"])
            for key in ("authenticated", "g5_claim", "cbz_claim", "lemma_closed", "custody"):
                self.assertNotIn(key, record)
        self.assertEqual(tuple(self.frame), ("schema", "graph", "artifacts"))

    def test_projection_preserves_records_without_mutation_or_status_promotion(self):
        self.positive()
        released = json.loads((ROOT / "tests/fixtures/node_status/math-463-released.json").read_bytes())
        records = self.changed(HARD, lambda r: r.update(
            owner=released["owner"], predecessor=released["predecessor"],
            successor=None, carried_forward=released["carried_forward"],
            refresh_error=None))
        expected = self.expected(records)
        before = copy.deepcopy((records, self.frame))
        result = project_v1_population(list(reversed(records)), self.frame)
        self.assertEqual(result, encoded(expected))
        self.assertEqual(json.loads(result), expected)
        self.assertTrue(result.endswith(b"\n"))
        self.assertFalse(result.endswith(b"\n\n"))
        self.assertEqual((records, self.frame), before)
        self.assertIsInstance(expected[HARD]["owner"], dict)
        self.assertIn("released_url", expected[HARD]["owner"])
        self.assertIsNone(expected[HARD]["successor"])
        for node_id in (D0, LIFE, SIDE, PARENT, E1, E2, CAP, REC):
            self.assertNotIn("artifacts", expected[node_id])
        invalid = self.changed(HARD, lambda r: r.update(extra=True), records)
        self.refused("INVALID_RECORD", project_v1_population, invalid, self.frame)

    def test_same_count_dependency_target_relation_and_required_substitutions_refuse(self):
        self.positive()
        changes = (
            ("DEPENDENCY_MISMATCH", lambda r: (
                r["depends_on"].__setitem__(0, CAP),
                r["dependency_evidence"][0].update(node_id=CAP))),
            ("DEPENDENCY_MISMATCH", lambda r:
                r["dependency_evidence"][0].update(relation="TEST-substituted-relation")),
            ("DEPENDENCY_MISMATCH", lambda r: r["depends_on"].__setitem__(0, CAP)),
            ("INVALID_RECORD", lambda r: r["dependency_evidence"][0].update(required=False)),
            ("INVALID_RECORD", lambda r: r["dependency_evidence"][0].update(required=1)),
        )
        for code, mutate in changes:
            records = self.changed(LIFE, mutate)
            self.refused(code, check_v1_record, *self.one(LIFE, records), self.frame)
        records = self.changed(PARENT, lambda r: r["dependency_evidence"].pop())
        self.refused("DEPENDENCY_MISMATCH", check_v1_record,
                     *self.one(PARENT, records), self.frame)

    def test_source_frames_and_declared_path_kinds_cannot_override_bindings(self):
        self.positive()
        for mutate in (
            lambda f: f.update(extra=True),
            lambda f: f["graph"].update(extra=True),
            lambda f: f["graph"]["identity"].update(extra=True),
            lambda f: f["artifacts"][0].update(extra=True),
            lambda f: f["artifacts"][0]["identity"].update(extra=True),
            lambda f: f["graph"].update(raw="TEST-not-bytes"),
            lambda f: f["artifacts"][0].update(raw="TEST-not-bytes"),
            lambda f: f["artifacts"][0]["identity"].update(git_mode="120000"),
            lambda f: f["graph"]["identity"].update(bytes=True),
        ):
            frame = copy.deepcopy(self.frame)
            mutate(frame)
            self.refused("INVALID_INPUTS", check_v1_population, self.records, frame)
        frame = copy.deepcopy(self.frame)
        frame["graph"]["raw"] += b" "
        self.refused("GRAPH_IDENTITY_MISMATCH", check_v1_population, self.records, frame)
        for key, wrong in (("repository", "TEST/wrong"), ("path", "TEST/GRAPH.json"),
                           ("git_blob", "0" * 40), ("sha256", "0" * 64), ("bytes", 1)):
            records = self.changed(HARD, lambda r: r["graph_source"].update({key: wrong}))
            self.refused("GRAPH_IDENTITY_MISMATCH", check_v1_record,
                         *self.one(HARD, records), self.frame)
        frame = copy.deepcopy(self.frame)
        frame["artifacts"][1] = copy.deepcopy(frame["artifacts"][0])
        self.refused("INVALID_INPUTS", check_v1_population, self.records, frame)
        frame = copy.deepcopy(self.frame)
        frame["artifacts"].pop()
        self.refused("INVALID_INPUTS", check_v1_population, self.records, frame)

    def test_synthetic_cycles_unknown_targets_and_reversed_edges_are_refused(self):
        self.positive()
        acyclic = copy.deepcopy(self.graph)
        acyclic["edges"].append({"from": E1, "to": CAP, "required": True,
                                  "relation": "TEST-acyclic-required-edge"})
        positive_records, positive_frame = self.rebound_graph(acyclic)
        self.positive(positive_records, positive_frame)
        for target, code in ((E1, "DEPENDENCY_CYCLE"), (PARENT, "DEPENDENCY_CYCLE"),
                             ("math.TEST-unknown-target", "GRAPH_DECLARATION_MISMATCH")):
            graph = copy.deepcopy(acyclic)
            graph["edges"].append({"from": E1, "to": target, "required": True,
                                   "relation": "TEST-adverse-edge"})
            records, frame = self.rebound_graph(graph)
            self.refused(code, check_v1_population, records, frame)
        graph = copy.deepcopy(self.graph)
        graph["edges"].append({"from": E1, "to": CAP, "required": False,
                               "relation": "TEST-context-only"})
        records, frame = self.rebound_graph(graph)
        self.positive(records, frame)
        records = self.changed(E1, lambda r: r.update(
            depends_on=[CAP], dependency_evidence=[
                {"node_id": CAP, "relation": "TEST-context-only", "required": True}]), records)
        self.refused("DEPENDENCY_MISMATCH", check_v1_record,
                     *self.one(E1, records), frame)
        records = self.changed(D0, lambda r: r.update(
            depends_on=["eng.main-pr87-crosswalk"], dependency_evidence=[{
                "node_id": "eng.main-pr87-crosswalk",
                "relation": "ci_unblock_depends_on_allowlist_patch", "required": True}]))
        self.refused("DEPENDENCY_MISMATCH", check_v1_record,
                     *self.one(D0, records), self.frame)

    def test_v0_schema_and_original_accept_refuse_vectors_remain_unchanged(self):
        raw, _mode = original("architecture/node-status.schema.json", SCHEMA_V0_BLOB)
        schema = json.loads(raw)
        self.assertEqual(len(schema["required"]), 11)
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(schema["properties"]["schema_version"], {"const": 0})
        known = set(self.graph["nodes"])
        self.assertEqual(self.legacy.check_schema_file(ROOT), [])
        self.assertEqual(self.legacy.validate_directory(ROOT, ROOT / "records/node_status"), [])
        for node_id in (D0, HARD, LIFE, SIDE):
            value = json.loads((ROOT / "records/node_status" / (node_id + ".json")).read_bytes())
            self.assertEqual(value["schema_version"], 0)
            self.assertEqual(self.legacy.check_record(value, node_id + ".json", known), [])
            self.refused("UNSUPPORTED_VERSION", check_v1_record,
                         node_id + ".json", encoded(value), self.frame)
        released = json.loads((ROOT / "tests/fixtures/node_status/math-463-released.json").read_bytes())
        self.assertEqual(self.legacy.check_record(released, HARD + ".json", known), [])
        self.assertIsInstance(released["owner"], dict)
        self.assertIn("released_url", released["owner"])
        for extra in ("depends_on", "artifacts", "graph_source"):
            value = copy.deepcopy(released)
            value[extra] = ["TEST"]
            self.assertEqual(self.legacy.check_record(value, HARD + ".json", known),
                             ["unexpected fields: " + extra])
        value = copy.deepcopy(released)
        value["schema_version"] = 1
        self.assertEqual(self.legacy.check_record(value, HARD + ".json", known),
                         ["schema_version must be 0"])
        value = copy.deepcopy(released)
        value["owner"] = {"id": "TEST agent"}
        self.assertEqual(self.legacy.check_record(value, HARD + ".json", known),
                         ["owner requires evidence_url"])


if __name__ == "__main__":
    unittest.main()
