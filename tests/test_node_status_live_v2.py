"""Offline current-input v2 contracts; no production implementation is staged here.

Owned Git fixtures carry TEST manifests and records. The real captured source is
used, while explicitly synthetic GitHub replies exercise ContentObserver's real
path traversal. None authenticates HTTP, Actions, ownership, proof or freshness.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
# Existing module, absent names: initial native RED must stop HERE before fixtures.
from node_status_live_check import evaluate_fresh_records_v2, run_live_v2
import node_status_live_check as subject
from node_status_v1 import check_v1_population
from tests import node_status_legacy_v1_runner as legacy

if Path(subject.__file__).resolve() != ROOT / "tools/node_status_live_check.py":
    raise AssertionError("V2 subject escaped actual source checkout")

MATH = "d6g8k5htny-coder/Math-"
PIN = "dbe6c0a1b1a22c7404b0fd173ee0043b98a08cde"
GRAPH_PATH = "frontiers/downstream_gate_20260925/GRAPH.json"
GRAPH_BLOB = "220b56867f24c5ef6f55a3cce70490e9777cf3af"
GRAPH_SHA = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"
MANIFEST = "architecture/node-status-inputs.json"
SNAPSHOT = "docs/site/current.json"
RAW_GRAPH = "docs/site/dependency-source/GRAPH.json"
CODE = (
    "tools/node_status_live_check.py", "tools/node_status_refresh.py",
    "tools/node_status_v1.py", "tools/validate_node_status.py",
)
SCHEMAS = ("architecture/node-status.schema.json", "architecture/node-status-v1.schema.json")
SEEDS = (
    "eng.d0-packet-allowlist-fix", "eng.hard-gate",
    "math.lifetime-remainder", "math.side24-coefficient",
)
PARENT = "math.uniform-matrix-cap-lifetime"
COMPONENTS = (
    "math.d1-component.congruence-erratum",
    "math.d1-component.section9-replacement-v1_1",
    "math.d1-component.marked-cylinder-cap",
    "math.d1-component.reconciliation-record",
)
NODES = tuple(sorted(SEEDS + (PARENT,) + COMPONENTS))
SUPPORT = {
    PARENT: ("blob", "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md"),
    COMPONENTS[0]: ("blob", "imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md"),
    COMPONENTS[1]: ("blob", "reviews/d1_section9_borel_repair_20260925/REPAIR.md"),
    COMPONENTS[2]: ("blob", "imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md"),
    COMPONENTS[3]: ("tree", "reviews/d1_chain_reconciliation_20260928"),
}
ARTIFACTS = (
    "cbz/evidence/hard-gate-source-refresh-20261110/README.md",
    "cbz/evidence/hard-gate-source-refresh-20261110/current-SOURCE_FILES.json",
    "cbz/evidence/hard-gate-source-refresh-20261110/observation.json",
)
EDGES = (
    (SEEDS[2], PARENT, "imports_named_interfaces"),
    (SEEDS[3], PARENT, "evaluates_eq_15_2"),
    (PARENT, COMPONENTS[0], "reads_with_congruence_erratum"),
    (PARENT, COMPONENTS[1], "reads_section9_as_replaced_v1_1"),
    (PARENT, COMPONENTS[2], "consumes_cap_sections_1_5"),
    (PARENT, COMPONENTS[3], "reads_with_reconciled_rule_W1_radius"),
)
ROLE = "engineering-source-identity-evidence"
SCOPE = "ENGINEERING_OBJECT_IDENTITY_ONLY"
RECONCILIATION = "reviews/d1_chain_reconciliation_20260928"
# Public native tree metadata only, not acquired member bodies/proof review.
OLD_RECON = {
    "LEDGER.json": "f6c329fb0f21a6cc2cc690240e86aa0a86114d4e",
    "RECONCILIATION.md": "75da2597971510f843f8d90c743950cb8c177342",
    "RESULTS.json": "e2de6c41e87ce64196dfbe5077ead2f09d87b367",
    "SOURCE_FILES.json": "0ade05ab9218684955ad27a8c5c97f758dbc8e23",
    "reconciliation_check.py": "c2f413e47feef42d57034685b1f504dd48ed2ed3",
}
CURRENT_RECON = dict(OLD_RECON, **{
    "LEDGER.json": "e49a15aee0399e797388fd6f0fdcaa6845846e97",
    "RECOVERED_G2.md": "9e82dce83d81345b912a27cbc44fb913155f0302",
    "RECOVERED_REVIEWS.json": "a651092339661242a37860d37d5be851267dcae0",
    "RECOVERED_X3.md": "b5662098d08ad429cf0bb9e37e18f16722a11bf4",
    "RECOVERED_X4.md": "b1b934ff4404e05294921615c69e66fa53cd4758",
    "RECOVERED_X5.md": "d1efa5cb8ebd9e648d776203f7bd4150521e529d",
    "RESULTS.json": "9c66f6d5216edeff962d4716b506d450756abb14",
    "SCHEMA_NOTE.md": "3e2bc4a4c30beb854da14ec220f693e5b55220c6",
    "SOURCE_FILES.json": "66ce7f63390f28b0c2f54abde33fa682f1a51345",
})
V1_PAYLOAD = ROOT / "tests/fixtures/node_status_live_v1/payload"
EXPECTED_LEGACY_BLOBS = {
    "tools/node_status_live_check.py": "1e4852188f15060176b4458f58b27815f56845fd",
    "tools/node_status_refresh.py": "5fa7c8b6647938a0aca538a7735edede65de9ed5",
    "tools/validate_node_status.py": "e9d44b7620536b389a35ff6fad8aef2b0464b5d2",
    SCHEMAS[0]: "2bad95d30bcb89ea686d1ac5a50d52113dd730be",
    SNAPSHOT: "a64fb6d9768ada12b9884adac854e0ccdd5a910e",
    "records/node_status/eng.d0-packet-allowlist-fix.json": "0a7f209b58f4d309f8c8ae36d8dce1696c22ffbd",
    "records/node_status/eng.hard-gate.json": "9889637191bc01e26a6ad6dfec19b35de8b72a0b",
    "records/node_status/math.lifetime-remainder.json": "a6f82a36e4f9a31abf656661f7c8ee5201874522",
    "records/node_status/math.side24-coefficient.json": "ecf9c42723ee72b8e7d1a4d5b36470f158d6af0e",
    "tests/test_node_status_live_check.py": "c5d578ddb9b7699f24966784ebfcb518d2d7284a",
    "tests/test_node_status_live_cli.py": "14d0c84b1236fdc0d01083c62b791117b663c682",
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def graph_identity(raw):
    return dict(repository=MATH, commit=PIN, path=GRAPH_PATH, git_blob=blob(raw),
                bytes=len(raw), sha256=sha(raw))


def record_path(node):
    return "records/node_status/" + node + ".json"


def source_descriptor(root, relative):
    path = root / relative
    raw = path.read_bytes()
    return dict(git_blob=blob(raw),
                git_mode="100755" if path.stat().st_mode & stat.S_IXUSR else "100644",
                sha256=sha(raw))


def record_fixture(node, identity, graph, artifact_frames):
    if node in SUPPORT:
        kind, path = SUPPORT[node]
        primary = dict(url="https://github.com/" + MATH + "/" + kind + "/" + PIN + "/" + path,
                       rev=PIN)
        value = dict(schema_version=1, node_id=node, owner=None,
            next_action="TEST support view only; no canonical adoption.",
            missing_input=["TEST: no current native source/proof qualification."],
            stops=["TEST only; scientific effect NONE."], sources=[primary], source_rev=PIN,
            observed_at="2000-01-01T00:00:00Z", status="unknown", scientific_effect="NONE")
    else:
        raw = (V1_PAYLOAD / (record_path(node) + ".txt")).read_bytes()
        value = json.loads(raw)
        value["schema_version"] = 1
    value["graph_source"] = copy.deepcopy(identity)
    outgoing = [edge for edge in graph["edges"] if edge["from"] == node and edge["required"] is True]
    value.pop("depends_on", None)
    value.pop("dependency_evidence", None)
    if outgoing:
        value["depends_on"] = [e["to"] for e in outgoing]
        value["dependency_evidence"] = [
            dict(node_id=e["to"], relation=e["relation"], required=True) for e in outgoing]
    if node == SEEDS[1]:
        value["artifacts"] = list(ARTIFACTS)
        value["artifact_evidence"] = [
            dict(frame["identity"], role=ROLE, scope=SCOPE) for frame in artifact_frames]
    return value


class SyntheticMathAPI:
    """Complete TEST GitHub-shaped data; never a transport/authentication claim."""
    head = "f" * 40

    def __init__(self, records, *, wrong_graph=False, missing_graph=False,
                 bad_kind=False, bad_mode=False, truncated=False, hook=None):
        self.calls, self.record_calls = [], []
        self.hook, self.truncated = hook, truncated
        self.trees = {}
        leaves = {GRAPH_PATH: ("blob", "100644", GRAPH_BLOB)}
        self.discussions = {}
        for record in records:
            for source in record["sources"]:
                match = re.fullmatch(
                    r"https://github\.com/d6g8k5htny-coder/Math-/(blob|tree)/[0-9a-f]{40}/(.+)",
                    source["url"])
                if match:
                    kind, path = match.groups()
                    if kind == "blob" and path != GRAPH_PATH:
                        leaves[path] = ("blob", "100644", blob(("TEST " + path).encode()))
                    elif kind == "tree" and path != RECONCILIATION:
                        leaves[path + "/TEST.txt"] = ("blob", "100644", blob(b"TEST directory"))
                else:
                    comment = re.fullmatch(
                        r"https://github\.com/d6g8k5htny-coder/Math-/issues/([1-9]\d*)#issuecomment-([1-9]\d*)",
                        source["url"])
                    if comment:
                        issue, identity = comment.groups()
                        self.discussions[identity] = dict(id=int(identity), updated_at=source["rev"],
                            issue_url="https://api.github.com/repos/" + MATH + "/issues/" + issue)
        old = self.tree({p: ("blob", "100644", oid) for p, oid in OLD_RECON.items()})
        current = self.tree({p: ("blob", "100644", oid) for p, oid in CURRENT_RECON.items()})
        if old != "39734d39af1220069d3a29405106e2bc03c45b40" or current != "dacbb1ed2c6f61f22691fb81c5f3c93a75a6403e":
            raise AssertionError("native public reconciliation metadata identity changed")
        pinned = dict(leaves, **{RECONCILIATION: ("tree", "040000", old)})
        moving = dict(leaves, **{RECONCILIATION: ("tree", "040000", current)})
        if missing_graph:
            pinned.pop(GRAPH_PATH)
        elif wrong_graph or bad_kind or bad_mode:
            pinned[GRAPH_PATH] = (
                "tree" if bad_kind else "blob",
                "040000" if bad_kind else "100755" if bad_mode else "100644",
                "a" * 40 if wrong_graph else GRAPH_BLOB)
        self.pinned_root, self.current_root = self.tree(pinned), self.tree(moving)

    def tree(self, flat):
        nested = {}
        for path, item in flat.items():
            pieces = path.split("/")
            current = nested
            for piece in pieces[:-1]:
                current = current.setdefault(piece, {})
            current[pieces[-1]] = item
        def visit(node):
            entries, body = [], bytearray()
            for name in sorted(node, key=lambda p: p + ("/" if isinstance(node[p], dict) else "")):
                child = node[name]
                kind, mode, identity = ("tree", "040000", visit(child)) if isinstance(child, dict) else child
                entries.append(dict(path=name, type=kind, mode=mode, sha=identity))
                body.extend((mode.lstrip("0") + " " + name).encode() + b"\0" + bytes.fromhex(identity))
            raw = bytes(body)
            identity = hashlib.sha1(b"tree " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            self.trees[identity] = dict(sha=identity, tree=entries, truncated=False)
            return identity
        return visit(nested)

    def __call__(self, path):
        self.calls.append(path)
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        prefix = "/repos/" + MATH + "/"
        if not path.startswith(prefix):
            raise AssertionError("TEST API escaped fixed Math repository: " + path)
        relative = path[len(prefix):]
        if relative == "commits/HEAD":
            return dict(sha=self.head, commit=dict(tree=dict(sha=self.current_root)))
        if relative.startswith("git/commits/"):
            identity = relative.removeprefix("git/commits/")
            if re.fullmatch(r"[0-9a-f]{40}", identity) is None:
                raise AssertionError("TEST immutable commit expected")
            return dict(sha=identity, tree=dict(sha=self.current_root if identity == self.head else self.pinned_root))
        if relative.startswith("git/trees/"):
            value = copy.deepcopy(self.trees[relative.removeprefix("git/trees/")])
            if self.truncated:
                value["truncated"] = True
            return value
        if relative.startswith("issues/comments/"):
            return copy.deepcopy(self.discussions[relative.removeprefix("issues/comments/")])
        raise AssertionError("unlisted TEST API request: " + path)


REFRESH_GUARD = r'''
import json
from pathlib import Path
import runpy
import socket
import sys
import urllib.request
tool, marker = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError("TEST_REFRESH_NETWORK_FORBIDDEN")
def audit(event, args):
    if event.startswith("socket.") or event.startswith("urllib."):
        forbidden()
urllib.request.urlopen = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
sys.addaudithook(audit)
def trace(frame, event, arg):
    if (event == "call" and frame.f_code.co_filename == str(tool)
            and frame.f_code.co_name == "__init__"
            and type(frame.f_locals.get("self")).__name__ == "ContentObserver"):
        marker.write_text(json.dumps(dict(
            phase="ACTUAL_CONTENT_OBSERVER_ENTRY", filename=str(tool),
            root=str(tool.parent.parent), optimize=sys.flags.optimize)))
        raise RuntimeError("TEST_STOP_AT_ACTUAL_OBSERVER_BEFORE_HTTP")
    return trace
sys.argv = [str(tool), *sys.argv[3:]]
sys.path.insert(0, str(tool.parent))
sys.setprofile(trace)
try:
    runpy.run_path(str(tool), run_name="__main__")
finally:
    sys.setprofile(None)
'''


IMPORT_STOP = r'''
import json
from pathlib import Path
import runpy
import socket
import sys
import urllib.request
tool, checkpoint = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError("TEST_V2_NETWORK_FORBIDDEN")
def audit(event, args):
    if event.startswith("socket.") or event.startswith("urllib."):
        forbidden()
urllib.request.urlopen = forbidden
socket.create_connection = forbidden
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
sys.addaudithook(audit)
def trace(frame, event, arg):
    if (event == "call" and frame.f_code.co_name == "run_live_v2"
            and frame.f_globals.get("__name__") == "__main__"
            and Path(frame.f_code.co_filename).resolve() == tool):
        checkpoint.write_text(json.dumps(dict(
            phase="BOUND_RUN_LIVE_V2_ENTRY", filename=str(tool),
            root=str(Path(frame.f_locals["root"]).resolve()),
            identity=frame.f_locals["identity"], optimize=sys.flags.optimize)))
        raise RuntimeError("TEST_STOP_BEFORE_ACTUAL_V2_HTTP_BODY")
    return trace
sys.argv = [str(tool), *sys.argv[3:]]
sys.path.insert(0, str(tool.parent))
sys.settrace(trace)
try:
    runpy.run_path(str(tool), run_name="__main__")
finally:
    sys.settrace(None)
'''


class NodeStatusV2Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="node-status-v2-TEST-")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.denied = []
        def network(*args, **kwargs):
            self.denied.append(True)
            raise AssertionError("OFFLINE_V2_CONTRACT_NETWORK_FORBIDDEN")
        for target, name in (
            (urllib.request, "urlopen"), (socket, "create_connection"),
            (socket.socket, "connect"), (socket.socket, "connect_ex"),
        ):
            guard = patch.object(target, name, side_effect=network)
            guard.start()
            self.addCleanup(guard.stop)
        self.git_env = dict(PATH=os.environ.get("PATH", os.defpath),
            LANG="C.UTF-8", LC_ALL="C.UTF-8", GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
            GIT_AUTHOR_NAME="TEST V2 fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
            GIT_COMMITTER_NAME="TEST V2 fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid",
            GIT_AUTHOR_DATE="2000-01-01T00:00:00Z", GIT_COMMITTER_DATE="2000-01-01T00:00:00Z")
        self.serial = 0

    def git(self, root, *args):
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(root), *args],
            env=self.git_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result.stdout.decode().strip()

    def commit(self, root):
        self.git(root, "add", "--all")
        self.git(root, "commit", "--allow-empty", "-qm", "TEST source cut only")
        return self.git(root, "rev-parse", "HEAD")

    def manifest(self, profile, nodes):
        version = 0 if profile == "legacy-four-v0" else 1
        return dict(schema="node-status-inputs.v2", profile=profile,
            records=[dict(node_id=n, path=record_path(n), record_version=version) for n in sorted(nodes)],
            schema_paths=[dict(record_version=v, path=p) for v, p in enumerate(SCHEMAS)],
            code_paths=list(CODE), graph=dict(snapshot_path=SNAPSHOT, raw_path=RAW_GRAPH),
            artifact_paths=[] if version == 0 else list(ARTIFACTS))

    def fixture(self, profile="graph-closure-v1"):
        self.serial += 1
        root = self.home / ("selected-" + str(self.serial))
        root.mkdir()
        for relative in CODE + SCHEMAS + (SNAPSHOT, RAW_GRAPH) + ARTIFACTS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, path)
            path.chmod((ROOT / relative).stat().st_mode & 0o777)
        self.git(root, "-c", "init.defaultBranch=main", "init", "-q")
        hooks = self.home / ("hooks-" + str(self.serial))
        hooks.mkdir()
        for name, value in (
            ("user.name", "TEST V2 fixture"), ("user.email", "fixture@example.invalid"),
            ("commit.gpgsign", "false"), ("core.hooksPath", str(hooks)),
            ("core.attributesFile", os.devnull), ("core.fsmonitor", "false"), ("core.autocrlf", "false"),
        ):
            self.git(root, "config", name, value)
        artifact_commit = self.commit(root)
        raw = (root / RAW_GRAPH).read_bytes()
        self.assertEqual((blob(raw), len(raw), sha(raw)), (GRAPH_BLOB, 38753, GRAPH_SHA))
        graph = json.loads(raw)
        self.assertEqual([(e["from"], e["to"], e["relation"]) for e in graph["edges"][:6]], list(EDGES))
        identity = graph_identity(raw)
        artifacts = [dict(identity=dict(path=p, repository="d6g8k5htny-coder/main",
            commit=artifact_commit, git_mode="100644", git_blob=blob((root / p).read_bytes()),
            bytes=(root / p).stat().st_size, sha256=sha((root / p).read_bytes())),
            raw=(root / p).read_bytes()) for p in ARTIFACTS]
        records = {}
        for node in (SEEDS if profile == "legacy-four-v0" else NODES):
            if profile == "legacy-four-v0":
                value = json.loads((V1_PAYLOAD / (record_path(node) + ".txt")).read_bytes())
            else:
                value = record_fixture(node, identity, graph, artifacts)
            records[node] = value
            path = root / record_path(node)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded(value))
        manifest = self.manifest(profile, records)
        (root / MANIFEST).write_bytes(encoded(manifest))
        checked = self.commit(root)
        return dict(root=root, records=records, manifest=manifest, artifacts=artifacts,
            graph=graph, identity=identity, artifact_commit=artifact_commit,
            context=dict(repository="d6g8k5htny-coder/main", checked_commit=checked,
                run_id="123456", run_attempt="1", python_optimize=sys.flags.optimize))

    def resave(self, fixture, *, manifest=None, records=None):
        root = fixture["root"]
        if manifest is not None:
            fixture["manifest"] = copy.deepcopy(manifest)
            (root / MANIFEST).write_bytes(encoded(manifest))
        if records is not None:
            fixture["records"] = copy.deepcopy(records)
            directory = root / "records/node_status"
            for path in directory.iterdir():
                path.unlink()
            for node, record in records.items():
                (root / record_path(node)).write_bytes(encoded(record))
        fixture["context"]["checked_commit"] = self.commit(root)

    def observer(self, fixture, **options):
        api = SyntheticMathAPI(list(fixture["records"].values()), **options)
        class CountingObserver(subject.ContentObserver):
            def __call__(self, url):
                api.record_calls.append(url)
                return super().__call__(url)
        return CountingObserver(api=api), api

    def evaluate(self, fixture, observer=None):
        observer, _api = self.observer(fixture) if observer is None else (observer, None)
        return evaluate_fresh_records_v2(fixture["root"], observer=observer, identity=fixture["context"])

    def selected(self, fixture):
        return set(CODE + SCHEMAS + (MANIFEST, SNAPSHOT, RAW_GRAPH)) | {
            record_path(n) for n in fixture["records"]} | set(fixture["manifest"]["artifact_paths"])

    def all_bytes(self, fixture):
        return {str(p.relative_to(fixture["root"])): p.read_bytes()
                for p in fixture["root"].rglob("*") if p.is_file() and ".git" not in p.parts}

    def positive(self, fixture):
        before = self.all_bytes(fixture)
        observer, api = self.observer(fixture)
        report = self.evaluate(fixture, observer)
        records = fixture["records"]
        fresh = sorted(n for n, r in records.items() if r["status"] == "fresh")
        expected = {p: source_descriptor(fixture["root"], p) for p in self.selected(fixture)}
        self.assertEqual(report["schema"], "cbz.node-status-live-check.v2")
        self.assertEqual(report["scope"], "NODE_RECORD_FRESHNESS_ONLY")
        self.assertEqual(report["records_total"], len(records))
        self.assertEqual(report["record_versions"], {n: r["schema_version"] for n, r in records.items()})
        self.assertEqual(report["fresh_candidates"], len(fresh))
        self.assertEqual(report["fresh_checked"], len(fresh))
        self.assertEqual(report["fresh_nodes"], fresh)
        self.assertEqual(report["nonfresh_records"], sorted(
            [dict(node_id=n, declared_status=r["status"]) for n, r in records.items()
             if r["status"] != "fresh"], key=lambda r: r["node_id"]))
        self.assertEqual(report["source_bindings"], expected)
        mraw = (fixture["root"] / MANIFEST).read_bytes()
        self.assertEqual(report["manifest_binding"], dict(
            profile=fixture["manifest"]["profile"], path=MANIFEST, bytes=len(mraw), **expected[MANIFEST]))
        requested = [s["url"] for n in fresh for s in records[n]["sources"]]
        self.assertEqual(sorted(api.record_calls), sorted(requested))
        self.assertEqual(report["source_observation_count"], len(requested))
        self.assertEqual(sorted(r["url"] for r in report["source_observations"]), sorted(requested))
        self.assertEqual(report["identity"], fixture["context"])
        self.assertEqual(report["observed_heads"], {"Math-": api.head})
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["cbz_green"], False)
        self.assertEqual(report["pinned_graph_binding"], dict(
            scope="PINNED_MATH_GRAPH_OBJECT_IDENTITY_ONLY", repository=MATH, commit=PIN,
            path=GRAPH_PATH, git_kind="blob", git_mode="100644", git_blob=GRAPH_BLOB,
            bytes=38753, sha256=GRAPH_SHA, snapshot_path=SNAPSHOT, raw_path=RAW_GRAPH,
            snapshot_git_blob=expected[SNAPSHOT]["git_blob"],
            raw_git_blob=expected[RAW_GRAPH]["git_blob"], full_data_equal=True, complete=True))
        self.assertIn("/repos/" + MATH + "/git/commits/" + PIN, api.calls)
        self.assertEqual(self.all_bytes(fixture), before)
        self.assertEqual(self.denied, [])
        return report

    def refuses(self, fixture, observer=None):
        before = self.all_bytes(fixture)
        observer, api = self.observer(fixture) if observer is None else (observer, None)
        with self.assertRaises(subject.LiveCheckError):
            self.evaluate(fixture, observer)
        self.assertEqual(observer.observations, [])
        self.assertEqual(self.all_bytes(fixture), before)
        self.assertEqual(self.denied, [])
        return api

    def retain_invocation(self, label, result, context):
        selected = os.environ.get("NODE_STATUS_V2_TEST_EVIDENCE")
        root = self.home / "invocations" if selected is None else Path(selected)
        self.assertTrue(root.is_absolute() and not root.is_relative_to(ROOT))
        for parent in (root, *root.parents):
            self.assertFalse(parent.is_symlink(), "test evidence path traverses symlink")
        directory = root / self._testMethodName / label
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, body in (("stdout.bin", result.stdout), ("stderr.bin", result.stderr)):
            self.assertIs(type(body), bytes)
            with (directory / name).open("xb") as stream:
                stream.write(body)
        context = dict(context, returncode=result.returncode,
            scope="OWNED_TEST_CLI_INVOCATION_ONLY", http_credit=False,
            actual_actions_authority=False, current_v2_qualified=False, scientific_effect="NONE",
            stdout=dict(bytes=len(result.stdout), sha256=sha(result.stdout)),
            stderr=dict(bytes=len(result.stderr), sha256=sha(result.stderr)))
        with (directory / "observation.json").open("xb") as stream:
            stream.write(encoded(context))

    def refresh_cli(self, fixture, option, label):
        marker = self.home / ("refresh-observer-" + str(self.serial))
        tool = fixture["root"] / "tools/node_status_refresh.py"
        command = [sys.executable, "-B"] + (["-O"] if sys.flags.optimize else []) + [
            "-S", "-c", REFRESH_GUARD, str(tool), str(marker),
            "records/node_status", option]
        environment = dict(self.git_env, PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1", GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="d6g8k5htny-coder/main",
            GITHUB_SHA=fixture["context"]["checked_commit"],
            GITHUB_RUN_ID="123456", GITHUB_RUN_ATTEMPT="1")
        result = subprocess.run(command, cwd=fixture["root"], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.retain_invocation("refresh-" + str(self.serial), result,
            dict(command=command, label=label, option=option,
                 program_root=str(fixture["root"]), checked_commit=fixture["context"]["checked_commit"]))
        return result, marker

    def cli(self, fixture, *, overrides=None, output=None, probe=False):
        self.serial += 1
        output = self.home / ("report-" + str(self.serial) + ".json") if output is None else output
        environment = dict(self.git_env)
        environment.update(GITHUB_ACTIONS="true", GITHUB_REPOSITORY="d6g8k5htny-coder/main",
            GITHUB_SHA=fixture["context"]["checked_commit"], GITHUB_RUN_ID="123456",
            GITHUB_RUN_ATTEMPT="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        for key, value in (overrides or {}).items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        command = [sys.executable, "-B"] + (["-O"] if sys.flags.optimize else []) + ["-S"]
        tool = fixture["root"] / "tools/node_status_live_check.py"
        checkpoint = self.home / ("checkpoint-" + str(self.serial) + ".json")
        if probe:
            environment["GITHUB_TOKEN"] = "SYNTHETIC_TEST_IMPORT_PROBE_NOT_A_CREDENTIAL"
            command += ["-c", IMPORT_STOP, str(tool), str(checkpoint)]
        else:
            command += [str(tool)]
        command += ["--root", str(fixture["root"]), "--output", str(output)]
        existed_before = os.path.lexists(output)
        result = subprocess.run(command, cwd=self.home, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.retain_invocation("cli-" + str(self.serial), result,
            dict(command=command, checked_commit=fixture["context"]["checked_commit"],
                 program_root=str(fixture["root"]), probe=probe))
        return result, output, checkpoint, existed_before

    def refused_cli(self, fixture, **options):
        result, output, checkpoint, existed_before = self.cli(fixture, **options)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr, b"")
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], "cbz.node-status-live-cli.v2")
        self.assertEqual(report["status"], "REFUSED")
        self.assertEqual(report["scientific_effect"], "NONE")
        self.assertIs(report["cbz_green"], False)
        self.assertNotIn("freshness", report)
        if not existed_before and output.exists() and not output.is_symlink() and output.is_file():
            self.assertEqual(output.read_bytes(), result.stdout)
        return report, checkpoint

    def test_legacy_bundle_and_case_targets_are_exact(self):
        report = legacy.verify_bundle()
        self.assertEqual(set(report), {"schema", "source_repository", "source_commit", "source_tree",
            "checked_commit", "checked_tree", "members", "helper_ids", "cli_ids", "source_bindings"})
        self.assertEqual(report["source_commit"], "d09f3252e4f9eb91d624f42357e81e3e43a658e5")
        self.assertEqual(report["source_tree"], "111ec264e36b15a03bb3c5150247003198fc1609")
        members = report["members"]
        self.assertEqual([r["source_path"] for r in members], sorted(EXPECTED_LEGACY_BLOBS))
        self.assertEqual({r["source_path"]: r["git_blob"] for r in members}, EXPECTED_LEGACY_BLOBS)
        derived = []
        for relative, module_name, class_name, count in (
            ("tests/test_node_status_live_check.py", "tests.test_node_status_live_check",
             "NodeStatusLiveContract", 23),
            ("tests/test_node_status_live_cli.py", "tests.test_node_status_live_cli",
             "NodeStatusLiveCLIContract", 20),
        ):
            raw = (V1_PAYLOAD / (relative + ".txt")).read_bytes()
            self.assertEqual(blob(raw), EXPECTED_LEGACY_BLOBS[relative])
            tree = ast.parse(raw.decode("utf-8"), filename=relative)
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
            self.assertEqual(len(classes), 1)
            methods = [n.name for n in classes[0].body
                       if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
            self.assertEqual(len(methods), count)
            self.assertEqual(len(set(methods)), count)
            expected_ids = [module_name + "." + class_name + "." + name for name in methods]
            wrapper = importlib.import_module(module_name)
            actual_names = unittest.defaultTestLoader.getTestCaseNames(getattr(wrapper, class_name))
            self.assertEqual(set(actual_names), set(methods))
            derived.append(expected_ids)
        self.assertEqual(tuple(derived[0]), legacy.HELPER_IDS)
        self.assertEqual(tuple(derived[1]), legacy.CLI_IDS)
        self.assertEqual(report["helper_ids"], derived[0])
        self.assertEqual(report["cli_ids"], derived[1])
        for row in members:
            path = row["source_path"]
            self.assertEqual(row["payload_path"], "tests/fixtures/node_status_live_v1/payload/" + path + ".txt")
            self.assertEqual(row["materialized_path"], path)
            self.assertEqual(row["git_mode"], "100644")
            raw = (ROOT / row["payload_path"]).read_bytes()
            self.assertEqual((len(raw), sha(raw), blob(raw)), (row["bytes"], row["sha256"], row["git_blob"]))
        destination = self.home / "materialized"
        destination.mkdir()
        self.assertEqual(legacy.materialize_bundle(destination)["members"], members)
        self.assertEqual(legacy.verify_materialized(destination)["members"], members)
        changed = destination / "tools/node_status_live_check.py"
        changed.write_bytes(changed.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            legacy.verify_materialized(destination)

    def test_closed_manifest_fields_and_profile_types_are_enforced(self):
        for profile in ("legacy-four-v0", "graph-closure-v1"):
            self.positive(self.fixture(profile))
        changes = (
            lambda m: m.update(extra=True), lambda m: m.pop("profile"),
            lambda m: m.update(schema=2), lambda m: m.update(profile="current"),
            lambda m: m.update(records={}), lambda m: m["records"][0].update(record_version=True),
            lambda m: m["records"][0].update(record_version=1.0),
            lambda m: m["schema_paths"][0].update(record_version=False),
            lambda m: m["records"][0].update(extra="authority"),
            lambda m: m["records"][0].update(node_id=True),
            lambda m: m["records"][0].update(path="records/node_status/../escape.json"),
            lambda m: m["schema_paths"][1].update(record_version=1.0),
            lambda m: m.update(authenticated=True),
        )
        for change in changes:
            with self.subTest(change=change):
                f = self.fixture()
                value = copy.deepcopy(f["manifest"])
                change(value)
                self.resave(f, manifest=value)
                self.refuses(f)

    def test_manifest_cannot_select_code_schema_graph_or_unlisted_paths(self):
        self.positive(self.fixture())
        for field, value in (
            ("code_paths", list(CODE[:-1]) + ["tools/unrelated.py"]),
            ("code_paths", list(reversed(CODE))),
            ("schema_paths", [dict(record_version=0, path=SCHEMAS[0])]),
            ("graph", dict(snapshot_path=SNAPSHOT, raw_path="other/GRAPH.json")),
            ("graph", dict(snapshot_path=SNAPSHOT, raw_path="docs/site/dependency-source/./GRAPH.json")),
            ("graph", dict(snapshot_path=SNAPSHOT, raw_path=RAW_GRAPH, repository="other/repo")),
            ("artifact_paths", ["/tmp/evidence"]),
            ("artifact_paths", list(ARTIFACTS) + ["unrelated.txt"]),
            ("artifact_paths", list(reversed(ARTIFACTS))),
        ):
            with self.subTest(field=field, value=value):
                f = self.fixture()
                value_manifest = copy.deepcopy(f["manifest"])
                value_manifest[field] = value
                self.resave(f, manifest=value_manifest)
                self.refuses(f)

    def test_legacy_profile_requires_exact_four_v0_records(self):
        report = self.positive(self.fixture("legacy-four-v0"))
        self.assertEqual(report["records_total"], 4)
        self.assertEqual(report["fresh_checked"], 4)
        for mutate in (
            lambda records: records.pop(SEEDS[0]),
            lambda records: records.update({PARENT: copy.deepcopy(records[SEEDS[0]])}),
            lambda records: records[SEEDS[0]].update(schema_version=1),
            lambda records: records[SEEDS[0]].update(depends_on=[SEEDS[1]]),
            lambda records: records[SEEDS[0]].update(artifacts=[ARTIFACTS[0]]),
            lambda records: records[SEEDS[0]].update(lemma_closed=True),
        ):
            f = self.fixture("legacy-four-v0")
            records = copy.deepcopy(f["records"])
            mutate(records)
            self.resave(f, records=records)
            self.refuses(f)
        f = self.fixture("legacy-four-v0")
        (f["root"] / record_path(SEEDS[0])).rename(f["root"] / "records/node_status/unexpected.json")
        f["context"]["checked_commit"] = self.commit(f["root"])
        self.refuses(f)

    def test_graph_profile_requires_complete_ordered_nine_record_closure(self):
        f = self.fixture()
        report = self.positive(f)
        self.assertEqual(report["records_total"], 9)
        self.assertEqual(report["record_versions"], dict.fromkeys(NODES, 1))
        actual = [(n, e["node_id"], e["relation"]) for n, r in f["records"].items()
                  for e in r.get("dependency_evidence", [])]
        self.assertEqual(actual, list(EDGES))
        for mutation in ("omit", "target", "relation", "reorder", "required", "incoming"):
            with self.subTest(mutation=mutation):
                f = self.fixture()
                records = copy.deepcopy(f["records"])
                if mutation == "omit":
                    records.pop(COMPONENTS[0])
                elif mutation == "target":
                    records[PARENT]["depends_on"][0] = SEEDS[0]
                    records[PARENT]["dependency_evidence"][0]["node_id"] = SEEDS[0]
                elif mutation == "relation":
                    records[PARENT]["dependency_evidence"][0]["relation"] = "component_of"
                elif mutation == "reorder":
                    records[PARENT]["depends_on"].reverse()
                    records[PARENT]["dependency_evidence"].reverse()
                elif mutation == "required":
                    records[PARENT]["dependency_evidence"][0]["required"] = False
                else:
                    records[SEEDS[0]]["depends_on"] = [SEEDS[1]]
                    records[SEEDS[0]]["dependency_evidence"] = [
                        dict(node_id=SEEDS[1], relation="crosswalk", required=True)]
                self.resave(f, records=records)
                self.refuses(f)

    def test_missing_extra_duplicate_and_same_count_inputs_refuse(self):
        self.positive(self.fixture())
        for mutation in ("missing-manifest", "duplicate-key", "duplicate-record", "same-count",
                         "missing-artifact", "wrong-artifact-owner", "missing-closure-entry", "extra-file"):
            with self.subTest(mutation=mutation):
                f = self.fixture()
                root, manifest = f["root"], copy.deepcopy(f["manifest"])
                if mutation == "missing-manifest":
                    (root / MANIFEST).unlink()
                elif mutation == "duplicate-key":
                    raw = encoded(manifest).replace(b'"profile":', b'"profile":"graph-closure-v1","profile":', 1)
                    (root / MANIFEST).write_bytes(raw)
                elif mutation == "duplicate-record":
                    manifest["records"].append(copy.deepcopy(manifest["records"][0]))
                    (root / MANIFEST).write_bytes(encoded(manifest))
                elif mutation == "same-count":
                    manifest["records"][0]["node_id"] = "math.not-the-closure"
                    manifest["records"][0]["path"] = record_path("math.not-the-closure")
                    (root / MANIFEST).write_bytes(encoded(manifest))
                elif mutation == "missing-artifact":
                    (root / ARTIFACTS[0]).unlink()
                elif mutation == "wrong-artifact-owner":
                    records = copy.deepcopy(f["records"])
                    records[SEEDS[0]]["artifacts"] = list(ARTIFACTS)
                    records[SEEDS[0]]["artifact_evidence"] = copy.deepcopy(records[SEEDS[1]]["artifact_evidence"])
                    self.resave(f, records=records)
                elif mutation == "missing-closure-entry":
                    manifest["records"].pop()
                    (root / MANIFEST).write_bytes(encoded(manifest))
                else:
                    (root / "records/node_status/extra.txt").write_bytes(b"TEST")
                f["context"]["checked_commit"] = self.commit(root)
                self.refuses(f)
        for malformed in (
            b"\xff",
            b'{"schema":NaN}',
            b'{"schema":1e999}',
            encoded(self.manifest("graph-closure-v1", NODES)) + b"{}",
            b" " * 65537,
            b"[" * 1000 + b"0" + b"]" * 1000,
        ):
            with self.subTest(raw_bytes=len(malformed)):
                f = self.fixture()
                (f["root"] / MANIFEST).write_bytes(malformed)
                f["context"]["checked_commit"] = self.commit(f["root"])
                self.refuses(f)

        for false_claim in ("orphan", "ancestor-wrong-path-object"):
            with self.subTest(native_artifact_claim=false_claim):
                f = self.fixture()
                root = f["root"]
                if false_claim == "orphan":
                    tree = self.git(root, "rev-parse", f["artifact_commit"] + "^{tree}")
                    claimed = self.git(root, "commit-tree", tree, "-m", "TEST orphan artifact cut")
                    self.assertNotIn(claimed, self.git(root, "rev-list", "HEAD").splitlines())
                else:
                    path = root / ARTIFACTS[0]
                    original = path.read_bytes()
                    path.write_bytes(original + b"\nTEST different historical path object\n")
                    claimed = self.commit(root)
                    path.write_bytes(original)
                    historical = self.git(root, "ls-tree", claimed, "--", ARTIFACTS[0])
                    self.assertNotEqual(historical.split()[2], blob(original))
                records = copy.deepcopy(f["records"])
                frames = copy.deepcopy(f["artifacts"])
                for row in records[SEEDS[1]]["artifact_evidence"]:
                    row["commit"] = claimed
                for frame in frames:
                    frame["identity"]["commit"] = claimed
                self.resave(f, records=records)
                if false_claim == "ancestor-wrong-path-object":
                    self.assertIn(claimed, self.git(root, "rev-list", "HEAD").splitlines())
                else:
                    self.assertNotIn(claimed, self.git(root, "rev-list", "HEAD").splitlines())
                pure_inputs = dict(schema="node-status-v1.source-inputs.v1",
                    graph=dict(identity=f["identity"], raw=(root / RAW_GRAPH).read_bytes()), artifacts=frames)
                pure_records = [(n + ".json", encoded(r)) for n, r in sorted(records.items())]
                self.assertEqual(check_v1_population(pure_records, pure_inputs), records)
                self.refuses(f)

    def test_native_git_bytes_modes_and_final_capture_are_required(self):
        self.positive(self.fixture())
        for mutation in ("bytes", "mode", "symlink", "head"):
            with self.subTest(mutation=mutation):
                f = self.fixture()
                path = f["root"] / CODE[1]
                if mutation == "bytes":
                    self.git(f["root"], "update-index", "--assume-unchanged", "--", CODE[1])
                    path.write_bytes(path.read_bytes() + b"\n")
                elif mutation == "mode":
                    path.chmod(0o755 if not path.stat().st_mode & stat.S_IXUSR else 0o644)
                elif mutation == "symlink":
                    target = self.home / ("owned-target-" + str(self.serial))
                    target.write_bytes(path.read_bytes())
                    path.unlink()
                    path.symlink_to(target)
                else:
                    self.commit(f["root"])  # Current identity intentionally retains old HEAD.
                self.refuses(f)
        for relative in (MANIFEST, RAW_GRAPH, record_path(SEEDS[0]), CODE[2]):
            f = self.fixture()
            observer, api = self.observer(f, hook=lambda f=f, p=relative:
                (f["root"] / p).write_bytes((f["root"] / p).read_bytes() + b"\n"))
            with self.assertRaises(subject.LiveCheckError):
                self.evaluate(f, observer)
            self.assertEqual(self.denied, [])

    def test_project_imports_execute_only_captured_originals(self):
        for dependency in ("node_status_v1", "validate_node_status", "node_status_refresh"):
            for form in ("package", "bytecode"):
                with self.subTest(dependency=dependency, form=form):
                    f = self.fixture()
                    source = f["root"] / "tools" / (dependency + ".py")
                    marker = self.home / (dependency + "-" + form + ".executed")
                    wrapper = ("from pathlib import Path\nPath(" + repr(str(marker))
                        + ").write_text('TEST_UNBOUND_EXECUTED')\n__file__ = " + repr(str(source))
                        + "\nexec(compile(" + repr(source.read_bytes())
                        + ", __file__, 'exec', dont_inherit=True), globals())\n")
                    if form == "package":
                        package = source.with_suffix("")
                        package.mkdir()
                        (package / "__init__.py").write_text(wrapper)
                    else:
                        code = compile(wrapper, str(source), "exec", dont_inherit=True,
                                       optimize=sys.flags.optimize)
                        info = source.stat()
                        header = importlib.util.MAGIC_NUMBER + struct.pack(
                            "<III", 0, int(info.st_mtime) & 0xffffffff, info.st_size & 0xffffffff)
                        cache = Path(importlib.util.cache_from_source(str(source),
                            optimization="" if not sys.flags.optimize else str(sys.flags.optimize)))
                        cache.parent.mkdir(exist_ok=True)
                        cache.write_bytes(header + marshal.dumps(code))
                    report, checkpoint = self.refused_cli(f, probe=True)
                    self.assertTrue(checkpoint.is_file(), "bound run_live_v2 entry was not reached")
                    reached = json.loads(checkpoint.read_bytes())
                    self.assertEqual(reached, dict(phase="BOUND_RUN_LIVE_V2_ENTRY",
                        filename=str(source.parent / "node_status_live_check.py"),
                        root=str(f["root"]), identity=f["context"], optimize=sys.flags.optimize))
                    self.assertFalse(marker.exists(), "unbound project module executed")
                    self.assertEqual(report["source_bindings"], report["source_bindings_after"])

    def test_pinned_math_graph_joins_local_identity_raw_and_embedded_data(self):
        f = self.fixture()
        report = self.positive(f)
        self.assertEqual(report["pinned_graph_binding"]["git_blob"], f["identity"]["git_blob"])
        for option in ("wrong_graph", "missing_graph", "bad_kind", "bad_mode", "truncated"):
            f = self.fixture()
            observer, api = self.observer(f, **{option: True})
            with self.assertRaises(subject.LiveCheckError):
                self.evaluate(f, observer)
            self.assertEqual(api.record_calls, [], "record freshness borrowed before Graph join")
        f = self.fixture()
        snapshot = json.loads((f["root"] / SNAPSHOT).read_bytes())
        snapshot["graph"]["data"]["nodes"][PARENT]["TEST_embedded_only"] = "changed"
        (f["root"] / SNAPSHOT).write_bytes(encoded(snapshot))
        f["context"]["checked_commit"] = self.commit(f["root"])
        self.refuses(f)

    def test_graph_substitutions_cannot_borrow_unrelated_native_evidence(self):
        f = self.fixture()
        self.positive(f)
        for declared_pin in ("e" * 40, "9bdfc54b76173b57912503bfcf901a5f9d43b09c"):
            with self.subTest(coherent_source_commit=declared_pin):
                other = self.fixture()
                identity = dict(other["identity"], commit=declared_pin)
                records = copy.deepcopy(other["records"])
                for node, record in records.items():
                    record["graph_source"] = copy.deepcopy(identity)
                    if node in SUPPORT:
                        kind, relative = SUPPORT[node]
                        record["sources"][0] = dict(
                            url="https://github.com/" + MATH + "/" + kind + "/" + declared_pin + "/" + relative,
                            rev=declared_pin)
                        record["source_rev"] = declared_pin
                raw = (other["root"] / RAW_GRAPH).read_bytes()
                pure = dict(schema="node-status-v1.source-inputs.v1",
                    graph=dict(identity=identity, raw=raw), artifacts=other["artifacts"])
                files = [(n + ".json", encoded(r)) for n, r in sorted(records.items())]
                self.assertEqual(check_v1_population(files, pure), records)
                snapshot = json.loads((other["root"] / SNAPSHOT).read_bytes())
                snapshot["graph"]["identity"]["commit"] = declared_pin
                (other["root"] / SNAPSHOT).write_bytes(encoded(snapshot))
                self.resave(other, records=records)
                observer, api = self.observer(other)
                # Synthetic API can resolve ANY40hex to same original220b;
                # fixed dbe policy must not be selected from that declaration.
                with self.assertRaises(subject.LiveCheckError):
                    self.evaluate(other, observer)
                self.assertEqual(api.record_calls, [])
        # Change inert content outside the six-edge acceptance rules. Pure v1
        # consistency remains valid, while native pinned Math still names220b.
        graph = copy.deepcopy(f["graph"])
        graph["nodes"][PARENT]["TEST_coherent_substitution"] = "different original"
        raw = encoded(graph)
        identity = graph_identity(raw)
        records = copy.deepcopy(f["records"])
        for record in records.values():
            record["graph_source"] = copy.deepcopy(identity)
        frame = dict(schema="node-status-v1.source-inputs.v1",
            graph=dict(identity=identity, raw=raw), artifacts=f["artifacts"])
        record_files = [(node + ".json", encoded(record)) for node, record in sorted(records.items())]
        self.assertEqual(check_v1_population(record_files, frame), records)
        snapshot = json.loads((f["root"] / SNAPSHOT).read_bytes())
        snapshot["graph"] = dict(identity=dict(repository=MATH, commit=PIN, path=GRAPH_PATH,
            blob=blob(raw), bytes=len(raw), sha256=sha(raw)), data=graph)
        (f["root"] / RAW_GRAPH).write_bytes(raw)
        (f["root"] / SNAPSHOT).write_bytes(encoded(snapshot))
        self.resave(f, records=records)
        observer, api = self.observer(f)
        with self.assertRaises(subject.LiveCheckError):
            self.evaluate(f, observer)
        self.assertEqual(api.record_calls, [])
        # Correct Graph join cannot freshen whole old5/current11 reconciliation.
        f = self.fixture()
        records = copy.deepcopy(f["records"])
        records[COMPONENTS[3]]["status"] = "fresh"
        self.resave(f, records=records)
        observer, api = self.observer(f)
        with self.assertRaises(subject.LiveCheckError):
            self.evaluate(f, observer)
        self.assertEqual(len(OLD_RECON), 5)
        self.assertEqual(len(CURRENT_RECON), 11)

    def test_refresh_validation_precedes_observation_or_writes(self):
        self.positive(self.fixture())
        for mutation in ("invalid-version", "duplicate-json", "incomplete", "missing-manifest"):
            with self.subTest(mutation=mutation):
                f = self.fixture()
                if mutation == "invalid-version":
                    records = copy.deepcopy(f["records"])
                    records[SEEDS[0]]["schema_version"] = True
                    self.resave(f, records=records)
                elif mutation == "duplicate-json":
                    path = f["root"] / record_path(SEEDS[0])
                    path.write_bytes(path.read_bytes().replace(b'"owner":null', b'"owner":null,"owner":null', 1))
                    f["context"]["checked_commit"] = self.commit(f["root"])
                elif mutation == "incomplete":
                    records = copy.deepcopy(f["records"])
                    records.pop(COMPONENTS[0])
                    self.resave(f, records=records)
                else:
                    (f["root"] / MANIFEST).unlink()
                    f["context"]["checked_commit"] = self.commit(f["root"])
                observer, api = self.observer(f)
                before = self.all_bytes(f)
                with self.assertRaises(subject.LiveCheckError):
                    self.evaluate(f, observer)
                self.assertEqual(api.record_calls, [])
                self.assertEqual(observer.observations, [])
                self.assertEqual(self.all_bytes(f), before)
        for option in ("--check", "--write"):
            f = self.fixture()
            before = self.all_bytes(f)
            result, marker = self.refresh_cli(f, option, "VALID_PROFILE_BASELINE")
            self.assertNotEqual(result.returncode, 0)  # intentional trace stop, not success.
            self.assertTrue(marker.is_file(), "valid admission never reached actual observer constructor")
            self.assertEqual(json.loads(marker.read_bytes()), dict(
                phase="ACTUAL_CONTENT_OBSERVER_ENTRY",
                filename=str(f["root"] / "tools/node_status_refresh.py"),
                root=str(f["root"]), optimize=sys.flags.optimize))
            self.assertEqual(self.all_bytes(f), before)
        for mutation in ("missing-manifest", "invalid-v1", "incomplete"):
            for option in ("--check", "--write"):
                with self.subTest(refresh=option, invalid=mutation):
                    f = self.fixture()
                    if mutation == "missing-manifest":
                        (f["root"] / MANIFEST).unlink()
                        f["context"]["checked_commit"] = self.commit(f["root"])
                    else:
                        records = copy.deepcopy(f["records"])
                        if mutation == "invalid-v1":
                            records[SEEDS[0]]["schema_version"] = True
                        else:
                            records.pop(COMPONENTS[0])
                        self.resave(f, records=records)
                    before = self.all_bytes(f)
                    result, marker = self.refresh_cli(f, option, mutation)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stderr, b"")
                    refusal = json.loads(result.stdout)
                    self.assertEqual(refusal.get("error_code"), "VALIDATION_REFUSED")
                    self.assertEqual(refusal.get("scientific_effect"), "NONE")
                    self.assertTrue(isinstance(refusal.get("validation_errors"), list)
                                    and refusal["validation_errors"])
                    self.assertTrue(all(isinstance(row, str) and row for row in refusal["validation_errors"]))
                    self.assertFalse(refusal.get("report"))
                    self.assertFalse(marker.exists(), "observer entered before current validation refusal")
                    self.assertEqual(self.all_bytes(f), before)

        f = self.fixture()
        records = copy.deepcopy(f["records"])
        for record in records.values():
            record["status"] = "unknown"
        self.resave(f, records=records)
        self.refuses(f)  # no declared-fresh candidate is never an empty success.

    def test_v2_reports_and_current_cli_require_real_coverage_context_and_output(self):
        f = self.fixture()
        report = self.positive(f)
        self.assertEqual(len(report["nonfresh_records"]), 5)
        self.assertEqual(report["fresh_nodes"], sorted(SEEDS))
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(subject, "ContentObserver", side_effect=AssertionError("observer constructed")):
                with self.assertRaises(subject.LiveCheckError):
                    run_live_v2(f["root"], identity=f["context"])
        observer, api = self.observer(f)
        with patch.dict(os.environ, {"GITHUB_TOKEN": "SYNTHETIC_TEST_NOT_A_CREDENTIAL"}):
            with patch.object(subject, "ContentObserver", return_value=observer) as factory:
                observed = run_live_v2(f["root"], identity=f["context"])
        factory.assert_called_once_with()
        self.assertEqual(observed["record_versions"], report["record_versions"])
        # These are forged TEST contexts and no-token source/refusal controls,
        # not actual Actions or authenticated live observations.
        valid_refusal, _ = self.refused_cli(f)
        self.assertEqual(valid_refusal["error_code"], "AUTH_REQUIRED")
        self.assertEqual(set(valid_refusal["source_bindings"]), self.selected(f))
        for overrides in (
            {"GITHUB_ACTIONS": None}, {"GITHUB_REPOSITORY": "other/repo"},
            {"GITHUB_SHA": "0" * 40}, {"GITHUB_RUN_ID": "0"}, {"GITHUB_RUN_ATTEMPT": "true"},
        ):
            self.refused_cli(f, overrides=overrides)
        existing = self.home / "existing.json"
        existing.write_bytes(b"KEEP_ORIGINAL")
        self.refused_cli(f, output=existing)
        self.assertEqual(existing.read_bytes(), b"KEEP_ORIGINAL")
        self.refused_cli(f, output=f["root"] / "inside.json")
        target = self.home / "target.json"
        target.write_bytes(b"KEEP_TARGET")
        link = self.home / "link.json"
        link.symlink_to(target)
        self.refused_cli(f, output=link)
        self.assertEqual(target.read_bytes(), b"KEEP_TARGET")


if __name__ == "__main__":
    unittest.main()
