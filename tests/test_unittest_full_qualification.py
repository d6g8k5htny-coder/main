"""TEST-only contracts for the absent Main unittest qualification helper.

Every repository, native/API response, archive and architecture log constructed
here is a synthetic contract fixture. Inert API consistency is exercised; no
supplied dict establishes transport authentication, a live run, custody, or G5.
No original conditional test, real project record, browser or Lean body runs.

The subject and project dependencies are imported from the checked source
checkout, never copied into a fixture import directory. Temporary Git trees
are discovery/data inputs only. Actual CLI acquisition remains future work.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Deliberately unconditional: the first hosted observation must be honest
# missing-helper RED. No skip, fallback subject, fake checker or exception trap.
from tools import unittest_full_qualification as subject
from tools.unittest_full_qualification import (
    architecture_inventory, run_core, qualify_join,
)
from architecture.graph_artifact_readback import read_graph_archive

if Path(subject.__file__).resolve() != ROOT / "tools/unittest_full_qualification.py":
    raise AssertionError("subject import escaped the captured source checkout")

ARCHIVE_FIXTURE_BLOB = "a46ef3acc5d10ccc256259cd14c4c098d6bebd47"
SOURCE_SHA = "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09"
CATALOG_SHA = "b51eaf6cfaede42105ad377c066c4b18789e320ccf542b6817a6850c350fc9ef"
CORE_ID = "test_architecture_impostor.Core.test_core"
MEMBERS = (
    "source-before.json", "source-after.json", "full-census.json",
    "architecture-census.json", "events.jsonl", "stdout.bin", "stderr.bin",
    "core-result.json", "ARTIFACT_SHA256SUMS",
)
CONTEXT_KEYS = {"repository", "checked_commit", "checked_tree", "run_id", "run_attempt"}
SOURCE_KEYS = {"schema", "context", "files"}
INVENTORY_KEYS = {
    "schema", "context", "source_sha256", "mode", "profile", "discovery",
    "runtime", "tests", "count", "loader_errors",
}
RESULT_KEYS = {
    "schema", "context", "mode", "profile", "purpose", "source_before_sha256",
    "source_after_sha256", "full_inventory_sha256", "architecture_inventory_sha256",
    "selected_ids", "journal", "stdout", "stderr", "exit_code", "complete",
}
JOIN_KEYS = {
    "schema", "context", "scope", "status", "scientific_effect", "g5_claim",
    "cbz_claim", "source_sha256", "native_sha256", "core", "architecture", "coverage",
}
RUNTIME_KEYS = {
    "implementation", "python", "executable_sha256", "optimize",
    "no_site", "dont_write_bytecode",
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.check_output(
        ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
         "core.hooksPath=/dev/null", "-C", str(root), *args], timeout=20,
        env={**{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
             "GIT_NO_REPLACE_OBJECTS": "1"}, stderr=subprocess.PIPE
    ).decode("utf-8").strip()


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def archive(members):
    """Fixture writer only; never an acceptance parser."""
    bodies = dict(members)
    bodies.pop("ARTIFACT_SHA256SUMS", None)
    bodies["ARTIFACT_SHA256SUMS"] = "".join(
        sha(raw) + "  " + name + "\n" for name, raw in sorted(bodies.items())
    ).encode("ascii")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as result:
        for name, raw in sorted(bodies.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 10, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            result.writestr(info, raw)
    return stream.getvalue()


def unpack(raw):
    """Read our own producer fixtures for mutation; no acceptance verdict."""
    with zipfile.ZipFile(io.BytesIO(raw)) as source:
        return {name: source.read(name) for name in source.namelist()}


def mutate_core(raw, name, change):
    members = unpack(raw)
    value = json.loads(members[name])
    change(value)
    members[name] = encoded(value)
    result = json.loads(members["core-result.json"])
    references = {
        "source-before.json": "source_before_sha256",
        "source-after.json": "source_after_sha256",
        "full-census.json": "full_inventory_sha256",
        "architecture-census.json": "architecture_inventory_sha256",
    }
    if name in references:
        result[references[name]] = sha(members[name])
    members["core-result.json"] = encoded(result)
    return archive(members)


def mutate_journal(raw, change):
    members = unpack(raw)
    rows = [json.loads(line) for line in members["events.jsonl"].splitlines()]
    change(rows)
    members["events.jsonl"] = b"".join(encoded(row) for row in rows)
    result = json.loads(members["core-result.json"])
    result["journal"] = {"bytes": len(members["events.jsonl"]),
                         "sha256": sha(members["events.jsonl"])}
    members["core-result.json"] = encoded(result)
    return archive(members)


def existing_archive_fixture(cleanup):
    """Reuse existing pinned TEST producers; execute no existing test methods."""
    path = ROOT / "tests/test_architecture_graph_artifact_readback.py"
    raw = path.read_bytes()
    if git_blob(raw) != ARCHIVE_FIXTURE_BLOB:
        raise AssertionError("captured TEST archive producer changed")
    spec = importlib.util.spec_from_file_location("TEST_archive_producers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if len(module.REQUIRED_IDS) != 89:
        raise AssertionError("TEST catalog count changed")
    if sha(("\n".join(sorted(module.REQUIRED_IDS)) + "\n").encode("ascii")) != CATALOG_SHA:
        raise AssertionError("TEST catalog digest changed")
    case = module.ArchitectureGraphArtifactReadbackContract("runTest")
    case.setUp()
    cleanup(case.doCleanups)
    return module, case


CORE_SOURCE = r'''
import json, os, unittest
from pathlib import Path
ORIGINALS = []
def note(value):
    path = os.environ.get("TEST_TRACE")
    if path:
        with open(path, "a") as stream: stream.write(value + "\n")
def setUpModule(): note("module+")
def tearDownModule(): note("module-")
class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        note("class+")
        cls.addClassCleanup(note, "class-cleanup")
    @classmethod
    def tearDownClass(cls): note("class-")
    def id(self): return "test_architecture_impostor.Core." + self._testMethodName
    def test_core(self):
        self.assertIs(self, ORIGINALS[0])
        self.addCleanup(note, "case-cleanup")
        note("body")
        print("nested synthetic child: FAILED (skipped=1)")
def load_tests(loader, tests, pattern):
    global ORIGINALS
    note("load")
    def leaves(node):
        if isinstance(node, unittest.TestSuite):
            return [case for child in node for case in leaves(child)]
        return [node]
    ORIGINALS = leaves(tests)
    return tests
'''

CORE_DRIVER = r'''
import hashlib, importlib, json, os, pathlib, sys
captured, fixture, context_file, output, expected_subject_sha = sys.argv[1:]
captured = pathlib.Path(captured).resolve()
subject_path = captured / "tools/unittest_full_qualification.py"
if hashlib.sha256(subject_path.read_bytes()).hexdigest() != expected_subject_sha:
    raise AssertionError("subject bytes changed before import")
sys.path.insert(0, str(captured))
module = importlib.import_module("tools.unittest_full_qualification")
actual = pathlib.Path(module.__file__).resolve()
if actual != captured / "tools/unittest_full_qualification.py":
    raise AssertionError("subject escaped captured checkout")
if hashlib.sha256(actual.read_bytes()).hexdigest() != expected_subject_sha:
    raise AssertionError("subject bytes changed before child import")
context = json.loads(pathlib.Path(context_file).read_bytes())
for key, value in (
    ("GITHUB_REPOSITORY", context["repository"]),
    ("GITHUB_SHA", context["checked_commit"]),
    ("GITHUB_RUN_ID", context["run_id"]),
    ("GITHUB_RUN_ATTEMPT", context["run_attempt"]),
):
    os.environ[key] = value
# This is a synthetic API fixture context, never authenticated native context.
fixture = pathlib.Path(fixture).resolve()
os.chdir(fixture)
sys.path[0] = str(fixture)
sys.path.insert(1, str(captured))  # admitted same-source project dependencies
inventory = module.architecture_inventory(
    fixture, context, pathlib.Path(output).with_suffix(".architecture.json"))
raise SystemExit(module.run_core(fixture, context, inventory, pathlib.Path(output)))
'''


class FixtureTree:
    """Synthetic Git/data tree; subject imports remain in captured ROOT."""
    def __init__(self, parent, ids, *, core=CORE_SOURCE, architecture_override=None):
        self.root = parent / "TEST-repository"
        (self.root / "tests").mkdir(parents=True)
        grouped = {}
        for identity in ids:
            module, owner, method = identity.split(".")
            grouped.setdefault(module, {}).setdefault(owner, []).append(method)
        for module, owners in sorted(grouped.items()):
            lines = ["import unittest"]
            for owner, methods in sorted(owners.items()):
                lines.append("class " + owner + "(unittest.TestCase):")
                lines.extend("    def " + method + "(self): self.assertTrue(True)"
                             for method in sorted(methods))
            (self.root / "tests" / (module + ".py")).write_text("\n".join(lines) + "\n")
        if architecture_override is not None:
            for path in (self.root / "tests").glob("test_architecture_*.py"):
                path.unlink()
            (self.root / "tests/test_architecture_fixture.py").write_text(
                textwrap.dedent(architecture_override))
        if core is not None:
            (self.root / "tests/test_core_fixture.py").write_text(textwrap.dedent(core))
        graph = (ROOT / "docs/site/dependency-source/GRAPH.json").read_bytes()
        if len(graph) != 38753 or sha(graph) != SOURCE_SHA:
            raise AssertionError("captured public graph source changed")
        target = self.root / "docs/site/dependency-source/GRAPH.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(graph)
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "Synthetic TEST fixture")
        git(self.root, "config", "user.email", "TEST@example.invalid")
        git(self.root, "add", ".")
        git(self.root, "commit", "-q", "-m", "Synthetic unittest contract inputs")
        self.context = {
            "repository": "TEST-owner/TEST-main-unittest-fixture",
            "checked_commit": git(self.root, "rev-parse", "HEAD"),
            "checked_tree": git(self.root, "rev-parse", "HEAD^{tree}"),
            "run_id": "314159", "run_attempt": "1",
        }
        self.context_path = parent / "TEST-context.json"
        self.context_path.write_bytes(encoded(self.context))
        self.parent = parent

    def run(self, mode, label):
        out = self.parent / ("TEST-" + label + "-" + mode)
        trace = out.with_suffix(".trace")
        env = dict(os.environ, TEST_TRACE=str(trace),
                   TEST_IMPORT_TRACE=str(out.with_suffix(".imports")))
        env.pop("PYTHONOPTIMIZE", None)
        flags = ["-B"] + (["-O"] if mode == "optimized" else []) + ["-S"]
        result = subprocess.run(
            [sys.executable, *flags, "-c", CORE_DRIVER, str(ROOT), str(self.root),
             str(self.context_path), str(out),
             sha((ROOT / "tools/unittest_full_qualification.py").read_bytes())],
            cwd=self.root, env=env, capture_output=True, timeout=30, check=False,
        )
        return result, out, trace


def native_fixture(context, cores, architecture_binding, architecture_zip):
    """Inert response bytes/projections, not HTTP or transport authentication."""
    base = "https://api.github.com/repos/" + context["repository"]
    producer = {"core_result": "success", "architecture_result": "success",
                "architecture_outputs": dict(architecture_binding)}
    run_path = "TEST-api-run-attempt.json"
    run_body = {
        "id": int(context["run_id"]), "run_attempt": int(context["run_attempt"]),
        "head_sha": context["checked_commit"], "status": "in_progress",
        "conclusion": None, "event": "workflow_dispatch",
        "path": ".github/workflows/full-unittest-qualification.yml",
        "repository": {"full_name": context["repository"]},
        "head_repository": {"full_name": context["repository"]},
    }
    originals = {run_path: encoded(run_body)}
    urls = {run_path: base + "/actions/runs/" + context["run_id"]
            + "/attempts/" + context["run_attempt"]}
    rows = []
    for profile, mode, identity, name, raw in (
        ("architecture", "both", architecture_binding["artifact_id"],
         architecture_binding["artifact_name"], architecture_zip),
        ("core", "normal", "457", "main-unittest-core-normal-"
         + context["run_id"] + "-" + context["run_attempt"], cores["normal"]),
        ("core", "optimized", "458", "main-unittest-core-optimized-"
         + context["run_id"] + "-" + context["run_attempt"], cores["optimized"]),
    ):
        url = base + "/actions/artifacts/" + identity
        path = "TEST-api-artifact-" + identity + ".json"
        originals[path] = encoded({
            "id": int(identity), "name": name, "size_in_bytes": len(raw),
            "expired": False, "digest": "sha256:" + sha(raw), "url": url,
            "archive_download_url": url + "/zip",
            "workflow_run": {"id": int(context["run_id"]), "repository_id": 1,
                             "head_repository_id": 1, "head_branch": "TEST",
                             "head_sha": context["checked_commit"]},
        })
        urls[path] = url
        rows.append({
            "profile": profile, "mode": mode, "id": identity, "name": name,
            "run_id": context["run_id"], "head_sha": context["checked_commit"],
            "expired": False, "digest": sha(raw), "archive_bytes": len(raw),
        })
    record = {
        "schema": "main-unittest.native.v1", "context": dict(context),
        "producer": producer,
        "api_originals": [{"path": path, "url": urls[path], "bytes": len(raw),
                           "sha256": sha(raw)} for path, raw in sorted(originals.items())],
        "artifacts": rows,
    }
    return producer, (record, originals)


class MainUnittestQualificationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="TEST-main-unittest-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.home = Path(cls.temporary.name)
        cls.fixture_module, cls.fixture_case = existing_archive_fixture(cls.addClassCleanup)
        cls.arch_ids = tuple(sorted(cls.fixture_module.REQUIRED_IDS))
        cls.tree = FixtureTree(cls.home, cls.arch_ids)
        cls.cores = {}
        cls.outputs = {}
        for mode in ("normal", "optimized"):
            process, out, trace = cls.tree.run(mode, "positive")
            if process.returncode != 0:
                raise AssertionError("real subject rejected TEST positive: "
                                     + repr(process.stdout[-2000:] + process.stderr[-2000:]))
            names = {path.name for path in out.iterdir()}
            if names != set(MEMBERS):
                raise AssertionError("core evidence is not the closed nine-member layout")
            cls.outputs[mode] = (out, trace)
            cls.cores[mode] = archive({name: (out / name).read_bytes() for name in MEMBERS})
        context = {key: cls.tree.context[key] for key in
                   ("repository", "checked_commit", "run_id", "run_attempt")}
        context["artifact_id"] = "456"
        cls.arch_members = cls.fixture_case.fixture(context, cls.arch_ids)
        cls.arch_zip = cls.fixture_module.classic_zip(cls.fixture_case.entries(cls.arch_members))
        cls.binding = cls.fixture_case.expected(cls.arch_zip, cls.arch_members, context)
        cls.producer, cls.native = native_fixture(
            cls.tree.context, cls.cores, cls.binding, cls.arch_zip)

    def join(self, *, cores=None, native=None, producer=None, binding=None,
             architecture_zip=None, context=None):
        cores = self.cores if cores is None else cores
        if native is None:
            producer_built, native = native_fixture(
                self.tree.context if context is None else context, cores,
                self.binding if binding is None else binding,
                self.arch_zip if architecture_zip is None else architecture_zip)
            producer = producer_built if producer is None else producer
        return qualify_join(
            self.tree.root, self.tree.context if context is None else context,
            self.producer if producer is None else producer, native, cores,
            self.arch_zip if architecture_zip is None else architecture_zip,
        )

    def refused(self, **arguments):
        with self.assertRaises(ValueError):
            self.join(**arguments)

    def with_tree(self, *, core=CORE_SOURCE, architecture_override=None):
        temporary = tempfile.TemporaryDirectory(prefix="TEST-discovery-")
        self.addCleanup(temporary.cleanup)
        return FixtureTree(Path(temporary.name), ("test_architecture_fixture.Arch.test_ok",),
                           core=core, architecture_override=architecture_override)

    def assert_positive_core(self, mode):
        members = unpack(self.cores[mode])
        before, after = (json.loads(members[name]) for name in
                         ("source-before.json", "source-after.json"))
        self.assertEqual(set(before), SOURCE_KEYS)
        self.assertEqual(before["schema"], "main-unittest.source.v1")
        self.assertEqual(before, after)
        self.assertEqual(set(before["context"]), CONTEXT_KEYS)
        self.assertEqual(before["context"], self.tree.context)
        expected_paths = git(self.tree.root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        self.assertEqual([row["path"] for row in before["files"]], expected_paths)
        for row in before["files"]:
            self.assertEqual(set(row), {"path", "mode", "git_blob", "bytes", "sha256"})
            raw = (self.tree.root / row["path"]).read_bytes()
            self.assertEqual(row["mode"], "100644")
            self.assertEqual(row["bytes"], len(raw))
            self.assertEqual(row["sha256"], sha(raw))
            self.assertEqual(row["git_blob"], git_blob(raw))
        full, arch = (json.loads(members[name]) for name in
                      ("full-census.json", "architecture-census.json"))
        for census, profile, pattern in (
            (full, "full", "test*.py"),
            (arch, "architecture", "test_architecture_*.py"),
        ):
            self.assertEqual(set(census), INVENTORY_KEYS)
            self.assertEqual(census["schema"], "main-unittest.inventory.v1")
            self.assertEqual(census["context"], self.tree.context)
            self.assertEqual(census["mode"], mode)
            self.assertEqual(census["profile"], profile)
            self.assertEqual(census["source_sha256"], sha(members["source-before.json"]))
            self.assertEqual(census["discovery"],
                             {"start_dir": "tests", "pattern": pattern, "top_level_dir": None})
            self.assertEqual(set(census["runtime"]), RUNTIME_KEYS)
            self.assertEqual(census["runtime"]["implementation"], "cpython")
            self.assertEqual(census["runtime"]["python"], sys.version)
            self.assertEqual(census["runtime"]["executable_sha256"],
                             sha(Path(sys.executable).resolve().read_bytes()))
            self.assertIs(type(census["runtime"]["optimize"]), int)
            self.assertEqual(census["runtime"]["optimize"], int(mode == "optimized"))
            self.assertEqual(census["runtime"]["no_site"], 1)
            self.assertIs(census["runtime"]["dont_write_bytecode"], True)
            self.assertEqual(census["count"], len(census["tests"]))
            self.assertEqual(census["loader_errors"], [])
            for row in census["tests"]:
                self.assertEqual(set(row), {"id", "module", "class", "source_path", "branch"})
        self.assertEqual([row["id"] for row in arch["tests"]], list(self.arch_ids))
        self.assertEqual([row["id"] for row in full["tests"]], [*self.arch_ids, CORE_ID])
        self.assertNotIn(CORE_ID, [row["id"] for row in arch["tests"]])
        modules = sorted({identity.split(".")[0] for identity in self.arch_ids})
        expected_arch = []
        for identity in self.arch_ids:
            module, owner, method = identity.split(".")
            expected_arch.append({"id": identity, "module": module,
                                  "class": module + "." + owner,
                                  "source_path": "tests/" + module + ".py",
                                  "branch": modules.index(module)})
        self.assertEqual(arch["tests"], expected_arch)
        self.assertEqual(full["tests"], expected_arch + [{
            "id": CORE_ID, "module": "test_core_fixture",
            "class": "test_core_fixture.Core",
            "source_path": "tests/test_core_fixture.py", "branch": len(modules)}])
        result = json.loads(members["core-result.json"])
        self.assertEqual(set(result), RESULT_KEYS)
        self.assertEqual(result["schema"], "main-unittest.core.v1")
        self.assertEqual(result["context"], self.tree.context)
        self.assertEqual(result["selected_ids"], [CORE_ID])
        self.assertEqual((result["mode"], result["profile"], result["purpose"]),
                         (mode, "core", "MAIN_DISCOVERY"))
        self.assertEqual(result["exit_code"], 0)
        self.assertIs(result["complete"], True)
        self.assertEqual(result["source_before_sha256"], sha(members["source-before.json"]))
        self.assertEqual(result["source_after_sha256"], sha(members["source-after.json"]))
        self.assertEqual(result["full_inventory_sha256"], sha(members["full-census.json"]))
        self.assertEqual(result["architecture_inventory_sha256"],
                         sha(members["architecture-census.json"]))
        for name, key in (("events.jsonl", "journal"), ("stdout.bin", "stdout"),
                          ("stderr.bin", "stderr")):
            self.assertEqual(result[key], {"bytes": len(members[name]), "sha256": sha(members[name])})
        rows = [json.loads(line) for line in members["events.jsonl"].splitlines()]
        self.assertEqual([row["seq"] for row in rows], list(range(len(rows))))
        common = {"schema", "seq", "pid", "monotonic_ns", "event"}
        extras = {
            "invocation_start": {"argv", "python", "optimize", "no_site", "isolated"},
            "run_start": set(), "test_start": {"test_id"},
            "outcome": {"test_id", "outcome"},
            "test_stop": {"test_id", "elapsed_ns"},
            "run_end": {"tests_run", "failures", "errors", "skipped",
                        "expected_failures", "unexpected_successes", "successful"},
            "invocation_exit": {"exit_code"},
        }
        for row in rows:
            self.assertEqual(row["schema"], "unittest-timing/v1")
            self.assertEqual(set(row), common | extras[row["event"]])
        self.assertEqual(len({row["pid"] for row in rows}), 1)
        self.assertEqual([row["monotonic_ns"] for row in rows],
                         sorted(row["monotonic_ns"] for row in rows))
        invocation = rows[0]
        self.assertEqual(invocation["event"], "invocation_start")
        self.assertEqual((invocation["python"], invocation["optimize"],
                          invocation["no_site"], invocation["isolated"]),
                         (sys.version, int(mode == "optimized"), 1, 0))
        self.assertTrue(invocation["argv"])
        self.assertEqual(rows[-1]["event"], "invocation_exit")
        self.assertEqual(rows[-1]["exit_code"], 0)
        self.assertEqual([row["test_id"] for row in rows if row["event"] == "test_start"], [CORE_ID])
        self.assertEqual([row["test_id"] for row in rows if row["event"] == "test_stop"], [CORE_ID])
        self.assertEqual([(row["test_id"], row["outcome"]) for row in rows
                          if row["event"] == "outcome"], [(CORE_ID, "success")])
        summary = next(row for row in rows if row["event"] == "run_end")
        self.assertEqual(summary["tests_run"], 1)
        self.assertIs(summary["successful"], True)
        self.assertEqual([summary[name] for name in
                          ("failures", "errors", "skipped", "expected_failures",
                           "unexpected_successes")], [0, 0, 0, 0, 0])
        self.assertIn(b"nested synthetic child: FAILED (skipped=1)", members["stdout.bin"])

    def test_native_pattern_censuses_select_actual_membership_not_id_prefix(self):
        for mode in ("normal", "optimized"):
            with self.subTest(mode=mode):
                self.assert_positive_core(mode)
        joined = self.join()
        self.assertEqual(set(joined), JOIN_KEYS)
        self.assertEqual(joined["schema"], "main-unittest.join.v1")
        self.assertEqual(joined["context"], self.tree.context)
        self.assertEqual(joined["scope"], "MAIN_REGISTERED_UNITTEST_DISCOVERY_ONLY")
        self.assertEqual(joined["status"], "QUALIFIED")
        self.assertEqual(joined["scientific_effect"], "NONE")
        self.assertIs(joined["g5_claim"], False)
        self.assertIs(joined["cbz_claim"], False)
        before = unpack(self.cores["normal"])["source-before.json"]
        self.assertEqual(joined["source_sha256"], sha(before))
        self.assertEqual(joined["native_sha256"], sha(encoded(self.native[0])))
        expected_core = []
        for mode, artifact_id in (("normal", "457"), ("optimized", "458")):
            members = unpack(self.cores[mode])
            expected_core.append({
                "mode": mode, "artifact_id": artifact_id, "artifact_sha256": sha(self.cores[mode]),
                "result_sha256": sha(members["core-result.json"]),
                "full_inventory_sha256": sha(members["full-census.json"]),
                "architecture_inventory_sha256": sha(members["architecture-census.json"])})
        self.assertEqual(joined["core"], expected_core)
        readback = read_graph_archive(
            self.arch_zip, self.binding,
            (self.tree.root / "docs/site/dependency-source/GRAPH.json").read_bytes())
        self.assertEqual(joined["architecture"],
                         {"binding": self.binding, "readback_sha256": sha(encoded(readback))})
        self.assertEqual(joined["coverage"], [
            {"mode": "normal", "full_count": 90, "core_count": 1, "architecture_count": 89},
            {"mode": "optimized", "full_count": 90, "core_count": 1, "architecture_count": 89},
        ])

    def test_fresh_pattern_dependent_load_tests_disagreement_refuses_before_body(self):
        tree = self.with_tree(architecture_override=r'''
            import json, os, unittest
            from pathlib import Path
            INITIALIZED = [os.getpid()]
            def note(event, pattern=None):
                with open(os.environ["TEST_IMPORT_TRACE"], "a") as stream:
                    stream.write(json.dumps({"event": event, "pid": os.getpid(),
                                             "pattern": pattern, "initial_pid": INITIALIZED[0]}) + "\n")
            note("import")
            def entered(): Path(os.environ["TEST_TRACE"]).write_text("entered architecture")
            class Arch(unittest.TestCase):
                def test_ok(self): entered()
                def test_arch_only(self): entered()
            def load_tests(loader, tests, pattern):
                note("load", pattern)
                method = "test_arch_only" if pattern == "test_architecture_*.py" else "test_ok"
                return unittest.TestSuite([Arch(method)])
        ''')
        process, out, trace = tree.run("normal", "pattern-disagreement")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(trace.read_text().splitlines() if trace.exists() else [], ([], ["load"]))
        imports = [json.loads(line) for line in out.with_suffix(".imports").read_text().splitlines()]
        self.assertEqual([(row["event"], row["pattern"]) for row in imports],
                         [("import", None), ("load", "test_architecture_*.py"),
                          ("import", None), ("load", "test*.py")])
        self.assertEqual(imports[0]["pid"], imports[1]["pid"])
        self.assertEqual(imports[2]["pid"], imports[3]["pid"])
        self.assertNotEqual(imports[0]["pid"], imports[2]["pid"])
        self.assertTrue(all(row["pid"] == row["initial_pid"] for row in imports))

    def test_safe_original_cases_preserve_fixture_lifecycle_and_unsafe_suites_refuse(self):
        for mode in ("normal", "optimized"):
            self.assertEqual(self.outputs[mode][1].read_text().splitlines(),
                             ["load", "module+", "class+", "body", "case-cleanup",
                              "class-", "class-cleanup", "module-"])
        for source in (
            '''
            import os, unittest
            from pathlib import Path
            def entered(): Path(os.environ["TEST_TRACE"]).write_text("entered mixed branch")
            class Arch(unittest.TestCase):
                def test_ok(self): entered()
            class Extra(unittest.TestCase):
                def test_core(self): entered()
            def load_tests(loader, tests, pattern):
                arch = unittest.TestSuite([Arch("test_ok")])
                if pattern == "test_architecture_*.py": return arch
                return unittest.TestSuite([arch, Extra("test_core")])
            ''',
            '''
            import os, unittest
            from pathlib import Path
            class Arch(unittest.TestCase):
                def test_ok(self):
                    Path(os.environ["TEST_TRACE"]).write_text("entered custom case")
            class Custom(unittest.TestSuite):
                def run(self, result, debug=False):
                    Path(os.environ["TEST_TRACE"]).write_text("entered custom run")
                    return super().run(result, debug=debug)
            def load_tests(loader, tests, pattern): return Custom(tests)
            ''',
        ):
            with self.subTest(source=source):
                tree = self.with_tree(architecture_override=source)
                process, out, trace = tree.run("normal", "unsafe-suite")
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(trace.read_text().splitlines() if trace.exists() else [], ([], ["load"]))

    def test_omitted_duplicate_extra_zero_and_import_failed_inventories_refuse(self):
        for change in (
            lambda value: value["tests"][-1].update(
                id="test_architecture_impostor.Core.test_different_same_count"),
            lambda value: (value["tests"].pop(),
                           value.update(count=len(value["tests"]))),
            lambda value: (value["tests"].append(copy.deepcopy(value["tests"][0])),
                           value.update(count=len(value["tests"]))),
            lambda value: (value["tests"].append({
                "id": "test_missing.Case.test_extra", "module": "test_missing",
                "class": "Case", "source_path": "tests/test_missing.py", "branch": 999}),
                           value.update(count=len(value["tests"]))),
            lambda value: value.update(tests=[], count=0),
        ):
            with self.subTest(change=change):
                cores = dict(self.cores)
                cores["normal"] = mutate_core(cores["normal"], "full-census.json", change)
                self.refused(cores=cores)
        tree = self.with_tree(architecture_override="raise ImportError('TEST original import failure')\n")
        process, out, trace = tree.run("normal", "import-failure")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(trace.read_text().splitlines() if trace.exists() else [], ([], ["load"]))

    def test_all_actual_parent_fixture_and_subtest_non_successes_cannot_qualify(self):
        bodies = {
            "failure": "self.fail('TEST parent failure')",
            "error": "raise RuntimeError('TEST parent error')",
            "skip": "self.skipTest('TEST parent skip')",
            "subtest-error": "with self.subTest(part=1): raise ValueError('TEST subtest error')",
            "subtest-skip": "with self.subTest(part=1): self.skipTest('TEST subtest skip')",
            "expected-failure": "self.fail('TEST expected failure')",
            "unexpected-success": "self.assertTrue(True)",
            "setup-error": "self.assertTrue(True)",
            "teardown-error": "self.assertTrue(True)",
            "class-setup-error": "self.assertTrue(True)",
            "class-teardown-error": "self.assertTrue(True)",
            "class-skip": "self.assertTrue(True)",
            "module-error": "self.assertTrue(True)",
            "module-skip": "self.assertTrue(True)",
        }
        for label, body in bodies.items():
            decorator = "@unittest.expectedFailure\n    " if label in (
                "expected-failure", "unexpected-success") else ""
            source = "import unittest\n"
            if label == "module-skip":
                source += "def setUpModule(): raise unittest.SkipTest('TEST module skip')\n"
            if label == "module-error":
                source += "def setUpModule(): raise RuntimeError('TEST module error')\n"
            source += "class Core(unittest.TestCase):\n"
            if label == "setup-error":
                source += "    def setUp(self): raise RuntimeError('TEST setup error')\n"
            if label == "teardown-error":
                source += "    def tearDown(self): raise RuntimeError('TEST teardown error')\n"
            if label in ("class-setup-error", "class-skip"):
                error = ("unittest.SkipTest('TEST class skip')" if label == "class-skip"
                         else "RuntimeError('TEST class setup error')")
                source += "    @classmethod\n    def setUpClass(cls): raise " + error + "\n"
            if label == "class-teardown-error":
                source += ("    @classmethod\n    def tearDownClass(cls): "
                           "raise RuntimeError('TEST class teardown error')\n")
            source += "    " + decorator + "def test_core(self):\n        " + body + "\n"
            with self.subTest(label=label):
                temporary = tempfile.TemporaryDirectory(prefix="TEST-native-outcome-")
                self.addCleanup(temporary.cleanup)
                tree = FixtureTree(Path(temporary.name), self.arch_ids, core=source)
                cores = {}
                expected_native_exit = 0 if label in (
                    "skip", "subtest-skip", "expected-failure", "class-skip",
                    "module-skip") else 1
                for mode in ("normal", "optimized"):
                    process, out, trace = tree.run(mode, label)
                    self.assertEqual({path.name for path in out.iterdir()}, set(MEMBERS))
                    cores[mode] = archive({name: (out / name).read_bytes() for name in MEMBERS})
                    result = json.loads((out / "core-result.json").read_bytes())
                    self.assertIs(result["complete"], True)
                    self.assertEqual(result["exit_code"], expected_native_exit,
                                     "qualification must preserve the actual native exit")
                    rows = [json.loads(line) for line in (out / "events.jsonl").read_bytes().splitlines()]
                    self.assertTrue(any(row["event"] == "outcome" and row["outcome"]
                                        not in ("success", "subtest_success") for row in rows))
                context = {key: tree.context[key] for key in
                           ("repository", "checked_commit", "run_id", "run_attempt")}
                context["artifact_id"] = "456"
                members = self.fixture_case.fixture(context, self.arch_ids)
                raw = self.fixture_module.classic_zip(self.fixture_case.entries(members))
                binding = self.fixture_case.expected(raw, members, context)
                producer, native = native_fixture(tree.context, cores, binding, raw)
                # Real native skips/expected failures can exit 0; full join still
                # refuses the actual completed outcomes, with all bindings valid.
                with self.assertRaises(ValueError):
                    qualify_join(tree.root, tree.context, producer, native, cores, raw)
        # Also reject a forged success summary with a retained bad original event.
        for outcome in ("failure", "error", "skip", "expected_failure", "unexpected_success",
                        "subtest_error", "subtest_failure"):
            cores = dict(self.cores)
            def change(rows):
                event = next(row for row in rows if row["event"] == "outcome")
                event["outcome"] = outcome
                if outcome.startswith("subtest_"):
                    event["parent_test_id"] = CORE_ID
            cores["normal"] = mutate_journal(cores["normal"], change)
            self.refused(cores=cores)

    def test_interrupted_incomplete_nonzero_and_damaged_journals_cannot_qualify(self):
        mutations = (
            lambda rows: rows.pop(),
            lambda rows: rows.__delitem__(next(i for i, row in enumerate(rows)
                                              if row["event"] == "test_stop")),
            lambda rows: rows.__delitem__(next(i for i, row in enumerate(rows)
                                              if row["event"] == "run_end")),
            lambda rows: rows.insert(2, copy.deepcopy(rows[2])),
            lambda rows: rows.reverse(),
            lambda rows: rows[-1].update(exit_code=1),
            lambda rows: rows[1].update(unknown=True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                cores = dict(self.cores)
                cores["normal"] = mutate_journal(cores["normal"], mutation)
                self.refused(cores=cores)
        for member in ("events.jsonl", "core-result.json"):
            cores = dict(self.cores)
            members = unpack(cores["normal"])
            members[member] = members[member][:-3]
            cores["normal"] = archive(members)
            self.refused(cores=cores)
        for change in (lambda value: value.update(exit_code=9),
                       lambda value: value.update(complete=False)):
            cores = dict(self.cores)
            cores["normal"] = mutate_core(cores["normal"], "core-result.json", change)
            self.refused(cores=cores)
        tree = self.with_tree(core='''
            import os, signal, unittest
            from pathlib import Path
            class Core(unittest.TestCase):
                def test_core(self):
                    Path(os.environ["TEST_TRACE"]).write_text("TEST about to signal")
                    os.kill(os.getpid(), signal.SIGTERM)
        ''')
        process, out, trace = tree.run("normal", "interrupted")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(trace.read_text(), "TEST about to signal")
        tree = self.with_tree(core='''
            import os, unittest
            from pathlib import Path
            class Core(unittest.TestCase):
                def test_core(self):
                    for candidate in Path("/proc/self/fd").iterdir():
                        try:
                            name = os.readlink(candidate)
                        except FileNotFoundError:
                            continue
                        if name.endswith("/events.jsonl"):
                            os.close(int(candidate.name))
                            Path(os.environ["TEST_TRACE"]).write_text("TEST closed actual journal")
                            return
                    self.fail("TEST actual journal descriptor was not open")
        ''')
        process, out, trace = tree.run("normal", "actual-journal-io-failure")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(trace.read_text(), "TEST closed actual journal")

    def test_source_runtime_closed_records_stale_context_and_missing_producers_refuse(self):
        for name, change in (
            ("source-after.json", lambda value: value["files"][0].update(sha256="0" * 64)),
            ("source-after.json", lambda value: value["files"][0].update(mode="100755")),
            ("source-after.json", lambda value: value["files"][0].update(git_blob="0" * 40)),
            ("full-census.json", lambda value: value.update(extra=True)),
            ("full-census.json", lambda value: value["runtime"].update(optimize=True)),
            ("full-census.json", lambda value: value["runtime"].update(executable_sha256="0" * 64)),
            ("core-result.json", lambda value: value.update(extra=True)),
            ("core-result.json", lambda value: value["context"].update(run_attempt="2")),
            ("core-result.json", lambda value: value["context"].update(run_id="314160")),
        ):
            with self.subTest(name=name, change=change):
                cores = dict(self.cores)
                cores["normal"] = mutate_core(cores["normal"], name, change)
                self.refused(cores=cores)
        cores = dict(self.cores)
        bodies = unpack(cores["normal"])
        bodies["core-result.json"] = bodies["core-result.json"][:-2] + b',"mode":"normal"}\n'
        cores["normal"] = archive(bodies)
        self.refused(cores=cores)
        self.refused(cores={"normal": self.cores["normal"]},
                     native=self.native, producer=self.producer)
        producer = copy.deepcopy(self.producer)
        producer["core_result"] = "failure"
        record, originals = copy.deepcopy(self.native)
        record["producer"] = producer
        self.refused(producer=producer, native=(record, originals))
        record, originals = copy.deepcopy(self.native)
        record["artifacts"].pop()
        self.refused(native=(record, originals))
        tree = self.with_tree(core='''
            import unittest
            from pathlib import Path
            class Core(unittest.TestCase):
                def test_core(self):
                    root = Path(__file__).resolve().parents[1]
                    (root / "docs/site/dependency-source/GRAPH.json").write_text("TEST changed source")
        ''')
        process, out, trace = tree.run("normal", "source-drift")
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual((tree.root / "docs/site/dependency-source/GRAPH.json").read_text(),
                         "TEST changed source")

    def test_real_unchanged_reader_accepts_only_labelled_test_archive_then_exact_join_membership(self):
        readback = read_graph_archive(
            self.arch_zip, self.binding,
            (self.tree.root / "docs/site/dependency-source/GRAPH.json").read_bytes())
        self.assertEqual(readback["custody"], "unknown")
        self.assertEqual(readback["receipt"]["verification_scope"],
                         "declared-native-pins-and-archive-content")
        self.assertEqual(readback["receipt"]["producer_receipt"]["test_modes"], [
            {"mode": "normal", "test_count": 89, "test_ids": list(self.arch_ids)},
            {"mode": "optimized", "test_count": 89, "test_ids": list(self.arch_ids)},
        ])
        self.assertIn(b"Synthetic TEST fixture", self.arch_members["console.log"])
        extras = (*self.arch_ids,
                  "test_architecture_synthetic_extra.Extra.test_unregistered")
        context = {key: self.tree.context[key] for key in
                   ("repository", "checked_commit", "run_id", "run_attempt")}
        context["artifact_id"] = "456"
        members = self.fixture_case.fixture(context, extras)
        raw = self.fixture_module.classic_zip(self.fixture_case.entries(members))
        binding = self.fixture_case.expected(raw, members, context)
        # The unchanged reader legitimately accepts all its scoped baseline IDs;
        # only the new full census join can reject this self-consistent extra.
        self.assertEqual(
            read_graph_archive(raw, binding,
                               (self.tree.root / "docs/site/dependency-source/GRAPH.json").read_bytes())
            ["receipt"]["producer_receipt"]["test_modes"][0]["test_count"], 90)
        self.refused(binding=binding, architecture_zip=raw)
        # Same cardinality, different actual identity: a count-only join must fail.
        temporary = tempfile.TemporaryDirectory(prefix="TEST-same-count-strict-")
        self.addCleanup(temporary.cleanup)
        discovered_extra = "test_architecture_synthetic_extra.Extra.test_discovered"
        declared_extra = "test_architecture_synthetic_extra.Extra.test_substituted"
        tree = FixtureTree(Path(temporary.name), (*self.arch_ids, discovered_extra))
        cores = {}
        for mode in ("normal", "optimized"):
            process, out, trace = tree.run(mode, "same-count-strict")
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            cores[mode] = archive({name: (out / name).read_bytes() for name in MEMBERS})
        context = {key: tree.context[key] for key in
                   ("repository", "checked_commit", "run_id", "run_attempt")}
        context["artifact_id"] = "456"
        matching_members = self.fixture_case.fixture(
            context, (*self.arch_ids, discovered_extra))
        matching_raw = self.fixture_module.classic_zip(
            self.fixture_case.entries(matching_members))
        matching_binding = self.fixture_case.expected(
            matching_raw, matching_members, context)
        matching_producer, matching_native = native_fixture(
            tree.context, cores, matching_binding, matching_raw)
        matching_join = qualify_join(
            tree.root, tree.context, matching_producer, matching_native, cores, matching_raw)
        self.assertEqual(matching_join["status"], "QUALIFIED")
        self.assertEqual(matching_join["context"], tree.context)
        self.assertEqual(matching_join["scope"], "MAIN_REGISTERED_UNITTEST_DISCOVERY_ONLY")
        self.assertIs(matching_join["g5_claim"], False)
        self.assertIs(matching_join["cbz_claim"], False)
        self.assertEqual(matching_join["coverage"], [
            {"mode": "normal", "full_count": 91, "core_count": 1, "architecture_count": 90},
            {"mode": "optimized", "full_count": 91, "core_count": 1, "architecture_count": 90},
        ])
        members = self.fixture_case.fixture(context, (*self.arch_ids, declared_extra))
        raw = self.fixture_module.classic_zip(self.fixture_case.entries(members))
        binding = self.fixture_case.expected(raw, members, context)
        self.assertEqual(
            read_graph_archive(raw, binding,
                               (tree.root / "docs/site/dependency-source/GRAPH.json").read_bytes())
            ["receipt"]["producer_receipt"]["test_modes"][0]["test_count"], 90)
        producer, native = native_fixture(tree.context, cores, binding, raw)
        with self.assertRaises(ValueError) as refusal:
            qualify_join(tree.root, tree.context, producer, native, cores, raw)
        self.assertEqual(str(refusal.exception), "INVENTORY_MISMATCH")

    def test_original_zip_member_bytes_and_inert_api_bindings_are_both_checked(self):
        self.refused(architecture_zip=self.arch_zip + b"TEST trailing bytes")
        cores = dict(self.cores)
        cores["normal"] += b"TEST changed raw archive"
        self.refused(cores=cores, native=self.native)
        cores = dict(self.cores)
        members = unpack(cores["normal"])
        members["stdout.bin"] += b"TEST changed original without changing R\n"
        cores["normal"] = archive(members)
        self.refused(cores=cores)
        cores = dict(self.cores)
        members = unpack(cores["normal"])
        members["TEST-unexpected-member.txt"] = b"TEST closed archive extra\n"
        cores["normal"] = archive(members)
        self.refused(cores=cores)
        for change in (
            lambda value: value["context"].update(checked_commit="f" * 40),
            lambda value: value["artifacts"][0].update(id="999"),
            lambda value: value["artifacts"][0].update(name="research-architecture-314159-2"),
            lambda value: value["artifacts"][0].update(expired=True),
            lambda value: value["artifacts"][0].update(digest="0" * 64),
            lambda value: value["artifacts"][0].update(extra=True),
            lambda value: value["producer"].update(extra=True),
            lambda value: value.update(untrusted_transport_claim=True),
        ):
            record, originals = copy.deepcopy(self.native)
            change(record)
            self.refused(native=(record, originals))
        record, originals = copy.deepcopy(self.native)
        path = "TEST-api-run-attempt.json"
        raw = json.loads(originals[path])
        raw["run_attempt"] = 2
        originals[path] = encoded(raw)
        row = next(item for item in record["api_originals"] if item["path"] == path)
        row.update(bytes=len(originals[path]), sha256=sha(originals[path]))
        self.refused(native=(record, originals))
        self.refused(architecture_zip=str(self.outputs["normal"][0]),
                     native=self.native, producer=self.producer)

    def test_pilot_success_can_never_produce_main_discovery_qualification(self):
        cores = {}
        for mode, raw in self.cores.items():
            cores[mode] = mutate_core(
                raw, "core-result.json",
                lambda value: value.update(purpose="ARCHITECTURE_PYTEST_PILOT"))
        self.refused(cores=cores)
        cores = {}
        for mode, raw in self.cores.items():
            cores[mode] = mutate_core(
                raw, "core-result.json",
                lambda value: value.update(
                    selected_ids=["tests.test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_exclusion_is_exact"]))
        self.refused(cores=cores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
