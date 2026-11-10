"""Synthetic TEST acquisition contracts; no HTTP, token, custody or G5 credit.

The real unchanged helper and reader admit the TEST producer archives. No
supplied record authenticates transport or proves completed native execution.
"""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Unconditional missing-module RED precedes fixture setup; no skip/fallback.
from tools import unittest_native_acquisition as subject
from tools.unittest_native_acquisition import verify_capture, write_original_frames
from tools.unittest_full_qualification import qualify_join
from architecture.graph_artifact_readback import read_graph_archive

if Path(subject.__file__).resolve() != ROOT / "tools/unittest_native_acquisition.py":
    raise AssertionError("subject escaped checked source")

FIXTURE_SOURCE = "2fe09abce9a8ef9356fd0af85780dcde75179386"
FIXED_REF = "refs/heads/codex/g5-native-qualification-20261110"
WORKFLOW = ".github/workflows/full-unittest-qualification.yml"
NEEDS_KEYS = {"architecture", "core_normal", "core_optimized"}
NATIVE_KEYS = {"event_name", "ref", "repository", "sha", "run_id", "run_attempt",
               "workflow_ref", "workflow_sha"}
CAPTURE_KEYS = {"schema", "context", "native", "needs_sha256", "responses", "downloads"}
RESPONSE_KEYS = {"path", "url", "bytes", "sha256", "exit_code", "timed_out", "complete"}
DOWNLOAD_KEYS = RESPONSE_KEYS | {"profile", "mode", "id", "name"}
CHECK_KEYS = {
    "schema", "context", "status", "transport_authority", "completion_authority",
    "supplied_run_status", "native_completion_required", "scientific_effect",
    "g5_claim", "cbz_claim", "capture_sha256", "api_originals", "archives", "join",
}
FRAME_KEYS = {"schema", "context", "capture_sha256", "transport_authority", "complete", "members"}
FRAME_ROW_KEYS = {"path", "original_bytes", "sha256", "encoding", "printed_original_bytes"}
PRODUCERS = {
    "architecture": ("architecture / research-architecture", 101),
    "core_normal": ("Core full discovery (normal)", 102),
    "core_optimized": ("Core full discovery (optimized)", 103),
}
OTHERS = (("Stage 1 targeted contracts (normal)", 104),
          ("Stage 1 targeted contracts (optimized)", 105),
          ("Acquire original Main unittest evidence", 106))
ARCHIVES = {"architecture.zip": ("architecture", "both", "456"),
            "core-normal.zip": ("core", "normal", "457"),
            "core-optimized.zip": ("core", "optimized", "458")}

# Exact inert APIs; actual CLI transport remains separate future native proof:
# verify_capture(root: Path, context: dict, needs: dict, capture: bytes,
#                originals: dict[str,bytes], archives: dict[str,bytes]) -> dict
# write_original_frames(root: Path, context: dict, needs: dict, capture: bytes,
#                       originals: dict[str,bytes], archives: dict[str,bytes],
#                       output: BinaryIO) -> dict
# capture exactly CAPTURE_KEYS; native exactly NATIVE_KEYS; needs exactly3 rows
# {result,outputs}, core outputs exactly{artifact_id,artifact_name,artifact_sha256},
# architecture outputs existing8. Response/download rows have exact keys above,
# sorted paths, integer0 exit (not bool), false timed_out, true complete.
# API bodies remain original bytes and accept unrelated GitHub fields; wrapper
# keys are closed. Exactly7 original response paths and3 raw ZIP paths below.
# verify_capture calls REAL qualify_join, returns exactly CHECK_KEYS with status
# INERT_CAPTURE_VERIFIED; both authority fields NONE_FROM_SUPPLIED_BYTES,
# native_completion_required=True, scientific_effect=NONE, G5/CBZ false.
# Fixed ValueError reasons: INVALID_CAPTURE, STALE_CONTEXT,
# PRODUCER_NOT_COMPLETE, ARCHIVE_MISMATCH, SOURCE_MISMATCH, INVENTORY_MISMATCH,
# NON_SUCCESS, INCOMPLETE, PILOT_SCOPE, INVALID_RECORD, UNSAFE_PARTITION and
# STALE_NATIVE. Existing helper refusals are preserved.
# Completed producer rows are required while the current run may be in_progress.
# Supplied completed-run JSON still establishes no native completion authority.
#
# Framing admits first through REAL helper/reader, then prints capture.json,
# source/GRAPH.json, all7 api originals and31 admitted ZIP members in sorted order.
# Existing helper9/reader13 limits apply; no new execution/runtime budget.
# ::stop-commands::<random collision-free48lowerhex>\n brackets all frames.
# NATIVE_ORIGINAL_BEGIN JSON{path,original_bytes,sha256,encoding}\n
# EXACT UTF8 bytes or canonical base64 for nonUTF8, then ONE delimiter LF.
# NATIVE_ORIGINAL_END JSON{path,printed_original_bytes,sha256}\n
# ::<nonce>::\n. Never print raw ZIP, replacement decoding or rebuilt outcomes.


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unpack(raw):
    """Producer-side mutation/readback only; supplies no acceptance verdict."""
    with zipfile.ZipFile(io.BytesIO(raw)) as stream:
        return {name: stream.read(name) for name in stream.namelist()}


class NativeAcquisitionBoundaryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "tests/test_unittest_full_qualification.py"
        raw = path.read_bytes()
        identity = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if identity != FIXTURE_SOURCE:
            raise AssertionError("frozen TEST producer source changed")
        spec = importlib.util.spec_from_file_location("TEST_acquisition_producers", path)
        cls.fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fixture)
        cls.data = cls.fixture.MainUnittestQualificationContract
        cls.addClassCleanup(cls.data.doClassCleanups)
        cls.data.setUpClass()
        cls.root, cls.context = cls.data.tree.root, copy.deepcopy(cls.data.tree.context)
        cls.archives = {"architecture.zip": cls.data.arch_zip,
                        "core-normal.zip": cls.data.cores["normal"],
                        "core-optimized.zip": cls.data.cores["optimized"]}
        cls.capture, cls.needs, cls.originals = cls.make_capture(cls.archives)

    @classmethod
    def make_capture(cls, archives):
        """Synthetic producer ONLY; no HTTP or acceptance checker."""
        cores = {mode: archives["core-" + mode + ".zip"] for mode in ("normal", "optimized")}
        mini = {key: cls.context[key] for key in
                ("repository", "checked_commit", "run_id", "run_attempt")}
        mini["artifact_id"] = "456"
        binding = cls.data.fixture_case.expected(
            archives["architecture.zip"], unpack(archives["architecture.zip"]), mini)
        producer, (record, old) = cls.fixture.native_fixture(
            cls.context, cores, binding, archives["architecture.zip"])
        base = "https://api.github.com/repos/" + cls.context["repository"]
        originals = {path.removeprefix("TEST-"): raw for path, raw in old.items()}
        urls = {row["path"].removeprefix("TEST-"): row["url"] for row in record["api_originals"]}
        run = json.loads(originals["api-run-attempt.json"])
        run.update(event="push", head_branch=FIXED_REF.removeprefix("refs/heads/"),
                   status="in_progress", conclusion=None, unrelated_future_field={"retained": True})
        run["repository"]["id"] = run["head_repository"]["id"] = 1
        originals["api-run-attempt.json"] = encoded(run)
        for identity in ("456", "457", "458"):
            path = "api-artifact-" + identity + ".json"
            artifact = json.loads(originals[path])
            artifact["workflow_run"]["head_branch"] = FIXED_REF.removeprefix("refs/heads/")
            originals[path] = encoded(artifact)
        jobs = []
        for name, identity in list(PRODUCERS.values()) + list(OTHERS):
            producer_job = identity <= 103
            jobs.append(dict(id=identity, run_id=int(cls.context["run_id"]),
                             run_attempt=int(cls.context["run_attempt"]),
                             head_sha=cls.context["checked_commit"], name=name,
                             status="completed" if producer_job else "in_progress",
                             conclusion="success" if producer_job else None))
        originals["api-attempt-jobs.json"] = encoded(dict(total_count=6, jobs=jobs))
        urls["api-attempt-jobs.json"] = (base + "/actions/runs/" + cls.context["run_id"]
            + "/attempts/" + cls.context["run_attempt"] + "/jobs?per_page=100")
        originals["api-artifacts.json"] = encoded(dict(total_count=3, artifacts=[
            json.loads(originals["api-artifact-" + identity + ".json"])
            for identity in ("456", "457", "458")]))
        urls["api-artifacts.json"] = base + "/actions/runs/" + cls.context["run_id"] + "/artifacts?per_page=100"
        originals["api-git-commit.json"] = encoded(dict(
            sha=cls.context["checked_commit"], tree=dict(sha=cls.context["checked_tree"]),
            parents=[], verification=dict(verified=False, reason="unsigned")))
        urls["api-git-commit.json"] = base + "/git/commits/" + cls.context["checked_commit"]
        needs = {"architecture": dict(result="success", outputs=binding)}
        for mode, identity in (("normal", "457"), ("optimized", "458")):
            needs["core_" + mode] = dict(result="success", outputs=dict(
                artifact_id=identity, artifact_sha256=sha(archives["core-" + mode + ".zip"]),
                artifact_name="main-unittest-core-" + mode + "-" + cls.context["run_id"]
                              + "-" + cls.context["run_attempt"]))
        capture = dict(
            schema="main-unittest.capture.v1", context=copy.deepcopy(cls.context),
            native=dict(event_name="push", ref=FIXED_REF, repository=cls.context["repository"],
                        sha=cls.context["checked_commit"], run_id=cls.context["run_id"],
                        run_attempt=cls.context["run_attempt"],
                        workflow_ref=cls.context["repository"] + "/" + WORKFLOW + "@" + FIXED_REF,
                        workflow_sha=cls.context["checked_commit"]),
            needs_sha256=sha(encoded(needs)),
            responses=[dict(path=path, url=urls[path], bytes=len(raw), sha256=sha(raw),
                            exit_code=0, timed_out=False, complete=True)
                       for path, raw in sorted(originals.items())],
            downloads=[dict(path=path, url=base + "/actions/artifacts/" + identity + "/zip",
                            bytes=len(archives[path]), sha256=sha(archives[path]), exit_code=0,
                            timed_out=False, complete=True, profile=profile, mode=mode, id=identity,
                            name=binding["artifact_name"] if profile == "architecture"
                            else needs["core_" + mode]["outputs"]["artifact_name"])
                       for path, (profile, mode, identity) in sorted(ARCHIVES.items())])
        return capture, needs, originals

    def fresh(self):
        return copy.deepcopy(self.capture), copy.deepcopy(self.needs), dict(self.originals), dict(self.archives)

    def call(self, *, capture=None, needs=None, originals=None, archives=None, context=None, root=None):
        return verify_capture(
            self.root if root is None else root, self.context if context is None else context,
            self.needs if needs is None else needs, encoded(self.capture) if capture is None else capture,
            self.originals if originals is None else originals,
            self.archives if archives is None else archives)

    def refused(self, reason, **arguments):
        with self.assertRaises(ValueError) as error:
            self.call(**arguments)
        self.assertEqual(str(error.exception), reason)

    def replace_response(self, capture, originals, path, value):
        originals[path] = encoded(value)
        next(row for row in capture["responses"] if row["path"] == path).update(
            bytes=len(originals[path]), sha256=sha(originals[path]))

    def direct_join(self, capture, needs, originals, archives):
        producer = dict(core_result="success", architecture_result="success",
                        architecture_outputs=needs["architecture"]["outputs"])
        four = [row for row in capture["responses"]
                if row["path"] == "api-run-attempt.json" or row["path"].startswith("api-artifact-")]
        native = dict(schema="main-unittest.native.v1", context=self.context, producer=producer,
            api_originals=[{key: row[key] for key in ("path", "url", "bytes", "sha256")} for row in four],
            artifacts=[dict(profile=row["profile"], mode=row["mode"], id=row["id"], name=row["name"],
                            run_id=self.context["run_id"], head_sha=self.context["checked_commit"],
                            expired=False, digest=row["sha256"], archive_bytes=row["bytes"])
                       for row in capture["downloads"]])
        return qualify_join(self.root, self.context, producer,
                            (native, {row["path"]: originals[row["path"]] for row in four}),
                            {mode: archives["core-" + mode + ".zip"] for mode in ("normal", "optimized")},
                            archives["architecture.zip"])

    def test_actual_helper_reader_join_accepts_only_inert_same_source_push(self):
        readback = read_graph_archive(self.archives["architecture.zip"],
            self.needs["architecture"]["outputs"],
            (self.root / "docs/site/dependency-source/GRAPH.json").read_bytes())
        self.assertEqual(readback["custody"], "unknown")
        self.assertEqual(readback["receipt"]["verification_scope"],
                         "declared-native-pins-and-archive-content")
        self.assertIn(b"Synthetic TEST fixture", unpack(self.archives["architecture.zip"])["console.log"])
        result = self.call()
        self.assertEqual(set(result), CHECK_KEYS)
        self.assertEqual(result["schema"], "main-unittest.acquisition-check.v1")
        self.assertEqual(result["context"], self.context)
        self.assertEqual(result["status"], "INERT_CAPTURE_VERIFIED")
        self.assertEqual(result["capture_sha256"], sha(encoded(self.capture)))
        self.assertEqual(result["join"],
                         self.direct_join(self.capture, self.needs, self.originals, self.archives))
        self.assertEqual(result["join"]["scope"], "MAIN_REGISTERED_UNITTEST_DISCOVERY_ONLY")
        self.assertEqual(result["api_originals"], [
            {key: row[key] for key in ("path", "url", "bytes", "sha256")}
            for row in self.capture["responses"]])
        self.assertEqual(result["archives"], [
            {key: row[key] for key in ("path", "profile", "mode", "id", "name", "bytes", "sha256")}
            for row in self.capture["downloads"]])

    def test_completed_producers_are_required_while_current_run_may_be_in_progress(self):
        self.assertEqual(self.call()["supplied_run_status"], "in_progress")
        for status, conclusion in (("in_progress", None), ("queued", None),
                ("completed", "failure"), ("completed", "cancelled"),
                ("completed", "timed_out"), ("completed", "skipped"),
                ("completed", "neutral"), ("completed", None)):
            for name, _ in PRODUCERS.values():
                with self.subTest(name=name, status=status, conclusion=conclusion):
                    cap, needs, originals, archives = self.fresh()
                    jobs = json.loads(originals["api-attempt-jobs.json"])
                    next(row for row in jobs["jobs"] if row["name"] == name).update(
                        status=status, conclusion=conclusion)
                    self.replace_response(cap, originals, "api-attempt-jobs.json", jobs)
                    self.refused("PRODUCER_NOT_COMPLETE", capture=encoded(cap), originals=originals)
        cap, needs, originals, archives = self.fresh()
        run = json.loads(originals["api-run-attempt.json"])
        run.update(status="completed", conclusion="success")
        self.replace_response(cap, originals, "api-run-attempt.json", run)
        jobs = json.loads(originals["api-attempt-jobs.json"])
        for row in jobs["jobs"]:
            row.update(status="completed", conclusion="success")
        self.replace_response(cap, originals, "api-attempt-jobs.json", jobs)
        result = self.call(capture=encoded(cap), originals=originals)
        self.assertEqual(result["supplied_run_status"], "completed")
        self.assertIs(result["native_completion_required"], True)
        run["conclusion"] = "failure"
        self.replace_response(cap, originals, "api-run-attempt.json", run)
        self.refused("NON_SUCCESS", capture=encoded(cap), originals=originals)
        run.update(status="queued", conclusion=None)
        self.replace_response(cap, originals, "api-run-attempt.json", run)
        self.refused("INCOMPLETE", capture=encoded(cap), originals=originals)

    def test_fixed_native_push_ref_workflow_and_source_context_refuse_substitution(self):
        for key, value in (
                ("event_name", "pull_request"), ("event_name", "workflow_dispatch"),
                ("event_name", "workflow_run"), ("ref", "refs/heads/main"),
                ("repository", "TEST/other"), ("sha", "f" * 40),
                ("run_id", "314160"), ("run_attempt", "2"),
                ("workflow_ref", "TEST/other/" + WORKFLOW + "@" + FIXED_REF),
                ("workflow_ref", self.context["repository"] + "/.github/workflows/other.yml@" + FIXED_REF),
                ("workflow_sha", "f" * 40), ("run_id", True)):
            with self.subTest(key=key, value=value):
                cap = copy.deepcopy(self.capture)
                cap["native"][key] = value
                self.refused("STALE_CONTEXT", capture=encoded(cap))
        for key, value in (("checked_commit", "f" * 40), ("checked_tree", "f" * 40),
                           ("run_attempt", "2"), ("repository", "TEST/other")):
            cap = copy.deepcopy(self.capture)
            cap["context"][key] = value
            self.refused("STALE_CONTEXT", capture=encoded(cap))
        self.refused("SOURCE_MISMATCH", root=self.root.parent)
        self.refused("STALE_CONTEXT", context={**self.context, "checked_tree": "f" * 40})

    def test_native_needs_are_closed_and_two_core_artifact_identities_cannot_overwrite(self):
        for key in NEEDS_KEYS:
            for result in ("failure", "cancelled", "skipped", None, "unknown"):
                with self.subTest(key=key, result=result):
                    cap, needs, originals, archives = self.fresh()
                    needs[key]["result"] = result
                    cap["needs_sha256"] = sha(encoded(needs))
                    self.refused("PRODUCER_NOT_COMPLETE", capture=encoded(cap), needs=needs)
        for mutate in (lambda value: value.pop("core_optimized"),
                lambda value: value.update(extra=dict(result="success", outputs={})),
                lambda value: value["core_normal"].update(extra=True),
                lambda value: value["core_normal"]["outputs"].update(extra=True)):
            cap, needs, originals, archives = self.fresh()
            mutate(needs)
            cap["needs_sha256"] = sha(encoded(needs))
            self.refused("INVALID_CAPTURE", capture=encoded(cap), needs=needs)
        cap, needs, originals, archives = self.fresh()
        needs["core_optimized"]["outputs"] = copy.deepcopy(needs["core_normal"]["outputs"])
        cap["needs_sha256"] = sha(encoded(needs))
        self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), needs=needs)
        cap, needs, originals, archives = self.fresh()
        needs["core_normal"]["outputs"]["artifact_id"] = "999"
        cap["needs_sha256"] = sha(encoded(needs))
        self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), needs=needs)
        needs = copy.deepcopy(self.needs)
        needs["core_normal"]["outputs"]["artifact_sha256"] = "0" * 64
        self.refused("INVALID_CAPTURE", needs=needs)
        for value in (True, 457, "0457", "0", "-1"):
            cap, needs, originals, archives = self.fresh()
            needs["core_normal"]["outputs"]["artifact_id"] = value
            cap["needs_sha256"] = sha(encoded(needs))
            self.refused("INVALID_CAPTURE", capture=encoded(cap), needs=needs)

    def test_closed_capture_and_exact_original_response_bytes_are_bound(self):
        for mutate in (
                lambda value: value.update(authenticated=True),
                lambda value: value.update(extra=True),
                lambda value: value["native"].update(extra=True),
                lambda value: value["responses"][0].update(extra=True),
                lambda value: value["downloads"][0].update(extra=True),
                lambda value: value.update(schema="main-unittest.capture.v2"),
                lambda value: value["responses"].reverse(),
                lambda value: value["downloads"].reverse(),
                lambda value: value["responses"].append(copy.deepcopy(value["responses"][0])),
                lambda value: value["downloads"].append(copy.deepcopy(value["downloads"][0]))):
            cap = copy.deepcopy(self.capture)
            mutate(cap)
            self.refused("INVALID_CAPTURE", capture=encoded(cap))
        raw = encoded(self.capture)
        self.refused("INVALID_CAPTURE", capture=raw[:-2] + b',"schema":"main-unittest.capture.v1"}\n')
        self.refused("INVALID_CAPTURE", capture=raw + b"{}")
        self.refused("INVALID_CAPTURE", capture=raw.replace(b'"needs_sha256":', b'"needs_sha256":NaN,"x":', 1))
        self.refused("INVALID_CAPTURE", capture=raw.decode("ascii"))
        for path in self.originals:
            originals = dict(self.originals)
            originals[path] += b" "
            self.refused("INVALID_CAPTURE", originals=originals)
        originals = dict(self.originals)
        originals.pop("api-git-commit.json")
        self.refused("INVALID_CAPTURE", originals=originals)
        self.refused("INVALID_CAPTURE", originals={**self.originals, "arbitrary.json": b"{}\n"})
        for field, value, reason in (("exit_code", 1, "INCOMPLETE"),
                ("timed_out", True, "INCOMPLETE"), ("complete", False, "INCOMPLETE"),
                ("bytes", True, "INVALID_CAPTURE"), ("exit_code", True, "INVALID_CAPTURE")):
            cap = copy.deepcopy(self.capture)
            cap["responses"][0][field] = value
            self.refused(reason, capture=encoded(cap))
        cap = copy.deepcopy(self.capture)
        cap["responses"][0]["url"] = cap["responses"][0]["url"].replace("api.github.com", "example.invalid")
        self.refused("STALE_CONTEXT", capture=encoded(cap))

    def test_original_api_repository_workflow_head_and_git_tree_are_checked(self):
        cases = (
            ("api-run-attempt.json", lambda row: row.update(head_sha="f" * 40)),
            ("api-run-attempt.json", lambda row: row.update(path=".github/workflows/other.yml")),
            ("api-run-attempt.json", lambda row: row.update(run_attempt=2)),
            ("api-run-attempt.json", lambda row: row.update(id=True)),
            ("api-run-attempt.json", lambda row: row.update(event="pull_request")),
            ("api-run-attempt.json", lambda row: row.update(head_branch="main")),
            ("api-run-attempt.json", lambda row: row["repository"].update(full_name="TEST/other")),
            ("api-run-attempt.json", lambda row: row["head_repository"].update(id=2)),
            ("api-git-commit.json", lambda row: row.update(sha="f" * 40)),
            ("api-git-commit.json", lambda row: row["tree"].update(sha="f" * 40)),
            ("api-artifact-457.json", lambda row: row["workflow_run"].update(head_sha="f" * 40)),
            ("api-artifact-457.json", lambda row: row["workflow_run"].update(repository_id=2)),
            ("api-artifact-457.json", lambda row: row["workflow_run"].update(head_repository_id=2)),
            ("api-artifact-457.json", lambda row: row["workflow_run"].update(head_branch="main")),
        )
        for path, mutate in cases:
            cap, needs, originals, archives = self.fresh()
            value = json.loads(originals[path])
            mutate(value)
            self.replace_response(cap, originals, path, value)
            self.refused("STALE_CONTEXT", capture=encoded(cap), originals=originals)
        cap, needs, originals, archives = self.fresh()
        row = json.loads(originals["api-run-attempt.json"])
        row.update(head_sha="f" * 40, checked_tree=self.context["checked_tree"])
        self.replace_response(cap, originals, "api-run-attempt.json", row)
        self.refused("STALE_CONTEXT", capture=encoded(cap), originals=originals)
        cap, needs, originals, archives = self.fresh()
        row = json.loads(originals["api-git-commit.json"])
        row["future_github_field"] = {"not_a_wrapper_field": [1, 2, 3]}
        self.replace_response(cap, originals, "api-git-commit.json", row)
        self.assertEqual(self.call(capture=encoded(cap), originals=originals)["status"],
                         "INERT_CAPTURE_VERIFIED")
        artifact = json.loads(originals["api-artifact-457.json"])
        artifact["unrelated_metadata_field"] = {"retained": True}
        self.replace_response(cap, originals, "api-artifact-457.json", artifact)
        self.assertEqual(self.call(capture=encoded(cap), originals=originals)["status"],
                         "INERT_CAPTURE_VERIFIED")
        row["future_github_field"] = "TEST bounded response " + "x" * (2 * 1024 * 1024)
        self.replace_response(cap, originals, "api-git-commit.json", row)
        self.refused("INVALID_CAPTURE", capture=encoded(cap), originals=originals)

    def test_fresh_artifact_names_ids_attempts_and_complete_listing_are_required(self):
        for path, mutate in (
                ("api-artifacts.json", lambda row: row.update(total_count=4)),
                ("api-artifacts.json", lambda row: row["artifacts"].pop()),
                ("api-artifacts.json", lambda row: row["artifacts"].append(copy.deepcopy(row["artifacts"][0]))),
                ("api-attempt-jobs.json", lambda row: row.update(total_count=101)),
                ("api-attempt-jobs.json", lambda row: row["jobs"].pop(0)),
                ("api-attempt-jobs.json", lambda row: row["jobs"].append(copy.deepcopy(row["jobs"][0]))),
                ("api-attempt-jobs.json", lambda row: row["jobs"][0].update(head_sha="f" * 40)),
                ("api-attempt-jobs.json", lambda row: row["jobs"][0].update(run_attempt=2)),
                ("api-attempt-jobs.json", lambda row: row["jobs"][0].update(name="unrecognized producer"))):
            cap, needs, originals, archives = self.fresh()
            value = json.loads(originals[path])
            mutate(value)
            self.replace_response(cap, originals, path, value)
            self.refused("INCOMPLETE", capture=encoded(cap), originals=originals)
        for key, value in (
                ("name", "main-unittest-core-normal-" + self.context["run_id"] + "-2"),
                ("id", 999), ("expired", True), ("digest", "sha256:" + "0" * 64),
                ("size_in_bytes", 0)):
            cap, needs, originals, archives = self.fresh()
            row = json.loads(originals["api-artifact-457.json"])
            row[key] = value
            self.replace_response(cap, originals, "api-artifact-457.json", row)
            self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), originals=originals)
        cap, needs, originals, archives = self.fresh()
        listing = json.loads(originals["api-artifacts.json"])
        unrelated = copy.deepcopy(listing["artifacts"][0])
        unrelated.update(id=999, name="unrelated-retained-diagnostic")
        listing["artifacts"].append(unrelated)
        listing["total_count"] = 4
        self.replace_response(cap, originals, "api-artifacts.json", listing)
        self.assertEqual(self.call(capture=encoded(cap), originals=originals)["status"],
                         "INERT_CAPTURE_VERIFIED")
        cap, needs, originals, archives = self.fresh()
        listing = json.loads(originals["api-artifacts.json"])
        duplicate_name = copy.deepcopy(listing["artifacts"][1])
        duplicate_name["id"] = 999
        listing["artifacts"].append(duplicate_name)
        listing["total_count"] = 4
        self.replace_response(cap, originals, "api-artifacts.json", listing)
        self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), originals=originals)

    def test_raw_zip_digest_size_and_real_existing_archive_admission_are_required(self):
        for name in ARCHIVES:
            archives = dict(self.archives)
            archives[name] += b"TEST changed raw ZIP"
            self.refused("ARCHIVE_MISMATCH", archives=archives)
        for archives in (
                {key: raw for key, raw in self.archives.items() if key != "core-optimized.zip"},
                {**self.archives, "extra.zip": b"TEST"},
                {**self.archives, "core-normal.zip": str(self.data.outputs["normal"][0])}):
            self.refused("INVALID_CAPTURE", archives=archives)
        for change in (lambda row: row.update(exit_code=1),
                       lambda row: row.update(timed_out=True),
                       lambda row: row.update(complete=False)):
            cap = copy.deepcopy(self.capture)
            change(cap["downloads"][0])
            self.refused("INCOMPLETE", capture=encoded(cap))
        archives = dict(self.archives)
        members = unpack(archives["core-normal.zip"])
        members["stdout.bin"] += b"TEST member changed without R update\n"
        archives["core-normal.zip"] = self.fixture.archive(members)
        cap, needs, originals = self.make_capture(archives)
        self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), needs=needs,
                     originals=originals, archives=archives)
        # Same strict fixture writer as the admitted positive; only extra member.
        archives = dict(self.archives)
        members = unpack(archives["architecture.zip"])
        members["TEST-extra.txt"] = b"TEST closed strict archive extra\n"
        members.pop("ARTIFACT_SHA256SUMS")  # pinned entries() appends the sole new manifest
        archives["architecture.zip"] = self.data.fixture_module.classic_zip(
            self.data.fixture_case.entries(members))
        with zipfile.ZipFile(io.BytesIO(archives["architecture.zip"])) as stream:
            names = stream.namelist()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(unpack(self.archives["architecture.zip"])) | {"TEST-extra.txt"})
        cap, needs, originals = self.make_capture(archives)
        self.refused("ARCHIVE_MISMATCH", capture=encoded(cap), needs=needs,
                     originals=originals, archives=archives)

    def test_runtime_source_inventory_and_pilot_refusals_survive_wrapper(self):
        for member, change, reason in (
                ("full-census.json", lambda row: row["runtime"].update(executable_sha256="0" * 64), "SOURCE_MISMATCH"),
                ("source-after.json", lambda row: row["files"][0].update(sha256="0" * 64), "SOURCE_MISMATCH"),
                ("full-census.json", lambda row: row["tests"].pop(), "INVENTORY_MISMATCH"),
                ("core-result.json", lambda row: row.update(complete=False), "INCOMPLETE"),
                ("core-result.json", lambda row: row.update(exit_code=9), "NON_SUCCESS"),
                ("core-result.json", lambda row: row.update(purpose="ARCHITECTURE_PYTEST_PILOT"), "PILOT_SCOPE")):
            with self.subTest(member=member, reason=reason):
                archives = dict(self.archives)
                archives["core-normal.zip"] = self.fixture.mutate_core(archives["core-normal.zip"], member, change)
                cap, needs, originals = self.make_capture(archives)
                self.refused(reason, capture=encoded(cap), needs=needs, originals=originals, archives=archives)
        archives = dict(self.archives)
        archives["core-normal.zip"] = self.fixture.mutate_journal(
            archives["core-normal.zip"],
            lambda rows: next(row for row in rows if row["event"] == "outcome").update(outcome="skip"))
        cap, needs, originals = self.make_capture(archives)
        self.refused("NON_SUCCESS", capture=encoded(cap), needs=needs, originals=originals, archives=archives)

    def frame_inputs(self, capture, originals, archives):
        result = {"capture.json": encoded(capture),
                  "source/GRAPH.json": (self.root / "docs/site/dependency-source/GRAPH.json").read_bytes()}
        result.update({"api/" + path: raw for path, raw in originals.items()})
        for archive_name, raw in archives.items():
            result.update({archive_name.removesuffix(".zip") + "/" + name: body
                           for name, body in unpack(raw).items()})
        return result

    def decode_frames(self, raw, expected):
        import base64
        start, separator, tail = raw.partition(b"\n")
        self.assertTrue(separator)
        match = re.fullmatch(rb"::stop-commands::([0-9a-f]{48})", start)
        self.assertIsNotNone(match)
        nonce = match.group(1)
        end = b"::" + nonce + b"::\n"
        self.assertTrue(tail.endswith(end))
        tail = tail[:-len(end)]
        observed, rows = {}, []
        while tail:
            line, separator, tail = tail.partition(b"\n")
            self.assertTrue(line.startswith(b"NATIVE_ORIGINAL_BEGIN "))
            header = json.loads(line[len(b"NATIVE_ORIGINAL_BEGIN "):])
            self.assertEqual(set(header), {"path", "original_bytes", "sha256", "encoding"})
            path = header["path"]
            self.assertNotIn(path, observed)
            self.assertIn(path, expected)
            original = expected[path]
            self.assertEqual(header["original_bytes"], len(original))
            self.assertEqual(header["sha256"], sha(original))
            try:
                original.decode("utf-8")
                encoding, body = "utf-8", original
            except UnicodeDecodeError:
                encoding, body = "base64", base64.b64encode(original)
            self.assertEqual(header["encoding"], encoding)
            self.assertEqual(tail[:len(body)], body)
            self.assertEqual(tail[len(body):len(body) + 1], b"\n")
            tail = tail[len(body) + 1:]
            line, separator, tail = tail.partition(b"\n")
            self.assertTrue(line.startswith(b"NATIVE_ORIGINAL_END "))
            footer = json.loads(line[len(b"NATIVE_ORIGINAL_END "):])
            self.assertEqual(footer, dict(path=path, printed_original_bytes=len(original), sha256=sha(original)))
            self.assertNotIn(b"::" + nonce + b"::", body)
            observed[path] = original
            rows.append(dict(path=path, original_bytes=len(original), sha256=sha(original),
                             encoding=encoding, printed_original_bytes=len(original)))
        self.assertEqual(observed, expected)
        self.assertEqual([row["path"] for row in rows], sorted(expected))
        return nonce, rows

    def test_original_frames_preserve_utf8_binary_empty_and_github_command_bytes(self):
        archives = dict(self.archives)
        members = unpack(archives["core-normal.zip"])
        members["stdout.bin"] = "TEST café without terminal LF ::error::not-a-command".encode("utf-8")
        members["stderr.bin"] = b"TEST binary\x00\xff\xfe"
        result = json.loads(members["core-result.json"])
        for name, key in (("stdout.bin", "stdout"), ("stderr.bin", "stderr")):
            result[key] = dict(bytes=len(members[name]), sha256=sha(members[name]))
        members["core-result.json"] = encoded(result)
        archives["core-normal.zip"] = self.fixture.archive(members)
        cap, needs, originals = self.make_capture(archives)
        expected = self.frame_inputs(cap, originals, archives)
        self.assertIn(b"", expected.values())
        nonces = []
        for _ in range(2):
            output = io.BytesIO()
            receipt = write_original_frames(self.root, self.context, needs, encoded(cap),
                                            originals, archives, output)
            nonce, rows = self.decode_frames(output.getvalue(), expected)
            nonces.append(nonce)
            self.assertEqual(set(receipt), FRAME_KEYS)
            self.assertEqual(receipt["schema"], "main-unittest.original-frames.v1")
            self.assertEqual(receipt["context"], self.context)
            self.assertEqual(receipt["capture_sha256"], sha(encoded(cap)))
            self.assertEqual(receipt["transport_authority"], "NONE_FROM_SUPPLIED_BYTES")
            self.assertIs(receipt["complete"], True)
            self.assertEqual(receipt["members"], rows)
            self.assertTrue(all(set(row) == FRAME_ROW_KEYS for row in rows))
        self.assertNotEqual(nonces[0], nonces[1])

    def test_original_frames_refuse_unadmitted_inputs_before_printing(self):
        for changed_archive in (True, False):
            cap, needs, originals, archives = self.fresh()
            if changed_archive:
                archives["core-normal.zip"] += b"TEST changed raw ZIP"
                reason = "ARCHIVE_MISMATCH"
            else:
                cap["responses"][0]["complete"] = False
                reason = "INCOMPLETE"
            output = io.BytesIO()
            with self.assertRaises(ValueError) as refusal:
                write_original_frames(self.root, self.context, needs, encoded(cap),
                                      originals, archives, output)
            self.assertEqual(str(refusal.exception), reason)
            self.assertEqual(output.getvalue(), b"")

    def test_wrapper_never_establishes_http_authority_or_run_completion(self):
        result = self.call()
        for field in ("transport_authority", "completion_authority"):
            self.assertEqual(result[field], "NONE_FROM_SUPPLIED_BYTES")
        self.assertIs(result["native_completion_required"], True)
        self.assertEqual(result["scientific_effect"], "NONE")
        for value in (result, result["join"]):
            self.assertIs(value["g5_claim"], False)
            self.assertIs(value["cbz_claim"], False)
            self.assertEqual(value["scientific_effect"], "NONE")
        for forbidden in ("authenticated", "token_present", "native_completed", "g5_claim"):
            cap = copy.deepcopy(self.capture)
            cap[forbidden] = True
            self.refused("INVALID_CAPTURE", capture=encoded(cap))
        cap, needs, originals, archives = self.fresh()
        run = json.loads(originals["api-run-attempt.json"])
        run["TEST_claimed_authenticated"] = True
        self.replace_response(cap, originals, "api-run-attempt.json", run)
        self.assertEqual(self.call(capture=encoded(cap), originals=originals)["transport_authority"],
                         "NONE_FROM_SUPPLIED_BYTES")


if __name__ == "__main__":
    unittest.main(verbosity=2)
