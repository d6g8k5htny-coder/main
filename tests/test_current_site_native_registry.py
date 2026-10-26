"""Private Task1 behavioral contract; execute ONLY on hosted ephemeral runners.

Fixtures are independently constructed metadata/byte joins, not ZIP admission
or scientific evidence. No production import happens during test collection.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import json
import io
import os
import socket
import subprocess
import unittest
import urllib.request
from unittest.mock import patch


REPOSITORY = "d6g8k5htny-coder/main"
ORIGIN = "https://d6g8k5htny-coder.github.io/main/site/"
BUILD = "b" * 40
PRODUCER = "a" * 40
TREE = "c" * 40
ARTIFACT = "1" * 64
MIB = 1024 * 1024
SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "7858329974e28be79f29b22644370084ff43da4f",
    "captured_at": "2026-10-03T15:07:03Z",
    "graph_sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
    "gate_sha256": "a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8",
}
BINDING = {
    "checked_commit": PRODUCER, "repository": REPOSITORY,
    "run_id": "100", "run_attempt": "2", "receipt_sha256": "2" * 64,
    "artifact_id": "300", "artifact_sha256": ARTIFACT,
    "artifact_name": "research-architecture-100-2",
}
SELECTION = {
    "workflow_id": "400", "workflow_path": ".github/workflows/workspace-landing.yml",
    "event": "push", "original_created_at": "2026-10-25T12:00:00Z",
    "producer_tree": TREE,
}
CONTEXT = {"repository": REPOSITORY, "ref": "refs/heads/main", "commit": BUILD,
           "run_id": 500, "run_attempt": 1, "event": "schedule"}
BASE = {"receipt_commit": None, "public_receipt_commit": None,
        "registry_sha256": None, "sequence": 0}
PAYLOAD = b'{"fixture":"current-payload"}\n'


def canonical(value):
    """Independent serialization only; never calls a production helper."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def descriptor(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": digest(raw)}


def entry(prefix=ARTIFACT, lengths=(1, 1, 1)):
    """Metadata policy fixture; no corresponding archive is claimed."""
    binding = dict(BINDING, artifact_sha256=prefix)
    names = ("architecture-run-receipt.json", "graph.json", "projection.json")
    files = [{"path": "native-graph/" + prefix + "/" + name,
              "bytes": length, "sha256": "3" * 64}
             for name, length in zip(names, lengths)]
    return {"binding": binding, "selection": dict(SELECTION), "files": files}


def registry(entries=None, current=ARTIFACT, sequence=1, previous=None):
    if entries is None:
        entries = [entry()]
    total = sum(item["bytes"] for row in entries for item in row["files"])
    return {
        "schema_version": 1, "scientific_effect": "NONE",
        "scientific_status_authority": False, "custody": "unknown", "origin": ORIGIN,
        "sequence": sequence, "previous_registry_sha256": previous,
        "current_artifact_sha256": current, "entries": entries, "immutable_bytes": total,
        "capacity": {"archives": len(entries), "maximum_archives": 8192,
                     "bytes": total, "maximum_bytes": 536870912},
    }


def bundle_fixture(prefix=ARTIFACT, producer=PRODUCER):
    """Small pure-byte closure fixture, never an acquired or admitted ZIP."""
    graph = {"schema_version": 1, "source": dict(SOURCE), "nodes": {}, "edges": [],
             "dimensions": {}, "scientific_effect": "NONE", "scientific_status_authority": False}
    graph_raw = canonical(graph)
    image = [{"Id": "sha256:"+"4"*64,
              "RepoDigests": ["python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"],
              "Os": "linux", "Architecture": "amd64"}]
    # Thirteen closed archive member descriptors; these fixture bytes are never zipped.
    member_raws = {
        "generated/graph.json": graph_raw, "checked-commit.txt": (producer+"\n").encode("ascii"),
        "run-id.txt": b"100\n", "run-attempt.txt": b"2\n", "container-exit-status.txt": b"0\n",
        "tests-normal.log": b"fixture-only-normal-not-an-admission-log\n",
        "tests-optimized.log": b"fixture-only-optimized-not-an-admission-log\n",
        "runtime-image.json": canonical(image), "exporter.stdout": b"/output/generated/graph.json\n",
        "exporter.stderr": b"", "console.log": b"fixture-only-console\n",
    }
    receipt = {
        "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        "checked_commit": producer, "repository": REPOSITORY, "run_id": "100", "run_attempt": "2",
        "runtime_image": {"id": "sha256:" + "4" * 64,
            "repo_digest": "python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"},
        "test_modes": [{"mode": mode, "test_count": 1,
                        "test_ids": ["test_architecture_fixture.Fixture.test_fixture"]}
                       for mode in ("normal", "optimized")],
        "graph_source": dict(SOURCE),
        "evidence_file_sha256": {name: digest(member_raws[name]) for name in (
            "generated/graph.json", "checked-commit.txt", "run-id.txt", "run-attempt.txt",
            "container-exit-status.txt", "tests-normal.log", "tests-optimized.log", "runtime-image.json")},
    }
    receipt["evidence_file_sha256"]["generated/graph.json"] = digest(graph_raw)
    receipt_raw = canonical(receipt)
    binding = dict(BINDING, artifact_sha256=prefix, checked_commit=producer,
                   receipt_sha256=digest(receipt_raw))
    member = descriptor("generated/graph.json", graph_raw)
    member_raws["architecture-run-receipt.json"] = receipt_raw
    member_raws["ARTIFACT_SHA256SUMS"] = "".join(
        digest(member_raws[name])+"  "+name+"\n" for name in sorted(member_raws)).encode("ascii")
    admission = {
        "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        "custody": "unknown", "verification_scope": "declared-native-pins-and-archive-content",
        "binding": binding,
        "archive": {"sha256": prefix, "bytes": 64000,
                    "members": [descriptor(name, member_raws[name]) for name in sorted(member_raws)]},
        "graph_member": member, "producer_receipt_sha256": digest(receipt_raw),
        "producer_receipt": receipt,
    }
    projection = {
        "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        "custody": "unknown", "graph": {"identity": {"kind": "native-artifact-member",
        "binding": binding, "member": member}, "data": graph}, "admission": admission,
    }
    stem = "native-graph/" + prefix + "/"
    files = {stem + "graph.json": graph_raw, stem + "architecture-run-receipt.json": receipt_raw,
             stem + "projection.json": canonical(projection),
             "proof-tree.html": b'<!doctype html><html lang="en"><title>Fixture</title></html>\n'}
    manifest = {
        "schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
        "custody": "unknown", "verification_scope": "declared-build-context-and-admitted-native-archive",
        "build": {"repository": REPOSITORY, "commit": BUILD,
                  "current_payload": descriptor("current.json", PAYLOAD)},
        "native": {"binding": binding, "graph_source": dict(SOURCE)},
        "files": [descriptor(path, files[path]) for path in sorted(files)],
    }
    files["current-build-manifest.json"] = canonical(manifest)
    return files, binding, dict(SELECTION), manifest


def reframe_manifest(files):
    """Rehash a deliberate mutation so its deeper identity join gets exercised."""
    manifest = json.loads(files["current-build-manifest.json"])
    manifest["files"] = [descriptor(path, files[path]) for path in sorted(files)
                         if path != "current-build-manifest.json"]
    files["current-build-manifest.json"] = canonical(manifest)



def publication_candidate(version=2, build_commit=BUILD, binding=None, selection=None):
    """Full candidate shape independently copied from frozen v1 Source/R04."""
    if binding is None:
        binding = dict(BINDING)
    if selection is None:
        selection = dict(SELECTION)
    candidate = {
        "schema_version": version, "scientific_effect": "NONE", "repository": REPOSITORY,
        "receipt_branch": "codex/site-publication-receipts",
        "source": {"commit": build_commit, "ref": "refs/heads/main",
                   "workflow": ".github/workflows/current-site.yml"},
        "run": {"id": 500, "attempt": 1},
        "artifact": {"sha256": "6"*64, "files": 12, "bytes": 10000},
        "manifest": {"bytes": 100, "sha256": "7"*64},
        "metadata": {"bytes": 100, "sha256": "8"*64},
        "payload": {"bytes": len(PAYLOAD), "sha256": digest(PAYLOAD)},
        "snapshot_ids": ["9"*64],
    }
    if version == 2:
        candidate["native"] = {
            "base": dict(BASE), "sequence": 1, "binding": copy.deepcopy(binding),
            "selection": copy.deepcopy(selection),
            "registry": {"path": "native-graph/index.json", "bytes": 1000, "sha256": "a"*64},
            "build_manifest": {"path": "current-build-manifest.json", "bytes": 2000, "sha256": "b"*64},
            "readback_scope": "legacy3+current-native5+registry;retained-native-assets-not-observed",
        }
    return candidate


def progress_current(build_commit=BUILD, producer=PRODUCER, prefix=ARTIFACT,
                     run_id="100", attempt="2", created_at="2026-10-25T12:00:00Z",
                     build_tree=TREE, producer_tree=TREE):
    binding = dict(BINDING, checked_commit=producer, artifact_sha256=prefix,
                   run_id=run_id, run_attempt=attempt,
                   artifact_name="research-architecture-"+run_id+"-"+attempt)
    selection = dict(SELECTION, original_created_at=created_at, producer_tree=producer_tree)
    return {"build": {"repository": REPOSITORY, "commit": build_commit, "tree": build_tree},
            "producer": {"binding": binding, "selection": selection}}


def native_commit(sha, tree, parents=()):
    """Ordinary complete native compare commit object, not a bool verdict."""
    api = "https://api.github.com/repos/"+REPOSITORY
    web = "https://github.com/"+REPOSITORY
    person = {"name": "Fixture", "email": "fixture@example.invalid", "date": "2026-10-25T12:00:00Z"}
    return {
        "sha": sha, "node_id": "fixture-native-commit-"+sha,
        "commit": {"author": dict(person), "committer": dict(person), "message": "Fixture commit",
                   "tree": {"sha": tree, "url": api+"/git/trees/"+tree},
                   "url": api+"/git/commits/"+sha, "comment_count": 0,
                   "verification": {"verified": False, "reason": "unsigned", "signature": None,
                                    "payload": None, "verified_at": None}},
        "url": api+"/commits/"+sha, "html_url": web+"/commit/"+sha,
        "comments_url": api+"/commits/"+sha+"/comments", "author": None, "committer": None,
        "parents": [{"sha": parent, "url": api+"/commits/"+parent,
                     "html_url": web+"/commit/"+parent} for parent in parents],
    }


def native_comparison(base, head, count=1, base_tree=TREE, head_tree="d"*40):
    """Complete independently authored GitHub response object; no trust marker."""
    api = "https://api.github.com/repos/"+REPOSITORY
    web = "https://github.com/"+REPOSITORY
    previous = base
    commits = []
    for index in range(count):
        sha = head if index == count-1 else format(index+1000, "040x")
        commits.append(native_commit(sha, head_tree, (previous,)))
        previous = sha
    original = native_commit(base, base_tree)
    return {
        "url": api+"/compare/"+base+"..."+head,
        "html_url": web+"/compare/"+base+"..."+head,
        "permalink_url": web+"/compare/"+base+"..."+head,
        "diff_url": web+"/compare/"+base+"..."+head+".diff",
        "patch_url": web+"/compare/"+base+"..."+head+".patch",
        "base_commit": original, "merge_base_commit": copy.deepcopy(original),
        "status": "ahead", "ahead_by": count, "behind_by": 0,
        "total_commits": count, "commits": commits, "files": [],
    }


class CurrentSiteNativeRegistryContract(unittest.TestCase):
    def shortDescription(self):
        return None

    def api(self, name):
        module_path = Path(__file__).resolve().parents[1] / "tools" / "current_site_native_registry.py"
        if not module_path.is_file():
            self.fail("missing Task1 exact Source file inside collected assertion: " + str(module_path))
        module_name = "_task1_current_site_native_registry_exact_source"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                self.fail("cannot load Task1 exact Source file inside collected assertion: " + str(module_path))
            module = importlib.util.module_from_spec(spec)
            # Register before execution, as normal imports do; dataclasses may
            # consult sys.modules while processing postponed annotations.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                if sys.modules.get(module_name) is module:
                    del sys.modules[module_name]
                raise
        else:
            self.assertEqual(getattr(module, "__file__", None), str(module_path),
                             "cached Task1 module must come from the exact Source file")
        function = getattr(module, name, None)
        self.assertTrue(callable(function), "missing Task1 callable: " + name)
        return function

    def admit(self, function, files, binding, selection, context=None, payload=PAYLOAD):
        return function(files, context=dict(CONTEXT) if context is None else context,
                        current_payload_bytes=payload, binding=binding, selection=selection)

    def build(self, function, previous, files, binding, selection, base=None, context=None):
        return function(previous, files, base=dict(BASE) if base is None else base,
                        context=dict(CONTEXT) if context is None else context,
                        current_payload_bytes=PAYLOAD, binding=binding, selection=selection)

    def test_registry_round_trip_exact_closed_bytes(self):
        # Catches loss of any closed field or alteration of independent golden bytes.
        validate = self.api("validate_native_registry")
        expected = registry()
        raw = canonical(expected)
        self.assertEqual(validate(raw), expected)
        self.assertEqual(canonical(validate(raw)), raw)
        for field in expected:
            bad = copy.deepcopy(expected); del bad[field]
            with self.subTest(missing=field), self.assertRaises(ValueError): validate(canonical(bad))
        bad = dict(expected, extra="refuse")
        with self.assertRaises(ValueError): validate(canonical(bad))

    def test_registry_strict_json_types_and_duplicate_nonfinite_documents(self):
        # Catches permissive JSON parsing and bool/int coercion.
        validate = self.api("validate_native_registry")
        raw = canonical(registry())
        bad_raws = [raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
                    raw.replace(b'"sequence":1', b'"sequence":NaN'),
                    raw.replace(b'"sequence":1', b'"sequence":Infinity'),
                    raw + b'{}\n', b'\xff' + raw, raw.decode("ascii"), bytearray(raw),
                    b'null\n', b'[]\n', b'1\n', b'true\n']
        for bad in bad_raws:
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError): validate(bad)
        for field, value in [("schema_version", True), ("sequence", True), ("immutable_bytes", True),
                             ("scientific_status_authority", 0), ("sequence", 0),
                             ("sequence", 2**63), ("sequence", "1"), ("sequence", 1.0),
                             ("custody", "known"), ("scientific_effect", "PASS"),
                             ("origin", ORIGIN.replace("https:", "http:"))]:
            bad = registry(); bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): validate(canonical(bad))
        high = registry(sequence=2**63-1)
        self.assertEqual(validate(canonical(high)), high)

    def test_registry_nested_closed_fields_and_pointer_capacity_joins(self):
        # Catches forged current pointers, aggregate counters or nested extras.
        validate = self.api("validate_native_registry")
        changes = [lambda r: r.update(current_artifact_sha256="f"*64),
                   lambda r: r.update(previous_registry_sha256="F"*64),
                   lambda r: r.update(immutable_bytes=4),
                   lambda r: r["capacity"].update(archives=2),
                   lambda r: r["capacity"].update(bytes=4),
                   lambda r: r["capacity"].update(maximum_archives=8193),
                   lambda r: r["capacity"].update(maximum_bytes=536870913),
                   lambda r: r["capacity"].update(extra=0),
                   lambda r: r["entries"][0].update(extra=0),
                   lambda r: r["entries"][0]["binding"].update(extra="x"),
                   lambda r: r["entries"][0]["selection"].update(extra="x"),
                   lambda r: r["entries"][0]["files"][0].update(extra="x"),
                   lambda r: r["entries"][0]["files"][0].update(bytes=True),
                   lambda r: r["entries"][0]["files"][0].update(sha256="A"*64),
                   lambda r: r["entries"][0]["files"][0].update(sha256=True),
                   lambda r: r["capacity"].update(archives=True),
                   lambda r: r.update(entries=[])]
        for index, mutate in enumerate(changes):
            bad = registry(); mutate(bad)
            with self.subTest(case=index), self.assertRaises(ValueError): validate(canonical(bad))
        for section in ("capacity",):
            for key in registry()[section]:
                bad = registry(); del bad[section][key]
                with self.subTest(section=section, missing=key), self.assertRaises(ValueError): validate(canonical(bad))
        for key in ("path", "bytes", "sha256"):
            bad = registry(); del bad["entries"][0]["files"][0][key]
            with self.subTest(descriptor_missing=key), self.assertRaises(ValueError): validate(canonical(bad))
        for section in ("binding", "selection"):
            for key in entry()[section]:
                bad = registry(); del bad["entries"][0][section][key]
                with self.subTest(section=section, missing=key), self.assertRaises(ValueError): validate(canonical(bad))

    def test_registry_binding_selection_grammar_and_native_utc(self):
        # Catches stale artifact-name joins and permissive decimal/date grammar.
        validate = self.api("validate_native_registry")
        cases = [("binding", "run_id", "01"), ("binding", "run_attempt", 2),
                 ("binding", "artifact_id", "0"), ("binding", "artifact_sha256", "A"*64),
                 ("binding", "checked_commit", "A"*40), ("binding", "receipt_sha256", "a"*63),
                 ("binding", "repository", "../main"), ("binding", "artifact_name", "research-architecture-100-1"),
                 ("selection", "workflow_id", "01"), ("selection", "workflow_id", 400),
                 ("selection", "workflow_path", ".github/workflows/other.yml"),
                 ("selection", "event", "schedule"), ("selection", "producer_tree", "C"*40),
                 ("selection", "original_created_at", "2026-02-30T12:00:00Z"),
                 ("selection", "original_created_at", "2026-10-25T12:00:00+00:00"),
                 ("selection", "original_created_at", "2026-10-25T12:00:00.0Z")]
        for section, key, value in cases:
            bad = registry(); bad["entries"][0][section][key] = value
            with self.subTest(section=section, field=key, value=value), self.assertRaises(ValueError):
                validate(canonical(bad))

    def test_registry_exact_sorted_paths_and_unique_prefixes(self):
        # Catches path traversal, case drift, duplicate descriptors and unsorted archives.
        validate = self.api("validate_native_registry")
        for path in ["native-graph/"+ARTIFACT+"/../graph.json", "native-graph/"+ARTIFACT+"/%67raph.json",
                     "native-graph/"+ARTIFACT+"/GRAPH.json", "native-graph\\"+ARTIFACT+"\\graph.json",
                     "native-graph/"+"f"*64+"/graph.json", "/native-graph/"+ARTIFACT+"/graph.json"]:
            bad = registry(); bad["entries"][0]["files"][1]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError): validate(canonical(bad))
        for files in [entry()["files"][::-1], entry()["files"][:-1], entry()["files"]+[entry()["files"][0]],
                      [entry()["files"][0]]*3]:
            bad = registry(); bad["entries"][0]["files"] = files
            with self.subTest(files=files), self.assertRaises(ValueError): validate(canonical(bad))
        for entries in [[entry(), entry()], [entry("2"*64), entry()]]:
            with self.subTest(prefixes=[r["binding"]["artifact_sha256"] for r in entries]), self.assertRaises(ValueError):
                validate(canonical(registry(entries)))

    def test_manifest_payload_build_and_binding_joins_refuse_before_output(self):
        # Catches independent manifest/binding/context or exact payload-byte substitution.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, expected = bundle_fixture()
        admitted = self.admit(admit, files, binding, selection)
        self.assertIs(type(admitted), dict)
        self.assertEqual(admitted["build"], expected["build"])
        self.assertEqual(admitted["native"], expected["native"])
        self.assertEqual(admitted["files"], expected["files"])
        for field, value in [("commit", "d"*40), ("repository", "other/repository")]:
            bad = dict(CONTEXT); bad[field] = value
            with self.subTest(context=field), self.assertRaises(ValueError): self.admit(admit, files, binding, selection, bad)
        with self.assertRaises(ValueError): self.admit(admit, files, binding, selection, payload=PAYLOAD+b" ")
        for field, value in [("checked_commit", "d"*40), ("artifact_id", "301"),
                             ("receipt_sha256", "f"*64), ("run_attempt", "3")]:
            bad = dict(binding); bad[field] = value
            with self.subTest(binding=field), self.assertRaises(ValueError): self.admit(admit, files, bad, selection)

    def test_manifest_exact_four_descriptors_excludes_self_and_five_buffers(self):
        # Catches sixth-output adoption, missing output, forged descriptors and self hashing.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, _ = bundle_fixture()
        self.admit(admit, files, binding, selection)
        for path in files:
            bad = dict(files); del bad[path]
            with self.subTest(missing=path), self.assertRaises(ValueError): self.admit(admit, bad, binding, selection)
        for path in ["extra.txt", "native-graph/"+"f"*64+"/graph.json"]:
            bad = dict(files); bad[path] = b"x"
            with self.subTest(extra=path), self.assertRaises(ValueError): self.admit(admit, bad, binding, selection)
        changes = [lambda m: m["files"].append(descriptor("current-build-manifest.json", b"x")),
                   lambda m: m["files"].reverse(),
                   lambda m: m["files"][0].update(bytes=True),
                   lambda m: m["files"][0].update(bytes=m["files"][0]["bytes"]+1),
                   lambda m: m["files"][0].update(sha256="f"*64),
                   lambda m: m["files"][0].update(extra="x"),
                   lambda m: m.update(extra="x"), lambda m: m["build"].update(extra="x"),
                   lambda m: m["native"].update(extra="x"),
                   lambda m: m["native"]["graph_source"].update(commit="d"*40),
                   lambda m: m["build"]["current_payload"].update(path="other.json"),
                   lambda m: m.update(scientific_status_authority=0)]
        for index, mutate in enumerate(changes):
            bad = dict(files); manifest = json.loads(bad["current-build-manifest.json"]); mutate(manifest)
            bad["current-build-manifest.json"] = canonical(manifest)
            with self.subTest(case=index), self.assertRaises(ValueError): self.admit(admit, bad, binding, selection)

    def test_manifest_strict_json_and_exact_byte_map_types(self):
        # Catches forgiving duplicate/nonfinite parsers or mutable/nonbyte buffers.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, _ = bundle_fixture()
        manifest = files["current-build-manifest.json"]
        for raw in [manifest.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
                    manifest.replace(b'"schema_version":1', b'"schema_version":NaN'),
                    manifest+b'{}', b'\xff', bytearray(manifest), manifest.decode("ascii")]:
            bad = dict(files); bad["current-build-manifest.json"] = raw
            with self.subTest(kind=type(raw).__name__), self.assertRaises(ValueError): self.admit(admit, bad, binding, selection)
        for value in [bytearray(files["proof-tree.html"]), "page", None]:
            bad = dict(files); bad["proof-tree.html"] = value
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError): self.admit(admit, bad, binding, selection)

    def test_manifest_and_original_receipt_projection_identity_joins(self):
        # Catches accepting internally rehashed content with a different admitted binding.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, _ = bundle_fixture()
        prefix = "native-graph/"+ARTIFACT+"/"
        for path, location, field, value in [
            (prefix+"projection.json", ("admission", "binding"), "artifact_id", "301"),
            (prefix+"projection.json", ("graph", "identity", "binding"), "checked_commit", "d"*40),
            (prefix+"projection.json", ("admission",), "producer_receipt_sha256", "f"*64),
            (prefix+"projection.json", ("graph", "identity", "member"), "sha256", "f"*64),
            (prefix+"projection.json", ("admission", "graph_member"), "bytes", 1),
            (prefix+"architecture-run-receipt.json", (), "run_id", "101"),
            (prefix+"graph.json", ("source",), "commit", "d"*40)]:
            bad = dict(files); document = json.loads(bad[path]); target = document
            for key in location: target = target[key]
            target[field] = value; bad[path] = canonical(document); reframe_manifest(bad)
            with self.subTest(path=path, location=location, field=field), self.assertRaises(ValueError):
                self.admit(admit, bad, binding, selection)

    def test_execution_context_is_closed_exact_and_main_only(self):
        # Catches widened events/refs/repositories and integer coercion.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, _ = bundle_fixture()
        for event in ["push", "schedule", "workflow_dispatch"]:
            self.admit(admit, files, binding, selection, dict(CONTEXT, event=event))
        for field in CONTEXT:
            bad = dict(CONTEXT); del bad[field]
            with self.subTest(missing=field), self.assertRaises(ValueError): self.admit(admit, files, binding, selection, bad)
        for field, value in [("extra", "x"), ("event", "pull_request"), ("ref", "refs/heads/other"),
                             ("repository", "D6g8k5htny-coder/main"), ("run_id", True),
                             ("run_id", "500"), ("run_id", 0), ("run_attempt", 0), ("run_attempt", 1.0),
                             ("commit", BUILD.upper())]:
            bad = dict(CONTEXT); bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.admit(admit, files, binding, selection, bad)

    def test_page_and_manifest_capacity_boundaries(self):
        # Catches page/manifest raw-byte caps; no archive or aggregate-boundary claim.
        admit = self.api("admit_current_native_bundle")
        files, binding, selection, _ = bundle_fixture()
        exact = dict(files); exact["proof-tree.html"] = b"x"*(4*MIB); reframe_manifest(exact)
        self.admit(admit, exact, binding, selection)
        over = dict(exact); over["proof-tree.html"] += b"x"; reframe_manifest(over)
        with self.assertRaises(ValueError): self.admit(admit, over, binding, selection)
        # A manifest may contain JSON whitespace on input; produced bytes are canonical.
        # Exact manifest capacity is independently tested without adding a schema field.
        raw = files["current-build-manifest.json"]
        exact = dict(files); exact["current-build-manifest.json"] = raw[:-1] + b" "*(65536-len(raw)) + b"\n"
        self.admit(admit, exact, binding, selection)
        over = dict(exact); over["current-build-manifest.json"] += b" "
        with self.assertRaises(ValueError): self.admit(admit, over, binding, selection)

    def test_build_registry_exact_independent_golden_and_original_base(self):
        # Catches a changed serializer, sequence bootstrap or accidental sixth bundle output.
        build = self.api("build_native_registry")
        files, binding, selection, _ = bundle_fixture()
        before = copy.deepcopy(files)
        expected_entry = {"binding": binding, "selection": selection,
                          "files": [descriptor(path, files[path]) for path in sorted(files)
                                    if path.startswith("native-graph/")]}
        expected = canonical(registry([expected_entry]))
        first = self.build(build, None, files, binding, selection)
        second = self.build(build, None, files, binding, selection)
        self.assertIs(type(first), bytes)
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(files, before)
        self.assertEqual(len(files), 5)

    def test_union_conserves_all_old_entries_and_reuse_adds_zero_versions(self):
        # Catches dropping old entries or treating each refresh as a new archive.
        build = self.api("build_native_registry")
        files, binding, selection, _ = bundle_fixture()
        current_entry = {"binding": binding, "selection": selection,
                         "files": [descriptor(path, files[path]) for path in sorted(files)
                                   if path.startswith("native-graph/")]}
        older = entry("0"*64)
        old = registry([older, current_entry], sequence=7)
        old_raw = canonical(old)
        base = {"receipt_commit": "d"*40, "public_receipt_commit": "e"*40,
                "registry_sha256": digest(old_raw), "sequence": 7}
        expected = registry([older, current_entry], sequence=8, previous=digest(old_raw))
        result = self.build(build, old_raw, files, binding, selection, base)
        self.assertEqual(result, canonical(expected))
        self.assertEqual(json.loads(result)["immutable_bytes"], old["immutable_bytes"])
        self.assertEqual(json.loads(result)["capacity"]["archives"], 2)
        added_files, added_binding, added_selection, _ = bundle_fixture("f"*64)
        added_entry = {"binding": added_binding, "selection": added_selection,
                       "files": [descriptor(path, added_files[path]) for path in sorted(added_files)
                                 if path.startswith("native-graph/")]}
        want = registry([older, current_entry, added_entry], "f"*64, 8, digest(old_raw))
        self.assertEqual(self.build(build, old_raw, added_files, added_binding, added_selection, base), canonical(want))

    def test_same_prefix_entire_old_binding_selection_and_descriptors_are_conserved(self):
        # Catches overwriting a retained identity under the same original-ZIP prefix.
        build = self.api("build_native_registry")
        files, binding, selection, _ = bundle_fixture()
        good = {"binding": binding, "selection": selection,
                "files": [descriptor(path, files[path]) for path in sorted(files) if path.startswith("native-graph/")]}
        changes = [lambda e: e["binding"].update(artifact_id="301"),
                   lambda e: e["selection"].update(workflow_id="401"),
                   lambda e: e["selection"].update(producer_tree="d"*40),
                   lambda e: e["files"][0].update(sha256="f"*64),
                   lambda e: e["files"][0].update(bytes=e["files"][0]["bytes"]+1)]
        for index, mutate in enumerate(changes):
            old_entry = copy.deepcopy(good); mutate(old_entry)
            old_raw = canonical(registry([old_entry], sequence=4))
            base = {"receipt_commit": "d"*40, "public_receipt_commit": "e"*40,
                    "registry_sha256": digest(old_raw), "sequence": 4}
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.build(build, old_raw, files, binding, selection, base)

    def test_unregistered_prefix_is_not_adopted(self):
        # Catches silently adopting an orphan immutable path in the five-output input.
        build = self.api("build_native_registry")
        files, binding, selection, _ = bundle_fixture()
        files["native-graph/"+"f"*64+"/graph.json"] = b"orphan"
        with self.assertRaises(ValueError): self.build(build, None, files, binding, selection)

    def test_base_exact_closed_lineage_hash_sequence_and_types(self):
        # Catches resetting lineage after unknown/mismatching prior state.
        build = self.api("build_native_registry")
        files, binding, selection, _ = bundle_fixture()
        for field, value in [("sequence", True), ("sequence", 1), ("sequence", -1),
                             ("registry_sha256", "f"*64), ("receipt_commit", "D"*40),
                             ("public_receipt_commit", "x"), ("extra", None)]:
            bad = dict(BASE); bad[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.build(build, None, files, binding, selection, bad)
        for field in BASE:
            bad = dict(BASE); del bad[field]
            with self.subTest(missing=field), self.assertRaises(ValueError): self.build(build, None, files, binding, selection, bad)
        old_entry = {"binding": binding, "selection": selection,
                     "files": [descriptor(path, files[path]) for path in sorted(files)
                               if path.startswith("native-graph/")]}
        old_raw = canonical(registry([old_entry], sequence=7))
        good_base = {"receipt_commit": "d"*40, "public_receipt_commit": "e"*40,
                     "registry_sha256": digest(old_raw), "sequence": 7}
        self.assertEqual(json.loads(self.build(build, old_raw, files, binding, selection, good_base))["sequence"], 8)
        for sequence, sha in [(0, digest(old_raw)), (7, "f"*64), (6, digest(old_raw))]:
            base = {"receipt_commit": "d"*40, "public_receipt_commit": "e"*40,
                    "registry_sha256": sha, "sequence": sequence}
            with self.subTest(sequence=sequence, sha=sha), self.assertRaises(ValueError): self.build(build, old_raw, files, binding, selection, base)

    def test8192_exact_and8193_refused(self):
        # Catches an off-by-one archive count policy; metadata only, no ZIPs/reads.
        validate = self.api("validate_native_registry")
        rows = [entry(format(n, "064x")) for n in range(8192)]
        expected = registry(rows, rows[-1]["binding"]["artifact_sha256"])
        self.assertEqual(validate(canonical(expected)), expected)
        rows.append(entry(format(8192, "064x")))
        with self.assertRaises(ValueError): validate(canonical(registry(rows, rows[-1]["binding"]["artifact_sha256"])))

    def test512mib_exact_and_one_over_refused_before_reads(self):
        # Catches aggregate capacity overflow; API is pure and no archive is read.
        validate = self.api("validate_native_registry")
        rows = [entry(format(n, "064x"), (32*MIB-2, 1, 1)) for n in range(16)]
        expected = registry(rows, rows[-1]["binding"]["artifact_sha256"])
        with patch("builtins.open", side_effect=AssertionError("pure validation attempted a read")):
            self.assertEqual(validate(canonical(expected)), expected)
            rows.append(entry("f"*64))
            # Use exactly +1 aggregate, preserving positive three-buffer descriptors:
            rows[-2]["files"][0]["bytes"] -= 2
            over = registry(rows, "f"*64)
            self.assertEqual(over["immutable_bytes"], 536870913)
            with self.assertRaises(ValueError): validate(canonical(over))

    def test_entry32mib_exact_and_one_over_and_nonpositive_descriptors(self):
        # Catches invented 16MiB per-asset limits and entry aggregate overflow.
        validate = self.api("validate_native_registry")
        expected = registry([entry(lengths=(32*MIB-2, 1, 1))])
        self.assertEqual(validate(canonical(expected)), expected)
        for length in [32*MIB-1, 32*MIB+1, 0, -1, True, "1", 1.0]:
            bad = registry([entry(lengths=(length, 1, 1))]) if type(length) is int else registry()
            if type(length) is not int: bad["entries"][0]["files"][0]["bytes"] = length
            with self.subTest(length=length), self.assertRaises(ValueError): validate(canonical(bad))

    def test_registry16mib_exact_and_one_over(self):
        # Catches registry raw-byte off-by-one independently of archive capacity.
        validate = self.api("validate_native_registry")
        expected = registry()
        # Positive workflow_id is a string without a specified digit-count cap.
        raw = canonical(expected)
        expected["entries"][0]["selection"]["workflow_id"] += "0"*(16*MIB-len(raw))
        exact = canonical(expected)
        self.assertEqual(len(exact), 16777216)
        self.assertEqual(validate(exact), expected)
        expected["entries"][0]["selection"]["workflow_id"] += "0"
        over = canonical(expected)
        self.assertEqual(len(over), 16777217)
        with self.assertRaises(ValueError): validate(over)


    def test_progress_none_and_same_producer_need_no_comparison(self):
        # Catches rejecting first native publication or charging fake repeated versions.
        progress = self.api("validate_native_progress")
        current = progress_current()
        self.assertIsNone(progress(None, current, comparisons={}))
        previous = publication_candidate()
        before = copy.deepcopy((previous, current))
        self.assertIsNone(progress(previous, current, comparisons={}))
        self.assertEqual((previous, current), before)

    def test_progress_v1_has_build_obligation_without_prior_native_producer(self):
        # Catches incorrectly skipping build ancestry on a v1-to-v2 migration.
        progress = self.api("validate_native_progress")
        previous = publication_candidate(1, "e"*40)
        current = progress_current()
        proof = native_comparison("e"*40, BUILD, head_tree=TREE)
        self.assertIsNone(progress(previous, current, comparisons={"build": proof}))
        with self.assertRaises(ValueError): progress(previous, current, comparisons={})
        self.assertIsNone(progress(publication_candidate(1), current, comparisons={}))

    def test_progress_older_original_run_is_refused_and_numeric_ties_ordered(self):
        # Catches timestamp-only, lexical-ID, or older-success producer fallback.
        progress = self.api("validate_native_progress")
        previous = publication_candidate(binding=dict(BINDING, run_id="100", artifact_name="research-architecture-100-2"))
        for current in [progress_current(prefix="f"*64, run_id="999", created_at="2026-10-25T11:59:59Z"),
                        progress_current(prefix="f"*64, run_id="99")]:
            with self.subTest(run=current["producer"]["binding"]["run_id"]), self.assertRaises(ValueError):
                progress(previous, current, comparisons={})
        self.assertIsNone(progress(previous, progress_current(prefix="f"*64, run_id="101"), comparisons={}))
        self.assertIsNone(progress(previous, progress_current(prefix="f"*64, run_id="9", created_at="2026-10-25T12:00:01Z"), comparisons={}))

    def test_progress_same_original_run_attempt_must_not_decrease(self):
        # Catches same-run attempt rewind, including lexical 10-vs-2 ordering.
        progress = self.api("validate_native_progress")
        binding = dict(BINDING, run_attempt="10", artifact_name="research-architecture-100-10")
        previous = publication_candidate(binding=binding)
        with self.assertRaises(ValueError):
            progress(previous, progress_current(prefix="f"*64, attempt="2"), comparisons={})
        self.assertIsNone(progress(previous, progress_current(prefix="f"*64, attempt="10"), comparisons={}))
        self.assertIsNone(progress(previous, progress_current(prefix="f"*64, attempt="11"), comparisons={}))

    def test_progress_same_prefix_preserves_entire_binding_and_selection(self):
        # Catches allowing retained-prefix identity drift merely because P is unchanged.
        progress = self.api("validate_native_progress")
        previous = publication_candidate()
        for section, field, value in [("binding", "artifact_id", "301"),
                                      ("binding", "receipt_sha256", "f"*64),
                                      ("selection", "workflow_id", "401"),
                                      ("selection", "producer_tree", "e"*40),
                                      ("selection", "original_created_at", "2026-10-25T12:00:01Z")]:
            current = progress_current(); current["producer"][section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                progress(previous, current, comparisons={})

    def test_progress_different_producer_requires_complete_ancestry_but_not_equal_trees(self):
        # Catches assuming source-tree equality is needed for forward progress.
        progress = self.api("validate_native_progress")
        previous = publication_candidate()
        current = progress_current(producer="e"*40, prefix="f"*64, producer_tree="d"*40)
        proof = native_comparison(PRODUCER, "e"*40, base_tree=TREE, head_tree="d"*40)
        self.assertNotEqual(proof["base_commit"]["commit"]["tree"]["sha"], proof["commits"][-1]["commit"]["tree"]["sha"])
        self.assertIsNone(progress(previous, current, comparisons={"producer": proof}))
        with self.assertRaises(ValueError): progress(previous, current, comparisons={})

    def test_progress_different_build_has_independent_ancestry_obligation(self):
        # Catches accepting a rewound build because producer progress succeeds.
        progress = self.api("validate_native_progress")
        previous = publication_candidate()
        current = progress_current(build_commit="e"*40, build_tree="d"*40)
        proof = native_comparison(BUILD, "e"*40)
        self.assertIsNone(progress(previous, current, comparisons={"build": proof}))
        with self.assertRaises(ValueError): progress(previous, current, comparisons={})
        rewind = native_comparison(BUILD, "e"*40); rewind.update(status="behind", ahead_by=0, behind_by=1)
        with self.assertRaises(ValueError): progress(previous, current, comparisons={"build": rewind})

    def test_progress_identical_build_and_producer_pair_reuses_same_full_response(self):
        # Catches a false independent-read requirement for an identical immutable pair.
        progress = self.api("validate_native_progress")
        previous = publication_candidate(build_commit=PRODUCER)
        current = progress_current(build_commit="e"*40, producer="e"*40, prefix="f"*64,
                                   build_tree="d"*40, producer_tree="d"*40)
        proof = native_comparison(PRODUCER, "e"*40)
        self.assertIsNone(progress(previous, current, comparisons={"build": proof, "producer": proof}))

    def test_progress_bool_partial_forged_or_truncated_comparison_never_qualifies(self):
        # Catches caller bool verdicts, missing immutable endpoints and truncated commit lists.
        progress = self.api("validate_native_progress")
        previous = publication_candidate()
        current = progress_current(producer="e"*40, prefix="f"*64, producer_tree="d"*40)
        original = native_comparison(PRODUCER, "e"*40)
        invalid = [True, False, None, {"status": "ahead"}, {"qualified": True}]
        mutations = [lambda c: c.update(status="diverged"), lambda c: c.update(behind_by=1),
                     lambda c: c.update(ahead_by=True), lambda c: c.update(total_commits=True),
                     lambda c: c.update(behind_by=False), lambda c: c.update(ahead_by=0),
                     lambda c: c.update(total_commits=2), lambda c: c.update(commits=[]),
                     lambda c: c["base_commit"].update(sha="f"*40),
                     lambda c: c["merge_base_commit"].update(sha="f"*40),
                     lambda c: c["commits"][-1].update(sha="f"*40),
                     lambda c: c["commits"][-1]["commit"]["tree"].update(sha="D"*40),
                     lambda c: c["commits"][-1]["commit"]["tree"].update(sha="f"*40),
                     lambda c: c["base_commit"]["commit"]["tree"].update(sha="f"*40),
                     lambda c: c["commits"][-1].update(parents=[{"sha": "not-a-sha"}]),
                     lambda c: c["commits"][-1].pop("commit"),
                     lambda c: c["commits"][-1].pop("parents")]
        for mutate in mutations:
            bad = copy.deepcopy(original); mutate(bad); invalid.append(bad)
        duplicate = native_comparison(PRODUCER, "e"*40, 2)
        duplicate["commits"][0] = copy.deepcopy(duplicate["commits"][1]); invalid.append(duplicate)
        truncated = native_comparison(PRODUCER, "e"*40, 2)
        truncated["commits"].pop(0); invalid.append(truncated)
        for index, bad in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                progress(previous, current, comparisons={"producer": bad})

    def test_progress250_exact_and251_refused_without_reset_or_fallback(self):
        # Catches accepting overbound proof or mutating base identity after refusal.
        progress = self.api("validate_native_progress")
        previous = publication_candidate()
        current = progress_current(producer="e"*40, prefix="f"*64, producer_tree="d"*40)
        exact = native_comparison(PRODUCER, "e"*40, 250)
        self.assertIsNone(progress(previous, current, comparisons={"producer": exact}))
        before = copy.deepcopy((previous, current))
        over = native_comparison(PRODUCER, "e"*40, 251)
        with self.assertRaises(ValueError): progress(previous, current, comparisons={"producer": over})
        self.assertEqual((previous, current), before)
        with self.assertRaises(ValueError): progress(previous, current, comparisons={})
        self.assertEqual((previous, current), before)

    def test_progress_current_and_previous_identity_types_are_not_coerced(self):
        # Catches bypassing identity/schema admission on first publication or equal commits.
        progress = self.api("validate_native_progress")
        for section, field, value in [("build", "repository", "other/repo"),
                                      ("build", "commit", "B"*40), ("build", "tree", True),
                                      ("build", "extra", True)]:
            current = progress_current(); current[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                progress(None, current, comparisons={})
        for value in [True, [], {"source": {"commit": BUILD}}]:
            with self.subTest(previous=value), self.assertRaises(ValueError):
                progress(value, progress_current(), comparisons={})
        previous = publication_candidate(); previous["source"]["commit"] = True
        with self.assertRaises(ValueError): progress(previous, progress_current(), comparisons={})
        current = progress_current(); current["producer"]["binding"]["run_id"] = 100
        with self.assertRaises(ValueError): progress(None, current, comparisons={})



    def test_four_pure_apis_have_no_filesystem_network_or_subprocess_effects(self):
        # Catches accidental I/O/project orchestration in these pure metadata APIs.
        validate = self.api("validate_native_registry")
        admit = self.api("admit_current_native_bundle")
        build = self.api("build_native_registry")
        progress = self.api("validate_native_progress")
        files, binding, selection, manifest = bundle_fixture()
        current_entry = {"binding": binding, "selection": selection,
                         "files": [descriptor(path, files[path]) for path in sorted(files)
                                   if path.startswith("native-graph/")]}
        expected_raw = canonical(registry([current_entry]))
        failure = AssertionError("pure Task1 API attempted an external effect")
        with patch("builtins.open", side_effect=failure), patch("io.open", side_effect=failure), \
             patch("os.open", side_effect=failure), patch("socket.socket", side_effect=failure), \
             patch("urllib.request.urlopen", side_effect=failure), patch("subprocess.run", side_effect=failure):
            self.assertEqual(validate(expected_raw), registry([current_entry]))
            self.assertEqual(self.admit(admit, files, binding, selection), manifest)
            self.assertEqual(self.build(build, None, files, binding, selection), expected_raw)
            self.assertIsNone(progress(None, progress_current(), comparisons={}))



if __name__ == "__main__":
    unittest.main()
