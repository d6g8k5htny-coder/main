"""R04 Task2 original Git/base-proof contracts; hosted execution only.

All fixtures are small inert metadata and byte strings. They are not service
observations, acquired artifacts, HTTP attestations, or scientific evidence.
Production code is loaded only inside each collected behavioral test.
"""
from __future__ import annotations

import base64
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
import urllib.request
from unittest.mock import patch


REPOSITORY = "d6g8k5htny-coder/main"
BRANCH = "codex/site-publication-receipts"
WORKFLOW = ".github/workflows/current-site.yml"
CONSUMER = "c" * 40
SNAPSHOT_A = "a" * 64
SNAPSHOT_X = "b" * 64
BOT = {"name": "Current research site publication",
       "email": "41898282+github-actions[bot]@users.noreply.github.com"}
DATE = "2026-10-25T00:00:00Z"
LEGACY_PATHS = {"manifest": "snapshots/index.json", "metadata": "current-meta.json",
                "payload": "current.json"}
NATIVE_ABSENCE = ("current-build-manifest.json", "native-graph/index.json", "proof-tree.html")
BASE_FILES = {"anchor.json", "public-anchor.json", "base.json", "git-proof.json",
              "pending-plan.json", "qualification.json"}


def canonical(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git_sha(kind, raw):
    return hashlib.sha1(kind.encode("ascii") + b" " + str(len(raw)).encode("ascii")
                        + b"\0" + raw).hexdigest()


def descriptor(path, raw):
    return {"path": path, "bytes": len(raw), "sha256": digest(raw)}


def registry_fixture():
    """An independently framed policy registry, not a ZIP-admission claim."""
    prefix = "1" * 64
    binding = {"checked_commit": "d" * 40, "repository": REPOSITORY,
               "run_id": "101", "run_attempt": "1", "receipt_sha256": "2" * 64,
               "artifact_id": "301", "artifact_sha256": prefix,
               "artifact_name": "research-architecture-101-1"}
    selection = {"workflow_id": "401", "workflow_path": ".github/workflows/workspace-landing.yml",
                 "event": "push", "original_created_at": DATE, "producer_tree": "e" * 40}
    assets = {"native-graph/" + prefix + "/" + name: raw for name, raw in
              (("architecture-run-receipt.json", b"r"), ("graph.json", b"g"),
               ("projection.json", b"p"))}
    files = [descriptor(path, raw) for path, raw in sorted(assets.items())]
    registry = {"schema_version": 1, "scientific_effect": "NONE",
                "scientific_status_authority": False, "custody": "unknown",
                "origin": "https://d6g8k5htny-coder.github.io/main/site/",
                "sequence": 1, "previous_registry_sha256": None,
                "current_artifact_sha256": prefix,
                "entries": [{"binding": binding, "selection": selection, "files": files}],
                "immutable_bytes": 3,
                "capacity": {"archives": 1, "maximum_archives": 8192,
                             "bytes": 3, "maximum_bytes": 536870912}}
    return canonical(registry), binding, selection, assets


def candidate(label="A", *, native=False, base=None, snapshots=None):
    """Fixtures derive no expected result through a production function."""
    source = hashlib.sha1(("inert-source-" + label).encode("ascii")).hexdigest()
    raw = {"manifest": b'{"fixture":"manifest"}\n',
           "metadata": b'{"fixture":"metadata"}\n',
           "payload": b'{"fixture":"payload"}\n'}
    value = {"schema_version": 2 if native else 1, "scientific_effect": "NONE",
             "repository": REPOSITORY, "receipt_branch": BRANCH,
             "source": {"commit": source, "ref": "refs/heads/main", "workflow": WORKFLOW},
             "run": {"id": 100 + ord(label[0]) + (1000 * int(label[1:]) if label[1:] else 0),
                     "attempt": 1},
             "artifact": {"sha256": digest(label.encode("ascii")), "files": 9, "bytes": 100},
             **{name: {"bytes": len(body), "sha256": digest(body)} for name, body in raw.items()},
             "snapshot_ids": [SNAPSHOT_A] if snapshots is None else sorted(snapshots)}
    if native:
        registry_raw, binding, selection, _ = registry_fixture()
        if base is not None:
            pending_registry = json.loads(registry_raw)
            pending_registry["sequence"] = base["sequence"] + 1
            pending_registry["previous_registry_sha256"] = base["registry_sha256"]
            registry_raw = canonical(pending_registry)
        value["native"] = {
            "base": {"receipt_commit": None, "public_receipt_commit": None,
                     "registry_sha256": None, "sequence": 0} if base is None else copy.deepcopy(base),
            "sequence": 1 if base is None else base["sequence"] + 1,
            "binding": binding, "selection": selection,
            "registry": descriptor("native-graph/index.json", registry_raw),
            "build_manifest": descriptor("current-build-manifest.json", b'{"fixture":"build"}\n'),
            "readback_scope": "legacy3+current-native5+registry;retained-native-assets-not-observed"}
    return value


def receipt(value, *, phase="observed", previous="f" * 40, reconciliation=None):
    result = {**copy.deepcopy(value), "phase": phase, "recorded_at": DATE,
              "recorded_by": {"source_commit": CONSUMER, "run_id": 501, "run_attempt": 1},
              "previous_receipt_commit": previous}
    if value["schema_version"] == 2:
        result["reconciliation"] = copy.deepcopy(reconciliation)
    return result


def frame_record(value, *, receipt_raw=None, extra_commit_lf=False, parents_override=None,
                 person_override=None, message_override=None, tree_override=None):
    """Independent original Git framing with the held publisher's BOT grammar."""
    raw = canonical(value) if receipt_raw is None else receipt_raw
    blob = git_sha("blob", raw)
    tree_raw = b"100644 latest.json\0" + bytes.fromhex(blob)
    tree = git_sha("tree", tree_raw) if tree_override is None else tree_override
    parents = ([] if value["previous_receipt_commit"] is None else
               [value["previous_receipt_commit"]]) if parents_override is None else parents_override
    person = dict(BOT, date=value["recorded_at"]) if person_override is None else person_override
    message = (f"Site publication {value['phase']} run {value['run']['id']} "
               f"attempt {value['run']['attempt']}\n") if message_override is None else message_override
    epoch = int(datetime.strptime(person["date"], "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=timezone.utc).timestamp())
    lines = ["tree " + tree, *("parent " + parent for parent in parents)]
    for role in ("author", "committer"):
        lines.append(f"{role} {person['name']} <{person['email']}> {epoch} +0000")
    commit_raw = ("\n".join(lines) + "\n\n" + message).encode("utf-8")
    if extra_commit_lf:
        commit_raw += b"\n"
    return {"head": git_sha("commit", commit_raw),
            "commit": {"tree": tree, "parents": list(parents), "message": message,
                       "author": dict(person), "committer": dict(person)},
            "receipt_b64": base64.b64encode(raw).decode("ascii")}


def anchor(record):
    return {"head": record["head"],
            "receipt": json.loads(base64.b64decode(record["receipt_b64"]))}


def candidate_of(value):
    return {key: copy.deepcopy(item) for key, item in value.items()
            if key not in {"phase", "recorded_at", "recorded_by",
                           "previous_receipt_commit", "reconciliation"}}


def superseded_chain(pairs=1, *, public_native=False, extra_commit_lf=False):
    """Actual parent chain S_n->P_n->...->S_1->P_1->A, effective public A."""
    public = frame_record(receipt(candidate(native=public_native)), extra_commit_lf=extra_commit_lf)
    records = [public]
    operational = public
    for index in range(pairs):
        a = anchor(public)["receipt"]
        base = {"receipt_commit": operational["head"], "public_receipt_commit": public["head"],
                "registry_sha256": a["native"]["registry"]["sha256"] if public_native else None,
                "sequence": a["native"]["sequence"] if public_native else 0}
        current = candidate("P" + str(index), native=True, base=base,
                            snapshots=[SNAPSHOT_A, SNAPSHOT_X])
        prepared = frame_record(receipt(current, phase="prepared", previous=operational["head"]))
        reconciliation = {"kind": "unstarted-deploy",
                          "effective_public_receipt_commit": public["head"],
                          "run_id": str(current["run"]["id"]), "run_attempt": "1",
                          "job_id": str(7000 + current["run"]["id"]), "proof_sha256": "7" * 64}
        operational = frame_record(receipt(current, phase="superseded", previous=prepared["head"],
                                             reconciliation=reconciliation))
        records.extend([prepared, operational])
    return records, operational, public


def captured_bridge(*, bridge_kind="exact-public-recovery", bridge_count=1,
                    alter_bridge_candidate=False):
    """P captured prepared A; exactly one factual recovery R may bridge P->R->A."""
    public = frame_record(receipt(candidate(native=True), phase="prepared", previous=None))
    records = [public]
    bridge = public
    for index in range(bridge_count):
        value = candidate_of(anchor(public)["receipt"])
        if alter_bridge_candidate:
            value["artifact"]["sha256"] = "9" * 64
        reconciliation = None if bridge_kind is None else {
            "kind": bridge_kind, "effective_public_receipt_commit": public["head"],
            "run_id": None, "run_attempt": None, "job_id": None, "proof_sha256": "8" * 64}
        bridge = frame_record(receipt(value, previous=bridge["head"], reconciliation=reconciliation))
        records.append(bridge)
    public_value = anchor(public)["receipt"]
    base = {"receipt_commit": public["head"], "public_receipt_commit": public["head"],
            "registry_sha256": public_value["native"]["registry"]["sha256"], "sequence": 1}
    value = candidate("P", native=True, base=base, snapshots=[SNAPSHOT_A, SNAPSHOT_X])
    prepared = frame_record(receipt(value, phase="prepared", previous=bridge["head"]))
    reconciliation = {"kind": "unstarted-deploy", "effective_public_receipt_commit": public["head"],
                      "run_id": str(value["run"]["id"]), "run_attempt": "1",
                      "job_id": "701", "proof_sha256": "7" * 64}
    operational = frame_record(receipt(value, phase="superseded", previous=prepared["head"],
                                         reconciliation=reconciliation))
    return records + [prepared, operational], operational, public


def metadata_observation(records):
    """Inert response descriptors; their hashes do not attest HTTP provenance."""
    rows = []
    # Fixed repository-relative route grammar from the adopted R04 clarification.
    operational = records[-1]["head"] if records else None
    ref_body = (b'{"message":"Not Found"}\n' if operational is None else canonical({
        "ref": "refs/heads/" + BRANCH, "object": {"type": "commit", "sha": operational}}))
    main_body = canonical({"ref": "refs/heads/main", "object": {"type": "commit", "sha": CONSUMER}})
    for kind, locator, body in (
            ("receipt-ref", "git/ref/heads/" + BRANCH, ref_body),
            ("main-ref", "git/ref/heads/main", main_body),
            ("receipt-ref", "git/ref/heads/" + BRANCH, ref_body),
            ("main-ref", "git/ref/heads/main", main_body)):
        rows.append({"route_kind": kind, "locator": locator, "bytes": len(body),
                     "sha256": digest(body), "http_exchanges": 1})
    for row in sorted(records, key=lambda item: item["head"]):
        receipt_raw = base64.b64decode(row["receipt_b64"])
        blob = git_sha("blob", receipt_raw)
        for kind, locator, body in (
                ("receipt-commit", "git/commits/" + row["head"], canonical({
                    "sha": row["head"], **row["commit"], "tree": {"sha": row["commit"]["tree"]},
                    "parents": [{"sha": parent} for parent in row["commit"]["parents"]]})),
                ("receipt-tree", "git/trees/" + row["commit"]["tree"],
                 canonical({"sha": row["commit"]["tree"], "truncated": False,
                     "tree": [{"path": "latest.json", "mode": "100644", "type": "blob",
                               "sha": blob, "size": len(receipt_raw)}]})),
                ("receipt-blob", "git/blobs/" + blob, canonical({"sha": blob, "encoding": "base64",
                    "content": row["receipt_b64"], "size": len(receipt_raw)}))):
            rows.append({"route_kind": kind, "locator": locator, "bytes": len(body),
                         "sha256": digest(body), "http_exchanges": 1})
    return {"logical_calls": len(rows), "physical_exchanges": len(rows),
            "received_bytes": sum(row["bytes"] for row in rows), "elapsed_ms": 10,
            "responses": rows}


def public_observation(public):
    if public is None:
        paths = ("snapshots/index.json", "current-meta.json", *NATIVE_ABSENCE)
        rows = [{"path": path, "status": 404, "bytes": 0, "sha256": None}
                for path in sorted(paths)]
        return {"target_receipt_head": None, "target_candidate_sha256": None,
                "scope": "initial-absence5", "records": rows,
                "requests": 5, "received_bytes": 0, "elapsed_ms": 10}
    value = anchor(public)["receipt"]
    rows = [{"path": path, "status": 200, **copy.deepcopy(value[field])}
            for field, path in LEGACY_PATHS.items()]
    if value["schema_version"] == 1:
        rows.extend({"path": path, "status": 404, "bytes": 0, "sha256": None}
                    for path in NATIVE_ABSENCE)
        scope = "legacy3+native-absence3"
    else:
        registry_raw, _, _, assets = registry_fixture()
        rows.extend({"status": 200, **descriptor(path, raw)} for path, raw in assets.items())
        rows.extend({"status": 200, **copy.deepcopy(value["native"][field])}
                    for field in ("registry", "build_manifest"))
        rows.append({"status": 200, **descriptor("proof-tree.html", b"<html>fixture</html>\n")})
        scope = "current9"
    # New logical-candidate digest retains full v2 native fields, not receipt extras.
    return {"target_receipt_head": public["head"],
            "target_candidate_sha256": digest(canonical(candidate_of(value))),
            "scope": scope, "records": sorted(rows, key=lambda row: row["path"]),
            "requests": len(rows), "received_bytes": sum(row["bytes"] for row in rows),
            "elapsed_ms": 10}


def handoff(records=None, operational=None, public=None, *, registry_override=None):
    """Closed six-file pure handoff plus exact registry only for native-v1."""
    records = [] if records is None else copy.deepcopy(records)
    actual = None if operational is None else anchor(operational)
    effective = None if public is None else anchor(public)
    proof = {"schema_version": 1, "records": sorted(records, key=lambda row: row["head"])}
    native = effective is not None and effective["receipt"]["schema_version"] == 2
    registry_raw = (registry_fixture()[0] if registry_override is None else registry_override) if native else None
    base = {"schema_version": 1,
            "receipt_commit": None if actual is None else actual["head"],
            "public_receipt_commit": None if effective is None else effective["head"],
            "receipt_phase": None if actual is None else actual["receipt"]["phase"],
            "native_sequence": effective["receipt"]["native"]["sequence"] if native else 0,
            "registry_sha256": digest(registry_raw) if native else None,
            "mode": "initial" if effective is None else "native-v1" if native else "legacy-v1"}
    operation = "observe-prepared" if actual is not None and actual["receipt"]["phase"] == "prepared" else "none"
    qualification = {"schema_version": 1, "repository": REPOSITORY, "consumer_commit": CONSUMER,
                     "initial_receipt_head": base["receipt_commit"],
                     "final_receipt_head": base["receipt_commit"],
                     "initial_main_commit": CONSUMER, "final_main_commit": CONSUMER,
                     "operation": operation, "target_public_head": base["public_receipt_commit"],
                     "no_start": None, "public_observation": public_observation(public),
                     "metadata": metadata_observation(records)}
    pending = {"schema_version": 1, "operation": operation,
               "operational_head": base["receipt_commit"],
               "effective_public_head": base["public_receipt_commit"],
               "pending_prepared_head": base["receipt_commit"] if operation == "observe-prepared" else None,
               "consumer_commit": CONSUMER,
               "git_proof_sha256": digest(canonical(proof)),
               "qualification_sha256": digest(canonical(qualification)),
               "served_snapshot_ids": [] if effective is None else effective["receipt"]["snapshot_ids"],
               "unserved_snapshot_ids": []}
    files = {"anchor.json": canonical(actual), "public-anchor.json": canonical(effective),
             "base.json": canonical(base), "git-proof.json": canonical(proof),
             "pending-plan.json": canonical(pending), "qualification.json": canonical(qualification)}
    if native:
        files["registry.json"] = registry_raw
    expected = {"anchor": actual, "public_anchor": effective, "registry_raw": registry_raw,
                "base": base, "disposition": pending, "git_proof": proof}
    return files, actual, expected


def rewrite(files, name, change, *, refresh_hash=True):
    """Change one parsed fixture and rebind its outer hash to test inner checks."""
    value = json.loads(files[name])
    change(value)
    files[name] = canonical(value)
    if refresh_hash and name in ("qualification.json", "git-proof.json"):
        plan = json.loads(files["pending-plan.json"])
        field = "qualification_sha256" if name == "qualification.json" else "git_proof_sha256"
        plan[field] = digest(files[name])
        files["pending-plan.json"] = canonical(plan)


class CurrentSiteNativeBaseContract(unittest.TestCase):
    def shortDescription(self):
        return None

    def api(self):
        module_path = Path(__file__).resolve().parents[1] / "tools" / "current_site_native_registry.py"
        if not module_path.is_file():
            self.fail("missing Task2 exact Source file inside collected assertion: " + str(module_path))
        module_name = "_task2_current_site_native_registry_exact_source"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                self.fail("cannot load Task2 exact Source file inside collected assertion: " + str(module_path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                if sys.modules.get(module_name) is module:
                    del sys.modules[module_name]
                raise
        else:
            self.assertEqual(getattr(module, "__file__", None), str(module_path),
                             "cached Task2 module must come from the exact Source file")
        function = getattr(module, "validate_publication_base", None)
        self.assertTrue(callable(function), "missing Task2 callable: validate_publication_base")
        return function

    def test_initial_absence_returns_exact_six_key_verified_base(self):
        # Catches default-success output that omits the authenticated empty proof.
        validate = self.api()
        files, native_anchor, expected = handoff()
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(set(expected), {"anchor", "public_anchor", "registry_raw", "base", "disposition", "git_proof"})

    def test_legacy_public_anchor_needs_no_unrelated_historical_parent_record(self):
        # Catches unbounded recursion behind the final admitted public anchor.
        validate = self.api()
        public = frame_record(receipt(candidate()))
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(expected["base"]["mode"], "legacy-v1")

    def test_native_public_anchor_pins_exact_original_registry(self):
        # Catches accepting a native base without its original registry byte pin.
        validate = self.api()
        public = frame_record(receipt(candidate(native=True)))
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(expected["base"]["native_sequence"], 1)

    def test_prepared_chosen_public_anchor_has_explicit_observe_prepared_plan(self):
        # Catches silently representing a pending exact-public recovery as no action.
        validate = self.api()
        public = frame_record(receipt(candidate(native=True), phase="prepared", previous=None))
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(expected["disposition"]["operation"], "observe-prepared")
        self.assertEqual(expected["disposition"]["pending_prepared_head"], public["head"])

    def test_valid_superseded_prepared_public_chain_resolves_effective_legacy_base(self):
        # Catches resolving S directly through a free public pointer without P.
        validate = self.api()
        records, operational, public = superseded_chain()
        files, native_anchor, expected = handoff(records, operational, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertNotEqual(expected["anchor"]["head"], expected["public_anchor"]["head"])
        self.assertEqual(expected["disposition"]["served_snapshot_ids"], [SNAPSHOT_A])
        self.assertEqual(expected["disposition"]["unserved_snapshot_ids"], [])

    def test_valid_multi_link_superseded_chain_resolves_captured_bases(self):
        # Catches only handling one disposition instead of the reachable closure.
        validate = self.api()
        records, operational, public = superseded_chain(3, public_native=True)
        files, native_anchor, expected = handoff(records, operational, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(len(expected["git_proof"]["records"]), 7)

    def test_one_captured_base_exact_public_recovery_bridge_is_accepted(self):
        # Catches requiring P.actual_parent == captured H despite the exact R rule.
        validate = self.api()
        records, operational, public = captured_bridge()
        files, native_anchor, expected = handoff(records, operational, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(expected["public_anchor"]["receipt"]["phase"], "prepared")

    def test_existing_raw_plus_one_lf_commit_form_is_accepted(self):
        # Catches admitting only one of the existing helper's two Git forms.
        validate = self.api()
        public = frame_record(receipt(candidate()), extra_commit_lf=True)
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)

    def test_existing_reader_accepts_differently_worded_bounded_commit_message(self):
        # Catches tightening old read admission to the writer's one emitted text.
        validate = self.api()
        public = frame_record(receipt(candidate()), message_override="Historical metadata message\n")
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)

    def test_noncanonical_but_strict_original_receipt_bytes_are_preserved(self):
        # Catches regenerating originals from normalized JSON before Git framing.
        validate = self.api()
        value = receipt(candidate())
        raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n\n").encode("utf-8")
        public = frame_record(value, receipt_raw=raw)
        files, native_anchor, expected = handoff([public], public, public)
        self.assertEqual(validate(files, native_anchor=native_anchor), expected)
        self.assertEqual(base64.b64decode(expected["git_proof"]["records"][0]["receipt_b64"]), raw)

    def test_required_file_set_and_bytes_values_are_exact(self):
        # Catches arbitrary extra handoff inputs and coerced text/bytearray values.
        validate = self.api()
        original, native_anchor, _ = handoff()
        for name in sorted(BASE_FILES):
            bad = dict(original); del bad[name]
            with self.subTest(missing=name), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)
        for name, raw in (("extra.json", b"{}\n"), ("registry.json", b"{}\n")):
            bad = dict(original); bad[name] = raw
            with self.subTest(extra=name), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)
        for wrong in (original["base.json"].decode("ascii"), bytearray(original["base.json"]), None):
            bad = dict(original); bad["base.json"] = wrong
            with self.subTest(value_type=type(wrong).__name__), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_strict_json_rejects_duplicates_nonfinite_invalid_utf8_and_extra_document(self):
        # Catches permissive JSON decoding at the pure handoff boundary.
        validate = self.api()
        original, native_anchor, _ = handoff()
        raw = original["base.json"]
        for bad_raw in (raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
                        raw.replace(b'"native_sequence":0', b'"native_sequence":NaN'),
                        raw.replace(b'"native_sequence":0', b'"native_sequence":Infinity'),
                        b"\xff" + raw, raw + b"{}\n", b"[]\n", b"null\n"):
            bad = dict(original); bad["base.json"] = bad_raw
            with self.subTest(raw=bad_raw[:60]), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_closed_base_plan_qualification_and_proof_schemas(self):
        # Catches silently ignoring unrecognized or missing handoff fields.
        validate = self.api()
        original, native_anchor, _ = handoff()
        for name in ("base.json", "pending-plan.json", "qualification.json", "git-proof.json"):
            fields = list(json.loads(original[name]))
            for field in fields:
                bad = dict(original); rewrite(bad, name, lambda row, key=field: row.pop(key))
                with self.subTest(file=name, missing=field), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)
            bad = dict(original); rewrite(bad, name, lambda row: row.update(extra=True))
            with self.subTest(file=name, extra=True), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_base_fields_are_recomputed_from_authenticated_effective_anchor(self):
        # Catches trusting mode, heads, phase, sequence, or registry pins directly.
        validate = self.api()
        records, operational, public = superseded_chain(public_native=True)
        original, native_anchor, _ = handoff(records, operational, public)
        changes = [("receipt_commit", public["head"]), ("public_receipt_commit", operational["head"]),
                   ("receipt_phase", "observed"), ("native_sequence", 2),
                   ("registry_sha256", "9" * 64), ("mode", "legacy-v1"),
                   ("schema_version", True), ("native_sequence", True)]
        for key, value in changes:
            bad = dict(original); rewrite(bad, "base.json", lambda row, k=key, v=value: row.update({k: v}))
            with self.subTest(field=key, value=value), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_native_registry_is_required_pinned_and_not_replaced_by_normalization(self):
        # Catches missing registry bytes or accepting a semantically equal re-encoding.
        validate = self.api()
        public = frame_record(receipt(candidate(native=True)))
        original, native_anchor, _ = handoff([public], public, public)
        bad = dict(original); del bad["registry.json"]
        with self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)
        bad = dict(original); bad["registry.json"] += b"\n"
        with self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_pinned_registry_still_requires_strict_native_registry_admission(self):
        # Catches trusting a matching outer digest without validating registry authority.
        validate = self.api()
        registry = json.loads(registry_fixture()[0]); registry["custody"] = "known"
        invalid_raw = canonical(registry)
        value = candidate(native=True)
        value["native"]["registry"] = descriptor("native-graph/index.json", invalid_raw)
        public = frame_record(receipt(value))
        files, native_anchor, _ = handoff([public], public, public, registry_override=invalid_raw)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_native_anchor_must_equal_fresh_fixed_ref_anchor(self):
        # Catches treating an arbitrary caller head as the frozen operational head.
        validate = self.api()
        records, operational, public = superseded_chain()
        files, native_anchor, _ = handoff(records, operational, public)
        for wrong in (None, anchor(public), {"head": operational["head"]},
                      dict(native_anchor, head="9" * 40), dict(native_anchor, extra=True)):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError): validate(files, native_anchor=wrong)

    def test_anchor_objects_must_match_authenticated_original_receipts(self):
        # Catches trusting normalized private anchor content independently of Git.
        validate = self.api()
        records, operational, public = superseded_chain()
        original, native_anchor, _ = handoff(records, operational, public)
        for name in ("anchor.json", "public-anchor.json"):
            for mutate in (lambda row: row.update(head="9" * 40),
                           lambda row: row["receipt"].update(recorded_at="2026-10-24T00:00:00Z"),
                           lambda row: row.update(extra=True)):
                bad = dict(original); rewrite(bad, name, mutate)
                with self.subTest(file=name), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_missing_prepared_record_cannot_be_replaced_by_pointer_and_digest(self):
        # Catches S->A shortcut acceptance when S's actual prepared parent is absent.
        validate = self.api()
        records, operational, public = superseded_chain()
        missing = [row for row in records if row["head"] != operational["commit"]["parents"][0]]
        files, native_anchor, _ = handoff(missing, operational, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_missing_intermediate_captured_superseded_base_is_rejected(self):
        # Catches resolving a multi-link chain without each captured-base witness.
        validate = self.api()
        records, operational, public = superseded_chain(3)
        missing = [row for row in records if row["head"] != records[2]["head"]]
        files, native_anchor, _ = handoff(missing, operational, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_original_receipt_lf_tamper_breaks_git_identity(self):
        # Catches hashing normalized JSON rather than the exact original blob bytes.
        validate = self.api()
        records, operational, public = superseded_chain()
        files, native_anchor, _ = handoff(records, operational, public)
        def tamper(proof):
            row = proof["records"][0]
            raw = base64.b64decode(row["receipt_b64"])
            row["receipt_b64"] = base64.b64encode(raw[:-1]).decode("ascii")
        rewrite(files, "git-proof.json", tamper)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_original_receipt_must_have_strict_closed_v1_or_v2_schema(self):
        # Catches framing a real Git object then accepting invalid receipt content.
        validate = self.api()
        for change in (lambda row: row.update(extra=True),
                       lambda row: row.update(scientific_effect="PASS"),
                       lambda row: row.update(schema_version=True),
                       lambda row: row.update(phase="superseded")):
            value = receipt(candidate()); change(value)
            public = frame_record(value)
            files, native_anchor, _ = handoff([public], public, public)
            with self.subTest(change=change), self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_noncanonical_or_whitespace_base64_is_rejected(self):
        # Catches a decoder that accepts whitespace or nonstandard Base64 spelling.
        validate = self.api()
        records, operational, public = superseded_chain()
        original, native_anchor, _ = handoff(records, operational, public)
        for change in (lambda raw: raw + "\n", lambda raw: " " + raw,
                       lambda raw: raw[:-1] + "_", lambda raw: raw + "="):
            bad = dict(original)
            rewrite(bad, "git-proof.json", lambda proof, f=change: proof["records"][0].update(
                receipt_b64=f(proof["records"][0]["receipt_b64"])))
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_actual_git_parent_must_equal_original_receipt_previous_pointer(self):
        # Catches checking only receipt pointers while ignoring the framed parent.
        validate = self.api()
        value = receipt(candidate())
        public = frame_record(value, parents_override=["9" * 40])
        files, native_anchor, _ = handoff([public], public, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_one_file_100644_latest_json_tree_is_derived_not_trusted(self):
        # Catches trusting an arbitrary commit.tree instead of deriving its bytes.
        validate = self.api()
        public = frame_record(receipt(candidate()), tree_override="9" * 40)
        files, native_anchor, _ = handoff([public], public, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_receipt_commit_bot_timestamp_and_bounded_message_grammar_is_enforced(self):
        # Catches accepting arbitrary Git authors or unbounded/NUL commit messages.
        validate = self.api()
        for kwargs in ({"person_override": dict(BOT, name="Another writer", date=DATE)},
                       {"message_override": ""}, {"message_override": "x" * 257},
                       {"message_override": "Unrelated\0metadata"}):
            public = frame_record(receipt(candidate()), **kwargs)
            files, native_anchor, _ = handoff([public], public, public)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)
        public = frame_record(receipt(candidate()))
        files, native_anchor, _ = handoff([public], public, public)
        rewrite(files, "git-proof.json", lambda proof: proof["records"][0]["commit"]["author"].update(
            date="2026-10-25T00:00:01Z"))
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_proof_records_and_commit_people_have_closed_schemas(self):
        # Catches accepting caller annotations as part of an authenticated record.
        validate = self.api()
        public = frame_record(receipt(candidate()))
        original, native_anchor, _ = handoff([public], public, public)
        for target in ("record", "commit", "author", "committer"):
            bad = dict(original)
            def mutate(proof, which=target):
                value = proof["records"][0]
                if which != "record": value = value["commit"]
                if which in ("author", "committer"): value = value[which]
                value["extra"] = True
            rewrite(bad, "git-proof.json", mutate)
            with self.subTest(target=target), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_reordered_or_duplicate_proof_heads_are_rejected(self):
        # Catches normalizing caller ordering or silently deduplicating records.
        validate = self.api()
        records, operational, public = superseded_chain()
        original, native_anchor, _ = handoff(records, operational, public)
        for change in (lambda proof: proof["records"].reverse(),
                       lambda proof: proof["records"].insert(1, copy.deepcopy(proof["records"][0]))):
            bad = dict(original); rewrite(bad, "git-proof.json", change)
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_spliced_valid_unrelated_commit_is_rejected_as_unused(self):
        # Catches treating every cryptographically valid supplied vertex as useful.
        validate = self.api()
        records, operational, public = superseded_chain()
        unrelated = frame_record(receipt(candidate("Z"), previous=None, phase="prepared"))
        files, native_anchor, _ = handoff(records + [unrelated], operational, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_disconnected_valid_public_record_cannot_substitute_for_reachable_public_base(self):
        # Catches selecting a valid but unrelated public anchor from the proof bag.
        validate = self.api()
        records, operational, public = superseded_chain()
        unrelated = frame_record(receipt(candidate("Z")))
        files, native_anchor, _ = handoff(records + [unrelated], operational, unrelated)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_superseded_record_conserves_entire_prepared_candidate(self):
        # Catches matching only source or native pins while losing unserved Math X.
        validate = self.api()
        records, operational, public = superseded_chain()
        for change in (lambda row: row.update(snapshot_ids=[SNAPSHOT_A]),
                       lambda row: row["artifact"].update(sha256="9" * 64),
                       lambda row: row["native"]["binding"].update(receipt_sha256="9" * 64)):
            value = anchor(operational)["receipt"]; change(value)
            replacement = frame_record(value)
            changed = [replacement if row["head"] == operational["head"] else row for row in records]
            files, native_anchor, _ = handoff(changed, replacement, public)
            with self.subTest(change=change), self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_supersession_requires_prepared_parent_and_exact_public_pointer(self):
        # Catches unsupported S parent phases and substituted effective pointers.
        validate = self.api()
        records, operational, public = superseded_chain()
        value = anchor(operational)["receipt"]
        value["reconciliation"]["effective_public_receipt_commit"] = "9" * 40
        replacement = frame_record(value)
        files, native_anchor, _ = handoff([*records[:-1], replacement], replacement, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)
        value = anchor(operational)["receipt"]
        value["previous_receipt_commit"] = public["head"]
        replacement = frame_record(value)
        files, native_anchor, _ = handoff([public, replacement], replacement, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_missing_captured_base_recovery_bridge_record_is_rejected(self):
        # Catches accepting a prepared actual parent pointer without original R.
        validate = self.api()
        records, operational, public = captured_bridge()
        files, native_anchor, _ = handoff([row for row in records if row["head"] != records[1]["head"]],
                                         operational, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_recovery_bridge_must_have_correct_kind_and_conserved_candidate(self):
        # Catches treating arbitrary observed descendants as the one allowed R.
        validate = self.api()
        for kwargs in ({"bridge_kind": None}, {"alter_bridge_candidate": True}):
            records, operational, public = captured_bridge(**kwargs)
            files, native_anchor, _ = handoff(records, operational, public)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_two_reconciliation_children_do_not_expand_the_single_bridge_rule(self):
        # Catches allowing arbitrary descendant distance between captured H and P.
        validate = self.api()
        records, operational, public = captured_bridge(bridge_count=2)
        files, native_anchor, _ = handoff(records, operational, public)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_more_than_sixteen_distinct_material_edges_is_rejected(self):
        # Five complete pairs create at least 19 material pairs with 11 records.
        validate = self.api()
        records, operational, public = superseded_chain(5)
        files, native_anchor, _ = handoff(records, operational, public)
        self.assertEqual(len(records), 11)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_more_than_seventeen_original_records_is_rejected(self):
        # A genuine 19-record closure exceeds both caps; no isolated branch claim.
        validate = self.api()
        records, operational, public = superseded_chain(9)
        files, native_anchor, _ = handoff(records, operational, public)
        self.assertEqual(len(records), 19)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_cyclic_pointer_tamper_is_rejected_without_claiming_valid_git_cycle(self):
        # A self-parent receipt cannot be rehashed to the same Git head in this fixture.
        validate = self.api()
        records, operational, public = superseded_chain()
        files, native_anchor, _ = handoff(records, operational, public)
        def mutate(proof):
            row = next(row for row in proof["records"] if row["head"] == operational["head"])
            value = json.loads(base64.b64decode(row["receipt_b64"]))
            value["previous_receipt_commit"] = row["head"]
            row["receipt_b64"] = base64.b64encode(canonical(value)).decode("ascii")
            row["commit"]["parents"] = [row["head"]]
        rewrite(files, "git-proof.json", mutate)
        with self.assertRaises(ValueError): validate(files, native_anchor=native_anchor)

    def test_private_proof_and_qualification_hashes_pin_original_file_bytes(self):
        # Catches accepting a semantically equal re-encoding without its exact hash.
        validate = self.api()
        files, native_anchor, _ = handoff()
        for name in ("git-proof.json", "qualification.json"):
            bad = dict(files); bad[name] += b"\n"
            with self.subTest(file=name), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_qualification_requires_unchanged_operational_and_main_fences(self):
        # Catches ignoring the frozen resolver's initial/final identity agreement.
        validate = self.api()
        records, operational, public = superseded_chain()
        original, native_anchor, _ = handoff(records, operational, public)
        changes = [("repository", "other/repository"), ("consumer_commit", "9" * 40),
                   ("initial_receipt_head", public["head"]), ("final_receipt_head", "9" * 40),
                   ("initial_main_commit", "9" * 40), ("final_main_commit", "9" * 40),
                   ("target_public_head", operational["head"]), ("operation", "observe-prepared")]
        for key, value in changes:
            bad = dict(original); rewrite(bad, "qualification.json", lambda row, k=key, v=value: row.update({k: v}))
            with self.subTest(field=key), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_pending_plan_is_recomputed_including_served_and_unserved_inventory(self):
        # Catches borrowing unserved X into served history or inventing free eligibility.
        validate = self.api()
        records, operational, public = superseded_chain()
        original, native_anchor, _ = handoff(records, operational, public)
        changes = [("operational_head", public["head"]), ("effective_public_head", operational["head"]),
                   ("pending_prepared_head", operational["head"]), ("consumer_commit", "9" * 40),
                   ("served_snapshot_ids", [SNAPSHOT_A, SNAPSHOT_X]),
                   ("unserved_snapshot_ids", [SNAPSHOT_X]), ("operation", "supersede-unstarted")]
        for key, value in changes:
            bad = dict(original); rewrite(bad, "pending-plan.json", lambda row, k=key, v=value: row.update({k: v}))
            with self.subTest(field=key), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)
        bad = dict(original); rewrite(bad, "pending-plan.json", lambda row: row.update(eligible=True))
        with self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_active_supersession_cannot_use_missing_boolean_or_free_no_start_eligibility(self):
        # Catches replacing the complete current qualification with a caller boolean.
        validate = self.api()
        records, _, public = superseded_chain()
        prepared = records[1]
        original, native_anchor, _ = handoff([public, prepared], prepared, public)
        for no_start in (None, True, {"eligible": True}):
            bad = dict(original)
            rewrite(bad, "qualification.json", lambda row, value=no_start: row.update(
                operation="supersede-unstarted", no_start=value))
            rewrite(bad, "pending-plan.json", lambda row: row.update(
                operation="supersede-unstarted", pending_prepared_head=prepared["head"],
                unserved_snapshot_ids=[SNAPSHOT_X]))
            with self.subTest(no_start=no_start), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_public_observation_requires_complete_sorted_unique_target_closure(self):
        # Catches accepting missing, duplicated, reordered or foreign final paths.
        validate = self.api()
        public = frame_record(receipt(candidate(native=True)))
        original, native_anchor, _ = handoff([public], public, public)
        changes = [lambda rows: rows.pop(), lambda rows: rows.reverse(),
                   lambda rows: rows.append(copy.deepcopy(rows[0])),
                   lambda rows: rows[0].update(path="foreign.json"),
                   lambda rows: rows[0].update(status=403),
                   lambda rows: rows[0].update(bytes=True)]
        for change in changes:
            bad = dict(original)
            rewrite(bad, "qualification.json", lambda row, f=change: f(row["public_observation"]["records"]))
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_migration_and_initial_absence_require_exact_404_zero_null_rows(self):
        # Catches treating denied/existing native pointers as migration absence.
        validate = self.api()
        public = frame_record(receipt(candidate()))
        for original, native_anchor, _ in (handoff(), handoff([public], public, public)):
            for change in (lambda row: row.update(status=200, bytes=1, sha256="1" * 64),
                           lambda row: row.update(status=403), lambda row: row.update(bytes=1),
                           lambda row: row.update(sha256="1" * 64)):
                bad = dict(original)
                def mutate(qualification, f=change):
                    row = next(row for row in qualification["public_observation"]["records"] if row["status"] == 404)
                    f(row)
                rewrite(bad, "qualification.json", mutate)
                with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_public_observation_target_and_known_descriptor_pins_must_join(self):
        # Catches accepting another target candidate or altered known asset hashes.
        validate = self.api()
        public = frame_record(receipt(candidate(native=True)))
        original, native_anchor, _ = handoff([public], public, public)
        for change in (lambda row: row.update(target_receipt_head="9" * 40),
                       lambda row: row.update(target_candidate_sha256="9" * 64),
                       lambda row: row.update(scope="legacy3"),
                       lambda row: next(item for item in row["records"] if item["path"] == "current.json").update(sha256="9" * 64)):
            bad = dict(original); rewrite(bad, "qualification.json", lambda row, f=change: f(row["public_observation"]))
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_recorded_metadata_and_public_counters_respect_declared_caps(self):
        # Catches bool coercion, phase budget resets, and counters below held rows.
        validate = self.api()
        original, native_anchor, _ = handoff()
        changes = [("metadata", "logical_calls", 65), ("metadata", "physical_exchanges", 65),
                   ("metadata", "received_bytes", 16 * 1024 * 1024 + 1),
                   ("metadata", "elapsed_ms", 120001), ("metadata", "logical_calls", True),
                   ("metadata", "physical_exchanges", 0), ("metadata", "received_bytes", 0),
                   ("public_observation", "requests", 202),
                   ("public_observation", "received_bytes", 128 * 1024 * 1024 + 1),
                   ("public_observation", "elapsed_ms", 660001),
                   ("public_observation", "requests", True), ("public_observation", "requests", 4)]
        for section, key, value in changes:
            bad = dict(original)
            rewrite(bad, "qualification.json", lambda row, s=section, k=key, v=value: row[s].update({k: v}))
            with self.subTest(section=section, field=key, value=value), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_metadata_descriptor_rejects_url_capability_and_unbounded_response(self):
        # Catches a private handoff that carries arbitrary origins or unsafe locators.
        validate = self.api()
        original, native_anchor, _ = handoff()
        for change in (lambda row: row.update(locator="https://example.invalid/capability"),
                       lambda row: row.update(route_kind="arbitrary-network"),
                       lambda row: row.update(bytes=1024 * 1024 + 1),
                       lambda row: row.update(http_exchanges=True),
                       lambda row: row.update(sha256="not-a-digest"),
                       lambda row: row.update(extra=True)):
            bad = dict(original)
            rewrite(bad, "qualification.json", lambda row, f=change: f(row["metadata"]["responses"][0]))
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bad, native_anchor=native_anchor)

    def test_pure_validation_performs_no_network_subprocess_or_filesystem_reads(self):
        # Catches an implementation that silently reopens historical service/IO units.
        validate = self.api()
        records, operational, public = superseded_chain()
        files, native_anchor, expected = handoff(records, operational, public)
        denied = AssertionError("pure validate_publication_base attempted IO")
        targets = ("builtins.open", "io.open", "os.open", "os.stat", "os.lstat",
                   "pathlib.Path.read_bytes", "pathlib.Path.write_bytes",
                   "socket.create_connection", "socket.socket", "urllib.request.urlopen",
                   "urllib.request.OpenerDirector.open", "subprocess.run", "subprocess.Popen")
        with ExitStack() as guards:
            for target in targets:
                guards.enter_context(patch(target, side_effect=denied))
            self.assertEqual(validate(files, native_anchor=native_anchor), expected)


if __name__ == "__main__":
    unittest.main()
