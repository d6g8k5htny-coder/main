#!/usr/bin/env python3
"""Durable prepared/observed Pages receipts on one fixed metadata-only Git branch.

No downloaded code is executed, no source branch is changed, and no force update
is allowed. A prepared receipt survives failed or unknown deployment outcomes.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from current_site_data import parse_json

REPOSITORY = "d6g8k5htny-coder/main"
RECEIPT_BRANCH = "codex/site-publication-receipts"
REF_NAME = "refs/heads/" + RECEIPT_BRANCH
REF_READ = "/git/ref/heads/" + RECEIPT_BRANCH
REF_WRITE = "/git/refs/heads/" + RECEIPT_BRANCH
API_ORIGIN = "https://api.github.com/repos/" + REPOSITORY
PUBLIC_ORIGIN = "https://d6g8k5htny-coder.github.io/main/site/"
WORKFLOW = ".github/workflows/current-site.yml"
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_PUBLIC_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOTS = 512
REF_READBACK_ATTEMPTS = 4
REF_READBACK_SECONDS = 10
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PATHS = {"manifest": "snapshots/index.json", "metadata": "current-meta.json", "payload": "current.json"}
CANDIDATE_FIELDS = {"schema_version", "scientific_effect", "repository", "receipt_branch", "source", "run",
                    "artifact", "manifest", "metadata", "payload", "snapshot_ids"}
BOT = {"name": "Current research site publication", "email": "41898282+github-actions[bot]@users.noreply.github.com"}


def closed(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("invalid closed " + label + " schema")


def hex_value(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError("invalid " + label)
    return value


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def git_sha(kind, raw):
    return hashlib.sha1(kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def encode(data):
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value):
        raise ValueError("invalid receipt timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("invalid " + label)


def validate_candidate(candidate):
    closed(candidate, CANDIDATE_FIELDS, "publication candidate")
    if (type(candidate["schema_version"]) is not int or candidate["schema_version"] != 1
            or candidate["scientific_effect"] != "NONE" or candidate["repository"] != REPOSITORY
            or candidate["receipt_branch"] != RECEIPT_BRANCH):
        raise ValueError("invalid publication authority")
    closed(candidate["source"], ("commit", "ref", "workflow"), "source")
    hex_value(candidate["source"]["commit"], HEX40, "source commit")
    if candidate["source"]["ref"] != "refs/heads/main" or candidate["source"]["workflow"] != WORKFLOW:
        raise ValueError("invalid publication source")
    closed(candidate["run"], ("id", "attempt"), "run")
    integer(candidate["run"]["id"], 1, 2**63-1, "run id")
    integer(candidate["run"]["attempt"], 1, 100_000, "run attempt")
    closed(candidate["artifact"], ("sha256", "files", "bytes"), "artifact")
    hex_value(candidate["artifact"]["sha256"], HEX64, "artifact fingerprint")
    integer(candidate["artifact"]["files"], 1, 1_000_000, "artifact file count")
    integer(candidate["artifact"]["bytes"], 1, 1024**3, "artifact bytes")
    for field in PATHS:
        closed(candidate[field], ("bytes", "sha256"), field)
        integer(candidate[field]["bytes"], 1, MAX_PUBLIC_BYTES, field + " bytes")
        hex_value(candidate[field]["sha256"], HEX64, field + " digest")
    snapshots = candidate["snapshot_ids"]
    if not isinstance(snapshots, list) or not 1 <= len(snapshots) <= MAX_SNAPSHOTS:
        raise ValueError("invalid retained snapshot inventory")
    for snapshot in snapshots:
        hex_value(snapshot, HEX64, "snapshot id")
    if snapshots != sorted(set(snapshots)):
        raise ValueError("snapshot ids must be unique and sorted")
    return candidate


def candidate_of(receipt):
    return {field: receipt[field] for field in CANDIDATE_FIELDS}


def validate_receipt(receipt):
    closed(receipt, CANDIDATE_FIELDS | {"phase", "recorded_at", "recorded_by", "previous_receipt_commit"}, "publication receipt")
    validate_candidate(candidate_of(receipt))
    if receipt["phase"] not in ("prepared", "observed"):
        raise ValueError("invalid publication phase")
    timestamp(receipt["recorded_at"])
    closed(receipt["recorded_by"], ("source_commit", "run_id", "run_attempt"), "receipt observer")
    hex_value(receipt["recorded_by"]["source_commit"], HEX40, "receipt observer source")
    integer(receipt["recorded_by"]["run_id"], 1, 2**63-1, "receipt observer run")
    integer(receipt["recorded_by"]["run_attempt"], 1, 100_000, "receipt observer attempt")
    if receipt["previous_receipt_commit"] is not None:
        hex_value(receipt["previous_receipt_commit"], HEX40, "previous receipt commit")
    if receipt["phase"] == "observed" and receipt["previous_receipt_commit"] is None:
        raise ValueError("observed receipt requires its prepared parent")
    return receipt


def source_guard(candidate, context):
    validate_candidate(candidate)
    closed(context, ("repository", "ref", "commit", "run_id", "run_attempt"), "execution context")
    hex_value(context["commit"], HEX40, "execution source commit")
    integer(context["run_id"], 1, 2**63-1, "execution run id")
    integer(context["run_attempt"], 1, 100_000, "execution run attempt")
    if (context["repository"] != REPOSITORY or context["ref"] != "refs/heads/main"
            or candidate["source"]["commit"] != context["commit"]
            or candidate["run"]["id"] != context["run_id"]
            or candidate["run"]["attempt"] > context["run_attempt"]):
        raise ValueError("publication source/run differs from qualified execution")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("publication redirect refused")


def tree_bytes(blob):
    return b"100644 latest.json\0" + bytes.fromhex(hex_value(blob, HEX40, "receipt blob"))


def commit_bytes(tree, parents, message, author, committer):
    hex_value(tree, HEX40, "receipt tree")
    if not isinstance(parents, list) or len(parents) > 1:
        raise ValueError("receipt commit must have at most one parent")
    lines = ["tree " + tree, *("parent " + hex_value(parent, HEX40, "receipt parent") for parent in parents)]
    for label, person in (("author", author), ("committer", committer)):
        closed(person, ("name", "email", "date"), "receipt " + label)
        if {key: person[key] for key in BOT} != BOT:
            raise ValueError("invalid receipt author identity")
        epoch = int(timestamp(person["date"]).timestamp())
        lines.append(f"{label} {BOT['name']} <{BOT['email']}> {epoch} +0000")
    if not isinstance(message, str) or not 1 <= len(message) <= 256 or "\0" in message:
        raise ValueError("invalid receipt commit message")
    return ("\n".join(lines) + "\n\n" + message).encode()


def guard_request(method, path, body):
    immutable = re.fullmatch(r"/git/(commits|trees|blobs)/[0-9a-f]{40}", path if isinstance(path, str) else "")
    if method == "GET" and (path == REF_READ or immutable) and body is None:
        return
    if method == "POST" and path == "/git/blobs":
        closed(body, ("encoding", "content"), "blob write")
        if body["encoding"] != "base64" or not isinstance(body["content"], str) or len(body["content"]) > MAX_RECEIPT_BYTES * 2:
            raise ValueError("invalid receipt blob write")
        raw = base64.b64decode(body["content"], validate=True)
        if len(raw) > MAX_RECEIPT_BYTES:
            raise ValueError("receipt blob capacity exceeded")
        validate_receipt(parse_json(raw))
        return
    if method == "POST" and path == "/git/trees":
        closed(body, ("tree",), "tree write")
        if not isinstance(body["tree"], list) or len(body["tree"]) != 1:
            raise ValueError("metadata tree must contain latest.json only")
        row = body["tree"][0]
        closed(row, ("path", "mode", "type", "sha"), "tree entry")
        if row["path"] != "latest.json" or row["mode"] != "100644" or row["type"] != "blob":
            raise ValueError("nonmetadata tree write refused")
        hex_value(row["sha"], HEX40, "receipt blob")
        return
    if method == "POST" and path == "/git/commits":
        closed(body, ("message", "tree", "parents", "author", "committer"), "commit write")
        commit_bytes(body["tree"], body["parents"], body["message"], body["author"], body["committer"])
        return
    if method == "POST" and path == "/git/refs":
        closed(body, ("ref", "sha"), "branch creation")
        if body["ref"] != REF_NAME:
            raise ValueError("nonmetadata branch creation refused")
        hex_value(body["sha"], HEX40, "receipt commit")
        return
    if method == "PATCH" and path == REF_WRITE:
        closed(body, ("sha", "force"), "branch advance")
        if body["force"] is not False:
            raise ValueError("force update refused")
        hex_value(body["sha"], HEX40, "receipt commit")
        return
    raise ValueError("nonmetadata GitHub route refused")


class GitHubAPI:
    def __init__(self, *, token=None, opener=None):
        self.token = os.environ.get("GITHUB_TOKEN") if token is None else token
        self.opener = opener if opener is not None else urllib.request.build_opener(NoRedirect())
        self.requests = self.total_bytes = 0

    def request(self, method, path, body=None):
        guard_request(method, path, body)
        if self.requests >= 64:
            raise ValueError("publication API request bound exceeded")
        self.requests += 1
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "current-site-publication-receipt",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        raw_body = encode(body) if body is not None else None
        if raw_body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(API_ORIGIN + path, data=raw_body, headers=headers, method=method)
        with self.opener.open(request, timeout=20) as response:
            if response.geturl() != request.full_url or response.status not in (200, 201):
                raise ValueError("publication API response identity mismatch")
            raw = response.read(MAX_RECEIPT_BYTES + 1)
        self.total_bytes += len(raw)
        if len(raw) > MAX_RECEIPT_BYTES or self.total_bytes > 16 * MAX_RECEIPT_BYTES:
            raise ValueError("publication API byte bound exceeded")
        result = parse_json(raw)
        if not isinstance(result, dict):
            raise ValueError("publication API object response required")
        return result


class PublicReader:
    def __init__(self, *, opener=None, cache_key="", expected_sizes=None, deadline=None, clock=time.monotonic):
        self.opener = opener if opener is not None else urllib.request.build_opener(NoRedirect())
        self.cache_key = hex_value(cache_key, HEX64, "public cache key") if cache_key else ""
        self.expected_sizes = expected_sizes or {}
        self.deadline = deadline
        self.clock = clock
        self.requests = self.total_bytes = 0

    def get_bytes(self, path):
        if path not in PATHS.values() or self.requests >= 201:
            raise ValueError("publication public request bound or path refused")
        self.requests += 1
        url = PUBLIC_ORIGIN + path + ("?publication=" + self.cache_key if self.cache_key else "")
        request = urllib.request.Request(url, headers={"User-Agent": "current-site-publication-readback", "Cache-Control": "no-cache",
                                                      "Pragma": "no-cache", "Accept-Encoding": "identity"})
        remaining = 20 if self.deadline is None else min(20, self.deadline-self.clock())
        if remaining <= 0:
            raise ValueError("publication public readback deadline exceeded")
        with self.opener.open(request, timeout=remaining) as response:
            if response.geturl() != url or response.status != 200 or response.headers.get("Content-Encoding", "identity") not in ("", "identity"):
                raise ValueError("publication public response identity mismatch")
            limit = self.expected_sizes.get(path, MAX_PUBLIC_BYTES)
            integer(limit, 1, MAX_PUBLIC_BYTES, "expected public byte count")
            raw = response.read(limit + 1)
        self.total_bytes += len(raw)
        if len(raw) > limit or self.total_bytes > 128 * 1024 * 1024:
            raise ValueError("publication public byte bound exceeded")
        return raw


def verified_commit(api, commit):
    data = api.request("GET", "/git/commits/" + hex_value(commit, HEX40, "anchor commit"))
    if data.get("sha") != commit or not isinstance(data.get("tree"), dict) or not isinstance(data.get("parents"), list) or len(data["parents"]) > 1:
        raise ValueError("invalid anchor commit response")
    parents = [row.get("sha") for row in data["parents"] if isinstance(row, dict)]
    if len(parents) != len(data["parents"]):
        raise ValueError("invalid anchor commit parent")
    people = []
    for field in ("author", "committer"):
        person = data.get(field)
        if not isinstance(person, dict):
            raise ValueError("invalid anchor author")
        # GitHub normalizes UTC commit dates to the same ISO form.
        people.append({key: person.get(key) for key in ("name", "email", "date")})
    raw = commit_bytes(data["tree"].get("sha"), parents, data.get("message"), *people)
    if commit not in {git_sha("commit", raw), git_sha("commit", raw + b"\n")}:
        raise ValueError("anchor Git commit integrity mismatch")
    return data, parents


def read_ref_head(api, *, allow_bootstrap=False):
    try:
        ref = api.request("GET", REF_READ)
    except urllib.error.HTTPError as error:
        if error.code == 404 and allow_bootstrap:
            return None
        raise ValueError("publication receipt branch unavailable; bootstrap requires explicit initial 404") from error
    if ref.get("ref") != REF_NAME or not isinstance(ref.get("object"), dict) or ref["object"].get("type") != "commit":
        raise ValueError("invalid publication anchor ref")
    return hex_value(ref["object"].get("sha"), HEX40, "anchor ref commit")


def read_anchor_at(api, head):
    commit, parents = verified_commit(api, head)
    tree_sha = commit["tree"]["sha"]
    tree = api.request("GET", "/git/trees/" + tree_sha)
    rows = tree.get("tree")
    if tree.get("sha") != tree_sha or tree.get("truncated") is not False or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("invalid metadata-only anchor tree")
    row = rows[0]
    if not isinstance(row, dict) or row.get("path") != "latest.json" or row.get("mode") != "100644" or row.get("type") != "blob":
        raise ValueError("anchor tree must contain latest.json only")
    blob_sha = hex_value(row.get("sha"), HEX40, "anchor blob")
    if git_sha("tree", tree_bytes(blob_sha)) != tree_sha:
        raise ValueError("anchor Git tree integrity mismatch")
    blob = api.request("GET", "/git/blobs/" + blob_sha)
    if (blob.get("sha") != blob_sha or blob.get("encoding") != "base64" or not isinstance(blob.get("content"), str)
            or len(blob["content"]) > MAX_RECEIPT_BYTES * 2):
        raise ValueError("invalid receipt blob response")
    raw = base64.b64decode(blob["content"].replace("\n", ""), validate=True)
    if len(raw) > MAX_RECEIPT_BYTES or type(blob.get("size")) is not int or blob["size"] != len(raw) or git_sha("blob", raw) != blob_sha:
        raise ValueError("anchor Git blob integrity mismatch")
    receipt = validate_receipt(parse_json(raw))
    if parents != ([receipt["previous_receipt_commit"]] if receipt["previous_receipt_commit"] else []):
        raise ValueError("anchor receipt differs from actual Git parent")
    return {"head": head, "receipt": receipt}


def read_anchor(api=None, *, allow_bootstrap=False):
    api = api if api is not None else GitHubAPI()
    head = read_ref_head(api, allow_bootstrap=allow_bootstrap)
    return None if head is None else read_anchor_at(api, head)


def check_public(candidate, raw):
    validate_candidate(candidate)
    for field, path in PATHS.items():
        content = raw[path]
        if len(content) != candidate[field]["bytes"] or sha256(content) != candidate[field]["sha256"]:
            raise ValueError("public " + field + " digest/byte count differs from qualified artifact")
    manifest, meta = parse_json(raw[PATHS["manifest"]]), parse_json(raw[PATHS["metadata"]])
    if not isinstance(manifest, dict) or not isinstance(manifest.get("archives"), list) or len(manifest["archives"]) > MAX_SNAPSHOTS:
        raise ValueError("invalid publication manifest inventory")
    ids = sorted(row.get("id") for row in manifest["archives"] if isinstance(row, dict) and isinstance(row.get("id"), str))
    if len(ids) != len(manifest["archives"]) or ids != candidate["snapshot_ids"]:
        raise ValueError("publication manifest snapshot inventory mismatch")
    if (not isinstance(meta, dict) or meta.get("archive") != manifest.get("current")
            or meta.get("history_manifest_sha256") != candidate["manifest"]["sha256"]
            or meta.get("sha256") != candidate["payload"]["sha256"] or type(meta.get("bytes")) is not int
            or meta["bytes"] != candidate["payload"]["bytes"]):
        raise ValueError("publication manifest/metadata/payload pointers disagree")
    parse_json(raw[PATHS["payload"]])


def public_readback(receipt, *, reader=None, attempts=67, pause=time.sleep, clock=time.monotonic):
    candidate = candidate_of(receipt) if "phase" in receipt else validate_candidate(receipt)
    integer(attempts, 1, 67, "public readback attempts")
    deadline = clock() + 660
    reader = reader if reader is not None else PublicReader(cache_key=candidate["manifest"]["sha256"], expected_sizes={path:candidate[field]["bytes"] for field,path in PATHS.items()}, deadline=deadline, clock=clock)
    for attempt in range(attempts):
        try:
            raw = {}
            for field, path in PATHS.items():
                if clock() >= deadline:
                    raise ValueError("public readback deadline exceeded")
                raw[path] = reader.get_bytes(path)
                if clock() >= deadline:
                    raise ValueError("public readback deadline exceeded")
                if len(raw[path]) != candidate[field]["bytes"] or sha256(raw[path]) != candidate[field]["sha256"]:
                    raise ValueError("public " + field + " differs from anchored digest")
            check_public(candidate, raw)
            return raw
        except (ValueError, OSError) as error:
            if attempt + 1 == attempts or clock() >= deadline:
                raise ValueError("public readback failed; prepared publication recovery needed, anchor retained") from error
            pause(min(10, max(0, deadline - clock())))


def append_receipt(api, receipt, old_head, *, read_attempts=REF_READBACK_ATTEMPTS,
                   read_pause=time.sleep, read_clock=time.monotonic):
    raw = encode(validate_receipt(receipt))
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("receipt capacity exceeded")
    blob = git_sha("blob", raw)
    response = api.request("POST", "/git/blobs", {"encoding": "base64", "content": base64.b64encode(raw).decode()})
    if response.get("sha") != blob:
        raise ValueError("created receipt blob identity mismatch")
    tree = git_sha("tree", tree_bytes(blob))
    response = api.request("POST", "/git/trees", {"tree": [{"path": "latest.json", "mode": "100644", "type": "blob", "sha": blob}]})
    if response.get("sha") != tree:
        raise ValueError("created receipt tree identity mismatch")
    parents = [old_head] if old_head else []
    person = {**BOT, "date": receipt["recorded_at"]}
    message = f"Site publication {receipt['phase']} run {receipt['run']['id']} attempt {receipt['run']['attempt']}\n"
    response = api.request("POST", "/git/commits", {"tree": tree, "parents": parents, "message": message, "author": person, "committer": person})
    commit = hex_value(response.get("sha"), HEX40, "created receipt commit")
    actual, actual_parents = verified_commit(api, commit)
    if actual["tree"]["sha"] != tree or actual_parents != parents:
        raise ValueError("created receipt commit differs from requested immutable tree/parent")
    if old_head:
        api.request("PATCH", REF_WRITE, {"sha": commit, "force": False})
        integer(read_attempts, 1, REF_READBACK_ATTEMPTS, "receipt ref readback attempts")
        deadline = read_clock() + REF_READBACK_SECONDS
        for attempt in range(read_attempts):
            visible = read_ref_head(api)
            if visible == commit:
                observed = read_anchor_at(api, commit)
                if observed != {"head": commit, "receipt": receipt}:
                    raise ValueError("receipt target differs from immutable requested receipt")
                return observed
            if visible != old_head:
                raise ValueError("receipt branch advanced to a third-party head; publication race")
            if attempt + 1 == read_attempts or read_clock() >= deadline:
                raise ValueError("receipt target visibility readback exhausted; publication recovery needed")
            read_pause(min(1, max(0, deadline - read_clock())))
    else:
        api.request("POST", "/git/refs", {"ref": REF_NAME, "sha": commit})
    observed = read_anchor_at(api, commit)
    if observed != {"head": commit, "receipt": receipt}:
        raise ValueError("receipt branch changed during write; publication recovery needed")
    return observed


def prepare_publication(candidate, *, api=None, context, allow_bootstrap=False, reader=None, now=None, attempts=67):
    source_guard(candidate, context)
    api = api if api is not None else GitHubAPI()
    anchor = read_anchor(api, allow_bootstrap=allow_bootstrap)
    if anchor and candidate_of(anchor["receipt"]) == candidate:
        return anchor
    if anchor is None:
        require_initial_absence(reader if reader is not None else PublicReader())
    if anchor:
        previous = anchor["receipt"]
        if not set(previous["snapshot_ids"]) <= set(candidate["snapshot_ids"]):
            raise ValueError("publication snapshot inventory regresses below persistent anchor")
        if previous["phase"] == "prepared":
            public_readback(previous, reader=reader, attempts=attempts)
            recovered = {**candidate_of(previous), "phase": "observed", "recorded_at": now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "recorded_by": observer(context), "previous_receipt_commit": anchor["head"]}
            anchor = append_receipt(api, recovered, anchor["head"])
    receipt = {**candidate, "phase": "prepared", "recorded_at": now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "recorded_by": observer(context),
               "previous_receipt_commit": anchor["head"] if anchor else None}
    return append_receipt(api, receipt, receipt["previous_receipt_commit"])


def observer(context):
    return {"source_commit": context["commit"], "run_id": context["run_id"], "run_attempt": context["run_attempt"]}


def observe_publication(candidate, prepared, *, api=None, context, reader=None, now=None, attempts=67):
    source_guard(candidate, context)
    closed(prepared, ("head", "receipt"), "prepared state")
    validate_receipt(prepared["receipt"])
    if candidate_of(prepared["receipt"]) != candidate:
        raise ValueError("qualified artifact differs from prepared receipt")
    api = api if api is not None else GitHubAPI()
    anchor = read_anchor(api)
    if (anchor["receipt"]["phase"] == "observed" and candidate_of(anchor["receipt"]) == candidate
            and (anchor == prepared or anchor["receipt"]["previous_receipt_commit"] == prepared["head"])):
        return anchor
    if anchor != prepared or prepared["receipt"]["phase"] != "prepared":
        raise ValueError("publication receipt branch advanced; no observed receipt inferred")
    public_readback(prepared["receipt"], reader=reader, attempts=attempts)
    receipt = {**candidate, "phase": "observed", "recorded_at": now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "recorded_by": observer(context),
               "previous_receipt_commit": prepared["head"]}
    return append_receipt(api, receipt, prepared["head"])


def require_initial_absence(reader):
    for path in ("snapshots/index.json", "current-meta.json"):
        try:
            reader.get_bytes(path)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise ValueError("initial public publication state unavailable; bootstrap refused") from error
        raise ValueError("public history exists without a persistent receipt; bootstrap refused")


def restore_publication(site, *, api=None, reader=None, history_reader=None, allow_bootstrap=False, attempts=67):
    anchor = read_anchor(api, allow_bootstrap=allow_bootstrap)
    if anchor is None:
        require_initial_absence(reader if reader is not None else PublicReader())
        from current_site_history import restore_history
        return restore_history(site, reader=history_reader)
    public_readback(anchor["receipt"], reader=reader, attempts=attempts)
    from current_site_history import restore_history
    return restore_history(site, reader=history_reader, expected_manifest_sha=anchor["receipt"]["manifest"]["sha256"])


def safe_read(path, maximum=MAX_PUBLIC_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("unsafe or oversized local publication input")
    return path.read_bytes()


def atomic(path, raw):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlinked publication output refused")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path); temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def artifact_fingerprint(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("unsafe assembled artifact")
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    total = 0
    tree = hashlib.sha256(b"current-site-generated-browser-input-v1\0")
    for path in files:
        name = path.relative_to(root).as_posix().encode(); size = path.stat().st_size
        total += size
        if total > 1024**3 or len(files) > 1_000_000:
            raise ValueError("assembled artifact capacity exceeded")
        tree.update(len(name).to_bytes(8, "big")); tree.update(name); tree.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                tree.update(chunk)
    return {"sha256": tree.hexdigest(), "files": len(files), "bytes": total}


def package_metadata(site, artifact_input, browser_report, output, *, context):
    site = Path(site)
    raw = {path: safe_read(site / path) for path in PATHS.values()}
    inputs = parse_json(safe_read(artifact_input, MAX_RECEIPT_BYTES))
    report = parse_json(safe_read(browser_report, MAX_RECEIPT_BYTES))
    if not isinstance(inputs, dict) or not isinstance(report, dict):
        raise ValueError("qualified artifact evidence required")
    artifact = artifact_fingerprint(site.parent)
    if (inputs.get("scientific_effect") != "NONE" or inputs.get("checked_commit") != context["commit"]
            or inputs.get("assembled_docs_sha256") != artifact["sha256"] or inputs.get("assembled_docs_files") != artifact["files"]
            or inputs.get("assembled_docs_bytes") != artifact["bytes"] or inputs.get("source_payload_sha256") != sha256(raw["current.json"])):
        raise ValueError("assembled artifact differs from qualified browser input")
    if (report.get("scientific_effect") != "NONE" or report.get("checked_commit") != context["commit"]
            or report.get("source_payload_sha256") != sha256(raw["current.json"]) or report.get("error")
            or not isinstance(report.get("cases"), list) or not report["cases"]
            or len(report["cases"]) > 128 or any(not isinstance(case, dict) or case.get("passed") is not True for case in report["cases"])):
        raise ValueError("qualified generated-artifact browser admission missing")
    manifest = parse_json(raw["snapshots/index.json"])
    from current_site_history import validate_manifest
    validate_manifest(manifest)
    candidate = {"schema_version": 1, "scientific_effect": "NONE", "repository": REPOSITORY, "receipt_branch": RECEIPT_BRANCH,
                 "source": {"commit": context["commit"], "ref": context["ref"], "workflow": WORKFLOW},
                 "run": {"id": context["run_id"], "attempt": context["run_attempt"]}, "artifact": artifact,
                 **{field: {"bytes": len(raw[path]), "sha256": sha256(raw[path])} for field, path in PATHS.items()},
                 "snapshot_ids": sorted(entry["id"] for entry in manifest["archives"])}
    source_guard(candidate, context); check_public(candidate, raw)
    output = Path(output)
    for path, content in raw.items():
        atomic(output / path, content)
    atomic(output / "candidate.json", encode(candidate))
    return candidate


def load_metadata(directory):
    directory = Path(directory)
    candidate = validate_candidate(parse_json(safe_read(directory / "candidate.json", MAX_RECEIPT_BYTES)))
    check_public(candidate, {path: safe_read(directory / path) for path in PATHS.values()})
    return candidate


def execution_context():
    if os.environ.get("GITHUB_EVENT_NAME") not in ("push", "schedule", "workflow_dispatch"):
        raise ValueError("publication requires a main publication workflow event")
    context = {"repository": os.environ.get("GITHUB_REPOSITORY"), "ref": os.environ.get("GITHUB_REF"),
               "commit": os.environ.get("GITHUB_SHA"), "run_id": int(os.environ.get("GITHUB_RUN_ID", "0")),
               "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "0"))}
    if context["repository"] != REPOSITORY or context["ref"] != "refs/heads/main":
        raise ValueError("publication execution repository/ref refused")
    hex_value(context["commit"], HEX40, "checked source commit")
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != context["commit"]:
        raise ValueError("publication tool checkout differs from tested source commit")
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("restore", "package", "prepare", "observe"))
    parser.add_argument("--site", type=Path)
    parser.add_argument("--artifact-input", type=Path)
    parser.add_argument("--browser-report", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--allow-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    try:
        context = execution_context()
        if args.action == "restore":
            if args.site is None: raise ValueError("restore requires --site")
            result = restore_publication(args.site, allow_bootstrap=args.allow_bootstrap)
        elif args.action == "package":
            if None in (args.site, args.artifact_input, args.browser_report, args.output): raise ValueError("package inputs missing")
            result = package_metadata(args.site, args.artifact_input, args.browser_report, args.output, context=context)
        else:
            if args.metadata is None or args.prepared is None: raise ValueError("publication metadata/prepared inputs missing")
            candidate = load_metadata(args.metadata)
            if args.action == "prepare":
                result = prepare_publication(candidate, context=context, allow_bootstrap=args.allow_bootstrap)
                atomic(args.prepared, encode(result))
            else:
                prepared = parse_json(safe_read(args.prepared, MAX_RECEIPT_BYTES))
                result = observe_publication(candidate, prepared, context=context)
        print(json.dumps({"action": args.action, "scientific_effect": "NONE", "receipt_commit": result.get("head"),
                          "phase": result.get("receipt", {}).get("phase"), "snapshots": len(result.get("receipt", result).get("snapshot_ids", []))}))
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.exit(1, f"publication custody refused: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
