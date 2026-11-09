#!/usr/bin/env python3
"""Q-9006 helper phase: real-record freshness, never scientific admission.

No CLI is implemented in this phase. The evaluator accepts an observation
boundary for offline contract controls; only run_live constructs the default
production HTTPS observer. Unit results are not actual live qualification.
"""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import urlsplit

import node_status_refresh as refresh
from node_status_refresh import ContentObserver
import validate_node_status as validator

RECORD_NAMES = (
    "eng.d0-packet-allowlist-fix.json",
    "eng.hard-gate.json",
    "math.lifetime-remainder.json",
    "math.side24-coefficient.json",
)
METADATA = ("architecture/node-status.schema.json", "docs/site/current.json")
IDENTITY_FIELDS = {"repository", "checked_commit", "run_id", "run_attempt", "python_optimize"}
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
POSITIVE = re.compile(r"[1-9][0-9]*\Z")


class LiveCheckError(ValueError):
    """A required input or actual freshness comparison could not be accepted."""


def require(condition, message):
    if not condition:
        raise LiveCheckError(message)


def _identity(value):
    require(isinstance(value, dict) and set(value) == IDENTITY_FIELDS,
            "invalid live-check identity fields")
    require(value["repository"] == "d6g8k5htny-coder/main", "wrong source repository")
    require(isinstance(value["checked_commit"], str)
            and HEX40.fullmatch(value["checked_commit"]), "immutable checked commit required")
    for key in ("run_id", "run_attempt"):
        require(isinstance(value[key], str) and POSITIVE.fullmatch(value[key]),
                "positive native " + key + " required")
    require(type(value["python_optimize"]) is int and value["python_optimize"] in (0, 1),
            "interpreter mode must be integer 0 or 1")
    return dict(value)


def _path(root, relative):
    path = root
    for part in Path(relative).parts:
        require(part not in ("", ".", ".."), "noncanonical selected input path")
        path = path / part
        require(not path.is_symlink(), "symlink selected input refused")
    return path


def _read(root, relative, limit):
    path = _path(root, relative)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
            "selected input is not a bounded regular file: " + relative)
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(tuple(getattr(before, name) for name in fields)
            == tuple(getattr(after, name) for name in fields)
            and len(raw) == before.st_size, "selected input changed during read: " + relative)
    return raw


def _capture(root):
    require(root.is_dir() and not root.is_symlink(), "selected root is missing or symlink")
    directory = _path(root, "records/node_status")
    require(directory.is_dir(), "mandatory records directory missing")
    require({path.name for path in directory.iterdir()} == set(RECORD_NAMES),
            "selected four-record input set differs")
    captured = {
        "records/node_status/" + name: _read(root, "records/node_status/" + name, validator.MAX_BYTES)
        for name in RECORD_NAMES
    }
    for relative in METADATA:
        limit = 2_000_000 if relative.endswith("current.json") else validator.MAX_BYTES
        captured[relative] = _read(root, relative, limit)
    return captured


def _validated_records(captured):
    # Production validation observes the same captured bytes, not a second
    # mutable read from the selected checkout. Only this owned temporary is written.
    with tempfile.TemporaryDirectory(prefix="node-status-live-input-") as temporary:
        root = Path(temporary)
        for relative, raw in captured.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        problems = validator.validate_directory(root, root / "records/node_status")
        require(not problems, "production record/schema validation refused: " + "; ".join(problems))
        return [
            validator.load_json(root / "records/node_status" / name)
            for name in RECORD_NAMES
        ]


def _observation_report(observer, requested):
    observations = getattr(observer, "observations", None)
    heads = getattr(observer, "heads", None)
    require(isinstance(observations, list) and isinstance(heads, dict),
            "observer did not retain observations and heads")
    require(all(isinstance(row, dict) and isinstance(row.get("url"), str)
                for row in observations), "malformed source observation")
    require(Counter(row["url"] for row in observations) == Counter(requested),
            "source observation counters or URL multiplicities differ")
    git_repositories = set()
    for row in observations:
        url = row["url"]
        if validator.PINNED_URL.fullmatch(url):
            parts = urlsplit(url).path.split("/")
            repository, kind = parts[2], parts[3]
            git_repositories.add(repository)
            head = heads.get(repository)
            require(isinstance(head, str) and HEX40.fullmatch(head)
                    and row.get("head") == head, "observation differs from captured repository head")
            before, after = row.get("pinned_object"), row.get("current_object")
            require(isinstance(before, (tuple, list)) and isinstance(after, (tuple, list))
                    and len(before) == len(after) == 3, "incomplete Git object observation")
            require(before[0] == after[0] == kind
                    and before[1] == after[1]
                    and before[1] in (("040000",) if kind == "tree" else ("100644", "100755"))
                    and isinstance(before[2], str) and HEX40.fullmatch(before[2])
                    and tuple(before) == tuple(after), "cited Git object is not unchanged")
        else:
            require(isinstance(row.get("updated_at"), str)
                    and validator.OBSERVED.fullmatch(row["updated_at"]),
                    "discussion timestamp observation missing")
    require(set(heads) == git_repositories, "captured repository head coverage differs")
    # Freeze returned evidence values, so a later caller change cannot rewrite this report.
    return copy.deepcopy(heads), copy.deepcopy(observations)


def evaluate_fresh_records(root, *, observer, identity):
    """Check real captured inputs using the supplied source-observation boundary.

    A supplied observer or identity is not proof of actual HTTPS or hosted
    execution. The separately qualified live entrypoint must establish those.
    """
    try:
        identity = _identity(identity)
        root = Path(root).absolute()
        captured = _capture(root)
        records = _validated_records(captured)
        fresh = [record for record in records if record["status"] == "fresh"]
        require(fresh, "no real declared-fresh candidate; live coverage is unexercised")
        require(callable(observer) and getattr(observer, "observations", None) == [],
                "a fresh observation boundary is required for this mode")
        requested = []
        checked = []
        for record in fresh:
            requested.extend(source["url"] for source in record["sources"])
            result = refresh.refresh_record(record, observer)
            require(result.get("status") == "fresh" and not result.get("refresh_error"),
                    "declared-fresh source is stale, unknown or unavailable: " + record["node_id"])
            require(result.get("source_rev") == record["source_rev"]
                    and result.get("sources") == record["sources"],
                    "immutable source pins changed during observation")
            checked.append(record["node_id"])
        require(len(checked) == len(fresh), "fresh cohort was not completely checked")
        heads, observations = _observation_report(observer, requested)
        require(_capture(root) == captured, "selected record/schema/graph bytes changed during observation")
        return {
            "schema": "cbz.node-status-live-check.v1",
            "scope": "NODE_RECORD_FRESHNESS_ONLY",
            "records_total": len(records),
            "fresh_candidates": len(fresh),
            "fresh_checked": len(checked),
            "fresh_nodes": sorted(checked),
            "nonfresh_records": sorted(
                ({"node_id": record["node_id"], "declared_status": record["status"]}
                 for record in records if record["status"] != "fresh"),
                key=lambda row: row["node_id"],
            ),
            "input_sha256": {
                name: hashlib.sha256(captured["records/node_status/" + name]).hexdigest()
                for name in RECORD_NAMES
            },
            "schema_sha256": hashlib.sha256(captured[METADATA[0]]).hexdigest(),
            "graph_sha256": hashlib.sha256(captured[METADATA[1]]).hexdigest(),
            "identity": identity,
            "observed_heads": heads,
            "source_observation_count": len(observations),
            "source_observations": observations,
            "scientific_effect": "NONE",
            "cbz_green": False,
        }
    except LiveCheckError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise LiveCheckError("freshness check refused: " + str(error)) from error


def run_live(root, *, identity):
    """Require authentication and construct exactly one default HTTPS observer.

    There is deliberately no observer/API/transcript parameter on this wrapper,
    no anonymous fallback and no retry. This helper phase has no CLI yet.
    """
    token = os.environ.get("GITHUB_TOKEN")
    require(isinstance(token, str) and bool(token.strip()), "GITHUB_TOKEN is required")
    return evaluate_fresh_records(root, observer=ContentObserver(), identity=identity)
