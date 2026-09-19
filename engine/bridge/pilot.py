"""Bounded, operator-mediated bridge pilot; no service or credential holder.

The executing session supplies freshly observed owner-boundary facts and
actual command results. This adapter checks their consistency and byte
identity, and constructs existing v1 receipts. It does not authenticate the
caller, fetch Drive, execute commands, enforce process limits, or establish
authorization, review, scientific truth, or acceptance. Those boundaries
must not be replaced by a caller-written JSON assertion in production.

Only the existing append-only store writes receipts. EXECUTED and RECORDED
are immutable delivery stages in separate stores, never an in-place edit.
The records role must upload/read back all outputs as well as the receipt.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from engine.bridge import run_receipt as rr, work_order as wo
from engine.bridge.common import is_commit, is_drive_id, parse_timestamp


class PilotRefused(ValueError):
    """Missing, stale, inconsistent or out-of-scope pilot evidence."""


@dataclass(frozen=True)
class BoundarySnapshot:
    """Facts observed by the trusted session, not authenticated by this module."""
    repository_id: int
    commit_sha: str
    tree_sha: str
    dirty_worktree: bool
    principal_id: str
    claim_event_id: str
    claim_lease_until_utc: str
    claim_active: bool
    output_head: str | None
    observed_utc: str


def preflight(order: dict, boundary: BoundarySnapshot,
              source_bytes: Mapping[str, bytes], now: str) -> None:
    """Refuse drift before the host executes; performs no authorization grant.

    Five minutes is the maximum age of an observation for this manual pilot.
    Freshness is not atomicity: concurrent remote writes remain an owner-side
    concern. Sources are the freshly fetched bytes under the order's exact
    extraction rules; native revisions must be checked by the fetching role.
    """
    problems = wo.validate_work_order(order)
    if problems:
        raise PilotRefused("invalid work order: " + "; ".join(problems))
    if order["status"] != wo.STATUS_EXECUTABLE:
        raise PilotRefused("order is not executable")
    auth = order["authorization"]
    if auth["verification_status"] != wo.VERIFIED_BY_OWNER_BOUNDARY:
        raise PilotRefused("owner boundary has not verified authorization")
    current, observed = parse_timestamp(now), parse_timestamp(boundary.observed_utc)
    lease = parse_timestamp(boundary.claim_lease_until_utc)
    if current is None or observed is None or lease is None:
        raise PilotRefused("invalid observation or lease timestamp")
    if not 0 <= (current - observed).total_seconds() <= 300:
        raise PilotRefused("stale or future boundary observation")
    if lease <= current or boundary.claim_active is not True:
        raise PilotRefused("claim expired or inactive")
    claim = order["claim"]
    if (boundary.claim_event_id != claim["work_events_event_id"] or
            boundary.claim_lease_until_utc != claim["lease_until_utc"]):
        raise PilotRefused("claim identity or lease changed")
    repo = order["repository"]
    if (type(boundary.repository_id) is not int or
            boundary.repository_id != repo["id"] or
            boundary.commit_sha != repo["base_commit_sha"] or
            not is_commit(boundary.tree_sha)):
        raise PilotRefused("repository, commit or tree identity mismatch")
    if boundary.dirty_worktree is not False:
        raise PilotRefused("tested checkout is not clean")
    if boundary.principal_id not in auth["approved_principal_ids"]:
        raise PilotRefused("principal is outside the owner-approved list")
    if boundary.output_head != order["expected_old_output_head"]:
        raise PilotRefused("stale output head")
    sources = order["governing_sources"]
    ids = [s["drive_id"] for s in sources]
    if len(set(ids)) != len(ids) or set(source_bytes) != set(ids):
        raise PilotRefused("source set is incomplete, duplicated or unexpected")
    if auth["authorizing_record_drive_id"] not in ids:
        raise PilotRefused("authorizing record must be a byte-pinned source")
    for source in sources:
        data = source_bytes[source["drive_id"]]
        if (not isinstance(data, bytes) or len(data) != source["bytes"] or
                sha256(data).hexdigest() != source["sha256"]):
            raise PilotRefused("governing source bytes changed: " + source["drive_id"])


def executed_receipt(order: dict, boundary: BoundarySnapshot,
                     source_bytes: Mapping[str, bytes], *, run_id: str,
                     actor: dict, environment: str, commands: list[str],
                     exit_codes: list[int], start_utc: str, end_utc: str,
                     outputs: Mapping[str, bytes], negative_controls: list[str],
                     coverage: str, dirty_after: bool) -> dict:
    """Record host-run acceptance checks, one count per command exit code.

    The host must preflight before running and retain stdout/stderr. This
    rechecks input consistency at the declared start. Counts are command-level
    checks, not the test-case totals in a pytest log. Every output is hashed;
    output paths are logical capsule paths checked against the order scope.
    Caller-supplied times/results are observations, not execution proof.
    """
    preflight(order, boundary, source_bytes, start_utc)
    if type(dirty_after) is not bool:
        raise PilotRefused("dirty_after must be observed as a boolean")
    start, end = parse_timestamp(start_utc), parse_timestamp(end_utc)
    lease = parse_timestamp(boundary.claim_lease_until_utc)
    if end is None or end < start or end >= lease:
        raise PilotRefused("execution end is invalid or outside the lease")
    if (end - start).total_seconds() > order["limits"]["wall_time_seconds"]:
        raise PilotRefused("execution exceeded the order's wall-time budget")
    if actor.get("authenticated_principal_id") != boundary.principal_id:
        raise PilotRefused("receipt principal differs from observed principal")
    if (not commands or len(commands) != len(exit_codes) or
            any(type(code) is not int for code in exit_codes)):
        raise PilotRefused("one integer exit code is required per actual command")
    if any(not isinstance(data, bytes) for data in outputs.values()):
        raise PilotRefused("output bodies must be bytes")
    receipt = rr.finalize({
        "schema": rr.SCHEMA, "record_kind": "RECORD", "status": rr.EXECUTED,
        "task_id": order["task_id"], "work_order_digest": order["work_order_digest"],
        "run_id": run_id, "actor": deepcopy(actor),
        "execution": {"repository_id": boundary.repository_id,
                      "commit_sha": boundary.commit_sha, "tree_sha": boundary.tree_sha,
                      "dirty_worktree": dirty_after, "environment_identity": environment,
                      "commands": list(commands), "exit_codes": list(exit_codes),
                      "start_utc": start_utc, "end_utc": end_utc},
        "verification": {"tests_passed": sum(c == 0 for c in exit_codes),
                         "tests_failed": sum(c != 0 for c in exit_codes),
                         "negative_controls": list(negative_controls),
                         "coverage": "Command-level acceptance checks. " + coverage,
                         "excluded_scope": ["Scientific correctness and promotion",
                                            "Independent review and owner acceptance",
                                            "Production identity and process isolation"]},
        "outputs": [{"path": path, "bytes": len(data), "sha256": sha256(data).hexdigest()}
                    for path, data in sorted(outputs.items())],
        "review": {"technical_verdict": rr.NOT_REVIEWED,
                   "authorship_exposure": "Same executing session authored the pilot; zero independent review credit.",
                   "independence_credit": 0, "organizational_independence_record_id": None},
        "drive_return": {"delivery_id": None, "receipt_file_id": None,
                         "expected_old_head": order["expected_old_output_head"],
                         "readback_sha256": None, "readback_utc": None,
                         "acceptance_status": rr.ACCEPTANCE_NOT},
        "accepted_by_drive_record": None, "scientific_status_change": "UNCHANGED",
        "does_not_establish": "This operator-mediated pilot records supplied execution observations and exact byte identities. It does not establish independent review, scientific truth, acceptance, authenticated service deployment, or enforcement of process or repository access limits.",
    })
    problems = rr.validate_run_receipt(receipt) + rr.check_receipt_against_order(receipt, order)
    if problems:
        raise PilotRefused("; ".join(problems))
    return receipt


def recorded_receipt(executed: dict, *, receipt_file_id: str, delivery_id: str,
                     receipt_readback: bytes, output_readbacks: Mapping[str, bytes],
                     readback_utc: str, observed_output_head: str | None) -> dict:
    """Construct a distinct archival stage after exact raw-byte readback.

    The records role supplies real connector downloads. The receipt_file_id
    names the immutable EXECUTED receipt, whose bytes were read back. This
    function does not overwrite that receipt or accept the work. Store the
    returned RECORDED stage separately; same-key content changes in one store
    are conflicts and must continue to be held by store_receipt.
    """
    problems = rr.validate_run_receipt(executed)
    if problems or executed.get("status") != rr.EXECUTED:
        raise PilotRefused("a valid EXECUTED receipt is required")
    if not is_drive_id(receipt_file_id) or not is_drive_id(delivery_id):
        raise PilotRefused("Drive receipt and delivery ids are required")
    if receipt_readback != rr.to_json(executed).encode("utf-8"):
        raise PilotRefused("receipt readback differs from uploaded canonical bytes")
    when = parse_timestamp(readback_utc)
    if when is None or when < parse_timestamp(executed["execution"]["end_utc"]):
        raise PilotRefused("readback time precedes execution or is invalid")
    if observed_output_head != executed["drive_return"]["expected_old_head"]:
        raise PilotRefused("stale output head at archival boundary")
    outputs = executed["outputs"]
    if len({o["path"] for o in outputs}) != len(outputs):
        raise PilotRefused("duplicate output paths")
    if set(output_readbacks) != {o["path"] for o in outputs}:
        raise PilotRefused("every output requires a readback; no extra output is accepted")
    for output in outputs:
        data = output_readbacks[output["path"]]
        if (not isinstance(data, bytes) or len(data) != output["bytes"] or
                sha256(data).hexdigest() != output["sha256"]):
            raise PilotRefused("output readback differs: " + output["path"])
    result = deepcopy(executed)
    result["status"] = rr.RECORDED
    result["drive_return"].update({"delivery_id": delivery_id,
                                   "receipt_file_id": receipt_file_id,
                                   "readback_sha256": sha256(receipt_readback).hexdigest(),
                                   "readback_utc": readback_utc})
    result = rr.finalize(result)
    if rr.validate_run_receipt(result):
        raise PilotRefused("constructed archival stage is invalid")
    return result
