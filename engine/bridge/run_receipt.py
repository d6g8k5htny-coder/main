#!/usr/bin/env python3
"""The run-receipt record, ``q0.bridge.run_receipt/v1``, its validator and
its append-only, content-addressed store.

A **run receipt** is the return record of one attempt at one work order: who
actually ran (the authenticated principal, and separately what the run
*declared* itself to be), exactly which commit and tree, in what environment,
with which commands and exit codes, what the tests and negative controls
reported, which outputs came out with which digests, what a reviewer said
about it, and how far it got back to Drive. It follows the conventions of
``engine/receipt.py`` (``q0.engine.receipt/v1``): closed vocabularies at
every level, a body digest, exclusive-creation writes, and a mandatory
``does_not_establish``. It ports the DG-EXEC handoff of 2026-09-18, whose
disposition is PROPOSED EXECUTION CONTRACT / NOT DEPLOYED; nothing here
deploys it.

THE STATE MODEL
---------------
``status`` is one of six words, each meaning exactly this and no more:

``NOT_RUN``
    Nothing executed. A template is not a receipt; a NOT_RUN receipt is a
    receipt that says so.
``CANNOT_VERIFY``
    Something may have run, but the evidence needed to say what is missing.
    Missing evidence is NOT_RUN or CANNOT_VERIFY, **never PASS**.
``PREPARED``
    The work order resolved and the execution capsule was assembled; no
    command ran.
``EXECUTED``
    Commands ran at a named commit. Requires ``repository_id``,
    ``commit_sha``, ``tree_sha``, ``dirty_worktree`` (a boolean),
    ``environment_identity``, ``commands`` (each a real command, none a
    placeholder or a no-op), ``exit_codes`` (one per command), start and end
    (end not before start), test counts that count at least one test, at
    least one negative control that is a statement, a coverage statement
    and an authenticated principal. A blank, a placeholder (``n/a``,
    ``none``, ``unknown``, ``:``) or a single character in any of those is
    refused: a receipt that ran nothing and says so in shape only is not an
    execution. An execution can succeed while archival is pending.
``RECORDED``
    The receipt and its evidence reached Drive AND were read back
    (``drive_return.readback_sha256`` / ``readback_utc``). A delivery without
    a readback is not recorded.
``ACCEPTED``
    An owner-side acceptance transaction happened in Drive. **This
    repository cannot verify an acceptance.** A receipt claiming ACCEPTED is
    refused unless ``accepted_by_drive_record`` names a Drive-id-shaped
    record (id and SHA-256); with one, the claim is transcribed unverified
    and the checker prints it as a NOTE. Whether that record exists and
    accepted anything is checked at the owner boundary, not here.

``scientific_status_change`` admits exactly one value, ``UNCHANGED``. Any
other value is refused: FW-NO-RECEIPT-PROMOTION (``tools/claims_check.py``)
says a receipt is a record that something ran, not that something is true,
and the same rule applies to a receipt that went through Drive and back.

``review.technical_verdict`` ``PASS_TECHNICAL`` requires an executed status,
``tests_passed >= 1``, ``tests_failed == 0``, every exit code zero, a clean
worktree (``dirty_worktree: false`` -- a reviewer cannot pass a tree that is
not the named commit) and an authorship exposure that is a statement.

``review.independence_credit`` is 0 unless ``organizational_independence_
record_id`` names a record: a Drive-id-shaped id or a ``reviews/`` record id
(``REV-...``), never a placeholder, never anything containing
"same-provider", and never on a ``NOT_REVIEWED`` verdict (a credit on a review
that did not happen). Even then it is transcribed, not awarded (CLAUDE.md
rule 6): the independence-requiring gate is not moved by a receipt.

AGAINST ITS ORDER
-----------------
:func:`check_receipt_against_order` ties a receipt to the order its digest
names: same ``task_id``, same ``repository_id``, every executed command
present verbatim in the order's ``allowed_commands``, every output path
within the order's ``allowed_paths`` and never on a policy or protected
surface, and no executed status against an order that is not
``EXECUTABLE`` and ``VERIFIED_BY_OWNER_BOUNDARY`` (a NOT_VERIFIED order is
PREPARED-only; a receipt that says something ran under it records a run the
contract did not permit, and the checker fails on it). Agreement is still
not evidence: a receipt that agrees with its order is a consistent record,
not a true one.

THE IDEMPOTENCY KEY
-------------------
``idempotency_key = sha256(canonical_json([task_id, work_order_digest,
tested_commit, run_id]))`` where ``tested_commit`` is ``execution.commit_sha``
(``null`` for a receipt that names none). The receipt file is named by this
key. Changing any of the four inputs changes the key; the validator
recomputes it and refuses a mismatch.

APPEND-ONLY, STRUCTURALLY
-------------------------
:func:`store_receipt` creates files with ``open(path, "x")`` and nothing
else. A second delivery for the same key with identical content writes
nothing and reports ``IDENTICAL_RETRY``. A second delivery with *different*
content is held beside the first as a ``q0.bridge.held_delivery/v1`` wrapper
named ``<key>.NEEDS_RECONCILIATION.<held-digest-prefix>.json``, preserving
the conflicting record verbatim inside it; the first receipt is never
replaced, and reconciliation is an owner-side task the wrapper does not do.
There is no code path in this module that opens an existing file for
writing, truncates, renames or deletes. The store writes to exactly one
place inside this repository, :data:`RECEIPTS_ROOT`; any other destination
must lie entirely outside the repository (tests use temporary directories),
and every root under :data:`FORBIDDEN_WRITE_ROOTS` -- including
``engine/bridge/orders/`` -- is refused by name as well: the receipt writer
structurally cannot manufacture a work order. Git history is the second
line: ``tools/bridge_check.py`` fails on any receipt, order or example that
was rewritten or removed in any reachable commit.

WHAT A RECEIPT DOES NOT ESTABLISH
---------------------------------
That anything is true. A receipt with ``tests_failed: 0`` and a
``PASS_TECHNICAL`` verdict records that a program exited zero and a reviewer
said the object holds at review scope. It is not a proof, not a certificate,
not independence (``review.independence_credit`` must be 0 unless an
organizational-independence record id is given, and even then it is
transcribed, not awarded), not acceptance, and not a movement of any claim,
premise or obligation. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND`` (hi side),
``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE`` and both Pieces of
``D3-LEMMA-RN-UNIF`` are OPEN, and no receipt this module can store says
otherwise.

Standard library only. Python 3.11.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.bridge.common import (
    PLACEHOLDERS, body_digest, canonical_json, check_does_not_establish, check_evidence_string,
    check_keys, is_commit, is_drive_id, is_int, is_nonempty_str, is_sha256,
    is_str_list, is_timestamp, load_json_strict, normalize_scope_path,
    parse_timestamp, sha256_of, utc_now,
)
from engine.bridge.work_order import (
    STATUS_EXECUTABLE, VERIFIED_BY_OWNER_BOUNDARY, path_within_scope,
    policy_surfaces_touched, protected_surfaces_touched,
)

__all__ = [
    "SCHEMA", "HELD_SCHEMA", "RECORD_KINDS", "STATUSES", "EXECUTED_STATUSES",
    "TECHNICAL_VERDICTS", "ACCEPTANCE_STATUSES", "SCIENTIFIC_STATUS_CHANGE",
    "DISPOSITION_NEEDS_RECONCILIATION", "RECEIPTS_ROOT", "FORBIDDEN_WRITE_ROOTS",
    "REVIEW_RECORD_ID_RE",
    "ReceiptRefused", "ForbiddenDestination", "StoreResult",
    "idempotency_key", "canonical_body", "receipt_body_digest", "finalize",
    "validate_run_receipt", "run_receipt_notes", "check_receipt_against_order",
    "validate_held_delivery", "held_file_name", "parse_receipt_file_name",
    "store_receipt", "load_receipt", "to_json",
]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECEIPTS_ROOT = os.path.join(ROOT, "engine", "bridge", "receipts")

#: Destinations this module refuses to write under, whatever root a caller
#: passes. A receipt writer that could write a work order, a lane, a claim, a
#: register, a governance file or a workflow could decide something. It cannot.
#: (Since the store also refuses every in-repository destination other than
#: :data:`RECEIPTS_ROOT`, this list is a second, named refusal, not the only one.)
FORBIDDEN_WRITE_ROOTS = (
    os.path.join(ROOT, "engine", "bridge", "orders"),
    os.path.join(ROOT, "engine", "bridge", "examples"),
    os.path.join(ROOT, "engine", "lanes"),
    os.path.join(ROOT, "engine", "receipts"),
    os.path.join(ROOT, "claims"),
    os.path.join(ROOT, "registers"),
    os.path.join(ROOT, "drive"),
    os.path.join(ROOT, "governance"),
    os.path.join(ROOT, "docs"),
    os.path.join(ROOT, ".github"),
)

SCHEMA = "q0.bridge.run_receipt/v1"
HELD_SCHEMA = "q0.bridge.held_delivery/v1"

RECORD_EXAMPLE = "EXAMPLE"
RECORD_RECORD = "RECORD"
RECORD_KINDS = (RECORD_EXAMPLE, RECORD_RECORD)

NOT_RUN = "NOT_RUN"
CANNOT_VERIFY = "CANNOT_VERIFY"
PREPARED = "PREPARED"
EXECUTED = "EXECUTED"
RECORDED = "RECORDED"
ACCEPTED = "ACCEPTED"
STATUSES = (NOT_RUN, CANNOT_VERIFY, PREPARED, EXECUTED, RECORDED, ACCEPTED)
#: Statuses that assert commands ran.
EXECUTED_STATUSES = (EXECUTED, RECORDED, ACCEPTED)

NOT_REVIEWED = "NOT_REVIEWED"
PASS_TECHNICAL = "PASS_TECHNICAL"
AMEND_REQUIRED = "AMEND_REQUIRED"
TECHNICAL_VERDICTS = (NOT_REVIEWED, PASS_TECHNICAL, AMEND_REQUIRED, CANNOT_VERIFY)

ACCEPTANCE_NOT = "NOT_ACCEPTED"
ACCEPTANCE_STATUSES = (ACCEPTANCE_NOT, ACCEPTED)

#: The only value ``scientific_status_change`` may carry.
SCIENTIFIC_STATUS_CHANGE = "UNCHANGED"

#: The id shape of a record under ``reviews/records/`` (``reviews/
#: review_record.schema.json``). An organizational-independence record is
#: either such a record or a Drive-id-shaped id; nothing else names one.
REVIEW_RECORD_ID_RE = re.compile(r"^REV-[A-Z0-9][A-Z0-9._-]{2,60}$")
_SAME_PROVIDER_RE = re.compile(r"same[ _-]?provider")

DISPOSITION_NEEDS_RECONCILIATION = "NEEDS_RECONCILIATION"
_HELD_INFIX = "." + DISPOSITION_NEEDS_RECONCILIATION + "."
_HELD_PREFIX_LEN = 16

_TOP_KEYS = (
    "schema", "record_kind", "status", "task_id", "work_order_digest", "run_id",
    "idempotency_key", "actor", "execution", "verification", "outputs", "review",
    "drive_return", "accepted_by_drive_record", "scientific_status_change",
    "does_not_establish",
)
_ACTOR_KEYS = ("authenticated_principal_id", "declared_provider", "declared_model",
               "session_id", "exposure")
_EXEC_KEYS = ("repository_id", "commit_sha", "tree_sha", "dirty_worktree",
              "environment_identity", "commands", "start_utc", "end_utc", "exit_codes")
_VERIF_KEYS = ("tests_passed", "tests_failed", "negative_controls", "coverage",
               "excluded_scope")
_OUTPUT_KEYS = ("path", "bytes", "sha256")
_REVIEW_KEYS = ("technical_verdict", "authorship_exposure", "independence_credit",
                "organizational_independence_record_id")
_RETURN_KEYS = ("delivery_id", "receipt_file_id", "expected_old_head",
                "readback_sha256", "readback_utc", "acceptance_status")
_ACCEPTED_KEYS = ("drive_id", "sha256")
_HELD_KEYS = ("schema", "disposition", "idempotency_key", "conflicts_with",
              "first_body_sha256", "held_body_sha256", "held_utc", "delivered", "note")

HELD_NOTE = (
    "A second delivery for the same idempotency key arrived with different "
    "content. It is held here beside the first receipt, verbatim. Neither "
    "replaces the other. Reconciliation is an owner-side task under governance/; "
    "this file performs none of it and grants nothing."
)


class ReceiptRefused(ValueError):
    """The store refused a receipt: it does not satisfy the schema."""


class ForbiddenDestination(ReceiptRefused):
    """A write was aimed outside the receipts root, or at a governed path."""


# ---------------------------------------------------------------------------
# key, canonical form, digest
# ---------------------------------------------------------------------------

def idempotency_key(task_id: str, work_order_digest: str,
                    tested_commit: Optional[str], run_id: str) -> str:
    """SHA-256 of the canonical JSON list of the four inputs.

    A list, not a joined string, so that no separator can collide and a
    ``null`` commit is distinguishable from the string ``"null"``.
    """
    return sha256_of(canonical_json([task_id, work_order_digest, tested_commit, run_id]))


def canonical_body(obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The hashed part of a receipt: everything but ``body_sha256``."""
    return {k: v for k, v in obj.items() if k != "body_sha256"}


def receipt_body_digest(obj: Mapping[str, Any]) -> str:
    return body_digest(canonical_body(obj))


def finalize(body: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of ``body`` with ``idempotency_key`` and ``body_sha256`` filled in.

    Pure: computes in memory from the body's own ``task_id``,
    ``work_order_digest``, ``execution.commit_sha`` and ``run_id``. It does
    not validate; :func:`validate_run_receipt` does.
    """
    b = canonical_body(body)
    ex = b.get("execution") if isinstance(b.get("execution"), dict) else {}
    b["idempotency_key"] = idempotency_key(
        b.get("task_id"), b.get("work_order_digest"), ex.get("commit_sha"), b.get("run_id"))
    return {**b, "body_sha256": body_digest(b)}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _opt(pred, value: Any) -> bool:
    return value is None or pred(value)


def _is_independence_record_id(x: Any) -> bool:
    """Drive-id shaped or a ``reviews/`` record id, and never a placeholder."""
    if not isinstance(x, str):
        return False
    s = x.strip()
    if not s or s.casefold() in PLACEHOLDERS:
        return False
    if _SAME_PROVIDER_RE.search(s.casefold()):
        return False
    return is_drive_id(s) or bool(REVIEW_RECORD_ID_RE.match(s))


def validate_run_receipt(obj: Any) -> List[str]:
    """Every problem with a run receipt, as a list of strings. Never raises.

    Empty list means the record is well-formed under the state model above.
    It does not mean the run was correct, the tests were the right tests, the
    reviewer was independent, or anything was accepted or changed.
    """
    p: List[str] = []
    if not isinstance(obj, dict):
        return ["run receipt is not a JSON object"]

    top = check_keys(canonical_body(obj), _TOP_KEYS, "run_receipt", p)
    if "body_sha256" not in obj:
        p.append("run_receipt.body_sha256 is missing")
    elif not is_sha256(obj["body_sha256"]):
        p.append(f"body_sha256 {obj['body_sha256']!r} is not a sha256 hex digest")
    else:
        recomputed = receipt_body_digest(obj)
        if recomputed != obj["body_sha256"]:
            p.append(
                f"body_sha256 mismatch: stored {obj['body_sha256'][:16]}..., recomputed "
                f"{recomputed[:16]}... -- this receipt was modified after it was written "
                "(append-only violation)")
    if top is None:
        return p

    if obj["schema"] != SCHEMA:
        p.append(f"schema is {obj['schema']!r}, expected {SCHEMA!r}")
    kind = obj["record_kind"]
    if kind not in RECORD_KINDS:
        p.append(f"record_kind {kind!r} is not one of {list(RECORD_KINDS)}")
    status = obj["status"]
    if status not in STATUSES:
        p.append(f"status {status!r} is not one of {list(STATUSES)}")
    if not is_nonempty_str(obj["task_id"]):
        p.append("task_id is empty")
    if not is_sha256(obj["work_order_digest"]):
        p.append(f"work_order_digest {obj['work_order_digest']!r} is not a sha256 hex digest")
    if not is_nonempty_str(obj["run_id"]):
        p.append("run_id is empty")
    executed = status in EXECUTED_STATUSES

    # the one permitted value
    ssc = obj["scientific_status_change"]
    if ssc != SCIENTIFIC_STATUS_CHANGE:
        p.append(
            f"scientific_status_change is {ssc!r}; the only permitted value is "
            f"{SCIENTIFIC_STATUS_CHANGE!r}. FW-NO-RECEIPT-PROMOTION: a receipt records "
            "that something ran, not that something is true, and it changes no status.")

    actor = check_keys(obj["actor"], _ACTOR_KEYS, "actor", p)
    if actor is not None:
        for k in _ACTOR_KEYS:
            if not _opt(is_nonempty_str, actor[k]):
                p.append(f"actor.{k} must be a non-empty string or null")
        if executed:
            if actor["authenticated_principal_id"] is None:
                p.append(f"status is {status} but actor.authenticated_principal_id is null; "
                         "a declared model name is attribution, not authentication")
            else:
                check_evidence_string(actor["authenticated_principal_id"],
                                      "actor.authenticated_principal_id", p)

    ex = check_keys(obj["execution"], _EXEC_KEYS, "execution", p)
    commit = None
    exit_codes: List[Any] = []
    commands: List[Any] = []
    dirty: Optional[bool] = None
    if ex is not None:
        if not _opt(lambda v: is_int(v) and v > 0, ex["repository_id"]):
            p.append("execution.repository_id must be a positive integer or null")
        if not _opt(is_commit, ex["commit_sha"]):
            p.append(f"execution.commit_sha {ex['commit_sha']!r} is not a 40-hex commit sha or null")
        else:
            commit = ex["commit_sha"]
        if not _opt(is_commit, ex["tree_sha"]):
            p.append(f"execution.tree_sha {ex['tree_sha']!r} is not a 40-hex tree sha or null")
        if not _opt(lambda v: isinstance(v, bool), ex["dirty_worktree"]):
            p.append("execution.dirty_worktree must be a boolean or null")
        else:
            dirty = ex["dirty_worktree"]
        if not _opt(is_nonempty_str, ex["environment_identity"]):
            p.append("execution.environment_identity must be a non-empty string or null")
        if not is_str_list(ex["commands"]):
            p.append("execution.commands must be a list of strings")
        else:
            commands = ex["commands"]
            for i, cmd in enumerate(commands):
                check_evidence_string(cmd, f"execution.commands[{i}]", p, min_len=2)
        for k in ("start_utc", "end_utc"):
            if not _opt(is_timestamp, ex[k]):
                p.append(f"execution.{k} {ex[k]!r} is not a UTC timestamp or null")
        start, end = parse_timestamp(ex["start_utc"]), parse_timestamp(ex["end_utc"])
        if start is not None and end is not None and end < start:
            p.append(f"execution.end_utc {ex['end_utc']!r} is before start_utc "
                     f"{ex['start_utc']!r}; a run does not end before it starts")
        if not isinstance(ex["exit_codes"], list) or not all(is_int(c) for c in ex["exit_codes"]):
            p.append("execution.exit_codes must be a list of integers")
        else:
            exit_codes = ex["exit_codes"]

    # idempotency key: recomputed from the four inputs
    key = obj["idempotency_key"]
    if not is_sha256(key):
        p.append(f"idempotency_key {key!r} is not a sha256 hex digest")
    elif ex is not None and is_nonempty_str(obj["task_id"]) and is_sha256(obj["work_order_digest"]) \
            and is_nonempty_str(obj["run_id"]) and _opt(is_commit, ex["commit_sha"]):
        expect = idempotency_key(obj["task_id"], obj["work_order_digest"],
                                 ex["commit_sha"], obj["run_id"])
        if key != expect:
            p.append(
                f"idempotency_key mismatch: stored {key[:16]}..., recomputed {expect[:16]}... "
                "from (task_id, work_order_digest, execution.commit_sha, run_id)")

    ver = check_keys(obj["verification"], _VERIF_KEYS, "verification", p)
    tests_passed = tests_failed = None
    negative_controls: List[Any] = []
    if ver is not None:
        for k in ("tests_passed", "tests_failed"):
            if not _opt(lambda v: is_int(v) and v >= 0, ver[k]):
                p.append(f"verification.{k} must be a non-negative integer or null")
        tests_passed, tests_failed = ver["tests_passed"], ver["tests_failed"]
        if not is_str_list(ver["negative_controls"]) or any(not s.strip() for s in ver["negative_controls"]):
            p.append("verification.negative_controls must be a list of non-empty strings")
        else:
            negative_controls = ver["negative_controls"]
            for i, nc in enumerate(negative_controls):
                check_evidence_string(nc, f"verification.negative_controls[{i}]", p)
        if not _opt(is_nonempty_str, ver["coverage"]):
            p.append("verification.coverage must be a non-empty string or null")
        if not is_str_list(ver["excluded_scope"]):
            p.append("verification.excluded_scope must be a list of strings")

    outputs = obj["outputs"]
    if not isinstance(outputs, list):
        p.append("outputs must be a list")
    else:
        for i, out in enumerate(outputs):
            o = check_keys(out, _OUTPUT_KEYS, f"outputs[{i}]", p)
            if o is None:
                continue
            if not is_nonempty_str(o["path"]):
                p.append(f"outputs[{i}].path is empty")
            else:
                norm, reason = normalize_scope_path(o["path"])
                if norm is None:
                    p.append(f"outputs[{i}].path {o['path']!r} {reason}; an output path is a "
                             "literal repository-relative path")
                elif norm == ".":
                    p.append(f"outputs[{i}].path {o['path']!r} names the whole repository")
            if not is_int(o["bytes"]) or o["bytes"] < 0:
                p.append(f"outputs[{i}].bytes {o['bytes']!r} is not a byte count")
            if not is_sha256(o["sha256"]):
                p.append(f"outputs[{i}].sha256 {o['sha256']!r} is not a sha256 hex digest")

    rev = check_keys(obj["review"], _REVIEW_KEYS, "review", p)
    verdict = None
    if rev is not None:
        verdict = rev["technical_verdict"]
        if verdict not in TECHNICAL_VERDICTS:
            p.append(f"review.technical_verdict {verdict!r} is not one of {list(TECHNICAL_VERDICTS)}")
        if not _opt(is_nonempty_str, rev["authorship_exposure"]):
            p.append("review.authorship_exposure must be a non-empty string or null")
        credit = rev["independence_credit"]
        rec_id = rev["organizational_independence_record_id"]
        if not is_int(credit) or credit not in (0, 1):
            p.append(f"review.independence_credit {credit!r} must be 0 or 1")
        elif credit == 1:
            if not _is_independence_record_id(rec_id):
                p.append(
                    f"review.independence_credit is 1 but organizational_independence_"
                    f"record_id {rec_id!r} names no record (a Drive-id-shaped id or a "
                    "reviews/ REV-... id; never a placeholder, never a same-provider "
                    "reviewer). Organizational independence cannot be manufactured here; "
                    "without a record it is 0.")
            if verdict == NOT_REVIEWED:
                p.append("review.independence_credit is 1 on a NOT_REVIEWED verdict; a credit "
                         "on a review that did not happen is manufactured independence")
        if not _opt(is_nonempty_str, rec_id):
            p.append("review.organizational_independence_record_id must be a non-empty string or null")
        elif rec_id is not None and not _is_independence_record_id(rec_id):
            p.append(f"review.organizational_independence_record_id {rec_id!r} is not a "
                     "record id (a Drive-id-shaped id or a reviews/ REV-... id)")
        if verdict == PASS_TECHNICAL:
            if rev["authorship_exposure"] is None:
                p.append("review.technical_verdict is PASS_TECHNICAL but authorship_exposure is "
                         "null; a verdict without an exposure disclosure is not a review record")
            else:
                check_evidence_string(rev["authorship_exposure"], "review.authorship_exposure", p)

    ret = check_keys(obj["drive_return"], _RETURN_KEYS, "drive_return", p)
    acceptance = None
    if ret is not None:
        for k in ("delivery_id", "receipt_file_id", "expected_old_head"):
            if not _opt(is_nonempty_str, ret[k]):
                p.append(f"drive_return.{k} must be a non-empty string or null")
        if not _opt(is_sha256, ret["readback_sha256"]):
            p.append("drive_return.readback_sha256 must be a sha256 hex digest or null")
        if not _opt(is_timestamp, ret["readback_utc"]):
            p.append("drive_return.readback_utc must be a UTC timestamp or null")
        acceptance = ret["acceptance_status"]
        if acceptance not in ACCEPTANCE_STATUSES:
            p.append(f"drive_return.acceptance_status {acceptance!r} is not one of "
                     f"{list(ACCEPTANCE_STATUSES)}")

    abr = obj["accepted_by_drive_record"]
    if abr is not None:
        a = check_keys(abr, _ACCEPTED_KEYS, "accepted_by_drive_record", p)
        if a is not None:
            if not is_drive_id(a["drive_id"]):
                p.append(f"accepted_by_drive_record.drive_id {a['drive_id']!r} is not a Drive id")
            if not is_sha256(a["sha256"]):
                p.append(f"accepted_by_drive_record.sha256 {a['sha256']!r} is not a sha256 hex digest")

    check_does_not_establish(obj["does_not_establish"], "does_not_establish", p)

    # -- the state model ---------------------------------------------------

    # missing evidence is never PASS
    if verdict == PASS_TECHNICAL:
        if tests_passed is None:
            p.append("review.technical_verdict is PASS_TECHNICAL while verification."
                     "tests_passed is null: missing evidence is NOT_RUN or CANNOT_VERIFY, "
                     "never PASS")
        elif is_int(tests_passed) and tests_passed < 1:
            p.append("review.technical_verdict is PASS_TECHNICAL while verification."
                     "tests_passed is 0: a run that passed no test passed nothing; missing "
                     "evidence is NOT_RUN or CANNOT_VERIFY, never PASS")
        if tests_failed is None:
            p.append("review.technical_verdict is PASS_TECHNICAL while verification."
                     "tests_failed is null: an uncounted failure count is missing evidence")
        if not executed:
            p.append(f"review.technical_verdict is PASS_TECHNICAL on a {status} receipt; "
                     "nothing that did not execute can pass")
        if tests_failed not in (None, 0):
            p.append("review.technical_verdict is PASS_TECHNICAL with tests_failed > 0")
        if any(c != 0 for c in exit_codes):
            p.append("review.technical_verdict is PASS_TECHNICAL with a nonzero exit code")
        if dirty is True:
            p.append("review.technical_verdict is PASS_TECHNICAL while execution.dirty_worktree "
                     "is true: the tested tree is not the named commit, and a reviewer cannot "
                     "pass a tree the commit does not identify")

    if any(c != 0 for c in exit_codes) and tests_failed == 0:
        p.append(
            f"execution.exit_codes {exit_codes} contain a nonzero code while "
            "verification.tests_failed is 0; a failed command cannot be reported as "
            "zero failures")

    if executed:
        if commit is None:
            p.append(f"status is {status} but execution.commit_sha is null")
        if ex is not None:
            if ex["repository_id"] is None:
                p.append(f"status is {status} but execution.repository_id is null; the exact "
                         "executed commit is a commit of a named repository")
            if ex["tree_sha"] is None:
                p.append(f"status is {status} but execution.tree_sha is null; the contract "
                         "requires the exact executed commit and tree")
            if ex["dirty_worktree"] is None:
                p.append(f"status is {status} but execution.dirty_worktree is null; whether "
                         "the tree was the commit is evidence, not an option")
            if ex["environment_identity"] is None:
                p.append(f"status is {status} but execution.environment_identity is null")
            else:
                check_evidence_string(ex["environment_identity"], "execution.environment_identity", p)
            if ex["start_utc"] is None or ex["end_utc"] is None:
                p.append(f"status is {status} but start_utc/end_utc are not both set")
        if not commands:
            p.append(f"status is {status} but execution.commands is empty")
        if not exit_codes:
            p.append(f"status is {status} but execution.exit_codes is empty")
        elif commands and len(exit_codes) != len(commands):
            p.append(f"status is {status} but exit_codes ({len(exit_codes)}) and commands "
                     f"({len(commands)}) differ in count; one exit code per command")
        if not negative_controls:
            p.append(f"status is {status} but verification.negative_controls is empty; "
                     "negative controls are the deliverable")
        if tests_passed is None or tests_failed is None:
            p.append(f"status is {status} but test counts are null: missing evidence is "
                     "CANNOT_VERIFY, never an execution with a blank")
        elif is_int(tests_passed) and is_int(tests_failed) and tests_passed + tests_failed < 1:
            p.append(f"status is {status} but tests_passed + tests_failed is 0: a run that "
                     "counted no test is CANNOT_VERIFY, not an execution")
        if ver is not None:
            if ver["coverage"] is None:
                p.append(f"status is {status} but verification.coverage is null; covered "
                         "versus excluded scope is evidence the contract requires")
            else:
                check_evidence_string(ver["coverage"], "verification.coverage", p)
    else:
        if exit_codes:
            p.append(f"status is {status} but execution.exit_codes is not empty; "
                     "a receipt that reports exit codes executed")

    if status in (RECORDED, ACCEPTED) and ret is not None:
        for k in ("delivery_id", "receipt_file_id", "readback_sha256", "readback_utc"):
            if ret[k] is None:
                p.append(f"status is {status} but drive_return.{k} is null; RECORDED "
                         "requires a Drive readback, not a delivery attempt")

    if status == ACCEPTED:
        if abr is None:
            p.append(
                "status is ACCEPTED without accepted_by_drive_record. ACCEPTED is an "
                "owner-side acceptance in Drive that this repository cannot verify; a "
                "receipt claiming it must name the Drive record (id and sha256) it "
                "attributes the acceptance to, and even then the claim is transcribed, "
                "not verified.")
        if acceptance != ACCEPTED:
            p.append("status is ACCEPTED but drive_return.acceptance_status is not ACCEPTED")
    else:
        if abr is not None:
            p.append(f"status is {status} but accepted_by_drive_record is set")
        if acceptance == ACCEPTED:
            p.append(f"status is {status} but drive_return.acceptance_status is ACCEPTED")

    if kind == RECORD_EXAMPLE:
        if status != NOT_RUN:
            p.append(f"record_kind is EXAMPLE but status is {status}; an example is NOT_RUN")
        if verdict != NOT_REVIEWED:
            p.append("record_kind is EXAMPLE but technical_verdict is not NOT_REVIEWED")

    return p


def run_receipt_notes(obj: Any) -> List[str]:
    """Facts RECORDED about a well-formed receipt, granted by nobody here.

    Never raises on garbage; the checker calls it only for a receipt that
    validated.
    """
    notes: List[str] = []
    if not isinstance(obj, dict):
        return notes
    key = str(obj.get("idempotency_key") or "?")[:16]
    rev = obj.get("review") if isinstance(obj.get("review"), dict) else {}
    ex = obj.get("execution") if isinstance(obj.get("execution"), dict) else {}
    ver = obj.get("verification") if isinstance(obj.get("verification"), dict) else {}
    abr = obj.get("accepted_by_drive_record")
    if rev.get("independence_credit") == 1:
        notes.append(
            f"{key}: independence_credit 1 is transcribed from organizational-independence "
            f"record {rev.get('organizational_independence_record_id')}; this validator "
            "awards no independence and the independence-requiring gate is not moved by it.")
    if obj.get("status") == ACCEPTED and isinstance(abr, dict):
        notes.append(
            f"{key}: status ACCEPTED is transcribed from Drive record {abr.get('drive_id')}; "
            "this repository did not accept anything, cannot verify that the record exists "
            "or accepted anything, and the claim is recorded unverified.")
    if obj.get("status") in EXECUTED_STATUSES:
        if ex.get("dirty_worktree") is True:
            notes.append(f"{key}: execution.dirty_worktree is true; the tested tree is not the "
                         f"named commit {str(ex.get('commit_sha'))[:12]} and this run cannot "
                         "be reconstructed from that commit alone.")
        if ver.get("tests_passed") == 0:
            notes.append(f"{key}: tests_passed is 0 on an executed receipt; nothing passed.")
    return notes


def check_receipt_against_order(receipt: Mapping[str, Any],
                                order: Optional[Mapping[str, Any]]) -> List[str]:
    """Cross-checks between a receipt and the order its digest names.

    Agreement is consistency, not evidence: a receipt that agrees with its
    order is still a record that something ran, not that it ran correctly.
    """
    p: List[str] = []
    if order is None:
        p.append(
            f"work_order_digest {str(receipt.get('work_order_digest'))[:16]}... resolves to "
            "no committed work order; a receipt for an order this repository does not "
            "hold cannot be checked against it")
        return p
    if order.get("task_id") != receipt.get("task_id"):
        p.append(f"task_id {receipt.get('task_id')!r} differs from the referenced order's "
                 f"{order.get('task_id')!r}")
    if receipt.get("record_kind") == RECORD_RECORD and order.get("record_kind") == RECORD_EXAMPLE:
        p.append("a RECORD receipt references an EXAMPLE order; examples authorize nothing")

    repo = order.get("repository") if isinstance(order.get("repository"), dict) else {}
    auth = order.get("authorization") if isinstance(order.get("authorization"), dict) else {}
    scope = order.get("scope") if isinstance(order.get("scope"), dict) else {}
    ex = receipt.get("execution") if isinstance(receipt.get("execution"), dict) else {}
    status = receipt.get("status")

    rid = ex.get("repository_id")
    if rid is not None and repo.get("id") is not None and rid != repo.get("id"):
        p.append(f"execution.repository_id {rid!r} differs from the order's repository.id "
                 f"{repo.get('id')!r}; a run in another repository is not a run of this order")

    if status in EXECUTED_STATUSES:
        if order.get("status") != STATUS_EXECUTABLE or \
                auth.get("verification_status") != VERIFIED_BY_OWNER_BOUNDARY:
            p.append(
                f"status is {status} but the referenced order is {order.get('status')!r} with "
                f"verification_status {auth.get('verification_status')!r}: an order that is "
                "not EXECUTABLE and VERIFIED_BY_OWNER_BOUNDARY is PREPARED-only, and a "
                "receipt that says something ran under it records a run the contract did "
                "not permit")

    allowed_cmds = scope.get("allowed_commands")
    cmds = ex.get("commands")
    if is_str_list(cmds) and is_str_list(allowed_cmds):
        for i, cmd in enumerate(cmds):
            if cmd not in allowed_cmds:
                p.append(f"execution.commands[{i}] {cmd!r} is not present verbatim in the "
                         "order's scope.allowed_commands; a receipt reports what the order "
                         "allowed or it reports a run outside the order")

    allowed_paths = scope.get("allowed_paths")
    outputs = receipt.get("outputs")
    if isinstance(outputs, list):
        for i, out in enumerate(outputs):
            path = out.get("path") if isinstance(out, dict) else None
            if not isinstance(path, str):
                continue
            protected = protected_surfaces_touched([path])
            policy = policy_surfaces_touched([path])
            if protected or policy:
                p.append(f"outputs[{i}].path {path!r} lies on surface(s) {protected + policy}; "
                         "no output of a run lands on a policy or protected surface, "
                         "whatever the order says")
            elif not path_within_scope(path, allowed_paths):
                p.append(f"outputs[{i}].path {path!r} is not within the order's "
                         "scope.allowed_paths; an output outside the scope is a run outside "
                         "the order")
    return p


# ---------------------------------------------------------------------------
# held (conflicting) deliveries
# ---------------------------------------------------------------------------

def held_file_name(key: str, held_body_sha256: str) -> str:
    return f"{key}{_HELD_INFIX}{held_body_sha256[:_HELD_PREFIX_LEN]}.json"


def parse_receipt_file_name(name: str) -> Tuple[str, Optional[str], Optional[str]]:
    """``("receipt", key, None)``, ``("held", key, prefix)`` or ``("other", None, None)``."""
    if not name.endswith(".json"):
        return "other", None, None
    stem = name[:-5]
    if is_sha256(stem):
        return "receipt", stem, None
    if _HELD_INFIX in stem:
        key, _, prefix = stem.partition(_HELD_INFIX)
        if is_sha256(key) and len(prefix) == _HELD_PREFIX_LEN and \
                all(c in "0123456789abcdef" for c in prefix):
            return "held", key, prefix
    return "other", None, None


def validate_held_delivery(obj: Any, first: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Problems with a held-delivery wrapper (and the receipt inside it)."""
    p: List[str] = []
    h = check_keys(obj, _HELD_KEYS, "held_delivery", p)
    if h is None:
        return p
    if h["schema"] != HELD_SCHEMA:
        p.append(f"held_delivery.schema is {h['schema']!r}, expected {HELD_SCHEMA!r}")
    if h["disposition"] != DISPOSITION_NEEDS_RECONCILIATION:
        p.append(f"held_delivery.disposition is {h['disposition']!r}, expected "
                 f"{DISPOSITION_NEEDS_RECONCILIATION!r}")
    if not is_sha256(h["idempotency_key"]):
        p.append("held_delivery.idempotency_key is not a sha256 hex digest")
    if h["conflicts_with"] != f"{h['idempotency_key']}.json":
        p.append("held_delivery.conflicts_with does not name <idempotency_key>.json")
    if not is_sha256(h["first_body_sha256"]) or not is_sha256(h["held_body_sha256"]):
        p.append("held_delivery body digests must be sha256 hex digests")
    if not is_timestamp(h["held_utc"]):
        p.append("held_delivery.held_utc is not a UTC timestamp")
    if not is_nonempty_str(h["note"]):
        p.append("held_delivery.note is empty")
    d = h["delivered"]
    for msg in validate_run_receipt(d):
        p.append(f"delivered: {msg}")
    if isinstance(d, dict):
        if d.get("idempotency_key") != h["idempotency_key"]:
            p.append("held_delivery.delivered carries a different idempotency_key")
        if d.get("body_sha256") != h["held_body_sha256"]:
            p.append("held_delivery.held_body_sha256 does not match delivered.body_sha256")
        if d.get("record_kind") == RECORD_EXAMPLE:
            p.append("held_delivery.delivered is an EXAMPLE; examples are never delivered")
    if first is not None:
        if first.get("body_sha256") != h["first_body_sha256"]:
            p.append("held_delivery.first_body_sha256 does not match the first receipt on disk")
        if isinstance(d, dict) and d.get("body_sha256") == first.get("body_sha256"):
            p.append("held_delivery.delivered is identical to the first receipt; an identical "
                     "retry is not a conflict and is not held")
    return p


# ---------------------------------------------------------------------------
# the append-only store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoreResult:
    """Where a delivery landed and how. ``disposition`` is one of ``STORED``,
    ``IDENTICAL_RETRY`` (nothing written) or ``NEEDS_RECONCILIATION`` (held
    beside the first, first untouched)."""
    path: str
    disposition: str
    wrote: bool


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _assert_destination_allowed(path: str, root: str) -> None:
    """The store writes to :data:`RECEIPTS_ROOT` or to a directory entirely
    outside this repository; nothing else. Every root under
    :data:`FORBIDDEN_WRITE_ROOTS` is additionally refused by name."""
    real = os.path.realpath(path)
    if not _inside(real, root):
        raise ForbiddenDestination(
            f"receipt destination {real!r} is outside the receipts root {root!r}")
    for forbidden in FORBIDDEN_WRITE_ROOTS:
        f = os.path.realpath(forbidden)
        if real == f or _inside(real, f):
            raise ForbiddenDestination(
                f"refusing to write under {forbidden!r}. A receipt records; it does not "
                "decide, and it does not author work orders, lanes, claims, registers, "
                "governance or workflows.")
    repo = os.path.realpath(ROOT)
    receipts_root = os.path.realpath(RECEIPTS_ROOT)
    if _inside(real, repo) and not _inside(real, receipts_root):
        raise ForbiddenDestination(
            f"refusing to write {real!r}: inside this repository the store writes under "
            f"{receipts_root!r} only. A receipt anywhere else in the tree is clutter at best "
            "and a record out of place at worst.")


def store_receipt(obj: Mapping[str, Any], receipts_dir: Optional[str] = None,
                  now: Optional[str] = None) -> StoreResult:
    """Store a receipt, or refuse. Never overwrites; never deletes.

    Refuses a receipt that fails :func:`validate_run_receipt`, a receipt whose
    ``record_kind`` is EXAMPLE (examples live under ``examples/`` and are never
    delivered), any destination under :data:`FORBIDDEN_WRITE_ROOTS`, and any
    destination inside this repository other than :data:`RECEIPTS_ROOT`.
    """
    problems = validate_run_receipt(obj)
    if problems:
        raise ReceiptRefused(
            f"receipt refused ({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems))
    if obj.get("record_kind") == RECORD_EXAMPLE:
        raise ReceiptRefused("receipt refused: record_kind EXAMPLE is never stored as a delivery")

    root = os.path.realpath(receipts_dir or RECEIPTS_ROOT)
    key = obj["idempotency_key"]
    primary = os.path.join(root, key + ".json")
    _assert_destination_allowed(primary, root)
    os.makedirs(root, exist_ok=True)

    if not os.path.exists(primary):
        try:
            with open(primary, "x", encoding="utf-8") as f:
                f.write(to_json(obj))
            return StoreResult(primary, "STORED", True)
        except FileExistsError:
            pass  # lost a race with a concurrent delivery; fall through and compare

    first = load_receipt(primary)
    if first.get("body_sha256") == obj.get("body_sha256") and \
            canonical_json(first) == canonical_json(dict(obj)):
        return StoreResult(primary, "IDENTICAL_RETRY", False)

    held = {
        "schema": HELD_SCHEMA,
        "disposition": DISPOSITION_NEEDS_RECONCILIATION,
        "idempotency_key": key,
        "conflicts_with": key + ".json",
        "first_body_sha256": first.get("body_sha256"),
        "held_body_sha256": obj["body_sha256"],
        "held_utc": now or utc_now(),
        "delivered": dict(obj),
        "note": HELD_NOTE,
    }
    held_path = os.path.join(root, held_file_name(key, obj["body_sha256"]))
    _assert_destination_allowed(held_path, root)
    try:
        with open(held_path, "x", encoding="utf-8") as f:
            f.write(to_json(held))
        return StoreResult(held_path, DISPOSITION_NEEDS_RECONCILIATION, True)
    except FileExistsError:
        # the same conflicting content delivered again: already held
        return StoreResult(held_path, DISPOSITION_NEEDS_RECONCILIATION, False)


def load_receipt(path: str) -> Dict[str, Any]:
    """Read one receipt file. Duplicate keys and NaN/Infinity raise ``ValueError``."""
    return load_json_strict(path)


def to_json(obj: Mapping[str, Any]) -> str:
    """Pretty, sorted, trailing newline: the only serialisation a committed
    record may have on disk (the checker refuses any other bytes)."""
    return json.dumps(dict(obj), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
