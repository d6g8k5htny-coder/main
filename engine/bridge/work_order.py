#!/usr/bin/env python3
"""The work-order record, ``q0.bridge.work_order/v1``, and its validator.

A **work order** is the frozen statement of one bounded task whose
authorization was recorded at the owner boundary: which repository at which
base commit, which Drive sources govern it (each by stable id, exact SHA-256,
byte count, extraction rule and native revision), which R17 Work Events claim
and lease it runs under, which paths and commands are in scope, which tests
and negative controls it must produce, what its limits are, what may be
disclosed, and what it may not do. It is the git-side pin of a Drive object.
The handoff it ports (``drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/
DRIVE_GITHUB_EXECUTION_HANDOFF.md``, line 9, disposition PROPOSED EXECUTION
CONTRACT / NOT DEPLOYED) puts it as "a GitHub copy of an instruction is a
pinned execution input, not a second governing policy". That sentence is the
handoff's, not ``governance/GIT_ADAPTATION.md``'s, and porting it does not
deploy it.

WHAT A WORK ORDER IS NOT, AND DOES NOT ESTABLISH
------------------------------------------------
It does not establish anything mathematical: no bound, no premise, no
closure. It is not an authorization. The authorization is the Drive record it
names in ``authorization.authorizing_record_drive_id``; this file is a pinned
copy of what that record says, and a hash of it "establishes identity, not
truth or authorization" (OP-PROT-019 §2). It is not a task queue: the R17 Work
Events register remains the claim/disposition log. It is not enforcement:
nothing here stops a push, holds a credential or protects a branch. And it
cannot authorize a change of scientific status: ``scientific_status_change_
authorized`` admits exactly one value, ``false``, and a work order carrying
``true`` is refused, because no work order can authorize a status change ---
only an operator decision under ``governance/`` can.

THE RULES THE VALIDATOR ENFORCES
--------------------------------
All of them can only REFUSE. None grants, verifies or executes anything.

1. ``scientific_status_change_authorized`` must be exactly ``false``.
2. ``authorization.verification_status`` is ``NOT_VERIFIED`` or
   ``VERIFIED_BY_OWNER_BOUNDARY``. A ``NOT_VERIFIED`` order is PREPARED-only:
   its ``status`` cannot be ``EXECUTABLE``.
3. A work order must not be created or edited by the PR it authorizes.
   ``authorization.authorized_outside_this_repository`` must be ``true``.
   ``source_ref``, ``authorizing_record_drive_id`` and
   ``policy_change_authorized_by`` may not name ``engine/bridge/orders/`` in
   any spelling (case, ``//``, ``/./``), any top-level path of this repository
   (:data:`REPOSITORY_TOP_LEVEL`), a file name, a GitHub URL, a pull request
   or a PR number: none of those is an owner-side record. A
   ``VERIFIED_BY_OWNER_BOUNDARY`` order must carry a Drive id in
   ``authorizing_record_drive_id`` and a Drive id or Drive URL in
   ``source_ref``.
4. Every governing source carries all five identity fields: ``drive_id``,
   ``sha256``, ``bytes``, ``extraction_rule``, ``revision_if_native``. A
   source without them is invalid; a title or a modified time is not identity.
5. Scope paths (``scope.allowed_paths``, ``public_disclosure.approved_paths``)
   are literal repository-relative paths: an absolute path, ``~``, a ``..``
   segment, a glob, an empty segment or a control character is refused, not
   normalized, and an unnormalizable entry counts as reaching every surface.
   Comparison is casefolded and segment-wise. Two lists of surfaces apply:

   * :data:`PROTECTED_SURFACES` -- the bridge itself (``engine/bridge/``:
     orders, receipts, examples, validators, store), ``engine/lanes/``,
     ``engine/receipts/``, ``drive/``, ``tools/``, ``tests/test_bridge.py``,
     ``quarantine/``, ``CLAUDE.md``, ``AGENTS.md``. Reaching one is refused
     outright. No disclosure approval or policy reference unlocks it: an
     order that could authorize writing an order, a receipt, the checker or
     its negative controls could authorize itself.
   * :data:`POLICY_SURFACES` -- ``.github/``, ``governance/``, ``registers/``,
     ``claims/graph.json``, ``docs/OPEN_PROBLEMS.md``, the policy and status
     surfaces the contract names. Reaching one is refused unless
     ``public_disclosure.approved`` is true AND
     ``authorization.policy_change_authorized_by`` names a Drive id. Even
     then the validator only *records* that fact (:func:`work_order_notes`);
     it never grants the change. Passing validation with a policy surface in
     scope means "the references are present", not "the change is permitted".

   ``approved_paths`` must lie within ``allowed_paths``: nothing outside the
   scope can be disclosed.
6. ``scope.allowed_commands`` are literal commands, none a placeholder or a
   no-op (``:``, ``true``), none a git or gh mutation (``git push``, ``git
   commit``, ...: committing is a separately authorized step, not something a
   work order grants), and none that lexically writes into a surface (a
   redirect, ``open(``, ``tee``, ``cp``, ``sed -i``, ... preceding a surface
   path). This is a lexical heuristic that can only refuse; passing it does
   not establish that a command is safe. What a command may write is bounded
   by ``allowed_paths`` and, under the contract, by server-side controls this
   repository does not hold.
7. The record is closed at every level, every identifier is well-formed, and
   ``work_order_digest`` equals the SHA-256 of the canonical body. The digest
   establishes identity only; the freeze is git history, which
   ``tools/bridge_check.py`` reads. An order edited in place and re-digested
   is refused by the checker, not by this function.
8. ``does_not_establish`` is mandatory and must be a statement.
9. ``record_kind`` is ``EXAMPLE`` or ``RECORD``. An EXAMPLE is never
   ``EXECUTABLE`` and is never ``VERIFIED``; the checker refuses one under
   ``engine/bridge/orders/``.

THERE IS NO WRITER
------------------
This module opens no file for writing. A work order is authored at the owner
boundary and committed by a separately authorized step, never manufactured by
the code it would govern. :func:`with_digest` attaches a digest to a body in
memory; :func:`load_work_order` reads one file. That is all the I/O there is.

Standard library only. Python 3.11.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.bridge.common import (
    body_digest, check_does_not_establish, check_evidence_string, check_keys,
    is_commit, is_drive_id, is_drive_url, is_int, is_nonempty_str, is_sha256,
    is_str_list, is_timestamp, load_json_strict, normalize_scope_path,
)

__all__ = [
    "SCHEMA", "RECORD_KINDS", "STATUSES", "VERIFICATION_STATUSES",
    "NETWORK_POLICIES", "POLICY_SURFACES", "PROTECTED_SURFACES",
    "REPOSITORY_TOP_LEVEL", "SOURCE_FIELDS", "OWN_ORDERS_DIR",
    "MUTATING_COMMANDS",
    "canonical_body", "work_order_digest", "with_digest", "to_json",
    "validate_work_order", "work_order_notes", "policy_surfaces_touched",
    "protected_surfaces_touched", "malformed_scope_paths", "path_within_scope",
    "names_repository_path", "command_problems", "load_work_order",
]

SCHEMA = "q0.bridge.work_order/v1"

RECORD_EXAMPLE = "EXAMPLE"
RECORD_RECORD = "RECORD"
RECORD_KINDS = (RECORD_EXAMPLE, RECORD_RECORD)

STATUS_PREPARED = "PREPARED"
STATUS_EXECUTABLE = "EXECUTABLE"
STATUSES = (STATUS_PREPARED, STATUS_EXECUTABLE)

NOT_VERIFIED = "NOT_VERIFIED"
VERIFIED_BY_OWNER_BOUNDARY = "VERIFIED_BY_OWNER_BOUNDARY"
VERIFICATION_STATUSES = (NOT_VERIFIED, VERIFIED_BY_OWNER_BOUNDARY)

NETWORK_POLICIES = ("NONE", "ALLOWLIST", "UNRESTRICTED")

#: Paths whose contents are policy or status, not research code. A work order
#: whose scope touches one is refused unless it carries both a disclosure
#: approval and a policy-change authorization reference -- and carrying them is
#: recorded, not granted. These are the surfaces the contract names.
POLICY_SURFACES = (
    ".github/",
    "governance/",
    "registers/",
    "claims/graph.json",
    "docs/OPEN_PROBLEMS.md",
)

#: Paths no work order may reach, whatever references it carries. The bridge
#: itself (an order that could authorize writing orders, receipts, the
#: validators or the store could authorize itself), the other engine records,
#: the Drive exports (CLAUDE.md rule 7), the checkers CI runs and this
#: bridge's negative controls, the quarantine list and the agent instructions.
PROTECTED_SURFACES = (
    "engine/bridge/",
    "engine/lanes/",
    "engine/receipts/",
    "drive/",
    "tools/",
    "tests/test_bridge.py",
    "quarantine/",
    "CLAUDE.md",
    "AGENTS.md",
    "requirements-ci.lock",
)

#: Every top-level entry of this repository. A reference that names one is a
#: repository path, and a repository path is never an owner-side record.
#: ``tests/test_bridge.py`` asserts this list matches the checkout.
REPOSITORY_TOP_LEVEL = (
    ".github", ".gitignore", "requirements-ci.lock", "AGENTS.md", "CITATION.cff", "CLAUDE.md",
    "CONTRIBUTING.md", "LICENSE", "README.md", "architecture", "attestations", "claims", "docs", "drive",
    "engine", "governance", "legacy", "packages", "quarantine", "recovery",
    "registers", "research", "reviews", "sandbox", "tests", "tools",
)

#: The directory a committed order lives in. An order naming it as its own
#: authority would be authorizing itself.
OWN_ORDERS_DIR = "engine/bridge/orders/"

#: Command prefixes a work order never allows: every git/gh verb that writes a
#: ref, the index, the worktree or the remote. Committing, pushing and opening
#: pull requests are separately authorized steps, not something an order grants.
MUTATING_COMMANDS = (
    "git push", "git commit", "git add", "git rm", "git mv", "git reset",
    "git rebase", "git merge", "git cherry-pick", "git revert", "git restore",
    "git checkout", "git switch", "git clean", "git stash", "git apply",
    "git am", "git tag", "git update-ref", "git symbolic-ref", "git filter",
    "git replace", "git notes", "git worktree", "git submodule", "git gc",
    "git prune", "git reflog", "git config", "git remote", "git fetch",
    "git pull", "git init", "git clone", "gh pr", "gh api", "gh release",
    "gh repo", "gh workflow", "gh run", "gh auth", "gh secret", "gh variable",
)

#: The five identity fields of a governing source. All required.
SOURCE_FIELDS = ("drive_id", "sha256", "bytes", "extraction_rule", "revision_if_native")

_TOP_KEYS = (
    "schema", "record_kind", "status", "task_id", "repository", "authorization",
    "governing_sources", "claim", "scope", "limits", "public_disclosure",
    "expected_old_output_head", "scientific_status_change_authorized",
    "does_not_establish",
)
_REPO_KEYS = ("id", "full_name", "base_commit_sha")
_AUTH_KEYS = (
    "approved_principal_ids", "source_ref", "verification_status",
    "authorized_outside_this_repository", "authorizing_record_drive_id",
    "policy_change_authorized_by",
)
_CLAIM_KEYS = ("work_events_event_id", "lease_until_utc")
_SCOPE_KEYS = ("objective", "allowed_paths", "allowed_commands",
               "acceptance_tests", "required_negative_controls")
_LIMIT_KEYS = ("wall_time_seconds", "memory_mib", "network_policy")
_DISCLOSURE_KEYS = ("approved", "approved_paths")

# ``engine/bridge/orders`` in any spelling: case, repeated slashes, ``./``
# segments, backslashes (folded before matching), inside free text.
_OWN_ORDERS_RE = re.compile(
    r"(?<![a-z0-9_])engine/+(?:\./+)*bridge/+(?:\./+)*orders(?![a-z0-9_])")

_REPO_PATH_RE = re.compile(
    r"(?<![a-z0-9_.-])(?:\./)?(?:"
    + "|".join(re.escape(t.casefold()) + (r"(?:/|$)" if "." not in t else r"(?![a-z0-9_-])")
               for t in REPOSITORY_TOP_LEVEL)
    + r")")
_PR_RE = re.compile(r"github\.com|/pull/|pull[ _-]request|merge[ _-]request|\bpr\s*#\s*\d|\bpr\s+\d")
_FILE_NAME_RE = re.compile(r"\.(?:py|json|jsonl|md|yml|yaml|txt|csv|patch|diff|toml|cfg|ini|sh)$")

# Lexical write markers, matched before a surface mention in a command.
_WRITE_TOKEN_RE = re.compile(
    r">|\bopen\(|\bwrite|\.write\b|write_text|write_bytes|\bsed\s+-[a-z]*i|\bperl\s+-[a-z]*i"
    r"|\bof=|shutil|unlink|\brename|\.replace\(|\bcurl\b[^|;&]*\s-o\b"
    r"|(?<![a-z0-9_.-])(?:rm|cp|mv|tee|ln|touch|mkdir|rsync|patch|install|chmod|chown"
    r"|dd|truncate|wget|tar|unzip|git|gh)\s")


# ---------------------------------------------------------------------------
# canonical form and digest
# ---------------------------------------------------------------------------

def canonical_body(obj: Mapping[str, Any]) -> Dict[str, Any]:
    """The hashed part of a work order: everything but ``work_order_digest``."""
    return {k: v for k, v in obj.items() if k != "work_order_digest"}


def work_order_digest(obj: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical body. Byte identity, nothing more."""
    return body_digest(canonical_body(obj))


def with_digest(body: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of ``body`` with ``work_order_digest`` attached (in memory)."""
    b = canonical_body(body)
    return {**b, "work_order_digest": body_digest(b)}


# ---------------------------------------------------------------------------
# scope paths and surfaces
# ---------------------------------------------------------------------------

def _surface_key(surface: str) -> str:
    return surface.casefold().rstrip("/")


def _touches(norm: str, surface: str) -> bool:
    """Does the normalized path ``norm`` reach ``surface``?

    Directly (equal, or inside a surface directory), or from above (a
    directory that contains the surface, or ``.`` for the whole repository).
    """
    s = _surface_key(surface)
    if norm == ".":
        return True
    return norm == s or norm.startswith(s + "/") or s.startswith(norm + "/")


def malformed_scope_paths(paths: Any) -> List[Tuple[str, str]]:
    """``[(entry, reason)]`` for every entry that cannot be normalized."""
    if not is_str_list(paths):
        return []
    out: List[Tuple[str, str]] = []
    for p in paths:
        norm, reason = normalize_scope_path(p)
        if norm is None:
            out.append((p, reason))
    return out


def _surfaces_touched(paths: Any, surfaces: Tuple[str, ...]) -> List[str]:
    if not is_str_list(paths):
        return []
    norms: List[str] = []
    for p in paths:
        norm, _ = normalize_scope_path(p)
        if norm is None:
            return list(surfaces)  # an unnormalizable entry reaches everything
        norms.append(norm)
    return [s for s in surfaces if any(_touches(n, s) for n in norms)]


def policy_surfaces_touched(allowed_paths: Any) -> List[str]:
    """Policy surfaces reached by any entry of ``allowed_paths``, in order."""
    return _surfaces_touched(allowed_paths, POLICY_SURFACES)


def protected_surfaces_touched(allowed_paths: Any) -> List[str]:
    """Protected surfaces reached by any entry of ``allowed_paths``, in order."""
    return _surfaces_touched(allowed_paths, PROTECTED_SURFACES)


def path_within_scope(path: Any, allowed_paths: Any) -> bool:
    """Is ``path`` equal to or inside some entry of ``allowed_paths``?

    Both sides normalized; a malformed path or a malformed scope entry is
    never within scope. ``.`` as a scope entry is refused elsewhere and is
    not honoured here.
    """
    norm, _ = normalize_scope_path(path)
    if norm is None or norm == "." or not is_str_list(allowed_paths):
        return False
    for entry in allowed_paths:
        e, _ = normalize_scope_path(entry)
        if e is None or e == ".":
            continue
        if norm == e or norm.startswith(e + "/"):
            return True
    return False


def _names_own_orders(ref: Any) -> bool:
    if not isinstance(ref, str):
        return False
    return bool(_OWN_ORDERS_RE.search(ref.replace("\\", "/").casefold()))


def names_repository_path(ref: Any) -> Optional[str]:
    """Why ``ref`` is a repository path, a file name or a pull request, or None.

    A Drive id and a Drive URL are never repository paths. Anything that names
    a top-level entry of this repository, ends in a source/data file
    extension, or points at GitHub or a pull request is: none of those is an
    owner-side record, and a work order cannot be its own authority.
    """
    if not isinstance(ref, str):
        return None
    if is_drive_id(ref) or is_drive_url(ref):
        return None
    s = ref.replace("\\", "/").casefold().strip()
    if _names_own_orders(s):
        return f"names a path under {OWN_ORDERS_DIR}"
    if _PR_RE.search(s):
        return "names GitHub or a pull request; a PR is what an order governs, not what authorizes it"
    if _REPO_PATH_RE.search(s):
        return "names a path in this repository"
    if _FILE_NAME_RE.search(s):
        return "is a file name, not a record"
    return None


def command_problems(cmd: Any) -> List[str]:
    """Why an allowed command is refused (each reason is a full sentence)."""
    p: List[str] = []
    if not check_evidence_string(cmd, "command", p, min_len=2):
        return p
    folded = " ".join(cmd.casefold().split())
    for verb in MUTATING_COMMANDS:
        if re.search(r"(?<![a-z0-9_.-])" + re.escape(verb) + r"(?![a-z0-9_-])", folded):
            p.append(f"command {cmd!r} runs {verb!r}: a work order does not authorize a git "
                     "or gh mutation; committing and pushing are separately authorized steps")
            break
    for surface in PROTECTED_SURFACES + POLICY_SURFACES:
        key = _surface_key(surface)
        pat = r"(?<![a-z0-9_./-])" + re.escape(key) + (r"(?:/|(?![a-z0-9_.-]))")
        for m in re.finditer(pat, folded):
            if _WRITE_TOKEN_RE.search(folded[:m.start()]):
                p.append(f"command {cmd!r} writes into {surface!r} (a write marker precedes "
                         "the path); a surface is not written by a work order's command")
                break
        else:
            continue
        break
    return p


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_work_order(obj: Any) -> List[str]:
    """Every problem with a work order, as a list of strings. Never raises.

    Empty list means the record is well-formed. It does not mean the work is
    authorized (the Drive record it names is what authorizes), permitted
    (allowed paths are recorded, not granted) or executed.
    """
    p: List[str] = []
    if not isinstance(obj, dict):
        return ["work order is not a JSON object"]

    # closed top level, plus the digest which sits beside the body
    top = check_keys({k: v for k, v in obj.items() if k != "work_order_digest"},
                     _TOP_KEYS, "work_order", p)
    if "work_order_digest" not in obj:
        p.append("work_order.work_order_digest is missing")
    elif not is_sha256(obj["work_order_digest"]):
        p.append(f"work_order_digest {obj['work_order_digest']!r} is not a sha256 hex digest")
    else:
        recomputed = work_order_digest(obj)
        if recomputed != obj["work_order_digest"]:
            p.append(
                f"work_order_digest mismatch: stored {obj['work_order_digest'][:16]}..., "
                f"recomputed {recomputed[:16]}... -- this order was edited after its "
                "digest was taken; a work order is frozen, not maintained")
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
    elif "/" in obj["task_id"] or obj["task_id"] != obj["task_id"].strip():
        p.append(f"task_id {obj['task_id']!r} must be a bare identifier (it names the file)")

    # rule 1: no work order can authorize a status change
    ssca = obj["scientific_status_change_authorized"]
    if ssca is not False:
        p.append(
            f"scientific_status_change_authorized is {ssca!r}; the only permitted "
            "value is false. No work order can authorize a change of scientific "
            "status -- only an operator decision under governance/ can.")

    # repository
    repo = check_keys(obj["repository"], _REPO_KEYS, "repository", p)
    if repo is not None:
        if not is_int(repo["id"]) or repo["id"] <= 0:
            p.append(f"repository.id {repo['id']!r} must be the numeric repository id")
        if not is_nonempty_str(repo["full_name"]) or repo["full_name"].count("/") != 1:
            p.append(f"repository.full_name {repo['full_name']!r} must be owner/name")
        if repo["base_commit_sha"] is not None and not is_commit(repo["base_commit_sha"]):
            p.append(f"repository.base_commit_sha {repo['base_commit_sha']!r} is not a "
                     "40-hex commit sha or null")

    # authorization (rules 2 and 3)
    auth = check_keys(obj["authorization"], _AUTH_KEYS, "authorization", p)
    verification = None
    if auth is not None:
        verification = auth["verification_status"]
        if verification not in VERIFICATION_STATUSES:
            p.append(f"authorization.verification_status {verification!r} is not one of "
                     f"{list(VERIFICATION_STATUSES)}")
        if not is_str_list(auth["approved_principal_ids"]) or \
                any(not s.strip() for s in auth["approved_principal_ids"]):
            p.append("authorization.approved_principal_ids must be a list of non-empty strings")
        if auth["source_ref"] is not None and not is_nonempty_str(auth["source_ref"]):
            p.append("authorization.source_ref must be a non-empty string or null")
        if auth["authorized_outside_this_repository"] is not True:
            p.append(
                "authorization.authorized_outside_this_repository must be true: a work "
                "order created or edited by the PR it authorizes cannot authorize itself")
        for field in ("source_ref", "authorizing_record_drive_id", "policy_change_authorized_by"):
            ref = auth[field]
            why = names_repository_path(ref)
            if why is not None:
                p.append(f"authorization.{field} {ref!r} {why}: a work order cannot be "
                         "its own authority")
            elif field != "source_ref" and ref is not None and not is_drive_id(ref):
                p.append(f"authorization.{field} {ref!r} is not a Drive id "
                         "(a repository path is not an authorizing record)")
        if verification == VERIFIED_BY_OWNER_BOUNDARY:
            if not is_drive_id(auth["authorizing_record_drive_id"]):
                p.append("authorization.verification_status is VERIFIED_BY_OWNER_BOUNDARY "
                         "but authorizing_record_drive_id names no Drive record")
            if not auth["approved_principal_ids"]:
                p.append("authorization.verification_status is VERIFIED_BY_OWNER_BOUNDARY "
                         "but approved_principal_ids is empty")
            if auth["source_ref"] is None:
                p.append("authorization.verification_status is VERIFIED_BY_OWNER_BOUNDARY "
                         "but source_ref is null")
            elif not (is_drive_id(auth["source_ref"]) or is_drive_url(auth["source_ref"])):
                p.append(f"authorization.source_ref {auth['source_ref']!r} is neither a Drive "
                         "id nor a Drive URL; a VERIFIED_BY_OWNER_BOUNDARY order names the "
                         "owner-side record its authorization came from, not free text, a "
                         "repository path or a pull request")
        if verification == NOT_VERIFIED and status == STATUS_EXECUTABLE:
            p.append(
                "status is EXECUTABLE while authorization.verification_status is "
                "NOT_VERIFIED: an unverified work order is PREPARED-only and cannot "
                "be marked executable")
        if kind == RECORD_EXAMPLE and verification != NOT_VERIFIED:
            p.append("record_kind is EXAMPLE but verification_status is not NOT_VERIFIED: "
                     "an example is not an authorization")
    if kind == RECORD_EXAMPLE and status == STATUS_EXECUTABLE:
        p.append("record_kind is EXAMPLE but status is EXECUTABLE: an example is never executable")

    # governing sources (rule 4)
    sources = obj["governing_sources"]
    if not isinstance(sources, list):
        p.append("governing_sources is not a list")
        sources = []
    for i, src in enumerate(sources):
        s = check_keys(src, SOURCE_FIELDS, f"governing_sources[{i}]", p)
        if s is None:
            continue
        if not is_drive_id(s["drive_id"]):
            p.append(f"governing_sources[{i}].drive_id {s['drive_id']!r} is not a Drive id")
        if not is_sha256(s["sha256"]):
            p.append(f"governing_sources[{i}].sha256 {s['sha256']!r} is not a sha256 hex digest")
        if not is_int(s["bytes"]) or s["bytes"] < 0:
            p.append(f"governing_sources[{i}].bytes {s['bytes']!r} is not a byte count")
        if not is_nonempty_str(s["extraction_rule"]):
            p.append(f"governing_sources[{i}].extraction_rule is empty; a source without "
                     "an extraction rule has no identity")
        rev = s["revision_if_native"]
        if rev is not None and not is_nonempty_str(rev):
            p.append(f"governing_sources[{i}].revision_if_native must be a non-empty "
                     "string or null")

    # claim
    claim = check_keys(obj["claim"], _CLAIM_KEYS, "claim", p)
    if claim is not None:
        ev = claim["work_events_event_id"]
        if ev is not None and not is_nonempty_str(ev):
            p.append("claim.work_events_event_id must be a non-empty string or null")
        lease = claim["lease_until_utc"]
        if lease is not None and not is_timestamp(lease):
            p.append(f"claim.lease_until_utc {lease!r} is not a UTC timestamp or null")

    # scope (rules 5 and 6)
    scope = check_keys(obj["scope"], _SCOPE_KEYS, "scope", p)
    if scope is not None:
        if scope["objective"] is not None and not is_nonempty_str(scope["objective"]):
            p.append("scope.objective must be a non-empty string or null")
        for k in ("allowed_paths", "allowed_commands", "acceptance_tests",
                  "required_negative_controls"):
            if not is_str_list(scope[k]):
                p.append(f"scope.{k} must be a list of strings")
        if is_str_list(scope["allowed_commands"]):
            for i, cmd in enumerate(scope["allowed_commands"]):
                for msg in command_problems(cmd):
                    p.append(f"scope.allowed_commands[{i}]: {msg}")
        if is_str_list(scope["required_negative_controls"]):
            for i, nc in enumerate(scope["required_negative_controls"]):
                check_evidence_string(nc, f"scope.required_negative_controls[{i}]", p)
        if is_str_list(scope["acceptance_tests"]):
            for i, t in enumerate(scope["acceptance_tests"]):
                check_evidence_string(t, f"scope.acceptance_tests[{i}]", p, min_len=2)

    disclosure = check_keys(obj["public_disclosure"], _DISCLOSURE_KEYS, "public_disclosure", p)
    if disclosure is not None:
        if not isinstance(disclosure["approved"], bool):
            p.append("public_disclosure.approved must be a boolean")
        if not is_str_list(disclosure["approved_paths"]):
            p.append("public_disclosure.approved_paths must be a list of strings")
        elif disclosure["approved"] is False and disclosure["approved_paths"]:
            p.append("public_disclosure.approved is false but approved_paths is not empty")

    allowed = scope["allowed_paths"] if scope is not None else None
    if is_str_list(allowed):
        for entry, reason in malformed_scope_paths(allowed):
            p.append(f"scope.allowed_paths entry {entry!r} {reason}; a scope path is a "
                     "literal repository-relative path and is refused, not normalized")
        protected = protected_surfaces_touched(allowed)
        if protected:
            p.append(
                f"scope.allowed_paths reaches protected surface(s) {protected}: the bridge, "
                "the engine records, the Drive exports, the checkers, this bridge's negative "
                "controls, the quarantine list and the agent instructions are never in a "
                "work order's scope, whatever references it carries")
        touched = policy_surfaces_touched(allowed)
        if touched:
            approved = disclosure is not None and disclosure["approved"] is True
            policy_ref = auth["policy_change_authorized_by"] if auth is not None else None
            if not approved or not is_drive_id(policy_ref):
                p.append(
                    f"scope.allowed_paths touches policy/status surface(s) {touched} "
                    "without both public_disclosure.approved and a Drive id in "
                    "authorization.policy_change_authorized_by. Those surfaces carry "
                    "policy and status, not research code.")
    if disclosure is not None and is_str_list(disclosure["approved_paths"]):
        for entry, reason in malformed_scope_paths(disclosure["approved_paths"]):
            p.append(f"public_disclosure.approved_paths entry {entry!r} {reason}; refused, "
                     "not normalized")
        protected = protected_surfaces_touched(disclosure["approved_paths"])
        if protected:
            p.append(f"public_disclosure.approved_paths reaches protected surface(s) "
                     f"{protected}; nothing there is disclosed under a work order")
        if is_str_list(allowed):
            for entry in disclosure["approved_paths"]:
                if not path_within_scope(entry, allowed):
                    p.append(f"public_disclosure.approved_paths entry {entry!r} is not within "
                             "scope.allowed_paths; nothing outside the scope can be disclosed")

    # limits
    limits = check_keys(obj["limits"], _LIMIT_KEYS, "limits", p)
    if limits is not None:
        for k in ("wall_time_seconds", "memory_mib"):
            v = limits[k]
            if v is not None and (not is_int(v) or v <= 0):
                p.append(f"limits.{k} {v!r} must be a positive integer or null")
        np_ = limits["network_policy"]
        if np_ is not None and np_ not in NETWORK_POLICIES:
            p.append(f"limits.network_policy {np_!r} is not one of {list(NETWORK_POLICIES)} or null")

    head = obj["expected_old_output_head"]
    if head is not None and not is_nonempty_str(head):
        p.append("expected_old_output_head must be a non-empty string or null")

    check_does_not_establish(obj["does_not_establish"], "does_not_establish", p)

    # the strict branch: what an EXECUTABLE order must carry
    if status == STATUS_EXECUTABLE:
        if repo is not None and not is_commit(repo["base_commit_sha"]):
            p.append("status is EXECUTABLE but repository.base_commit_sha is not pinned")
        if not sources:
            p.append("status is EXECUTABLE but governing_sources is empty")
        if claim is not None:
            if claim["work_events_event_id"] is None:
                p.append("status is EXECUTABLE but claim.work_events_event_id is null "
                         "(the R17 Work Events register is the claim log)")
            if claim["lease_until_utc"] is None:
                p.append("status is EXECUTABLE but claim.lease_until_utc is null")
        if scope is not None:
            if not is_nonempty_str(scope["objective"]):
                p.append("status is EXECUTABLE but scope.objective is empty")
            for k in ("allowed_paths", "allowed_commands", "acceptance_tests",
                      "required_negative_controls"):
                if not scope[k]:
                    p.append(f"status is EXECUTABLE but scope.{k} is empty")
        if limits is not None and any(limits[k] is None for k in _LIMIT_KEYS):
            p.append("status is EXECUTABLE but limits are not all set")

    return p


def work_order_notes(obj: Any) -> List[str]:
    """Facts the validator RECORDS about a well-formed order without granting.

    A policy surface in scope with both references present is noted here; it
    is not a problem and it is not a permission. The checker prints these as
    notes, and only for an order that validated.
    """
    notes: List[str] = []
    if not isinstance(obj, dict):
        return notes
    tid = str(obj.get("task_id") or "?")
    scope = obj.get("scope") if isinstance(obj.get("scope"), dict) else {}
    auth = obj.get("authorization") if isinstance(obj.get("authorization"), dict) else {}
    disc = obj.get("public_disclosure") if isinstance(obj.get("public_disclosure"), dict) else {}
    touched = policy_surfaces_touched(scope.get("allowed_paths"))
    if touched and disc.get("approved") is True and is_drive_id(auth.get("policy_change_authorized_by")):
        notes.append(
            f"{tid}: allowed_paths touch policy/status surface(s) {touched}; "
            f"public_disclosure.approved and policy_change_authorized_by="
            f"{auth.get('policy_change_authorized_by')} are RECORDED. Nothing is granted "
            "by this validator; the change requires the owner-side authorization the "
            "reference names.")
    if auth.get("verification_status") == VERIFIED_BY_OWNER_BOUNDARY:
        notes.append(
            f"{tid}: verification_status VERIFIED_BY_OWNER_BOUNDARY is "
            f"transcribed from authorizing record {auth.get('authorizing_record_drive_id')}; "
            "this repository did not verify it and cannot.")
    return notes


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load_work_order(path: str) -> Dict[str, Any]:
    """Read one work-order file as a plain dict (the only file access here).

    Duplicate keys and NaN/Infinity in the text raise ``ValueError``.
    """
    return load_json_strict(path)


def to_json(obj: Mapping[str, Any]) -> str:
    """Pretty, sorted, trailing newline: how a committed order is formatted.

    The checker refuses a committed file whose bytes are not this
    serialisation of its own parsed content: the text on disk is the record.
    """
    return json.dumps(dict(obj), indent=2, ensure_ascii=False, sort_keys=True) + "\n"
