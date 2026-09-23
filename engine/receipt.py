#!/usr/bin/env python3
"""The receipt schema and its append-only writer.

A **receipt** is a record that a computation in this repository ran: which lane
it was run under, which module and entry point were called, with which
arguments, at which commit, when, for how long, what numbers came out and --
mandatorily -- what those numbers do **not** establish.

WHAT A RECEIPT IS NOT
---------------------
A receipt is **not evidence**. It is not a proof, not a certificate, not a
review verdict, not a status and not a promotion. `engine/README.md` says this
of the whole directory and it is repeated here because this module is where a
receipt is manufactured:

    Neither a lane nor a receipt is evidence. A green run is a run, not a
    proof.

Accordingly the schema has a ``verdict`` field whose **only** permitted value
is ``NO-MATHEMATICAL-VERDICT``, and a ``status_effect`` field whose only
permitted value says that nothing changed. They are not decoration: they give
``tools/receipts_check.py`` a fixed place to police, so that a hand-edited
receipt claiming a discharge, a closure or a promotion is a schema violation
rather than a matter of taste. ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND``,
``OBL-H5-REMOTE-THRESHOLD``, ``OBL-D1-PROMOTE`` and **both** Pieces of
``D3-LEMMA-RN-UNIF`` are OPEN, and no receipt this module can produce is
capable of saying otherwise.

``does_not_establish`` is mandatory, is validated non-empty, is rejected when
it is a placeholder (``n/a``, ``none``, ``tbd``, ...) and is rejected when it is
too short to be a sentence. A receipt that does not say what it fails to
establish **is refused by the writer**.

APPEND-ONLY, STRUCTURALLY
-------------------------
Receipts follow the append-only pattern the repository already tests for
``registers/json/work_events.json`` (``tests/test_registers.py``). Here it is
enforced three ways, none of them a convention:

1. The only file-creating call in this module is ``open(path, "x")`` --
   exclusive creation. There is no code path that opens a receipt for writing
   or truncation, and no code path in this module deletes or renames anything.
   Writing over an existing receipt raises :class:`AppendOnlyViolation`.
2. Every receipt carries the SHA-256 of its own canonical JSON body, so a later
   edit is detectable by recomputation alone, with no reference copy needed.
   ``tools/receipts_check.py`` recomputes every one.
3. The writer refuses any destination outside the receipts root, and refuses
   outright to write anywhere under ``engine/lanes/`` or ``claims/`` --- see
   :data:`FORBIDDEN_WRITE_ROOTS`. ``engine/run.py`` imports this writer and no
   other, so "a run records, it does not decide" is a property of the code
   rather than a promise in a docstring.

PROVENANCE OF NUMBERS
---------------------
Every numeric result carries one of three provenances, and nothing else is
accepted:

``certified_interval``
    A two-sided enclosure from ``research/interval/``, whose contract is that
    containment is unconditional. ``lo`` and ``hi`` are exact rationals.
``exact_rational``
    A ``fractions.Fraction`` computed in exact arithmetic.
``float_noncertifying``
    A binary float, an ``mpmath`` value, a Monte Carlo estimate, a fitted
    exponent, a dense sampling, a display or a probe. **This is not a bound.**
    High precision is not certification; the sources say so themselves. Any
    result on this path is labelled NON-CERTIFYING in the code and in the
    receipt, and :attr:`NumericResult.certifying` is False for it.

``runtime_seconds`` is outside this scheme on purpose: it is a wall-clock
measurement of the machine that ran, NON-CERTIFYING and a bound on nothing
mathematical. It is stored as fixed three-decimal text so that no float ever
enters the canonical body, and it appears in no result list.

Standard library only (``dataclasses``, ``fractions``, ``hashlib``, ``json``,
``os``, ``re``, ``subprocess``, ``datetime``). Python 3.11.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "SCHEMA", "OUTCOMES", "PROVENANCES", "VERDICT_NONE", "ALLOWED_VERDICTS",
    "STATUS_EFFECT", "FORBIDDEN_STATUS_WORDS", "VERDICT_FIELDS",
    "RECEIPTS_ROOT", "FORBIDDEN_WRITE_ROOTS",
    "OUTCOME_RAN", "OUTCOME_FAILED", "OUTCOME_UNAVAILABLE", "OUTCOME_SKIPPED",
    "OUTCOME_DRY_RUN",
    "CERTIFIED_INTERVAL", "EXACT_RATIONAL", "FLOAT_NONCERTIFYING",
    "NumericResult", "Receipt",
    "ReceiptRejected", "AppendOnlyViolation", "ForbiddenDestination",
    "canonical_json", "body_sha256", "argument_digest",
    "new_receipt_id", "repository_commit", "utc_now",
    "validate_body", "validate_receipt_object", "scan_status_words",
    "write_receipt", "receipt_path", "load_receipt", "iter_receipt_files",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECEIPTS_ROOT = os.path.join(ROOT, "engine", "receipts")

#: Destinations this module refuses to write to under any circumstances, even
#: if a caller passes one as ``receipts_root``. ``engine/lanes/`` carries the
#: transcribed statuses and ``claims/graph.json`` carries the claim graph; a
#: runner that could write either could decide something. It cannot.
FORBIDDEN_WRITE_ROOTS = (
    os.path.join(ROOT, "engine", "lanes"),
    os.path.join(ROOT, "claims"),
    os.path.join(ROOT, "registers"),
    os.path.join(ROOT, "drive"),
)

SCHEMA = "q0.engine.receipt/v1"

OUTCOME_RAN = "RAN"
OUTCOME_FAILED = "FAILED"
OUTCOME_UNAVAILABLE = "UNAVAILABLE"
OUTCOME_SKIPPED = "SKIPPED"
OUTCOME_DRY_RUN = "DRY_RUN"

#: What happened to the *process*. Deliberately free of any word that could be
#: read as a mathematical judgement.
OUTCOMES = (
    OUTCOME_RAN,
    OUTCOME_FAILED,
    OUTCOME_UNAVAILABLE,
    OUTCOME_SKIPPED,
    OUTCOME_DRY_RUN,
)

CERTIFIED_INTERVAL = "certified_interval"
EXACT_RATIONAL = "exact_rational"
FLOAT_NONCERTIFYING = "float_noncertifying"
PROVENANCES = (CERTIFIED_INTERVAL, EXACT_RATIONAL, FLOAT_NONCERTIFYING)

#: The only verdict a receipt may carry. See the module docstring.
VERDICT_NONE = "NO-MATHEMATICAL-VERDICT"
ALLOWED_VERDICTS = (VERDICT_NONE,)

STATUS_EFFECT = (
    "NONE. This receipt records that a computation ran. It promotes, closes, "
    "discharges and reclassifies nothing. Only an operator decision under "
    "governance/ can change a mathematical status."
)

#: Words a verdict field may never contain. The check is word-boundary based
#: and applies ONLY to the fields in :data:`VERDICT_FIELDS`, never to prose:
#: ``does_not_establish`` legitimately contains "does not close Piece 2", and
#: an entry point may legitimately be named ``radial_gaussian_closed_form``.
FORBIDDEN_STATUS_WORDS = (
    "discharge", "discharged", "discharges",
    "promote", "promoted", "promotes", "promotion",
    "close", "closed", "closes", "closure",
    "reclassify", "reclassified",
    "proven", "proved", "certified",
    "resolved", "solved",
)

#: The fields :data:`FORBIDDEN_STATUS_WORDS` is scanned in. A closed set, so
#: the scan can never accidentally fire on a docstring or a caveat.
VERDICT_FIELDS = ("outcome", "verdict", "status_effect")

_PLACEHOLDERS = frozenset({
    "", "-", ".", "n/a", "na", "n.a.", "none", "null", "nil", "tbd", "todo",
    "unknown", "pending", "see above", "as above", "same", "nothing",
})

#: A ``does_not_establish`` shorter than this is not a sentence, and the field
#: is the most load-bearing one in the schema (CLAUDE.md rule 2).
MIN_DOES_NOT_ESTABLISH = 40

_RECEIPT_ID_RE = re.compile(r"^RCPT-[A-Za-z0-9]+-[0-9]{8}T[0-9]+Z-[0-9a-f]{8}$")
_LANE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^([0-9a-f]{40}|unknown)$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_RUNTIME_RE = re.compile(r"^\d+\.\d{3}$")


class ReceiptRejected(ValueError):
    """The writer refused a receipt: it does not satisfy the schema."""


class AppendOnlyViolation(ReceiptRejected):
    """A write would have modified or replaced an existing receipt."""


class ForbiddenDestination(ReceiptRejected):
    """A write was aimed outside the receipts root, or at a governed path."""


# ---------------------------------------------------------------------------
# canonical form and digests
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    """The canonical serialisation a receipt body is hashed in.

    Sorted keys, no insignificant whitespace, UTF-8. Every number in a receipt
    body is stored as a *string* (exact rationals as ``"num/den"``, runtimes as
    fixed three-decimal text), so no float ever reaches this function and the
    canonical form is byte-stable across platforms and Python builds.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def body_sha256(body: Mapping[str, Any]) -> str:
    """SHA-256 of a receipt body in canonical form."""
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def argument_digest(arguments: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical form of a call's arguments.

    Two runs with the same digest were called the same way. It is a labelling
    device for receipts, not a hash of any mathematical content.
    """
    return hashlib.sha256(canonical_json(dict(arguments)).encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Current UTC time, ``YYYY-MM-DDTHH:MM:SS.ffffffZ``."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def new_receipt_id(lane: str, module: str, entry_point: str,
                   arg_digest: str, timestamp: str) -> str:
    """A receipt id: ``RCPT-<lane>-<compact UTC stamp>-<8 hex>``."""
    stamp = re.sub(r"[-:.]", "", timestamp)
    stamp = stamp[:8] + "T" + stamp[9:].rstrip("Z") + "Z"
    tag = hashlib.sha256(
        "|".join((lane, module, entry_point, arg_digest, timestamp)).encode("utf-8")
    ).hexdigest()[:8]
    return f"RCPT-{lane}-{stamp}-{tag}"


def repository_commit(root: str = ROOT) -> Tuple[str, bool]:
    """``(commit, dirty)`` for the working tree, or ``("unknown", True)``.

    Read-only: ``git rev-parse`` and ``git status --porcelain`` and nothing
    else. A receipt taken from a dirty tree says so, because the commit alone
    would then not identify the code that ran.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return "unknown", True
        commit = head.stdout.strip()
        st = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                            capture_output=True, text=True, timeout=60)
        dirty = st.returncode != 0 or bool(st.stdout.strip())
        return (commit if _COMMIT_RE.match(commit) else "unknown"), dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", True


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericResult:
    """One number a run produced, with the provenance that qualifies it.

    ``value`` is a string in every case. Exact rationals render as
    ``"num/den"`` via ``str(Fraction)``; a float renders as ``repr(float)``
    and is marked NON-CERTIFYING. ``lo``/``hi`` are the exact rational
    endpoints of an enclosure and are required when the provenance is
    ``certified_interval``.
    """

    name: str
    provenance: str
    value: Optional[str] = None
    lo: Optional[str] = None
    hi: Optional[str] = None
    note: str = ""

    @property
    def certifying(self) -> bool:
        """False on the float path. A float is never a certified bound."""
        return self.provenance != FLOAT_NONCERTIFYING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provenance": self.provenance,
            "certifying": self.certifying,
            "value": self.value,
            "lo": self.lo,
            "hi": self.hi,
            "note": self.note,
        }

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_interval(cls, name: str, interval: Any, note: str = "") -> "NumericResult":
        """A certified enclosure from ``research/interval``.

        Duck-typed on ``.lo`` / ``.hi`` so this module keeps no dependency on
        the interval package; the endpoints must be exact rationals.
        """
        lo, hi = Fraction(interval.lo), Fraction(interval.hi)
        if lo > hi:
            raise ReceiptRejected(f"{name}: enclosure has lo > hi")
        return cls(name=name, provenance=CERTIFIED_INTERVAL,
                   value=None, lo=str(lo), hi=str(hi), note=note)

    @classmethod
    def from_fraction(cls, name: str, x: Fraction, note: str = "") -> "NumericResult":
        """An exactly computed rational."""
        return cls(name=name, provenance=EXACT_RATIONAL,
                   value=str(Fraction(x)), note=note)

    @classmethod
    def from_int(cls, name: str, n: int, note: str = "") -> "NumericResult":
        """A count. Exact, hence an exact rational with denominator 1."""
        return cls(name=name, provenance=EXACT_RATIONAL,
                   value=str(Fraction(int(n))), note=note)

    @classmethod
    def from_float(cls, name: str, x: float, note: str = "") -> "NumericResult":
        """A float. **NON-CERTIFYING**: this is not a bound on anything.

        The label is forced into the note so that it cannot be dropped by a
        caller who forgets, and :attr:`certifying` is False.
        """
        prefix = "NON-CERTIFYING (binary float; not a bound)."
        full = prefix if not note else f"{prefix} {note}"
        return cls(name=name, provenance=FLOAT_NONCERTIFYING,
                   value=repr(float(x)), note=full)


# ---------------------------------------------------------------------------
# the receipt
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Receipt:
    """One execution, recorded. See the module docstring for what it is not."""

    receipt_id: str
    lane: str
    module: str
    entry_point: str
    arguments: Dict[str, Any]
    arg_digest: str
    commit: str
    dirty: bool
    timestamp_utc: str
    outcome: str
    #: Wall-clock seconds, fixed three decimals. A NON-CERTIFYING measurement
    #: of the machine that ran; it bounds nothing mathematical.
    runtime_seconds: str
    does_not_establish: str
    results: Tuple[NumericResult, ...] = ()
    notes: Tuple[str, ...] = ()
    error: Optional[str] = None
    verdict: str = VERDICT_NONE
    status_effect: str = STATUS_EFFECT
    schema: str = SCHEMA

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, *, lane: str, module: str, entry_point: str,
              arguments: Mapping[str, Any], outcome: str,
              runtime_seconds: float, does_not_establish: str,
              results: Iterable[NumericResult] = (),
              notes: Iterable[str] = (), error: Optional[str] = None,
              commit: Optional[str] = None, dirty: Optional[bool] = None,
              timestamp: Optional[str] = None,
              root: str = ROOT) -> "Receipt":
        """Assemble a receipt, filling id, digest, commit and timestamp."""
        args = dict(arguments)
        digest = argument_digest(args)
        stamp = timestamp or utc_now()
        if commit is None or dirty is None:
            c, d = repository_commit(root)
            commit = commit if commit is not None else c
            dirty = d if dirty is None else dirty
        return cls(
            receipt_id=new_receipt_id(lane, module, entry_point, digest, stamp),
            lane=lane, module=module, entry_point=entry_point,
            arguments=args, arg_digest=digest,
            commit=commit, dirty=bool(dirty), timestamp_utc=stamp,
            outcome=outcome,
            runtime_seconds=f"{max(float(runtime_seconds), 0.0):.3f}",
            does_not_establish=does_not_establish,
            results=tuple(results), notes=tuple(notes), error=error,
        )

    # -- serialisation -----------------------------------------------------

    def body(self) -> Dict[str, Any]:
        """The hashed part of the receipt: everything but ``body_sha256``."""
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "lane": self.lane,
            "run": {
                "module": self.module,
                "entry_point": self.entry_point,
                "arguments": self.arguments,
                "argument_digest": self.arg_digest,
            },
            "repository": {"commit": self.commit, "dirty": self.dirty},
            "timestamp_utc": self.timestamp_utc,
            "outcome": self.outcome,
            "verdict": self.verdict,
            "status_effect": self.status_effect,
            "runtime_seconds": self.runtime_seconds,
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
            "error": self.error,
            "does_not_establish": self.does_not_establish,
        }

    def to_dict(self) -> Dict[str, Any]:
        """The full receipt object, body plus its own SHA-256."""
        b = self.body()
        return {**b, "body_sha256": body_sha256(b)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False,
                          sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

_BODY_KEYS = {
    "schema", "receipt_id", "lane", "run", "repository", "timestamp_utc",
    "outcome", "verdict", "status_effect", "runtime_seconds", "results",
    "notes", "error", "does_not_establish",
}
_RUN_KEYS = {"module", "entry_point", "arguments", "argument_digest"}
_REPO_KEYS = {"commit", "dirty"}
_RESULT_KEYS = {"name", "provenance", "certifying", "value", "lo", "hi", "note"}


def scan_status_words(text: str) -> List[str]:
    """Forbidden status words present in ``text``, matched on word boundaries.

    Call this ONLY on the fields in :data:`VERDICT_FIELDS`. Run over prose it
    fires on every honest caveat, because an honest caveat is exactly where the
    words "does not close" and "does not discharge" belong.
    """
    low = str(text).lower()
    return [w for w in FORBIDDEN_STATUS_WORDS
            if re.search(r"\b" + re.escape(w) + r"\b", low)]


_word_scan = scan_status_words  # internal alias


def _fraction_or_none(s: Any) -> Optional[Fraction]:
    try:
        return Fraction(str(s))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def validate_body(body: Any) -> List[str]:
    """Every schema problem with a receipt body, as a list of strings.

    Empty list means the body satisfies the schema. This function is the one
    definition of the schema: the writer calls it before creating a file and
    ``tools/receipts_check.py`` calls it on every file on disk, so the two can
    never drift.
    """
    p: List[str] = []
    if not isinstance(body, dict):
        return ["receipt body is not a JSON object"]

    extra = set(body) - _BODY_KEYS
    missing = _BODY_KEYS - set(body)
    for k in sorted(extra):
        p.append(f"unexpected top-level field {k!r} (the schema is closed)")
    for k in sorted(missing):
        p.append(f"missing required field {k!r}")
    if missing:
        return p

    if body["schema"] != SCHEMA:
        p.append(f"schema is {body['schema']!r}, expected {SCHEMA!r}")

    rid = body["receipt_id"]
    if not isinstance(rid, str) or not _RECEIPT_ID_RE.match(rid):
        p.append(f"receipt_id {rid!r} is not a well-formed receipt id")

    lane = body["lane"]
    if not isinstance(lane, str) or not _LANE_RE.match(lane):
        p.append(f"lane {lane!r} is not a well-formed lane key")
    elif isinstance(rid, str) and not rid.startswith(f"RCPT-{lane}-"):
        p.append(f"receipt_id {rid!r} does not carry its own lane {lane!r}")

    run = body["run"]
    if not isinstance(run, dict):
        p.append("run is not an object")
    else:
        for k in sorted(_RUN_KEYS - set(run)):
            p.append(f"run.{k} is missing")
        for k in sorted(set(run) - _RUN_KEYS):
            p.append(f"unexpected field run.{k!r}")
        if _RUN_KEYS <= set(run):
            if not isinstance(run["module"], str) or not run["module"].strip():
                p.append("run.module is empty")
            if not isinstance(run["entry_point"], str) or not run["entry_point"].strip():
                p.append("run.entry_point is empty")
            if not isinstance(run["arguments"], dict):
                p.append("run.arguments is not an object")
            d = run["argument_digest"]
            if not isinstance(d, str) or not _SHA256_RE.match(d):
                p.append(f"run.argument_digest {d!r} is not a sha256 hex digest")
            elif isinstance(run["arguments"], dict):
                recomputed = argument_digest(run["arguments"])
                if recomputed != d:
                    p.append(
                        f"run.argument_digest does not match run.arguments "
                        f"(stored {d[:16]}..., recomputed {recomputed[:16]}...)")

    repo = body["repository"]
    if not isinstance(repo, dict):
        p.append("repository is not an object")
    else:
        for k in sorted(_REPO_KEYS - set(repo)):
            p.append(f"repository.{k} is missing")
        for k in sorted(set(repo) - _REPO_KEYS):
            p.append(f"unexpected field repository.{k!r}")
        if isinstance(repo.get("commit"), str) and not _COMMIT_RE.match(repo["commit"]):
            p.append(f"repository.commit {repo['commit']!r} is not a commit sha or 'unknown'")
        if "dirty" in repo and not isinstance(repo["dirty"], bool):
            p.append("repository.dirty is not a boolean")

    ts = body["timestamp_utc"]
    if not isinstance(ts, str) or not _TIMESTAMP_RE.match(ts):
        p.append(f"timestamp_utc {ts!r} is not an ISO-8601 UTC timestamp ending in Z")

    if body["outcome"] not in OUTCOMES:
        p.append(f"outcome {body['outcome']!r} is not one of {list(OUTCOMES)}")
    if body["verdict"] not in ALLOWED_VERDICTS:
        p.append(
            f"verdict {body['verdict']!r} is not permitted; the only permitted "
            f"verdict is {VERDICT_NONE!r} -- a receipt carries no mathematical verdict")
    if body["status_effect"] != STATUS_EFFECT:
        p.append("status_effect has been altered; it must be the fixed "
                 "constant stating that this receipt changes no status")

    # The forbidden-word scan, on the closed set of verdict fields only.
    for fieldname in VERDICT_FIELDS:
        val = body.get(fieldname)
        if not isinstance(val, str):
            continue
        if fieldname == "status_effect" and val == STATUS_EFFECT:
            continue  # the fixed constant names the words in order to refuse them
        hits = _word_scan(val)
        if hits:
            p.append(
                f"{fieldname} claims a status change: contains {sorted(set(hits))}. "
                "A receipt records a computation and may not promote, close, "
                "discharge or reclassify anything.")

    rt = body["runtime_seconds"]
    if not isinstance(rt, str) or not _RUNTIME_RE.match(rt):
        p.append(f"runtime_seconds {rt!r} must be a string with three decimals")

    results = body["results"]
    if not isinstance(results, list):
        p.append("results is not a list")
    else:
        if body.get("outcome") == OUTCOME_RAN and not results:
            p.append("outcome is RAN but no numeric results were recorded")
        for i, r in enumerate(results):
            if not isinstance(r, dict):
                p.append(f"results[{i}] is not an object")
                continue
            for k in sorted(_RESULT_KEYS - set(r)):
                p.append(f"results[{i}].{k} is missing")
            for k in sorted(set(r) - _RESULT_KEYS):
                p.append(f"unexpected field results[{i}].{k!r}")
            prov = r.get("provenance")
            if prov not in PROVENANCES:
                p.append(f"results[{i}].provenance {prov!r} is not one of {list(PROVENANCES)}")
                continue
            if not isinstance(r.get("name"), str) or not r["name"].strip():
                p.append(f"results[{i}].name is empty")
            expect_certifying = prov != FLOAT_NONCERTIFYING
            if r.get("certifying") is not expect_certifying:
                p.append(
                    f"results[{i}].certifying is {r.get('certifying')!r} but "
                    f"provenance {prov!r} forces {expect_certifying!r}")
            if prov == FLOAT_NONCERTIFYING:
                if "NON-CERTIFYING" not in str(r.get("note", "")):
                    p.append(
                        f"results[{i}] is on the float path but its note is not "
                        "labelled NON-CERTIFYING")
            if prov == CERTIFIED_INTERVAL:
                lo, hi = _fraction_or_none(r.get("lo")), _fraction_or_none(r.get("hi"))
                if lo is None or hi is None:
                    p.append(f"results[{i}] is a certified_interval without exact rational lo/hi")
                elif lo > hi:
                    p.append(f"results[{i}] has lo > hi, which is not an enclosure")
            if prov == EXACT_RATIONAL and _fraction_or_none(r.get("value")) is None:
                p.append(f"results[{i}].value {r.get('value')!r} is not an exact rational")

    notes = body["notes"]
    if not isinstance(notes, list) or any(not isinstance(n, str) for n in notes):
        p.append("notes must be a list of strings")
    if body["error"] is not None and not isinstance(body["error"], str):
        p.append("error must be a string or null")

    dne = body["does_not_establish"]
    if not isinstance(dne, str):
        p.append("does_not_establish must be a string")
    else:
        s = dne.strip()
        if not s:
            p.append(
                "does_not_establish is empty. A receipt that does not say what "
                "it fails to establish is rejected: that field is the most "
                "load-bearing one in the schema.")
        elif s.lower() in _PLACEHOLDERS:
            p.append(f"does_not_establish is the placeholder {s!r}, not a statement")
        elif len(s) < MIN_DOES_NOT_ESTABLISH:
            p.append(
                f"does_not_establish is {len(s)} characters, shorter than the "
                f"{MIN_DOES_NOT_ESTABLISH} required for it to be a statement")

    return p


def validate_receipt_object(obj: Any) -> List[str]:
    """Validate a full receipt file object: the body plus its own hash."""
    if not isinstance(obj, dict):
        return ["receipt is not a JSON object"]
    stored = obj.get("body_sha256")
    body = {k: v for k, v in obj.items() if k != "body_sha256"}
    p = validate_body(body)
    if not isinstance(stored, str) or not _SHA256_RE.match(stored or ""):
        p.append(f"body_sha256 {stored!r} is not a sha256 hex digest")
        return p
    recomputed = body_sha256(body)
    if recomputed != stored:
        p.append(
            f"body_sha256 mismatch: stored {stored[:16]}..., recomputed "
            f"{recomputed[:16]}... -- this receipt was modified after it was "
            "written (append-only violation)")
    return p


# ---------------------------------------------------------------------------
# the append-only writer
# ---------------------------------------------------------------------------

def _resolve_root(receipts_root: Optional[str]) -> str:
    return os.path.realpath(receipts_root or RECEIPTS_ROOT)


def _assert_destination_allowed(path: str, root: str) -> None:
    """Refuse a destination outside ``root`` or inside a governed path."""
    real = os.path.realpath(path)
    if os.path.commonpath([real, root]) != root:
        raise ForbiddenDestination(
            f"receipt destination {real!r} is outside the receipts root {root!r}")
    for forbidden in FORBIDDEN_WRITE_ROOTS:
        f = os.path.realpath(forbidden)
        if real == f or os.path.commonpath([real, f]) == f:
            raise ForbiddenDestination(
                f"refusing to write under {forbidden!r}. A run records; it does "
                "not decide. Lane statuses and the claim graph are changed by "
                "an operator under governance/, never by this writer.")


def receipt_path(receipt: Receipt, receipts_root: Optional[str] = None) -> str:
    """Where a receipt lands: ``<root>/<lane>/<receipt_id>.json``."""
    root = _resolve_root(receipts_root)
    return os.path.join(root, receipt.lane, receipt.receipt_id + ".json")


def write_receipt(receipt: Receipt, receipts_root: Optional[str] = None) -> str:
    """Write a receipt, or refuse. Returns the path written.

    Refuses when the body fails :func:`validate_body` (in particular when
    ``does_not_establish`` is missing, empty, a placeholder or too short),
    when the destination is outside the receipts root or under a governed
    path, and when a receipt with that id already exists.

    The file is created with ``open(path, "x")``: exclusive creation. This
    module contains no other file-creating call, no truncating open, no
    ``os.remove``, no ``os.replace`` and no ``shutil`` import, so there is no
    code path here that can modify or delete an existing receipt.
    """
    problems = validate_body(receipt.body())
    if problems:
        raise ReceiptRejected(
            f"receipt rejected ({len(problems)} problem(s)):\n  - "
            + "\n  - ".join(problems))

    root = _resolve_root(receipts_root)
    path = os.path.join(root, receipt.lane, receipt.receipt_id + ".json")
    _assert_destination_allowed(path, root)

    if os.path.exists(path):
        raise AppendOnlyViolation(
            f"receipt {receipt.receipt_id} already exists at {path!r}; receipts "
            "are append-only and are never rewritten")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(receipt.to_json())
    except FileExistsError as exc:  # lost a race with a concurrent writer
        raise AppendOnlyViolation(
            f"receipt {receipt.receipt_id} already exists at {path!r}") from exc
    return path


def iter_receipt_files(receipts_root: Optional[str] = None) -> List[str]:
    """Every ``*.json`` receipt under the receipts root, sorted."""
    root = _resolve_root(receipts_root)
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__",)]
        for fn in filenames:
            if fn.endswith(".json"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_receipt(path: str) -> Dict[str, Any]:
    """Read one receipt file as a plain dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
