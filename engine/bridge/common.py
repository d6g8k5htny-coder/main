"""Shared vocabulary for the bridge records: canonical JSON, digests, shapes.

Everything both ``work_order.py`` and ``run_receipt.py`` need and neither
owns: the canonical serialisation a record is hashed in, the SHA-256 of it,
the regular expressions for the identifiers the records carry (commit SHAs,
Drive ids, UTC timestamps, hex digests), the placeholder list a
``does_not_establish`` and every evidence string is refused against, the
scope-path normaliser both validators compare paths with, the strict JSON
loader that refuses a file whose text says two different things, and the
small type-checking helpers the validators are written with. The conventions
are those of ``engine/receipt.py`` (``q0.engine.receipt/v1``): sorted keys,
no insignificant whitespace, UTF-8, no floats in a hashed body, a closed key
set at every level.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
A digest computed here establishes byte identity and nothing else. It does
not establish authorization, truth, review or status. A string that passes
:func:`check_evidence_string` is a string that is not a placeholder; it is not
verified evidence. A path that :func:`normalize_scope_path` accepts is a
well-formed relative path; nothing here checks that it exists or that writing
to it is permitted. Nothing in this module writes a file, and the contract the
records implement is PROPOSED / NOT DEPLOYED.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import posixpath
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "canonical_json", "sha256_of", "body_digest", "utc_now",
    "SHA256_RE", "COMMIT_RE", "DRIVE_ID_RE", "DRIVE_URL_RE", "TIMESTAMP_RE",
    "MIN_DOES_NOT_ESTABLISH", "MIN_EVIDENCE_STRING", "PLACEHOLDERS",
    "VACUOUS_COMMANDS",
    "is_sha256", "is_commit", "is_drive_id", "is_drive_url", "is_timestamp",
    "parse_timestamp", "is_int", "is_str", "is_nonempty_str", "is_str_list",
    "check_keys", "check_does_not_establish", "check_evidence_string",
    "normalize_scope_path", "load_json_strict",
]

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
#: A Drive file, folder or shared-drive id. Anything containing a slash is a
#: path, and a path in this repository is never a Drive record. A 40- or
#: 64-hex digest is a hash, not an id, and is refused by :func:`is_drive_id`.
DRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{12,128}$")
#: A Drive or Docs URL: the only URL shape accepted where a Drive record is
#: expected. A GitHub URL, a pull request or a repository path is never one.
DRIVE_URL_RE = re.compile(r"^https://(?:drive|docs)\.google\.com/\S+$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,9})?Z$")

#: A ``does_not_establish`` shorter than this is not a sentence (CLAUDE.md
#: rule 2: the field is the most load-bearing one in any record).
MIN_DOES_NOT_ESTABLISH = 40
#: An evidence string on an executed receipt (a negative control, the
#: authenticated principal, the environment identity, the coverage statement,
#: the authorship exposure) shorter than this is a label, not evidence.
MIN_EVIDENCE_STRING = 8
PLACEHOLDERS = frozenset({
    "", "-", ".", "n/a", "na", "n.a.", "none", "null", "nil", "tbd", "todo",
    "unknown", "pending", "see above", "as above", "same", "nothing",
    "anonymous", "x", "xxx", "test", "fixme", "placeholder", "?",
})
#: Commands that run nothing. A receipt listing one as what executed, or an
#: order allowing one, reports a run in shape only.
VACUOUS_COMMANDS = frozenset({
    ":", "true", "false", "exit", "exit 0", "exit 1", "pass", "echo", "echo ok",
    "echo done", "echo pass", "noop", "no-op", "nop", "sleep 0", "sleep 1",
    "[ ]", "[]", "return", "return 0",
})

_GLOB_CHARS = frozenset("*?[]{}")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def canonical_json(obj: Any) -> str:
    """Sorted keys, no insignificant whitespace, UTF-8, ``ensure_ascii`` off."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body_digest(body: Mapping[str, Any]) -> str:
    """SHA-256 of a record body in canonical form."""
    return sha256_of(canonical_json(dict(body)))


def utc_now() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# -- predicates -------------------------------------------------------------

def is_sha256(x: Any) -> bool:
    return isinstance(x, str) and bool(SHA256_RE.match(x))


def is_commit(x: Any) -> bool:
    return isinstance(x, str) and bool(COMMIT_RE.match(x))


def is_drive_id(x: Any) -> bool:
    """Drive-id shaped: the id character set, 12-128 long, and not a digest.

    Shape only. That a string has this shape does not mean a Drive record
    exists under it; nothing in this repository can check that.
    """
    return (isinstance(x, str) and bool(DRIVE_ID_RE.match(x))
            and not SHA256_RE.match(x) and not COMMIT_RE.match(x))


def is_drive_url(x: Any) -> bool:
    return isinstance(x, str) and bool(DRIVE_URL_RE.match(x))


def parse_timestamp(x: Any) -> Optional[datetime.datetime]:
    """The UTC datetime a ``YYYY-MM-DDTHH:MM:SS[.frac]Z`` string names, or None.

    The regular expression is the shape; ``strptime`` is the calendar. A
    string like ``2026-99-99T99:99:99Z`` has the shape and names no instant.
    """
    if not isinstance(x, str) or not TIMESTAMP_RE.match(x):
        return None
    base, _, frac = x[:-1].partition(".")
    try:
        dt = datetime.datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    if frac:
        dt = dt.replace(microsecond=int(frac[:6].ljust(6, "0")))
    return dt.replace(tzinfo=datetime.timezone.utc)


def is_timestamp(x: Any) -> bool:
    return parse_timestamp(x) is not None


def is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def is_str_list(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(e, str) for e in x)


# -- shape helpers ----------------------------------------------------------

def check_keys(obj: Any, required: Sequence[str], where: str,
               problems: List[str]) -> Optional[Dict[str, Any]]:
    """Assert ``obj`` is an object with exactly the keys in ``required``.

    Appends a problem for every missing and every unexpected key (the key set
    of each record level is closed, so a status field cannot be smuggled in
    under a new name). Returns the object when it is a dict and every required
    key is present, else ``None`` so the caller can stop descending.
    """
    if not isinstance(obj, dict):
        problems.append(f"{where} is not a JSON object")
        return None
    req = set(required)
    for k in sorted(req - set(obj)):
        problems.append(f"{where}.{k} is missing")
    for k in sorted(set(obj) - req):
        problems.append(f"unexpected field {where}.{k!r} (the schema is closed)")
    if req - set(obj):
        return None
    return obj


def check_does_not_establish(value: Any, where: str, problems: List[str]) -> None:
    """The same three lines of defence ``engine/receipt.py`` uses."""
    if not isinstance(value, str):
        problems.append(f"{where} must be a string")
        return
    s = value.strip()
    if not s:
        problems.append(
            f"{where} is empty. A record that does not say what it fails to "
            "establish is refused: that field is the most load-bearing one.")
    elif s.lower() in PLACEHOLDERS:
        problems.append(f"{where} is the placeholder {s!r}, not a statement")
    elif len(s) < MIN_DOES_NOT_ESTABLISH:
        problems.append(
            f"{where} is {len(s)} characters, shorter than the "
            f"{MIN_DOES_NOT_ESTABLISH} required for it to be a statement")


def check_evidence_string(value: Any, where: str, problems: List[str],
                          min_len: int = MIN_EVIDENCE_STRING) -> bool:
    """Refuse a string that is not evidence; True when it passes.

    Not a string, blank, a placeholder (``n/a``, ``none``, ``-``, ``unknown``,
    ...), a command that runs nothing (``:``, ``true``), shorter than
    ``min_len``, or without a single letter or digit: each is refused with a
    problem naming ``where``. Passing means "not a placeholder". It does not
    mean the string is true.
    """
    if not isinstance(value, str):
        problems.append(f"{where} must be a string")
        return False
    s = value.strip()
    folded = " ".join(s.casefold().split())
    if not s:
        problems.append(f"{where} is blank; missing evidence is recorded as missing, "
                        "not as an empty string")
    elif folded in PLACEHOLDERS or folded in VACUOUS_COMMANDS:
        problems.append(f"{where} is the placeholder {s!r}, not evidence")
    elif not any(c.isalnum() for c in s):
        problems.append(f"{where} {s!r} contains no letter or digit; it is not evidence")
    elif len(s) < min_len:
        problems.append(f"{where} {s!r} is {len(s)} characters, shorter than the "
                        f"{min_len} required for it to say anything")
    else:
        return True
    return False


def normalize_scope_path(p: Any) -> Tuple[Optional[str], str]:
    """``(normalized, "")`` for a well-formed repository-relative path, else
    ``(None, reason)``.

    Normalized means: backslashes folded to ``/``, ``./`` prefixes and the
    trailing slash removed, casefolded (the repository may be checked out on a
    case-insensitive filesystem, so ``Governance/`` must reach
    ``governance/``). Refused rather than normalized: an absolute or
    home-relative path, a Windows drive, a ``..`` segment, an empty or ``.``
    segment inside the path, a glob metacharacter, a control character, and
    surrounding whitespace. ``.`` (the whole repository) normalizes to ``.``.
    """
    if not isinstance(p, str):
        return None, "is not a string"
    s = p.replace("\\", "/")
    if not s.strip():
        return None, "is blank"
    if s != s.strip():
        return None, "has surrounding whitespace"
    if any(ord(c) < 32 or c == "\x7f" for c in s):
        return None, "contains a control character"
    if s.startswith("/") or s.startswith("~") or _WINDOWS_DRIVE_RE.match(s):
        return None, "is absolute or home-relative; a scope path is repository-relative"
    if any(c in _GLOB_CHARS for c in s):
        return None, "contains a glob metacharacter; a scope path is literal"
    if "//" in s or "/./" in s or s.endswith("/."):
        return None, "contains an empty or '.' segment"
    if ".." in s.split("/"):
        return None, "contains a '..' segment"
    norm = posixpath.normpath(s)
    if norm.startswith("../") or norm == "..":
        return None, "escapes the repository"
    return norm.casefold(), ""


# -- strict reading ---------------------------------------------------------

def _refuse_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}: the file text says two things")
        out[k] = v
    return out


def _refuse_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value this record may carry")


def load_json_strict(path: str) -> Any:
    """``json.load`` that refuses duplicate keys and NaN/Infinity.

    A parser that keeps the last duplicate reads a different record than the
    bytes on disk say (the first ``"status"`` in the text is what another
    reader sees). Raises ``ValueError`` (of which ``JSONDecodeError`` is a
    subclass) or ``OSError``; callers report, never repair.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=_refuse_duplicate_keys,
                         parse_constant=_refuse_constant)
