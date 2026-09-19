#!/usr/bin/env python3
"""Validate the git-side operations registry and the trial ledger.

`engine/operations/REGISTRY.json` transcribes the register tab
`reusable_operations` (GP-REG-032-v1.2, sheet 42) cell for cell and adds a
`git_side` block per entry saying whether this repository can reproduce the
entry's displayed identity in exact arithmetic. `engine/operations/trials/`
holds `q0.operation.trial/v1` records in the shape of the register's own
`operation_trials` tab (sheet 43). This checker keeps both honest. It asserts:

  1. **The registry is the register.** There is one entry per register row and
     no more; every entry's twelve cells equal the row cell for cell under the
     header names; the row digest is sha256 of
     `json.dumps(row, ensure_ascii=False, separators=(",", ":"))`; the header
     is the register's header. In particular Utility and Novelty are the
     register's cells (`UNMEASURED` / `NOT_ASSESSED` at this export): a registry
     that says anything stronger, in a cell, in `git_side` or in its top-level
     `authority`, `utility_and_novelty` or `does_not_establish` text, fails.
     The top-level text is not free: `authority` and `utility_and_novelty` are
     `trial.py`'s `REGISTRY_AUTHORITY` / `REGISTRY_UTILITY_AND_NOVELTY`
     verbatim and `does_not_establish` contains every sentence of
     `REGISTRY_DOES_NOT_ESTABLISH` (extra sentences allowed, scanned). The
     provenance blocks are compared with the files the checker was pointed
     at: `register.path` / `tab` / `sheet_index` are the loaded register's,
     `register.export` and `row_sha256_rule` are the fixed sentences,
     `trial_ledger.path` / `tab` / `sheet_index` are the loaded trials tab's,
     `trial_ledger.columns` is the eighteen-column header, `record_schema`
     is the trial schema and `records_dir` is the trials directory.

  2. **git_side says only what it may.** Exactly the four keys
     `exact_identity_checkable`, `trial_kind`, `why`, `research_module`;
     `trial_kind` in the closed vocabulary and consistent with the boolean; a
     `research_module` that names an existing file under `research/`; and every
     `EXACT_IDENTITY` entry has an identity spec in `engine/operations/trial.py`
     whose displayed text is verbatim in the entry's `Action / output` cell and
     whose `scope_displayed` text, when named, is verbatim in `Required scope`
     (and conversely no spec exists for a non-checkable entry).

  3. **A trial is a record, in the ledger's shape, checked against the catalog
     it names.** Top-level keys are exactly the eighteen `operation_trials`
     columns plus `does_not_establish` and `authority` (and the register tab's
     header still reads those eighteen columns). `Library / catalog SHA-256`
     must be the digest of the current `REGISTRY.json` or of a version of it in
     git history (`git log --all -- <registry>`); a digest that matches neither
     fails, because the record cannot be checked against a catalog nobody can
     read. `Problem ID / SHA-256` must be that catalog's row digest for the
     operation. Every further check runs against the resolved catalog's cells:
     `Verified result` is one of `IDENTITY_REPRODUCED`, `IDENTITY_NOT_REPRODUCED`,
     `NOT_RUN`; a `NOT_MACHINE_CHECKABLE_HERE` operation carries only `NOT_RUN`
     and an `EXACT_IDENTITY` operation never carries `NOT_RUN` (the runner
     writes neither pairing, so a record with one was written by hand);
     for a run, the recorded `identity_spec` is displayed in the catalog's
     `Action / output` cell, is re-run here, and **the re-run's steps must equal
     the recorded steps field for field** (statement, lhs, rhs, equal) with the
     same failure and the same verdict; `IDENTITY_REPRODUCED` additionally
     requires the spec to be the one `trial.py` binds to the operation, every
     step equal and a zero failure charge; `Run evidence.trial_kind`,
     `identity_displayed`, `checked_equalities` and the `Trial ID` digest are
     recomputed from the record; `Scope / constraints` is the `Required scope`
     cell verbatim and `Notes` contains the `Do not infer` cell and the
     `Utility: <cell>` / `Novelty: <cell>` phrases verbatim; `Split`, `Arm`, the
     costs and the failure charge are in their vocabularies **and agree with
     the verdict**: a run carries `Split` `EXPOSED_DEVELOPMENT`, `Arm`
     `NONE_SINGLE_EXACT_REPLAY` and `Verification cost` = number of steps;
     `NOT_RUN` carries `NOT_APPLICABLE` for both and `NOT_MEASURED` for the
     cost, records no steps, no failure and no spec, and its `reason_not_run`
     is the catalog's `git_side.why` verbatim (a run's is `null`); the four
     unmeasured cost columns are exactly `NOT_MEASURED`, and `Budget / cost
     unit` and `Run evidence.arithmetic` are the runner's fixed sentences,
     because nothing here measures a search, retrieval, acquisition or
     maintenance cost and no other arithmetic or budget was run; `authority`
     is the fixed sentence; `does_not_establish` contains, verbatim, every
     sentence `trial.py`'s `required_does_not_establish(op, verdict)` returns
     (the generic sentences, the operation's own sentence, and the NOT_RUN
     sentence for `NOT_RUN`), with extra sentences allowed and scanned; `Seed /
     environment` has exactly the runner's six keys with the fixed `seed` and
     `runner` sentences, a null or 40-hex `repository_commit_at_run` and a
     version-shaped `python`. **Time**: `Run evidence.utc` is the time the
     runner ran. It may not be later than this check's own clock, and, for a
     record present in `git HEAD`, may not be later than the commit that added
     it (`git log --diff-filter=A`), because a record cannot be committed before
     it was run. A stamp fixed with the runner's tests-only `--utc` option is
     not the time the runner ran and has no place in a committed record.
     Under the trials directory only `README.md` and `*.json` may exist, and
     `README.md` is scanned with the status-word and usefulness-word scans.

  4. **No trial or registry text uses a status word, or a usefulness word
     without the register's marker.** This is a word-list scan, not a reading
     of the sentence: the repository-authored fields (everything except
     `Scope / constraints`, and `Notes` with the quoted `Do not infer` cell
     removed; keys included, except a string that is exactly a register column
     name) may contain none of the words in `trial.py`'s `STATUS_WORDS` (the
     five rule-1 families PROVE/PROOF, CERTIFY/CERTIFICATE, CLOSE/CLOSURE,
     PROMOTE/PROMOTION, DISCHARGE in participle, verb and noun forms, and the
     review-verdict families INDEPENDENT, RATIFY, VERIFY, ACCEPT, ADMIT, PASS,
     APPROVE, SATISFY, VALIDATE, CONFIRM, plus ESTABLISHED / ESTABLISHES;
     case-insensitive, whole word), and every sentence that uses a word in
     `USEFULNESS_WORDS` (utility, novelty, useful, gain, valuable, beneficial,
     improve, important, ...) must carry the register's `UNMEASURED` /
     `NOT_ASSESSED` verbatim and pair the word with no value word (HIGH, LOW,
     NEW, MEASURED, ASSESSED, POSITIVE, ...). Only the exact header phrase
     `Next useful step` is exempt; the bare words `Utility` and `Novelty` are
     not. A status claimed in words outside the lists is not caught; the
     record's own `does_not_establish` is what says it moves nothing.

  5. **Trials are append-only.** Every trial present in `git HEAD` is present,
     byte for byte, in the working tree (the pattern of
     `tools/receipts_check.py`).

A trial whose catalog digest is not the current registry's but is a version in
git history is HISTORICAL: it is reported by name and checked in full against
that version's cells. It is not deleted (append-only) and is not evidence of
anything. A change to an identity spec or evaluator in `trial.py` makes
existing records of that operation fail the re-run comparison; that failure
is the point, and this checker defines no remedy for it.

Exit status is non-zero on any violation. Prints one line:
`operations_check: registry=<n> checkable=<k> trials=<m> problems=<p>`.

WHAT A PASS DOES NOT MEAN. It does not mean any operation is correct, useful,
new or admissible anywhere, that any trial is evidence, or that any status
moved. Utility is UNMEASURED and Novelty is NOT_ASSESSED because the register
says so. The five validity premises of Theorem D1 v2.2(2) are OPEN and
`D3-LEMMA-RN-UNIF` remains open; a green checker is not a step toward
changing either.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.operations import trial as T  # noqa: E402

DEFAULT_REGISTRY = T.DEFAULT_REGISTRY
DEFAULT_REGISTER = T.DEFAULT_REGISTER
DEFAULT_TRIALS_TAB = T.DEFAULT_TRIALS_TAB
DEFAULT_TRIALS = os.path.join(ROOT, "engine", "operations", "trials")

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MIN_SENTENCE = 40
STEP_FIELDS = ("statement", "lhs", "rhs", "equal")
#: The only files allowed under the trials directory besides the records.
TRIALS_DIR_TEXT_FILES = ("README.md",)


def _rel(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace(os.sep, "/")


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def markdown_paragraphs(text: str) -> List[str]:
    """Paragraphs of a markdown file with their line wrapping undone, so the
    sentence scans see sentences and not lines."""
    out: List[str] = []
    for block in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if joined:
            out.append(joined)
    return out


def _git(root: str, *args: str, timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def head_blobs(root: str, directory: str) -> Optional[Dict[str, bytes]]:
    """JSON blobs under ``directory`` as of ``git HEAD``, or None if git cannot answer."""
    rel = os.path.relpath(directory, root).replace(os.sep, "/")
    listing = _git(root, "ls-tree", "-r", "--name-only", "HEAD", "--", rel)
    if listing is None or listing.returncode != 0:
        return None
    out: Dict[str, bytes] = {}
    for name in listing.stdout.decode("utf-8", "replace").splitlines():
        name = name.strip()
        if not name.endswith(".json"):
            continue
        blob = _git(root, "show", f"HEAD:{name}")
        if blob is not None and blob.returncode == 0:
            out[name] = blob.stdout
    return out


def registry_history(root: str, registry_path: str) -> Dict[str, bytes]:
    """Every version of the registry file in git history (all refs), keyed by
    the sha256 of its bytes. Empty when git cannot answer."""
    rel = os.path.relpath(registry_path, root).replace(os.sep, "/")
    if rel.startswith(".."):
        return {}
    log = _git(root, "log", "--all", "--format=%H", "--", rel, timeout=120)
    if log is None or log.returncode != 0:
        return {}
    out: Dict[str, bytes] = {}
    for commit in log.stdout.decode("utf-8", "replace").split():
        blob = _git(root, "show", f"{commit}:{rel}")
        if blob is not None and blob.returncode == 0:
            out.setdefault(hashlib.sha256(blob.stdout).hexdigest(), blob.stdout)
    return out


def added_at(root: str, name: str) -> Optional[datetime.datetime]:
    """The committer time of the commit on HEAD's first-parent history that
    added ``name`` (the earliest `--diff-filter=A` entry), or None when git
    cannot answer. In a shallow clone the grafted root counts as the adding
    commit, whose time is never earlier than the true one, so the floor this
    gives is only ever looser, never a false failure."""
    log = _git(root, "log", "HEAD", "--format=%ct", "--diff-filter=A", "--", name)
    if log is None or log.returncode != 0:
        return None
    stamps = log.stdout.decode("utf-8", "replace").split()
    if not stamps:
        return None
    try:
        return datetime.datetime.fromtimestamp(int(stamps[-1]), datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def _entries_by_id(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in registry.get("entries") or []:
        if isinstance(e, dict) and isinstance(e.get("operation_id"), str):
            out.setdefault(e["operation_id"], e)
    return out


def check_registry(registry_path: str, register_path: str, root: str,
                   trials_tab: Optional[str] = None, trials_dir: Optional[str] = None,
                   ) -> Tuple[List[str], Dict[str, Dict[str, Any]], int]:
    problems: List[str] = []
    rel = os.path.relpath(registry_path, root)
    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
        with open(register_path, encoding="utf-8") as f:
            register = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{rel}: unreadable ({exc})"], {}, 0

    if not isinstance(registry, dict) or registry.get("schema") != T.REGISTRY_SCHEMA:
        problems.append(f"{rel}: schema is not {T.REGISTRY_SCHEMA!r}")
        return problems, {}, 0
    header = register.get("header")
    if header != list(T.REGISTER_COLUMNS):
        problems.append(f"{os.path.relpath(register_path, root)}: header is not the twelve "
                        "reusable_operations columns this checker knows")
    if registry.get("header") != header:
        problems.append(f"{rel}: header differs from the register's header")
    for key in ("authority", "does_not_establish", "utility_and_novelty"):
        if not registry.get(key):
            problems.append(f"{rel}: missing {key}")
    # The top-level sentences are pinned: a registry whose authority read
    # "canonical and final" once passed a check that only wanted the field
    # non-empty, and a bland one-sentence does_not_establish passed too.
    if registry.get("authority") != T.REGISTRY_AUTHORITY:
        problems.append(f"{rel}: authority must be exactly trial.py's REGISTRY_AUTHORITY "
                        f"({T.REGISTRY_AUTHORITY[:40]!r}...); the registry carries no other "
                        "authority sentence")
    dne = registry.get("does_not_establish")
    if not (isinstance(dne, list) and dne and all(isinstance(s, str) and len(s) >= MIN_SENTENCE
                                                  for s in dne)):
        problems.append(f"{rel}: does_not_establish must be a non-empty list of sentences")
    else:
        for i, sentence in enumerate(T.REGISTRY_DOES_NOT_ESTABLISH):
            if sentence not in dne:
                problems.append(f"{rel}: does_not_establish is missing trial.py's fixed sentence "
                                f"REGISTRY_DOES_NOT_ESTABLISH[{i}] ({sentence[:50]!r}...) "
                                "verbatim; the fixed sentences may be added to, never removed "
                                "or replaced")
    uan = registry.get("utility_and_novelty")
    if not (isinstance(uan, str) and all(m in uan for m in T.REGISTER_MARKERS)):
        problems.append(f"{rel}: utility_and_novelty must carry the register's words "
                        f"{' and '.join(T.REGISTER_MARKERS)} verbatim")
    if uan != T.REGISTRY_UTILITY_AND_NOVELTY:
        problems.append(f"{rel}: utility_and_novelty must be exactly trial.py's "
                        "REGISTRY_UTILITY_AND_NOVELTY")

    # Provenance blocks: compared with the files this check was pointed at,
    # not taken on trust (a registry once claimed its cells came from another
    # tab and its ledger had two columns, and passed).
    prov = registry.get("register")
    if not isinstance(prov, dict):
        problems.append(f"{rel}: register block missing")
    else:
        expected_prov = {
            "path": _rel(register_path, root),
            "tab": register.get("tab"),
            "sheet_index": register.get("sheet_index"),
            "export": T.REGISTER_EXPORT,
            "row_sha256_rule": T.ROW_SHA256_RULE,
        }
        if list(prov.keys()) != list(expected_prov):
            problems.append(f"{rel}: register block keys must be exactly "
                            f"{list(expected_prov)} (got {list(prov.keys())})")
        for k, want in expected_prov.items():
            if prov.get(k) != want:
                problems.append(f"{rel}: register.{k} is {prov.get(k)!r}, but the register this "
                                f"check reads says {want!r}")
    ledger = registry.get("trial_ledger")
    if not isinstance(ledger, dict):
        problems.append(f"{rel}: trial_ledger block missing")
    else:
        tab_obj: Dict[str, Any] = {}
        if trials_tab is not None:
            try:
                with open(trials_tab, encoding="utf-8") as f:
                    loaded = json.load(f)
                tab_obj = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                tab_obj = {}
        expected_ledger = {
            "path": _rel(trials_tab, root) if trials_tab is not None else ledger.get("path"),
            "tab": tab_obj.get("tab"),
            "sheet_index": tab_obj.get("sheet_index"),
            "columns": list(T.TRIAL_COLUMNS),
            "record_schema": T.SCHEMA,
            "records_dir": (_rel(trials_dir, root) + "/") if trials_dir is not None
            else ledger.get("records_dir"),
        }
        if list(ledger.keys()) != list(expected_ledger):
            problems.append(f"{rel}: trial_ledger block keys must be exactly "
                            f"{list(expected_ledger)} (got {list(ledger.keys())})")
        for k, want in expected_ledger.items():
            got = ledger.get(k)
            if k == "records_dir" and isinstance(got, str) and isinstance(want, str):
                got, want = got.rstrip("/"), want.rstrip("/")
            if got != want:
                problems.append(f"{rel}: trial_ledger.{k} is {ledger.get(k)!r}, but this check "
                                f"reads {want!r}")
    # 4 (registry): the top-level repository-authored text claims nothing.
    top = {k: v for k, v in registry.items() if k not in ("entries", "header")}
    hits = T.status_word_hits(top)
    if hits:
        problems.append(f"{rel}: top-level text uses status words {hits}")
    for reason in T.usefulness_hits_in(top):
        problems.append(f"{rel}: top-level text speaks of {reason}; Utility and Novelty are "
                        "the register's words")

    rows = register.get("rows") or []
    entries = registry.get("entries")
    if not isinstance(entries, list):
        problems.append(f"{rel}: entries is not a list")
        return problems, {}, 0
    if len(entries) != len(rows):
        problems.append(f"{rel}: {len(entries)} entries but the register has {len(rows)} rows "
                        "(the registry has exactly the register's rows, no more, no fewer)")

    by_id: Dict[str, Dict[str, Any]] = {}
    seen_rows = set()
    checkable = 0
    for n, e in enumerate(entries):
        tag = f"{rel}: entries[{n}]"
        if not isinstance(e, dict):
            problems.append(f"{tag}: not an object")
            continue
        if set(e.keys()) != {"operation_id", "register_row_index", "row_sha256", "cells",
                             "git_side"}:
            problems.append(f"{tag}: keys must be exactly operation_id, register_row_index, "
                            f"row_sha256, cells, git_side (got {sorted(e.keys())})")
        op_id = e.get("operation_id")
        idx = e.get("register_row_index")
        cells = e.get("cells")
        if not (isinstance(op_id, str) and T.OP_ID_RE.match(op_id)):
            problems.append(f"{tag}: bad operation_id {op_id!r}")
            continue
        if op_id in by_id:
            problems.append(f"{tag}: duplicate operation_id {op_id}")
        by_id[op_id] = e
        if not (isinstance(idx, int) and 0 <= idx < len(rows)):
            problems.append(f"{tag} ({op_id}): register_row_index {idx!r} is not a register row")
            continue
        if idx in seen_rows:
            problems.append(f"{tag} ({op_id}): register row {idx} transcribed twice")
        seen_rows.add(idx)
        row = rows[idx]
        if not isinstance(cells, dict) or list(cells.keys()) != list(header or []):
            problems.append(f"{tag} ({op_id}): cells are not the twelve header names in order")
            continue
        for h, cell in zip(header, row):
            if cells.get(h) != cell:
                which = " (Utility/Novelty are the register's words, never stronger)" \
                    if h in ("Utility", "Novelty") else ""
                problems.append(f"{tag} ({op_id}): cell {h!r} differs from the register row "
                                f"{idx}{which}: registry={cells.get(h)!r} register={cell!r}")
        if not str(cells.get("Operation", "")).startswith(op_id + " "):
            problems.append(f"{tag}: operation_id {op_id} does not head the Operation cell")
        if e.get("row_sha256") != T.row_sha256(row):
            problems.append(f"{tag} ({op_id}): row_sha256 does not match the canonical row digest")

        g = e.get("git_side")
        if not isinstance(g, dict) or tuple(g.keys()) != T.GIT_SIDE_KEYS:
            problems.append(f"{tag} ({op_id}): git_side keys must be exactly {T.GIT_SIDE_KEYS}")
            continue
        kind = g.get("trial_kind")
        flag = g.get("exact_identity_checkable")
        if kind not in T.TRIAL_KINDS:
            problems.append(f"{tag} ({op_id}): trial_kind {kind!r} not in {sorted(T.TRIAL_KINDS)}")
        if not isinstance(flag, bool) or flag != (kind == "EXACT_IDENTITY"):
            problems.append(f"{tag} ({op_id}): exact_identity_checkable must be the boolean of "
                            "trial_kind == EXACT_IDENTITY")
        if not (isinstance(g.get("why"), str) and len(g["why"]) >= MIN_SENTENCE):
            problems.append(f"{tag} ({op_id}): git_side.why must be a sentence")
        mod = g.get("research_module")
        if mod is not None and not (isinstance(mod, str) and mod.startswith("research/")
                                    and os.path.isfile(os.path.join(root, mod))):
            problems.append(f"{tag} ({op_id}): research_module {mod!r} is not an existing file "
                            "under research/")
        hits = T.status_word_hits(g)
        if hits:
            problems.append(f"{tag} ({op_id}): git_side uses status words {hits}")
        for reason in T.usefulness_hits_in(g):
            problems.append(f"{tag} ({op_id}): git_side speaks of {reason}; Utility and "
                            "Novelty are the register's words")
        if kind == "EXACT_IDENTITY":
            checkable += 1
            spec = T.IDENTITIES.get(op_id)
            if spec is None:
                problems.append(f"{tag} ({op_id}): EXACT_IDENTITY but engine/operations/trial.py "
                                "has no identity spec for it")
            elif not T.spec_bound_to_cell(spec, cells.get("Action / output", ""),
                                          cells.get("Required scope", "")):
                problems.append(f"{tag} ({op_id}): the identity spec's displayed text is not "
                                "verbatim in the Action / output cell (or its scope_displayed "
                                "text not in Required scope)")
        elif op_id in T.IDENTITIES:
            problems.append(f"{tag} ({op_id}): NOT_MACHINE_CHECKABLE_HERE but trial.py carries an "
                            "identity spec for it (classify it or drop the spec)")
    for op_id in T.IDENTITIES:
        if op_id not in by_id:
            problems.append(f"{rel}: trial.py has an identity spec for {op_id}, which is not in "
                            "the registry")
    return problems, by_id, checkable


# ---------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------

def _is_cost(v: Any) -> bool:
    return v == T.NOT_MEASURED or (isinstance(v, int) and not isinstance(v, bool) and v >= 0)


def _step_view(steps: List[Any]) -> List[Any]:
    return [tuple(s.get(k) for k in STEP_FIELDS) if isinstance(s, dict) else s for s in steps]


Resolver = Callable[[str], Optional[Dict[str, Dict[str, Any]]]]


def check_trial(path: str, by_id: Dict[str, Dict[str, Any]], registry_sha: str, root: str,
                resolve: Resolver) -> Tuple[List[str], Optional[str]]:
    """Returns ``(problems, historical)`` where ``historical`` is the digest of
    the registry version the record was resolved against when that is not the
    current one, else None."""
    problems: List[str] = []
    rel = os.path.relpath(path, root)
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{rel}: unreadable ({exc})"], None
    if not isinstance(rec, dict):
        return [f"{rel}: not an object"], None

    expected = list(T.TRIAL_COLUMNS) + list(T.REPOSITORY_FIELDS)
    if list(rec.keys()) != expected:
        missing = [k for k in expected if k not in rec]
        extra = [k for k in rec if k not in expected]
        problems.append(f"{rel}: top-level keys must be exactly the 18 operation_trials columns "
                        f"plus does_not_establish and authority, in order (missing={missing} "
                        f"extra={extra})")
        return problems, None

    op_id = rec["Operation ID"]
    if op_id not in by_id:
        problems.append(f"{rel}: Operation ID {op_id!r} is not in the registry")
        return problems, None

    # Resolve the catalog the record names; every cell-bound check runs against it.
    historical: Optional[str] = None
    pid = rec["Problem ID / SHA-256"]
    lib = rec["Library / catalog SHA-256"]
    if not (isinstance(pid, str) and SHA256_RE.match(pid)):
        problems.append(f"{rel}: Problem ID / SHA-256 is not a sha256")
        return problems, None
    if not (isinstance(lib, str) and SHA256_RE.match(lib)):
        problems.append(f"{rel}: Library / catalog SHA-256 is not a sha256")
        return problems, None
    if lib == registry_sha:
        catalog = by_id
    else:
        catalog = resolve(lib)
        if catalog is None:
            problems.append(f"{rel}: Library / catalog SHA-256 {lib[:12]}... matches neither the "
                            "current REGISTRY.json nor any version of it in git history; the "
                            "record names a catalog nobody can read and cannot be checked")
            return problems, None
        historical = lib
    entry = catalog.get(op_id)
    if entry is None or not isinstance(entry.get("cells"), dict) \
            or not isinstance(entry.get("git_side"), dict):
        problems.append(f"{rel}: {op_id} is not an entry of the registry version the record "
                        "names")
        return problems, historical
    cells = entry["cells"]
    kind = entry["git_side"].get("trial_kind")
    if pid != entry.get("row_sha256"):
        problems.append(f"{rel}: Problem ID / SHA-256 is not the row digest of {op_id} in the "
                        "registry version the record names (Library / catalog SHA-256)")

    tid = rec["Trial ID"]
    if not (isinstance(tid, str) and tid.startswith(f"TRIAL-{op_id}-")):
        problems.append(f"{rel}: Trial ID {tid!r} does not name {op_id}")
    if os.path.basename(path) != f"{tid}.json":
        problems.append(f"{rel}: filename does not match Trial ID {tid!r}")

    verdict = rec["Verified result"]
    if verdict not in T.VERIFIED_RESULTS:
        problems.append(f"{rel}: Verified result {verdict!r} is not in the closed vocabulary "
                        f"{sorted(T.VERIFIED_RESULTS)}")
    elif kind == "NOT_MACHINE_CHECKABLE_HERE" and verdict != "NOT_RUN":
        problems.append(f"{rel}: {op_id} is NOT_MACHINE_CHECKABLE_HERE and may only carry "
                        f"NOT_RUN, not {verdict}")
    elif kind == "EXACT_IDENTITY" and verdict == "NOT_RUN":
        problems.append(f"{rel}: {op_id} is EXACT_IDENTITY in the catalog the record names and "
                        "the runner never records NOT_RUN for it; a NOT_RUN record of a "
                        "checkable operation was written by hand")
    if rec["Split"] not in T.SPLITS:
        problems.append(f"{rel}: Split {rec['Split']!r} not in {sorted(T.SPLITS)}")
    if rec["Arm"] not in T.ARMS:
        problems.append(f"{rel}: Arm {rec['Arm']!r} not in {sorted(T.ARMS)}")
    if not (isinstance(rec["Budget / cost unit"], str) and rec["Budget / cost unit"].strip()):
        problems.append(f"{rel}: Budget / cost unit must be a non-empty string")
    for col in T.COST_COLUMNS:
        if not _is_cost(rec[col]):
            problems.append(f"{rel}: {col} must be a non-negative integer of the stated unit or "
                            f"{T.NOT_MEASURED!r}")
        elif col != "Verification cost" and rec[col] != T.NOT_MEASURED:
            problems.append(f"{rel}: {col} must be exactly {T.NOT_MEASURED!r}; nothing here "
                            "measures it, so a number there is a claim the runner never made")
    if rec["Budget / cost unit"] != T.BUDGET:
        problems.append(f"{rel}: Budget / cost unit is not the runner's fixed sentence; no other "
                        "budget or failure rule was run here")
    # Split and Arm must say what the verdict says: a run is an exposed
    # development replay with no comparison arm; NOT_RUN has neither.
    if verdict in ("IDENTITY_REPRODUCED", "IDENTITY_NOT_REPRODUCED"):
        if rec["Split"] != "EXPOSED_DEVELOPMENT":
            problems.append(f"{rel}: a run must carry Split 'EXPOSED_DEVELOPMENT', not "
                            f"{rec['Split']!r}")
        if rec["Arm"] != "NONE_SINGLE_EXACT_REPLAY":
            problems.append(f"{rel}: a run must carry Arm 'NONE_SINGLE_EXACT_REPLAY', not "
                            f"{rec['Arm']!r}")
    elif verdict == "NOT_RUN":
        if rec["Split"] != "NOT_APPLICABLE":
            problems.append(f"{rel}: NOT_RUN must carry Split 'NOT_APPLICABLE', not "
                            f"{rec['Split']!r}; no problem was posed")
        if rec["Arm"] != "NOT_APPLICABLE":
            problems.append(f"{rel}: NOT_RUN must carry Arm 'NOT_APPLICABLE', not "
                            f"{rec['Arm']!r}; nothing was replayed")
        if rec["Verification cost"] != T.NOT_MEASURED:
            problems.append(f"{rel}: NOT_RUN must carry Verification cost {T.NOT_MEASURED!r}, "
                            f"not {rec['Verification cost']!r}; no equality was checked on an "
                            "operation that was not run")
    fc = rec["Timeout / failure charge"]
    if not (isinstance(fc, int) and not isinstance(fc, bool) and fc >= 0):
        problems.append(f"{rel}: Timeout / failure charge must be a non-negative integer")
    env = rec["Seed / environment"]
    if not isinstance(env, dict) or tuple(env.keys()) != T.ENVIRONMENT_KEYS:
        problems.append(f"{rel}: Seed / environment must be an object with exactly the keys "
                        f"{list(T.ENVIRONMENT_KEYS)} in order (got "
                        f"{list(env.keys()) if isinstance(env, dict) else type(env).__name__})")
    else:
        if env["seed"] != T.SEED:
            problems.append(f"{rel}: Seed / environment.seed is not the runner's fixed sentence "
                            f"{T.SEED!r}; the computation is deterministic and no seed exists")
        if env["runner"] != T.RUNNER_PATH:
            problems.append(f"{rel}: Seed / environment.runner is {env['runner']!r}, not "
                            f"{T.RUNNER_PATH!r}")
        commit = env["repository_commit_at_run"]
        if commit is not None and not (isinstance(commit, str) and T.COMMIT_RE.match(commit)):
            problems.append(f"{rel}: Seed / environment.repository_commit_at_run must be null or "
                            f"a 40-hex commit, not {commit!r}")
        if not (isinstance(env["python"], str) and T.PYTHON_VERSION_RE.match(env["python"])):
            problems.append(f"{rel}: Seed / environment.python {env['python']!r} is not a "
                            "version string")
        for k in ("implementation", "platform"):
            if not (isinstance(env[k], str) and env[k].strip()):
                problems.append(f"{rel}: Seed / environment.{k} must be a non-empty string")

    ev = rec["Run evidence"]
    steps: List[Any] = []
    if not isinstance(ev, dict) or ev.get("record_schema") != T.SCHEMA:
        problems.append(f"{rel}: Run evidence must be an object carrying record_schema "
                        f"{T.SCHEMA!r}")
    else:
        steps = ev.get("steps") if isinstance(ev.get("steps"), list) else []
        if not isinstance(ev.get("steps"), list):
            problems.append(f"{rel}: Run evidence.steps must be a list")
        for i, s in enumerate(steps):
            if not (isinstance(s, dict) and set(STEP_FIELDS) <= set(s)
                    and isinstance(s["equal"], bool)):
                problems.append(f"{rel}: Run evidence.steps[{i}] is not a recorded equality")
        all_equal = bool(steps) and all(isinstance(s, dict) and s.get("equal") is True
                                        for s in steps)
        if ev.get("all_equal") is not all_equal:
            problems.append(f"{rel}: Run evidence.all_equal disagrees with the recorded steps")
        if ev.get("checked_equalities") != len(steps):
            problems.append(f"{rel}: Run evidence.checked_equalities is not the number of "
                            "recorded steps")
        if ev.get("trial_kind") != kind:
            problems.append(f"{rel}: Run evidence.trial_kind {ev.get('trial_kind')!r} is not the "
                            f"registry's {kind!r} for {op_id}")
        spec = ev.get("identity_spec")
        displayed = spec.get("displayed") if isinstance(spec, dict) else None
        if ev.get("identity_displayed") != displayed:
            problems.append(f"{rel}: Run evidence.identity_displayed is not the recorded "
                            "identity_spec's displayed text")
        if ev.get("reproduces") != (spec.get("reproduces") if isinstance(spec, dict) else None):
            problems.append(f"{rel}: Run evidence.reproduces is not the recorded identity_spec's "
                            "reproduces text")
        if ev.get("arithmetic") != T.ARITHMETIC:
            problems.append(f"{rel}: Run evidence.arithmetic is not the runner's fixed sentence; "
                            "the only arithmetic run here is exact rational, and a record may "
                            "not describe another")
        utc = ev.get("utc")
        when = T.parse_utc(utc)
        if when is None:
            problems.append(f"{rel}: Run evidence.utc is not a UTC timestamp")
        else:
            if tid != T.trial_id(op_id, pid, lib, spec, utc):
                problems.append(f"{rel}: Trial ID digest does not recompute from (operation, "
                                "row digest, catalog digest, identity_spec, utc)")
            now = _now()
            if when > now:
                problems.append(f"{rel}: Run evidence.utc {utc} is later than this check's clock "
                                f"({_iso(now)}); a record's utc is the time the runner ran, and "
                                "a run in the future did not happen")

        if verdict == "IDENTITY_REPRODUCED":
            if not all_equal or ev.get("failure") is not None:
                problems.append(f"{rel}: IDENTITY_REPRODUCED with a failed step, no steps or a "
                                "recorded failure")
            if fc != 0:
                problems.append(f"{rel}: IDENTITY_REPRODUCED with a non-zero failure charge")
        elif verdict == "IDENTITY_NOT_REPRODUCED":
            if all_equal and ev.get("failure") is None:
                problems.append(f"{rel}: IDENTITY_NOT_REPRODUCED although every step is equal and "
                                "no failure is recorded")
        elif verdict == "NOT_RUN":
            if steps or ev.get("failure") is not None or spec is not None:
                problems.append(f"{rel}: NOT_RUN must record no steps, no failure and no spec")
            if not ev.get("reason_not_run"):
                problems.append(f"{rel}: NOT_RUN without reason_not_run")
            elif ev.get("reason_not_run") != entry["git_side"].get("why"):
                problems.append(f"{rel}: NOT_RUN reason_not_run is not the catalog's git_side.why "
                                f"for {op_id} verbatim; the registry's reason is the only one "
                                "the runner records")
        if verdict in ("IDENTITY_REPRODUCED", "IDENTITY_NOT_REPRODUCED") \
                and ev.get("reason_not_run") is not None:
            problems.append(f"{rel}: a run carries reason_not_run null, not "
                            f"{ev.get('reason_not_run')!r}")
        if verdict in ("IDENTITY_REPRODUCED", "IDENTITY_NOT_REPRODUCED"):
            if rec["Verification cost"] != len(steps):
                problems.append(f"{rel}: Verification cost must be the number of checked "
                                "equalities")
            if kind != "EXACT_IDENTITY":
                pass                      # already refused above: only NOT_RUN is allowed
            elif not isinstance(spec, dict):
                problems.append(f"{rel}: Run evidence.identity_spec missing")
            elif not T.spec_bound_to_cell(spec, cells.get("Action / output", ""),
                                          cells.get("Required scope", "")):
                problems.append(f"{rel}: the recorded identity is not displayed in the "
                                "register's Action / output cell (or Required scope) of the "
                                "catalog the record names")
            else:
                re_steps, re_fail = T.evaluate(copy.deepcopy(spec))
                re_verdict = T.verdict_from(re_steps, re_fail)
                if re_verdict != verdict:
                    problems.append(f"{rel}: re-running the recorded identity here gives "
                                    f"{re_verdict}, the record says {verdict}")
                if _step_view(re_steps) != _step_view(steps):
                    problems.append(f"{rel}: re-running the recorded identity here gives steps "
                                    "that differ from the recorded steps (statement, lhs, rhs, "
                                    "equal); the record's arithmetic is not this arithmetic")
                if (re_fail is None) != (ev.get("failure") is None):
                    problems.append(f"{rel}: re-running the recorded identity here "
                                    f"{'raises' if re_fail else 'does not raise'} but the record "
                                    f"{'records no failure' if re_fail else 'records a failure'}")
                if verdict == "IDENTITY_REPRODUCED" and spec != T.IDENTITIES.get(op_id):
                    problems.append(f"{rel}: IDENTITY_REPRODUCED for a spec that is not the "
                                    f"one trial.py binds to {op_id}")

    if rec["Scope / constraints"] != cells.get("Required scope"):
        problems.append(f"{rel}: Scope / constraints is not the Required scope cell verbatim")
    notes = rec["Notes"]
    if not (isinstance(notes, str) and cells.get("Do not infer", "\0") in notes):
        problems.append(f"{rel}: Notes does not carry the register's 'Do not infer' cell "
                        "verbatim")
    for col in ("Utility", "Novelty"):
        phrase = f"{col}: {cells.get(col)}"
        if not (isinstance(notes, str) and phrase in notes):
            problems.append(f"{rel}: Notes does not carry the register's {col} cell as "
                            f"{phrase!r} verbatim")

    # 4: no status word and no usefulness claim in the repository-authored fields.
    authored = T.authored_fields(rec, cells)
    hits = T.status_word_hits(authored)
    if hits:
        problems.append(f"{rel}: repository-authored fields use status words {hits}; a trial "
                        "records a computation and claims nothing")
    for reason in T.usefulness_hits_in(authored):
        problems.append(f"{rel}: repository-authored fields speak of {reason}; Utility and "
                        "Novelty are the register's words and a trial measures neither")

    dne = rec["does_not_establish"]
    if not (isinstance(dne, list) and dne and all(isinstance(s, str) and len(s) >= MIN_SENTENCE
                                                  for s in dne)):
        problems.append(f"{rel}: does_not_establish must be a non-empty list of sentences")
    elif verdict in T.VERIFIED_RESULTS:
        # The load-bearing field is pinned: every sentence the runner writes
        # must still be there, verbatim. Extra sentences are allowed (and were
        # scanned above); the fixed ones may not be removed or replaced.
        for sentence in T.required_does_not_establish(op_id, verdict):
            if sentence not in dne:
                which = ("the NOT_RUN sentence" if sentence == T.NOT_RUN_DOES_NOT_ESTABLISH
                         else f"the {op_id} sentence" if sentence == T.OP_DOES_NOT_ESTABLISH.get(op_id)
                         else "a generic sentence")
                problems.append(f"{rel}: does_not_establish is missing {which} the runner writes "
                                f"({sentence[:50]!r}...) verbatim; the fixed sentences may be "
                                "added to, never removed or replaced")
    if rec["authority"] != T.AUTHORITY:
        problems.append(f"{rel}: authority must be exactly {T.AUTHORITY!r}")
    return problems, historical


def check(registry_path: str = DEFAULT_REGISTRY, register_path: str = DEFAULT_REGISTER,
          trials_dir: str = DEFAULT_TRIALS, trials_tab: str = DEFAULT_TRIALS_TAB,
          root: str = ROOT, check_git: bool = True) -> Tuple[List[str], Dict[str, Any]]:
    problems, by_id, checkable = check_registry(registry_path, register_path, root,
                                                trials_tab=trials_tab, trials_dir=trials_dir)
    counts: Dict[str, Any] = {"registry": len(by_id), "checkable": checkable, "trials": 0,
                              "historical": []}

    try:
        with open(trials_tab, encoding="utf-8") as f:
            tab = json.load(f)
        if tab.get("header") != list(T.TRIAL_COLUMNS):
            problems.append(f"{os.path.relpath(trials_tab, root)}: header is not the eighteen "
                            "operation_trials columns the trial record is shaped on")
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{os.path.relpath(trials_tab, root)}: unreadable ({exc})")

    registry_sha = T.sha256_file(registry_path) if os.path.isfile(registry_path) else ""

    history: Dict[str, Optional[Dict[str, Dict[str, Any]]]] = {}
    loaded = {"done": False, "blobs": {}}

    def resolve(sha: str) -> Optional[Dict[str, Dict[str, Any]]]:
        if sha in history:
            return history[sha]
        if not loaded["done"]:
            loaded["blobs"] = registry_history(root, registry_path)
            loaded["done"] = True
        blob = loaded["blobs"].get(sha)
        parsed: Optional[Dict[str, Dict[str, Any]]] = None
        if blob is not None:
            try:
                obj = json.loads(blob.decode("utf-8"))
                if isinstance(obj, dict) and obj.get("schema") == T.REGISTRY_SCHEMA:
                    parsed = _entries_by_id(obj)
            except (ValueError, UnicodeDecodeError):
                parsed = None
        history[sha] = parsed
        return parsed

    # Only records and the README live under the trials directory. Anything
    # else (a SUMMARY.md, a subdirectory, a stray text file) would be a place
    # to write what no record may say and no scan would read; it is refused.
    paths: List[str] = []
    if os.path.isdir(trials_dir):
        for name in sorted(os.listdir(trials_dir)):
            full = os.path.join(trials_dir, name)
            tag = _rel(full, root)
            if name.endswith(".json") and os.path.isfile(full):
                paths.append(full)
            elif name in TRIALS_DIR_TEXT_FILES and os.path.isfile(full):
                try:
                    with open(full, encoding="utf-8") as f:
                        text = f.read()
                except (OSError, UnicodeDecodeError) as exc:
                    problems.append(f"{tag}: unreadable ({exc})")
                    continue
                paragraphs = markdown_paragraphs(text)
                hits = T.status_word_hits(paragraphs)
                if hits:
                    problems.append(f"{tag}: uses status words {hits}; the trials directory "
                                    "claims nothing")
                for reason in T.usefulness_hits_in(paragraphs):
                    problems.append(f"{tag}: speaks of {reason}; Utility and Novelty are the "
                                    "register's words")
            else:
                problems.append(f"{tag}: only trial records (*.json) and "
                                f"{'/'.join(TRIALS_DIR_TEXT_FILES)} may exist under the trials "
                                "directory; nothing else is scanned, so nothing else is allowed")
    seen: Dict[str, str] = {}
    for path in paths:
        counts["trials"] += 1
        p, historical = check_trial(path, by_id, registry_sha, root, resolve)
        problems.extend(p)
        if historical:
            counts["historical"].append((os.path.relpath(path, root), historical))
        try:
            with open(path, encoding="utf-8") as f:
                tid = json.load(f).get("Trial ID")
        except (OSError, json.JSONDecodeError, AttributeError):
            tid = None
        if isinstance(tid, str):
            if tid in seen:
                problems.append(f"{os.path.relpath(path, root)}: Trial ID {tid} already used by "
                                f"{seen[tid]}")
            seen[tid] = os.path.relpath(path, root)

    if check_git:
        head = head_blobs(root, trials_dir)
        if head is not None:
            for name, blob in sorted(head.items()):
                cur = os.path.join(root, name)
                if not os.path.exists(cur):
                    problems.append(f"{name}: present in git HEAD and deleted from the working "
                                    "tree (trials are append-only)")
                    continue
                with open(cur, "rb") as f:
                    if f.read() != blob:
                        problems.append(f"{name}: differs from its git HEAD content (trials are "
                                        "append-only and are never rewritten)")
                        continue
                # A committed record was run before it was committed: its utc
                # may not be later than the commit that added it.
                try:
                    utc = json.loads(blob.decode("utf-8")).get("Run evidence", {}).get("utc")
                except (ValueError, UnicodeDecodeError, AttributeError):
                    utc = None
                when = T.parse_utc(utc)
                added = added_at(root, name)
                # git records commit times to the second; compare at that resolution
                if when is not None and added is not None \
                        and when.replace(microsecond=0) > added:
                    problems.append(f"{name}: Run evidence.utc {utc} is later than the commit "
                                    f"that added the record ({_iso(added)}); a record's utc is "
                                    "the time the runner ran and cannot follow its own commit")
    return problems, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--register", default=DEFAULT_REGISTER,
                    help="the exported reusable_operations tab")
    ap.add_argument("--trials", default=DEFAULT_TRIALS, help="directory of trial records")
    ap.add_argument("--trials-tab", default=DEFAULT_TRIALS_TAB,
                    help="the exported operation_trials tab (header shape)")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--no-git", action="store_true",
                    help="skip the git HEAD append-only comparison (a historical catalog digest "
                         "is still resolved through git history; without git it cannot be)")
    args = ap.parse_args(argv)

    problems, counts = check(args.registry, args.register, args.trials, args.trials_tab,
                             args.root, check_git=not args.no_git)
    for p in problems:
        print(p)
    for name, sha in counts["historical"]:
        print(f"note: {name}: HISTORICAL (checked in full against registry version {sha[:12]}... "
              "from git history, not the current one; not evidence of anything)")
    print(f"operations_check: registry={counts['registry']} checkable={counts['checkable']} "
          f"trials={counts['trials']} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
