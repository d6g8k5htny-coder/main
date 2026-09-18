#!/usr/bin/env python3
"""Validate every bridge work order and run receipt under `engine/bridge/`.

The bridge (`engine/bridge/`) is the repository-side half of a PROPOSED, NOT
DEPLOYED execution contract. This checker keeps its records honest. Over
`engine/bridge/orders/`, `engine/bridge/receipts/` and
`engine/bridge/examples/` (any of which may be empty) it asserts:

  1. **Schema.** Every order satisfies `engine.bridge.work_order.validate_
     work_order` and every receipt `engine.bridge.run_receipt.validate_run_
     receipt` -- the same functions the store calls, so the two cannot drift.
     In particular: no order authorizes a status change, no NOT_VERIFIED order
     is EXECUTABLE, no order is its own authority, no scope reaches the bridge
     or a policy surface without its references, every governing source has
     its five identity fields, no receipt reports PASS on missing or
     placeholder evidence, no receipt is ACCEPTED without naming the Drive
     record it attributes the acceptance to, independence credit names a
     record or is 0, and every `scientific_status_change` reads UNCHANGED.

  2. **Resolution.** Each receipt's `work_order_digest` resolves to an order
     committed under `orders/` (or, for an example receipt, under
     `examples/`), the two agree on `task_id` and `repository_id`, every
     executed command is in the order's `allowed_commands`, every output lies
     within its `allowed_paths` and on no surface, and no receipt reports an
     execution against an order that is not EXECUTABLE and verified.

  3. **Content addressing and hygiene.** An order's file stem is its
     `task_id`; a receipt's file stem is its `idempotency_key`; a held
     delivery is named `<key>.NEEDS_RECONCILIATION.<held-digest-prefix>.json`,
     sits beside an existing `<key>.json`, and differs from it. No task_id,
     digest or key is used twice. The live directories are flat and hold only
     `README.md` and `.json` files (lowercase extension); every `.json` under
     `orders/` is an order, every one under `receipts/` a receipt or a held
     delivery, every one under `examples/` an order or a receipt -- any other
     schema, a non-object, a file with duplicate keys or NaN, or a file whose
     bytes are not the canonical serialisation of its own content is a
     problem, never skipped. `skipped=` in the summary counts the README
     files and nothing else.

  4. **Examples stay examples.** A record under `examples/` has `record_kind`
     EXAMPLE; a record under `orders/` or `receipts/` does not. An example
     receipt is NOT_RUN; an example order is NOT_VERIFIED and PREPARED.

  5. **Append-only, in history.** Every `.json` under `orders/`, `receipts/`
     and `examples/` that appears in any commit reachable from HEAD was only
     ever added: `git log --diff-filter=MDT` over each directory must list
     nothing. A rewrite or a removal that was committed is caught exactly as
     an uncommitted one is; the working-tree-versus-HEAD comparison is kept
     as a local convenience for the uncommitted case. A digest establishes a
     record's identity; git history is its freeze. (`--no-git` skips both.)

Facts a validator RECORDS without granting -- a policy surface in scope with
its references present, a transcribed verification, a transcribed
independence credit, a transcribed acceptance, a dirty worktree, a run that
passed no test -- are printed as `NOTE:` lines, only for records that
validated, and do not fail the run.

It writes nothing. Exit status is non-zero on any problem, in the style of
`tools/receipts_check.py`.

WHAT A PASS DOES NOT ESTABLISH. It does not establish that any work order is
authorized (the Drive record it names is what authorizes), that any run
happened or was correct, that any reviewer was independent, that anything was
accepted, or that any claim, premise or obligation moved. It does not deploy
the contract, which remains PROPOSED / NOT DEPLOYED: no branch protection,
credential, access audit or Drive route exists because this checker is green.
History that was force-pushed away is not history this checker can read;
that is what server-side protection, which this repository does not hold, is
for. `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side), `OBL-H5-REMOTE-THRESHOLD`,
`OBL-D1-PROMOTE` and both Pieces of `D3-LEMMA-RN-UNIF` are OPEN.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.bridge import run_receipt as RR  # noqa: E402
from engine.bridge import work_order as WO  # noqa: E402
from engine.bridge.common import load_json_strict  # noqa: E402

DEFAULT_ORDERS = os.path.join(ROOT, "engine", "bridge", "orders")
DEFAULT_RECEIPTS = os.path.join(ROOT, "engine", "bridge", "receipts")
DEFAULT_EXAMPLES = os.path.join(ROOT, "engine", "bridge", "examples")

#: The only non-record file a live directory may hold.
ALLOWED_OTHER = ("README.md",)


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def list_directory(directory: Optional[str], root: str,
                   problems: List[str]) -> Tuple[List[str], int]:
    """``(json files, skipped)``: every entry accounted for, nothing silent.

    A subdirectory, a non-``.json`` file other than README.md, or a file whose
    extension is not exactly lowercase ``.json`` is a problem. README.md
    files are counted as skipped.
    """
    if not directory or not os.path.isdir(directory):
        return [], 0
    files: List[str] = []
    skipped = 0
    for fn in sorted(os.listdir(directory)):
        path = os.path.join(directory, fn)
        rel = _rel(path, root)
        if os.path.isdir(path):
            problems.append(f"{rel}: a subdirectory; the live directories are flat and hold "
                            "records only")
            continue
        if fn in ALLOWED_OTHER:
            skipped += 1
            continue
        if fn.casefold().endswith(".json"):
            if not fn.endswith(".json"):
                problems.append(f"{rel}: extension is not lowercase .json; a record is named "
                                "<task_id>.json or <idempotency_key>.json exactly")
                continue
            files.append(path)
            continue
        problems.append(f"{rel}: not a record file (only README.md and *.json belong here)")
    return files, skipped


def _load(path: str, rel: str, problems: List[str]) -> Optional[Any]:
    """Strict load: duplicate keys, NaN and unreadable text are problems."""
    try:
        return load_json_strict(path)
    except (OSError, ValueError) as exc:
        problems.append(f"{rel}: unreadable ({exc})")
        return None


def check_canonical_bytes(path: str, rel: str, obj: Any, problems: List[str]) -> None:
    """The bytes on disk must be the canonical serialisation of the content."""
    try:
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{rel}: unreadable ({exc})")
        return
    if raw != RR.to_json(obj):
        problems.append(f"{rel}: file bytes are not the canonical serialisation (sorted keys, "
                        "two-space indent, trailing newline) of the record they parse to; "
                        "the committed text is the record, and it is written by the store "
                        "or by the owner boundary, not edited")


def head_files(root: str, directory: str) -> Optional[Dict[str, bytes]]:
    """``{repo-relative name: blob}`` for the directory as of git HEAD, or None."""
    rel = os.path.relpath(directory, root).replace(os.sep, "/")
    if rel.startswith(".."):
        return None
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", rel],
            cwd=root, capture_output=True, text=True, timeout=60)
        if listing.returncode != 0:
            return None
        out: Dict[str, bytes] = {}
        for name in listing.stdout.splitlines():
            name = name.strip()
            if not name.casefold().endswith(".json"):
                continue
            blob = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root,
                                  capture_output=True, timeout=60)
            if blob.returncode == 0:
                out[name] = blob.stdout
        return out
    except (OSError, subprocess.SubprocessError):
        return None


def history_changes(root: str, directory: str) -> Optional[List[Tuple[str, str]]]:
    """``[(repo-relative name, commit)]`` for every ``.json`` under
    ``directory`` that some commit reachable from HEAD modified, deleted or
    type-changed. Renames are not detected, so a renamed record shows as a
    deletion. None when git is unavailable or the directory is outside."""
    rel = os.path.relpath(directory, root).replace(os.sep, "/")
    if rel.startswith(".."):
        return None
    try:
        log = subprocess.run(
            ["git", "log", "--no-renames", "--diff-filter=MDT", "--name-only",
             "--format=%H", "HEAD", "--", rel],
            cwd=root, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if log.returncode != 0:
        return None
    out: List[Tuple[str, str]] = []
    commit = "?"
    for line in log.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line):
            commit = line
            continue
        if line.casefold().endswith(".json"):
            out.append((line, commit))
    return out


def load_directory(files: List[str], root: str, problems: List[str]) -> Dict[str, Any]:
    """Strictly load every file once; ``{path: parsed}`` for the readable ones.

    Unreadable text, duplicate keys, NaN and non-canonical bytes are reported
    here, so that the order and receipt passes below see the same objects.
    """
    loaded: Dict[str, Any] = {}
    for path in files:
        rel = _rel(path, root)
        obj = _load(path, rel, problems)
        if obj is None:
            continue
        check_canonical_bytes(path, rel, obj, problems)
        loaded[path] = obj
    return loaded


def check_orders(loaded: Dict[str, Any], root: str, expect_kind: str,
                 problems: List[str], notes: List[str],
                 only: bool) -> Dict[str, Dict[str, Any]]:
    """Validate every order among ``loaded``; return ``{digest: order}``.

    With ``only``, every file must be an order; otherwise (the examples
    directory) receipts are left to :func:`check_receipts` and any other
    schema is a problem.
    """
    by_digest: Dict[str, Dict[str, Any]] = {}
    seen_task: Dict[str, str] = {}
    for path in sorted(loaded):
        rel = _rel(path, root)
        obj = loaded[path]
        schema = obj.get("schema") if isinstance(obj, dict) else None
        if schema != WO.SCHEMA:
            if only:
                problems.append(f"{rel}: not a {WO.SCHEMA} record (schema {schema!r}); every "
                                "file under the orders directory is a work order and is "
                                "validated as one, nothing is skipped")
            elif schema not in (RR.SCHEMA,):
                problems.append(f"{rel}: schema {schema!r} is neither a work order nor a "
                                "run receipt; nothing else belongs here")
            continue
        wo_problems = WO.validate_work_order(obj)
        for msg in wo_problems:
            problems.append(f"{rel}: {msg}")
        if not wo_problems:
            for msg in WO.work_order_notes(obj):
                notes.append(f"{rel}: {msg}")
        kind = obj.get("record_kind")
        if kind != expect_kind:
            problems.append(f"{rel}: record_kind {kind!r} under a directory that holds "
                            f"{expect_kind} records only")
        tid = obj.get("task_id")
        stem = os.path.basename(path)[:-5]
        if isinstance(tid, str):
            if expect_kind == WO.RECORD_RECORD and stem != tid:
                problems.append(f"{rel}: file stem {stem!r} is not its task_id {tid!r} "
                                "(an order is named by its task)")
            if tid in seen_task:
                problems.append(f"{rel}: task_id {tid!r} is already used by {seen_task[tid]}")
            else:
                seen_task[tid] = rel
        digest = obj.get("work_order_digest")
        if isinstance(digest, str):
            if digest in by_digest:
                problems.append(f"{rel}: work_order_digest already used by another order")
            else:
                by_digest[digest] = obj
    return by_digest


def check_receipts(loaded: Dict[str, Any], root: str, expect_kind: str,
                   orders: Dict[str, Dict[str, Any]], problems: List[str],
                   notes: List[str], only: bool) -> Tuple[int, int]:
    """Validate every receipt and held delivery among ``loaded``; return
    ``(receipts, held)``. With ``only``, an order among the files is a
    problem (an order under receipts/ is out of place, not skipped)."""
    n_receipts = n_held = 0
    primaries: Dict[str, Dict[str, Any]] = {}
    for obj in loaded.values():
        if isinstance(obj, dict) and obj.get("schema") == RR.SCHEMA:
            key = obj.get("idempotency_key")
            if isinstance(key, str):
                primaries[key] = obj

    for path in sorted(loaded):
        rel = _rel(path, root)
        obj = loaded[path]
        schema = obj.get("schema") if isinstance(obj, dict) else None
        if schema == WO.SCHEMA:
            if only:
                problems.append(f"{rel}: a {WO.SCHEMA} record under the receipts directory; "
                                "an order is committed under orders/ by a separately "
                                "authorized step, never delivered as a receipt")
            continue  # under examples/ it was handled by check_orders
        if schema not in (RR.SCHEMA, RR.HELD_SCHEMA):
            if only:
                problems.append(f"{rel}: not a {RR.SCHEMA} record (schema {schema!r}); every "
                                "file under the receipts directory is a receipt or a held "
                                "delivery and is validated as one, nothing is skipped")
            continue  # under examples/ check_orders already reported it
        name = os.path.basename(path)
        shape, key, prefix = RR.parse_receipt_file_name(name)

        if schema == RR.HELD_SCHEMA:
            n_held += 1
            if shape != "held":
                problems.append(f"{rel}: a held delivery must be named "
                                "<key>.NEEDS_RECONCILIATION.<16 hex>.json")
            first = primaries.get(obj.get("idempotency_key")) if isinstance(
                obj.get("idempotency_key"), str) else None
            if first is None:
                problems.append(f"{rel}: held delivery has no first receipt "
                                f"{obj.get('idempotency_key')}.json beside it")
            for msg in RR.validate_held_delivery(obj, first):
                problems.append(f"{rel}: {msg}")
            if shape == "held":
                if obj.get("idempotency_key") != key:
                    problems.append(f"{rel}: file name key does not match idempotency_key")
                held_sha = obj.get("held_body_sha256")
                if isinstance(held_sha, str) and not held_sha.startswith(prefix or ""):
                    problems.append(f"{rel}: file name digest prefix does not match "
                                    "held_body_sha256")
            d = obj.get("delivered")
            if isinstance(d, dict):
                order = orders.get(d.get("work_order_digest")) if isinstance(
                    d.get("work_order_digest"), str) else None
                for msg in RR.check_receipt_against_order(d, order):
                    problems.append(f"{rel}: delivered: {msg}")
            notes.append(f"{rel}: held for reconciliation; the first receipt is unchanged "
                         "and nothing here decides between them")
            continue

        n_receipts += 1
        rr_problems = RR.validate_run_receipt(obj)
        for msg in rr_problems:
            problems.append(f"{rel}: {msg}")
        if not rr_problems:
            for msg in RR.run_receipt_notes(obj):
                notes.append(f"{rel}: {msg}")
        kind = obj.get("record_kind")
        if kind != expect_kind:
            problems.append(f"{rel}: record_kind {kind!r} under a directory that holds "
                            f"{expect_kind} records only")
        ikey = obj.get("idempotency_key")
        if expect_kind == RR.RECORD_RECORD:
            if shape != "receipt":
                problems.append(f"{rel}: a receipt is named by its idempotency_key "
                                "(<64 hex>.json); this file is not")
            elif ikey != key:
                problems.append(f"{rel}: file stem does not equal idempotency_key "
                                f"{str(ikey)[:16]}...")
        if isinstance(ikey, str) and sum(
                1 for o in loaded.values() if isinstance(o, dict)
                and o.get("schema") == RR.SCHEMA and o.get("idempotency_key") == ikey) > 1:
            problems.append(f"{rel}: idempotency_key is used by more than one receipt file")
        order = orders.get(obj.get("work_order_digest")) if isinstance(
            obj.get("work_order_digest"), str) else None
        for msg in RR.check_receipt_against_order(obj, order):
            problems.append(f"{rel}: {msg}")
    return n_receipts, n_held


def check_append_only(root: str, directory: Optional[str], problems: List[str]) -> None:
    """History first (a committed rewrite or removal), then HEAD versus the
    working tree (an uncommitted one)."""
    if not directory:
        return
    what = "records"
    changed = history_changes(root, directory)
    if changed is not None:
        for name, commit in changed:
            problems.append(f"{name}: rewritten or removed in commit {commit[:12]} "
                            f"({what} are append-only in history, not only against HEAD; "
                            "a record that must change gets a successor, never an edit)")
    head = head_files(root, directory)
    if head is None:
        return
    for name, blob in sorted(head.items()):
        cur = os.path.join(root, name)
        if not os.path.exists(cur):
            problems.append(f"{name}: present in git HEAD and deleted from the working "
                            f"tree ({what} are append-only)")
            continue
        with open(cur, "rb") as f:
            if f.read() != blob:
                problems.append(f"{name}: differs from its git HEAD content ({what} are "
                                "append-only and are never rewritten)")


def check(orders_dir: str = DEFAULT_ORDERS, receipts_dir: str = DEFAULT_RECEIPTS,
          examples_dir: Optional[str] = DEFAULT_EXAMPLES, root: str = ROOT,
          check_git: bool = True) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Return ``(problems, notes, counts)``. Empty ``problems`` means every check passed."""
    problems: List[str] = []
    notes: List[str] = []
    order_files, skipped_o = list_directory(orders_dir, root, problems)
    receipt_files, skipped_r = list_directory(receipts_dir, root, problems)
    orders = check_orders(load_directory(order_files, root, problems), root,
                          WO.RECORD_RECORD, problems, notes, only=True)
    n_receipts, n_held = check_receipts(load_directory(receipt_files, root, problems), root,
                                        RR.RECORD_RECORD, orders, problems, notes, only=True)
    ex_orders: Dict[str, Dict[str, Any]] = {}
    n_ex_receipts = 0
    skipped_e = 0
    if examples_dir:
        example_files, skipped_e = list_directory(examples_dir, root, problems)
        examples = load_directory(example_files, root, problems)
        ex_orders = check_orders(examples, root, WO.RECORD_EXAMPLE, problems, notes, only=False)
        n_ex_receipts, ex_held = check_receipts(examples, root, WO.RECORD_EXAMPLE,
                                                ex_orders, problems, notes, only=False)
        if ex_held:
            problems.append(f"{_rel(examples_dir, root)}: holds a held delivery; examples "
                            "are never delivered")
    if check_git:
        for d in (orders_dir, receipts_dir, examples_dir):
            check_append_only(root, d, problems)
    counts = {"orders": len(orders), "receipts": n_receipts, "held": n_held,
              "example_orders": len(ex_orders), "example_receipts": n_ex_receipts,
              "skipped": skipped_o + skipped_r + skipped_e, "notes": len(notes)}
    return problems, notes, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--orders", default=DEFAULT_ORDERS)
    ap.add_argument("--receipts", default=DEFAULT_RECEIPTS)
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES,
                    help="examples directory; pass an empty string to skip")
    ap.add_argument("--no-git", action="store_true",
                    help="skip the git history and HEAD append-only comparisons")
    args = ap.parse_args(argv)

    problems, notes, counts = check(args.orders, args.receipts, args.examples or None,
                                    ROOT, check_git=not args.no_git)
    for n in notes:
        print(f"NOTE: {n}")
    for p in problems:
        print(p)
    print(f"orders={counts['orders']} receipts={counts['receipts']} held={counts['held']} "
          f"example_orders={counts['example_orders']} "
          f"example_receipts={counts['example_receipts']} skipped={counts['skipped']} "
          f"notes={counts['notes']} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
