#!/usr/bin/env python3
"""Tip-repo vault and quarantine path hygiene. Metadata only.

`drive/vault_tree.txt` is a hand-kept listing. Until this check, nothing
compared it to `drive/inventory.jsonl`, so a digest column, a dropped row, or
an invented id would still leave CI green. `tools/quarantine_check.py` refuses
a manifest row whose `drive_path` contains `99_DO_NOT_OPEN`. A row can store a
vault id under some other path string and that check does not see it.

This check does not open Drive, does not fetch bytes, and does not decide that
any carrier is present beyond the inventory and manifests already on the tip.
It does not add an exclusion, restore a claim, or move a status.

For an inventable agent the same bound holds. A vault id, a quarantine path,
and a green run of this check do not activate an inventable source of truth
and do not discharge OBL-H5-JETMOD. `tools/quarantine_check.py` owns the
exclusion register. Its `vault_rows` refusal sees a stored manifest row only
when `drive_path` contains `99_DO_NOT_OPEN`. This check also refuses a stored
row whose Drive id is a vault id when that path string omits the token, and
refuses a vault or `90_QUARANTINE_AND_TRIAGE` path copied into `engine/`,
`research/`, `packages/`, or `claims/`. Neither checker scans
`docs/math_status_probes/`. Silence there is not activation. Quarantine is
not a source of truth. See `docs/math_status_probes/README.md` and
`quarantine/PATHS.md`.

The letters `DO_NOT_OPEN` are not the vault. The vault is a path segment that
starts with `99_DO_NOT_OPEN`. The external `00_DO_NOT_OPEN_MANIFEST` sits
outside that folder. `DO_NOT_OPEN_BEFORE_HASH_FREEZE` is a different folder.

Run:  python3 tools/vault_hygiene_check.py [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY_REL = os.path.join("drive", "inventory.jsonl")
TREE_REL = os.path.join("drive", "vault_tree.txt")
COVERAGE_REL = os.path.join(
    "drive", "deltas", "2026-09-19", "DG-MIGRATION-20260919", "coverage.jsonl")
ACTIVE_ROOTS = ("engine", "research", "packages", "claims")
VAULT_PREFIX = "99_DO_NOT_OPEN"
EXTERNAL_PREFIX = "00_DO_NOT_OPEN_MANIFEST"
HASH_FREEZE = "DO_NOT_OPEN_BEFORE_HASH_FREEZE"
QUARANTINE_LANE = "90_QUARANTINE_AND_TRIAGE"
FOLDER_MIME = "application/vnd.google-apps.folder"
# Same stored statuses as tools/drive_coverage.py. A vault id in one of these
# means that ledger claims the bytes were fetched.
STORED_COVERAGE = frozenset({
    "EXACT_SOURCE_BYTES", "EXISTING_EXACT_BYTES", "NATIVE_EXPORT", "RAW_SOURCE_SNAPSHOT",
})
# The 2026-09-19 ledger's HELD_NO_OPEN rows, classified by path. The reason
# text says "Standing DO_NOT_OPEN boundary" for every one of them. Only `vault`
# is the vault. Locked so a later edit of that dated ledger cannot silently
# reclassify the vault as something else, or the reverse.
LOCKED_HELD_NO_OPEN = {
    "vault": 5,
    "external_manifest": 1,
    "hash_freeze": 32,
    "other_do_not_open": 2,
}
# HELD_NO_OPEN ids with no stored manifest row. The hash-freeze id is indexed
# tree-only (BULK_DATA_OVER_STORE_SIZE_LIMIT); this check does not fetch it.
LOCKED_UNSTORED_NONVAULT = frozenset({"1cPEMAUyvOlqH8-PT7TnOV420uYHDUYn8"})


def segments(path: str) -> list[str]:
    return [part for part in (path or "").split("/") if part]


def is_vault_path(path: str) -> bool:
    return any(part.startswith(VAULT_PREFIX) for part in segments(path))


def is_external_manifest_path(path: str) -> bool:
    return (any(part.startswith(EXTERNAL_PREFIX) for part in segments(path))
            and not is_vault_path(path))


def coverage_class(path: str) -> str:
    if is_vault_path(path):
        return "vault"
    if is_external_manifest_path(path):
        return "external_manifest"
    if HASH_FREEZE in (path or ""):
        return "hash_freeze"
    if "DO_NOT_OPEN" in (path or ""):
        return "other_do_not_open"
    return "no_token"


def load_inventory(path: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not isinstance(record, dict) or not record.get("id"):
                raise ValueError(f"{path}:{number}: inventory row without an id")
            if record["id"] in found:
                raise ValueError(f"{path}:{number}: duplicate id {record['id']}")
            found[record["id"]] = record
    return found


def render_bytes(record: dict) -> str:
    folder = (record.get("access_status") == "FOLDER"
              or record.get("mimeType") == FOLDER_MIME)
    if folder:
        if record.get("bytes") not in (None, "", "-"):
            raise ValueError(
                f"{record.get('id')}: folder bytes {record.get('bytes')!r} are not empty")
        return "-"
    size = record.get("bytes")
    if type(size) is not int or size < 0:
        raise ValueError(f"{record.get('id')}: non-folder bytes {size!r}")
    return str(size)


def expected_rows(inventory: dict[str, dict]) -> dict[str, tuple[str, str, str, str]]:
    """id -> (id, access_status, bytes, path) for the vault plus the external manifest."""
    rows: dict[str, tuple[str, str, str, str]] = {}
    external = []
    for record in inventory.values():
        path = record.get("path") or ""
        if is_vault_path(path) or is_external_manifest_path(path):
            status = record.get("access_status")
            if not isinstance(status, str) or not status:
                raise ValueError(f"{record['id']}: no access_status")
            row = (record["id"], status, render_bytes(record), path)
            rows[record["id"]] = row
            if is_external_manifest_path(path):
                external.append(record["id"])
    if len(external) != 1:
        raise ValueError(
            f"expected one external 00_DO_NOT_OPEN_MANIFEST, found {len(external)}")
    return rows


def parse_tree(path: str) -> dict[str, tuple[str, str, str, str]]:
    rows: dict[str, tuple[str, str, str, str]] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            raw = line.rstrip("\n")
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{number}: expected 4 tab fields, found {len(parts)}")
            ident, status, size, drive_path = parts
            if any(len(part) == 64 and all(c in "0123456789abcdef" for c in part.lower())
                   for part in parts):
                raise ValueError(f"{path}:{number}: a digest does not belong on a vault listing")
            if ident in rows:
                raise ValueError(f"{path}:{number}: duplicate id {ident}")
            rows[ident] = (ident, status, size, drive_path)
    return rows


def manifest_rows(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames
                       if name not in (".git", "__pycache__", ".pytest_cache", "node_modules")]
        for name in filenames:
            if name not in ("_MANIFEST.jsonl", "MANIFEST.jsonl"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield os.path.relpath(path, root), number, row


def active_lane_copies(root: str) -> list[str]:
    found = []
    forbidden = (VAULT_PREFIX, QUARANTINE_LANE)
    for lane in ACTIVE_ROOTS:
        base = os.path.join(root, lane)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if name not in (".git", "__pycache__")]
            for name in dirnames + filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root)
                if any(token in part for part in rel.split(os.sep) for token in forbidden):
                    found.append(rel.replace(os.sep, "/"))
    return sorted(set(found))


def stored_ids(root: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for rel, _number, row in manifest_rows(root):
        ident = row.get("id")
        if row.get("stored") and ident:
            found.setdefault(ident, []).append(rel)
    return found


def check(root: str) -> tuple[list[str], dict]:
    problems: list[str] = []
    inventory_path = os.path.join(root, INVENTORY_REL)
    tree_path = os.path.join(root, TREE_REL)
    try:
        inventory = load_inventory(inventory_path)
        expected = expected_rows(inventory)
        listed = parse_tree(tree_path)
    except (OSError, ValueError) as exc:
        return [str(exc)], {}

    missing = sorted(set(expected) - set(listed))
    extra = sorted(set(listed) - set(expected))
    for ident in missing:
        problems.append(f"vault_tree.txt omits inventory id {ident}")
    for ident in extra:
        problems.append(f"vault_tree.txt lists {ident}, which is not a vault or external-manifest inventory row")
    for ident in sorted(set(expected) & set(listed)):
        if listed[ident] != expected[ident]:
            problems.append(
                f"vault_tree.txt {ident} is {listed[ident][1:]!r}, inventory is {expected[ident][1:]!r}")

    vault_ids = {ident for ident, row in expected.items() if is_vault_path(row[3])}
    external_ids = set(expected) - vault_ids
    held = stored_ids(root)
    for rel, _number, row in manifest_rows(root):
        ident = row.get("id")
        drive_path = row.get("drive_path") or ""
        if not row.get("stored"):
            continue
        if ident in vault_ids or is_vault_path(drive_path):
            problems.append(
                f"{rel}: stored row {ident} is vault material; metadata only, never opened")
        if ident in external_ids:
            if row.get("exact") is True:
                problems.append(
                    f"{rel}: external manifest {ident} is stored exact:true; "
                    "the inventory declares no digest for it")
            if is_vault_path(drive_path):
                problems.append(
                    f"{rel}: external manifest {ident} is stored under a vault path")

    for rel in active_lane_copies(root):
        problems.append(
            f"{rel}: vault or quarantine lane path copied into an active lane; "
            "copying does not activate it and is refused")

    stats = {
        "vault_tree_rows": len(listed),
        "vault_ids": len(vault_ids),
        "external_manifest_ids": len(external_ids),
        "external_manifest_stored": sum(1 for ident in external_ids if ident in held),
        "vault_bytes_stored": sum(1 for ident in vault_ids if ident in held),
    }

    coverage_path = os.path.join(root, COVERAGE_REL)
    if os.path.isfile(coverage_path):
        problems.extend(check_coverage(coverage_path, inventory, held, stats))
    elif os.path.isfile(inventory_path) and root == ROOT:
        problems.append(f"missing tip coverage ledger {COVERAGE_REL}")

    stats["problems"] = len(problems)
    return problems, stats


def check_coverage(path: str, inventory: dict[str, dict], held: dict[str, list[str]],
                   stats: dict) -> list[str]:
    problems = []
    counts = {key: 0 for key in LOCKED_HELD_NO_OPEN}
    unstored_nonvault = []
    vault_unstored = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                return [f"{path}:{number}: {exc}"]
            ident = row.get("id")
            record = inventory.get(ident) if ident else None
            drive_path = (record or {}).get("path") or ""
            if ident and is_vault_path(drive_path) and row.get("status") in STORED_COVERAGE:
                problems.append(
                    f"{path}:{number}: vault id {ident} has coverage status {row.get('status')}, "
                    "which claims the bytes were fetched")
            if row.get("status") != "HELD_NO_OPEN":
                continue
            kind = coverage_class(drive_path)
            if kind not in counts:
                problems.append(f"{path}:{number}: HELD_NO_OPEN id {ident} class {kind}")
                continue
            counts[kind] += 1
            if ident not in held:
                if kind == "vault":
                    vault_unstored.append(ident)
                else:
                    unstored_nonvault.append(ident)
    if counts != LOCKED_HELD_NO_OPEN:
        problems.append(
            f"HELD_NO_OPEN class counts {counts} != locked {LOCKED_HELD_NO_OPEN}")
    if frozenset(unstored_nonvault) != LOCKED_UNSTORED_NONVAULT:
        problems.append(
            "HELD_NO_OPEN ids with no stored manifest row, outside the vault: "
            f"{sorted(unstored_nonvault)} != {sorted(LOCKED_UNSTORED_NONVAULT)}")
    vault_docs = sum(1 for ident, record in inventory.items()
                     if is_vault_path(record.get("path") or "")
                     and record.get("access_status") != "FOLDER")
    if len(vault_unstored) != vault_docs:
        problems.append(
            f"unstored vault HELD_NO_OPEN rows {len(vault_unstored)} != vault docs {vault_docs}")
    stats["held_no_open"] = counts
    stats["unstored_nonvault"] = sorted(unstored_nonvault)
    return problems


def format_stats(stats: dict) -> str:
    counts = stats.get("held_no_open") or {}
    return (
        f"vault_tree_rows={stats.get('vault_tree_rows', 0)} "
        f"vault_ids={stats.get('vault_ids', 0)} "
        f"external_manifest_stored={stats.get('external_manifest_stored', 0)} "
        f"vault_bytes_stored={stats.get('vault_bytes_stored', 0)} "
        f"held_no_open_vault={counts.get('vault', 0)} "
        f"held_no_open_external={counts.get('external_manifest', 0)} "
        f"held_no_open_hash_freeze={counts.get('hash_freeze', 0)} "
        f"held_no_open_other={counts.get('other_do_not_open', 0)} "
        f"problems={stats.get('problems', 0)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args(argv)
    problems, stats = check(os.path.abspath(args.root))
    stats["problems"] = len(problems)
    for problem in problems:
        print(problem)
    print(format_stats(stats))
    print("quarantine ≠ SoT; vault metadata ≠ contents; a pass activates nothing and discharges nothing.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
