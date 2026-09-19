#!/usr/bin/env python3
"""Strict, read-only validation of the existing sha256sum/JSONL manifests.

This supplements (does not rewrite) verify_manifests.py and frozen manifests.
At least one manifest and one successfully checked payload entry are required.
Use --coverage to pin reviewed manifest identities, or repeat
--require-manifest to require named paths. Newly discovered manifests are
always scanned as well. CI uses .github/manifest-coverage.json.
Exclusions remain declarations: they are reported, never counted as verified.

Scope: named files and formats only, not completeness of the research corpus,
mathematical correctness, authorization of exclusions, or a security sandbox.
No basename fallback, absolute paths, parent traversal, or symlinks are used.
Run against a stable checkout; concurrent filesystem mutation is out of scope.
Standard library only; no payload is imported or executed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"[0-9a-fA-F]{64}\Z")
SUM_LINE = re.compile(r"([0-9a-fA-F]{64}) ([ *])(.+)\Z")
EXCLUSION = re.compile(r"^(skipped|failed|tree-only)(?:\s|:)(.+)$")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache"}


@dataclass
class Report:
    manifests: int = 0
    pinned_manifests: int = 0
    verified_entries: int = 0
    excluded_entries: int = 0
    payload_paths: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "manifests": self.manifests,
            "pinned_manifests": self.pinned_manifests,
            "verified_entries": self.verified_entries,
            "unique_payload_paths": len(self.payload_paths),
            "excluded_entries": self.excluded_entries,
            "problems": self.problems,
            "exclusions": self.exclusions,
            "scope": "Manifest integrity only; no scientific or authorization verdict.",
        }


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def local_file(base: Path, value: object) -> Path:
    """Resolve a literal portable relative file path, rejecting all symlinks."""
    if not isinstance(value, str) or not value:
        raise ValueError("missing or non-string file path")
    if (value.startswith("/") or PureWindowsPath(value).drive or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("non-local or non-portable file path")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("empty, dot or parent path segment")
    path = base
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink file or directory")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("payload is not a regular file")
    return path


def verify_entry(base: Path, dest: object, digest: object, size: object,
                 check_size: bool, seen: set[Path], report: Report) -> None:
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("missing or invalid SHA-256")
    if check_size and (type(size) is not int or size < 0):
        raise ValueError("bytes must be a nonnegative integer (not bool)")
    path = local_file(base, dest)
    if path in seen:
        raise ValueError("duplicate destination in one manifest")
    seen.add(path)
    h = hashlib.sha256()
    actual_size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
            actual_size += len(chunk)
    if check_size and actual_size != size:
        raise ValueError("SIZE MISMATCH")
    if h.hexdigest() != digest.lower():
        raise ValueError("SHA MISMATCH")
    report.verified_entries += 1
    report.payload_paths.add(str(path))


def check_manifest(path: Path, report: Report) -> None:
    report.manifests += 1
    if path.is_symlink() or not path.is_file():
        report.problems.append(f"{path}: manifest is not a regular non-symlink file")
        return
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        report.problems.append(f"{path}: unreadable manifest: {exc}")
        return
    seen: set[Path] = set()
    nonempty = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if path.name.endswith(".sha256") and line.lstrip().startswith("#"):
            continue
        nonempty += 1
        location = f"{path}:{number}"
        try:
            if path.name.endswith(".sha256"):
                match = SUM_LINE.fullmatch(line)
                if not match:
                    raise ValueError("malformed or unsupported sha256sum line")
                verify_entry(path.parent, match[3], match[1], None, False, seen, report)
                continue
            row = json.loads(line, object_pairs_hook=strict_object,
                             parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("JSONL entry is not an object")
            if "stored" in row and type(row["stored"]) is not bool:
                raise ValueError("stored must be a boolean")
            note = row.get("note", "")
            if not isinstance(note, str):
                raise ValueError("note is not a string")
            exclusion = EXCLUSION.match(note)
            excluded = bool(exclusion and exclusion[2].strip(" :"))
            # Frozen tree-only metadata uses null for unknown exactness.
            # It is no payload assertion and must never count as verified.
            if ("exact" in row and type(row["exact"]) is not bool and not
                    (excluded and row.get("stored") is False and row["exact"] is None)):
                raise ValueError("exact must be a boolean, or null on an explicit non-stored exclusion")
            if excluded:
                if row.get("stored") is True:
                    raise ValueError("stored=true contradicts exclusion")
                report.excluded_entries += 1
                report.exclusions.append(f"{location}: {note}")
                continue
            if row.get("stored") is False:
                raise ValueError("not-stored record lacks an explicit exclusion reason")
            if "bytes_stored" in row and (type(row["bytes_stored"]) is not int
                    or row["bytes_stored"] != row.get("bytes")):
                raise ValueError("bytes_stored must be an integer agreeing with bytes")
            verify_entry(path.parent, row.get("dest"), row.get("sha256"),
                         row.get("bytes"), True, seen, report)
        except (OSError, ValueError, TypeError, OverflowError, RecursionError) as exc:
            report.problems.append(f"{location}: {exc}")
    if nonempty == 0:
        report.problems.append(f"{path}: empty manifest")


def is_manifest(name: str) -> bool:
    return name.endswith(".sha256") or name.endswith("_MANIFEST.jsonl") or name == "MANIFEST.jsonl"


def coverage_paths(root: Path, value: str, report: Report) -> list[str]:
    """Pin the reviewed existing manifest set; never derive it from this scan.

    A deleted or rewritten manifest cannot quietly reduce coverage. New
    manifests are still discovered and validated. This is a CI input, not
    branch protection or authority to exclude a source from migration.
    """
    try:
        path = local_file(root, value)
        config = json.loads(path.read_text(encoding="utf-8"),
                            object_pairs_hook=strict_object, parse_constant=reject_constant)
        if (not isinstance(config, dict)
                or set(config) != {"schema", "base_commit", "scope", "manifests"}
                or config["schema"] != "q0.manifest-coverage/v1"
                or not isinstance(config["base_commit"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", config["base_commit"])
                or not isinstance(config["scope"], str) or not config["scope"].strip()
                or not isinstance(config["manifests"], list) or not config["manifests"]):
            raise ValueError("invalid or empty coverage configuration")
        required = []
        for row in config["manifests"]:
            if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
                    or not isinstance(row["sha256"], str)
                    or not DIGEST.fullmatch(row["sha256"])):
                raise ValueError("invalid pinned manifest record")
            pinned = local_file(root, row["path"])
            if not is_manifest(pinned.name) or row["path"] in required:
                raise ValueError("duplicate or unsupported pinned manifest")
            if hashlib.sha256(pinned.read_bytes()).hexdigest() != row["sha256"].lower():
                raise ValueError(f"pinned manifest changed: {row['path']}")
            required.append(row["path"])
        report.pinned_manifests = len(required)
        return required
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        report.problems.append(f"coverage {value!r}: {exc}")
        return []


def audit(root: Path, required: list[str], coverage: str | None = None) -> Report:
    report = Report()
    if root.is_symlink() or not root.is_dir():
        report.problems.append(f"invalid scan root: {root}")
        return report
    root = root.resolve()
    required = list(required)
    if coverage is not None:
        required += coverage_paths(root, coverage, report)
    required_paths: set[Path] = set()
    for rel in required:
        try:
            path = local_file(root, rel)
            if not is_manifest(path.name):
                raise ValueError("required path is not a supported manifest")
            required_paths.add(path)
        except (OSError, ValueError) as exc:
            report.problems.append(f"required manifest {rel!r}: {exc}")
    discovered: set[Path] = set()
    def walk_error(exc: OSError) -> None:
        report.problems.append(f"scan error: {exc}")
    for directory, dirs, files in os.walk(root, onerror=walk_error, followlinks=False):
        kept = []
        for name in sorted(dirs):
            if name in SKIP_DIRS:
                continue
            child = Path(directory) / name
            if child.is_symlink():
                report.problems.append(f"symlink scan directory: {child}")
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            if is_manifest(name):
                path = Path(directory) / name
                discovered.add(path)
                check_manifest(path, report)
    for path in sorted(required_paths - discovered):
        report.problems.append(f"required manifest outside scanned scope: {path}")
    if report.manifests == 0:
        report.problems.append("no manifests discovered")
    if report.verified_entries == 0:
        report.problems.append("no payload entries verified")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--require-manifest", action="append", default=[])
    parser.add_argument("--coverage", help="repository-relative pinned manifest coverage configuration")
    parser.add_argument("--json", action="store_true", help="emit a structured report on stdout")
    args = parser.parse_args(argv)
    report = audit(args.root if args.root is not None else ROOT, args.require_manifest, args.coverage)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        for item in report.exclusions:
            print(f"EXCLUDED {item}")
        for item in report.problems:
            print(f"FAIL {item}")
        print("Manifest integrity only; no scientific or authorization verdict.")
        print(f"manifest_integrity: manifests={report.manifests} "
              f"pinned_manifests={report.pinned_manifests} "
              f"verified_entries={report.verified_entries} "
              f"unique_payload_paths={len(report.payload_paths)} "
              f"excluded_entries={report.excluded_entries} problems={len(report.problems)}")
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
