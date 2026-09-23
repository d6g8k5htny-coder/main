#!/usr/bin/env python3
"""Build, deep-audit, fresh-extract, and attest the Q0-C093 release."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
INVENTORY = BASE / "C093_RELEASE_INVENTORY.csv"
MANIFEST = BASE / "Q0_C093_SHA256_MANIFEST.json"
BUNDLE = BASE / "Q0_C093_FINAL_RELEASE.zip"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(base: Path, script: str, *args: str, timeout: int = 420) -> dict:
    completed = subprocess.run(
        [sys.executable, str(base / script), *args],
        cwd=base,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "script": script,
        "args": list(args),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1600:],
        "stderr_tail": completed.stderr[-1600:],
    }


def inventory_files() -> list[str]:
    with INVENTORY.open(newline="", encoding="utf-8") as handle:
        return [row["filename"] for row in csv.DictReader(handle)]


def main() -> None:
    prechecks = [
        run(BASE, "validate_q0_llm_verifier_v4.py"),
        run(BASE, "q0_c092_contract_checker.py"),
        run(BASE, "audit_q0_c093_release.py", "--deep", timeout=900),
    ]
    failed = [item for item in prechecks if item["returncode"] != 0]
    if failed:
        raise RuntimeError(json.dumps(failed, indent=2))

    files = inventory_files()
    missing = [name for name in files if not (BASE / name).exists()]
    if missing:
        raise FileNotFoundError(f"inventory files missing: {missing}")

    c092 = json.loads((BASE / "q0_c092_contract_check_report.json").read_text())
    audit = json.loads((BASE / "Q0_C093_AUDIT_REPORT.json").read_text())
    if not c092.get("valid") or not audit.get("valid"):
        raise RuntimeError("semantic or release audit invalid")

    manifest = {
        "release_id": "Q0-C093-CLOSEOUT",
        "release_date": "2026-07-17",
        "theorem_contract": "Q0-C092-FINAL",
        "declared_file_count": len(files),
        "zip_entry_count": len(files) + 1,
        "manifest_self_excluded": True,
        "core_root_hashes": c092["semantic_core"]["root_hashes"],
        "unowned_items": audit.get("unowned_items"),
        "files": [],
    }
    for name in files:
        path = BASE / name
        manifest["files"].append(
            {"name": name, "bytes": path.stat().st_size, "sha256": hash_file(path)}
        )
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in manifest["files"]:
            archive.write(BASE / item["name"], arcname=item["name"])
        archive.write(MANIFEST, arcname=MANIFEST.name)

    verification = run(BASE, "verify_q0_c093_release.py")
    if verification["returncode"] != 0:
        raise RuntimeError(json.dumps(verification, indent=2))

    # True fresh extraction: no predecessor ZIP and no parent imports.
    with tempfile.TemporaryDirectory(prefix="q0_c093_fresh_") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(BUNDLE, "r") as archive:
            archive.extractall(temp)
        fresh_checks = [
            run(temp, "validate_q0_llm_verifier_v4.py"),
            run(temp, "q0_c092_contract_checker.py"),
            run(temp, "audit_q0_c093_release.py", "--no-write"),
            run(temp, "verify_q0_c093_release.py"),
        ]
        fresh_pass = all(item["returncode"] == 0 for item in fresh_checks)
        fresh = {
            "release_id": "Q0-C093-CLOSEOUT",
            "test": "fresh extraction cold start",
            "all_pass": fresh_pass,
            "checks": fresh_checks,
            "declared_files": len(files),
            "zip_entries": len(files) + 1,
            "bundle_sha256": hash_file(BUNDLE),
            "core_root_hashes": manifest["core_root_hashes"],
        }
        (BASE / "Q0_C093_FRESH_EXTRACTION_TEST.json").write_text(
            json.dumps(fresh, indent=2), encoding="utf-8"
        )
        if not fresh_pass:
            raise RuntimeError(json.dumps(fresh, indent=2))

    result = {
        "release_id": "Q0-C093-CLOSEOUT",
        "bundle": str(BUNDLE),
        "bundle_bytes": BUNDLE.stat().st_size,
        "bundle_sha256": hash_file(BUNDLE),
        "manifest": str(MANIFEST),
        "declared_files": len(files),
        "zip_entries": len(files) + 1,
        "unowned_items": audit.get("unowned_items"),
        "core_root_hashes": manifest["core_root_hashes"],
        "prechecks": prechecks,
        "fresh_extraction_pass": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
