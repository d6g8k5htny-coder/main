#!/usr/bin/env python3
"""Verify the Q0-C093 manifest in ZIP or extracted-directory mode."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "Q0_C093_SHA256_MANIFEST.json"
BUNDLE = BASE / "Q0_C093_FINAL_RELEASE.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    issues: list[str] = []
    if not MANIFEST.exists():
        issues.append("missing manifest")
        manifest = {"files": []}
    else:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    verified = 0
    mode = "zip" if BUNDLE.exists() else "extracted-directory"
    if mode == "zip":
        with zipfile.ZipFile(BUNDLE, "r") as archive:
            names = set(archive.namelist())
            for item in manifest.get("files", []):
                name = item["name"]
                if name not in names:
                    issues.append(f"missing from ZIP: {name}")
                    continue
                data = archive.read(name)
                if len(data) != item["bytes"]:
                    issues.append(f"size mismatch: {name}")
                if sha256(data) != item["sha256"]:
                    issues.append(f"hash mismatch: {name}")
                verified += 1
            if MANIFEST.name not in names:
                issues.append("manifest not included in ZIP")
            declared = {item["name"] for item in manifest.get("files", [])}
            extras = sorted(names - declared - {MANIFEST.name})
            if extras:
                issues.append(f"undeclared ZIP entries: {extras}")
    else:
        for item in manifest.get("files", []):
            path = BASE / item["name"]
            if not path.exists():
                issues.append(f"missing extracted file: {item['name']}")
                continue
            data = path.read_bytes()
            if len(data) != item["bytes"]:
                issues.append(f"size mismatch: {item['name']}")
            if sha256(data) != item["sha256"]:
                issues.append(f"hash mismatch: {item['name']}")
            verified += 1

    output = {
        "release_id": manifest.get("release_id"),
        "mode": mode,
        "valid": not issues,
        "issues": issues,
        "declared_files": len(manifest.get("files", [])),
        "verified_files": verified,
        "zip_entries_expected": len(manifest.get("files", [])) + 1,
        "bundle_sha256": sha256(BUNDLE.read_bytes()) if BUNDLE.exists() else None,
        "core_root_hashes": manifest.get("core_root_hashes", {}),
    }
    out = BASE / "Q0_C093_RELEASE_VERIFICATION.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
