#!/usr/bin/env python3
"""Offline integrity and receipt verification of a Research Flow snapshot."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from flow import ROOT, Coordinator, verify_runtime
from model import validate
from workers import verify_receipt


def verify(root=ROOT, replay=False):
    root = Path(root)
    coordinator = Coordinator(root / "runtime")
    structural = validate(coordinator.base)
    foundation = json.loads((root / "FOUNDATION.json").read_text())
    actual = hashlib.sha256((root / "engine/dataset.json").read_bytes()).hexdigest()
    if actual != foundation["files"]["dataset.json"]:
        raise ValueError("reference mathematical dataset differs from the retained foundation")
    custody = verify_runtime(coordinator)
    if custody["errors"]:
        raise ValueError("research custody verification failed: " + str(custody["errors"]))
    receipts = []
    for path in sorted((root / "runtime/artifacts").glob("*/*/receipt.json")):
        receipts.append(verify_receipt(path, replay=replay))
    state = coordinator.snapshot()
    discoveries = {c["source_key"].split("/", 1)[1]: c for c in state["research_discoveries"]}
    archive_name = "carriers/CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip"
    member_checks = 0
    if archive_name in discoveries:
        archive = discoveries[archive_name]
        expected = "eb5fc2c2279787df628ac763c5f48df6cddf9cd8763b103074dc4f6cae0d3a6b"
        if archive["sha256"] != expected:
            raise ValueError("RN-FIELD archive identity differs from its acquired source")
        with zipfile.ZipFile(root / "runtime" / archive["artifact_path"]) as carrier:
            for member in ["RN_FIELD_PROOF.md", "RN_FIELD_CERTIFICATE.json", "rn_field.py", "rn_field_audit.py"]:
                relative = "carriers/CLOSE-20260917-b9c2/closure_round2/" + member
                if relative not in discoveries:
                    raise ValueError("missing captured RN-FIELD member: " + member)
                raw = carrier.read("closure_round2/" + member)
                if hashlib.sha256(raw).hexdigest() != discoveries[relative]["sha256"]:
                    raise ValueError("RN-FIELD member differs from original archive: " + member)
                member_checks += 1
    manifest_count = 0
    manifest_path = root / "RELEASE_SHA256SUMS"
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = root / relative
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("release manifest path escapes package")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("release checksum mismatch: " + relative)
            manifest_count += 1
    return {"result": "passed", "canonical_records": structural["mathematical_records"],
            "canonical_links": structural["direct_source_explicit_links"],
            "reference_dataset_bytes_preserved": True, "custody": custody,
            "rn_field_members_matched_to_original_zip": member_checks,
            "worker_receipts_verified": len(receipts), "worker_receipts_replayed": sum(r["replayed"] for r in receipts),
            "release_files_hashed": manifest_count,
            "configured_evidence_links": sum(len(values) for values in state["research_evidence_attachments"].values()),
            "records_requiring_mathematical_review": state["changed_count"],
            "current_completed_diagnostics": state["diagnostic_completed_count"],
            "superseded_jobs": sum(j["state"] == "stale" for j in state["jobs"]),
            "full_theorem_certificates": state["full_theorem_certificates"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(replay=args.replay), indent=2))
