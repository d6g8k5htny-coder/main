#!/usr/bin/env python3
"""Resource-bounded deterministic diagnostic subprocess for research-flow-worker-1."""
import argparse
import json
from pathlib import Path
import sys

from model import canonical, digest, fingerprints, structures, validate


def run(job, data, attempt, execution_dataset_sha256):
    # Resource setup executes in the child, avoiding preexec_fn in threaded servers.
    try:
        import resource
    except ImportError as exc:
        raise ValueError("CPU/memory limits unsupported on this execution platform") from exc
    bounds = job["resource_limits"]
    resource.setrlimit(resource.RLIMIT_CPU, (bounds["cpu_seconds"], bounds["cpu_seconds"]))
    resource.setrlimit(resource.RLIMIT_AS, (bounds["memory_bytes"], bounds["memory_bytes"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (bounds["max_artifact_bytes"], bounds["max_artifact_bytes"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    validation = validate(data, require_stored_fingerprints=False)
    record_id = job["record_id"]
    records, obs, imports, corrections, local = structures(data)
    if fingerprints(data)[record_id] != job["inputs"]["obligation_fingerprint"]:
        raise ValueError("subprocess input fingerprint mismatch")
    if digest(data) != execution_dataset_sha256:
        raise ValueError("subprocess dataset snapshot hash mismatch")
    for note in job["inputs"]["proof_notes"]:
        import hashlib
        path = attempt / "notes" / note["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != note["sha256"]:
            raise ValueError("referenced proof note bytes changed")
    for number, source in enumerate(job["inputs"].get("source_inputs", [])):
        relative = source.get("relative_path") or source.get("artifact_path") or source.get("local_path")
        expected = source.get("sha256") or source.get("content_sha256") or source.get("source_content_sha256")
        if relative and expected:
            import hashlib
            if hashlib.sha256((attempt / "sources" / (str(number) + ".bin")).read_bytes()).hexdigest() != expected:
                raise ValueError("acquired mathematical source bytes changed")
    from research_evidence import validate_capture, verify_capture
    for capture in job.get("research_inputs", []):
        validate_capture(capture, data)
        verify_capture(attempt, capture)
    ancestry = sorted(set(job["inputs"]["transitive_prerequisite_fingerprints"]) | {record_id})
    import_ids = sorted({i for rid in ancestry for i in obs[rid]["external_theorem_imports"]})
    correction_ids = sorted({i for rid in ancestry for i in obs[rid]["source_corrections"]})
    return {"protocol_version": "research-flow-worker-1", "result": "diagnostic_completed",
            "record_id": record_id, "obligation_id": job["obligation_id"],
            "binding_sha256": job["inputs"]["binding_sha256"],
            "validated_prerequisite_branch": ancestry,
            "structural_validation": validation,
            "proof_note_files_checked": len(job["inputs"]["proof_notes"]),
            "source_inputs_recorded": len(job["inputs"].get("source_inputs", [])),
            "research_inputs_checked": len(job.get("research_inputs", [])),
            "unresolved_imports": [i for i in import_ids if imports[i]["import_status"] != "independently_certified"],
            "unresolved_corrections": [i for i in correction_ids if not corrections[i]["validated"]],
            "undischarged_side_conditions": {rid: [c["id"] for c in obs[rid]["side_conditions"]
                                                   if c.get("validation") != "independently_discharged"]
                                             for rid in ancestry},
            "proof_state": "needs_mathematical_reverification",
            "formal_certification": "not_certified", "theorem_certificates_awarded": 0,
            "resource_limits_enforced": ["wall_by_parent", "cpu", "address_space", "file_size", "no_core_dump"],
            "scope": "Schema, dependency fingerprints and evidence custody only; no mathematical theorem proof"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.input).read_text(encoding="utf8"))
    data = json.loads(Path(args.dataset).read_text(encoding="utf8"))
    try:
        result = run(job, data, Path(args.input).parent, args.dataset_sha256)
        code = 0
    except Exception as exc:
        result = {"protocol_version": "research-flow-worker-1", "result": "failed",
                  "reason": str(exc), "error_type": type(exc).__name__,
                  "formal_certification": "not_certified", "theorem_certificates_awarded": 0}
        code = 1
    Path(args.output).write_bytes(canonical(result))
    print(json.dumps({"result": result["result"], "formal_certification": "not_certified"}, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
