#!/usr/bin/env python3
"""Version-pinned bounded local diagnostics. A receipt is never a theorem proof.

The coordinator owns durable scheduling; this module owns immutable attempt
inputs, subprocess execution, artifact custody and independent receipt replay.
Only the built-in deterministic diagnostic can execute in this release.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from model import digest, fingerprints, structures

ROOT = Path(__file__).resolve().parent
PROTOCOL = "research-flow-worker-1"
WORKER = "local-diagnostic"
VERSION = "1.0"
DEFAULT_LIMITS = {"wall_seconds": 15, "cpu_seconds": 8,
                  "memory_bytes": 268435456, "max_output_bytes": 65536,
                  "max_artifact_bytes": 2097152}
SHA = re.compile(r"^[0-9a-f]{64}$")


class WorkerError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf8")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _inside(root, relative):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise WorkerError("artifact path must be relative: " + str(relative))
    root = Path(root).resolve()
    target = root / p
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise WorkerError("artifact path escapes directory")
    return target


def _limits(limits=None):
    result = dict(DEFAULT_LIMITS)
    if limits:
        if set(limits) - set(result):
            raise WorkerError("unknown resource limit")
        result.update(limits)
    bounds = {"wall_seconds": (1, 60), "cpu_seconds": (1, 30),
              "memory_bytes": (67108864, 1073741824),
              "max_output_bytes": (1024, 1048576),
              "max_artifact_bytes": (65536, 8388608)}
    for key, value in result.items():
        lo, hi = bounds[key]
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise WorkerError("out-of-range resource limit: " + key)
    return result


def _ancestry(data, record_id):
    records, obligations, _, _, _ = structures(data)
    if record_id not in records:
        raise WorkerError("unknown obligation " + str(record_id))
    result = set()

    def visit(rid):
        if rid in result:
            return
        result.add(rid)
        for parent in obligations[rid]["premises"]:
            visit(parent)

    visit(record_id)
    return sorted(result)


def _source_inputs(data, ancestry, root):
    # Source bytes are an additional custody axis, never mathematical acceptance.
    if not (root / "source_provenance.json").exists():
        return []
    try:
        from source_custody import source_inputs
        values = {}
        for rid in ancestry:
            for item in source_inputs(data, rid, root=root):
                key = item.get("source_id") or item.get("url") or digest(item)
                values[key] = item
        return [values[k] for k in sorted(values)]
    except ImportError as exc:
        raise WorkerError("source custody module unavailable") from exc


def capture_inputs(data, record_id, baseline_dataset_sha256=None, root=ROOT):
    """Capture exact obligation inputs and only its transitive prerequisite branch."""
    root = Path(root)
    records, obs, imports, corrections, local = structures(data)
    fp = fingerprints(data)
    ancestry = _ancestry(data, record_id)
    obligation = obs[record_id]
    index_path = root / "LOCAL_EVIDENCE_NOTES_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf8"))
    note_index = {n["path"]: n for n in index["notes"]}
    local_ids = sorted({eid for rid in ancestry for eid in obs[rid]["local_evidence_refs"]})
    note_paths = sorted({local[eid]["note_file"] for eid in local_ids})
    notes = []
    for path in note_paths:
        if path not in note_index:
            raise WorkerError("proof note absent from authenticated local index: " + path)
        entry = note_index[path]
        notes.append({"path": path, "sha256": entry["sha256"],
                      "cited_by": sorted(set(entry["cited_by"]) & set(local_ids))})
    sources = _source_inputs(data, ancestry, root)
    if baseline_dataset_sha256 is None:
        baseline_dataset_sha256 = digest(json.loads((root / "dataset.json").read_text(encoding="utf8")))
    branch_imports = sorted({i for rid in ancestry for i in obs[rid]["external_theorem_imports"]})
    branch_corrections = sorted({i for rid in ancestry for i in obs[rid]["source_corrections"]})
    binding = {
        "record_id": record_id, "obligation_id": obligation["id"],
        "baseline_dataset_sha256": baseline_dataset_sha256,
        "effective_dataset_sha256": digest(data),
        "obligation_fingerprint": fp[record_id],
        "statement_sha256": digest({k: records[record_id][k] for k in ("statement", "scope", "equations")}),
        "record_sha256": digest(records[record_id]),
        "proof_hyperedge_sha256": digest({k: v for k, v in obligation.items() if k != "fingerprint"}),
        "prerequisite_fingerprints": {rid: fp[rid] for rid in sorted(obligation["premises"])},
        "transitive_prerequisite_fingerprints": {rid: fp[rid] for rid in ancestry if rid != record_id},
        "side_condition_fingerprints": {rid: digest(obs[rid]["side_conditions"]) for rid in ancestry},
        "import_fingerprints": {eid: digest(imports[eid]) for eid in branch_imports},
        "correction_fingerprints": {eid: digest(corrections[eid]) for eid in branch_corrections},
        "local_evidence_fingerprints": {eid: digest(local[eid]) for eid in local_ids},
        "proof_notes": notes, "source_inputs": sources,
        "source_fingerprints": {str(s.get("source_id") or s.get("url") or i): digest(s)
                                for i, s in enumerate(sources)},
    }
    # A change on an unrelated branch must not invalidate this binding.
    binding["binding_sha256"] = digest({k: v for k, v in binding.items()
                                       if k != "effective_dataset_sha256"})
    return binding


def checker_identity(worker=WORKER, root=ROOT):
    if worker != WORKER:
        return {"name": worker, "version": "unconfigured", "protocol": PROTOCOL,
                "certificate_kind": "none", "formal_certification": False}
    root = Path(root)
    files = {}
    for name in ("workers.py", "worker_process.py", "model.py", "research_evidence.py"):
        p = root / name
        files[name] = sha_file(p) if p.is_file() else None
    return {"name": WORKER, "version": VERSION, "protocol": PROTOCOL,
            "implementation_sha256": files,
            "certificate_kind": "diagnostic_receipt_only", "formal_certification": False}


def make_job(data, record_id, triggering_event, worker=WORKER, root=ROOT, limits=None):
    inputs = capture_inputs(data, record_id, root=root)
    identity = checker_identity(worker, root)
    event = copy.deepcopy(triggering_event)
    if not isinstance(event, dict):
        event = {"event_id": str(event)}
    key = {"binding": inputs["binding_sha256"], "checker": identity,
           "triggering_event": event, "limits": _limits(limits)}
    return {"protocol_version": PROTOCOL, "job_id": "j-" + digest(key)[:32],
            "obligation_id": inputs["obligation_id"], "record_id": record_id,
            "triggering_event": event, "inputs": inputs,
            "worker": identity, "checker": identity, "resource_limits": _limits(limits),
            "mathematical_certification": "not_certified"}


def detect_adapters():
    """Detect installed tools without invoking an unconfigured proof checker.

    Installed Lean/Coq programs are deliberately not executed merely to obtain
    a banner. A future reviewed adapter must pin and bound that probe itself.
    """
    result = []
    for name, executable in ((WORKER, sys.executable), ("lean", shutil.which("lean")),
                              ("coq", shutil.which("coqc"))):
        installed = executable is not None
        version = "Python " + platform.python_version() if name == WORKER else None
        result.append({"name": name, "installed": installed, "executable": executable,
                       "reported_version": version, "probe_error": None,
                       "version_state": "runtime_inspected" if name == WORKER else "not_invoked_unconfigured",
                       "configured": name == WORKER,
                       "formal_certification": False,
                       "reason": "deterministic diagnostic only" if name == WORKER else
                                 "No reviewed version-pinned statement-aligned kernel adapter configured"})
    return result


def _environment():
    return {"python_version": platform.python_version(), "python_executable": sys.executable,
            "python_executable_sha256": sha_file(sys.executable),
            "implementation": platform.python_implementation(), "platform": platform.platform(),
            "machine": platform.machine(), "environment_policy": "minimal explicit environment; no inherited credentials"}


def _write_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _descriptor(directory, relative):
    p = _inside(directory, relative)
    return {"path": relative, "sha256": sha_file(p), "bytes": p.stat().st_size}


def _assert_binding(current, captured):
    if current.get("binding_sha256") != captured.get("binding_sha256"):
        raise WorkerError("stale or mismatched mathematical/source input binding")


def execute_job(job, data, workdir, root=ROOT, research_root=None):
    """Execute an actual resource-bounded child checker in a new attempt directory.

    Failed attempts still retain their immutable captured inputs and a hashed
    receipt. They cannot clear the coordinator's mathematical reinspection flag.
    """
    root, directory = Path(root), Path(workdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise WorkerError("attempt directory must be empty (immutable artifacts)")
    limits = _limits(job.get("resource_limits"))
    receipt = {"protocol_version": PROTOCOL, "job_id": job.get("job_id"),
               "obligation_id": job.get("obligation_id"), "record_id": job.get("record_id"),
               "triggering_event": job.get("triggering_event"), "inputs": job.get("inputs"),
               "execution_dataset_sha256": digest(data),
               "worker": job.get("worker"), "checker": job.get("checker", job.get("worker")),
               "environment": _environment(), "resource_limits": limits,
               "artifact_directory": str(directory), "input_files": [], "artifacts": [],
               "stdout": "", "stderr": "", "stdout_truncated": False,
               "stderr_truncated": False, "exit_code": None, "retryable": False,
               "result": "failed", "formal_certification": "not_certified",
               "theorem_certificates_awarded": 0,
               "replay_command": "python workers.py replay RECEIPT_PATH"}
    started = time.monotonic()
    try:
        if job.get("protocol_version") != PROTOCOL:
            raise WorkerError("unsupported worker protocol")
        _assert_binding(capture_inputs(data, job["record_id"], root=root), job["inputs"])
        if job.get("obligation_id") != job["inputs"]["obligation_id"]:
            raise WorkerError("job/obligation mismatch")
        _write_new(directory / "job.json", canonical(job))
        _write_new(directory / "dataset.json", canonical(data))
        receipt["input_files"] += [_descriptor(directory, "job.json"), _descriptor(directory, "dataset.json")]
        for note in job["inputs"]["proof_notes"]:
            original = _inside(root, note["path"])
            if not original.is_file() or sha_file(original) != note["sha256"]:
                raise WorkerError("missing or modified referenced proof note: " + note["path"])
            relative = "notes/" + note["path"]
            _write_new(_inside(directory, relative), original.read_bytes())
            receipt["input_files"].append(_descriptor(directory, relative))
        for number, source in enumerate(job["inputs"].get("source_inputs", [])):
            relative = source.get("relative_path") or source.get("artifact_path") or source.get("local_path")
            expected = source.get("sha256") or source.get("content_sha256") or source.get("source_content_sha256")
            if not relative or not expected:
                continue
            original = _inside(root, relative)
            if not original.is_file() or sha_file(original) != expected:
                raise WorkerError("missing or modified acquired source bytes: " + str(relative))
            target = "sources/" + str(number) + ".bin"
            _write_new(directory / target, original.read_bytes())
            receipt["input_files"].append(_descriptor(directory, target))
        research_inputs = job.get("research_inputs", [])
        receipt["research_inputs"] = research_inputs
        for capture in research_inputs:
            from research_evidence import validate_capture, verify_capture
            validate_capture(capture, data)
            if research_root is None:
                raise WorkerError("research evidence custody root is required")
            raw = verify_capture(research_root, capture)
            relative = capture["artifact_path"]
            if not (directory / relative).exists():
                _write_new(_inside(directory, relative), raw)
                receipt["input_files"].append(_descriptor(directory, relative))
        if job.get("worker", {}).get("name") != WORKER:
            receipt["result"] = "blocked"
            receipt["reason"] = "Formal checker adapter is unconfigured; presence alone cannot certify a theorem"
            receipt["adapter_detection"] = detect_adapters()
        else:
            if job["worker"] != checker_identity(WORKER, root):
                raise WorkerError("worker/checker implementation version or hash mismatch")
            args = [sys.executable, "-B", str((root / "worker_process.py").resolve()),
                    "--input", str(directory / "job.json"),
                    "--dataset", str(directory / "dataset.json"),
                    "--dataset-sha256", receipt["execution_dataset_sha256"],
                    "--output", str(directory / "diagnostic.json")]
            receipt["command"] = args
            env = {"PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PATH": os.defpath, "LANG": "C.UTF-8"}
            with (directory / "stdout.txt").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
                child = subprocess.Popen(args, cwd=directory, env=env, stdout=stdout, stderr=stderr,
                                         start_new_session=os.name == "posix")
                try:
                    receipt["exit_code"] = child.wait(timeout=limits["wall_seconds"])
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                    child.wait()
                    receipt.update(result="failed", retryable=True, reason="worker wall-time limit exceeded",
                                   exit_code=child.returncode)
            for name in ("stdout", "stderr"):
                p = directory / (name + ".txt")
                receipt[name] = p.read_bytes()[:limits["max_output_bytes"]].decode("utf8", "replace")
                receipt[name + "_truncated"] = p.stat().st_size > limits["max_output_bytes"]
                receipt["artifacts"].append(_descriptor(directory, name + ".txt"))
            output = directory / "diagnostic.json"
            if output.is_file():
                if output.stat().st_size > limits["max_artifact_bytes"]:
                    raise WorkerError("worker artifact exceeds resource limit")
                result = json.loads(output.read_text(encoding="utf8"))
                receipt["artifacts"].append(_descriptor(directory, "diagnostic.json"))
                receipt["diagnostic"] = result
                if receipt["exit_code"] == 0 and result.get("result") == "diagnostic_completed":
                    receipt["result"] = "diagnostic_completed"
                else:
                    receipt["reason"] = result.get("reason", "diagnostic failed")
            elif "reason" not in receipt:
                receipt["reason"] = "worker exited without a replayable diagnostic artifact"
                receipt["retryable"] = receipt["exit_code"] is not None and receipt["exit_code"] < 0
            if receipt["stdout_truncated"] or receipt["stderr_truncated"]:
                receipt.update(result="failed", reason="worker output limit exceeded", retryable=False)
    except (WorkerError, KeyError, ValueError, OSError) as exc:
        receipt.update(result="failed", reason=str(exc), retryable=isinstance(exc, OSError))
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
    receipt["receipt_sha256"] = digest(receipt)
    _write_new(directory / "receipt.json", canonical(receipt))
    return receipt


def verify_receipt(receipt, job=None, data=None, workdir=None, root=ROOT, replay=False):
    """Validate receipt integrity and optionally repeat the actual child diagnostic.

    A valid receipt may describe a failure. Callers must inspect ``result``;
    neither integrity nor a passing diagnostic is a mathematical certificate.
    """
    if isinstance(receipt, (str, Path)):
        receipt_path = Path(receipt).resolve()
        if workdir is None:
            workdir = receipt_path.parent
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    receipt = copy.deepcopy(receipt)
    expected = receipt.pop("receipt_sha256", None)
    if not expected or not SHA.fullmatch(expected) or digest(receipt) != expected:
        raise WorkerError("invalid worker receipt hash")
    receipt["receipt_sha256"] = expected
    if receipt.get("protocol_version") != PROTOCOL:
        raise WorkerError("unknown worker receipt protocol")
    if receipt.get("result") not in {"diagnostic_completed", "failed", "blocked"}:
        raise WorkerError("unknown worker execution result")
    _limits(receipt.get("resource_limits"))
    if receipt.get("formal_certification") != "not_certified" or receipt.get("theorem_certificates_awarded") != 0:
        raise WorkerError("diagnostic receipt falsely claims theorem certification")
    directory = Path(workdir or receipt["artifact_directory"]).resolve()
    names = set()
    for descriptor in receipt["input_files"] + receipt["artifacts"]:
        relative = descriptor["path"]
        if relative in names:
            raise WorkerError("duplicate artifact descriptor")
        names.add(relative)
        p = _inside(directory, relative)
        if not p.is_file() or p.stat().st_size != descriptor["bytes"] or sha_file(p) != descriptor["sha256"]:
            raise WorkerError("invalid worker artifact hash: " + relative)
    stored_job = None
    if "job.json" in names:
        stored_job = json.loads((directory / "job.json").read_text(encoding="utf8"))
        if job is not None and stored_job != job:
            raise WorkerError("recorded job input differs from scheduled job")
        if job is None:
            job = stored_job
    if job is not None:
        for key in ("job_id", "record_id", "obligation_id", "triggering_event", "inputs", "worker", "resource_limits"):
            if receipt.get(key) != job.get(key):
                raise WorkerError("receipt/job mismatch: " + key)
    if job is not None and receipt.get("research_inputs", []) != job.get("research_inputs", []):
        if receipt["result"] == "diagnostic_completed":
            raise WorkerError("research input receipt/job mismatch")
    if receipt["result"] == "diagnostic_completed":
        from research_evidence import verify_capture
        for capture in (job or {}).get("research_inputs", []):
            if capture["artifact_path"] not in names:
                raise WorkerError("passing receipt is missing research input bytes")
            verify_capture(directory, capture)
    if data is not None:
        _assert_binding(capture_inputs(data, receipt["record_id"], root=root), receipt["inputs"])
    if receipt["result"] == "diagnostic_completed":
        if receipt["exit_code"] != 0 or not {"job.json", "dataset.json", "diagnostic.json", "stdout.txt", "stderr.txt"} <= names:
            raise WorkerError("passing receipt lacks successful diagnostic artifact")
        if receipt.get("stdout_truncated") or receipt.get("stderr_truncated"):
            raise WorkerError("passing receipt has truncated worker output")
        if receipt.get("worker") != checker_identity(WORKER, root) or receipt.get("checker") != receipt.get("worker"):
            raise WorkerError("diagnostic checker version or implementation mismatch")
        stored_data = json.loads((directory / "dataset.json").read_text(encoding="utf8"))
        if digest(stored_data) != receipt.get("execution_dataset_sha256"):
            raise WorkerError("recorded dataset input differs from captured snapshot")
        _assert_binding(capture_inputs(stored_data, receipt["record_id"], root=root), receipt["inputs"])
        output = json.loads((directory / "diagnostic.json").read_text(encoding="utf8"))
        if output != receipt.get("diagnostic") or output.get("result") != "diagnostic_completed":
            raise WorkerError("diagnostic/receipt mismatch")
        if output.get("binding_sha256") != receipt["inputs"]["binding_sha256"]:
            raise WorkerError("diagnostic statement binding mismatch")
        if output.get("formal_certification") != "not_certified" or output.get("theorem_certificates_awarded") != 0:
            raise WorkerError("diagnostic artifact falsely claims theorem certification")
    if replay:
        if job is None or "dataset.json" not in names:
            raise WorkerError("receipt has no replayable input set")
        original_data = json.loads((directory / "dataset.json").read_text(encoding="utf8"))
        with tempfile.TemporaryDirectory(prefix="ul-worker-replay-") as attempt:
            repeated = execute_job(job, original_data, attempt, root=root, research_root=directory)
        if repeated["result"] != receipt["result"] or repeated.get("diagnostic") != receipt.get("diagnostic"):
            raise WorkerError("worker replay differs from recorded diagnostic")
    return {"valid": True, "receipt_sha256": expected, "result": receipt["result"],
            "replayed": bool(replay), "formal_certification": "not_certified"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("detect")
    repeat = sub.add_parser("replay")
    repeat.add_argument("receipt")
    args = parser.parse_args()
    if args.action == "detect":
        print(json.dumps(detect_adapters(), indent=2))
    else:
        print(json.dumps(verify_receipt(args.receipt, replay=True), indent=2))


if __name__ == "__main__":
    main()
