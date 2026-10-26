#!/usr/bin/env python3
"""Create a consistent portable snapshot without stopping research acquisition."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
import uuid
from pathlib import Path

from flow import ROOT, Coordinator


def release(destination, workdir=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    coordinator = Coordinator(workdir or ROOT / "runtime")
    excluded_dirs = {"__pycache__", "smoke-runtime", "snapshots"}
    engine_excluded = {"examples", "baseline", "validation_logs"}
    engine_legacy = {"manifest.json", "SHA256SUMS", "verification_report.json", "index.html", "conductor.html"}
    with tempfile.TemporaryDirectory(prefix="research-flow-release-") as temporary:
        target = Path(temporary) / "Research_Flow"
        target.mkdir()
        with coordinator.journal.transaction():
            state = coordinator.snapshot()
            if any(job["state"] == "running" for job in state["jobs"]):
                raise ValueError("workers are running; retry snapshot after current attempts finish")
            for path in ROOT.rglob("*"):
                relative = path.relative_to(ROOT)
                if not path.is_file() or path.is_symlink():
                    continue
                if relative.parts[0] == "runtime" or any(part in excluded_dirs for part in relative.parts) or path.name.startswith("."):
                    continue
                if relative.name == "RELEASE_SHA256SUMS" or relative.suffix in {".zip", ".tmp"} or relative.name.endswith(".sha256"):
                    continue
                if relative.parts[0] == "engine" and (any(part in engine_excluded for part in relative.parts[1:])
                                                      or relative.name in engine_legacy):
                    continue
                output = target / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
            for path in coordinator.journal.workdir.rglob("*"):
                relative = path.relative_to(coordinator.journal.workdir)
                if not path.is_file() or path.is_symlink() or any(part.startswith(".") for part in relative.parts):
                    continue
                output = target / "runtime" / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
            # current_state is a projection, regenerated from the frozen journal.
            (target / "runtime/current_state.json").write_text(json.dumps(state, sort_keys=True))
        manifest = []
        for path in sorted(target.rglob("*")):
            if path.is_file():
                manifest.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(target).as_posix())
        (target / "RELEASE_SHA256SUMS").write_text("\n".join(manifest) + "\n")
        temporary_zip = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(target.rglob("*")):
                if path.is_file():
                    archive.write(path, "Research_Flow/" + path.relative_to(target).as_posix())
        os.replace(temporary_zip, destination)
    checksum = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(checksum + "  " + destination.name + "\n")
    return {"zip": str(destination), "sha256": checksum,
            "research_sources": len(state["research_discoveries"]), "journal_sequence": state["last_sequence"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(release(parser.parse_args().output), indent=2))
