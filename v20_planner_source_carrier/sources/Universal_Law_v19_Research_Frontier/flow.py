#!/usr/bin/env python3
"""Continuous local research intake connected to a durable proof-worker engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))
from conductor import Coordinator, Handler, Server, canonical
from model import digest
from research_evidence import MAX_BYTES, revision_id, validate_capture, verify_capture


def atomic_new(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("stored evidence object differs from its content address")
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class Intake:
    def __init__(self, coordinator, source_root, config):
        self.coord = coordinator
        self.source_root = Path(source_root).resolve()
        self.config = config
        if type(config.get("auto_register_evidence")) is not bool:
            raise ValueError("auto_register_evidence must be an explicit boolean")
        patterns = config.get("patterns", [])
        if not patterns or any(Path(p).is_absolute() or ".." in Path(p).parts for p in patterns):
            raise ValueError("watch patterns must remain inside the source root")
        if not isinstance(config.get("rules", {}), dict):
            raise ValueError("watch rules must be an object")
        namespace = config.get("source_namespace", "research")
        if (not isinstance(namespace, str) or not namespace or namespace != namespace.strip()
                or namespace in {".", ".."} or "/" in namespace or "\\" in namespace
                or any(ord(c) < 32 or ord(c) == 127 for c in namespace)):
            raise ValueError("source_namespace must be a nonempty single segment")
        self.source_namespace = namespace
        self.stability = {}

    def _owns_capture(self, capture):
        relative = capture.get("provenance", {}).get("source_relative_path")
        return (isinstance(relative, str)
                and capture.get("source_key") == self.source_namespace + "/" + relative)

    def _failure(self, source_key, reason):
        with self.coord._mutex, self.coord.journal.transaction():
            if self.coord.snapshot()["research_failures"].get(source_key) != reason:
                self.coord.journal.append("RESEARCH_CAPTURE_FAILED", {"source_key": source_key,
                                          "reason": reason})
                self.coord.publish_snapshot()

    def _recovered(self, source_key):
        with self.coord._mutex, self.coord.journal.transaction():
            if source_key in self.coord.snapshot()["research_failures"]:
                self.coord.journal.append("RESEARCH_CAPTURE_RECOVERED", {"source_key": source_key})

    def capture(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source path escapes watch root")
        path = self.source_root / relative
        cursor = path
        while cursor != self.source_root:
            if cursor.is_symlink():
                raise ValueError("symlinked research source refused")
            cursor = cursor.parent
        if not path.resolve().is_relative_to(self.source_root):
            raise ValueError("source path escapes watch root")
        before = path.stat()
        if not path.is_file() or before.st_size > MAX_BYTES:
            raise ValueError("research source exceeds bound or is not a regular file")
        raw = path.read_bytes()
        after = path.stat()
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after) or len(raw) != after.st_size:
            raise ValueError("source changed during capture; retry on the next scan")
        sha = hashlib.sha256(raw).hexdigest()
        name = relative.as_posix()
        rule = self.config.get("rules", {}).get(name, {})
        capture = {"source_key": self.source_namespace + "/" + name,
                   "title": rule.get("title", name), "sha256": sha, "bytes": len(raw),
                   "artifact_path": "research/objects/" + sha,
                   "classification": rule.get("classification", "unreviewed"),
                   "scope": rule.get("scope", "Unreviewed research artifact; mathematical scope remains to be assessed."),
                   "limitations": rule.get("limitations", ["Capture authenticates retained bytes only; applicability and mathematical correctness remain unvalidated."]),
                   "links": rule.get("links", []),
                   "provenance": {"acquisition_method": "local immutable byte capture",
                                  "source_relative_path": name,
                                  "source_root_at_capture": str(self.source_root),
                                  "declared_source_metadata": rule.get("provenance", {})}}
        capture["revision_id"] = revision_id(capture)
        validate_capture(capture, self.coord.base)
        with self.coord._mutex, self.coord.journal.transaction():
            destination = self.coord.journal.workdir / capture["artifact_path"]
            if destination.is_symlink() or not destination.resolve().is_relative_to(self.coord.journal.workdir.resolve()):
                raise ValueError("stored research path escapes runtime")
            atomic_new(destination, raw)
            verify_capture(self.coord.journal.workdir, capture)
            state = self.coord.snapshot()
            same = any(c["revision_id"] == capture["revision_id"] for c in state["research_discoveries"])
            failed = capture["source_key"] in state["research_failures"]
            if not same or failed:
                # A -> B -> A is a new source transition even though A's bytes
                # already exist. The serialized predecessor distinguishes it.
                eid = "research-capture:" + capture["revision_id"] + ":after:" + str(state["last_sequence"])
                self.coord.journal.append("RESEARCH_CAPTURED", capture, event_id=eid)
            return capture

    def recover_links(self):
        """Recover this namespace's outbox under this intake's registration policy."""
        count = 0
        for discovery in self.coord.snapshot()["research_discoveries"]:
            if (not self._owns_capture(discovery)
                    or discovery["capture_state"] != "captured"):
                continue
            capture = {k: v for k, v in discovery.items() if k != "capture_state"}
            for link in capture["links"]:
                with self.coord._mutex, self.coord.journal.transaction():
                    state = self.coord.snapshot()
                    current = next(c for c in state["research_discoveries"] if c["source_key"] == capture["source_key"])
                    if current["revision_id"] != capture["revision_id"] or current["capture_state"] != "captured":
                        continue
                    matches = [p for p in state["proposals"] if p["kind"] == "research_evidence"
                               and p["target"] == link["record_id"]
                               and json.loads(p["replacement"])["capture"]["revision_id"] == capture["revision_id"]]
                    active = state["research_evidence_attachments"].get(link["record_id"], [])
                    if any(v["capture"]["revision_id"] == capture["revision_id"] for v in active):
                        continue
                    if any(p["status"] == "rejected" for p in matches):
                        continue
                    pending = next((p for p in matches if p["ui_status"] == "awaiting_approval"), None)
                    if pending is None:
                        value = {"kind": "research_evidence", "record_id": link["record_id"],
                                 "capture": capture, "relation": link["relation"],
                                 "applicability": "unvalidated", "formal_status": "not_certified"}
                        capture_sequence = max(e["seq"] for e in self.coord.journal.since(0)
                            if e["type"] == "RESEARCH_CAPTURED" and e["payload"]["source_key"] == capture["source_key"])
                        eid = "research-link:" + digest({"revision": capture["revision_id"],
                                    "target": link["record_id"], "capture_sequence": capture_sequence,
                                    "branch": state["branch_sha256"]})
                        pending = self.coord.propose("research_evidence", link["record_id"], value,
                                                     actor="agent:research-intake", event_id=eid)
                        count += 1
                    if self.config["auto_register_evidence"]:
                        # Owner-configured administrative evidence registration;
                        # no theorem, import applicability or proof is accepted.
                        self.coord.decide(pending["proposal_id"], True,
                                          actor="owner:configured-evidence-registration",
                                          event_id="research-register:" + pending["proposal_id"])
        return count

    def scan(self, require_stable=False):
        captured = 0
        if not self.source_root.is_dir():
            self._failure(self.source_namespace + "/", "watch root is unavailable")
            return {"files_observed": 0, "links_staged": self.recover_links(), "root_unavailable": True}
        self._recovered(self.source_namespace + "/")
        paths = {p for pattern in self.config["patterns"] for p in self.source_root.glob(pattern)}
        observed = set()
        for path in sorted(paths):
            relative = path.relative_to(self.source_root)
            key = self.source_namespace + "/" + relative.as_posix()
            observed.add(key)
            try:
                stat = path.lstat()
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                previous = self.stability.get(key)
                self.stability[key] = signature
                if require_stable and signature != previous:
                    continue
                self.capture(relative)
                captured += 1
            except (OSError, ValueError) as error:
                self._failure(key, type(error).__name__ + ": " + str(error))
        for capture in self.coord.snapshot()["research_discoveries"]:
            if self._owns_capture(capture) and capture["source_key"] not in observed:
                self._failure(capture["source_key"], "source is no longer present in the watched selection; retained bytes remain preserved")
        links = self.recover_links()
        self.coord.publish_snapshot()
        return {"files_observed": len(paths), "files_captured_or_unchanged": captured, "links_staged": links}

    def watch(self, stop, interval=2):
        while not stop.is_set():
            try:
                self.scan(require_stable=True)
                self._recovered("intake/service")
            except Exception as error:
                self._failure("intake/service", type(error).__name__ + ": " + str(error))
            stop.wait(interval)


def verify_runtime(coordinator):
    state = coordinator.snapshot()
    captures = {e["payload"]["revision_id"]: e["payload"] for e in coordinator.journal.since(0)
                if e["type"] == "RESEARCH_CAPTURED"}
    errors = []
    for capture in captures.values():
        try:
            validate_capture(capture, coordinator.base)
            verify_capture(coordinator.journal.workdir, capture)
        except (ValueError, OSError) as error:
            errors.append({"revision_id": capture["revision_id"], "reason": str(error)})
    return {"result": "passed" if not errors else "failed", "evidence_revisions_checked": len(captures),
            "latest_discoveries": len(state["research_discoveries"]), "errors": errors,
            "full_theorem_certificates": state["full_theorem_certificates"]}


def snapshot_loop(coordinator, stop, directory, interval):
    """Periodically retain a consistent ZIP when the journal has advanced."""
    from release import release
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    last_sequence = None
    while not stop.wait(interval):
        try:
            state = coordinator.snapshot()
            if state["last_sequence"] == last_sequence or any(j["state"] == "running" for j in state["jobs"]):
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            result = release(directory / ("Research_Flow_" + stamp + ".zip"), coordinator.journal.workdir)
            last_sequence = result["journal_sequence"]
            temporary = directory / ("LATEST." + uuid.uuid4().hex + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(temporary, directory / "LATEST.json")
        except Exception as error:
            # A busy attempt can postpone a snapshot; recording the failure does
            # not change the evidence journal or interrupt research intake.
            (directory / "last-error.txt").write_text(type(error).__name__ + ": " + str(error) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=ROOT / "runtime")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "watch.json")
    sub = parser.add_subparsers(dest="command", required=True)
    once = sub.add_parser("once")
    once.add_argument("--checks", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8771)
    serve.add_argument("--interval", type=float, default=2)
    serve.add_argument("--snapshot-dir", type=Path, default=ROOT / "snapshots")
    serve.add_argument("--snapshot-seconds", type=float, default=300,
                       help="periodic portable snapshot interval; 0 disables")
    sub.add_parser("state")
    sub.add_parser("verify")
    sub.add_parser("check")
    args = parser.parse_args()
    coord = Coordinator(args.workdir, max_workers=2)
    if args.command in {"once", "serve"}:
        if args.source_root is None and args.command == "once":
            parser.error("once requires --source-root")
        config = json.loads(args.config.read_text())
        intake = Intake(coord, args.source_root, config) if args.source_root else None
        if args.command == "once":
            result = intake.scan()
            if args.checks:
                result["checks"] = coord.run_checks()
            result["custody"] = verify_runtime(coord)
            print(json.dumps(result, indent=2))
            return
        if args.interval < 0.25 or args.interval > 60:
            parser.error("interval must be between 0.25 and 60 seconds")
        if args.snapshot_seconds and args.snapshot_seconds < 30:
            parser.error("snapshot interval must be at least 30 seconds, or 0")
        coord.start()
        stop = threading.Event()
        thread = threading.Thread(target=intake.watch, args=(stop, args.interval), daemon=True) if intake else None
        if thread:
            thread.start()
        snapshotter = threading.Thread(target=snapshot_loop,
            args=(coord, stop, args.snapshot_dir, args.snapshot_seconds), daemon=True) if args.snapshot_seconds else None
        if snapshotter:
            snapshotter.start()
        server = Server(("127.0.0.1", args.port), Handler, coord)
        print(json.dumps({"dashboard": "http://127.0.0.1:" + str(server.server_port),
                          "continuous_intake": intake is not None,
                          "snapshot_interval_seconds": args.snapshot_seconds,
                          "source_root": str(args.source_root) if args.source_root else None}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            stop.set()
            if thread:
                thread.join(timeout=20)
            if snapshotter:
                snapshotter.join(timeout=20)
            coord.stop()
        return
    if args.command == "state":
        result = coord.snapshot()
    elif args.command == "check":
        result = coord.run_checks()
    else:
        result = verify_runtime(coord)
    print(json.dumps(result, indent=2))
    if args.command == "verify" and result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
