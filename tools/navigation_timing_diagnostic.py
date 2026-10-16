#!/usr/bin/env python3
"""One exposed optimized navigation profile; no product/required-CI verdict."""
import argparse
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = "c448b1d8328610aa6af3f323ae75859561200283"
BASE_TREE = "998cc3df85f83fa4753c2ea2b80aa2a05ed60311"
ADDITIONS = {".github/workflows/navigation-timing-diagnostic.yml", "tools/navigation_timing_diagnostic.py"}
BRANCH = "refs/heads/codex/navigation-timing-20261016"
MIB, LOG_LIMIT, EXPECTED = 2**20, 16 * 2**20, 832


def stamp():
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "monotonic": time.monotonic(), "process_cpu": time.process_time()}


def save(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True)
        stream.write("\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=10)
    require(result.returncode == 0 and len(result.stdout) <= 8 * MIB, "bounded native Git read failed")
    return result.stdout


def tree_hash(rows):
    tree = {}
    for mode, kind, digest, path in rows:
        current = tree
        parts = path.split("/")
        require(not path.startswith("/") and all(p not in ("", ".", "..") for p in parts), "unsafe tracked path")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = (mode, digest)
    def encode(node):
        raw = b""
        for name, value in sorted(node.items(), key=lambda item: (item[0] + ("/" if isinstance(item[1], dict) else "")).encode()):
            mode, digest = ("40000", encode(value)) if isinstance(value, dict) else value
            raw += mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(digest)
        return hashlib.sha1(b"tree " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False).hexdigest()
    return encode(tree)


def inventory():
    rows, selected, total = [], {}, 0
    for entry in git("ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0"):
        if not entry:
            continue
        identity, raw_path = entry.split(b"\t", 1)
        mode, kind, blob = identity.decode("ascii").split()
        path = raw_path.decode("utf-8")
        rows.append([mode, kind, blob, path])
        if path.startswith(("tests/", "tools/", "architecture/", "docs/")) or path in ADDITIONS:
            source = ROOT / path
            info = source.lstat()
            require(mode in ("100644", "100755") and stat.S_ISREG(info.st_mode)
                    and info.st_size <= 4 * MIB, "selected source regular-file/size refusal")
            raw = source.read_bytes()
            total += len(raw)
            require(total <= 32 * MIB and hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw,
                    usedforsecurity=False).hexdigest() == blob, "selected worktree/source identity refusal")
            selected[path] = {"bytes": len(raw), "git_blob": blob, "sha256": hashlib.sha256(raw).hexdigest()}
    require(0 < len(rows) <= 20000, "tracked inventory bound")
    return {"tracked_tree": rows, "reconstructed_tree": tree_hash(rows), "selected_sources": selected,
            "selected_bytes": total}


def prepare(out):
    require(sys.version_info[:3] == (3, 11, 17), "Python3.11.17 required")
    require(os.environ.get("GITHUB_REPOSITORY") == "d6g8k5htny-coder/main"
            and os.environ.get("GITHUB_REF") == BRANCH
            and os.environ.get("GITHUB_EVENT_NAME") in ("push", "workflow_dispatch"), "native context refusal")
    raw_commit = git("cat-file", "commit", "HEAD")
    header, separator, _ = raw_commit.partition(b"\n\n")
    require(len(raw_commit) <= MIB and separator and len(header) <= 64 * 1024,
            "bounded raw HEAD commit header refusal")
    fields = [line.partition(b" ") for line in header.split(b"\n") if not line.startswith(b" ")]
    trees = [value.decode("ascii") for key, _, value in fields if key == b"tree"]
    parents = [value.decode("ascii") for key, _, value in fields if key == b"parent"]
    require(len(trees) == 1 and len(trees[0]) == 40
            and all(c in "0123456789abcdef" for c in trees[0]), "raw HEAD tree header refusal")
    tree = trees[0]
    commit = hashlib.sha1(b"commit " + str(len(raw_commit)).encode("ascii") + b"\0" + raw_commit,
                          usedforsecurity=False).hexdigest()
    require(commit == git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
            == os.environ.get("GITHUB_SHA") and parents == [BASE], "raw HEAD identity/sole-parent refusal")
    source = inventory()
    require(source["reconstructed_tree"] == tree, "native checked tree refusal")
    require(ADDITIONS <= {row[3] for row in source["tracked_tree"]}
            and tree_hash([row for row in source["tracked_tree"] if row[3] not in ADDITIONS]) == BASE_TREE,
            "only the two diagnostic additions may differ from exact base tree")
    save(out / "source-before.json", source)
    with (out / "raw-head-commit.txt").open("xb") as stream:
        stream.write(raw_commit)
    save(out / "native-context.json", {**stamp(), "repository": os.environ["GITHUB_REPOSITORY"],
         "ref": os.environ["GITHUB_REF"], "run_id": os.environ["GITHUB_RUN_ID"],
         "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "checked_commit": commit, "tree": tree,
         "parents": parents, "base": BASE, "base_tree": BASE_TREE, "expected_tests": EXPECTED,
         "head_identity_method": "raw commit headers and recomputed Git commit object hash; shallow-safe",
         "raw_head_commit_bytes": len(raw_commit), "raw_head_commit_sha256": hashlib.sha256(raw_commit).hexdigest(),
         "diagnostic_only": True, "scientific_effect": "NONE", "organizational_independence": 0,
         "cpu_clock_scope": "current worker process only; excludes subprocess CPU",
         "stdio_threshold_bytes": LOG_LIMIT, "threshold_kind": "externally observed; overshoot possible"})


def worker(out):
    require(sys.flags.optimize == 1 and sys.flags.no_site and sys.flags.dont_write_bytecode, "worker flags refusal")
    os.chdir(ROOT)
    sys.path[0] = str(ROOT)  # Match python -m unittest before discovery inserts tests/.
    events = (out / "timing.jsonl").open("xb", buffering=0)
    stacks = (out / "parent-stacks.log").open("xb", buffering=0)
    def event(kind, **fields):
        events.write((json.dumps({**stamp(), "event": kind, **fields}, ensure_ascii=True) + "\n").encode("ascii"))
    class TimedResult(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.starts, self.statuses = {}, {}
        def startTest(self, test):
            self.starts[test.id()] = (time.monotonic(), time.process_time())
            event("TEST_START", test_id=test.id())
            super().startTest(test)
        def stopTest(self, test):
            super().stopTest(test)
            wall, cpu = self.starts.pop(test.id())
            event("TEST_STOP", test_id=test.id(), status=self.statuses.get(test.id(), "completed"),
                  wall_seconds=time.monotonic() - wall, process_cpu_seconds=time.process_time() - cpu)
        def mark(self, test, status, **fields):
            owner = getattr(test, "test_case", test)
            self.statuses[owner.id()] = status
            event("TEST_STATUS", test_id=test.id(), parent_id=owner.id(), is_subtest=owner is not test,
                  status=status, **fields)
        def addSuccess(self, test):
            self.mark(test, "ok"); super().addSuccess(test)
        def addFailure(self, test, err):
            self.mark(test, "failure"); super().addFailure(test, err)
        def addError(self, test, err):
            self.mark(test, "error"); super().addError(test, err)
        def addSkip(self, test, reason):
            self.mark(test, "skip", reason=reason); super().addSkip(test, reason)
        def addExpectedFailure(self, test, err):
            self.mark(test, "expected_failure"); super().addExpectedFailure(test, err)
        def addUnexpectedSuccess(self, test):
            self.mark(test, "unexpected_success"); super().addUnexpectedSuccess(test)
        def addSubTest(self, test, subtest, err):
            status = "ok" if err is None else "failure" if issubclass(err[0], test.failureException) else "error"
            event("SUBTEST_STATUS", test_id=subtest.id(), parent_id=test.id(), status=status)
            if err is not None:
                self.statuses[test.id()] = "subtest_" + status
            super().addSubTest(test, subtest, err)
    class ProfileRunner(unittest.TextTestRunner):
        resultclass = TimedResult
    faulthandler.enable(file=stacks, all_threads=True)
    faulthandler.dump_traceback_later(30, repeat=True, file=stacks)
    try:
        event("DISCOVERY_START", optimize=sys.flags.optimize, python=sys.version,
              no_site=sys.flags.no_site, dont_write_bytecode=sys.flags.dont_write_bytecode,
              isolated=sys.flags.isolated, sys_path_0=sys.path[0], cwd=str(Path.cwd()),
              argv=["discover", "-s", "tests", "-p", "test_*.py", "-v"])
        suite = unittest.defaultTestLoader.discover(start_dir="tests", pattern="test_*.py")
        count = suite.countTestCases()
        result = ProfileRunner(verbosity=2, warnings="default" if not sys.warnoptions else None).run(suite)
        completed = count == EXPECTED and result.testsRun == count and count > 0
        save(out / "worker-result.json", {**stamp(), "profile_complete": completed,
             "discovered": count, "tests_run": result.testsRun, "unittest_success": result.wasSuccessful(),
             "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
             "full_test_skips": sum(not hasattr(test, "test_case") for test, _ in result.skipped),
             "subtest_skips": sum(hasattr(test, "test_case") for test, _ in result.skipped),
             "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses),
             "scope": "instrumented full optimized discovery; not required-CI or scientific acceptance"})
        return 0 if completed and result.wasSuccessful() else 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        stacks.close(); events.close()


def present(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def run(out):
    command = [sys.executable, "-B", "-O", "-S", str(Path(__file__).resolve()), "worker", str(out)]
    save(out / "command.json", {**stamp(), "argv": command, "deadline_seconds": 900, "term_grace_seconds": 10})
    process, reason, signals = None, None, []
    started = time.monotonic()
    try:
        with (out / "worker.stdout").open("xb") as stdout, (out / "worker.stderr").open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            save(out / "owned-process.json", {**stamp(), "pid": process.pid, "pgid": process.pid})
            while process.poll() is None:
                sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
                if any(sizes.get(name, 0) > LOG_LIMIT for name in ("worker.stdout", "worker.stderr", "timing.jsonl", "parent-stacks.log")):
                    reason = "OUTPUT_LIMIT"; break
                if time.monotonic() - started >= 900:
                    reason = "TIMEOUT"; break
                time.sleep(0.2)
    except BaseException as error:
        reason = type(error).__name__
    finally:
        if process is not None and present(process.pid):
            for signum, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 2)):
                if not present(process.pid):
                    break
                os.killpg(process.pid, signum); signals.append(signal.Signals(signum).name)
                until = time.monotonic() + grace
                while present(process.pid) and time.monotonic() < until:
                    process.poll()  # Reap the leader while waiting for its owned group.
                    time.sleep(0.1)
        if process is not None:
            process.wait(timeout=2)
    absent = process is not None and not present(process.pid)
    final_sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    overshoot = {name: final_sizes.get(name, 0) - LOG_LIMIT for name in
                 ("worker.stdout", "worker.stderr", "timing.jsonl", "parent-stacks.log") if final_sizes.get(name, 0) > LOG_LIMIT}
    if overshoot:
        reason = reason or "OUTPUT_LIMIT"
    after = inventory()
    save(out / "source-after.json", after)
    unchanged = after == json.loads((out / "source-before.json").read_text())
    result = json.loads((out / "worker-result.json").read_text()) if (out / "worker-result.json").is_file() else None
    passed = (process is not None and process.returncode == 0 and reason is None and not signals and absent
              and unchanged and result is not None and result["profile_complete"] and result["unittest_success"])
    save(out / "controller-result.json", {**stamp(), "elapsed_seconds": time.monotonic() - started,
         "returncode": process.returncode if process else None, "stop_reason": reason, "cleanup_signals": signals,
         "owned_process_group_absent": absent, "sources_unchanged": unchanged, "worker_result_present": result is not None,
         "passed_scoped_instrumented_profile": passed, "observed_file_bytes": final_sizes,
         "output_limit_observed": bool(overshoot), "threshold_overshoot_bytes": overshoot})
    return 0 if passed else 1


def seal(out):
    owned = json.loads((out / "owned-process.json").read_text()) if (out / "owned-process.json").is_file() else None
    require((owned is not None and not present(owned["pgid"])) or
            (owned is None and not (out / "command.json").exists()), "owned disposal unconfirmed; no upload clearance")
    rows = []
    for path in sorted(out.iterdir()):
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 64 * MIB, "unsafe/oversized retained output")
        rows.append({"path": path.name, "bytes": info.st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    require(len(rows) <= 32 and sum(r["bytes"] for r in rows) <= 128 * MIB, "retention inventory bound")
    save(out / "manifest.json", {**stamp(), "files": rows, "native_original_archive_readback": "REQUIRED_AFTER_UPLOAD",
         "scope": "retained diagnostic originals only; no required-CI or scientific acceptance"})
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="ascii") as stream:
        stream.write("safe_to_upload=true\n")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("action", choices=("prepare", "run", "worker", "seal")); parser.add_argument("output")
    args = parser.parse_args()
    out = Path(args.output)
    expected = Path(os.environ["RUNNER_TEMP"]) / ("navigation-timing-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"])
    require(out == expected and out.is_absolute() and out.is_dir() and not out.is_symlink(), "fixed owned output refusal")
    require(stat.S_IMODE(out.stat().st_mode) == 0o700 and out.stat().st_uid == os.getuid(), "private output ownership refusal")
    if args.action == "run":
        def interrupted(_signum, _frame):
            raise InterruptedError("diagnostic controller interrupted")
        signal.signal(signal.SIGTERM, interrupted)
    return {"prepare": prepare, "run": run, "worker": worker, "seal": seal}[args.action](out) or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
