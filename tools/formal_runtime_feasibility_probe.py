"""Fixed, instrumented feasibility observation of two unchanged Lean methods.

This does not implement a Main regression controller, generated-input admission,
required formal check, G5/CBZ promotion, or scientific acceptance. No original
method, source input, expected result, or environment identity is rewritten.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import traceback

PURPOSE = "FORMAL_RUNTIME_FEASIBILITY_ONLY"
REPOSITORY = "d6g8k5htny-coder/main"
BRANCH = "refs/heads/codex/g5-formal-runtime-feasibility-20261110"
SUBJECT = "02973f3975a1c4cc91b7c51d372f65717c9b3c5c"
SUBJECT_TREE = "5d3160d982b4513f17848de45e0f4569233acd4e"
TOOLCHAIN = "leanprover/lean4:v4.34.1"
LEAN_COMMIT = "5045d0056413266e57c625dcd7c365b10e377c52"
ELAN_URL = "https://github.com/leanprover/elan/releases/download/v4.2.4/elan-x86_64-unknown-linux-gnu.tar.gz"
ELAN_SHA256 = "42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63"
METHODS = (
    "tests.test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt",
    "tests.test_formal_compiled_inventory.PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants",
)
PINS = {
    "tests/test_formal_gate.py": "aa8acd86661183a4d0b97704efaf025ee385bc3e",
    "tests/test_formal_compiled_inventory.py": "ce3b77bebd084c711acf5509ca56410ea3e56071",
    "tools/formal_gate_check.py": "aa2f90376f349ec6fca3d35a115990829d3d3f54",
    "tools/formal_compiled_inventory.py": "c037a8c3cdab114089a37f67aeb5d109277083ea",
    "formal/manifest.json": "7bd51fe201ade413fcd18314854c40ee4aae3df6",
    "formal/lake-manifest.json": "fc7655fad5ede55ad51ac99d0801a13a21f78dd7",
    "formal/lakefile.toml": "d986aa8d776636f3f47deaa6d17d41e79eb5b0d3",
    "formal/lean-toolchain": "ba8ebf2dbaf6a668cd2a0e086186d6d569b69ff5",
}
MIB = 1024 * 1024
# Initial observation ceilings, not measured adequate production budgets.
CHILD_SECONDS = 1800
TRACE_LIMIT = 128 * MIB
LOG_LIMIT = 16 * MIB
SOURCE_FILE_LIMIT = 256 * MIB
GENERATED_TOTAL_LIMIT = 256 * MIB
LEAN_BIN = None
TRACE_CALLS = (
    "open,openat,openat2,creat,rename,renameat,renameat2,unlink,unlinkat,"
    "mkdir,mkdirat,rmdir,link,linkat,symlink,symlinkat,truncate,ftruncate,"
    "chmod,fchmod,fchmodat,chown,fchown,fchownat,lchown,utime,utimes,"
    "utimensat,chdir,fchdir"
)
# No execve/execveat, process, read/write buffer, trace=all or environment dump.
MODIFY_CALLS = {
    "creat", "rename", "renameat", "renameat2", "unlink", "unlinkat",
    "mkdir", "mkdirat", "rmdir", "link", "linkat", "symlink", "symlinkat",
    "truncate", "ftruncate", "chmod", "fchmod", "fchmodat", "chown",
    "fchown", "fchownat", "lchown", "utime", "utimes", "utimensat",
}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def actual_directory(path):
    path = Path(os.path.abspath(path))
    require(path == path.resolve(strict=True), "directory path traverses a symlink")
    for component in (path, *path.parents):
        require(component.is_dir() and not component.is_symlink(),
                "directory component is not a real directory")
    return path


def regular(path, limit):
    path = Path(path)
    actual_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 <= before.st_size <= limit, "nonregular or oversized file")
        blocks = []
        remaining = limit + 1
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            blocks.append(block)
            remaining -= len(block)
        raw = b"".join(blocks)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        require(len(raw) == before.st_size <= limit
                and all(getattr(before, key) == getattr(after, key)
                        == getattr(named, key) for key in fields),
                "file changed during read")
        return raw
    finally:
        os.close(fd)


def write_json(path, value):
    raw = encoded(value)
    temporary = path.with_suffix(path.suffix + ".new")
    actual_directory(temporary.parent)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.geteuid(), "unsafe observation temporary")
        stream.write(raw)
    os.replace(temporary, path)


def native_context():
    require(os.environ.get("GITHUB_ACTIONS") == "true", "native Actions required")
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "wrong repository")
    require(os.environ.get("GITHUB_EVENT_NAME") == "push", "fixed push event required")
    require(os.environ.get("GITHUB_REF") == BRANCH, "wrong fixed probe branch")
    result = {key: os.environ.get(key) for key in
              ("GITHUB_REPOSITORY", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
               "GITHUB_SHA", "GITHUB_REF", "GITHUB_EVENT_NAME")}
    require(re.fullmatch(r"[0-9a-f]{40}", result["GITHUB_SHA"] or ""), "invalid native SHA")
    require(all(re.fullmatch(r"[1-9][0-9]*", result[key] or "")
                for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")), "invalid native run")
    return result


def environment(scratch=None):
    # HOME is inherited unchanged. Do not pass tokens, arbitrary Actions
    # service variables, Python/Git overrides, or injected shell startup.
    selected = {key: os.environ[key] for key in
                ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ")
                if key in os.environ}
    selected.update(PATH=str(LEAN_BIN if LEAN_BIN is not None else ELAN / "bin")
                    + os.pathsep + os.defpath,
                    ELAN_HOME=str(ELAN), GIT_CONFIG_NOSYSTEM="1",
                    GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1",
                    GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1",
                    PYTHONNOUSERSITE="1")
    selected.update(CONTEXT)
    selected["GITHUB_ACTIONS"] = "true"
    if scratch is not None:
        selected["TMPDIR"] = str(scratch)
    return selected


def git(root, *args):
    result = subprocess.run(
        ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
         "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
         "-C", str(root), *args], env=environment(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    require(result.returncode == 0 and len(result.stdout) <= 8 * MIB
            and len(result.stderr) <= MIB, "bounded fixed Git read failed")
    return result.stdout


def tool(label, command, cwd, limit_seconds, env=None):
    require(re.fullmatch(r"[a-z0-9-]+", label), "unsafe internal tool label")
    stdout = EVIDENCE / (label + ".stdout.bin")
    stderr = EVIDENCE / (label + ".stderr.bin")
    started = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env or environment(),
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=limit_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
    record = {"command": command, "cwd": str(cwd), "exit_code": code,
              "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started}
    for name, path in (("stdout", stdout), ("stderr", stderr)):
        raw = regular(path, LOG_LIMIT)
        record[name] = {"bytes": len(raw), "sha256": sha(raw)}
    write_json(EVIDENCE / (label + ".json"), record)
    require(code == 0 and not timed_out, "fixed native tool failed: " + label)
    return regular(stdout, LOG_LIMIT)


def mount_rows():
    raw = Path("/proc/self/mountinfo").read_bytes()
    require(len(raw) <= 8 * MIB, "mountinfo exceeds bound")
    def unescape(value):
        return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)
    result = {}
    for line in raw.decode("utf-8").splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        result[unescape(fields[4])] = {"mount_id": fields[0], "device": fields[2],
                                      "root": unescape(fields[3]),
                                      "options": fields[5], "filesystem": right.split()[0]}
    return result


def save_control():
    write_json(EVIDENCE / "control.json", CONTROL)


def bind(source, target, readonly):
    require(str(target) not in mount_rows(), "unexpected existing probe mount")
    tool("bind-" + str(len(CONTROL["mounts"])), ["sudo", "-n", "mount", "--bind",
         str(source), str(target)], BASE, 30)
    CONTROL["mounts"].append(str(target))
    save_control()
    if readonly:
        tool("seal-" + str(len(CONTROL["mounts"])), ["sudo", "-n", "mount", "-o",
             "remount,bind,ro", str(target)], BASE, 30)
    rows = mount_rows()
    require(str(target) in rows and ("ro" in rows[str(target)]["options"].split(","))
            == readonly, "requested mount state was not established")


def snapshot(root):
    require(git(root, "rev-parse", "--show-toplevel").decode().strip() == str(root),
            "subject is not actual worktree top-level")
    require(git(root, "rev-parse", "--verify", "HEAD").decode().strip() == SUBJECT
            and git(root, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
            == SUBJECT_TREE, "fixed subject HEAD/tree mismatch")
    rows = []
    for item in git(root, "ls-tree", "-r", "--full-tree", "-z", SUBJECT).split(b"\0"):
        if not item:
            continue
        metadata, relative = item.split(b"\t", 1)
        mode, kind, identity = metadata.decode("ascii").split()
        name = relative.decode("utf-8")
        require(kind == "blob" and mode in ("100644", "100755"), "unsupported tracked source")
        path = root / name
        raw = regular(path, SOURCE_FILE_LIMIT)
        actual_blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        require(actual_blob == identity
                and bool(path.stat().st_mode & stat.S_IXUSR) == (mode == "100755")
                and (name not in PINS or identity == PINS[name]), "tracked source bytes/mode changed")
        rows.append({"path": name, "git_blob": identity, "git_mode": mode,
                     "actual_mode": stat.S_IMODE(path.stat().st_mode),
                     "bytes": len(raw), "sha256": sha(raw)})
    require(set(PINS) <= {row["path"] for row in rows}, "missing exact original source")
    others = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    ignored = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    names = sorted(set(part.decode("utf-8") for part in (others + ignored).split(b"\0") if part))
    require(all(name.startswith("formal/.lake/") for name in names),
            "unexpected generated path outside exact real .lake")
    return {"subject_commit": SUBJECT, "subject_tree": SUBJECT_TREE, "files": rows,
            "generated_paths_observed": names}


def generated_inventory(root, destination):
    rows, total = [], 0
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in sorted(directories + names):
            path = Path(directory) / name
            info = path.lstat()
            relative = str(path.relative_to(root))
            require(len(rows) < 10000, "generated path count exceeds observation ceiling")
            if stat.S_ISDIR(info.st_mode):
                rows.append({"path": relative, "type": "directory"})
                continue
            require(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                    "generated special/symlink output requires HOLD")
            raw = regular(path, TRACE_LIMIT)
            total += len(raw)
            require(total <= GENERATED_TOTAL_LIMIT, "generated byte ceiling reached")
            rows.append({"path": relative, "type": "regular", "bytes": len(raw),
                         "sha256": sha(raw), "mode": stat.S_IMODE(info.st_mode)})
    write_json(destination, {"root": str(root), "bytes": total, "entries": rows})
    return rows


def bootstrap():
    global LEAN_BIN
    require(os.geteuid() != 0, "probe tests must not execute as root")
    require(all(shutil.which(name) for name in ("git", "sudo", "mount", "umount", "curl", "strace", "setpriv")),
            "fixed native tooling unavailable; no fallback")
    require(git(CONTROLLER, "rev-parse", "HEAD").decode().strip() == CONTEXT["GITHUB_SHA"],
            "controller native SHA differs from actual checkout")
    git(CONTROLLER, "cat-file", "-e", SUBJECT + "^{commit}")
    require(git(CONTROLLER, "rev-parse", SUBJECT + "^{tree}").decode().strip() == SUBJECT_TREE,
            "fixed Main object/tree absent")
    ELAN.mkdir(parents=True)
    archive = EVIDENCE / "elan-v4.2.4.tar.gz"
    tool("download-elan", ["curl", "--disable", "-sSfL", "--max-time", "300",
         "--max-filesize", str(16 * MIB), ELAN_URL, "-o", str(archive)], BASE, 330)
    raw = regular(archive, 16 * MIB)
    require(sha(raw) == ELAN_SHA256, "pinned elan archive digest mismatch")
    with tarfile.open(archive, "r:gz") as source:
        entries = [entry for entry in source.getmembers()
                   if entry.name in ("elan-init", "./elan-init")]
        require(len(entries) == 1 and entries[0].isfile()
                and entries[0].size <= 64 * MIB,
                "pinned elan-init member refused")
        raw_init = source.extractfile(entries[0]).read(64 * MIB + 1)
    require(len(raw_init) == entries[0].size <= 64 * MIB, "elan-init extent mismatch")
    installer = BASE / "elan-init"
    with installer.open("xb") as stream:
        stream.write(raw_init)
    installer.chmod(0o700)
    tool("install-elan", [str(installer), "-y", "--no-modify-path",
         "--default-toolchain", "none"], BASE, 300)
    tool("install-lean", [str(ELAN / "bin/elan"), "toolchain", "install", TOOLCHAIN], BASE, 900)
    selected_lake = tool("resolve-real-lake", [str(ELAN / "bin/elan"), "which",
                         "--toolchain", TOOLCHAIN, "lake"], BASE, 60).decode("utf-8").strip()
    lake_path = Path(selected_lake)
    require(lake_path.is_absolute() and lake_path == lake_path.resolve(strict=True)
            and (ELAN / "toolchains") in lake_path.parents and lake_path.name == "lake",
            "elan did not select the genuine pinned toolchain lake")
    LEAN_BIN = actual_directory(lake_path.parent)
    binaries = {}
    for name in ("lean", "lake", "leanchecker"):
        path = LEAN_BIN / name
        raw_binary = regular(path, SOURCE_FILE_LIMIT)
        binaries[name] = {"path": str(path), "bytes": len(raw_binary),
                          "sha256": sha(raw_binary), "mode": stat.S_IMODE(path.stat().st_mode)}
    write_json(EVIDENCE / "runtime-binaries.json", {
        "elan_release_archive_sha256": ELAN_SHA256, "toolchain": TOOLCHAIN,
        "toolchain_archive_sha256": None, "toolchain_archive_hash_claim": False,
        "selected_binaries": binaries, "uid": os.geteuid(), "gid": os.getegid(),
        "python": sys.version, "python_executable": sys.executable})
    version = tool("actual-lean-version", [str(LEAN_BIN / "lean"), "--version"],
                   BASE, 60).decode("utf-8").strip()
    require(re.fullmatch(r"Lean \(version 4\.34\.1, [A-Za-z0-9_.+-]+, commit "
                         + LEAN_COMMIT + r", Release\)", version), "real Lean version mismatch")
    tool("strace-version", ["strace", "--version"], BASE, 30)
    tool("setpriv-version", ["setpriv", "--version"], BASE, 30)
    for mode in ("normal", "optimized"):
        root = SUBJECTS / mode
        tool("worktree-" + mode, ["git", "--no-replace-objects", "-C", str(CONTROLLER),
             "worktree", "add", "--detach", str(root), SUBJECT], BASE, 120)
        actual_directory(root)
        require(root.stat().st_uid == os.geteuid(), "subject ownership differs from test UID")
        (root / "formal/.lake").mkdir()
        (GENERATED / mode).mkdir()
        (SCRATCH / mode).mkdir()
        (EVIDENCE / mode).mkdir()
        before = snapshot(root)
        require(not before["generated_paths_observed"], "subject .lake is not initially fresh")
        write_json(EVIDENCE / mode / "source-preparation.json", before)
    common = Path(git(CONTROLLER, "rev-parse", "--git-common-dir").decode().strip())
    if not common.is_absolute():
        common = CONTROLLER / common
    common = actual_directory(common)
    require(common == actual_directory(CONTROLLER / ".git")
            and common.stat().st_uid == os.geteuid(), "unexpected common Git storage")
    require(not (common / "objects/info/alternates").exists(),
            "unadmitted alternate Git object store")
    CONTROL["common_git"] = str(common)
    save_control()
    bind(common, common, True)
    bind(ELAN, ELAN, True)
    for mode in ("normal", "optimized"):
        root = SUBJECTS / mode
        bind(root, root, True)
        bind(GENERATED / mode, root / "formal/.lake", False)


def trace_census(raw, roots):
    """Bounded conservative index of actual original mutation/intent lines."""
    raw.decode("utf-8", errors="strict")
    records, concerns, unresolved = [], [], 0
    offset, indexed_bytes, budget = 0, 0, 0
    partial = None
    lines = raw.splitlines(keepends=True)
    for number, raw_line in enumerate(lines, 1):
        line = raw_line.decode("utf-8")
        match = re.search(r"\b([a-z][a-z0-9_]*)\(", line)
        resumed = re.search(r"<\.\.\. ([a-z][a-z0-9_]*) resumed>", line)
        operation = match[1] if match else (resumed[1] if resumed else None)
        intent = operation in MODIFY_CALLS or (
            operation in ("open", "openat", "openat2")
            and (resumed is not None or any(flag in line for flag in
                 ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"))))
        if not intent:
            offset += len(raw_line)
            indexed_bytes = offset
            continue
        paths, parse_error = [], None
        try:
            for literal in re.findall(r'"(?:\\.|[^"\\])*"', line):
                value = ast.literal_eval(literal)
                if isinstance(value, str) and value.startswith("/"):
                    paths.append(value)
            paths.extend(path.removesuffix(" (deleted)")
                         for path in re.findall(r"<(/[^>]*)>", line))
        except Exception as error:
            parse_error = type(error).__name__ + ": " + str(error)
        paths = sorted(set(paths))
        outside = [path for path in paths if path != "/dev/null"
                   and not any(Path(path) == root or root in Path(path).parents for root in roots)]
        denied = "= -1 " in line
        ambiguous = not paths or resumed is not None or "<unfinished ...>" in line
        row = {"operation": operation, "trace_line_number": number,
               "trace_byte_offset": offset, "trace_original_bytes": len(raw_line),
               "trace_line_sha256": sha(raw_line), "original_trace_utf8": line,
               "write_intent_or_path_mutation": True, "absolute_paths": paths,
               "outside_fixed_writable_roots": outside, "native_failure": denied,
               "unresolved_or_incomplete": ambiguous, "parse_error": parse_error}
        concern = {"record_index": len(records), "outside_paths": outside,
                   "native_failure": denied, "unresolved": ambiguous,
                   "parse_error": parse_error}
        increment = len(encoded(row)) + len(encoded(concern))
        if len(records) >= 4096 or len(raw_line) > 16384 or budget + increment > 4 * MIB:
            partial = "HOLD_WRITE_INDEX_OBSERVATION_CEILING"
            break
        records.append(row)
        budget += increment
        if ambiguous:
            unresolved += 1
        if outside or denied or ambiguous or parse_error is not None:
            concerns.append(concern)
        offset += len(raw_line)
        indexed_bytes = offset
    result = {"scope": "PATH_SYSCALLS_AND_WRITE_INTENTS_NOT_PAYLOAD_TRACE",
            "status": "PARTIAL_HOLD" if partial else "COMPLETE_BOUNDED_INDEX",
            "partial_reason": partial, "raw_trace_bytes": len(raw),
            "raw_trace_sha256": sha(raw), "raw_trace_lines": len(lines),
            "indexed_trace_prefix_bytes": indexed_bytes,
            "all_trace_lines_considered": partial is None,
            "review_index_only": True, "manual_write_review_required": True,
            "unresolved_write_path_lines": unresolved, "records": records,
            "hold_concerns": concerns}
    # The printed/retained original JSON has a hard encoded bound, including
    # array separators and envelope fields, not just the incremental estimate.
    while len(encoded(result)) > 4 * MIB:
        require(bool(records), "fixed census envelope exceeds observation bound")
        removed = records.pop()
        result.update(status="PARTIAL_HOLD",
                      partial_reason="HOLD_WRITE_INDEX_SERIALIZED_CEILING",
                      indexed_trace_prefix_bytes=min(result["indexed_trace_prefix_bytes"],
                                                     removed["trace_byte_offset"]),
                      all_trace_lines_considered=False,
                      unresolved_write_path_lines=sum(bool(row["unresolved_or_incomplete"])
                                                      for row in records))
        result["hold_concerns"] = [row for row in result["hold_concerns"]
                                   if row["record_index"] < len(records)]
    return result


def collect_receipts(mode):
    """Preserve bounded originals even when the observation is already HOLD."""
    source = GENERATED / mode / "formal-evidence"
    result = {"status": "NOT_PRODUCED", "copied": [], "refusals": [],
              "receipt_status": "NOT_PRODUCED"}
    if not os.path.lexists(source):
        return result
    try:
        actual_directory(source)
        target = EVIDENCE / mode / "formal-evidence"
        target.mkdir(exist_ok=True)
        actual_directory(target)
        paths = sorted(source.iterdir())
        if len(paths) > 128:
            result["refusals"].append("receipt/log count exceeds initial observation ceiling")
        total = 0
        for path in paths[:128]:
            try:
                require(path.is_file() and not path.is_symlink(), "nonregular receipt/log")
                raw = regular(path, 8 * MIB)
                total += len(raw)
                require(total <= 64 * MIB, "receipt/log bytes exceed initial observation ceiling")
                destination = target / path.name
                if os.path.lexists(destination):
                    require(regular(destination, 8 * MIB) == raw,
                            "retained original differs; refusing overwrite")
                else:
                    with destination.open("xb") as stream:
                        stream.write(raw)
                result["copied"].append({"name": path.name, "bytes": len(raw), "sha256": sha(raw)})
            except Exception as error:
                result["refusals"].append({"name": path.name, "error_type": type(error).__name__,
                                           "error": str(error)})
        receipt_path = target / "receipt.json"
        if receipt_path.is_file() and not receipt_path.is_symlink():
            receipt = json.loads(regular(receipt_path, 8 * MIB).decode("utf-8"))
            valid = (receipt.get("checked_commit") == SUBJECT
                     and receipt.get("formalization_status") == "kernel-checked"
                     and receipt.get("alignment_status") == "PENDING_INDEPENDENT_REVIEW"
                     and receipt.get("scientific_effect") == "NONE"
                     and receipt.get("scientific_status_authority") is False)
            result["receipt_status"] = "SOURCE_BOUND_ORIGINAL" if valid else "REFUSED"
            if not valid:
                result["refusals"].append("original receipt source/scope mismatch")
        result["status"] = "PARTIAL_OR_REFUSED" if result["refusals"] else "ORIGINALS_RETAINED"
    except Exception as error:
        result["status"] = "PARTIAL_OR_REFUSED"
        result["refusals"].append({"error_type": type(error).__name__, "error": str(error)})
    return result


def emit_original(path, label, budget, trace_prefix=False):
    """Print complete bounded original UTF-8, or an honestly labelled prefix."""
    if not os.path.lexists(path):
        print("FORMAL_PROBE_NOT_PRODUCED", label, flush=True)
        return {"label": label, "status": "NOT_PRODUCED"}
    raw = regular(path, TRACE_LIMIT)
    raw.decode("utf-8", errors="strict")
    full = not trace_prefix and len(raw) <= 8 * MIB and len(raw) <= budget["remaining"]
    count = len(raw) if full else min(65536, len(raw), budget["remaining"])
    shown = raw[:count]
    try:
        shown.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        # Full raw text was valid; only an incomplete trailing prefix codepoint
        # may be removed. No replacement or reconstruction of original bytes.
        require(error.reason == "unexpected end of data" and error.end == len(shown),
                "unexpected UTF-8 prefix refusal")
        shown = shown[:error.start]
        shown.decode("utf-8", errors="strict")
    budget["remaining"] -= len(shown)
    record = {"label": label, "status": "FULL_ORIGINAL" if full else
              ("RAW_TRACE_PREFIX_ONLY" if trace_prefix else "HOLD_VISIBILITY_CEILING"),
              "original_bytes": len(raw), "original_sha256": sha(raw),
              "printed_original_bytes": len(shown), "full_original_printed": full,
              "full_original_retained_in_artifact": True}
    nonce = os.urandom(32).hex()
    while nonce.encode("ascii") in shown:
        nonce = os.urandom(32).hex()
    print("::stop-commands::" + nonce, flush=True)
    try:
        print("FORMAL_PROBE_ORIGINAL " + json.dumps(record, sort_keys=True), flush=True)
        sys.stdout.buffer.write(shown)
        if shown and not shown.endswith(b"\n"):
            sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        if not full:
            print("FORMAL_PROBE_PRINT_PREFIX_ONLY", label,
                  "full_original_retained_in_artifact=true", flush=True)
    finally:
        print("::" + nonce + "::", flush=True)
    return record


def print_available_originals():
    """One fixed always phase: no tests, downloads, or original-file rewrites."""
    if not EVIDENCE.is_dir():
        print("FORMAL_PROBE_EVIDENCE_NOT_PRODUCED; HOLD, no assumed execution", flush=True)
        return 1
    budget = {"remaining": 16 * MIB}
    visibility = {"purpose": PURPOSE, "status": "AVAILABLE_ORIGINALS_PRINTED",
                  "native_controller": CONTEXT, "material_byte_ceiling": 16 * MIB,
                  "raw_trace_full_printed": False, "snapshots_full_printed": False,
                  "no_new_test_execution": True, "records": []}
    def show(path, label, trace_prefix=False):
        try:
            record = emit_original(path, label, budget, trace_prefix)
        except Exception as error:
            record = {"label": label, "status": "HOLD_READ_OR_UTF8_REFUSAL",
                      "error_type": type(error).__name__, "error": str(error)}
            print("FORMAL_PROBE_VISIBILITY_HOLD " + json.dumps(record, sort_keys=True), flush=True)
        visibility["records"].append(record)
        if record["status"].startswith("HOLD"):
            visibility["status"] = "HOLD"
    for name in ("probe.json", "cleanup.json", "control.json", "runtime-binaries.json",
                 "observer-error.log"):
        show(EVIDENCE / name, name)
    # Fixed internally named native tool streams; no source-tree or arbitrary
    # user path is accepted. Environment values and tokens were not dumped.
    for path in sorted(EVIDENCE.iterdir()):
        if re.fullmatch(r"[a-z0-9-]+\.(?:stdout|stderr)\.bin", path.name):
            show(path, path.name)
    for mode in ("normal", "optimized"):
        destination = EVIDENCE / mode
        for name in ("stdout.bin", "stderr.bin", "observation.json",
                     "mounts-before.json", "mounts-after.json", "write-census.json",
                     "generated-after.json", "scratch-after.json", "observer-error.log",
                     "retention-cleanup-1.json", "retention-cleanup-2.json"):
            show(destination / name, mode + "/" + name)
        for name in ("source-preparation.json", "source-before.json", "source-after.json"):
            path, label = destination / name, mode + "/" + name
            if not os.path.lexists(path):
                print("FORMAL_PROBE_NOT_PRODUCED", label, flush=True)
                continue
            try:
                raw = regular(path, 8 * MIB)
                document = json.loads(raw.decode("utf-8"))
                rows = {row["path"]: row for row in document["files"]}
                summary = {"label": label, "original_bytes": len(raw),
                           "original_sha256": sha(raw), "tracked_count": len(rows),
                           "subject_commit": document["subject_commit"],
                           "subject_tree": document["subject_tree"],
                           "known_original_pins": {name: rows[name]["git_blob"] for name in PINS},
                           "full_snapshot_member_printed": False,
                           "full_snapshot_retained_in_artifact": True}
                print("FORMAL_PROBE_SOURCE_SNAPSHOT_SUMMARY "
                      + json.dumps(summary, sort_keys=True), flush=True)
            except Exception as error:
                visibility["status"] = "HOLD"
                print("FORMAL_PROBE_SOURCE_SNAPSHOT_HOLD", label,
                      type(error).__name__, str(error), flush=True)
        # Full bounded original receipt and all retained original gate logs.
        receipts = destination / "formal-evidence"
        show(receipts / "receipt.json", mode + "/formal-evidence/receipt.json")
        if receipts.is_dir() and not receipts.is_symlink():
            for path in sorted(receipts.iterdir()):
                if path.suffix == ".log":
                    show(path, mode + "/formal-evidence/" + path.name)
        # Retain 128 MiB trace ceiling but disclose only its safe UTF-8 prefix.
        show(destination / "file-syscalls.log", mode + "/file-syscalls.log", True)
    visibility["original_material_bytes_printed"] = 16 * MIB - budget["remaining"]
    write_json(EVIDENCE / "visibility.json", visibility)
    print("FORMAL_PROBE_VISIBILITY " + json.dumps(visibility, sort_keys=True), flush=True)
    return 0 if visibility["status"] != "HOLD" else 1


def run_mode(mode):
    root, scratch, destination = SUBJECTS / mode, SCRATCH / mode, EVIDENCE / mode
    result = {"mode": mode, "status": "STARTED", "executed": False,
              "subject_commit": SUBJECT, "subject_tree": SUBJECT_TREE,
              "native_context": CONTEXT, "expected_original_ids": list(METHODS),
              "timing_kind": "INSTRUMENTED_OBSERVATION_ONLY",
              "child_observation_seconds": CHILD_SECONDS,
              "production_budget_guarantee": False}
    process = None
    try:
        rows = mount_rows()
        require("ro" in rows[str(root)]["options"].split(",")
                and "ro" not in rows[str(root / "formal/.lake")]["options"].split(","),
                "source/generated mount state changed")
        require(not any((GENERATED / mode).iterdir()), "mode .lake is not fresh")
        executable = scratch / "fixed-exec-check.sh"
        executable.write_bytes(b"#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        tool("exec-scratch-" + mode, [str(executable)], scratch, 15, environment(scratch))
        before = snapshot(root)
        write_json(destination / "source-before.json", before)
        write_json(destination / "mounts-before.json", mount_rows())
        command = [sys.executable, "-B"] + (["-O"] if mode == "optimized" else [])
        command += ["-S", "-m", "unittest", "-v", *METHODS]
        trace = destination / "file-syscalls.log"
        traced = ["setpriv", "--no-new-privs", "--", "strace", "-f", "-qq", "-ttt",
                  "-T", "-yy", "-s", "4096", "-e", "trace=" + TRACE_CALLS,
                  "-o", str(trace), "--", *command]
        def limits():
            resource.setrlimit(resource.RLIMIT_FSIZE, (TRACE_LIMIT, TRACE_LIMIT))
        started = time.monotonic()
        with (destination / "stdout.bin").open("xb") as stdout, \
                (destination / "stderr.bin").open("xb") as stderr:
            process = subprocess.Popen(traced, cwd=root, env=environment(scratch),
                                       stdin=subprocess.DEVNULL, stdout=stdout,
                                       stderr=stderr, start_new_session=True,
                                       preexec_fn=limits)
            result["executed"] = True
            result["command"] = command
            result["trace_command"] = traced
            try:
                starttime = Path("/proc/" + str(process.pid) + "/stat").read_text().split()[21]
            except FileNotFoundError:
                starttime = None
            CONTROL["sessions"][mode] = {"pid": process.pid, "starttime": starttime,
                                        "quiescent": False}
            save_control()
            result["timed_out"] = False
            try:
                result["native_exit_code"] = process.wait(timeout=CHILD_SECONDS)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                os.killpg(process.pid, signal.SIGKILL)
                result["native_exit_code"] = process.wait(timeout=10)
        result["instrumented_elapsed_seconds"] = time.monotonic() - started
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            result["writers_quiescent"] = True
            CONTROL["sessions"][mode]["quiescent"] = True
            save_control()
        else:
            raise RuntimeError("tracked process group still exists; collection held")
        after = snapshot(root)
        write_json(destination / "source-after.json", after)
        require(before["files"] == after["files"], "tracked source bytes/modes changed")
        result["tracked_source_unchanged"] = True
        write_json(destination / "mounts-after.json", mount_rows())
        raw_trace = regular(trace, TRACE_LIMIT)
        require(len(raw_trace) < TRACE_LIMIT, "trace/file ceiling reached")
        census = trace_census(raw_trace, (root / "formal/.lake", GENERATED / mode, scratch, destination))
        write_json(destination / "write-census.json", census)
        generated_inventory(GENERATED / mode, destination / "generated-after.json")
        generated_inventory(scratch, destination / "scratch-after.json")
        raw_out = regular(destination / "stdout.bin", LOG_LIMIT)
        raw_err = regular(destination / "stderr.bin", LOG_LIMIT)
        text = raw_err.decode("utf-8", errors="strict")
        ids = re.findall(r"^\S+ \(([^()]*)\) \.\.\. ok$", text, re.M)
        summaries = re.findall(r"^Ran ([0-9]+) tests? in [0-9]+(?:\.[0-9]+)?s$", text, re.M)
        terminal = [line for line in text.splitlines() if line.strip()]
        result["observed_original_ok_ids"] = ids
        result["original_methods_successful"] = bool(
            result["native_exit_code"] == 0 and not result["timed_out"]
            and ids == list(METHODS) and len(set(ids)) == 2 and summaries == ["2"]
            and terminal and terminal[-1] == "OK")
        result["streams"] = {
            "stdout": {"bytes": len(raw_out), "sha256": sha(raw_out)},
            "stderr": {"bytes": len(raw_err), "sha256": sha(raw_err)},
            "trace": {"bytes": len(raw_trace), "sha256": sha(raw_trace)}}
        result["manual_write_review_required"] = True
        result["write_concern_count"] = len(census["hold_concerns"])
        result["unresolved_write_path_lines"] = census["unresolved_write_path_lines"]
        require(result["original_methods_successful"], "original two-method outcome is not clean success")
        require(census["status"] == "COMPLETE_BOUNDED_INDEX"
                and not census["hold_concerns"] and not census["unresolved_write_path_lines"],
                "HOLD for outside/failed/unresolved native write census")
        result["status"] = "OBSERVATION_COMPLETE_REQUIRES_INDEPENDENT_WRITE_REVIEW"
    except Exception as error:
        result["status"] = "HOLD"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        (destination / "observer-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        if process is not None and not result.get("writers_quiescent"):
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                result["writers_quiescent"] = True
                CONTROL["sessions"][mode]["quiescent"] = True
                save_control()
        # Retention is independent of successful census parsing/ceilings.
        # Never replace the primary original/observer HOLD with a later error.
        if result.get("writers_quiescent"):
            retained = collect_receipts(mode)
            result["formal_evidence_retention"] = retained
            if retained["refusals"] or (result.get("original_methods_successful")
                                      and retained["receipt_status"] != "SOURCE_BOUND_ORIGINAL"):
                if result["status"] != "HOLD":
                    result.update(status="HOLD", error="original evidence retention refused")
        else:
            result["formal_evidence_retention"] = {
                "status": "NOT_READ_WRITER_QUIESCENCE_UNCONFIRMED"}
        write_json(destination / "observation.json", result)
    return result


def cleanup():
    failures = []
    # Never trust a recorded arbitrary mount path or PID as an authority.
    valid = {str(ELAN), str(CONTROLLER / ".git")}
    valid |= {str(SUBJECTS / mode) for mode in ("normal", "optimized")}
    valid |= {str(SUBJECTS / mode / "formal/.lake") for mode in ("normal", "optimized")}
    attempt = CONTROL.get("cleanup_attempt", 0) + 1
    require(type(attempt) is int and 1 <= attempt <= 2, "unexpected cleanup repetition")
    CONTROL["cleanup_attempt"] = attempt
    save_control()
    for mode, session in CONTROL.get("sessions", {}).items():
        if session.get("quiescent") is True:
            continue
        pid = session["pid"]
        try:
            current = Path("/proc/" + str(pid) + "/stat").read_text().split()
        except FileNotFoundError:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                continue
            failures.append("leader absent but process group remains: " + mode)
            continue
        if current[21] != session["starttime"] or int(current[4]) != pid:
            failures.append("process identity changed; not signalling: " + mode)
            continue
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(40):
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            failures.append("process group not quiescent: " + mode)
    if not failures:
        for mode in ("normal", "optimized"):
            if (EVIDENCE / mode).is_dir():
                retained = collect_receipts(mode)
                write_json(EVIDENCE / mode / ("retention-cleanup-" + str(attempt) + ".json"), retained)
        for target in reversed(CONTROL.get("mounts", [])):
            if target not in valid:
                failures.append("unexpected recorded mount, not unmounting")
                continue
            if target not in mount_rows():
                continue
            try:
                tool("unmount-" + str(attempt) + "-" + str(CONTROL["mounts"].index(target)),
                     ["sudo", "-n", "umount", target], BASE, 30)
            except Exception as error:
                failures.append(type(error).__name__ + ": " + str(error))
    write_json(EVIDENCE / "cleanup.json", {"status": "HOLD" if failures else "COMPLETE",
               "failures": failures, "remaining_probe_mounts":
               sorted(set(CONTROL.get("mounts", [])) & set(mount_rows()))})
    return not failures


def main():
    global CONTEXT, CONTROLLER, BASE, EVIDENCE, ELAN, SUBJECTS, GENERATED, SCRATCH, CONTROL
    require(len(sys.argv) == 1 or sys.argv[1:] in (["cleanup"], ["print"]),
            "only fixed run/cleanup/print phases; no generic command inputs")
    CONTEXT = native_context()
    CONTROLLER = actual_directory(os.environ["GITHUB_WORKSPACE"])
    temporary = actual_directory(os.environ["RUNNER_TEMP"])
    BASE = temporary / "g5-formal-runtime-feasibility-20261110"
    require(not BASE.is_relative_to(CONTROLLER), "probe outputs would enter source")
    EVIDENCE, ELAN = BASE / "evidence", BASE / "runtime/elan"
    SUBJECTS, GENERATED, SCRATCH = BASE / "subjects", BASE / "generated", BASE / "scratch"
    if sys.argv[1:] == ["print"]:
        return print_available_originals()
    if sys.argv[1:] == ["cleanup"]:
        if not (EVIDENCE / "control.json").is_file():
            print("FORMAL_PROBE_CLEANUP_NOT_PRODUCED; no assumed successful run")
            return 0
        CONTROL = json.loads(regular(EVIDENCE / "control.json", MIB).decode("utf-8"))
        require(CONTROL["native_context"] == CONTEXT, "cleanup context mismatch")
        return 0 if cleanup() else 1
    BASE.mkdir(mode=0o700)
    for path in (EVIDENCE, SUBJECTS, GENERATED, SCRATCH, BASE / "runtime"):
        path.mkdir(mode=0o700)
    CONTROL = {"native_context": CONTEXT, "mounts": [], "sessions": {}}
    save_control()
    report = {"purpose": PURPOSE, "native_controller": CONTEXT,
              "fixed_original_subject": {"commit": SUBJECT, "tree": SUBJECT_TREE, "pins": PINS},
              "methods": list(METHODS), "originals_modified": False,
              "initial_limits_not_measured_adequacy": {"child_seconds_per_mode": CHILD_SECONDS,
                   "trace_and_child_file_bytes": TRACE_LIMIT},
              "timing_kind": "INSTRUMENTED_OBSERVATION_ONLY", "production_budget_guarantee": False,
              "G5_claim": False, "CBZ_claim": False, "scientific_effect": "NONE",
              "feasibility_acceptance": "INDEPENDENT_WRITE_REVIEW_REQUIRED", "modes": []}
    try:
        bootstrap()
        for mode in ("normal", "optimized"):
            report["modes"].append(run_mode(mode))
        report["status"] = ("OBSERVATIONS_RETAINED_REVIEW_REQUIRED"
                            if all(row["status"] != "HOLD" for row in report["modes"]) else "HOLD")
    except Exception as error:
        report["status"] = "HOLD"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        (EVIDENCE / "observer-error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if not cleanup():
            report["status"] = "HOLD"
        write_json(EVIDENCE / "probe.json", report)
        print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] != "HOLD" else 1


if __name__ == "__main__":
    raise SystemExit(main())
