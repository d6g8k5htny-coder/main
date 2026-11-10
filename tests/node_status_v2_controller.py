import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
import sys

MODE = os.environ["NODE_STATUS_V2_MODE"]
if MODE not in ("normal", "optimized") or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise ValueError("offline mode/bytecode prerequisites missing")
if os.environ["GITHUB_REPOSITORY"] != "d6g8k5htny-coder/main":
    raise ValueError("wrong native repository")
ROOT = Path(os.environ["GITHUB_WORKSPACE"]).resolve(strict=True)
TEMP = Path(os.environ["RUNNER_TEMP"])
if (not TEMP.is_absolute() or TEMP.is_symlink()
        or TEMP.resolve(strict=True) != TEMP or TEMP.is_relative_to(ROOT)):
    raise ValueError("evidence root is not external RUNNER_TEMP")
RUN, ATTEMPT = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
if not all(re.fullmatch(r"[1-9][0-9]*", v) for v in (RUN, ATTEMPT)):
    raise ValueError("invalid native run identity")
OUT = TEMP / ("node-status-v2-offline-" + MODE + "-" + RUN + "-" + ATTEMPT)
OUT.mkdir(mode=0o700, exist_ok=True)
LEGACY = TEMP / ("node-status-legacy-v1-" + MODE)
TEST_INVOCATIONS = TEMP / ("node-status-v2-test-invocations-" + MODE)
if Path(os.environ["NODE_STATUS_V2_TEST_EVIDENCE"]) != TEST_INVOCATIONS:
    raise ValueError("fixed TEST invocation root differs")
if Path(os.environ["NODE_STATUS_LEGACY_V1_EVIDENCE"]) != LEGACY:
    raise ValueError("fixed legacy evidence root differs")
PROTECTED = {
    ".github/workflows/research-architecture.yml": "69ad3b47537ab8e4f8936affa3e09fccbdcfd1e5",
    "tools/unittest_timing_probe.py": "05fb481b5538c3f517174b00948de45d2849387a",
    "tools/architecture_run_receipt.py": "a6062afa9261958b8d8b85853a9c43031385b2c1",
    "architecture/graph_artifact_readback.py": "c39ffd0fdcde1fd5f895c57df5f770a582918a8a",
    "tools/architecture_check_binding.py": "53ea34e4fffc1087770584744939904844a41cff",
    "tools/unittest_full_qualification.py": "02dc2f3222aaf3e3d4d024659fffff22d6fb5e66",
    "tests/test_unittest_full_qualification.py": "2fe09abce9a8ef9356fd0af85780dcde75179386",
    ".github/workflows/full-unittest-qualification.yml": "ff8226cdeea57ee2ddb41c58c8c93e644343557d",
    "tools/unittest_native_acquisition.py": "1532884b5a2b0f1bc4040c61763c894cbadc8730",
    "tests/test_unittest_native_acquisition.py": "210242d2fcb5a2045db035c9eed4fbee9d43dad0",
    ".github/workflows/unittest-native-acquisition-contracts.yml": "c57d7868ae509acd2e5dd465a87072783983ce0d",
    "tools/node_status_v1.py": "e63dc0dad57172158927c5bd61e3bed7c18d509e",
    "architecture/node-status-v1.schema.json": "05f5dbf110497bbc89fd6030df2c21c1d8ae52ac",
    "tests/test_node_status_v1.py": "debc4b3f07f8d47dbf1b1b6f3be93ce225627b7a",
    ".github/workflows/node-status-v1-contract.yml": "1b2b021170254f28dc5e809100c53c198240d237",
    "architecture/node-status.schema.json": "2bad95d30bcb89ea686d1ac5a50d52113dd730be",
    "docs/site/dependency-source/GRAPH.json": "220b56867f24c5ef6f55a3cce70490e9777cf3af",
    "records/node_status/eng.d0-packet-allowlist-fix.json": "0a7f209b58f4d309f8c8ae36d8dce1696c22ffbd",
    "records/node_status/eng.hard-gate.json": "9889637191bc01e26a6ad6dfec19b35de8b72a0b",
    "records/node_status/math.lifetime-remainder.json": "a6f82a36e4f9a31abf656661f7c8ee5201874522",
    "records/node_status/math.side24-coefficient.json": "ecf9c42723ee72b8e7d1a4d5b36470f158d6af0e",
    "tests/test_node_status_live_v2.py": "dc38155aaff65537a8d9d5c739ec83217469ecdb",
    "tests/fixtures/node_status_live_v1/payload/tools/node_status_live_check.py.txt": "1e4852188f15060176b4458f58b27815f56845fd",
    "tests/fixtures/node_status_live_v1/payload/tools/node_status_refresh.py.txt": "5fa7c8b6647938a0aca538a7735edede65de9ed5",
    "tests/fixtures/node_status_live_v1/payload/tools/validate_node_status.py.txt": "e9d44b7620536b389a35ff6fad8aef2b0464b5d2",
    "tests/fixtures/node_status_live_v1/payload/architecture/node-status.schema.json.txt": "2bad95d30bcb89ea686d1ac5a50d52113dd730be",
    "tests/fixtures/node_status_live_v1/payload/docs/site/current.json.txt": "a64fb6d9768ada12b9884adac854e0ccdd5a910e",
    "tests/fixtures/node_status_live_v1/payload/records/node_status/eng.d0-packet-allowlist-fix.json.txt": "0a7f209b58f4d309f8c8ae36d8dce1696c22ffbd",
    "tests/fixtures/node_status_live_v1/payload/records/node_status/eng.hard-gate.json.txt": "9889637191bc01e26a6ad6dfec19b35de8b72a0b",
    "tests/fixtures/node_status_live_v1/payload/records/node_status/math.lifetime-remainder.json.txt": "a6f82a36e4f9a31abf656661f7c8ee5201874522",
    "tests/fixtures/node_status_live_v1/payload/records/node_status/math.side24-coefficient.json.txt": "ecf9c42723ee72b8e7d1a4d5b36470f158d6af0e",
    "tests/fixtures/node_status_live_v1/payload/tests/test_node_status_live_check.py.txt": "c5d578ddb9b7699f24966784ebfcb518d2d7284a",
    "tests/fixtures/node_status_live_v1/payload/tests/test_node_status_live_cli.py.txt": "14d0c84b1236fdc0d01083c62b791117b663c682",
    "tests/node_status_legacy_v1_runner.py": "77e036b5e8e816cec67720318d61fb95d445bc47",
    "tests/fixtures/node_status_live_v1/bundle.json": "0723223a945bea1593b6e42cb82c957f64b5488b",
    "tests/test_node_status_live_check.py": "3591d1088f22148fea71e25f829ba27ae4524d08",
    "tests/test_node_status_live_cli.py": "4d0df19642d66ccd22eb008325c15d6b370ac7e8"
}
METHODS = [
    "test_legacy_bundle_and_case_targets_are_exact",
    "test_closed_manifest_fields_and_profile_types_are_enforced",
    "test_manifest_cannot_select_code_schema_graph_or_unlisted_paths",
    "test_legacy_profile_requires_exact_four_v0_records",
    "test_graph_profile_requires_complete_ordered_nine_record_closure",
    "test_missing_extra_duplicate_and_same_count_inputs_refuse",
    "test_native_git_bytes_modes_and_final_capture_are_required",
    "test_project_imports_execute_only_captured_originals",
    "test_pinned_math_graph_joins_local_identity_raw_and_embedded_data",
    "test_graph_substitutions_cannot_borrow_unrelated_native_evidence",
    "test_refresh_validation_precedes_observation_or_writes",
    "test_v2_reports_and_current_cli_require_real_coverage_context_and_output"
]


def write(path, value):
    with path.open("x", encoding="ascii", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def git(*args):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1")
    return subprocess.check_output(
        ["git", "--no-replace-objects", "--literal-pathspecs", "-c",
         "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null",
         "-C", str(ROOT), *args], env=environment, timeout=30)


def source():
    head = git("rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = git("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    if head != os.environ["GITHUB_SHA"]:
        raise ValueError("checked HEAD differs from native context")
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise ValueError("wrong Git top-level")
    rows = []
    for entry in git("ls-tree", "-r", "--full-tree", "-z", head).split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, kind, identity = metadata.decode("ascii").split()
        relative = name.decode("utf-8")
        path = ROOT
        for part in relative.split("/"):
            path = path / part
            if path.is_symlink():
                raise ValueError("symlink tracked source")
        before = path.stat()
        if (kind != "blob" or mode not in ("100644", "100755")
                or not stat.S_ISREG(before.st_mode)
                or bool(before.st_mode & stat.S_IXUSR) != (mode == "100755")):
            raise ValueError("tracked kind/mode refused")
        raw = path.read_bytes()
        after = path.stat()
        fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
        if tuple(getattr(before, f) for f in fields) != tuple(getattr(after, f) for f in fields):
            raise ValueError("tracked source changed during capture")
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != identity or (relative in PROTECTED and PROTECTED[relative] != identity):
            raise ValueError("tracked/protected source differs: " + relative)
        rows.append(dict(path=relative, git_mode=mode, git_blob=identity,
                         bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    if set(PROTECTED) - {r["path"] for r in rows}:
        raise ValueError("missing protected input")
    if git("ls-files", "--others", "--exclude-standard", "-z") or git(
            "ls-files", "--others", "--ignored", "--exclude-standard", "-z"):
        raise ValueError("untracked/ignored checkout input")
    if git("rev-parse", "--verify", "HEAD").decode().strip() != head:
        raise ValueError("HEAD changed during capture")
    return dict(repository=os.environ["GITHUB_REPOSITORY"], checked_commit=head,
                checked_tree=tree, run_id=RUN, run_attempt=ATTEMPT, files=rows)


def legacy_ids():
    # Derive expected names from HASH-VERIFIED unchanged original test bodies,
    # independently of the new runner/wrapper target tables.
    populations = []
    for relative, module, owner, count in (
        ("tests/test_node_status_live_check.py", "tests.test_node_status_live_check",
         "NodeStatusLiveContract", 23),
        ("tests/test_node_status_live_cli.py", "tests.test_node_status_live_cli",
         "NodeStatusLiveCLIContract", 20),
    ):
        path = "tests/fixtures/node_status_live_v1/payload/" + relative + ".txt"
        raw = (ROOT / path).read_bytes()
        identity = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if identity != PROTECTED[path]:
            raise ValueError("original legacy test body differs")
        tree = ast.parse(raw.decode("utf-8"), filename=path)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner]
        if len(classes) != 1:
            raise ValueError("original legacy class differs")
        names = [n.name for n in classes[0].body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        if len(names) != count or len(set(names)) != count:
            raise ValueError("original legacy population differs")
        populations.append(tuple(module + "." + owner + "." + name for name in names))
    return tuple(populations)


def observe(group):
    if group not in ("helper23", "cli20", "current12"):
        raise ValueError("unlisted fixed group")
    destination = OUT / group
    destination.mkdir(mode=0o700, exist_ok=False)
    record = dict(schema="node-status-v2.offline-observation.v1", group=group, mode=MODE,
        scope="TEST_CONTRACTS_ONLY", scientific_effect="NONE", http_credit=False,
        native_acquisition=False, data_adoption=False, g5_claim=False, cbz_claim=False,
        status="STARTED", complete=False, timeout_policy="EXISTING_15MIN_OUTER_JOB_LIMIT_ONLY",
        stdout_complete=False, stderr_complete=False)
    write(destination / "started.json", record)
    process = None
    try:
        before = source()
        write(destination / "source-before.json", before)
        helper, cli = legacy_ids()
        expected = sorted(helper if group == "helper23" else cli if group == "cli20" else [
            "test_node_status_live_v2.NodeStatusV2Contract." + name for name in METHODS])
        command = [sys.executable, "-B"] + (["-O"] if MODE == "optimized" else []) + ["-S", "-m", "unittest"]
        command += (["discover", "-s", "tests", "-p", "test_node_status_live_v2.py", "-v"]
            if group == "current12" else [
                "tests.test_node_status_live_check" if group == "helper23" else
                "tests.test_node_status_live_cli", "-v"])
        environment = dict(os.environ)
        for key in ("GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"):
            environment.pop(key, None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        record.update(command=command, cwd=str(ROOT), expected_parent_ids=expected)
        with (destination / "stdout.bin").open("xb") as stdout, (
                destination / "stderr.bin").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdout=stdout, stderr=stderr, start_new_session=True)
            record["native_exit_code"] = process.wait()
        record.update(timed_out=False, stdout_complete=True, stderr_complete=True)
        after = source()
        write(destination / "source-after.json", after)
        record["source_unchanged"] = before == after
        text = (destination / "stderr.bin").read_bytes().decode("utf-8", errors="strict")
        observed, bad, malformed = [], [], []
        for line in text.splitlines():
            if " ... " not in line and not line.startswith("test"):
                continue
            match = re.fullmatch(r"(\S+) \(([^()]*)\) \.\.\. (.+)", line)
            if not match:
                malformed.append(line)
                continue
            short, identity, disposition = match.groups()
            observed.append(identity)
            if short != identity.rsplit(".", 1)[-1] or disposition != "ok":
                bad.append(line)
        summaries = re.findall(r"^Ran ([1-9][0-9]*) tests? in [0-9]+(?:\.[0-9]+)?s$", text, re.M)
        nonempty = [line for line in text.splitlines() if line.strip()]
        record.update(observed_parent_ids=observed, bad_result_lines=bad,
            malformed_result_lines=malformed,
            reported_test_count=int(summaries[0]) if len(summaries) == 1 else None,
            terminal_ok=bool(nonempty and nonempty[-1] == "OK"))
        record["exact_parent_results"] = (
            observed == expected and not bad and not malformed and len(summaries) == 1
            and int(summaries[0]) == len(expected))
        accepted = (record["native_exit_code"] == 0 and before == after
            and record["exact_parent_results"] and record["terminal_ok"]
            and (destination / "stdout.bin").stat().st_size == 0)
        record.update(status="PASS" if accepted else "FAIL", complete=True)
    except BaseException as error:
        record.update(status="HOLD", controller_error=type(error).__name__, complete=False)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        for label in ("stdout", "stderr"):
            path = destination / (label + ".bin")
            if path.is_file():
                raw = path.read_bytes()
                record[label] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        write(destination / "observation.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
    return 0 if record["status"] == "PASS" else 1


def readback():
    # Whole originals only. Missing/omitted bytes remain explicit HOLD.
    limit = 2 * 1024 * 1024
    originals, missing, errors = [], [], []
    def add(path, name):
        try:
            if not path.is_relative_to(TEMP):
                raise ValueError("readback escaped fixed root")
            current = path
            while current != TEMP:
                if current.is_symlink():
                    raise ValueError("symlink readback")
                current = current.parent
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
                raise ValueError("nonregular/oversize original")
            raw = path.read_bytes()
            if len(raw) != info.st_size:
                raise ValueError("original extent changed")
            originals.append((name, raw))
            return raw
        except FileNotFoundError:
            missing.append(name)
        except Exception as error:
            errors.append(dict(path=name, error=type(error).__name__))
        return None
    for group in ("helper23", "cli20", "current12"):
        for name in ("stdout.bin", "stderr.bin", "observation.json"):
            add(OUT / group / name, "outer/" + group + "/" + name)
    shared_refs = {}
    native = None
    expected_bootstrap = None
    try:
        native = source()
        write(OUT / "readback-source.json", native)
        runner_raw = (ROOT / "tests/node_status_legacy_v1_runner.py").read_bytes()
        runner_tree = ast.parse(runner_raw.decode("utf-8"))
        constants = [n.value for n in runner_tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_BOOTSTRAP" for t in n.targets)]
        if len(constants) != 1:
            raise ValueError("fixed bootstrap constant population differs")
        bootstrap = ast.literal_eval(constants[0])
        if type(bootstrap) is not str:
            raise ValueError("fixed bootstrap is not original source text")
        bootstrap = bootstrap.encode("ascii")
        expected_bootstrap = dict(path="_shared/bootstrap.py", bytes=len(bootstrap),
            sha256=hashlib.sha256(bootstrap).hexdigest())
    except Exception as error:
        errors.append(dict(path="readback-source-or-bootstrap", error=type(error).__name__))
    def shared_reference(ref, bootstrap=False):
        keys = {"path", "bytes", "sha256"} | (set() if bootstrap else {"observed_at"})
        if (type(ref) is not dict or set(ref) != keys
                or type(ref["path"]) is not str or type(ref["bytes"]) is not int
                or not 0 < ref["bytes"] <= 65536 or type(ref["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]) is None):
            raise ValueError("invalid closed shared reference")
        if bootstrap:
            if expected_bootstrap is None or ref != expected_bootstrap:
                raise ValueError("shared bootstrap differs from pinned runner original")
        else:
            match = re.fullmatch(r"_shared/source-([0-9a-f]{64})\.json", ref["path"])
            if (match is None or match[1] != ref["sha256"]
                    or type(ref["observed_at"]) is not str
                    or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z",
                                    ref["observed_at"]) is None):
                raise ValueError("noncanonical shared source identity/time")
        identity = {k: ref[k] for k in ("path", "bytes", "sha256")}
        previous = shared_refs.setdefault(ref["path"], identity)
        if previous != identity:
            raise ValueError("conflicting shared body reference")
    try:
        helper, cli = legacy_ids()
        for target in helper + cli:
            directory = LEGACY / (target + "-opt" + ("1" if MODE == "optimized" else "0"))
            observed = add(directory / "observation.json", "legacy/" + directory.name + "/observation.json")
            for name in ("supervisor.json", "stdout.bin", "stderr.bin"):
                add(directory / name, "legacy/" + directory.name + "/" + name)
            if observed is not None:
                try:
                    case = json.loads(observed)
                    shared_reference(case["source_before"])
                    shared_reference(case["source_after"])
                    shared_reference(case["bootstrap"], bootstrap=True)
                except Exception as error:
                    errors.append(dict(path="legacy/" + directory.name + "/shared-refs",
                                       error=type(error).__name__))
            raw = add(directory / "inner-outcome.json", "legacy/" + directory.name + "/inner-outcome.json")
            if raw is None:
                continue
            inner = json.loads(raw)
            captures = inner.get("program_captures")
            if (type(captures) is not list or len(captures) > 128
                    or len(set(captures)) != len(captures)):
                raise ValueError("invalid fixed program capture inventory")
            for number, name in enumerate(captures):
                if name != "program-" + str(number).zfill(3):
                    raise ValueError("unlisted program capture")
                for leaf in ("observation.json", "stdout.bin", "stderr.bin"):
                    add(directory / name / leaf, "legacy/" + directory.name + "/" + name + "/" + leaf)
    except Exception as error:
        errors.append(dict(path="legacy-inventory", error=type(error).__name__))
    for name, ref in sorted(shared_refs.items()):
        raw = add(LEGACY / name, "legacy/" + name)
        if raw is None:
            continue
        try:
            if len(raw) != ref["bytes"] or hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ValueError("shared original extent/digest differs")
            if name.startswith("_shared/source-"):
                body = json.loads(raw)
                if (native is None or body["checked_commit"] != native["checked_commit"]
                        or body["checked_tree"] != native["checked_tree"]):
                    raise ValueError("shared source belongs to another native cut")
        except Exception as error:
            errors.append(dict(path="legacy/" + name, error=type(error).__name__))
    invocation_counts = {}
    try:
        if TEST_INVOCATIONS.exists():
            if not TEST_INVOCATIONS.is_dir() or TEST_INVOCATIONS.is_symlink():
                raise ValueError("invalid TEST invocation root")
            for case in sorted(TEST_INVOCATIONS.iterdir()):
                if case.name not in METHODS or not case.is_dir() or case.is_symlink():
                    raise ValueError("unlisted TEST invocation case")
                invocation_counts[case.name] = 0
                for directory in sorted(case.iterdir()):
                    if (not directory.is_dir() or directory.is_symlink()
                            or re.fullmatch(r"(cli|refresh)-[1-9][0-9]*", directory.name) is None):
                        raise ValueError("unlisted TEST invocation path")
                    prefix = "test-invocations/" + case.name + "/" + directory.name + "/"
                    raw = add(directory / "observation.json", prefix + "observation.json")
                    stdout = add(directory / "stdout.bin", prefix + "stdout.bin")
                    stderr = add(directory / "stderr.bin", prefix + "stderr.bin")
                    if raw is None or stdout is None or stderr is None:
                        continue
                    record = json.loads(raw)
                    if (record.get("scope") != "OWNED_TEST_CLI_INVOCATION_ONLY"
                            or record.get("http_credit") is not False
                            or record.get("actual_actions_authority") is not False
                            or record.get("current_v2_qualified") is not False):
                        raise ValueError("TEST invocation scope differs")
                    for name, body in (("stdout", stdout), ("stderr", stderr)):
                        if record[name] != dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest()):
                            raise ValueError("TEST invocation original digest differs")
                    invocation_counts[case.name] += 1
        current_path = OUT / "current12/observation.json"
        if current_path.is_file() and json.loads(current_path.read_bytes()).get("status") == "PASS":
            expected_invocations = {
                "test_project_imports_execute_only_captured_originals": 6,
                "test_refresh_validation_precedes_observation_or_writes": 8,
                "test_v2_reports_and_current_cli_require_real_coverage_context_and_output": 9,
            }
            if invocation_counts != expected_invocations:
                raise ValueError("successful current12 lacks exact23 TEST CLI originals")
    except Exception as error:
        errors.append(dict(path="TEST-invocation-inventory", error=type(error).__name__))
    nonce = secrets.token_hex(24)
    terminal = ("::" + nonce + "::").encode("ascii")
    while any(terminal in raw for _, raw in originals):
        nonce = secrets.token_hex(24)
        terminal = ("::" + nonce + "::").encode("ascii")
    frames, omitted, size = [], [], 0
    for name, raw in originals:
        try:
            raw.decode("utf-8", errors="strict")
            encoding, body = "utf-8", raw
        except UnicodeError:
            encoding, body = "base64", base64.b64encode(raw)
        metadata = dict(path=name, original_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            encoding=encoding, original_complete=True, scientific_effect="NONE", http_credit=False)
        frame = (b"NODE_STATUS_V2_ORIGINAL " + (json.dumps(metadata, sort_keys=True) + "\n").encode()
            + body + b"\n" + b"NODE_STATUS_V2_ORIGINAL_END "
            + (json.dumps(dict(path=name), sort_keys=True) + "\n").encode())
        if size + len(frame) + 131072 > limit:
            omitted.append(dict(path=name, original_bytes=len(raw), sha256=metadata["sha256"],
                                printed_original_bytes=0, original_complete=False))
        else:
            frames.append(frame)
            size += len(frame)
    summary = dict(schema="node-status-v2.offline-readback.v1", mode=MODE,
        complete=not (missing or errors or omitted), selected_whole_originals=len(frames),
        missing=missing, errors=errors, omitted=omitted, output_limit=limit,
        scope="LEGACY_FIXTURE_AND_V2_TEST_ORIGINALS_ONLY", scientific_effect="NONE",
        http_credit=False, native_acquisition=False, current_v2_qualified=False,
        g5_claim=False, cbz_claim=False, test_invocation_counts=invocation_counts,
        unique_shared_originals=len(shared_refs))
    summary_raw = (json.dumps(summary, sort_keys=True) + "\n").encode()
    if len(summary_raw) > 131072:
        raise ValueError("complete readback summary ceiling exceeded")
    stream = (("::stop-commands::" + nonce + "\n").encode()
        + b"NODE_STATUS_V2_READBACK " + summary_raw + b"".join(frames) + terminal + b"\n")
    if len(stream) > limit:
        raise ValueError("full framed readback ceiling exceeded")
    (OUT / "readback-stream.bin").write_bytes(stream)
    write(OUT / "readback-observation.json", summary)
    if sys.stdout.buffer.write(stream) != len(stream):
        raise OSError("short original readback")
    sys.stdout.buffer.flush()
    return 0 if summary["complete"] else 1


if len(sys.argv) != 2:
    raise ValueError("fixed supervisor requires one group")
raise SystemExit(readback() if sys.argv[1] == "readback" else observe(sys.argv[1]))
