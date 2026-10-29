"""Collect source-bound mutant/control transcripts; never certify mutation kills."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

SCOPES = {
    "lexer": {
        "registry": "tests/test_cbz_layer_a_scan_mutants.py",
        "tool": "tools/cbz/layer_a_scan.py",
        "env": "CBZ_LAYER_A_SCAN",
        "tests": ("tests.test_cbz_layer_a_scan", "tests.test_cbz_auditor_hardening"),
    },
    "a3": {
        "registry": "tests/test_cbz_a3_float_census.py",
        "tool": "tools/cbz/a3_float_census.py",
        "env": "CBZ_A3_CENSUS",
        "tests": ("tests.test_cbz_a3_float_census", "tests.test_cbz_auditor_hardening"),
    },
}


def parse_registry(source):
    """Admit only the explicit literal lists/dicts and their literal extensions."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("invalid registry source") from error
    values = {}

    def literal(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys = []
                for key in child.keys:
                    if key is None:
                        raise ValueError("dictionary expansion is unsupported")
                    keys.append(ast.literal_eval(key))
                try:
                    if len(set(keys)) != len(keys):
                        raise ValueError("duplicate literal dictionary key")
                except TypeError as error:
                    raise ValueError("unhashable dictionary key") from error
        return ast.literal_eval(node)

    try:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            used = {item.id for item in ast.walk(node)
                    if isinstance(item, ast.Name) and item.id in ("MUTANTS", "EXPECTED_FAILURE")}
            if not used:
                continue
            if isinstance(node, ast.Assign):
                named = [t.id for t in node.targets
                         if isinstance(t, ast.Name) and t.id in ("MUTANTS", "EXPECTED_FAILURE")]
                if len(node.targets) != 1 or len(named) != 1 or named[0] in values:
                    raise ValueError("ambiguous registry assignment")
                values[named[0]] = literal(node.value)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                        and func.value.id in ("MUTANTS", "EXPECTED_FAILURE")):
                    raise ValueError("unsupported registry expression")
                name = func.value.id
                method = "extend" if name == "MUTANTS" else "update"
                if (name not in values or func.attr != method or len(node.value.args) != 1
                        or node.value.keywords):
                    raise ValueError("unsupported registry update")
                addition = literal(node.value.args[0])
                if name == "MUTANTS":
                    if not isinstance(values[name], list) or not isinstance(addition, list):
                        raise ValueError("invalid mutant extension")
                    values[name].extend(addition)
                else:
                    if not isinstance(values[name], dict) or not isinstance(addition, dict):
                        raise ValueError("invalid target update")
                    if set(values[name]).intersection(addition):
                        raise ValueError("duplicate target")
                    values[name].update(addition)
            else:
                raise ValueError("unsupported mutation of registry")
    except (TypeError, SyntaxError, ValueError) as error:
        raise ValueError("invalid or nonliteral mutation registry") from error
    mutants, targets = values.get("MUTANTS"), values.get("EXPECTED_FAILURE")
    if not isinstance(mutants, list) or not mutants or not isinstance(targets, dict):
        raise ValueError("missing or empty mutation registry")
    names = []
    for item in mutants:
        if (not isinstance(item, tuple) or len(item) != 3
                or not all(isinstance(value, str) for value in item)
                or not item[0] or not item[1] or item[1] == item[2]):
            raise ValueError("invalid mutation")
        names.append(item[0])
    if (len(set(names)) != len(names) or set(names) != set(targets)
            or any(not isinstance(value, str) or re.fullmatch(r"test_[A-Za-z0-9_]+", value) is None
                   for value in targets.values())):
        raise ValueError("duplicate or unmapped mutation")
    return mutants, targets


def resolve_target(sources, wanted):
    """Find exactly one top-level unittest method with the given name."""
    if not isinstance(wanted, str) or re.fullmatch(r"test_[A-Za-z0-9_]+", wanted) is None:
        raise ValueError("invalid target name")
    matches = []
    try:
        for module, source in sources.items():
            tree = ast.parse(source)
            for cls in tree.body:
                if isinstance(cls, ast.ClassDef):
                    for node in cls.body:
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wanted:
                            matches.append(module + "." + cls.name + "." + node.name)
    except (SyntaxError, AttributeError) as error:
        raise ValueError("invalid test source") from error
    if len(matches) != 1:
        raise ValueError("expected target is absent or ambiguous")
    return matches[0]


def make_mutant(source, old, new):
    if not old or old == new or source.count(old) != 1:
        raise ValueError("mutation anchor missing, ambiguous or unchanged")
    return source.replace(old, new, 1)


def _stream(raw):
    raw = raw or b""
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    try:
        return raw.decode("utf-8"), encoded, None
    except UnicodeDecodeError:
        return None, encoded, "non-UTF-8 stream"


def observe(command, cwd, env, timeout=30):
    """Record the actual process and both byte streams; never infer a kill."""
    try:
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=timeout)
        kind, code, out, err, error = "COMPLETED", result.returncode, result.stdout, result.stderr, None
    except subprocess.TimeoutExpired as exc:
        kind, code, out, err, error = "TIMEOUT", None, exc.stdout, exc.stderr, str(exc)
    except OSError as exc:
        kind, code, out, err, error = "LAUNCH_ERROR", None, b"", b"", str(exc)
    stdout, stdout_b64, stdout_error = _stream(out)
    stderr, stderr_b64, stderr_error = _stream(err)
    return dict(command=list(command), cwd=str(cwd), kind=kind, returncode=code, stdout=stdout, stderr=stderr,
                stdout_b64=stdout_b64, stderr_b64=stderr_b64, error=error,
                stream_errors=[item for item in (stdout_error, stderr_error) if item])


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(root, scope, temporary):
    cfg = SCOPES[scope]
    registry_path, tool_path = root / cfg["registry"], root / cfg["tool"]
    source_inputs = list(dict.fromkeys(
        [registry_path, tool_path, root / "tools/cbz/layer_a_scan.py",
         root / "tools/cbz/a3_float_census.py"]
        + [root / (name.replace(".", "/") + ".py") for name in cfg["tests"]]))
    original = {str(path.relative_to(root)): _sha(path.read_bytes()) for path in source_inputs}
    mutants, targets = parse_registry(registry_path.read_text(encoding="utf-8"))
    source = tool_path.read_bytes().decode("utf-8")
    modules = {name: (root / (name.replace(".", "/") + ".py")).read_text(encoding="utf-8")
               for name in cfg["tests"]}
    full_targets = {name: resolve_target(modules, name) for name in set(targets.values())}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "GIT_")) and key not in ("CBZ_LAYER_A_SCAN", "CBZ_A3_CENSUS")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(temporary / "empty-bytecode-cache")
    interpreter = [sys.executable, *(["-O"] if sys.flags.optimize else []), "-B", "-S",
                   "-m", "unittest", "-v"]
    baselines = {}
    for target in sorted(set(full_targets.values())):
        baseline_env = dict(env, **{cfg["env"]: str(tool_path)})
        baselines[target] = observe(interpreter + [target], root, baseline_env)
    rows = []
    for index, (name, old, new) in enumerate(mutants):
        target = full_targets[targets[name]]
        row = dict(name=name, target=target, original_source_sha256=_sha(source.encode("utf-8")))
        try:
            changed = make_mutant(source, old, new).encode("utf-8")
            mutant = temporary / (scope + "-" + str(index) + ".py")
            mutant.write_bytes(changed)
            row["mutant_source_sha256"] = _sha(changed)
            row["replacement"] = dict(old=old, new=new)
            mutant_env = dict(env, **{cfg["env"]: str(mutant)})
            row["observation"] = observe(interpreter + [target], root, mutant_env)
        except (OSError, ValueError) as error:
            row["construction_error"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
    after = {}
    for path in source_inputs:
        try:
            after[str(path.relative_to(root))] = _sha(path.read_bytes())
        except OSError:
            after[str(path.relative_to(root))] = None
    return dict(scope=scope, registry_count=len(mutants), source_before=original,
                source_after=after, targets=full_targets, baselines=baselines, mutants=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scope", choices=("lexer", "a3", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    if output == root or output.is_relative_to(root):
        parser.error("output must be outside the source root")
    try:
        stream = output.open("x", encoding="utf-8")
    except OSError as error:
        parser.error("new output required: " + str(error))
    report = dict(schema="cbz.auditor-mutant-diagnostic.v1",
                  scope="NON_CERTIFYING_RAW_PROCESS_OBSERVATIONS",
                  source_root=str(root), python=sys.version, collections=[], errors=[])
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    git_env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    git_base = ["git", "--no-replace-objects", "--no-optional-locks", "-c", "core.fsmonitor=false",
                "-c", "core.untrackedCache=false", "-c", "core.hooksPath=/dev/null", "-C", str(root)]
    head_command = git_base + ["rev-parse", "--verify", "HEAD^{commit}"]
    status_command = git_base + ["status", "--porcelain=v1", "--untracked-files=all"]
    report["git_head_before"] = observe(head_command, root, git_env)
    report["git_status_before"] = observe(status_command, root, git_env)
    try:
        with tempfile.TemporaryDirectory(prefix="cbz-auditor-diagnostic-") as directory:
            temporary = Path(directory)
            for scope in (("lexer", "a3") if args.scope == "both" else (args.scope,)):
                try:
                    report["collections"].append(collect(root, scope, temporary))
                except (OSError, ValueError, UnicodeError) as error:
                    report["errors"].append(dict(scope=scope, error=type(error).__name__ + ": " + str(error)))
    except Exception as error:
        report["errors"].append(dict(scope="collector", error=type(error).__name__ + ": " + str(error)))
    finally:
        report["git_head_after"] = observe(head_command, root, git_env)
        report["git_status_after"] = observe(status_command, root, git_env)
    incomplete = bool(report["errors"])
    for prefix in ("git_head", "git_status"):
        before, after = report[prefix + "_before"], report[prefix + "_after"]
        if (before["kind"] != "COMPLETED" or after["kind"] != "COMPLETED"
                or before["returncode"] != 0 or after["returncode"] != 0
                or before["stdout"] != after["stdout"] or before["stream_errors"] or after["stream_errors"]):
            incomplete = True
        if prefix == "git_status" and (before["stdout"] or after["stdout"]):
            incomplete = True
    for collection in report["collections"]:
        if collection["source_before"] != collection["source_after"]:
            incomplete = True
        if any(row["kind"] != "COMPLETED" or row["returncode"] != 0 or row["stream_errors"]
               for row in collection["baselines"].values()):
            incomplete = True
        if any(not re.search(r"^Ran 1 test in ", row["stderr"] or "", re.MULTILINE)
               or not re.search(r"^OK$", row["stderr"] or "", re.MULTILINE)
               for row in collection["baselines"].values()):
            incomplete = True
        if any("construction_error" in row or row.get("observation", {}).get("kind") != "COMPLETED"
               or row.get("observation", {}).get("stream_errors") for row in collection["mutants"]):
            incomplete = True
    report["collection_incomplete"] = incomplete
    json.dump(report, stream, indent=2, ensure_ascii=True)
    stream.write("\n")
    stream.close()
    print(json.dumps(dict(schema=report["schema"], scope=report["scope"],
                          observed_mutants=sum(len(row["mutants"]) for row in report["collections"]),
                          collection_incomplete=incomplete)))
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
