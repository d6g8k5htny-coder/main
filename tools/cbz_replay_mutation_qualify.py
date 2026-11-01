#!/usr/bin/env python3
"""Qualify recorded outputs of the frozen three-mutant replay runner.

This bounded parser checks source pins, report structure and eight complete
CPython 3.11.16 unittest transcripts. It executes neither source nor tests.
Unsigned consistent input is not authenticated execution evidence. Preserve
the original report/logs and external artifact custody; this is no proof gate.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat

PINS = {
    "replay_tool.py": "b97bee53a590651b83d68056c090ba70e8a055157e65d2bd3fcc5cd0219840b3",
    "test_replay_tool.py": "a07eaab8b6fd40a911d313197fae688927ebb3e1d74644628e4888a7e6505fc0",
    "run_mutations.py": "4b3d72c125ac038f8b5662b3f5e3edb58d80a7d993addc3eb418bbbfe45050d6",
}
MODES = ("normal", "optimized")
CAUSES = {
    "owned-names": (60, "owned names differ",
        'AssertionError: "owned names differ" does not match "module assignment differs from complete compiled census"'),
    "module-assignment": (65, "module assignment differs", "AssertionError: ValueError not raised"),
    "axioms": (69, "forbidden axiom", "AssertionError: ValueError not raised"),
}
THRESHOLD = "100% of the three declared validator-source mutants in each Python mode"
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "cbz/evidence/kernel/additional/replay-tool"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular_bytes(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError("unreadable or symlink input: " + str(path)) from error
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "input must be a regular file")
        return stream.read()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def forbidden_number(value):
    raise ValueError("non-integer/nonfinite JSON number: " + value)


def keys(value, expected, context):
    require(type(value) is dict and set(value) == set(expected), "unknown/missing fields: " + context)


def check_transcript(raw, tests, label, method):
    text = raw.decode("utf-8")
    failed_id = "test_replay_tool.TranscriptTests." + method if method else None
    progress = "".join("%s (%s) ... %s\n" % (item.rsplit(".", 1)[1], item,
                        "FAIL" if item == failed_id else "ok") for item in tests)
    pattern = re.escape(progress)
    if method:
        line, reason, assertion = CAUSES[label]
        detail = "\n" + "=" * 70 + "\nFAIL: %s (%s)\n" % (method, failed_id)
        detail += "-" * 70 + "\n"
        if label == "owned-names":
            detail += ("ValueError: module assignment differs from complete compiled census\n\n"
                       "During handling of the above exception, another exception occurred:\n\n")
        detail += ('Traceback (most recent call last):\n'
                   '  File "SOURCE_PATH", line %d, in %s\n' % (line, method))
        detail += "    with self.assertRaisesRegex(ValueError, '%s'):\n%s\n" % (reason, assertion)
        pattern += re.escape(detail).replace("SOURCE_PATH", r'[^"\r\n]+/test_replay_tool\.py')
    pattern += re.escape("\n" + "-" * 70 + "\n")
    pattern += r"Ran 31 tests in [0-9]+(?:\.[0-9]+)?s\n\n"
    pattern += re.escape("FAILED (failures=1)" if method else "OK") + r"\n\n?"
    require(re.fullmatch(pattern, text) is not None, "transcript identity/count/cause mismatch: " + label)


def qualify(run_dir, source_dir=DEFAULT_SOURCE):
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    require(not run_dir.is_symlink() and not source_dir.is_symlink(), "symlink input directory")
    sources = {name: regular_bytes(source_dir / name) for name in PINS}
    require(all(sha(sources[name]) == pin for name, pin in PINS.items()), "frozen source hash mismatch")
    source_ast = ast.parse(sources["test_replay_tool.py"])
    tests = sorted("test_replay_tool.%s.%s" % (cls.name, node.name)
                   for cls in source_ast.body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"))
    require(len(tests) == 31 and len(set(tests)) == 31, "frozen test census mismatch")
    runner_ast = ast.parse(sources["run_mutations.py"])
    mutations = [ast.literal_eval(node.value) for node in runner_ast.body
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MUTATIONS" for t in node.targets)]
    require(len(mutations) == 1, "frozen mutation declaration mismatch")
    mutations = mutations[0]
    require({row[0] for row in mutations} == set(CAUSES) and len(mutations) == 3, "mutation census mismatch")
    labels = ("baseline", *CAUSES)
    names = {"REPORT.json"} | {label + "-" + mode + "." + stream
                               for label in labels for mode in MODES for stream in ("stdout", "stderr")}
    require({p.name for p in run_dir.iterdir()} == names, "missing/unknown raw input file")
    raw = {name: regular_bytes(run_dir / name) for name in names}
    report = json.loads(raw["REPORT.json"].decode("utf-8"), object_pairs_hook=unique_object,
                        parse_constant=forbidden_number, parse_float=forbidden_number)
    keys(report, ("schema", "source_sha256", "test_sha256", "python", "python_executable",
                  "threshold", "baseline", "mutants", "status"), "report")
    require(report["schema"] == "cbz-replay-source-mutations-v1"
            and report["status"] == "PASS_ALL_DECLARED_MUTANTS_KILLED"
            and report["threshold"] == THRESHOLD, "report contract mismatch")
    require(report["source_sha256"] == PINS["replay_tool.py"]
            and report["test_sha256"] == PINS["test_replay_tool.py"], "report source binding mismatch")
    require(type(report["python"]) is str and report["python"].startswith("3.11.16 (")
            and type(report["python_executable"]) is str and report["python_executable"].startswith("/"),
            "unsupported reported runtime (requires CPython 3.11.16)")
    keys(report["baseline"], MODES, "baseline modes")
    keys(report["mutants"], CAUSES, "mutant identities")
    for label, old, new, method in mutations:
        source = sources["replay_tool.py"].decode("utf-8")
        require(source.count(old) == 1, "frozen mutation anchor mismatch")
        mutant = report["mutants"][label]
        keys(mutant, ("source_sha256", "modes"), label)
        require(mutant["source_sha256"] == sha(source.replace(old, new, 1).encode("utf-8")),
                "mutated source binding mismatch: " + label)
        keys(mutant["modes"], MODES, label + " modes")
    methods = {row[0]: row[3] for row in mutations}
    for label in labels:
        for mode in MODES:
            row = report["baseline"][mode] if label == "baseline" else report["mutants"][label]["modes"][mode]
            expected_keys = ("exit_code", "stderr_sha256") if label == "baseline" else (
                "exit_code", "stderr_sha256", "expected_failed_test", "reason_bound_kill")
            keys(row, expected_keys, label + "/" + mode)
            expected_exit = 0 if label == "baseline" else 1
            require(type(row["exit_code"]) is int and row["exit_code"] == expected_exit, "wrong recorded exit")
            method = methods.get(label)
            if method:
                require(row["expected_failed_test"] == method and row["reason_bound_kill"] is True,
                        "declared intended failure mismatch")
            stderr = raw[label + "-" + mode + ".stderr"]
            require(row["stderr_sha256"] == sha(stderr), "raw stderr digest mismatch")
            require(raw[label + "-" + mode + ".stdout"] == b"", "unexpected stdout")
            check_transcript(stderr, tests, label, method)
    return {
        "schema": "cbz-replay-recorded-qualification-v1",
        "status": "PASS_RECORDED_MUTATION_QUALIFICATION",
        "qualified_runs": 8,
        "source_sha256": dict(PINS),
        "input_sha256": {name: sha(raw[name]) for name in sorted(raw)},
        "reported_python": report["python"],
        "limit": "Recorded consistency only; unsigned inputs do not authenticate execution or establish proof/tool/global-gate admission.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file outside the input/source directories")
    args = parser.parse_args()
    destination = args.output.resolve()
    for directory in (args.input.resolve(), args.source_dir.resolve()):
        if destination == directory or directory in destination.parents:
            parser.error("output must be outside the input/source directories")
    # Reserve a new output before any admission; never truncate an earlier receipt.
    with args.output.open("x", encoding="utf-8") as out:
        try:
            result = qualify(args.input, args.source_dir)
            code = 0
        except (ValueError, OSError) as error:
            result = {"schema": "cbz-replay-recorded-qualification-v1", "status": "FAIL",
                      "failure": str(error)}
            code = 1
        out.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(result["status"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
