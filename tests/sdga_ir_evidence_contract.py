#!/usr/bin/env python3
"""Separate Task D output-admission contracts: 2 Python methods / 6 real I/O routes.

No Lean/semantic-mutant credit. The absent helper fails only these methods.
Subject execution uses the exact captured bytes and canonical filename; no
package loader, sys.modules registration, pyc or alternate implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "tools" / "sdga_ir_evidence.py"
ROUTES = (
    "new-external-accept", "existing-dir-reject", "existing-file-reject",
    "dangling-root-symlink-reject", "source-descendant-reject",
    "parent-alias-into-source-reject",
)
STATE = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def capture(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    if stat.S_ISLNK(info.st_mode):
        return {"exists": True, "kind": "symlink", "target": os.readlink(path)}
    if stat.S_ISDIR(info.st_mode):
        return {"exists": True, "kind": "directory"}
    if stat.S_ISREG(info.st_mode):
        raw = path.read_bytes()
        return {"exists": True, "kind": "file", "bytes": len(raw), "sha256": sha(raw)}
    return {"exists": True, "kind": "other"}


class Evidence:
    def __init__(self, output):
        # This is test-evidence setup, not the production helper under test.
        try:
            output.lstat()
        except FileNotFoundError:
            pass
        else:
            raise ValueError("TEST_OUTPUT_NOT_FRESH: " + str(output))
        output.mkdir(parents=True)
        self.output = output.resolve(strict=True)
        self.records = {name: {"route": name, "status": "not_run",
                              "reason": "method body not reached"} for name in ROUTES}
        self.binding_error = None
        self.source = self.bind_source(SUBJECT)
        self.test_source = self.bind_source(Path(__file__).resolve())
        self.flush()

    def bind_source(self, path):
        try:
            raw = path.read_bytes()
            canonical = path.resolve(strict=True)
        except FileNotFoundError:
            return {"path": str(path.absolute()), "exists": False}
        except OSError as error:
            self.binding_error = type(error).__name__ + ": " + str(error)
            return {"path": str(path.absolute()), "exists": None,
                    "measurement_error": self.binding_error}
        return {"path": str(canonical), "exists": True, "bytes": len(raw),
                "sha256": sha(raw), "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()}

    def subject(self, test):
        if self.binding_error is not None:
            test.fail("SOURCE_MEASUREMENT: " + self.binding_error)
        if not self.source["exists"]:
            test.fail("IMPLEMENTATION_ABSENCE: tools/sdga_ir_evidence.py; no I/O refusal credit")
        canonical = SUBJECT.resolve(strict=True)
        test.assertEqual(canonical, SUBJECT.absolute(), "subject path alias is not admitted")
        raw = canonical.read_bytes()
        test.assertEqual((len(raw), sha(raw)),
                         (self.source["bytes"], self.source["sha256"]),
                         "subject changed after capture")
        namespace = {"__name__": "sdga_ir_evidence_captured_subject",
                     "__file__": str(canonical), "__package__": None}
        exec(compile(raw, str(canonical), "exec"), namespace)
        helper = namespace.get("admit_output_root")
        test.assertTrue(callable(helper),
                        "IMPLEMENTATION_ABSENCE: exact admit_output_root API")
        test.assertEqual(canonical.read_bytes(), raw, "subject changed during captured execution")
        return helper

    def flush(self):
        write_json(self.output / "source-identities.json", {
            "record": "sdga-ir-admission-source-identities", "subject": self.source,
            "test": self.test_source, "python": sys.version,
            "python_optimized": bool(sys.flags.optimize),
            "outside_67": True, "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False})
        with (self.output / "routes.jsonl").open("w", encoding="utf-8") as stream:
            for row in self.records.values():
                stream.write(json.dumps(row, sort_keys=True) + "\n")


class AdmissionContracts(unittest.TestCase):
    def route(self, helper, name, expected):
        sandbox = STATE.output / name
        sandbox.mkdir()
        source = sandbox / "source"
        source.mkdir()
        source_file = source / "source.txt"
        source_file.write_bytes(b"actual output-admission source fixture\n")
        candidate = sandbox / "new-output"
        alias = None
        if name == "existing-dir-reject":
            candidate.mkdir()
        elif name == "existing-file-reject":
            candidate.write_bytes(b"existing file must survive\n")
        elif name == "dangling-root-symlink-reject":
            candidate.symlink_to(sandbox / "nonexistent-target", target_is_directory=True)
        elif name == "source-descendant-reject":
            candidate = source / "absent-child"
        elif name == "parent-alias-into-source-reject":
            alias = sandbox / "alias"
            alias.symlink_to(source, target_is_directory=True)
            candidate = alias / "absent-child"
        before = capture(candidate)
        source_before = capture(source_file)
        alias_before = capture(alias) if alias is not None else None
        row = {"route": name, "status": "running", "requested_path": str(candidate),
               "resolved_parent": str(candidate.parent.resolve(strict=True)),
               "source_root": str(source.resolve(strict=True)), "before": before,
               "source_before": source_before, "alias_before": alias_before,
               "expected": expected, "subject": STATE.source,
               "executor": "Python_output_admission_only", "outside_67": True}
        STATE.records[name] = row
        STATE.flush()
        try:
            if expected == "accept":
                actual = helper(candidate, (source,))
                self.assertIsInstance(actual, Path)
                self.assertEqual(actual, candidate.resolve(strict=True))
                self.assertTrue(actual.is_dir())
                self.assertFalse(before["exists"])
                row["returned_path"] = str(actual)
            else:
                # OSError/PermissionError/other failures cannot satisfy a policy refusal.
                with self.assertRaises(ValueError) as caught:
                    helper(candidate, (source,))
                self.assertTrue(str(caught.exception).startswith(expected + ":"),
                                "wrong policy cause: " + str(caught.exception))
                row["observed_error"] = type(caught.exception).__name__ + ": " + str(caught.exception)
                self.assertEqual(capture(candidate), before, "rejected leaf was changed")
            self.assertEqual(capture(source_file), source_before, "source bytes changed")
            if alias is not None:
                self.assertEqual(capture(alias), alias_before, "source parent alias changed")
            row["status"] = "pass"
        except BaseException as error:
            row["status"] = "fail"
            row["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            row["after"] = capture(candidate)
            row["source_after"] = capture(source_file)
            row["alias_after"] = capture(alias) if alias is not None else None
            STATE.flush()

    def test_external_and_existing_paths(self):
        helper = STATE.subject(self)
        self.route(helper, "new-external-accept", "accept")
        self.route(helper, "existing-dir-reject", "OUTPUT_NOT_FRESH")
        self.route(helper, "existing-file-reject", "OUTPUT_NOT_FRESH")

    def test_source_and_symlink_paths(self):
        helper = STATE.subject(self)
        self.route(helper, "dangling-root-symlink-reject", "OUTPUT_NOT_FRESH")
        self.route(helper, "source-descendant-reject", "OUTPUT_IN_SOURCE")
        self.route(helper, "parent-alias-into-source-reject", "OUTPUT_IN_SOURCE")


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    STATE = Evidence(args.output)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdmissionContracts)
    method_ids = [test.id() for test in suite]
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    after = STATE.bind_source(SUBJECT)
    test_after = STATE.bind_source(Path(__file__).resolve())
    drift = after != STATE.source or test_after != STATE.test_source
    STATE.flush()
    final = {
        "record": "sdga-ir-admission-result", "methods_run": result.testsRun,
        "method_ids": method_ids,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "declared_routes": 6,
        "route_pass": sum(row["status"] == "pass" for row in STATE.records.values()),
        "route_fail": sum(row["status"] == "fail" for row in STATE.records.values()),
        "route_not_run": sum(row["status"] == "not_run" for row in STATE.records.values()),
        "source_before": STATE.source, "source_after": after,
        "test_before": STATE.test_source, "test_after": test_after, "source_drift": drift,
        "outside_67": True, "scientific_effect": "NONE", "qualification": "BLOCKED",
        "classification_status": "BLOCKED", "execution_authenticated": False,
        "credit": "real output-admission I/O only; no Lean or semantic-mutant credit",
    }
    write_json(STATE.output / "result.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if (result.wasSuccessful() and result.testsRun == 2 and not result.skipped
                 and final["route_pass"] == 6 and final["route_fail"] == 0
                 and final["route_not_run"] == 0 and not drift and STATE.binding_error is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
