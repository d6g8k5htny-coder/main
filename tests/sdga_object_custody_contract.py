#!/usr/bin/env python3
"""Three Python-only object-custody methods; separate from the six/39 Lean census.

These are real file-I/O byte fixtures, not Lean object or semantic-refusal evidence.
Missing shared helpers fail methods before any of the six custody routes execute.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tests" / "sdga_component_contract.py"
ROUTES = ("intact", "changed-before", "changed-after", "missing-after",
          "expected-missing", "appeared-after")
STATE = None
CHILD = (
    "import hashlib,json,sys,time\n"
    "from pathlib import Path\n"
    "rows={}\n"
    "for name in json.loads(sys.argv[1]):\n"
    " p=Path(name)\n"
    " try: raw=p.read_bytes()\n"
    " except FileNotFoundError: rows[name]={'exists':False}\n"
    " else: rows[name]={'exists':True,'bytes':len(raw),"
    "'sha256':hashlib.sha256(raw).hexdigest()}\n"
    "print(json.dumps({'record':'custody-read-ready','manifest':rows,"
    "'read_monotonic_ns':time.monotonic_ns()},sort_keys=True),flush=True)\n"
    "if sys.stdin.readline()!='continue\\n': raise SystemExit(2)\n"
    "print(json.dumps({'record':'custody-read-complete'}),flush=True)\n"
)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def source_identity():
    result = {}
    for path in (Path(__file__).resolve(), LAUNCHER,
                 ROOT / ".github" / "workflows" / "sdga-component-contract.yml"):
        raw = path.read_bytes()
        result[str(path.relative_to(ROOT))] = {
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": hashlib.sha1(
                b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest(),
        }
    return result


class Evidence:
    def __init__(self, output):
        output = output.resolve()
        if output == ROOT or ROOT in output.parents or output.exists():
            raise ValueError("OUTPUT_NOT_FRESH_EXTERNAL: " + str(output))
        output.mkdir(parents=True)
        self.output = output
        self.records = output / "records"
        self.records.mkdir()
        self.before_sources = source_identity()
        self.rows = {route: {
            "route": route, "status": "not_run", "reason": "body not invoked",
            "executor": "Python_object_custody_only",
        } for route in ROUTES}
        spec = importlib.util.spec_from_file_location("sdga_custody_launcher", LAUNCHER)
        if spec is None or spec.loader is None:
            raise ImportError("launcher loader unavailable")
        self.launcher = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.launcher
        spec.loader.exec_module(self.launcher)
        self.flush()

    def flush(self):
        (self.output / "cases.jsonl").write_text(
            "".join(json.dumps(self.rows[key], sort_keys=True) + "\n"
                    for key in ROUTES), encoding="utf-8")
        write_json(self.output / "source-identities.json", {
            "sources_before": self.before_sources,
            "scope": "Python_object_custody_only",
            "outside_lean_method_and_subcase_census": True,
            "python_optimized": bool(sys.flags.optimize),
            "scientific_effect": "NONE", "qualification": "BLOCKED",
            "classification_status": "BLOCKED", "execution_authenticated": False,
        })

    def files(self, route):
        case = self.output / ("case-" + route)
        case.mkdir()
        objects = (case / "sample.olean", case / "sample.ir")
        objects[0].write_bytes(b"Python byte-custody fixture OLean label only\n")
        objects[1].write_bytes(b"Python byte-custody fixture IR label only\n")
        request = case / "request.json"
        write_json(request, {
            "fixture_kind": "Python_byte_custody_only",
            "objects": [{"path": str(path), "sha256": hashlib.sha256(
                path.read_bytes()).hexdigest()} for path in objects],
        })
        return tuple(path.resolve() for path in (*objects, request))

    def observe(self, route, paths, mutation=None):
        command = [sys.executable, "-B", "-S", "-c", CHILD,
                   json.dumps([str(path) for path in paths])]
        entry = {"route": route, "argv": command, "exit_code": None,
                 "ready": None, "mutation_monotonic_ns": None, "error": None}
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
        first = ""
        rest = ""
        stderr = ""
        communicated = False
        try:
            if not select.select([process.stdout], [], [], 10)[0]:
                raise TimeoutError("child read handshake timed out")
            first = process.stdout.readline()
            ready = json.loads(first)
            if ready.get("record") != "custody-read-ready":
                raise ValueError("child did not report actual file read")
            entry["ready"] = ready
            if mutation is not None:
                mutation()
                entry["mutation_monotonic_ns"] = time.monotonic_ns()
            rest, stderr = process.communicate("continue\n", timeout=10)
            communicated = True
            if process.returncode != 0 or stderr:
                raise ValueError("fixed child read failed")
            if json.loads(rest).get("record") != "custody-read-complete":
                raise ValueError("child did not complete after handshake")
        except BaseException as exc:
            entry["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            if process.poll() is None:
                process.kill()
            if not communicated:
                tail, err = process.communicate()
                rest += tail
                stderr += err
            entry["exit_code"] = process.returncode
            (self.records / (route + ".stdout")).write_text(first + rest, encoding="utf-8")
            (self.records / (route + ".stderr")).write_text(stderr, encoding="utf-8")
            write_json(self.records / (route + "-child.json"), entry)
        return entry


class ObjectCustodyContracts(unittest.TestCase):
    def require_helpers(self):
        missing = [name for name in ("measure_objects", "object_custody")
                   if not callable(getattr(STATE.launcher, name, None))]
        if missing:
            reason = "IMPLEMENTATION_ABSENCE: " + ", ".join(missing)
            for row in STATE.rows.values():
                if row["status"] == "not_run":
                    row["reason"] = reason + "; no semantic-negative or mutation-kill credit"
            STATE.flush()
            self.fail(reason + " — custody routes not executed")

    def route(self, name, body):
        try:
            detail = body()
        except Exception as exc:
            STATE.rows[name].update(status="fail", reason=type(exc).__name__ + ": " + str(exc))
            STATE.flush()
            raise
        STATE.rows[name].update(status="pass", reason="actual route completed", detail=detail)
        STATE.flush()

    def verify_child(self, entry, expected, *, mutated):
        self.assertEqual(entry["exit_code"], 0)
        self.assertIsNone(entry["error"])
        self.assertEqual(entry["ready"]["manifest"], expected)
        if mutated:
            self.assertGreater(entry["mutation_monotonic_ns"],
                               entry["ready"]["read_monotonic_ns"])
        else:
            self.assertIsNone(entry["mutation_monotonic_ns"])

    def exercise(self, name, *, mutation_index=None, delete=False,
                 expected_missing=False, appearance=False):
        paths = STATE.files(name)
        target = paths[0] if mutation_index is None else paths[mutation_index]
        if expected_missing:
            target.unlink()  # Establish intentional absence BEFORE the expected snapshot.
        expected = STATE.launcher.measure_objects(paths)
        saved_expected = copy.deepcopy(expected)
        evidence_path = STATE.records / (name + "-custody.json")
        observed = None

        def mutate():
            if appearance:
                target.write_bytes(b"unexpected appearance after child read\n")
            elif delete:
                target.unlink()
            else:
                target.write_bytes(target.read_bytes() + b"actual changed bytes\n")

        drift_expected = mutation_index is not None or delete or appearance
        try:
            with STATE.launcher.object_custody(paths, expected, evidence_path):
                observed = STATE.observe(name, paths, mutate if drift_expected else None)
        except RuntimeError as exc:
            if not drift_expected:
                raise
            self.assertTrue(str(exc).startswith("OBJECT_DRIFT:"), str(exc))
        else:
            self.assertFalse(drift_expected, "changed/missing bytes escaped custody refusal")
        self.assertIsNotNone(observed, "required actual child observation did not complete")
        self.verify_child(observed, saved_expected, mutated=drift_expected)
        record = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(record["before"], saved_expected)
        self.assertEqual(record["after"], STATE.launcher.measure_objects(paths))
        self.assertEqual(record["drift"], [str(target)] if drift_expected else [])
        self.assertEqual(record["status"], "fail" if drift_expected else "pass")
        self.assertEqual(expected, saved_expected, "expected manifest was silently reset")
        return {"custody_record": str(evidence_path), "child_exit": observed["exit_code"]}

    def test_intact_actual_observation(self):
        self.require_helpers()

        def intact():
            detail = self.exercise("intact")
            paths = STATE.files("intact-body-error")
            expected = STATE.launcher.measure_objects(paths)
            saved = copy.deepcopy(expected)
            evidence = STATE.records / "intact-body-error-custody.json"
            observed = None
            with self.assertRaisesRegex(ValueError, "^OBSERVER_FAILURE:"):
                with STATE.launcher.object_custody(paths, expected, evidence):
                    observed = STATE.observe("intact-body-error", paths)
                    raise ValueError("OBSERVER_FAILURE: original body error retained")
            self.verify_child(observed, saved, mutated=False)
            record = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(record["before"], saved)
            self.assertEqual(record["after"], saved)
            self.assertEqual(record["drift"], [])
            self.assertEqual(record["status"], "fail")
            self.assertIn("OBSERVER_FAILURE:", record["observation_error"])
            self.assertEqual(expected, saved)
            detail["original_error_record"] = str(evidence)
            return detail
        self.route("intact", intact)

    def test_changed_before_and_after_actual_observation(self):
        self.require_helpers()

        def before():
            paths = STATE.files("changed-before")
            expected = STATE.launcher.measure_objects(paths)
            saved = copy.deepcopy(expected)
            paths[2].write_bytes(paths[2].read_bytes() + b" \n")  # Change actual request bytes.
            launched = []
            evidence = STATE.records / "changed-before-custody.json"
            with self.assertRaisesRegex(RuntimeError, "^OBJECT_DRIFT:"):
                with STATE.launcher.object_custody(paths, expected, evidence):
                    launched.append(True)
            self.assertEqual(launched, [], "before mismatch must prevent the child launch")
            record = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "fail")
            self.assertEqual(record["drift"], [str(paths[2])])
            self.assertNotEqual(record["before"], saved)
            self.assertEqual(expected, saved)
            return {"child_launched": False, "custody_record": str(evidence)}
        self.route("changed-before", before)
        self.route("changed-after", lambda: self.exercise("changed-after", mutation_index=1))

    def test_missing_and_intentionally_missing_actual_observations(self):
        self.require_helpers()
        self.route("missing-after", lambda: self.exercise("missing-after", delete=True))
        self.route("expected-missing", lambda: self.exercise(
            "expected-missing", expected_missing=True))
        self.route("appeared-after", lambda: self.exercise(
            "appeared-after", expected_missing=True, appearance=True))


def main():
    global STATE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    STATE = Evidence(args.output)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ObjectCustodyContracts)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    after_sources = source_identity()
    drift = [path for path, before in STATE.before_sources.items()
             if after_sources.get(path) != before]
    final = {
        "record": "python-object-custody-result", "methods_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": len(result.skipped), "declared_routes": len(ROUTES),
        "route_pass": sum(row["status"] == "pass" for row in STATE.rows.values()),
        "route_fail": sum(row["status"] == "fail" for row in STATE.rows.values()),
        "route_not_run": sum(row["status"] == "not_run" for row in STATE.rows.values()),
        "sources_after": after_sources, "source_drift": drift,
        "outside_lean_method_and_subcase_census": True,
        "scientific_effect": "NONE", "qualification": "BLOCKED",
        "classification_status": "BLOCKED", "execution_authenticated": False,
    }
    STATE.flush()
    write_json(STATE.output / "result.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return 0 if (result.wasSuccessful() and result.testsRun == 3 and not result.skipped
                 and final["route_pass"] == 6 and not drift) else 1


if __name__ == "__main__":
    raise SystemExit(main())
