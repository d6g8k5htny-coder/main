"""Fixed Task E test support; no production import may depend on this file."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/sdga_component"
COMPONENT = ROOT / "tools/sdga/Component.lean"
CONTROLLER = ROOT / "tools/sdga_component.py"
LAUNCHER = ROOT / "tests/sdga_component_contract.py"
IR_HELPER = ROOT / "tools/sdga_ir_evidence.py"
CONTROLS = FIXTURE / "Controls.lean"
RELEASE = "5045d0056413266e57c625dcd7c365b10e377c52"
CLASS_SHA256 = "3b3ded1e79b33227db7da5c25175bb65af1ad423af4878b7b9ad3c56d4565cbd"
COMPONENT_BLOB = "ccf340774887d612ef5956405fb53f925eb1f424"
CONTROLS_BLOB = "9e14a94302f3c5988f0af51bc9fc1f74836a9fae"
MODULES = ("Fixture.BadDependency", "Fixture.Safe", "Fixture.AxiomUse",
           "Fixture.Skipped", "Fixture.OwnedAxiom")
LEAVES = ("BadDependency", "Safe", "AxiomUse", "Skipped", "OwnedAxiom")
RELATIVE_FILES = tuple("Fixture/" + leaf + ".lean" for leaf in LEAVES)
GLOBAL = dict(scientific_effect="NONE", qualification="BLOCKED",
              classification_status="BLOCKED", execution_authenticated=False)
E_IDS = {
    "E01": ("deps-json-header-graph", "compile-fixed-argv",
            "fresh-parts-source", "reused-output-cli-refused"),
    "E02": ("inventory-BadDependency", "inventory-Safe", "inventory-AxiomUse",
            "inventory-Skipped", "inventory-OwnedAxiom", "cohort-custody",
            "existing-c08-replay-preserved", "unknown-mode-refused"),
    "E03": ("complete-native-composition", "counts-and-axiom-union",
            "missing-module-refused", "duplicate-module-refused",
            "scope-mismatch-refused", "malformed-jsonl-refused",
            "negative-fixture-findings-preserved"),
}
SCOPE_A = {
    "scope_id": "sdga-task-a-fixture",
    "modules": [{"module": "Fixture.Safe", "path": "Fixture/Safe.lean", "role": "production"}],
    "targets": ["Fixture.Safe.proof", "Fixture.Safe.propValue", "Fixture.Safe.natValue"],
    "expected_counts": {"public_targets": 3, "proposition_targets": 1, "data_targets": 2},
}
SCOPE_E = {
    "scope_id": "sdga-task-e-fixture-cohort-v1",
    "modules": [{"module": module, "path": path, "role": "production"}
                for module, path in zip(MODULES, RELATIVE_FILES)],
    "targets": SCOPE_A["targets"][:], "expected_counts": SCOPE_A["expected_counts"].copy(),
}
API_PARAMETERS = {
    "compile_owned_module": ("source_root", "source_file", "module", "output_root",
                             "dependency_path", "evidence_root"),
    "observe_module": ("mode", "request_path", "output_root", "dependency_path", "evidence_root"),
    "combine_inventory": ("transcripts", "scope"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_file() and path.resolve(strict=True) == path, "canonical file required: " + str(path))
    raw = path.read_bytes()
    return {"exists": True, "bytes": len(raw), "sha256": sha(raw), "git_blob": blob(raw)}


def measure(path):
    state = identity(path)
    return {key: state[key] for key in ("exists", "bytes", "sha256")}


def json_write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def captured(path, name):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(path), None
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(path.read_bytes() == raw, "captured source drift")
    return module, identity(path)


def original_class(raw):
    text = raw.decode("utf-8")
    start = text.index("class TaskAContracts(")
    end = text.index("\n\nclass Result(", start)
    return text[start:end].encode("utf-8")


def current_launcher(root=ROOT, name="sdga_e_original_launcher"):
    path = root / "tests/sdga_component_contract.py"
    raw = path.read_bytes()
    require(sha(original_class(raw)) == CLASS_SHA256, "original TaskAContracts class changed")
    module, binding = captured(path, name)
    require(module.ROOT == root, "launcher must execute at its real source root")
    return module, binding


def part_paths(owned, leaf):
    base = owned / "Fixture" / (leaf + ".olean")
    return (base, Path(str(base) + ".server"), Path(str(base) + ".private"),
            base.with_suffix(".ir.sig"), base.with_suffix(".ir"))


def request(task1, owned, leaf):
    paths = part_paths(owned, leaf)
    return {
        "schema": 1, "scope_id": SCOPE_E["scope_id"],
        "profile_sha256": sha(json.dumps(SCOPE_E, sort_keys=True).encode("utf-8")),
        "module": task1.encode_name(("Fixture", leaf)),
        "olean_parts": [
            {"level": level, "path": str(path), "sha256": sha(path.read_bytes())}
            for level, path in zip(("exported", "server", "private"), paths[:3])],
        "ir_parts": [
            {"level": level, "path": str(path), "sha256": sha(path.read_bytes())}
            for level, path in zip(("ir.sig", "ir"), paths[3:])],
    }


def fixed_output_paths(root):
    base = root / "Fixture/Safe.olean"
    return (base, Path(str(base) + ".server"), Path(str(base) + ".private"),
            base.with_suffix(".ir.sig"), base.with_suffix(".ir"), base.with_suffix(".ilean"))


class EContext:
    def __init__(self, launcher, dstate, dresult):
        self.launcher, self.dstate, self.dresult = launcher, dstate, dresult
        subject, self.subject_binding = captured(CONTROLLER, "sdga_e_captured_controller")
        self.subject = subject
        self.task1 = dstate.task1
        self.io, self.io_binding = captured(IR_HELPER, "sdga_e_captured_ir_io")
        self.root = self.io.admit_output_root(
            Path(str(dstate.output) + "-controller"), (ROOT,))
        self.api_root = self.root / "acquisition"
        self.compile_root = self.api_root / "compile-owned"
        self.supervisor = launcher.Evidence(self.root / "supervision")
        self.supervisor.enable_subreaper()
        self.supervisor.cleanup_only_preflight()
        self.records = {
            case + "/" + subcase: dict(
                record="sdga-e-subcase", schema=1, case_id=case, subcase_id=subcase,
                executor="Python_controller", status="not_run", command_label=None,
                evidence=[], error=None, **GLOBAL)
            for case, names in E_IDS.items() for subcase in names
        }
        self.compile_result = None
        self.compile_expected = None
        self.compile_producers = None
        self.input_expected = {}
        self.generated_expected = {}
        self.observations = []
        self.transcripts = []
        self.requests = []
        self.combined = None
        self.d_input_before = dict(dstate.object_expected)
        self.d_file_before = {
            str(path): measure(path)
            for path in dstate.output.rglob("*") if path.is_file()
        }
        self.dependencies_before = {
            str(path): measure(path)
            for path in dstate.owned.rglob("*") if path.is_file()
        }
        for path in self.dependencies_before:
            Path(path).chmod(0o444)
        source_paths = {ROOT / relative for relative in dstate.identities["sources"]}
        source_paths.update((Path(__file__).resolve(), CONTROLLER, COMPONENT, LAUNCHER,
                             IR_HELPER, CONTROLS,
                             ROOT / "tests/sdga_component_controller_contract.py",
                             ROOT / "tests/sdga_component_mutations.py"))
        self.source_before = {str(path): identity(path) for path in sorted(source_paths)}


    @contextlib.contextmanager
    def case(self, case_id, subcase_id):
        row = self.records[case_id + "/" + subcase_id]
        require(row["status"] == "not_run", "duplicate E subcase")
        try:
            yield row
        except BaseException as error:
            row["status"] = "fail"
            row["error"] = type(error).__module__ + "." + type(error).__qualname__ + ": " + str(error)
            raise
        else:
            row["status"] = "pass"

    def evidence(self, row, path):
        path = Path(path)
        row["evidence"].append({"path": str(path), **measure(path)})

    def stream(self, command, which):
        path = Path(command[which + "_path"])
        raw = path.read_bytes()
        require(len(raw) == command[which + "_bytes"] and sha(raw) == command[which + "_sha256"],
                "original stream differs: " + str(path))
        require(command["cleanup"]["confirmed"], "unconfirmed native cleanup")
        return raw

    def finish(self, result):
        source_after = {path: identity(Path(path)) for path in self.source_before}
        d_files_after = {str(path): measure(path) for path in self.dstate.output.rglob("*")
                         if path.is_file()}
        dependencies_after = {path: measure(Path(path)) for path in self.dependencies_before}
        errors = []
        if source_after != self.source_before:
            errors.append("SOURCE_DRIFT")
        d_preserved = (d_files_after == self.d_file_before
                       and self.dstate.object_expected == self.d_input_before)
        if not d_preserved:
            errors.append("FINALIZED_D_CHANGED")
        if dependencies_after != self.dependencies_before:
            errors.append("DEPENDENCY_DRIFT")
        compile_outputs = None
        ir_outputs = None
        final_inputs = {}
        final_generated = {}
        try:
            if self.compile_expected is not None:
                compile_outputs = json_read(self.compile_result["output_registry_path"])
                if (compile_outputs["status"] != "pass"
                        or compile_outputs["expected"] != self.compile_expected
                        or compile_outputs["after"] != self.compile_expected
                        or compile_outputs["producer_labels"] != self.compile_producers
                        or {str(path) for path in self.compile_root.rglob("*") if path.is_file()}
                           != set(self.compile_expected)
                        or {path: measure(Path(path)) for path in self.compile_expected}
                           != self.compile_expected):
                    errors.append("COMPILED_OUTPUT_MISSING_RESET_OR_DRIFT")
            final_inputs = {path: measure(Path(path)) for path in self.input_expected}
            if final_inputs != self.input_expected:
                errors.append("E_INPUT_DRIFT")
            generated_path = self.api_root / "ir-evidence-registry.json"
            if self.generated_expected:
                if not generated_path.is_file():
                    errors.append("IR_REGISTRY_REMOVED")
                else:
                    ir_outputs = json_read(generated_path)
                    if (ir_outputs["failures"]
                            or ir_outputs["expected_generated_outputs"] != self.generated_expected):
                        errors.append("IR_REGISTRY_RESET_OR_DRIFT")
                final_generated = {
                    path: measure(Path(path)) for path in self.generated_expected}
                if final_generated != {
                        path: value["measurement"] for path, value in self.generated_expected.items()}:
                    errors.append("IR_OUTPUT_DRIFT")
                actual_artifacts = {
                    str(path) for path in self.api_root.rglob("*.ir-manifest.json") if path.is_file()}
                actual_artifacts.update(str(path) for path in
                                        (self.api_root / "ir-evidence").glob("*.json")
                                        if path.is_file())
                if actual_artifacts != set(self.generated_expected):
                    errors.append("IR_ARTIFACT_SET_CHANGED")
            elif generated_path.is_file():
                ir_outputs = json_read(generated_path)
                if ir_outputs["expected_generated_outputs"]:
                    errors.append("IR_EXPECTATION_NOT_INDEPENDENTLY_FROZEN")
        except BaseException as error:
            errors.append("FINAL_REMEASUREMENT: " + type(error).__name__ + ": " + str(error))
        api_paths = set()
        actual_observations = actual_requests = 0
        api_registry = self.api_root / "object-custody.json"
        if api_registry.is_file():
            try:
                registered = json_read(api_registry)
                api_paths = set(registered["expected"])
                if registered["expected"] != self.input_expected:
                    errors.append("API_INPUT_REGISTRY_RESET_OR_INCOMPLETE")
                if registered["failures"] or any(
                        value["status"] != "pass" for value in registered["observations"]):
                    errors.append("API_INPUT_CUSTODY_RECORDED_FAILURE")
                actual_observations = len(registered["observations"])
                actual_requests = len(registered["request_bindings"])
            except BaseException as error:
                errors.append("API_CUSTODY_INVALID: " + type(error).__name__ + ": " + str(error))
                actual_observations = actual_requests = None
        elif self.api_root.exists():
            errors.append("API_CUSTODY_NOT_PRODUCED")
            actual_observations = actual_requests = None
        union = set(self.d_input_before) | api_paths
        observed = sum(row["status"] != "not_run" for row in self.records.values())
        passed = sum(row["status"] == "pass" for row in self.records.values())
        record = {
            "record": "sdga-e-controller-result", "schema": 1,
            "mode": "optimized" if sys.flags.optimize else "normal",
            "candidate_commit": os.environ["GITHUB_SHA"],
            "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
            "methods_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "declared_subcases": 19, "observed_subcases": observed, "subcase_pass": passed,
            "subcases": list(self.records.values()),
            "d_original": {"methods": 15, "subcases": 67, "guarded": 28,
                           "requests": 10, "paths": 55, "preserved": d_preserved,
                           "result_path": str(self.dstate.output / "result.json"),
                           "result_sha256": sha((self.dstate.output / "result.json").read_bytes())},
            "cohort_expected": {"observations": 5, "requests": 5, "paths": 30},
            "cohort_actual": {"observations": actual_observations,
                              "requests": actual_requests, "paths": len(api_paths),
                              "written_requests": len(self.requests),
                              "validated_observations": len(self.observations)},
            "union_expected": {"observations": 33, "requests": 15, "paths": 60},
            "union_actual": {"observations": (None if actual_observations is None else 28 + actual_observations),
                             "requests": (None if actual_requests is None else 10 + actual_requests),
                             "paths": len(union)},
            "source_before": self.source_before, "source_after": source_after,
            "dependency_before": self.dependencies_before, "dependency_after": dependencies_after,
            "d_files_before": self.d_file_before, "d_files_after": d_files_after,
            "frozen_compile_expected": self.compile_expected,
            "frozen_input_expected": self.input_expected,
            "frozen_generated_expected": self.generated_expected,
            "final_input_measurements": final_inputs,
            "final_generated_measurements": final_generated,
            "compile_outputs": compile_outputs, "ir_outputs": ir_outputs,
            "validation_errors": errors, **GLOBAL,
        }
        complete = (
            result.wasSuccessful() and result.testsRun == 3 and not result.skipped
            and observed == passed == 19 and not errors
            and len(self.observations) == len(self.requests) == 5
            and actual_observations == actual_requests == 5
            and len(api_paths) == 30 and len(union) == 60)
        record["status"] = "pass" if complete else "fail"
        json_write(self.root / "result.json", record)
        return complete
