#!/usr/bin/env python3
"""Fixed selected-C02 M2 reachability probe; no full D/E qualification."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback
import types
import unittest

BASE_COMMIT = "02973f3975a1c4cc91b7c51d372f65717c9b3c5c"
BASE_TREE = "5d3160d982b4513f17848de45e0f4569233acd4e"
RECIPE_DESIGN_BLOB = "792e27b601997d28c9f4385667dae2aa28b0b083"
COMPONENT = "tools/sdga/Component.lean"
LAUNCHER = "tests/sdga_component_contract.py"
PINS = {
    COMPONENT: "ccf340774887d612ef5956405fb53f925eb1f424",
    LAUNCHER: "c45b685f21f05232dad1e84898cac06f354e336c",
    "tools/sdga_ir_evidence.py": "b29702b941d03f1db34464a6bd9db2cf65de5632",
    "tools/sdga_scope.py": "14d4617b523c29007da4bdb95ee6ee4dc588a86c",
    "tools/sdga_component.py": "372dab47ade30bc2202ebf554ab1def9b1900211",
    "tests/fixtures/sdga_component/Controls.lean": "9e14a94302f3c5988f0af51bc9fc1f74836a9fae",
}
ORIGINAL_SHA256 = "3491f417b9746daebf9059d98967e059a542733a57fa93bfd9fc47a2b49fd7ee"
MUTANT_SHA256 = "49dba7ad8bc96adadb53daba08db72b74bb50d4b66f7d9edd8c47ef997d59162"
OLD_SHA256 = "c930d35ec522f29a98cc297317d5e16992a70861d84a41885b1a86305b665f3e"
NEW_SHA256 = "33716209c37cfdbbc3fcfe63abe6ea1b1ebf1e9b1f01d03b1747351b4fba04d3"
OLD = "def inventory (loaded : LoadedModule) : IO Unit := do\n  let observed ← fromAudit (collectObjectExtras loaded)\n  persistObjectEvidence loaded \"inventory\" observed\n  emit <| .mkObj [\n    (\"record\", .str \"header\"), (\"schema\", toJson (1 : Nat)),\n    (\"scope_id\", .str loaded.request.scopeId)\n  ]\n  let final := loaded.parts[loaded.parts.size - 1]!.2\n  let extras := observed.oleanNames\n  let mut objectParts : Array Json := #[]\n  for h : i in [:loaded.parts.size] do\n    let (level, part) := loaded.parts[i]\n    let requested := loaded.request.oleanParts[i]!\n    objectParts := objectParts.push <| .mkObj [\n      (\"level\", .str level), (\"sha256\", .str requested.sha256),\n      (\"const_names\", nameArray part.constNames),\n      (\"constant_names\", nameArray (part.constants.map (·.name)))\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"module\"), (\"name\", .str (encodeName loaded.request.module)),\n    (\"const_count\", toJson final.constants.size), (\"object_parts\", .arr objectParts),\n    (\"extra_const_names\", nameArray extras)\n  ]\n  let lookup := fun name => loaded.owned[name]? <|> loaded.env.find? name\n  let finalDigest := loaded.request.oleanParts[loaded.request.oleanParts.size - 1]!.sha256\n  let mut axiomUnion : Array Name := #[]\n  let mut axiomSeen : Std.HashSet Name := {}\n  for ci in final.constants do\n    let row ← observeConstant loaded ci\n    let axioms ← fromAudit (strictAxioms lookup #[ci.name])\n    for name in axioms do\n      if !axiomSeen.contains name then\n        axiomSeen := axiomSeen.insert name\n        axiomUnion := axiomUnion.push name\n    emit <| .mkObj [\n      (\"record\", .str \"declaration\"), (\"module\", .str (encodeName row.owner)),\n      (\"name\", .str (encodeName row.name)), (\"kind\", .str row.kind),\n      (\"is_prop\", .bool row.isProp), (\"is_private\", .bool row.isPrivate),\n      (\"unsafe\", .bool row.isUnsafe), (\"partial\", .bool row.isPartial),\n      (\"type_display\", .str row.typeDisplay), (\"object_part_sha256\", .str finalDigest),\n      (\"axioms\", nameArray axioms)\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"end\"), (\"module_count\", toJson (1 : Nat)),\n    (\"declaration_count\", toJson final.constants.size), (\"axioms\", nameArray axiomUnion)\n  ]\n"
NEW = "def inventory (loaded : LoadedModule) : IO Unit := do\n  let observed ← fromAudit (collectObjectExtras loaded)\n  persistObjectEvidence loaded \"inventory\" observed\n  emit <| .mkObj [\n    (\"record\", .str \"header\"), (\"schema\", toJson (1 : Nat)),\n    (\"scope_id\", .str loaded.request.scopeId)\n  ]\n  let exported := loaded.parts[0]!.2\n  let mut finalConstants : Array ConstantInfo := #[]\n  for name in exported.constNames do\n    let some ci := loaded.owned[name]? | fromAudit (.error (.missingConstant name))\n    finalConstants := finalConstants.push ci\n  let final := { exported with constants := finalConstants }\n  let extras := observed.oleanNames\n  let mut objectParts : Array Json := #[]\n  for h : i in [:loaded.parts.size] do\n    let (level, _part) := loaded.parts[i]\n    let requested := loaded.request.oleanParts[i]!\n    objectParts := objectParts.push <| .mkObj [\n      (\"level\", .str level), (\"sha256\", .str requested.sha256),\n      (\"const_names\", nameArray final.constNames),\n      (\"constant_names\", nameArray (final.constants.map (·.name)))\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"module\"), (\"name\", .str (encodeName loaded.request.module)),\n    (\"const_count\", toJson final.constants.size), (\"object_parts\", .arr objectParts),\n    (\"extra_const_names\", nameArray extras)\n  ]\n  let lookup := fun name => loaded.owned[name]? <|> loaded.env.find? name\n  let finalDigest := loaded.request.oleanParts[loaded.request.oleanParts.size - 1]!.sha256\n  let mut axiomUnion : Array Name := #[]\n  let mut axiomSeen : Std.HashSet Name := {}\n  for ci in final.constants do\n    let row ← observeConstant loaded ci\n    let axioms ← fromAudit (strictAxioms lookup #[ci.name])\n    for name in axioms do\n      if !axiomSeen.contains name then\n        axiomSeen := axiomSeen.insert name\n        axiomUnion := axiomUnion.push name\n    emit <| .mkObj [\n      (\"record\", .str \"declaration\"), (\"module\", .str (encodeName row.owner)),\n      (\"name\", .str (encodeName row.name)), (\"kind\", .str row.kind),\n      (\"is_prop\", .bool row.isProp), (\"is_private\", .bool row.isPrivate),\n      (\"unsafe\", .bool row.isUnsafe), (\"partial\", .bool row.isPartial),\n      (\"type_display\", .str row.typeDisplay), (\"object_part_sha256\", .str finalDigest),\n      (\"axioms\", nameArray axioms)\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"end\"), (\"module_count\", toJson (1 : Nat)),\n    (\"declaration_count\", toJson final.constants.size), (\"axioms\", nameArray axiomUnion)\n  ]\n"
C02 = ("raw-tables", "foreign-owner", "explicit-private", "module-private",
       "generated-present", "dependency-excluded", "final-rows-owner-census")
BOOT_LABELS = (
    "cleanup-only-preflight", "main_commit", "main_tree", "lean-githash",
    "fixture-bad-dependency-compile", "fixture-safe-compile",
    "fixture-axiomuse-compile", "fixture-skipped-compile",
    "fixture-ownedaxiom-compile", "bridge-compile", "inventory",
)
COMMAND_LABELS = (*BOOT_LABELS, "m2-source-before-control", "control-C02",
                  "m2-source-after-control")
EXPECTED = {
    "selected_methods": 1, "selected_native_subcases": 7,
    "guarded_observations": 2, "request_bindings": 2, "input_paths": 12,
    "ir_contexts": 1, "ir_outputs": 2, "commands": 14,
    "genuine_final_declarations": 15, "genuine_exported_declarations": 12,
    "other_original_subcases_not_run": 60,
}
GLOBAL = {
    "scientific_effect": "NONE", "qualification": "BLOCKED",
    "classification_status": "BLOCKED", "execution_authenticated": False,
}
SELECTED_CODE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def file_identity(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and path.resolve(strict=True) == path,
            "canonical regular source required: " + str(path))
    raw = path.read_bytes()
    return {"exists": True, "bytes": len(raw), "sha256": sha(raw),
            "git_blob": git_blob(raw)}, raw


def verify_pins(tree):
    require((tree / ".git").is_file(), "actual git worktree file missing")
    require((tree / ".git").read_text().startswith("gitdir: "), "not a git worktree")
    for relative, expected in PINS.items():
        identity, _ = file_identity(tree / relative)
        require(identity["git_blob"] == expected, "source pin differs: " + relative)
    require(not any(tree.rglob("*.pyc")), "fresh worktree has pre-existing bytecode")


def load_original(tree, variant):
    path = tree / LAUNCHER
    identity, raw = file_identity(path)
    require(identity["git_blob"] == PINS[LAUNCHER], "launcher source differs")
    module = types.ModuleType("sdga_m2_original_" + variant)
    module.__file__ = str(path)
    module.__package__ = None
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(path.read_bytes() == raw and module.ROOT == tree, "launcher captured-source drift")
    return module


class ProbeResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_details = []

    def addFailure(self, test, err):
        kind, error, tb = err
        frames = []
        original_frame = False
        while tb is not None:
            code = tb.tb_frame.f_code
            frames.append({"file": code.co_filename, "line": tb.tb_lineno,
                           "function": code.co_name})
            if code is SELECTED_CODE and tb.tb_lineno == 1166:
                original_frame = True
            tb = tb.tb_next
        self.failure_details.append({
            "type": kind.__module__ + "." + kind.__qualname__,
            "message": str(error), "original_method_line1166": original_frame,
            "frames": frames, "traceback": "".join(traceback.format_exception(*err)),
        })
        super().addFailure(test, err)


def capture_state_sources(module, state):
    captured = {}
    destination = state.output / "probe-sources"
    destination.mkdir()
    for relative, before in state.identities["sources"].items():
        identity, raw = file_identity(module.ROOT / relative)
        require(identity == before, "captured source mismatch: " + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        captured[relative] = identity
    return captured


def current_sources(module, captured):
    return {relative: file_identity(module.ROOT / relative)[0] for relative in captured}


def inspect_population(module, state, variant):
    control = json.loads((state.output / "control-observation-control-C02.json").read_text())
    require(control["exit_code"] == 0 and not control["setup_rows"]
            and not control["protocol_errors"], "original native C02 failed setup/protocol")
    rows = control["rows"]
    require([row["subcase_id"] for row in rows] == list(C02)
            and all(row["status"] == "pass" for row in rows), "native C02 oracle incomplete")
    actual = rows[-1]["detail"]["final_names"]
    raw_parts = rows[-1]["detail"]["raw_parts"]
    require(len(actual) == 15 and len(set(actual)) == 15, "genuine final census differs")
    require([part["level"] for part in raw_parts] == ["exported", "server", "private"],
            "genuine raw part order differs")
    require(len(raw_parts[0]["const_names"]) == 12, "genuine exported census differs")
    manifest_path = Path(str(state.request_path) + ".inventory.ir-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["olean_parts"][-1]["const_names"] == actual,
            "genuine manifest differs from unchanged native oracle")
    require(manifest["olean_parts"][0]["const_names"] == raw_parts[0]["const_names"],
            "genuine exported manifest differs from unchanged oracle")
    envelope_path = state.output / "ir-evidence" / "01-inventory.json"
    envelope = json.loads(envelope_path.read_text())
    require(envelope["validation_status"] == "valid" and not envelope["errors"]
            and envelope["manifest_completeness"] == "complete"
            and envelope["registration"] == "new"
            and envelope["observation_error"] is None, "D retention did not remain healthy")
    protocol = state.inventory
    emitted = [row["name"] for row in protocol["declarations"]]
    wanted_count = 15 if variant == "baseline" else 12
    require(len(emitted) == wanted_count and len(set(emitted)) == wanted_count,
            "protocol projection census differs")
    require(len(protocol["modules"]) == 1, "protocol module census differs")
    projected = protocol["modules"][0]
    require([part["sha256"] for part in projected["object_parts"]] ==
            [part["sha256"] for part in state.request["olean_parts"]],
            "M2 changed actual part hashes")
    require(all(row["object_part_sha256"] == state.request["olean_parts"][-1]["sha256"]
                for row in protocol["declarations"]), "private declaration digest changed")
    if variant == "baseline":
        require(emitted == actual, "healthy baseline population differs")
    else:
        require(emitted == raw_parts[0]["const_names"] and set(emitted) < set(actual),
                "actual exported projection was not observed")
        require(all(part["const_names"] == emitted and part["constant_names"] == emitted
                    for part in projected["object_parts"]), "M2 projection is incoherent")
    return {
        "native_c02_subcases": rows, "native_final_names": actual,
        "native_exported_names": raw_parts[0]["const_names"], "protocol_names": emitted,
        "missing_names": sorted(set(actual) - set(emitted)),
        "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path.read_bytes()),
        "envelope_path": str(envelope_path), "envelope_sha256": sha(envelope_path.read_bytes()),
        "manifest_final_count": len(manifest["olean_parts"][-1]["const_names"]),
        "protocol_count": len(emitted),
    }


def narrow_snapshot(module, state, captured):
    value = {
        "source_before": captured, "source_after": None,
        "input_expected": state.object_expected, "input_after": None,
        "observations": state.object_observations, "request_bindings": state.request_bindings,
        "generated_expected": state.ir_expected, "generated_after": None,
        "measurement_errors": [], "full_d_finalizers_called": False,
        "actual": {"guarded_observations": len(state.object_observations),
                   "request_bindings": len(state.request_bindings),
                   "input_paths": len(state.object_expected), "ir_contexts": len(state.ir_contexts),
                   "ir_outputs": len(state.ir_expected), "commands": len(state.commands)},
    }
    try:
        value["source_after"] = current_sources(module, captured)
    except BaseException as error:
        value["measurement_errors"].append("sources: " + type(error).__name__ + ": " + str(error))
    try:
        value["input_after"] = module.measure_objects(
            tuple(Path(path) for path in state.object_expected))
    except BaseException as error:
        value["measurement_errors"].append("inputs: " + type(error).__name__ + ": " + str(error))
    try:
        value["generated_after"] = state._check_generated("m2-selected-final")
    except BaseException as error:
        value["measurement_errors"].append("outputs: " + type(error).__name__ + ": " + str(error))
    return value


def validate_narrow_snapshot(state, value):
    require(not value["measurement_errors"], "selected final measurement failed")
    require(value["source_after"] == value["source_before"], "source drift during selected probe")
    after_inputs = value["input_after"]
    require(after_inputs == state.object_expected, "final input measurement differs")
    require(len(state.object_observations) == 2 and len(state.request_bindings) == 2
            and len(state.object_expected) == 12, "selected input census differs")
    require([row["native_label"] for row in state.object_observations] ==
            ["inventory", "control-C02"]
            and all(row["status"] == "pass" for row in state.object_observations)
            and not state.object_failures, "selected custody failed")
    require(len(state.ir_contexts) == 1 and len(state.ir_expected) == 2
            and len(state.ir_observations) == 1 and not state.ir_failures,
            "selected generated-output census/failure differs")
    require(not state.scheduling_blocked and state.cleanup_preflight["status"] == "pass",
            "cleanup unavailable or uncertain")
    require([row["label"] for row in state.commands] == list(COMMAND_LABELS),
            "fixed command census/order differs")
    for command in state.commands:
        if command["label"] == "cleanup-only-preflight":
            require(command["timeout_expected_cleanup_only"] is True,
                    "cleanup-only timeout identity differs")
        else:
            require(command["exit_code"] == 0 and not command["timed_out"]
                    and command["launch_error"] is None and not command.get("interruption"),
                    "nonsemantic command failure: " + command["label"])
        require(command["cleanup"]["confirmed"], "child cleanup not confirmed")
    selected = [row for row in state.records.values() if row["status"] != "not_run"]
    require(len(selected) == 7 and all(row["case_id"] == "C02"
            and row["status"] == "pass" for row in selected), "unexpected native case credit")


def selected_state(mode_root, tree, variant, patch):
    global SELECTED_CODE
    record = {
        "record": "sdga-m2-selected-c02-state", "schema": 1,
        "mode": "optimized" if sys.flags.optimize else "normal", "variant": variant,
        "base_commit": BASE_COMMIT, "base_tree": BASE_TREE, "worktree": str(tree),
        "status": "invalid", "expected": EXPECTED, "bootstrap": None,
        "unittest": None, "population": None, "narrow_custody": None,
        "errors": [], "full_d_qualification_claim": False,
        "full_e_qualification_claim": False, "mutation_kill_claim": False, **GLOBAL,
    }
    state = None
    module = None
    captured = None
    try:
        verify_pins(tree)
        original = (tree / COMPONENT).read_bytes()
        require(sha(original) == ORIGINAL_SHA256, "original Component SHA differs")
        if variant == "mutant":
            text = original.decode("utf-8")
            require(sha(OLD.encode()) == OLD_SHA256 and sha(NEW.encode()) == NEW_SHA256,
                    "literal preimage recipe digest differs")
            require(text.count(OLD) == 1, "preimage is not unique")
            altered = text.replace(OLD, NEW).encode("utf-8")
            require(sha(altered) == MUTANT_SHA256
                    and altered.decode().replace(NEW, OLD).encode() == original,
                    "M2 source/inverse differs")
            with (tree / COMPONENT).open("wb") as stream:
                stream.write(altered)
            patch["applied_once"] = True
        module = load_original(tree, variant)
        state = module.Evidence(mode_root / variant)
        module.STATE = state
        captured = capture_state_sources(module, state)
        state.bootstrap()
        record["bootstrap"] = {
            "bootstrap_errors": state.bootstrap_errors,
            "task_b_errors": state.task_b_errors,
            "controller_error": state.controller_error,
            "actual_commit": state.identities.get("main_commit"),
            "actual_tree": state.identities.get("main_tree"),
            "actual_lean_githash": state.identities.get("actual_lean_githash"),
        }
        require(not state.bootstrap_errors and not state.task_b_errors
                and state.controller_error is None and state.controller is not None,
                "healthy original bootstrap was not established")
        require(state.identities["main_commit"] == BASE_COMMIT
                and state.identities["main_tree"] == BASE_TREE, "actual worktree Git identity differs")
        require(tuple(row["label"] for row in state.commands) == BOOT_LABELS,
                "original bootstrap command roster differs")
        diff_before, diff_raw, _ = state.command(
            "m2-source-before-control",
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--"],
            cwd=module.ROOT)
        require(diff_before["exit_code"] == 0, "actual source diff failed")
        if variant == "baseline":
            require(diff_raw == b"", "baseline tracked source is dirty")
        else:
            require(diff_raw.count(b"diff --git ") == 1
                    and diff_raw.startswith(b"diff --git a/tools/sdga/Component.lean "
                                            b"b/tools/sdga/Component.lean\n"),
                    "variant has another tracked source change")
        SELECTED_CODE = module.TaskAContracts.test_C02_owned_population.__code__
        require(SELECTED_CODE.co_firstlineno == 1158
                and SELECTED_CODE.co_filename == str(tree / LAUNCHER),
                "original method code identity differs")
        with (state.output / "selected-unittest.log").open("x", encoding="utf-8") as stream:
            suite = unittest.TestSuite([
                module.TaskAContracts("test_C02_owned_population")])
            result = unittest.TextTestRunner(stream=stream, verbosity=2,
                                            resultclass=ProbeResult).run(suite)
        record["unittest"] = {
            "tests_run": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "failure_details": result.failure_details,
        }
        after_command, diff_after, _ = state.command(
            "m2-source-after-control",
            ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--"],
            cwd=module.ROOT)
        require(after_command["exit_code"] == 0 and diff_after == diff_raw,
                "tracked source diff drifted")
        require(result.testsRun == 1 and not result.errors and not result.skipped,
                "selected unittest did not complete exactly once")
        if variant == "baseline":
            require(result.wasSuccessful() and not result.failures,
                    "unmutated original C02 did not pass")
        else:
            require(len(result.failures) == 1 and len(result.failure_details) == 1,
                    "variant did not fail only one original Python assertion")
            failure = result.failure_details[0]
            require(failure["type"] == "builtins.AssertionError"
                    and failure["original_method_line1166"]
                    and failure["message"].endswith(" : inventory omitted actual final constants"),
                    "variant failed for an unrelated cause")
        record["population"] = inspect_population(module, state, variant)
        record["status"] = "baseline_pass" if variant == "baseline" else "reachability_confirmed"
    except BaseException as error:
        record["errors"].append({
            "type": type(error).__module__ + "." + type(error).__qualname__,
            "message": str(error), "traceback": traceback.format_exc(),
        })
    finally:
        if state is not None:
            try:
                record["narrow_custody"] = narrow_snapshot(
                    module, state, captured if captured is not None else state.identities["sources"])
                validate_narrow_snapshot(state, record["narrow_custody"])
            except BaseException as error:
                record["status"] = "invalid"
                record["errors"].append({
                    "type": type(error).__module__ + "." + type(error).__qualname__,
                    "message": "selected final measurement: " + str(error),
                    "traceback": traceback.format_exc(),
                })
            try:
                state.flush()
            except BaseException as error:
                record["status"] = "invalid"
                record["errors"].append({
                    "type": type(error).__module__ + "." + type(error).__qualname__,
                    "message": "original Evidence.flush: " + str(error),
                    "traceback": traceback.format_exc(),
                })
        write_json(mode_root / (variant + "-state.json"), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(sys.flags.dont_write_bytecode and sys.flags.no_site
            and sys.flags.optimize in (0, 1), "exact -B [-O] -S invocation required")
    mode = "optimized" if sys.flags.optimize else "normal"
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True)
    checkout = Path(__file__).resolve().parents[1]
    baseline_tree = runner_temp / ("sdga-m2-" + mode + "-baseline-tree")
    mutant_tree = runner_temp / ("sdga-m2-" + mode + "-mutant-tree")
    verify_pins(baseline_tree)
    verify_pins(mutant_tree)
    loader = load_original(baseline_tree, "admission")
    api, _ = loader._load_ir_io_subject()
    root = api["admit_output_root"](
        args.output, (checkout, baseline_tree, mutant_tree))
    context = {
        "record": "sdga-m2-probe-context", "schema": 1,
        "candidate_commit": os.environ["GITHUB_SHA"],
        "run_id": os.environ["GITHUB_RUN_ID"], "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"],
        "github_job": os.environ["GITHUB_JOB"], "python": sys.version,
        "python_optimized": bool(sys.flags.optimize), "worker_path": str(Path(__file__).resolve()),
        "worker_sha256": sha(Path(__file__).read_bytes()),
        "workflow_path": ".github/workflows/sdga-m2-reachability-probe.yml",
        "workflow_sha256": sha((checkout / ".github/workflows/sdga-m2-reachability-probe.yml").read_bytes()),
        "base_commit": BASE_COMMIT, "base_tree": BASE_TREE,
        "recipe_design_blob": RECIPE_DESIGN_BLOB, "expected": EXPECTED, **GLOBAL,
    }
    write_json(root / "probe-context.json", context)
    patch = {
        "record": "sdga-m2-source-edit", "schema": 1, "path": COMPONENT,
        "original_git_blob": PINS[COMPONENT], "original_sha256": ORIGINAL_SHA256,
        "mutant_sha256": MUTANT_SHA256, "old_sha256": OLD_SHA256,
        "new_sha256": NEW_SHA256, "old": OLD, "new": NEW,
        "expected_occurrences": 1, "applied_once": False,
    }
    original = (baseline_tree / COMPONENT).read_bytes()
    altered = original.decode().replace(OLD, NEW).encode()
    with (root / "m2-source.diff").open("x", encoding="utf-8") as stream:
        stream.writelines(difflib.unified_diff(
            original.decode().splitlines(True), altered.decode().splitlines(True),
            fromfile="a/" + COMPONENT, tofile="b/" + COMPONENT))
    baseline = selected_state(root, baseline_tree, "baseline", patch)
    mutant = None
    if baseline["status"] == "baseline_pass":
        mutant = selected_state(root, mutant_tree, "mutant", patch)
    write_json(root / "m2-source-edit.json", patch)
    confirmed = (baseline["status"] == "baseline_pass" and mutant is not None
                 and mutant["status"] == "reachability_confirmed" and patch["applied_once"])
    result = {
        "record": "sdga-m2-reachability-result", "schema": 1, "mode": mode,
        "status": "confirmed" if confirmed else "hold",
        "baseline_status": baseline["status"],
        "mutant_status": "not_run" if mutant is None else mutant["status"],
        "expected": EXPECTED, "attempted_states": 1 if mutant is None else 2,
        "reachability_only": True, "mutation_kill_claim": False,
        "full_d_qualification_claim": False, "full_e_qualification_claim": False, **GLOBAL,
    }
    write_json(root / "probe-result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())
