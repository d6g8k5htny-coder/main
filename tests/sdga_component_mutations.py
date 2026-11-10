#!/usr/bin/env python3
"""Eight fixed Task E source mutations; no user selector or production implementation."""
from __future__ import annotations
import argparse
import difflib
import json
import os
from pathlib import Path
import sys
import traceback
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
support_path = ROOT / "tests/sdga_component_e_support.py"
support_raw = support_path.read_bytes()
S = types.ModuleType("sdga_e_mutation_support")
S.__file__, S.__package__ = str(support_path), None
sys.modules[S.__name__] = S
exec(compile(support_raw, str(support_path), "exec"), S.__dict__)
S.require(support_path.read_bytes() == support_raw, "mutation support drift")
PATCHES = [
  {
    "id": "M1",
    "old": "def encodeName (name : Name) : String :=\n  String.intercalate \"/\" (nameComponents name)\n",
    "new": "def encodeName (name : Name) : String :=\n  match name with\n  | .anonymous => \"\"\n  | _ => name.toString\n",
    "old_sha256": "586d4e4c8f598eea6b54991b5491c5765009bf00cf3ca255826e79bb5fd36813",
    "new_sha256": "45f8d42cd5b747d4ab9ce1912341bf2803857b4881a513b23fbc55c3069b0ce0"
  },
  {
    "id": "M2",
    "old": "def inventory (loaded : LoadedModule) : IO Unit := do\n  let observed ← fromAudit (collectObjectExtras loaded)\n  persistObjectEvidence loaded \"inventory\" observed\n  emit <| .mkObj [\n    (\"record\", .str \"header\"), (\"schema\", toJson (1 : Nat)),\n    (\"scope_id\", .str loaded.request.scopeId)\n  ]\n  let final := loaded.parts[loaded.parts.size - 1]!.2\n  let extras := observed.oleanNames\n  let mut objectParts : Array Json := #[]\n  for h : i in [:loaded.parts.size] do\n    let (level, part) := loaded.parts[i]\n    let requested := loaded.request.oleanParts[i]!\n    objectParts := objectParts.push <| .mkObj [\n      (\"level\", .str level), (\"sha256\", .str requested.sha256),\n      (\"const_names\", nameArray part.constNames),\n      (\"constant_names\", nameArray (part.constants.map (·.name)))\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"module\"), (\"name\", .str (encodeName loaded.request.module)),\n    (\"const_count\", toJson final.constants.size), (\"object_parts\", .arr objectParts),\n    (\"extra_const_names\", nameArray extras)\n  ]\n  let lookup := fun name => loaded.owned[name]? <|> loaded.env.find? name\n  let finalDigest := loaded.request.oleanParts[loaded.request.oleanParts.size - 1]!.sha256\n  let mut axiomUnion : Array Name := #[]\n  let mut axiomSeen : Std.HashSet Name := {}\n  for ci in final.constants do\n    let row ← observeConstant loaded ci\n    let axioms ← fromAudit (strictAxioms lookup #[ci.name])\n    for name in axioms do\n      if !axiomSeen.contains name then\n        axiomSeen := axiomSeen.insert name\n        axiomUnion := axiomUnion.push name\n    emit <| .mkObj [\n      (\"record\", .str \"declaration\"), (\"module\", .str (encodeName row.owner)),\n      (\"name\", .str (encodeName row.name)), (\"kind\", .str row.kind),\n      (\"is_prop\", .bool row.isProp), (\"is_private\", .bool row.isPrivate),\n      (\"unsafe\", .bool row.isUnsafe), (\"partial\", .bool row.isPartial),\n      (\"type_display\", .str row.typeDisplay), (\"object_part_sha256\", .str finalDigest),\n      (\"axioms\", nameArray axioms)\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"end\"), (\"module_count\", toJson (1 : Nat)),\n    (\"declaration_count\", toJson final.constants.size), (\"axioms\", nameArray axiomUnion)\n  ]\n",
    "new": "def inventory (loaded : LoadedModule) : IO Unit := do\n  let observed ← fromAudit (collectObjectExtras loaded)\n  persistObjectEvidence loaded \"inventory\" observed\n  emit <| .mkObj [\n    (\"record\", .str \"header\"), (\"schema\", toJson (1 : Nat)),\n    (\"scope_id\", .str loaded.request.scopeId)\n  ]\n  let exported := loaded.parts[0]!.2\n  let mut finalConstants : Array ConstantInfo := #[]\n  for name in exported.constNames do\n    let some ci := loaded.owned[name]? | fromAudit (.error (.missingConstant name))\n    finalConstants := finalConstants.push ci\n  let final := { exported with constants := finalConstants }\n  let extras := observed.oleanNames\n  let mut objectParts : Array Json := #[]\n  for h : i in [:loaded.parts.size] do\n    let (level, _part) := loaded.parts[i]\n    let requested := loaded.request.oleanParts[i]!\n    objectParts := objectParts.push <| .mkObj [\n      (\"level\", .str level), (\"sha256\", .str requested.sha256),\n      (\"const_names\", nameArray final.constNames),\n      (\"constant_names\", nameArray (final.constants.map (·.name)))\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"module\"), (\"name\", .str (encodeName loaded.request.module)),\n    (\"const_count\", toJson final.constants.size), (\"object_parts\", .arr objectParts),\n    (\"extra_const_names\", nameArray extras)\n  ]\n  let lookup := fun name => loaded.owned[name]? <|> loaded.env.find? name\n  let finalDigest := loaded.request.oleanParts[loaded.request.oleanParts.size - 1]!.sha256\n  let mut axiomUnion : Array Name := #[]\n  let mut axiomSeen : Std.HashSet Name := {}\n  for ci in final.constants do\n    let row ← observeConstant loaded ci\n    let axioms ← fromAudit (strictAxioms lookup #[ci.name])\n    for name in axioms do\n      if !axiomSeen.contains name then\n        axiomSeen := axiomSeen.insert name\n        axiomUnion := axiomUnion.push name\n    emit <| .mkObj [\n      (\"record\", .str \"declaration\"), (\"module\", .str (encodeName row.owner)),\n      (\"name\", .str (encodeName row.name)), (\"kind\", .str row.kind),\n      (\"is_prop\", .bool row.isProp), (\"is_private\", .bool row.isPrivate),\n      (\"unsafe\", .bool row.isUnsafe), (\"partial\", .bool row.isPartial),\n      (\"type_display\", .str row.typeDisplay), (\"object_part_sha256\", .str finalDigest),\n      (\"axioms\", nameArray axioms)\n    ]\n  emit <| .mkObj [\n    (\"record\", .str \"end\"), (\"module_count\", toJson (1 : Nat)),\n    (\"declaration_count\", toJson final.constants.size), (\"axioms\", nameArray axiomUnion)\n  ]\n",
    "old_sha256": "c930d35ec522f29a98cc297317d5e16992a70861d84a41885b1a86305b665f3e",
    "new_sha256": "33716209c37cfdbbc3fcfe63abe6ea1b1ebf1e9b1f01d03b1747351b4fba04d3"
  },
  {
    "id": "M3",
    "old": "    let some ci := lookup name | throw (.missingConstant name)\n",
    "new": "    let some ci := lookup name | return #[]\n",
    "old_sha256": "66e501853bc57053bc00248310d38bcdbb5881f1b0765ea02bc287d396c0fd6d",
    "new_sha256": "e1131ca4d18d46f68126d4b292f01af7b2cc131083f3281001718e7c8dee3701"
  },
  {
    "id": "M4",
    "old": "  for name in ci.getUsedConstantsAsSet do\n    result := result.push name\n",
    "new": "  if let some value := ci.value? (allowOpaque := true) then\n    for name in value.getUsedConstantsAsSet do\n      result := result.push name\n",
    "old_sha256": "661cad6acaf4755c18ecd5faae5b0073fc9442d21d9a8a8aa592d064982cb68e",
    "new_sha256": "55f5aec93890bd0696d8349eb8fb356e84825063e2f5a2ec8bac8f5d36096cb6"
  },
  {
    "id": "M5",
    "old": "def replayOwned (base : Kernel.Environment) (owned : Std.HashMap Name ConstantInfo) :\n    IO Kernel.Environment := do\n  fromAudit (guardReplayInput base owned)\n  let replayed ← base.replay owned\n  fromAudit (verifyReplayOutput replayed owned)\n  return replayed\n",
    "new": "def replayOwned (base : Kernel.Environment) (owned : Std.HashMap Name ConstantInfo) :\n    IO Kernel.Environment := do\n  let _ := owned\n  return base\n",
    "old_sha256": "a5fee0edee91b247e2fe3f04723195d195f703ba65cf04108f06be3effeab4f7",
    "new_sha256": "d557cbdb8be06b81f99874ecc8af1a63e6bf54ae48b677fe40e755c410fc54a8"
  },
  {
    "id": "M6",
    "old": "def guardReplayConstant (ci : ConstantInfo) : Except AuditError Unit := do\n  if ci.isUnsafe then throw (.unsafeReplaySkip ci.name)\n  if ci.isPartial then throw (.partialReplaySkip ci.name)\n",
    "new": "def guardReplayConstant (ci : ConstantInfo) : Except AuditError Unit := do\n  let _ := ci\n  return ()\n",
    "old_sha256": "da1050d07613d6a6a76cd360657f1ed06b07736168a31e01ea4c18f5926f0a18",
    "new_sha256": "4447e746e1b2423228a4f9df75d58879876b77c3d91c1a5365b965fd5aac3e88"
  },
  {
    "id": "M7",
    "old": "            residue := residue.push name\n",
    "new": "            residue := #[]\n",
    "old_sha256": "80b67ae1447cd678c67eb75384fbfe4a2358eaa6785eec66180a7306cfe9d18b",
    "new_sha256": "fc66e6ecf9eca48ad5e0aa3dcf115b3f0674ced29d8c5260218b421eed13bae4"
  },
  {
    "id": "M8",
    "old": "    if (base.find? name).isSome then throw (.duplicateBase name)\n",
    "new": "",
    "old_sha256": "2956455b7826722f9f1f03a7d31e5cc5b192f4f380a142a1e916abb1e82b52c8",
    "new_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
]
class Survived(Exception):
    """The valid selected original assertion passed under the intended source edit."""


EXPECTED_COMMANDS = 57
M2_SHA256 = "49dba7ad8bc96adadb53daba08db72b74bb50d4b66f7d9edd8c47ef997d59162"
IDS = ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
SELECTIONS = {
    "M1": ("C01", ("anon-roundtrip", "anon-identity-reject", "numeric-vs-string",
                    "empty-component", "dot-component", "slash-component",
                    "unicode-normalization-distinct", "nonbmp", "large-nat",
                    "uppercase-hex-reject", "leading-zero-reject", "invalid-utf8-reject"),
           "numeric-vs-string", "wire bytes differ: expected n12"),
    "M3": ("C05", ("missing-type", "missing-value", "missing-link"), "missing-type",
           "strictAxioms accepted an actual-CI alteration with absent dependency"),
    "M4": ("C04", ("type-route", "value-route", "opaque-route"), "type-route",
           "strict closure omitted exact BadProof through declaration type"),
    "M5": ("C09", ("actual-source-ci", "singleton-resolved-refs", "clean-base-absence",
                   "kernel-type-rejection"), "kernel-type-rejection",
           "shared replay accepted singleton malformed actual theorem"),
    "M7": ("C14", ("actual-ir-binding", "raw-names-preserved", "ci-backed-accounted",
                   "residue-retained-blocked"), "residue-retained-blocked",
           "shared IR evidence differs from independent raw/owned oracle"),
    "M8": ("C10", ("good-base-setup", "singleton-resolved-refs", "duplicate-base-guard"),
           "duplicate-base-guard",
           "expected exact pre-upstream duplicateBase for bad singleton; observed SDGA_ERROR MALFORMED_CONSTANT:"),
}
M6 = (("unsafe-replay-skip", "shared production guard accepted actual unsafe CI"),
      ("partial-replay-skip", "shared production guard accepted actual partial CI"))
C02_ASSERTION = '        self.assertEqual(set(emitted), set(actual), "inventory omitted actual final constants")'
BOOT_LABELS = ("cleanup-only-preflight", "main_commit", "main_tree", "lean-githash",
               "fixture-bad-dependency-compile", "fixture-safe-compile",
               "fixture-axiomuse-compile", "fixture-skipped-compile",
               "fixture-ownedaxiom-compile", "bridge-compile", "inventory")


def patch_for(family):
    return next(value for value in PATCHES if value["id"] == family)


def apply_patch(family, original):
    patch = patch_for(family)
    text = original.decode("utf-8")
    S.require(S.sha(patch["old"].encode()) == patch["old_sha256"], "old edit digest differs")
    S.require(S.sha(patch["new"].encode()) == patch["new_sha256"], "new edit digest differs")
    S.require(text.count(patch["old"]) == 1, "native preimage is not unique: " + family)
    altered = text.replace(patch["old"], patch["new"])
    if patch["new"]:
        inverse = altered.replace(patch["new"], patch["old"])
    else:
        position = text.index(patch["old"])
        inverse = altered[:position] + patch["old"] + altered[position:]
    S.require(inverse == text, "native edit inverse differs")
    return altered.encode("utf-8")


def freeze_files(destination, paths, expected=None):
    """Only fixed source/executable/evidence paths selected by the eight-family code."""
    for path in paths:
        path = str(path)
        measured = S.measure(Path(path))
        if expected is not None:
            S.require(measured == expected[path], "first-observed artifact differs: " + path)
        if path in destination:
            S.require(destination[path] == measured, "cannot reset frozen artifact: " + path)
        else:
            destination[path] = S.copy.deepcopy(measured)


def stream(command, which):
    path = Path(command[which + "_path"])
    raw = path.read_bytes()
    S.require(S.sha(raw) == command[which + "_sha256"]
              and len(raw) == command[which + "_bytes"], "original native stream drift")
    S.require(command["cleanup"]["confirmed"], "native cleanup uncertainty")
    S.require(not command["timed_out"] and command["launch_error"] is None
              and not command.get("interruption"), "nonsemantic native failure")
    return raw


def command_success(command):
    out, err = stream(command, "stdout"), stream(command, "stderr")
    S.require(command["exit_code"] == 0, "native setup/compile failed")
    return out, err


def strict_object(pairs):
    value = dict(pairs)
    S.require(len(value) == len(pairs), "duplicate native JSON key")
    return value


def control_rows(command, case, ids, target, fragment, mutated, controls):
    """Only original structured assertion plus its exact #eval diagnostic can kill."""
    raw, err = stream(command, "stdout"), stream(command, "stderr")
    S.require(err == b"", "unrelated native stderr is not an intended assertion")
    lines = raw.decode("utf-8").splitlines()
    rows, diagnostics = [], []
    for line in lines:
        if line.startswith("{"):
            row = json.loads(line, object_pairs_hook=strict_object)
            S.require(set(row) in (
                {"record", "case_id", "subcase_id", "executor", "status", "detail"},
                {"record", "case_id", "subcase_id", "executor", "status", "detail", "error_code"}),
                "unexpected native record/schema; no setup credit")
            S.require(row["record"] == "subcase" and row["case_id"] == case
                      and row["executor"] == "Lean", "wrong native case/executor")
            rows.append(row)
        else:
            diagnostics.append(line)
    if not mutated:
        S.require(command["exit_code"] == 0 and not diagnostics, "baseline control failed")
        S.require([row["subcase_id"] for row in rows] == list(ids)
                  and all(row["status"] == "pass" for row in rows), "baseline rows incomplete")
    else:
        end = ids.index(target)
        if command["exit_code"] == 0:
            S.require(not diagnostics and [row["subcase_id"] for row in rows] == list(ids)
                      and all(row["status"] == "pass" for row in rows),
                      "successful mutant command has invalid/incomplete oracle rows")
            raise Survived("selected original assertion passed: " + case + "/" + target)
        S.require(type(command["exit_code"]) is int and command["exit_code"] == 1,
                  "only ordinary pinned #eval assertion exit1 may kill; signal/OOM/crash is INVALID")
        S.require([row["subcase_id"] for row in rows] == list(ids[:end + 1]),
                  "selected assertion not reached through original predecessor rows")
        S.require(all(row["status"] == "pass" for row in rows[:-1]), "earlier assertion failed")
        failure = rows[-1]
        S.require(failure["status"] == "fail" and failure.get("error_code") == "CONTROL_ASSERT"
                  and type(failure["detail"]) is str and fragment in failure["detail"],
                  "unrelated failure is INVALID, not KILL")
        if case == "C10":
            S.require("replayed ConstantInfo differs from complete input" in failure["detail"],
                      "M8 requires the actual postcheck cause, not generic rejection")
        eval_line = controls.read_text().splitlines().index("#eval SDGA.runControls") + 1
        expected = (str(controls) + ":" + str(eval_line) + ":0: error: CONTROL_ASSERT: "
                    + case + "/" + target + ": " + failure["detail"])
        S.require(diagnostics == [expected], "unexpected compiler/import/warning diagnostic")
    return rows


class M2Result(unittest.TextTestResult):
    original_code = None
    assertion_line = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_failures = []

    def addFailure(self, test, err):
        kind, error, tb = err
        frames, exact = [], False
        while tb:
            code = tb.tb_frame.f_code
            frames.append({"file": code.co_filename, "line": tb.tb_lineno, "function": code.co_name})
            exact = exact or (code is self.original_code and tb.tb_lineno == self.assertion_line)
            tb = tb.tb_next
        self.original_failures.append({
            "type": kind.__module__ + "." + kind.__qualname__, "message": str(error),
            "same_original_code_and_actual_line": exact,
            "actual_assertion_line": self.assertion_line, "frames": frames,
            "traceback": "".join(traceback.format_exception(*err)),
        })
        super().addFailure(test, err)


def m2_state(tree, output, variant, commit):
    record = {"variant": variant, "status": "INVALID", "bootstrap": None,
              "unittest": None, "population": None, "final_measurements": None, "errors": [],
              "historical_R1_relabelled": False, **S.GLOBAL}
    state = None
    before_sources = None
    owned_before = None
    frozen_inputs = None
    frozen_generated = None
    record["frozen_artifacts"] = {}
    try:
        component = tree / "tools/sdga/Component.lean"
        original = component.read_bytes()
        S.require(S.blob(original) == S.COMPONENT_BLOB, "M2 actual source preimage differs")
        if variant == "mutant":
            changed = apply_patch("M2", original)
            S.require(S.sha(changed) == M2_SHA256, "M2 full source digest differs")
            component.write_bytes(changed)
        S.require(component.read_bytes() == (changed if variant == "mutant" else original),
                  "M2 actual source write/readback differs")
        component.chmod(0o444)
        module, binding = S.current_launcher(tree, "sdga_e_m2_" + variant)
        state = module.Evidence(output)
        module.STATE = state
        before_sources = S.copy.deepcopy(state.identities["sources"])
        copied_sources = []
        for relative, expected in before_sources.items():
            source_path = tree / relative
            S.require(S.identity(source_path) == expected, "M2 source capture differs")
            copy_path = state.output / "probe-sources" / relative
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            raw = source_path.read_bytes()
            with copy_path.open("xb") as copy:
                copy.write(raw)
            S.require(copy_path.read_bytes() == raw, "M2 source-copy readback differs")
            copied_sources.append(copy_path)
        freeze_files(record["frozen_artifacts"], copied_sources)
        state.bootstrap()
        record["bootstrap"] = {
            "errors": state.bootstrap_errors, "task_b_errors": state.task_b_errors,
            "controller_error": state.controller_error,
            "commit": state.identities.get("main_commit"), "tree": state.identities.get("main_tree"),
        }
        S.require(not state.bootstrap_errors and not state.task_b_errors
                  and state.controller_error is None, "M2 healthy bootstrap absent")
        S.require(state.identities["main_commit"] == commit, "M2 worktree source commit differs")
        frozen_inputs = S.copy.deepcopy(state.object_expected)
        frozen_generated = S.copy.deepcopy(state.ir_expected)
        owned_before = {
            str(path): S.measure(path) for path in state.owned.rglob("*") if path.is_file()}
        S.require(len(owned_before) == 36, "M2 fresh owned fixture/helper family incomplete")
        for path in owned_before:
            Path(path).chmod(0o444)
        S.require(tuple(row["label"] for row in state.commands) == BOOT_LABELS,
                  "M2 original bootstrap command roster changed")
        bridge_compile = next(row for row in state.commands if row["label"] == "bridge-compile")
        S.require(bridge_compile["argv"][-1] == str(component),
                  "M2 must compile the actual current worktree Component source")
        before_command, before_diff, _ = state.command(
            "m2-source-before-control", ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--"],
            cwd=tree)
        command_success(before_command)
        if variant == "baseline":
            S.require(before_diff == b"", "M2 baseline dirty source")
        else:
            S.require(before_diff.count(b"diff --git ") == 1 and before_diff.startswith(
                b"diff --git a/tools/sdga/Component.lean b/tools/sdga/Component.lean\n"),
                "M2 unrelated tracked edit")
        launcher_source = (tree / "tests/sdga_component_contract.py").read_text()
        S.require(launcher_source.count(C02_ASSERTION) == 1, "original assertion text not unique")
        M2Result.original_code = module.TaskAContracts.test_C02_owned_population.__code__
        M2Result.assertion_line = launcher_source[:launcher_source.index(C02_ASSERTION)].count("\n") + 1
        with (state.output / "selected-unittest.log").open("x", encoding="utf-8") as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=M2Result).run(
                unittest.TestSuite([module.TaskAContracts("test_C02_owned_population")]))
        record["unittest"] = {
            "methods": result.testsRun, "failures": len(result.failures),
            "errors": len(result.errors), "skipped": len(result.skipped),
            "original_failures": result.original_failures,
        }
        after_command, after_diff, _ = state.command(
            "m2-source-after-control", ["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "--"],
            cwd=tree)
        command_success(after_command)
        S.require(before_diff == after_diff, "M2 source diff drift")
        S.require(result.testsRun == 1 and not result.errors and not result.skipped,
                  "M2 original selected case did not complete")
        if variant == "baseline":
            S.require(result.wasSuccessful() and not result.failures, "M2 baseline did not pass")
        else:
            if result.wasSuccessful():
                raise Survived("M2 original population assertion passed")
            S.require(len(result.failures) == len(result.original_failures) == 1,
                      "M2 unrelated unittest failure")
            failure = result.original_failures[0]
            S.require(failure["type"] == "builtins.AssertionError"
                      and failure["same_original_code_and_actual_line"]
                      and failure["message"].endswith(" : inventory omitted actual final constants"),
                      "M2 wrong original failure cause")
        oracle = S.json_read(state.output / "control-observation-control-C02.json")
        S.require(oracle["exit_code"] == 0 and not oracle["setup_rows"] and not oracle["protocol_errors"],
                  "M2 native setup/oracle refusal")
        S.require([row["subcase_id"] for row in oracle["rows"]] ==
                  list(module.CASES["C02"]) and all(row["status"] == "pass" for row in oracle["rows"]),
                  "M2 native C02 incomplete")
        actual = oracle["rows"][-1]["detail"]["final_names"]
        exported = oracle["rows"][-1]["detail"]["raw_parts"][0]["const_names"]
        emitted = [row["name"] for row in state.inventory["declarations"]]
        manifest = S.json_read(Path(str(state.request_path) + ".inventory.ir-manifest.json"))
        envelope_path = state.output / "ir-evidence/01-inventory.json"
        envelope = S.json_read(envelope_path)
        S.require(len(actual) == 15 and len(exported) == 12
                  and manifest["olean_parts"][-1]["const_names"] == actual,
                  "M2 genuine raw population changed")
        S.require(emitted == (actual if variant == "baseline" else exported),
                  "M2 selected actual population/order differs")
        S.require(envelope["validation_status"] == "valid" and envelope["errors"] == []
                  and envelope["manifest_completeness"] == "complete"
                  and envelope["registration"] == "new", "M2 D retention invalid")
        record["population"] = {"native_final_names": actual, "native_exported_names": exported,
                                "protocol_names": emitted, "manifest_final_count": 15,
                                "protocol_count": len(emitted), "native_rows": oracle["rows"]}
        record["status"] = "PASS" if variant == "baseline" else "KILL"
    except BaseException as error:
        if isinstance(error, Survived):
            record["status"] = "SURVIVED"
        record["errors"].append({"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()})
    finally:
        if state is not None:
            try:
                if before_sources is None:
                    before_sources = dict(state.identities["sources"])
                source_after = {path: S.identity(tree / path) for path in before_sources}
                expected_inputs = (frozen_inputs if frozen_inputs is not None
                                   else S.copy.deepcopy(state.object_expected))
                expected_generated = (frozen_generated if frozen_generated is not None
                                      else S.copy.deepcopy(state.ir_expected))
                inputs_after = module.measure_objects(tuple(Path(path) for path in expected_inputs))
                generated = state._check_generated("e-m2-final")
                S.require(state.object_expected == expected_inputs
                          and state.ir_expected == expected_generated, "M2 expected registry reset")
                S.require(all(S.measure(Path(path)) == value["measurement"]
                              for path, value in expected_generated.items()),
                          "M2 generated artifact differs from first observation")
                owned_after = ({path: S.measure(Path(path)) for path in owned_before}
                               if owned_before is not None else None)
                record["final_measurements"] = {
                    "owned_before": owned_before, "owned_after": owned_after,
                    "source_before": before_sources, "source_after": source_after,
                    "input_expected": expected_inputs, "input_after": inputs_after,
                    "generated_expected": expected_generated, "generated_after": generated,
                    "observations": state.object_observations, "requests": state.request_bindings,
                    "commands": state.commands, "old_full_finalizers_called": False,
                }
                S.require(source_after == before_sources and inputs_after == expected_inputs
                          and owned_before is not None and owned_after == owned_before,
                          "M2 source/input/owned dependency drift")
                S.require(len(state.commands) == 14 and len(state.object_observations) == 2
                          and len(state.request_bindings) == 2 and len(state.object_expected) == 12
                          and len(state.ir_contexts) == 1 and len(state.ir_expected) == 2
                          and not state.object_failures and not state.ir_failures,
                          "M2 narrow source-derived census incomplete")
                state.flush()
                freeze_files(record["frozen_artifacts"], [tree / path for path in before_sources])
                freeze_files(record["frozen_artifacts"], owned_before, owned_before)
                freeze_files(record["frozen_artifacts"], expected_inputs, expected_inputs)
                freeze_files(record["frozen_artifacts"], expected_generated,
                             {path: value["measurement"] for path, value in expected_generated.items()})
                for command in state.commands:
                    freeze_files(record["frozen_artifacts"],
                                 [command["stdout_path"], command["stderr_path"]])
                freeze_files(record["frozen_artifacts"], [
                    state.output / "selected-unittest.log",
                    state.output / "control-observation-control-C02.json",
                    state.output / "object-custody/01-inventory.json",
                    state.output / "object-custody/02-control-C02.json",
                    state.output / "object-custody.json", state.output / "ir-evidence-registry.json"])
            except BaseException as error:
                record["status"] = "INVALID"
                record["errors"].append({"type": type(error).__name__, "message": str(error),
                                         "traceback": traceback.format_exc()})
        if state is None:
            output.mkdir(parents=True, exist_ok=True)
        S.json_write(output / "m2-selected-result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    S.require(sys.flags.dont_write_bytecode and sys.flags.no_site and sys.flags.optimize in (0, 1),
              "exact -B [-O] -S invocation required")
    mode = "optimized" if sys.flags.optimize else "normal"
    droot = Path(os.environ["RUNNER_TEMP"]).resolve(strict=True) / ("sdga-task-a-" + mode)
    eroot = Path(str(droot) + "-controller")
    io, io_binding = S.captured(S.IR_HELPER, "sdga_e_mutant_io")
    root = io.admit_output_root(args.output, (S.ROOT,))
    report = {
        "record": "sdga-e-mutation-batch", "schema": 1, "mode": mode,
        "candidate_commit": os.environ["GITHUB_SHA"], "run_id": os.environ["GITHUB_RUN_ID"],
        "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "status": "BLOCKED",
        "expected_families": list(IDS), "expected_commands": EXPECTED_COMMANDS,
        "families": {family: {"status": "not_run", "reason": "healthy D+E baseline prerequisite"}
                     for family in IDS},
        "prerequisite_error": None, "commands": None, "input_custody": None,
        "dependency_before": None, "dependency_after": None,
        "source_before": None, "source_after": None, "actual_commands": 0,
        "full_E_qualification_claim": False,
        "scientific_promotion": False, **S.GLOBAL,
    }
    supervisor = None
    try:
        d = S.json_read(droot / "result.json")
        e = S.json_read(eroot / "result.json")
        S.require(d["methods_run"] == 15 and d["subcase_pass"] == 67
                  and not d["failures"] and not d["errors"] and not d["skipped"],
                  "original D baseline incomplete")
        S.require(e["status"] == "pass" and e["methods_run"] == 3 and e["subcase_pass"] == 19
                  and e["candidate_commit"] == os.environ["GITHUB_SHA"]
                  and e["mode"] == mode, "current same-mode E API baseline incomplete")
        for path, before in e["source_after"].items():
            S.require(S.identity(Path(path)) == before, "prerequisite source drift")
        launcher, launcher_binding = S.current_launcher()
        S.require(S.blob(S.COMPONENT.read_bytes()) == S.COMPONENT_BLOB
                  and S.blob(S.CONTROLS.read_bytes()) == S.CONTROLS_BLOB,
                  "native source preimages changed")
        supervisor = launcher.Evidence(root / "supervision")
        supervisor.enable_subreaper()
        supervisor.cleanup_only_preflight()
        for expression, label in (("HEAD^{commit}", "main_commit"), ("HEAD^{tree}", "main_tree")):
            command, raw, err = supervisor.command(label, ["git", "rev-parse", "--verify", expression],
                                                    cwd=S.ROOT)
            command_success(command)
            S.require(not err, "Git identity stderr")
            if label == "main_commit":
                commit = raw.decode().strip()
                S.require(commit == os.environ["GITHUB_SHA"], "actual mutation source commit differs")
        command, raw, err = supervisor.command("lean-githash", ["lean", "--githash"])
        command_success(command)
        S.require(raw.decode().strip() == S.RELEASE and not err, "actual Lean source revision differs")
        report["dependency_before"] = {
            str(path): S.measure(path) for path in (droot / "owned").rglob("*") if path.is_file()}
        report["source_before"] = {path: S.identity(Path(path)) for path in e["source_after"]}
        for path in report["dependency_before"]:
            Path(path).chmod(0o444)
        request_paths = {
            "Safe": droot / "request.json",
            "AxiomUse": droot / "task-b-AxiomUse-request.json",
            "Skipped": droot / "task-b-Skipped-request.json",
            "BadDependency": droot / "task-b-BadDependency-request.json",
        }
        for path in request_paths.values():
            supervisor.bind_request(path, S.json_read(path))
        wt_baseline = root.parent / (root.name + "-M2-baseline-tree")
        wt_mutant = root.parent / (root.name + "-M2-mutant-tree")
        for variant, tree in (("baseline", wt_baseline), ("mutant", wt_mutant)):
            command, _, _ = supervisor.command(
                "e-m2-worktree-" + variant,
                ["git", "worktree", "add", "--detach", str(tree), commit], cwd=S.ROOT)
            command_success(command)
        original = S.COMPONENT.read_bytes()
        for family in IDS:
            family_root = root / family
            family_root.mkdir()
            record = {"status": "INVALID", "source_edit": patch_for(family),
                      "source_original_sha256": S.sha(original), "source_mutant_sha256": None,
                      "baseline": [], "mutant": [], "dependency_before": None,
                      "dependency_after": None, "frozen_artifacts": {}, "error": None,
                      "global_final_custody_error": None}
            report["families"][family] = record
            try:
                changed = apply_patch(family, original)
                record["source_mutant_sha256"] = S.sha(changed)
                (family_root / "Component.original.lean").write_bytes(original)
                (family_root / "Component.mutant.lean").write_bytes(changed)
                S.require((family_root / "Component.original.lean").read_bytes() == original
                          and (family_root / "Component.mutant.lean").read_bytes() == changed,
                          "retained source copy readback differs")
                with (family_root / "source.diff").open("x", encoding="utf-8") as log:
                    log.writelines(difflib.unified_diff(
                        original.decode().splitlines(True), changed.decode().splitlines(True),
                        fromfile="a/tools/sdga/Component.lean", tofile="b/tools/sdga/Component.lean"))
                freeze_files(record["frozen_artifacts"], [
                    family_root / "Component.original.lean", family_root / "Component.mutant.lean",
                    family_root / "source.diff"])
                if family == "M2":
                    baseline = m2_state(wt_baseline, family_root / "baseline", "baseline", commit)
                    record["baseline"].append(baseline)
                    record["frozen_artifacts"].update(S.copy.deepcopy(baseline["frozen_artifacts"]))
                    freeze_files(record["frozen_artifacts"],
                                 [family_root / "baseline/m2-selected-result.json"])
                    S.require(baseline["status"] == "PASS", "M2 same-route baseline failed")
                    mutant = m2_state(wt_mutant, family_root / "mutant", "mutant", commit)
                    record["mutant"].append(mutant)
                    record["frozen_artifacts"].update(S.copy.deepcopy(mutant["frozen_artifacts"]))
                    freeze_files(record["frozen_artifacts"],
                                 [family_root / "mutant/m2-selected-result.json"])
                    if mutant["status"] == "SURVIVED":
                        raise Survived("M2 original population assertion passed")
                    S.require(mutant["status"] == "KILL", "M2 original population assertion not reached")
                else:
                    source_root = family_root / "source"
                    source_root.mkdir()
                    component = source_root / "Component.lean"
                    component.write_bytes(changed)
                    S.require(component.read_bytes() == changed, "executed copy write/readback differs")
                    component.chmod(0o444)
                    freeze_files(record["frozen_artifacts"], [component])
                    out = family_root / "owned"
                    out.mkdir()
                    command, _, _ = supervisor.lean(
                        family + "-compile",
                        ["--root=" + str(source_root), "-o", str(out / "Component.olean"),
                         "-i", str(out / "Component.ilean"), str(component)],
                        search_root=str(out) + ":" + str(droot / "owned"))
                    compiled_out, compiled_err = command_success(command)
                    S.require(compiled_out == compiled_err == b"", "mutant compile warning/output")
                    objects = tuple(path for path in out.rglob("*") if path.is_file())
                    S.require(len(objects) == 6, "mutant compiled family incomplete")
                    record["dependency_before"] = {str(path): S.measure(path) for path in objects}
                    freeze_files(record["frozen_artifacts"], objects, record["dependency_before"])
                    S.require(S.measure(component) == record["frozen_artifacts"][str(component)],
                              "executed source drift during compile")
                    for path in objects:
                        path.chmod(0o444)
                    if family == "M6":
                        cases = [("C07", (subcase,), subcase, fragment) for subcase, fragment in M6]
                    else:
                        cases = [SELECTIONS[family]]
                    for case, expected_ids, target, fragment in cases:
                        if case == "C04":
                            selected_requests = (request_paths["AxiomUse"], request_paths["BadDependency"])
                        elif case == "C07":
                            selected_requests = (request_paths["Skipped"],)
                        else:
                            selected_requests = (request_paths["Safe"],)
                        for variant in ("baseline", "mutant"):
                            environment = {"SDGA_CONTROL_CASE": case,
                                           "SDGA_CONTROL_REQUEST": str(request_paths["Safe"])}
                            if case in ("C04", "C07"):
                                environment["SDGA_CONTROL_TASK_B_REQUEST"] = str(
                                    request_paths["AxiomUse" if case == "C04" else "Skipped"])
                            if case == "C07":
                                environment["SDGA_CONTROL_SUBCASE"] = target
                            search = str(droot / "owned") if variant == "baseline" else (
                                str(out) + ":" + str(droot / "owned"))
                            label = family + "-" + variant + "-" + target
                            shared_before = {
                                path: S.measure(Path(path)) for path in report["dependency_before"]}
                            S.require(shared_before == report["dependency_before"],
                                      "shared owned dependencies drifted before selected variant")
                            command, _, _ = supervisor.observe_lean(
                                label, ["--root=" + str(S.FIXTURE), str(S.CONTROLS)],
                                requests=selected_requests, search_root=search, env_extra=environment)
                            rows = None
                            pending = None
                            try:
                                rows = control_rows(command, case, expected_ids, target, fragment,
                                                    variant == "mutant", S.CONTROLS)
                            except BaseException as error:
                                pending = error
                            try:
                                shared_after = {
                                    path: S.measure(Path(path)) for path in report["dependency_before"]}
                                S.require(shared_after == shared_before, "shared dependencies drifted in variant")
                                helper_after = {str(path): S.measure(path) for path in objects}
                                S.require(helper_after == record["dependency_before"],
                                          "executable dependency drifted in selected variant")
                                S.require(S.measure(component) ==
                                          record["frozen_artifacts"][str(component)],
                                          "executed source drifted in selected variant")
                                freeze_files(record["frozen_artifacts"],
                                             [command["stdout_path"], command["stderr_path"]])
                            except BaseException as resource_error:
                                if pending is not None:
                                    raise resource_error from pending
                                raise
                            if pending is not None:
                                raise pending
                            record[variant].append({"command": command, "rows": rows,
                                                    "selected_case": case, "target": target,
                                                    "shared_dependencies_before": shared_before,
                                                    "shared_dependencies_after": shared_after,
                                                    "helper_dependencies_after": helper_after})
                    record["dependency_after"] = {str(path): S.measure(path) for path in objects}
                    S.require(record["dependency_after"] == record["dependency_before"],
                              "mutant executable dependency changed")
                record["status"] = "KILL"
            except BaseException as error:
                if isinstance(error, Survived):
                    record["status"] = "SURVIVED"
                record["error"] = {"type": type(error).__name__, "message": str(error),
                                   "traceback": traceback.format_exc()}
            S.json_write(family_root / "observed-result.json",
                         {"phase": "pending-global-final-custody", **record})
        report["dependency_after"] = {
            path: S.measure(Path(path)) for path in report["dependency_before"]}
        report["source_after"] = {path: S.identity(Path(path)) for path in report["source_before"]}
        S.require(report["source_after"] == report["source_before"], "current candidate source drift")
        S.require(report["dependency_after"] == report["dependency_before"], "shared owned dependency drift")
        for family, value in report["families"].items():
            for path, expected in value.get("frozen_artifacts", {}).items():
                S.require(S.measure(Path(path)) == expected,
                          "batch-final family artifact drift: " + family + ": " + path)
        for command in supervisor.commands:
            for which in ("stdout", "stderr"):
                S.require(S.measure(Path(command[which + "_path"])) == {
                    "exists": True, "bytes": command[which + "_bytes"],
                    "sha256": command[which + "_sha256"]}, "batch-final native stream drift")
        inputs_after = launcher.measure_objects(tuple(Path(path) for path in supervisor.object_expected))
        S.require(inputs_after == supervisor.object_expected and not supervisor.object_failures,
                  "fixed mutation control input custody failed")
        all_killed = all(value["status"] == "KILL" for value in report["families"].values())
        if all_killed:
            S.require(len(supervisor.object_observations) == 16
                      and len(supervisor.request_bindings) == 4 and len(supervisor.object_expected) == 24,
                      "non-M2 fixed selected census differs")
        report["commands"] = supervisor.commands
        report["input_custody"] = {
            "expected": supervisor.object_expected, "after": inputs_after,
            "observations": supervisor.object_observations,
            "request_bindings": supervisor.request_bindings, "failures": supervisor.object_failures}
        m2_commands = sum(len(value["final_measurements"]["commands"])
                          for value in report["families"]["M2"]["baseline"]
                          + report["families"]["M2"]["mutant"]
                          if value["final_measurements"] is not None)
        actual = len(supervisor.commands) + m2_commands
        report["actual_commands"] = actual
        if all_killed:
            S.require(actual == EXPECTED_COMMANDS, "fixed total command census differs")
            report["status"] = "pass"
    except BaseException as error:
        report["prerequisite_error"] = {"type": type(error).__name__, "message": str(error),
                                         "traceback": traceback.format_exc()}
        for value in report["families"].values():
            if value["status"] in ("KILL", "SURVIVED"):
                value["status"] = "INVALID"
                value["global_final_custody_error"] = str(error)
    finally:
        if supervisor is not None:
            try:
                supervisor.flush()
            except BaseException as error:
                report["status"] = "BLOCKED"
                report["prerequisite_error"] = {
                    "type": type(error).__name__, "message": "final Evidence.flush: " + str(error),
                    "traceback": traceback.format_exc()}
                for value in report["families"].values():
                    if value["status"] in ("KILL", "SURVIVED"):
                        value["status"] = "INVALID"
                        value["global_final_custody_error"] = str(error)
        for family, value in report["families"].items():
            family_root = root / family
            if family_root.is_dir():
                S.json_write(family_root / "result.json", value)
        S.json_write(root / "result.json", report)
    print(json.dumps({"record": report["record"], "mode": mode, "status": report["status"],
                      "result": str(root / "result.json"), **S.GLOBAL}, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
