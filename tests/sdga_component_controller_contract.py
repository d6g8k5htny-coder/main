#!/usr/bin/env python3
"""Fixed Task E test-first frontdoor: unchanged D suite, then three API contracts."""
from __future__ import annotations
import inspect
import json
import os
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
support_path = ROOT / "tests/sdga_component_e_support.py"
support_raw = support_path.read_bytes()
S = types.ModuleType("sdga_e_fixed_support")
S.__file__, S.__package__ = str(support_path), None
sys.modules[S.__name__] = S
exec(compile(support_raw, str(support_path), "exec"), S.__dict__)
S.require(support_path.read_bytes() == support_raw, "E support source drift")
CTX = None


class ControllerContracts(unittest.TestCase):
    """Real public helpers only. Missing APIs fail lazily inside their owning body."""

    def api(self, name):
        function = getattr(CTX.subject, name, None)
        if not callable(function):
            self.fail("IMPLEMENTATION_ABSENCE: tools/sdga_component.py." + name)
        self.assertEqual(tuple(inspect.signature(function).parameters), S.API_PARAMETERS[name])
        self.assertEqual(Path(function.__code__.co_filename), S.CONTROLLER)
        return function

    def command_ok(self, command):
        self.assertEqual(command["exit_code"], 0)
        self.assertFalse(command["timed_out"])
        self.assertIsNone(command["launch_error"])
        self.assertIsNone(command.get("interruption"))
        self.assertTrue(command["cleanup"]["confirmed"])
        return CTX.stream(command, "stdout"), CTX.stream(command, "stderr")

    def test_E01_compile_owned_module(self):
        """Catch missing native acquisition, wrong compile argv or stale/copied output paths."""
        with CTX.case("E01", "deps-json-header-graph") as row:
            paths = tuple(str(S.FIXTURE / path) for path in S.RELATIVE_FILES)
            command, raw, error = CTX.supervisor.lean(
                "e-deps-json-five", ["--deps-json", *paths], search_root=CTX.dstate.owned)
            self.command_ok(command)
            self.assertEqual(error, b"")
            document = json.loads(raw)
            self.assertEqual(len(document["imports"]), 5)
            self.assertTrue(all(value["errors"] == [] and value["result"]["isModule"] is True
                                for value in document["imports"]))
            parsed = CTX.subject.parse_dependency_headers(raw.decode("utf-8"), paths)
            edges = {}
            for module, path in zip(S.MODULES, paths):
                self.assertTrue(all(name in S.MODULES or name in ("Init", "Lean")
                                    for name in parsed[path]), "unknown dependency header import")
                edges[module] = tuple(name for name in parsed[path] if name in S.MODULES)
            self.assertEqual(edges, {
                "Fixture.BadDependency": (), "Fixture.Safe": (),
                "Fixture.AxiomUse": ("Fixture.BadDependency",),
                "Fixture.Skipped": (), "Fixture.OwnedAxiom": (),
            })
            self.assertEqual(CTX.subject.topological_modules(edges, S.MODULES), S.MODULES)
            S.json_write(CTX.root / "actual-dependency-headers.json", {
                "record": "sdga-e-actual-dependency-headers", "schema": 1,
                "files": paths, "raw": document, "local_edges": edges,
                "derived_order": S.MODULES, "command_label": command["label"], **S.GLOBAL})
            row["command_label"] = command["label"]
            CTX.evidence(row, command["stdout_path"])
            CTX.evidence(row, command["stderr_path"])
        with CTX.case("E01", "compile-fixed-argv") as row:
            function = self.api("compile_owned_module")
            self.assertFalse(CTX.api_root.exists())
            result = function(
                source_root=S.FIXTURE, source_file=S.FIXTURE / "Fixture/Safe.lean",
                module="Fixture.Safe", output_root=CTX.compile_root,
                dependency_path=str(CTX.dstate.owned), evidence_root=CTX.api_root)
            self.assertEqual(set(result), {
                "record", "schema", "module", "source_file", "source_before", "source_after",
                "output_root", "outputs", "command", "custody_path", "output_registry_path"})
            self.assertIs(type(result["schema"]), int)
            self.assertEqual((result["record"], result["schema"], result["module"]),
                             ("sdga-controller-compile", 1, "Fixture.Safe"))
            self.assertEqual(result["source_file"], str(S.FIXTURE / "Fixture/Safe.lean"))
            self.assertEqual(result["output_root"], str(CTX.compile_root))
            out, error = self.command_ok(result["command"])
            self.assertEqual(error, b"")
            baseline_compile = next(value for value in CTX.dstate.commands
                                    if value["label"] == "fixture-safe-compile")
            baseline_out, baseline_error = self.command_ok(baseline_compile)
            self.assertEqual(baseline_error, b"")
            self.assertEqual(out, baseline_out,
                             "extra/missing diagnostics versus actual same-source healthy Safe compile")
            CTX.evidence(row, baseline_compile["stdout_path"])
            CTX.evidence(row, baseline_compile["stderr_path"])
            expected = [
                "lake", "env", "env",
                "LEAN_PATH=" + str(CTX.compile_root) + ":" + str(CTX.dstate.owned),
                "lean", "--threads=2", "--root=" + str(S.FIXTURE),
                "-o", str(CTX.compile_root / "Fixture/Safe.olean"),
                "-i", str(CTX.compile_root / "Fixture/Safe.ilean"),
                str(S.FIXTURE / "Fixture/Safe.lean"),
            ]
            self.assertEqual(result["command"]["argv"], expected)
            self.assertEqual(result["command"]["cwd"], str(S.FIXTURE))
            CTX.compile_result = S.copy.deepcopy(result)
            S.json_write(CTX.root / "compile-result.json", result)
            row["command_label"] = result["command"]["label"]
            CTX.evidence(row, result["command"]["stdout_path"])
            CTX.evidence(row, result["command"]["stderr_path"])
        with CTX.case("E01", "fresh-parts-source") as row:
            result = CTX.compile_result
            self.assertEqual(result["source_before"], result["source_after"])
            self.assertEqual(result["source_before"], S.measure(S.FIXTURE / "Fixture/Safe.lean"))
            expected_paths = S.fixed_output_paths(CTX.compile_root)
            self.assertEqual(set(result["outputs"]), {str(path) for path in expected_paths})
            for path in expected_paths:
                self.assertTrue(path.is_relative_to(CTX.compile_root))
                self.assertFalse(path.is_relative_to(CTX.dstate.owned))
                self.assertEqual(result["outputs"][str(path)], S.measure(path))
            custody_path = Path(result["custody_path"])
            custody = S.json_read(custody_path)
            self.assertEqual(custody["status"], "pass")
            self.assertEqual(custody["source_path"], str(S.FIXTURE / "Fixture/Safe.lean"))
            self.assertEqual(custody["source_before"], result["source_before"])
            self.assertEqual(custody["source_after"], result["source_after"])
            self.assertEqual(custody["dependencies_expected"], CTX.dependencies_before)
            self.assertEqual(custody["dependencies_expected"], custody["dependencies_before"])
            self.assertEqual(custody["dependencies_expected"], custody["dependencies_after"])
            self.assertEqual(custody["errors"], [])
            self.assertEqual(custody["command_label"], result["command"]["label"])
            self.assertEqual(custody["record"], "sdga-controller-compile-custody")
            outputs_path = Path(result["output_registry_path"])
            outputs = S.json_read(outputs_path)
            self.assertEqual(outputs["status"], "pass")
            self.assertEqual(outputs["expected"], result["outputs"])
            self.assertEqual(outputs["after"], outputs["expected"])
            self.assertEqual(outputs["errors"], [])
            self.assertEqual(outputs["record"], "sdga-controller-compiled-outputs")
            self.assertEqual(set(outputs["producer_labels"]), set(result["outputs"]))
            CTX.compile_expected = S.copy.deepcopy(result["outputs"])
            CTX.compile_producers = S.copy.deepcopy(outputs["producer_labels"])
            CTX.evidence(row, custody_path)
            CTX.evidence(row, outputs_path)
        with CTX.case("E01", "reused-output-cli-refused") as row:
            command, raw, error = CTX.supervisor.command(
                "e-cli-existing-output",
                [sys.executable, "-B", *(["-O"] if sys.flags.optimize else []), "-S",
                 str(S.CONTROLLER), "fixtures", "--output", str(CTX.dstate.output)],
                cwd=S.ROOT)
            self.assertNotEqual(command["exit_code"], 0)
            self.assertFalse(command["timed_out"])
            self.assertIsNone(command["launch_error"])
            self.assertTrue(command["cleanup"]["confirmed"])
            self.assertEqual(raw, b"")
            self.assertIn("OUTPUT_NOT_FRESH:", error.decode("utf-8"))
            self.assertEqual({path: S.measure(Path(path)) for path in CTX.d_file_before},
                             CTX.d_file_before)
            row["command_label"] = command["label"]
            CTX.evidence(row, command["stdout_path"])
            CTX.evidence(row, command["stderr_path"])

    def test_E02_observe_module(self):
        """Catch wrong module resolution, missing custody/IR retention or relabelled cohorts."""
        for index, leaf in enumerate(S.LEAVES):
            with CTX.case("E02", "inventory-" + leaf) as row:
                function = self.api("observe_module")
                self.assertIsNotNone(CTX.compile_result, "E02_SETUP: actual API compile prerequisite failed")
                if not CTX.requests:
                    folder = CTX.api_root / "requests"
                    folder.mkdir()
                    for fixed_leaf in S.LEAVES:
                        path = folder / (fixed_leaf + ".json")
                        S.json_write(path, S.request(CTX.task1, CTX.dstate.owned, fixed_leaf))
                        CTX.requests.append(path)
                        CTX.input_expected[str(path)] = S.measure(path)
                        for part in S.json_read(path)["olean_parts"] + S.json_read(path)["ir_parts"]:
                            actual = S.measure(Path(part["path"]))
                            self.assertEqual(actual, CTX.d_input_before[part["path"]])
                            if part["path"] in CTX.input_expected:
                                self.assertEqual(actual, CTX.input_expected[part["path"]])
                            else:
                                CTX.input_expected[part["path"]] = S.copy.deepcopy(actual)
                path = CTX.requests[index]
                before = S.measure(path)
                result = function(
                    mode="inventory", request_path=path, output_root=CTX.dstate.owned,
                    dependency_path="", evidence_root=CTX.api_root)
                self.assertEqual(set(result), {
                    "record", "schema", "mode", "request_path", "command",
                    "custody_path", "envelope_path"})
                self.assertIs(type(result["schema"]), int)
                self.assertEqual((result["record"], result["schema"], result["mode"]),
                                 ("sdga-controller-observation", 1, "inventory"))
                self.assertEqual(result["request_path"], str(path))
                self.assertEqual(S.measure(path), before)
                raw, error = self.command_ok(result["command"])
                self.assertEqual(error, b"")
                self.assertEqual(result["command"]["argv"], [
                    "lake", "env", "env", "LEAN_PATH=" + str(CTX.dstate.owned),
                    "lean", "--threads=2", "--run", str(S.COMPONENT), "inventory", str(path)])
                self.assertEqual(result["command"]["cwd"], str(S.FIXTURE))
                parsed = CTX.task1.parse_inventory(raw.decode("utf-8"))
                self.assertEqual(parsed["scope_id"], S.SCOPE_E["scope_id"])
                self.assertEqual(len(parsed["modules"]), 1)
                self.assertEqual(parsed["modules"][0]["name"], CTX.task1.encode_name(("Fixture", leaf)))
                requested = S.json_read(path)
                custody = S.json_read(result["custody_path"])
                self.assertEqual(custody["status"], "pass")
                self.assertEqual(custody["expected"], custody["before"])
                self.assertEqual(custody["before"], custody["after"])
                self.assertEqual((custody["drift"], custody["measurement_errors"]), ([], []))
                self.assertEqual(custody["expected"][str(path)], CTX.input_expected[str(path)])
                self.assertEqual(custody["native_outcome"]["stdout_sha256"],
                                 result["command"]["stdout_sha256"])
                self.assertEqual(custody["native_outcome"]["stderr_sha256"],
                                 result["command"]["stderr_sha256"])
                for part in requested["olean_parts"] + requested["ir_parts"]:
                    self.assertEqual(custody["expected"][part["path"]]["sha256"], part["sha256"])
                envelope = S.json_read(result["envelope_path"])
                self.assertEqual((envelope["validation_status"], envelope["manifest_completeness"],
                                  envelope["registration"]), ("valid", "complete", "new"))
                self.assertEqual(envelope["errors"], [])
                self.assertEqual(envelope["producer_label"], result["command"]["label"])
                self.assertEqual(envelope["request_path"], str(path))
                self.assertEqual(envelope["input_custody"]["status"], "pass")
                self.assertEqual(envelope["native_outcome"]["stdout"]["sha256"], S.sha(raw))
                manifest_path = Path(str(path) + ".inventory.ir-manifest.json")
                manifest = S.json_read(manifest_path)
                self.assertEqual(manifest["request_path"], str(path))
                self.assertEqual(manifest["profile_sha256"], requested["profile_sha256"])
                self.assertEqual(manifest["scope_id"], S.SCOPE_E["scope_id"])
                self.assertEqual(manifest["olean_parts"][-1]["const_names"],
                                 [entry["name"] for entry in parsed["declarations"]])
                self.assertEqual(len(manifest["olean_parts"]), 3)
                self.assertEqual([entry["level"] for entry in manifest["ir_parts"]], ["ir.sig", "ir"])
                for artifact, kind in ((manifest_path, "manifest"),
                                       (Path(result["envelope_path"]), "envelope")):
                    self.assertFalse(str(artifact) in CTX.generated_expected, "artifact path reused")
                    CTX.generated_expected[str(artifact)] = {
                        "measurement": S.measure(artifact),
                        "producer_label": result["command"]["label"], "kind": kind}
                transcript_path = CTX.root / (leaf + ".inventory.jsonl")
                with transcript_path.open("xb") as stream:
                    stream.write(raw)
                CTX.transcripts.append(raw.decode("utf-8"))
                CTX.observations.append(result)
                row["command_label"] = result["command"]["label"]
                for evidence in (path, transcript_path, result["custody_path"],
                                 result["envelope_path"], manifest_path,
                                 result["command"]["stdout_path"], result["command"]["stderr_path"]):
                    CTX.evidence(row, evidence)
        with CTX.case("E02", "cohort-custody") as row:
            registry_path = CTX.api_root / "object-custody.json"
            registry = S.json_read(registry_path)
            self.assertEqual(len(registry["observations"]), 5)
            self.assertEqual(len(registry["request_bindings"]), 5)
            self.assertEqual(len(registry["expected"]), 30)
            self.assertEqual(registry["failures"], [])
            self.assertEqual(registry["expected"], CTX.input_expected)
            expected_paths = {str(path) for path in CTX.requests}
            for leaf in S.LEAVES:
                expected_paths.update(str(path) for path in S.part_paths(CTX.dstate.owned, leaf))
            self.assertEqual(set(registry["expected"]), expected_paths)
            self.assertEqual({path: S.measure(Path(path)) for path in expected_paths}, registry["expected"])
            self.assertEqual(len(set(CTX.d_input_before) | expected_paths), 60)
            generated_path = CTX.api_root / "ir-evidence-registry.json"
            generated = S.json_read(generated_path)
            self.assertEqual((len(generated["contexts"]), len(generated["observations"]),
                              len(generated["expected_generated_outputs"])), (5, 5, 10))
            self.assertEqual(generated["failures"], [])
            self.assertEqual(generated["expected_generated_outputs"], CTX.generated_expected)
            for path, value in generated["expected_generated_outputs"].items():
                self.assertEqual(S.measure(Path(path)), value["measurement"])
            sources_path = CTX.api_root / "source-identities.json"
            sources = S.json_read(sources_path)
            self.assertEqual(sources["main_commit"], CTX.dstate.identities["main_commit"])
            self.assertEqual(sources["main_tree"], CTX.dstate.identities["main_tree"])
            self.assertEqual(sources["actual_lean_githash"], S.RELEASE)
            self.assertEqual(sources["sources"]["tools/sdga_component.py"]["sha256"],
                             CTX.subject_binding["sha256"])
            commands_path = CTX.api_root / "commands.json"
            commands = S.json_read(commands_path)
            self.assertEqual(len(commands), 10)
            self.assertEqual([value["label"] for value in commands[:4]],
                             ["cleanup-only-preflight", "main_commit", "main_tree", "lean-githash"])
            self.assertEqual(commands[4], CTX.compile_result["command"])
            self.assertEqual(commands[5:], [value["command"] for value in CTX.observations])
            for evidence in (registry_path, generated_path, sources_path, commands_path):
                CTX.evidence(row, evidence)
        with CTX.case("E02", "existing-c08-replay-preserved") as row:
            # Current c45 R0 still has no public API and fails earlier. The future
            # D adapter must retain the actual observe_module return from this
            # EXISTING C08 invocation; no additional replay is allowed.
            replay = CTX.dresult["cli_replay"]
            self.assertEqual(replay["status"], "pass")
            self.assertEqual(replay["receipt"]["helper"], "SDGA.replayOwned")
            self.assertEqual(replay["receipt"]["scientific_effect"], "NONE")
            self.assertEqual(len(replay["receipt"]), 12)
            observation = replay.get("controller_observation")
            self.assertIsInstance(observation, dict, "C08 production observe_module route not retained")
            self.assertEqual(set(observation), {
                "record", "schema", "mode", "request_path", "command",
                "custody_path", "envelope_path"})
            self.assertEqual((observation["record"], observation["schema"], observation["mode"]),
                             ("sdga-controller-observation", 1, "replay"))
            self.assertEqual(observation["command"], replay["command"])
            self.assertEqual(observation["request_path"], str(CTX.dstate.request_path))
            self.assertEqual(S.json_read(observation["custody_path"])["status"], "pass")
            self.assertEqual(S.json_read(observation["envelope_path"])["mode"], "replay")
            CTX.evidence(row, observation["custody_path"])
            CTX.evidence(row, observation["envelope_path"])
            CTX.evidence(row, CTX.dstate.output / "result.json")
        with CTX.case("E02", "unknown-mode-refused") as row:
            before = {str(path): S.measure(path) for path in CTX.api_root.rglob("*")
                      if path.is_file()}
            with self.assertRaisesRegex(ValueError, r"^OBSERVATION_MODE:"):
                self.api("observe_module")(
                    mode="not-a-mode", request_path=CTX.requests[1],
                    output_root=CTX.dstate.owned, dependency_path="", evidence_root=CTX.api_root)
            after = {str(path): S.measure(path) for path in CTX.api_root.rglob("*")
                     if path.is_file()}
            self.assertEqual(after, before, "unknown mode changed acquisition files/registries")
            CTX.evidence(row, CTX.api_root / "commands.json")

    def combination_refusal(self, transcripts):
        with self.assertRaisesRegex(ValueError, r"^INVENTORY_COMBINATION:"):
            self.api("combine_inventory")(transcripts=transcripts, scope=S.SCOPE_E)

    def test_E03_combine_inventory(self):
        """Catch incomplete/duplicate/foreign cohorts and lost negative fixture facts."""
        with CTX.case("E03", "complete-native-composition") as row:
            function = self.api("combine_inventory")
            self.assertEqual(len(CTX.transcripts), 5, "E03_SETUP: five actual inventories required")
            combined = function(transcripts=tuple(CTX.transcripts), scope=S.SCOPE_E)
            self.assertIsInstance(combined, str)
            parsed = CTX.task1.parse_inventory(combined)
            self.assertEqual(parsed["scope_id"], S.SCOPE_E["scope_id"])
            self.assertEqual([value["name"] for value in parsed["modules"]],
                             [CTX.task1.encode_name(("Fixture", leaf)) for leaf in S.LEAVES])
            CTX.combined = parsed
            with (CTX.root / "combined.inventory.jsonl").open("x", encoding="utf-8") as stream:
                stream.write(combined)
            CTX.evidence(row, CTX.root / "combined.inventory.jsonl")
        with CTX.case("E03", "counts-and-axiom-union") as row:
            originals = [CTX.task1.parse_inventory(value) for value in CTX.transcripts]
            self.assertEqual(CTX.combined["modules"],
                             [part for value in originals for part in value["modules"]])
            self.assertEqual(CTX.combined["declarations"],
                             [part for value in originals for part in value["declarations"]])
            self.assertEqual(set(CTX.combined["axioms"]),
                             {name for value in originals for name in value["axioms"]})
            self.assertEqual(len({value["name"] for value in CTX.combined["declarations"]}),
                             len(CTX.combined["declarations"]))
            CTX.evidence(row, CTX.root / "combined.inventory.jsonl")
        with CTX.case("E03", "missing-module-refused"):
            self.combination_refusal(tuple(CTX.transcripts[:-1]))
        with CTX.case("E03", "duplicate-module-refused"):
            self.combination_refusal(tuple(CTX.transcripts[:4] + [CTX.transcripts[0]]))
        with CTX.case("E03", "scope-mismatch-refused"):
            lines = CTX.transcripts[0].splitlines()
            header = json.loads(lines[0])
            header["scope_id"] = "sdga-task-e-wrong-scope"
            altered = json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n"
            self.combination_refusal((altered, *CTX.transcripts[1:]))
        with CTX.case("E03", "malformed-jsonl-refused"):
            self.combination_refusal((CTX.transcripts[0] + '{"record":"extra"}\n',
                                      *CTX.transcripts[1:]))
        with CTX.case("E03", "negative-fixture-findings-preserved") as row:
            classified = CTX.task1.classify_inventory(CTX.combined, S.SCOPE_E)
            self.assertEqual(classified["classification_status"], "BLOCKED")
            self.assertEqual((classified["counts"]["public_targets"],
                              classified["counts"]["proposition_targets"],
                              classified["counts"]["data_targets"]), (3, 1, 2))
            codes = {entry["code"] for entry in classified["findings"]}
            self.assertTrue({"OWNED_AXIOM", "FORBIDDEN_AXIOM", "UNSAFE_REPLAY_SKIP",
                             "PARTIAL_REPLAY_SKIP"}.issubset(codes))
            S.json_write(CTX.root / "combined-classification.json", classified)
            CTX.evidence(row, CTX.root / "combined-classification.json")


def main():
    global CTX
    launcher, binding = S.current_launcher()
    d_exit = launcher.main()  # fixed original main parses the existing --output/-v argv
    state = launcher.STATE
    if d_exit:
        return d_exit
    result = S.json_read(state.output / "result.json")
    S.require(result["methods_run"] == 15 and result["subcase_pass"] == 67
              and result["failures"] == result["errors"] == result["skipped"] == 0
              and result["object_custody"]["status"] == result["ir_evidence"]["status"] == "pass",
              "original D prerequisite not healthy")
    CTX = S.EContext(launcher, state, result)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ControllerContracts)
    actual = unittest.TextTestRunner(verbosity=2).run(suite)
    complete = CTX.finish(actual)
    print(json.dumps({"record": "sdga-e-frontdoor-result", "schema": 1,
                      "status": "pass" if complete else "fail",
                      "result": str(CTX.root / "result.json"), **S.GLOBAL}, sort_keys=True), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
