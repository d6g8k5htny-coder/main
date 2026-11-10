"""SDGA pure dependency parsing and fixed native controller acquisition.

Scientific effect NONE; classification and qualification remain BLOCKED.
"""
from __future__ import annotations

import json


def parse_dependency_headers(
    payload: str, expected_files: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    """Bind strict pinned Lean deps-json rows to files by invocation position."""
    prefix = "DEPENDENCY_HEADERS: "

    def fail(message):
        raise ValueError(prefix + message)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail("duplicate JSON key " + repr(key))
            result[key] = value
        return result

    def invalid_constant(value):
        fail("invalid JSON constant " + value)

    def exact_object(value, keys, label):
        if type(value) is not dict or set(value) != keys:
            fail(label + " has missing, unknown or invalid fields")

    if not isinstance(payload, str):
        fail("payload must be text")
    if not isinstance(expected_files, tuple) or any(
        not isinstance(path, str) or not path for path in expected_files
    ):
        fail("expected_files must be a tuple of nonempty paths")
    if len(set(expected_files)) != len(expected_files):
        fail("duplicate expected file path")
    try:
        document = json.loads(
            payload, object_pairs_hook=unique_object, parse_constant=invalid_constant
        )
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(prefix):
            raise
        fail("invalid JSON: " + str(exc))

    exact_object(document, {"imports"}, "document")
    rows = document["imports"]
    if type(rows) is not list or len(rows) != len(expected_files):
        fail("imports row census does not match expected_files")

    result = {}
    for path, row in zip(expected_files, rows):
        exact_object(row, {"result", "errors"}, "row")
        if type(row["errors"]) is not list or row["errors"]:
            fail("row errors must be an empty array")
        header = row["result"]
        exact_object(header, {"imports", "isModule"}, "result")
        if type(header["isModule"]) is not bool:
            fail("isModule must be a JSON Boolean")
        imports = header["imports"]
        if type(imports) is not list:
            fail("result imports must be an array")
        modules = []
        for entry in imports:
            exact_object(
                entry, {"module", "importAll", "isExported", "isMeta"}, "import"
            )
            module = entry["module"]
            if not isinstance(module, str) or not module:
                fail("import module must be a nonempty string")
            for field in ("importAll", "isExported", "isMeta"):
                if type(entry[field]) is not bool:
                    fail(field + " must be a JSON Boolean")
            modules.append(module)
        result[path] = tuple(modules)
    return result


def topological_modules(
    edges: dict[str, tuple[str, ...]], expected: tuple[str, ...]
) -> tuple[str, ...]:
    """Order a complete local graph, choosing the earliest currently ready vertex."""
    if not isinstance(expected, tuple) or any(
        not isinstance(module, str) or not module for module in expected
    ):
        raise ValueError("MISSING_LOCAL_EDGE: expected must contain nonempty module names")
    vertices = set(expected)
    if len(vertices) != len(expected):
        raise ValueError("DUPLICATE_MODULE: repeated expected vertex")
    if type(edges) is not dict or set(edges) != vertices:
        raise ValueError("MISSING_LOCAL_EDGE: adjacency keys must equal fixed vertices")

    dependencies = {}
    for module in expected:
        imports = edges[module]
        if not isinstance(imports, tuple) or any(
            not isinstance(dependency, str) or dependency not in vertices
            for dependency in imports
        ):
            raise ValueError(
                "MISSING_LOCAL_EDGE: invalid or unknown local dependency for " + repr(module)
            )
        dependencies[module] = set(imports)

    pending = set(expected)
    ordered = []
    while pending:
        ready = next(
            (module for module in expected
             if module in pending and not dependencies[module]),
            None,
        )
        if ready is None:
            raise ValueError("DEPENDENCY_CYCLE: no remaining local vertex is ready")
        pending.remove(ready)
        ordered.append(ready)
        for module in pending:
            dependencies[module].discard(ready)
    return tuple(ordered)


# Acquisition is lazy: importing the pure controller does not create files,
# execute native commands, or initialize a test launcher.
def _runtime():
    import hashlib
    from pathlib import Path
    import sys
    import types
    path = Path(__file__).resolve().parent / "sdga_runtime.py"
    if path.resolve(strict=True) != path:
        raise ValueError("SOURCE_PATH: production runtime alias")
    raw = path.read_bytes()
    name = "sdga_production_runtime_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    cached = sys.modules.get(name)
    if cached is None:
        cached = types.ModuleType(name)
        cached.__file__, cached.__package__ = str(path), None
        # Register before execution to give every captured controller the same
        # production state for this exact source root.
        sys.modules[name] = cached
        try:
            exec(compile(raw, str(path), "exec"), cached.__dict__)
        except BaseException:
            del sys.modules[name]
            raise
        cached._captured_source = raw
    if cached._captured_source != raw or path.read_bytes() != raw:
        raise RuntimeError("SOURCE_DRIFT: production runtime changed after capture")
    return cached


def compile_owned_module(*, source_root, source_file, module, output_root,
                         dependency_path, evidence_root):
    """Compile one fixed fixture, retaining original command and byte custody."""
    R = _runtime()
    source_root, source_file = R.Path(source_root), R.Path(source_file)
    output_root = R.Path(output_root)
    leaves = ("BadDependency", "Safe", "AxiomUse", "Skipped", "OwnedAxiom")
    if (module not in tuple("Fixture." + leaf for leaf in leaves)
            or source_root != R.FIXTURE or source_root.resolve(strict=True) != source_root
            or source_file != source_root / (module.replace(".", "/") + ".lean")
            or source_file.resolve(strict=True) != source_file):
        raise ValueError("COMPILE_SOURCE: fixed canonical fixture required")
    state = R.acquisition_runtime(evidence_root)
    custody_path = state.output / "compile-custody.json"
    registry_path = state.output / "compiled-outputs.json"
    if custody_path.exists() or registry_path.exists():
        raise ValueError("COMPILE_EVIDENCE_REUSE: fixed compile already observed")
    if not output_root.is_absolute() or not output_root.is_relative_to(state.output):
        raise ValueError("COMPILE_OUTPUT: fresh acquisition child required")
    io, _ = R._load_ir_io_subject(state.ir_subject_binding)
    output_root = io["admit_output_root"](output_root, state.source_roots)
    dependency = R.Path(dependency_path)
    if not dependency.is_absolute() or dependency.resolve(strict=True) != dependency:
        raise ValueError("COMPILE_DEPENDENCIES: canonical owned output root required")
    dependency_files = tuple(sorted(path for path in dependency.rglob("*") if path.is_file()))
    if any(path.resolve(strict=True) != path for path in dependency_files):
        raise ValueError("COMPILE_DEPENDENCIES: dependency alias")
    dependencies = R.measure_objects(dependency_files)
    dependency_producers = {str(path): state.producer_for(path) for path in dependency_files}
    source_before = R.measure_objects((source_file,))[str(source_file)]
    target = output_root / (module.replace(".", "/") + ".olean")
    target.parent.mkdir(parents=True)
    outputs = (target, R.Path(str(target) + ".server"), R.Path(str(target) + ".private"),
               target.with_suffix(".ir.sig"), target.with_suffix(".ir"),
               target.with_suffix(".ilean"))
    command = None
    result_outputs = {}
    errors = []
    pending = None
    try:
        command, _, _ = state.command(
            "compile-" + module.replace(".", "-"),
            ["lake", "env", "env", "LEAN_PATH=" + str(output_root) + ":" + str(dependency),
             "lean", "--threads=2", "--root=" + str(source_root),
             "-o", str(target), "-i", str(target.with_suffix(".ilean")), str(source_file)],
            cwd=source_root)
        if command["exit_code"] != 0 or command["launch_error"] or command["timed_out"]:
            raise RuntimeError("COMPILE_NATIVE: retained unsuccessful native outcome")
        result_outputs = R.measure_objects(outputs)
        if result_outputs != command["compile_outputs"]:
            raise RuntimeError("COMPILE_OUTPUT_DRIFT: native producer snapshot differs")
        if any(not value["exists"] for value in result_outputs.values()):
            raise RuntimeError("COMPILE_OUTPUTS: fresh six-file output family incomplete")
        state.compiled_expected = R.copy.deepcopy(result_outputs)
        state.compiled_producers = {name: command["label"] for name in result_outputs}
        for path in outputs:
            path.chmod(0o444)
    except BaseException as error:
        pending = error
        errors.append(type(error).__name__ + ": " + str(error))
        raise
    finally:
        source_after = dependencies_after = after = None
        try:
            source_after = R.measure_objects((source_file,))[str(source_file)]
            dependencies_after = R.measure_objects(dependency_files)
            population_after = tuple(sorted(path for path in dependency.rglob("*") if path.is_file()))
            if source_before != source_after:
                errors.append("SOURCE_DRIFT: compile source changed")
            if dependencies_after != dependencies or population_after != dependency_files:
                errors.append("DEPENDENCY_DRIFT: compile dependency bytes/population changed")
            after = R.measure_objects(outputs)
            if result_outputs and after != result_outputs:
                errors.append("COMPILED_OUTPUT_DRIFT: fresh outputs changed")
        except BaseException as measurement_error:
            errors.append("COMPILE_MEASUREMENT: " + type(measurement_error).__name__
                          + ": " + str(measurement_error))
            if pending is not None:
                R._append_ir_cause(pending, measurement_error)
        if errors:
            state.compiled_failures.extend(errors)
        custody = {
            "record": "sdga-controller-compile-custody", "schema": 1,
            "status": "pass" if not errors and command is not None else "fail",
            "source_path": str(source_file), "source_before": source_before,
            "source_after": source_after, "dependencies_expected": dependencies,
            "dependencies_before": dependencies, "dependencies_after": dependencies_after,
            "dependency_producers": dependency_producers,
            "command_label": None if command is None else command["label"], "errors": errors,
        }
        registry = {
            "record": "sdga-controller-compiled-outputs", "schema": 1,
            "status": "pass" if not errors and result_outputs else "fail",
            "expected": R.copy.deepcopy(result_outputs), "after": after, "errors": errors,
            "producer_labels": R.copy.deepcopy(state.compiled_producers),
        }
        try:
            R.json_write(custody_path, custody)
            R.json_write(registry_path, registry)
            state.flush()
        except BaseException as retention_error:
            state.compiled_failures.append(
                "COMPILE_RETENTION: " + type(retention_error).__name__ + ": " + str(retention_error))
            if pending is not None:
                R._append_ir_cause(pending, retention_error)
                raise pending
            raise
        if errors and pending is None:
            raise RuntimeError("COMPILE_CUSTODY: " + "; ".join(errors))
    return {
        "record": "sdga-controller-compile", "schema": 1, "module": module,
        "source_file": str(source_file), "source_before": source_before,
        "source_after": source_after, "output_root": str(output_root),
        "outputs": R.copy.deepcopy(result_outputs), "command": command,
        "custody_path": str(custody_path), "output_registry_path": str(registry_path),
    }


def observe_module(*, mode, request_path, output_root, dependency_path, evidence_root):
    """Observe the actual native CLI through shared input and IR-output custody."""
    if mode not in ("inventory", "replay"):
        raise ValueError("OBSERVATION_MODE: inventory or replay required")
    R = _runtime()
    state = R.acquisition_runtime(evidence_root)
    request_path, output_root = R.Path(request_path), R.Path(output_root)
    if (not request_path.is_absolute() or request_path.resolve(strict=True) != request_path
            or not request_path.is_relative_to(state.output)
            or not output_root.is_absolute() or output_root.resolve(strict=True) != output_root
            or dependency_path != ""):
        raise ValueError("OBSERVATION_INPUT: canonical fixed owned request required")
    task1 = R.load_module(R.ROOT / "tools/sdga_scope.py", "sdga_controller_observation_task1")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    leaves = ("BadDependency", "Safe", "AxiomUse", "Skipped", "OwnedAxiom")
    leaf = next((leaf for leaf in leaves
                 if request["module"] == task1.encode_name(("Fixture", leaf))), None)
    if leaf is None:
        raise ValueError("OBSERVATION_MODULE: fixed fixture module required")
    base = output_root / "Fixture" / (leaf + ".olean")
    expected = (base, R.Path(str(base) + ".server"), R.Path(str(base) + ".private"),
                base.with_suffix(".ir.sig"), base.with_suffix(".ir"))
    parts = request["olean_parts"] + request["ir_parts"]
    if (tuple(part["path"] for part in parts) != tuple(map(str, expected))
            or tuple(part["level"] for part in parts)
               != ("exported", "server", "private", "ir.sig", "ir")):
        raise ValueError("OBSERVATION_PARTS: exact fixed part order and origin required")
    for path in expected:
        if path.resolve(strict=True) != path or not path.is_file():
            raise ValueError("OBSERVATION_PARTS: canonical actual part required")
        state.producer_for(path)
    state.bind_request(request_path, request)
    label = "cli-safe-replay" if mode == "replay" else "inventory-" + leaf
    command, _, _ = R.Runtime.observe_lean(
        state, label, ["--run", str(R.BRIDGE), mode, str(request_path)],
        requests=(request_path,), search_root=output_root)
    context = state.ir_contexts[-1]
    custody = state.custody / ("%02d-%s.json" %
                                (len(state.object_observations), label))
    return {
        "record": "sdga-controller-observation", "schema": 1,
        "mode": mode, "request_path": str(request_path), "command": command,
        "custody_path": str(custody), "envelope_path": str(context["envelope_path"]),
    }


def combine_inventory(*, transcripts, scope):
    """Compose the complete fixed native cohort without changing its row populations."""
    try:
        R = _runtime()
        task1 = R.load_module(R.ROOT / "tools/sdga_scope.py", "sdga_controller_combination_task1")
        leaves = ("BadDependency", "Safe", "AxiomUse", "Skipped", "OwnedAxiom")
        modules = tuple("Fixture." + leaf for leaf in leaves)
        if (type(transcripts) is not tuple or len(transcripts) != len(modules)
                or type(scope) is not dict
                or tuple(row["module"] for row in scope["modules"]) != modules
                or any(row["path"] != module.replace(".", "/") + ".lean"
                       or row["role"] != "production"
                       for row, module in zip(scope["modules"], modules))):
            raise ValueError("complete ordered fixed module cohort required")
        parsed = tuple(task1.parse_inventory(value) for value in transcripts)
        encoded = tuple(task1.encode_name(("Fixture", leaf)) for leaf in leaves)
        if any(value["scope_id"] != scope["scope_id"]
               or len(value["modules"]) != 1
               or value["modules"][0]["name"] != expected
               for value, expected in zip(parsed, encoded)):
            raise ValueError("scope, module order or module uniqueness differs")
        declarations = [row for value in parsed for row in value["declarations"]]
        if len({row["name"] for row in declarations}) != len(declarations):
            raise ValueError("duplicate declaration across native modules")
        rows = [{"record": "header", "schema": 1, "scope_id": scope["scope_id"]}]
        for value in parsed:
            rows.extend(value["modules"])
            rows.extend(value["declarations"])
        rows.append({"record": "end", "module_count": len(modules),
                     "declaration_count": len(declarations),
                     "axioms": sorted({name for row in declarations for name in row["axioms"]})})
        result = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        task1.parse_inventory(result)
        return result
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        raise ValueError("INVENTORY_COMBINATION: " + str(error)) from error


def main():
    """Fixed fixture CLI: one same-interpreter exec, without importing test classes."""
    import argparse
    import os
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixtures",))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        print("OUTPUT_NOT_FRESH: " + str(args.output), file=sys.stderr)
        return 2
    frontdoor = Path(__file__).resolve().parents[1] / "tests/sdga_component_controller_contract.py"
    if frontdoor.resolve(strict=True) != frontdoor:
        raise ValueError("SOURCE_PATH: fixed fixture frontdoor alias")
    os.execv(sys.executable, [
        sys.executable, "-B", *(["-O"] if sys.flags.optimize else []), "-S",
        str(frontdoor), "--output", str(args.output.absolute()),
        *(["-v"] if args.verbose else []),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
