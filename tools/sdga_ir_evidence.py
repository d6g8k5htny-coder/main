"""Bounded output admission and retention of native SDGA IR evidence.

This module performs no command acquisition or execution. The launcher owns
immutable input/output registries and invokes retention in its finally path.
Validation preserves failed native observations; it never qualifies a replay.
"""
from __future__ import annotations

import hashlib as _hashlib
import json as _json
import os as _os
import re as _re
import stat as _stat
from pathlib import Path

__all__ = ("admit_output_root", "retain_ir_manifest")

_GLOBAL = {
    "scientific_effect": "NONE",
    "qualification": "BLOCKED",
    "classification_status": "BLOCKED",
    "execution_authenticated": False,
}
_FIELDS = ("const_names", "constant_names", "extra_const_names")
_ERROR_STAGES = (
    "admission", "before", "native", "after", "schema", "binding",
    "retention", "final",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, fields):
    _need(type(value) is dict and set(value) == set(fields.split()),
          "closed object fields differ: " + fields)


def _string(value):
    _need(type(value) is str, "string required")
    value.encode("utf-8")
    return value


def _nullable_string(value):
    if value is not None:
        _string(value)


def _nat(value):
    _need(type(value) is int and value >= 0, "nonnegative integer required")


def _boolean(value):
    _need(type(value) is bool, "boolean required")


def _hex(value, length=64):
    _string(value)
    _need(len(value) == length and
          all(char in "0123456789abcdef" for char in value),
          "noncanonical lowercase digest")


def _array(value):
    _need(type(value) is list, "array required")
    return value


def _wire(value, anonymous=False):
    _string(value)
    if value == "":
        _need(anonymous, "anonymous declaration/module identity")
        return ()
    components = []
    for component in value.split("/"):
        if _re.fullmatch(r"s(?:[0-9a-f]{2})*", component):
            text = bytes.fromhex(component[1:]).decode("utf-8")
            _need("s" + text.encode("utf-8").hex() == component,
                  "noncanonical string Name component")
            components.append(text)
        elif _re.fullmatch(r"n(?:0|[1-9][0-9]*)", component):
            # The wire check needs no bounded Python integer conversion.
            components.append(None)
        else:
            raise ValueError("noncanonical Name component")
    return tuple(components)


def _names(value, anonymous=False):
    for name in _array(value):
        _wire(name, anonymous=anonymous)


def _unique_names(value):
    _names(value)
    _need(len(set(value)) == len(value), "duplicate summary identity")


def _path_string(value):
    _string(value)
    path = Path(value)
    _need(path.is_absolute() and str(path) == value and
          not value.startswith("//") and "\0" not in value and
          ".." not in path.parts, "absolute normalized path required")
    return path


def _file_state(value, nullable=False):
    if nullable and value is None:
        return
    _need(type(value) is dict, "FileState object required")
    _boolean(value.get("exists"))
    if value["exists"]:
        _keys(value, "exists bytes sha256")
        _nat(value["bytes"])
        _hex(value["sha256"])
    else:
        _keys(value, "exists")


def _policy(value):
    _keys(value, "olean_extras ir_residue combined")
    _need(all(type(item) is str and item in ("clear", "blocked")
              for item in value.values()), "invalid policy")
    expected = ("blocked" if "blocked" in (
        value["olean_extras"], value["ir_residue"]) else "clear")
    _need(value["combined"] == expected, "combined policy differs")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "duplicate JSON object key: " + key)
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError("non-JSON numeric constant: " + value)


def _decode(raw):
    try:
        return _json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds parser capacity") from error


def _sha(raw):
    return _hashlib.sha256(raw).hexdigest()


def _read(path):
    """Only a genuinely absent leaf is an absent measurement."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"exists": False}, None
    if not _stat.S_ISREG(info.st_mode):
        raise OSError("evidence path is not a regular file: " + str(path))
    raw = path.read_bytes()
    return {"exists": True, "bytes": len(raw), "sha256": _sha(raw)}, raw


def _roots(source_roots):
    _need(type(source_roots) is tuple and bool(source_roots),
          "nonempty source_roots tuple required")
    result = []
    for root in source_roots:
        canonical = Path(root).resolve(strict=True)
        if not _stat.S_ISDIR(canonical.stat().st_mode):
            raise NotADirectoryError(str(canonical))
        result.append(canonical)
    return tuple(result)


def _overlap(path, roots):
    return any(path == root or path.is_relative_to(root) or
               root.is_relative_to(path) for root in roots)


def _canonical_origin(path):
    path = Path(path)
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        # A missing request is measurable failure evidence, never a fresh root.
        return path.parent.resolve(strict=True) / path.name


def admit_output_root(output_root: Path, source_roots: tuple[Path, ...]) -> Path:
    """Create one fresh canonical external directory, preserving refusals."""
    output_root = Path(output_root)
    try:
        output_root.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("OUTPUT_NOT_FRESH: " + str(output_root))
    roots = _roots(source_roots)
    parent = output_root.parent.resolve(strict=True)
    if not _stat.S_ISDIR(parent.stat().st_mode):
        raise NotADirectoryError(str(parent))
    canonical = parent / output_root.name
    if _overlap(canonical, roots):
        raise ValueError("OUTPUT_IN_SOURCE: " + str(canonical))
    # No exist_ok, parent creation, retry, rename, or deletion.
    canonical.mkdir()
    resolved = output_root.resolve(strict=True)
    if resolved != canonical or not _stat.S_ISDIR(resolved.stat().st_mode):
        raise OSError("OUTPUT_CANONICAL_DRIFT: " + str(output_root))
    if _overlap(resolved, roots):
        raise ValueError("OUTPUT_IN_SOURCE: " + str(resolved))
    return resolved


def _request(value):
    """Match native structural parsing; level-order refusal stays native."""
    _keys(value, "schema scope_id profile_sha256 module olean_parts ir_parts")
    _need(type(value["schema"]) is int and value["schema"] == 1,
          "unsupported request schema")
    _string(value["scope_id"])
    _need(bool(value["scope_id"]), "empty scope identity")
    _hex(value["profile_sha256"])
    module = _wire(value["module"])
    _need(all(type(component) is str and component and
              component not in (".", "..") and
              not any(char in component for char in ("/", "\\", "\0"))
              for component in module), "unsafe module import Name")
    for family, allowed in (
            ("olean_parts", ("exported", "server", "private")),
            ("ir_parts", ("ir.sig", "ir"))):
        for part in _array(value[family]):
            _keys(part, "level path sha256")
            _string(part["level"])
            _need(part["level"] in allowed, "unknown object part level")
            _path_string(part["path"])
            _need(part["path"] != "/", "object path requires a leaf")
            _hex(part["sha256"])
    return value


def _manifest(value, request, origin, mode):
    _keys(value, "record schema mode request_path scope_id profile_sha256 module "
          "hash_semantics completeness olean_parts olean_extra_union ir_parts "
          "ir_occurrences ci_backed_names unexplained_residue policy "
          "scientific_effect qualification classification_status "
          "execution_authenticated")
    _need(value["record"] == "sdga-ir-manifest", "wrong native record")
    _need(type(value["schema"]) is int and value["schema"] == 1,
          "unsupported native schema")
    _need(value["mode"] == mode, "native mode differs")
    _need(value["request_path"] == str(origin), "native request origin differs")
    _path_string(value["request_path"])
    _string(value["scope_id"])
    _hex(value["profile_sha256"])
    _wire(value["module"])
    for key in ("scope_id", "profile_sha256", "module"):
        _need(value[key] == request[key], "native request identity differs: " + key)
    _need(value["hash_semantics"] == "request-declared",
          "wrong native hash semantics")
    _need(value["completeness"] == "complete", "incomplete native document")
    _global(value)
    for family in ("olean_parts", "ir_parts"):
        parts = _array(value[family])
        requested = request[family]
        _need(len(parts) == len(requested), "native part count differs")
        for part, declared in zip(parts, requested):
            _keys(part, "level path declared_sha256 is_module const_names "
                  "constant_names extra_const_names extension_entries")
            _string(part["level"])
            _path_string(part["path"])
            _hex(part["declared_sha256"])
            _need(part["level"] == declared["level"] and
                  part["path"] == declared["path"] and
                  part["declared_sha256"] == declared["sha256"],
                  "native part binding differs")
            _boolean(part["is_module"])
            for field in _FIELDS:
                _names(part[field])
            for extension in _array(part["extension_entries"]):
                _keys(extension, "name entry_count")
                _wire(extension["name"], anonymous=True)
                _nat(extension["entry_count"])
    olean = value["olean_parts"]
    ir = value["ir_parts"]
    levels = [part["level"] for part in olean]
    _need(levels in (["exported"], ["exported", "server", "private"]),
          "native OLean level vector differs")
    _need([part["level"] for part in ir] in ([], ["ir.sig", "ir"]),
          "native IR level vector differs")
    if olean[0]["is_module"]:
        _need(levels == ["exported", "server", "private"] and len(ir) == 2,
              "module-system native parts incomplete")
    else:
        _need(levels == ["exported"], "legacy native parts differ")
    owned = set(olean[-1]["constant_names"])
    for part in olean:
        _need(part["const_names"] == part["constant_names"],
              "OLean parallel constant arrays differ")
        _need(len(set(part["const_names"])) == len(part["const_names"]),
              "duplicate OLean declaration identity")
        _need(set(part["const_names"]).issubset(owned),
              "OLean final-owned population differs")
    extras = list(dict.fromkeys(
        name for part in olean for name in part["extra_const_names"]))
    _unique_names(value["olean_extra_union"])
    _need(value["olean_extra_union"] == extras, "OLean extra union differs")
    occurrences = []
    backed, residue = [], []
    for part_index, part in enumerate(ir):
        for field in _FIELDS:
            for index, name in enumerate(part[field]):
                ci_backed = name in owned
                occurrences.append({
                    "part_index": part_index, "field": field, "index": index,
                    "name": name, "ci_backed": ci_backed,
                })
                population = backed if ci_backed else residue
                if name not in population:
                    population.append(name)
    for occurrence in _array(value["ir_occurrences"]):
        _keys(occurrence, "part_index field index name ci_backed")
        _nat(occurrence["part_index"])
        _nat(occurrence["index"])
        _string(occurrence["field"])
        _need(occurrence["field"] in _FIELDS, "unknown IR occurrence field")
        _wire(occurrence["name"])
        _boolean(occurrence["ci_backed"])
    _need(value["ir_occurrences"] == occurrences,
          "IR occurrence order, coordinates, or CI partition differs")
    _unique_names(value["ci_backed_names"])
    _unique_names(value["unexplained_residue"])
    _need(value["ci_backed_names"] == backed and
          value["unexplained_residue"] == residue, "IR summaries differ")
    _policy(value["policy"])
    _need(value["policy"] == {
        "olean_extras": "blocked" if extras else "clear",
        "ir_residue": "blocked" if residue else "clear",
        "combined": "blocked" if extras or residue else "clear",
    }, "native policy does not match raw evidence")
    return value


def _global(value):
    for key, wanted in _GLOBAL.items():
        _need(type(value[key]) is type(wanted) and value[key] == wanted,
              "global evidence status differs: " + key)


def _stream(value):
    _keys(value, "path bytes sha256")
    _path_string(value["path"])
    _nat(value["bytes"])
    _hex(value["sha256"])


def _outcome(value):
    if value is None:
        return
    _keys(value, "label argv cwd exit_code timed_out launch_error interruption "
          "cleanup_confirmed stdout stderr")
    _string(value["label"])
    _array(value["argv"])
    for arg in value["argv"]:
        _string(arg)
    _path_string(value["cwd"])
    _need(value["exit_code"] is None or type(value["exit_code"]) is int,
          "native exit code must be an integer or null")
    _boolean(value["timed_out"])
    _nullable_string(value["launch_error"])
    _nullable_string(value["interruption"])
    _need(value["cleanup_confirmed"] is None or
          type(value["cleanup_confirmed"]) is bool,
          "native cleanup confirmation must be boolean or null")
    _stream(value["stdout"])
    _stream(value["stderr"])


def _source(value):
    _keys(value, "path exists bytes sha256 git_blob")
    name = _string(value["path"])
    _need(name and not Path(name).is_absolute() and str(Path(name)) == name
          and ".." not in Path(name).parts,
          "repository-relative source binding required")
    _boolean(value["exists"])
    if value["exists"]:
        if value["bytes"] is not None:
            _nat(value["bytes"])
        if value["sha256"] is not None:
            _hex(value["sha256"])
        if value["git_blob"] is not None:
            _hex(value["git_blob"], 40)
    else:
        _need(all(value[key] is None for key in ("bytes", "sha256", "git_blob")),
              "absent source has metadata")


def _envelope(value):
    _keys(value, "record schema mode request_path request_measurement manifest_path "
          "manifest_before manifest_after manifest_completeness registration "
          "producer_label native_outcome observation_error input_custody "
          "source_bindings policy validation_status errors scientific_effect "
          "qualification classification_status execution_authenticated")
    _need(value["record"] == "sdga-ir-envelope", "wrong envelope record")
    _need(type(value["schema"]) is int and value["schema"] == 1,
          "unsupported envelope schema")
    _need(type(value["mode"]) is str and value["mode"] in ("inventory", "replay"),
          "invalid envelope mode")
    _path_string(value["request_path"])
    _path_string(value["manifest_path"])
    for key in ("request_measurement", "manifest_before", "manifest_after"):
        _file_state(value[key], nullable=True)
    _need(type(value["manifest_completeness"]) is str and
          value["manifest_completeness"] in
          ("complete", "missing", "invalid", "unreadable"),
          "invalid manifest completeness")
    _need(type(value["registration"]) is str and value["registration"] in
          ("new", "preexisting", "none"), "invalid registration")
    _nullable_string(value["producer_label"])
    _nullable_string(value["observation_error"])
    _outcome(value["native_outcome"])
    custody = value["input_custody"]
    _keys(custody, "path status sha256")
    _path_string(custody["path"])
    _need(type(custody["status"]) is str and
          custody["status"] in ("pass", "fail", "not_produced"),
          "invalid custody reference")
    if custody["sha256"] is not None:
        _hex(custody["sha256"])
    names = []
    for source in _array(value["source_bindings"]):
        _source(source)
        names.append(source["path"])
    _need(names == sorted(set(names)), "source bindings not sorted/unique")
    if value["policy"] is not None:
        _policy(value["policy"])
    _need(type(value["validation_status"]) is str and
          value["validation_status"] in ("valid", "invalid"),
          "invalid validation status")
    for error in _array(value["errors"]):
        _keys(error, "stage code message")
        _need(type(error["stage"]) is str and error["stage"] in _ERROR_STAGES,
              "invalid error stage")
        _string(error["code"])
        _string(error["message"])
    _global(value)
    return value


def _error(errors, stage, code, error):
    errors.append({"stage": stage, "code": code,
                   "message": str(error)})


def _normalized_command(command, label):
    _need(type(command) is dict, "actual command object required")
    _need(command["label"] == label, "command label differs from observation")
    cleanup = command.get("cleanup")
    _need(cleanup is None or type(cleanup) is dict,
          "actual command cleanup object required")
    value = {
        "label": command["label"], "argv": command["argv"][:],
        "cwd": command["cwd"], "exit_code": command["exit_code"],
        "timed_out": command["timed_out"], "launch_error": command["launch_error"],
        "interruption": command.get("interruption"),
        "cleanup_confirmed": None if cleanup is None else cleanup["confirmed"],
        "stdout": {
            "path": command["stdout_path"], "bytes": command["stdout_bytes"],
            "sha256": command["stdout_sha256"],
        },
        "stderr": {
            "path": command["stderr_path"], "bytes": command["stderr_bytes"],
            "sha256": command["stderr_sha256"],
        },
    }
    _outcome(value)
    return value


def _sources(bindings, errors):
    _need(type(bindings) is dict, "captured source map required")
    result = []
    for path in sorted(bindings):
        value = bindings[path]
        _need(type(value) is dict and "exists" in value and
              set(value).issubset({"exists", "bytes", "sha256", "git_blob"}),
              "captured source fields differ")
        source = {
            "path": path, "exists": value["exists"],
            "bytes": value.get("bytes"), "sha256": value.get("sha256"),
            "git_blob": value.get("git_blob"),
        }
        _source(source)
        result.append(source)
        if not source["exists"] or any(source[key] is None for key in
                                     ("bytes", "sha256", "git_blob")):
            _error(errors, "binding", "SOURCE_UNMEASURED", path)
    _need(bool(result), "empty source binding map")
    return result


def _custody(value, label, outcome):
    # This input is the existing launcher's finalized custody record. Its
    # schema is independent of the closed CustodyRef stored in the envelope.
    _keys(value, "record expected before after drift status precheck_error "
          "observation_error measurement_errors native_label native_outcome")
    _need(value["record"] == "object-custody", "wrong input custody record")
    _need(value["native_label"] == label, "custody native label differs")
    _need(type(value["status"]) is str and value["status"] in ("pass", "fail"),
          "invalid input custody status")
    for field in ("expected", "before", "after"):
        rows = value[field]
        _need(type(rows) is dict and bool(rows), "unavailable input " + field)
        for name, state in rows.items():
            _path_string(name)
            _file_state(state)
    _need(value["before"] == value["expected"] == value["after"],
          "immutable before/expected/after input bindings differ")
    _need(type(value["drift"]) is list and not value["drift"],
          "input custody reports drift")
    _need(type(value["measurement_errors"]) is list and
          not value["measurement_errors"], "input custody measurement failed")
    _nullable_string(value["precheck_error"])
    _nullable_string(value["observation_error"])
    _need(value["status"] == "pass" and value["precheck_error"] is None and
          value["observation_error"] is None, "input custody did not pass")
    if outcome is None:
        _need(value["native_outcome"] is None,
              "custody has an unrecovered native outcome")
    else:
        summary = value["native_outcome"]
        _keys(summary, "exit_code timed_out launch_error stdout_sha256 stderr_sha256")
        _need(type(summary["exit_code"]) is type(outcome["exit_code"]) and
              summary["exit_code"] == outcome["exit_code"] and
              type(summary["timed_out"]) is bool and
              summary["timed_out"] == outcome["timed_out"] and
              summary["launch_error"] == outcome["launch_error"] and
              summary["stdout_sha256"] == outcome["stdout"]["sha256"] and
              summary["stderr_sha256"] == outcome["stderr"]["sha256"],
              "custody original native outcome differs")
    return value["expected"]


def _envelope_root(output_root, source_roots):
    roots = _roots(source_roots)
    path = Path(output_root)
    info = path.lstat()
    if not _stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(str(path))
    canonical = path.resolve(strict=True)
    if canonical != Path(_os.path.abspath(path)):
        raise ValueError("OUTPUT_CANONICAL_DRIFT: " + str(path))
    if _overlap(canonical, roots):
        raise ValueError("OUTPUT_IN_SOURCE: " + str(canonical))
    evidence = canonical / "ir-evidence"
    try:
        evidence.mkdir()
    except FileExistsError:
        if not _stat.S_ISDIR(evidence.lstat().st_mode):
            raise NotADirectoryError(str(evidence))
    if evidence.resolve(strict=True) != evidence:
        raise OSError("IR_EVIDENCE_CANONICAL_DRIFT: " + str(evidence))
    return evidence


def _retained_write(path, value):
    # Validate our complete closed output before exclusive creation. Readback
    # uses a separate read, including duplicate/key/type validation again.
    _envelope(value)
    raw = (_json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        written = stream.write(raw)
        if written != len(raw):
            raise OSError("IR_ENVELOPE_SHORT_WRITE: " + str(path))
        stream.flush()
    measured, actual = _read(path)
    if actual != raw or measured != {
            "exists": True, "bytes": len(raw), "sha256": _sha(raw)}:
        raise OSError("IR_ENVELOPE_READBACK: " + str(path))
    recovered = _envelope(_decode(actual))
    if recovered != value:
        raise OSError("IR_ENVELOPE_READBACK_SCHEMA: " + str(path))
    return recovered


def retain_ir_manifest(*, mode, request_path, output_root, source_roots,
                       native_outcome, custody_record, source_bindings,
                       observation_error) -> dict:
    """Retain one actual native observation without altering its sidecar.

    native_outcome is the launcher's transport, not the serialized outcome:
    label, ordinal, command, manifest_before, manifest_before_error,
    producer_label. The launcher owns registration and all final drift checks.
    Evidence validation errors return an invalid closed envelope. An unsafe
    destination, unusable observation identity, or envelope I/O failure raises.
    """
    _need(type(mode) is str and mode in ("inventory", "replay"), "invalid mode")
    _keys(native_outcome, "label ordinal command manifest_before "
          "manifest_before_error producer_label")
    label = _string(native_outcome["label"])
    _need(_re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) is not None,
          "unsafe observation label")
    ordinal = native_outcome["ordinal"]
    _need(type(ordinal) is int and ordinal >= 1, "positive observation ordinal required")
    evidence_root = _envelope_root(output_root, source_roots)
    destination = evidence_root / (f"{ordinal:02d}-{label}.json")
    errors = []
    try:
        origin = _canonical_origin(request_path)
    except (OSError, ValueError) as error:
        origin = Path(_os.path.abspath(request_path))
        _error(errors, "binding", "REQUEST_PATH", error)
    manifest_path = Path(str(origin) + "." + mode + ".ir-manifest.json")
    try:
        custody_path = _canonical_origin(custody_record)
    except (OSError, ValueError) as error:
        custody_path = Path(_os.path.abspath(custody_record))
        _error(errors, "binding", "CUSTODY_PATH", error)
    if observation_error is not None and type(observation_error) is not str:
        _error(errors, "binding", "OBSERVATION_ERROR_TYPE",
               "formatted original exception string required")
        observation_error = str(observation_error)
    value = {
        "record": "sdga-ir-envelope", "schema": 1, "mode": mode,
        "request_path": str(origin), "request_measurement": None,
        "manifest_path": str(manifest_path),
        "manifest_before": None, "manifest_after": None,
        "manifest_completeness": "unreadable", "registration": "none",
        "producer_label": None, "native_outcome": None,
        "observation_error": observation_error,
        "input_custody": {"path": str(custody_path), "status": "not_produced",
                         "sha256": None},
        "source_bindings": [], "policy": None, "validation_status": "invalid",
        "errors": errors, **_GLOBAL,
    }

    before = native_outcome["manifest_before"]
    try:
        _file_state(before, nullable=True)
        value["manifest_before"] = None if before is None else before.copy()
    except (ValueError, UnicodeError) as error:
        _error(errors, "before", "MANIFEST_BEFORE_SCHEMA", error)
    before_error = native_outcome["manifest_before_error"]
    try:
        _nullable_string(before_error)
        if before_error is not None:
            _error(errors, "before", "MANIFEST_BEFORE_READ", before_error)
        elif before is None:
            _error(errors, "before", "MANIFEST_BEFORE_UNAVAILABLE",
                   "no pre-command sidecar measurement")
    except (ValueError, UnicodeError) as error:
        _error(errors, "before", "MANIFEST_BEFORE_ERROR_SCHEMA", error)
    previous_producer = native_outcome["producer_label"]
    try:
        _nullable_string(previous_producer)
        if previous_producer is not None:
            _need(_re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",
                               previous_producer) is not None,
                  "unsafe previous producer label")
    except (ValueError, UnicodeError) as error:
        _error(errors, "binding", "PRODUCER_LABEL", error)
        previous_producer = None

    stdout = stderr = None
    command = native_outcome["command"]
    if command is None:
        _error(errors, "binding", "NATIVE_OUTCOME_UNAVAILABLE",
               "no actual completed command record recovered")
    else:
        try:
            value["native_outcome"] = _normalized_command(command, label)
        except (KeyError, TypeError, ValueError, UnicodeError) as error:
            _error(errors, "binding", "NATIVE_OUTCOME_SCHEMA", error)
    outcome = value["native_outcome"]
    if outcome is not None:
        for field in ("stdout", "stderr"):
            stream = outcome[field]
            try:
                measured, raw = _read(Path(stream["path"]))
                _need(measured == {"exists": True, "bytes": stream["bytes"],
                                   "sha256": stream["sha256"]},
                      "original stream bytes differ: " + field)
                if field == "stdout":
                    stdout = raw
                else:
                    stderr = raw
            except (OSError, ValueError) as error:
                _error(errors, "binding", "NATIVE_STREAM_" + field.upper(), error)
        if (outcome["exit_code"] is not None and
                outcome["cleanup_confirmed"] is not True):
            _error(errors, "binding", "NATIVE_CLEANUP_UNCONFIRMED",
                   "actual process cleanup is not confirmed")

    native_code = None
    native_message = None
    if outcome is not None:
        if outcome["timed_out"]:
            native_code, native_message = "NATIVE_TIMEOUT", "native command timed out"
        elif outcome["launch_error"] is not None:
            native_code, native_message = "NATIVE_LAUNCH", outcome["launch_error"]
        elif outcome["interruption"] is not None:
            native_code, native_message = "NATIVE_INTERRUPTION", outcome["interruption"]
        elif outcome["exit_code"] != 0:
            native_code = "NATIVE_EXIT"
            native_message = "native exit code: " + str(outcome["exit_code"])
            if stderr is not None:
                try:
                    text = stderr.decode("utf-8").strip()
                    native_message = text or native_message
                    match = _re.fullmatch(r"SDGA_ERROR ([A-Z][A-Z0-9_]*):[^\r\n]*",
                                          text)
                    if match is not None:
                        native_code = match.group(1)
                except UnicodeError:
                    native_message = "native stderr is not UTF-8; original bytes retained"
        elif outcome["exit_code"] == 0 and outcome["launch_error"] is None:
            if stderr:
                _error(errors, "binding", "NATIVE_SUCCESS_STDERR",
                       "successful native command produced stderr")
    if native_code is not None:
        _error(errors, "native", native_code, native_message)

    expected = None
    try:
        custody_state, custody_raw = _read(custody_path)
        if not custody_state["exists"]:
            _error(errors, "binding", "CUSTODY_NOT_PRODUCED",
                   "finalized input custody record missing")
        else:
            # Keep the available actual digest even when JSON/schema fails.
            value["input_custody"] = {
                "path": str(custody_path), "status": "fail",
                "sha256": custody_state["sha256"],
            }
            custody_value = _decode(custody_raw)
            status = (custody_value.get("status") if
                      type(custody_value) is dict else None)
            if status in ("pass", "fail"):
                value["input_custody"]["status"] = status
            expected = _custody(custody_value, label, outcome)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        if value["input_custody"]["status"] == "not_produced":
            value["input_custody"]["status"] = "fail"
        _error(errors, "binding", "CUSTODY_INVALID", error)
    try:
        value["source_bindings"] = _sources(source_bindings, errors)
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        _error(errors, "binding", "SOURCE_BINDINGS_INVALID", error)

    request_raw = None
    try:
        value["request_measurement"], request_raw = _read(origin)
    except (OSError, ValueError) as error:
        _error(errors, "binding", "REQUEST_READ", error)
    manifest_raw = None
    try:
        value["manifest_after"], manifest_raw = _read(manifest_path)
        value["manifest_completeness"] = (
            "missing" if manifest_raw is None else "invalid")
    except (OSError, ValueError) as error:
        _error(errors, "after", "MANIFEST_READ", error)

    request_schema_refusal = (
        outcome is not None and type(outcome["exit_code"]) is int and
        outcome["exit_code"] != 0 and not outcome["timed_out"] and
        outcome["launch_error"] is None and outcome["interruption"] is None and
        native_code == "REQUEST_SCHEMA" and stdout == b"" and
        value["manifest_after"] == {"exists": False} and
        value["manifest_before"] == {"exists": False} and
        previous_producer is None)
    request = None
    early_refusal = False
    try:
        _need(request_raw is not None, "request bytes are unavailable")
        request = _request(_decode(request_raw))
    except (ValueError, KeyError, TypeError, UnicodeError) as error:
        # A native code alone does not establish phase. Only available bytes
        # independently rejected by the closed request parser prove this
        # REQUEST_SCHEMA refusal occurred before module observation.
        early_refusal = request_schema_refusal and request_raw is not None
        if not early_refusal:
            _error(errors, "schema", "REQUEST_SCHEMA_INVALID", error)

    measured_inputs = {}
    if expected is not None:
        for name, original in expected.items():
            try:
                _need(_canonical_origin(Path(name)) == Path(name),
                      "input canonical path differs: " + name)
                actual, _ = _read(Path(name))
                measured_inputs[name] = actual
                _need(actual == original, "immutable input drift: " + name)
            except (OSError, ValueError) as error:
                _error(errors, "binding", "INPUT_MEASUREMENT", error)
        if (str(origin) not in expected or
                value["request_measurement"] != expected.get(str(origin))):
            _error(errors, "binding", "REQUEST_CUSTODY_BINDING",
                   "actual request differs from immutable custody")
    if request is not None:
        for part in request["olean_parts"] + request["ir_parts"]:
            name = part["path"]
            try:
                _need(_canonical_origin(Path(name)) == Path(name),
                      "part canonical path differs: " + name)
                actual, _ = _read(Path(name))
                _need(expected is not None and name in expected,
                      "request part missing from immutable custody: " + name)
                _need(actual == expected[name] == measured_inputs.get(name),
                      "part measurement differs from immutable custody: " + name)
                if actual["exists"]:
                    _need(actual["sha256"] == part["sha256"],
                          "declared request part digest differs: " + name)
                else:
                    _need(early_refusal,
                          "native observation requires a present input part: " + name)
            except (OSError, ValueError) as error:
                _error(errors, "binding", "PART_CUSTODY_BINDING", error)

    manifest_valid = False
    if manifest_raw is not None:
        try:
            _need(request is not None, "valid request required to reconcile manifest")
            manifest = _manifest(_decode(manifest_raw), request, origin, mode)
            value["policy"] = dict(manifest["policy"])
            value["manifest_completeness"] = "complete"
            manifest_valid = True
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            _error(errors, "schema", "MANIFEST_SCHEMA_INVALID", error)
    elif value["manifest_completeness"] == "missing" and not early_refusal:
        _error(errors, "after", "MANIFEST_REQUIRED",
               "complete matching native sidecar required for this observation")

    after = value["manifest_after"]
    before = value["manifest_before"]
    if before is not None:
        if before["exists"]:
            if after != before:
                _error(errors, "after", "PREEXISTING_MANIFEST_DRIFT",
                       "preexisting sidecar bytes changed or became unavailable")
            if previous_producer is None:
                _error(errors, "binding", "PREEXISTING_PRODUCER_REQUIRED",
                       "preexisting sidecar lacks its immutable producer binding")
            if manifest_valid and after == before and previous_producer is not None:
                value["registration"] = "preexisting"
                value["producer_label"] = previous_producer
            if outcome is not None and outcome["exit_code"] == 0:
                _error(errors, "binding", "PREEXISTING_NATIVE_SUCCESS",
                       "exclusive native sidecar writer cannot succeed on an existing leaf")
        elif previous_producer is not None:
            _error(errors, "binding", "ABSENT_MANIFEST_PRODUCER",
                   "previous producer supplied for an absent pre-command sidecar")
        elif manifest_valid:
            value["registration"] = "new"
            value["producer_label"] = label
    if native_code == "SIDECAR_COLLISION":
        if not (before is not None and before["exists"] and after == before and
                manifest_valid and previous_producer is not None):
            _error(errors, "binding", "COLLISION_BINDING_INVALID",
                   "native collision lacks unchanged validated original sidecar")
    invalid = any(error["stage"] != "native" for error in errors)
    value["validation_status"] = "invalid" if invalid else "valid"
    if invalid and value["registration"] == "new":
        # Preserve the measured failed bytes; do not assert validated new
        # registration. The launcher still records/checks all observed outputs.
        value["registration"] = "none"
        value["producer_label"] = None
    return _retained_write(destination, value)
