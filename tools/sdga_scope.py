"""Pure SDGA source/protocol observations; this module never certifies execution.

No process, Git, network, filesystem-read/write or Lean execution API is provided.
Caller-provided bindings are declarations, not independent provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
import tomllib

__all__ = [
    "ScopeError", "NameProtocolError", "InventoryProtocolError",
    "extract_scope", "verify_source_files", "encode_name", "decode_name",
    "parse_inventory", "classify_inventory", "classify_receipt",
]

_BLOCKERS = (
    "EXACT_TYPE_SEMANTICS_UNRESOLVED",
    "FULL_DECLARATION_ROLES_UNRESOLVED",
    "EXECUTION_NOT_OBSERVED",
    "KERNEL_COVERAGE_UNESTABLISHED",
)
_INPUTS = {"lakefile.toml", "lake-manifest.json", "lean-toolchain", "SOURCE_MANIFEST.json"}
_LEAN = b"leanprover/lean4:v4.34.1\n"
_LEAN_COMMIT = "5045d0056413266e57c625dcd7c365b10e377c52"
_HASH = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_MODULE_COMPONENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_KINDS = {"theorem", "definition", "opaque", "axiom", "inductive",
          "constructor", "recursor", "quotient"}


class _ContractError(ValueError):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(code + (": " + message if message else ""))


class ScopeError(_ContractError):
    """Source bytes or a declared source scope do not match their bindings."""


class NameProtocolError(_ContractError):
    """A lossless Name is malformed or noncanonical."""


class InventoryProtocolError(_ContractError):
    """No inventory has been accepted from this malformed transcript."""


def _need(condition, error, code, note):
    if not condition:
        raise error(code, note)


def _nat(value):
    return type(value) is int and value >= 0


def _hex(value, pattern=_HASH):
    return type(value) is str and pattern.fullmatch(value) is not None


def _blocked():
    return {
        "scientific_effect": "NONE",
        "qualification": "BLOCKED",
        "classification_status": "BLOCKED",
        "execution_authenticated": False,
        "blockers": list(_BLOCKERS),
    }


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _finite_float(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    return value


def _constant(raw):
    raise ValueError("nonfinite JSON constant: " + raw)


def _json(text, error, code):
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_float=_finite_float,
                          parse_constant=_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise error(code, str(exc)) from exc


def _source_path(value):
    _need(type(value) is str and bool(value), ScopeError, "PATH_SCOPE", "empty source path")
    fields = value.split("/")
    _need(not value.startswith("/") and "\\" not in value and "\0" not in value
          and all(field not in ("", ".", "..") for field in fields)
          and value.endswith(".lean"),
          ScopeError, "PATH_SCOPE", "unsafe source path")
    return value


def _module_name(value):
    _need(type(value) is str and bool(value)
          and all(_MODULE_COMPONENT.fullmatch(field) for field in value.split(".")),
          ScopeError, "MODULE_SCOPE", "invalid module name")
    return value


def _target_name(value):
    _need(type(value) is str and bool(value), ScopeError, "TARGET_SCOPE", "empty target")
    fields = value.split(".")
    _need(all(field and (field[0].isalpha() or field[0] == "_")
              and all(c.isalnum() or c in "_'" for c in field) for field in fields),
          ScopeError, "TARGET_SCOPE", "unsupported dotted target name")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ScopeError("TARGET_SCOPE", "invalid target Unicode") from exc
    return tuple(fields)


def extract_scope(raw_inputs: dict[str, bytes], profile: dict,
                  observed_identity: dict) -> dict:
    """Check declared raw bindings and independently extract the frozen source scope."""
    _need(type(raw_inputs) is dict and set(raw_inputs) == _INPUTS
          and type(profile) is dict and type(profile.get("inputs")) is dict
          and set(profile["inputs"]) == _INPUTS,
          ScopeError, "INPUT_BINDING", "exactly four bound inputs are required")
    for name in sorted(_INPUTS):
        raw, binding = raw_inputs[name], profile["inputs"][name]
        _need(type(raw) is bytes and type(binding) is dict,
              ScopeError, "INPUT_BINDING", name)
        _need(_nat(binding.get("bytes")) and len(raw) == binding["bytes"]
              and _hex(binding.get("sha256")) and _hex(binding.get("git_blob"), _GIT),
              ScopeError, "INPUT_BINDING", name)
        digest = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        _need(digest == binding["sha256"] and blob == binding["git_blob"],
              ScopeError, "INPUT_BINDING", name)
    identity = profile.get("source")
    _need(type(identity) is dict and type(observed_identity) is dict
          and observed_identity == identity,
          ScopeError, "SOURCE_IDENTITY", "source identity differs")
    _need(raw_inputs["lean-toolchain"] == _LEAN
          and profile.get("lean_release_commit") == _LEAN_COMMIT,
          ScopeError, "TOOLCHAIN_SCOPE", "pinned Lean differs")
    try:
        config = tomllib.loads(raw_inputs["lakefile.toml"].decode("utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ScopeError("MODULE_SCOPE", "invalid TOML") from exc
    manifest = _json(raw_inputs["SOURCE_MANIFEST.json"], ScopeError, "JSON_PROTOCOL")
    lock = _json(raw_inputs["lake-manifest.json"], ScopeError, "JSON_PROTOCOL")
    _need(type(manifest) is dict and type(lock) is dict,
          ScopeError, "JSON_PROTOCOL", "JSON object required")

    packages = lock.get("packages")
    _need(type(packages) is list and all(type(p) is dict for p in packages),
          ScopeError, "DEPENDENCY_SCOPE", "missing package lock")
    pins = [{"name": p.get("name"), "rev": p.get("rev")} for p in packages]
    _need(pins == profile.get("dependency_pins") and len(pins) == 9
          and all(type(p["name"]) is str and _hex(p["rev"], _GIT) for p in pins)
          and len({p["name"] for p in pins}) == len(pins),
          ScopeError, "DEPENDENCY_SCOPE", "dependency pins differ")
    mathlib = next((p["rev"] for p in pins if p["name"] == "mathlib"), None)
    required = config.get("require")
    _need(type(required) is list and len(required) == 1
          and type(required[0]) is dict and required[0].get("name") == "mathlib"
          and required[0].get("rev") == mathlib
          and required[0].get("git") == "https://github.com/leanprover-community/mathlib4.git",
          ScopeError, "DEPENDENCY_SCOPE", "Mathlib requirement differs")

    sources = manifest.get("source_modules")
    _need(type(sources) is list, ScopeError, "MODULE_SCOPE", "missing source modules")
    for path in sources:
        _source_path(path)
    _need(config.get("name") == "source_derivative_gaussian",
          ScopeError, "MODULE_SCOPE", "package name differs")
    defaults, libraries = config.get("defaultTargets"), config.get("lean_lib")
    _need(type(defaults) is list and type(libraries) is list
          and all(type(lib) is dict for lib in libraries),
          ScopeError, "MODULE_SCOPE", "missing library registration")
    for name in defaults:
        _module_name(name)
    _need(len(defaults) == 67 and len(set(defaults)) == len(defaults),
          ScopeError, "MODULE_SCOPE", "production roots differ")
    by_name = {}
    for lib in libraries:
        name = _module_name(lib.get("name"))
        _need(name not in by_name, ScopeError, "MODULE_SCOPE", "duplicate library")
        by_name[name] = lib
    _need(set(by_name) == set(defaults) | {"SourceDerivativeInputs", "SourceElderInputs"},
          ScopeError, "MODULE_SCOPE", "unexpected library registration")
    rows = []
    for name in defaults:
        lib = by_name[name]
        _need(lib.get("srcDir", ".") == "." and lib.get("roots", [name]) == [name],
              ScopeError, "MODULE_SCOPE", "production library changed")
        rows.append({"module": name, "path": name.replace(".", "/") + ".lean",
                     "role": "production"})
    production_paths = [row["path"] for row in rows]
    _need(sources == production_paths, ScopeError, "MODULE_SCOPE",
          "source modules differ from production roots")
    for library, directory, count in (
        ("SourceDerivativeInputs", "inputs", 25),
        ("SourceElderInputs", "inputs/elder", 7),
    ):
        lib = by_name[library]
        roots = lib.get("roots")
        _need(lib.get("srcDir") == directory and type(roots) is list and len(roots) == count,
              ScopeError, "MODULE_SCOPE", "input registration differs")
        for name in roots:
            _module_name(name)
            rows.append({"module": name, "path": directory + "/" + name.replace(".", "/") + ".lean",
                         "role": "input"})
    _need(len(rows) == 99 and len({row["module"] for row in rows}) == 99
          and len({row["path"] for row in rows}) == 99
          and type(manifest.get("inherited_input_count")) is int
          and manifest["inherited_input_count"] == 32,
          ScopeError, "MODULE_SCOPE", "duplicate/missing input roots")
    wanted = profile.get("modules")
    _need(type(wanted) is list and all(type(row) is dict for row in wanted),
          ScopeError, "MODULE_SCOPE", "missing frozen module matrix")
    identities = [{key: row.get(key) for key in ("module", "path", "role")} for row in wanted]
    _need(rows == identities, ScopeError, "MODULE_SCOPE", "module/path/role binding differs")
    files = manifest.get("files")
    _need(type(files) is dict, ScopeError, "SOURCE_BINDING", "missing source hashes")
    for row, expected in zip(rows, wanted):
        value = files.get(row["path"])
        _need(_hex(value) and value == expected.get("source_sha256"),
              ScopeError, "SOURCE_BINDING", row["path"])
        row["source_sha256"] = value
    matrix = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    _need(hashlib.sha256(matrix).hexdigest() == profile.get("module_vector_sha256"),
          ScopeError, "SOURCE_BINDING", "module matrix digest differs")

    targets = manifest.get("targets")
    _need(type(targets) is list and all(type(x) is str for x in targets),
          ScopeError, "TARGET_SCOPE", "missing target vector")
    for target in targets:
        _target_name(target)
    _need(len(targets) == 1137 and len(set(targets)) == len(targets)
          and targets == profile.get("targets"),
          ScopeError, "TARGET_SCOPE", "literal target vector differs")
    vector = ("\n".join(targets) + "\n").encode("utf-8")
    _need(hashlib.sha256(vector).hexdigest() == profile.get("target_vector_sha256"),
          ScopeError, "TARGET_SCOPE", "target vector digest differs")
    counts = profile.get("expected_counts")
    expected_counts = {"production_modules": 67, "input_modules": 32, "ordinary_inputs": 25,
                       "elder_inputs": 7, "public_targets": 1137,
                       "proposition_targets": 1128, "data_targets": 9}
    _need(type(counts) is dict and counts == expected_counts
          and all(type(x) is int for x in counts.values()),
          ScopeError, "MODULE_SCOPE", "frozen expected counts differ")
    _need(type(profile.get("scope_id")) is str and bool(profile["scope_id"]),
          ScopeError, "SOURCE_IDENTITY", "missing scope identity")
    result = _blocked()
    result.update(scope_id=profile["scope_id"], source=dict(identity), modules=rows,
                  targets=list(targets), expected_counts=dict(counts))
    return result


def verify_source_files(scope: dict, source_files: dict[str, bytes]) -> None:
    """Check an exact caller-provided byte map; never read paths from the filesystem."""
    _need(type(scope) is dict and type(scope.get("modules")) is list
          and type(source_files) is dict,
          ScopeError, "SOURCE_BINDING", "source maps required")
    expected = {}
    for row in scope["modules"]:
        _need(type(row) is dict, ScopeError, "SOURCE_BINDING", "invalid source row")
        path = _source_path(row.get("path"))
        _need(path not in expected and _hex(row.get("source_sha256")),
              ScopeError, "SOURCE_BINDING", "duplicate/invalid source binding")
        expected[path] = row["source_sha256"]
    _need(set(source_files) == set(expected), ScopeError, "SOURCE_BINDING",
          "selected source set differs")
    for path, digest in expected.items():
        raw = source_files[path]
        _need(type(raw) is bytes and hashlib.sha256(raw).hexdigest() == digest,
              ScopeError, "SOURCE_BINDING", path)


def encode_name(name: tuple[str | int, ...]) -> str:
    """Preserve string/numeric components and Unicode code points exactly."""
    _need(type(name) is tuple, NameProtocolError, "NAME_PROTOCOL", "tuple required")
    encoded = []
    try:
        for component in name:
            if type(component) is str:
                encoded.append("s" + component.encode("utf-8").hex())
            elif _nat(component):
                encoded.append("n" + str(component))
            else:
                raise NameProtocolError("NAME_PROTOCOL", "invalid component type/value")
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, NameProtocolError):
            raise
        raise NameProtocolError("NAME_PROTOCOL", str(exc)) from exc
    return "/".join(encoded)


def decode_name(encoded: str) -> tuple[str | int, ...]:
    _need(type(encoded) is str, NameProtocolError, "NAME_PROTOCOL", "string required")
    if encoded == "":
        return ()
    result = []
    try:
        for component in encoded.split("/"):
            if re.fullmatch(r"s(?:[0-9a-f]{2})*", component):
                result.append(bytes.fromhex(component[1:]).decode("utf-8"))
            elif re.fullmatch(r"n(?:0|[1-9][0-9]*)", component):
                result.append(int(component[1:]))
            else:
                raise NameProtocolError("NAME_PROTOCOL", "noncanonical component")
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, NameProtocolError):
            raise
        raise NameProtocolError("NAME_PROTOCOL", str(exc)) from exc
    return tuple(result)


def _wire_name(value):
    try:
        name = decode_name(value)
    except NameProtocolError as exc:
        raise InventoryProtocolError("INVENTORY_PROTOCOL", str(exc)) from exc
    _need(bool(name), InventoryProtocolError, "INVENTORY_PROTOCOL", "anonymous identity")
    return value


def _wire_names(value, unique=True):
    _need(type(value) is list, InventoryProtocolError, "INVENTORY_PROTOCOL", "name list required")
    for name in value:
        _wire_name(name)
    _need(not unique or len(set(value)) == len(value),
          InventoryProtocolError, "INVENTORY_PROTOCOL", "duplicate identity")
    return value


def _keys(value, keys):
    _need(type(value) is dict and set(value) == set(keys),
          InventoryProtocolError, "INVENTORY_PROTOCOL", "record fields differ")


def parse_inventory(text: str) -> dict:
    """Strict declared JSONL inventory; parser acceptance does not prove collection."""
    _need(type(text) is str and bool(text), InventoryProtocolError,
          "INVENTORY_PROTOCOL", "empty transcript")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    _need(all(lines), InventoryProtocolError, "INVENTORY_PROTOCOL", "blank record")
    records = [_json(line, InventoryProtocolError, "INVENTORY_PROTOCOL") for line in lines]
    _need(len(records) >= 3 and all(type(row) is dict for row in records),
          InventoryProtocolError, "INVENTORY_PROTOCOL", "incomplete transcript")
    header, end = records[0], records[-1]
    _keys(header, ("record", "schema", "scope_id"))
    _keys(end, ("record", "module_count", "declaration_count", "axioms"))
    _need(header["record"] == "header" and type(header["schema"]) is int
          and header["schema"] == 1 and type(header["scope_id"]) is str
          and bool(header["scope_id"]) and end["record"] == "end",
          InventoryProtocolError, "INVENTORY_PROTOCOL", "header/end required")
    modules, declarations, by_module, by_name = [], [], {}, {}
    for row in records[1:-1]:
        kind = row.get("record")
        if kind == "module":
            _keys(row, ("record", "name", "const_count", "object_parts", "extra_const_names"))
            name = _wire_name(row["name"])
            _need(name not in by_module and _nat(row["const_count"]),
                  InventoryProtocolError, "INVENTORY_PROTOCOL", "duplicate module or invalid count")
            parts = row["object_parts"]
            _need(type(parts) is list and bool(parts), InventoryProtocolError,
                  "INVENTORY_PROTOCOL", "object parts required")
            levels = []
            for part in parts:
                _keys(part, ("level", "sha256", "const_names", "constant_names"))
                level = part["level"]
                _need(type(level) is str and level in ("exported", "server", "private")
                      and _hex(part["sha256"]),
                      InventoryProtocolError, "INVENTORY_PROTOCOL", "invalid part")
                levels.append(("exported", "server", "private").index(level))
                left, right = _wire_names(part["const_names"]), _wire_names(part["constant_names"])
                _need(left == right, InventoryProtocolError, "INVENTORY_PROTOCOL",
                      "parallel constant tables differ")
            _need(levels[0] == 0 and levels == sorted(set(levels)),
                  InventoryProtocolError, "INVENTORY_PROTOCOL", "part visibility order differs")
            final = parts[-1]["const_names"]
            _need(row["const_count"] == len(final)
                  and all(set(part["const_names"]).issubset(final) for part in parts),
                  InventoryProtocolError, "INVENTORY_PROTOCOL", "part population differs")
            _wire_names(row["extra_const_names"])
            modules.append(row)
            by_module[name] = row
        elif kind == "declaration":
            _keys(row, ("record", "module", "name", "kind", "is_prop", "is_private", "unsafe",
                        "partial", "type_display", "object_part_sha256", "axioms"))
            _wire_name(row["module"])
            name = _wire_name(row["name"])
            _need(name not in by_name and type(row["kind"]) is str and row["kind"] in _KINDS
                  and all(type(row[key]) is bool for key in ("is_prop", "is_private", "unsafe", "partial"))
                  and type(row["type_display"]) is str and bool(row["type_display"])
                  and _hex(row["object_part_sha256"]),
                  InventoryProtocolError, "INVENTORY_PROTOCOL", "invalid/duplicate declaration")
            _wire_names(row["axioms"])
            declarations.append(row)
            by_name[name] = row
        else:
            raise InventoryProtocolError("INVENTORY_PROTOCOL", "unexpected record")
    _need(bool(modules) and _nat(end["module_count"]) and _nat(end["declaration_count"])
          and end["module_count"] == len(modules) and end["declaration_count"] == len(declarations),
          InventoryProtocolError, "INVENTORY_PROTOCOL", "terminal counts differ")
    populations = {name: set() for name in by_module}
    for row in declarations:
        module = by_module.get(row["module"])
        _need(module is not None, InventoryProtocolError, "INVENTORY_PROTOCOL", "unknown owner")
        _need(row["object_part_sha256"] == module["object_parts"][-1]["sha256"],
              InventoryProtocolError, "INVENTORY_PROTOCOL", "declaration object binding differs")
        populations[row["module"]].add(row["name"])
    for name, module in by_module.items():
        _need(populations[name] == set(module["object_parts"][-1]["const_names"]),
              InventoryProtocolError, "INVENTORY_PROTOCOL", "declaration coverage differs")
    axioms = _wire_names(end["axioms"])
    union = {name for row in declarations for name in row["axioms"]}
    _need(set(axioms) == union, InventoryProtocolError, "INVENTORY_PROTOCOL", "axiom union differs")
    return {"scope_id": header["scope_id"], "modules": modules,
            "declarations": declarations, "axioms": axioms}


def classify_inventory(inventory: dict, scope: dict) -> dict:
    """Classify declared fields only; unknown roles/types/coverage always block."""
    result = _blocked()
    findings = set()
    def flag(code, name):
        findings.add((code, name))
    _need(type(inventory) is dict and type(scope) is dict
          and all(key in inventory for key in ("scope_id", "modules", "declarations", "axioms"))
          and type(scope.get("modules")) is list and type(scope.get("targets")) is list,
          InventoryProtocolError, "INVENTORY_PROTOCOL", "parsed inventory and scope required")
    expected_modules = {}
    for row in scope["modules"]:
        _need(type(row) is dict and row.get("role") in ("production", "input"),
              ScopeError, "MODULE_SCOPE", "invalid scope module")
        module = encode_name(tuple(_module_name(row.get("module")).split(".")))
        _need(module not in expected_modules, ScopeError, "MODULE_SCOPE", "duplicate scope module")
        expected_modules[module] = row
    actual_modules = {row["name"] for row in inventory["modules"]}
    if inventory["scope_id"] != scope.get("scope_id"):
        flag("MODULE_SCOPE", "scope_id")
    for name in actual_modules.symmetric_difference(expected_modules):
        flag("MODULE_SCOPE", name)
    wanted = {encode_name(_target_name(name)) for name in scope["targets"]}
    _need(len(wanted) == len(scope["targets"]), ScopeError, "TARGET_SCOPE", "duplicate scope target")
    declarations = {row["name"]: row for row in inventory["declarations"]}
    for name in wanted - declarations.keys():
        flag("TARGET_SCOPE", name)
    public = [declarations[name] for name in sorted(wanted & declarations.keys())]
    for row in public:
        owner = expected_modules.get(row["module"])
        if owner is None or owner["role"] != "production" or row["is_private"]:
            flag("TARGET_OWNER", row["name"])
    propositions = sum(row["is_prop"] for row in public)
    counts = {"owned_declarations": len(inventory["declarations"]), "public_targets": len(public),
              "proposition_targets": propositions, "data_targets": len(public) - propositions}
    expected_counts = scope.get("expected_counts", {})
    if any(counts[key] != expected_counts.get(key) for key in
           ("public_targets", "proposition_targets", "data_targets")):
        flag("TARGET_KIND_COUNTS", "targets")
    for module in inventory["modules"]:
        if module["extra_const_names"]:
            flag("UNSUPPORTED_EXTRAS", module["name"])
    for row in inventory["declarations"]:
        if row["kind"] == "axiom":
            flag("OWNED_AXIOM", row["name"])
        if row["unsafe"]:
            flag("UNSAFE_REPLAY_SKIP", row["name"])
        if row["partial"]:
            flag("PARTIAL_REPLAY_SKIP", row["name"])
    allowed = {encode_name(("propext",)), encode_name(("Classical", "choice")),
               encode_name(("Quot", "sound"))}
    for name in set(inventory["axioms"]) - allowed:
        flag("FORBIDDEN_AXIOM", name)
    result.update(counts=counts, findings=[{"code": code, "name": name}
                                         for code, name in sorted(findings)])
    return result


def _receipt_need(condition, note):
    if not condition:
        raise ValueError(note)


def _receipt_keys(value, keys):
    _receipt_need(type(value) is dict and set(value) == set(keys), "receipt fields differ")


def _utc(value):
    _receipt_need(type(value) is str and re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z", value
    ) is not None, "full extended UTC timestamp required")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    _receipt_need(parsed.tzinfo == timezone.utc, "aware UTC timestamp required")
    return parsed


def _receipt_schema(value):
    _receipt_keys(value, (
        "schema", "scope_id", "scientific_effect", "source", "adapter_sha256", "profile_sha256",
        "run", "toolchain", "inputs_before_sha256", "inputs_after_sha256",
        "cache_before_sha256", "cache_after_sha256", "owned_objects_sha256",
        "module_coverage", "unsafe_skipped", "partial_skipped", "steps",
    ))
    _receipt_need(type(value["schema"]) is int and value["schema"] == 1
                  and type(value["scope_id"]) is str and bool(value["scope_id"])
                  and value["scientific_effect"] == "NONE", "receipt schema/scope differs")
    source = value["source"]
    _receipt_keys(source, ("repository", "commit", "tree"))
    _receipt_need(type(source["repository"]) is str and bool(source["repository"])
                  and _hex(source["commit"], _GIT) and _hex(source["tree"], _GIT), "source schema")
    run = value["run"]
    _receipt_keys(run, ("repository", "id", "attempt", "job_id"))
    _receipt_need(type(run["repository"]) is str and bool(run["repository"])
                  and all(_nat(run[k]) and run[k] > 0 for k in ("id", "attempt", "job_id")),
                  "run schema")
    toolchain = value["toolchain"]
    _receipt_keys(toolchain, ("commit", "lean_sha256", "lake_sha256"))
    _receipt_need(_hex(toolchain["commit"], _GIT) and _hex(toolchain["lean_sha256"])
                  and _hex(toolchain["lake_sha256"]), "toolchain schema")
    for key in ("adapter_sha256", "profile_sha256", "inputs_before_sha256", "inputs_after_sha256",
                "cache_before_sha256", "cache_after_sha256", "owned_objects_sha256"):
        _receipt_need(_hex(value[key]), "hash schema: " + key)
    for key in ("module_coverage", "unsafe_skipped", "partial_skipped"):
        _wire_names(value[key], unique=False)
    _receipt_need(type(value["steps"]) is list, "step list required")
    for step in value["steps"]:
        _receipt_keys(step, ("name", "exit_code", "signal", "timed_out", "stdout_sha256",
                             "stderr_sha256", "started_utc", "ended_utc", "duration_ns", "max_rss_kib"))
        _receipt_need(type(step["name"]) is str and bool(step["name"])
                      and type(step["exit_code"]) is int
                      and (step["signal"] is None or _nat(step["signal"]) and step["signal"] > 0)
                      and type(step["timed_out"]) is bool
                      and _hex(step["stdout_sha256"]) and _hex(step["stderr_sha256"])
                      and _nat(step["duration_ns"]) and _nat(step["max_rss_kib"]),
                      "step schema")
        _receipt_need(_utc(step["ended_utc"]) >= _utc(step["started_utc"]), "backwards step UTC")


def classify_receipt(receipt: dict, expected: dict) -> dict:
    """Validate declared bindings; expected must be independently measured for later use.

    Even identical caller assertions only produce DECLARED_FIELDS_VALID, never proof
    of execution. No filesystem/log authentication or qualified result exists here.
    """
    result = _blocked()
    result.update(binding_status="INVALID", findings=[])
    def reject(code, name):
        result["findings"] = [{"code": code, "name": name}]
        return result
    for name, value in (("receipt", receipt), ("expected", expected)):
        try:
            _receipt_schema(value)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return reject("RECEIPT_SCHEMA", name + ": " + str(exc))
    expected_names = [step["name"] for step in expected["steps"]]
    names = [step["name"] for step in receipt["steps"]]
    required = ["build", "inventory"] + ["kernel:" + name for name in expected["module_coverage"]]
    if (expected_names != required or names != expected_names
            or len(set(names)) != len(names)):
        return reject("STEP_COVERAGE", "steps")
    if any(step["exit_code"] != 0 or step["signal"] is not None or step["timed_out"]
           for value in (receipt, expected) for step in value["steps"]):
        return reject("STEP_FAILED", "steps")
    if any(value["inputs_before_sha256"] != value["inputs_after_sha256"]
           for value in (receipt, expected)):
        return reject("SOURCE_DRIFT", "inputs")
    if any(value["cache_before_sha256"] != value["cache_after_sha256"]
           for value in (receipt, expected)):
        return reject("CACHE_DRIFT", "cache")
    coverage = receipt["module_coverage"]
    if (not coverage or not expected["module_coverage"]
            or coverage != expected["module_coverage"] or len(set(coverage)) != len(coverage)):
        return reject("MODULE_COVERAGE", "modules")
    if any(value["unsafe_skipped"] or value["partial_skipped"] for value in (receipt, expected)):
        return reject("REPLAY_SKIPPED", "declarations")
    if receipt != expected:
        return reject("RECEIPT_BINDING", "expected")
    result["binding_status"] = "DECLARED_FIELDS_VALID"
    return result
