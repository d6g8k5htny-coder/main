"""Strict validator for the Reactor A freeze package."""

import hashlib
import json
from pathlib import Path


CONTROL_KEYS = {
    "schema_version", "scope", "scientific_status_effect", "fixture",
    "sources", "closure", "alpha", "amplitude_scale", "threshold",
    "expected_control", "campaign_status",
}
SOURCE_KEYS = {
    "repository", "commit", "path", "byte_length", "raw_sha256",
    "git_blob_sha", "pull_request", "theorem_conclusion_consumed",
    "source_status",
}
RATIONAL_KEYS = {"numerator", "denominator"}
REQUIRED_CLOSURE = {
    ".github/workflows/reactor-v0-controls.yml",
    "experiments/reactor_v0/README.md",
    "experiments/reactor_v0/control.json",
    "experiments/reactor_v0/source_manifest.json",
    "experiments/reactor_v0/fixture.json",
    "experiments/reactor_v0/exact_helpers.py",
    "experiments/reactor_v0/evaluator.py",
    "experiments/reactor_v0/validator.py",
    "experiments/reactor_v0/run_control.py",
    "experiments/reactor_v0/test_reactor_v0.py",
    "experiments/reactor_v0/mutation_harness.py",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid UTF-8 JSON: " + str(path)) from error


def _strict_keys(document, expected, name):
    _require(type(document) is dict and set(document) == expected,
             name + " has missing or extra fields")


def _rational(document, name):
    _strict_keys(document, RATIONAL_KEYS, name)
    _require(type(document["numerator"]) is int and
             type(document["denominator"]) is int and
             document["denominator"] > 0, name + " must be an exact rational")


def validate_document(document):
    _strict_keys(document, CONTROL_KEYS, "control")
    _require(document["schema_version"] == 1 and type(document["schema_version"]) is int,
             "unsupported control schema")
    _require(document["scope"] == "ENGINEERING_ONLY_FINITE_AMPLITUDE_TRANSFORM_CONTROL",
             "unexpected control scope")
    _require(document["scientific_status_effect"] == "NONE",
             "scientific_status_effect must be NONE")
    _require(document["campaign_status"] == "B_UNRESOLVED_NO_RUN",
             "campaign B must remain UNRESOLVED/no-run")
    for key in ("fixture", "sources", "closure"):
        _require(type(document[key]) is str and document[key] and "/" not in document[key],
                 key + " must be a local filename")
    for key in ("alpha", "amplitude_scale", "threshold", "expected_control"):
        _rational(document[key], key)
    return True


def _validate_fixture(document):
    expected = {"schema_version", "grid_side", "values", "expected_barcode", "exclusions"}
    _strict_keys(document, expected, "fixture")
    _require(document["schema_version"] == 1 and type(document["schema_version"]) is int,
             "unsupported fixture schema")
    side = document["grid_side"]
    values = document["values"]
    _require(type(side) is int and side == 4, "exact 4x4 fixture required")
    _require(type(values) is list and len(values) == 16 and
             all(type(value) is int for value in values), "strict 4x4 integer values required")
    _require(document["expected_barcode"] == {"intervals": [[1, 0], [8, 0]], "essential": [27]},
             "unexpected exposed barcode oracle")
    _require(document["exclusions"] == ["essential_class"],
             "essential class must be excluded")


def _validate_sources(document):
    _require(type(document) is dict and set(document) == {"schema_version", "sources"},
             "source manifest has missing or extra fields")
    _require(document["schema_version"] == 1 and type(document["schema_version"]) is int,
             "unsupported source manifest schema")
    sources = document["sources"]
    _require(type(sources) is list and len(sources) == 2, "exactly two parent sources required")
    for source in sources:
        _strict_keys(source, SOURCE_KEYS, "source")
        _require(source["repository"] == "d6g8k5htny-coder/main", "unexpected source repository")
        _require(type(source["commit"]) is str and len(source["commit"]) == 40,
                 "full source commit required")
        _require(type(source["path"]) is str and source["path"].endswith("/PROOF.md"),
                 "proof source path required")
        _require(type(source["byte_length"]) is int and source["byte_length"] > 0,
                 "positive source byte length required")
        _require(type(source["raw_sha256"]) is str and len(source["raw_sha256"]) == 64,
                 "raw SHA-256 required")
        _require(type(source["git_blob_sha"]) is str and len(source["git_blob_sha"]) == 40,
                 "Git blob membership identity required separately")
        _require(type(source["pull_request"]) is int and source["pull_request"] in (330, 334),
                 "parent PR allowlist mismatch")
        _require(source["theorem_conclusion_consumed"] is False,
                 "A may consume no theorem conclusion")
        _require(source["source_status"] == "OPEN_UNMERGED_AT_FREEZE",
                 "source status must remain explicit")
    return sources


def verify_closure(package_directory, closure):
    repository_root = Path(package_directory).resolve().parents[1]
    _require(type(closure) is dict and set(closure) == REQUIRED_CLOSURE,
             "consumed closure path set mismatch")
    for relative, digest in closure.items():
        _require(type(digest) is str and len(digest) == 64,
                 "closure SHA-256 must be 64 lowercase hex characters")
        path = repository_root / relative
        _require(path.is_file(), "missing closure path: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == digest, "closure hash mismatch: " + relative)
    return True


def load_and_validate(package_directory):
    directory = Path(package_directory).resolve()
    control = _load_json(directory / "control.json")
    validate_document(control)
    fixture = _load_json(directory / control["fixture"])
    _validate_fixture(fixture)
    source_manifest = _load_json(directory / control["sources"])
    sources = _validate_sources(source_manifest)
    closure = _load_json(directory / control["closure"])
    verify_closure(directory, closure)
    return {"control": control, "fixture": fixture, "sources": sources, "closure": closure}
