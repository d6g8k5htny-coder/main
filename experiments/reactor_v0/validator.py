"""Strict validator for the Reactor A freeze package."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from fractions import Fraction


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
SOURCE_PINS = {
    330: (
        "88ff9223e57bf38c7fc33cb3559a88793a7b5087",
        "experiments/universality/spectral_gaussian_admission/PROOF.md",
    ),
    334: (
        "6bd133d591e43e75a9292463bb4bfde7a34984dc",
        "experiments/universality/spectral_amplitude_crossover/PROOF.md",
    ),
}
LOWER_HEX = re.compile(r"[0-9a-f]+\Z")


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
    return Fraction(document["numerator"], document["denominator"])


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
        _require(type(document[key]) is str and document[key] not in ("", ".", "..") and
                 "/" not in document[key] and "\\" not in document[key],
                 key + " must be a local filename")
    alpha = _rational(document["alpha"], "alpha")
    amplitude_scale = _rational(document["amplitude_scale"], "amplitude_scale")
    threshold = _rational(document["threshold"], "threshold")
    expected_control = _rational(document["expected_control"], "expected_control")
    _require(alpha > 0, "alpha must be positive")
    _require(amplitude_scale > 0, "amplitude_scale must be positive")
    _require(threshold >= 0, "threshold must be nonnegative")
    _require(expected_control >= 0, "expected_control must be nonnegative")
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
    commits = []
    pull_requests = []
    for source in sources:
        _strict_keys(source, SOURCE_KEYS, "source")
        _require(source["repository"] == "d6g8k5htny-coder/main", "unexpected source repository")
        _require(type(source["pull_request"]) is int and
                 source["pull_request"] in SOURCE_PINS,
                 "parent PR allowlist mismatch")
        pull_requests.append(source["pull_request"])
        _require(type(source["commit"]) is str and len(source["commit"]) == 40 and
                 LOWER_HEX.fullmatch(source["commit"]) is not None,
                 "full source commit required")
        _require((source["commit"], source["path"]) ==
                 SOURCE_PINS[source["pull_request"]],
                 "source PR, commit, and path pin mismatch")
        commits.append(source["commit"])
        _require(type(source["byte_length"]) is int and source["byte_length"] > 0,
                 "positive source byte length required")
        _require(type(source["raw_sha256"]) is str and len(source["raw_sha256"]) == 64 and
                 LOWER_HEX.fullmatch(source["raw_sha256"]) is not None,
                 "raw SHA-256 required")
        _require(type(source["git_blob_sha"]) is str and len(source["git_blob_sha"]) == 40 and
                 LOWER_HEX.fullmatch(source["git_blob_sha"]) is not None,
                 "Git blob membership identity required separately")
        _require(source["theorem_conclusion_consumed"] is False,
                 "A may consume no theorem conclusion")
        _require(source["source_status"] == "OPEN_UNMERGED_AT_FREEZE",
                 "source status must remain explicit")
    _require(set(pull_requests) == set(SOURCE_PINS) and
             len(set(pull_requests)) == len(pull_requests),
             "parent PR identities must be unique and exact")
    _require(len(set(commits)) == len(commits), "parent source commits must be unique")
    return sources


def _git(repo_root, *arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(completed.returncode == 0,
             "git source-object verification failed: " +
             completed.stderr.decode("utf-8", errors="replace").strip())
    return completed.stdout


def verify_source_objects(repository_root, sources):
    root = Path(repository_root).resolve()
    for source in sources:
        object_name = source["commit"] + ":" + source["path"]
        raw = _git(root, "cat-file", "blob", object_name)
        blob = _git(root, "rev-parse", object_name).decode("ascii").strip()
        _require(len(raw) == source["byte_length"],
                 "source byte length mismatch: " + object_name)
        _require(hashlib.sha256(raw).hexdigest() == source["raw_sha256"],
                 "source raw SHA-256 mismatch: " + object_name)
        _require(blob == source["git_blob_sha"],
                 "source Git blob mismatch: " + object_name)
    return True


def verify_closure(package_directory, closure):
    package_directory = Path(package_directory)
    _require(not package_directory.is_symlink(), "package directory symlink is forbidden")
    repository_root = package_directory.resolve().parents[1]
    _require(type(closure) is dict and set(closure) == REQUIRED_CLOSURE,
             "consumed closure path set mismatch")
    expected_package_files = {
        Path(relative).relative_to("experiments/reactor_v0").as_posix()
        for relative in REQUIRED_CLOSURE
        if relative.startswith("experiments/reactor_v0/")
    } | {"closure.json"}
    package_entries = list(package_directory.rglob("*"))
    _require(all(not path.is_symlink() for path in package_entries),
             "package symlink is forbidden")
    actual_package_files = {
        path.relative_to(package_directory).as_posix()
        for path in package_entries
        if not path.is_dir()
    }
    _require(actual_package_files == expected_package_files,
             "package file set differs from frozen closure")
    for relative, digest in closure.items():
        _require(type(digest) is str and len(digest) == 64 and
                 LOWER_HEX.fullmatch(digest) is not None,
                 "closure SHA-256 must be 64 lowercase hex characters")
        path = repository_root / relative
        _require(not path.is_symlink(), "closure member symlink is forbidden: " + relative)
        _require(path.is_file(), "missing closure path: " + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == digest, "closure hash mismatch: " + relative)
    return True


def load_and_validate(package_directory):
    supplied_directory = Path(package_directory)
    _require(not supplied_directory.is_symlink(),
             "package directory symlink is forbidden")
    directory = supplied_directory.resolve()
    control = _load_json(directory / "control.json")
    validate_document(control)
    fixture = _load_json(directory / control["fixture"])
    _validate_fixture(fixture)
    source_manifest = _load_json(directory / control["sources"])
    sources = _validate_sources(source_manifest)
    closure = _load_json(directory / control["closure"])
    verify_closure(directory, closure)
    if os.environ.get("REACTOR_VERIFY_SOURCES") == "1":
        verify_source_objects(directory.parents[1], sources)
    return {"control": control, "fixture": fixture, "sources": sources, "closure": closure}
