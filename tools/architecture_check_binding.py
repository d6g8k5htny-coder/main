#!/usr/bin/env python3
"""Bind producer outputs to current native identities; no scientific authority.

Run in the hosted ephemeral runner. This validates the producer's identity
strings, not archive bytes, the contents of a receipt, or mathematical claims.
"""
from __future__ import annotations

import argparse
import json
import re
import sys


STDIN_LIMIT = 64 * 1024
OUTPUT_FIELDS = (
    "checked_commit", "repository", "run_id", "run_attempt",
    "receipt_sha256", "artifact_id", "artifact_sha256", "artifact_name",
)
NATIVE_FIELDS = (
    ("commit", "checked_commit"),
    ("repository", "repository"),
    ("run-id", "run_id"),
    ("run-attempt", "run_attempt"),
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
IDENTITY_PATTERNS = {
    "checked_commit": HEX40,
    "repository": REPOSITORY,
    "run_id": POSITIVE_DECIMAL,
    "run_attempt": POSITIVE_DECIMAL,
    "receipt_sha256": HEX64,
    "artifact_id": POSITIVE_DECIMAL,
    "artifact_sha256": HEX64,
}


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document = {}
    for key, value in pairs:
        require(key not in document, "duplicate JSON key")
        document[key] = value
    return document


def reject_number(_: str) -> None:
    # Every value allowed by this input contract is a string or object. Reject
    # numbers during parsing, including nonstandard NaN/Infinity and exponent
    # overflow such as 1e999, without converting or coercing any identity.
    raise ValueError("JSON numbers are not identity strings")


class Once(argparse.Action):
    """Refuse ambiguous repeated native identity options."""

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


def validate_identity(field: str, value: object) -> None:
    require(type(value) is str, f"{field} must be a string")
    require(IDENTITY_PATTERNS[field].fullmatch(value), f"invalid {field} format")
    if field == "repository":
        require(all(part not in (".", "..") for part in value.split("/")),
                "invalid repository format")


def bind(document: object, expected: dict[str, str]) -> dict[str, object]:
    require(type(document) is dict, "input must be an object")
    require(set(document) == {"result", "outputs"}, "invalid input fields")
    require(type(document["result"]) is str and document["result"] == "success",
            "producer result is not success")
    outputs = document["outputs"]
    require(type(outputs) is dict, "outputs must be an object")
    require(set(outputs) == set(OUTPUT_FIELDS), "invalid output fields")
    for field in OUTPUT_FIELDS:
        require(type(outputs[field]) is str and outputs[field] != "",
                f"{field} must be a nonempty string")
    for field in IDENTITY_PATTERNS:
        validate_identity(field, outputs[field])
    for field, value in expected.items():
        require(outputs[field] == value, f"{field} differs from current native identity")
    artifact_name = f"research-architecture-{expected['run_id']}-{expected['run_attempt']}"
    require(outputs["artifact_name"] == artifact_name,
            "artifact_name differs from current run and attempt")
    return {
        "schema_version": 1,
        "scientific_effect": "NONE",
        "scientific_status_authority": False,
        "binding": {field: outputs[field] for field in OUTPUT_FIELDS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for option, _ in NATIVE_FIELDS:
        parser.add_argument(f"--expected-{option}", required=True, action=Once)
    arguments = parser.parse_args(argv)
    try:
        expected = {
            field: getattr(arguments, "expected_" + option.replace("-", "_"))
            for option, field in NATIVE_FIELDS
        }
        for field, value in expected.items():
            validate_identity(field, value)
        raw = sys.stdin.buffer.read(STDIN_LIMIT + 1)
        require(len(raw) <= STDIN_LIMIT, "stdin exceeds 64 KiB")
        document = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_object,
            parse_int=reject_number, parse_float=reject_number,
            parse_constant=reject_number,
        )
        envelope = bind(document, expected)
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    except (ValueError, UnicodeError, OSError, RecursionError) as exc:
        parser.exit(1, f"architecture binding refused: {exc}\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
