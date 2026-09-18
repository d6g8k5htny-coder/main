#!/usr/bin/env python3
"""Check every review record in `reviews/records/` against the R17 §4 form.

OP-PROT-019-v1.1 (R17) §4 says what a valid technical review records, and says
just as plainly what it does not do: "A task may finish its technical review
while an external-independence predicate remains open. Never relabel an
independence-required theorem terminal solely because its technical review
passed." This checker turns that sentence into an assertion a CI run can fail
on, so a review written here cannot quietly overclaim.

It enforces:

  * conformance to `reviews/review_record.schema.json` (a self-contained subset
    of JSON Schema, evaluated by `validate_instance` below — no third-party
    dependency);
  * `independence_credit == 0` for an Anthropic-family reviewer, with a
    non-empty reason; and 0 likewise when reviewer and author share a family or
    the author lineage is unknown; a credit of 1 requires documented
    OP-PROT-012 §5 evidence;
  * `gate_status_after == "UNCHANGED"`, always, on every record;
  * `technical_verdict` drawn from the register's own R17 status set and
    nothing else (and that the register itself still uses only that set);
  * `obtained == false` implies `technical_verdict == "CANNOT_VERIFY"` — you may
    not pass or fail an object you did not read — and `obtained == true`
    requires a 64-hex digest and a positive byte count;
  * no promotion language in any free-text field, and no confidence voting
    (R17 §4: "No confidence voting.");
  * every `route_key` exists verbatim in `registers/json/review_queue.json`, and
    a digest that disagrees with that row's `Body SHA-256` is explained;
  * `does_not_establish` and `exposure_disclosure` are present, long enough,
    lexically varied, and not shared verbatim with another record.

This checker verifies the FORM of a review. It does not verify any mathematics,
it awards no independence credit, and it moves no gate. A record that passes it
is still only a verdict on an object.

Exit status is non-zero when any record violates any rule.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(ROOT, "reviews", "review_record.schema.json")
RECORDS_DIR = os.path.join(ROOT, "reviews", "records")
QUEUE_PATH = os.path.join(ROOT, "registers", "json", "review_queue.json")

# R17 §4, verbatim: "Technical statuses: READY, IN_REVIEW, PASS_TECHNICAL,
# AMEND, FAIL, CANNOT_VERIFY, or NEEDS_RECONCILIATION."
R17_TECH_STATUS = ("READY", "IN_REVIEW", "PASS_TECHNICAL", "AMEND", "FAIL",
                   "CANNOT_VERIFY", "NEEDS_RECONCILIATION")

# Families that can never award themselves organizational independence here.
SAME_FAMILY_AS_THIS_SESSION = "anthropic"

# Free-text phrases that would convert a verdict on an object into a movement of
# a gate. The list is the one the task and the program's own prose name; it is
# deliberately literal, so a record must phrase an honest negative differently
# ("no obligation moves from OPEN", "does not discharge", "the gate remains
# open") rather than reusing promotion wording under a negation.
PROMOTION_PHRASES = (
    "discharged",
    "closed the",
    "promotes",
    "premise is now",
    "gate satisfied",
    "independence satisfied",
)

# R17 §4: "No confidence voting."
CONFIDENCE_PATTERNS = (
    re.compile(r"\b\d{1,3}\s*%\s*(?:confiden|sure|certain)", re.I),
    re.compile(r"\bconfidence\s+(?:score|level|vote|rating|interval of belief)\b", re.I),
    re.compile(r"\b(?:high|medium|low|moderate)\s+confidence\b", re.I),
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
WORD = re.compile(r"[A-Za-z0-9_./-]+")

# Anti-boilerplate floors. Length alone is easy to pad, so a minimum number of
# DISTINCT tokens is required too.
MIN_DISTINCT_TOKENS = {"exposure_disclosure": 40, "does_not_establish": 25}


# --------------------------------------------------------------------------- #
# A small, self-contained JSON Schema subset evaluator.                        #
# --------------------------------------------------------------------------- #

def _type_ok(value, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type {expected!r}")


def validate_instance(value, schema: dict, where: str = "$") -> list[str]:
    """Validate `value` against the supported JSON Schema subset.

    Supported keywords: type, const, enum, pattern, minLength, minItems,
    minimum, maximum, required, properties, additionalProperties (false only),
    items. Anything else in the schema is ignored, deliberately: an unsupported
    keyword must never be read as a silent pass of a rule this file claims to
    enforce, so the rules that matter are ALSO asserted explicitly in
    `check_record` below.
    """
    problems: list[str] = []

    if "const" in schema and value != schema["const"]:
        problems.append(f"{where}: must be {schema['const']!r}, found {value!r}")
        return problems
    if "enum" in schema and value not in schema["enum"]:
        problems.append(
            f"{where}: {value!r} is not one of {sorted(map(str, schema['enum']))}")
        return problems

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_type_ok(value, t) for t in types):
            problems.append(f"{where}: expected type {expected}, found "
                            f"{type(value).__name__}")
            return problems

    if isinstance(value, str):
        if "minLength" in schema and len(value.strip()) < schema["minLength"]:
            problems.append(
                f"{where}: {len(value.strip())} characters, minimum "
                f"{schema['minLength']} — too short to be a real disclosure")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            problems.append(f"{where}: {value!r} does not match /{pattern}/")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{where}: {value} is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{where}: {value} is above maximum {schema['maximum']}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{where}: {len(value)} item(s), minimum "
                            f"{schema['minItems']}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                problems += validate_instance(item, item_schema, f"{where}[{i}]")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"{where}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    problems.append(f"{where}: unknown field {key!r} "
                                    f"(the schema is closed)")
        for key, sub in props.items():
            if key in value:
                problems += validate_instance(value[key], sub, f"{where}.{key}")

    return problems


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #

def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def queue_index(queue_path: str) -> tuple[dict[str, dict], list[str]]:
    """Map review key -> row dict from the exported Review Queue tab.

    Also returns any problem with the register's own status vocabulary: if the
    export ever grows a technical status outside the R17 set, the records that
    cite it must not be silently blessed.
    """
    data = load_json(queue_path)
    header = data["header"]
    rows = {}
    problems: list[str] = []
    for row in data["rows"]:
        record = {header[i]: (row[i] if i < len(row) else "")
                  for i in range(len(header))}
        rows[record["Review key"]] = record
        status = (record.get("Technical status") or "").strip()
        if status and status not in R17_TECH_STATUS:
            problems.append(
                f"{os.path.relpath(queue_path, ROOT)}: row {record['Review key']!r} "
                f"carries technical status {status!r}, which is outside the R17 §4 "
                f"set {list(R17_TECH_STATUS)}")
    return rows, problems


def walk_strings(value, where: str = "$"):
    """Yield (field_path, string) for every string anywhere in the record."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, sub in value.items():
            yield from walk_strings(sub, f"{where}.{key}")
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            yield from walk_strings(sub, f"{where}[{i}]")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- #
# Per-record rules                                                             #
# --------------------------------------------------------------------------- #

def check_record(record: dict, schema: dict, queue: dict[str, dict],
                 filename: str) -> list[str]:
    problems = [f"{filename}: {p}" for p in
                validate_instance(record, schema, "$")]

    def fail(message: str) -> None:
        problems.append(f"{filename}: {message}")

    # Field access that tolerates a record already failing schema validation.
    def get(key, default=None):
        return record.get(key, default)

    # -- identity ---------------------------------------------------------- #
    review_id = get("review_id")
    stem = os.path.splitext(os.path.basename(filename))[0]
    if isinstance(review_id, str) and review_id and stem != review_id:
        fail(f"filename stem {stem!r} does not match review_id {review_id!r}")

    # -- R17 §4: gate_status_after is always UNCHANGED ---------------------- #
    gate = get("gate_status_after")
    if gate != "UNCHANGED":
        fail(f"gate_status_after is {gate!r}; a technical review is a verdict on "
             f"an object and never moves a gate — the only admissible value is "
             f'"UNCHANGED"')

    # -- R17 §4: the technical status vocabulary --------------------------- #
    verdict = get("technical_verdict")
    if verdict not in R17_TECH_STATUS:
        fail(f"technical_verdict {verdict!r} is not one of the register's own R17 "
             f"statuses {list(R17_TECH_STATUS)}")

    # -- you may not pass or fail an object you did not read ---------------- #
    obtained = get("obtained")
    if obtained is False and verdict != "CANNOT_VERIFY":
        fail(f"obtained is false but technical_verdict is {verdict!r}: an object "
             f"that was not read can only yield CANNOT_VERIFY")
    if obtained is True:
        digest = get("object_sha256") or ""
        if not HEX64.match(digest):
            fail("obtained is true but object_sha256 is not a lowercase 64-hex "
                 "digest: a review is only real if the exact bytes were hashed")
        if not isinstance(get("object_bytes"), int) or isinstance(get("object_bytes"), bool) \
                or (get("object_bytes") or 0) <= 0:
            fail("obtained is true but object_bytes is not a positive count")
    if isinstance(get("obtained_how"), str) and not get("obtained_how").strip():
        fail("obtained_how is empty")

    # -- R17 §4: organizational independence -------------------------------- #
    credit = get("independence_credit")
    reviewer_family = get("reviewer_family")
    author_family = get("author_family")
    reason = (get("independence_reason") or "") if isinstance(
        get("independence_reason"), str) else ""
    if isinstance(credit, bool) or not isinstance(credit, int):
        fail(f"independence_credit must be an integer, found {credit!r}")
    else:
        if reviewer_family == SAME_FAMILY_AS_THIS_SESSION and credit != 0:
            fail(f"reviewer_family is {SAME_FAMILY_AS_THIS_SESSION!r} but "
                 f"independence_credit is {credit}: R17 §4 records same provider "
                 f"as ZERO organizational independence")
        if reviewer_family and reviewer_family == author_family and credit != 0:
            fail(f"reviewer_family and author_family are both "
                 f"{reviewer_family!r} but independence_credit is {credit}: a "
                 f"same-family reviewer earns zero organizational independence")
        if author_family == "unknown" and credit != 0:
            fail("author_family is 'unknown' but independence_credit is nonzero: "
                 "independence cannot be claimed against an unestablished author "
                 "lineage (R17 §4, OP-PROT-012 §5)")
        if credit != 0 and not (get("independence_evidence") or "").strip():
            fail("independence_credit is nonzero without independence_evidence: "
                 "R17 §4 — 'Different provider alone does not establish "
                 "independence'; OP-PROT-012 §5 (a)-(j) must be documented")
    if credit == 0 and not reason.strip():
        fail("independence_credit is 0 with an empty independence_reason: the "
             "reason for zero credit must be stated, not implied")

    # -- R17 §4: negative controls 'if applicable' -------------------------- #
    controls = get("negative_controls_executed")
    if isinstance(controls, list) and not controls:
        if not (get("negative_controls_not_applicable_reason") or "").strip():
            fail("negative_controls_executed is empty and "
                 "negative_controls_not_applicable_reason is empty: R17 §4 allows "
                 "'if applicable', not silence")

    # -- route_key must exist in the register ------------------------------- #
    route_key = get("route_key")
    row = queue.get(route_key) if isinstance(route_key, str) else None
    if row is None:
        fail(f"route_key {route_key!r} is not a Review key in "
             f"registers/json/review_queue.json")
    else:
        register_digest = (row.get("Body SHA-256") or "").strip().lower()
        record_digest = (get("object_sha256") or "").strip().lower()
        if HEX64.match(register_digest) and record_digest and \
                register_digest != record_digest:
            if not (get("sha_mismatch_explanation") or "").strip():
                fail(f"object_sha256 {record_digest[:12]}… differs from the "
                     f"Body SHA-256 {register_digest[:12]}… that the review_queue "
                     f"row {route_key!r} carries, with no sha_mismatch_explanation: "
                     f"say which bytes were reviewed and why")

    # -- non-boilerplate disclosures ---------------------------------------- #
    for field, floor in MIN_DISTINCT_TOKENS.items():
        text = get(field)
        if not isinstance(text, str) or not text.strip():
            fail(f"{field} is empty")
            continue
        distinct = {w.lower() for w in WORD.findall(text)}
        if len(distinct) < floor:
            fail(f"{field} uses only {len(distinct)} distinct tokens (minimum "
                 f"{floor}): it reads as boilerplate or as padding by repetition")

    # -- promotion language and confidence voting --------------------------- #
    for path, text in walk_strings(record):
        low = text.lower()
        for phrase in PROMOTION_PHRASES:
            if phrase in low:
                fail(f"promotion language {phrase!r} in field {path}: a review "
                     f"record states a verdict on an object and may not describe "
                     f"any premise, obligation or gate as moved")
        for pattern in CONFIDENCE_PATTERNS:
            hit = pattern.search(text)
            if hit:
                fail(f"confidence voting {hit.group(0)!r} in field {path}: "
                     f"R17 §4 — 'No confidence voting.'")

    return problems


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def check_all(records_dir: str, schema_path: str, queue_path: str) -> list[str]:
    schema = load_json(schema_path)
    queue, problems = queue_index(queue_path)

    if not os.path.isdir(records_dir):
        return problems + [f"{records_dir}: records directory does not exist"]

    filenames = sorted(fn for fn in os.listdir(records_dir)
                       if fn.endswith(".json"))
    seen_ids: dict[str, str] = {}
    seen_text: dict[str, dict[str, str]] = {f: {} for f in MIN_DISTINCT_TOKENS}

    for fn in filenames:
        path = os.path.join(records_dir, fn)
        try:
            record = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{fn}: not readable as JSON: {exc}")
            continue
        if not isinstance(record, dict):
            problems.append(f"{fn}: top level is {type(record).__name__}, "
                            f"expected an object")
            continue

        problems += check_record(record, schema, queue, fn)

        review_id = record.get("review_id")
        if isinstance(review_id, str):
            if review_id in seen_ids:
                problems.append(f"{fn}: review_id {review_id!r} is already used "
                                f"by {seen_ids[review_id]}")
            else:
                seen_ids[review_id] = fn

        # Boilerplate shared across records is the failure mode this catches:
        # one honest disclosure copied into every subsequent record is not a
        # disclosure.
        for field in MIN_DISTINCT_TOKENS:
            text = record.get(field)
            if isinstance(text, str) and text.strip():
                key = normalize(text)
                if key in seen_text[field]:
                    problems.append(
                        f"{fn}: {field} is verbatim identical to the {field} in "
                        f"{seen_text[field][key]}: a disclosure copied between "
                        f"records is boilerplate, not a disclosure")
                else:
                    seen_text[field][key] = fn

    return problems


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    records_dir, schema_path, queue_path = RECORDS_DIR, SCHEMA_PATH, QUEUE_PATH
    while argv:
        flag = argv.pop(0)
        if flag == "--records" and argv:
            records_dir = argv.pop(0)
        elif flag == "--schema" and argv:
            schema_path = argv.pop(0)
        elif flag == "--queue" and argv:
            queue_path = argv.pop(0)
        else:
            print(f"usage: reviews_check.py [--records DIR] [--schema PATH] "
                  f"[--queue PATH]  (unrecognized: {flag!r})", file=sys.stderr)
            return 2

    problems = check_all(records_dir, schema_path, queue_path)
    count = len([fn for fn in sorted(os.listdir(records_dir))
                 if fn.endswith(".json")]) if os.path.isdir(records_dir) else 0
    for problem in problems:
        print(f"PROBLEM  {problem}")
    print(f"reviews_check: {count} record(s) in "
          f"{os.path.relpath(records_dir, ROOT) if records_dir.startswith(ROOT) else records_dir}, "
          f"{len(problems)} problem(s).")
    if not problems:
        print("reviews_check: form only. No mathematics verified, no "
              "independence credit awarded, no gate moved.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
