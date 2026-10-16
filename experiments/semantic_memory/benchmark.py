"""Task1 synthetic fixture validation only; no model, retrieval or installation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

META_LIMIT = 4 * 2**20
TEXT_LIMIT = 384 * 2**10
SOURCE_KEYS = frozenset(("source_id", "revision_id", "source_sha256", "extraction_sha256",
    "extractor_version", "text_sha256", "text_bytes", "content_ref", "observed_at",
    "policy_epoch", "access_state", "exclusion_state", "supersession_state"))
CASE_KEYS = frozenset(("case_id", "stratum", "query", "exact_source_id", "exact_ref", "task_scope"))
SPAN_KEYS = frozenset(("source_id", "revision_id", "text_sha256", "start_byte", "end_byte", "sha256"))
MANIFEST_KEYS = frozenset(("schema_version", "seed", "text_transform", "sources", "case_files",
    "development_case_ids", "heldout_case_ids", "generation_case_ids", "query_order",
    "schedule_algorithm", "baseline_definitions", "gates", "pending_obligations"))
CASE_FILES = ("development-cases.json", "heldout-cases.json", "heldout-oracle.json")
STRATA = {"P": "paraphrase", "D": "deep", "V": "confusable_revision",
          "X": "exact_ref", "N": "no_answer"}
DEV_IDS = frozenset("DEV%02d" % i for i in range(1, 9))
HELD_IDS = frozenset(p + "%02d" % i for p in STRATA for i in range(1, 9))
ABSENT_IDS = frozenset("N%02d" % i for i in range(1, 9)) | frozenset(
    ("DEV05", "DEV06", "DEV07", "DEV08"))
BASELINES = {"E": "first12000_characters_then_full_prompt_admission",
    "L": "exact_ref_first_then_deterministic_literal_or_SQLite_FTS5",
    "S": "exact_ref_first_then_literal_fallback_and_semantic_top4"}
GATES = {"exact_ref_successes": 8, "exact_ref_cases": 8, "citation_fidelity": 1.0,
    "no_answer_generation_abstentions_per_profile": 4, "answerable_retrieval_min": 29,
    "answerable_cases": 32, "paraphrase_deep_recall_min": 14, "paraphrase_deep_cases": 16,
    "improvement_over_better_baseline_min": 2, "generation_score_min": 18,
    "generation_cases_per_profile": 20, "unsupported_answer_max": 0,
    "semantic_vs_literal_warm_complete_task_p95_ratio_max": 1.25,
    "break_even_queries_max": 500, "peak_experiment_rss_bytes_max": 12 * 2**30,
    "fairness_p95_increase_ratio_max": 0.1, "starvation_max": 0, "task1_quality_measured": False}
OBLIGATIONS = ["tokenizer_parity", "final_chunk_count_le_128", "500_token_chunks",
    "semantic_retrieval", "quality_cost_fairness", "actual_W_C_integration"]
SCHEDULE = "materialized-v1; authoring Python3 random.Random(20261008) shuffle; rows are authority"


def _require(condition):
    if not condition:
        raise ValueError("fixture contract")


def _keys(value, fields):
    _require(type(value) is dict and set(value) == set(fields))


def _integer(value, low=0, high=2**63 - 1):
    _require(type(value) is int and low <= value <= high)
    return value


def _text(value, maximum=2048):
    _require(type(value) is str and bool(value.strip()))
    _require(len(value.encode("utf-8")) <= maximum)
    return value


def _identity(value):
    _require(type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) is not None)
    return value


def _hash(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_report(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _same(value, expected):
    # Serialized equality preserves int/float/bool distinctions.
    _require(canonical_report(value) == canonical_report(expected))


def _strict_json(raw):
    _require(type(raw) is bytes and len(raw) <= META_LIMIT)
    text = raw.decode("utf-8")
    depth = 0
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            _require(depth <= 64)
        elif character in "]}":
            depth -= 1
            _require(depth >= 0)
    _require(not quoted and depth == 0)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result)
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite JSON")

    value = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
    pending = [(value, 0)]
    while pending:
        item, level = pending.pop()
        if type(item) in (dict, list):
            _require(level < 64)
            if type(item) is dict:
                for key in item:
                    key.encode("utf-8")
                children = item.values()
            else:
                children = item
            pending.extend((child, level + 1) for child in children)
        elif type(item) is str:
            item.encode("utf-8")  # Reject escaped unpaired surrogates.
        elif type(item) is float:
            _require(math.isfinite(item))
        else:
            _require(item is None or type(item) in (bool, int))
    return value


def _relative(name):
    _text(name, 1024)
    _require(not name.startswith("/") and "\\" not in name and ":" not in name)
    pieces = name.split("/")
    _require(all(piece not in ("", ".", "..") and not any(ord(c) < 32 for c in piece)
                 for piece in pieces))
    return pieces


def _directory(path):
    raw = os.fspath(path)
    _require(type(raw) is str and 0 < len(raw.encode("utf-8")) <= 4096 and "\0" not in raw)
    absolute = raw.startswith("/")
    pieces = raw.split("/")[1:] if absolute else raw.split("/")
    _require(all(piece not in ("", ".", "..") for piece in pieces))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open("/" if absolute else ".", flags)
    try:
        for piece in pieces:
            child = os.open(piece, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(root, name, limit):
    pieces = _relative(name)
    fd = os.dup(root)
    try:
        for piece in pieces[:-1]:
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        source = os.open(pieces[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=fd)
        try:
            before = os.fstat(source)
            _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit)
            chunks = []
            size = 0
            while True:
                block = os.read(source, min(65536, limit + 1 - size))
                if not block:
                    break
                chunks.append(block)
                size += len(block)
                _require(size <= limit)
            after = os.fstat(source)
            fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
            _require(all(getattr(before, field) == getattr(after, field) for field in fields))
            _require(size == before.st_size)
            return b"".join(chunks)
        finally:
            os.close(source)
    finally:
        os.close(fd)


def _inventory(fd, expected, directories=()):
    seen = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            _require(len(seen) < len(expected) and entry.name in expected and entry.name not in seen)
            seen.add(entry.name)
            item = os.stat(entry.name, dir_fd=fd, follow_symlinks=False)
            _require(stat.S_ISDIR(item.st_mode) if entry.name in directories else stat.S_ISREG(item.st_mode))
    _require(seen == set(expected))


def _source(value):
    _keys(value, SOURCE_KEYS)
    _identity(value["source_id"])
    _identity(value["revision_id"])
    for key in ("source_sha256", "extraction_sha256", "text_sha256"):
        _hash(value[key])
    _integer(value["text_bytes"], 1, TEXT_LIMIT)
    _integer(value["policy_epoch"], 1)
    _require(type(value["observed_at"]) is str and re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value["observed_at"]) is not None)
    _require(value["extractor_version"] == "synthetic-identity-utf8/v1")
    _require(value["access_state"] == "available" and value["exclusion_state"] == "included")
    _require(value["supersession_state"] in ("current", "superseded"))
    _relative(value["content_ref"])
    _require(re.fullmatch(r"corpus/[A-Za-z0-9_.-]+\.txt", value["content_ref"]) is not None)
    return value["source_id"], value["revision_id"]


def _case(value):
    _keys(value, CASE_KEYS)
    identifier = _identity(value["case_id"])
    _text(value["query"])
    _require(value["task_scope"] == "focused_excerpt")
    if identifier in DEV_IDS:
        _require(value["stratum"] == "development")
    else:
        _require(identifier in HELD_IDS and value["stratum"] == STRATA[identifier[0]])
    source, revision = value["exact_source_id"], value["exact_ref"]
    _require((source is None) == (revision is None))
    if source is not None:
        _identity(source)
        _identity(revision)
    if identifier in DEV_IDS:
        _require(source is None)
    elif value["stratum"] == "exact_ref":
        _require(source is not None)
    else:
        _require(source is None)
    if identifier in ABSENT_IDS:
        query = value["query"].casefold()
        _require(not any(cue in query for cue in ("absent", "no-answer", "unanswerable", identifier.casefold())))
    return value


def case_request(case: dict, profile: str) -> dict:
    """Project synthetic request fields; does not render or call either worker."""
    _case(case)
    _require(type(profile) is str and profile in ("W", "C"))
    return {key: case[key] for key in ("query", "exact_source_id", "exact_ref", "task_scope")} | {
        "profile": profile}


def _case_document(value, expected):
    _keys(value, ("schema_version", "cases"))
    _integer(value["schema_version"], 1, 1)
    _require(type(value["cases"]) is list and len(value["cases"]) == len(expected))
    cases = {}
    for row in value["cases"]:
        _case(row)
        identifier = row["case_id"]
        _require(identifier in expected and identifier not in cases)
        cases[identifier] = row
    _require(set(cases) == set(expected))
    return cases


def _gold(records, expected, cases, sources, texts):
    _require(type(records) is list and len(records) == len(expected))
    seen = set()
    for row in records:
        _keys(row, ("case_id", "no_answer", "expected_answer", "spans"))
        identifier = row["case_id"]
        _require(type(identifier) is str and identifier in expected and identifier not in seen)
        seen.add(identifier)
        _require(type(row["no_answer"]) is bool and row["no_answer"] == (identifier in ABSENT_IDS))
        _require(type(row["spans"]) is list and len(row["spans"]) <= 32)
        if row["no_answer"]:
            _require(row["expected_answer"] is None and row["spans"] == [])
            continue
        _text(row["expected_answer"], 8192)
        _require(bool(row["spans"]))
        case = cases[identifier]
        for span in row["spans"]:
            _keys(span, SPAN_KEYS)
            key = (_identity(span["source_id"]), _identity(span["revision_id"]))
            _require(key in sources)
            source, raw = sources[key], texts[key]
            _require(_hash(span["text_sha256"]) == source["text_sha256"])
            start = _integer(span["start_byte"], 0, len(raw))
            end = _integer(span["end_byte"], start + 1, len(raw))
            piece = raw[start:end]
            _require(bool(piece.strip()) and _hash(span["sha256"]) == _sha(piece))
            prefix = raw[:start].decode("utf-8")
            piece.decode("utf-8")
            raw[end:].decode("utf-8")
            if case["stratum"] == "deep":
                _require(len(prefix) > 12000 and len(prefix) != start)
                _require(piece.strip() not in raw.decode("utf-8")[:12000].encode("utf-8"))
            if case["stratum"] == "confusable_revision":
                _require(source["supersession_state"] == "current")
                older = [s for other, s in sources.items() if other[0] == key[0]
                         and other != key and s["supersession_state"] == "superseded"]
                _require(bool(older) and all(s["text_sha256"] != source["text_sha256"] for s in older))
            if case["exact_source_id"] is not None:
                _require(key == (case["exact_source_id"], case["exact_ref"]))
    _require(seen == set(expected))


def _representation(controls):
    _require(type(controls) is list and len(controls) == 1)
    value = controls[0]
    _keys(value, ("source_representation", "source_sha256", "extracted_text",
                  "text_sha256", "extractor_version"))
    raw = _text(value["source_representation"], META_LIMIT).encode("utf-8")
    source = _strict_json(raw)
    _keys(source, ("number", "title", "body"))
    _integer(source["number"], 1)
    _text(source["title"])
    _text(source["body"])
    _require(value["extractor_version"] == "synthetic-github-body-title/v1")
    _require(_hash(value["source_sha256"]) == _sha(raw))
    derived = (source["body"] + "\n" + source["title"]).encode("utf-8")
    _require(type(value["extracted_text"]) is str and value["extracted_text"].encode("utf-8") == derived)
    _require(_hash(value["text_sha256"]) == _sha(derived))


def validate_fixtures(fixture_dir: Path) -> dict:
    """Validate declared public fixture bytes; make no runtime or science claim."""
    root = _directory(fixture_dir)
    try:
        _inventory(root, {"corpus", "corpus-manifest.json", *CASE_FILES}, {"corpus"})
        raw_manifest = _read(root, "corpus-manifest.json", META_LIMIT)
        manifest = _strict_json(raw_manifest)
        _keys(manifest, MANIFEST_KEYS)
        _integer(manifest["schema_version"], 1, 1)
        _integer(manifest["seed"], 20261008, 20261008)
        _same(manifest["text_transform"], {"version": "synthetic-identity-utf8/v1", "normalization": "none"})
        _same(manifest["baseline_definitions"], BASELINES)
        _same(manifest["gates"], GATES)
        _same(manifest["pending_obligations"], OBLIGATIONS)
        _require(manifest["schedule_algorithm"] == SCHEDULE)
        _require(type(manifest["sources"]) is list and len(manifest["sources"]) == 32)
        sources, texts, files, projection = {}, {}, {}, []
        files["corpus-manifest.json"] = {"bytes": len(raw_manifest), "sha256": _sha(raw_manifest)}
        paths = set()
        total = 0
        for row in manifest["sources"]:
            key = _source(row)
            path = row["content_ref"]
            _require(key not in sources and path not in paths)
            paths.add(path)
            raw = _read(root, path, TEXT_LIMIT)
            _require(len(raw) == row["text_bytes"])
            total += len(raw)
            _require(total <= TEXT_LIMIT)
            digest = _sha(raw)
            _require(all(row[name] == digest for name in ("text_sha256", "source_sha256", "extraction_sha256")))
            text = raw.decode("utf-8")
            sources[key], texts[key] = row, raw
            files[path] = {"bytes": len(raw), "sha256": digest}
            projection.append(dict(row, text=text))
        directory = os.open("corpus", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=root)
        try:
            _inventory(directory, {path.split("/")[1] for path in paths})
        finally:
            os.close(directory)
        _keys(manifest["case_files"], CASE_FILES)
        documents = {}
        for name in CASE_FILES:
            identity = manifest["case_files"][name]
            _keys(identity, ("bytes", "sha256"))
            _integer(identity["bytes"], 1, META_LIMIT)
            _hash(identity["sha256"])
            raw = _read(root, name, META_LIMIT)
            _require(len(raw) == identity["bytes"] and _sha(raw) == identity["sha256"])
            files[name] = {"bytes": len(raw), "sha256": _sha(raw)}
            documents[name] = _strict_json(raw)
        development = _case_document(documents["development-cases.json"], DEV_IDS)
        heldout = _case_document(documents["heldout-cases.json"], HELD_IDS)
        cases = development | heldout
        for case in cases.values():
            if case["exact_source_id"] is not None:
                _require((case["exact_source_id"], case["exact_ref"]) in sources)
        for field, expected in (("development_case_ids", DEV_IDS), ("heldout_case_ids", HELD_IDS)):
            values = manifest[field]
            _require(type(values) is list and len(values) == len(expected)
                     and all(type(value) is str for value in values) and set(values) == set(expected))
        generation = manifest["generation_case_ids"]
        _require(type(generation) is list and len(generation) == 20
                 and all(type(value) is str and value in HELD_IDS for value in generation)
                 and len(set(generation)) == 20)
        _require(all(sum(value.startswith(prefix) for value in generation) == 4 for prefix in STRATA))
        order = manifest["query_order"]
        _require(type(order) is list and len(order) == 40)
        seen = set()
        for row in order:
            _keys(row, ("case_id", "arms", "profiles"))
            _require(type(row["case_id"]) is str and row["case_id"] in HELD_IDS and row["case_id"] not in seen)
            seen.add(row["case_id"])
            for name, expected in (("arms", {"E", "L", "S"}), ("profiles", {"W", "C"})):
                _require(type(row[name]) is list and len(row[name]) == len(expected)
                         and all(type(value) is str for value in row[name]) and set(row[name]) == expected)
        oracle = documents["heldout-oracle.json"]
        _keys(oracle, ("schema_version", "oracle_only_sentinel", "records", "development_records", "representation_controls"))
        _integer(oracle["schema_version"], 1, 1)
        _text(oracle["oracle_only_sentinel"])
        _gold(oracle["records"], HELD_IDS, cases, sources, texts)
        _gold(oracle["development_records"], DEV_IDS, cases, sources, texts)
        _representation(oracle["representation_controls"])
        requests = {identifier: {profile: case_request(case, profile) for profile in ("W", "C")}
                    for identifier, case in cases.items()}
        return {"schema_version": 1, "scientific_effect": "NONE", "scientific_status_authority": False,
            "fixture_manifest_sha256": _sha(raw_manifest), "files": files,
            "counts": {"documents": 32, "text_bytes": total, "development": 8, "heldout": 40,
                       "by_stratum": {value: 8 for value in STRATA.values()}, "generation": 20},
            "seed": 20261008, "query_order": order, "baseline_definitions": manifest["baseline_definitions"],
            "gates": manifest["gates"], "corpus_projection": projection, "request_projections": requests,
            "validation": {"inventory": "passed", "gold": "passed", "deep": "passed", "representation": "passed"},
            "pending": {"tokenizer_parity": "NOT_EVALUATED", "chunk_count": None,
                        "retrieval": "NOT_EVALUATED", "quality_cost_fairness": "NOT_EVALUATED",
                        "installed_consumers": "NOT_EVALUATED"}}
    finally:
        os.close(root)


def admitted_corpus(fixture_dir: Path) -> list[dict]:
    """Admit file identities only; old annotations remain, no live policy runs."""
    return validate_fixtures(fixture_dir)["corpus_projection"]


def verify_report(fixture_dir: Path, report_path: Path) -> bytes:
    """Verify an explicit regular report against recomputed canonical bytes."""
    path = os.fspath(report_path)
    _require(type(path) is str and bool(path) and not path.endswith("/"))
    parent, name = os.path.split(path)
    root = _directory(parent) if parent else os.open(
        ".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        supplied = _read(root, name, META_LIMIT)
    finally:
        os.close(root)
    _strict_json(supplied)
    expected = canonical_report(validate_fixtures(fixture_dir))
    _require(supplied == expected)
    return expected
