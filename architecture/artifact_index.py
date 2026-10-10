"""Project a pinned native Artifact Index capture into fixed public fields.

This module treats captures as inert JSON data. It performs no I/O, interprets
no cell content, and authenticates no external custody or scientific status.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
import math
import re


STDIN_LIMIT = 16 * 1024 * 1024
CAPTURE_LIMIT = 4 * 1024 * 1024
DEPTH_LIMIT = 64
COLLECTION_LIMIT = 4096
CELL_LIMIT = 8192
GRID_ROW_LIMIT = 10_000_000

HEADER = (
    "Artifact ID", "Title", "Org", "Class", "Priority", "Topics / object tags",
    "Status", "Authority / canonical impact", "Dependencies or supersession",
    "Modified UTC", "Source", "Notes",
)
PUBLIC_COLUMNS = (
    "Artifact ID", "Title", "Org", "Class", "Status",
    "Authority / canonical impact", "Dependencies or supersession", "Modified UTC",
)
PUBLIC_INDICES = tuple(HEADER.index(column) for column in PUBLIC_COLUMNS)
ROLES = frozenset(("source", "review", "kernel", "computation", "alignment", "teaching-source"))
PIN_KEYS = frozenset((
    "capture_sha256", "metadata_sha256", "policy_sha256",
    "bindings_sha256", "graph_nodes_sha256",
))
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}")
SHA256 = re.compile(r"[0-9a-f]{64}")
UTC_SECOND = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})Z")


class ArtifactIndexRefused(ValueError):
    """An opaque refusal whose message never includes caller data."""

    def __init__(self):
        super().__init__("ARTIFACT_INDEX_REFUSED")


def _require(condition):
    if not condition:
        raise ArtifactIndexRefused()


def _validate_json_value(value, depth=0):
    """Validate JSON types, finite numbers, Unicode, and container depth."""
    kind = type(value)
    if kind is dict:
        _require(depth < DEPTH_LIMIT)
        for key, child in value.items():
            _require(type(key) is str)
            key.encode("utf-8", errors="strict")
            _validate_json_value(child, depth + 1)
    elif kind is list:
        _require(depth < DEPTH_LIMIT)
        for child in value:
            _validate_json_value(child, depth + 1)
    elif kind is str:
        value.encode("utf-8", errors="strict")
    elif kind is float:
        _require(math.isfinite(value))
    else:
        _require(kind in (int, bool, type(None)))


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _nonfinite_constant(_token):
    raise ArtifactIndexRefused()


def _finite_float(token):
    value = float(token)
    _require(math.isfinite(value))
    return value


def _strict_json(raw, limit):
    try:
        _require(type(raw) is bytes and 0 < len(raw) <= limit)
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_object_pairs,
            parse_constant=_nonfinite_constant,
            parse_float=_finite_float,
        )
        _validate_json_value(value)
        return value
    except Exception:
        raise ArtifactIndexRefused() from None


def parse_packet(raw):
    """Read one bounded strict JSON packet from caller-supplied bytes."""
    return _strict_json(raw, STDIN_LIMIT)


def canonical_json(value):
    """Return deterministic ASCII JSON bytes, preserving array order/types."""
    try:
        _validate_json_value(value)
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except Exception:
        raise ArtifactIndexRefused() from None


def _object(value, keys=None):
    _require(type(value) is dict)
    if keys is not None:
        _require(value.keys() == keys)
    return value


def _integer(value, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum)
    if maximum is not None:
        _require(value <= maximum)
    return value


def _identifier(value):
    _require(type(value) is str and IDENTIFIER.fullmatch(value) is not None)
    return value


def _bounded_list(value, minimum=0, maximum=COLLECTION_LIMIT):
    _require(type(value) is list and minimum <= len(value) <= maximum)
    return value


def _unique_identifiers(value, minimum=0):
    values = _bounded_list(value, minimum)
    seen = set()
    for item in values:
        _identifier(item)
        _require(item not in seen)
        seen.add(item)
    return seen


def _expected_pins(expected):
    _object(expected, PIN_KEYS)
    for value in expected.values():
        _require(type(value) is str and SHA256.fullmatch(value) is not None)


def _capture_metadata(capture):
    _object(capture, {"capture_id", "captured_at", "source", "raw_base64"})
    _identifier(capture["capture_id"])
    timestamp = capture["captured_at"]
    _require(type(timestamp) is str)
    match = UTC_SECOND.fullmatch(timestamp)
    _require(match is not None)
    # A numeric constructor checks calendar validity without parsing cell text.
    datetime(*(int(part) for part in match.groups()))
    source = _object(capture["source"], {
        "spreadsheet_id", "worksheet_id", "title", "ranges", "cell_fields",
    })
    _identifier(source["spreadsheet_id"])
    _integer(source["worksheet_id"])
    title = source["title"]
    _require(type(title) is str and 0 < len(title.encode("utf-8")) <= 256)
    _require("'" not in title and not any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in title))
    _require(source["cell_fields"] == "formattedValue" and type(source["cell_fields"]) is str)
    ranges = _bounded_list(source["ranges"], 1, COLLECTION_LIMIT + 1)
    range_pattern = re.compile(r"'" + re.escape(title) + r"'!A([1-9][0-9]*):L([1-9][0-9]*)")
    intervals = []
    prior_end = 0
    for declared in ranges:
        _require(type(declared) is str)
        match = range_pattern.fullmatch(declared)
        _require(match is not None)
        # Bounds precede integer conversion, including enormous decimal text.
        _require(all(len(part) <= 8 for part in match.groups()))
        first, last = (int(part) for part in match.groups())
        _require(prior_end < first <= last <= GRID_ROW_LIMIT)
        intervals.append((first - 1, last))
        prior_end = last
    _require(sum(stop - start for start, stop in intervals) <= COLLECTION_LIMIT + 1)
    return source, intervals


def _capture_bytes(encoded):
    _require(type(encoded) is str and 0 < len(encoded) <= 4 * ((CAPTURE_LIMIT + 2) // 3))
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    _require(0 < len(raw) <= CAPTURE_LIMIT)
    _require(base64.b64encode(raw).decode("ascii") == encoded)
    return raw


def _native_rows(native, source, intervals):
    _object(native)
    _require(type(native.get("spreadsheetId")) is str and native["spreadsheetId"] == source["spreadsheet_id"])
    sheets = _bounded_list(native.get("sheets"), 1, 1)
    sheet = _object(sheets[0])
    properties = _object(sheet.get("properties"))
    _integer(properties.get("sheetId"))
    _require(properties["sheetId"] == source["worksheet_id"])
    _require(type(properties.get("title")) is str and properties["title"] == source["title"])
    grid = _object(properties.get("gridProperties"))
    row_count = _integer(grid.get("rowCount"), 1, GRID_ROW_LIMIT)
    _require(_integer(grid.get("columnCount"), 1) == len(HEADER))
    blocks = _bounded_list(sheet.get("data"), len(intervals), len(intervals))
    rows_by_id = {}
    header_count = 0
    for block, (start, stop) in zip(blocks, intervals):
        _object(block)
        _require(stop <= row_count)
        _require(_integer(block.get("startRow", 0)) == start)
        _require(_integer(block.get("startColumn", 0)) == 0)
        rows = _bounded_list(block.get("rowData"), stop - start, stop - start)
        for offset, row in enumerate(rows):
            _object(row)
            cells = _bounded_list(row.get("values"), len(HEADER), len(HEADER))
            values = []
            for cell in cells:
                _object(cell)
                value = cell.get("formattedValue", "")
                _require(type(value) is str and len(value.encode("utf-8")) <= CELL_LIMIT)
                values.append(value)
            position = start + offset
            if position == 0:
                _require(values == list(HEADER))
                header_count += 1
            else:
                artifact_id = _identifier(values[0])
                _require(artifact_id not in rows_by_id)
                rows_by_id[artifact_id] = (position + 1, values)
                _require(len(rows_by_id) <= COLLECTION_LIMIT)
    _require(header_count == 1)
    return rows_by_id


def _project(packet, expected):
    _validate_json_value(packet)
    _expected_pins(expected)
    _object(packet, {"schema_version", "capture", "policy", "bindings", "graph_nodes"})
    _require(_integer(packet["schema_version"]) == 1)
    capture = packet["capture"]
    source, intervals = _capture_metadata(capture)
    raw = _capture_bytes(capture["raw_base64"])
    metadata = {key: value for key, value in capture.items() if key != "raw_base64"}
    identities = {
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "metadata_sha256": hashlib.sha256(canonical_json(metadata)).hexdigest(),
        "policy_sha256": hashlib.sha256(canonical_json(packet["policy"])).hexdigest(),
        "bindings_sha256": hashlib.sha256(canonical_json(packet["bindings"])).hexdigest(),
        "graph_nodes_sha256": hashlib.sha256(canonical_json(packet["graph_nodes"])).hexdigest(),
    }
    _require(identities == expected)
    policy = _object(packet["policy"], {"schema_version", "artifact_ids", "columns"})
    _require(_integer(policy["schema_version"]) == 1)
    _require(type(policy["columns"]) is list and policy["columns"] == list(PUBLIC_COLUMNS))
    selected = _unique_identifiers(policy["artifact_ids"], 1)
    graph_nodes = _unique_identifiers(packet["graph_nodes"])
    bindings = []
    pairs = set()
    bound_nodes = set()
    for binding in _bounded_list(packet["bindings"]):
        _object(binding, {"node_id", "artifact_id", "roles"})
        node = _identifier(binding["node_id"])
        artifact = _identifier(binding["artifact_id"])
        _require(node in graph_nodes and artifact in selected)
        roles = _bounded_list(binding["roles"], 1, len(ROLES))
        _require(all(type(role) is str and role in ROLES for role in roles))
        _require(len(set(roles)) == len(roles))
        pair = (node, artifact)
        _require(pair not in pairs)
        pairs.add(pair)
        bound_nodes.add(node)
        bindings.append({"node_id": node, "artifact_id": artifact, "roles": sorted(roles)})
    native = _strict_json(raw, CAPTURE_LIMIT)
    native_rows = _native_rows(native, source, intervals)
    _require(selected <= native_rows.keys())
    rows = []
    for artifact in sorted(selected):
        source_row, values = native_rows[artifact]
        rows.append({
            "artifact_id": artifact, "source_row": source_row,
            "fields": {column: values[index] for column, index in zip(PUBLIC_COLUMNS, PUBLIC_INDICES)},
        })
    return {
        "schema_version": 1, "scientific_effect": "NONE",
        "scientific_status_authority": False, "custody": "unknown",
        "source": {
            "capture_id": capture["capture_id"], "captured_at": capture["captured_at"],
            "spreadsheet_id": source["spreadsheet_id"], "worksheet_id": source["worksheet_id"],
            "worksheet_title": source["title"], "ranges": list(source["ranges"]),
            "cell_fields": source["cell_fields"], "representation": "native-api-formattedValue-json",
        },
        "identities": identities, "columns": list(PUBLIC_COLUMNS), "rows": rows,
        "bindings": sorted(bindings, key=lambda item: (item["node_id"], item["artifact_id"])),
        "unmapped_nodes": sorted(graph_nodes - bound_nodes),
    }


def project_artifact_index(packet, expected):
    """Return the exact public projection, or raise an opaque refusal.

    Independent expected pins are mandatory; no expectation is learned from
    the packet. Input objects are not mutated and private data is not returned.
    """
    try:
        return _project(packet, expected)
    except Exception:
        raise ArtifactIndexRefused() from None
