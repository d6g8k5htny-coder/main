"""Pure dependency helpers for SDGA Task A.

Native acquisition, compilation, observation and controller CLI interfaces are
not implemented here. An empty pure result is not production coverage.
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
