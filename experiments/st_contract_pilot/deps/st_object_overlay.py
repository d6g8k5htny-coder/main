"""Pure, bounded object-member overlays for ULHI R2 case construction.

This is not RFC 6902, a mathematical diagnostic, or an evidence-admission gate.
It evaluates no formulas and performs no filesystem or network operations.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

MAX_DEPTH = 64
MAX_NODES = 16384
MAX_OPERATIONS = 256
MAX_JSON_CHARS = 1048576
_BAD_ESCAPE = re.compile(r'~(?:[^01]|$)')


class OverlayError(ValueError):
    """Input refusal with a stable code, without echoing input values."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _clone(value: Any, budget: list[int], ancestors: set[int], depth: int = 0) -> Any:
    """Validate plain JSON values and copy every container independently."""
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise OverlayError('INPUT_LIMIT')
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float:
        if not math.isfinite(value):
            raise OverlayError('NONFINITE_NUMBER')
        return value
    if kind not in (dict, list):
        raise OverlayError('NON_JSON_VALUE')
    identity = id(value)
    if identity in ancestors:
        raise OverlayError('CYCLIC_INPUT')
    ancestors.add(identity)
    try:
        if kind is list:
            return [_clone(item, budget, ancestors, depth + 1) for item in value]
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise OverlayError('NON_STRING_KEY')
            result[key] = _clone(item, budget, ancestors, depth + 1)
        return result
    finally:
        ancestors.remove(identity)


def _tokens(pointer: str) -> tuple[str, ...]:
    if type(pointer) is not str or not pointer.startswith('/'):
        raise OverlayError('INVALID_POINTER')
    if len(pointer) > MAX_JSON_CHARS:
        raise OverlayError('INPUT_LIMIT')
    if _BAD_ESCAPE.search(pointer):
        raise OverlayError('INVALID_POINTER')
    # The order matters: ~01 denotes the literal key ~1, not the key /.
    tokens = tuple(token.replace('~1', '/').replace('~0', '~')
                   for token in pointer[1:].split('/'))
    if len(tokens) > MAX_DEPTH + 1:
        raise OverlayError('INPUT_LIMIT')
    return tokens


def _parent(root: dict[str, Any], tokens: tuple[str, ...]) -> dict[str, Any]:
    current = root
    for token in tokens[:-1]:
        if type(current) is not dict:
            raise OverlayError('NON_OBJECT_PARENT')
        if token not in current:
            raise OverlayError('MISSING_PARENT')
        current = current[token]
    if type(current) is not dict:
        raise OverlayError('NON_OBJECT_PARENT')
    return current


def apply_overlay(base: dict[str, Any], set_values: dict[str, Any],
                  remove_paths: list[str]) -> dict[str, Any]:
    """Return an independent copy, or refuse without modifying any input.

    Set adds or replaces a terminal member; it never deep-merges. Remove needs
    an existing member. Parents must already be objects in the original base.
    All decoded targets must be disjoint: equality and ancestry are refused.
    Only built-in JSON types are accepted. Arrays may be values, not parents.
    Caps are implementation limits, not new mathematical assumptions.
    """
    if type(base) is not dict:
        raise OverlayError('INVALID_ROOT')
    if type(set_values) is not dict or type(remove_paths) is not list:
        raise OverlayError('INVALID_OPERATIONS')
    if len(set_values) + len(remove_paths) > MAX_OPERATIONS:
        raise OverlayError('OPERATION_LIMIT')
    assignments = [(_tokens(path), value) for path, value in set_values.items()]
    deletions = [_tokens(path) for path in remove_paths]
    targets = [tokens for tokens, _ in assignments] + deletions
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            common = min(len(left), len(right))
            if left[:common] == right[:common]:
                raise OverlayError('CONFLICTING_TARGETS')
    budget = [MAX_NODES]
    result = _clone(base, budget, set())
    prepared_sets = []
    prepared_removes = []
    for tokens, value in assignments:
        parent = _parent(result, tokens)
        prepared_sets.append((parent, tokens[-1], _clone(value, budget, set())))
    for tokens in deletions:
        parent = _parent(result, tokens)
        if tokens[-1] not in parent:
            raise OverlayError('MISSING_REMOVE')
        prepared_removes.append((parent, tokens[-1]))
    # Nothing is applied until all pointers, parents, values and removals pass.
    for parent, key, value in prepared_sets:
        parent[key] = value
    for parent, key in prepared_removes:
        del parent[key]
    return result


def loads_object(text: str) -> dict[str, Any]:
    """Parse a bounded JSON object, refusing duplicate keys and nonfinite values."""
    if type(text) is not str:
        raise OverlayError('INVALID_JSON_INPUT')
    if len(text) > MAX_JSON_CHARS:
        raise OverlayError('INPUT_LIMIT')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise OverlayError('DUPLICATE_KEY')
            result[key] = value
        return result

    def constant(_):
        raise OverlayError('NONFINITE_NUMBER')

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except OverlayError:
        raise
    except RecursionError:
        raise OverlayError('INPUT_LIMIT') from None
    except (ValueError, TypeError):
        raise OverlayError('INVALID_JSON') from None
    if type(value) is not dict:
        raise OverlayError('INVALID_ROOT')
    return _clone(value, [MAX_NODES], set())
