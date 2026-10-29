#!/usr/bin/env python3
"""Lexical Lean 4 declaration extractor for the CBZ-0 fuzzing harness and drift crawler.

Scientific effect: NONE. Works on the Layer A scanner's masked text, so
comments and string text can neither hide nor invent a declaration. For each
declaration it returns the kind, the namespace-qualified name, the binder text,
the type (the statement, for a theorem) and the value text. Signatures are
lexical: whitespace-normalized token text, not elaborated types. Elaborated
types need a Lean toolchain; that boundary is recorded in TOOLING.md.
"""
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_a_scan  # noqa: E402

KINDS = ('theorem', 'lemma', 'def', 'abbrev', 'instance', 'structure', 'class', 'inductive', 'axiom',
         'opaque', 'example')
MODIFIERS = ('private', 'protected', 'noncomputable', 'partial', 'unsafe', 'nonrec')
HEADER = re.compile(
    r'^[ \t]*(?:@\[[^\]\n]*\][ \t\n]*)*(?P<mods>(?:(?:' + '|'.join(MODIFIERS) + r')[ \t]+)*)'
    r'(?P<kind>' + '|'.join(KINDS) + r')(?![A-Za-z0-9_\'!?])', re.M)
SCOPE = re.compile(r'^[ \t]*(?P<word>namespace|section|end)(?:[ \t]+(?P<name>[^\s]+))?[ \t]*$', re.M)
OPEN = {'(': ')', '[': ']', '{': '}', '⟨': '⟩', '⦃': '⦄'}
CLOSE = {v: k for k, v in OPEN.items()}


def mask(text):
    masked, errors = layer_a_scan.Masker(text).run()
    return masked, errors


def read_name(masked, i):
    """Read a dotted Lean name starting at or after i; return (name, end) or (None, i)."""
    n = len(masked)
    while i < n and masked[i] in ' \t\n':
        i += 1
    start = i
    parts = []
    while i < n:
        if masked[i] == '«':
            j = masked.find('»', i + 1)
            if j < 0:
                return None, start
            parts.append(masked[i:j + 1])
            i = j + 1
        elif layer_a_scan.is_id_first(masked[i]):
            j = i + 1
            while j < n and layer_a_scan.is_id_rest(masked[j]):
                j += 1
            parts.append(masked[i:j])
            i = j
        else:
            break
        if i + 1 < n and masked[i] == '.' and (masked[i + 1] == '«' or layer_a_scan.is_id_first(masked[i + 1])):
            i += 1
            continue
        break
    if not parts:
        return None, start
    return '.'.join(parts), i


def split_top(text, stop_tokens):
    """Index of the first stop token at bracket depth 0, or None."""
    depth = 0
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth = max(depth - 1, 0)
        elif depth == 0:
            for token in stop_tokens:
                if text.startswith(token, i):
                    if token == '|' and not match_arm(text, i):
                        continue  # `|x|` inside a statement, not a pattern-matching arm
                    if token.isalpha():
                        before = text[i - 1] if i > 0 else ' '
                        after = text[i + len(token)] if i + len(token) < n else ' '
                        if layer_a_scan.is_id_rest(before) or layer_a_scan.is_id_rest(after):
                            continue
                    return i, token
        i += 1
    return None


def match_arm(text, i):
    """True when the `|` at i starts a pattern-matching arm.

    An arm needs a non-empty type before it (a top-level `:` followed by text)
    and reaches `=>` at bracket depth 0 before the next top-level `|`.
    """
    head = text[:i]
    colon = top_colon(head)
    if colon is None or not head[colon + 1:].strip():
        return False
    depth, j, n = 0, i + 1, len(text)
    while j < n:
        c = text[j]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth = max(depth - 1, 0)
        elif depth == 0 and c == '|':
            return False
        elif depth == 0 and text.startswith('=>', j):
            return True
        j += 1
    return False


def top_colon(text):
    """Index of the first `:` at depth 0 that is not part of `:=`, or None."""
    depth = 0
    for i, c in enumerate(text):
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth = max(depth - 1, 0)
        elif c == ':' and depth == 0 and not text.startswith(':=', i):
            return i
    return None


def normalize(text):
    return ' '.join(text.split())


def declarations(text):
    """Return (declarations, errors) for one Lean source text."""
    masked, errors = mask(text)
    scopes = []
    events = []
    for m in SCOPE.finditer(masked):
        events.append((m.start(), 'scope', m))
    headers = list(HEADER.finditer(masked))
    for m in headers:
        events.append((m.start(), 'decl', m))
    events.sort(key=lambda e: e[0])
    starts = [m.start() for m in headers]
    where = layer_a_scan.Located(text)
    out = []
    for index, (offset, kind, m) in enumerate(events):
        if kind == 'scope':
            word, name = m.group('word'), m.group('name')
            if word in ('namespace', 'section'):
                scopes.append((word, name or ''))
            elif scopes:
                scopes.pop()
            continue
        decl_kind = m.group('kind')
        name, after = read_name(masked, m.end())
        if decl_kind in ('instance', 'example') and (name is None or name in ('where',)):
            name, after = None, m.end()
        if name is None and decl_kind not in ('instance', 'example'):
            continue
        following = [s for s in starts if s > m.start()]
        next_scope = [e[0] for e in events if e[1] == 'scope' and e[0] > m.start()]
        end = min(following[:1] + next_scope[:1] + [len(masked)])
        body = masked[after:end]
        stop = split_top(body, (':=', ' where', '\nwhere', '|'))
        if stop is None:
            signature, value = body, None
        else:
            cut, token = stop
            signature = body[:cut]
            value = body[cut + len(token):] if token == ':=' else None
        colon = top_colon(signature)
        binders = signature[:colon] if colon is not None else signature
        statement = signature[colon + 1:] if colon is not None else ''
        prefix = '.'.join(n for w, n in scopes if w == 'namespace' and n)
        if name is None:
            qualified = None
        elif name.startswith('_root_.'):
            qualified = name[len('_root_.'):]
        else:
            qualified = (prefix + '.' + name) if prefix else name
        line, col = where(m.start('kind'))
        out.append({
            'kind': 'theorem' if decl_kind == 'lemma' else decl_kind,
            'name': name,
            'qualified': qualified,
            'modifiers': m.group('mods').split(),
            'line': line,
            'binders': normalize(binders),
            'type': normalize(statement),
            'value': normalize(value) if value is not None else None,
            'signature_sha256': hashlib.sha256(normalize(binders + ' : ' + statement).encode('utf-8')).hexdigest(),
        })
    return out, errors


def package_root(path, tree_paths):
    """Nearest ancestor directory of `path` that holds a lakefile, or '' for none."""
    parts = path.split('/')[:-1]
    while True:
        base = '/'.join(parts)
        for lake in ('lakefile.lean', 'lakefile.toml'):
            if (base + '/' + lake if base else lake) in tree_paths:
                return base
        if not parts:
            return None
        parts.pop()
