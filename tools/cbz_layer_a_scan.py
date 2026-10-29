#!/usr/bin/env python3
"""Layer A static scanner for Lean sources (CBZ-0 gate G1).

It reports, per file, every occurrence of the Layer A tokens below that is
outside comments, string literals and character literals:

    sorry, admit, sorryAx    placeholders (A1)
    axiom                    axiom declarations (A2)
    native_decide, +native   compiler-trusted decision (A2: adds Lean.ofReduceBool);
                             ``decide +native`` is reported as ``+native``
    ofReduceBool             direct use of the compiler-trust axiom

The lexer understands Lean 4 line comments (``--``), nested block comments
(``/- -/``, including ``/--`` and ``/-!``), string literals with escapes,
raw strings (``r"..."``, ``r#"..."#``) and character literals.  A token is a
maximal run of identifier characters; dotted names are split on ``.``, so
``Lean.sorryAx`` reports ``sorryAx`` while ``sorryish`` and ``my_admit`` report
nothing.

Every file under the given roots whose name ends in ``.lean`` is scanned.  A
file that cannot be decoded as UTF-8 is reported as an error, never skipped,
and an error makes the exit status nonzero.  All counting is exact integer
arithmetic.  The scanner's own tests are ``tests/test_cbz_layer_a_scan.py``;
``tools/cbz_scanner_mutation_check.py`` checks that those tests kill every
listed mutant of this file.

Usage:
  cbz_layer_a_scan.py ROOT [ROOT ...] [--json] [--allow-axiom NAME ...]

Exit status: 0 if there are no hits and no errors, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata

LAYER_A_TOKENS = ('sorry', 'admit', 'sorryAx', 'axiom', 'native_decide', '+native',
                  'ofReduceBool')


def _is_ident_char(ch: str) -> bool:
    if ch in "_'!?":
        return True
    if ch.isalnum():
        return True
    # Lean identifiers admit Unicode letters and subscripts.
    return unicodedata.category(ch)[0] in ('L', 'N')


def _raw_string_end(text: str, i: int) -> int | None:
    """If text[i] starts a raw string r"..." or r#"..."#, return the index
    just past its end (or len(text) if unterminated); else None."""
    if text[i] != 'r' or (i > 0 and _is_ident_char(text[i - 1])):
        return None
    j = i + 1
    hashes = 0
    while j < len(text) and text[j] == '#':
        hashes += 1
        j += 1
    if j >= len(text) or text[j] != '"':
        return None
    terminator = '"' + '#' * hashes
    end = text.find(terminator, j + 1)
    return len(text) if end < 0 else end + len(terminator)


def code_tokens(text: str):
    """Yield (token, line) for identifier tokens outside comments/strings."""
    i, n, line = 0, len(text), 1
    while i < n:
        ch = text[i]
        if ch == '\n':
            line += 1
            i += 1
            continue
        if text.startswith('--', i):
            end = text.find('\n', i)
            i = n if end < 0 else end
            continue
        if text.startswith('/-', i):
            depth = 1
            i += 2
            while i < n and depth:
                if text.startswith('/-', i):
                    depth += 1
                    i += 2
                elif text.startswith('-/', i):
                    depth -= 1
                    i += 2
                else:
                    if text[i] == '\n':
                        line += 1
                    i += 1
            continue
        raw_end = _raw_string_end(text, i)
        if raw_end is not None:
            line += text.count('\n', i, raw_end)
            i = raw_end
            continue
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                if text[i] == '\\':
                    i += 1
                if i < n and text[i] == '\n':
                    line += 1
                i += 1
            i += 1
            continue
        if ch == "'" and (i == 0 or not _is_ident_char(text[i - 1])):
            # Character literal: 'c' or an escape such as '\n', '\'' or '\x41'.
            if i + 2 < n and text[i + 1] != '\\' and text[i + 2] == "'":
                i += 3
                continue
            if i + 1 < n and text[i + 1] == '\\':
                end = text.find("'", i + 3)
                if end > 0 and end - i <= 12:
                    i = end + 1
                    continue
        if _is_ident_char(ch):
            start = i
            while i < n and (_is_ident_char(text[i]) or text[i] == '.'):
                i += 1
            k = start - 1
            while k >= 0 and text[k] in ' \t':
                k -= 1
            plus = k >= 0 and text[k] == '+'
            for part in text[start:i].split('.'):
                if part:
                    yield ('+native' if plus and part == 'native' else part), line
                    plus = False
            continue
        i += 1


def scan_text(text: str):
    """Return [(token, line)] Layer A hits in one Lean source text."""
    return [(tok, line) for tok, line in code_tokens(text) if tok in LAYER_A_TOKENS]


def scan_roots(roots):
    """Scan every *.lean file under roots. Returns a result dict."""
    files, hits, errors = [], [], []
    for root in roots:
        if os.path.isfile(root):
            candidates = [root]
        else:
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in ('.git', '.lake'))
                for name in sorted(filenames):
                    if name.endswith('.lean'):
                        candidates.append(os.path.join(dirpath, name))
        for path in candidates:
            files.append(path)
            try:
                with open(path, 'rb') as handle:
                    text = handle.read().decode('utf-8')
            except (OSError, UnicodeDecodeError) as exc:
                errors.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            for tok, line in scan_text(text):
                hits.append({'path': path, 'line': line, 'token': tok})
    return {'files_scanned': len(files), 'hits': hits, 'errors': errors}


def main(argv):
    args = list(argv[1:])
    as_json = '--json' in args
    args = [a for a in args if a != '--json']
    allowed = set()
    if '--allow-axiom' in args:
        k = args.index('--allow-axiom')
        allowed = set(args[k + 1:])
        args = args[:k]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    result = scan_roots(args)
    # An allowed axiom is still reported, but does not fail the gate.  The
    # allow-list names a file path suffix and line, e.g. "Foo.lean:12".
    blocking = [h for h in result['hits']
                if not (h['token'] == 'axiom'
                        and any(f"{h['path']}:{h['line']}".endswith(a) for a in allowed))]
    result['blocking_hits'] = len(blocking)
    if as_json:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        for h in result['hits']:
            print(f"{h['path']}:{h['line']}: {h['token']}")
        for e in result['errors']:
            print(f"ERROR {e['path']}: {e['error']}")
        print(f"files_scanned={result['files_scanned']} hits={len(result['hits'])} "
              f"blocking_hits={len(blocking)} errors={len(result['errors'])}")
    return 0 if not blocking and not result['errors'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
