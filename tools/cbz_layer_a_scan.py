#!/usr/bin/env python3
"""Layer A static scanner for Lean sources (CBZ-0 gate G1).

It reports, per file, every occurrence of the Layer A tokens below that is
in code, that is, outside comments, plain string data and character literals:

    sorry, admit, sorryAx         placeholders (A1)
    axiom                         axiom declarations (A2)
    native_decide, +native,       compiler-trusted decisions (A2): ``decide
      native:=                    +native`` and ``(config := {native := true})``
    ofReduceBool, ofReduceNat,    compiler-trust axioms (A2)
      trustCompiler
    skipKernelTC,                 kernel-check bypasses (A2): the
      addDeclWithoutChecking      ``debug.skipKernelTC`` option and the
                                  unchecked declaration API

The lexer understands Lean 4 line comments (``--``), nested block comments
(``/- -/``, including ``/--`` and ``/-!``), string literals with escapes, raw
strings (``r"..."``, ``r#"..."#``), character literals and interpolated
strings.  A string after an identifier ending in ``!`` (``s!``, ``m!``,
``f!``) is lexed as interpolated: each ``{...}`` hole is code, with nested
braces, strings and comments.  For any other string literal, each balanced
``{...}`` segment is also scanned for Layer A tokens, because macros such as
``throwError`` interpolate plain-looking literals; a false positive there is
in the safe direction.  A token is a maximal run of identifier characters;
dotted names are split on ``.``, so ``Lean.sorryAx`` reports ``sorryAx``
while ``sorryish`` and ``my_admit`` report nothing.

It fails closed.  An unterminated string (plain, raw or interpolated) or
block comment raises ``LexError``; ``scan_roots`` records it as an error
and the exit status is nonzero.  A file that cannot be decoded as UTF-8 is an
error, never skipped.  Every ``*.lean`` file under the given roots is
scanned, except under ``.git/`` and ``.lake/`` (dependencies and build
output).  All counting is exact integer arithmetic.

Domain (CBZ-0 governance B-006): this is a source-level lexical check.  It
cannot see placeholders or axioms introduced by macro expansion or failed
elaboration, so it does not establish kernel closure or an axiom set.  The
compiled-environment audit (``#print axioms``) is the authority for those.

Its tests are ``tests/test_cbz_layer_a_scan.py``;
``tools/cbz_scanner_mutation_check.py`` checks that those tests kill every
listed mutant of this file.

Usage:
  cbz_layer_a_scan.py ROOT [ROOT ...] [--json] [--allow-axiom NAME ...]

Exit status: 0 if there are no blocking hits and no errors, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import unicodedata

LAYER_A_TOKENS = ('sorry', 'admit', 'sorryAx', 'axiom', 'native_decide', '+native',
                  'native:=', 'ofReduceBool', 'ofReduceNat', 'trustCompiler',
                  'skipKernelTC', 'addDeclWithoutChecking')


class LexError(ValueError):
    """An unterminated literal, comment or interpolation hole."""


def _is_ident_char(ch: str) -> bool:
    if ch in "_'!?":
        return True
    if ch.isalnum():
        return True
    # Lean identifiers admit Unicode letters and subscripts.
    return unicodedata.category(ch)[0] in ('L', 'N')


class _Lexer:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.i = 0
        self.line = 1
        self.tokens = []

    def _advance(self, k: int = 1):
        for _ in range(k):
            if self.i < self.n and self.text[self.i] == '\n':
                self.line += 1
            self.i += 1

    def _prev_code_char(self, start: int) -> str:
        k = start - 1
        while k >= 0 and self.text[k] in ' \t':
            k -= 1
        return self.text[k] if k >= 0 else ''

    def _next_code_text(self, end: int) -> str:
        k = end
        while k < self.n and self.text[k] in ' \t':
            k += 1
        return self.text[k:k + 2]

    def _block_comment(self):
        start_line = self.line
        depth = 1
        self._advance(2)
        while depth:
            if self.i >= self.n:
                raise LexError(f'unterminated block comment starting at line {start_line}')
            if self.text.startswith('/-', self.i):
                depth += 1
                self._advance(2)
            elif self.text.startswith('-/', self.i):
                depth -= 1
                self._advance(2)
            else:
                self._advance()

    def _raw_string(self) -> bool:
        """Consume r"..." or r#"..."# at self.i if present."""
        text, i = self.text, self.i
        if text[i] != 'r' or (i > 0 and _is_ident_char(text[i - 1])):
            return False
        j = i + 1
        hashes = 0
        while j < self.n and text[j] == '#':
            hashes += 1
            j += 1
        if j >= self.n or text[j] != '"':
            return False
        terminator = '"' + '#' * hashes
        end = text.find(terminator, j + 1)
        if end < 0:
            raise LexError(f'unterminated raw string starting at line {self.line}')
        self._advance(end + len(terminator) - i)
        return True

    def _plain_string(self):
        """Consume a non-interpolated string literal; scan balanced {...} for tokens."""
        start_line = self.line
        self._advance()
        content_start = self.i
        while True:
            if self.i >= self.n:
                raise LexError(f'unterminated string literal starting at line {start_line}')
            ch = self.text[self.i]
            if ch == '\\':
                self._advance(2)
                continue
            if ch == '"':
                break
            self._advance()
        content_end = self.i
        self._advance()
        self._scan_brace_segments(content_start, content_end, start_line)

    def _scan_brace_segments(self, start: int, end: int, line: int):
        """Conservatively tokenise each balanced {...} segment of string data."""
        text = self.text
        k = start
        while k < end:
            ch = text[k]
            if ch == '\\':
                k += 2
                continue
            if ch == '\n':
                line += 1
            if ch == '{':
                depth, j, seg_line = 1, k + 1, line
                while j < end and depth:
                    if text[j] == '\\':
                        j += 2
                        continue
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    j += 1
                if depth == 0:
                    inner = _Lexer(text[k + 1:j - 1])
                    inner.line = seg_line
                    try:
                        inner.code(until_brace=False)
                    except LexError:
                        pass  # brace text that is not code; the literal itself was closed
                    self.tokens.extend(inner.tokens)
                    line += text.count('\n', k + 1, j)
                    k = j
                    continue
            k += 1

    def _interpolated_string(self):
        """Consume an interpolated literal; each {...} hole is lexed as code."""
        start_line = self.line
        self._advance()
        while True:
            if self.i >= self.n:
                raise LexError(f'unterminated interpolated string starting at line {start_line}')
            ch = self.text[self.i]
            if ch == '\\':
                self._advance(2)
                continue
            if ch == '"':
                self._advance()
                return
            if ch == '{':
                self._advance()
                self.code(until_brace=True)
                continue
            self._advance()

    def _char_literal(self) -> bool:
        text, i = self.text, self.i
        if i > 0 and _is_ident_char(text[i - 1]):
            return False
        # 'c' or an escape such as '\n', '\'' or '\x41'.
        if i + 2 < self.n and text[i + 1] != '\\' and text[i + 2] == "'":
            self._advance(3)
            return True
        if i + 1 < self.n and text[i + 1] == '\\':
            end = text.find("'", i + 3)
            if 0 < end and end - i <= 12:
                self._advance(end + 1 - i)
                return True
        return False

    def code(self, until_brace: bool):
        """Lex code; with until_brace, stop after the brace closing this hole."""
        depth = 0
        text = self.text
        while self.i < self.n:
            ch = text[self.i]
            if until_brace and ch == '}' and depth == 0:
                self._advance()
                return
            if until_brace and ch == '{':
                depth += 1
            elif until_brace and ch == '}':
                depth -= 1
            if ch == '\n':
                self._advance()
                continue
            if text.startswith('--', self.i):
                end = text.find('\n', self.i)
                self._advance((self.n if end < 0 else end) - self.i)
                continue
            if text.startswith('/-', self.i):
                self._block_comment()
                continue
            if self._raw_string():
                continue
            if ch == '"':
                if self._prev_code_char(self.i).endswith('!') and self.i > 0 and text[self.i - 1] == '!':
                    self._interpolated_string()
                else:
                    self._plain_string()
                continue
            if ch == "'" and self._char_literal():
                continue
            if _is_ident_char(ch):
                start = self.i
                while self.i < self.n and (_is_ident_char(text[self.i]) or text[self.i] == '.'):
                    self.i += 1
                # An identifier ending in '!' directly before '"' opens an interpolated string;
                # the '!' stays part of the identifier for the check above.
                plus = self._prev_code_char(start) == '+'
                config = (self._next_code_text(self.i) == ':='
                          and self._prev_code_char(start) in ('{', ','))
                for part in text[start:self.i].split('.'):
                    if not part:
                        continue
                    name = part.rstrip('!')
                    if plus and name == 'native':
                        name = '+native'
                    elif config and name == 'native':
                        name = 'native:='
                    self.tokens.append((name, self.line))
                    plus = False
                continue
            self._advance()
        # A hole that reaches end of file leaves its string unterminated;
        # _interpolated_string raises for that.


def code_tokens(text: str):
    """Return [(token, line)] for identifier tokens in code. Raises LexError."""
    lexer = _Lexer(text)
    lexer.code(until_brace=False)
    return lexer.tokens


def scan_text(text: str):
    """Return [(token, line)] Layer A hits in one Lean source text. Raises LexError."""
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
                found = scan_text(text)
            except (OSError, UnicodeDecodeError, LexError) as exc:
                errors.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            for tok, line in found:
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
