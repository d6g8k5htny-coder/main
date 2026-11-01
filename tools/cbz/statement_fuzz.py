#!/usr/bin/env python3
"""CBZ-0 G12 — empirical statement check for core-Lean arithmetic skeletons (stdlib only).

The owner's directive addendum of 2026-10-29 requires an empirical validation layer before a proof
obligation may be marked CLOSED. Kernel verification is not strengthened by sampling, so this harness
tests the *statements*, not the proofs: for every theorem of a formal package it parses the Lean
statement and the package's own definitions in a small interpreter for the core-Lean arithmetic subset
these skeletons use, evaluates the statement independently of Lean in exact arithmetic (Python
integers and fractions, never floats — Layer A3), enumerates finite quantifier domains exhaustively,
probes boundary values outside a hypothesis domain (information: tightness and vacuity, A6), and
reports PASS / FAIL / UNSUPPORTED per target.

A FAIL is a misstatement or a transcription defect in the Lean text; it is never evidence about the
kernel. UNSUPPORTED is fail-closed: the construct is outside the interpreter (Mathlib structures,
measures and functions, transcendental functions, unbounded quantifiers inside a statement, notation
the subset does not define) and the target cannot count as fuzz-validated.

Statements with binders before the colon — `theorem t (x y : ℝ) (hx : 0 < x) : …` — are *sampled*: the
numeric binders (ℕ ℤ ℚ ℝ) are drawn from exact pathological values (0, ±1, ±tiny, ±huge, squares and
non-squares, the boundary values the hypotheses name) and seeded random rationals, in declaration order
with every hypothesis checked as soon as its variables are assigned; the conclusion is evaluated exactly
at each instance that satisfies the hypotheses. The verdict is FAIL with the exact witness, or
CORROBORATED when no instance refutes it — admitted evidence of the statement, never PASS and never a
proof — or UNSUPPORTED when a binder is not a sampled numeric type or decidable hypothesis, or no
sample satisfies the hypotheses (possibly vacuous).

Real arithmetic is exact: rationals as Fractions, and `Real.sqrt` of a rational as an element of the
multi-quadratic field ℚ(√d₁, …, √dₖ) (canonical form over squarefree radicands, so equality is decided
exactly; the sign of a nonzero element is decided by refining integer-square-root enclosures). Nested
radicals, `Real.exp`/`log`/`pi` and real exponents are UNSUPPORTED (an interval backend may admit them).

Lean semantics implemented: Nat subtraction truncates at 0; Nat and Int division by zero is 0 and
x % 0 = x (totalized); Nat division floors; Int `/` and `%` are Euclidean (`Int.ediv`/`Int.emod`:
0 ≤ r < |b|), which is Lean 4 core's convention — not Python's floor division and not T-rounding;
field division and inverse are totalized (`x / 0 = 0`, `0⁻¹ = 0`); `Real.sqrt x = 0` for `x ≤ 0`;
`a ^ n` types its exponent on its own (`x ^ (1/2)` is `x ^ 0`, as in Lean), an `ℤ` exponent is `zpow`;
numerals default to Nat unless an ascription, a binder or a definition's result type makes the
arithmetic Int/Rat/Real, following the maximal-type rule of Lean's `binop%` elaborator, and an ascription
`(e : T)` elaborates `e` at `T` (`(n - 1 : ℝ)` is `↑n - 1`; `((n - 1 : ℕ) : ℝ)` casts a truncated Nat).

Usage:
    python3 tools/cbz/statement_fuzz.py --package formal [--json report.json] [--selftest]
                                        [--seed 1] [--samples 64] [--admit-corroborated]
Exit 0 only when every registered target is PASS — or PASS/CORROBORATED with --admit-corroborated —
(and, with --selftest, every required mutant is killed and the harness control passes).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import random
import re
import sys
from fractions import Fraction

sys.setrecursionlimit(10000)

# --------------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Token:
    kind: str   # 'num', 'ident', 'op', 'punct'
    value: str
    line: int
    col: int


_MULTI_OPS = ('<=', '>=', '!=', '->', ':=', '=>', '/\\', '\\/', '<->', '||', '&&', '..', '⁻¹')
_SINGLE = set('+-*/%^=<>()[],:|.{}!∀∃∧∨¬→↔≤≥≠∈∉λ√')
_IDENT_START = re.compile(r'[A-Za-z_α-ωΑ-Ωℕℤℚℝℂ«]')
_IDENT_CHAR = re.compile(r'[A-Za-z0-9_\'!?α-ωΑ-Ωℕℤℚℝℂ₀-₉»]')


def tokenize(text: str) -> list:
    """Tokens with line/col; `--` line comments and (nested) `/- -/` block comments are dropped."""
    toks = []
    i, n = 0, len(text)
    line, col = 1, 0
    def advance(k):
        nonlocal i, line, col
        for ch in text[i:i + k]:
            if ch == '\n':
                line += 1; col = 0
            else:
                col += 1
        i += k
    while i < n:
        ch = text[i]
        if text.startswith('--', i):
            j = text.find('\n', i)
            advance((j if j != -1 else n) - i)
            continue
        if text.startswith('/-', i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith('/-', j):
                    depth += 1; j += 2
                elif text.startswith('-/', j):
                    depth -= 1; j += 2
                else:
                    j += 1
            advance(j - i)
            continue
        if ch.isspace():
            advance(1); continue
        if ch == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            toks.append(Token('str', text[i:j + 1], line, col)); advance(j + 1 - i); continue
        if ch.isdigit():
            j = i
            while j < n and (text[j].isdigit() or text[j] == '_'):
                j += 1
            # a scientific literal `1.5` (Lean `OfScientific`): exact decimal, kept as a Fraction downstream
            if j + 1 < n and text[j] == '.' and text[j + 1].isdigit():
                j += 1
                while j < n and text[j].isdigit():
                    j += 1
            toks.append(Token('num', text[i:j].replace('_', ''), line, col)); advance(j - i); continue
        if _IDENT_START.match(ch):
            j = i + 1
            while j < n and (_IDENT_CHAR.match(text[j]) or (text[j] == '.' and j + 1 < n and _IDENT_START.match(text[j + 1]))):
                j += 1
            toks.append(Token('ident', text[i:j], line, col)); advance(j - i); continue
        matched = False
        for op in _MULTI_OPS:
            if text.startswith(op, i):
                toks.append(Token('op', op, line, col)); advance(len(op)); matched = True; break
        if matched:
            continue
        if ch in _SINGLE:
            toks.append(Token('op', ch, line, col)); advance(1); continue
        if ch == '@':
            advance(1); continue
        # Anything else (Mathlib notation such as ←, ∑, ↦, ⟨⟩) is outside the subset: keep it as an
        # 'other' token so the statement that contains it is reported UNSUPPORTED, never skipped.
        toks.append(Token('other', ch, line, col)); advance(1)
    return toks


# --------------------------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Num:
    value: object           # int, or a Fraction for a scientific literal such as `1.5`

@dataclasses.dataclass
class Var:
    name: str

@dataclasses.dataclass
class App:
    fn: object
    args: list

@dataclasses.dataclass
class BinOp:
    op: str
    left: object
    right: object

@dataclasses.dataclass
class Neg:
    expr: object

@dataclasses.dataclass
class Not:
    expr: object

@dataclasses.dataclass
class Ascribe:
    expr: object
    type: str

@dataclasses.dataclass
class ListLit:
    items: list

@dataclasses.dataclass
class Lambda:
    params: list
    body: object

@dataclasses.dataclass
class Tuple:
    items: list

@dataclasses.dataclass
class Proj:
    expr: object
    index: int

@dataclasses.dataclass
class IfThenElse:
    cond: object
    then: object
    other: object

@dataclasses.dataclass
class BoolNot:
    expr: object

@dataclasses.dataclass
class Forall:
    var: str
    type: str | None
    domain: object | None   # expression whose value is a list, when written `∀ x ∈ l, …`
    body: object


_NORMALIZE = {'<=': '≤', '>=': '≥', '!=': '≠', '->': '→', '/\\': '∧', '\\/': '∨', '<->': '↔'}
_PREC = {'→': 25, '↔': 20, '∨': 30, '∧': 35, '||': 30, '&&': 35, '=': 50, '≠': 50, '<': 50, '>': 50, '≤': 50, '≥': 50,
         '∈': 50, '∉': 50, '+': 65, '-': 65, '*': 70, '/': 70, '%': 70, '^': 75}
_RIGHT = {'→', '↔', '∨', '∧', '||', '&&', '^'}
_COMPARISONS = {'=', '≠', '<', '>', '≤', '≥', '∈', '∉'}
_LOGIC = {'→', '↔', '∨', '∧'}


class Unsupported(Exception):
    pass


class Parser:
    def __init__(self, toks, start=0, end=None):
        self.toks = toks
        self.i = start
        self.end = len(toks) if end is None else end
        self.abs_depth = 0

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < self.end else None

    def at(self, value):
        t = self.peek()
        return t is not None and t.value == value

    def eat(self, value=None):
        t = self.peek()
        if t is None or (value is not None and t.value != value):
            raise SyntaxError(f'expected {value!r}, got {t.value if t else "end"}')
        self.i += 1
        return t

    def parse_expr(self, min_prec=0):
        left = self.parse_unary()
        while True:
            t = self.peek()
            if t is not None and t.kind == 'other':
                raise Unsupported(f'notation {t.value!r}')
            if t is None or t.kind != 'op':
                break
            op = _NORMALIZE.get(t.value, t.value)
            if op not in _PREC or _PREC[op] < min_prec:
                break
            prec = _PREC[op]
            self.i += 1
            right = self.parse_expr(prec if op in _RIGHT else prec + 1)
            left = BinOp(op, left, right)
        return left

    def parse_unary(self):
        t = self.peek()
        if t is None:
            raise SyntaxError('unexpected end of expression')
        if t.value == '¬':
            self.i += 1
            return Not(self.parse_expr(40))
        if t.value == '!':
            self.i += 1
            return BoolNot(self.parse_expr(80))
        if t.value == 'if':
            self.i += 1
            cond = self.parse_expr(0)
            self.eat('then')
            then = self.parse_expr(0)
            self.eat('else')
            other = self.parse_expr(0)
            return IfThenElse(cond, then, other)
        if t.value == '-':
            self.i += 1
            return Neg(self.parse_expr(75))
        if t.value == '√':
            # Mathlib `prefix:100 "√" => Real.sqrt`: binds tighter than `^` (75), looser than application
            self.i += 1
            return App(Var('Real.sqrt'), [self.parse_expr(100)])
        if t.value in ('∀', '∃'):
            return self.parse_quantifier()
        if t.value in ('fun', 'λ'):
            return self.parse_lambda()
        return self.parse_application()

    def parse_quantifier(self):
        q = self.eat().value
        if q == '∃':
            raise Unsupported('existential quantifier')
        var = self.eat().value
        typ, domain = None, None
        if self.at(':'):
            self.eat(':')
            typ = self.eat().value
        elif self.at('∈'):
            self.eat('∈')
            domain = self.parse_expr(51)
        self.eat(',')
        body = self.parse_expr(0)
        return Forall(var, typ, domain, body)

    def parse_lambda(self):
        self.eat()
        params = []
        while not self.at('=>'):
            t = self.eat()
            if t.value == '(':
                params.append(self.eat().value)
                if self.at(':'):
                    self.eat(':'); self.eat()
                self.eat(')')
            else:
                params.append(t.value)
        self.eat('=>')
        return Lambda(params, self.parse_expr(0))

    def parse_application(self):
        head = self.parse_atom()
        args = []
        while True:
            t = self.peek()
            if t is None:
                break
            if t.kind == 'ident' and t.value in ('if', 'then', 'else', 'fun', 'λ'):
                break
            if t.kind in ('num', 'ident') or (t.kind == 'op' and t.value in ('(', '[')):
                args.append(self.parse_atom())
                continue
            # `f |x|`: an opening bar is an argument only outside any `|…|` and when, as in Lean's
            # `atomic("|" noWs)`, no whitespace separates it from the next token
            nxt = self.peek(1)
            if (t.kind == 'op' and t.value == '|' and self.abs_depth == 0 and nxt is not None
                    and nxt.line == t.line and nxt.col == t.col + 1):
                args.append(self.parse_atom())
                continue
            break
        if not args:
            return head
        if isinstance(head, App) and isinstance(head.fn, Var) and head.fn.name.startswith('.'):
            return App(head.fn, head.args + args)      # receiver.method arg1 arg2 …
        return App(head, args)

    def parse_postfix(self, atom):
        """Dot notation binds tighter than application: `p.1`, `xs.foldl`, `(e).filter`; so does `x⁻¹`."""
        while True:
            t = self.peek()
            if t is not None and t.kind == 'op' and t.value == '⁻¹':
                self.i += 1
                atom = App(Var('Inv.inv'), [atom])
                continue
            if t is not None and t.kind == 'op' and t.value == '.' and self.peek(1) is not None:
                nxt = self.peek(1)
                if nxt.kind == 'num':
                    self.i += 2
                    atom = Proj(atom, int(nxt.value))
                    continue
                if nxt.kind == 'ident':
                    self.i += 2
                    atom = App(Var('.' + nxt.value), [atom])
                    continue
            return atom

    def parse_atom(self):
        return self.parse_postfix(self._parse_atom_core())

    def _parse_atom_core(self):
        t = self.eat()
        if t.kind == 'other':
            raise Unsupported(f'notation {t.value!r}')
        if t.kind == 'num':
            return Num(Fraction(t.value) if '.' in t.value else int(t.value))
        if t.kind == 'ident':
            if t.value in ('fun', 'λ'):
                self.i -= 1
                return self.parse_lambda()
            return Var(t.value)
        if t.value == '|':
            # Mathlib absolute value `|e|`; the inner expression stops at the closing bar because `|` is
            # not an infix operator of the subset (`||x| - |y||` is not parsed: Lean needs parentheses too)
            self.abs_depth += 1
            e = self.parse_expr(0)
            self.eat('|')
            self.abs_depth -= 1
            return App(Var('abs'), [e])
        if t.value == '(':
            e = self.parse_expr(0)
            if self.at(':'):
                self.eat(':')
                typ = self.eat().value
                self.eat(')')
                return Ascribe(e, typ)
            if self.at(','):
                items = [e]
                while self.at(','):
                    self.eat(',')
                    items.append(self.parse_expr(0))
                self.eat(')')
                return Tuple(items)
            self.eat(')')
            return e
        if t.value == '[':
            items = []
            while not self.at(']'):
                items.append(self.parse_expr(0))
                if self.at(','):
                    self.eat(',')
            self.eat(']')
            return ListLit(items)
        raise SyntaxError(f'unexpected token {t.value!r} at {t.line}:{t.col}')


# --------------------------------------------------------------------------------------------
# Module structure
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Def:
    name: str
    params: list            # [(name, type)]
    result_type: str | None
    body: object | None     # expression, or None when pattern-matching
    clauses: list           # [(patterns, expr)] for pattern-matching defs

@dataclasses.dataclass
class Binder:
    """A binder before the colon of a theorem: a sampled numeric variable (`kind == 'var'`, `type` is the
    numeric type name) or a hypothesis (`kind == 'hyp'`, `type` is the proposition's AST)."""
    kind: str
    name: str
    type: object
    text: str = ''

@dataclasses.dataclass
class Theorem:
    name: str
    statement: object | None
    error: str | None = None
    text: str = ''
    binders: list = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Module:
    namespace: str
    defs: dict
    theorems: list


_DECL_KEYWORDS = {'theorem', 'lemma', 'def', 'abbrev', 'example', 'instance', 'structure', 'inductive',
                  'namespace', 'end', 'open', 'import', 'section', 'variable', 'universe', 'set_option',
                  'noncomputable', 'private', 'protected', 'attribute', 'macro', 'syntax', 'notation', 'axiom'}


def _split_declarations(toks):
    starts = [k for k, t in enumerate(toks) if t.col == 0 and ((t.kind == 'ident' and t.value in _DECL_KEYWORDS)
                                                               or t.value == '@')]
    starts.append(len(toks))
    return [(starts[k], starts[k + 1]) for k in range(len(starts) - 1)]


def _find(toks, value, start, end):
    depth = 0
    for k in range(start, end):
        v = toks[k].value
        if v in ('(', '[', '{'):
            depth += 1
        elif v in (')', ']', '}'):
            depth -= 1
        elif depth == 0 and v == value:
            return k
    return -1


def _matching(toks, k, end):
    """Index of the bracket closing the one at ``k`` (or -1)."""
    depth = 0
    for j in range(k, end):
        v = toks[j].value
        if v in ('(', '[', '{'):
            depth += 1
        elif v in (')', ']', '}'):
            depth -= 1
            if depth == 0:
                return j
    return -1


def _parse_binders(toks, start, end):
    """(name : T) (m : T) ... -> [(name, T)], position after binders."""
    params, k = [], start
    while k < end and toks[k].value == '(':
        close = k
        while close < end and toks[close].value != ')':
            close += 1
        if close >= end:
            break
        names = []
        j = k + 1
        while toks[j].value != ':':
            names.append(toks[j].value); j += 1
        typ = ' '.join(t.value for t in toks[j + 1:close])
        params.extend((nm, typ) for nm in names)
        k = close + 1
    return params, k


def parse_module(source: str) -> Module:
    toks = tokenize(source)
    namespace, defs, theorems = '', {}, []
    for start, end in _split_declarations(toks):
        kw = toks[start].value
        k = start
        while k < end and toks[k].value in ('noncomputable', 'private', 'protected'):
            k += 1
        kw = toks[k].value if k < end else kw
        if kw == 'namespace':
            namespace = toks[k + 1].value
            continue
        if kw in ('def', 'abbrev'):
            name = toks[k + 1].value
            params, j = _parse_binders(toks, k + 2, end)
            result_type = None
            assign = _find(toks, ':=', j, end)
            bar = _find(toks, '|', j, end)
            if toks[j].value == ':':
                type_end = assign if assign != -1 else (bar if bar != -1 else end)
                result_type = ' '.join(t.value for t in toks[j + 1:type_end])
            if assign != -1 and (bar == -1 or assign < bar):
                try:
                    body = Parser(toks, assign + 1, end).parse_expr(0)
                except (SyntaxError, Unsupported):
                    body = None
                defs[name] = Def(name, params, result_type, body, [])
            elif bar != -1:
                clauses = []
                bars = [m for m in range(j, end) if toks[m].value == '|' and toks[m].col <= 2]
                bars.append(end)
                try:
                    for b0, b1 in zip(bars, bars[1:]):
                        arrow = _find(toks, '=>', b0, b1)
                        if arrow == -1:
                            raise Unsupported('clause without =>')
                        pats = _split_patterns(toks[b0 + 1:arrow])
                        try:
                            expr = Parser(toks, arrow + 1, b1).parse_expr(0)
                        except (SyntaxError, Unsupported):
                            expr = None
                        clauses.append((pats, expr))
                except (Unsupported, SyntaxError, IndexError):
                    clauses = []
                defs[name] = Def(name, params, result_type, None, clauses)
            continue
        if kw in ('theorem', 'lemma'):
            name = toks[k + 1].value
            colon = _find(toks, ':', k + 2, end)
            assign = _find(toks, ':=', k + 2, end)
            full = f'{namespace}.{name}' if namespace else name
            if colon == -1 or assign == -1 or assign < colon:
                theorems.append(Theorem(full, None, 'malformed theorem'))
                continue
            binder_text = ' '.join(t.value for t in toks[k + 2:colon])
            text = ' '.join(t.value for t in toks[colon + 1:assign])
            if binder_text:
                text = f'{binder_text} : {text}'
            binders, binder_error = _parse_theorem_binders(toks, k + 2, colon, defs)
            if binder_error:
                theorems.append(Theorem(full, None, binder_error, text, binders))
                continue
            try:
                parser = Parser(toks, colon + 1, assign)
                stmt = parser.parse_expr(0)
                if parser.i != assign:
                    raise SyntaxError(f'trailing tokens from {toks[parser.i].value!r}')
                theorems.append(Theorem(full, _desugar(stmt), None, text, binders))
            except Unsupported as e:
                theorems.append(Theorem(full, None, f'unsupported syntax: {e}', text, binders))
            except SyntaxError as e:
                theorems.append(Theorem(full, None, f'unparsed statement: {e}', text, binders))
            continue
    return Module(namespace, defs, theorems)


def _split_patterns(toks):
    """Top-level comma-separated patterns, each either a literal, a variable or `var + k`."""
    pats, cur, depth = [], [], 0
    for t in toks + [Token('op', ',', 0, 0)]:
        if t.value == ',' and depth == 0:
            pats.append(_pattern(cur)); cur = []
            continue
        if t.value in ('(', '['):
            depth += 1
        elif t.value in (')', ']'):
            depth -= 1
        cur.append(t)
    return pats


def _pattern(toks):
    toks = [t for t in toks if t.value not in ('(', ')')]
    if len(toks) == 1 and toks[0].kind == 'num':
        return ('lit', int(toks[0].value))
    if len(toks) == 1:
        return ('var', toks[0].value)
    if len(toks) == 3 and toks[1].value == '+' and toks[2].kind == 'num':
        return ('succ', toks[0].value, int(toks[2].value))
    raise Unsupported('pattern ' + ' '.join(t.value for t in toks))


# --------------------------------------------------------------------------------------------
# Binders before the colon: sampled numeric variables and decidable hypotheses
# --------------------------------------------------------------------------------------------

_SAMPLED_TYPE_TOKENS = {'Nat': 'Nat', 'ℕ': 'Nat', 'Int': 'Int', 'ℤ': 'Int', 'Rat': 'Rat', 'ℚ': 'Rat',
                        'Real': 'Real', 'ℝ': 'Real'}

_BUILTIN_NAMES = {'true', 'false', 'True', 'False', 'abs', 'min', 'max', 'Real.sqrt', 'Inv.inv', 'List.range',
                  'List.foldl', 'List.filter', 'List.sum', 'List.length', 'List.map', 'Nat.choose', 'Nat.factorial',
                  'Nat.succ', 'Nat.pred', 'Nat.min', 'Nat.max', 'Nat.add', 'Nat.mul', 'Nat.sub', 'Int.natAbs',
                  'Int.toNat', 'Int.ofNat', 'Set.Icc', 'Set.Ico', 'Set.Ioc', 'Set.Ioo', 'Set.Ici', 'Set.Iic',
                  'Set.Ioi', 'Set.Iio'}


def _parse_theorem_binders(toks, start, end, defs):
    """Binder groups between the theorem name and its colon → ([Binder], error or None).

    `(x y : ℝ)` / `{x : ℕ}` with a sampled numeric type declare variables; `(h : p)` whose type parses as a
    proposition over names declared so far (variables, the module's definitions, the interpreter's
    builtins) is a hypothesis; anything else — an instance binder `[Fact p]`, a function or structure type
    `(f : Ω → ℝ)`, `(μ : Measure Ω)`, a predicate the interpreter does not define — makes the theorem
    UNSUPPORTED, fail-closed, naming the binder."""
    binders, declared = [], set()
    k = start
    while k < end:
        t = toks[k]
        if t.value == '[':
            close = _matching(toks, k, end)
            txt = ' '.join(x.value for x in toks[k:close + 1 if close != -1 else end])
            return binders, f'instance binder {txt} cannot be sampled'
        if t.value not in ('(', '{'):
            return binders, f'binder {t.value!r} without a type is outside the interpreter'
        close = _matching(toks, k, end)
        if close == -1:
            return binders, 'unbalanced binder brackets'
        txt = ' '.join(x.value for x in toks[k:close + 1])
        colon = _find(toks, ':', k + 1, close)
        if colon == -1:
            return binders, f'binder {txt} without a type is outside the interpreter'
        names = [x.value for x in toks[k + 1:colon]]
        if not names or any(x.kind != 'ident' for x in toks[k + 1:colon]):
            return binders, f'binder pattern {txt} is outside the interpreter'
        type_toks = toks[colon + 1:close]
        if len(type_toks) == 1 and type_toks[0].value in _SAMPLED_TYPE_TOKENS:
            typ = _SAMPLED_TYPE_TOKENS[type_toks[0].value]
            for nm in names:
                binders.append(Binder('var', nm, typ, txt))
                declared.add(nm)
            k = close + 1
            continue
        try:
            parser = Parser(toks, colon + 1, close)
            node = parser.parse_expr(0)
            if parser.i != close:
                raise SyntaxError('trailing tokens in a binder type')
        except (SyntaxError, Unsupported):
            node = None
        if node is not None and _is_prop_shape(node):
            known = declared | set(defs) | _BUILTIN_NAMES | _TRANSCENDENTAL
            unknown = sorted(v for v in _free_vars(node) if v not in known and v.split('.')[0] not in known)
            if not unknown:
                node = _desugar(node)
                for nm in names:
                    binders.append(Binder('hyp', nm, node, txt))
                k = close + 1
                continue
            if any(u in _SAMPLED_TYPE_TOKENS for u in unknown):
                return binders, f'binder {txt} is neither a sampled numeric type (ℕ ℤ ℚ ℝ) nor a decidable hypothesis'
            return binders, f'hypothesis {txt} mentions undeclared names {unknown}'
        return binders, f'binder {txt} is neither a sampled numeric type (ℕ ℤ ℚ ℝ) nor a decidable hypothesis'
    return binders, None


def _is_prop_shape(node) -> bool:
    """A syntactic proposition: comparisons and membership, closed under ¬ ∧ ∨ → ↔ and ∀."""
    if isinstance(node, BinOp):
        if node.op in _COMPARISONS:
            return True
        if node.op in _LOGIC:
            return _is_prop_shape(node.left) and _is_prop_shape(node.right)
        return False
    if isinstance(node, Not):
        return _is_prop_shape(node.expr)
    if isinstance(node, Forall):
        return _is_prop_shape(node.body)
    return isinstance(node, Var) and node.name in ('True', 'False')


def _free_vars(node, bound=frozenset()) -> set:
    if isinstance(node, Var):
        return set() if node.name in bound or node.name.startswith('.') else {node.name}
    if isinstance(node, Forall):
        inner = _free_vars(node.body, bound | {node.var})
        return inner | (_free_vars(node.domain, bound) if node.domain is not None else set())
    if isinstance(node, Lambda):
        return _free_vars(node.body, bound | set(node.params))
    out = set()
    for child in _children(node):
        out |= _free_vars(child, bound)
    return out


def _children(node):
    if isinstance(node, (Num, Var)):
        return []
    if isinstance(node, App):
        return [node.fn] + list(node.args)
    if isinstance(node, BinOp):
        return [node.left, node.right]
    if isinstance(node, (Neg, Not, BoolNot, Ascribe)):
        return [node.expr]
    if isinstance(node, (ListLit, Tuple)):
        return list(node.items)
    if isinstance(node, Proj):
        return [node.expr]
    if isinstance(node, IfThenElse):
        return [node.cond, node.then, node.other]
    if isinstance(node, Lambda):
        return [node.body]
    if isinstance(node, Forall):
        return ([node.domain] if node.domain is not None else []) + [node.body]
    return []


_SET_INTERVALS = {'Set.Icc': ('≤', '≤'), 'Set.Ico': ('≤', '<'), 'Set.Ioc': ('<', '≤'), 'Set.Ioo': ('<', '<'),
                  'Set.Ici': ('≤', None), 'Set.Ioi': ('<', None), 'Set.Iic': (None, '≤'), 'Set.Iio': (None, '<')}


def _desugar(node):
    """Rewrite Mathlib interval membership into comparisons: `x ∈ Set.Icc a b` ↦ `a ≤ x ∧ x ≤ b`, etc.
    Everything else is rebuilt unchanged."""
    if isinstance(node, BinOp) and node.op in ('∈', '∉') and isinstance(node.right, App) \
            and isinstance(node.right.fn, Var) and node.right.fn.name in _SET_INTERVALS:
        lo_op, hi_op = _SET_INTERVALS[node.right.fn.name]
        args = node.right.args
        x = _desugar(node.left)
        parts = []
        if lo_op is not None and args:
            parts.append(BinOp(lo_op, _desugar(args[0]), x))
        if hi_op is not None:
            bound = args[-1] if (lo_op is None and args) else (args[1] if len(args) > 1 else None)
            if bound is None:
                return node
            parts.append(BinOp(hi_op, x, _desugar(bound)))
        if not parts:
            return node
        out = parts[0]
        for p in parts[1:]:
            out = BinOp('∧', out, p)
        return Not(out) if node.op == '∉' else out
    if isinstance(node, App):
        return App(_desugar(node.fn), [_desugar(a) for a in node.args])
    if isinstance(node, BinOp):
        return BinOp(node.op, _desugar(node.left), _desugar(node.right))
    if isinstance(node, Neg):
        return Neg(_desugar(node.expr))
    if isinstance(node, Not):
        return Not(_desugar(node.expr))
    if isinstance(node, BoolNot):
        return BoolNot(_desugar(node.expr))
    if isinstance(node, Ascribe):
        return Ascribe(_desugar(node.expr), node.type)
    if isinstance(node, ListLit):
        return ListLit([_desugar(x) for x in node.items])
    if isinstance(node, Tuple):
        return Tuple([_desugar(x) for x in node.items])
    if isinstance(node, Proj):
        return Proj(_desugar(node.expr), node.index)
    if isinstance(node, IfThenElse):
        return IfThenElse(_desugar(node.cond), _desugar(node.then), _desugar(node.other))
    if isinstance(node, Lambda):
        return Lambda(node.params, _desugar(node.body))
    if isinstance(node, Forall):
        return Forall(node.var, node.type, None if node.domain is None else _desugar(node.domain), _desugar(node.body))
    return node


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

_LIST_METHODS = {'foldl', 'filter', 'map', 'sum', 'length'}
_INT_TYPES = {'Int', 'ℤ'}
_NAT_TYPES = {'Nat', 'ℕ'}
_RAT_TYPES = {'Rat', 'ℚ'}
_REAL_TYPES = {'Real', 'ℝ'}
_FIELD_TYPES = {'Rat', 'Real'}          # numeric types with exact (totalized) division
_TRANSCENDENTAL = {'Real.exp', 'Real.log', 'Real.pi', 'Real.sin', 'Real.cos', 'Real.tan', 'Real.arctan',
                   'Real.sinh', 'Real.cosh', 'Real.rpow', 'Real.logb', 'π'}

# Resource caps (Layer A: an exact evaluation that cannot finish is reported, never approximated)
_MAX_RESULT_BITS = 2_000_000
_MAX_LOOP = 200_000           # List.range
_MAX_PRODUCT_LOOP = 5_000      # Nat.factorial, Nat.choose (products of thousands of big integers)


class Infeasible(Unsupported):
    """An exact evaluation that would exceed the resource caps (an astronomically large power, factorial or
    range). A sampled instance that is infeasible is skipped and counted; a closed statement is UNSUPPORTED."""


class Closure:
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env


_BUILTIN_BINARY = {
    'min': min, 'max': max, 'Nat.min': min, 'Nat.max': max,
    'Nat.add': lambda a, b: a + b, 'Nat.mul': lambda a, b: a * b,
    'Nat.sub': lambda a, b: a - b if a >= b else 0,
}


class Bound:
    """A bound variable's value with its numeric type ('Nat', 'Int', 'Rat' or 'Bool')."""
    __slots__ = ('value', 'type')

    def __init__(self, value, type_):
        self.value, self.type = value, type_


class Evaluator:
    def __init__(self, module: Module):
        self.module = module
        self.values_seen = []
        self.instances = 0
        self.vacuous = False
        self.failures = []
        self.probes = []

    # ---- typing: the maximal numeric type among the leaves of an arithmetic tree
    def arith_type(self, node, env) -> str:
        if isinstance(node, Ascribe):
            if node.type in _INT_TYPES:
                return 'Int'
            if node.type in _NAT_TYPES:
                return 'Nat'
            if node.type in _RAT_TYPES:
                return 'Rat'
            if node.type in _REAL_TYPES:
                return 'Real'
            raise Unsupported(f'ascription to {node.type}')
        if isinstance(node, Num):
            # a scientific literal (`1.5`) takes the field type of its context; alone, Lean would default it to
            # Float, which the interpreter refuses (see num)
            return 'Nat' if isinstance(node.value, int) else 'Sci'
        if isinstance(node, Var):
            if node.name in env:
                v = env[node.name]
                return v.type if isinstance(v, Bound) else 'Nat'
            d = self.module.defs.get(node.name)
            if d is not None and d.result_type:
                return self._type_name(d.result_type)
            return 'Nat'
        if isinstance(node, (BinOp,)):
            if node.op in _COMPARISONS or node.op in _LOGIC:
                return 'Prop'
            if node.op == '^':
                return self.arith_type(node.left, env)      # the exponent is typed on its own (Lean `rightact%`)
            return _max_type(self.arith_type(node.left, env), self.arith_type(node.right, env))
        if isinstance(node, Neg):
            return 'Int' if self.arith_type(node.expr, env) == 'Nat' else self.arith_type(node.expr, env)
        if isinstance(node, (Proj, Tuple, ListLit, BoolNot)):
            return 'Nat'
        if isinstance(node, IfThenElse):
            return _max_type(self.arith_type(node.then, env), self.arith_type(node.other, env))
        if isinstance(node, App):
            fn = node.fn
            if isinstance(fn, Var):
                d = self.module.defs.get(fn.name)
                if d is not None and d.result_type:
                    return self._type_name(d.result_type)
                if fn.name == 'Real.sqrt':
                    return 'Real'
                if fn.name in ('abs', 'min', 'max', 'Inv.inv') and node.args:
                    out = 'Nat'
                    for a in node.args:
                        out = _max_type(out, self.arith_type(a, env))
                    return out
                if fn.name in ('Int.natAbs', 'Int.toNat', 'Nat.choose', 'Nat.factorial', 'Nat.succ', 'Nat.pred',
                               'List.length', 'List.sum', 'Nat.min', 'Nat.max', 'Nat.sqrt'):
                    return 'Nat'
                if fn.name in ('Int.ofNat', 'Int.negSucc'):
                    return 'Int'
            return 'Nat'
        return 'Nat'

    @staticmethod
    def _type_name(t: str) -> str:
        last = t.split('→')[-1].strip()
        if last in _INT_TYPES:
            return 'Int'
        if last in _RAT_TYPES:
            return 'Rat'
        if last in _REAL_TYPES:
            return 'Real'
        if last in _NAT_TYPES:
            return 'Nat'
        if last == 'Bool':
            return 'Bool'
        if last.startswith('List') or '×' in last:
            return 'Nat'   # containers: element arithmetic defaults to Nat
        raise Unsupported(f'result type {t}')

    # ---- numeric evaluation at a given type
    def num(self, node, env, typ):
        if isinstance(node, Num):
            if typ in _FIELD_TYPES:
                return self._rec(Fraction(node.value))
            if not isinstance(node.value, int):
                raise Unsupported('scientific literal outside a rational or real context (Lean defaults it to Float)')
            return self._rec(node.value)
        if isinstance(node, Ascribe):
            # `(e : T)` elaborates e at T (Lean pushes the expected type to the leaves: `(n - 1 : ℝ)` is real
            # subtraction of ↑n, `((n - 1 : ℕ) : ℝ)` casts a truncated Nat); the value is then used at typ
            inner = self._type_name(node.type)
            return self._rec(_coerce(self.num(node.expr, env, inner), inner, typ))
        if isinstance(node, Var):
            if node.name in ('true', 'false'):
                return node.name == 'true'
            if node.name in env:
                v = env[node.name]
                return self._rec(v.value if isinstance(v, Bound) else v)
            if '.' in node.name and not node.name.startswith('.'):
                recv, method = node.name.rsplit('.', 1)
                if (recv in env or recv in self.module.defs) and method in _LIST_METHODS:
                    return self._rec(self.apply(App(Var('.' + method), [Var(recv)]), env, typ))
            d = self.module.defs.get(node.name)
            if d is not None:
                if d.params or d.clauses:
                    raise Unsupported(f'definition {node.name} used without arguments')
                if d.body is None:
                    raise Unsupported(f'definition {node.name} has an unparsed body')
                dtyp = self._type_name(d.result_type) if d.result_type else typ
                return self._rec(self.num(d.body, {}, dtyp))
            raise Unsupported(f'unknown name {node.name}')
        if isinstance(node, Neg):
            if typ == 'Nat':
                raise Unsupported('negation at type Nat')
            return self._rec(-self.num(node.expr, env, typ))
        if isinstance(node, BinOp):
            if node.op == '^':
                return self._rec(self._power(node, env, typ))
            a = self.num(node.left, env, typ)
            b = self.num(node.right, env, typ)
            return self._rec(self._arith(node.op, a, b, typ))
        if isinstance(node, App):
            return self._rec(self.apply(node, env, typ))
        if isinstance(node, ListLit):
            return [self.num(x, env, typ) for x in node.items]
        if isinstance(node, Tuple):
            return tuple(self.num(x, env, typ) for x in node.items)
        if isinstance(node, Proj):
            v = self.num(node.expr, env, typ)
            if not isinstance(v, tuple) or node.index < 1 or node.index > len(v):
                raise Unsupported('projection of a non-pair')
            return self._rec(v[node.index - 1])
        if isinstance(node, IfThenElse):
            return self.num(node.then if self.truth(node.cond, env) else node.other, env, typ)
        if isinstance(node, BoolNot):
            return not self.truth(node.expr, env)
        raise Unsupported(f'expression {type(node).__name__}')

    def truth(self, node, env) -> bool:
        """A Bool-valued expression (`true`, `!`, `&&`, `||`, a Bool def) or a decidable proposition."""
        if isinstance(node, Var) and node.name in ('true', 'false'):
            return node.name == 'true'
        if isinstance(node, BoolNot):
            return not self.truth(node.expr, env)
        if isinstance(node, BinOp) and node.op in ('&&', '||'):
            a = self.truth(node.left, env)
            return (a and self.truth(node.right, env)) if node.op == '&&' else (a or self.truth(node.right, env))
        if isinstance(node, BinOp) and (node.op in _COMPARISONS or node.op in _LOGIC):
            return self.prop(node, env)
        if isinstance(node, Not):
            return self.prop(node, env)
        v = self.num(node, env, 'Nat')
        if isinstance(v, bool):
            return v
        raise Unsupported('condition is not Bool-valued')

    def _rec(self, v):
        if isinstance(v, float):
            raise AssertionError('floating point value produced')  # pragma: no cover - A3 guard
        if isinstance(v, (int, Fraction, Alg)) and not isinstance(v, bool):
            self.values_seen.append(v)
        if isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, float):
                    raise AssertionError('floating point value produced')  # pragma: no cover - A3 guard
        return v

    @staticmethod
    def _arith(op, a, b, typ):
        if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
            raise Unsupported('arithmetic on a non-numeric value')
        if op == '+':
            return a + b
        if op == '-':
            if typ == 'Nat':
                return a - b if a >= b else 0
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if typ in _FIELD_TYPES:
                # field division, totalized: x / 0 = 0 (`div_zero`); ints are coerced so no float can arise
                return Fraction(0) if b == 0 else _exact(a) / _exact(b)
            if b == 0:
                return 0                      # totalized: x / 0 = 0 (Nat.div, Int.ediv)
            if typ == 'Nat':
                return a // b
            return _ediv(a, b)
        if op == '%':
            if typ in _FIELD_TYPES:
                raise Unsupported(f'modulus at type {typ}')
            if b == 0:
                return a                      # totalized: x % 0 = x (Nat.mod, Int.emod)
            if typ == 'Nat':
                return a % b
            return a - b * _ediv(a, b)        # Int.emod: 0 ≤ r < |b|
        raise Unsupported(f'operator {op}')

    def _power(self, node: BinOp, env, typ):
        """`a ^ b`: the exponent is elaborated on its own (Lean's `rightact%`), so a literal exponent is a Nat
        (`Monoid.npow`) whatever the base; an Int exponent is `zpow` on a field (`x ^ (-n : ℤ) = (x ^ n)⁻¹`,
        with `0⁻¹ = 0`); a real exponent is `rpow`, outside the exact backend."""
        base = self.num(node.left, env, typ)
        if isinstance(base, (list, tuple, bool)):
            raise Unsupported('arithmetic on a non-numeric value')
        et = self.arith_type(node.right, env)
        if et == 'Nat':
            exp = self.num(node.right, env, 'Nat')
        elif et == 'Int':
            exp = self.num(node.right, env, 'Int')
            if exp < 0 and typ not in _FIELD_TYPES:
                raise Unsupported('negative exponent outside a field')
        else:
            raise Unsupported('real exponent (Real.rpow) is outside the exact backend')
        if isinstance(exp, (list, tuple, bool)):
            raise Unsupported('arithmetic on a non-numeric value')
        if exp < 0:
            return _inv(_pow(base, -exp))
        return _pow(base, exp)

    def apply(self, node: App, env, typ):
        fn = node.fn
        args = node.args
        if isinstance(fn, Lambda):
            return self.call_closure(Closure(fn.params, fn.body, env), args, env, typ)
        if not isinstance(fn, Var):
            raise Unsupported('application of a non-name')
        name = fn.name
        if '.' in name and not name.startswith('.'):
            recv, method = name.rsplit('.', 1)
            if (recv in env or recv in self.module.defs) and method in _LIST_METHODS:
                return self.apply(App(Var('.' + method), [Var(recv)] + list(args)), env, typ)
        if name in env and isinstance(env[name], Closure):
            return self.call_closure(env[name], args, env, typ)
        if name in _TRANSCENDENTAL:
            raise Unsupported(f'transcendental {name} is outside the exact algebraic backend')
        if name == 'Real.sqrt' and len(args) == 1:
            return _real_sqrt(self.num(args[0], env, 'Real'))
        if name == 'abs' and len(args) == 1:
            t = _max_type(self.arith_type(args[0], env), typ) if typ != 'Nat' else self.arith_type(args[0], env)
            v = self.num(args[0], env, t)
            if isinstance(v, (list, tuple, bool)):
                raise Unsupported('abs of a non-numeric value')
            return _abs(v)
        if name == 'Inv.inv' and len(args) == 1:
            if typ not in _FIELD_TYPES:
                raise Unsupported(f'inverse at type {typ}')
            return _inv(self.num(args[0], env, typ))
        if name == 'List.range' and len(args) == 1:
            n = self.num(args[0], env, 'Nat')
            if n > _MAX_LOOP:
                raise Infeasible(f'List.range {n} exceeds the loop cap')
            return list(range(n))
        if name in ('.foldl', 'List.foldl'):
            recv, f, init = (args[0], args[1], args[2]) if name == '.foldl' else (args[2], args[0], args[1])
            lst = self.num(recv, env, 'Nat')
            if not isinstance(lst, list):
                raise Unsupported('foldl on a non-list')
            acc = self.num(init, env, typ)
            closure = self._closure(f, env)
            for x in lst:
                acc = self.call_values(closure, [acc, x], typ)
            return acc
        if name in ('.filter', 'List.filter'):
            recv, f = (args[0], args[1]) if name == '.filter' else (args[1], args[0])
            lst = self.num(recv, env, typ)
            if not isinstance(lst, list):
                raise Unsupported('filter on a non-list')
            closure = self._closure(f, env)
            kept = []
            for x in lst:
                r = self.call_values(closure, [x], typ)
                if not isinstance(r, bool):
                    raise Unsupported('filter predicate is not Bool-valued')
                if r:
                    kept.append(x)
            return kept
        if name in ('.sum', 'List.sum'):
            lst = self.num(args[0], env, typ)
            return sum(lst)
        if name in ('.length', 'List.length'):
            lst = self.num(args[0], env, 'Nat')
            return len(lst)
        if name in ('.map', 'List.map'):
            recv, f = (args[0], args[1]) if name == '.map' else (args[1], args[0])
            lst = self.num(recv, env, typ)
            closure = self._closure(f, env)
            return [self.call_values(closure, [x], typ) for x in lst]
        if name in ('Nat.choose',) and len(args) == 2:
            n, k = self.num(args[0], env, 'Nat'), self.num(args[1], env, 'Nat')
            return _choose(n, k)
        if name in ('Nat.factorial',) and len(args) == 1:
            n = self.num(args[0], env, 'Nat')
            if n > _MAX_PRODUCT_LOOP:
                raise Infeasible(f'Nat.factorial {n} exceeds the product-loop cap')
            out = 1
            for m in range(2, n + 1):
                out *= m
            return out
        if name == 'Nat.succ' and len(args) == 1:
            return self.num(args[0], env, 'Nat') + 1
        if name == 'Nat.pred' and len(args) == 1:
            v = self.num(args[0], env, 'Nat')
            return v - 1 if v > 0 else 0
        if name in ('Nat.min', 'min') and len(args) == 2:
            return min(self.num(args[0], env, typ), self.num(args[1], env, typ))
        if name in ('Nat.max', 'max') and len(args) == 2:
            return max(self.num(args[0], env, typ), self.num(args[1], env, typ))
        if name == 'Int.natAbs' and len(args) == 1:
            return abs(self.num(args[0], env, 'Int'))
        if name == 'Int.toNat' and len(args) == 1:
            return max(0, self.num(args[0], env, 'Int'))
        if name == 'Int.ofNat' and len(args) == 1:
            return self.num(args[0], env, 'Nat')
        d = self.module.defs.get(name)
        if d is None:
            raise Unsupported(f'unknown function {name}')
        dtyp = self._type_name(d.result_type) if d.result_type else typ
        if d.clauses:
            n_lead = len(d.params)
            n_pats = len(d.clauses[0][0])
            if len(args) != n_lead + n_pats:
                raise Unsupported(f'{name} applied to {len(args)} of {n_lead + n_pats} arguments')
            local = {}
            for (pname, ptype), a in zip(d.params, args[:n_lead]):
                pt = self._type_name(ptype)
                local[pname] = Bound(self.num(a, env, pt), pt)
            arrow_types = [t.strip() for t in (d.result_type or '').split('→')[:-1]] if d.result_type else []
            vals = []
            for k, a in enumerate(args[n_lead:]):
                at = self._type_name(arrow_types[k]) if k < len(arrow_types) else 'Nat'
                vals.append((self.num(a, env, at), at))
            for pats, expr in d.clauses:
                bindings = [_match(p, v) for p, (v, _) in zip(pats, vals)]
                if any(b is None for b in bindings):
                    continue
                if expr is None:
                    raise Unsupported(f'unparsed clause of {name}')
                scope = dict(local)
                for b, (_, at) in zip(bindings, vals):
                    scope.update({k: Bound(val, at) for k, val in b.items()})
                return self.num(expr, scope, dtyp)
            raise Unsupported(f'no clause of {name} matches {[v for v, _ in vals]}')
        if d.body is None:
            raise Unsupported(f'definition {name} has an unparsed body')
        if len(args) != len(d.params):
            raise Unsupported(f'{name} applied to {len(args)} of {len(d.params)} arguments')
        local = {}
        for (pname, ptype), a in zip(d.params, args):
            pt = self._type_name(ptype)
            local[pname] = Bound(self.num(a, env, pt), pt)
        return self.num(d.body, local, dtyp)

    def _closure(self, f, env):
        if isinstance(f, Lambda):
            return Closure(f.params, f.body, env)
        if isinstance(f, Var) and f.name in _BUILTIN_BINARY:
            return _BUILTIN_BINARY[f.name]
        if isinstance(f, Var) and f.name in env and isinstance(env[f.name], Closure):
            return env[f.name]
        if isinstance(f, Var) and f.name in self.module.defs:
            d = self.module.defs[f.name]
            if d.body is not None and d.params:
                return Closure([p for p, _ in d.params], d.body, {})
        raise Unsupported('higher-order argument is not a lambda')

    def call_closure(self, closure, arg_nodes, env, typ):
        vals = [self.num(a, env, typ) for a in arg_nodes]
        return self.call_values(closure, vals, typ)

    def call_values(self, closure, vals, typ):
        if callable(closure) and not isinstance(closure, Closure):
            return closure(*vals)
        if len(vals) != len(closure.params):
            raise Unsupported('lambda arity mismatch')
        local = dict(closure.env)
        for p, v in zip(closure.params, vals):
            local[p] = Bound(v, typ)
        return self.num(closure.body, local, typ)

    # ---- bounded universal quantification
    def forall(self, node: 'Forall', env) -> bool:
        var_type = 'Int' if node.type in _INT_TYPES else 'Nat'
        if node.type is not None and node.type not in _INT_TYPES | _NAT_TYPES:
            raise Unsupported(f'quantifier over {node.type}')
        domain, body = _domain_of(node, self, env)
        if domain is None:
            raise Unsupported(f'unbounded quantifier over {node.var}')
        self.instances += len(domain)
        if not domain:
            self.vacuous = True
        ok = True
        for v in domain:
            scope = dict(env)
            scope[node.var] = Bound(v, var_type)
            if not self.prop(body, scope):
                self.failures.append(f'{node.var} = {v}')
                ok = False
        if ok and domain and all(isinstance(v, int) and not isinstance(v, bool) for v in domain):
            lo, hi = min(domain), max(domain)
            for cand in sorted({lo - 1, hi + 1, 0, 2 * hi + 1}):
                if cand in domain or (cand < 0 and var_type == 'Nat'):
                    continue
                scope = dict(env)
                scope[node.var] = Bound(cand, var_type)
                try:
                    holds = self.prop(body, scope)
                except Unsupported:
                    continue
                self.probes.append({'var': node.var, 'value': cand, 'holds': holds})
        return ok

    # ---- propositions
    def prop(self, node, env) -> bool:
        if isinstance(node, Not):
            return not self.prop(node.expr, env)
        if isinstance(node, Forall):
            return self.forall(node, env)
        if isinstance(node, (BoolNot, Var)) or (isinstance(node, BinOp) and node.op in ('&&', '||')):
            if isinstance(node, Var) and node.name not in ('true', 'false'):
                v = self.num(node, env, 'Nat')
                if isinstance(v, bool):
                    return v
                raise Unsupported(f'proposition {node.name} is not Bool-valued')
            return self.truth(node, env)
        if isinstance(node, BinOp):
            if node.op == '∧':
                return self.prop(node.left, env) and self.prop(node.right, env)
            if node.op == '∨':
                return self.prop(node.left, env) or self.prop(node.right, env)
            if node.op == '→':
                return (not self.prop(node.left, env)) or self.prop(node.right, env)
            if node.op == '↔':
                return self.prop(node.left, env) == self.prop(node.right, env)
            if node.op in ('∈', '∉'):
                lst = self.num(node.right, env, 'Nat')
                if not isinstance(lst, list):
                    raise Unsupported('membership in a non-list')
                v = self.num(node.left, env, 'Nat')
                return (v in lst) if node.op == '∈' else (v not in lst)
            if node.op in _COMPARISONS:
                typ = _max_type(self.arith_type(node.left, env), self.arith_type(node.right, env))
                if typ == 'Bool':
                    typ = 'Nat'
                if typ == 'Sci':
                    raise Unsupported('scientific literal outside a rational or real context (Lean defaults it to Float)')
                a = self.num(node.left, env, typ)
                b = self.num(node.right, env, typ)
                if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
                    if node.op == '=':
                        return a == b
                    if node.op == '≠':
                        return a != b
                    raise Unsupported('ordering of non-numeric values')
                if typ in _FIELD_TYPES:
                    a, b = _exact(a), _exact(b)       # an int meets a Fraction or an Alg: compare exactly
                return {'=': a == b, '≠': a != b, '<': a < b, '>': a > b, '≤': a <= b, '≥': a >= b}[node.op]
            raise Unsupported(f'operator {node.op} in a proposition')
        if isinstance(node, Var) and node.name in ('True', 'False'):
            return node.name == 'True'
        if isinstance(node, Ascribe):
            return self.prop(node.expr, env)
        raise Unsupported(f'proposition of form {type(node).__name__}')


def _ediv(a: int, b: int) -> int:
    """Lean 4 core `Int./` is `Int.ediv` (Euclidean): the quotient q with a = b*q + r and 0 ≤ r < |b|.
    For b > 0 this is floor division; for b < 0 it is -(a // -b). Python's `//` floors for every sign of
    b and T-division rounds toward zero, so both differ from this for some sign combination
    (goldens: 5 / -2 = -2, -5 / -2 = 3, -5 / 2 = -3; main#684/6511137449)."""
    return a // b if b > 0 else -(a // (-b))


def _max_type(a, b):
    order = {'Sci': -2, 'Bool': -1, 'Nat': 0, 'Int': 1, 'Rat': 2, 'Real': 3, 'Prop': 4}
    if 'Prop' in (a, b):
        raise Unsupported('arithmetic on propositions')
    return a if order[a] >= order[b] else b


def _coerce(v, from_type, to_type):
    """Value of type ``from_type`` used where ``to_type`` is expected: ℕ/ℤ embed into ℚ/ℝ; a rational or
    real value has no coercion back to ℕ/ℤ in Lean (the statement would not elaborate)."""
    if isinstance(v, (list, tuple, bool)):
        return v
    if to_type in _FIELD_TYPES and from_type not in _FIELD_TYPES:
        return Fraction(v)
    if from_type in _FIELD_TYPES and to_type not in _FIELD_TYPES and to_type != 'Sci':
        raise Unsupported(f'a {from_type} value ascribed where a {to_type} is expected')
    return v


def _exact(v):
    """An int as a Fraction (field arithmetic never touches Python's `/` on ints)."""
    return Fraction(v) if isinstance(v, int) and not isinstance(v, bool) else v


def _bits(v) -> int:
    if isinstance(v, int):
        return v.bit_length()
    if isinstance(v, Fraction):
        return v.numerator.bit_length() + v.denominator.bit_length()
    if isinstance(v, Alg):
        return sum(_bits(c) for c in v.terms.values()) + 8 * len(v.terms)
    return 0


def _pow(a, n: int):
    """a ^ n for a Nat exponent, with a cap on the exact result size (an exponent such as 2^63 from a
    pathological sample is infeasible, not approximated)."""
    if n == 0:
        return 1 if isinstance(a, int) else Fraction(1)
    if _bits(a) * n > _MAX_RESULT_BITS or (isinstance(a, Alg) and n > 10_000):
        raise Infeasible(f'exponent {n} exceeds the exact-size cap')
    if isinstance(a, Alg):
        out, base = Fraction(1), a
        while n:
            if n & 1:
                out = out * base
            n >>= 1
            if n:
                base = base * base
        return out
    return a ** n


def _inv(v):
    """Lean's field inverse, totalized: `0⁻¹ = 0`."""
    if isinstance(v, Alg):
        return v.inverse()
    v = _exact(v)
    return Fraction(0) if v == 0 else 1 / v


def _abs(v):
    if isinstance(v, Alg):
        return v if v.sign() > 0 else -v
    return abs(v)


def _choose(n, k):
    if k < 0 or k > n:
        return 0
    if k > _MAX_PRODUCT_LOOP:
        raise Infeasible(f'Nat.choose with k = {k} exceeds the product-loop cap')
    out = 1
    for i in range(1, k + 1):
        out = out * (n - k + i) // i
    return out


def _match(pat, v):
    kind = pat[0]
    if kind == 'lit':
        return {} if v == pat[1] else None
    if kind == 'var':
        return {pat[1]: v}
    if kind == 'succ':
        return {pat[1]: v - pat[2]} if v >= pat[2] else None
    return None


# --------------------------------------------------------------------------------------------
# Exact real arithmetic in multi-quadratic extensions ℚ(√d₁, …, √dₖ)
# --------------------------------------------------------------------------------------------
# `Real.sqrt` of a rational is exact here: √(a/b) = (s/b)·√d with a·b = s²·d and d squarefree. Sums and
# products of such values stay in the field, division rationalizes the denominator, and equality is decided
# by the canonical form (the √d for distinct squarefree d are linearly independent over ℚ, so an element is
# zero iff every coefficient is zero). The sign of a nonzero element is decided by refining rational
# enclosures of the √d (integer square roots), never by floating point. Nested radicals (√ of an irrational
# value) and transcendental functions are outside this backend and reported UNSUPPORTED.

_SMALL_PRIMES = None


def _primes_up_to(n: int) -> list:
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(range(p * p, n + 1, p)))
    return [i for i, f in enumerate(sieve) if f]


def _isqrt_exact(n: int):
    r = _isqrt(n)
    return r if r * r == n else None


def _isqrt(n: int) -> int:
    if n < 0:
        raise ValueError('negative')
    if n < 2:
        return n
    x = 1 << ((n.bit_length() + 1) // 2)
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y


_SQUAREFREE_CACHE = {}


def _squarefree_split(n: int):
    """n = s² · d with d squarefree, for n ≥ 1, exactly: trial division by primes up to B removes the small
    square factors; the cofactor m then has all prime factors > B, so when m < B³ it is either a perfect
    square, a prime, or a product of two distinct primes — in every case its squarefree part is known. A
    cofactor outside that range is infeasible (not approximated)."""
    if n in _SQUAREFREE_CACHE:
        return _SQUAREFREE_CACHE[n]
    global _SMALL_PRIMES
    if _SMALL_PRIMES is None:
        _SMALL_PRIMES = _primes_up_to(100_000)
    s, d, m = 1, 1, n
    cofactor_is_prime = False
    for p in _SMALL_PRIMES:
        if p * p > m:
            cofactor_is_prime = m > 1       # every smaller prime has been removed
            break
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            s *= p ** (e // 2)
            if e % 2:
                d *= p
    if m > 1:
        r = _isqrt_exact(m)
        if cofactor_is_prime or r is None and m < 100_000 ** 3:
            d *= m           # a prime, or a product of two distinct primes > 100000: squarefree
        elif r is not None:
            s *= r
        else:
            raise Infeasible(f'radicand cofactor {m} is too large to split into square and squarefree parts exactly')
    _SQUAREFREE_CACHE[n] = (s, d)
    return s, d


class Alg:
    """Σ c_d √d over squarefree d ≥ 1 (d = 1 is the rational part), coefficients nonzero Fractions. Always
    irrational: results that collapse to a rational are returned as a Fraction by ``_alg``."""
    __slots__ = ('terms',)

    def __init__(self, terms):
        self.terms = terms

    def __repr__(self):
        return ' + '.join(f'{c}' if d == 1 else f'{c}·√{d}' for d, c in sorted(self.terms.items()))

    # ---- construction
    @staticmethod
    def _terms(x):
        if isinstance(x, Alg):
            return x.terms
        if isinstance(x, bool):
            raise Unsupported('arithmetic on a Bool')
        return {1: Fraction(x)} if x != 0 else {}

    def __hash__(self):
        return hash(tuple(sorted(self.terms.items())))

    # ---- ring operations
    def __add__(self, other):
        out = dict(self.terms)
        for d, c in Alg._terms(other).items():
            out[d] = out.get(d, Fraction(0)) + c
        return _alg(out)

    __radd__ = __add__

    def __neg__(self):
        return Alg({d: -c for d, c in self.terms.items()})

    def __sub__(self, other):
        return self + (-other if isinstance(other, Alg) else -Fraction(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        out = {}
        for d1, c1 in self.terms.items():
            for d2, c2 in Alg._terms(other).items():
                g = _gcd(d1, d2)
                d = (d1 // g) * (d2 // g)
                out[d] = out.get(d, Fraction(0)) + c1 * c2 * g
        return _alg(out)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Alg):
            return self * other.inverse()
        other = Fraction(other)
        if other == 0:
            return Fraction(0)                         # Lean: x / 0 = 0
        return Alg({d: c / other for d, c in self.terms.items()})

    def __rtruediv__(self, other):
        return Fraction(other) * self.inverse()

    def __pow__(self, n):
        return _pow(self, n)

    def inverse(self):
        """x⁻¹ by conjugation over a coprime base of the radicands: x = A + B√b with A, B free of √b, so
        x · (A − B√b) = A² − b·B² lies in the smaller field (and is nonzero when x is), and recursion ends at
        a rational."""
        radicands = [d for d in self.terms if d != 1]
        b = _coprime_base(radicands)[0]
        conj = Alg({d: (-c if d % b == 0 else c) for d, c in self.terms.items()})
        norm = self * conj
        if isinstance(norm, Alg) and any(d % b == 0 for d in norm.terms):
            raise AssertionError('conjugation did not eliminate the radical')   # pragma: no cover
        return conj * _inv(norm)

    # ---- order
    def sign(self) -> int:
        """Exact sign: the element is nonzero (canonical form), so an enclosure of each √d by integer square
        roots at increasing precision eventually separates the sum from 0."""
        for bits in (64, 128, 256, 512, 1024, 2048, 4096):
            lo = hi = Fraction(0)
            scale = 1 << bits
            for d, c in self.terms.items():
                if d == 1:
                    lo += c
                    hi += c
                    continue
                r = _isqrt(d << (2 * bits))
                a, b = Fraction(r, scale), Fraction(r + 1, scale)
                if c >= 0:
                    lo += c * a
                    hi += c * b
                else:
                    lo += c * b
                    hi += c * a
            if lo > 0:
                return 1
            if hi < 0:
                return -1
        raise Unsupported('sign of an algebraic value undecided at 4096 bits')

    def _cmp(self, other):
        diff = self - other
        if isinstance(diff, Alg):
            return diff.sign()
        return (diff > 0) - (diff < 0)

    def __eq__(self, other):
        return isinstance(other, Alg) and self.terms == other.terms

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ge__(self, other):
        return self._cmp(other) >= 0


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _alg(terms: dict):
    terms = {d: c for d, c in terms.items() if c != 0}
    if not terms:
        return Fraction(0)
    if set(terms) == {1}:
        return terms[1]
    return Alg(terms)


def _coprime_base(radicands: list) -> list:
    """Pairwise coprime integers > 1 such that every radicand is a product of a subset (gcd refinement)."""
    base = sorted({d for d in radicands if d > 1})
    changed = True
    while changed:
        changed = False
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                a, b = base[i], base[j]
                g = _gcd(a, b)
                if g > 1:
                    base = sorted({x for x in base if x not in (a, b)} | {x for x in (g, a // g, b // g) if x > 1})
                    changed = True
                    break
            if changed:
                break
    return base


def _real_sqrt(v):
    """Mathlib `Real.sqrt`: 0 on non-positive arguments (`Real.sqrt_eq_zero'`), exact otherwise."""
    if isinstance(v, Alg):
        if v.sign() <= 0:
            return Fraction(0)
        raise Unsupported('nested radical: Real.sqrt of an irrational value is outside the exact backend')
    v = _exact(v)
    if isinstance(v, bool) or not isinstance(v, Fraction):
        raise Unsupported('Real.sqrt of a non-numeric value')
    if v <= 0:
        return Fraction(0)
    s, d = _squarefree_split(v.numerator * v.denominator)
    coeff = Fraction(s, v.denominator)
    return coeff if d == 1 else Alg({d: coeff})


# --------------------------------------------------------------------------------------------
# Theorem evaluation with quantifier enumeration and boundary probes
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class EvalResult:
    status: str                 # PASS | CORROBORATED | FAIL | UNSUPPORTED
    detail: str
    instances: int
    vacuous: bool
    probes: list                # [{'value': v, 'holds': bool}] outside-domain probes (information only)
    values: list                # every intermediate numeric value (A3 audit: never a float)
    sampling: dict | None = None    # for statements with sampled binders: seed, draws, accepted, rejected


HOLDS = ('PASS', 'CORROBORATED')    # the two verdicts under which no counterexample was found


def _domain_of(forall: Forall, ev: Evaluator, env=None):
    """The finite domain of a universally quantified variable, or None when it is not finite."""
    env = env or {}
    if forall.domain is not None:
        lst = ev.num(forall.domain, env, 'Nat')
        return (lst, forall.body) if isinstance(lst, list) else (None, forall.body)
    body = forall.body
    if isinstance(body, BinOp) and body.op == '→':
        hyp = body.left
        if isinstance(hyp, BinOp) and hyp.op == '∈' and isinstance(hyp.left, Var) and hyp.left.name == forall.var:
            lst = ev.num(hyp.right, env, 'Nat')
            if isinstance(lst, list):
                return lst, body.right
        # A bound gives a finite domain only for an explicit Nat binder: an Int or Rat binder, or an
        # untyped one that Lean may unify with Int elsewhere in the body, is unbounded below.
        if (isinstance(hyp, BinOp) and hyp.op in ('≤', '<') and isinstance(hyp.left, Var)
                and hyp.left.name == forall.var and forall.type in _NAT_TYPES):
            bound = ev.num(hyp.right, env, 'Nat')
            if isinstance(bound, int) and not isinstance(bound, bool) and bound <= 100000:
                return list(range(bound + 1 if hyp.op == '≤' else bound)), body.right
    return None, body


_EVAL_ERRORS = (SyntaxError, RecursionError, ZeroDivisionError, OverflowError, KeyError, IndexError, TypeError, ValueError)


def evaluate_theorem(module: Module, thm: Theorem, seed: int = 1, samples: int = 64) -> EvalResult:
    """A closed statement is decided exactly: PASS / FAIL. A statement with sampled binders before the colon
    is tested on ``samples`` hypothesis-satisfying instances drawn deterministically from ``seed``:
    FAIL with the exact witness, CORROBORATED when no instance refutes it (admitted evidence, never PASS),
    UNSUPPORTED when a binder, a hypothesis or the conclusion is outside the interpreter or no sample
    satisfies the hypotheses (possibly vacuous: A5)."""
    if thm.statement is None:
        return EvalResult('UNSUPPORTED', thm.error or 'no statement', 0, False, [], [])
    if thm.binders:
        return _evaluate_sampled(module, thm, seed, samples)
    ev = Evaluator(module)
    try:
        ok = ev.prop(thm.statement, {})
        quantified = ev.instances > 0 or ev.vacuous
        if not ok:
            detail = ('fails at ' + ', '.join(ev.failures)) if ev.failures else 'closed statement evaluated false'
            return EvalResult('FAIL', detail, ev.instances if quantified else 1, False, ev.probes, ev.values_seen)
        if quantified:
            detail = f'holds on all {ev.instances} quantified instances' + (' (an empty domain: vacuous)' if ev.vacuous else '')
        else:
            detail = 'closed statement evaluated true'
        return EvalResult('PASS', detail, ev.instances if quantified else 1, ev.vacuous, ev.probes, ev.values_seen)
    except Unsupported as e:
        return EvalResult('UNSUPPORTED', str(e), 0, False, [], ev.values_seen)
    except _EVAL_ERRORS as e:
        return EvalResult('UNSUPPORTED', f'evaluation error: {type(e).__name__}: {e}', 0, False, [], ev.values_seen)


def check_module(source: str, seed: int = 1, samples: int = 64):
    module = parse_module(source)
    return [evaluate_theorem(module, t, seed, samples) for t in module.theorems]


# --------------------------------------------------------------------------------------------
# Sampled evaluation of statements with binders before the colon
# --------------------------------------------------------------------------------------------
# The sampler never proves anything: it looks for a counterexample among exact rationals (and small
# integers for ℕ/ℤ binders) chosen to be pathological — 0, ±1, ±tiny, ±huge, perfect squares and
# non-squares, repeating decimals — plus the boundary values the hypotheses themselves name (`0 ≤ r`,
# `r ≤ 1`, `|a - b| ≤ δ` contribute 0, 1, b ± δ …), then random rationals. Variables are drawn in
# declaration order and each hypothesis is checked as soon as its variables are assigned, so constrained
# domains (`m2 ^ 2 < m4`) are sampled efficiently without any float.


_F = Fraction
_REAL_POOL = [_F(0), _F(1), _F(-1), _F(1, 2), _F(-1, 2), _F(2), _F(-2), _F(10 ** 6), _F(-10 ** 6), _F(1, 10 ** 6),
              _F(-1, 10 ** 6), _F(3), _F(-3), _F(1, 3), _F(2, 3), _F(1, 4), _F(3, 4), _F(-1, 4), _F(4), _F(9), _F(-4),
              _F(10), _F(-10), _F(1, 10), _F(1, 7), _F(22, 7), _F(3, 2), _F(5, 4), _F(100), _F(-100), _F(10 ** 12),
              _F(1, 2 ** 40), _F(355, 113), _F(-7, 3), _F(2 ** 31), _F(-(2 ** 31)), _F(999999, 1000000),
              _F(1000001, 1000000), _F(9, 4), _F(1, 9), _F(8), _F(27), _F(-1, 1000)]
_INT_POOL = [0, 1, -1, 2, -2, 10 ** 6, -10 ** 6, 3, -3, 7, -7, 10, -10, 100, -100, 2 ** 31 - 1, -(2 ** 31),
             2 ** 63, -(2 ** 63), 12, 60, -60, 1000, -1000]
_NAT_POOL = [0, 1, 2, 3, 4, 5, 10 ** 6, 6, 7, 8, 9, 10, 16, 31, 32, 63, 64, 100, 127, 128, 255, 256, 1000, 1023,
             1024, 65535, 65536, 2 ** 31 - 1, 2 ** 31, 2 ** 32, 2 ** 63, 10 ** 18, 12, 24, 60]
_POOLS = {'Real': _REAL_POOL, 'Rat': _REAL_POOL, 'Int': _INT_POOL, 'Nat': _NAT_POOL}
_DIAGONAL = 8                   # structured points first: every variable takes pool[i] for i < _DIAGONAL
_DRAW_RETRIES = 24              # redraws of one variable against the hypotheses it completes
_MAX_ATTEMPTS_PER_SAMPLE = 16


def _fmt(v) -> str:
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f'{v.numerator}/{v.denominator}'
    return str(v)


def _coerce_sample(v, typ):
    """A pool value as a sample of ``typ`` (None when it is not representable there)."""
    if typ in _FIELD_TYPES:
        return Fraction(v)
    if isinstance(v, Fraction):
        if v.denominator != 1:
            return None
        v = v.numerator
    if typ == 'Nat' and v < 0:
        return None
    return int(v)


def _random_value(rng: random.Random, typ):
    k = rng.randint(1, 9)
    num = rng.randint(-10 ** k, 10 ** k)
    if typ in _FIELD_TYPES:
        den = rng.randint(1, 10 ** rng.randint(0, 6))
        return Fraction(num, den)
    return abs(num) if typ == 'Nat' else num


def _split_conjuncts(node):
    if isinstance(node, BinOp) and node.op == '∧':
        return _split_conjuncts(node.left) + _split_conjuncts(node.right)
    return [node]


def _boundary_candidates(var: str, typ: str, hyps: list, ev: Evaluator, env: dict, var_names: set) -> list:
    """Exact values at and next to the boundaries that the hypotheses give ``var`` once every other
    variable they mention is assigned: `c ≤ x`, `x < c`, `x = c`, `x ≠ c`, `|x - e| ≤ c` (and `|e - x|`)."""
    out = []
    eps = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 10 ** 6), Fraction(-1, 10 ** 6)] if typ in _FIELD_TYPES \
        else [0, 1, -1]

    def closed(node):
        return all(v in env for v in _free_vars(node) & var_names)

    def value(node):
        try:
            v = ev.num(node, env, typ)
        except Unsupported:
            return None
        if isinstance(v, (list, tuple, bool, Alg)):
            return None
        return v

    def centre(node, other):
        # node is `|x - e|` or `|e - x|`: candidates e ± other
        if isinstance(node, App) and isinstance(node.fn, Var) and node.fn.name == 'abs' and len(node.args) == 1 \
                and isinstance(node.args[0], BinOp) and node.args[0].op == '-':
            l, r = node.args[0].left, node.args[0].right
            e = r if isinstance(l, Var) and l.name == var else (l if isinstance(r, Var) and r.name == var else None)
            if e is not None and closed(e) and closed(other):
                ev_, ov = value(e), value(other)
                if ev_ is not None and ov is not None:
                    return [ev_ + ov, ev_ - ov, ev_]
        return []

    for hyp in hyps:
        for h in _split_conjuncts(hyp):
            if not (isinstance(h, BinOp) and h.op in ('≤', '<', '≥', '>', '=', '≠')):
                continue
            for side, other in ((h.left, h.right), (h.right, h.left)):
                if isinstance(side, Var) and side.name == var and closed(other):
                    c = value(other)
                    if c is not None:
                        out.extend(c + e for e in eps)
                else:
                    out.extend(centre(side, other))
    seen, uniq = set(), []
    for v in out:
        v = _coerce_sample(v, typ)
        if v is not None and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _evaluate_sampled(module: Module, thm: Theorem, seed: int, samples: int) -> EvalResult:
    variables = [(b.name, b.type) for b in thm.binders if b.kind == 'var']
    hyps = [b.type for b in thm.binders if b.kind == 'hyp']
    ev = Evaluator(module)
    rng = random.Random(seed)
    accepted, rejected, infeasible, draws = 0, 0, 0, 0
    seen = set()
    probes = []
    try:
        if not variables:
            # hypotheses only: a closed statement under closed hypotheses
            for h in hyps:
                if not ev.prop(h, {}):
                    return EvalResult('UNSUPPORTED', 'no sample satisfied the hypotheses: a closed hypothesis is false '
                                      '(the theorem is vacuous, A5)', 0, True, [], ev.values_seen,
                                      {'seed': seed, 'draws': 0, 'accepted': 0, 'rejected': 1})
            return evaluate_theorem(module, dataclasses.replace(thm, binders=[]))

        var_names = {n for n, _ in variables}

        def hyps_completed_by(index):
            # the hypotheses whose last sampled variable (in declaration order) is variables[index]
            name = variables[index][0]
            later = {n for n, _ in variables[index + 1:]}
            return [h for h in hyps if name in _free_vars(h) and not (_free_vars(h) & later)]

        completed = [hyps_completed_by(i) for i in range(len(variables))]
        for h in hyps:
            if not (_free_vars(h) & var_names) and not ev.prop(h, {}):
                return EvalResult('UNSUPPORTED', 'no sample satisfied the hypotheses: a closed hypothesis is false '
                                  '(the theorem is vacuous, A5)', 0, True, [], ev.values_seen,
                                  {'seed': seed, 'draws': 0, 'accepted': 0, 'rejected': 1})

        def hyps_hold(hs, env):
            try:
                return all(ev.prop(h, env) for h in hs)
            except Infeasible:
                return False

        def draw(index, env):
            name, typ = variables[index]
            pool = _POOLS[typ]
            boundaries = _boundary_candidates(name, typ, hyps, ev, env, var_names)
            for _ in range(_DRAW_RETRIES):
                r = rng.random()
                if boundaries and r < 0.35:
                    v = rng.choice(boundaries)
                elif r < 0.75:
                    v = _coerce_sample(rng.choice(pool), typ)
                    if v is None:
                        continue
                else:
                    v = _random_value(rng, typ)
                env[name] = Bound(v, typ)
                if hyps_hold(completed[index], env):
                    return True
            return False

        def check(env):
            nonlocal accepted, rejected, infeasible, draws
            draws += 1
            key = tuple(env[n].value for n, _ in variables)
            if key in seen:
                return None
            seen.add(key)
            try:
                ok = ev.prop(thm.statement, env)
            except Infeasible:
                infeasible += 1
                return None
            accepted += 1
            return ok

        def refuted(env):
            witness = ', '.join(f'{n} = {_fmt(env[n].value)}' for n, _ in variables)
            detail = (f'refuted at {witness} (sample {draws}; {accepted - 1} sampled instances held before it); '
                      'the hypotheses hold at the witness')
            return EvalResult('FAIL', detail, accepted, False, probes, ev.values_seen,
                              {'seed': seed, 'draws': draws, 'accepted': accepted, 'rejected': rejected,
                               'infeasible': infeasible, 'witness': {n: _fmt(env[n].value) for n, _ in variables}})

        # structured points: every variable takes the i-th pool value
        for i in range(_DIAGONAL):
            env = {}
            for name, typ in variables:
                v = _coerce_sample(_POOLS[typ][i], typ)
                if v is None:
                    break
                env[name] = Bound(v, typ)
            if len(env) != len(variables):
                continue
            if not hyps_hold(hyps, env):
                rejected += 1
                continue
            ok = check(env)
            if ok is False:
                return refuted(env)
        # hypothesis-directed random samples
        attempts = 0
        while accepted < samples and attempts < samples * _MAX_ATTEMPTS_PER_SAMPLE:
            attempts += 1
            env = {}
            if not all(draw(i, env) for i in range(len(variables))):
                rejected += 1
                continue
            ok = check(env)
            if ok is False:
                return refuted(env)
        # information only (A6): the conclusion at hypothesis boundaries the hypotheses exclude
        for name, typ in variables:
            if len(probes) >= 8:
                break
            env = {n: Bound(_coerce_sample(_POOLS[t][1], t), t) for n, t in variables}
            for v in _boundary_candidates(name, typ, hyps, ev, env, var_names):
                env[name] = Bound(v, typ)
                try:
                    if all(ev.prop(h, env) for h in hyps):
                        continue
                    holds = ev.prop(thm.statement, env)
                except Unsupported:
                    continue
                probes.append({'var': name, 'value': _fmt(v), 'holds': holds, 'outside_hypotheses': True})
                if len(probes) >= 8:
                    break
        sampling = {'seed': seed, 'draws': draws, 'accepted': accepted, 'rejected': rejected, 'infeasible': infeasible}
        if accepted == 0:
            why = 'every drawn instance was infeasible to evaluate exactly' if infeasible and not rejected else \
                'the hypotheses may be unsatisfiable (vacuous, A5) or outside what the sampler can hit'
            return EvalResult('UNSUPPORTED', f'no sample satisfied the hypotheses and evaluated ({draws + rejected} draws): {why}',
                              0, False, probes, ev.values_seen, sampling)
        detail = (f'holds on {accepted} sampled instances satisfying the hypotheses ({rejected} draws rejected by the '
                  f'hypotheses, {infeasible} infeasible, seed {seed}): sampling is admitted evidence of the statement, '
                  'not a proof of it')
        return EvalResult('CORROBORATED', detail, accepted, False, probes, ev.values_seen, sampling)
    except Unsupported as e:
        return EvalResult('UNSUPPORTED', str(e), 0, False, [], ev.values_seen)
    except _EVAL_ERRORS as e:
        return EvalResult('UNSUPPORTED', f'evaluation error: {type(e).__name__}: {e}', 0, False, [], ev.values_seen)


# --------------------------------------------------------------------------------------------
# Mutation self-test (A8): every guaranteed-false mutant of a PASS statement must be reported FAIL
# --------------------------------------------------------------------------------------------

_NEGATE = {'=': '≠', '≠': '=', '<': '≥', '≥': '<', '≤': '>', '>': '≤'}
_WEAKEN = {'<': '≤', '>': '≥'}


def _walk_comparisons(stmt, visit):
    """Call ``visit(node, rebuild)`` for every comparison reachable through ∧, → (consequent) and ∀."""
    def walk(node, rebuild):
        if isinstance(node, Forall):
            walk(node.body, lambda b: rebuild(Forall(node.var, node.type, node.domain, b)))
        elif isinstance(node, BinOp) and node.op == '∧':
            walk(node.left, lambda l: rebuild(BinOp(node.op, l, node.right)))
            walk(node.right, lambda r: rebuild(BinOp(node.op, node.left, r)))
        elif isinstance(node, BinOp) and node.op == '→':
            walk(node.right, lambda r: rebuild(BinOp(node.op, node.left, r)))
        elif isinstance(node, BinOp) and node.op in _NEGATE:
            visit(node, rebuild)
    walk(stmt, lambda s: s)


def required_mutants(stmt):
    """Class 1 — guaranteed-false mutants: false whenever the original is true. A comparison replaced by
    its negation (one conjunct at a time), and the right-hand literal of an equality shifted by one.
    Every one of these must be reported FAIL."""
    out = []

    def visit(node, rebuild):
        out.append((f'negate {node.op}', rebuild(BinOp(_NEGATE[node.op], node.left, node.right))))
        if node.op == '=' and isinstance(node.right, Num):
            out.append(('shift rhs literal +1', rebuild(BinOp('=', node.left, Num(node.right.value + 1)))))
    _walk_comparisons(stmt, visit)
    return out


def weakening_mutants(stmt):
    """Class 2 — truth-preserving weakenings (`<`→`≤`, `>`→`≥`): true whenever the original is true, so
    every one of these must still be reported PASS; a rejection is an evaluator defect, not a kill
    (Codex finding main#684/6510829073)."""
    out = []

    def visit(node, rebuild):
        if node.op in _WEAKEN:
            out.append((f'weaken {node.op} to {_WEAKEN[node.op]}', rebuild(BinOp(_WEAKEN[node.op], node.left, node.right))))
    _walk_comparisons(stmt, visit)
    return out


def mutation_selftest(module: Module, evaluate=evaluate_theorem):
    """Harness control (unmutated set PASS or CORROBORATED) + class 1 (must FAIL) + class 2 (must hold). An
    UNSUPPORTED or errored mutant is a survivor / a rejection, never a kill. Mutants keep the theorem's
    binders: a class-1 mutant of a sampled statement is false at every instance, so sampling must refute it;
    a class-2 weakening stays true at every instance, so sampling must not."""
    baseline = [(t, evaluate(module, t)) for t in module.theorems]
    control_passed = all(r.status in HOLDS for _, r in baseline)
    mutants = weakenings = 0
    survivor_list, rejected = [], []
    for thm, res in baseline:
        if res.status not in HOLDS or thm.statement is None:
            continue
        for label, mutated in required_mutants(thm.statement):
            mutants += 1
            r = evaluate(module, dataclasses.replace(thm, statement=mutated, error=None))
            if r.status != 'FAIL':
                survivor_list.append({'theorem': thm.name, 'mutant': label, 'status': r.status})
        for label, mutated in weakening_mutants(thm.statement):
            weakenings += 1
            r = evaluate(module, dataclasses.replace(thm, statement=mutated, error=None))
            if r.status not in HOLDS:
                rejected.append({'theorem': thm.name, 'mutant': label, 'status': r.status})
    return {'theorems': len(baseline), 'control_passed': control_passed,
            'mutants': mutants, 'killed': mutants - len(survivor_list), 'survivors': survivor_list,
            'weakenings': weakenings, 'weakenings_rejected': rejected,
            'harness_ok': control_passed and not survivor_list and not rejected and mutants > 0}


def declared_controls(source: str, module_rel: str, controls: dict):
    """Class 3 — the manifest's declared negative controls: `replace`→`with` text edits whose falsity the
    package records. Each edit that changes a theorem *statement* must evaluate FAIL; an edit elsewhere
    (proof text, an injected `sorry` or axiom) is classified non-statement and not evaluated here."""
    base = parse_module(source)
    base_by_name = {t.name: t.text for t in base.theorems}
    out = {}
    for name, ctl in controls.items():
        if ctl.get('module') != module_rel:
            continue
        if source.count(ctl['replace']) != 1:
            out[name] = {'class': 'inapplicable', 'detail': f"replacement text occurs {source.count(ctl['replace'])} times"}
            continue
        mutated_src = source.replace(ctl['replace'], ctl['with'])
        mutated = parse_module(mutated_src)
        changed = [t for t in mutated.theorems if base_by_name.get(t.name) != t.text]
        if len(changed) != 1:
            out[name] = {'class': 'non-statement', 'detail': 'the edit changes no theorem statement' if not changed
                         else 'the edit changes several statements'}
            continue
        res = evaluate_theorem(mutated, changed[0])
        out[name] = {'class': 'statement', 'theorem': changed[0].name, 'status': res.status, 'detail': res.detail}
    statement = [v for v in out.values() if v['class'] == 'statement']
    out['_summary'] = {'statement_controls': len(statement),
                       'all_statement_controls_fail': bool(statement) and all(v['status'] == 'FAIL' for v in statement),
                       'non_statement_controls': sum(1 for v in out.values() if v.get('class') == 'non-statement')}
    return out


# --------------------------------------------------------------------------------------------
# Package check against the manifest
# --------------------------------------------------------------------------------------------

def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_package(root, selftest=False, seed=1, samples=64, admit_corroborated=False):
    """Exit 0 only when every registered target is PASS — or, with ``admit_corroborated`` (CLI
    ``--admit-corroborated``), PASS or CORROBORATED — and nothing is unregistered or fails the self-test.
    The registered G12 contract (GOVERNANCE.md) is the strict form; the admitting form is for packages whose
    statements carry sampled binders and is recorded in the report as ``admit_corroborated``."""
    root = pathlib.Path(root)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    modules = {}
    module_hashes = {}
    for rel in manifest.get('source_modules', []):
        p = root / rel
        module_hashes[rel] = _sha256(p)
        modules[rel] = parse_module(p.read_text(encoding='utf-8'))
    per_target = {}
    counts = {}
    selftests = {}
    normalized_targets = []
    for target in manifest.get('targets', []):
        if isinstance(target, str):
            rel = next((r for r, m in modules.items() if any(t.name == target for t in m.theorems)), None)
            normalized_targets.append({'name': target, 'module': rel})
        else:
            normalized_targets.append({'name': target['name'], 'module': target.get('module')})
    for target in normalized_targets:
        name, rel = target['name'], target['module']
        module = modules.get(rel)
        thm = None if module is None else next((t for t in module.theorems if t.name == name), None)
        if thm is None:
            res = EvalResult('UNSUPPORTED', f'target {name} not found in {rel}', 0, False, [], [])
        else:
            res = evaluate_theorem(module, thm, seed, samples)
        per_target[name] = {'module': rel, 'status': res.status, 'detail': res.detail, 'instances': res.instances,
                            'vacuous': res.vacuous, 'probes': res.probes,
                            'float_free': not any(isinstance(v, float) for v in res.values)}
        if res.sampling is not None:
            per_target[name]['sampling'] = res.sampling
        counts[res.status] = counts.get(res.status, 0) + 1
    unregistered = []
    registered = {t['name'] for t in normalized_targets}
    for rel, module in modules.items():
        for t in module.theorems:
            if t.name not in registered:
                unregistered.append(t.name)
    controls_report = {}
    if selftest:
        for rel, module in modules.items():
            selftests[rel] = mutation_selftest(module)
        nc = manifest.get('negative_controls') or {}
        if isinstance(nc, dict):
            for rel in modules:
                rep = declared_controls((root / rel).read_text(encoding='utf-8'), rel, nc)
                for k, v in rep.items():
                    if k != '_summary':
                        controls_report[k] = v
        statement_controls = [v for v in controls_report.values() if v.get('class') == 'statement']
        controls_ok = all(v['status'] == 'FAIL' for v in statement_controls)
    else:
        controls_ok = True
    allowed = set(HOLDS) if admit_corroborated else {'PASS'}
    exit_code = 0 if counts and set(counts) <= allowed and not unregistered and controls_ok and all(
        s['harness_ok'] for s in selftests.values()) else 1
    return {'package_root': str(root), 'manifest_sha256': _sha256(manifest_path), 'module_sha256': module_hashes,
            'targets': len(per_target), 'status_counts': counts, 'per_target': per_target,
            'unregistered_theorems': unregistered, 'manifest_targets': normalized_targets,
            'sampling': {'seed': seed, 'samples': samples, 'admit_corroborated': admit_corroborated},
            'selftest': selftests, 'declared_controls': controls_report, 'exit_code': exit_code,
            'note': ('G12 empirical statement check: PASS means the closed statement, read independently of Lean, '
                     'evaluates true in exact arithmetic; CORROBORATED means no sampled instance of a statement with '
                     'binders refuted it (admitted evidence, never a proof); neither is kernel evidence. FAIL names an '
                     'exact witness. UNSUPPORTED is fail-closed.')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--package', required=True, help='formal package root containing manifest.json')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--selftest', action='store_true', help='also run the mutation self-test per module')
    ap.add_argument('--seed', type=int, default=1, help='seed of the deterministic sampler for binders before the colon')
    ap.add_argument('--samples', type=int, default=64, help='hypothesis-satisfying instances to test per sampled statement')
    ap.add_argument('--admit-corroborated', action='store_true',
                    help='exit 0 when every target is PASS or CORROBORATED (default: PASS only, the registered contract)')
    a = ap.parse_args(argv)
    report = check_package(a.package, selftest=a.selftest, seed=a.seed, samples=a.samples,
                           admit_corroborated=a.admit_corroborated)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, default=str) + '\n')
    summary = {k: report[k] for k in ('package_root', 'targets', 'status_counts', 'unregistered_theorems', 'sampling', 'exit_code')}
    if a.selftest:
        summary['selftest'] = {rel: {k: v for k, v in s.items() if k not in ('survivors', 'weakenings_rejected')}
                               | {'survivors': len(s['survivors']), 'weakenings_rejected': len(s['weakenings_rejected'])}
                               for rel, s in report['selftest'].items()}
        dc = report['declared_controls']
        summary['declared_controls'] = {
            'statement_controls_fail': [k for k, v in dc.items() if v.get('class') == 'statement' and v['status'] == 'FAIL'],
            'statement_controls_not_failing': [k for k, v in dc.items() if v.get('class') == 'statement' and v['status'] != 'FAIL'],
            'non_statement_controls': [k for k, v in dc.items() if v.get('class') == 'non-statement']}
    print(json.dumps(summary, indent=1))
    for name, r in report['per_target'].items():
        if r['status'] != 'PASS':
            print(f"{r['status']}: {name}: {r['detail']}")
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
