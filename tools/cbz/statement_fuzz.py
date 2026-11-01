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

Measure-theoretic statements — binders `(μ : Measure Ω)`, `(W : Ω → ℝ)`, `(A : Set Ω)` over a carrier
declared by `variable {Ω : Type*}`, instance hypotheses `[IsProbabilityMeasure μ]`, integrals `∫ x, f x ∂μ`
and `∫ x in A, …`, event masses `(μ A).toReal`, a.e. relations `0 ≤ᵐ[μ] W`, set-builders `{x | p x}`,
`⊆`, `≪`, `μ.withDensity`, `c • μ`, `ENNReal.ofReal`/`⁻¹`/`.toReal` — are sampled on *finite discrete
models*: Ω = Fin n (n ≤ 4) with the discrete σ-algebra, non-negative rational weights (normalised when the
measure is hypothesised to be a probability measure), functions as tables, sets as subsets. Every such
model is an instance of the Lean statement: integrals are finite sums, measurability and integrability are
automatic, a.e. relations hold off the null points, measure equality and `≪` are pointwise. A conclusion
that is only a structural predicate (`MeasurableSet …`) is UNSUPPORTED, since it is trivial there; a
real-indexed family `(μ : ℝ → Measure Ω)` is UNSUPPORTED. The mutation contract applies unchanged.

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
_SINGLE = set('+-*/%^=<>()[],:|.{}!∀∃∧∨¬→↔≤≥≠∈∉λ√∫∂ᵐ•⊆≪∩∪ᶜ∅')
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

@dataclasses.dataclass
class Integral:
    var: str
    domain: object | None   # `∫ x in A, body ∂μ` restricts to the set A
    body: object
    measure: object

@dataclasses.dataclass
class SetBuilder:
    var: str
    body: object            # `{x | body}` on the model's points

@dataclasses.dataclass
class AERel:
    op: str                 # `f ≤ᵐ[μ] g`: the relation holds at every point of nonzero mass
    left: object
    right: object
    measure: object


_NORMALIZE = {'<=': '≤', '>=': '≥', '!=': '≠', '->': '→', '/\\': '∧', '\\/': '∨', '<->': '↔'}
_PREC = {'→': 25, '↔': 20, '∨': 30, '∧': 35, '||': 30, '&&': 35, '=': 50, '≠': 50, '<': 50, '>': 50, '≤': 50, '≥': 50,
         '∈': 50, '∉': 50, '⊆': 50, '≪': 50, '+': 65, '-': 65, '∪': 65, '*': 70, '/': 70, '%': 70, '∩': 70, '•': 73, '^': 75}
_RIGHT = {'→', '↔', '∨', '∧', '||', '&&', '^', '•'}
_COMPARISONS = {'=', '≠', '<', '>', '≤', '≥', '∈', '∉', '⊆', '≪'}
_AE_OPS = {'=', '≠', '<', '>', '≤', '≥'}
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
            if op in _AE_OPS and self.at('ᵐ'):
                # Mathlib `f ≤ᵐ[μ] g` (`Filter.EventuallyLE` along `ae μ`)
                self.eat('ᵐ')
                self.eat('[')
                measure = self.parse_expr(0)
                self.eat(']')
                right = self.parse_expr(prec + 1)
                left = AERel(op, left, right, measure)
                continue
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
        if t.value == '∫':
            return self.parse_integral()
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
            parts = []
            while not self.at(','):
                parts.append(self.eat().value)
            typ = ' '.join(parts)
        elif self.at('∈'):
            self.eat('∈')
            domain = self.parse_expr(51)
        self.eat(',')
        body = self.parse_expr(0)
        return Forall(var, typ, domain, body)

    def parse_integral(self):
        """`∫ x, f x ∂μ` and `∫ x in A, f x ∂μ` (the Bochner integral and the set integral); the binder may be
        typed, `∫ (x : Ω), …`. The measure after `∂` is one application (`μ`, `(μ r)`, `μ.restrict A`)."""
        self.eat('∫')
        if self.at('('):
            self.eat('(')
            var = self.eat().value
            if self.at(':'):
                self.eat(':')
                while not self.at(')'):
                    self.eat()
            self.eat(')')
        else:
            var = self.eat().value
        domain = None
        if self.at('in'):
            self.eat('in')
            domain = self.parse_expr(51)
        self.eat(',')
        body = self.parse_expr(0)
        self.eat('∂')
        measure = self.parse_application()
        return Integral(var, domain, body, measure)

    def parse_lambda(self):
        self.eat()
        params = []
        while not self.at('=>'):
            t = self.eat()
            if t.value == '(':
                params.append(self.eat().value)
                if self.at(':'):
                    self.eat(':')
                    while not self.at(')'):
                        self.eat()
                self.eat(')')
            else:
                params.append(t.value)
                if self.at(':'):
                    self.eat(':')
                    while not self.at('=>'):
                        self.eat()
        self.eat('=>')
        return Lambda(params, self.parse_expr(0))

    def parse_application(self):
        head = self.parse_atom()
        args = []
        while True:
            t = self.peek()
            if t is None:
                break
            if t.kind == 'ident' and t.value in ('if', 'then', 'else', 'fun', 'λ', 'in'):
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
            if t is not None and t.kind == 'op' and t.value == 'ᶜ':
                self.i += 1
                atom = App(Var('.compl'), [atom])
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
        if t.value == '∅':
            return Var('∅')
        if t.value == '{':
            # set-builder `{x | p x}` / `{x : Ω | p x}` over the model's points; other brace terms are outside
            nxt, nxt2 = self.peek(), self.peek(1)
            if nxt is not None and nxt.kind == 'ident' and nxt2 is not None and nxt2.value in ('|', ':'):
                var = self.eat().value
                if self.at(':'):
                    self.eat(':')
                    while not self.at('|'):
                        self.eat()
                self.eat('|')
                self.abs_depth += 1
                body = self.parse_expr(0)
                self.abs_depth -= 1
                self.eat('}')
                return SetBuilder(var, body)
            raise Unsupported('brace term that is not a set-builder')
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
    point_types: set = dataclasses.field(default_factory=set)   # `variable {Ω : Type*}`: finite-model carriers


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
    point_types = set()
    for start, end in _split_declarations(toks):
        kw = toks[start].value
        k = start
        while k < end and toks[k].value in ('noncomputable', 'private', 'protected'):
            k += 1
        kw = toks[k].value if k < end else kw
        if kw == 'namespace':
            namespace = toks[k + 1].value
            continue
        if kw == 'variable':
            # `variable {Ω : Type*} [MeasurableSpace Ω]`: a type variable becomes a finite-model carrier
            j = k + 1
            while j < end and toks[j].value in ('(', '{', '['):
                close = _matching(toks, j, end)
                if close == -1:
                    break
                colon = _find(toks, ':', j + 1, close)
                if toks[j].value in ('(', '{') and colon != -1 and toks[colon + 1].value == 'Type':
                    point_types.update(x.value for x in toks[j + 1:colon] if x.kind == 'ident')
                j = close + 1
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
            binders, binder_error = _parse_theorem_binders(toks, k + 2, colon, defs, point_types)
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
    return Module(namespace, defs, theorems, point_types)


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

# Predicates on finite discrete models. Structural ones hold automatically there (discrete σ-algebra, finite
# measure, finite sums): as hypotheses they filter nothing; as a whole conclusion they are reported UNSUPPORTED.
_STRUCTURAL_PREDICATES = {'Measurable', 'MeasurableSet', 'Integrable', 'MemLp', 'AEMeasurable', 'AEStronglyMeasurable',
                          'StronglyMeasurable', 'Memℒp'}
_MEASURE_PREDICATES = {'IsProbabilityMeasure', 'IsFiniteMeasure'} | _STRUCTURAL_PREDICATES   # the first two are decided
_IGNORED_INSTANCES = {'MeasurableSpace', 'MeasurableSingletonClass', 'TopologicalSpace', 'BorelSpace', 'Nonempty',
                      'DecidableEq', 'Fintype', 'Countable'}

_BUILTIN_NAMES = {'true', 'false', 'True', 'False', 'abs', 'min', 'max', 'Real.sqrt', 'Inv.inv', 'List.range',
                  'List.foldl', 'List.filter', 'List.sum', 'List.length', 'List.map', 'Nat.choose', 'Nat.factorial',
                  'Nat.succ', 'Nat.pred', 'Nat.min', 'Nat.max', 'Nat.add', 'Nat.mul', 'Nat.sub', 'Int.natAbs',
                  'Int.toNat', 'Int.ofNat', 'Set.Icc', 'Set.Ico', 'Set.Ioc', 'Set.Ioo', 'Set.Ici', 'Set.Iic',
                  'Set.Ioi', 'Set.Iio', 'Set.univ', '∅', 'ENNReal.ofReal', 'ENNReal.toReal'} | _MEASURE_PREDICATES

_MODEL_KINDS = ('Measure', 'Func', 'Set')     # binder kinds sampled as finite discrete models


def _kind_of_type(t: str, point_types) -> str | None:
    """The interpreter's kind for a Lean type written as a string: a numeric type ('Nat', 'Int', 'Rat', 'Real'),
    'ENNReal', 'Bool', 'Point' (a finite-model carrier Ω), 'Measure' (`Measure Ω`), 'Set' (`Set Ω`), 'Func'
    (`Ω → ℝ`, a table on the carrier), or None when the type is outside the interpreter (`ℝ → Measure Ω`,
    `Fin n → ℝ`, `Ω → Ω`, structures)."""
    t = t.strip()
    parts = [x.strip() for x in t.split('→')]
    if len(parts) == 2 and parts[0] in point_types:
        return 'Func' if parts[1] in _SAMPLED_TYPE_TOKENS or parts[1] in ('ENNReal', 'ℝ≥0∞') else None
    if len(parts) >= 2:
        return None
    if t in _SAMPLED_TYPE_TOKENS:
        return _SAMPLED_TYPE_TOKENS[t]
    if t in ('ENNReal', 'ℝ≥0∞'):
        return 'ENNReal'
    if t == 'Bool':
        return 'Bool'
    if t in point_types:
        return 'Point'
    words = t.split()
    if len(words) == 2 and words[0] == 'Measure' and words[1] in point_types:
        return 'Measure'
    if len(words) == 2 and words[0] == 'Set' and words[1] in point_types:
        return 'Set'
    return None


def _parse_theorem_binders(toks, start, end, defs, point_types=frozenset()):
    """Binder groups between the theorem name and its colon → ([Binder], error or None).

    `(x y : ℝ)` / `{x : ℕ}` with a sampled numeric type declare variables; `(h : p)` whose type parses as a
    proposition over names declared so far (variables, the module's definitions, the interpreter's
    builtins) is a hypothesis; anything else — an instance binder `[Fact p]`, a function or structure type
    `(f : Ω → ℝ)`, `(μ : Measure Ω)`, a predicate the interpreter does not define — makes the theorem
    UNSUPPORTED, fail-closed, naming the binder."""
    binders, declared = [], set()
    point_types = set(point_types)
    k = start
    while k < end:
        t = toks[k]
        if t.value == '[':
            close = _matching(toks, k, end)
            txt = ' '.join(x.value for x in toks[k:close + 1 if close != -1 else end])
            if close == -1:
                return binders, 'unbalanced binder brackets'
            # `[IsProbabilityMeasure μ]` is a hypothesis on a declared measure; `[MeasurableSpace Ω]` is structure
            inner = toks[k + 1:close]
            if inner and inner[0].kind == 'ident' and inner[0].value in _IGNORED_INSTANCES:
                k = close + 1
                continue
            if inner and inner[0].kind == 'ident' and inner[0].value in _MEASURE_PREDICATES:
                try:
                    parser = Parser(toks, k + 1, close)
                    node = parser.parse_expr(0)
                    ok = parser.i == close and isinstance(node, App) and all(
                        isinstance(a, Var) and a.name in declared for a in node.args)
                except (SyntaxError, Unsupported):
                    ok = False
                if ok:
                    binders.append(Binder('hyp', '_inst', node, txt))
                    k = close + 1
                    continue
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
        if type_toks and type_toks[0].value == 'Type':
            point_types.update(names)          # `{Ω : Type*}` declared on the theorem itself
            k = close + 1
            continue
        kind = _kind_of_type(' '.join(x.value for x in type_toks), point_types)
        if kind in _SAMPLED_TYPE_TOKENS.values() or kind in _MODEL_KINDS:
            for nm in names:
                binders.append(Binder('var', nm, kind, txt))
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
            known = declared | set(defs) | _BUILTIN_NAMES | _TRANSCENDENTAL | point_types
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
    """A syntactic proposition: comparisons, membership, a.e. relations and the model predicates, closed
    under ¬ ∧ ∨ → ↔ and ∀."""
    if isinstance(node, BinOp):
        if node.op in _COMPARISONS:
            return True
        if node.op in _LOGIC:
            return _is_prop_shape(node.left) and _is_prop_shape(node.right)
        return False
    if isinstance(node, AERel):
        return True
    if isinstance(node, App):
        return isinstance(node.fn, Var) and node.fn.name in _MEASURE_PREDICATES
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
    if isinstance(node, Integral):
        out = _free_vars(node.body, bound | {node.var}) | _free_vars(node.measure, bound)
        return out | (_free_vars(node.domain, bound) if node.domain is not None else set())
    if isinstance(node, SetBuilder):
        return _free_vars(node.body, bound | {node.var})
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
    if isinstance(node, Integral):
        return ([node.domain] if node.domain is not None else []) + [node.body, node.measure]
    if isinstance(node, SetBuilder):
        return [node.body]
    if isinstance(node, AERel):
        return [node.left, node.right, node.measure]
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
    if isinstance(node, Integral):
        return Integral(node.var, None if node.domain is None else _desugar(node.domain), _desugar(node.body),
                        _desugar(node.measure))
    if isinstance(node, SetBuilder):
        return SetBuilder(node.var, _desugar(node.body))
    if isinstance(node, AERel):
        return AERel(node.op, _desugar(node.left), _desugar(node.right), _desugar(node.measure))
    return node


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------

_LIST_METHODS = {'foldl', 'filter', 'map', 'sum', 'length', 'withDensity', 'restrict', 'toReal', 'compl'}
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


# --------------------------------------------------------------------------------------------
# Finite discrete models: ENNReal values, measures as weight vectors, functions as tables, sets as subsets
# --------------------------------------------------------------------------------------------
# Ω = Fin n with the discrete σ-algebra. Every set is measurable, every function measurable, every function
# integrable against a finite measure, so the structural hypotheses of a Mathlib statement hold and the
# numeric content is a finite sum. An instance refuted here is refuted in Lean's reading of the statement.

@dataclasses.dataclass(frozen=True)
class ENN:
    """An extended non-negative real: a non-negative Fraction, or ∞ (`value is None`). Mathlib's conventions:
    `0⁻¹ = ∞`, `∞⁻¹ = 0`, `∞ * 0 = 0`, truncated subtraction, `ENNReal.toReal ∞ = 0`."""
    value: object = Fraction(0)

    @property
    def inf(self):
        return self.value is None

    def __repr__(self):
        return '∞' if self.inf else _fmt(self.value)

    def __add__(self, other):
        other = _enn(other)
        return ENN(None) if self.inf or other.inf else ENN(self.value + other.value)

    __radd__ = __add__

    def __mul__(self, other):
        other = _enn(other)
        if (self.inf and other.value == 0) or (other.inf and self.value == 0):
            return ENN(Fraction(0))
        return ENN(None) if self.inf or other.inf else ENN(self.value * other.value)

    __rmul__ = __mul__

    def __sub__(self, other):
        other = _enn(other)
        if other.inf:
            return ENN(Fraction(0))
        if self.inf:
            return ENN(None)
        return ENN(max(self.value - other.value, Fraction(0)))

    def inverse(self):
        if self.inf:
            return ENN(Fraction(0))
        return ENN(None) if self.value == 0 else ENN(1 / self.value)

    def __truediv__(self, other):
        return self * _enn(other).inverse()

    def to_real(self):
        return Fraction(0) if self.inf else self.value

    def _key(self):
        return (1, Fraction(0)) if self.inf else (0, self.value)

    def __lt__(self, other):
        return self._key() < _enn(other)._key()

    def __le__(self, other):
        return self._key() <= _enn(other)._key()

    def __gt__(self, other):
        return self._key() > _enn(other)._key()

    def __ge__(self, other):
        return self._key() >= _enn(other)._key()


def _enn(x):
    if isinstance(x, ENN):
        return x
    if isinstance(x, bool) or isinstance(x, (list, tuple, frozenset)):
        raise Unsupported('ENNReal arithmetic on a non-numeric value')
    x = _exact(x)
    if isinstance(x, Alg):
        if x.sign() < 0:
            raise Unsupported('a negative value where an ENNReal is expected')
        raise Unsupported('an irrational ENNReal value is outside the finite-model backend')
    if x < 0:
        raise Unsupported('a negative value where an ENNReal is expected')
    return ENN(Fraction(x))


@dataclasses.dataclass(frozen=True)
class Meas:
    """A measure on Fin n: one ENNReal weight per point."""
    weights: tuple

    def mass(self, points=None):
        total = ENN(Fraction(0))
        for i, w in enumerate(self.weights):
            if points is None or i in points:
                total = total + w
        return total

    def __repr__(self):
        return '[' + ', '.join(repr(w) for w in self.weights) + ']'


@dataclasses.dataclass(frozen=True)
class Func:
    """A function on Fin n as a table; `codomain` is the interpreter type of its values."""
    table: tuple
    codomain: str = 'Real'

    def __repr__(self):
        return '[' + ', '.join(_fmt(v) if not isinstance(v, ENN) else repr(v) for v in self.table) + ']'


def _fmt_set(points, n=None) -> str:
    return '{' + ', '.join(str(i) for i in sorted(points)) + '}'


def _fmt_value(v) -> str:
    if isinstance(v, frozenset):
        return _fmt_set(v)
    if isinstance(v, (Meas, Func, ENN)):
        return repr(v)
    return _fmt(v)


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
        self.model_n = None          # the carrier size of the current finite model, when one is in force

    def points(self):
        if self.model_n is None:
            raise Unsupported('a measure, set or function value without a finite model in force')
        return range(self.model_n)

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
            if node.name in ('Set.univ', '∅'):
                return 'Set'
            d = self.module.defs.get(node.name)
            if d is not None and d.result_type:
                return self._type_name(d.result_type)
            return 'Nat'
        if isinstance(node, (Integral,)):
            return 'Real'
        if isinstance(node, SetBuilder):
            return 'Set'
        if isinstance(node, AERel):
            return 'Prop'
        if isinstance(node, (BinOp,)):
            if node.op in _COMPARISONS or node.op in _LOGIC:
                return 'Prop'
            if node.op == '^':
                return self.arith_type(node.left, env)      # the exponent is typed on its own (Lean `rightact%`)
            if node.op == '•':
                return self.arith_type(node.right, env)
            if node.op in ('∩', '∪'):
                return 'Set'
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
                if fn.name in env and isinstance(env[fn.name], Bound):
                    b = env[fn.name]
                    if b.type == 'Measure':
                        return 'ENNReal'                      # `μ A`
                    if b.type == 'Func':
                        return b.value.codomain if isinstance(b.value, Func) else 'Real'
                d = self.module.defs.get(fn.name)
                if d is not None and d.result_type:
                    rt = self._type_name(d.result_type)
                    if rt == 'Measure' and len(node.args) == len(d.params) + 1:
                        return 'ENNReal'                      # a defined measure applied to a set
                    return rt
                if fn.name == 'Real.sqrt':
                    return 'Real'
                if fn.name in ('.toReal', 'ENNReal.toReal'):
                    return 'Real'
                if fn.name in ('ENNReal.ofReal',):
                    return 'ENNReal'
                if fn.name in ('.withDensity', '.restrict'):
                    return 'Measure'
                if fn.name in ('.compl',):
                    return 'Set'
                if fn.name in _MEASURE_PREDICATES:
                    return 'Prop'
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

    def _type_name(self, t: str) -> str:
        kind = _kind_of_type(t, self.module.point_types)
        if kind is not None:
            return kind
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
        kind = _kind_of_type(last, self.module.point_types)
        if kind is not None:
            return kind
        raise Unsupported(f'result type {t}')

    # ---- numeric evaluation at a given type
    def num(self, node, env, typ):
        if isinstance(node, Num):
            if typ in _FIELD_TYPES:
                return self._rec(Fraction(node.value))
            if typ == 'ENNReal':
                return ENN(Fraction(node.value))
            if typ == 'Measure':
                if node.value == 0:
                    return Meas(tuple(ENN(Fraction(0)) for _ in self.points()))
                raise Unsupported('a nonzero numeral where a measure is expected')
            if not isinstance(node.value, int):
                raise Unsupported('scientific literal outside a rational or real context (Lean defaults it to Float)')
            return self._rec(node.value)
        if isinstance(node, Integral):
            return self._rec(self.integral(node, env))
        if isinstance(node, SetBuilder):
            return frozenset(i for i in self.points() if self.prop(node.body, {**env, node.var: Bound(i, 'Point')}))
        if isinstance(node, Lambda) and typ == 'Func':
            return self.materialize(Closure(node.params, node.body, env))
        if isinstance(node, Ascribe):
            # `(e : T)` elaborates e at T (Lean pushes the expected type to the leaves: `(n - 1 : ℝ)` is real
            # subtraction of ↑n, `((n - 1 : ℕ) : ℝ)` casts a truncated Nat); the value is then used at typ
            inner = self._type_name(node.type)
            return self._rec(_coerce(self.num(node.expr, env, inner), inner, typ))
        if isinstance(node, Var):
            if node.name in ('true', 'false'):
                return node.name == 'true'
            if node.name == 'Set.univ':
                return frozenset(self.points())
            if node.name == '∅':
                return frozenset()
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
            if node.op == '•':
                c = self.num(node.left, env, 'ENNReal')
                m = self.num(node.right, env, 'Measure')
                if not isinstance(m, Meas):
                    raise Unsupported('scalar action on a non-measure')
                return Meas(tuple(_enn(c) * w for w in m.weights))
            if node.op in ('∩', '∪'):
                a, b = self.num(node.left, env, 'Set'), self.num(node.right, env, 'Set')
                if not isinstance(a, frozenset) or not isinstance(b, frozenset):
                    raise Unsupported('set operation on non-sets')
                return a & b if node.op == '∩' else a | b
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
        if typ == 'ENNReal' or isinstance(a, ENN) or isinstance(b, ENN):
            a, b = _enn(a), _enn(b)
            if op == '+':
                return a + b
            if op == '-':
                return a - b
            if op == '*':
                return a * b
            if op == '/':
                return a / b
            raise Unsupported(f'operator {op} on ENNReal')
        if isinstance(a, (Meas, Func, frozenset)) or isinstance(b, (Meas, Func, frozenset)):
            raise Unsupported(f'operator {op} on a measure, function or set')
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
        if name in env and isinstance(env[name], Bound) and env[name].type == 'Measure' and len(args) == 1:
            return self.measure_apply(env[name].value, args[0], env)
        if name in env and isinstance(env[name], Bound) and env[name].type == 'Func' and len(args) == 1:
            return self.func_apply(env[name].value, args[0], env)
        if name in _TRANSCENDENTAL:
            raise Unsupported(f'transcendental {name} is outside the exact algebraic backend')
        if name in ('.toReal', 'ENNReal.toReal') and len(args) == 1:
            v = self.num(args[0], env, 'ENNReal')
            return _enn(v).to_real()
        if name == 'ENNReal.ofReal' and len(args) == 1:
            v = self.num(args[0], env, 'Real')
            if isinstance(v, Alg):
                if v.sign() <= 0:
                    return ENN(Fraction(0))
                raise Unsupported('an irrational ENNReal value is outside the finite-model backend')
            return ENN(max(_exact(v), Fraction(0)))
        if name == '.withDensity' and len(args) == 2:
            m = self.num(args[0], env, 'Measure')
            f = self.num(args[1], env, 'Func')
            if not isinstance(m, Meas) or not isinstance(f, Func):
                raise Unsupported('withDensity on a non-measure or non-function')
            return Meas(tuple(w * _enn(f.table[i]) for i, w in enumerate(m.weights)))
        if name == '.restrict' and len(args) == 2:
            m = self.num(args[0], env, 'Measure')
            a = self.num(args[1], env, 'Set')
            if not isinstance(m, Meas) or not isinstance(a, frozenset):
                raise Unsupported('restrict on a non-measure or non-set')
            return Meas(tuple(w if i in a else ENN(Fraction(0)) for i, w in enumerate(m.weights)))
        if name == '.compl' and len(args) == 1:
            a = self.num(args[0], env, 'Set')
            if not isinstance(a, frozenset):
                raise Unsupported('complement of a non-set')
            return frozenset(self.points()) - a
        if name in _MEASURE_PREDICATES:
            raise Unsupported(f'predicate {name} used as a value')
        if name == 'Real.sqrt' and len(args) == 1:
            return _real_sqrt(self.num(args[0], env, 'Real'))
        if name == 'abs' and len(args) == 1:
            t = _max_type(self.arith_type(args[0], env), typ) if typ != 'Nat' else self.arith_type(args[0], env)
            v = self.num(args[0], env, t)
            if isinstance(v, (list, tuple, bool)):
                raise Unsupported('abs of a non-numeric value')
            return _abs(v)
        if name == 'Inv.inv' and len(args) == 1:
            if typ == 'ENNReal':
                return _enn(self.num(args[0], env, 'ENNReal')).inverse()
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
        if dtyp == 'Measure' and not d.clauses and len(args) == len(d.params) + 1:
            measure = self.apply(App(fn, list(args[:-1])), env, 'Measure')
            return self.measure_apply(measure, args[-1], env)
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

    # ---- finite models
    def integral(self, node: Integral, env):
        """`∫ x, f x ∂μ` on a finite model: Σ w_i f(i) over the points of nonzero mass (restricted to the set
        for `∫ x in A, …`). A point of infinite mass makes the integral undefined here (Lean's Bochner integral
        of a non-integrable function is 0; the hypotheses of the statements in scope exclude it): UNSUPPORTED."""
        m = self.num(node.measure, env, 'Measure')
        if not isinstance(m, Meas):
            raise Unsupported('integral against a non-measure')
        domain = None
        if node.domain is not None:
            domain = self.num(node.domain, env, 'Set')
            if not isinstance(domain, frozenset):
                raise Unsupported('set integral over a non-set')
        total = Fraction(0)
        for i, w in enumerate(m.weights):
            if domain is not None and i not in domain:
                continue
            if w.inf:
                raise Unsupported('integral against a point of infinite mass')
            if w.value == 0:
                continue
            v = self.num(node.body, {**env, node.var: Bound(i, 'Point')}, 'Real')
            if isinstance(v, (bool, list, tuple, frozenset, Meas, Func, ENN)):
                raise Unsupported('integrand is not real-valued')
            total = total + w.value * _exact(v)
        return total

    def measure_apply(self, measure, set_node, env):
        if not isinstance(measure, Meas):
            raise Unsupported('application of a non-measure')
        a = self.num(set_node, env, 'Set')
        if not isinstance(a, frozenset):
            raise Unsupported('a measure applied to a non-set')
        return measure.mass(a)

    def func_apply(self, func, point_node, env):
        if not isinstance(func, Func):
            raise Unsupported('application of a non-function')
        i = self.num(point_node, env, 'Point')
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(func.table):
            raise Unsupported('a function applied to a non-point')
        return func.table[i]

    def materialize(self, closure: Closure) -> Func:
        """A lambda on the carrier as a table, typed by its body."""
        if len(closure.params) != 1:
            raise Unsupported('a function of several arguments on the carrier')
        table, codomain = [], 'Real'
        for i in self.points():
            scope = dict(closure.env)
            scope[closure.params[0]] = Bound(i, 'Point')
            codomain = self.arith_type(closure.body, scope)
            if codomain in ('Prop', 'Measure', 'Set', 'Func'):
                raise Unsupported('a function on the carrier with a non-numeric codomain')
            table.append(self.num(closure.body, scope, codomain if codomain != 'Nat' else 'Real'))
        return Func(tuple(table), codomain if codomain != 'Nat' else 'Real')

    def at_point(self, node, env, i, typ='Real'):
        """The value at point i of a function-or-constant expression (`0 ≤ᵐ[μ] W` compares 0 with W i)."""
        if isinstance(node, Var) and node.name in env and isinstance(env[node.name], Bound) \
                and env[node.name].type == 'Func':
            return env[node.name].value.table[i]
        if isinstance(node, Lambda):
            return self.func_apply(self.materialize(Closure(node.params, node.body, env)), Num(i), env)
        t = self.arith_type(node, env)
        if t == 'Func':
            f = self.num(node, env, 'Func')
            return f.table[i] if isinstance(f, Func) else f
        return self.num(node, env, typ if t in ('Nat', 'Int', 'Sci') else t)

    def predicate(self, node: App, env) -> bool:
        """Mathlib predicates on a finite discrete model. `IsProbabilityMeasure μ` is total mass 1; the
        structural ones (measurability, integrability, finiteness) hold for every value the model can carry."""
        name = node.fn.name
        if name == 'IsProbabilityMeasure':
            if len(node.args) != 1:
                raise Unsupported('IsProbabilityMeasure with an unexpected arity')
            m = self.num(node.args[0], env, 'Measure')
            if not isinstance(m, Meas):
                raise Unsupported('IsProbabilityMeasure of a non-measure')
            return m.mass() == ENN(Fraction(1))
        if name == 'IsFiniteMeasure':
            if len(node.args) != 1:
                raise Unsupported('IsFiniteMeasure with an unexpected arity')
            m = self.num(node.args[0], env, 'Measure')
            if not isinstance(m, Meas):
                raise Unsupported('IsFiniteMeasure of a non-measure')
            return not any(w.inf for w in m.weights)
        if name in _STRUCTURAL_PREDICATES:
            measures = []
            for a in node.args:            # the arguments must at least be values the model carries
                t = self.arith_type(a, env)
                if t not in ('Measure', 'Set', 'Func', 'Nat', 'Real', 'ENNReal') and not isinstance(a, Lambda):
                    raise Unsupported(f'{name} of a {t}')
                if t == 'Measure':
                    measures.append(self.num(a, env, 'Measure'))
            # a derived measure (`c • μ` with c = ∞) may carry a point of infinite mass: finiteness is then false
            # and integrability is not automatic, so nothing is assumed there (a8, 743/5729046258 N2)
            infinite = any(isinstance(m, Meas) and any(w.inf for w in m.weights) for m in measures)
            if infinite:
                raise Unsupported(f'{name} on a measure with a point of infinite mass is not decided by the finite-model backend')
            return True
        raise Unsupported(f'predicate {name}')

    def ae_rel(self, node: AERel, env) -> bool:
        m = self.num(node.measure, env, 'Measure')
        if not isinstance(m, Meas):
            raise Unsupported('a.e. relation along a non-measure')
        for i, w in enumerate(m.weights):
            if not w.inf and w.value == 0:
                continue
            a = _exact(self.at_point(node.left, env, i))
            b = _exact(self.at_point(node.right, env, i))
            if isinstance(a, (bool, frozenset, Meas, Func)) or isinstance(b, (bool, frozenset, Meas, Func)):
                raise Unsupported('a.e. relation between non-numeric values')
            if not {'=': a == b, '≠': a != b, '<': a < b, '>': a > b, '≤': a <= b, '≥': a >= b}[node.op]:
                return False
        return True

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
            kind = _kind_of_type(node.type, self.module.point_types)
            if kind == 'Set':
                # every subset of the finite carrier: 2^n instances
                pts = list(self.points())
                subsets = [frozenset(i for i in pts if mask >> i & 1) for mask in range(1 << len(pts))]
                return self._forall_over(node, env, subsets, 'Set')
            if kind == 'Point':
                return self._forall_over(node, env, list(self.points()), 'Point')
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

    def _forall_over(self, node: 'Forall', env, domain, var_type) -> bool:
        self.instances += len(domain)
        ok = True
        for v in domain:
            scope = dict(env)
            scope[node.var] = Bound(v, var_type)
            if not self.prop(node.body, scope):
                self.failures.append(f'{node.var} = {_fmt_value(v)}')
                ok = False
        return ok

    # ---- propositions
    def prop(self, node, env) -> bool:
        if isinstance(node, Not):
            return not self.prop(node.expr, env)
        if isinstance(node, Forall):
            return self.forall(node, env)
        if isinstance(node, AERel):
            return self.ae_rel(node, env)
        if isinstance(node, App) and isinstance(node.fn, Var) and node.fn.name in _MEASURE_PREDICATES:
            return self.predicate(node, env)
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
                rt = self.arith_type(node.right, env)
                if rt == 'Set':
                    a = self.num(node.right, env, 'Set')
                    if not isinstance(a, frozenset):
                        raise Unsupported('membership in a non-set')
                    i = self.num(node.left, env, 'Point')
                    return (i in a) if node.op == '∈' else (i not in a)
                lst = self.num(node.right, env, 'Nat')
                if not isinstance(lst, list):
                    raise Unsupported('membership in a non-list')
                v = self.num(node.left, env, 'Nat')
                return (v in lst) if node.op == '∈' else (v not in lst)
            if node.op == '⊆':
                a, b = self.num(node.left, env, 'Set'), self.num(node.right, env, 'Set')
                if not isinstance(a, frozenset) or not isinstance(b, frozenset):
                    raise Unsupported('⊆ between non-sets')
                return a <= b
            if node.op == '≪':
                nu, mu = self.num(node.left, env, 'Measure'), self.num(node.right, env, 'Measure')
                if not isinstance(nu, Meas) or not isinstance(mu, Meas):
                    raise Unsupported('≪ between non-measures')
                # absolute continuity on a finite carrier: every μ-null point is ν-null
                return all(w_nu == ENN(Fraction(0)) for w_nu, w_mu in zip(nu.weights, mu.weights)
                           if w_mu == ENN(Fraction(0)))
            if node.op in _COMPARISONS:
                typ = _max_type(self.arith_type(node.left, env), self.arith_type(node.right, env))
                if typ == 'Bool':
                    typ = 'Nat'
                if typ == 'Sci':
                    raise Unsupported('scientific literal outside a rational or real context (Lean defaults it to Float)')
                if typ in ('Measure', 'Set', 'Func'):
                    a = self.num(node.left, env, typ)
                    b = self.num(node.right, env, typ)
                    if node.op == '=':
                        return a == b
                    if node.op == '≠':
                        return a != b
                    raise Unsupported(f'ordering of {typ} values')
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
    order = {'Sci': -2, 'Bool': -1, 'Nat': 0, 'Int': 1, 'Rat': 2, 'Real': 3, 'ENNReal': 4, 'Point': 5, 'Set': 6,
             'Func': 7, 'Measure': 8, 'Prop': 9}
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

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
_MR_DETERMINISTIC_BELOW = 3_317_044_064_679_887_385_961_981       # Miller–Rabin with these bases is exact below this


def _is_prime(n: int) -> bool:
    """Deterministic Miller–Rabin for n < 3.3·10²⁴ (Sorenson–Webster bases); larger n is infeasible here."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41):
        if n % p == 0:
            return n == p
    if n >= _MR_DETERMINISTIC_BELOW:
        raise Infeasible(f'primality of {n} is beyond the deterministic Miller–Rabin range')
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_brent(n: int) -> int:
    """A nontrivial factor of a composite n (Brent's variant of Pollard's rho, deterministic parameters)."""
    if n % 2 == 0:
        return 2
    for c in range(1, 1000):
        y, m, g, r, q = 2, 128, 1, 1, 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = _gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = _gcd(abs(x - ys), n)
        if g != n:
            return g
    raise Infeasible(f'no factor of {n} found')       # pragma: no cover - rho with 999 polynomials


def _prime_factors(n: int) -> dict:
    """Exact prime factorization with multiplicities (trial division, then Miller–Rabin + Brent rho)."""
    global _SMALL_PRIMES
    if _SMALL_PRIMES is None:
        _SMALL_PRIMES = _primes_up_to(100_000)
    out = {}
    for p in _SMALL_PRIMES:
        if p * p > n:
            break
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
    stack = [n] if n > 1 else []
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_prime(m):
            out[m] = out.get(m, 0) + 1
            continue
        r = _isqrt_exact(m)
        if r is not None:
            stack.extend([r, r])
            continue
        f = _pollard_brent(m)
        stack.extend([f, m // f])
    return out


def _small_square_split(n: int):
    """n = s² · m with every prime below 100000 appearing in m at most once; m need not be squarefree."""
    global _SMALL_PRIMES
    if _SMALL_PRIMES is None:
        _SMALL_PRIMES = _primes_up_to(100_000)
    s, m = 1, n
    for p in _SMALL_PRIMES:
        if p * p > m:
            break
        while m % (p * p) == 0:
            m //= p * p
            s *= p
    return s, m


def _squarefree_split(n: int):
    """n = s² · d with d squarefree, for n ≥ 1, exactly, from the prime factorization."""
    if n in _SQUAREFREE_CACHE:
        return _SQUAREFREE_CACHE[n]
    s, d = 1, 1
    for p, e in _prime_factors(n).items():
        s *= p ** (e // 2)
        if e % 2:
            d *= p
    _SQUAREFREE_CACHE[n] = (s, d)
    return s, d


class Alg:
    """Σ c_d √d over squarefree d ≥ 1 (d = 1 is the rational part), coefficients nonzero Fractions. Always
    irrational: results that collapse to a rational are returned as a Fraction by ``_alg``.

    ``canonical`` is False when some radicand could not be split into square and squarefree parts exactly
    (its cofactor is beyond the deterministic primality range): the value is still exact and every
    ordering is still decided by enclosures, but equality by form is unavailable and an exact zero cannot
    be recognised — those cases are reported Infeasible, never guessed."""
    __slots__ = ('terms', 'canonical')

    def __init__(self, terms, canonical=True):
        self.terms = terms
        self.canonical = canonical

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
        return _alg(out, self.canonical and _canonical(other))

    __radd__ = __add__

    def __neg__(self):
        return Alg({d: -c for d, c in self.terms.items()}, self.canonical)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Alg) else -Fraction(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        out = {}
        for d1, c1 in self.terms.items():
            for d2, c2 in Alg._terms(other).items():
                g = _gcd(d1, d2)
                d = (d1 // g) * (d2 // g)          # √d₁√d₂ = g·√(d₁d₂/g²) holds for any d₁, d₂
                out[d] = out.get(d, Fraction(0)) + c1 * c2 * g
        return _alg(out, self.canonical and _canonical(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Alg):
            return self * other.inverse()
        other = Fraction(other)
        if other == 0:
            return Fraction(0)                         # Lean: x / 0 = 0
        return Alg({d: c / other for d, c in self.terms.items()}, self.canonical)

    def __rtruediv__(self, other):
        return Fraction(other) * self.inverse()

    def __pow__(self, n):
        return _pow(self, n)

    def inverse(self):
        """x⁻¹ by conjugation over a coprime base of the radicands: x = A + B√b with A, B free of √b, so
        x · (A − B√b) = A² − b·B² lies in the smaller field (and is nonzero when x is), and recursion ends at
        a rational."""
        if not self.canonical:
            raise Infeasible('inverse of a value with an unsplit radicand')
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
        raise Infeasible('sign of an algebraic value undecided at 4096 bits'
                         + ('' if self.canonical else ' (an unsplit radicand: the value may be exactly zero)'))

    def _cmp(self, other):
        diff = self - other
        if isinstance(diff, Alg):
            return diff.sign()
        return (diff > 0) - (diff < 0)

    def __eq__(self, other):
        if self.canonical and _canonical(other):
            return isinstance(other, Alg) and self.terms == other.terms
        diff = self - other                   # no canonical form: equal iff the difference is exactly zero
        if not isinstance(diff, Alg):
            return diff == 0
        return diff.sign() == 0              # never 0: a decided sign means unequal, otherwise Infeasible

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


def _alg(terms: dict, canonical: bool = True):
    terms = {d: c for d, c in terms.items() if c != 0}
    if not terms:
        return Fraction(0)
    if set(terms) == {1}:
        return terms[1]
    return Alg(terms, canonical)


def _canonical(x) -> bool:
    return x.canonical if isinstance(x, Alg) else True


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
    # √(a/b) = √a / √b = (sₐ√dₐ)/(s_b√d_b) with dₐ·d_b = g²·d, g = gcd(dₐ, d_b): the numerator and the
    # denominator are split separately (each is far smaller than their product)
    ra, rb = _isqrt_exact(v.numerator), _isqrt_exact(v.denominator)
    if ra is not None and rb is not None:
        return Fraction(ra, rb)
    canonical = True
    try:
        sa, da = _squarefree_split(v.numerator)
        sb, db = _squarefree_split(v.denominator)
    except Infeasible:
        # beyond exact factoring: remove the small square factors only; the radicand may keep a square part
        canonical = False
        sa, da = _small_square_split(v.numerator)
        sb, db = _small_square_split(v.denominator)
    g = _gcd(da, db)
    d = (da // g) * (db // g)
    coeff = Fraction(sa * g, sb * db)
    return coeff if d == 1 else Alg({d: coeff}, canonical)


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


# Finite-model sampling: carrier sizes, weights, tables and subsets. n = 1 is the degenerate model every
# statement must survive; n ≤ 4 keeps 2^n subset enumeration and witness printing small.
_MODEL_SIZES = [1, 2, 2, 3, 1, 3, 4, 2]
_WEIGHT_POOL = [_F(1), _F(0), _F(1, 2), _F(1, 3), _F(2), _F(1, 4), _F(3, 4), _F(1, 10 ** 6), _F(10 ** 6), _F(1, 8), _F(5), _F(3)]
_TABLE_POOL = [_F(0), _F(1), _F(-1), _F(1, 2), _F(2), _F(-2), _F(10), _F(1, 10), _F(3), _F(-1, 2), _F(10 ** 6),
               _F(1, 10 ** 6), _F(999999, 1000000), _F(4), _F(9), _F(1, 4), _F(22, 7), _F(-3)]


def _structured_model_value(kind, i, n, probability):
    """The i-th structured model value of a kind on a carrier of size n (deterministic, no RNG)."""
    if kind == 'Measure':
        if probability:
            return Meas(tuple(ENN(Fraction(1, n)) for _ in range(n)))
        w = [_F(1), _F(1, 2), _F(2), _F(0), _F(1), _F(1, 3), _F(10 ** 6), _F(1)][i % 8]
        return Meas(tuple(ENN(w) for _ in range(n)))
    if kind == 'Func':
        c = [_F(1), _F(0), _F(-1), _F(1, 2), _F(2), _F(-2), _F(10), _F(1, 10)][i % 8]
        return Func(tuple(c + (k if i % 3 == 2 else 0) for k in range(n)))
    if kind == 'Set':
        return [frozenset(range(n)), frozenset(), frozenset(range(n)), frozenset([0]), frozenset(range(n)),
                frozenset(range(1, n)), frozenset([0]), frozenset(range(n))][i % 8]
    raise Unsupported(f'model kind {kind}')


def _neighbour_model_value(base, rng: random.Random):
    """A value next to an already drawn one of the same kind: the same table with one entry moved by a
    pathological amount (or unchanged), a subset with one point flipped, a measure with one weight moved.
    Perturbation statements (`∫ |W - V| ∂μ ≤ δ`) are tight exactly there."""
    deltas = [_F(0), _F(1, 10 ** 6), _F(-1, 10 ** 6), _F(1), _F(-1), _F(1, 2), _F(-1, 2), _F(10)]
    if isinstance(base, Func):
        table = list(base.table)
        if table:
            i = rng.randrange(len(table))
            d = rng.choice(deltas)
            table[i] = table[i] + d if not isinstance(table[i], ENN) else table[i] + _enn(abs(d))
        return Func(tuple(table), base.codomain)
    if isinstance(base, frozenset):
        return base
    if isinstance(base, Meas):
        w = list(base.weights)
        if w:
            i = rng.randrange(len(w))
            d = rng.choice(deltas)
            if not w[i].inf and w[i].value + d >= 0:
                w[i] = ENN(w[i].value + d)
        return Meas(tuple(w))
    return base


def _random_model_value(kind, rng: random.Random, n, probability):
    if kind == 'Measure':
        for _ in range(16):
            w = [rng.choice(_WEIGHT_POOL) if rng.random() < 0.8 else Fraction(rng.randint(0, 10 ** 6), rng.randint(1, 10 ** 3))
                 for _ in range(n)]
            total = sum(w)
            if probability:
                if total == 0:
                    continue
                w = [x / total for x in w]
            return Meas(tuple(ENN(x) for x in w))
        return Meas(tuple(ENN(Fraction(1, n)) for _ in range(n)))
    if kind == 'Func':
        r = rng.random()
        if r < 0.2:
            c = rng.choice(_TABLE_POOL)
            return Func(tuple(c for _ in range(n)))
        return Func(tuple(rng.choice(_TABLE_POOL) if rng.random() < 0.8 else _random_value(rng, 'Real') for _ in range(n)))
    if kind == 'Set':
        r = rng.random()
        if r < 0.15:
            return frozenset(range(n))
        if r < 0.3:
            return frozenset()
        return frozenset(i for i in range(n) if rng.random() < 0.5)
    raise Unsupported(f'model kind {kind}')


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
    model = any(t in _MODEL_KINDS for _, t in variables)
    if model:
        conjuncts = _split_conjuncts(thm.statement)
        if all(isinstance(c, App) and isinstance(c.fn, Var) and c.fn.name in _STRUCTURAL_PREDICATES for c in conjuncts):
            names = sorted({c.fn.name for c in conjuncts})
            return EvalResult('UNSUPPORTED', f'conclusion is a structural predicate {names}: trivial on finite discrete '
                              'models (every set is measurable, every function measurable and integrable), so sampling '
                              'has no content here', 0, False, [], [])
    probability = {h.args[0].name for h in hyps if isinstance(h, App) and isinstance(h.fn, Var)
                   and h.fn.name == 'IsProbabilityMeasure' and len(h.args) == 1 and isinstance(h.args[0], Var)}
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
            if typ in _MODEL_KINDS:
                others = [env[n].value for n, t in variables[:index] if t == typ and n in env] if typ != 'Measure' else []
                for _ in range(_DRAW_RETRIES):
                    if others and rng.random() < 0.4:
                        v = _neighbour_model_value(rng.choice(others), rng)      # W near (or equal to) V
                    else:
                        v = _random_model_value(typ, rng, ev.model_n, name in probability)
                    env[name] = Bound(v, typ)
                    if hyps_hold(completed[index], env):
                        return True
                return False
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
            key = (ev.model_n,) + tuple(env[n].value for n, _ in variables)
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
            witness = ', '.join(f'{n} = {_fmt_value(env[n].value)}' for n, _ in variables)
            if model:
                witness = f'Ω = Fin {ev.model_n}, ' + witness
            detail = (f'refuted at {witness} (sample {draws}; {accepted - 1} sampled instances held before it); '
                      'the hypotheses hold at the witness')
            record = {n: _fmt_value(env[n].value) for n, _ in variables}
            if model:
                record['Ω'] = f'Fin {ev.model_n}'
            return EvalResult('FAIL', detail, accepted, False, probes, ev.values_seen,
                              {'seed': seed, 'draws': draws, 'accepted': accepted, 'rejected': rejected,
                               'infeasible': infeasible, 'witness': record})

        # structured points: every variable takes the i-th pool value (and the i-th carrier size)
        for i in range(_DIAGONAL):
            env = {}
            if model:
                ev.model_n = _MODEL_SIZES[i]
            for name, typ in variables:
                if typ in _MODEL_KINDS:
                    env[name] = Bound(_structured_model_value(typ, i, ev.model_n, name in probability), typ)
                    continue
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
            if model:
                ev.model_n = rng.choice(_MODEL_SIZES)
            if not all(draw(i, env) for i in range(len(variables))):
                rejected += 1
                continue
            ok = check(env)
            if ok is False:
                return refuted(env)
        # information only (A6): the conclusion at hypothesis boundaries the hypotheses exclude
        for name, typ in variables:
            if len(probes) >= 8 or model:
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


def required_mutants(stmt, typing=None):
    """Class 1 — guaranteed-false mutants: false whenever the original is true. A comparison replaced by
    its negation (one conjunct at a time), and the right-hand literal of an equality shifted by one — only
    where the left-hand side is numeric (``typing(node)``): `weightedLaw μ W = 0` compares measures, and
    `= 1` there is not a Lean term. Every one of these must be reported FAIL."""
    out = []

    def visit(node, rebuild):
        out.append((f'negate {node.op}', rebuild(BinOp(_NEGATE[node.op], node.left, node.right))))
        if node.op == '=' and isinstance(node.right, Num):
            t = typing(node.left) if typing is not None else None
            if t not in ('Measure', 'Set', 'Func', 'Prop', 'Point'):
                out.append(('shift rhs literal +1', rebuild(BinOp('=', node.left, Num(node.right.value + 1)))))
    _walk_comparisons(stmt, visit)
    return out


def _static_typing(module: Module, thm: Theorem):
    """A best-effort type oracle for mutant generation: the theorem's binders as placeholder values."""
    ev = Evaluator(module)
    env = {}
    for b in thm.binders:
        if b.kind == 'var':
            placeholder = Func((), 'Real') if b.type == 'Func' else None
            env[b.name] = Bound(placeholder, b.type)

    def typing(node):
        try:
            return ev.arith_type(node, env)
        except Unsupported:
            return None
    return typing


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
        for label, mutated in required_mutants(thm.statement, _static_typing(module, thm)):
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
    # a package's modules import one another: definitions and carrier types are visible package-wide
    shared_defs, shared_points = {}, set()
    for m in modules.values():
        for name, d in m.defs.items():
            shared_defs.setdefault(name, d)
        shared_points |= m.point_types
    for m in modules.values():
        m.defs = shared_defs
        m.point_types = shared_points
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
    sampled = {name: r['sampling'] for name, r in per_target.items() if r.get('sampling')}
    sampling_totals = {
        'sampled_targets': len(sampled),
        'accepted_total': sum(x.get('accepted', 0) for x in sampled.values()),
        'rejected_total': sum(x.get('rejected', 0) for x in sampled.values()),
        'infeasible_total': sum(x.get('infeasible', 0) for x in sampled.values()),
        # a8 (742/5728819617 nit 1): an infeasible draw is skipped, never approximated — say where it happened
        'targets_with_infeasible_draws': sorted(n for n, x in sampled.items() if x.get('infeasible', 0))}
    return {'package_root': str(root), 'manifest_sha256': _sha256(manifest_path), 'module_sha256': module_hashes,
            'targets': len(per_target), 'status_counts': counts, 'per_target': per_target,
            'unregistered_theorems': unregistered, 'manifest_targets': normalized_targets,
            'sampling': {'seed': seed, 'samples': samples, 'admit_corroborated': admit_corroborated},
            'sampling_totals': sampling_totals,
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
    summary = {k: report[k] for k in ('package_root', 'targets', 'status_counts', 'unregistered_theorems', 'sampling',
                                      'sampling_totals', 'exit_code')}
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
