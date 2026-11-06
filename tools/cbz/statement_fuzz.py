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
    python3 tools/cbz/statement_fuzz.py --lean-dir <dir> [--exclude <substring>]… [--json report.json]
Exit 0 only when every registered target is PASS — or PASS/CORROBORATED with --admit-corroborated —
(and, with --selftest, every required mutant is killed and the harness control passes). The census mode
`--lean-dir` takes every theorem under a directory as a target without a manifest (coverage evidence, not a
gate): byte-identical files are evaluated once and UNSUPPORTED reasons are classified.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
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


_ASCII_DIGITS = set('0123456789')
_MULTI_OPS = ('<=', '>=', '!=', '->', ':=', '=>', '/\\', '\\/', '<->', '||', '&&', '..', '⁻¹', '⬝ᵥ')
_SINGLE = set('+-*/%^=<>()[],:|.{}!∀∃∧∨¬→↔≤≥≠∈∉λ√∫∂ᵐ•⊆≪∩∪ᶜ∅;‖∑∏↑')
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
        if ch in _ASCII_DIGITS:             # not str.isdigit(): subscript digits such as `₀` are identifier chars
            j = i
            while j < n and (text[j] in _ASCII_DIGITS or text[j] == '_'):
                j += 1
            # a scientific literal `1.5` (Lean `OfScientific`): exact decimal, kept as a Fraction downstream —
            # but not after a projection dot (`p.2.1` is two projections)
            after_dot = bool(toks) and toks[-1].value == '.' and toks[-1].kind == 'op'
            if not after_dot and j + 1 < n and text[j] == '.' and text[j + 1] in _ASCII_DIGITS:
                j += 1
                while j < n and text[j] in _ASCII_DIGITS:
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
    types: list | None = None     # explicit binder types `fun (n : ℕ) => …` (None per parameter when absent)

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

@dataclasses.dataclass
class VecLit:
    items: list             # `![a, b, c]` : Fin 3 → α

@dataclasses.dataclass
class MatLit:
    rows: list              # `!![a, b; c, d]` : Matrix (Fin 2) (Fin 2) α

@dataclasses.dataclass
class BigOp:
    op: str                 # '∑' or '∏'
    var: str
    type: str | None        # `∑ i : Fin n, …`
    domain: object | None   # `∑ i ∈ s, …` / `∑ i in s, …`
    body: object

@dataclasses.dataclass
class Norm:
    expr: object            # `‖v‖`: the sup norm on `Fin n → ℝ`, the Euclidean norm on `EuclideanSpace ℝ (Fin n)`

@dataclasses.dataclass
class Coe:
    expr: object            # `↑e`: the coercion Lean inserts, elaborated at the context's type


_NORMALIZE = {'<=': '≤', '>=': '≥', '!=': '≠', '->': '→', '/\\': '∧', '\\/': '∨', '<->': '↔'}
_PREC = {'→': 25, '↔': 20, '∨': 30, '∧': 35, '||': 30, '&&': 35, '=': 50, '≠': 50, '<': 50, '>': 50, '≤': 50, '≥': 50,
         '∈': 50, '∉': 50, '⊆': 50, '≪': 50, '+': 65, '-': 65, '∪': 65, '*': 70, '/': 70, '%': 70, '∩': 70, '⬝ᵥ': 72, '•': 73,
         '^': 75}
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
        self.norm_depth = 0

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
        if t.value == '!' and not self._at_vector_literal(0):
            self.i += 1
            return BoolNot(self.parse_expr(80))
        if t.value in ('∑', '∏'):
            return self.parse_bigop()
        if t.value == '↑':
            self.i += 1
            return Coe(self.parse_atom())
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

    def _at_vector_literal(self, k):
        """Whether the tokens at offset k start `![` or `!![`."""
        a, b, c = self.peek(k), self.peek(k + 1), self.peek(k + 2)
        return a is not None and a.value == '!' and b is not None and (
            b.value == '[' or (b.value == '!' and c is not None and c.value == '['))

    def parse_vector_literal(self):
        """`![a, b, c]` (Mathlib `Matrix.vecCons` notation): the opening `![` has been consumed."""
        items = []
        while not self.at(']'):
            items.append(self.parse_expr(0))
            if self.at(','):
                self.eat(',')
        self.eat(']')
        return VecLit(items)

    def parse_matrix_literal(self):
        """`!![a, b; c, d]` (Mathlib `Matrix.of` notation): rows separated by `;`, entries by `,`."""
        rows, row = [], []
        while not self.at(']'):
            row.append(self.parse_expr(0))
            if self.at(','):
                self.eat(',')
            elif self.at(';'):
                self.eat(';')
                rows.append(row)
                row = []
        self.eat(']')
        if row:
            rows.append(row)
        if rows and any(len(r) != len(rows[0]) for r in rows):
            raise Unsupported('ragged matrix literal')
        return MatLit(rows)

    def parse_bigop(self):
        """`∑ i, f i`, `∑ i : Fin n, f i`, `∑ i ∈ s, f i`, `∑ i in s, f i` (and `∏`): BigOperators notation
        over a finite type or a Finset. A single binder only; the body extends as far as an expression does."""
        op = self.eat().value
        if self.at("'"):
            raise Unsupported(f"infinite {'sum' if op == '∑' else 'product'} {op}' (tsum / tprod) is outside the interpreter")
        if self.at('('):
            self.eat('(')
            var = self.eat().value
            typ = None
            if self.at(':'):
                self.eat(':')
                parts = []
                while not self.at(')'):
                    parts.append(self.eat().value)
                typ = ' '.join(parts)
            self.eat(')')
            domain = None
        else:
            vt = self.eat()
            if vt.kind != 'ident':
                raise Unsupported(f'big operator binder {vt.value!r}')
            var = vt.value
            typ, domain = None, None
            if self.at(':'):
                self.eat(':')
                parts = []
                while not self.at(','):
                    parts.append(self.eat().value)
                typ = ' '.join(parts)
            elif self.at('∈') or self.at('in'):
                self.eat()
                domain = self.parse_expr(51)
            elif not self.at(','):
                raise Unsupported('big operator with several binders')
        self.eat(',')
        body = self.parse_expr(67)       # Mathlib `notation3 "∑ " … r:67`: `∑ i, f i ≤ 1` is `(∑ i, f i) ≤ 1`
        return BigOp(op, var, typ, domain, body)

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
        """`fun x y => e`, `fun (x : T) => e`, `fun x : T => e`: the binder types are kept (Codex 752/4471516437 —
        a returned function's parameters take their domain types, never the codomain's)."""
        self.eat()
        params, types = [], []
        while not self.at('=>'):
            t = self.eat()
            if t.value == '(':
                names = []
                while not self.at(':') and not self.at(')'):
                    nt = self.eat()
                    if nt.kind != 'ident':
                        raise Unsupported(f'lambda binder pattern {nt.value!r}')
                    names.append(nt.value)
                typ = None
                if self.at(':'):
                    self.eat(':')
                    parts = []
                    while not self.at(')'):
                        parts.append(self.eat().value)
                    typ = ' '.join(parts)
                self.eat(')')
                params.extend(names)
                types.extend([typ] * len(names))
            else:
                if t.kind != 'ident':
                    raise Unsupported(f'lambda binder pattern {t.value!r}')
                params.append(t.value)
                typ = None
                if self.at(':'):
                    self.eat(':')
                    parts = []
                    while not self.at('=>'):
                        parts.append(self.eat().value)
                    typ = ' '.join(parts)
                types.append(typ)
        self.eat('=>')
        return Lambda(params, self.parse_expr(0), types if any(t is not None for t in types) else None)

    def parse_application(self):
        head = self.parse_atom()
        args = []
        while True:
            t = self.peek()
            if t is None:
                break
            if t.kind == 'ident' and t.value in ('if', 'then', 'else', 'fun', 'λ', 'in'):
                break
            if t.kind in ('num', 'ident') or (t.kind == 'op' and t.value in ('(', '[')) or self._at_vector_literal(0) \
                    or (t.kind == 'op' and t.value == '‖' and self.norm_depth == 0):
                args.append(self.parse_atom())
                continue
            if t.kind == 'op' and t.value == '↑':
                self.i += 1
                args.append(Coe(self.parse_atom()))
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
                    for part in nxt.value.split('.'):
                        atom = Proj(atom, int(part))
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
        if t.value == '‖':
            # `‖e‖`: inside the bars a `‖` closes (as for `|e|`); outside, it opens an argument
            self.norm_depth += 1
            e = self.parse_expr(0)
            self.norm_depth -= 1
            self.eat('‖')
            return Norm(e)
        if t.value == '!':
            if self.at('['):
                self.eat('[')
                return self.parse_vector_literal()
            if self.at('!') and self.peek(1) is not None and self.peek(1).value == '[':
                self.eat('!')
                self.eat('[')
                return self.parse_matrix_literal()
            raise SyntaxError("'!' is not an atom")
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
                parts = []
                while not self.at(')'):
                    parts.append(self.eat().value)
                self.eat(')')
                return Ascribe(e, ' '.join(parts))
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
    params: list            # [(name, type)] explicit parameters
    result_type: str | None
    body: object | None     # expression, or None when pattern-matching
    clauses: list           # [(patterns, expr)] for pattern-matching defs
    namespace: str = ''     # names in the body resolve from the declaration's namespace, not the caller's
    opens: tuple = ()       # namespaces `open`ed where the definition was declared
    implicits: list = dataclasses.field(default_factory=list)   # [(name, type)] `{n : ℕ}`: inferred from the arguments

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
    namespace: str = ''     # the namespace the declaration sits in: names in its statement resolve from here outward
    opens: tuple = ()       # namespaces `open`ed in its scope


class DefTable:
    """Definitions by qualified name, resolved the way Lean resolves an identifier written in a statement: from
    the statement's namespace outward (`f` inside `namespace A.B` → `A.B.f`, `A.f`, `f`), then through the
    `open`ed namespaces, in the module itself first and then in the other modules of the group when exactly one
    definition (up to identical text) carries the name. A name defined differently in two other modules is
    ambiguous: using it raises Unsupported, never a silent pick (Codex 752/4471516451)."""

    def __init__(self, local=None, shared=None):
        self.local = dict(local or {})            # qualified name → Def, this module
        self.shared = dict(shared or {})          # qualified name → [(module, Def)], the whole group

    @staticmethod
    def candidates(name, namespace='', opens=()):
        parts = namespace.split('.') if namespace else []
        out = ['.'.join(parts[:k] + [name]) for k in range(len(parts), 0, -1)] + [name]
        out += [f'{o}.{name}' for o in opens]
        return out

    def resolve(self, name, namespace='', opens=()):
        """The definition ``name`` denotes from ``namespace`` (None when there is none); Unsupported when the
        group defines it in incompatible ways."""
        for c in self.candidates(name, namespace, opens):      # innermost namespace first, then `open`s
            if c in self.local:                                  # the module's own definition under this name
                return self.local[c]
            if c in self.shared:
                distinct = []
                for mod, d in self.shared[c]:
                    if not any(d == e for _, e in distinct):
                        distinct.append((mod, d))
                if len(distinct) == 1:
                    return distinct[0][1]
                mods = sorted(m for m, _ in distinct)
                raise Unsupported(f'ambiguous definition {name}: {c} is defined differently in {mods}')
        return None

    def get(self, name, default=None):
        """Namespace-agnostic lookup (parser-time helpers): the module's own definition under the exact name."""
        return self.local.get(name, default)

    def __contains__(self, name):
        return name in self.local or name in self.shared

    def __iter__(self):
        return iter(self.local)

    def __len__(self):
        return len(self.local)

    def items(self):
        return self.local.items()

    def keys(self):
        return self.local.keys()

    def denotes(self, name, namespace='', opens=()):
        """Whether ``name``, written in ``namespace``, denotes a definition (ambiguity counts as denoting)."""
        try:
            return self.resolve(name, namespace, opens) is not None
        except Unsupported:
            return True


def share_definitions(modules):
    """Package / census: a group's modules import one another, so every module sees the group's definitions
    (resolved by qualified name, ambiguity fail-closed) and carrier types."""
    shared, points = {}, set()
    aliases = {}
    for rel, m in modules.items():
        for q, d in m.defs.items():
            shared.setdefault(q, []).append((rel, d))
        points |= m.point_types
        for k, v in m.aliases.items():
            aliases.setdefault(k, v)            # a type abbreviation defined twice differently stays the first;
    for m in modules.values():                  # the binder then names a type the interpreter checks structurally
        m.defs = DefTable(m.defs.local, shared)
        m.point_types = points
        m.aliases = {**aliases, **m.aliases}


@dataclasses.dataclass
class Module:
    namespace: str
    defs: DefTable
    theorems: list
    point_types: set = dataclasses.field(default_factory=set)   # `variable {Ω : Type*}`: finite-model carriers
    aliases: dict = dataclasses.field(default_factory=dict)     # `abbrev DM2Jet := Fin 9 → ℝ`: type abbreviations


_DECL_KEYWORDS = {'theorem', 'lemma', 'def', 'abbrev', 'example', 'instance', 'structure', 'inductive',
                  'namespace', 'end', 'open', 'import', 'section', 'variable', 'universe', 'set_option',
                  'noncomputable', 'private', 'protected', 'attribute', 'macro', 'syntax', 'notation', 'axiom'}


_INDENTED_DECL_KEYWORDS = {'theorem', 'lemma', 'def', 'abbrev', 'example', 'instance', 'structure', 'inductive',
                           'axiom', 'namespace', 'section', 'end', 'private', 'protected', 'noncomputable'}
_DECL_START_KEYWORDS = {'theorem', 'lemma', 'def', 'abbrev', 'example', 'instance', 'structure', 'inductive', 'axiom',
                        'private', 'protected', 'noncomputable'}


def _split_declarations(toks):
    """Declaration starts (Codex 752/4471516464 — a declaration the splitter misses is a target the census would
    silently drop): a declaration keyword at column 0; a declaration keyword that is the first token on its line at
    any indentation (declarations inside an indented `namespace`/`section`/`mutual` block — `open … in` and
    `set_option … in` are left out of this rule because they also occur as tactics); a declaration keyword right
    after the `in` of such a prefix on the same line (`open Real in theorem t …`); or one that follows an attribute
    group `@[simp] theorem …` whose bracket opens the line (the tokenizer drops `@`)."""
    starts = []
    for k, t in enumerate(toks):
        if t.kind != 'ident' or t.value not in _DECL_KEYWORDS:
            continue
        if t.col == 0:
            starts.append(k)
            continue
        first_on_line = k == 0 or toks[k - 1].line != t.line
        if first_on_line and t.value in _INDENTED_DECL_KEYWORDS:
            starts.append(k)
            continue
        if k > 0 and toks[k - 1].value == 'in' and t.value in _DECL_START_KEYWORDS:
            starts.append(k)
            continue
        if k > 0 and toks[k - 1].value == ']':
            j = k - 1
            depth = 0
            while j >= 0:
                if toks[j].value == ']':
                    depth += 1
                elif toks[j].value == '[':
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j >= 0 and toks[j].line == t.line and (toks[j].col <= 1 or j == 0 or toks[j - 1].line != toks[j].line):
                starts.append(j)                                             # `@` is dropped: `[` opens the line
    starts = sorted(set(starts)) + [len(toks)]
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


def _parse_binders(toks, start, end, implicits=None):
    """`(name : T) {m : U} [inst] …` → explicit [(name, T)] and the position after the binders; implicit binders
    `{m : U}` / `⦃m : U⦄` are appended to ``implicits`` (inferred at application: a `{n : ℕ}` that is the
    dimension of a vector argument takes that vector's length); instance binders are skipped."""
    params, k = [], start
    closers = {'(': ')', '{': '}', '[': ']', '⦃': '⦄'}
    while k < end and toks[k].value in closers:
        opener = toks[k].value
        close = k
        depth = 0
        while close < end:
            if toks[close].value == opener:
                depth += 1
            elif toks[close].value == closers[opener]:
                depth -= 1
                if depth == 0:
                    break
            close += 1
        if close >= end:
            break
        if opener == '[':
            k = close + 1
            continue
        names = []
        j = k + 1
        while j < close and toks[j].value != ':':
            names.append(toks[j].value); j += 1
        typ = ' '.join(t.value for t in toks[j + 1:close])
        if opener == '(':
            params.extend((nm, typ) for nm in names)
        elif implicits is not None:
            implicits.extend((nm, typ) for nm in names)
        k = close + 1
    return params, k


def _qualify(namespace: str, name: str) -> str:
    if name.startswith('_root_.'):
        return name[len('_root_.'):]
    return f'{namespace}.{name}' if namespace else name


def parse_module(source: str) -> Module:
    """Definitions (by qualified name) and theorems of one Lean file. Scopes are tracked: `namespace A.B` /
    `section` push, `end` pops, so a declaration's full name and the namespace its statement resolves from are
    the ones Lean gives it. Every `theorem`/`lemma` keyword the splitter did not reach (a declaration
    mid-line after another one) is still recorded, as an UNSUPPORTED target, so a census accounts for every
    declaration in the file (Codex 752/4471516464)."""
    toks = tokenize(source)
    defs, theorems = {}, []
    point_types = set()
    aliases = {}                     # type abbreviations by short name (and qualified name)
    scopes = []                      # [(kind, name)], kind in ('namespace', 'section')
    opens = [()]                     # `open` namespaces per scope depth
    pending_opens = ()               # `open X in` applies to the next declaration only

    def namespace_of(stack):
        return '.'.join(nm for kind, nm in stack if kind == 'namespace' and nm)

    for start, end in _split_declarations(toks):
        namespace = namespace_of(scopes)
        k = start
        if toks[k].value == '[':
            close = _matching(toks, k, end)
            k = close + 1 if close != -1 else k
        kw = toks[k].value if k < end else toks[start].value
        while k < end and toks[k].value in ('noncomputable', 'private', 'protected'):
            k += 1
        kw = toks[k].value if k < end else kw
        decl_keyword_at = k
        # a second `theorem`/`lemma` keyword inside this range (Lean accepts declarations mid-line) is a
        # declaration the splitter could not separate: account for it, fail-closed
        for j in range(decl_keyword_at + 1, end):
            tj = toks[j]
            if tj.kind == 'ident' and tj.value in ('theorem', 'lemma') and j + 1 < end and toks[j + 1].kind == 'ident' \
                    and _find(toks, ':', j + 2, end) != -1:
                theorems.append(Theorem(_qualify(namespace, toks[j + 1].value), None,
                                        f'declaration not reached by the parser: `{tj.value} {toks[j + 1].value}` '
                                        f'begins mid-line at line {tj.line}', '', [], namespace, tuple(opens[-1])))
        if kw == 'namespace':
            scopes.append(('namespace', toks[k + 1].value if k + 1 < end else ''))
            opens.append(opens[-1])
            continue
        if kw == 'section':
            scopes.append(('section', toks[k + 1].value if k + 1 < end and toks[k + 1].kind == 'ident' else None))
            opens.append(opens[-1])
            continue
        if kw == 'end':
            if scopes:
                scopes.pop()
            if len(opens) > 1:
                opens.pop()
            continue
        if kw == 'open':
            names = tuple(t.value for t in toks[k + 1:end]
                          if t.kind == 'ident' and t.value not in ('scoped', 'in', 'hiding', 'renaming'))
            if toks[end - 1].value == 'in':
                pending_opens = pending_opens + names
            else:
                opens[-1] = tuple(opens[-1]) + names
            continue
        if kw == 'set_option':           # `set_option … in` is a prefix too: it does not consume a pending `open … in`
            continue
        scope = tuple(opens[-1]) + pending_opens
        pending_opens = ()
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
            implicits = []
            params, j = _parse_binders(toks, k + 2, end, implicits)
            result_type = None
            assign = _find(toks, ':=', j, end)
            bar = _find(toks, '|', j, end)
            if toks[j].value == ':':
                type_end = assign if assign != -1 else (bar if bar != -1 else end)
                result_type = ' '.join(t.value for t in toks[j + 1:type_end])
            if assign != -1 and (bar == -1 or assign < bar):
                try:
                    body = Parser(toks, assign + 1, end).parse_expr(0)
                except (SyntaxError, Unsupported, ValueError, IndexError, RecursionError):
                    body = None
                rhs = ' '.join(t.value for t in toks[assign + 1:end])
                if not params and (result_type is None or result_type.split()[0] == 'Type') \
                        and _kind_of_type(rhs, point_types, aliases) is not None:
                    aliases[name] = rhs                      # `abbrev DM2Jet := Fin 9 → ℝ`
                    aliases[_qualify(namespace, name)] = rhs
                defs[_qualify(namespace, name)] = Def(_qualify(namespace, name), params, result_type, body, [],
                                                      namespace, scope, implicits)
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
                        except (SyntaxError, Unsupported, ValueError, IndexError, RecursionError):
                            expr = None
                        clauses.append((pats, expr))
                except (Unsupported, SyntaxError, IndexError, ValueError, RecursionError):
                    clauses = []
                defs[_qualify(namespace, name)] = Def(_qualify(namespace, name), params, result_type, None, clauses,
                                                      namespace, scope, implicits)
            continue
        if kw in ('theorem', 'lemma'):
            name = toks[k + 1].value
            colon = _find(toks, ':', k + 2, end)
            assign = _find(toks, ':=', k + 2, end)
            full = _qualify(namespace, name)
            if colon == -1 or assign == -1 or assign < colon:
                theorems.append(Theorem(full, None, 'malformed theorem', '', [], namespace, scope))
                continue
            binder_text = ' '.join(t.value for t in toks[k + 2:colon])
            text = ' '.join(t.value for t in toks[colon + 1:assign])
            if binder_text:
                text = f'{binder_text} : {text}'
            try:
                binders, binder_error = _parse_theorem_binders(toks, k + 2, colon, DefTable(defs), point_types,
                                                               namespace, scope, aliases)
            except (SyntaxError, Unsupported, ValueError, IndexError, RecursionError) as e:
                binders, binder_error = [], f'unparsed binders: {type(e).__name__}: {e}'
            if binder_error:
                theorems.append(Theorem(full, None, binder_error, text, binders, namespace, scope))
                continue
            try:
                parser = Parser(toks, colon + 1, assign)
                stmt = parser.parse_expr(0)
                if parser.i != assign:
                    raise SyntaxError(f'trailing tokens from {toks[parser.i].value!r}')
                theorems.append(Theorem(full, _desugar(stmt), None, text, binders, namespace, scope))
            except Unsupported as e:
                theorems.append(Theorem(full, None, f'unsupported syntax: {e}', text, binders, namespace, scope))
            except (SyntaxError, ValueError, IndexError, RecursionError) as e:
                theorems.append(Theorem(full, None, f'unparsed statement: {type(e).__name__}: {e}', text, binders,
                                        namespace, scope))
            continue
    return Module(namespace_of(scopes), DefTable(defs), theorems, point_types, aliases)


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
_VECTOR_KINDS = ('Vec', 'Fin', 'Mat')         # binder kinds sampled as exact finite vectors, indices and matrices
_DIM_RE = re.compile(r'^(?:\d+|[A-Za-z_][A-Za-z0-9_\']*)$')


def _base(typ: str) -> str:
    """The kind of an interpreter type string: 'Vec:Real:6' → 'Vec' (plain kinds are their own base)."""
    return typ.split(':', 1)[0] if isinstance(typ, str) else typ


def _vec_info(typ: str):
    """(element type, dimension token, space) of a 'Vec:…' type; the dimension token is a literal or the name
    of the ℕ binder that carries it; space is 'pi' (`Fin n → ℝ`, sup norm, pointwise order) or 'euclid'
    (`EuclideanSpace ℝ (Fin n)`, L² norm)."""
    parts = typ.split(':')
    return parts[1], parts[2], (parts[3] if len(parts) > 3 else 'pi')


def _mat_info(typ: str):
    parts = typ.split(':')
    return parts[1], parts[2], parts[3]


def _dims_of(typ: str) -> list:
    """The dimension tokens of a vector / index / matrix type ('Fin:d' → ['d']; 'Mat:Real:2:n' → ['2', 'n'])."""
    b = _base(typ)
    if b == 'Fin':
        return [typ.split(':')[1]]
    if b == 'Vec':
        return [_vec_info(typ)[1]]
    if b == 'Mat':
        return list(_mat_info(typ)[1:])
    return []


def _expand_aliases(t: str, aliases) -> str:
    """Replace type abbreviations (`abbrev DM2Jet := Fin 9 → ℝ`) by their right-hand side, whole words only."""
    if not aliases:
        return t
    out = []
    for w in t.split():
        a = aliases.get(w) or aliases.get(w.rsplit('.', 1)[-1])
        out.append(f'( {a} )' if a is not None and a != w else w)
    return ' '.join(out)


def _strip_parens(t: str) -> str:
    t = t.strip()
    while t.startswith('(') and t.endswith(')') and _matching_paren(t, 0) == len(t) - 1:
        t = t[1:-1].strip()
    return t


def _matching_paren(t: str, i: int) -> int:
    depth = 0
    for j in range(i, len(t)):
        if t[j] == '(':
            depth += 1
        elif t[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    return -1


def _fin_dim(t: str):
    """'Fin 6' / 'Fin d' / '( Fin 6 )' → the dimension token, else None."""
    w = _strip_parens(t).split()
    if len(w) == 2 and w[0] == 'Fin' and _DIM_RE.match(w[1]):
        return w[1]
    return None


def _arrow_parts(t: str) -> list:
    """The components of an arrow type at bracket depth 0: `ℝ → ℝ → ℝ` → ['ℝ', 'ℝ', 'ℝ'];
    `Set ( Fin d → ℤ )` → ['Set ( Fin d → ℤ )'] (an arrow inside brackets is not a function type)."""
    parts, cur, depth = [], [], 0
    for ch in t:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == '→' and depth == 0:
            parts.append(''.join(cur).strip()); cur = []
        else:
            cur.append(ch)
    parts.append(''.join(cur).strip())
    return parts


def _kind_of_type(t: str, point_types, aliases=None) -> str | None:
    """The interpreter's kind for a Lean type written as a string: a numeric type ('Nat', 'Int', 'Rat', 'Real'),
    'ENNReal', 'Bool', 'Point' (a finite-model carrier Ω), 'Measure' (`Measure Ω`), 'Set' (`Set Ω`), 'Func'
    (`Ω → ℝ`, a table on the carrier); a finite vector 'Vec:<elem>:<dim>[:euclid]' (`Fin 6 → ℝ`, `Fin d → ℤ`,
    `EuclideanSpace ℝ (Fin n)`), an index 'Fin:<dim>' (`Fin 6`), a matrix 'Mat:<elem>:<rows>:<cols>'
    (`Matrix (Fin 2) (Fin 2) ℝ`); or None when the type is outside the interpreter (`ℝ → Measure Ω`,
    `Ω → Ω`, structures). Type abbreviations are expanded first."""
    t = _strip_parens(_expand_aliases(t.strip(), aliases))
    parts = _arrow_parts(t)
    if len(parts) == 2 and parts[0] in point_types:
        return 'Func' if parts[1] in _SAMPLED_TYPE_TOKENS or parts[1] in ('ENNReal', 'ℝ≥0∞') else None
    if len(parts) == 2 and _fin_dim(parts[0]) is not None and _strip_parens(parts[1]) in _SAMPLED_TYPE_TOKENS:
        return f'Vec:{_SAMPLED_TYPE_TOKENS[_strip_parens(parts[1])]}:{_fin_dim(parts[0])}'
    if len(parts) >= 2:
        return None
    dim = _fin_dim(t)
    if dim is not None:
        return f'Fin:{dim}'
    words = t.split()
    if len(words) >= 3 and words[0] == 'EuclideanSpace' and words[1] in _REAL_TYPES:
        d = _fin_dim(' '.join(words[2:]))
        if d is not None:
            return f'Vec:Real:{d}:euclid'
    if words and words[0] == 'Matrix':
        rest = t[len('Matrix'):].strip()
        # Matrix ( Fin m ) ( Fin n ) α
        if rest.startswith('('):
            c1 = _matching_paren(rest, 0)
            r2 = rest[c1 + 1:].strip()
            if c1 != -1 and r2.startswith('('):
                c2 = _matching_paren(r2, 0)
                elem = r2[c2 + 1:].strip() if c2 != -1 else ''
                m_, n_ = _fin_dim(rest[:c1 + 1]), _fin_dim(r2[:c2 + 1]) if c2 != -1 else None
                if m_ is not None and n_ is not None and elem in _SAMPLED_TYPE_TOKENS:
                    return f'Mat:{_SAMPLED_TYPE_TOKENS[elem]}:{m_}:{n_}'
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


def _parse_theorem_binders(toks, start, end, defs, point_types=frozenset(), namespace='', opens=(), aliases=None):
    """Binder groups between the theorem name and its colon → ([Binder], error or None).

    `(x y : ℝ)` / `{x : ℕ}` with a sampled numeric type declare variables; `(h : p)` whose type parses as a
    proposition over names declared so far (variables, the module's definitions, the interpreter's
    builtins) is a hypothesis; anything else — an instance binder `[Fact p]`, a function or structure type
    `(f : Ω → ℝ)`, `(μ : Measure Ω)`, a predicate the interpreter does not define — makes the theorem
    UNSUPPORTED, fail-closed, naming the binder."""
    binders, declared = [], set()
    nat_vars = {}                 # declared variable → kind (dimension binders must be ℕ)
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
        kind = _kind_of_type(' '.join(x.value for x in type_toks), point_types, aliases)
        if kind is not None and _base(kind) in _VECTOR_KINDS:
            # a dimension that is a name must be a ℕ binder declared before it (`(n : ℕ) (j : Fin n → ℝ)`)
            for dim in _dims_of(kind):
                if not dim.isdigit() and nat_vars.get(dim) != 'Nat':
                    return binders, f'binder {txt}: dimension {dim} is not a ℕ binder declared before it'
        if kind in _SAMPLED_TYPE_TOKENS.values() or kind in _MODEL_KINDS or (kind is not None and _base(kind) in _VECTOR_KINDS):
            for nm in names:
                binders.append(Binder('var', nm, kind, txt))
                declared.add(nm)
                nat_vars[nm] = kind
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
            known = declared | _BUILTIN_NAMES | _TRANSCENDENTAL | point_types

            def is_known(v):
                if v in known or v.split('.')[0] in known:
                    return True
                if isinstance(defs, DefTable):
                    return defs.denotes(v, namespace, opens)
                return v in defs
            unknown = sorted(v for v in _free_vars(node) if not is_known(v))
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
    """A lambda with its environment. ``types`` are the interpreter type names of the parameters when they are
    known (explicit binder types, or the domain of the arrow type a definition declares); a parameter without
    one is bound at the type the call site evaluates at."""
    def __init__(self, params, body, env, types=None, namespace='', opens=()):
        self.params, self.body, self.env = params, body, env
        self.types = list(types) if types else [None] * len(params)
        self.namespace, self.opens = namespace, tuple(opens)


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


class Vec:
    """An exact finite vector `Fin n → α` (entries are ints, Fractions or Alg); ``space`` is 'pi' for the Pi type
    (sup norm, pointwise order) or 'euclid' for `EuclideanSpace ℝ (Fin n)` (L² norm, no order)."""
    __slots__ = ('entries', 'elem', 'space')

    def __init__(self, entries, elem='Real', space='pi'):
        self.entries, self.elem, self.space = tuple(entries), elem, space

    def __eq__(self, other):
        return isinstance(other, Vec) and self.entries == other.entries

    def __hash__(self):
        return hash(('Vec', self.entries))

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return '![' + ', '.join(_fmt(v) for v in self.entries) + ']'


class Mat:
    """An exact matrix `Matrix (Fin m) (Fin n) α` as rows."""
    __slots__ = ('rows', 'elem')

    def __init__(self, rows, elem='Real'):
        self.rows, self.elem = tuple(tuple(r) for r in rows), elem

    def __eq__(self, other):
        return isinstance(other, Mat) and self.rows == other.rows

    def __hash__(self):
        return hash(('Mat', self.rows))

    @property
    def shape(self):
        return len(self.rows), (len(self.rows[0]) if self.rows else 0)

    def __repr__(self):
        return '!![' + '; '.join(', '.join(_fmt(v) for v in r) for r in self.rows) + ']'


def _fmt_set(points, n=None) -> str:
    return '{' + ', '.join(str(i) for i in sorted(points)) + '}'


def _fmt_value(v) -> str:
    if isinstance(v, frozenset):
        return _fmt_set(v)
    if isinstance(v, (Meas, Func, ENN, Vec, Mat, ExpPoly, Iv)):
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
    def __init__(self, module: Module, namespace: str = '', opens=()):
        self.module = module
        self.namespace, self.opens = namespace, tuple(opens)
        self.values_seen = []
        self.instances = 0
        self.vacuous = False
        self.failures = []
        self.probes = []
        self.model_n = None          # the carrier size of the current finite model, when one is in force
        self.bits = _ENCLOSURE_BITS[0]   # precision of transcendental enclosures (refined on Undecided)

    def defn(self, name):
        """The definition a name written in the statement denotes (None if none; Unsupported if ambiguous)."""
        defs = self.module.defs
        if isinstance(defs, DefTable):
            return defs.resolve(name, self.namespace, self.opens)
        return defs.get(name)

    def has_def(self, name):
        return self.defn(name) is not None

    @contextlib.contextmanager
    def declaration_scope(self, namespace, opens):
        """Resolve names in a definition or closure where it was declared, restoring the caller afterward."""
        previous = self.namespace, self.opens
        self.namespace, self.opens = namespace, tuple(opens)
        try:
            yield
        finally:
            self.namespace, self.opens = previous

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
            kind = _kind_of_type(node.type, self.module.point_types, self.module.aliases)
            if kind is not None:
                return kind
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
            if node.name in ('Real.pi', 'π'):
                return 'Real'
            d = self.defn(node.name)
            if d is not None and d.result_type:
                return self._type_name(d.result_type)
            return 'Nat'
        if isinstance(node, (Integral, Norm)):
            return 'Real'
        if isinstance(node, SetBuilder):
            return 'Set'
        if isinstance(node, AERel):
            return 'Prop'
        if isinstance(node, Coe):
            t = self.arith_type(node.expr, env)
            return 'Nat' if _base(t) == 'Fin' else t
        if isinstance(node, VecLit):
            elem = 'Nat'
            for x in node.items:
                elem = _max_type(elem, self.arith_type(x, env))
            if _base(elem) in ('Vec', 'Mat', 'Set', 'Func', 'Measure', 'Point', 'Function'):
                raise Unsupported('vector literal of non-scalar entries')
            return f'Vec:{"Real" if elem in ("Sci", "Bool") else elem}:{len(node.items)}'
        if isinstance(node, MatLit):
            elem = 'Nat'
            for r in node.rows:
                for x in r:
                    elem = _max_type(elem, self.arith_type(x, env))
            return f'Mat:{"Real" if elem in ("Sci", "Bool") else elem}:{len(node.rows)}:{len(node.rows[0]) if node.rows else 0}'
        if isinstance(node, BigOp):
            return self.arith_type(node.body, {**env, node.var: Bound(0, 'Fin:?')})
        if isinstance(node, (BinOp,)):
            if node.op in _COMPARISONS or node.op in _LOGIC:
                return 'Prop'
            if node.op == '^':
                return self.arith_type(node.left, env)      # the exponent is typed on its own (Lean `rightact%`)
            if node.op == '•':
                return self.arith_type(node.right, env)
            if node.op == '⬝ᵥ':
                t = _max_type(self.arith_type(node.left, env), self.arith_type(node.right, env))
                return _vec_info(t)[0] if _base(t) == 'Vec' else 'Real'
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
                    if _base(b.type) == 'Vec':
                        return _vec_info(b.type)[0]           # `v i`
                    if _base(b.type) == 'Mat':
                        elem, _, cols = _mat_info(b.type)
                        return elem if len(node.args) >= 2 else f'Vec:{elem}:{cols}'   # `M i k`, `M i`
                d = self.defn(fn.name)
                if d is not None and d.result_type:
                    rt = self._type_name(d.result_type)
                    if rt == 'Measure' and len(node.args) == len(d.params) + 1:
                        return 'ENNReal'                      # a defined measure applied to a set
                    if _base(rt) == 'Vec' and len(node.args) == len(d.params) + 1:
                        return _vec_info(rt)[0]               # a defined vector applied to an index
                    if _base(rt) == 'Mat' and len(node.args) >= len(d.params) + 2:
                        return _mat_info(rt)[0]
                    if not d.clauses and _base(rt) not in ('Vec', 'Mat') and len(_arrow_parts(d.result_type)) > 1 \
                            and len(node.args) == len(d.params):
                        return 'Function'                     # a function value, not carried
                    return rt
                if fn.name in ('.det', 'Matrix.det', '.trace', 'Matrix.trace') and node.args:
                    t = self.arith_type(node.args[-1], env)
                    return _mat_info(t)[0] if _base(t) == 'Mat' else 'Real'
                if fn.name in ('Pi.single', 'EuclideanSpace.single') and len(node.args) == 2:
                    return f'Vec:{self.arith_type(node.args[1], env)}:?'
                if fn.name in ('dotProduct', 'Matrix.dotProduct', 'inner') and node.args:
                    t = self.arith_type(node.args[-1], env)
                    return _vec_info(t)[0] if _base(t) == 'Vec' else 'Real'
                if fn.name == 'Real.sqrt' or fn.name in _TRANSCENDENTAL:
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
        kind = _kind_of_type(t, self.module.point_types, self.module.aliases)
        if kind is not None:
            return kind
        last = _arrow_parts(t)[-1]
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
        kind = _kind_of_type(last, self.module.point_types, self.module.aliases)
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
            if _base(typ) == 'Vec':
                # a numeral in a Pi type is the constant function (`0 : Fin n → ℝ` is the zero vector)
                elem, dim, space = _vec_info(typ)
                n = self._dim(dim, env)
                c = self.num(node, env, elem)
                return Vec([c] * n, elem, space)
            if _base(typ) == 'Mat':
                elem, m_, n_ = _mat_info(typ)
                m, n = self._dim(m_, env), self._dim(n_, env)
                zero, one = self.num(Num(0), env, elem), self.num(Num(1), env, elem)
                if node.value == 0:
                    return Mat([[zero] * n for _ in range(m)], elem)
                if node.value == 1 and m == n:
                    return Mat([[one if i == k else zero for k in range(n)] for i in range(m)], elem)
                raise Unsupported('a numeral other than 0 or 1 where a matrix is expected')
            if _base(typ) == 'Fin':
                # a numeral at `Fin n` is a `Fin n` literal: `(7 : Fin 6) = 1` wraps, so only numerals below n are read
                n = _resolve_dim(typ.split(':')[1], env)
                if not isinstance(node.value, int) or (n is not None and node.value >= n):
                    raise Unsupported(f'numeral {node.value} as a Fin {n} literal wraps modulo {n}')
                return self._rec(node.value)
            if not isinstance(node.value, int):
                raise Unsupported('scientific literal outside a rational or real context (Lean defaults it to Float)')
            return self._rec(node.value)
        if isinstance(node, Integral):
            return self._rec(self.integral(node, env))
        if isinstance(node, SetBuilder):
            return frozenset(i for i in self.points() if self.prop(node.body, {**env, node.var: Bound(i, 'Point')}))
        if isinstance(node, Lambda) and typ == 'Func':
            return self.materialize(Closure(node.params, node.body, env, node.types, self.namespace, self.opens))
        if isinstance(node, Lambda) and _base(typ) == 'Vec':
            return self._rec(self.vector_of_lambda(node, env, typ))
        if isinstance(node, VecLit):
            return self._rec(self.vector_literal(node, env, typ))
        if isinstance(node, MatLit):
            return self._rec(self.matrix_literal(node, env, typ))
        if isinstance(node, BigOp):
            return self._rec(self.bigop(node, env, typ))
        if isinstance(node, Norm):
            return self._rec(self.norm(node, env))
        if isinstance(node, Coe):
            src = self.arith_type(node.expr, env)
            v = self.num(node.expr, env, 'Nat' if _base(src) == 'Fin' else src)
            return self._rec(_coerce(v, src, typ))
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
            if node.name in ('Real.pi', 'π') and node.name not in env:
                return _pi_enclosure(self.bits)
            if node.name in env:
                v = env[node.name]
                return self._rec(v.value if isinstance(v, Bound) else v)
            if '.' in node.name and not node.name.startswith('.'):
                recv, method = node.name.rsplit('.', 1)
                if (recv in env or self.has_def(recv)) and method in _LIST_METHODS:
                    return self._rec(self.apply(App(Var('.' + method), [Var(recv)]), env, typ))
            d = self.defn(node.name)
            if d is not None:
                if d.params or d.clauses:
                    raise Unsupported(f'definition {node.name} used without arguments')
                if d.body is None:
                    raise Unsupported(f'definition {node.name} has an unparsed body')
                dtyp = self._type_name(d.result_type) if d.result_type else typ
                with self.declaration_scope(d.namespace, d.opens):
                    return self._rec(self.num(d.body, {}, dtyp))
            raise Unsupported(f'unknown name {node.name}')
        if isinstance(node, Neg):
            if typ == 'Nat':
                raise Unsupported('negation at type Nat')
            if _base(typ) in ('Vec', 'Mat'):
                v = self.num(node.expr, env, typ)
                if isinstance(v, Vec):
                    return Vec([-x for x in v.entries], v.elem, v.space)
                if isinstance(v, Mat):
                    return Mat([[-x for x in r] for r in v.rows], v.elem)
                raise Unsupported('negation of a non-vector at a vector type')
            return self._rec(-self.num(node.expr, env, typ))
        if isinstance(node, BinOp):
            if _base(typ) == 'Fin' and node.op in ('+', '-', '*', '/', '%', '^'):
                # arithmetic in `Fin n` wraps around (`(2 : Fin 3) + 1 = 0`); only `(i : ℕ)` / `↑i` arithmetic is read
                raise Unsupported('arithmetic on Fin values wraps modulo n: outside the interpreter (cast with (i : ℕ))')
            if node.op == '^':
                return self._rec(self._power(node, env, typ))
            if node.op == '•':
                rt = self.arith_type(node.right, env)
                if _base(rt) in ('Vec', 'Mat'):
                    target = typ if _base(typ) == _base(rt) and '?' not in typ else rt
                    lt = self.arith_type(node.left, env)
                    if _base(lt) in ('Vec', 'Mat', 'Set', 'Func', 'Measure', 'Point'):
                        raise Unsupported('scalar action by a non-scalar')
                    c = self.num(node.left, env, lt if lt not in ('Sci', 'Bool', 'Fin') else 'Nat')
                    v = self.num(node.right, env, target)
                    return self._scale(c, v)
                c = self.num(node.left, env, 'ENNReal')
                m = self.num(node.right, env, 'Measure')
                if not isinstance(m, Meas):
                    raise Unsupported('scalar action on a non-measure')
                return Meas(tuple(_enn(c) * w for w in m.weights))
            if node.op == '⬝ᵥ':
                return self._rec(self.apply(App(Var('dotProduct'), [node.left, node.right]), env, typ))
            if _base(typ) in ('Vec', 'Mat') and node.op in ('+', '-', '*', '/'):
                return self._rec(self.vector_arith(node.op, node.left, node.right, env, typ))
            if node.op in ('+', '-', '*', '/', '%') and (_base(self.arith_type(node.left, env)) == 'Fin'
                                                         or _base(self.arith_type(node.right, env)) == 'Fin'):
                raise Unsupported('arithmetic on Fin values wraps modulo n: outside the interpreter (cast with (i : ℕ))')
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

    # ---- finite vectors, indices and matrices (exact; `Fin n → ℝ`, `Fin n`, `Matrix (Fin m) (Fin n) ℝ`)
    def _dim(self, token: str, env) -> int:
        """The value of a dimension token: a literal, or a ℕ variable bound in ``env`` (`(n : ℕ) (j : Fin n → ℝ)`)."""
        if token == '?':
            raise Unsupported('a vector whose dimension the context does not fix')
        if token.isdigit():
            return int(token)
        if token in env and isinstance(env[token], Bound) and isinstance(env[token].value, int) \
                and not isinstance(env[token].value, bool):
            return env[token].value
        d = self.defn(token)
        if d is not None and not d.params and d.body is not None:
            v = self.num(d.body, {}, 'Nat')
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        raise Unsupported(f'dimension {token} is not bound')

    def _scale(self, c, v):
        if isinstance(v, Vec):
            return Vec([_exact(c) * _exact(x) if v.elem in _FIELD_TYPES else c * x for x in v.entries], v.elem, v.space)
        if isinstance(v, Mat):
            return Mat([[_exact(c) * _exact(x) if v.elem in _FIELD_TYPES else c * x for x in r] for r in v.rows], v.elem)
        raise Unsupported('scalar action on a non-vector')

    def vector_arith(self, op, left, right, env, typ):
        a, b = self.num(left, env, typ), self.num(right, env, typ)
        if isinstance(a, Vec) and isinstance(b, Vec):
            if len(a) != len(b):
                raise Unsupported('vectors of different dimensions')
            if op == '*' and (a.space == 'euclid' or b.space == 'euclid'):
                raise Unsupported('pointwise product on EuclideanSpace (no Mul instance)')
            elem = a.elem
            return Vec([self._arith(op, x, y, elem) for x, y in zip(a.entries, b.entries)], elem, a.space)
        if isinstance(a, Mat) and isinstance(b, Mat):
            if op in ('+', '-'):
                if a.shape != b.shape:
                    raise Unsupported('matrices of different shapes')
                return Mat([[self._arith(op, x, y, a.elem) for x, y in zip(ra, rb)] for ra, rb in zip(a.rows, b.rows)], a.elem)
            if op == '*':
                m, k = a.shape
                k2, n = b.shape
                if k != k2:
                    raise Unsupported('matrix product of incompatible shapes')
                rows = []
                for i in range(m):
                    row = []
                    for j in range(n):
                        acc = self.num(Num(0), env, a.elem)
                        for t in range(k):
                            acc = self._arith('+', acc, self._arith('*', a.rows[i][t], b.rows[t][j], a.elem), a.elem)
                        row.append(acc)
                    rows.append(row)
                return Mat(rows, a.elem)
            raise Unsupported(f'operator {op} on matrices')
        raise Unsupported(f'operator {op} between a vector and a non-vector')

    def vector_literal(self, node: VecLit, env, typ):
        if _base(typ) == 'Vec':
            elem, dim, space = _vec_info(typ)
            n = _resolve_dim(dim, env) if dim != '?' else None
            if n is not None and n != len(node.items):
                raise Unsupported(f'vector literal of {len(node.items)} entries where Fin {dim} is expected')
        else:
            elem, space = _vec_info(self.arith_type(node, env))[0], 'pi'
        return Vec([self.num(x, env, elem) for x in node.items], elem, space)

    def matrix_literal(self, node: MatLit, env, typ):
        if _base(typ) == 'Mat':
            elem, m_, n_ = _mat_info(typ)
            if self._dim(m_, env) != len(node.rows) or (node.rows and self._dim(n_, env) != len(node.rows[0])):
                raise Unsupported('matrix literal of a different shape than its type')
        else:
            elem = _mat_info(self.arith_type(node, env))[0]
        return Mat([[self.num(x, env, elem) for x in r] for r in node.rows], elem)

    def vector_of_lambda(self, node: Lambda, env, typ):
        """`fun i => e` where a vector `Fin n → α` is expected: the table of e over i = 0 … n-1."""
        if len(node.params) != 1:
            raise Unsupported('a vector given as a lambda of several parameters')
        elem, dim, space = _vec_info(typ)
        n = self._dim(dim, env)
        fin = f'Fin:{dim}'
        return Vec([self.num(node.body, {**env, node.params[0]: Bound(k, fin)}, elem) for k in range(n)], elem, space)

    def fix_unknown_dimension(self, typ, sides, env):
        """A vector type whose dimension the operands did not fix ('Vec:Real:?', a sum of `Pi.single`s against a
        lambda) takes the dimension of the index a big operator on either side ranges over."""
        if _base(typ) != 'Vec' or '?' not in typ:
            return typ
        elem, _, space = _vec_info(typ)
        for side in sides:
            if isinstance(side, BigOp):
                try:
                    idx, _ = self.finite_index_domain(side.var, side.type, side.domain, side.body, env)
                except Unsupported:
                    continue
                return f'Vec:{elem}:{len(idx)}' + (':euclid' if space == 'euclid' else '')
        return typ

    def index_dimension(self, var: str, body, env):
        """The dimension of the vector (or the rows / columns of the matrix) that ``var`` indexes in ``body``, when
        exactly one such application fixes it: `∑ i, |j i|` sums over `Fin 6` when `j : Fin 6 → ℝ`."""
        found = set()

        def walk(node):
            if isinstance(node, App) and isinstance(node.fn, Var):
                positions = [k for k, a in enumerate(node.args) if isinstance(a, Var) and a.name == var]
                if positions:
                    fn = node.fn.name
                    if fn in env and isinstance(env[fn], Bound):
                        bt, bv = env[fn].type, env[fn].value
                        if _base(bt) == 'Vec' and positions == [0]:
                            found.add(str(len(bv)) if isinstance(bv, Vec) else _vec_info(bt)[1])
                        elif _base(bt) == 'Mat':
                            _, rows, cols = _mat_info(bt)
                            for pos in positions:
                                if isinstance(bv, Mat):
                                    found.add(str(bv.shape[0] if pos == 0 else bv.shape[1]))
                                else:
                                    found.add(rows if pos == 0 else cols)
                    else:
                        d = self.defn(fn)
                        if d is not None and d.result_type:
                            rt = self._type_name(d.result_type)
                            extra = [pos - len(d.params) for pos in positions if pos >= len(d.params)]
                            if _base(rt) == 'Vec' and extra == [0]:
                                found.add(_vec_info(rt)[1])
                            elif _base(rt) == 'Mat' and extra:
                                _, rows, cols = _mat_info(rt)
                                for pos in extra:
                                    found.add(rows if pos == 0 else cols)
            if dataclasses.is_dataclass(node):
                for f in dataclasses.fields(node):
                    walk(getattr(node, f.name))
            elif isinstance(node, (list, tuple)):
                for x in node:
                    walk(x)
        walk(body)
        dims = {self._dim(t, env) for t in found}
        if len(dims) == 1:
            return dims.pop()
        if len(dims) > 1:
            raise Unsupported(f'{var} indexes vectors of different dimensions {sorted(dims)}')
        return None

    def finite_index_domain(self, var, type_text, domain_node, body, env):
        """Indices of a big operator or a quantifier and the type the index is bound at: `Fin n` from the binder
        type, a `Finset.range k` domain (a ℕ index) or `Finset.univ`, or the vectors the variable indexes."""
        if type_text is not None:
            kind = _kind_of_type(type_text, self.module.point_types, self.module.aliases)
            if kind is not None and _base(kind) == 'Fin':
                return list(range(self._dim(kind.split(':')[1], env))), kind
            if kind == 'Bool':
                raise Unsupported('sum over Bool')
            raise Unsupported(f'big operator over {type_text}')
        if domain_node is not None:
            if isinstance(domain_node, Var) and domain_node.name in ('Finset.univ', 'univ'):
                n = self.index_dimension(var, body, env)
                if n is None:
                    raise Unsupported(f'{var} ranges over Finset.univ of an unknown type')
                return list(range(n)), f'Fin:{n}'
            dom = self.num(domain_node, env, 'Nat')
            if isinstance(dom, list):
                return dom, 'Nat'
            raise Unsupported('big operator over a non-finite domain')
        n = self.index_dimension(var, body, env)
        if n is None:
            raise Unsupported(f'unbounded big operator over {var}')
        return list(range(n)), f'Fin:{n}'

    def bigop(self, node: BigOp, env, typ):
        """`∑ i, e` / `∏ i, e` over a finite index set, exactly, at the context's type (a vector-valued sum adds
        vectors). The empty sum is 0 and the empty product 1 (the constant vector at a vector type)."""
        idx, index_type = self.finite_index_domain(node.var, node.type, node.domain, node.body, env)
        if len(idx) > _MAX_LOOP:
            raise Infeasible(f'big operator over {len(idx)} indices exceeds the loop cap')
        if _base(typ) == 'Fin':
            typ = 'Nat'
        acc = self.num(Num(0 if node.op == '∑' else 1), env, typ)
        for k in idx:
            term = self.num(node.body, {**env, node.var: Bound(k, index_type)}, typ)
            if isinstance(acc, Vec) or isinstance(acc, Mat):
                if node.op == '∑':
                    acc = self._vm_arith('+', acc, term)
                else:
                    raise Unsupported('product of vectors')
            else:
                acc = self._arith('+' if node.op == '∑' else '*', acc, term, typ)
        return acc

    def _vm_arith(self, op, a, b):
        if isinstance(a, Vec) and isinstance(b, Vec) and len(a) == len(b):
            return Vec([self._arith(op, x, y, a.elem) for x, y in zip(a.entries, b.entries)], a.elem, a.space)
        if isinstance(a, Mat) and isinstance(b, Mat) and a.shape == b.shape:
            return Mat([[self._arith(op, x, y, a.elem) for x, y in zip(ra, rb)] for ra, rb in zip(a.rows, b.rows)], a.elem)
        raise Unsupported(f'operator {op} on vectors of different shapes')

    def norm(self, node: Norm, env):
        """`‖v‖`: the sup norm on the Pi type `Fin n → ℝ`, the Euclidean norm on `EuclideanSpace ℝ (Fin n)`
        (exact in ℚ(√d)), the absolute value on a scalar."""
        t = self.arith_type(node.expr, env)
        if _base(t) == 'Vec':
            v = self.num(node.expr, env, t)
            if not isinstance(v, Vec):
                raise Unsupported('norm of a non-vector')
            if v.space == 'euclid':
                total = Fraction(0)
                for x in v.entries:
                    total = total + _exact(x) * _exact(x)
                return self._square_root(total)
            out = Fraction(0)
            for x in v.entries:
                if isinstance(x, (ExpPoly, Iv)):
                    raise Unsupported('sup norm of a vector with transcendental entries')
                ax = _abs(_exact(x))
                if ax > out:                  # Alg orders exactly against Fractions and other Alg values
                    out = ax
            return out
        if _base(t) == 'Mat':
            raise Unsupported('matrix norm')
        if _base(t) not in ('Nat', 'Int', 'Rat', 'Real', 'Fin', 'Sci'):
            raise Unsupported(f'norm of a {_base(t)} value')
        v = self.num(node.expr, env, 'Real' if _base(t) in ('Nat', 'Int', 'Fin', 'Sci') else t)
        if isinstance(v, (list, tuple, bool, Vec, Mat, ENN)):
            raise Unsupported('norm of a non-numeric value')
        return self._absolute(v)

    def index_vector(self, v, index_node, env, typ):
        i = self.num(index_node, env, 'Nat')
        if isinstance(i, bool) or not isinstance(i, int):
            raise Unsupported('a non-integer index')
        if isinstance(v, Vec):
            if not 0 <= i < len(v):
                raise Unsupported(f'index {i} outside Fin {len(v)}')
            return _coerce(v.entries[i], v.elem, typ)
        if isinstance(v, Mat):
            if not 0 <= i < v.shape[0]:
                raise Unsupported(f'row index {i} outside Fin {v.shape[0]}')
            return Vec(v.rows[i], v.elem)
        raise Unsupported('indexing a non-vector')

    def determinant(self, m: Mat):
        rows, cols = m.shape
        if rows != cols:
            raise Unsupported('determinant of a non-square matrix')
        if rows > 6:
            raise Infeasible(f'determinant of a {rows}×{rows} matrix exceeds the expansion cap')
        elem = m.elem

        def det(a):
            n = len(a)
            if n == 0:
                return self.num(Num(1), {}, elem)
            if n == 1:
                return a[0][0]
            total = None
            for j in range(n):
                minor = [r[:j] + r[j + 1:] for r in a[1:]]
                term = self._arith('*', a[0][j], det(minor), elem)
                if j % 2:
                    term = -term if elem != 'Nat' else term
                    if elem == 'Nat':
                        raise Unsupported('determinant over ℕ')
                total = term if total is None else self._arith('+', total, term, elem)
            return total
        return det([list(r) for r in m.rows])

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
        if isinstance(v, Iv):
            if v.magnitude_bits() > _MAX_RESULT_BITS or _bits(v.lo) > 2 * _MAX_RESULT_BITS or _bits(v.hi) > 2 * _MAX_RESULT_BITS:
                raise Infeasible('an enclosure exceeds the result-size cap')
            return v
        if isinstance(v, ExpPoly):
            return v
        if isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, float):
                    raise AssertionError('floating point value produced')  # pragma: no cover - A3 guard
        if isinstance(v, Vec):
            for x in v.entries:
                if isinstance(x, float):
                    raise AssertionError('floating point value produced')  # pragma: no cover - A3 guard
        if isinstance(v, Mat):
            for r in v.rows:
                for x in r:
                    if isinstance(x, float):
                        raise AssertionError('floating point value produced')  # pragma: no cover - A3 guard
        return v

    def _arith(self, op, a, b, typ):
        if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
            raise Unsupported('arithmetic on a non-numeric value')
        if isinstance(a, (ExpPoly, Iv)) or isinstance(b, (ExpPoly, Iv)):
            return self._trans_arith(op, a, b)
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

    # ---- transcendental values: exact exponential polynomials where possible, rigorous enclosures otherwise
    def _trans_arith(self, op, a, b):
        """Arithmetic with an exponential polynomial or an enclosure on either side (always real: Lean's
        `Real.exp` etc. live in ℝ, so division is field division and `x / 0 = 0` applies to an exact zero)."""
        if isinstance(a, (Meas, Func, frozenset, Vec, Mat, ENN)) or isinstance(b, (Meas, Func, frozenset, Vec, Mat, ENN)):
            raise Unsupported(f'operator {op} between a transcendental value and a non-numeric one')
        if op == '%':
            raise Unsupported('modulus of a transcendental value')
        exact = not isinstance(a, (Iv, Alg)) and not isinstance(b, (Iv, Alg))
        if exact:
            x, y = _ep_of(a), _ep_of(b)
            if op == '+':
                return _ep(x + y)
            if op == '-':
                return _ep(x - y)
            if op == '*':
                return _ep(x * y)
            if op == '/':
                if y.is_rational():
                    r = y.rational()
                    return Fraction(0) if r == 0 else _ep(x * ExpPoly({Fraction(0): 1 / r}))
                try:
                    return _ep(x * y.inverse())
                except Undecided:
                    pass            # a sum of exponentials in the divisor: fall through to enclosures
            else:
                raise Unsupported(f'operator {op} on exponential values')
        A, B = Iv.of(a, self.bits), Iv.of(b, self.bits)
        if op == '+':
            return (A + B).round(self.bits + _IV_GUARD_BITS)
        if op == '-':
            return (A - B).round(self.bits + _IV_GUARD_BITS)
        if op == '*':
            return (A * B).round(self.bits + _IV_GUARD_BITS)
        if op == '/':
            return (A / B).round(self.bits + _IV_GUARD_BITS)
        raise Unsupported(f'operator {op} on enclosures')

    def _inverse(self, v):
        """Lean's field inverse, totalized at an exact zero; exponential monomials exactly, sums by enclosure."""
        if isinstance(v, ExpPoly):
            try:
                return _ep(v.inverse())
            except Undecided:
                return Iv.of(v, self.bits).inverse()
        if isinstance(v, Iv):
            return v.inverse()
        return _inv(v)

    def _absolute(self, v):
        if isinstance(v, ExpPoly):
            sgn = _iv_compare('<', Iv.of(v, self.bits), Iv(0, 0))
            if sgn is None:
                raise Undecided(f'the sign of {v} is not decided at {self.bits} bits')
            return -v if sgn else v
        if isinstance(v, Iv):
            return v.abs()
        return _abs(v)

    def _minmax(self, want_min: bool, a, b):
        """`min` / `max`: exact when the order is decided (including between exponential polynomials); when two
        enclosures overlap, the hull of the pointwise minima / maxima, which contains the value whichever side wins."""
        if isinstance(a, (ExpPoly, Iv)) or isinstance(b, (ExpPoly, Iv)):
            if isinstance(a, (list, tuple, bool, Meas, Func, frozenset, Vec, Mat, ENN)) or \
                    isinstance(b, (list, tuple, bool, Meas, Func, frozenset, Vec, Mat, ENN)):
                raise Unsupported('min/max between a transcendental value and a non-numeric one')
            try:
                le = self._trans_compare('≤', _exact(a), _exact(b))
            except Undecided:
                A, B = Iv.of(a, self.bits), Iv.of(b, self.bits)
                return Iv(min(A.lo, B.lo), min(A.hi, B.hi)) if want_min else Iv(max(A.lo, B.lo), max(A.hi, B.hi))
            return (a if le else b) if want_min else (b if le else a)
        if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
            raise Unsupported('min/max of a non-numeric value')
        return min(a, b) if want_min else max(a, b)

    def _square_root(self, v):
        """`Real.sqrt`: exact in ℚ(√d) for a rational radicand; an enclosure for a transcendental radicand and for
        a nested radical (√ of an irrational algebraic value, where the exact field would have to grow)."""
        if isinstance(v, (ExpPoly, Iv)):
            return _sqrt_iv(Iv.of(v, self.bits), self.bits)
        if isinstance(v, Alg):
            return _sqrt_iv(Iv.of(v, self.bits), self.bits) if v.sign() > 0 else Fraction(0)
        return _real_sqrt(v)

    def _transcendental(self, name, args, env):
        """`Real.exp` of a rational is an exact exponential monomial (and `Real.sinh`/`cosh` exact exponential
        polynomials); `Real.log` of `e^q` is q exactly; everything else is a rigorous enclosure at the current
        precision. Mathlib's conventions: `Real.log 0 = 0`, `Real.log (-x) = Real.log x`."""
        if name == 'Real.rpow':
            raise Unsupported('real exponent (Real.rpow) is outside the exact backend')
        if name == 'Real.logb':
            if len(args) != 2:
                raise Unsupported(f'{name} applied to {len(args)} arguments')
            # Mathlib: `Real.logb b x = Real.log x / Real.log b`, with the totalised log and `x / 0 = 0`
            num_ = self._transcendental('Real.log', [args[1]], env)
            den_ = self._transcendental('Real.log', [args[0]], env)
            return self._trans_arith('/', num_, den_) if isinstance(num_, (ExpPoly, Iv)) or isinstance(den_, (ExpPoly, Iv)) \
                else self._arith('/', num_, den_, 'Real')
        if len(args) != 1:
            raise Unsupported(f'transcendental {name} applied to {len(args)} arguments')
        x = self.num(args[0], env, 'Real')
        if isinstance(x, (list, tuple, bool, Meas, Func, frozenset, Vec, Mat, ENN)):
            raise Unsupported(f'{name} of a non-numeric value')
        x = _ep(_exact(x)) if isinstance(x, ExpPoly) else _exact(x)
        if name == 'Real.exp':
            if isinstance(x, Fraction):
                return _ep(ExpPoly.exp(x))
            return _exp_iv(Iv.of(x, self.bits), self.bits)
        if name == 'Real.log':
            if isinstance(x, Fraction):
                if x == 0:
                    return Fraction(0)
                return _log_point(abs(x), self.bits)
            if isinstance(x, ExpPoly) and len(x.terms) == 1:
                (q, c), = x.terms.items()
                if c == 1:
                    return q                                   # log (e^q) = q, exactly
                if c == -1:
                    return q                                   # log |−e^q| = q
                base = _log_point(abs(c), self.bits)
                return base + Iv(q, q)
            return _log_iv(Iv.of(x, self.bits), self.bits)
        if name in ('Real.sin', 'Real.cos'):
            return _trig_iv(Iv.of(x, self.bits), self.bits, 'sin' if name == 'Real.sin' else 'cos')
        if name == 'Real.tan':
            X = Iv.of(x, self.bits)
            return _trig_iv(X, self.bits, 'sin') / _trig_iv(X, self.bits, 'cos')
        if name in ('Real.sinh', 'Real.cosh'):
            if isinstance(x, Fraction):
                e, einv = ExpPoly.exp(x), ExpPoly.exp(-x)
                return _ep((e - einv) * ExpPoly({Fraction(0): Fraction(1, 2)}) if name == 'Real.sinh'
                           else (e + einv) * ExpPoly({Fraction(0): Fraction(1, 2)}))
            X = Iv.of(x, self.bits)
            E, Einv = _exp_iv(X, self.bits), _exp_iv(-X, self.bits)
            return (E - Einv) * Fraction(1, 2) if name == 'Real.sinh' else (E + Einv) * Fraction(1, 2)
        if name == 'Real.arctan':
            X = Iv.of(x, self.bits)
            return self._arctan_iv(X)
        raise Unsupported(f'transcendental {name} is outside the exact algebraic backend')

    def _arctan_iv(self, X: Iv) -> Iv:
        """arctan on an interval (monotone): series for |x| ≤ 1, π/2 − arctan(1/x) beyond."""
        def point(x):
            x = _exact(x)
            if abs(x) <= 1:
                return _atan_series(x, self.bits)
            pi = _pi_enclosure(self.bits + 4)
            inner = _atan_series(1 / abs(x), self.bits)
            half = pi * Fraction(1, 2)
            v = half - inner
            return v if x > 0 else -v
        lo, hi = point(X.lo), point(X.hi)
        return Iv(lo.lo, hi.hi)

    def _trans_compare(self, op, a, b):
        """A comparison with an exponential polynomial or an enclosure on a side: exact when both sides are
        exponential polynomials (a difference that is rational, or `=`/`≠` by normal form), otherwise by
        enclosures at the current precision; undecided enclosures raise Undecided."""
        exact = not isinstance(a, (Iv, Alg)) and not isinstance(b, (Iv, Alg))
        if exact:
            d = _ep(_ep_of(a) - _ep_of(b))
            if op == '=':
                return d == 0
            if op == '≠':
                return d != 0
            if isinstance(d, Fraction):
                return {'<': d < 0, '>': d > 0, '≤': d <= 0, '≥': d >= 0}[op]
            res = _iv_compare(op, Iv.of(d, self.bits), Iv(0, 0))
            if res is None:
                raise Undecided(f'{op} between exponential values is not decided at {self.bits} bits (difference enclosed in {Iv.of(d, self.bits)})')
            return res
        A, B = Iv.of(a, self.bits), Iv.of(b, self.bits)
        res = _iv_compare(op, A, B)
        if res is None:
            hint = ''
            if op in ('=', '≠') and (A.width or B.width):
                hint = (' — an equality between transcendental values is confirmed only in the exponential normal '
                        'form (Real.exp of rationals under + − × and integer powers); enclosures can refute it, '
                        'never confirm it')
            raise Undecided(f'{op} is not decided at {self.bits} bits by the enclosures {A} and {B}{hint}')
        return res

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
        if isinstance(base, (ExpPoly, Iv)):
            if abs(exp) > _MAX_EXP_ARGUMENT:
                raise Infeasible(f'exponent {exp} on a transcendental value exceeds the cap')
            if isinstance(base, Iv):
                p = self.bits + _IV_GUARD_BITS + 2 * abs(exp).bit_length()
                powered = base.pow_rounded(abs(exp), p)
                return powered.inverse() if exp < 0 else powered
            return self._inverse(base ** (-exp)) if exp < 0 else _ep(base ** exp)
        if exp < 0:
            return _inv(_pow(base, -exp))
        return _pow(base, exp)

    def apply(self, node: App, env, typ):
        fn = node.fn
        args = node.args
        if isinstance(fn, Lambda):
            return self.call_closure(Closure(fn.params, fn.body, env, fn.types, self.namespace, self.opens),
                                     args, env, typ)
        if not isinstance(fn, Var):
            ft = self.arith_type(fn, env)
            if _base(ft) in ('Vec', 'Mat') and len(args) in (1, 2):
                value = self.num(fn, env, ft)
                for a in args:
                    value = self.index_vector(value, a, env, typ if a is args[-1] else ft)
                return value
            raise Unsupported('application of a non-name')
        name = fn.name
        if '.' in name and not name.startswith('.'):
            recv, method = name.rsplit('.', 1)
            if (recv in env or self.has_def(recv)) and method in _LIST_METHODS:
                return self.apply(App(Var('.' + method), [Var(recv)] + list(args)), env, typ)
        if name in env and isinstance(env[name], Closure):
            return self.call_closure(env[name], args, env, typ)
        if name in env and isinstance(env[name], Bound) and env[name].type == 'Measure' and len(args) == 1:
            return self.measure_apply(env[name].value, args[0], env)
        if name in env and isinstance(env[name], Bound) and env[name].type == 'Func' and len(args) == 1:
            return self.func_apply(env[name].value, args[0], env)
        if name in env and isinstance(env[name], Bound) and _base(env[name].type) == 'Vec' and len(args) == 1:
            return self.index_vector(env[name].value, args[0], env, typ)
        if name in env and isinstance(env[name], Bound) and _base(env[name].type) == 'Mat' and len(args) in (1, 2):
            row = self.index_vector(env[name].value, args[0], env, typ)
            return row if len(args) == 1 else self.index_vector(row, args[1], env, typ)
        if name in ('.det', 'Matrix.det') and len(args) == 1:
            mt = self.arith_type(args[0], env)
            m = self.num(args[0], env, mt if _base(mt) == 'Mat' else 'Mat:Real:?:?')
            if not isinstance(m, Mat):
                raise Unsupported('determinant of a non-matrix')
            return _coerce(self.determinant(m), m.elem, typ)
        if name in ('.trace', 'Matrix.trace') and len(args) == 1:
            mt = self.arith_type(args[0], env)
            m = self.num(args[0], env, mt)
            if not isinstance(m, Mat) or m.shape[0] != m.shape[1]:
                raise Unsupported('trace of a non-square matrix')
            acc = self.num(Num(0), env, m.elem)
            for i in range(m.shape[0]):
                acc = self._arith('+', acc, m.rows[i][i], m.elem)
            return _coerce(acc, m.elem, typ)
        if name in ('Pi.single', 'EuclideanSpace.single') and len(args) == 2:
            if _base(typ) != 'Vec':
                raise Unsupported(f'{name} outside a vector context')
            elem, dim, space = _vec_info(typ)
            n = self._dim(dim, env)
            i = self.num(args[0], env, 'Nat')
            if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < n:
                raise Unsupported(f'{name} at an index outside Fin {n}')
            zero = self.num(Num(0), env, elem)
            x = self.num(args[1], env, elem)
            return Vec([x if k == i else zero for k in range(n)], elem, space)
        if name in ('dotProduct', 'Matrix.dotProduct', 'inner') and len(args) in (2, 3):
            va, vb = args[-2], args[-1]
            ta, tb = self.arith_type(va, env), self.arith_type(vb, env)
            vt = _max_type(ta, tb) if _base(ta) == 'Vec' or _base(tb) == 'Vec' else None
            if vt is None or _base(vt) != 'Vec':
                raise Unsupported(f'{name} of non-vectors')
            a, b = self.num(va, env, vt), self.num(vb, env, vt)
            if not isinstance(a, Vec) or not isinstance(b, Vec) or len(a) != len(b):
                raise Unsupported(f'{name} of vectors of different dimensions')
            acc = self.num(Num(0), env, a.elem)
            for x, y in zip(a.entries, b.entries):
                acc = self._arith('+', acc, self._arith('*', x, y, a.elem), a.elem)
            return _coerce(acc, a.elem, typ)
        if name == 'Finset.range' and len(args) == 1:
            n = self.num(args[0], env, 'Nat')
            if isinstance(n, bool) or not isinstance(n, int):
                raise Unsupported('Finset.range of a non-natural')
            if n > _MAX_LOOP:
                raise Infeasible(f'Finset.range {n} exceeds the loop cap')
            return list(range(n))
        if name in _TRANSCENDENTAL:
            return self._transcendental(name, args, env)
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
            return self._square_root(self.num(args[0], env, 'Real'))
        if name == 'abs' and len(args) == 1:
            t = _max_type(self.arith_type(args[0], env), typ) if typ != 'Nat' else self.arith_type(args[0], env)
            v = self.num(args[0], env, t)
            if isinstance(v, (list, tuple, bool)):
                raise Unsupported('abs of a non-numeric value')
            return self._absolute(v)
        if name == 'Inv.inv' and len(args) == 1:
            if typ == 'ENNReal':
                return _enn(self.num(args[0], env, 'ENNReal')).inverse()
            if typ not in _FIELD_TYPES:
                raise Unsupported(f'inverse at type {typ}')
            return self._inverse(self.num(args[0], env, typ))
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
        if name in ('Nat.min', 'min', 'Nat.max', 'max') and len(args) == 2:
            return self._minmax(name.endswith('min'), self.num(args[0], env, typ), self.num(args[1], env, typ))
        if name == 'Int.natAbs' and len(args) == 1:
            return abs(self.num(args[0], env, 'Int'))
        if name == 'Int.toNat' and len(args) == 1:
            return max(0, self.num(args[0], env, 'Int'))
        if name == 'Int.ofNat' and len(args) == 1:
            return self.num(args[0], env, 'Nat')
        d = self.defn(name)
        if d is None:
            raise Unsupported(f'unknown function {name}')
        dtyp = self._type_name(d.result_type) if d.result_type else typ
        if dtyp == 'Measure' and not d.clauses and len(args) == len(d.params) + 1:
            measure = self.apply(App(fn, list(args[:-1])), env, 'Measure')
            return self.measure_apply(measure, args[-1], env)
        if _base(dtyp) in ('Vec', 'Mat') and len(args) > len(d.params):
            value = self.apply(App(fn, list(args[:len(d.params)])), env, dtyp)
            for a in args[len(d.params):]:
                value = self.index_vector(value, a, env, typ if a is args[-1] else dtyp)
            return value
        if not d.clauses and d.result_type and _base(dtyp) not in ('Vec', 'Mat') and len(_arrow_parts(d.result_type)) > 1:
            # `def f (L : ℝ) : ℝ → ℝ := fun x => …`: the result is a function; extra arguments are applied to it
            arity = len(_arrow_parts(d.result_type)) - 1
            if len(args) == len(d.params):
                raise Unsupported(f'{name} is a function value the interpreter does not carry')
            if len(args) == len(d.params) + arity:
                if d.body is None:
                    raise Unsupported(f'definition {name} has an unparsed body')
                local = self.bind_parameters(d, args[:len(d.params)], env)
                fn_value = d.body
                if not isinstance(fn_value, Lambda):
                    raise Unsupported(f'{name} returns a function that is not a lambda')
                arg_types = _arrow_parts(d.result_type)[:-1]
                if len(fn_value.params) != arity:
                    raise Unsupported(f'{name} returns a lambda of {len(fn_value.params)} parameters for an arrow type of {arity}')
                # the lambda's parameters take the domain types (an explicit binder type wins), the body the codomain
                types = [lt if lt is not None else at for lt, at in zip(fn_value.types or [None] * arity, arg_types)]
                vals = [self.num(a, env, self._type_name(at)) for a, at in zip(args[len(d.params):], types)]
                return self.call_values(Closure(fn_value.params, fn_value.body, local, types,
                                                d.namespace, d.opens), vals, dtyp)
            raise Unsupported(f'{name} applied to {len(args)} arguments')
        if d.clauses:
            n_lead = len(d.params)
            n_pats = len(d.clauses[0][0])
            if len(args) != n_lead + n_pats:
                raise Unsupported(f'{name} applied to {len(args)} of {n_lead + n_pats} arguments')
            local = self.bind_parameters(d, args[:n_lead], env)
            arrow_types = _arrow_parts(d.result_type)[:-1] if d.result_type else []
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
                with self.declaration_scope(d.namespace, d.opens):
                    return self.num(expr, scope, dtyp)
            raise Unsupported(f'no clause of {name} matches {[v for v, _ in vals]}')
        if d.body is None:
            raise Unsupported(f'definition {name} has an unparsed body')
        if len(args) != len(d.params):
            raise Unsupported(f'{name} applied to {len(args)} of {len(d.params)} arguments')
        local = self.bind_parameters(d, args, env)
        with self.declaration_scope(d.namespace, d.opens):
            return self.num(d.body, local, dtyp)

    def bind_parameters(self, d: Def, args, env) -> dict:
        """The local environment of a definition applied to ``args``: explicit parameters at their declared types,
        then every implicit `{n : ℕ}` that is the dimension of a vector / index / matrix parameter, read off the
        argument's actual size (`def gm8Radius {n : ℕ} (j : Fin n → ℝ)` applied to a `Fin 6 → ℝ` binds n = 6).
        An implicit the arguments do not fix stays unbound: a body that mentions it is UNSUPPORTED."""
        local = {}
        for (pname, ptype), a in zip(d.params, args):
            pt = self._type_name(ptype)
            local[pname] = Bound(self.num(a, env, pt), pt)
        for iname, itype in d.implicits:
            if itype.strip() not in _NAT_TYPES:
                continue
            for pname, ptype in d.params:
                pt = local[pname].type
                v = local[pname].value
                dims = _dims_of(pt)
                if iname not in dims:
                    continue
                if isinstance(v, Vec) and _base(pt) == 'Vec':
                    local[iname] = Bound(len(v), 'Nat')
                elif isinstance(v, Mat) and _base(pt) == 'Mat':
                    rows, cols = _mat_info(pt)[1:]
                    local[iname] = Bound(v.shape[0] if rows == iname else v.shape[1], 'Nat')
                if iname in local:
                    break
        return local

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
        with self.declaration_scope(closure.namespace, closure.opens):
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
            return self.func_apply(self.materialize(Closure(node.params, node.body, env, node.types,
                                                            self.namespace, self.opens)), Num(i), env)
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
            # true only for values the finite model carries: a table, a subset, a lambda on the carrier, a model
            # measure; exponents such as the 2 of `MemLp f 2 μ` are numerals. Anything else (a function on ℝ,
            # `gaussianReal 0 1`, a structure) is outside the model and the predicate is not decided.
            measures, carried = [], False
            for a in node.args:
                if isinstance(a, Lambda):
                    carried = True
                    continue
                t = self.arith_type(a, env)
                if t in ('Measure', 'Set', 'Func'):
                    carried = True
                    if t == 'Measure':
                        measures.append(self.num(a, env, 'Measure'))
                    elif t == 'Func':
                        if not isinstance(self.num(a, env, 'Func'), Func):
                            raise Unsupported(f'{name} of a function value the model does not carry')
                    continue
                if isinstance(a, Num) or (isinstance(a, Var) and a.name in env and env[a.name].type in ('Nat', 'Int', 'Rat', 'Real', 'ENNReal')):
                    continue
                raise Unsupported(f'{name} of a value outside the finite model ({t})')
            if not carried:
                raise Unsupported(f'{name} with no model-carried argument')
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
            return Closure(f.params, f.body, env, f.types, self.namespace, self.opens)
        if isinstance(f, Var) and f.name in _BUILTIN_BINARY:
            return _BUILTIN_BINARY[f.name]
        if isinstance(f, Var) and f.name in env and isinstance(env[f.name], Closure):
            return env[f.name]
        if isinstance(f, Var) and self.has_def(f.name):
            d = self.defn(f.name)
            if d.body is not None and d.params:
                return Closure([p for p, _ in d.params], d.body, {}, [pt for _, pt in d.params],
                               d.namespace, d.opens)
        raise Unsupported('higher-order argument is not a lambda')

    def call_closure(self, closure, arg_nodes, env, typ):
        types = self._param_types(closure, len(arg_nodes), typ)
        vals = [self.num(a, env, t) for a, t in zip(arg_nodes, types)]
        return self.call_values(closure, vals, typ)

    def _param_types(self, closure, n, typ):
        """The type each parameter is bound at: its declared type when the closure carries one, else ``typ``."""
        if callable(closure) and not isinstance(closure, Closure):
            return [typ] * n
        return [self._type_name(t) if t is not None else typ for t in closure.types[:n]] + [typ] * (n - len(closure.types))

    def call_values(self, closure, vals, typ):
        """Apply ``closure`` to evaluated ``vals`` and evaluate its body at ``typ``. Parameters are bound at their
        own types (Codex 752/4471516437): `def f : ℕ → ℝ := fun n => …` binds `n : ℕ`, so `n - 1` truncates."""
        if callable(closure) and not isinstance(closure, Closure):
            return closure(*vals)
        if len(vals) != len(closure.params):
            raise Unsupported('lambda arity mismatch')
        local = dict(closure.env)
        for p, v, t in zip(closure.params, vals, self._param_types(closure, len(vals), typ)):
            local[p] = Bound(v, t)
        with self.declaration_scope(closure.namespace, closure.opens):
            return self.num(closure.body, local, typ)

    # ---- bounded universal quantification
    def forall(self, node: 'Forall', env) -> bool:
        var_type = 'Int' if node.type in _INT_TYPES else 'Nat'
        if node.type is not None and node.type not in _INT_TYPES | _NAT_TYPES:
            kind = _kind_of_type(node.type, self.module.point_types, self.module.aliases)
            if kind == 'Set':
                # every subset of the finite carrier: 2^n instances
                pts = list(self.points())
                subsets = [frozenset(i for i in pts if mask >> i & 1) for mask in range(1 << len(pts))]
                return self._forall_over(node, env, subsets, 'Set')
            if kind == 'Point':
                return self._forall_over(node, env, list(self.points()), 'Point')
            if kind is not None and _base(kind) == 'Fin':
                n = self._dim(kind.split(':')[1], env)
                return self._forall_over(node, env, list(range(n)), kind)
            raise Unsupported(f'quantifier over {node.type}')
        domain, body = _domain_of(node, self, env)
        if domain is None and node.type is None and node.domain is None:
            n = self.index_dimension(node.var, node.body, env)
            if n is not None:
                return self._forall_over(node, env, list(range(n)), f'Fin:{n}')
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
    def connective(self, node: BinOp, env) -> bool:
        """`∧`, `∨`, `→` with Lean's short-circuit reading from the left. A side whose comparison the current
        enclosure precision leaves undecided does not decide the connective by itself: the other side is still
        evaluated, and settles it when it can (`A ∨ B` is true when B is true, `A ∧ B` false when B is false,
        `A → B` true when B is true, whatever A is); otherwise the undecided comparison is re-raised, so the
        statement is refined or reported, never guessed. Unsupported constructs propagate unchanged."""
        try:
            left = self.prop(node.left, env)
        except Undecided as undecided:
            right = self.prop(node.right, env)
            if (node.op == '∨' and right) or (node.op == '→' and right):
                return True
            if node.op == '∧' and not right:
                return False
            raise undecided
        if node.op == '∧':
            return left and self.prop(node.right, env)
        if node.op == '∨':
            return left or self.prop(node.right, env)
        return (not left) or self.prop(node.right, env)

    def prop(self, node, env) -> bool:
        if isinstance(node, Not):
            return not self.prop(node.expr, env)
        if isinstance(node, Forall):
            return self.forall(node, env)
        if isinstance(node, AERel):
            return self.ae_rel(node, env)
        if isinstance(node, App) and isinstance(node.fn, Var) and node.fn.name in _MEASURE_PREDICATES:
            return self.predicate(node, env)
        if isinstance(node, Var) and node.name in ('True', 'False'):
            return node.name == 'True'
        if isinstance(node, (BoolNot, Var)) or (isinstance(node, BinOp) and node.op in ('&&', '||')):
            if isinstance(node, Var) and node.name not in ('true', 'false'):
                v = self.num(node, env, 'Nat')
                if isinstance(v, bool):
                    return v
                raise Unsupported(f'proposition {node.name} is not Bool-valued')
            return self.truth(node, env)
        if isinstance(node, BinOp):
            if node.op in ('∧', '∨', '→'):
                return self.connective(node, env)
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
                lt, rt = self.arith_type(node.left, env), self.arith_type(node.right, env)
                typ = _max_type(lt, rt)
                # a `Fin n` index against numerals only: the numerals are `Fin n` literals (`i = 2`, `i ≠ 7 - 5`),
                # which wrap; against a ℕ term Lean coerces the index (`↑i < n`)
                if _base(lt) == 'Fin' and rt == 'Nat' and _literal_only(node.right):
                    typ = lt
                elif _base(rt) == 'Fin' and lt == 'Nat' and _literal_only(node.left):
                    typ = rt
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
                if _base(typ) in ('Vec', 'Mat'):
                    typ = self.fix_unknown_dimension(typ, (node.left, node.right), env)
                    a = self.num(node.left, env, typ)
                    b = self.num(node.right, env, typ)
                    if node.op == '=':
                        return a == b
                    if node.op == '≠':
                        return a != b
                    if isinstance(a, Vec) and isinstance(b, Vec) and a.space == 'pi' and b.space == 'pi' and len(a) == len(b):
                        # the Pi order: v ≤ w ↔ ∀ i, v i ≤ w i; v < w ↔ v ≤ w ∧ ¬ w ≤ v
                        le = lambda x, y: all(_exact(p) <= _exact(q) if a.elem in _FIELD_TYPES else p <= q
                                              for p, q in zip(x.entries, y.entries))
                        return {'≤': le(a, b), '≥': le(b, a), '<': le(a, b) and not le(b, a),
                                '>': le(b, a) and not le(a, b)}[node.op]
                    raise Unsupported(f'ordering of {_base(typ)} values' + (' on EuclideanSpace' if isinstance(a, Vec) else ''))
                a = self.num(node.left, env, typ)
                b = self.num(node.right, env, typ)
                if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
                    if node.op == '=':
                        return a == b
                    if node.op == '≠':
                        return a != b
                    raise Unsupported('ordering of non-numeric values')
                if isinstance(a, (ExpPoly, Iv)) or isinstance(b, (ExpPoly, Iv)):
                    return self._trans_compare(node.op, _exact(a), _exact(b))
                if typ in _FIELD_TYPES:
                    a, b = _exact(a), _exact(b)       # an int meets a Fraction or an Alg: compare exactly
                return {'=': a == b, '≠': a != b, '<': a < b, '>': a > b, '≤': a <= b, '≥': a >= b}[node.op]
            raise Unsupported(f'operator {node.op} in a proposition')
        if isinstance(node, Var) and node.name in ('True', 'False'):
            return node.name == 'True'
        if isinstance(node, Ascribe):
            return self.prop(node.expr, env)
        if isinstance(node, App) and isinstance(node.fn, Var):
            d = self.defn(node.fn.name)
            if d is not None and d.result_type and d.result_type.strip() == 'Prop' and not d.clauses:
                # `def dm2Negative (H : …) : Prop := ∀ v, …`: the proposition is the body at the arguments
                if d.body is None:
                    raise Unsupported(f'definition {node.fn.name} has an unparsed body')
                if len(node.args) != len(d.params):
                    raise Unsupported(f'{node.fn.name} applied to {len(node.args)} of {len(d.params)} arguments')
                local = self.bind_parameters(d, node.args, env)
                with self.declaration_scope(d.namespace, d.opens):
                    return self.prop(d.body, local)
        raise Unsupported(f'proposition of form {type(node).__name__}')


def _ediv(a: int, b: int) -> int:
    """Lean 4 core `Int./` is `Int.ediv` (Euclidean): the quotient q with a = b*q + r and 0 ≤ r < |b|.
    For b > 0 this is floor division; for b < 0 it is -(a // -b). Python's `//` floors for every sign of
    b and T-division rounds toward zero, so both differ from this for some sign combination
    (goldens: 5 / -2 = -2, -5 / -2 = 3, -5 / 2 = -3; main#684/6511137449)."""
    return a // b if b > 0 else -(a // (-b))


_TYPE_ORDER = {'Sci': -2, 'Bool': -1, 'Fin': -0.5, 'Nat': 0, 'Int': 1, 'Rat': 2, 'Real': 3, 'ENNReal': 4, 'Point': 5,
               'Set': 6, 'Vec': 6.5, 'Mat': 6.7, 'Func': 7, 'Measure': 8, 'Function': 9, 'Prop': 10}


def _max_type(a, b):
    """The `binop%` maximal type of two operand types; vector / matrix types carry their dimension, and when
    both sides are vectors the one whose dimension is known wins ('Vec:Real:?' is a `Pi.single` without
    context)."""
    ba, bb = _base(a), _base(b)
    if 'Prop' in (ba, bb):
        raise Unsupported('arithmetic on propositions')
    if ba == bb and ba in ('Vec', 'Mat'):
        return b if '?' in a else a
    return a if _TYPE_ORDER[ba] >= _TYPE_ORDER[bb] else b


def _coerce(v, from_type, to_type):
    """Value of type ``from_type`` used where ``to_type`` is expected: ℕ/ℤ (and a `Fin n` index, `↑i`) embed into
    ℚ/ℝ; a rational or real value has no coercion back to ℕ/ℤ in Lean (the statement would not elaborate)."""
    if isinstance(v, (list, tuple, bool, Vec, Mat)):
        return v
    from_type, to_type = _base(from_type), _base(to_type)
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
    def enclosure(self, bits: int):
        """(lo, hi) rational bounds from integer square roots of each radicand at ``bits`` bits."""
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
        return lo, hi

    def sign(self) -> int:
        """Exact sign: the element is nonzero (canonical form), so an enclosure of each √d by integer square
        roots at increasing precision eventually separates the sum from 0."""
        for bits in (64, 128, 256, 512, 1024, 2048, 4096):
            lo, hi = self.enclosure(bits)
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


# --------------------------------------------------------------------------------------------
# Transcendental values: exact exponential polynomials and rigorous rational enclosures (Layer A: every
# endpoint is a Fraction, every remainder bound is a theorem of the series used; nothing is approximated
# without its error term, and a comparison the enclosure does not decide is reported, never guessed)
# --------------------------------------------------------------------------------------------

class Undecided(Unsupported):
    """A comparison the current enclosure precision does not decide. A closed statement is retried at higher
    precision and reported UNSUPPORTED when the cap is reached; a sampled instance is skipped and counted."""


_ENCLOSURE_BITS = (64, 128, 256, 512, 1024)      # refinement ladder for closed statements
_MAX_EXP_ARGUMENT = 1_000_000                     # |q| beyond this: e^q has more bits than the result caps allow (e^q ≈ 2^(1.443·q))
_MAX_EXP_TERMS = 20_000                           # distinct exponents an exponential-polynomial product may reach


def _fact(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


_IV_GUARD_BITS = 32           # significant bits kept beyond the working precision when an endpoint is rounded


def _round_out(x: Fraction, p: int, up: bool) -> Fraction:
    """x rounded to about p significant bits, toward −∞ (``up`` False) or +∞ (``up`` True): the result is a
    dyadic rational on the outer side of x, so an interval rounded this way still contains every value it did."""
    x = _exact(x)
    if x == 0:
        return x
    n, d = x.numerator, x.denominator
    s = p - (n.bit_length() - d.bit_length())        # |x|·2^s has about p bits
    if s >= 0:
        q, r = divmod(n << s, d)
    else:
        q, r = divmod(n, d << (-s))
    if up and r:
        q += 1                                       # floor → ceiling when the division was inexact
    return Fraction(q, 1 << s) if s >= 0 else Fraction(q << (-s))


def _magnitude_bits(x) -> int:
    """About log2 |x| for a rational x (0 for |x| < 1 and for 0)."""
    x = _exact(x)
    if not isinstance(x, Fraction) or x == 0:
        return 0
    return max(0, x.numerator.bit_length() - x.denominator.bit_length())


def _approx(x, digits: int = 18) -> str:
    """A decimal approximation of a rational for messages (never for arithmetic), safe for huge values: the
    integer that is printed never has more than ``digits`` + 1 digits, whatever the size of x."""
    x = _exact(x)
    if not isinstance(x, Fraction):
        return str(x)
    if x == 0:
        return '0'
    sign = '-' if x < 0 else ''
    a = abs(x)
    b = a.numerator.bit_length() - a.denominator.bit_length()
    if abs(b) > 100_000:                                        # astronomically large or small: the exponent is enough
        return f'{sign}2^{b}…' if b > 0 else f'{sign}2^{b - 1}…'
    # decimal exponent k with 10^k ≤ a < 10^(k+1): estimated from the bit lengths, then corrected
    k = int(b * 0.30103) - 1
    while a >= Fraction(10) ** (k + 1):
        k += 1
    while a < Fraction(10) ** k:
        k -= 1
    scaled = a * Fraction(10) ** (digits - 1 - k)            # floor(scaled) has exactly ``digits`` digits
    m = scaled.numerator // scaled.denominator
    s, exact = str(m), m == scaled
    if 0 <= k < digits:
        ip, fp = s[:k + 1], s[k + 1:].rstrip('0')
        out = ip + ('.' + fp if fp else '')
    elif -4 <= k < 0:
        out = '0.' + '0' * (-k - 1) + s.rstrip('0')
    else:
        fp = s[1:].rstrip('0')
        out = s[0] + ('.' + fp if fp else '') + f'e{k}'
    return sign + out + ('' if exact else '…')


class Iv:
    """A closed interval [lo, hi] with exact rational endpoints: every operation is an exact hull, so a value
    inside the operands' intervals has its image inside the result. ``round`` moves the endpoints outward to a
    bounded number of significant bits (containment is kept, only the width grows), which keeps the rational
    endpoints small across long computations such as e^n for large n."""
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi):
        lo, hi = _exact(lo), _exact(hi)
        if lo > hi:
            lo, hi = hi, lo
        self.lo, self.hi = lo, hi

    @staticmethod
    def of(v, bits: int) -> 'Iv':
        if isinstance(v, Iv):
            return v
        if isinstance(v, ExpPoly):
            return v.enclosure(bits)
        if isinstance(v, Alg):
            lo, hi = v.enclosure(bits)
            return Iv(lo, hi)
        if isinstance(v, bool) or not isinstance(v, (int, Fraction)):
            raise Unsupported('enclosure of a non-numeric value')
        return Iv(v, v)

    @property
    def width(self):
        return self.hi - self.lo

    def round(self, p: int) -> 'Iv':
        """Outward rounding of both endpoints to about p significant bits (a superset of self)."""
        return Iv(_round_out(self.lo, p, False), _round_out(self.hi, p, True))

    def __neg__(self):
        return Iv(-self.hi, -self.lo)

    def __add__(self, o):
        o = Iv.of(o, 64) if not isinstance(o, Iv) else o
        return Iv(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __sub__(self, o):
        o = Iv.of(o, 64) if not isinstance(o, Iv) else o
        return Iv(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        o = Iv.of(o, 64) if not isinstance(o, Iv) else o
        if self.lo >= 0 and o.lo >= 0:                       # the common case needs no comparisons
            return Iv(self.lo * o.lo, self.hi * o.hi)
        if self.hi <= 0 and o.hi <= 0:
            return Iv(self.hi * o.hi, self.lo * o.lo)
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        return Iv(min(c), max(c))

    __rmul__ = __mul__

    def inverse(self):
        if self.lo == 0 == self.hi:
            return Iv(0, 0)                    # Lean's totalised inverse: 0⁻¹ = 0, x / 0 = 0 (an exact zero only)
        if self.lo <= 0 <= self.hi:
            raise Undecided(f'the divisor\'s enclosure {self} contains 0 (Lean: x / 0 = 0)')
        return Iv(1 / self.hi, 1 / self.lo)

    def __truediv__(self, o):
        o = Iv.of(o, 64) if not isinstance(o, Iv) else o
        return self * o.inverse()

    def __rtruediv__(self, o):
        return Iv.of(o, 64) * self.inverse()

    def __pow__(self, n: int):
        if n == 0:
            return Iv(1, 1)
        if n < 0:
            return (self ** (-n)).inverse()
        if n % 2 == 0 and self.lo < 0 < self.hi:
            m = max(-self.lo, self.hi)
            return Iv(0, m ** n)
        a, b = self.lo ** n, self.hi ** n
        return Iv(min(a, b), max(a, b))

    def magnitude_bits(self) -> int:
        """About log2 of the largest absolute endpoint (0 for an interval inside (-1, 1))."""
        return max(_magnitude_bits(self.lo), _magnitude_bits(self.hi))

    def pow_rounded(self, n: int, p: int) -> 'Iv':
        """self^n for n ≥ 0 by binary exponentiation, rounding outward to p significant bits after each product
        (so the mantissas stay about p bits long however large n is); every intermediate is a superset. The
        magnitude of the result is capped like every exact result (an endpoint beyond 2^(2·10⁶) is Infeasible)."""
        if n * (self.magnitude_bits() + 1) > _MAX_RESULT_BITS:
            raise Infeasible(f'exponent {n} on an enclosure of magnitude 2^{self.magnitude_bits()} exceeds the result cap')
        out, base = Iv(1, 1), self.round(p)
        while n:
            if n & 1:
                out = (out * base).round(p)
            n >>= 1
            if n:
                base = (base * base).round(p)
        return out

    def abs(self):
        if self.lo >= 0:
            return self
        if self.hi <= 0:
            return -self
        return Iv(0, max(-self.lo, self.hi))

    def __repr__(self):
        return f'[{_approx(self.lo)}, {_approx(self.hi)}]'


class ExpPoly:
    """Σ c · e^q over distinct rational exponents q with nonzero rational coefficients c: the exact normal form
    of the values `Real.exp` builds from rational arguments under + - * and integer powers. Equality of normal
    forms is equality of values (the e^q for distinct rational q are linearly independent over ℚ —
    Lindemann–Weierstrass), so `exp 2 = exp 1 ^ 2` is decided exactly; an ordering is decided by enclosure."""
    __slots__ = ('terms',)

    def __init__(self, terms: dict):
        self.terms = {q: c for q, c in terms.items() if c != 0}

    @staticmethod
    def exp(q) -> 'ExpPoly':
        return ExpPoly({Fraction(q): Fraction(1)})

    def is_rational(self):
        return all(q == 0 for q in self.terms)

    def rational(self):
        return self.terms.get(Fraction(0), Fraction(0))

    def __eq__(self, o):
        if isinstance(o, (int, Fraction)) and not isinstance(o, bool):
            return self.is_rational() and self.rational() == o
        return isinstance(o, ExpPoly) and self.terms == o.terms

    def __hash__(self):
        return hash(frozenset(self.terms.items()))

    def __neg__(self):
        return ExpPoly({q: -c for q, c in self.terms.items()})

    def __add__(self, o):
        o = _ep_of(o)
        t = dict(self.terms)
        for q, c in o.terms.items():
            t[q] = t.get(q, Fraction(0)) + c
        return ExpPoly(t)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-_ep_of(o))

    def __rsub__(self, o):
        return _ep_of(o) + (-self)

    def __mul__(self, o):
        o = _ep_of(o)
        if len(self.terms) * len(o.terms) > _MAX_EXP_TERMS:
            raise Infeasible(f'a product of exponential polynomials with {len(self.terms)} × {len(o.terms)} terms exceeds the cap')
        t = {}
        for q1, c1 in self.terms.items():
            for q2, c2 in o.terms.items():
                t[q1 + q2] = t.get(q1 + q2, Fraction(0)) + c1 * c2
        return ExpPoly(t)

    __rmul__ = __mul__

    def __pow__(self, n: int):
        if n < 0:
            return self.inverse() ** (-n)
        out, base = ExpPoly({Fraction(0): Fraction(1)}), self
        while n:
            if n & 1:
                out = out * base
            n >>= 1
            if n:
                base = base * base
        return out

    def inverse(self):
        """Exact for a monomial c·e^q (1/(c e^q) = e^{-q}/c); otherwise the caller falls back to enclosures."""
        if len(self.terms) == 1:
            (q, c), = self.terms.items()
            return ExpPoly({-q: 1 / c})
        raise Undecided('inverse of a sum of exponentials is not an exponential polynomial')

    def enclosure(self, bits: int) -> Iv:
        total = Iv(0, 0)
        for q, c in self.terms.items():
            total = total + Iv(c, c) * _exp_point(q, bits)
        return total

    def __repr__(self):
        return ' + '.join(f'{_fmt(c)}·e^{_fmt(q)}' if q != 0 else _fmt(c) for q, c in sorted(self.terms.items())) or '0'


def _ep_of(v) -> ExpPoly:
    if isinstance(v, ExpPoly):
        return v
    if isinstance(v, bool) or not isinstance(v, (int, Fraction)):
        raise Unsupported('exponential arithmetic with a non-rational value')
    return ExpPoly({Fraction(0): Fraction(v)})


def _ep(v):
    """Normalise: an exponential polynomial that is rational is a Fraction."""
    if isinstance(v, ExpPoly):
        return v.rational() if v.is_rational() else v
    return v


def _ubound(x, bits: int = 16) -> Fraction:
    """A short dyadic upper bound of |x|, for remainder estimates (never for the value itself)."""
    return _round_out(abs(_exact(x)), bits, True)


def _series_precision(bits: int) -> int:
    """Significant bits kept by the per-term rounding inside a series: the rounding errors of at most 2^16
    terms, each at most 2^-p relative, stay far below the remainder target 2^-(bits+4)."""
    return bits + _IV_GUARD_BITS + 16


@functools.lru_cache(maxsize=4096)
def _e_enclosure(bits: int) -> Iv:
    """e = Σ 1/k!: for N ≥ 1 the tail Σ_{k>N} 1/k! < 1/(N!·N)."""
    N, fact, acc = 1, 1, Fraction(1)
    target = Fraction(1, 1 << (bits + 4))
    while True:
        fact *= N
        acc += Fraction(1, fact)
        if Fraction(1, fact * N) < target:
            return Iv(acc, acc + Fraction(1, fact * N))
        N += 1


@functools.lru_cache(maxsize=4096)
def _exp_point(q, bits: int) -> Iv:
    """An enclosure of e^q for rational q: q = n + r with |r| ≤ 1/2; e^r by its Taylor series with the Lagrange
    remainder |r|^{N+1}/(N+1)! · e^{|r|} ≤ 2·|r|^{N+1}/(N+1)!, each term enclosed and rounded outward; e^n by the
    rounded power of the enclosure of e. Every intermediate interval contains the exact quantity it stands for."""
    q = _exact(q)
    if abs(q) > _MAX_EXP_ARGUMENT:
        raise Infeasible(f'Real.exp of {_approx(q)}: the exact enclosure exceeds the size caps')
    n = round(q)
    r = q - n
    p = _series_precision(bits)
    R, rb = Iv(r, r).round(p), _ubound(r)                        # rb ≥ |r|, rb < 0.5001 so e^rb < 2
    target = Fraction(1, 1 << (bits + 4))
    acc, term, N = Iv(1, 1), Iv(1, 1), 0
    while True:
        N += 1
        if N > _MAX_LOOP:
            raise Infeasible('exp series does not converge within the loop cap')
        term = (term * R * Fraction(1, N)).round(p)              # encloses r^N / N!
        acc = (acc + term).round(p)                              # encloses Σ_{k≤N} r^k / k!
        bound = 2 * rb ** (N + 1) / _fact(N + 1)
        if bound < target:
            break
    er = Iv(acc.lo - bound, acc.hi + bound)
    if n == 0:
        return er
    pe = bits + _IV_GUARD_BITS + 2 * abs(n).bit_length()         # rounding slack for the ≤ 2·log2(n) products
    e = _e_enclosure(bits + abs(n).bit_length() + 4)
    en = e.pow_rounded(abs(n), pe)                               # e > 0: e^n = [lo^n, hi^n], outward-rounded
    if n < 0:
        en = en.inverse()
    return (er * en).round(bits + _IV_GUARD_BITS)


@functools.lru_cache(maxsize=4096)
def _log2_enclosure(bits: int) -> Iv:
    return _atanh_series(Fraction(1, 3), bits) * 2


@functools.lru_cache(maxsize=4096)
def _atanh_series(z, bits: int) -> Iv:
    """atanh(z) = Σ_{j≥0} z^{2j+1}/(2j+1) for 0 ≤ z < 1. After the terms j = 0 … N the omitted tail is positive
    and at most z^{2N+3}/(2N+3) · Σ_{i≥0} z^{2i} = z^{2N+3}/((2N+3)(1 − z²)) (consecutive terms shrink by more
    than z²), so [S_N, S_N + tail] encloses the value; the partial sums are themselves enclosed because every
    power is rounded outward."""
    z = _exact(z)
    if not 0 <= z < 1:
        raise Unsupported('atanh series outside [0, 1)')
    p = _series_precision(bits)
    Z, zb = Iv(z, z).round(p), _ubound(z)
    if zb >= 1:
        raise Unsupported('atanh series too close to 1')
    Z2, zb2 = (Z * Z).round(p), zb * zb
    target = Fraction(1, 1 << (bits + 4))
    acc, N = Iv(0, 0), -1
    power, pb = Z, zb                           # power encloses z^{2N+1}; pb bounds it from above
    while True:
        N += 1
        if N > _MAX_LOOP:
            raise Infeasible('atanh series does not converge within the loop cap')
        acc = (acc + power * Fraction(1, 2 * N + 1)).round(p)
        power, pb = (power * Z2).round(p), pb * zb2             # z^{2N+3}: the first omitted term's numerator
        tail = pb / ((2 * N + 3) * (1 - zb2))
        if tail < target:
            return Iv(acc.lo, acc.hi + tail)


@functools.lru_cache(maxsize=4096)
def _log_point(q, bits: int) -> Iv:
    """An enclosure of log q for rational q > 0: q = 2^k·m with m ∈ [1, 2), log m = 2·atanh((m−1)/(m+1))."""
    q = _exact(q)
    if q <= 0:
        raise ValueError('log of a non-positive rational')
    k = q.numerator.bit_length() - q.denominator.bit_length()
    m = q / Fraction(2) ** k
    if m < 1:
        k -= 1
        m = m * 2
    if m >= 2:
        k += 1
        m = m / 2
    log_m = _atanh_series((m - 1) / (m + 1), bits) * 2
    if k == 0:
        return log_m
    return log_m + _log2_enclosure(bits + abs(k).bit_length() + 2) * k


@functools.lru_cache(maxsize=4096)
def _atan_series(x, bits: int) -> Iv:
    """atan(x) = Σ (−1)^j x^{2j+1}/(2j+1) for |x| ≤ 1/2: an alternating series whose terms decrease in absolute
    value, so the exact remainder is bounded by the first omitted term. Arguments in (1/2, 1] are split first:
    atan x = atan(1/2) + atan((x − 1/2)/(1 + x/2)) with the second argument in (0, 1/3]."""
    x = _exact(x)
    if abs(x) > 1:
        raise Unsupported('atan series outside [-1, 1]')
    if abs(x) > Fraction(1, 2):
        half = _atan_series(Fraction(1, 2), bits + 2)
        rest = _atan_series((abs(x) - Fraction(1, 2)) / (1 + abs(x) / 2), bits + 2)
        total = half + rest
        return total if x > 0 else -total
    p = _series_precision(bits)
    X, xb = Iv(x, x).round(p), _ubound(x)
    X2, xb2 = (X * X).round(p), xb * xb
    target = Fraction(1, 1 << (bits + 4))
    acc, j = Iv(0, 0), 0
    power, pb = X, xb                           # power encloses x^{2j+1} (with the sign of x); pb bounds |·|
    while True:
        if j > _MAX_LOOP:
            raise Infeasible('atan series does not converge within the loop cap')
        term = power * Fraction(1, 2 * j + 1)
        acc = (acc + (term if j % 2 == 0 else -term)).round(p)
        power, pb = (power * X2).round(p), pb * xb2
        nxt = pb / (2 * j + 3)                  # bounds the first omitted term
        if nxt < target:
            return Iv(acc.lo - nxt, acc.hi + nxt)
        j += 1


@functools.lru_cache(maxsize=4096)
def _pi_enclosure(bits: int) -> Iv:
    """Machin: π = 16·atan(1/5) − 4·atan(1/239)."""
    return _atan_series(Fraction(1, 5), bits + 6) * 16 - _atan_series(Fraction(1, 239), bits + 4) * 4


@functools.lru_cache(maxsize=4096)
def _sin_cos_point(y, bits: int, want: str) -> Iv:
    """sin or cos of a rational y by its Taylor series at 0. Every derivative of sin and cos is bounded by 1, so
    the Lagrange remainder of the polynomial through degree N−1 is at most |y|^N/N!."""
    y = _exact(y)
    if abs(y) > 64:
        raise Infeasible('trigonometric series at an argument beyond the reduction range')
    p = _series_precision(bits)
    Y, yb = Iv(y, y).round(p), _ubound(y)
    target = Fraction(1, 1 << (bits + 4))
    acc, N = Iv(0, 0), 0
    term, tb = Iv(1, 1), Fraction(1)            # term encloses y^N / N!; tb = yb^N / N! bounds it from above
    while True:
        if N > _MAX_LOOP:
            raise Infeasible('trigonometric series does not converge within the loop cap')
        if want == 'sin' and N % 2 == 1:
            acc = (acc + (term if N % 4 == 1 else -term)).round(p)
        if want == 'cos' and N % 2 == 0:
            acc = (acc + (term if N % 4 == 0 else -term)).round(p)
        N += 1
        term, tb = (term * Y * Fraction(1, N)).round(p), tb * yb / N
        if tb < target and N > 2:
            return Iv(acc.lo - tb, acc.hi + tb)


def _reduce_angle(x: Iv, bits: int):
    """x − 2kπ with k the nearest integer to x/(2π), as an interval (π is an interval)."""
    pi = _pi_enclosure(bits + 8)
    mid = (x.lo + x.hi) / 2
    k = round(mid / (2 * pi.lo)) if pi.lo > 0 else 0
    if abs(k) > 1 << 40:
        raise Infeasible('angle reduction of an astronomically large argument')
    return x - pi * (2 * k), k


def _trig_iv(x: Iv, bits: int, want: str) -> Iv:
    """sin/cos on an interval: reduce to |y| ≲ π + width, evaluate at the endpoints, and widen to ±1 when a
    critical point may lie inside (checked against the π enclosure conservatively)."""
    y, _ = _reduce_angle(x, bits)
    if y.width > 4:
        return Iv(-1, 1)
    pi = _pi_enclosure(bits + 8)
    lo_v, hi_v = _sin_cos_point(y.lo, bits, want), _sin_cos_point(y.hi, bits, want)
    out = Iv(min(lo_v.lo, hi_v.lo), max(lo_v.hi, hi_v.hi))
    # critical points: sin at π/2 + kπ, cos at kπ, for k with the point possibly inside [y.lo, y.hi]
    for k in range(-3, 4):
        if want == 'sin':
            c_lo, c_hi = pi.lo / 2 + pi.lo * k if k >= 0 else pi.lo / 2 + pi.hi * k, pi.hi / 2 + pi.hi * k if k >= 0 else pi.hi / 2 + pi.lo * k
        else:
            c_lo, c_hi = (pi.lo * k, pi.hi * k) if k >= 0 else (pi.hi * k, pi.lo * k)
        if c_hi >= y.lo and c_lo <= y.hi:                        # the critical point may lie inside
            extreme = 1 if (want == 'sin' and k % 2 == 0) or (want == 'cos' and k % 2 == 0) else -1
            out = Iv(min(out.lo, extreme), max(out.hi, extreme))
    return Iv(max(out.lo, Fraction(-1)), min(out.hi, Fraction(1)))


def _exp_iv(x: Iv, bits: int) -> Iv:
    return Iv(_exp_point(x.lo, bits).lo, _exp_point(x.hi, bits).hi)


def _log_iv(x: Iv, bits: int) -> Iv:
    """Mathlib `Real.log`: log |x| for x ≠ 0 and log 0 = 0. An enclosure straddling 0 is undecided."""
    if x.lo <= 0 <= x.hi:
        if x.lo == 0 and x.hi == 0:
            return Iv(0, 0)
        raise Undecided(f'Real.log of an enclosure containing 0: {x}')
    a = x.abs()
    return Iv(_log_point(a.lo, bits).lo, _log_point(a.hi, bits).hi)


@functools.lru_cache(maxsize=4096)
def _sqrt_point(q, bits: int) -> Iv:
    q = _exact(q)
    if q <= 0:
        return Iv(0, 0)
    scale = 1 << bits
    n = (q * scale * scale)
    r = _isqrt(n.numerator // n.denominator)
    return Iv(Fraction(r, scale), Fraction(r + 1, scale))


def _sqrt_iv(x: Iv, bits: int) -> Iv:
    """Mathlib `Real.sqrt`: 0 on non-positive arguments; monotone."""
    lo = _sqrt_point(x.lo, bits).lo if x.lo > 0 else Fraction(0)
    hi = _sqrt_point(x.hi, bits).hi if x.hi > 0 else Fraction(0)
    return Iv(lo, hi)


def _iv_compare(op: str, a: Iv, b: Iv):
    """True / False when the enclosures decide the comparison, else None."""
    if op == '<':
        return True if a.hi < b.lo else (False if a.lo >= b.hi else None)
    if op == '≤':
        return True if a.hi <= b.lo else (False if a.lo > b.hi else None)
    if op == '>':
        return _iv_compare('<', b, a)
    if op == '≥':
        return _iv_compare('≤', b, a)
    if op == '=':
        if a.hi < b.lo or b.hi < a.lo:
            return False
        if a.lo == a.hi == b.lo == b.hi:
            return True
        return None
    if op == '≠':
        r = _iv_compare('=', a, b)
        return None if r is None else not r
    raise Unsupported(f'operator {op} on enclosures')


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
    structural = _structural_conclusion(thm.statement)
    if structural:
        return EvalResult('UNSUPPORTED', f'conclusion is a structural predicate {structural}: trivial on finite discrete '
                          'models (every set is measurable, every function measurable and integrable), so sampling '
                          'has no content here', 0, False, [], [])
    if thm.binders:
        return _evaluate_sampled(module, thm, seed, samples)
    ev = Evaluator(module, thm.namespace, thm.opens)
    try:
        for bits in _ENCLOSURE_BITS:
            # transcendental enclosures are refined until they decide every comparison or the cap is reached
            ev.bits = bits
            ev.instances, ev.vacuous, ev.failures, ev.probes = 0, False, [], []
            try:
                ok = ev.prop(thm.statement, {})
                break
            except Undecided as e:
                undecided = e
        else:
            return EvalResult('UNSUPPORTED', f'enclosures to {_ENCLOSURE_BITS[-1]} bits do not decide the statement: '
                              f'{undecided}', 0, False, [], ev.values_seen)
        quantified = ev.instances > 0 or ev.vacuous
        if not ok:
            detail = ('fails at ' + ', '.join(ev.failures)) if ev.failures else 'closed statement evaluated false'
            if ev.bits != _ENCLOSURE_BITS[0]:
                detail += f' (enclosures at {ev.bits} bits)'
            return EvalResult('FAIL', detail, ev.instances if quantified else 1, False, ev.probes, ev.values_seen)
        if quantified:
            detail = f'holds on all {ev.instances} quantified instances' + (' (an empty domain: vacuous)' if ev.vacuous else '')
        else:
            detail = 'closed statement evaluated true'
        if ev.bits != _ENCLOSURE_BITS[0]:
            detail += f' (transcendental enclosures at {ev.bits} bits)'
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
# a ℕ binder that is the dimension of a later `Fin n → ℝ` / `Fin n` binder is drawn from here: statements over
# every dimension are tested at the small ones (0 included), where a vector is enumerable and a witness readable
_DIM_POOL = [2, 1, 3, 0, 2, 4, 1, 3]
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


def _resolve_dim(token: str, env):
    """A dimension token's value under the current sample, or None when its ℕ binder is not yet drawn."""
    if token.isdigit():
        return int(token)
    b = env.get(token)
    return b.value if isinstance(b, Bound) and isinstance(b.value, int) and not isinstance(b.value, bool) else None


def _structured_vector(typ, i, env):
    """The i-th structured value of a vector / index / matrix binder (deterministic): the zero vector, the ones,
    a unit vector, the pool along the diagonal, alternating signs, a constant; the first / last / middle index;
    the zero / identity / pool-diagonal matrix. None when a dimension is not drawn yet or is empty for an index."""
    b = _base(typ)
    if b == 'Fin':
        n = _resolve_dim(typ.split(':')[1], env)
        if not n:
            return None
        return [0, n - 1, 1 % n, n // 2, 0, n - 1, 2 % n, (n - 2) % n][i % 8]
    if b == 'Vec':
        elem, dim, space = _vec_info(typ)
        n = _resolve_dim(dim, env)
        if n is None:
            return None
        pool = _POOLS[elem]
        def pv(k):
            v = _coerce_sample(pool[k % len(pool)], elem)
            return v if v is not None else _coerce_sample(0, elem)
        patterns = [
            lambda k: _coerce_sample(0, elem),
            lambda k: _coerce_sample(1, elem),
            lambda k: _coerce_sample(1 if k == 0 else 0, elem),
            lambda k: pv(k + 1),
            lambda k: _coerce_sample(1 if k % 2 == 0 else -1, elem) if elem != 'Nat' else _coerce_sample(k % 2, elem),
            lambda k: pv(7),
            lambda k: pv(2 * k + 3),
            lambda k: _coerce_sample(1 if k == n - 1 else 0, elem),
        ]
        return Vec([patterns[i % 8](k) for k in range(n)], elem, space)
    if b == 'Mat':
        elem, m_, n_ = _mat_info(typ)
        m, n = _resolve_dim(m_, env), _resolve_dim(n_, env)
        if m is None or n is None:
            return None
        pool = _POOLS[elem]
        def pv(k):
            v = _coerce_sample(pool[k % len(pool)], elem)
            return v if v is not None else _coerce_sample(0, elem)
        if i % 4 == 0:
            rows = [[_coerce_sample(0, elem) for _ in range(n)] for _ in range(m)]
        elif i % 4 == 1:
            rows = [[_coerce_sample(1 if r == c else 0, elem) for c in range(n)] for r in range(m)]
        elif i % 4 == 2:
            rows = [[pv(r * n + c + 1) for c in range(n)] for r in range(m)]
        else:
            rows = [[pv(r + c + i) for c in range(n)] for r in range(m)]
        return Mat(rows, elem)
    return None


def _random_vector(typ, rng: random.Random, env, boundaries=None):
    """A random vector / index / matrix: each entry a pool value, a hypothesis boundary of that component when
    one is known, or a random rational; None when a dimension is not drawn yet or is empty for an index."""
    b = _base(typ)
    if b == 'Fin':
        n = _resolve_dim(typ.split(':')[1], env)
        return None if not n else rng.randrange(n)
    boundaries = boundaries or {}

    def entry(elem, k):
        bs = boundaries.get(k)
        r = rng.random()
        if bs and r < 0.4:
            return rng.choice(bs)
        if r < 0.8:
            v = _coerce_sample(rng.choice(_POOLS[elem]), elem)
            if v is not None:
                return v
        return _random_value(rng, elem)
    if b == 'Vec':
        elem, dim, space = _vec_info(typ)
        n = _resolve_dim(dim, env)
        if n is None:
            return None
        if rng.random() < 0.15:
            c = entry(elem, 0)
            return Vec([c] * n, elem, space)
        return Vec([entry(elem, k) for k in range(n)], elem, space)
    if b == 'Mat':
        elem, m_, n_ = _mat_info(typ)
        m, n = _resolve_dim(m_, env), _resolve_dim(n_, env)
        if m is None or n is None:
            return None
        return Mat([[entry(elem, r * n + c) for c in range(n)] for r in range(m)], elem)
    return None


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


def _literal_only(node) -> bool:
    """Whether an expression is built from numerals alone (its type comes from the other side of a comparison)."""
    if isinstance(node, Num):
        return isinstance(node.value, int)
    if isinstance(node, BinOp):
        return _literal_only(node.left) and _literal_only(node.right)
    if isinstance(node, Neg):
        return _literal_only(node.expr)
    return False


def _split_conjuncts(node):
    if isinstance(node, BinOp) and node.op == '∧':
        return _split_conjuncts(node.left) + _split_conjuncts(node.right)
    return [node]


def _structural_conclusion(stmt):
    """The sorted predicate names when every conjunct of the conclusion is a structural predicate
    (measurability, integrability, …), else None. Such a conclusion has no sampled content on any model."""
    conjuncts = _split_conjuncts(stmt)
    if conjuncts and all(isinstance(c, App) and isinstance(c.fn, Var) and c.fn.name in _STRUCTURAL_PREDICATES
                         for c in conjuncts):
        return sorted({c.fn.name for c in conjuncts})
    return None


def _boundary_candidates(var: str, typ: str, hyps: list, ev: Evaluator, env: dict, var_names: set,
                         component=None) -> list:
    """Exact values at and next to the boundaries that the hypotheses give ``var`` once every other
    variable they mention is assigned: `c ≤ x`, `x < c`, `x = c`, `x ≠ c`, `|x - e| ≤ c` (and `|e - x|`).
    With ``component`` = k the target is the entry `var k` of a vector binder (`v 0 ∈ Set.Ioo a b`)."""
    out = []
    eps = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 10 ** 6), Fraction(-1, 10 ** 6)] if typ in _FIELD_TYPES \
        else [0, 1, -1]

    def is_target(node):
        if component is None:
            return isinstance(node, Var) and node.name == var
        return (isinstance(node, App) and isinstance(node.fn, Var) and node.fn.name == var and len(node.args) == 1
                and isinstance(node.args[0], Num) and node.args[0].value == component)

    def closed(node):
        return all(v in env for v in _free_vars(node) & var_names) and not (component is not None and var in _free_vars(node))

    def values(node):
        """The exact value of a closed bound, as a list of rational candidates: a transcendental bound (`r ≤ Real.exp (-1)`)
        contributes the two endpoints of its 64-bit enclosure — exact rationals just inside and just outside it."""
        try:
            v = ev.num(node, env, typ)
        except Unsupported:
            return []
        if isinstance(v, (ExpPoly, Iv)):
            try:
                enclosure = Iv.of(v, _ENCLOSURE_BITS[0]).round(_ENCLOSURE_BITS[0])     # short dyadic endpoints
            except Unsupported:
                return []
            return [enclosure.lo, enclosure.hi]
        if isinstance(v, (list, tuple, bool, Alg)):
            return []
        return [v]

    def centre(node, other):
        # node is `|x - e|` or `|e - x|`: candidates e ± other
        if isinstance(node, App) and isinstance(node.fn, Var) and node.fn.name == 'abs' and len(node.args) == 1 \
                and isinstance(node.args[0], BinOp) and node.args[0].op == '-':
            l, r = node.args[0].left, node.args[0].right
            e = r if is_target(l) else (l if is_target(r) else None)
            if e is not None and closed(e) and closed(other):
                return [c for ev_ in values(e) for ov in values(other) for c in (ev_ + ov, ev_ - ov, ev_)]
        return []

    for hyp in hyps:
        for h in _split_conjuncts(hyp):
            if not (isinstance(h, BinOp) and h.op in ('≤', '<', '≥', '>', '=', '≠')):
                continue
            for side, other in ((h.left, h.right), (h.right, h.left)):
                if is_target(side) and closed(other):
                    for c in values(other):
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
    ev = Evaluator(module, thm.namespace, thm.opens)
    rng = random.Random(seed)
    accepted, rejected, infeasible, undecided, hyp_undecided, draws = 0, 0, 0, 0, 0, 0
    seen = set()
    probes = []
    model = any(t in _MODEL_KINDS for _, t in variables)
    probability = {h.args[0].name for h in hyps if isinstance(h, App) and isinstance(h.fn, Var)
                   and h.fn.name == 'IsProbabilityMeasure' and len(h.args) == 1 and isinstance(h.args[0], Var)}
    vectors = any(_base(t) in _VECTOR_KINDS for _, t in variables)
    dim_vars = {d for _, t in variables if _base(t) in _VECTOR_KINDS for d in _dims_of(t) if not d.isdigit()}
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
            nonlocal hyp_undecided
            try:
                try:
                    ev.bits = _ENCLOSURE_BITS[0]
                    return all(ev.prop(h, env) for h in hs)
                except Undecided:
                    ev.bits = _ENCLOSURE_BITS[2]        # a boundary sample sits within 2^-64 of a transcendental bound:
                    try:                                # 256 bits usually decide it; an identity stays undecided
                        return all(ev.prop(h, env) for h in hs)
                    except Undecided:
                        hyp_undecided += 1              # not decided: the sample is not used (never counted as holding)
                        return False
                    finally:
                        ev.bits = _ENCLOSURE_BITS[0]
            except Infeasible:
                return False

        def draw(index, env):
            name, typ = variables[index]
            if _base(typ) in _VECTOR_KINDS:
                bounds = {}
                if _base(typ) == 'Vec':
                    elem, dim, _ = _vec_info(typ)
                    n = _resolve_dim(dim, env)
                    for k in range(n or 0):
                        bs = _boundary_candidates(name, elem, hyps, ev, env, var_names, component=k)
                        if bs:
                            bounds[k] = bs
                for _ in range(_DRAW_RETRIES):
                    v = _random_vector(typ, rng, env, bounds)
                    if v is None:
                        return False
                    env[name] = Bound(v, typ)
                    if hyps_hold(completed[index], env):
                        return True
                return False
            if name in dim_vars:
                for _ in range(_DRAW_RETRIES):
                    env[name] = Bound(rng.choice(_DIM_POOL), 'Nat')
                    if hyps_hold(completed[index], env):
                        return True
                return False
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
            nonlocal accepted, rejected, infeasible, undecided, draws
            draws += 1
            key = (ev.model_n,) + tuple(env[n].value for n, _ in variables)
            if key in seen:
                return None
            seen.add(key)
            try:
                try:
                    ev.bits = _ENCLOSURE_BITS[0]
                    ok = ev.prop(thm.statement, env)
                except Undecided:
                    ev.bits = _ENCLOSURE_BITS[2]
                    try:
                        ok = ev.prop(thm.statement, env)
                    except Undecided:
                        undecided += 1          # the enclosures at 256 bits do not decide this instance: skipped
                        return None
                    finally:
                        ev.bits = _ENCLOSURE_BITS[0]
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
                               'infeasible': infeasible, 'undecided': undecided, 'witness': record})

        # structured points: every variable takes the i-th pool value (and the i-th carrier size)
        for i in range(_DIAGONAL):
            env = {}
            if model:
                ev.model_n = _MODEL_SIZES[i]
            for name, typ in variables:
                if typ in _MODEL_KINDS:
                    env[name] = Bound(_structured_model_value(typ, i, ev.model_n, name in probability), typ)
                    continue
                if _base(typ) in _VECTOR_KINDS:
                    v = _structured_vector(typ, i, env)
                    if v is None:
                        break
                    env[name] = Bound(v, typ)
                    continue
                if name in dim_vars:
                    env[name] = Bound(_DIM_POOL[i % len(_DIM_POOL)], 'Nat')
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
            if len(probes) >= 8 or model or vectors:
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
        sampling = {'seed': seed, 'draws': draws, 'accepted': accepted, 'rejected': rejected, 'infeasible': infeasible,
                    'undecided': undecided}
        if hyp_undecided:
            sampling['hypotheses_undecided'] = hyp_undecided
        if accepted == 0:
            if undecided and not rejected and not infeasible:
                why = (f'the enclosures to {_ENCLOSURE_BITS[2]} bits did not decide any of the {undecided} sampled '
                       'instances (transcendental values too close to the compared side)')
            elif infeasible and not rejected:
                why = 'every drawn instance was infeasible to evaluate exactly'
            elif hyp_undecided and hyp_undecided >= rejected:
                why = (f'the enclosures did not decide the hypotheses on {hyp_undecided} draws (transcendental hypotheses '
                       'too close to their bound), so no draw was admitted')
            else:
                why = 'the hypotheses may be unsatisfiable (vacuous, A5) or outside what the sampler can hit'
            return EvalResult('UNSUPPORTED', f'no sample satisfied the hypotheses and evaluated ({draws + rejected} draws): {why}',
                              0, False, probes, ev.values_seen, sampling)
        if undecided > accepted:
            # the enclosures decide only a minority of the instances: the statement is outside what this layer
            # can corroborate (typically an equality between transcendental values), not weakly supported
            return EvalResult('UNSUPPORTED', f'the enclosures to {_ENCLOSURE_BITS[2]} bits did not decide {undecided} of the '
                              f'sampled instances ({accepted} decided, all held): sampling cannot corroborate the statement',
                              accepted, False, probes, ev.values_seen, sampling)
        if hyp_undecided and accepted * 4 < samples:
            # the hypotheses themselves are undecided on most draws (an identity between transcendental values as a
            # hypothesis): the few admitted instances are not evidence about the hypotheses' domain
            return EvalResult('UNSUPPORTED', f'the enclosures did not decide the hypotheses on {hyp_undecided} draws and only '
                              f'{accepted} instances were admitted: sampling cannot corroborate the statement',
                              accepted, False, probes, ev.values_seen, sampling)
        undecided_note = f', {undecided} undecided by enclosures and skipped' if undecided else ''
        if hyp_undecided:
            undecided_note += f', the hypotheses undecided on {hyp_undecided} draws'
        detail = (f'holds on {accepted} sampled instances satisfying the hypotheses ({rejected} draws rejected by the '
                  f'hypotheses, {infeasible} infeasible{undecided_note}, seed {seed}): sampling is admitted evidence '
                  'of the statement, not a proof of it')
        if accepted * 4 < samples:
            detail += f' — WEAK: only {accepted} instances, the hypotheses are hard to hit by sampling'
            sampling['weak'] = True
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
    ev = Evaluator(module, thm.namespace, thm.opens)
    env = {}
    for b in thm.binders:
        if b.kind == 'var':
            placeholder = Func((), 'Real') if b.type == 'Func' else None
            if _base(b.type) == 'Vec':
                placeholder = Vec((), _vec_info(b.type)[0], _vec_info(b.type)[2])
            elif _base(b.type) == 'Mat':
                placeholder = Mat((), _mat_info(b.type)[0])
            elif _base(b.type) == 'Fin':
                placeholder = 0
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
    share_definitions(modules)
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


def check_lean_dir(root, exclude=(), seed=1, samples=64):
    """Census mode: every `theorem`/`lemma` in every `.lean` file under ``root`` (recursively, sorted; paths
    containing any ``exclude`` substring skipped; byte-identical files evaluated once and recorded as copies) is
    a target, with definitions and carrier types shared across the group as in a package. The report has the
    shape of ``check_package`` plus a histogram of UNSUPPORTED reasons; the exit code is 0 only when every target
    PASSes (there is no manifest to register against, so this mode is coverage evidence, never a gate)."""
    root = pathlib.Path(root)
    files = sorted(p for p in root.rglob('*.lean') if not any(x in str(p.relative_to(root)) for x in exclude))
    modules, hashes, copies, seen_hash = {}, {}, {}, {}
    for p in files:
        rel = str(p.relative_to(root))
        h = _sha256(p)
        hashes[rel] = h
        if h in seen_hash:
            copies[rel] = seen_hash[h]
            continue
        seen_hash[h] = rel
        modules[rel] = parse_module(p.read_text(encoding='utf-8', errors='replace'))
    share_definitions(modules)
    per_target, counts, reasons = {}, {}, {}
    for rel, module in modules.items():
        for thm in module.theorems:
            res = evaluate_theorem(module, thm, seed, samples)
            key = f'{rel}::{thm.name}'
            per_target[key] = {'module': rel, 'name': thm.name, 'status': res.status, 'detail': res.detail,
                               'instances': res.instances, 'vacuous': res.vacuous,
                               'float_free': not any(isinstance(v, float) for v in res.values)}
            if res.sampling is not None:
                per_target[key]['sampling'] = res.sampling
            counts[res.status] = counts.get(res.status, 0) + 1
            if res.status == 'UNSUPPORTED':
                reason = _reason_class(res.detail)
                reasons[reason] = reasons.get(reason, 0) + 1
    return {'lean_dir': str(root), 'files': len(files), 'files_evaluated': len(modules), 'copies': copies,
            'file_sha256': hashes, 'targets': len(per_target), 'status_counts': counts,
            'unsupported_reasons': dict(sorted(reasons.items(), key=lambda kv: -kv[1])), 'per_target': per_target,
            'sampling': {'seed': seed, 'samples': samples},
            'exit_code': 0 if counts and set(counts) == {'PASS'} else 1,
            'note': ('G12 census: coverage of the empirical statement check over every declaration in a directory, '
                     'with no manifest. PASS/CORROBORATED/FAIL/UNSUPPORTED as in a package check; UNSUPPORTED reasons '
                     'are classified. Not a gate and not kernel evidence.')}


def _reason_class(detail: str) -> str:
    """A short class for an UNSUPPORTED detail, for the census histogram."""
    d = detail
    for needle, cls in (('not reached by the parser', 'declaration not reached by the parser'),
                        ('ambiguous definition', 'ambiguous definition across modules'),
                        ('do not decide', 'enclosures do not decide the statement'),
                        ('did not decide', 'enclosures do not decide the statement'),
                        ('binder', 'binder outside the interpreter'), ('instance binder', 'binder outside the interpreter'),
                        ('hypothesis', 'hypothesis outside the interpreter'), ('existential', 'existential quantifier'),
                        ('unbounded quantifier', 'unbounded quantifier inside the statement'),
                        ('quantifier over', 'quantifier over an unsupported type'),
                        ('notation', 'Mathlib notation'), ('unparsed statement', 'unparsed statement'),
                        ('malformed theorem', 'malformed or term-mode theorem'), ('structural predicate', 'structural conclusion'),
                        ('Real.rpow', 'real exponent (Real.rpow)'),
                        ('transcendental', 'transcendental function'), ('nested radical', 'nested radical'),
                        ('unknown name', 'unknown name'), ('unknown function', 'unknown function'),
                        ('result type', 'definition result type outside the interpreter'),
                        ('no sample', 'no sample satisfied the hypotheses'), ('evaluation error', 'evaluation error'),
                        ('exceeds', 'resource cap'), ('ascription to', 'ascription to an unsupported type'),
                        ('scientific literal', 'scientific literal'), ('definition', 'definition outside the interpreter'),
                        ('application of', 'application outside the interpreter'), ('proposition', 'proposition outside the interpreter')):
        if needle in d:
            return cls
    return 'other'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--package', help='formal package root containing manifest.json')
    group.add_argument('--lean-dir', help='census mode: every theorem in every .lean file under this directory')
    ap.add_argument('--exclude', action='append', default=[],
                    help='census mode: skip files whose relative path contains this substring (repeatable)')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--selftest', action='store_true', help='also run the mutation self-test per module')
    ap.add_argument('--seed', type=int, default=1, help='seed of the deterministic sampler for binders before the colon')
    ap.add_argument('--samples', type=int, default=64, help='hypothesis-satisfying instances to test per sampled statement')
    ap.add_argument('--admit-corroborated', action='store_true',
                    help='exit 0 when every target is PASS or CORROBORATED (default: PASS only, the registered contract)')
    a = ap.parse_args(argv)
    if a.lean_dir:
        report = check_lean_dir(a.lean_dir, exclude=a.exclude, seed=a.seed, samples=a.samples)
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(report, indent=1, default=str) + '\n')
        summary = {k: report[k] for k in ('lean_dir', 'files', 'files_evaluated', 'targets', 'status_counts',
                                          'unsupported_reasons', 'sampling', 'exit_code')}
        summary['copies'] = len(report['copies'])
        print(json.dumps(summary, indent=1))
        for name, r in report['per_target'].items():
            if r['status'] != 'PASS':
                print(f"{r['status']}: {name}: {r['detail'][:200]}")
        return report['exit_code']
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
