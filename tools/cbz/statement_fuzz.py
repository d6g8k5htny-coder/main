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
kernel. UNSUPPORTED is fail-closed: the construct is outside the interpreter (Mathlib, reals,
unbounded quantifiers, ambiguous Int division) and the target cannot count as fuzz-validated.

Lean semantics implemented: Nat subtraction truncates at 0; Nat and Int division by zero is 0;
Nat division floors; Int division is evaluated only when the T-rounding and Euclidean conventions
agree (otherwise UNSUPPORTED); numerals default to Nat unless an ascription or a definition's result
type makes the arithmetic Int, following the maximal-type rule of Lean's `binop%` elaborator.

Usage:
    python3 tools/cbz/statement_fuzz.py --package formal [--json report.json] [--selftest]
Exit 0 only when every registered target is PASS (and, with --selftest, every required mutant is
killed and the harness control passes).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
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


_MULTI_OPS = ('<=', '>=', '!=', '->', ':=', '=>', '/\\', '\\/', '<->', '||', '&&', '..')
_SINGLE = set('+-*/%^=<>()[],:|.{}!∀∃∧∨¬→↔≤≥≠∈∉λ')
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
    value: int

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
            break
        if not args:
            return head
        if isinstance(head, App) and isinstance(head.fn, Var) and head.fn.name.startswith('.'):
            return App(head.fn, head.args + args)      # receiver.method arg1 arg2 …
        return App(head, args)

    def parse_postfix(self, atom):
        """Dot notation binds tighter than application: `p.1`, `xs.foldl`, `(e).filter`."""
        while True:
            t = self.peek()
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
            return Num(int(t.value))
        if t.kind == 'ident':
            if t.value in ('fun', 'λ'):
                self.i -= 1
                return self.parse_lambda()
            return Var(t.value)
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
class Theorem:
    name: str
    statement: object | None
    error: str | None = None
    text: str = ''

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
        if v in ('(', '['):
            depth += 1
        elif v in (')', ']'):
            depth -= 1
        elif depth == 0 and v == value:
            return k
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
            if colon == -1 or assign == -1:
                theorems.append(Theorem(full, None, 'malformed theorem'))
                continue
            if colon != k + 2:
                theorems.append(Theorem(full, None, 'theorem with binders before the colon (unbounded domain)'))
                continue
            text = ' '.join(t.value for t in toks[colon + 1:assign])
            try:
                stmt = Parser(toks, colon + 1, assign).parse_expr(0)
                theorems.append(Theorem(full, stmt, None, text))
            except Unsupported as e:
                theorems.append(Theorem(full, None, f'unsupported syntax: {e}', text))
            except SyntaxError as e:
                theorems.append(Theorem(full, None, f'unparsed statement: {e}', text))
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
# Evaluation
# --------------------------------------------------------------------------------------------

_LIST_METHODS = {'foldl', 'filter', 'map', 'sum', 'length'}
_INT_TYPES = {'Int', 'ℤ'}
_NAT_TYPES = {'Nat', 'ℕ'}
_RAT_TYPES = {'Rat', 'ℚ'}


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
            raise Unsupported(f'ascription to {node.type}')
        if isinstance(node, Num):
            return 'Nat'
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
                if fn.name in ('Int.natAbs', 'Int.toNat', 'Nat.choose', 'Nat.factorial', 'Nat.succ', 'Nat.pred',
                               'List.length', 'List.sum', 'Nat.min', 'Nat.max', 'min', 'max', 'Nat.sqrt'):
                    return 'Nat' if not fn.name.startswith('Int.') or fn.name in ('Int.natAbs', 'Int.toNat') else 'Int'
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
            return self._rec(node.value if typ != 'Rat' else Fraction(node.value))
        if isinstance(node, Ascribe):
            return self.num(node.expr, env, typ)
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
        if isinstance(v, (int, Fraction)) and not isinstance(v, bool):
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
        if op == '^':
            if b < 0:
                raise Unsupported('negative exponent')
            return a ** b
        if op == '/':
            if typ == 'Rat':
                return Fraction(0) if b == 0 else a / b
            if b == 0:
                return 0
            if typ == 'Nat':
                return a // b
            t_div = int(a / Fraction(b)) if (a * b) >= 0 else -((-a) // b if a < 0 else a // (-b))
            # T-rounding (toward zero) and Euclidean division
            tdiv = (abs(a) // abs(b)) * (1 if (a >= 0) == (b > 0) else -1)
            ediv = a // b if b > 0 else -(a // (-b))
            if tdiv != ediv:
                raise Unsupported('Int division with a negative operand: T-rounding and Euclidean conventions disagree')
            return ediv
        if op == '%':
            if b == 0:
                return a
            if typ == 'Nat':
                return a % b
            tmod = a - ((abs(a) // abs(b)) * (1 if (a >= 0) == (b > 0) else -1)) * b
            emod = a - (a // b if b > 0 else -(a // (-b))) * b
            if tmod != emod:
                raise Unsupported('Int modulus with a negative operand: conventions disagree')
            return emod
        raise Unsupported(f'operator {op}')

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
        if name == 'List.range' and len(args) == 1:
            n = self.num(args[0], env, 'Nat')
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
                a = self.num(node.left, env, typ)
                b = self.num(node.right, env, typ)
                if isinstance(a, (list, tuple, bool)) or isinstance(b, (list, tuple, bool)):
                    if node.op == '=':
                        return a == b
                    if node.op == '≠':
                        return a != b
                    raise Unsupported('ordering of non-numeric values')
                return {'=': a == b, '≠': a != b, '<': a < b, '>': a > b, '≤': a <= b, '≥': a >= b}[node.op]
            raise Unsupported(f'operator {node.op} in a proposition')
        if isinstance(node, Var) and node.name in ('True', 'False'):
            return node.name == 'True'
        if isinstance(node, Ascribe):
            return self.prop(node.expr, env)
        raise Unsupported(f'proposition of form {type(node).__name__}')


def _max_type(a, b):
    order = {'Bool': -1, 'Nat': 0, 'Int': 1, 'Rat': 2, 'Prop': 3}
    if 'Prop' in (a, b):
        raise Unsupported('arithmetic on propositions')
    return a if order[a] >= order[b] else b


def _choose(n, k):
    if k < 0 or k > n:
        return 0
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
# Theorem evaluation with quantifier enumeration and boundary probes
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class EvalResult:
    status: str                 # PASS | FAIL | UNSUPPORTED
    detail: str
    instances: int
    vacuous: bool
    probes: list                # [{'value': v, 'holds': bool}] outside-domain probes (information only)
    values: list                # every intermediate numeric value (A3 audit: never a float)


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
        if isinstance(hyp, BinOp) and hyp.op in ('≤', '<') and isinstance(hyp.left, Var) and hyp.left.name == forall.var:
            bound = ev.num(hyp.right, env, 'Nat')
            if isinstance(bound, int) and not isinstance(bound, bool) and bound <= 100000:
                return list(range(bound + 1 if hyp.op == '≤' else bound)), body.right
    return None, body


def evaluate_theorem(module: Module, thm: Theorem) -> EvalResult:
    if thm.statement is None:
        return EvalResult('UNSUPPORTED', thm.error or 'no statement', 0, False, [], [])
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
    except (SyntaxError, RecursionError, ZeroDivisionError, OverflowError, KeyError, IndexError, TypeError, ValueError) as e:
        return EvalResult('UNSUPPORTED', f'evaluation error: {type(e).__name__}: {e}', 0, False, [], ev.values_seen)


def check_module(source: str):
    module = parse_module(source)
    return [evaluate_theorem(module, t) for t in module.theorems]


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
    """Harness control (unmutated set PASS) + class 1 (must FAIL) + class 2 (must PASS). An UNSUPPORTED or
    errored mutant is a survivor / a rejection, never a kill."""
    baseline = [(t, evaluate(module, t)) for t in module.theorems]
    control_passed = all(r.status == 'PASS' for _, r in baseline)
    mutants = weakenings = 0
    survivor_list, rejected = [], []
    for thm, res in baseline:
        if res.status != 'PASS' or thm.statement is None:
            continue
        for label, mutated in required_mutants(thm.statement):
            mutants += 1
            r = evaluate(module, Theorem(thm.name, mutated, None, thm.text))
            if r.status != 'FAIL':
                survivor_list.append({'theorem': thm.name, 'mutant': label, 'status': r.status})
        for label, mutated in weakening_mutants(thm.statement):
            weakenings += 1
            r = evaluate(module, Theorem(thm.name, mutated, None, thm.text))
            if r.status != 'PASS':
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


def check_package(root, selftest=False):
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
            res = evaluate_theorem(module, thm)
        per_target[name] = {'module': rel, 'status': res.status, 'detail': res.detail, 'instances': res.instances,
                            'vacuous': res.vacuous, 'probes': res.probes,
                            'float_free': not any(isinstance(v, float) for v in res.values)}
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
    exit_code = 0 if counts and set(counts) == {'PASS'} and not unregistered and controls_ok and all(
        s['harness_ok'] for s in selftests.values()) else 1
    return {'package_root': str(root), 'manifest_sha256': _sha256(manifest_path), 'module_sha256': module_hashes,
            'targets': len(per_target), 'status_counts': counts, 'per_target': per_target,
            'unregistered_theorems': unregistered, 'manifest_targets': normalized_targets,
            'selftest': selftests, 'declared_controls': controls_report, 'exit_code': exit_code,
            'note': ('G12 empirical statement check: PASS means the statement, read independently of Lean, '
                     'evaluates true in exact arithmetic; it is not kernel evidence. UNSUPPORTED is fail-closed.')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--package', required=True, help='formal package root containing manifest.json')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--selftest', action='store_true', help='also run the mutation self-test per module')
    a = ap.parse_args(argv)
    report = check_package(a.package, selftest=a.selftest)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1, default=str) + '\n')
    summary = {k: report[k] for k in ('package_root', 'targets', 'status_counts', 'unregistered_theorems', 'exit_code')}
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
