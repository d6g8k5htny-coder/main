#!/usr/bin/env python3
"""Empirical fuzzing harness for Lean bounds: CBZ-0 gate G6, fuzz part (stdlib only).

Scientific effect: NONE. This layer falsifies; it never certifies. Layer A
invariant A3 forbids floating-point certification, so every evaluation here is
exact (Python integers and fractions.Fraction) with Lean's own semantics:
truncated `Nat` subtraction, floor `Nat` division, Euclidean `Int` division,
and `x / 0 = 0`. A counterexample it reports is an exact value assignment that
makes the statement false. A PASS means only that no counterexample was found
in the recorded search; it adds no proof. Kernel checking stays the authority.

What it does, per theorem of every Lean package at an exact commit:
  1. Translates the statement (binders plus type) when it lies in the
     evaluable fragment: numerals, `+ - * / % ^`, comparisons, `∧ ∨ ¬ → ↔`,
     `∣`, list membership, `∀`/`∃` over `ℕ ℤ ℚ ℝ` (binder predicates
     `∈ [..]`, `<`, `≤`), type ascriptions, coercion arrows, `min`/`max`/`abs`,
     and non-recursive package definitions in the same fragment (expanded).
  2. Evaluates finite domains exhaustively. Prenex universal variables over
     infinite domains are sampled: pathological edge values first (0, ±1,
     powers of two and ten, tiny and huge rationals, values near hypothesis
     bounds), then seeded random values at many scales.
  3. Reports, per theorem: PASS-exact, PASS-sampled, FAIL (exact
     counterexample), VACUITY_SUSPECT (no sample satisfied the hypotheses;
     invariant A6) or NA (outside the fragment, with the exact reason).

Gate rule (GOVERNANCE.md): no theorem may be indexed CLOSED unless its record
here is PASS-exact, PASS-sampled or NA. Exit status 0 only when there is no
FAIL, no unacknowledged VACUITY_SUSPECT and no internal error.
"""
import argparse
import itertools
import json
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_a_scan  # noqa: E402
import lean_decls  # noqa: E402

SCHEMA = 1
TYPES = {'ℕ': 'nat', 'Nat': 'nat', 'ℤ': 'int', 'Int': 'int', 'ℚ': 'rat', 'Rat': 'rat', 'ℝ': 'real', 'Real': 'real'}
RANK = {'nat': 0, 'int': 1, 'rat': 2, 'real': 3}
MAX_POW_BITS = 4_000_000


class NA(Exception):
    """The statement lies outside the evaluable fragment."""


class TooLarge(Exception):
    """One evaluation exceeds the size limit; the sample is skipped and counted."""


# ---------------------------------------------------------------- tokens

TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>0x[0-9a-fA-F]+|0b[01]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<sym><->|->|<=|>=|!=|&&|\|\||[≠≤≥<>=+\-*/%^∧∨¬→↔∀∃∈∉,()\[\]:∣↑|∑∏!])
""", re.X)


def tokenize(text):
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '«' or layer_a_scan.is_id_first(c):
            name, j = lean_decls.read_name(text, i)
            if name is None:
                raise NA('unreadable name')
            out.append(('ident', name))
            i = j
            continue
        m = TOKEN.match(text, i)
        if not m:
            raise NA(f'unsupported symbol {c!r}')
        if m.lastgroup == 'num':
            out.append(('num', m.group()))
        elif m.lastgroup == 'sym':
            sym = {'<->': '↔', '->': '→', '<=': '≤', '>=': '≥', '!=': '≠', '&&': '∧', '||': '∨'}.get(m.group(), m.group())
            out.append(('sym', sym))
        i = m.end()
    out.append(('end', ''))
    return out


# ---------------------------------------------------------------- parser
# Lean 4 levels: `syntax:L term:Lleft op term:Lright`.
INFIX = {
    '↔': (20, 21, 21), '→': (25, 26, 25), '∨': (30, 31, 30), '∧': (35, 36, 35),
    '=': (50, 51, 51), '≠': (50, 51, 51), '<': (50, 51, 51), '>': (50, 51, 51), '≤': (50, 51, 51),
    '≥': (50, 51, 51), '∈': (50, 51, 51), '∉': (50, 51, 51), '∣': (50, 51, 51),
    '+': (65, 65, 66), '-': (65, 65, 66), '*': (70, 70, 71), '/': (70, 70, 71), '%': (70, 70, 71),
    '^': (75, 76, 75),
}
MAX = 1024
ATOM_START = {'num', 'ident'}


class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i]

    def take(self, kind=None, value=None):
        tok = self.toks[self.i]
        if (kind and tok[0] != kind) or (value is not None and tok[1] != value):
            raise NA(f'expected {value or kind}, found {tok[1]!r}')
        self.i += 1
        return tok

    def parse(self):
        node, _ = self.expr(0)
        if self.peek()[0] != 'end':
            raise NA(f'unparsed trailing token {self.peek()[1]!r}')
        return node

    def is_atom_start(self):
        kind, value = self.peek()
        return kind in ATOM_START or (kind == 'sym' and value in ('(', '[', '↑'))

    def expr(self, minimum):
        left, level = self.prefix()
        while True:
            kind, value = self.peek()
            if (kind, value) == ('sym', '!'):
                self.take()
                left, level = ('app', 'Nat.factorial', [left]), MAX
                continue
            if kind != 'sym' or value not in INFIX:
                break
            lvl, need_left, rhs = INFIX[value]
            if lvl < minimum or level < need_left:
                break
            self.take()
            right, _ = self.expr(rhs)
            left, level = ('bin', value, left, right), lvl
        return left, level

    def prefix(self):
        kind, value = self.peek()
        if kind == 'sym' and value == '-':
            self.take()
            operand, _ = self.expr(75)
            return ('neg', operand), 75
        if kind == 'sym' and value == '¬':
            self.take()
            operand, _ = self.expr(40)
            return ('not', operand), MAX
        if kind == 'sym' and value in ('∀', '∃'):
            return self.quantifier(), 0
        if kind == 'sym' and value in ('∑', '∏'):
            self.take()
            name = self.take('ident')[1]
            if self.peek() == ('sym', '∈') or self.peek() == ('ident', 'in'):
                self.take()
            else:
                raise NA('big operator without a finite index set')
            index_set, _ = self.expr(51)
            self.take('sym', ',')
            body, _ = self.expr(67)
            return ('bigop', value, name, index_set, body), 67
        node = self.atom()
        if node[0] == 'ident':
            args = []
            while self.is_atom_start():
                args.append(self.atom_expr())
            node = ('app', node[1], args) if args else ('var', node[1])
            return node, MAX
        return node, MAX

    def atom(self):
        kind, value = self.peek()
        if kind == 'num':
            self.take()
            return ('num', value)
        if kind == 'ident':
            self.take()
            return ('ident', value)
        if kind == 'sym' and value == '↑':
            self.take()
            return ('coe', self.atom_expr())
        if kind == 'sym' and value == '(':
            self.take()
            inner, _ = self.expr(0)
            if self.peek() == ('sym', ':'):
                self.take()
                type_name = self.type_name()
                self.take('sym', ')')
                return ('asc', inner, type_name)
            self.take('sym', ')')
            return inner
        if kind == 'sym' and value == '|':
            self.take()
            inner, _ = self.expr(0)
            self.take('sym', '|')
            return ('app', 'abs', [inner])
        if kind == 'sym' and value == '[':
            self.take()
            items = []
            if self.peek() != ('sym', ']'):
                while True:
                    item, _ = self.expr(0)
                    items.append(item)
                    if self.peek() == ('sym', ','):
                        self.take()
                        continue
                    break
            self.take('sym', ']')
            return ('list', items)
        raise NA(f'unsupported term start {value!r}')

    def atom_expr(self):
        node = self.atom()
        if node[0] == 'ident':
            return ('var', node[1])
        return node

    def type_name(self):
        kind, value = self.take('ident')
        if value not in TYPES:
            raise NA(f'unsupported type {value}')
        return TYPES[value]

    def quantifier(self):
        _, q = self.take('sym')
        groups = []
        # `∀ (x y : T) (z : U), body`, `∀ x y : T, body`, `∀ x ∈ s, body`, `∀ x < n, body`
        while True:
            kind, value = self.peek()
            if kind == 'sym' and value == '(':
                self.take()
                names = []
                while self.peek()[0] == 'ident':
                    names.append(self.take()[1])
                self.take('sym', ':')
                type_name = self.type_name()
                self.take('sym', ')')
                groups.extend((name, type_name, None) for name in names)
                continue
            if kind == 'ident':
                names = []
                while self.peek()[0] == 'ident':
                    names.append(self.take()[1])
                kind, value = self.peek()
                if (kind, value) == ('sym', ':'):
                    self.take()
                    type_name = self.type_name()
                    groups.extend((name, type_name, None) for name in names)
                elif kind == 'sym' and value in ('∈', '<', '≤'):
                    if len(names) != 1:
                        raise NA('binder predicate on several names')
                    self.take()
                    bound, _ = self.expr(51)
                    groups.append((names[0], None, (value, bound)))
                else:
                    groups.extend((name, None, None) for name in names)
                continue
            break
        self.take('sym', ',')
        body, _ = self.expr(0)
        return ('forall' if q == '∀' else 'exists', groups, body)


def parse(text):
    return Parser(tokenize(text)).parse()


# ---------------------------------------------------------------- evaluation

def to_type(value, type_name):
    if type_name == 'nat':
        if not isinstance(value, int) or value < 0:
            raise NA('value is not a natural number')
        return value
    if type_name == 'int':
        if isinstance(value, Fraction):
            if value.denominator != 1:
                raise NA('non-integer value in Int context')
            return int(value)
        return value
    return Fraction(value)


def literal(text, type_name):
    if text.startswith(('0x', '0b')):
        return to_type(int(text, 0), type_name)
    if re.fullmatch(r'\d+', text):
        return to_type(int(text), type_name)
    if type_name not in ('rat', 'real'):
        raise NA('decimal literal outside ℚ/ℝ')
    return Fraction(text)


def lean_div(a, b, type_name):
    if b == 0:
        return to_type(0, type_name)
    if type_name == 'nat':
        return a // b
    if type_name == 'int':
        r = a % abs(b)
        return (a - r) // b
    return Fraction(a) / Fraction(b)


def lean_mod(a, b, type_name):
    if type_name not in ('nat', 'int'):
        raise NA('% outside ℕ/ℤ')
    if b == 0:
        return a
    return a % abs(b) if type_name == 'int' else a % b


def lean_sub(a, b, type_name):
    if type_name == 'nat':
        return max(a - b, 0)
    return a - b


def lean_pow(base, exponent):
    if exponent < 0:
        raise NA('negative exponent')
    if exponent > 1:
        size = (abs(base.numerator).bit_length() + abs(base.denominator).bit_length()
                if isinstance(base, Fraction) else abs(base).bit_length())
        if size * exponent > MAX_POW_BITS:
            raise TooLarge('power exceeds the size limit')
    return base ** exponent


def _nat_sqrt(n):
    import math
    return math.isqrt(n)


def _factorial(n):
    import math
    if n > 5000:
        raise TooLarge('factorial argument exceeds the size limit')
    return math.factorial(n)


def _choose(n, k):
    import math
    if n > 100000:
        raise TooLarge('choose argument exceeds the size limit')
    return math.comb(n, k)


def _gcd(a, b):
    import math
    return math.gcd(a, b)


def _lcm(a, b):
    import math
    return a * b // math.gcd(a, b) if a and b else 0


# name -> (argument types, result type, exact function)
BUILTINS = {
    'Nat.factorial': (('nat',), 'nat', _factorial),
    'Nat.choose': (('nat', 'nat'), 'nat', _choose),
    'Nat.gcd': (('nat', 'nat'), 'nat', _gcd),
    'Nat.lcm': (('nat', 'nat'), 'nat', _lcm),
    'Nat.succ': (('nat',), 'nat', lambda n: n + 1),
    'Nat.pred': (('nat',), 'nat', lambda n: max(n - 1, 0)),
    'Nat.sqrt': (('nat',), 'nat', _nat_sqrt),
    'Int.natAbs': (('int',), 'nat', abs),
    'Int.toNat': (('int',), 'nat', lambda n: max(n, 0)),
}
INTERVALS = {'Icc': ('≤', '≤'), 'Ico': ('≤', '<'), 'Ioc': ('<', '≤'), 'Ioo': ('<', '<'),
             'Ici': ('≤', None), 'Iic': (None, '≤'), 'Ioi': ('<', None), 'Iio': (None, '<')}


def interval(node):
    """(lower_op, lower, upper_op, upper) for `Set.Icc a b` and friends, or None."""
    if node[0] not in ('app', 'var'):
        return None
    name = node[1]
    for prefix in ('Set.', 'Finset.'):
        if name.startswith(prefix) and name[len(prefix):] in INTERVALS:
            lo_op, hi_op = INTERVALS[name[len(prefix):]]
            args = node[2] if node[0] == 'app' else []
            if len(args) != (lo_op is not None) + (hi_op is not None):
                raise NA(f'{name} arity')
            lo = args[0] if lo_op else None
            hi = args[-1] if hi_op else None
            return lo_op, lo, hi_op, hi, prefix == 'Finset.'
    return None


def finite_set(node, env, ctx, limit, element_type=None):
    """(element type, values) for a finite index set, or None when the set is not finite here."""
    if node[0] == 'list':
        t = element_type or unify(*[leaf_type(x, env, ctx) for x in node[1]])
        return t, [num(x, t, env, ctx) for x in node[1]]
    if node[0] == 'app' and node[1] in ('Finset.range', 'List.range') and len(node[2]) == 1:
        n = num(node[2][0], 'nat', env, ctx)
        if n > limit:
            raise NA('index range exceeds the enumeration limit')
        return 'nat', list(range(n))
    bounds = interval(node)
    if bounds:
        lo_op, lo, hi_op, hi, _ = bounds
        t = element_type or unify(*[leaf_type(x, env, ctx) for x in (lo, hi) if x is not None])
        if t not in ('nat', 'int') or lo is None or hi is None:
            if t == 'nat' and lo is None and hi is not None:
                lo_value = 0
            else:
                return None
        else:
            lo_value = num(lo, t, env, ctx) + (1 if lo_op == '<' else 0)
        hi_value = num(hi, t, env, ctx) - (1 if hi_op == '<' else 0)
        if hi_value - lo_value + 1 > limit:
            raise NA('index range exceeds the enumeration limit')
        return t, list(range(lo_value, hi_value + 1))
    return None


def member(x_node, set_node, env, ctx, limit):
    bounds = interval(set_node)
    if bounds:
        lo_op, lo, hi_op, hi, _ = bounds
        t = unify(leaf_type(x_node, env, ctx), *[leaf_type(b, env, ctx) for b in (lo, hi) if b is not None])
        x = num(x_node, t, env, ctx)
        ok = True
        if lo is not None:
            ok = ok and COMPARE[lo_op](num(lo, t, env, ctx), x)
        if hi is not None:
            ok = ok and COMPARE[hi_op](x, num(hi, t, env, ctx))
        return ok
    fin = finite_set(set_node, env, ctx, limit)
    if fin is None:
        raise NA('membership in an unsupported set')
    t = unify(leaf_type(x_node, env, ctx), fin[0])
    return num(x_node, t, env, ctx) in [to_type(v, t) for v in fin[1]]


class Context:
    def __init__(self, defs, theorem_qualified, fallback=None):
        self.defs = defs
        self.fallback = fallback or {}
        self.namespaces = []
        if theorem_qualified and '.' in theorem_qualified:
            parts = theorem_qualified.split('.')[:-1]
            self.namespaces = ['.'.join(parts[:k]) for k in range(len(parts), 0, -1)]

    def lookup(self, name):
        for prefix in self.namespaces:
            if prefix + '.' + name in self.defs:
                return self.defs[prefix + '.' + name]
        if name in self.defs:
            return self.defs[name]
        matches = [d for q, d in self.defs.items() if q.endswith('.' + name)]
        if len(matches) == 1:
            return matches[0]
        for table in (self.fallback,):
            for prefix in self.namespaces:
                if prefix + '.' + name in table:
                    return table[prefix + '.' + name]
            if name in table:
                return table[name]
            matches = [d for q, d in table.items() if q.endswith('.' + name)]
            if len(matches) == 1:
                return matches[0]
        return None


def leaf_type(node, env, ctx):
    """The maximal known leaf type of an arithmetic tree, or None if only literals."""
    kind = node[0]
    if kind == 'num' or kind == 'coe':
        return None
    if kind == 'asc':
        return node[2]
    if kind == 'var':
        name = node[1]
        if name in env:
            return env[name][0]
        d = resolve(name, ctx)
        if d is None:
            raise NA(f'unknown name {name}')
        if d['type'] == 'prop':
            raise NA(f'proposition {name} used as a number')
        return d['type']
    if kind == 'bigop':
        _, _, name, index_set, body = node
        fin = finite_set(index_set, env, ctx, 10 ** 9)
        if fin is None:
            raise NA('big operator over a non-finite set')
        local = dict(env)
        local[name] = (fin[0], to_type(0, fin[0]))
        return leaf_type(body, local, ctx)
    if kind == 'app' and node[1] in BUILTINS and ctx.lookup(node[1]) is None:
        return BUILTINS[node[1]][1]
    if kind == 'app':
        d = resolve(node[1], ctx)
        if d is not None and d['type'] == 'prop':
            raise NA(f'proposition {node[1]} used as a number')
        if d is None:
            if node[1] in ('min', 'max', 'abs'):
                types = [t for t in (leaf_type(a, env, ctx) for a in node[2]) if t]
                return max(types, key=RANK.get) if types else None
            raise NA(f'unknown function {node[1]}')
        return d['type']
    if kind == 'neg':
        return leaf_type(node[1], env, ctx)
    if kind == 'bin' and node[1] in ('+', '-', '*', '/', '%'):
        types = [t for t in (leaf_type(node[2], env, ctx), leaf_type(node[3], env, ctx)) if t]
        return max(types, key=RANK.get) if types else None
    if kind == 'bin' and node[1] == '^':
        return leaf_type(node[2], env, ctx)
    raise NA(f'not an arithmetic term: {kind} {node[1] if len(node) > 1 else ""}')


def resolve(name, ctx):
    d = ctx.lookup(name)
    if d is None:
        return None
    if 'na' in d:
        raise NA(f'definition {name} is outside the fragment: {d["na"]}')
    return d


def num(node, type_name, env, ctx):
    kind = node[0]
    if kind == 'num':
        return literal(node[1], type_name)
    if kind == 'asc':
        return to_type(num(node[1], node[2], env, ctx), type_name)
    if kind == 'coe':
        inner = node[1]
        t = leaf_type(inner, env, ctx) or 'nat'
        return to_type(num(inner, t, env, ctx), type_name)
    if kind == 'var':
        name = node[1]
        if name in env:
            return to_type(env[name][1], type_name)
        d = resolve(name, ctx)
        if d is None:
            raise NA(f'unknown name {name}')
        if d['params']:
            raise NA(f'function {name} used without arguments')
        return to_type(num(d['body'], d['type'], {}, d['ctx']), type_name)
    if kind == 'bigop':
        _, op, name, index_set, body = node
        fin = finite_set(index_set, env, ctx, 200000)
        if fin is None:
            raise NA('big operator over a non-finite set')
        total = to_type(0 if op == '∑' else 1, type_name)
        for value in fin[1]:
            local = dict(env)
            local[name] = (fin[0], value)
            term = num(body, type_name, local, ctx)
            total = total + term if op == '∑' else total * term
        return total
    if kind == 'app' and node[1] in BUILTINS and ctx.lookup(node[1]) is None:
        arg_types, result_type, function = BUILTINS[node[1]]
        if len(node[2]) != len(arg_types):
            raise NA(f'{node[1]} arity')
        values = [num(a, t, env, ctx) for a, t in zip(node[2], arg_types)]
        return to_type(function(*values), type_name)
    if kind == 'app':
        name, args = node[1], node[2]
        if name in ('min', 'max', 'abs') and ctx.lookup(name) is None:
            values = [num(a, type_name, env, ctx) for a in args]
            if name == 'abs':
                if len(values) != 1:
                    raise NA('abs arity')
                return abs(values[0])
            if len(values) != 2:
                raise NA(f'{name} arity')
            return min(values) if name == 'min' else max(values)
        d = resolve(name, ctx)
        if d is None:
            raise NA(f'unknown function {name}')
        if len(args) != len(d['params']):
            raise NA(f'partial application of {name}')
        local = {}
        for (param, ptype), arg in zip(d['params'], args):
            local[param] = (ptype, num(arg, ptype, env, ctx))
        return to_type(num(d['body'], d['type'], local, d['ctx']), type_name)
    if kind == 'neg':
        if type_name == 'nat':
            raise NA('negation in ℕ')
        return -num(node[1], type_name, env, ctx)
    if kind == 'bin':
        op, left, right = node[1], node[2], node[3]
        if op == '^':
            base = num(left, type_name, env, ctx)
            et = leaf_type(right, env, ctx) or 'nat'
            exponent = num(right, et, env, ctx)
            if isinstance(exponent, Fraction):
                if exponent.denominator != 1:
                    raise NA('non-integer exponent')
                exponent = int(exponent)
            return lean_pow(base, exponent)
        a, b = num(left, type_name, env, ctx), num(right, type_name, env, ctx)
        if op == '+':
            return a + b
        if op == '-':
            return lean_sub(a, b, type_name)
        if op == '*':
            return a * b
        if op == '/':
            return lean_div(a, b, type_name)
        if op == '%':
            return lean_mod(a, b, type_name)
    raise NA(f'unsupported arithmetic node {kind}')


COMPARE = {'=': lambda a, b: a == b, '≠': lambda a, b: a != b, '<': lambda a, b: a < b,
           '>': lambda a, b: a > b, '≤': lambda a, b: a <= b, '≥': lambda a, b: a >= b}


def unify(*types):
    known = [t for t in types if t]
    return max(known, key=RANK.get) if known else 'nat'


def domain(bound, env, ctx, limit):
    """Finite values for a binder predicate, as (type, values), or None if infinite."""
    op, term = bound
    if op == '∈':
        return finite_set(term, env, ctx, limit)
    t = leaf_type(term, env, ctx) or 'nat'
    if t != 'nat':
        return None
    n = num(term, t, env, ctx)
    top = n if op == '<' else n + 1
    if top > limit:
        raise NA('bounded quantifier range exceeds the enumeration limit')
    return 'nat', list(range(top))


def prop(node, env, ctx, limit):
    kind = node[0]
    if kind == 'var' and node[1] in ('True', 'False') and node[1] not in env:
        return node[1] == 'True'
    if kind in ('var', 'app') and node[1] not in env:
        d = resolve(node[1], ctx)
        if d is not None and d['type'] == 'prop':
            args = node[2] if kind == 'app' else []
            if len(args) != len(d['params']):
                raise NA(f'partial application of {node[1]}')
            local = {param: (ptype, num(arg, ptype, env, ctx)) for (param, ptype), arg in zip(d['params'], args)}
            return prop(d['body'], local, d['ctx'], limit)
    if kind == 'not':
        return not prop(node[1], env, ctx, limit)
    if kind in ('forall', 'exists'):
        return quantified(node, env, ctx, limit)
    if kind == 'bin':
        op, left, right = node[1], node[2], node[3]
        if op == '∧':
            return prop(left, env, ctx, limit) and prop(right, env, ctx, limit)
        if op == '∨':
            return prop(left, env, ctx, limit) or prop(right, env, ctx, limit)
        if op == '→':
            return (not prop(left, env, ctx, limit)) or prop(right, env, ctx, limit)
        if op == '↔':
            return prop(left, env, ctx, limit) == prop(right, env, ctx, limit)
        if op in COMPARE:
            t = unify(leaf_type(left, env, ctx), leaf_type(right, env, ctx))
            return COMPARE[op](num(left, t, env, ctx), num(right, t, env, ctx))
        if op in ('∈', '∉'):
            inside = member(left, right, env, ctx, limit)
            return inside if op == '∈' else not inside
        if op == '∣':
            t = unify(leaf_type(left, env, ctx), leaf_type(right, env, ctx))
            if t not in ('nat', 'int'):
                raise NA('∣ outside ℕ/ℤ')
            a, b = num(left, t, env, ctx), num(right, t, env, ctx)
            return b == 0 if a == 0 else b % a == 0
    raise NA(f'not a proposition: {kind}')


def quantified(node, env, ctx, limit):
    kind, groups, body = node
    domains = []
    for name, type_name, bound in groups:
        if bound is not None:
            fin = domain(bound, env, ctx, limit)
            if fin is None:
                raise NA('nested quantifier over an infinite domain')
            domains.append((name, fin[0], fin[1]))
        else:
            raise NA('nested quantifier over an infinite domain')
    total = 1
    for _, _, values in domains:
        total *= max(len(values), 1)
    if total > limit:
        raise NA('nested quantifier range exceeds the enumeration limit')
    for combo in itertools.product(*[values for _, _, values in domains]):
        local = dict(env)
        for (name, t, _), value in zip(domains, combo):
            local[name] = (t, value)
        result = prop(body, local, ctx, limit)
        if kind == 'forall' and not result:
            return False
        if kind == 'exists' and result:
            return True
    return kind == 'forall'


# ---------------------------------------------------------------- statements

def binder_groups(text):
    """Parse theorem binders `(a b : T) {c : U} (h : P)` into variables and hypotheses."""
    variables, hypotheses = [], []
    i, n = 0, len(text)
    pairs = {'(': ')', '{': '}', '⦃': '⦄', '[': ']'}
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c not in pairs:
            raise NA('unsupported binder syntax')
        if c == '[':
            raise NA('instance binder')
        depth, j = 0, i
        while j < n:
            if text[j] in pairs:
                depth += 1
            elif text[j] in pairs.values():
                depth -= 1
                if depth == 0:
                    break
            j += 1
        inner = text[i + 1:j]
        colon = lean_decls.top_colon(inner)
        if colon is None:
            raise NA('binder without a type')
        names = inner[:colon].split()
        type_text = inner[colon + 1:].strip()
        if type_text in TYPES:
            variables.extend((name, TYPES[type_text], None) for name in names)
        else:
            node = parse(type_text)
            if is_type_expression(node):
                raise NA(f'unsupported binder type {type_text[:40]}')
            hypotheses.append(node)
        i = j + 1
    return variables, hypotheses


def is_type_expression(node):
    """True when a binder type mentions a type name outside an ascription (`ℝ → ℝ`, `Set ℝ`)."""
    if node[0] == 'var':
        return node[1] in TYPES or node[1] in ('Prop', 'Type')
    if node[0] == 'asc':
        return False
    for child in node[1:]:
        if isinstance(child, tuple) and is_type_expression(child):
            return True
        if isinstance(child, list) and any(isinstance(c, tuple) and is_type_expression(c) for c in child):
            return True
    return False


def prenex(statement):
    """Split `∀ ..., H → ... → C` into (variables, hypotheses, conclusion)."""
    variables, hypotheses = [], []
    node = statement
    while True:
        if node[0] == 'forall':
            variables.extend(node[1])
            node = node[2]
            continue
        if node[0] == 'bin' and node[1] == '→' and node[3] != ('var', 'False'):
            hypotheses.append(node[2])
            node = node[3]
            continue
        return variables, hypotheses, node


NAT_EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 24, 31, 32, 33, 63, 64, 65, 99, 100, 101, 127,
             128, 129, 255, 256, 1000, 1023, 1024, 10 ** 6, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1, 2 ** 32, 2 ** 32 + 1,
             2 ** 32 + 7, 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1, 2 ** 64, 2 ** 64 + 1, 10 ** 18, 10 ** 30]
RAT_EDGES = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3), Fraction(-1, 3),
             Fraction(2), Fraction(-2), Fraction(3), Fraction(-3), Fraction(1, 4), Fraction(3, 4), Fraction(7, 3),
             Fraction(22, 7), Fraction(-355, 113), Fraction(1, 10), Fraction(-1, 10), Fraction(10), Fraction(-10),
             Fraction(1, 10 ** 6), Fraction(-1, 10 ** 6), Fraction(10 ** 6), Fraction(-10 ** 6), Fraction(1, 10 ** 12),
             Fraction(10 ** 12), Fraction(-10 ** 12), Fraction(1, 2 ** 20), Fraction(2 ** 60), Fraction(-2 ** 60),
             Fraction(10 ** 9 + 1, 10 ** 9), Fraction(-(10 ** 9 + 1), 10 ** 9)]


def edges(type_name):
    if type_name == 'nat':
        return NAT_EDGES
    if type_name == 'int':
        return sorted(set(NAT_EDGES) | {-x for x in NAT_EDGES})
    return RAT_EDGES


def random_value(type_name, rng):
    scale = 10 ** rng.choice([0, 1, 2, 3, 6, 9, 18])
    if type_name == 'nat':
        return rng.randint(0, scale)
    if type_name == 'int':
        return rng.randint(-scale, scale)
    den = rng.choice([1, 2, 3, 4, 7, 10, 1000, 10 ** 6])
    return Fraction(rng.randint(-scale * den, scale * den), den)


def hypothesis_points(hypotheses, names):
    """Values that sit on or next to single-variable comparison bounds in the hypotheses."""
    points = {name: {} for name in names}
    for h in hypotheses:
        stack = [h]
        while stack:
            node = stack.pop()
            if node[0] == 'bin' and node[1] in ('∧', '∨', '↔', '→'):
                stack.extend([node[2], node[3]])
            elif node[0] == 'bin' and node[1] in COMPARE:
                for var, other in ((node[2], node[3]), (node[3], node[2])):
                    if var[0] == 'var' and var[1] in points and other[0] in ('num', 'neg', 'bin', 'asc'):
                        points[var[1]][repr(other)] = other
            elif node[0] == 'bin' and node[1] == '∈' and node[2][0] == 'var' and node[2][1] in points:
                bounds = interval(node[3])
                if bounds:
                    for b in (bounds[1], bounds[3]):
                        if b is not None:
                            points[node[2][1]][repr(b)] = b
    return {name: [v for _, v in sorted(found.items())] for name, found in points.items()}


def rational_sqrt(value):
    """The exact rational square root of a non-negative Fraction, or None."""
    import math
    if value < 0:
        return None
    num_root, den_root = math.isqrt(value.numerator), math.isqrt(value.denominator)
    if num_root * num_root == value.numerator and den_root * den_root == value.denominator:
        return Fraction(num_root, den_root)
    return None


def equalities(hypotheses):
    out, stack = [], list(hypotheses)
    while stack:
        node = stack.pop()
        if node[0] == 'bin' and node[1] == '∧':
            stack.extend([node[2], node[3]])
        elif node[0] == 'bin' and node[1] == '=':
            out.append(node)
    return out


def repair(env, hypotheses, infinite, ctx, rng):
    """Propose a witness for the hypotheses by solving one equality for one ℚ/ℝ variable.

    The equality is fitted as a polynomial of degree at most 2 in that variable
    from exact values at t = 0, 1, -1 and checked at t = 2 and t = 1/2 before an
    exact rational root is used. The caller re-evaluates every hypothesis
    exactly, so a wrong fit can only waste a sample, never admit a false witness.
    """
    eqs = equalities(hypotheses)
    names = [name for name, t in infinite if t in ('rat', 'real')]
    rng.shuffle(names)
    for eq in eqs:
        for name in names:
            t_name = env[name][0]

            def f(value):
                local = dict(env)
                local[name] = (t_name, Fraction(value))
                t = unify(leaf_type(eq[2], local, ctx), leaf_type(eq[3], local, ctx))
                if t not in ('rat', 'real'):
                    raise NA('equality outside ℚ/ℝ')
                return Fraction(num(eq[2], t, local, ctx)) - Fraction(num(eq[3], t, local, ctx))
            try:
                c0, f1, fm1 = f(0), f(1), f(-1)
                c2 = (f1 + fm1) / 2 - c0
                c1 = (f1 - fm1) / 2
                if f(2) != c0 + 2 * c1 + 4 * c2 or f(Fraction(1, 2)) != c0 + c1 / 2 + c2 / 4:
                    continue
                roots = []
                if c2 == 0:
                    if c1 != 0:
                        roots = [-c0 / c1]
                else:
                    root = rational_sqrt(c1 * c1 - 4 * c2 * c0)
                    if root is not None:
                        roots = [(-c1 + root) / (2 * c2), (-c1 - root) / (2 * c2)]
                if roots:
                    candidate = dict(env)
                    candidate[name] = (t_name, rng.choice(roots))
                    return candidate
            except (NA, TooLarge, ZeroDivisionError):
                continue
    return None


def check_theorem(decl, ctx, samples, seed, limit):
    """Return the fuzz record for one theorem declaration."""
    record = {'status': None}
    try:
        variables, hypotheses = binder_groups(decl['binders'])
        statement = parse(decl['type'])
        more_vars, more_hyps, conclusion = prenex(statement)
        variables = variables + more_vars
        hypotheses = hypotheses + more_hyps
        finite, infinite = [], []
        for name, type_name, bound in variables:
            if bound is not None:
                fin = domain(bound, {}, ctx, limit)
                if fin is None:
                    # An infinite domain such as `x ∈ Set.Icc a b` over ℝ: sample x and keep the bound as a hypothesis.
                    op, term = bound
                    if op == '∈':
                        bounds = interval(term)
                        if not bounds:
                            raise NA('prenex binder predicate over an unsupported set')
                        inferred = type_name or unify(*[leaf_type(b, {}, ctx) for b in (bounds[1], bounds[3]) if b is not None])
                    else:
                        inferred = type_name or leaf_type(term, {}, ctx)
                    if inferred not in ('int', 'rat', 'real'):
                        raise NA(f'cannot infer the type of binder {name}')
                    infinite.append((name, inferred))
                    hypotheses.append(('bin', op, ('var', name), term))
                    continue
                finite.append((name, fin[0], fin[1]))
            elif type_name is None:
                premise = next((h for h in hypotheses if h[0] == 'bin' and h[1] == '∈' and h[2] == ('var', name)), None)
                fin = finite_set(premise[3], {}, ctx, limit) if premise is not None else None
                if fin is None:
                    raise NA(f'untyped binder {name}')
                finite.append((name, fin[0], fin[1]))
            else:
                infinite.append((name, type_name))
        rng = random.Random(f'{seed}:{decl["qualified"]}')
        boundary = hypothesis_points(hypotheses, [n for n, _ in infinite])

        def assignments():
            fixed = [[]] if not finite else [list(c) for c in itertools.product(*[v for _, _, v in finite])]
            if len(fixed) > limit:
                raise NA('finite domain exceeds the enumeration limit')
            if not infinite:
                for combo in fixed:
                    yield {name: (t, v) for (name, t, _), v in zip(finite, combo)}
                return
            for k in range(samples):
                combo = fixed[k % len(fixed)]
                env = {name: (t, v) for (name, t, _), v in zip(finite, combo)}
                for index, (name, t) in enumerate(infinite):
                    if k < len(edges(t)) * 2 and index == k % max(len(infinite), 1):
                        value = edges(t)[(k // 2) % len(edges(t))]
                    elif k < 4:
                        value = to_type([0, 1, 2, 3][k], t)
                    elif boundary[name] and rng.random() < 0.3:
                        try:
                            centre = num(rng.choice(boundary[name]), t, env, ctx)
                            value = centre + to_type(rng.choice([0, 0, 1, -1]), t) if t != 'nat' or centre > 0 else centre
                            if t in ('rat', 'real') and rng.random() < 0.5:
                                value = centre + Fraction(rng.choice([1, -1]), rng.choice([2, 10, 10 ** 6]))
                        except (NA, TooLarge):
                            value = random_value(t, rng)
                    elif rng.random() < 0.35:
                        value = rng.choice(edges(t))
                    else:
                        value = random_value(t, rng)
                    env[name] = (t, to_type(value, t))
                yield env

        hypothesis_hits = refuted_hypotheses = skipped = 0
        for env in assignments():
            try:
                if hypotheses and not all(prop(h, env, ctx, limit) for h in hypotheses):
                    repaired = repair(env, hypotheses, infinite, ctx, rng) if infinite else None
                    if repaired is None or not all(prop(h, repaired, ctx, limit) for h in hypotheses):
                        refuted_hypotheses += 1
                        continue
                    env = repaired
                holds = prop(conclusion, env, ctx, limit)
            except TooLarge:
                skipped += 1
                continue
            hypothesis_hits += 1
            if not holds:
                record.update(status='FAIL', counterexample={k: str(v[1]) for k, v in sorted(env.items())})
                break
        else:
            if hypothesis_hits == 0 and refuted_hypotheses == 0:
                raise NA('every evaluation exceeded the size limit')
            if hypothesis_hits == 0:
                record['status'] = 'VACUITY_SUSPECT'
            else:
                record['status'] = 'PASS-sampled' if infinite else 'PASS-exact'
        evaluated = hypothesis_hits + refuted_hypotheses
        record.update(evaluated=evaluated, hypothesis_hits=hypothesis_hits, skipped_too_large=skipped,
                      sampled_variables=[n for n, _ in infinite], seed=seed if infinite else None)
    except NA as exc:
        record = {'status': 'NA', 'reason': str(exc)}
    except RecursionError:
        record = {'status': 'NA', 'reason': 'expression nesting too deep'}
    except Exception as exc:  # fail closed: an internal fault is an error, never a pass or an NA
        record = {'status': 'ERROR', 'reason': f'{type(exc).__name__}: {exc}'}
    return record


# ---------------------------------------------------------------- packages

def definition(decl):
    """A package definition as an expandable macro, or a record of why it is not one."""
    out = {'qualified': decl['qualified']}
    try:
        if decl['kind'] not in ('def', 'abbrev'):
            raise NA(f'{decl["kind"]} is not a definition')
        if decl['value'] is None:
            raise NA('pattern-matching or structure definition')
        type_text = decl['type'].strip()
        if type_text in TYPES:
            out['type'] = TYPES[type_text]
        elif type_text == 'Prop':
            out['type'] = 'prop'
        else:
            raise NA(f'unsupported definition type {type_text[:40]}')
        params = []
        if decl['binders']:
            variables, hypotheses = binder_groups(decl['binders'])
            if hypotheses:
                raise NA('definition with hypothesis binders')
            params = [(name, t) for name, t, _ in variables]
        short = decl['name'].split('.')[-1]
        body_tokens = tokenize(decl['value'])
        if any(tok == ('ident', short) or tok == ('ident', decl['name']) for tok in body_tokens):
            raise NA('recursive definition')
        out['params'] = params
        out['body'] = Parser(body_tokens).parse()
    except NA as exc:
        return {'qualified': decl['qualified'], 'na': str(exc)}
    return out


def fuzz_repository(name, path, rev, samples, seed, limit):
    commit, files, read_errors = layer_a_scan.read_repository(path, rev)
    tree = set(layer_a_scan.git(path, 'ls-tree', '-r', '--name-only', '--full-tree', commit).decode().split('\n'))
    packages = {}
    errors = [{'repository': name, **e} for e in read_errors]
    for file_path, raw in sorted(files.items()):
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            errors.append({'repository': name, 'path': file_path, 'error': 'not UTF-8'})
            continue
        root = lean_decls.package_root(file_path, tree)
        decls, lex_errors = lean_decls.declarations(text)
        for e in lex_errors:
            errors.append({'repository': name, 'path': file_path, 'error': e[0]})
        packages.setdefault(root, []).append((file_path, decls))
    records = []
    package_defs, everywhere, bodies = {}, {}, {}
    for root, entries in packages.items():
        defs = {}
        for file_path, decls in entries:
            for decl in decls:
                if decl['kind'] in ('def', 'abbrev') and decl['qualified']:
                    entry = definition(decl)
                    defs.setdefault(decl['qualified'], entry)
                    signature = repr((decl['binders'], decl['type'], decl['value']))
                    if decl['qualified'] in bodies and bodies[decl['qualified']] != signature:
                        everywhere[decl['qualified']] = {'qualified': decl['qualified'],
                                                         'na': 'defined differently in several packages'}
                    elif decl['qualified'] not in bodies:
                        bodies[decl['qualified']] = signature
                        everywhere[decl['qualified']] = entry
        package_defs[root] = defs
    for root, defs in package_defs.items():
        for entry in defs.values():
            if 'na' not in entry and 'ctx' not in entry:
                entry['ctx'] = Context(defs, entry['qualified'], everywhere)
    for entry in everywhere.values():
        if 'na' not in entry and 'ctx' not in entry:
            entry['ctx'] = Context(everywhere, entry['qualified'])
    for root, entries in sorted(packages.items(), key=lambda kv: (kv[0] is None, kv[0] or '')):
        defs = package_defs[root]
        for file_path, decls in entries:
            for decl in decls:
                if decl['kind'] != 'theorem' or not decl['qualified']:
                    continue
                result = check_theorem(decl, Context(defs, decl['qualified'], everywhere), samples, seed, limit)
                records.append({'repository': name, 'commit': commit, 'package': root, 'path': file_path,
                                'theorem': decl['qualified'], 'line': decl['line'],
                                'signature_sha256': decl['signature_sha256'], **result})
    return {'name': name, 'commit': commit, 'lean_files': len(files)}, records, errors


def fuzz_text(text, samples=300, seed='cbz0', limit=200000):
    """Fuzz every theorem of one Lean source text as a single package: {qualified name: record}."""
    decls, _ = lean_decls.declarations(text)
    defs = {}
    for decl in decls:
        if decl['kind'] in ('def', 'abbrev') and decl['qualified']:
            defs.setdefault(decl['qualified'], definition(decl))
    for entry in defs.values():
        if 'na' not in entry:
            entry['ctx'] = Context(defs, entry['qualified'])
    return {d['qualified']: check_theorem(d, Context(defs, d['qualified']), samples, seed, limit)
            for d in decls if d['kind'] == 'theorem' and d['qualified']}


def load_acknowledged(path):
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA or not isinstance(data.get('acknowledged'), list):
        raise ValueError('invalid acknowledged-vacuity registry')
    out = {}
    for row in data['acknowledged']:
        if set(row) != {'repository', 'theorem', 'signature_sha256', 'reason'} or len(row['reason'].strip()) < 20:
            raise ValueError('invalid acknowledged-vacuity entry')
        out[(row['repository'], row['theorem'], row['signature_sha256'])] = row['reason']
    return out


def run(repositories, samples, seed, limit, acknowledged):
    report = {'schema': SCHEMA, 'scientific_effect': 'NONE', 'certifies': False, 'samples': samples, 'seed': seed,
              'enumeration_limit': limit, 'repositories': [], 'records': [], 'errors': []}
    for name, path, rev in repositories:
        repo, records, errors = fuzz_repository(name, path, rev, samples, seed, limit)
        report['repositories'].append(repo)
        report['records'].extend(records)
        report['errors'].extend(errors)
    counts = {}
    for record in report['records']:
        key = (record['repository'], record['theorem'], record['signature_sha256'])
        if record['status'] == 'VACUITY_SUSPECT' and key in acknowledged:
            record['acknowledged'] = acknowledged[key]
        counts[record['status']] = counts.get(record['status'], 0) + 1
    failing = [r for r in report['records'] if r['status'] in ('FAIL', 'ERROR')
               or (r['status'] == 'VACUITY_SUSPECT' and 'acknowledged' not in r)]
    report['summary'] = {'theorems': len(report['records']), 'by_status': dict(sorted(counts.items())),
                         'blocking': len(failing), 'errors': len(report['errors'])}
    report['verdict'] = 'PASS' if not failing and not report['errors'] else 'FAIL'
    return report


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', action='append', type=layer_a_scan.parse_repo, default=[],
                        help='NAME=PATH[@REV]; repeatable. Default: main=<this repository>@HEAD')
    parser.add_argument('--samples', type=int, default=2000, help='samples per theorem with infinite variables')
    parser.add_argument('--seed', default='cbz0')
    parser.add_argument('--limit', type=int, default=200000, help='finite enumeration limit')
    parser.add_argument('--acknowledged', default=str(here / 'fuzz_acknowledged_vacuity.json'))
    parser.add_argument('--json', help='write the full report to this file')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    repositories = args.repo or [('main', here.parent.parent, 'HEAD')]
    try:
        report = run(repositories, args.samples, args.seed, args.limit, load_acknowledged(args.acknowledged))
    except (ValueError, RuntimeError, OSError) as exc:
        print('FUZZ_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    if not args.quiet:
        for repo in report['repositories']:
            print(f"fuzzed {repo['name']} @ {repo['commit']}: {repo['lean_files']} Lean files")
    for record in report['records']:
        if record['status'] in ('FAIL', 'VACUITY_SUSPECT', 'ERROR'):
            print(f"{record['status']}: {record['repository']}:{record['path']}:{record['line']} {record['theorem']} "
                  + json.dumps({k: record[k] for k in ('counterexample', 'acknowledged') if k in record}, sort_keys=True))
    for error in report['errors']:
        print('ERROR: ' + json.dumps(error, sort_keys=True))
    print('FUZZ ' + report['verdict'] + ' ' + json.dumps(report['summary'], sort_keys=True))
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
