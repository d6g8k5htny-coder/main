#!/usr/bin/env python3
"""Source and execution checker for the proof packet
`specifications/lean_statement_proofs_20261011` (standard library only).

    python3 check.py [source]        source mode: packet identity against git HEAD, the lexical
                                     refusal list on every module, the theorem inventory (every
                                     non-aux theorem must have as its type a registered `def … :
                                     Prop` of the statement register, applied to exactly its own
                                     parameters), docstring labels, root imports, dependency lock,
                                     byte-exact lakefile.toml, toolchain pin
    python3 check.py --execute       the same, then the exact Lean version, a fresh `lake build`,
                                     `leanchecker`, `#check` and `#print axioms` of every
                                     inventoried theorem, the environment audit (every constant
                                     Lean added for the modules, classified by its ConstantInfo
                                     and its type after `forallTelescope`), real negative controls,
                                     a receipt under .lake/evidence/
    python3 check.py --allow-dirty   local development only: the clean-worktree requirement is
                                     skipped; a receipt then records the worktree as unverified
                                     and must not be trusted as evidence about any commit. With
                                     LEAN_PROOF_SKIP_LEANCHECKER=1 (honoured only together with
                                     --allow-dirty, never in CI) the leanchecker replay, which
                                     needs roughly 14 GB of memory on the Mathlib-importing
                                     environment, is skipped and the receipt records the skip

Scientific effect: NONE. The packet holds proofs of registered statements: theorems whose types
are `def … : Prop` constants of `specifications/lean_statements_20261010/REGISTER.json`. A green
check means that those theorems were accepted by the Lean kernel at the pinned toolchain and
Mathlib, depend on no axiom outside propext, Classical.choice and Quot.sound, and that every
non-aux theorem of the modules proves a registered Prop applied to exactly its own parameters.
That is kernel evidence about those Props only. It is not a proof of any informal claim, not an
alignment review (alignment is PENDING_INDEPENDENT_REVIEW), not an acceptance, and it changes no
label of the statement register, which this packet never edits. Nothing here can raise any claim.

Two independent stages enforce the theorem shape. The source scan is a fail-closed pre-Lean stage
over comment-stripped text: whole-token refusals (sorry, axiom, native_decide, unsafe, opaque, …),
`set_option maxHeartbeats` above 400000 without a justifying comment, declaration keywords off
column 0, and a lexical parse of every theorem header (binders, result type, head constant resolved
through the opened namespaces, explicit binders passed as arguments). It is not a Lean elaborator.
The authority is the `--execute` environment audit: Lean's own `ConstantInfo` of every constant
the modules add (`.lake/evidence/Environment.lean`) refuses axioms, opaque and unsafe constants,
definitions whose type is a proposition, any transitive axiom outside the allowed three, and any
non-aux theorem whose type after `forallTelescope` is not a registered Prop constant applied to
exactly the telescope's variables (an extra hypothesis or a narrowed argument is a finding about
the statement, never a theorem of it); and it requires Lean's theorem inventory to equal the scan's.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PACKAGE = 'specifications/lean_statement_proofs_20261011'
STATEMENT_PACKAGE = 'specifications/lean_statements_20261010'
STATEMENT_DIR = '../lean_statements_20261010'  # the path dependency, relative to the packet
REPOSITORY = 'd6g8k5htny-coder/Math-'
WORKFLOW = '.github/workflows/lean-specification-proofs.yml'
ROOT_MODULE = 'SpecificationProofs'
STATEMENT_ROOT_MODULE = 'Specifications'
NAMESPACE = 'UniversalLaw.Proofs.'
STATEMENT_NAMESPACE = 'UniversalLaw.Spec.'
AUX = 'Aux'
CONTROL_NS = 'UniversalLaw.Proofs.Control'
TOOLCHAIN = 'leanprover/lean4:v4.34.1'
LEAN_COMMIT = '5045d0056413266e57c625dcd7c365b10e377c52'
MATHLIB_REV = 'd13f23b723b8a846827a245b89c10fc7d3f11612'
LOCK_NAME = 'universal_law_specification_proofs'
STATEMENT_LOCK_NAME = 'universal_law_specifications'
REGISTER_OBJECT = 'LEAN-STATEMENT-REGISTER-20261010-v1'
RECEIPT_OBJECT = 'LEAN-PROOF-RECEIPT-20261011-v1'
ALLOWED_AXIOMS = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
MAX_HEARTBEATS = 400000
LABELS = ('Proves:', 'Method:', 'Does not establish:')
LEAN_OPTIONS_LINE = 'leanOptions = { autoImplicit = false }'
PATH_DEPENDENCY = {'type': 'path', 'name': STATEMENT_LOCK_NAME, 'dir': STATEMENT_DIR, 'manifestFile': 'lake-manifest.json',
                   'inherited': False, 'configFile': 'lakefile.toml'}
# The packet's lakefile.toml, byte-exactly: Lake accepts options such as `moreLeanArgs =
# ["-Ddebug.skipKernelTC=true"]` at package or library level and a "contains these lines" check
# would let them through, so the whole file is pinned like lean-toolchain is.
LAKEFILE = (f'name = "{LOCK_NAME}"\nversion = "0.1.0"\ndefaultTargets = ["{ROOT_MODULE}"]\n\n'
            f'[[require]]\nname = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib4.git"\nrev = "{MATHLIB_REV}"\n\n'
            f'[[require]]\nname = "{STATEMENT_LOCK_NAME}"\npath = "{STATEMENT_DIR}"\n\n'
            f'[[lean_lib]]\nname = "{ROOT_MODULE}"\n{LEAN_OPTIONS_LINE}\n')
SKIP_LEANCHECKER_ENV = 'LEAN_PROOF_SKIP_LEANCHECKER'
# Lexical refusal list (DESIGN2, theorem shape): proof escapes (sorry and its axiom sorryAx, admit,
# axiom, native_decide and `decide +native` through the token `native`, its axiom ofReduceBool,
# trustCompiler), constants the kernel does not check (unsafe, opaque, partial, implemented_by,
# extern), options that skip or fake checking (debug.skipKernelTC, debug.byAsSorry),
# metaprogramming and syntax that can add declarations under other names (macro, syntax, elab,
# notation, run_cmd, initialize, export, alias, irreducible_def), `include` (forces a section
# hypothesis into every theorem) and `_root_` (escapes the module namespace). Matched as whole
# identifier tokens of the comment-stripped code, so a docstring may mention them. Not an
# elaboration check: the environment audit in --execute is the Lean-side authority.
REFUSED_TOKENS = ('sorry', 'sorryAx', 'admit', 'axiom', 'native_decide', 'native', 'ofReduceBool', 'trustCompiler', 'unsafe',
                  'opaque', 'partial', 'implemented_by', 'extern', 'skipKernelTC', 'byAsSorry', 'macro', 'macro_rules', 'syntax',
                  'elab', 'elab_rules', 'notation', 'infix', 'infixl', 'infixr', 'prefix', 'postfix', 'run_cmd', 'run_tac',
                  'run_elab', 'initialize', 'builtin_initialize', 'declare_syntax_cat', 'export', 'alias',
                  'irreducible_def', 'include', '_root_')
REFUSED_OPTION_PREFIXES = ('debug.', 'bootstrap.', 'compiler.')
REFUSED_OPTIONS = ('autoImplicit', 'relaxedAutoImplicit')
DECLARATION_KEYWORDS = ('theorem', 'lemma', 'def', 'abbrev', 'instance', 'structure', 'inductive', 'example')
MODIFIERS = ('private', 'protected', 'noncomputable')
SCOPE_KEYWORDS = ('namespace', 'section', 'end', 'variable', 'universe')
# Lean commands are not indentation-sensitive, so a declaration or scope keyword anywhere but at
# the start of its own line (after attributes and modifiers) is refused. `open … in` and
# `set_option … in` may chain a command onto the NEXT line and may also appear inside tactic blocks,
# so they are not policed here; the environment audit sees whatever Lean adds.
COLUMN0_KEYWORDS = DECLARATION_KEYWORDS + MODIFIERS + SCOPE_KEYWORDS
ALLOWED_COMMANDS = frozenset(COLUMN0_KEYWORDS) | {'import', 'open', 'set_option', 'attribute', 'deriving', 'omit'}
CONTINUATIONS = ('|', 'termination_by', 'decreasing_by')
COMMAND_KEYWORD = re.compile(r"(?<![A-Za-z0-9_.'])(" + '|'.join(sorted(COLUMN0_KEYWORDS)) + r")(?![A-Za-z0-9_'!?])")
IDENT = r"[^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*"
IDENT_RE = re.compile(IDENT)
DECL_LINE = re.compile(r'^((?:@\[[^\]\n]*\]\s*)*)((?:(?:private|protected|noncomputable)\s+)*)'
                       r'(theorem|lemma|def|abbrev|instance|structure|inductive|example)(?:\s+([^\s(\[{⦃:]+))?(.*)$')
IN_TOKEN = re.compile(r"(?<![A-Za-z0-9_.'])in(?![A-Za-z0-9_'])")
SET_OPTION = re.compile(r"(?<![A-Za-z0-9_.'])set_option\s+([A-Za-z_][\w.]*)\s+(\S+)")
HASH_COMMAND = re.compile(r'#[A-Za-z_]')
PLACEHOLDER = re.compile(r'<<[^<>\n]*>>')
FIXED_FILES = ('.gitignore', 'README.md', 'SCOPE.md', 'check.py', 'test_check.py', 'replay.sh', 'lakefile.toml',
               'lean-toolchain', 'lake-manifest.json', ROOT_MODULE + '.lean')
SKIP_DIRS = frozenset({'.lake', '__pycache__'})
HEX40 = re.compile(r'[0-9a-f]{40}')
AXIOM_LINE = re.compile(r"'([^'\s]+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
AUDIT_LINE = re.compile(r'\bAUDIT (\S+) (\S+)(?: (\S+))?$', re.M)
AUDIT_END = re.compile(r'\bAUDIT END (\d+) (\d+) (\d+)$', re.M)
AUDIT_CLASSES = frozenset({'theorem', 'aux_theorem', 'auxiliary', 'generated'})
LEAN_ERROR = re.compile(r':\d+:\d+: error(?:\([^)]*\))?:')  # Lean 4.34 may tag errors: `error(lean.unknownIdentifier):`
MODULE_WARNING = re.compile(r'warning: \S*' + ROOT_MODULE + r'/[A-Za-z0-9_]+\.lean:\d+:\d+:|' + ROOT_MODULE + r'/[A-Za-z0-9_]+\.lean:\d+:\d+: warning')
BRACKETS = {'(': ')', '[': ']', '{': '}', '⦃': '⦄', '⟨': '⟩'}
CLOSERS = frozenset(BRACKETS.values())
BINDER_KINDS = {'(': 'explicit', '{': 'implicit', '⦃': 'strict', '[': 'instance'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(text: str):
    """Strict JSON: duplicate keys, non-integer numbers and NaN/Infinity are refused."""
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, 'duplicate JSON key: ' + key)
            obj[key] = value
        return obj

    def reject(token):
        raise ValueError('non-integer or non-finite JSON number refused: ' + token)
    return json.loads(text, object_pairs_hook=unique, parse_float=reject, parse_constant=reject)


def safe_path(root: Path, relative) -> Path:
    require(isinstance(relative, str) and relative and '\\' not in relative, f'invalid path: {relative!r}')
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and '..' not in pure.parts and str(pure) == relative, f'unsafe path: {relative}')
    path = root
    for part in pure.parts:
        path = path / part
        require(not path.is_symlink(), f'symlink refused: {relative}')
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f'missing or escaping file: {relative}')
    return path


# ----------------------------------------------------------------------------- Lean source scan

def lex(text: str):
    """Comment-stripped code with every newline preserved, plus the docstring blocks `/-- … -/`
    as (first_line, last_line, raw_text). Nested block comments are supported; an unterminated
    block comment, string literal or guillemet name is refused. A string literal or guillemet
    name outside a comment is copied verbatim into the code, so a `/-`, `-/` or `--` inside it
    never opens or closes a comment."""
    out, docs, i, n, depth, line, start = [], [], 0, len(text), 0, 0, None
    while i < n:
        two = text[i:i + 2]
        if depth:
            if two == '/-':
                depth, i = depth + 1, i + 2
            elif two == '-/':
                depth, i = depth - 1, i + 2
                if depth == 0 and start is not None:
                    docs.append((start[0], line, text[start[1]:i]))
                    start = None
            else:
                if text[i] == '\n':
                    out.append('\n')
                    line += 1
                else:
                    out.append(' ')
                i += 1
        elif two == '/-':
            depth = 1
            if text.startswith('/--', i):
                start = (line, i)
            i += 2
        elif two == '--':
            end = text.find('\n', i)
            i = n if end < 0 else end
        elif text[i] in '"«':
            if text[i] == '"':
                j = i + 1
                while j < n and text[j] != '"':
                    j += 2 if text[j] == '\\' else 1
            else:
                j = text.find('»', i)
                j = n if j < 0 else j
            require(j < n, 'unterminated string literal or guillemet name')
            segment = text[i:j + 1]
            out.append(segment)
            line += segment.count('\n')
            i = j + 1
        else:
            if text[i] == '\n':
                line += 1
            out.append(text[i])
            i += 1
    require(depth == 0, 'unterminated block comment')
    return ''.join(out), docs


def refused_tokens(code: str) -> list:
    words = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', code))
    return sorted(set(REFUSED_TOKENS) & words)


def depth_zero(text: str) -> str:
    """`text` with everything inside brackets blanked (same length), for depth-zero tests."""
    depth, out = 0, []
    for ch in text:
        if ch in BRACKETS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
        out.append(ch if depth == 0 and ch not in CLOSERS else ' ')
    return ''.join(out)


def header(lines: list, i: int, after: str) -> str:
    """Declaration header text: from `after` (the rest of line i past the name) up to the first
    bracket-depth-zero `:=`, `where` or leading `| `, or up to the next column-0 line."""
    parts, depth, k, seg = [], 0, i, after
    while True:
        p = 0
        while p < len(seg):
            c = seg[p]
            if c in BRACKETS:
                depth += 1
            elif c in CLOSERS:
                depth -= 1
            elif depth == 0:
                if seg.startswith(':=', p):
                    parts.append(seg[:p])
                    return ''.join(parts)
                if (p == 0 or seg[p - 1].isspace()) and re.match(r'where(?![A-Za-z0-9_])', seg[p:]):
                    parts.append(seg[:p])
                    return ''.join(parts)
            p += 1
        parts.append(seg + '\n')
        k += 1
        if k >= len(lines):
            return ''.join(parts)
        seg = lines[k]
        if (seg and not seg[0].isspace()) or (depth == 0 and seg.lstrip().startswith('| ')):
            return ''.join(parts)


def first_colon(text: str):
    """Index of the first bracket-depth-zero `:` that is not part of `:=` or `::`, or None."""
    depth = 0
    for p, c in enumerate(text):
        if c in BRACKETS:
            depth += 1
        elif c in CLOSERS:
            depth -= 1
        elif c == ':' and depth == 0 and text[p:p + 2] != ':=' and (p == 0 or text[p - 1] != ':') and text[p + 1:p + 2] != ':':
            return p
    return None


def parse_binders(text: str) -> list:
    """Binder groups of a header: [{kind: explicit|implicit|strict|instance, names: [...]}].
    A bare identifier is an explicit binder; an instance binder may be unnamed."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in BRACKETS:
            require(c in BINDER_KINDS, f'unexpected binder bracket {c!r} in a theorem header')
            depth, j = 1, i + 1
            while j < n and depth:
                if text[j] in BRACKETS:
                    depth += 1
                elif text[j] in CLOSERS:
                    depth -= 1
                j += 1
            require(depth == 0, 'unbalanced brackets in a theorem header')
            inner = text[i + 1:j - 1]
            colon = first_colon(inner)
            if colon is None:
                names = [] if BINDER_KINDS[c] == 'instance' else inner.split()
            else:
                names = inner[:colon].split()
            out.append({'kind': BINDER_KINDS[c], 'names': names})
            i = j
        else:
            m = IDENT_RE.match(text, i)
            require(m is not None, f'unparsable binder text {text[i:i + 24]!r} in a theorem header')
            out.append({'kind': 'explicit', 'names': [m.group()]})
            i = m.end()
    return out


def result_head(type_text: str):
    """(head constant name, [argument names]) of a result type written as a constant applied to
    bare parameter names; anything else (a quantifier, an implication, a relation, a bracketed
    argument) is refused: the type of an inventory theorem must be exactly a registered Prop
    constant applied to the theorem's own parameters."""
    normalized = ' '.join(type_text.split())
    require(normalized, 'theorem without a result type')
    require(all(ch not in BRACKETS for ch in depth_zero(normalized) if ch != ' ') and not any(ch in BRACKETS for ch in normalized),
            f'result type {normalized!r}: arguments of the registered Prop must be bare parameter names (no bracketed '
            'expressions; a narrowed argument is a finding about the statement, not a theorem of it)')
    tokens = normalized.split()
    for t in tokens:
        require(IDENT_RE.fullmatch(t) is not None and not t.startswith('_root_'),
                f'result type {normalized!r}: must be a registered Prop constant applied to parameter names; found {t!r} '
                '(no `∀`, `→`, `∃`, relations or operators at depth zero: move quantifiers into binders)')
    return tokens[0], tokens[1:]


def head_candidates(head: str, opens: list, namespaces: list) -> set:
    """Every global name `head` may denote under Lean's resolution: as written, relative to a
    prefix of the current namespace, or through an opened namespace (itself possibly relative)."""
    prefixes = set()
    for ns in namespaces:
        parts = ns.split('.')
        for k in range(1, len(parts) + 1):
            prefixes.add('.'.join(parts[:k]))
    candidates = {head} | {p + '.' + head for p in prefixes}
    for o in opens:
        candidates.add(o + '.' + head)
        candidates.update(p + '.' + o + '.' + head for p in prefixes)
    return candidates


def resolve_head(head: str, opens: list, namespaces: list, registered: dict, local=frozenset()) -> str:
    """The unique registered Prop that `head` denotes: as written, or through an opened
    namespace, or relative to a prefix of the current namespace (Lean's name resolution). A
    candidate that is a declaration of the module itself (`local`) is refused first: Lean resolves
    a name of the current namespace before an opened one, so a module-local `def A : Prop` would
    silently make `theorem t : A` a theorem about the local definition, not the registered Prop."""
    candidates = head_candidates(head, opens, namespaces)
    shadow = sorted(candidates & set(local))
    require(not shadow, f'result type head {head!r} is shadowed by the module-local declaration(s) {shadow!r}: a proof module may '
            'not declare a name that a registered Prop resolves through (Lean resolves the current namespace before an opened one)')
    hits = sorted(candidates & set(registered))
    require(hits, f'result type head {head!r} is not a registered `def … : Prop` of the statement register '
            f'(resolution tried {len(candidates)} names through the opened namespaces)')
    require(len(hits) == 1, f'result type head {head!r} resolves to several registered Props: {hits!r}')
    return hits[0]


def names_prop(text: str, head: str) -> bool:
    """`text` names the registered Prop `head` as a whole token: its qualified name, or its last
    component preceded at most by a namespace prefix, and not as a prefix or suffix of a longer
    identifier (`AA`, `A.eq_1` do not name `A`)."""
    last = head.rsplit('.', 1)[-1]
    pattern = r"(?<![A-Za-z0-9_'!?])(?:" + re.escape(head) + '|' + re.escape(last) + r")(?![A-Za-z0-9_'!?]|\.[A-Za-z_])"
    return re.search(pattern, text) is not None


def theorem_doc(raw: str, name: str, head: str) -> dict:
    """The three labelled lines DESIGN2 requires of an inventory theorem's docstring; the
    `Proves:` line must name the registered Prop it proves (qualified or by its last component)
    as a whole token."""
    body = raw[3:-2]
    found = {}
    for label in LABELS:
        hits = re.findall(r'^[ \t]*' + re.escape(label) + r'(.*)$', body, re.M)
        require(len(hits) == 1, f'{name}: docstring must carry exactly one `{label}` line (found {len(hits)})')
        require(hits[0].strip(), f'{name}: empty `{label}` entry in docstring (the entry starts on the label\'s line)')
        found[label] = hits[0].strip()
    require(names_prop(found['Proves:'], head), f'{name}: the `Proves:` line must name the registered Prop {head} as a whole token')
    return found


def legal_head(text: str) -> bool:
    """A declaration or scope keyword may be preceded on its line only by attribute groups and
    modifiers, starting at column 0 (an indented keyword is refused)."""
    if text[:1].isspace():
        return False
    s = text
    while s.startswith('@['):
        depth, j = 0, 0
        while j < len(s):
            if s[j] == '[':
                depth += 1
            elif s[j] == ']':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(s):
            return False
        s = s[j + 1:].lstrip()
    return all(tok in MODIFIERS for tok in s.split())


def open_targets(block: str) -> list:
    """Namespaces named by an `open` command (the tokens before any `(`, `hiding`, `renaming`
    or `in`), `scoped` dropped."""
    out = []
    for tok in block.split()[1:]:
        if tok in ('hiding', 'renaming', 'in') or tok.startswith('('):
            break
        if tok != 'scoped':
            out.append(tok)
    return out


def check_set_options(code: str, raw_lines: list, stem: str):
    """Options a proof module may set: never a `debug.`, `bootstrap.` or `compiler.` option, never
    autoImplicit; `maxHeartbeats` above 400000 (or 0, unlimited) only with a justifying comment on
    the same line or on the line before. Returns the options seen."""
    seen = []
    for m in SET_OPTION.finditer(code):
        name, value = m.groups()
        line_no = code.count('\n', 0, m.start())
        require(not name.startswith(REFUSED_OPTION_PREFIXES) and name not in REFUSED_OPTIONS,
                f'{stem}:{line_no + 1}: refused option `set_option {name}` (checking-weakening or implicit-binding option)')
        if name == 'maxHeartbeats':
            require(value.isdigit(), f'{stem}:{line_no + 1}: maxHeartbeats must be a literal number')
            v = int(value)
            if v == 0 or v > MAX_HEARTBEATS:
                here = raw_lines[line_no]
                prev = next((ln for ln in reversed(raw_lines[:line_no]) if ln.strip()), '')
                justified = '--' in here[here.find('set_option'):] or prev.lstrip().startswith('--') or prev.rstrip().endswith('-/')
                require(justified, f'{stem}:{line_no + 1}: `set_option maxHeartbeats {value}` exceeds {MAX_HEARTBEATS} without a '
                        'justifying comment on the same line or the line before')
        seen.append({'line': line_no + 1, 'option': name, 'value': value})
    return seen


def scan_module(text: str, stem: str, registered: dict) -> dict:
    """One proof module: imports, namespaces, the theorem inventory and the auxiliary counts.
    Refuses the lexical list, `#` commands, refused options, declaration and scope keywords off
    column 0, column-0 commands outside the allowed set, unbalanced scoping, declarations outside
    the module namespace UniversalLaw.Proofs.<stem>, and any non-aux theorem (not private, not
    under an `Aux` namespace) whose header does not read `name <binders> : <registered Prop>
    <parameter names>` with every explicit binder passed, whose head is a parameter or a name the
    module itself declares (shadowing), or that lacks the three docstring labels. A pre-Lean
    stage: the environment audit of --execute is the authority on what the modules declare."""
    code, docs = lex(text)
    raw_lines = text.split('\n')
    bad = refused_tokens(code)
    require(not bad, f'refused token(s) in {stem}: {bad!r}')
    m = HASH_COMMAND.search(code)
    require(m is None, f'{stem}:{code.count(chr(10), 0, m.start()) + 1 if m else 0}: `#` commands have no place in a proof module')
    options = check_set_options(code, raw_lines, stem)
    dz = depth_zero(code)
    for m in COMMAND_KEYWORD.finditer(dz):
        head = code[code.rfind('\n', 0, m.start()) + 1:m.start()]
        require(legal_head(head), f'{stem}:{code.count(chr(10), 0, m.start()) + 1}: keyword `{m.group(1)}` off column 0 '
                '(every declaration and scope command must start its own line, after attributes and modifiers only)')
    expected_ns = NAMESPACE + stem
    doc_end = {end: raw for _, end, raw in docs}
    lines = code.split('\n')
    imports, stack, namespaces, theorems = [], [], [], []
    local_names = set()  # every name the module declares, fully qualified (shadowing check)
    aux_theorems = auxiliary = 0
    pending_opens = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line[0].isspace():
            i += 1
            continue
        tok = line.split()[0]
        if tok.startswith('@['):
            # an attribute line; the declaration follows on this or a later line (DECL_LINE handles both)
            if DECL_LINE.match(line) is None:
                i += 1
                continue
            tok = DECL_LINE.match(line).group(3)
        if tok in CONTINUATIONS or tok.startswith('|'):
            i += 1
            continue
        require(tok in ALLOWED_COMMANDS, f'{stem}:{i + 1}: column-0 command not allowed in a proof module: {tok!r}')
        if tok == 'import':
            m = re.match(r'^import\s+(\S+)\s*$', line)
            require(m is not None, f'{stem}:{i + 1}: unparsable import line')
            imports.append(m.group(1))
            i += 1
            continue
        if tok == 'namespace':
            m = re.match(r'^namespace\s+(\S+)\s*$', line)
            require(m is not None, f'{stem}:{i + 1}: anonymous or unparsable namespace')
            prefix = [n for k, n, _, _ in stack if k == 'namespace']
            fq = '.'.join(prefix + [m.group(1)])
            require(fq == expected_ns or fq.startswith(expected_ns + '.'),
                    f'{stem}:{i + 1}: namespace {fq} is outside the module namespace {expected_ns}')
            namespaces.append(fq)
            stack.append(('namespace', m.group(1), [], []))
            i += 1
            continue
        if tok == 'section' or (tok == 'noncomputable' and line.split()[1:2] == ['section']):
            m = re.match(r'^(?:noncomputable\s+)?section(?:\s+(\S+))?\s*$', line)
            require(m is not None, f'{stem}:{i + 1}: unparsable section line')
            stack.append(('section', m.group(1), [], []))
            i += 1
            continue
        if tok == 'end':
            m = re.match(r'^end(?:\s+(\S+))?\s*$', line)
            require(m is not None and stack, f'{stem}:{i + 1}: `end` without an open namespace or section')
            kind, name, _, _ = stack.pop()
            require(name == m.group(1), f'{stem}:{i + 1}: `end {m.group(1)}` does not close `{kind} {name}`')
            i += 1
            continue
        if tok in ('open', 'set_option', 'omit', 'variable', 'attribute', 'universe', 'deriving'):
            block, k = [line], i + 1
            while k < len(lines) and lines[k] and lines[k][0].isspace():
                block.append(lines[k])
                k += 1
            joined = '\n'.join(block)
            chained = IN_TOKEN.search(depth_zero(joined)) is not None
            if tok in ('open', 'set_option', 'omit'):
                require(not chained or re.search(r'(?<![A-Za-z0-9_.\'])in\s*$', depth_zero(joined)) is not None,
                        f'{stem}:{i + 1}: `{tok} … in` must end its line; the chained command starts the next line')
                if tok == 'open':
                    (pending_opens if chained else (stack[-1][2] if stack else pending_opens)).extend(open_targets(joined))
            elif tok == 'variable':
                require(not chained, f'{stem}:{i + 1}: `variable … in` is not allowed')
                require(stack, f'{stem}:{i + 1}: `variable` outside the module namespace')
                stack[-1][3].extend(parse_binders(joined[len('variable'):]))
            i = k
            continue
        m = DECL_LINE.match(line)
        require(m is not None, f'{stem}:{i + 1}: unparsable declaration line: {line[:80]!r}')
        modifiers, keyword, name, rest = m.group(2).split(), m.group(3), m.group(4), m.group(5)
        prefix = [n for k, n, _, _ in stack if k == 'namespace']
        require(prefix and prefix[0] == expected_ns, f'{stem}:{i + 1}: declaration outside namespace {expected_ns}')
        if name:
            local_names.add('.'.join(prefix + [name]))
        opens = [o for _, _, os_, _ in stack for o in os_] + pending_opens
        variables = [b for _, _, _, vs in stack for b in vs]
        pending_opens = []
        if keyword not in ('theorem', 'lemma'):
            auxiliary += keyword != 'example'
            i += 1
            continue
        require(name, f'{stem}:{i + 1}: theorem without a name')
        require(not any(part.startswith('_') for part in name.split('.')),
                f'{stem}:{i + 1}: theorem name components may not start with `_`: {name}')
        fq = '.'.join(prefix + [name])
        relative = fq[len(expected_ns) + 1:].split('.')
        if 'private' in modifiers or AUX in relative[:-1]:
            aux_theorems += 1
            i += 1
            continue
        hdr = header(lines, i, rest)
        colon = first_colon(hdr)
        require(colon is not None, f'{fq}: theorem without an explicit result type')
        try:
            binders = parse_binders(hdr[:colon])
            head, args = result_head(hdr[colon + 1:])
            pool = {n for b in binders + variables for n in b['names']}
            require(head not in pool, f'result type head {head!r} is a parameter or section variable of the theorem, not a '
                    'registered Prop constant (a bound-variable type is refused)')
            scopes = [n for n in namespaces if fq.startswith(n + '.')]
            proves = resolve_head(head, opens, scopes, registered)
        except ValueError as exc:
            raise ValueError(f'{stem}:{i + 1}: theorem {fq}: {exc}') from None
        require(len(args) == len(set(args)), f'{fq}: repeated argument in the result type')
        for a in args:
            require(a in pool, f'{fq}: argument {a!r} of {proves} is not a parameter of the theorem (arguments must be the '
                    'theorem\'s own binders or section variables)')
        for b in binders:
            if b['kind'] == 'explicit':
                for n in b['names']:
                    require(n in args, f'{fq}: explicit parameter {n!r} is not passed to {proves}: an extra hypothesis or a '
                            'narrowed statement is a finding about the statement, not a theorem of it')
        j = i - 1
        while j >= 0 and j not in doc_end and (not lines[j].strip() or lines[j].startswith('@[')):
            j -= 1
        require(j in doc_end, f'{fq}: inventory theorem has no docstring immediately before it')
        doc = theorem_doc(doc_end[j], fq, proves)
        theorems.append({'name': fq, 'line': i + 1, 'proves': proves, 'binders': binders, 'arguments': args,
                         'doc': {'proves': doc['Proves:'], 'method': doc['Method:'], 'does_not_establish': doc['Does not establish:']},
                         'resolution': (head, opens, scopes)})
        i += 1
    require(not stack, f'{stem}: unclosed namespace or section: {[(k, n) for k, n, _, _ in stack]!r}')
    for t in theorems:
        # shadowing is decided once every declaration of the module is known (a declaration after the
        # theorem is refused too: the rule is conservative, Lean's resolution order is the reason)
        head, opens, scopes = t.pop('resolution')
        try:
            resolve_head(head, opens, scopes, registered, local_names)
        except ValueError as exc:
            raise ValueError(f'{stem}:{t["line"]}: theorem {t["name"]}: {exc}') from None
    require(namespaces and namespaces[0] == expected_ns, f'{stem}: the first namespace must be {expected_ns}: {namespaces!r}')
    require(imports, f'{stem}: no import')
    for imp in imports:
        require(imp == 'Mathlib' or imp.startswith('Mathlib.')
                or (imp.startswith(STATEMENT_ROOT_MODULE + '.') and imp.count('.') == 1)
                or (imp.startswith(ROOT_MODULE + '.') and imp.count('.') == 1 and imp != ROOT_MODULE + '.' + stem),
                f'{stem}: imports must be `Specifications.<Module>`, Mathlib or a sibling `SpecificationProofs.<Module>` of this packet, nothing else: {imp!r}')
    require(any(imp.startswith(STATEMENT_ROOT_MODULE + '.') for imp in imports), f'{stem}: no statement module is imported')
    return {'stem': stem, 'imports': imports, 'namespaces': namespaces, 'theorems': theorems, 'aux_theorems': aux_theorems,
            'auxiliary_definitions': auxiliary, 'set_options': options}


# ----------------------------------------------------------------------------- packet checks

def git(repo: Path, *args, timeout=120):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=timeout)


def check_identity(repo: Path, allow_dirty: bool):
    proc = git(repo, 'rev-parse', 'HEAD')
    head = proc.stdout.strip() if proc.returncode == 0 and HEX40.fullmatch(proc.stdout.strip()) else None
    if allow_dirty:
        return head, 'UNVERIFIED'
    require(head is not None, 'not a git repository with a 40-hex HEAD (local development may pass --allow-dirty)')
    status = git(repo, 'status', '--porcelain', '--untracked-files=all', '--', PACKAGE, STATEMENT_PACKAGE, WORKFLOW)
    require(status.returncode == 0, 'git status failed: ' + status.stderr.strip()[:300])
    require(status.stdout.strip() == '',
            'proof packet, statement packet or workflow differ from HEAD (uncommitted or untracked):\n' + status.stdout[:2000])
    require(git(repo, 'ls-files', '--error-unmatch', WORKFLOW).returncode == 0, 'workflow is not tracked: ' + WORKFLOW)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        require(os.environ.get('GITHUB_SHA') == head, 'GITHUB_SHA differs from the checked-out HEAD')
        require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY, 'unexpected GITHUB_REPOSITORY')
    return head, 'CLEAN'


def load_register(statement_root: Path):
    """The registered `def … : Prop` declarations of the statement packet, {name: {module,
    claim_ids}}, plus the registered structures (never a theorem type). The register's own
    invariants are the statement packet's gate; here only the pins and the inventory are read."""
    raw = safe_path(statement_root, 'REGISTER.json').read_bytes()
    reg = load_json(raw.decode('utf-8'))
    require(isinstance(reg, dict) and reg.get('object') == REGISTER_OBJECT, 'REGISTER.json object must be ' + REGISTER_OBJECT)
    require(reg.get('scientific_effect') == 'NONE' and reg.get('scientific_status_authority') is False
            and reg.get('alignment_status') == 'PENDING_INDEPENDENT_REVIEW', 'REGISTER.json scope fields differ from the contract')
    require(reg.get('toolchain') == {'lean': TOOLCHAIN, 'lean_commit': LEAN_COMMIT, 'mathlib': MATHLIB_REV}, 'register toolchain pins differ')
    require(isinstance(reg.get('claims'), list), 'REGISTER.json claims must be a list')
    registered, structures = {}, set()
    for row in reg['claims']:
        require(isinstance(row, dict) and isinstance(row.get('claim_id'), str) and isinstance(row.get('statements'), list), 'malformed claim row')
        for st in row['statements']:
            require(isinstance(st, dict) and all(k in st for k in ('package', 'module', 'declaration', 'kind', 'formal_progress')), 'malformed statement row')
            if st['package'] != STATEMENT_PACKAGE:
                continue
            require(st['formal_progress'] == 'specified', f'{st["declaration"]}: a statement of the statement packet is specified; '
                    'the proof packet never relabels the register')
            require(st['declaration'].startswith(STATEMENT_NAMESPACE), f'{st["declaration"]}: registered name outside {STATEMENT_NAMESPACE}')
            if st['kind'] == 'def_prop':
                entry = registered.setdefault(st['declaration'], {'module': st['module'], 'claim_ids': []})
                require(entry['module'] == st['module'], f'{st["declaration"]}: registered in two modules')
                if row['claim_id'] not in entry['claim_ids']:
                    entry['claim_ids'].append(row['claim_id'])
            elif st['kind'] == 'structure':
                structures.add(st['declaration'])
    require(registered, 'the register names no `def … : Prop` of the statement packet')
    for name, entry in registered.items():
        safe_path(statement_root, entry['module'])
    return registered, structures, sha(raw)


def packet_files(root: Path) -> list:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            require(not (Path(dirpath) / d).is_symlink(), 'symlink directory refused: ' + d)
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in filenames:
            p = Path(dirpath) / f
            require(not p.is_symlink(), 'symlink refused: ' + p.relative_to(root).as_posix())
            files.append(p.relative_to(root).as_posix())
    return sorted(files)


def check_tree(root: Path) -> list:
    """The packet holds exactly the fixed files and `SpecificationProofs/<Module>.lean` modules;
    returns the module paths found."""
    present = packet_files(root)
    modules = [f for f in present if re.fullmatch(ROOT_MODULE + r'/[A-Z][A-Za-z0-9]*\.lean', f)]
    expected = sorted(set(FIXED_FILES) | set(modules))
    require(present == expected, 'packet tree differs from the fixed inventory: unexpected '
            + repr(sorted(set(present) - set(expected))) + ', missing ' + repr(sorted(set(expected) - set(present))))
    require(modules, 'no proof module under ' + ROOT_MODULE + '/')
    require('.lake/' in (root / '.gitignore').read_text(encoding='utf-8').splitlines(), '.gitignore must ignore .lake/')
    return modules


def check_lock(root: Path, statement_root: Path) -> dict:
    """lean-toolchain and lakefile.toml byte-exact (Mathlib rev, the path dependency, autoImplicit
    off, and nothing else: no `moreLeanArgs`, no extra dependency, library or executable) and
    lake-manifest.json == the statement packet's lock plus the path dependency. Returns the git
    dependency revisions."""
    require((root / 'lean-toolchain').read_bytes() == (TOOLCHAIN + '\n').encode(), 'lean-toolchain must read exactly ' + TOOLCHAIN)
    require((statement_root / 'lean-toolchain').read_bytes() == (TOOLCHAIN + '\n').encode(), 'statement packet lean-toolchain differs')
    lakefile = (root / 'lakefile.toml').read_bytes()
    if lakefile != LAKEFILE.encode():
        actual, expected = lakefile.decode('utf-8', 'replace').split('\n'), LAKEFILE.split('\n')
        k = next(i for i in range(max(len(actual), len(expected))) if actual[i:i + 1] != expected[i:i + 1])
        raise ValueError('lakefile.toml must be byte-exactly the pinned manifest (Mathlib rev, the path dependency, autoImplicit off and '
                         f'no other option, dependency or target): line {k + 1} reads {actual[k:k + 1]!r}, expected {expected[k:k + 1]!r}')
    lock = load_json((root / 'lake-manifest.json').read_text(encoding='utf-8'))
    require(isinstance(lock, dict) and lock.get('name') == LOCK_NAME, 'lake-manifest.json name must be ' + LOCK_NAME)
    ref = load_json(safe_path(statement_root, 'lake-manifest.json').read_text(encoding='utf-8'))
    require(ref.get('name') == STATEMENT_LOCK_NAME, 'statement packet lake-manifest.json name must be ' + STATEMENT_LOCK_NAME)
    packages = lock.get('packages')
    require(isinstance(packages, list) and packages and packages[-1] == PATH_DEPENDENCY,
            'lake-manifest.json must end with the path dependency on the statement packet exactly: ' + json.dumps(PATH_DEPENDENCY))
    require(packages[:-1] == ref['packages'], 'lake-manifest.json git dependencies differ from the statement packet lock')
    require({k: v for k, v in lock.items() if k not in ('name', 'packages')} == {k: v for k, v in ref.items() if k not in ('name', 'packages')},
            'lake-manifest.json differs from the statement packet lock beyond name and packages')
    revs = {p['name']: p['rev'] for p in packages[:-1]}
    require(len(revs) == len(packages) - 1 and all(isinstance(v, str) and HEX40.fullmatch(v) for v in revs.values()),
            'duplicate or unpinned dependency in lake-manifest.json')
    require(revs.get('mathlib') == MATHLIB_REV, 'lake-manifest.json Mathlib rev differs from the pin')
    return revs


def check_root(root: Path, module_files: list) -> list:
    """The root module imports exactly the module files present, one per line; returns the
    modules in root order."""
    code, _ = lex((root / (ROOT_MODULE + '.lean')).read_text(encoding='utf-8'))
    lines = [line for line in code.splitlines() if line.strip()]
    imports = []
    for line in lines:
        m = re.fullmatch(r'import ' + ROOT_MODULE + r'\.([A-Z][A-Za-z0-9]*)', line)
        require(m is not None, 'root module lines must read `import ' + ROOT_MODULE + '.<Module>`: ' + repr(line))
        imports.append(ROOT_MODULE + '/' + m.group(1) + '.lean')
    require(len(imports) == len(set(imports)), 'root module imports a module twice')
    require(set(imports) == set(module_files), 'root module must import exactly the modules present under ' + ROOT_MODULE
            + '/: imported ' + repr(sorted(set(imports) - set(module_files))) + ' missing, present but not imported '
            + repr(sorted(set(module_files) - set(imports))))
    return imports


def check_docs(root: Path):
    for name in ('README.md', 'SCOPE.md'):
        text = (root / name).read_text(encoding='utf-8')
        hits = PLACEHOLDER.findall(text)
        require(not hits, f'{name} still carries unfilled placeholder(s): {hits[:4]!r}')
        for phrase in ('Scientific effect', 'NONE', 'PENDING_INDEPENDENT_REVIEW'):
            require(phrase in text, f'{name} must state the scientific effect (`Scientific effect` … `NONE`) and the alignment status '
                    f'(`PENDING_INDEPENDENT_REVIEW`) literally; missing {phrase!r}')


def statement_modules(statement_root: Path) -> dict:
    files = sorted((statement_root / STATEMENT_ROOT_MODULE).glob('*.lean'))
    require(files, 'statement packet has no modules')
    return {STATEMENT_ROOT_MODULE + '/' + p.name: sha(p.read_bytes()) for p in files if not p.is_symlink()}


def source_check(root: Path = ROOT, repo: Path = REPO, allow_dirty: bool = False) -> dict:
    root, repo = Path(root).resolve(), Path(repo).resolve()
    head, worktree = check_identity(repo, allow_dirty)
    statement_root = (root / STATEMENT_DIR).resolve()
    require(statement_root.is_dir() and statement_root == (repo / STATEMENT_PACKAGE).resolve(),
            f'the path dependency {STATEMENT_DIR} must be the statement packet {STATEMENT_PACKAGE} of the same repository')
    registered, structures, register_sha = load_register(statement_root)
    stmt_hashes = statement_modules(statement_root)
    module_files = check_tree(root)
    revs = check_lock(root, statement_root)
    modules = check_root(root, module_files)
    scans = {m: scan_module(safe_path(root, m).read_text(encoding='utf-8'), PurePosixPath(m).stem, registered) for m in modules}
    for m, scan in scans.items():
        for imp in scan['imports']:
            if imp.startswith(STATEMENT_ROOT_MODULE + '.'):
                path = STATEMENT_ROOT_MODULE + '/' + imp.split('.', 1)[1] + '.lean'
                require(path in stmt_hashes, f'{m}: imports {imp}, which is not a module of the statement packet')
            elif imp.startswith(ROOT_MODULE + '.'):
                path = ROOT_MODULE + '/' + imp.split('.', 1)[1] + '.lean'
                require(path in modules, f'{m}: imports {imp}, which is not a proof module of this packet')
    check_docs(root)
    theorems = [dict(t, module=m) for m in modules for t in scans[m]['theorems']]
    names = [t['name'] for t in theorems]
    require(len(names) == len(set(names)), 'two inventory theorems share a name')
    for t in theorems:
        t['claim_ids'] = registered[t['proves']]['claim_ids']
        t['statement_module'] = registered[t['proves']]['module']
    covered = sorted({t['proves'] for t in theorems})
    duplicates = sorted({t['proves'] for t in theorems if sum(u['proves'] == t['proves'] for u in theorems) > 1})
    return {'scientific_effect': 'NONE', 'formalization_status': 'proved', 'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
            'checked_commit': head, 'worktree': worktree, 'register_sha256': register_sha, 'statement_module_sha256': stmt_hashes,
            'module_sha256': {m: sha((root / m).read_bytes()) for m in modules}, 'dependency_revisions': revs, 'modules': modules,
            'registered_props': len(registered), 'theorem_count': len(theorems), 'covered_props': covered, 'duplicate_proofs': duplicates,
            'aux_theorems': sum(s['aux_theorems'] for s in scans.values()),
            'auxiliary_definitions': sum(s['auxiliary_definitions'] for s in scans.values()),
            'set_options': {m: s['set_options'] for m, s in scans.items() if s['set_options']},
            'theorems': [{'name': t['name'], 'module': t['module'], 'line': t['line'], 'proves': t['proves'],
                          'statement_module': t['statement_module'], 'claim_ids': t['claim_ids'],
                          'binder_count': len(t['binders']), 'argument_count': len(t['arguments']),
                          'explicit_parameters': sum(len(b['names']) for b in t['binders'] if b['kind'] == 'explicit'),
                          'doc': t['doc']} for t in theorems],
            'registered': registered, 'scans': scans}


# ----------------------------------------------------------------------------- Lean execution

def lake_binary():
    found = shutil.which('lake')
    if found:
        return found
    candidate = Path.home() / '.elan/bin/lake'
    return str(candidate) if candidate.is_file() else None


def lean_env(lake: str) -> dict:
    return dict(os.environ, PATH=f'{Path(lake).parent}{os.pathsep}{os.environ.get("PATH", "")}')


def run(command, label, out: Path, cwd: Path, env, expect_success=True, timeout=3600):
    proc = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, env=env, encoding='utf-8', errors='replace')
    (out / (label + '.log')).write_text(proc.stdout, encoding='utf-8')
    print(label + ': exit ' + str(proc.returncode), flush=True)
    require((proc.returncode == 0) == expect_success, 'unexpected process outcome: ' + label + '\n' + proc.stdout[-6000:])
    return proc.stdout


def check_lean_version(text: str) -> str:
    """The last line of `lake env lean --version` (lake may print `info:` lines first) must be the
    pinned release exactly: version, release commit and Release build."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(lines and all(line.startswith('info:') for line in lines[:-1]), 'unexpected output around the Lean version: ' + repr(text[:200]))
    version = lines[-1]
    require(re.fullmatch(r'Lean \(version 4\.34\.1, [^()\r\n]+, commit ' + LEAN_COMMIT + r', Release\)', version) is not None,
            'running Lean is not the pinned release: ' + repr(version[:160]))
    return version


def dependency_state(root: Path, revs: dict) -> dict:
    """Every git dependency checkout under .lake/packages is at its locked revision and clean;
    the path dependency resolves to the statement packet. `.lake/packages` itself may be a symlink
    to a prebuilt dependency tree (recorded in the receipt), each package directory may not."""
    actual = {}
    for name, rev in revs.items():
        pkg = root / '.lake' / 'packages' / name
        require(pkg.is_dir() and not pkg.is_symlink(), 'dependency checkout missing: ' + name)
        actual[name] = subprocess.check_output(['git', '-C', str(pkg), 'rev-parse', 'HEAD'], text=True).strip()
        status = subprocess.check_output(['git', '-C', str(pkg), 'status', '--porcelain', '--untracked-files=normal'], text=True)
        require(status.strip() == '', f'dependency worktree is not clean: {name}: {status.strip()[:200]}')
    require(actual == revs, 'checked-out dependency revisions differ from lake-manifest.json: ' + repr(actual))
    return actual


def audit_axioms(text: str, targets: list) -> dict:
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets), 'empty or duplicate target list')
    records = {}
    for match in AXIOM_LINE.finditer(text):
        name, raw = match.groups()
        require(name in targets and name not in records, 'unexpected or duplicate axiom report: ' + name)
        axioms = [] if raw is None or not raw.strip() else [a.strip() for a in raw.split(',')]
        require(all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.']*", a) for a in axioms), 'malformed axiom name')
        require(set(axioms) <= ALLOWED_AXIOMS, 'forbidden transitive axiom: ' + repr(sorted(set(axioms) - ALLOWED_AXIOMS)))
        records[name] = sorted(set(axioms))
    require(not AXIOM_LINE.sub('', text).strip(), 'unrecognized audit output')
    require(set(records) == set(targets), 'missing axiom report: ' + repr(sorted(set(targets) - set(records))))
    return records


def signatures(log: str, names: list) -> dict:
    """One elaborated signature per name from the `#check` log: the block starting at the line
    that begins with the name, continued by indented lines (the pretty printer wraps)."""
    lines, found = log.splitlines(), {}
    for n in names:
        starts = [i for i, line in enumerate(lines) if re.match(re.escape(n) + r'(?![A-Za-z0-9_.])', line)]
        require(len(starts) == 1, 'no unique elaborated signature reported for ' + n)
        i = starts[0] + 1
        while i < len(lines) and lines[i] and lines[i][0].isspace():
            i += 1
        found[n] = '\n'.join(lines[starts[0]:i])
    return found


def kernel_audit(lake: str, root: Path, env: dict, out: Path, names: list, root_module: str = ROOT_MODULE):
    """`#print axioms` (parsed, allowed set only) and `#check` (logged, one signature per name) of
    every inventoried theorem. Returns the axiom records and the elaborated signatures."""
    audit = out / 'Audit.lean'
    audit.write_text('import ' + root_module + '\n' + ''.join(f'#print axioms {n}\n' for n in names), encoding='utf-8')
    axioms = audit_axioms(run([lake, 'env', 'lean', str(audit)], 'axioms', out, root, env), names)
    types = out / 'Types.lean'
    types.write_text('import ' + root_module + '\n' + ''.join(f'#check {n}\n' for n in names), encoding='utf-8')
    log = run([lake, 'env', 'lean', str(types)], 'types', out, root, env)
    require(LEAN_ERROR.search(log) is None, 'the elaborated-types log reports an error')
    return axioms, signatures(log, names)


AUDIT_PROGRAM = '''
/-! Every constant of the audited modules (or, when `auditModules` is empty, every constant of
this file under `auditPrefix`, private names included) is classified by Lean's own `ConstantInfo`
and type, never by source text. `theorem`: a theorem that is not private, not under an `Aux`
namespace and carries a declaration range, whose type after `forallTelescope` is a registered Prop
constant applied to exactly the telescope's variables (reported with that constant). `aux_theorem`:
a private theorem or one under `Aux`. `generated`: a range-less theorem (structure `injEq`,
`sizeOf_spec`, `_proof_*`, an equation lemma `eq_*` of an unfolded statement definition, which may
be generated in several modules and is audited once), refused if its head is a registered Prop. `auxiliary`: definitions,
instances, inductive types, constructors and recursors whose type is not a proposition.
`REFUSED:<why>`: an axiom, opaque, quotient or unsafe constant; a definition whose type is a
proposition (a proof written as `def`); a non-aux theorem whose head is not registered (`type`) or
whose arguments are not exactly its own parameters (`premise`: an extra hypothesis or a narrowed
argument); any transitive axiom outside propext, Classical.choice and Quot.sound (`axioms`).
One `AUDIT <class> <name> [<head>]` line per constant, then `AUDIT END <count> <refused>
<theorems>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci =>
      let user := (privateToUserName? ci.name).getD ci.name
      if auditPrefix.isPrefixOf user then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  -- an equation lemma of an imported definition (`foo.eq_1`) is generated on demand in every module
  -- that unfolds it and is one constant of the environment: audit each name once
  let mut seen : NameSet := {}
  let mut unique : Array ConstantInfo := #[]
  for ci in constants do
    unless seen.contains ci.name do
      seen := seen.insert ci.name
      unique := unique.push ci
  constants := unique
  let mut refused := 0
  let mut theorems := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    if verdict == "theorem" then
      theorems := theorems + 1
      logInfo m!"AUDIT theorem {n} {headStr}"
    else
      logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused} {theorems}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"
'''


def audit_source(root_module: str, module_names: list, registered: list, declarations: str = '') -> str:
    """The environment-audit file: imports, the audited module list (empty: this file), the
    registered Prop names, optional control declarations, and the fixed audit program."""
    modules = '#[' + ', '.join('`' + m for m in module_names) + ']'
    props = '#[' + ', '.join('`' + r for r in registered) + ']'
    return (f'import Lean\nimport {root_module}\nopen Lean Meta Elab Command\n\n'
            f'/-- Modules whose constants are audited; empty means the constants of this file under `auditPrefix`. -/\n'
            f'def auditModules : Array Name := {modules}\n'
            f'/-- Namespace of the control declarations audited when `auditModules` is empty. -/\n'
            f'def auditPrefix : Name := `{CONTROL_NS}\n'
            f'/-- The registered `def … : Prop` constants of the statement register. -/\n'
            f'def registeredProps : Array Name := {props}\n{declarations}{AUDIT_PROGRAM}')


def parse_audit(log: str):
    """`AUDIT <class> <name> [<head>]` lines as ({class: set(names)}, {theorem: head}); the
    `AUDIT END` line must account for every line, every refusal and every theorem."""
    classes, heads, seen = {}, {}, set()
    for m in AUDIT_LINE.finditer(log):
        verdict, name, head = m.groups()
        if verdict == 'END':
            continue
        require(name not in seen, 'duplicate audit line for ' + name)
        seen.add(name)
        classes.setdefault(verdict, set()).add(name)
        if verdict == 'theorem':
            require(head is not None, 'theorem audit line without its head: ' + name)
            heads[name] = head
    ends = AUDIT_END.findall(log)
    require(len(ends) == 1, 'the environment audit did not report exactly one END line')
    total, refused, theorems = (int(x) for x in ends[0])
    require(total == len(seen), f'environment audit reported {total} constants but {len(seen)} AUDIT lines')
    require(refused == sum(len(v) for k, v in classes.items() if k.startswith('REFUSED')), 'environment audit refusal count differs from its lines')
    require(theorems == len(heads), 'environment audit theorem count differs from its lines')
    return classes, heads


def environment_audit(lake: str, root: Path, env: dict, out: Path, module_names: list, registered: list, expected: dict,
                      root_module: str = ROOT_MODULE) -> dict:
    """Lean's own inventory of the modules (see AUDIT_PROGRAM) must pass with no refusal, and its
    theorem inventory {name: registered head} must equal the scan's exactly."""
    path = out / 'Environment.lean'
    path.write_text(audit_source(root_module, module_names, registered), encoding='utf-8')
    classes, heads = parse_audit(run([lake, 'env', 'lean', str(path)], 'environment', out, root, env))
    require(set(classes) <= AUDIT_CLASSES, 'environment audit refused constant(s): ' + repr(sorted(set(classes) - AUDIT_CLASSES)))
    require(heads == expected, 'environment audit: Lean\'s theorem inventory differs from the scan: unexpected '
            + repr(sorted(set(heads) - set(expected))[:8]) + ', missing ' + repr(sorted(set(expected) - set(heads))[:8])
            + ', head differs ' + repr(sorted(n for n in set(heads) & set(expected) if heads[n] != expected[n])[:8]))
    return {k: len(v) for k, v in sorted(classes.items())}


def parameterless_prop(statement_root: Path, registered: dict) -> str:
    """A registered Prop with no parameters (`def <Name> : Prop :=` in its statement module), used
    by the controls to state refused theorems of a registered type."""
    for module in sorted({e['module'] for e in registered.values()}):
        code, _ = lex(safe_path(statement_root, module).read_text(encoding='utf-8'))
        for m in re.finditer(r'^(?:noncomputable )?def\s+(\S+)\s*:\s*Prop\s*:=', code, re.M):
            name = STATEMENT_NAMESPACE + PurePosixPath(module).stem + '.' + m.group(1)
            if name in registered:
                return name
    raise ValueError('no parameterless registered Prop available for the controls')


HIDDEN_CONTROL = '''
namespace {ns}
theorem sorried : {prop} := sorry
axiom hidden : False
theorem usesAxiom : {prop} := hidden.elim
theorem unregistered : True := trivial
theorem smuggled (h : False) : {prop} := h.elim
theorem implication : {prop} → {prop} := fun h => h
def proofAsDef : (2 : Nat) + 2 = 4 := rfl
opaque opq : Nat
unsafe def unsafeDef : Nat := 0
end {ns}
'''
HONEST_CONTROL = '''
namespace {ns}
/-- Control: a theorem of a registered type, proved by the packet's own theorem. -/
theorem honest : {prop} := {witness}
/-- Control: a helper under Aux (not inventory). -/
theorem Aux.helper : True := trivial
/-- Control: a private helper (not inventory). -/
private theorem privateHelper : True := trivial
/-- Control: an auxiliary definition. -/
def helper (n : Nat) : Nat := n + 1
/-- Control: an instance is an auxiliary definition. -/
instance controlInst : Inhabited Nat := ⟨0⟩
end {ns}
'''


def lean_control_environment(lake: str, root: Path, env: dict, out: Path, registered: list, prop: str, witness,
                             root_module: str = ROOT_MODULE) -> str:
    """Two Lean-side controls of the environment audit itself, over the constants of a control
    file (never a packet module): a hidden file (sorry, custom axiom and a theorem using it, an
    unregistered type, a smuggled hypothesis, a `P → P` implication, a proof written as def, an
    opaque and an unsafe constant) must be refused for exactly those reasons; an honest file
    (a theorem of registered type proved by the packet's witness when one is parameterless, an
    Aux helper, a private helper, a definition, an instance) must be classified exactly."""
    hidden = out / 'controls' / 'ControlEnvironmentHidden.lean'
    hidden.write_text(audit_source(root_module, [], registered, HIDDEN_CONTROL.format(ns=CONTROL_NS, prop=prop)), encoding='utf-8')
    log = run([lake, 'env', 'lean', str(hidden)], 'control_environment_hidden', out, root, env, expect_success=False)
    require('AUDIT REFUSED' in log and LEAN_ERROR.search(log) is not None, 'hidden environment control failed for the wrong reason')
    classes, heads = parse_audit(log)
    c = CONTROL_NS + '.'
    expected = {'REFUSED:axioms:sorryAx': {c + 'sorried'}, 'REFUSED:axiom': {c + 'hidden'},
                'REFUSED:axioms:' + c + 'hidden': {c + 'usesAxiom'}, 'REFUSED:type:True': {c + 'unregistered'},
                'REFUSED:premise:' + prop: {c + 'smuggled', c + 'implication'}, 'REFUSED:proof_as_def': {c + 'proofAsDef'},
                'REFUSED:opaque': {c + 'opq'}, 'REFUSED:unsafe': {c + 'unsafeDef'}}
    require(classes == expected and not heads, 'hidden environment control was not refused for exactly the expected reasons: '
            + repr({k: sorted(v) for k, v in classes.items()}))
    honest = out / 'controls' / 'ControlEnvironmentHonest.lean'
    text = HONEST_CONTROL.format(ns=CONTROL_NS, prop=prop, witness=witness)
    if witness is None:
        text = re.sub(r'/-- Control: a theorem of a registered type[^\n]*\ntheorem honest[^\n]*\n', '', text)
    honest.write_text(audit_source(root_module, [], registered, text), encoding='utf-8')
    classes, heads = parse_audit(run([lake, 'env', 'lean', str(honest)], 'control_environment_honest', out, root, env))
    private = {n for n in classes.get('aux_theorem', set()) if n.startswith('_private.') and n.endswith('.privateHelper')}
    require(set(classes) <= AUDIT_CLASSES and heads == ({c + 'honest': prop} if witness else {})
            and classes.get('aux_theorem', set()) == {c + 'Aux.helper'} | private and len(private) == 1
            and classes.get('auxiliary') == {c + 'helper', c + 'controlInst'},
            'honest environment control was not classified exactly: ' + repr({k: sorted(v) for k, v in classes.items()}))
    return ('REFUSED_BY_ENVIRONMENT_AUDIT (sorry, custom axiom and its user, unregistered type, smuggled hypothesis, P → P '
            'implication, proof as def, opaque, unsafe) and the honest control classified exactly'
            + ('' if witness else ' (no parameterless inventory theorem: the honest control carries no registered-type theorem)'))


def lean_control_sorry_axioms(lake: str, root: Path, env: dict, out: Path, prop: str, root_module: str = ROOT_MODULE) -> str:
    """A `sorry` theorem of a registered type passes Lean with a warning; `#print axioms` reports
    sorryAx and the axiom audit refuses it."""
    path = out / 'controls' / 'ControlSorry.lean'
    name = CONTROL_NS + '.sorried'
    path.write_text(f'import {root_module}\n\ntheorem {name} : {prop} := sorry\n#print axioms {name}\n', encoding='utf-8')
    log = run([lake, 'env', 'lean', str(path)], 'control_sorry_axioms', out, root, env)
    require('declaration uses' in log and 'sorryAx' in log, 'sorry control did not report sorryAx: ' + log[-400:])
    try:
        audit_axioms('\n'.join(line for line in log.splitlines() if "'" in line and 'axioms' in line), [name])
    except ValueError as exc:
        require('forbidden transitive axiom' in str(exc) and 'sorryAx' in str(exc), 'sorry control refused for the wrong reason: ' + str(exc))
        return 'REFUSED_BY_AXIOM_AUDIT (sorryAx)'
    raise ValueError('the axiom audit accepted a sorry theorem')


def lean_control_false_arithmetic(lake: str, root: Path, env: dict, out: Path, root_module: str = ROOT_MODULE) -> str:
    """A true bound of the packet's kind tightened into a false one: `6/7 ≤ 6/7 - 10^-20` and
    `16·7 < 16·7`; Lean must leave the goal `False` unsolved."""
    path = out / 'controls' / 'ControlFalseArithmetic.lean'
    path.write_text(f'import {root_module}\n\n/-- Negative control: a tightened false arithmetic statement. Lean must refuse it. -/\n'
                    f'theorem {CONTROL_NS}.tightened : (6 : ℝ) / 7 ≤ 6 / 7 - 1 / 10 ^ 20 := by norm_num\n'
                    f'theorem {CONTROL_NS}.tightened2 : (16 : ℝ) * 7 < 16 * 7 := by norm_num\n', encoding='utf-8')
    log = run([lake, 'env', 'lean', str(path)], 'control_false_arithmetic', out, root, env, expect_success=False)
    require(len(re.findall(r'error: unsolved goals\s*\n⊢ False', log)) == 2 and LEAN_ERROR.search(log) is not None,
            'false-arithmetic control failed for the wrong reason: ' + log[-600:])
    return 'REJECTED_BY_LEAN (unsolved goal False, twice)'


def lean_control_missing(lake: str, root: Path, env: dict, out: Path, name: str, root_module: str = ROOT_MODULE) -> str:
    path = out / 'controls' / 'ControlMissing.lean'
    path.write_text('import ' + root_module + '\n#check ' + name + '\n', encoding='utf-8')
    log = run([lake, 'env', 'lean', str(path)], 'control_missing_declaration', out, root, env, expect_success=False)
    require(('unknown identifier' in log.lower() or 'unknown constant' in log.lower()) and LEAN_ERROR.search(log) is not None,
            'missing-declaration control failed for the wrong reason: ' + log[-600:])
    return 'REJECTED_BY_LEAN'


def expect_refusal(fn, fragment: str, label: str):
    try:
        fn()
    except ValueError as exc:
        require(fragment in str(exc), f'negative control {label} was refused for the wrong reason: {exc}')
        return 'REFUSED_BY_SOURCE_CHECK'
    raise ValueError('negative control escaped the source check: ' + label)


def source_controls(info: dict, ctrl: Path) -> dict:
    """Source-side negative controls (no Lean): module copies carrying a `sorry`, an `axiom`, a
    `native_decide`, an unjustified `maxHeartbeats 800000`, a theorem of unregistered type, a
    theorem with a smuggled explicit hypothesis, a `P → P` theorem and a theorem without the
    docstring labels, each run through `scan_module`; the packet itself is never modified."""
    registered = info['registered']
    module = info['modules'][0]
    stem = PurePosixPath(module).stem
    ns = NAMESPACE + stem
    prop = next(iter(sorted(registered)))
    stmt_module = STATEMENT_ROOT_MODULE + '.' + PurePosixPath(registered[prop]['module']).stem
    doc = f'/-- Proves: `{prop}`.\nMethod: control.\nDoes not establish: anything. -/\n'

    def mod(body):
        return f'import {stmt_module}\n\nnamespace {ns}\n\n{body}\n\nend {ns}\n'
    cases = {
        'sorry': (mod(f'{doc}theorem control : {prop} := sorry'), 'refused token'),
        'axiom': (mod(f'axiom hidden : False\n\n{doc}theorem control : {prop} := hidden.elim'), 'refused token'),
        'native_decide': (mod(f'{doc}theorem control : {prop} := by native_decide'), 'refused token'),
        'maxHeartbeats_unjustified': (mod(f'set_option maxHeartbeats 800000 in\n{doc}theorem control : {prop} := by trivial'), 'justifying comment'),
        'unregistered_type': (mod('/-- Proves: `True`.\nMethod: control.\nDoes not establish: anything. -/\ntheorem control : True := trivial'),
                              'not a registered'),
        'smuggled_hypothesis': (mod(f'{doc}theorem control (h : False) : {prop} := h.elim'), 'not passed to'),
        'implication': (mod(f'{doc}theorem control : {prop} → {prop} := fun h => h'), 'must be a registered Prop constant applied'),
        'missing_docstring_labels': (mod(f'/-- No labels. -/\ntheorem control : {prop} := by trivial'), 'docstring must carry'),
    }
    outcomes = {}
    for label, (text, fragment) in cases.items():
        (ctrl / f'Source_{label}.lean.txt').write_text(text, encoding='utf-8')
        outcomes[label] = expect_refusal(lambda: scan_module(text, stem, registered), fragment, label)
    justified = mod(f'-- justification: the control proof is heavy\nset_option maxHeartbeats 800000 in\n{doc}theorem control : {prop} := by trivial')
    scan = scan_module(justified, stem, registered)
    require(len(scan['theorems']) == 1 and scan['set_options'][0]['value'] == '800000', 'justified maxHeartbeats control was not accepted')
    outcomes['maxHeartbeats_justified'] = 'ACCEPTED_BY_SOURCE_CHECK (comment on the line before)'
    return outcomes


def negative_controls(lake: str, root: Path, env: dict, out: Path, info: dict, statement_root: Path) -> dict:
    ctrl = out / 'controls'
    ctrl.mkdir()
    outcomes = source_controls(info, ctrl)
    registered = sorted(info['registered'])
    # a parameterless registered Prop for the controls: preferably one the packet proves without parameters (then the
    # honest control re-proves it from the packet's theorem), else any parameterless Prop of the statement modules
    prop, witness = next(((t['proves'], t['name']) for t in info['theorems'] if t['binder_count'] == 0 and t['argument_count'] == 0), (None, None))
    if prop is None:
        prop = parameterless_prop(statement_root, info['registered'])
    outcomes['sorry_axiom_audit'] = lean_control_sorry_axioms(lake, root, env, out, prop)
    outcomes['false_arithmetic'] = lean_control_false_arithmetic(lake, root, env, out)
    outcomes['missing_declaration'] = lean_control_missing(lake, root, env, out, CONTROL_NS + '.DoesNotExist')
    outcomes['environment_audit'] = lean_control_environment(lake, root, env, out, registered, prop, witness)
    return outcomes


def leanchecker_mode(allow_dirty: bool) -> str:
    """`RUN`, or `SKIPPED_LOCAL_ONLY` when LEAN_PROOF_SKIP_LEANCHECKER=1 is set together with
    --allow-dirty outside CI. The leanchecker replay of the Mathlib-importing environment needs
    roughly 14 GB of memory; skipping it lets the remaining stages be exercised on a smaller
    machine, and the receipt it yields is already non-evidential (worktree UNVERIFIED). Setting
    the variable without --allow-dirty, or in CI, is refused rather than ignored."""
    if os.environ.get(SKIP_LEANCHECKER_ENV) != '1':
        return 'RUN'
    require(os.environ.get('GITHUB_ACTIONS') != 'true', SKIP_LEANCHECKER_ENV + ' is refused in CI: a receipt without leanchecker is local-only')
    require(allow_dirty, SKIP_LEANCHECKER_ENV + ' is honoured only together with --allow-dirty: a receipt without leanchecker is local-only')
    return 'SKIPPED_LOCAL_ONLY'


def execute(root: Path, repo: Path, info: dict, allow_dirty: bool) -> dict:
    root, repo = Path(root).resolve(), Path(repo).resolve()
    statement_root = (root / STATEMENT_DIR).resolve()
    leanchecker = leanchecker_mode(allow_dirty)
    lake = lake_binary()
    require(lake is not None, 'lake not found on PATH or in ~/.elan/bin')
    env = lean_env(lake)
    out = root / '.lake' / 'evidence'
    require(not (root / '.lake').is_symlink() and not out.is_symlink() and not (root / '.lake' / 'build').is_symlink(), 'symlink build directory')
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    version = check_lean_version(run([lake, 'env', 'lean', '--version'], 'version', out, root, env))
    revs = dependency_state(root, info['dependency_revisions'])
    if (root / '.lake' / 'build').exists():
        shutil.rmtree(root / '.lake' / 'build')  # fresh build of this packet only; dependency builds are untouched
    log = run([lake, 'build'], 'build', out, root, env)
    require(LEAN_ERROR.search(log) is None and re.search(r'\berror\b', log, re.I) is None, 'build log reports an error')
    require('declaration uses' not in log, 'build log reports sorry or a similar escape')
    require('is a proposition; use' not in log, 'build log reports a definition whose type is a proposition (a proof written as def)')
    require(MODULE_WARNING.search(log) is None, 'build log reports a warning in a proof module (modules are held to -DwarningAsError=true)')
    if leanchecker == 'RUN':
        run([lake, 'env', 'leanchecker', ROOT_MODULE], 'leanchecker', out, root, env)
    else:
        print('leanchecker: SKIPPED_LOCAL_ONLY (' + SKIP_LEANCHECKER_ENV + '=1 with --allow-dirty); the receipt records the skip and is '
              'not evidence about any commit', file=sys.stderr, flush=True)
    names = [t['name'] for t in info['theorems']]
    require(names, 'no inventory theorem to audit')
    axioms, sigs = kernel_audit(lake, root, env, out, names)
    expected = {t['name']: t['proves'] for t in info['theorems']}
    registered = sorted(info['registered'])
    environment = environment_audit(lake, root, env, out, [m[:-5].replace('/', '.') for m in info['modules']], registered, expected)
    outcomes = negative_controls(lake, root, env, out, info, statement_root)
    after = source_check(root, repo, allow_dirty)
    require(after['register_sha256'] == info['register_sha256'] and after['module_sha256'] == info['module_sha256']
            and after['statement_module_sha256'] == info['statement_module_sha256'] and after['checked_commit'] == info['checked_commit'],
            'packet changed during execution')
    theorems = [dict(t, axioms=axioms[t['name']], signature=sigs[t['name']], formal_progress='kernel-checked') for t in info['theorems']]
    (out / 'theorems.json').write_text(json.dumps(theorems, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    receipt = {'schema_version': 1, 'object': RECEIPT_OBJECT, 'scientific_effect': 'NONE', 'scientific_status_authority': False,
               'package': PACKAGE, 'statement_package': STATEMENT_PACKAGE, 'formalization_status': 'kernel-checked',
               'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
               'meaning': 'The inventoried theorems were accepted by the Lean kernel at the checked commit against the pinned Mathlib '
                          'and statement packet, depend on no axiom outside propext, Classical.choice and Quot.sound, and each has as '
                          'its type a registered Prop constant applied to exactly its own parameters (Lean\'s own inventory equals the '
                          'scan\'s). This is kernel evidence about those Props only: not that any informal claim is true, aligned with '
                          'its source, reviewed or accepted. The statement register is not edited by this packet and no claim, premise '
                          'or obligation changes status because of this receipt.',
               'checked_commit': info['checked_commit'], 'worktree': info['worktree'],
               'receipt_trust': 'CLEAN worktree at checked_commit' if info['worktree'] == 'CLEAN' else
                                'LOCAL_ONLY: the worktree was not verified against checked_commit; not evidence about any commit'
                                + ('; leanchecker was skipped' if leanchecker != 'RUN' else ''),
               'repository': os.environ.get('GITHUB_REPOSITORY'), 'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),
               'workflow_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'lean_version': version, 'toolchain': TOOLCHAIN,
               'dependency_revisions': revs, 'packages_dir': 'symlink' if (root / '.lake' / 'packages').is_symlink() else 'directory',
               'register_sha256': info['register_sha256'], 'statement_module_sha256': info['statement_module_sha256'],
               'module_sha256': info['module_sha256'], 'modules': info['modules'], 'registered_props': info['registered_props'],
               'theorem_count': len(names), 'covered_props': info['covered_props'], 'duplicate_proofs': info['duplicate_proofs'],
               'aux_theorems': info['aux_theorems'], 'auxiliary_definitions': info['auxiliary_definitions'],
               'theorems': [{'name': t['name'], 'module': t['module'], 'proves': t['proves'], 'claim_ids': t['claim_ids'],
                             'axioms': t['axioms'], 'formal_progress': 'kernel-checked'} for t in theorems],
               'axioms': axioms, 'environment_audit': environment, 'lean_options': {'autoImplicit': False},
               'negative_controls': outcomes, 'build': 'PASS', 'leanchecker': 'PASS' if leanchecker == 'RUN' else leanchecker,
               'dependency_worktrees': 'CLEAN'}
    receipt['logs'] = {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log'))}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', nargs='?', choices=['source'], help='optional; `source` names the default mode')
    p.add_argument('--root', type=Path, default=ROOT, help='packet directory')
    p.add_argument('--repo', type=Path, default=REPO, help='repository root (holds both packets and the workflow)')
    p.add_argument('--execute', action='store_true', help='fresh build, leanchecker, #check/#print axioms, environment audit, controls, receipt')
    p.add_argument('--allow-dirty', action='store_true', help='local development only: skip the clean-worktree requirement')
    args = p.parse_args(argv)
    try:
        info = source_check(args.root, args.repo, args.allow_dirty)
        if info['worktree'] != 'CLEAN':
            print('WORKTREE_UNVERIFIED: packet identity against HEAD was not checked (--allow-dirty); nothing printed here '
                  'is evidence about a commit', file=sys.stderr)
        summary = {k: v for k, v in info.items() if k not in ('scans', 'registered', 'module_sha256', 'statement_module_sha256')}
        if args.execute:
            receipt = execute(args.root, args.repo, info, args.allow_dirty)
            print(json.dumps({'passed': True, 'formalization_status': receipt['formalization_status'],
                              'alignment_status': receipt['alignment_status'], 'checked_commit': receipt['checked_commit'],
                              'worktree': receipt['worktree'], 'leanchecker': receipt['leanchecker'], 'theorem_count': receipt['theorem_count'],
                              'covered_props': receipt['covered_props'], 'negative_controls': receipt['negative_controls']},
                             indent=2, sort_keys=True))
        else:
            print('SOURCE_CHECK_PASS (proofs of registered statements only: not a Lean build, not a review, not an acceptance)')
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        print('LEAN_PROOF_CHECK_FAIL: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
