#!/usr/bin/env python3
"""Source and execution checker for the statement-only Lean package
`specifications/lean_statements_20261010` (standard library only).

    python3 check.py [source]        source mode: packet identity against git HEAD, the lexical
                                     refusal list on every module, root imports, the declaration
                                     scan against REGISTER.json (and back), vocabulary, verbatim
                                     anchors, pinned source bytes, dependency lock, toolchain pin
    python3 check.py --execute       the same, then a fresh `lake build`, `leanchecker`, `#check`
                                     and `#print axioms` of every registered declaration, the
                                     environment audit (every constant Lean added for the five
                                     modules, classified by its ConstantInfo and compared with the
                                     register), real negative controls, a receipt under .lake/evidence/
    python3 check.py --allow-dirty   local development only: the clean-worktree requirement is
                                     skipped; a receipt then records the worktree as unverified
                                     and must not be trusted as evidence about any commit

Scientific effect: NONE. The package holds statements (`def … : Prop` and interface
`structure`s), not proofs. A green check means that the registered statements elaborate against
the pinned Mathlib and that the inventory in REGISTER.json is complete and consistent with the
module text and the byte-pinned Layer 0 sources. Formal progress of every statement here is
`specified`; the checker refuses `proved` and `kernel-checked` for them. It is not a proof of any
claim, not an alignment review (alignment is PENDING_INDEPENDENT_REVIEW), not an acceptance, and
it transcribes dispositions without deciding them. Nothing here can raise any label.

Two independent stages enforce "statements only". The source scan is a fail-closed pre-Lean
stage over comment-stripped text: whole-token refusals, string literals and guillemet names
refused outright (they can desynchronise any comment lexer), every command keyword at column 0,
explicit result types, no proof-shaped result type, no `Sort`, no `_root_`. It is not a Lean
lexer. The authority is the `--execute` environment audit: Lean's own `ConstantInfo` of every
constant the modules add (`.lake/evidence/Environment.lean`) refuses theorems, axioms, opaque
constants, instances, non-structure inductives, definitions whose type is a proposition and any
transitive axiom outside propext, Classical.choice, Quot.sound (structure- and definition-
generated auxiliaries are tolerated only when range-less or flagged by the environment), and
requires the statement definitions (type syntactically `∀ …, Prop`, no reduction, so a
`Set`-valued definition is auxiliary exactly as the scan's `: Prop` rule has it), structures and
auxiliary definitions to equal the register and the scan exactly.
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
PACKAGE = 'specifications/lean_statements_20261010'
REPOSITORY = 'd6g8k5htny-coder/Math-'
WORKFLOW = '.github/workflows/lean-specifications.yml'
REFERENCE_LOCK = 'formal/lake-manifest.json'
ROOT_MODULE = 'Specifications'
MODULES = ('Specifications/Field.lean', 'Specifications/Lifetime.lean', 'Specifications/Side24.lean',
           'Specifications/RN.lean', 'Specifications/P15.lean')
NAMESPACE = 'UniversalLaw.Spec.'
TOOLCHAIN = 'leanprover/lean4:v4.34.1'
LEAN_COMMIT = '5045d0056413266e57c625dcd7c365b10e377c52'
MATHLIB_REV = 'd13f23b723b8a846827a245b89c10fc7d3f11612'
LOCK_NAME = 'universal_law_specifications'
REGISTER_OBJECT = 'LEAN-STATEMENT-REGISTER-20261010-v1'
RECEIPT_OBJECT = 'LEAN-STATEMENT-RECEIPT-20261010-v1'
LANDING_SOURCE = 'landing-claims'
FORMAL_PROGRESS = ('none', 'specified', 'proved', 'kernel-checked')
CLAIM_PROGRESS = ('none', 'specified')
CLAIM_KINDS = ('landing_claim', 'graph_node', 'proof_index_result', 'open_obligation',
               'existing_lean_package', 'main_experiment_proof', 'exact_certificate')
STATEMENT_KINDS = ('def_prop', 'structure', 'theorem')
LOCAL_STATEMENT_KINDS = ('def_prop', 'structure')
DISPOSITIONS = ('HOLD_WITH_DOMAIN', 'FAIL_CLOSED', 'EXACT_COUNTEREXAMPLE', 'REVIEWED_SCOPED', 'ENGINEERING_HOLD')
ALLOWED_AXIOMS = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
PUBLIC_REPOS = frozenset({'d6g8k5htny-coder/main', 'd6g8k5htny-coder/Math-', 'd6g8k5htny-coder/query-',
                          'd6g8k5htny-coder/meta-framework', 'd6g8k5htny-coder/Universal-Law-Workspace'})
TOP_KEYS = frozenset({'schema_version', 'object', 'scientific_effect', 'scientific_status_authority',
                      'alignment_status', 'meaning', 'toolchain', 'vocabulary', 'sources', 'claims'})
SOURCE_KEYS = frozenset({'id', 'repository', 'commit', 'path', 'bytes', 'sha256', 'local_copy'})
SOURCES_JSON_KEYS = SOURCE_KEYS | {'blob'}
ROW_KEYS = frozenset({'claim_id', 'kind', 'title', 'layer0', 'disposition_transcribed', 'formal_progress',
                      'statements', 'kernel_evidence', 'not_established', 'next_step'})
STATEMENT_KEYS = frozenset({'package', 'module', 'declaration', 'kind', 'formal_progress', 'anchor', 'does_not_claim'})
EVIDENCE_KEYS = frozenset({'package', 'receipt_or_workflow', 'note'})
# Lexical refusal list (DESIGN 3b): proof-bearing, trust-affecting or syntax-bending commands and
# terms, plus commands that add declarations under another name (export, alias, irreducible_def),
# `Sort` (Prop spelled as `Sort 0`) and `_root_` (escapes the module namespace). Matched as whole
# identifier tokens of the comment-stripped code, so a docstring may mention them. Not an
# elaboration check: the environment audit in --execute is the Lean-side authority.
REFUSED_TOKENS = ('theorem', 'lemma', 'example', 'instance', 'axiom', 'sorry', 'admit', 'native_decide',
                  'unsafe', 'opaque', 'partial', 'macro', 'macro_rules', 'notation', 'infix', 'infixl',
                  'infixr', 'prefix', 'postfix', 'syntax', 'elab', 'elab_rules', 'set_option', 'run_cmd',
                  'run_tac', 'run_elab', 'implemented_by', 'extern', 'decreasing_by', 'ofReduceBool',
                  'trustCompiler', 'initialize', 'builtin_initialize', 'deriving', 'attribute',
                  'include', 'omit', 'mutual', 'inductive', 'class', 'local', 'scoped',
                  'export', 'alias', 'irreducible_def', 'proof_wanted', 'suppress_compilation',
                  'initialize_simps_projections', 'register_simp_attr', 'declare_aesop_rule_sets',
                  'add_decl_doc', 'library_note', 'assert_not_exists', 'assert_exists', 'compile_inductive',
                  'compile_def', 'recall', 'whatsnew', 'count_heartbeats', 'unseal', 'seal', 'nonrec',
                  'declare_syntax_cat', 'binder_predicate', 'register_option', 'register_label_attr',
                  'register_hint', 'to_additive', 'simps', 'with_weak_namespace', 'Sort', '_root_')
# Characters that have no place in the code of a statement module: string literals and guillemet
# names (both can hide `/-`, `-/` or `--` from a comment lexer; Lean tokenizes them first),
# `#` commands (`#eval` runs code at build time) and attributes.
REFUSED_LITERALS = ('"', '«', '»', '#', '@[')
# Column-0 commands a statement module may use; everything else is refused. DESIGN 3b also lists
# `section` and `variable`, but no module uses them and a `variable (h : False)` mentioned in a
# body would add a vacuating hypothesis the scan cannot see (only `#check` in --execute shows the
# elaborated signature), so both are refused until a module needs them.
ALLOWED_COMMANDS = frozenset({'import', 'namespace', 'end', 'open', 'def', 'noncomputable', 'abbrev', 'structure'})
# Lean commands are not indentation-sensitive and `in` chains them, so every command keyword must
# start a line (the only exception: `def`/`abbrev`/`structure` right after `noncomputable ` at
# column 0). `private`, `protected`, `universe`, `section` and `variable` are then refused by the
# allow-list.
COMMAND_KEYWORDS = ALLOWED_COMMANDS | {'private', 'protected', 'universe', 'section', 'variable'}
COMMAND_KEYWORD = re.compile(r"(?<![A-Za-z0-9_.'])(" + '|'.join(sorted(COMMAND_KEYWORDS)) + r")(?![A-Za-z0-9_'])")
IN_TOKEN = re.compile(r"(?<![A-Za-z0-9_.'])in(?![A-Za-z0-9_'])")
# A result type with a bracket-depth-zero relation or a bare True/False is a proposition, so the
# definition would be a proof. Heuristic, pre-Lean; the environment audit decides by `isProp`.
PROOF_SHAPE = re.compile(r'(?<![A-Za-z0-9_.])(True|False)(?![A-Za-z0-9_])|[=≠<≤>≥∧∨↔¬∀∃∈∉⊆⊂∣≈]')
LEAN_OPTIONS_LINE = 'leanOptions = { autoImplicit = false }'
PLACEHOLDER = re.compile(r'<<[^<>\n]*>>')
LABELS = ('Source:', 'Anchor:', 'Interface:', 'Does not claim:')
# A verbatim anchor shorter than this locates nothing (a one-character anchor is verbatim in any
# source); docstring and register anchors are both held to it.
MIN_ANCHOR = 16
FIXED_FILES = ('.gitignore', 'README.md', 'SCOPE.md', 'REGISTER.json', 'check.py', 'test_check.py', 'replay.sh',
               'lakefile.toml', 'lean-toolchain', 'lake-manifest.json', ROOT_MODULE + '.lean', 'sources/SOURCES.json')
SKIP_DIRS = frozenset({'.lake', '__pycache__'})
HEX40 = re.compile(r'[0-9a-f]{40}')
HEX64 = re.compile(r'[0-9a-f]{64}')
AXIOM_LINE = re.compile(r"'([^'\s]+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
AUDIT_LINE = re.compile(r'\bAUDIT (\S+) (\S+)$', re.M)
AUDIT_END = re.compile(r'\bAUDIT END (\d+) (\d+)$', re.M)
AUDIT_CLASSES = frozenset({'statement', 'structure', 'auxiliary', 'generated'})
LEAN_ERROR = re.compile(r':\d+:\d+: error(?:\([^)]*\))?:')  # Lean 4.34 may tag errors: `error(lean.unknownIdentifier):`
BRACKETS = {'(': ')', '[': ']', '{': '}', '⦃': '⦄', '⟨': '⟩'}
CLOSERS = frozenset(BRACKETS.values())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


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
    block comment, string literal or guillemet name is refused. A string literal `"…"` (with
    backslash escapes) or a guillemet name `«…»` outside a comment is copied verbatim into the
    code, so a `/-`, `-/` or `--` inside it never opens or closes a comment and the literal
    itself remains visible to the refusal of `"` and `«` in statement modules."""
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


def result_type(hdr: str):
    """Text after the last bracket-depth-zero `:` of a header, whitespace-normalized, or None."""
    depth, last = 0, None
    for p, c in enumerate(hdr):
        if c in BRACKETS:
            depth += 1
        elif c in CLOSERS:
            depth -= 1
        elif c == ':' and depth == 0 and hdr[p:p + 2] != ':=' and (p == 0 or hdr[p - 1] != ':'):
            last = p
    return None if last is None else ' '.join(hdr[last + 1:].split())


def docstring_labels(raw: str, name: str, sources: dict) -> dict:
    """The four labelled lines DESIGN 3b requires of a specification docstring. The `Source:` line
    must name a registered pinned source exactly; the quoted `Anchor:` must be verbatim in it."""
    body = raw[3:-2]
    found = {}
    for label in LABELS:
        hits = re.findall(r'^' + re.escape(label) + r'(.*)$', body, re.M)
        require(len(hits) == 1, f'{name}: docstring must carry exactly one `{label}` line (found {len(hits)})')
        # `Interface:` and `Does not claim:` may continue on following lines; the two parsed labels may not.
        tail = body[re.search(r'^' + re.escape(label), body, re.M).end():]
        require(tail.strip(), f'{name}: empty `{label}` entry in docstring')
        require(label in ('Interface:', 'Does not claim:') or hits[0].strip(), f'{name}: `{label}` must carry its value on the same line')
        found[label] = hits[0].strip()
    m = re.fullmatch(r'(\S+) commit ([0-9a-f]{40}) path (\S+) sha256 ([0-9a-f]{64})', found['Source:'])
    require(m is not None, f'{name}: `Source:` must read `<repository> commit <40-hex> path <path> sha256 <64-hex>`')
    matches = [s for s in sources.values() if (s['repository'], s['commit'], s['path'], s['sha256']) == m.groups()]
    require(len(matches) == 1, f'{name}: `Source:` line names no registered pinned source')
    a = re.fullmatch(r'"(.+)"', found['Anchor:'])
    require(a is not None, f'{name}: `Anchor:` must be one double-quoted verbatim line')
    require(len(a.group(1)) >= MIN_ANCHOR, f'{name}: docstring anchor shorter than {MIN_ANCHOR} characters locates nothing')
    require(a.group(1) in matches[0]['text'],
            f'{name}: docstring anchor is not a verbatim substring of pinned source {matches[0]["id"]}')
    return {'source_id': matches[0]['id'], 'doc_anchor': a.group(1)}


def depth_zero(text: str) -> str:
    """`text` with everything inside brackets blanked, for depth-zero pattern tests."""
    depth, out = 0, []
    for ch in text:
        if ch in BRACKETS:
            depth += 1
        elif ch in CLOSERS:
            depth -= 1
        out.append(ch if depth == 0 and ch not in CLOSERS else ' ')
    return ''.join(out)


def scan_module(text: str, stem: str, sources: dict) -> list:
    """Top-level declarations of one statement module: fully qualified name, Lean keyword, result
    type, and kind `def_prop` (result exactly `Prop`), `structure` or None (auxiliary definition).
    Refuses the lexical list, string literals and guillemet names, `#` commands and attributes,
    any command keyword off column 0 (Lean commands are not indentation-sensitive), `open … in`,
    any column-0 command outside the allowed set (`section` and `variable` included, until a
    module needs them), imports other than `Mathlib`, a namespace other than
    UniversalLaw.Spec.<stem>, unbalanced scoping, declaration
    names with a `_`-prefixed component, definitions without an explicit result type, Prop-valued
    result types not written exactly as `Prop`, proof-shaped result types, and registered-kind
    declarations without the four labelled docstring lines. A pre-Lean stage: the environment
    audit of --execute is the authority on what the modules declare."""
    code, docs = lex(text)
    bad = refused_tokens(code)
    require(not bad, f'refused token(s) in {stem}: {bad!r}')
    literals = [lit for lit in REFUSED_LITERALS if lit in code]
    require(not literals, f'refused character(s) in the code of {stem}: {literals!r} (string literals, guillemet names, '
            '`#` commands and attributes have no place in a statement module)')
    for m in COMMAND_KEYWORD.finditer(code):
        head = code[code.rfind('\n', 0, m.start()) + 1:m.start()]
        require(head == '' or (head == 'noncomputable ' and m.group(1) in ('def', 'abbrev', 'structure')),
                f'{stem}:{code.count(chr(10), 0, m.start()) + 1}: command keyword `{m.group(1)}` off column 0 '
                '(every command must start its own line; `open … in` and indented commands are refused)')
    imports = re.findall(r'^import\s+(\S+)\s*$', code, re.M)
    require(imports == ['Mathlib'], f'{stem}: imports must be exactly `import Mathlib`: {imports!r}')
    expected_ns = NAMESPACE + stem
    doc_end = {end: raw for _, end, raw in docs}
    lines = code.split('\n')
    stack, namespaces, decls = [], [], []
    for i, line in enumerate(lines):
        if not line or line[0].isspace():
            continue
        tok = line.split()[0]
        require(tok in ALLOWED_COMMANDS, f'{stem}:{i + 1}: column-0 command not allowed in a statement module: {tok!r}')
        if tok == 'import':
            continue
        m = re.match(r'^namespace(?:\s+(\S+))?\s*$', line)
        if m:
            require(m.group(1), f'{stem}:{i + 1}: anonymous namespace')
            namespaces.append(m.group(1))
            stack.append(('namespace', m.group(1)))
            continue
        m = re.match(r'^end(?:\s+(\S+))?\s*$', line)
        if m:
            require(stack, f'{stem}:{i + 1}: `end` without an open namespace')
            kind, name = stack.pop()
            require(name == m.group(1), f'{stem}:{i + 1}: `end {m.group(1)}` does not close `{kind} {name}`')
            continue
        if tok == 'open':
            block, k = [line], i + 1
            while k < len(lines) and lines[k] and lines[k][0].isspace():
                block.append(lines[k])
                k += 1
            require(IN_TOKEN.search(depth_zero('\n'.join(block))) is None,
                    f'{stem}:{i + 1}: `open … in` is not allowed (it chains a command onto this line)')
            continue
        m = re.match(r'^(noncomputable )?(def|abbrev|structure)\s+([^\s(\[{⦃:]+)(.*)$', line)
        require(m is not None, f'{stem}:{i + 1}: unparsable declaration line: {line[:80]!r}')
        require(not any(part.startswith('_') for part in m.group(3).split('.')),
                f'{stem}:{i + 1}: declaration name components may not start with `_` (Lean treats them as internal details): {m.group(3)}')
        prefix = [n for k, n in stack if k == 'namespace']
        require(prefix and prefix[0] == expected_ns, f'{stem}:{i + 1}: declaration outside namespace {expected_ns}')
        fq = '.'.join(prefix + [m.group(3)])
        hdr = header(lines, i, m.group(4))
        res = None if m.group(2) == 'structure' else result_type(hdr)
        if m.group(2) == 'structure':
            kind = 'structure'
        elif res == 'Prop':
            kind = 'def_prop'
        else:
            require(res is not None, f'{fq}: every definition must carry an explicit result type (`: Prop` for a statement)')
            require(re.search(r'(?<![A-Za-z0-9_.])Prop(?![A-Za-z0-9_])', res) is None,
                    f'{fq}: a Prop-valued result type must be written exactly as `: Prop` (found {res!r})')
            require(PROOF_SHAPE.search(depth_zero(res)) is None,
                    f'{fq}: proof-shaped result type {res!r}: a definition whose type is a proposition is a proof, not a statement')
            kind = None
        decl = {'name': fq, 'lean': m.group(2), 'kind': kind, 'result': res, 'line': i + 1}
        if kind:
            j = i if i in doc_end else i - 1
            while j >= 0 and j not in doc_end and not lines[j].strip():
                j -= 1
            require(j in doc_end, f'{fq}: registered-kind declaration has no docstring immediately before it')
            decl.update(docstring_labels(doc_end[j], fq, sources))
        decls.append(decl)
    require(not stack, f'{stem}: unclosed namespace: {stack!r}')
    require(namespaces == [expected_ns], f'{stem}: exactly one namespace {expected_ns} is required: {namespaces!r}')
    return decls


# ----------------------------------------------------------------------------- packet checks

def git(repo: Path, *args, timeout=120):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=timeout)


def check_identity(repo: Path, allow_dirty: bool):
    proc = git(repo, 'rev-parse', 'HEAD')
    head = proc.stdout.strip() if proc.returncode == 0 and HEX40.fullmatch(proc.stdout.strip()) else None
    if allow_dirty:
        return head, 'UNVERIFIED'
    require(head is not None, 'not a git repository with a 40-hex HEAD (local development may pass --allow-dirty)')
    status = git(repo, 'status', '--porcelain', '--untracked-files=all', '--', PACKAGE, WORKFLOW, REFERENCE_LOCK)
    require(status.returncode == 0, 'git status failed: ' + status.stderr.strip()[:300])
    require(status.stdout.strip() == '',
            'packet, workflow or reference lock differ from HEAD (uncommitted or untracked):\n' + status.stdout[:2000])
    require(git(repo, 'ls-files', '--error-unmatch', WORKFLOW).returncode == 0, 'workflow is not tracked: ' + WORKFLOW)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        require(os.environ.get('GITHUB_SHA') == head, 'GITHUB_SHA differs from the checked-out HEAD')
        require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY, 'unexpected GITHUB_REPOSITORY')
    return head, 'CLEAN'


def load_sources(root: Path, repo, reg: dict):
    """REGISTER.json sources must equal sources/SOURCES.json (minus the git blob kept there);
    every local copy must match bytes, sha256 and blob; when the pinned commit is present in the
    repository history the copy must also equal `<commit>:<path>`."""
    sj = load_json(safe_path(root, 'sources/SOURCES.json').read_text(encoding='utf-8'))
    require(isinstance(sj, dict) and set(sj) == {'schema_version', 'scientific_effect', 'meaning', 'sources'}
            and sj['schema_version'] == 1 and sj['scientific_effect'] == 'NONE' and isinstance(sj['sources'], list),
            'sources/SOURCES.json schema')
    by_id = {}
    for s in sj['sources']:
        require(isinstance(s, dict) and set(s) == SOURCES_JSON_KEYS, 'sources/SOURCES.json rows have a fixed schema')
        require(s['id'] not in by_id, 'duplicate source id: ' + str(s['id']))
        by_id[s['id']] = s
    rows = reg['sources']
    require(isinstance(rows, list) and [r.get('id') for r in rows if isinstance(r, dict)] == list(by_id),
            'REGISTER.json sources must list exactly the sources/SOURCES.json ids in order')
    out, history = {}, 0
    for r in rows:
        require(set(r) == SOURCE_KEYS, 'register source rows have a fixed schema: ' + str(r.get('id')))
        s = by_id[r['id']]
        require(all(r[k] == s[k] for k in SOURCE_KEYS), f'register source {r["id"]} differs from sources/SOURCES.json')
        require(r['repository'] in PUBLIC_REPOS, 'nonpublic source repository refused: ' + str(r['repository']))
        require(isinstance(r['commit'], str) and HEX40.fullmatch(r['commit']), 'invalid commit for ' + r['id'])
        require(isinstance(r['sha256'], str) and HEX64.fullmatch(r['sha256']), 'invalid sha256 for ' + r['id'])
        require(type(r['bytes']) is int and r['bytes'] > 0, 'invalid byte count for ' + r['id'])
        require(isinstance(r['local_copy'], str) and r['local_copy'].startswith('sources/') and r['local_copy'] not in
                [o['local_copy'] for o in out.values()], 'local copies live under sources/ and are unique: ' + r['id'])
        data = safe_path(root, r['local_copy']).read_bytes()
        require(len(data) == r['bytes'] and sha(data) == r['sha256'] and git_blob(data) == s['blob'],
                f'local copy differs from pinned identity (bytes, sha256, blob): {r["id"]}')
        if repo is not None and git(repo, 'cat-file', '-e', f'{r["commit"]}:{r["path"]}').returncode == 0:
            blob = subprocess.run(['git', '-C', str(repo), 'cat-file', 'blob', f'{r["commit"]}:{r["path"]}'],
                                  capture_output=True, timeout=120)
            require(blob.returncode == 0 and blob.stdout == data,
                    f'local copy differs from {r["commit"][:12]}:{r["path"]} in git history: {r["id"]}')
            history += 1
        out[r['id']] = {**r, 'blob': s['blob'], 'text': data.decode('utf-8')}
    return out, history


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


def check_tree(root: Path, sources: dict, modules):
    present = packet_files(root)
    expected = sorted(set(FIXED_FILES) | set(modules) | {s['local_copy'] for s in sources.values()})
    require(present == expected, 'packet tree differs from the fixed inventory: unexpected '
            + repr(sorted(set(present) - set(expected))) + ', missing ' + repr(sorted(set(expected) - set(present))))
    require('.lake/' in (root / '.gitignore').read_text(encoding='utf-8').splitlines(), '.gitignore must ignore .lake/')


def check_lock(root: Path, repo: Path) -> dict:
    require((root / 'lean-toolchain').read_bytes() == (TOOLCHAIN + '\n').encode(), 'lean-toolchain must read exactly ' + TOOLCHAIN)
    lakefile = (root / 'lakefile.toml').read_text(encoding='utf-8')
    for pattern in (r'^name = "' + LOCK_NAME + r'"$', r'^defaultTargets = \["' + ROOT_MODULE + r'"\]$',
                    r'^\[\[require\]\]$', r'^name = "mathlib"$',
                    r'^git = "https://github\.com/leanprover-community/mathlib4\.git"$',
                    r'^rev = "' + MATHLIB_REV + r'"$', r'^\[\[lean_lib\]\]$', r'^name = "' + ROOT_MODULE + r'"$',
                    '^' + re.escape(LEAN_OPTIONS_LINE) + '$'):
        require(re.search(pattern, lakefile, re.M) is not None, 'lakefile.toml lacks the pinned line ' + pattern)
    require(lakefile.count('[[require]]') == 1 and lakefile.count('[[lean_lib]]') == 1,
            'lakefile.toml must declare exactly one dependency and one library')
    lock = load_json((root / 'lake-manifest.json').read_text(encoding='utf-8'))
    require(isinstance(lock, dict) and lock.get('name') == LOCK_NAME, 'lake-manifest.json name must be ' + LOCK_NAME)
    ref = load_json(safe_path(repo, REFERENCE_LOCK).read_text(encoding='utf-8'))
    require({k: v for k, v in lock.items() if k != 'name'} == {k: v for k, v in ref.items() if k != 'name'},
            'lake-manifest.json differs from ' + REFERENCE_LOCK + ' beyond the package name')
    packages = lock['packages']
    revs = {p['name']: p['rev'] for p in packages}
    require(len(revs) == len(packages) and all(isinstance(v, str) and HEX40.fullmatch(v) for v in revs.values()),
            'duplicate or unpinned dependency in lake-manifest.json')
    require(revs.get('mathlib') == MATHLIB_REV, 'lake-manifest.json Mathlib rev differs from the pin')
    return revs


def check_root(root: Path, modules):
    code, _ = lex((root / (ROOT_MODULE + '.lean')).read_text(encoding='utf-8'))
    lines = [line for line in code.splitlines() if line.strip()]
    expected = ['import ' + m[:-5].replace('/', '.') for m in modules]
    require(lines == expected, 'root module must import exactly the registered modules in order: ' + repr(lines))


def check_docs(root: Path):
    """README.md and SCOPE.md may not carry unfilled `<<…>>` placeholders: a committed packet
    whose tables are placeholders would describe itself as more than it is."""
    for name in ('README.md', 'SCOPE.md'):
        hits = PLACEHOLDER.findall((root / name).read_text(encoding='utf-8'))
        require(not hits, f'{name} still carries unfilled placeholder(s): {hits[:4]!r}')


def landing_dispositions(sources: dict) -> dict:
    if LANDING_SOURCE not in sources:
        return {}
    data = json.loads(sources[LANDING_SOURCE]['text'])
    out = {}
    for c in data['claims']:
        require(isinstance(c.get('claim_id'), str) and c['claim_id'] not in out and c.get('disposition') in DISPOSITIONS,
                'LANDING_CLAIMS copy has an unexpected claim row')
        out[c['claim_id']] = c['disposition']
    return out


def check_register(reg: dict, scans: dict, sources: dict) -> dict:
    """Vocabulary and the two-way inventory: every registered statement of this package is a
    scanned Prop definition or structure of the named module, and every scanned Prop definition
    or structure is registered. Dispositions are transcriptions of LANDING_CLAIMS, never values
    decided here. Claim-level and statement-level labels can never say proved or kernel-checked
    for this package."""
    require(isinstance(reg, dict) and set(reg) == TOP_KEYS, 'REGISTER.json top-level keys must be exactly ' + repr(sorted(TOP_KEYS)))
    require(type(reg['schema_version']) is int and reg['schema_version'] == 1, 'schema_version must be 1')
    require(reg['object'] == REGISTER_OBJECT, 'object must be ' + REGISTER_OBJECT)
    require(reg['scientific_effect'] == 'NONE', 'scientific_effect must be NONE')
    require(reg['scientific_status_authority'] is False, 'scientific_status_authority must be false')
    require(reg['alignment_status'] == 'PENDING_INDEPENDENT_REVIEW',
            'alignment_status must be PENDING_INDEPENDENT_REVIEW (a register cannot self-award a review)')
    require(isinstance(reg['meaning'], str) and reg['meaning'].strip(), 'meaning must be a nonempty string')
    require(reg['toolchain'] == {'lean': TOOLCHAIN, 'lean_commit': LEAN_COMMIT, 'mathlib': MATHLIB_REV}, 'toolchain pins differ')
    require(reg['vocabulary'] == {'formal_progress': list(FORMAL_PROGRESS)}, 'vocabulary must be exactly the four formal-progress labels')
    claims = reg['claims']
    require(isinstance(claims, list), 'claims must be a list')
    landing = landing_dispositions(sources)
    by_name = {(mod, d['name']): d for mod, decls in scans.items() for d in decls}
    required = {(mod, d['name'], d['kind']) for mod, decls in scans.items() for d in decls if d['kind']}
    registered, ids, n_statements = set(), set(), 0
    for row in claims:
        require(isinstance(row, dict) and set(row) == ROW_KEYS, 'claim rows have a fixed schema; offending keys: '
                + repr(sorted(set(row) ^ ROW_KEYS)) if isinstance(row, dict) else 'claim row must be an object')
        cid = row['claim_id']
        require(isinstance(cid, str) and cid.strip() and cid not in ids, 'claim_id must be a unique nonempty string: ' + repr(cid))
        ids.add(cid)
        require(row['kind'] in CLAIM_KINDS, f'{cid}: kind must be one of {CLAIM_KINDS}')
        require(isinstance(row['title'], str) and row['title'].strip(), f'{cid}: title must be nonempty')
        layer0 = row['layer0']
        require(isinstance(layer0, dict) and set(layer0) == {'source_id', 'heading'}, f'{cid}: layer0 must be {{source_id, heading}}')
        sid = layer0['source_id']
        require(sid is None or sid in sources, f'{cid}: layer0.source_id must be null or a registered source id')
        require(layer0['heading'] is None or (isinstance(layer0['heading'], str) and layer0['heading'].strip()),
                f'{cid}: layer0.heading must be null or a nonempty string')
        disp = row['disposition_transcribed']
        require(disp is None or disp in DISPOSITIONS, f'{cid}: disposition_transcribed must be null or a LANDING_CLAIMS disposition')
        if row['kind'] == 'landing_claim':
            require(cid in landing, f'{cid}: landing_claim rows must carry a LANDING_CLAIMS claim_id (pinned source {LANDING_SOURCE})')
            require(disp == landing[cid], f'{cid}: disposition_transcribed must equal the LANDING_CLAIMS disposition verbatim ({landing[cid]!r})')
        else:
            require(disp is None or (cid in landing and disp == landing[cid]),
                    f'{cid}: disposition_transcribed must be null or the verbatim LANDING_CLAIMS disposition of this claim_id')
        fp = row['formal_progress']
        require(fp in CLAIM_PROGRESS, f'{cid}: claim-level formal_progress must be none or specified; no research claim of this '
                'programme is proved, and arithmetic skeletons carry their own labels under statements')
        require(isinstance(row['statements'], list), f'{cid}: statements must be a list')
        has_spec = False
        for st in row['statements']:
            require(isinstance(st, dict) and set(st) == STATEMENT_KEYS, f'{cid}: statement rows have a fixed schema')
            for k in ('package', 'module', 'declaration', 'anchor', 'does_not_claim'):
                require(isinstance(st[k], str) and st[k].strip(), f'{cid}: statement {k} must be a nonempty string')
            require(st['kind'] in STATEMENT_KINDS, f'{cid}/{st["declaration"]}: statement kind must be one of {STATEMENT_KINDS}')
            require(st['formal_progress'] in ('specified', 'proved'), f'{cid}/{st["declaration"]}: a statement is specified or '
                    'proved; never none, and kernel-checked appears only in a run receipt')
            if st['package'] == PACKAGE:
                require(st['kind'] in LOCAL_STATEMENT_KINDS, f'{cid}/{st["declaration"]}: statements of this package are def_prop or structure, never theorem')
                require(st['formal_progress'] == 'specified', f'{cid}/{st["declaration"]}: statements of this package are never proved or kernel-checked')
                key = (st['module'], st['declaration'])
                require(key in by_name and by_name[key]['kind'] is not None,
                        f'{cid}: {st["declaration"]} is not declared as a Prop definition or structure in {st["module"]}')
                require(by_name[key]['kind'] == st['kind'], f'{cid}: {st["declaration"]} is a {by_name[key]["kind"]}, registered as {st["kind"]}')
                src = sources[by_name[key]['source_id']]
                require(len(st['anchor']) >= MIN_ANCHOR, f'{cid}: anchor of {st["declaration"]} shorter than {MIN_ANCHOR} characters locates nothing')
                require(st['anchor'] in src['text'], f'{cid}: anchor of {st["declaration"]} is not a verbatim substring of pinned source '
                        f'{src["id"]} (the source its docstring names)')
                registered.add((st['module'], st['declaration'], st['kind']))
            else:
                require((st['kind'] == 'theorem') == (st['formal_progress'] == 'proved'),
                        f'{cid}/{st["declaration"]}: outside this package a theorem is proved and a def_prop or structure is specified')
                require(sid is not None, f'{cid}: statements outside this package need a layer0 source for their anchor')
                require(len(st['anchor']) >= MIN_ANCHOR, f'{cid}: anchor of {st["declaration"]} shorter than {MIN_ANCHOR} characters locates nothing')
                require(st['anchor'] in sources[sid]['text'], f'{cid}: anchor of {st["declaration"]} is not a verbatim substring of pinned source {sid}')
            has_spec = has_spec or st['kind'] == 'def_prop'
            n_statements += 1
        require(fp != 'specified' or has_spec, f'{cid}: specified requires a def_prop statement of the claim itself')
        require(isinstance(row['kernel_evidence'], list), f'{cid}: kernel_evidence must be a list')
        for ev in row['kernel_evidence']:
            require(isinstance(ev, dict) and set(ev) == EVIDENCE_KEYS and all(isinstance(ev[k], str) and ev[k].strip() for k in EVIDENCE_KEYS),
                    f'{cid}: kernel_evidence rows are {{package, receipt_or_workflow, note}} with nonempty strings')
        for k in ('not_established', 'next_step'):
            require(isinstance(row[k], str) and row[k].strip(), f'{cid}: {k} must be a nonempty string')
    missing = sorted(required - registered)
    require(not missing, f'unregistered declaration(s): every Prop definition and structure of the modules must appear in '
            f'REGISTER.json; {len(missing)} missing, first {missing[:8]!r}')
    return {'claims': len(claims), 'statements': n_statements, 'registered_local_statements': len(registered)}


def source_check(root: Path = ROOT, repo: Path = REPO, allow_dirty: bool = False, modules=MODULES) -> dict:
    root, repo = Path(root).resolve(), Path(repo).resolve()
    head, worktree = check_identity(repo, allow_dirty)
    reg_raw = safe_path(root, 'REGISTER.json').read_bytes()
    reg = load_json(reg_raw.decode('utf-8'))
    require(isinstance(reg, dict) and set(reg) == TOP_KEYS, 'REGISTER.json top-level keys must be exactly ' + repr(sorted(TOP_KEYS)))
    sources, history = load_sources(root, repo if head else None, reg)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        require(history == len(sources), f'in CI every pinned source must be verified against its commit in git history '
                f'(fetch-depth 0): {history} of {len(sources)} verified')
    check_tree(root, sources, modules)
    revs = check_lock(root, repo)
    check_root(root, modules)
    scans = {m: scan_module(safe_path(root, m).read_text(encoding='utf-8'), PurePosixPath(m).stem, sources) for m in modules}
    stats = check_register(reg, scans, sources)
    check_docs(root)
    decls = [d for m in modules for d in scans[m]]
    return {'scientific_effect': 'NONE', 'formalization_status': 'specified', 'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
            'checked_commit': head, 'worktree': worktree, 'register_sha256': sha(reg_raw),
            'module_sha256': {m: sha((root / m).read_bytes()) for m in modules}, 'dependency_revisions': revs,
            'declarations': len(decls), 'prop_definitions': sum(d['kind'] == 'def_prop' for d in decls),
            'structures': sum(d['kind'] == 'structure' for d in decls), 'auxiliary_definitions': sum(d['kind'] is None for d in decls),
            'sources': len(sources), 'sources_verified_in_git_history': history, **stats, 'scans': scans}


# ----------------------------------------------------------------------------- Lean execution

def lake_binary():
    found = shutil.which('lake')
    if found:
        return found
    candidate = Path.home() / '.elan/bin/lake'
    return str(candidate) if candidate.is_file() else None


def lean_env(lake: str) -> dict:
    return dict(os.environ, PATH=f'{Path(lake).parent}{os.pathsep}{os.environ.get("PATH", "")}')


def run(command, label, out: Path, cwd: Path, env, expect_success=True, timeout=1800):
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
    """`#print axioms` (parsed, allowed set only) and `#check` (logged, one signature per name).
    Returns the axiom records and the elaborated signatures."""
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
this file under `auditPrefix`) is classified by Lean's own `ConstantInfo`, never by source text:
`structure`; `statement` (a definition whose type is syntactically `Prop` or `∀ …, Prop`, read
through `forallTelescope` without reduction, so that a `Set`-valued definition, whose type
unfolds to `… → Prop`, is `auxiliary` exactly as the scan's `: Prop` rule has it; the reducing
`isProp` still decides `REFUSED:proof`); `auxiliary` (any
other definition); `generated` (what `structure` and `def` elaboration add: projections,
recursors, `casesOn`, `noConfusion`, `injEq`/`inj`/`sizeOf_spec`, `_sizeOf_*`, `_flat_ctor`,
`match_*`, `_proof_*`; tolerated only when range-less, flagged by the environment, or an
internal name under its structure); or `REFUSED:<why>` for a theorem, axiom, opaque or quotient
constant, an inductive type that is not a structure, a constructor or recursor of a foreign
type, an instance, a definition whose type is a proposition (a proof), or any transitive axiom
outside propext, Classical.choice and Quot.sound. One `AUDIT <class> <name>` line per constant,
then `AUDIT END <count> <refused>`; any refusal makes the file fail. -/
run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut constants : Array ConstantInfo := #[]
  if auditModules.isEmpty then
    constants := env.constants.foldStage2 (fun acc _ ci => if auditPrefix.isPrefixOf ci.name then acc.push ci else acc) #[]
  else
    let mut found : Array Name := #[]
    for (modName, data) in env.header.moduleNames.zip env.header.moduleData do
      if auditModules.contains modName then
        found := found.push modName
        constants := constants ++ data.constants
    unless found.size == auditModules.size do
      throwError "AUDIT REFUSED: audited module not imported: {auditModules.filter (!found.contains ·)}"
  let structs : Array Name := constants.filterMap fun ci => match ci with
    | .inductInfo _ => if isStructure env ci.name then some ci.name else none
    | _ => none
  let underStructure (n : Name) : Bool := structs.any fun s => s != n && s.isPrefixOf n
  let generatedTheorem (n : Name) : Bool := structs.any fun s =>
    let c := (getStructureCtor env s).name
    n == c ++ `injEq || n == c ++ `inj || n == c ++ `sizeOf_spec
  let mut refused := 0
  for ci in constants do
    let n := ci.name
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let inst ← liftCoreM (isInstance n)
    let (statement, proof) ← liftTermElabM do
      pure ((← forallTelescope ci.type fun _ body => pure body.isProp), (← isProp ci.type))
    let kind : String :=
      match ci with
        | .inductInfo _ => if structs.contains n then "structure" else "REFUSED:inductive"
        | .ctorInfo v => if structs.contains v.induct then "generated" else "REFUSED:constructor"
        | .recInfo _ => if underStructure n then "generated" else "REFUSED:recursor"
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .thmInfo _ =>
          if (underStructure n && env.isProjectionFn n) || (range.isNone && (generatedTheorem n || n.isInternalDetail))
          then "generated" else "REFUSED:theorem"
        | .defnInfo _ =>
          if (underStructure n && (env.isProjectionFn n || isAuxRecursor env n || isNoConfusion env n
                || n.isInternalDetail || range.isNone)) || (range.isNone && n.isInternalDetail) then "generated"
          else if inst then "REFUSED:instance"
          else if proof then "REFUSED:proof"
          else if statement then "statement"
          else "auxiliary"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    logInfo m!"AUDIT {verdict} {n}"
  logInfo m!"AUDIT END {constants.size} {refused}"
  if refused > 0 then throwError "AUDIT REFUSED {refused} constant(s); see the AUDIT lines"
'''


def audit_source(root_module: str, module_names: list, declarations: str = '') -> str:
    """The environment-audit file: imports, the audited module list (empty: this file), optional
    control declarations, and the fixed audit program."""
    modules = '#[' + ', '.join('`' + m for m in module_names) + ']'
    return (f'import Lean\nimport {root_module}\nopen Lean Meta Elab Command\n\n'
            f'/-- Modules whose constants are audited; empty means the constants of this file under `auditPrefix`. -/\n'
            f'def auditModules : Array Name := {modules}\n'
            f'/-- Namespace of the control declarations audited when `auditModules` is empty. -/\n'
            f'def auditPrefix : Name := `{CONTROL_NS[:-1]}\n{declarations}{AUDIT_PROGRAM}')


def parse_audit(log: str) -> dict:
    """`AUDIT <class> <name>` lines as {class: set(names)}; the `AUDIT END` line must account for
    every line and every refusal."""
    classes, seen = {}, set()
    for m in AUDIT_LINE.finditer(log):
        verdict, name = m.groups()
        require(name not in seen, 'duplicate audit line for ' + name)
        seen.add(name)
        classes.setdefault(verdict, set()).add(name)
    ends = AUDIT_END.findall(log)
    require(len(ends) == 1, 'the environment audit did not report exactly one END line')
    total, refused = int(ends[0][0]), int(ends[0][1])
    require(total == len(seen), f'environment audit reported {total} constants but {len(seen)} AUDIT lines')
    require(refused == sum(len(v) for k, v in classes.items() if k.startswith('REFUSED')), 'environment audit refusal count differs from its lines')
    return classes


def environment_audit(lake: str, root: Path, env: dict, out: Path, module_names: list, expected: dict,
                      root_module: str = ROOT_MODULE) -> dict:
    """Lean's own inventory of the modules (see AUDIT_PROGRAM) must pass with no refusal, and its
    statements, structures and auxiliary definitions must equal the register and the scan."""
    path = out / 'Environment.lean'
    path.write_text(audit_source(root_module, module_names), encoding='utf-8')
    classes = parse_audit(run([lake, 'env', 'lean', str(path)], 'environment', out, root, env))
    require(set(classes) <= AUDIT_CLASSES, 'environment audit refused constant(s): ' + repr(sorted(set(classes) - AUDIT_CLASSES)))
    for kind in ('statement', 'structure', 'auxiliary'):
        got, want = classes.get(kind, set()), set(expected[kind])
        require(got == want, f'environment audit: Lean\'s {kind} constants differ from the register and scan: '
                f'unexpected {sorted(got - want)[:8]!r}, missing {sorted(want - got)[:8]!r}')
    return {k: len(v) for k, v in sorted(classes.items())}


HONEST_CONTROL = '''
namespace UniversalLaw.Spec.Control
/-- Control: a parametrized statement. -/
def IsTwo (n : Nat) : Prop := 0 < n ∧ n = 2
/-- Control: an interface with a Prop field (its projection is a generated theorem). -/
structure Iface (d : Nat) where
  x : Nat
  pos : 0 < x
/-- Control: an extending interface. -/
structure Ext (d : Nat) extends Iface d where
  y : Nat
  hy : y = x
/-- Control: an auxiliary definition. -/
def helper (x : Nat) : Nat := x + 1
/-- Control: an auxiliary abbreviation. -/
abbrev E (d : Nat) : Type := Fin d → Nat
end UniversalLaw.Spec.Control
'''
HIDDEN_CONTROL = '''
namespace UniversalLaw.Spec.Control
def Opener : Prop := "/-" = "x"
theorem smuggledBetweenStrings : True := trivial
def Closer : Prop := "-/" = "x"
 def HiddenIndented : Prop := False
def proofAsDef : (2 : Nat) + 2 = 4 := rfl
def usesSorry : Prop := sorry
axiom hidden : False
instance controlInstance : Inhabited Nat := ⟨0⟩
end UniversalLaw.Spec.Control
'''
CONTROL_NS = 'UniversalLaw.Spec.Control.'


def lean_control_environment(lake: str, root: Path, env: dict, out: Path, root_module: str = ROOT_MODULE) -> str:
    """Two Lean-side controls of the environment audit itself, run over the constants of a control
    file (never a packet module): an honest file must be classified exactly, and a file hiding a
    theorem between string literals, an indented Prop definition, a proof written as `def`, a
    `sorry`, an `axiom` and an `instance` must be refused for exactly those reasons, with the
    hidden Prop definition reported as a statement the register does not carry."""
    honest = out / 'controls' / 'ControlEnvironmentHonest.lean'
    honest.write_text(audit_source(root_module, [], HONEST_CONTROL), encoding='utf-8')
    classes = parse_audit(run([lake, 'env', 'lean', str(honest)], 'control_environment_honest', out, root, env))
    require(classes.get('statement') == {CONTROL_NS + 'IsTwo'} and classes.get('structure') == {CONTROL_NS + 'Iface', CONTROL_NS + 'Ext'}
            and classes.get('auxiliary') == {CONTROL_NS + 'helper', CONTROL_NS + 'E'} and set(classes) == AUDIT_CLASSES,
            'honest environment control was not classified exactly: ' + repr({k: sorted(v) for k, v in classes.items()}))
    hidden = out / 'controls' / 'ControlEnvironmentHidden.lean'
    hidden.write_text(audit_source(root_module, [], HIDDEN_CONTROL), encoding='utf-8')
    log = run([lake, 'env', 'lean', str(hidden)], 'control_environment_hidden', out, root, env, expect_success=False)
    require('AUDIT REFUSED' in log and LEAN_ERROR.search(log) is not None, 'hidden environment control failed for the wrong reason')
    classes = parse_audit(log)
    expected = {'REFUSED:theorem': {CONTROL_NS + 'smuggledBetweenStrings'}, 'REFUSED:proof': {CONTROL_NS + 'proofAsDef'},
                'REFUSED:axioms:sorryAx': {CONTROL_NS + 'usesSorry'}, 'REFUSED:axiom': {CONTROL_NS + 'hidden'},
                'REFUSED:instance': {CONTROL_NS + 'controlInstance'},
                'statement': {CONTROL_NS + 'Opener', CONTROL_NS + 'Closer', CONTROL_NS + 'HiddenIndented'}}
    require(classes == expected, 'hidden environment control was not refused for exactly the expected reasons: '
            + repr({k: sorted(v) for k, v in classes.items()}))
    return ('REFUSED_BY_ENVIRONMENT_AUDIT (theorem between string literals, proof as def, sorry, axiom, instance; the indented '
            'Prop definition surfaces as a statement absent from any register) and the honest control classified exactly')


def lean_control_type_error(lake: str, root: Path, env: dict, out: Path, root_module: str = ROOT_MODULE) -> str:
    path = out / 'ControlTypeError.lean'
    path.write_text('import ' + root_module + '\n\nnamespace UniversalLaw.Spec.Control\n\n'
                    '/-- Negative control: a Prop whose body has type ℕ. Lean must refuse it. -/\n'
                    'def TypeErrorControl : Prop := (1 : Nat)\n\nend UniversalLaw.Spec.Control\n', encoding='utf-8')
    log = run([lake, 'env', 'lean', str(path)], 'control_type_error', out, root, env, expect_success=False)
    require('type mismatch' in log.lower() and LEAN_ERROR.search(log) is not None,
            'type-error control failed for the wrong reason: ' + log[-600:])
    return 'REJECTED_BY_LEAN'


def lean_control_missing(lake: str, root: Path, env: dict, out: Path, name: str, root_module: str = ROOT_MODULE) -> str:
    path = out / 'ControlMissing.lean'
    path.write_text('import ' + root_module + '\n#check ' + name + '\n', encoding='utf-8')
    log = run([lake, 'env', 'lean', str(path)], 'control_missing_declaration', out, root, env, expect_success=False)
    require(('unknown identifier' in log.lower() or 'unknown constant' in log.lower()) and LEAN_ERROR.search(log) is not None,
            'missing-declaration control failed for the wrong reason: ' + log[-600:])
    return 'REJECTED_BY_LEAN'


def control_row(module: str, declaration: str, anchor: str) -> dict:
    """A register row naming `declaration` with a real anchor, so that the only defect of the row
    is the declaration it names (the anchor itself passes the length and verbatim checks)."""
    require(len(anchor) >= MIN_ANCHOR, 'control row anchor must itself be a valid anchor')
    return {'claim_id': 'negative-control', 'kind': 'graph_node', 'title': 'negative control row (never committed)',
            'layer0': {'source_id': None, 'heading': None}, 'disposition_transcribed': None, 'formal_progress': 'specified',
            'statements': [{'package': PACKAGE, 'module': module, 'declaration': declaration, 'kind': 'def_prop',
                            'formal_progress': 'specified', 'anchor': anchor, 'does_not_claim': 'anything'}],
            'kernel_evidence': [], 'not_established': 'everything', 'next_step': 'none'}


def expect_refusal(fn, fragment: str, label: str):
    try:
        fn()
    except ValueError as exc:
        require(fragment in str(exc), f'negative control {label} was refused for the wrong reason: {exc}')
        return 'REFUSED_BY_SOURCE_CHECK'
    raise ValueError('negative control escaped the source check: ' + label)


def source_controls(root: Path, reg: dict, scans: dict, sources: dict, modules, ctrl: Path):
    """The source-side negative controls (no Lean): an injected theorem, a register row naming a
    missing declaration, an unregistered Prop definition spliced into a module copy, and a
    register copy labelling a statement `proved`. Each mutation is written under `ctrl` and run
    through the same functions the packet is held to; the packet itself is never modified.
    Returns the outcomes, the missing declaration name and the injected module text."""
    module = modules[0]
    stem = PurePosixPath(module).stem
    ns = NAMESPACE + stem
    outcomes = {}
    injected = f'import Mathlib\n\nnamespace {ns}\n\ntheorem injected : True := trivial\n\nend {ns}\n'
    outcomes['injected_theorem'] = expect_refusal(lambda: scan_module(injected, stem, sources), 'refused token', 'injected_theorem')
    local = [d for d in scans[module] if d['kind'] == 'def_prop']
    require(local, 'no registered Prop definition available to derive the controls')
    src = sources[local[0]['source_id']]
    missing = ns + '.DoesNotExistControl'
    bad = copy.deepcopy(reg)
    bad['claims'].append(control_row(module, missing, local[0]['doc_anchor']))
    outcomes['missing_declaration'] = expect_refusal(lambda: check_register(bad, scans, sources), 'is not declared', 'missing_declaration')
    text = safe_path(root, module).read_text(encoding='utf-8')
    tail = text.rfind('\nend ' + ns)
    require(tail > 0, 'module copy has no closing namespace line')
    extra = (f'\n/-- Negative control.\n\nSource: {src["repository"]} commit {src["commit"]} path {src["path"]} sha256 {src["sha256"]}\n'
             f'Anchor: "{local[0]["doc_anchor"]}"\nInterface: none.\nDoes not claim: anything. -/\n'
             'def UnregisteredControl : Prop := True\n')
    mutated = text[:tail] + extra + text[tail:]
    (ctrl / 'Unregistered.lean.txt').write_text(mutated, encoding='utf-8')
    mutated_scans = dict(scans, **{module: scan_module(mutated, stem, sources)})
    outcomes['unregistered_declaration'] = expect_refusal(lambda: check_register(reg, mutated_scans, sources),
                                                          'unregistered declaration', 'unregistered_declaration')
    proved = copy.deepcopy(reg)
    rows = [st for row in proved['claims'] for st in row['statements'] if st['package'] == PACKAGE]
    require(rows, 'no registered statement available to derive the proved-label control')
    rows[0]['formal_progress'] = 'proved'
    outcomes['proved_label'] = expect_refusal(lambda: check_register(proved, scans, sources), 'never proved', 'proved_label')
    (ctrl / 'register_mutations.json').write_text(json.dumps({'missing_declaration': bad['claims'][-1], 'proved_label': rows[0]},
                                                             indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return outcomes, missing, injected


def negative_controls(lake: str, root: Path, env: dict, out: Path, reg: dict, scans: dict, sources: dict, modules) -> dict:
    """Real controls: the four source-side ones, Lean runs showing that the injected theorem is
    accepted by Lean (so the scan is the refusal point) and that a type error and a missing
    declaration are refused by Lean, and the two controls of the environment audit itself."""
    ctrl = out / 'controls'
    ctrl.mkdir()
    outcomes, missing, injected = source_controls(root, reg, scans, sources, modules, ctrl)
    path = ctrl / 'InjectedTheorem.lean'
    path.write_text(injected.replace('import Mathlib', 'import ' + ROOT_MODULE), encoding='utf-8')
    run([lake, 'env', 'lean', str(path)], 'control_injected_theorem', out, root, env)
    outcomes['injected_theorem'] += ' (Lean accepts the file: exit 0, so the scan is the refusal point)'
    outcomes['type_error'] = lean_control_type_error(lake, root, env, out)
    outcomes['missing_declaration'] += ' and ' + lean_control_missing(lake, root, env, out, missing)
    outcomes['environment_audit'] = lean_control_environment(lake, root, env, out)
    return outcomes


def execute(root: Path, repo: Path, info: dict, allow_dirty: bool, modules=MODULES) -> dict:
    root, repo = Path(root).resolve(), Path(repo).resolve()
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
        shutil.rmtree(root / '.lake' / 'build')  # fresh build of this package only; dependency caches are untouched
    log = run([lake, 'build'], 'build', out, root, env)
    require(LEAN_ERROR.search(log) is None and re.search(r'\berror\b', log, re.I) is None, 'build log reports an error')
    require('declaration uses' not in log, 'build log reports sorry or a similar escape')
    require('is a proposition; use' not in log, 'build log reports a definition whose type is a proposition (a proof written as def)')
    run([lake, 'env', 'leanchecker', ROOT_MODULE], 'leanchecker', out, root, env)
    names = [d['name'] for m in modules for d in info['scans'][m] if d['kind']]
    require(names, 'no registered declarations to audit')
    axioms, sigs = kernel_audit(lake, root, env, out, names)
    expected = {'statement': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] == 'def_prop'],
                'structure': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] == 'structure'],
                'auxiliary': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] is None]}
    environment = environment_audit(lake, root, env, out, [m[:-5].replace('/', '.') for m in modules], expected)
    reg = load_json(safe_path(root, 'REGISTER.json').read_text(encoding='utf-8'))
    sources, _ = load_sources(root, repo if info['checked_commit'] else None, reg)
    outcomes = negative_controls(lake, root, env, out, reg, info['scans'], sources, modules)
    after = source_check(root, repo, allow_dirty, modules)
    require(after['register_sha256'] == info['register_sha256'] and after['module_sha256'] == info['module_sha256']
            and after['checked_commit'] == info['checked_commit'], 'packet changed during execution')
    declarations = [{'name': d['name'], 'module': m, 'kind': d['kind'], 'lean': d['lean'], 'source_id': d['source_id'],
                     'doc_anchor': d['doc_anchor'], 'line': d['line'], 'signature': sigs[d['name']], 'axioms': axioms[d['name']],
                     'formal_progress': 'specified'} for m in modules for d in info['scans'][m] if d['kind']]
    (out / 'declarations.json').write_text(json.dumps(declarations, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    receipt = {'schema_version': 1, 'object': RECEIPT_OBJECT, 'scientific_effect': 'NONE', 'scientific_status_authority': False,
               'package': PACKAGE, 'formalization_status': 'specified', 'alignment_status': 'PENDING_INDEPENDENT_REVIEW',
               'meaning': 'The registered statements elaborated against the pinned Mathlib at the checked commit and depend on '
                          'no axiom outside propext, Classical.choice and Quot.sound; the environment audit found no theorem, '
                          'axiom, opaque constant, instance or proof-typed definition in the modules and Lean\'s own inventory '
                          'of Prop-former definitions, structures and auxiliary definitions equals the register and the scan. '
                          'This is kernel evidence that the statements are well formed, not that they are true, aligned with '
                          'their sources, reviewed or accepted. No claim, premise or obligation changes status because of this receipt.',
               'checked_commit': info['checked_commit'], 'worktree': info['worktree'],
               'receipt_trust': 'CLEAN worktree at checked_commit' if info['worktree'] == 'CLEAN' else
                                'LOCAL_ONLY: the worktree was not verified against checked_commit; not evidence about any commit',
               'repository': os.environ.get('GITHUB_REPOSITORY'), 'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),
               'workflow_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'lean_version': version, 'toolchain': TOOLCHAIN,
               'dependency_revisions': revs, 'register_sha256': info['register_sha256'], 'module_sha256': info['module_sha256'],
               'declarations': len(names), 'prop_definitions': info['prop_definitions'], 'structures': info['structures'],
               'auxiliary_definitions': info['auxiliary_definitions'], 'claims': info['claims'], 'statements': info['statements'],
               'axioms': axioms, 'environment_audit': environment, 'lean_options': {'autoImplicit': False},
               'negative_controls': outcomes, 'build': 'PASS', 'leanchecker': 'PASS', 'dependency_worktrees': 'CLEAN'}
    receipt['logs'] = {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log'))}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return receipt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', nargs='?', choices=['source'],
                   help='optional; `source` names the default mode (DESIGN 3b), so `check.py source` and `check.py` are the same check')
    p.add_argument('--root', type=Path, default=ROOT, help='packet directory')
    p.add_argument('--repo', type=Path, default=REPO, help='repository root holding formal/lake-manifest.json')
    p.add_argument('--execute', action='store_true', help='fresh build, leanchecker, #check/#print axioms, negative controls, receipt')
    p.add_argument('--allow-dirty', action='store_true', help='local development only: skip the clean-worktree requirement')
    args = p.parse_args(argv)
    try:
        info = source_check(args.root, args.repo, args.allow_dirty)
        if info['worktree'] != 'CLEAN':
            print('WORKTREE_UNVERIFIED: packet identity against HEAD was not checked (--allow-dirty); nothing printed here '
                  'is evidence about a commit', file=sys.stderr)
        summary = {k: v for k, v in info.items() if k not in ('scans', 'module_sha256')}
        if args.execute:
            receipt = execute(args.root, args.repo, info, args.allow_dirty)
            print(json.dumps({'passed': True, 'formalization_status': receipt['formalization_status'],
                              'alignment_status': receipt['alignment_status'], 'checked_commit': receipt['checked_commit'],
                              'worktree': receipt['worktree'], 'declarations': receipt['declarations'],
                              'negative_controls': receipt['negative_controls']}, indent=2, sort_keys=True))
        else:
            print('SOURCE_CHECK_PASS (statements only: not a Lean build, not a proof, not a review, not an acceptance)')
            print(json.dumps(summary, indent=2, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        print('LEAN_SPEC_CHECK_FAIL: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
