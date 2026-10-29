#!/usr/bin/env python3
"""Source-bound Lean evidence, not a scientific acceptance engine (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ALLOWED = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
NAME = r'[A-Za-z_][A-Za-z0-9_.]*'
AXIOM = re.compile(r"'(" + NAME + r")' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
SHA256 = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
ROOT = Path(__file__).resolve().parent
PACKAGE = 'ResearchFormalCoreR1'
# Environment-bound inventory lines emitted by the generated Lean audit (RF-GATE-01).
# Every Lean name is printed in a lossless component encoding, never with Name.toString:
# components are joined by `/`; `s<hex>` is a string component (lowercase hex of its UTF-8
# bytes) and `n<decimal>` a numeric component. Private names (`_private.<module>.0.<name>`),
# quoted names (`«.lake»`) and names that differ only by escaping therefore stay distinct.
ENC = r'[sn][0-9a-f]*(?:/[sn][0-9a-f]*)*'
INV = re.compile(r'^INV\|(theorem|def|axiom|opaque|inductive|ctor|rec|quot)\|(user|aux)\|(' + ENC + r')\|(' + ENC + r')$')
MODULE = re.compile(r'^MODULE\|(' + ENC + r')\|(true|false)\|([0-9]+)\|([0-9]+)$')
AXIOMS = re.compile(r'^AXIOMS\|((?:' + ENC + r')(?:,' + ENC + r')*)?$')
CLOSURE = re.compile(r'^CLOSURE\|([0-9]+)$')
SIMPLE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
# Suffix components a compiler-generated theorem may carry below its user-written parent.
# Used only together with Lean's own provenance (no declaration range) and a user-written
# parent in the same module; never on its own to classify a constant.
AUX_SUFFIX = re.compile(r'_proof_[0-9]+|_simp_[0-9]+_[0-9]+|eq_[0-9]+|eq_def|match_[0-9]+')
LOCAL_MODULE = ('_local',)
# Conservative declaration grammar the source scanner understands: public theorems
# start at column 0 as `theorem NAME` or `lemma NAME`, definitions as `def NAME` or
# `noncomputable def NAME`; every other spelling (indented, private, protected,
# attributed on the same line) is refused.
DECL_KEYWORD = re.compile(r'(?<![A-Za-z0-9_.])(?:theorem|lemma|def)(?![A-Za-z0-9_])')
CANONICAL = re.compile(r'^(?:(theorem|lemma)|(?:noncomputable )?def) (' + NAME + r')(?=\s|$)')
FORBIDDEN_TOKENS = re.compile(r'(?<![A-Za-z0-9_.])(?:sorry|axiom|native_decide|implemented_by|extern|unsafe)(?![A-Za-z0-9_])')
# Declaration-bearing or metaprogramming commands the pre-check does not inventory.
UNSUPPORTED_COMMAND = re.compile(r'^\s*(?:#|(?:@\[[^\]]*\]\s*)?(?:noncomputable\s+)?(?:private|protected|opaque|abbrev|structure|class|instance|'
                                 r'inductive|mutual|syntax|macro|macro_rules|elab|elab_rules|initialize|builtin_initialize|example|'
                                 r'attribute|notation|infix|infixl|infixr|prefix|postfix|deriving|local|scoped|section|run_cmd|run_elab|run_meta)(?![A-Za-z0-9_]))')
SCOPE_COMMAND = re.compile(r'^\s*(?:namespace|end)(?![A-Za-z0-9_]).*$')
# Metaprogramming, declaration-bearing and elaborator tokens refused anywhere in the code,
# not only at the start of a line, so `set_option ... in run_elab`, `open Lean in ...` and
# term-level `by_elab` cannot slip past the line-anchored check (RF312-SUCCESSOR-COMMAND-004).
UNSUPPORTED_TOKEN = re.compile(r'(?<![A-Za-z0-9_.])(?:run_elab|run_cmd|run_meta|run_tac|by_elab|elab|elab_rules|macro|macro_rules|syntax|'
                               r'declare_syntax_cat|initialize|builtin_initialize|private|protected|opaque|abbrev|structure|class|'
                               r'instance|inductive|mutual|example|attribute|notation|infix|infixl|infixr|prefix|postfix|deriving|'
                               r'section|Lean|MetaM|TermElabM|TacticM|CoreM|CommandElabM|addDecl|evalTactic)(?![A-Za-z0-9_])|#[A-Za-z]')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def audit_axioms(text, targets):
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets), 'empty or duplicate target list')
    records = {}
    for match in AXIOM.finditer(text):
        name, raw = match.groups()
        require(name in targets and name not in records, 'unexpected or duplicate axiom target: ' + name)
        axioms = [] if raw is None or not raw.strip() else [a.strip() for a in raw.split(',')]
        require(all(re.fullmatch(NAME, a) for a in axioms), 'malformed axiom name')
        require(set(axioms) <= ALLOWED, 'forbidden transitive axiom: ' + repr(set(axioms) - ALLOWED))
        records[name] = sorted(set(axioms))
    require(not AXIOM.sub('', text).strip(), 'unrecognized audit output')
    require(set(records) == set(targets), 'missing target axiom report')
    return records

def check_files(root, files):
    require(isinstance(files, dict) and files, 'empty file manifest')
    root = root.resolve()
    for name, digest in files.items():
        p = PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p) == name and '\\' not in name, 'unsafe manifest path')
        require(isinstance(digest, str) and SHA256.fullmatch(digest), 'invalid source hash')
        target = root / p
        require(all(not (root.joinpath(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts) + 1)), 'symlink source')
        require(target.is_file() and target.resolve().is_relative_to(root), 'missing or escaping source: ' + name)
        require(sha(target.read_bytes()) == digest, 'source hash mismatch: ' + name)

def check_blueprint(text, targets):
    links = re.findall(r'\\lean\{([^}]+)\}', text)
    require(links and all(n in targets for n in links), 'missing or unresolved Blueprint declaration')
    require('\\leanok' not in text, 'unreviewed Blueprint completion label')
    return links

IDENT_CHAR = re.compile(r"[A-Za-z0-9_'!?\u00c0-\U0010ffff]")
CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[nrt\\'\"])|[^\\'\n])'")

def strip_lean_comments(text):
    """Remove `--` line comments and `/- ... -/` block comments (nesting-aware) before scanning.

    String and character literals are lexed first, so comment markers inside them are not
    comments; their contents are kept verbatim, which makes keyword and token refusals see
    them (stricter, never more permissive). Plain strings with escapes, raw strings
    (`r"..."`, `r#"..."#`) and character literals are understood; interpolated strings
    (`s!"..."`, and any plain string containing a brace, since macros such as `throwError`
    interpolate plain literals) and unterminated strings, characters or block comments
    are refused rather than guessed. Newlines are kept, so line numbers in refusals refer to
    the original file, and a block comment is replaced by one space so that the tokens on
    either side are never joined (`opaque/-x-/hidden` stays two tokens).
    """
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        two = text[i:i + 2]
        if depth:
            if two == '/-':
                depth += 1; i += 2
            elif two == '-/':
                depth -= 1; i += 2
                if not depth:
                    out.append(' ')  # a comment separates tokens; never join its neighbours
            else:
                if text[i] == '\n':
                    out.append('\n')
                i += 1
            continue
        prev = text[i - 1] if i else ''
        if two == '--':
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif two == '/-':
            depth += 1; i += 2
        elif text[i] == 'r' and not IDENT_CHAR.match(prev) and re.match(r'r#*"', text[i:]):
            hashes = len(re.match(r'r(#*)"', text[i:]).group(1))
            close = text.find('"' + '#' * hashes, i + 2 + hashes)
            require(close >= 0, 'unterminated raw string literal')
            end = close + 1 + hashes
            out.append(text[i:end]); i = end
        elif text[i] == '"':
            require(prev != '!', 'interpolated string literal is not supported by the pre-check')
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            require(j < n, 'unterminated string literal')
            require('{' not in text[i:j + 1], 'string literal with a brace (possible interpolation) is not supported by the pre-check')
            out.append(text[i:j + 1]); i = j + 1
        elif text[i] == "'" and not IDENT_CHAR.match(prev):
            literal = CHAR_LITERAL.match(text, i)
            require(literal, 'unsupported apostrophe or character literal')
            out.append(literal.group()); i = literal.end()
        else:
            out.append(text[i]); i += 1
    require(depth == 0, 'unterminated block comment')
    return ''.join(out)

def grammar_check(text, path, definitions=None):
    """Refuse declaration spellings the column-0 scanner cannot inventory, and admission tokens.

    The source scanner is only a pre-check (see GATE_HARDENING.md); the authoritative
    package-wide admission check is `audit_inventory` on the compiled environment.
    Returns the canonical theorem names in source order; canonical definition names are
    appended to `definitions` when a list is given.
    """
    code = strip_lean_comments(text)
    require(not FORBIDDEN_TOKENS.search(code), 'forbidden token in source module: ' + path)
    token = UNSUPPORTED_TOKEN.search(code)
    if token:
        raise ValueError('unsupported metaprogramming or declaration token ' + repr(token.group()) + ' at ' + path + ':' + str(code.count('\n', 0, token.start()) + 1))
    names, declared, scopes = [], [], []
    for number, line in enumerate(code.splitlines(), 1):
        where = path + ':' + str(number)
        require(not UNSUPPORTED_COMMAND.match(line), 'unsupported declaration-bearing or metaprogramming command at ' + where)
        if SCOPE_COMMAND.match(line):
            scopes.append(line.rstrip())
        keywords = len(DECL_KEYWORD.findall(line))
        if not keywords:
            continue
        canonical = CANONICAL.match(line)
        require(canonical and keywords == 1, 'unsupported declaration spelling (indented, private, protected, attributed or shared line) at ' + where)
        declared.append(canonical.group(2))
        if canonical.group(1):
            names.append(canonical.group(2))
        elif definitions is not None:
            definitions.append(canonical.group(2))
    require(scopes == ['namespace ' + PACKAGE, 'end ' + PACKAGE], 'unsupported namespace structure (exactly one column-0 namespace ' + PACKAGE + ' block is allowed): ' + path)
    require(len(declared) == len(set(declared)), 'duplicate source declaration: ' + path)
    return names

def inventory_source(include_local=False):
    """Lean program that enumerates every constant of every package module from the compiled
    environment's module tables, records Lean's own provenance for each, and axiom-closes all
    of them in one shared traversal.

    The enumeration reads `env.header.moduleData[idx].constNames` for the modules under the
    package prefix (the whole-environment scan `env.constants.map₁` takes minutes in the
    interpreter; the module tables are the same data indexed by module). Provenance is
    `findDeclarationRanges?`: `user` when Lean stored a source range for the constant (every
    user-written declaration, private or not), `aux` when it did not (compiler-generated
    constants such as `_proof_N`, `_simp_N_M`, `eq_N`). Names are printed in the lossless
    component encoding described at `ENC`. The closure replicates the kernel-side rules of
    `Lean.CollectAxioms.collect` (axiom: record and visit its type; def/theorem/opaque: type
    and value; quot: nothing; ctor/rec: type; inductive: type and constructors) with one
    visited set, so the printed AXIOMS line is the union over every package constant.
    """
    local = ('  if includeLocal then\n'
             '    for (c, ci) in env.constants.map₂.toList do\n'
             '      unless (`_inv).isPrefixOf c do\n'
             '        consts := consts.push (c, ci, `_local)\n')
    return ('import ' + PACKAGE + '\nimport Lean\nopen Lean\n\n'
            'def _inv.kind : ConstantInfo → String\n'
            '  | .thmInfo _ => "theorem" | .defnInfo _ => "def" | .axiomInfo _ => "axiom" | .opaqueInfo _ => "opaque"\n'
            '  | .inductInfo _ => "inductive" | .ctorInfo _ => "ctor" | .recInfo _ => "rec" | .quotInfo _ => "quot"\n\n'
            'def _inv.hex (s : String) : String :=\n'
            '  s.toUTF8.foldl (fun acc b => acc ++ (if b.toNat < 16 then "0" else "") ++ String.ofList (Nat.toDigits 16 b.toNat)) ""\n\n'
            'def _inv.join (a b : String) : String := if a.isEmpty then b else a ++ "/" ++ b\n\n'
            'def _inv.enc : Name → String\n'
            '  | .anonymous => ""\n'
            '  | .str p s => _inv.join (_inv.enc p) ("s" ++ _inv.hex s)\n'
            '  | .num p k => _inv.join (_inv.enc p) ("n" ++ toString k)\n\n'
            'structure _inv.St where\n'
            '  visited : NameSet := {}\n'
            '  axioms : NameSet := {}\n\n'
            'partial def _inv.collect (env : Environment) (c : Name) : StateM _inv.St Unit := do\n'
            '  unless (← get).visited.contains c do\n'
            '    modify fun s => { s with visited := s.visited.insert c }\n'
            '    let collectExpr (e : Expr) : StateM _inv.St Unit := e.getUsedConstants.forM (_inv.collect env)\n'
            '    match env.find? c with\n'
            '    | some (.axiomInfo v)  => modify (fun s => { s with axioms := s.axioms.insert c }) *> collectExpr v.type\n'
            '    | some (.defnInfo v)   => collectExpr v.type *> collectExpr v.value\n'
            '    | some (.thmInfo v)    => collectExpr v.type *> collectExpr v.value\n'
            '    | some (.opaqueInfo v) => collectExpr v.type *> collectExpr v.value\n'
            '    | some (.quotInfo _)   => pure ()\n'
            '    | some (.ctorInfo v)   => collectExpr v.type\n'
            '    | some (.recInfo v)    => collectExpr v.type\n'
            '    | some (.inductInfo v) => collectExpr v.type *> v.ctors.forM (_inv.collect env)\n'
            '    | none                 => pure ()\n\n'
            'def _inv.main (pkg : Name) (includeLocal : Bool) : CoreM Unit := do\n'
            '  let env ← getEnv\n'
            '  let header := env.header\n'
            '  let names := header.moduleNames\n'
            '  let mut consts : Array (Name × ConstantInfo × Name) := #[]\n'
            '  for h : idx in [0:names.size] do\n'
            '    let m := names[idx]\n'
            '    if pkg.isPrefixOf m then\n'
            '      let data := header.moduleData[idx]!\n'
            '      IO.println s!"MODULE|{_inv.enc m}|{data.isModule}|{data.constNames.size}|{data.extraConstNames.size}"\n'
            '      for c in data.constNames do\n'
            '        match env.find? c with\n'
            '        | some ci => consts := consts.push (c, ci, m)\n'
            '        | none => throwError "constant {c} of module {m} is missing from the environment"\n'
            + local +
            '  let sorted := consts.qsort (fun a b => _inv.enc a.1 < _inv.enc b.1)\n'
            '  for (c, ci, m) in sorted do\n'
            '    let origin := if (← findDeclarationRanges? c).isSome then "user" else "aux"\n'
            '    IO.println s!"INV|{_inv.kind ci}|{origin}|{_inv.enc m}|{_inv.enc c}"\n'
            '  let (_, s) := (sorted.forM (fun x => _inv.collect env x.1)).run {}\n'
            '  let axs := (s.axioms.toArray.map _inv.enc).qsort (· < ·)\n'
            '  IO.println s!"AXIOMS|{",".intercalate axs.toList}"\n'
            '  IO.println s!"CLOSURE|{s.visited.size}"\n\n'
            '#eval _inv.main `' + PACKAGE + ' ' + ('true' if include_local else 'false') + '\n')

class InventoryProtocolError(ValueError):
    """The inventory output is not a well-formed protocol transcript (nothing was audited)."""

class InventoryRejected(ValueError):
    """A well-formed inventory violates the admission policy; `violations` lists every finding."""
    def __init__(self, violations, inventory):
        self.violations, self.inventory = violations, inventory
        super().__init__('environment inventory rejected: ' + '; '.join(code + ' ' + render(name) + (' ' + note if note else '') for code, name, note in violations))

def decode(enc):
    """Decode one ENC name into a tuple of components (str for string, int for numeric)."""
    parts = []
    for item in enc.split('/'):
        body = item[1:]
        if item[0] == 's':
            if len(body) % 2:
                raise InventoryProtocolError('malformed name encoding: ' + enc)
            try:
                parts.append(bytes.fromhex(body).decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                raise InventoryProtocolError('malformed name encoding: ' + enc) from None
            if body != body.lower() or bytes.fromhex(body).decode('utf-8').encode().hex() != body:
                raise InventoryProtocolError('non-canonical name encoding: ' + enc)
        else:
            if not body.isdigit() or str(int(body)) != body:
                raise InventoryProtocolError('malformed numeric name component: ' + enc)
            parts.append(int(body))
    return tuple(parts)

def encode(name):
    return '/'.join(('n' + str(c)) if isinstance(c, int) else ('s' + c.encode('utf-8').hex()) for c in name)

def render(name):
    """Human-readable name for messages and receipts only; identity is the component tuple."""
    return '.'.join(str(c) if isinstance(c, int) else (c if SIMPLE.fullmatch(c) else '«' + c + '»') for c in name)

def simple_name(text):
    """Component tuple of a dotted name made only of simple identifiers (targets, modules, axioms)."""
    parts = tuple(text.split('.'))
    require(all(SIMPLE.fullmatch(c) for c in parts), 'unrepresentable name: ' + repr(text))
    return parts

def user_name(name):
    """(user-facing name, is_private) for a private `_private.<module>.<n>.<name>` constant."""
    if name and name[0] == '_private':
        for index, c in enumerate(name):
            if isinstance(c, int):
                return name[index + 1:], True
    return name, False

ALLOWED_NAMES = frozenset(simple_name(a) for a in ALLOWED)

KINDS = ('theorem', 'def', 'axiom', 'opaque', 'inductive', 'ctor', 'rec', 'quot')

def bound_inventory(records, targets, modules):
    """The manifest's reviewed, exact non-target compiled inventory as a set of tuples.

    Each record is [kind, origin, module, name] with simple dotted names. Every constant of a
    package module that is not a user-written target theorem must appear here, and every
    entry must be present; admission therefore does not depend on the source pre-check or
    on Lean's provenance alone (RF312-SUCCESSOR-COMMAND-004).
    """
    require(isinstance(records, list), 'missing compiled inventory binding')
    registered, wanted = {simple_name(m) for m in modules}, {simple_name(t) for t in targets}
    result = set()
    for record in records:
        require(isinstance(record, list) and len(record) == 4 and all(isinstance(x, str) for x in record), 'malformed compiled inventory record: ' + repr(record))
        kind, origin, module, name = record
        require(kind in KINDS and kind != 'axiom' and origin in ('user', 'aux'), 'invalid compiled inventory kind or origin: ' + repr(record))
        entry = (kind, origin, simple_name(module), simple_name(name))
        require(entry[2] in registered, 'compiled inventory record outside registered modules: ' + repr(record))
        require(not (kind == 'theorem' and origin == 'user'), 'user-written theorems belong in targets, not the compiled inventory: ' + repr(record))
        require(entry[3] not in wanted and entry not in result, 'duplicate or target-shadowing compiled inventory record: ' + repr(record))
        result.add(entry)
    require(len({e[3] for e in result}) == len(result), 'duplicate name in compiled inventory')
    return frozenset(result)

def parse_inventory(text):
    """Strictly parse the protocol lines; any malformed or missing record is a protocol error."""
    lines = [line for line in text.splitlines() if line.startswith(('INV|', 'MODULE|', 'AXIOMS|', 'CLOSURE|'))]
    modules, constants, axioms, closures = [], [], [], []
    for line in lines:
        if line.startswith('MODULE|'):
            match = MODULE.fullmatch(line)
            if not match:
                raise InventoryProtocolError('malformed module line: ' + line)
            modules.append((decode(match.group(1)), match.group(2) == 'true', int(match.group(3)), int(match.group(4))))
        elif line.startswith('INV|'):
            match = INV.fullmatch(line)
            if not match:
                raise InventoryProtocolError('malformed inventory line: ' + line)
            kind, origin, module, name = match.groups()
            constants.append((kind, origin, decode(module), decode(name)))
        elif line.startswith('AXIOMS|'):
            match = AXIOMS.fullmatch(line)
            if not match:
                raise InventoryProtocolError('malformed axiom closure line: ' + line)
            axioms.append([] if not match.group(1) else [decode(a) for a in match.group(1).split(',')])
        else:
            match = CLOSURE.fullmatch(line)
            if not match:
                raise InventoryProtocolError('malformed closure line: ' + line)
            closures.append(int(match.group(1)))
    if not constants or not modules:
        raise InventoryProtocolError('empty environment inventory')
    if len(axioms) != 1 or len(closures) != 1:
        raise InventoryProtocolError('inventory must report exactly one axiom closure')
    return dict(modules=modules, constants=constants, axioms=axioms[0], closure=closures[0], lines=lines)

def inventory_violations(inventory, targets, modules, allow_local=False, reported_axioms=None, bound=frozenset()):
    """Every admission-policy finding on a parsed inventory, as sorted (code, name, note) triples.

    - the module table is exactly the registered modules plus an empty root module, none
      compiled under the module system, none with code-generator extras;
    - every constant sits in a registered module (or the current file in local mode), names
      are unique, per-module counts match the table, and no constant is itself an axiom;
    - the user-written (`user`) theorem-kind constants, private ones included, are exactly
      the manifest targets;
    - every compiler-generated (`aux`) theorem-kind constant hangs below a user-written
      constant of the same module through compiler suffix components only;
    - every other constant of a registered module (definitions, compiler theorems, anything
      added by a metaprogram) is exactly one of the manifest's bound `compiled_inventory`
      records, and every bound record is present;
    - the axiom closure of all constants together is inside ALLOWED, and contains every
      axiom the per-target audit reported.
    """
    require(isinstance(targets, list) and targets and len(set(targets)) == len(targets), 'empty or duplicate target list')
    require(isinstance(modules, (list, set, frozenset)) and modules, 'empty module registry')
    registered = {simple_name(m) for m in modules}
    root, wanted = simple_name(PACKAGE), {simple_name(t) for t in targets}
    found = set()
    def flag(code, name, note=''):
        found.add((code, encode(name), note))
    counts, seen_modules = {}, set()
    for name, is_module, constants, extra in inventory['modules']:
        if name in seen_modules:
            flag('duplicate_module', name)
        seen_modules.add(name)
        if is_module:
            flag('module_system', name)
        if name == root:
            if constants or extra:
                flag('root_not_empty', name)
        elif name not in registered:
            flag('unregistered_module', name)
        else:
            if extra:
                flag('codegen_extra', name, str(extra))
            counts[name] = constants
    for name in sorted(registered - seen_modules, key=encode):
        flag('missing_module', name)
    if root not in seen_modules:
        flag('missing_module', root)
    seen, per_module, user = set(), {name: 0 for name in registered}, {}
    for kind, origin, module, name in inventory['constants']:
        if name in seen:
            flag('duplicate_name', name)
        seen.add(name)
        if not (module in registered or (allow_local and module == LOCAL_MODULE)):
            flag('outside_module', name, render(module))
        if kind == 'axiom':
            flag('package_axiom', name)
        if module in per_module:
            per_module[module] += 1
        if origin == 'user':
            user.setdefault(module, set()).add(name)
    for name in sorted(registered, key=encode):
        if name in counts and per_module[name] != counts[name]:
            flag('count_mismatch', name, str(per_module[name]) + '/' + str(counts[name]))
    public = set()
    for kind, origin, module, name in inventory['constants']:
        if kind != 'theorem':
            continue
        if origin == 'user':
            public.add(name)
            if name not in wanted:
                flag('extra_theorem', name)
            continue
        parents = [name[:k] for k in range(len(name) - 1, 0, -1) if name[:k] in user.get(module, set())]
        if not parents or not all(isinstance(c, str) and AUX_SUFFIX.fullmatch(c) for c in name[len(parents[0]):]):
            flag('unbound_auxiliary', name)
    for name in sorted(wanted - public, key=encode):
        flag('missing_theorem', name)
    observed = set()
    for kind, origin, module, name in inventory['constants']:
        if module not in registered or (kind == 'theorem' and origin == 'user'):
            continue
        entry = (kind, origin, module, name)
        observed.add(entry)
        if entry not in bound:
            flag('unregistered_constant', name, kind + '/' + origin)
    for kind, origin, module, name in sorted(set(bound) - observed, key=lambda e: encode(e[3])):
        flag('missing_constant', name, kind + '/' + origin)
    closure = set(inventory['axioms'])
    if len(closure) != len(inventory['axioms']):
        flag('duplicate_axiom', ('AXIOMS',))
    for name in sorted(closure - ALLOWED_NAMES, key=encode):
        flag('forbidden_axiom', name)
    if reported_axioms is not None:
        reported = {simple_name(a) for values in reported_axioms.values() for a in values}
        for name in sorted(reported - closure, key=encode):
            flag('reported_axiom_omitted', name)
    if inventory['closure'] < len(inventory['constants']):
        flag('impossible_closure', ('CLOSURE',), str(inventory['closure']))
    return [(code, decode(name), note) for code, name, note in sorted(found)]

def audit_inventory(text, targets, modules, allow_local=False, reported_axioms=None, bound=frozenset()):
    """Package-wide admission check on the compiled environment, not on source text.

    Raises InventoryProtocolError on a malformed transcript and InventoryRejected (with the
    complete violation list) on any policy finding; see `inventory_violations`.
    """
    inventory = parse_inventory(text)
    violations = inventory_violations(inventory, targets, modules, allow_local, reported_axioms, bound)
    if violations:
        raise InventoryRejected(violations, inventory)
    public = sorted(render(name) for kind, origin, module, name in inventory['constants'] if kind == 'theorem' and origin == 'user')
    auxiliary = sum(1 for kind, origin, module, name in inventory['constants'] if kind == 'theorem' and origin == 'aux')
    return dict(declarations=len(inventory['constants']), public_theorems=public, auxiliary_theorems=auxiliary,
                axioms=sorted(render(a) for a in inventory['axioms']), closure=inventory['closure'],
                sha256=sha('\n'.join(inventory['lines']).encode()))

def lineage_identity(party, role):
    """Validate declared identity text, not the truth or completeness of authorship."""
    require(isinstance(party, dict), 'malformed review lineage: ' + role)
    placeholders = {'unknown', 'unverified', 'unspecified', 'tbd', 'n/a',
                    'none', 'null', '?', 'not known', 'not available'}
    result = {}
    for key in ('provider', 'family', 'agent'):
        value = party.get(key)
        require(isinstance(value, str) and value.strip(), 'missing review lineage: ' + role + '.' + key)
        normalized = ' '.join(value.split()).casefold()
        require(normalized not in placeholders, 'ambiguous review lineage: ' + role + '.' + key)
        result[key] = normalized
    return result

def check_alignment(review, manifest_digest, targets, scope_digest):
    require(isinstance(review, dict), 'review must be object')
    require(review.get('disposition') == 'ACCEPTED', 'alignment not accepted')
    require(review.get('manifest_sha256') == manifest_digest, 'stale alignment manifest')
    require(review.get('scope_sha256') == scope_digest, 'stale alignment scope')
    covered = review.get('targets', [])
    require(isinstance(covered, list) and len(covered) == len(set(covered)) and set(covered) == set(targets), 'partial or ambiguous review')
    r = lineage_identity(review.get('reviewer'), 'reviewer')
    authors = [('author', lineage_identity(review.get('author'), 'author'))]
    # Absence preserves the historical single-author schema. Declared proposers
    # are additional authors, never ignored metadata or a substitute for author.
    proposers = review.get('proposal_authors', [])
    require(isinstance(proposers, list), 'malformed review lineage: proposal_authors')
    for index, proposer in enumerate(proposers):
        role = 'proposal_authors[' + str(index) + ']'
        identity = lineage_identity(proposer, role)
        if 'targets' in proposer:
            scope = proposer['targets']
            require(isinstance(scope, list) and scope and
                    all(isinstance(name, str) and name in targets for name in scope) and
                    len(scope) == len(set(scope)), 'invalid proposal target scope: ' + role)
        authors.append((role, identity))
    for role, author in authors:
        for key in ('provider', 'family', 'agent'):
            require(author[key] != r[key], 'lineage not independent: ' + role + '.' + key)
    e = review.get('evidence', {})
    require(isinstance(e.get('repository'), str) and re.fullmatch(r'[^/\s]+/[^/\s]+', e['repository']), 'missing evidence repo')
    require(isinstance(e.get('commit'), str) and COMMIT.fullmatch(e['commit']), 'mutable review ref')
    require(isinstance(e.get('sha256'), str) and SHA256.fullmatch(e['sha256']), 'missing review hash')
    require(isinstance(e.get('path'), str) and e['path'] and not PurePosixPath(e['path']).is_absolute() and '..' not in PurePosixPath(e['path']).parts, 'missing or unsafe review path')
    # This checks structure, identity and scope. A controller must retrieve and
    # authenticate the review evidence; strings do not prove a review happened.

def load_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            require(key not in obj, 'duplicate JSON key: ' + key)
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)

def source_check():
    raw = (ROOT / 'manifest.json').read_bytes()
    m = load_json(raw)
    require(m.get('schema_version') == 1 and m.get('scientific_effect') == 'NONE', 'invalid evidence schema')
    require(m.get('formalization_status') == 'proved' and m.get('alignment_status') == 'PENDING_INDEPENDENT_REVIEW', 'source metadata cannot self-award execution or review')
    required = {'gate.py', 'tests/test_gate.py', 'lean-toolchain', 'lakefile.toml', 'lake-manifest.json', 'ResearchFormalCoreR1.lean', 'SCOPE.md', 'GLOSSARY.md', 'README.md', 'blueprint/src/content.tex', 'tests/test_alignment_lineage.py', 'LINEAGE_VALIDATION.md', 'tests/test_inventory_gate.py', 'GATE_HARDENING.md'}
    require(required <= set(m['files']), 'unbound control or scope file')
    originals = {'originals/Algebra.lean.txt': '4c196820c4db8fafc288dd35642828ba577e3544d60aba7d1d6d24f14ad1e8ae', 'originals/ProbabilityCompanions.lean.txt': '4ace6600a476859982c8851ae9097c89b082d3c96291b3c8330bd4cc00bae65d'}
    require(all(m['files'].get(path) == value for path, value in originals.items()), 'original source identity changed or omitted')
    require('COMPATIBILITY.md' in m['files'], 'unbound successor rationale')
    check_files(ROOT, m['files'])
    require((ROOT / 'lean-toolchain').read_text().strip() == 'leanprover/lean4:v4.34.1', 'wrong Lean toolchain')
    lock = load_json((ROOT / 'lake-manifest.json').read_text())
    packages = lock['packages']
    require(len(packages) == len({p['name'] for p in packages}), 'duplicate dependency')
    actual = {p['name']: p['rev'] for p in packages}
    require(actual == m['dependency_revisions'], 'dependency lock mismatch')
    require(all(COMMIT.fullmatch(v) for v in actual.values()), 'unpinned dependency')
    expected_root = ''.join('import ' + path[:-5].replace('/', '.') + '\n' for path in m['source_modules'])
    require((ROOT / 'ResearchFormalCoreR1.lean').read_text() == expected_root, 'root must only import registered modules')
    names, definitions = [], []
    for path in m['source_modules']:
        require(path in m['files'], 'unbound source module')
        require(path.startswith(PACKAGE + '/') and path.endswith('.lean') and path.count('/') == 1, 'source module outside the package directory: ' + path)
        text = (ROOT / path).read_text()
        # Pre-check only: the scanner refuses any theorem spelling it cannot count, and the
        # compiled environment is audited package-wide in execute() (RF-GATE-01).
        names.extend(PACKAGE + '.' + n for n in grammar_check(text, path, definitions))
    require(isinstance(m.get('targets'), list) and len(m['targets']) == len(set(m['targets'])), 'empty or duplicate target list')
    require(len(names) == len(set(names)), 'duplicate source theorem')
    require(names == m['targets'], 'target inventory differs from source declarations')
    modules = {PACKAGE + '.' + path[len(PACKAGE) + 1:-5] for path in m['source_modules']}
    bound = bound_inventory(m.get('compiled_inventory'), m['targets'], modules)
    bound_definitions = {render(e[3]) for e in bound if e[:2] == ('def', 'user')}
    require(bound_definitions == {PACKAGE + '.' + d for d in definitions} and len(definitions) == len(set(definitions)),
            'compiled inventory user definitions differ from source definitions')
    check_blueprint((ROOT / 'blueprint/src/content.tex').read_text(), names)
    lean_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.lean') if '.lake' not in p.relative_to(ROOT).parts}
    require(lean_files == set(m['source_modules']) | {'ResearchFormalCoreR1.lean'}, 'unregistered Lean module')
    return m, sha(raw)

def run(command, label, out, expect_success=True, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (out / (label + '.log')).write_text(result.stdout)
    print(label + ': exit ' + str(result.returncode), flush=True)
    require((result.returncode == 0) == expect_success, 'unexpected process outcome: ' + label + '\n' + result.stdout[-6000:])
    return result.stdout

def check_admission(text, label, injected, expected, targets, modules, allow_local, bound):
    """One admission experiment: the transcript must parse, contain the injected constant exactly
    once, and be rejected with exactly the expected findings.

    `injected` is (kind, origin, user-facing name, is_private, module or None); `expected` is a
    set of (code, user-facing name, is_private). The user-facing name only locates the injected
    constant. Findings are then compared by FULL decoded name and multiplicity: an expected
    entry naming the injection stands for that exact constant, every other expected entry must
    be a public name, and any further finding (an unrelated private constant with the same short
    name in another module, say) fails the experiment (RF312-CONTROL-PRIVATE-IDENTITY-006). A
    protocol error, a missing injection or a pass also fails instead of counting as a rejection.
    """
    inventory = parse_inventory(text)
    kind, origin, name, private, module = injected
    hits = [c for c in inventory['constants'] if c[0] == kind and c[1] == origin and user_name(c[3]) == (name, private) and (module is None or c[2] == module)]
    require(len(hits) == 1, 'admission control did not inject exactly one ' + kind + ' ' + render(name) + ': ' + label)
    full = hits[0][3]
    wanted = []
    for code, short, is_private in expected:
        if (short, is_private) == (name, private):
            wanted.append((code, encode(full)))
        else:
            require(not is_private, 'expected private finding other than the injection is not identifiable: ' + label)
            wanted.append((code, encode(short)))
    found = inventory_violations(inventory, targets, modules, allow_local, bound=bound)
    got = [(code, encode(n)) for code, n, note in found]
    require(sorted(got) == sorted(wanted), 'admission control rejected for an unexpected reason: ' + label + ' expected=' +
            repr(sorted(code + ' ' + render(decode(e)) for code, e in wanted)) + ' got=' + repr(sorted(code + ' ' + render(decode(e)) for code, e in got)))
    return dict(outcome='REJECTED_FOR_EXPECTED_REASON', injected=render(full),
                violations=[code + ' ' + render(n) for code, n, note in found])

COMPILED_INJECTIONS = (
    # A private theorem proved by sorry: kernel-valid, so build, recheck and the per-target
    # axiom audit all pass; only the package-wide inventory can reject it.
    ('compiled_admission', 'private theorem compiledAdmission : (1 : Nat) = 1 := by sorry\n',
     ('theorem', 'user', (PACKAGE, 'compiledAdmission'), True),
     {('forbidden_axiom', ('sorryAx',), False), ('extra_theorem', (PACKAGE, 'compiledAdmission'), True)}),
    # RF312-SUCCESSOR-COMMAND-004: a wrapped elaborator adds a range-less theorem under a real
    # user parent with a compiler-shaped suffix. It must fail the bound compiled inventory.
    ('compiled_metaprogram', 'set_option maxRecDepth 1000 in run_elab\n  Lean.addDecl <| .thmDecl {\n'
     '    name := `ResearchFormalCoreR1.ec005_fold_gap._proof_999\n    levelParams := []\n'
     '    type := Lean.mkConst ``True\n    value := Lean.mkConst ``True.intro\n  }\n',
     ('theorem', 'aux', (PACKAGE, 'ec005_fold_gap', '_proof_999'), False),
     {('unregistered_constant', (PACKAGE, 'ec005_fold_gap', '_proof_999'), False)}))

def compiled_admission(m, modules, baseline, bound, out):
    """Inject declarations into a compiled copy of the package and require each to be rejected.

    The package is copied (sources plus the fresh `.lake/build`, dependencies by symlink) and
    first audited unchanged: it must pass with the same inventory digest as the real package
    (positive baseline). Then, for each entry of COMPILED_INJECTIONS in turn, the first source
    module gains the declaration and the copy runs the same stages as the real package:
    `lake build`, `leanchecker`, the per-target `#print axioms` audit (each must succeed; an
    unrelated failure fails the run, it is never counted as a rejection), and finally the
    inventory, which must contain the injected constant and reject with exactly the expected
    findings (RF312-COMPILED-RECHECK-005). The copy lives under `.lake` outside the uploaded
    evidence directory and is removed afterwards; only its logs remain.
    """
    work = ROOT / '.lake' / 'compiled-admission'  # outside the uploaded evidence directory
    require(not work.is_symlink(), 'symlink compiled-admission directory')
    if work.exists():
        shutil.rmtree(work)
    path = m['source_modules'][0]
    module = simple_name(PACKAGE + '.' + path[len(PACKAGE) + 1:-5])
    results = {}
    try:
        pkg = work / 'formal'
        shutil.copytree(ROOT, pkg, ignore=lambda d, names: ['.lake'] if Path(d) == ROOT else [])
        shutil.copytree(ROOT / '.lake' / 'build', pkg / '.lake' / 'build')
        (pkg / '.lake' / 'packages').symlink_to(ROOT / '.lake' / 'packages', target_is_directory=True)
        program = work / 'Inventory.lean'
        program.write_text(inventory_source())
        audit = work / 'Audit.lean'
        audit.write_text('import ' + PACKAGE + '\n' + '\n'.join('#print axioms ' + n for n in m['targets']) + '\n')
        clean = audit_inventory(run(['lake', 'env', 'lean', str(program)], 'compiled_baseline', out, cwd=pkg), m['targets'], modules, bound=bound)
        require(clean['sha256'] == baseline['sha256'], 'compiled-admission copy does not reproduce the package inventory')
        original = (pkg / path).read_text()
        marker = '\nend ' + PACKAGE + '\n'
        require(original.count(marker) == 1, 'compiled-admission module has no unique namespace end: ' + path)
        for label, declaration, (kind, origin, name, private), expected in COMPILED_INJECTIONS:
            injected = original.replace(marker, '\n' + declaration + marker)
            (pkg / path).write_text(injected)
            run(['lake', 'build'], label + '_build', out, cwd=pkg)
            run(['lake', 'env', 'leanchecker', PACKAGE], label + '_leanchecker', out, cwd=pkg)
            audit_axioms(run(['lake', 'env', 'lean', str(audit)], label + '_axioms', out, cwd=pkg), m['targets'])
            log = run(['lake', 'env', 'lean', str(program)], label, out, cwd=pkg)
            result = check_admission(log, label, (kind, origin, name, private, module), expected, m['targets'], modules, False, bound)
            result.update(module=render(module), module_sha256=sha(injected.encode()),
                          stages=['build', 'leanchecker', 'axioms', 'inventory'])
            results[label] = result
            (pkg / path).write_text(original)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return dict(baseline_sha256=clean['sha256'], injections=results)

def execute(m, digest):
    out = ROOT / '.lake' / 'formal-evidence'
    out.mkdir(parents=True, exist_ok=True)
    require(not (ROOT / '.lake').is_symlink() and not (ROOT / '.lake/build').is_symlink(), 'symlink build directory')
    if (ROOT / '.lake/build').exists():
        shutil.rmtree(ROOT / '.lake/build')  # fresh local-package build; dependency cache is untouched
    run(['lake', 'build'], 'build', out)
    run(['lake', 'env', 'leanchecker', 'ResearchFormalCoreR1'], 'leanchecker', out)
    audit = out / 'Audit.lean'
    audit.write_text('import ResearchFormalCoreR1\n' + '\n'.join('#print axioms ' + n for n in m['targets']) + '\n')
    text = run(['lake', 'env', 'lean', str(audit)], 'axioms', out)
    axioms = audit_axioms(text, m['targets'])
    types = out / 'Types.lean'
    types.write_text('import ResearchFormalCoreR1\nset_option pp.explicit true\n' + '\n'.join('#check ' + n for n in m['targets']) + '\n')
    run(['lake', 'env', 'lean', str(types)], 'elaborated-types', out)
    # Package-wide admission check bound to the compiled environment, not to source text:
    # every constant in a registered module is enumerated by Lean itself and axiom-audited,
    # and the user-written theorems must be exactly the manifest targets (RF-GATE-01).
    modules = {PACKAGE + '.' + path[len(PACKAGE) + 1:-5] for path in m['source_modules']}
    inventory = out / 'Inventory.lean'
    inventory.write_text(inventory_source())
    bound = bound_inventory(m['compiled_inventory'], m['targets'], modules)
    inventory_report = audit_inventory(run(['lake', 'env', 'lean', str(inventory)], 'inventory', out), m['targets'], modules, reported_axioms=axioms, bound=bound)
    # Admission controls: declarations the source scanner could never see, or would
    # mis-prefix, must be caught by the environment inventory for their own reason. Each
    # compiles (sorry only warns), must appear in the inventory, and must be rejected with
    # exactly the expected findings; any other outcome fails the run.
    P = (PACKAGE,)
    admissions = {
        'hidden_sorry': ('namespace ' + PACKAGE + '\n  private theorem hiddenAdmission : (1 : Nat) = 1 := by sorry\nend ' + PACKAGE + '\n',
                         ('theorem', 'user', P + ('hiddenAdmission',), True, LOCAL_MODULE),
                         {('forbidden_axiom', ('sorryAx',), False), ('extra_theorem', P + ('hiddenAdmission',), True)}),
        'hidden_theorem': ('namespace ' + PACKAGE + '\n  @[simp] theorem indentedAdmission : (1 : Nat) = 1 := rfl\nend ' + PACKAGE + '\n',
                           ('theorem', 'user', P + ('indentedAdmission',), False, LOCAL_MODULE),
                           {('extra_theorem', P + ('indentedAdmission',), False)}),
        'hidden_def': ('namespace ' + PACKAGE + '\n  noncomputable def unusedAdmission : Nat := by sorry\nend ' + PACKAGE + '\n',
                       ('def', 'user', P + ('unusedAdmission',), False, LOCAL_MODULE),
                       {('forbidden_axiom', ('sorryAx',), False)}),
        'outside_namespace': ('theorem outsideNamespaceAdmission : (1 : Nat) = 1 := rfl\n',
                              ('theorem', 'user', ('outsideNamespaceAdmission',), False, LOCAL_MODULE),
                              {('extra_theorem', ('outsideNamespaceAdmission',), False)})}
    admission_outcomes = {}
    for label, (declaration, injected, expected) in admissions.items():
        path = out / (label + '.lean')
        path.write_text(inventory_source(include_local=True).replace('import Lean\n', 'import Lean\n' + declaration, 1))
        log = run(['lake', 'env', 'lean', str(path)], label, out)
        admission_outcomes[label] = check_admission(log, label, injected, expected, m['targets'], modules, True, bound)
    compiled = compiled_admission(m, modules, inventory_report, bound, out)
    # Real counterexample/mutation controls, not comparisons of fixed labels.
    cases = {
        'false_fold': ('import ResearchFormalCoreR1\nexample : ResearchFormalCoreR1.foldPotential 1 1 - ResearchFormalCoreR1.foldPotential 1 (-1) = (2 : Real)^3 / 7 := by\n  norm_num [ResearchFormalCoreR1.foldPotential]\n', False, None),
        'false_power': ('import ResearchFormalCoreR1\nexample : (2 : Real)^4 <= (2 : Real)^3 := by\n  norm_num\n', False, None),
        'sorry': ('import ResearchFormalCoreR1\ntheorem injected : False := by sorry\n#print axioms injected\n', True, 'injected'),
        'custom_imported': ('import ResearchFormalCoreR1\naxiom hiddenPremise : False\ntheorem middle : False := hiddenPremise\ntheorem injected : False := middle\n#print axioms injected\n', True, 'injected'),
        'native': ('import ResearchFormalCoreR1\ntheorem injected : (2 : Nat) + 2 = 4 := by native_decide\n#print axioms injected\n', True, 'injected')}
    outcomes = {}
    for label, (source, succeeds, target) in cases.items():
        path = out / (label + '.lean'); path.write_text(source)
        log = run(['lake', 'env', 'lean', str(path)], label, out, succeeds)
        if target:
            # Remove compiler diagnostics only by selecting the axiom line onward;
            # absence remains an error, never an empty allowlist.
            matches = list(AXIOM.finditer(log))
            require(matches, 'negative control missing axiom report: ' + label)
            payload = '\n'.join(x.group() for x in matches)
            try:
                audit_axioms(payload, [target])
            except ValueError:
                outcomes[label] = 'REJECTED_BY_AXIOM_GATE'
            else:
                raise ValueError('negative control escaped: ' + label)
        else:
            require('unsolved goals' in log or 'norm_num' in log and 'failed' in log, 'negative control did not fail on its proof goal')
            outcomes[label] = 'REJECTED_BY_LEAN'
    # Recheck after compilation: a build hook must not rewrite bound sources/lock.
    _, after = source_check(); require(after == digest, 'manifest changed during execution')
    version = run(['lake', 'env', 'lean', '--version'], 'version', out).strip()
    require('version 4.34.1' in version, 'unexpected running Lean version')
    actual_dependencies = {name: subprocess.check_output(['git', '-C', str(ROOT / '.lake/packages' / name), 'rev-parse', 'HEAD'], text=True).strip() for name in m['dependency_revisions']}
    require(actual_dependencies == m['dependency_revisions'], 'installed dependency HEAD differs from lock')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(COMMIT.fullmatch(head), 'missing exact checked commit')
    receipt = dict(schema_version=1, scientific_effect='NONE', formalization_status='kernel-checked',
                   alignment_status='PENDING_INDEPENDENT_REVIEW', manifest_sha256=digest,
                   checked_commit=head, repository=os.environ.get('GITHUB_REPOSITORY'),
                   workflow_run_id=os.environ.get('GITHUB_RUN_ID'), lean_version=version,
                   dependency_revisions=m['dependency_revisions'], axioms=axioms, negative_controls=outcomes,
                   inventory=dict(declarations=inventory_report['declarations'], public_theorems=len(inventory_report['public_theorems']),
                                  auxiliary_theorems=inventory_report['auxiliary_theorems'],
                                  axioms=inventory_report['axioms'], closure=inventory_report['closure'],
                                  sha256=inventory_report['sha256'], admission_controls=admission_outcomes,
                                  compiled_admission=compiled))
    receipt['logs'] = {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log'))}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='run build, transitive axiom audit and negative controls')
    p.add_argument('--alignment', type=Path, help='validate a separately authenticated alignment record; never promotes science')
    args = p.parse_args()
    try:
        m, digest = source_check()
        if args.alignment:
            check_alignment(load_json(args.alignment.read_text()), digest, m['targets'], m['files']['SCOPE.md'])
        if args.execute:
            execute(m, digest)
        else:
            print('SOURCE_IDENTITY_PASS (not a Lean build or scientific acceptance): ' + digest)
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as e:
        print('FORMAL_GATE_FAIL: ' + str(e), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
