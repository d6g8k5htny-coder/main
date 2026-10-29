"""Main all-constant inventory, adapted from Math- formal/gate_helper.py.

Origin: d6g8k5htny-coder/Math-@8151f592918a845bf19eb8eafaf8444540a32734.
Only the compiled enumeration, name protocol and admission policy are reused.
No runtime cross-repository import. Main accepts no non-protocol output.
"""
import hashlib
import re

ALLOWED = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
NAME = r'[A-Za-z_][A-Za-z0-9_.]*'
AXIOM = re.compile(r"'(" + NAME + r")' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
SHA256 = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
PACKAGE = 'UniversalLaw'
# Environment-bound inventory lines emitted by the generated Lean audit (RF-GATE-01).
# Every Lean name is printed in a lossless component encoding, never with Name.toString:
# components are joined by `/`; `s<hex>` is a string component (lowercase hex of its UTF-8
# bytes) and `n<decimal>` a numeric component. Private names (`_private.<module>.0.<name>`),
# quoted names (`«.lake»`) and names that differ only by escaping therefore stay distinct.
ENC = r'[sn][0-9a-f]*(?:/[sn][0-9a-f]*)*'
INV = re.compile(r'^INV\|(theorem|def|axiom|opaque|inductive|ctor|rec|quot)\|(user|aux)\|(' + ENC + r')\|(' + ENC + r')$')
MODULE = re.compile(r'^MODULE\|(' + ENC + r')\|(true|false)\|([0-9]+)\|([0-9]+)$')
AXIOMS = re.compile(r'^AXIOMS\|((?:' + ENC + r')(?:,' + ENC + r')*)?$')
EXTRA = re.compile(r'^EXTRA\|(' + ENC + r')\|(' + ENC + r')$')
CLOSURE = re.compile(r'^CLOSURE\|([0-9]+)$')
SIMPLE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
# Suffix components a compiler-generated theorem may carry below its user-written parent.
# Used only together with Lean's own provenance (no declaration range) and a user-written
# parent in the same module; never on its own to classify a constant.
AUX_SUFFIX = re.compile(r'_proof_[0-9]+|_simp_[0-9]+_[0-9]+|eq_[0-9]+|eq_def|match_[0-9]+')
LOCAL_MODULE = ('_local',)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def inventory_source(include_local=False, package=PACKAGE):
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
    return ('import ' + package + '\nimport Lean\nopen Lean\n\n'
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
            '  axioms : NameSet := {}\n'
            '  missing : NameSet := {}\n\n'
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
            '    | none                 => modify fun s => { s with missing := s.missing.insert c }\n\n'
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
            '      for c in data.extraConstNames do\n'
            '        IO.println s!"EXTRA|{_inv.enc m}|{_inv.enc c}"\n'
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
            '  unless s.missing.isEmpty do\n'
            '    throwError "axiom closure references missing constants"\n'
            '  let axs := (s.axioms.toArray.map _inv.enc).qsort (· < ·)\n'
            '  IO.println s!"AXIOMS|{",".intercalate axs.toList}"\n'
            '  IO.println s!"CLOSURE|{s.visited.size}"\n\n'
            '#eval _inv.main `' + package + ' ' + ('true' if include_local else 'false') + '\n')

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
        if not item or item[0] not in ('s', 'n'):
            raise InventoryProtocolError('malformed name encoding: ' + enc)
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
    lines = text.splitlines()
    modules, constants, axioms, closures, extras = [], [], [], [], []
    for line in lines:
        if line.startswith('EXTRA|'):
            match = EXTRA.fullmatch(line)
            if not match:
                raise InventoryProtocolError('malformed codegen name: ' + line)
            extras.append((decode(match.group(1)), decode(match.group(2))))
        elif line.startswith('MODULE|'):
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
    return dict(extras=extras, modules=modules, constants=constants, axioms=axioms[0], closure=closures[0], lines=lines)

def inventory_violations(inventory, targets, modules, allow_local=False, reported_axioms=None, bound=frozenset(), package=PACKAGE, codegen=frozenset()):
    """Every admission-policy finding on a parsed inventory, as sorted (code, name, note) triples.

    - the module table is exactly the registered modules plus an empty root module, none
      compiled under the module system; IR-only extra names are separately counted/bound;
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
    root, wanted = simple_name(package), {simple_name(t) for t in targets}
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
            observed_extras = [e for e in inventory['extras'] if e[0] == name]
            if extra != len(observed_extras):
                flag('codegen_extra', name, str(extra))
            counts[name] = constants
    for name in sorted(registered - seen_modules, key=encode):
        flag('missing_module', name)
    if root not in seen_modules:
        flag('missing_module', root)
    if len(set(inventory['extras'])) != len(inventory['extras']):
        flag('duplicate_codegen', root)
    for module, name in set(inventory['extras']) ^ set(codegen):
        flag('codegen_binding', name, render(module))
    for module, name in inventory['extras']:
        if module not in registered:
            flag('codegen_outside_module', name, render(module))
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

def audit_inventory(text, targets, modules, allow_local=False, reported_axioms=None, bound=frozenset(), package=PACKAGE, codegen=frozenset()):
    """Package-wide admission check on the compiled environment, not on source text.

    Raises InventoryProtocolError on a malformed transcript and InventoryRejected (with the
    complete violation list) on any policy finding; see `inventory_violations`.
    """
    inventory = parse_inventory(text)
    violations = inventory_violations(inventory, targets, modules, allow_local, reported_axioms, bound, package, codegen)
    if violations:
        raise InventoryRejected(violations, inventory)
    public = sorted(render(name) for kind, origin, module, name in inventory['constants'] if kind == 'theorem' and origin == 'user')
    auxiliary = sum(1 for kind, origin, module, name in inventory['constants'] if kind == 'theorem' and origin == 'aux')
    return dict(codegen_entries=len(inventory["extras"]), declarations=len(inventory['constants']), public_theorems=public, auxiliary_theorems=auxiliary,
                axioms=sorted(render(a) for a in inventory['axioms']), closure=inventory['closure'],
                sha256=sha('\n'.join(inventory['lines']).encode()))


def bound_codegen(records, modules):
    require(isinstance(records, list), 'missing codegen inventory binding')
    registered = {simple_name(m) for m in modules}
    result = set()
    for row in records:
        require(isinstance(row, list) and len(row) == 2 and all(isinstance(x, str) for x in row), 'invalid codegen record')
        item = tuple(simple_name(x) for x in row)
        require(item[0] in registered and item not in result, 'duplicate or unregistered codegen record')
        result.add(item)
    require(len({n for m,n in result}) == len(result), 'duplicate codegen name')
    return frozenset(result)
