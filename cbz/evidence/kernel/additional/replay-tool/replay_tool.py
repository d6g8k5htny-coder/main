#!/usr/bin/env python3
"""Bounded Lean proof-closure program generator and fail-closed transcript reader.

This tool does not run Lean or authenticate an execution receipt. Its validator
checks consistency of supplied streams/exit status with a separately established
complete compiled census. Mathematical evidence requires the actual Lean run,
source/object identities and the pinned toolchain; see README.md.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ALLOWED_AXIOMS = frozenset(('Classical.choice', 'Quot.sound', 'propext'))
SIMPLE_MODULE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z')
NAME = re.compile(r'[^\s\[\],]+\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json_unique(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON object key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes().decode('utf-8'), object_pairs_hook=object_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON constant: ' + value)))


def checked_names(names, label):
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or not SIMPLE_MODULE.fullmatch(n) for n in names) or len(names) != len(set(names)):
        raise ValueError(label + ' must be a nonempty unique list of simple Lean module names')
    return list(names)


def extract_collector(path, expected_sha256):
    data = Path(path).read_bytes()
    if not isinstance(expected_sha256, str) or not SHA256.fullmatch(expected_sha256) or digest(data) != expected_sha256:
        raise ValueError('collector hash mismatch')
    syntax = ast.parse(data.decode('utf-8'))
    functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == 'inventory_source']
    if len(functions) != 1:
        raise ValueError('expected one inventory_source function')
    start, stop = 'structure _inv.St where', 'def _inv.main ('
    literals = [node.value for node in ast.walk(functions[0]) if isinstance(node, ast.Constant) and isinstance(node.value, str) and start in node.value and stop in node.value]
    if len(literals) != 1 or literals[0].count(start) != 1 or literals[0].count(stop) != 1:
        raise ValueError('expected one unambiguous verbatim collector literal')
    text = literals[0]
    if text.index(start) >= text.index(stop):
        raise ValueError('collector markers are reversed')
    collector = text[text.index(start):text.index(stop)]
    return collector, digest(data)


def generate_program(collector_path, expected_collector_sha256, modules, imports):
    modules = checked_names(modules, 'modules')
    imports = checked_names(imports, 'imports')
    collector, helper_sha = extract_collector(collector_path, expected_collector_sha256)
    program = '\n'.join('import ' + m for m in imports) + '\nimport Lean\nimport Lean.Replay\nopen Lean\n\n' + collector + '''
def _inv.checkProofClosure (modules : Array Name) : CoreM Unit := do
  if modules.isEmpty then throwError "candidate module list is empty"
  if modules.toList.eraseDups.length != modules.size then
    throwError "candidate module list contains duplicates"
  let env := (← getEnv).setExporting false
  let mut seeds : Array Name := #[]
  for modName in modules do
    let some idx := env.getModuleIdx? modName
      | throwError "missing candidate module: {modName}"
    let names := (env.constants.toList.filterMap fun (name, _) =>
      if env.getModuleIdxFor? name == some idx then some name else none).toArray.qsort Name.lt
    if names.isEmpty then throwError "empty candidate module: {modName}"
    IO.println s!"KERNEL_CLOSURE_MODULE {modName} COUNT {names.size}"
    for name in names do
      if (env.checked.get.find? name).isNone then
        throwError "candidate has no kernel declaration: {name}"
      IO.println s!"KERNEL_CLOSURE_SEED {modName} {name}"
      seeds := seeds.push name
  let (_, state) := (seeds.forM (_inv.collect env)).run {}
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  for ax in state.axioms.toArray do
    unless allowed.contains ax do
      throwError "forbidden proof-closure axiom: {ax}"
  let mut selected : Std.HashMap Name ConstantInfo := {}
  for name in state.visited.toArray do
    let some ci := env.find? name
      | throwError "missing transitive proof dependency: {name}"
    if ci.isUnsafe || ci.isPartial then
      throwError "proof closure contains unsafe/partial constant: {name}"
    selected := selected.insert name ci
  IO.println s!"KERNEL_CLOSURE_REPLAY_BEGIN SEEDS {seeds.size} CLOSURE {selected.size}"
  let fresh ← mkEmptyEnvironment 0
  let checked ← fresh.toKernelEnv.replay selected
  for name in seeds do
    if (checked.find? name).isNone then
      throwError "kernel replay omitted candidate constant: {name}"
  let axioms := (state.axioms.toArray.map toString).qsort (· < ·)
  IO.println s!"KERNEL_CLOSURE_STANDARD_ONLY_PASS SEEDS {seeds.size} CLOSURE {selected.size} AXIOMS [{", ".intercalate axioms.toList}]"

#eval _inv.checkProofClosure MODULES_LITERAL
'''.replace('MODULES_LITERAL', '#[' + ', '.join('`' + m for m in modules) + ']')
    return program, {'schema': 'cbz-proof-closure-program-v1', 'program_sha256': digest(program.encode()),
                     'collector_file_sha256': helper_sha, 'collector_verbatim_sha256': digest(collector.encode()),
                     'seed_modules': modules, 'imports': imports, 'required_lean_version': '4.34.1',
                     'kernel_trust_level': 0, 'allowed_axioms': sorted(ALLOWED_AXIOMS),
                     'scope': 'All compiled constants in selected modules and complete transitive type/value dependencies; unrelated imported declarations are outside scope.'}


def expected_records(census):
    if not isinstance(census, dict):
        raise ValueError('expected census is not an object')
    declarations, counts = census.get('module_declarations'), census.get('module_counts')
    if not isinstance(declarations, dict) or not declarations or not isinstance(counts, dict) or set(counts) != set(declarations):
        raise ValueError('expected nonempty module declarations and matching counts')
    checked_names(list(declarations), 'expected modules')
    all_names = []
    for module, names in declarations.items():
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not NAME.fullmatch(n) for n in names) or len(names) != len(set(names)):
            raise ValueError('expected nonempty unique declaration names')
        if type(counts[module]) is not int or counts[module] != len(names):
            raise ValueError('expected module count does not match declarations')
        all_names.extend(names)
    if len(all_names) != len(set(all_names)):
        raise ValueError('expected declaration appears in multiple modules')
    if 'allowed_axioms' in census:
        raw_allowed = census['allowed_axioms']
        if not isinstance(raw_allowed, list) or any(not isinstance(ax, str) for ax in raw_allowed):
            raise ValueError('malformed axiom allowlist')
        if len(raw_allowed) != len(set(raw_allowed)) or set(raw_allowed) != ALLOWED_AXIOMS:
            raise ValueError('caller cannot expand or change the fixed axiom allowlist')
    return declarations, counts, set(all_names)


def validate_transcript(text, expected_census, program_sha256, *, exit_code, stderr):
    if not isinstance(text, str):
        raise ValueError('transcript must be text')
    if type(exit_code) is not int or exit_code != 0 or not isinstance(stderr, str) or stderr:
        raise ValueError('successful exit and empty stderr are required')
    if not isinstance(program_sha256, str) or not SHA256.fullmatch(program_sha256):
        raise ValueError('invalid program SHA256')
    declarations, expected_counts, expected_names = expected_records(expected_census)
    counts, observed, seen = {}, {}, set()
    current, beginning, completion = None, None, None
    for line in text.splitlines():
        if not line:
            continue
        if completion is not None:
            raise ValueError('records after completion')
        match = re.fullmatch(r'KERNEL_CLOSURE_MODULE (\S+) COUNT ([0-9]+)', line)
        if match:
            module, raw_count = match.groups()
            if beginning is not None or module not in declarations or module in counts or int(raw_count) < 1:
                raise ValueError('invalid or duplicate module record')
            current = module
            counts[module], observed[module] = int(raw_count), []
            continue
        match = re.fullmatch(r'KERNEL_CLOSURE_SEED (\S+) (\S+)', line)
        if match:
            module, name = match.groups()
            if beginning is not None or current != module or name in seen or not NAME.fullmatch(name):
                raise ValueError('invalid, duplicate or misplaced owned seed')
            observed[module].append(name)
            seen.add(name)
            continue
        match = re.fullmatch(r'KERNEL_CLOSURE_REPLAY_BEGIN SEEDS ([0-9]+) CLOSURE ([0-9]+)', line)
        if match:
            if beginning is not None or current is None:
                raise ValueError('invalid or duplicate replay beginning')
            beginning = tuple(map(int, match.groups()))
            continue
        match = re.fullmatch(r'KERNEL_CLOSURE_STANDARD_ONLY_PASS SEEDS ([0-9]+) CLOSURE ([0-9]+) AXIOMS \[([^\[\]]*)\]', line)
        if match:
            if beginning is None:
                raise ValueError('completion before replay beginning')
            seed_count, closure_count, raw = match.groups()
            axioms = [] if not raw else raw.split(', ')
            if any(not NAME.fullmatch(ax) for ax in axioms):
                raise ValueError('malformed axiom list')
            if len(axioms) != len(set(axioms)):
                raise ValueError('duplicate axiom')
            if not set(axioms) <= ALLOWED_AXIOMS:  # MUTATION_ANCHOR_AXIOMS
                raise ValueError('forbidden axiom')
            completion = (int(seed_count), int(closure_count), sorted(axioms))
            continue
        raise ValueError('unrecognized replay output: ' + line)
    if beginning is None or completion is None:
        raise ValueError('missing replay beginning or completion')
    if seen != expected_names:  # MUTATION_ANCHOR_OWNED_NAMES
        raise ValueError('owned names differ from complete compiled census')
    if set(counts) != set(expected_counts) or counts != expected_counts or any(len(observed[m]) != counts[m] for m in observed):
        raise ValueError('module counts differ from complete compiled census')
    if any(set(observed[m]) != set(declarations[m]) for m in declarations):  # MUTATION_ANCHOR_MODULE_ASSIGNMENT
        raise ValueError('module assignment differs from complete compiled census')
    own, closure, axioms = completion
    if beginning != (own, closure) or own != len(seen) or closure < own or own < 1:
        raise ValueError('replay beginning/completion/census counts disagree')
    names_blob = ('\n'.join(sorted(seen)) + '\n').encode()
    return {'schema': 'cbz-proof-closure-transcript-v1', 'status': 'CONSISTENT_SUPPLIED_EXECUTION_DATA',
            'execution_authentication': False, 'kernel_verification_by_python': False,
            'program_sha256': program_sha256, 'stdout_sha256': digest(text.encode()),
            'stderr_sha256': digest(stderr.encode()), 'exit_code': exit_code,
            'owned_constants': own, 'transitive_type_value_closure_constants': closure,
            'owned_names_sha256': digest(names_blob), 'axioms': axioms,
            'module_declarations': {m: sorted(observed[m]) for m in declarations},
            'scope': 'Transcript/compiled-census consistency only; actual Lean execution, source and object bindings must be supplied separately.'}


def ensure_distinct_paths(paths):
    resolved = [Path(p).resolve() for p in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError('input/output paths must be distinct')


def write_outputs_exclusive(outputs):
    ensure_distinct_paths(list(outputs))
    prepared, linked = [], []
    try:
        for destination, data in outputs.items():
            destination = Path(destination).resolve()
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(str(destination))
            if not isinstance(data, bytes):
                raise TypeError('output payload must be bytes')
            fd, temporary = tempfile.mkstemp(prefix='.cbz-replay-', dir=destination.parent)
            prepared.append((Path(temporary), destination))
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, destination in prepared:
            os.link(temporary, destination)
            linked.append((destination, temporary.stat().st_ino))
    except BaseException:
        for destination, inode in linked:
            if destination.exists() and destination.lstat().st_ino == inode:
                destination.unlink()
        raise
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    generate = sub.add_parser('generate')
    generate.add_argument('--collector', type=Path, required=True)
    generate.add_argument('--collector-sha256', required=True)
    generate.add_argument('--module', action='append', required=True)
    generate.add_argument('--import', dest='imports', action='append', required=True)
    generate.add_argument('--program', type=Path, required=True)
    generate.add_argument('--metadata', type=Path, required=True)
    validate = sub.add_parser('validate')
    validate.add_argument('--transcript', type=Path, required=True)
    validate.add_argument('--stderr', type=Path, required=True)
    validate.add_argument('--exit-code', type=int, required=True)
    validate.add_argument('--expected-census', type=Path, required=True)
    validate.add_argument('--program', type=Path, required=True)
    validate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'generate':
        ensure_distinct_paths([args.collector, args.program, args.metadata])
        program, metadata = generate_program(args.collector, args.collector_sha256, args.module, args.imports)
        write_outputs_exclusive({args.program: program.encode(), args.metadata: (json.dumps(metadata, indent=2) + '\n').encode()})
    else:
        ensure_distinct_paths([args.transcript, args.stderr, args.expected_census, args.program, args.output])
        receipt = validate_transcript(args.transcript.read_bytes().decode('utf-8'), load_json_unique(args.expected_census), digest(args.program.read_bytes()), exit_code=args.exit_code, stderr=args.stderr.read_bytes().decode('utf-8'))
        write_outputs_exclusive({args.output: (json.dumps(receipt, indent=2) + '\n').encode()})


if __name__ == '__main__':
    main()
