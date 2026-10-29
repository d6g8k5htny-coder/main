#!/usr/bin/env python3
"""Portable Main arithmetic CBZ gate. Run with pinned lake on PATH.

python -B -S tools/cbz_lean_check.py

Reuses formal_gate_check, then executes actual whole-module census controls,
theorem-kind/premise capture, finite-domain witnesses and full --fresh replay.
Receipt: formal/.lake/cbz-evidence/receipt.json. No paper/theorem alignment or
Math- closure is inferred. All failed phase logs remain available.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys

ALLOWED = {'propext', 'Classical.choice', 'Quot.sound'}
PREMISES = {
    'UniversalLaw.Side24.pairing_sum_le_six': 'q ∈ [0,1,2,3,4,5,6]',
    'UniversalLaw.Side24.exponent_bounds': 'd ∈ [2,3]',
}
MODULES_LINE = 'let modules : Array Name := #[`UniversalLaw, `UniversalLaw.Side24.Ledger, `UniversalLaw.Side24.ConeMoments, `UniversalLaw.Side24.Endpoints]'
WITNESSES = ['CBZNonvacuity.' + n for n in ('pairing_domain_inhabited', 'pairing_all_seven', 'dimensions_inhabited', 'dimensions_both')]


# Independent semantic contracts: witness names and an allowed axiom list alone
# are insufficient. These expected propositions do not come from the witness file.
WITNESS_TYPES = {
    'CBZNonvacuity.pairing_domain_inhabited': '∃ q, q ∈ [0, 1, 2, 3, 4, 5, 6]',
    'CBZNonvacuity.pairing_all_seven': ' ∧ '.join(str(q) + ' ∈ [0, 1, 2, 3, 4, 5, 6]' for q in range(7)),
    'CBZNonvacuity.dimensions_inhabited': '∃ d, d ∈ [2, 3]',
    'CBZNonvacuity.dimensions_both': '2 ∈ [2, 3] ∧ 3 ∈ [2, 3]',
}
WITNESS_PROPOSITIONS = {
    'CBZNonvacuity.pairing_domain_inhabited': '∃ q : Nat, q ∈ [0,1,2,3,4,5,6]',
    'CBZNonvacuity.pairing_all_seven': ' ∧ '.join('(' + str(q) + ':Nat) ∈ [0,1,2,3,4,5,6]' for q in range(7)),
    'CBZNonvacuity.dimensions_inhabited': '∃ d : Nat, d ∈ [2,3]',
    'CBZNonvacuity.dimensions_both': '(2:Nat) ∈ [2,3] ∧ (3:Nat) ∈ [2,3]',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def validate_census(log, targets, modules):
    records, summaries = {}, []
    for line in log.splitlines():
        if line.startswith('CBZ_RECORD '):
            row = load_json(line[len('CBZ_RECORD '):])
            require(set(row) == {'name', 'module', 'kind', 'axioms', 'statement', 'prop_premises'}, 'invalid census record')
            name = row['name']
            require(isinstance(name, str) and name and name not in records, 'duplicate/invalid census declaration')
            require(row['module'] in modules, 'unexpected module')
            require(row['kind'] in {'theorem', 'definition', 'generated'}, 'project axiom or unknown constant kind')
            require(isinstance(row['axioms'], list) and all(isinstance(a, str) for a in row['axioms']) and set(row['axioms']) <= ALLOWED, 'forbidden transitive axiom')
            require(isinstance(row['statement'], str) and row['statement'], 'missing elaborated statement')
            require(isinstance(row['prop_premises'], list), 'invalid premise inventory')
            if name in WITNESS_TYPES:
                require(row['kind'] == 'theorem' and not row['prop_premises'] and
                        ''.join(row['statement'].split()) == ''.join(WITNESS_TYPES[name].split()),
                        'witness type contract changed: ' + name)
            for premise in row['prop_premises']:
                require(isinstance(premise, dict) and set(premise) == {'name', 'type'} and all(isinstance(v, str) for v in premise.values()), 'invalid premise record')
            if row['kind'] == 'theorem' and row['prop_premises']:
                require(name in PREMISES and len(row['prop_premises']) == 1, 'conditional theorem lacks an explicit nonvacuity witness')
                actual = ''.join(row['prop_premises'][0]['type'].split())
                require(actual == ''.join(PREMISES[name].split()), 'witness does not cover changed theorem premise')
            records[name] = row
        elif line.startswith('CBZ_END '):
            summaries.append(load_json(line[len('CBZ_END '):]))
        elif line.startswith('CBZ_'):
            raise ValueError('unknown/refused audit output: ' + line)
    require(len(summaries) == 1, 'missing/duplicate census end')
    end = summaries[0]
    require(set(end) == {'count', 'modules'} and type(end['count']) is int and end['count'] == len(records), 'census count mismatch')
    require(isinstance(end['modules'], list) and len(end['modules']) == len(set(end['modules'])) and set(end['modules']) == set(modules), 'missing/duplicate selected module')
    proofs = [row for row in records.values() if row['kind'] == 'theorem']
    require({row['name'] for row in proofs} == set(targets), 'actual theorem inventory differs from manifest targets')
    return dict(constants=list(records.values()), proofs=proofs, modules=end['modules'])


def run(command, label, out, pkg, env, timeout, expected=0):
    result = subprocess.run(command, cwd=pkg, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    (out / (label + '.log')).write_text(result.stdout, encoding='utf-8')
    require((result.returncode == 0) == (expected == 0), 'unexpected phase outcome: ' + label + '\n' + result.stdout[-3000:])
    return result.stdout


def audit_program(source, module):
    require(source.count(MODULES_LINE) == 1, 'missing/duplicate audit module boundary')
    require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', module) is not None, 'unsafe audit module name')
    marker = 'open Lean Meta Elab Command'
    require(source.count(marker) == 1, 'missing/duplicate audit command boundary')
    return ('import Lean\nimport ' + module + '\n' + source[source.index(marker):]).replace(
        MODULES_LINE, 'let modules : Array Name := #[`' + module + ']')


def witness_contract_program(module):
    require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', module) is not None, 'unsafe witness contract module')
    return ('import ' + module + '\n' +
            '\n'.join('example : (' + WITNESS_PROPOSITIONS[name] + ') := ' + name for name in WITNESSES) + '\n')


def self_test(source, lake, out, pkg, env, timeout):
    cases = {
        'honest': ('theorem honest : True := True.intro\n', True),
        'unused_axiom': ('axiom hidden : False\ntheorem honest : True := True.intro\n', False),
        'sorry': ('theorem bad : False := by sorry\n', False),
        'native': ('theorem native : (2 : Nat) + 2 = 4 := by native_decide\n', False),
        'false_premise': ('theorem vacuous (h : False) : False := h\n', True),
    }
    outcomes = {}
    for label, (body, accepted) in cases.items():
        module = 'CBZControl_' + label
        fixture = out / (module + '.lean')
        fixture.write_text('import UniversalLaw\n' + body)
        run([lake, 'env', 'lean', '-o', str(pkg / '.lake/build/lib/lean' / (module + '.olean')), str(fixture)], label + '-compile', out, pkg, env, timeout)
        audit = out / ('Audit_' + label + '.lean')
        audit.write_text(source.replace('import UniversalLaw', 'import ' + module).replace(MODULES_LINE, 'let modules : Array Name := #[`' + module + ']'))
        log = run([lake, 'env', 'lean', str(audit)], label + '-audit', out, pkg, env, timeout, expected=0 if accepted else 1)
        if not accepted:
            require('CBZ_FORBIDDEN' in log, 'control failed outside its intended axiom boundary')
        elif label == 'false_premise':
            try:
                validate_census(log, {'vacuous'}, {module})
            except ValueError as exc:
                require('nonvacuity' in str(exc), 'false-premise control failed for an unrelated reason')
            else:
                raise ValueError('false-premise theorem escaped premise gate')
        else:
            validate_census(log, {'honest'}, {module})
        outcomes[label] = 'EXPECTED_REJECTION' if not accepted or label == 'false_premise' else 'EXPECTED_ACCEPTANCE'
    missing = out / 'MissingModule.lean'
    missing.write_text(source.replace(MODULES_LINE, 'let modules : Array Name := #[`UniversalLaw, `MissingModule]'))
    log = run([lake, 'env', 'lean', str(missing)], 'missing-module', out, pkg, env, timeout, expected=1)
    require('CBZ_MISSING_MODULE' in log, 'missing-module control failed outside the inventory boundary')
    outcomes['missing_module'] = 'EXPECTED_REJECTION'
    # R3: same public names, empty axiom lists and no premises still do not
    # establish nonvacuity if all four witness statements have become True.
    module = 'CBZWeakenedWitness'
    fixture = out / (module + '.lean')
    fixture.write_text('import UniversalLaw\nnamespace CBZNonvacuity\n' +
                       ''.join('theorem ' + name.split('.')[-1] + ' : True := True.intro\n' for name in WITNESSES) +
                       'end CBZNonvacuity\n')
    run([lake, 'env', 'lean', '-o', str(pkg / '.lake/build/lib/lean' / (module + '.olean')), str(fixture)],
        'weakened-witness-compile', out, pkg, env, timeout)
    audit = out / 'AuditWeakenedWitness.lean'
    audit.write_text(audit_program(source, module))
    log = run([lake, 'env', 'lean', str(audit)], 'weakened-witness-audit', out, pkg, env, timeout)
    try:
        validate_census(log, set(WITNESSES), {module})
    except ValueError as exc:
        require('witness type contract' in str(exc), 'weakened witness rejected for unrelated reason')
    else:
        raise ValueError('weakened witness escaped the exact-type census contract')
    contract = out / 'WeakenedWitnessContract.lean'
    contract.write_text(witness_contract_program(module))
    log = run([lake, 'env', 'lean', str(contract)], 'weakened-witness-contract', out, pkg, env, timeout, expected=1)
    require('type mismatch' in log.lower(), 'weakened witness contract failed outside its proof type')
    outcomes['weakened_witness_types'] = 'REJECTED_BY_EXACT_TYPE_CENSUS_AND_LEAN_ASCRIPTION'
    return outcomes


def execute(root, timeout):
    gate_path = root / 'tools/formal_gate_check.py'
    spec = importlib.util.spec_from_file_location('existing_main_gate', gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    manifest, digest, pkg = gate.source_check(root)
    out = pkg / '.lake/cbz-evidence'
    require(not (pkg / '.lake').is_symlink() and not out.is_symlink(), 'symlink evidence directory refused')
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / 'receipt.json'
    receipt_path.unlink(missing_ok=True)
    inputs = [gate_path, root / 'tools/CBZNonvacuity.lean', Path(__file__).resolve()]
    before = {str(p.relative_to(root)): sha(p) for p in inputs}
    source = inputs[1].read_text()
    require(source.count(MODULES_LINE) == 1, 'census module boundary changed')
    lake = gate.lake_binary()
    require(lake is not None, 'pinned lake missing from PATH')
    env = dict(os.environ, PATH=str(Path(lake).parent) + os.pathsep + os.environ.get('PATH', ''))
    baseline = gate.execute(manifest, digest, pkg, root)
    controls = self_test(source, lake, out, pkg, env, timeout)
    audit = out / 'CBZNonvacuity.lean'
    audit.write_text(source)
    log = run([lake, 'env', 'lean', '-o', str(pkg / '.lake/build/lib/lean/CBZNonvacuity.olean'), str(audit)], 'all-declarations-and-witnesses', out, pkg, env, timeout)
    targets = {t['name'] for t in manifest['targets']}
    modules = {p[:-5].replace('/', '.') for p in manifest['source_modules']} | {manifest['root_module']}
    census = validate_census(log, targets, modules)
    witness_lines = '\n'.join(m.group() for m in gate.AXIOM_LINE.finditer(log))
    witnesses = gate.audit_axioms(witness_lines, WITNESSES)
    self_audit = out / 'SelfAudit.lean'
    self_audit.write_text(audit_program(source, 'CBZNonvacuity'))
    self_log = run([lake, 'env', 'lean', str(self_audit)], 'auditor-module-census', out, pkg, env, timeout)
    auditor_census = validate_census(self_log, set(WITNESSES), {'CBZNonvacuity'})
    contract = out / 'WitnessTypeContract.lean'
    contract.write_text(witness_contract_program('CBZNonvacuity'))
    run([lake, 'env', 'lean', str(contract)], 'witness-type-contract', out, pkg, env, timeout)
    run([lake, 'env', 'leanchecker', '-v', 'CBZNonvacuity'], 'witness-kernel', out, pkg, env, timeout)
    run([lake, 'env', 'leanchecker', '-v', '--fresh', manifest['root_module']], 'fresh-kernel', out, pkg, env, timeout)
    _, after, _ = gate.source_check(root)
    require(after == digest and all(sha(p) == before[str(p.relative_to(root))] for p in inputs), 'source or audit inputs changed during execution')
    receipt = dict(schema_version=1, scope='31 literal Main arithmetic statements only; no analytic/source parent closure', checked_commit=baseline['checked_commit'], manifest_sha256=digest, lean_version=baseline['lean_version'], baseline_receipt_sha256=sha(pkg / '.lake/formal-evidence/receipt.json'), audit_input_sha256=before, all_project_constants=census['constants'], audit_module_constants=auditor_census['constants'], proofs=census['proofs'], witness_axioms=witnesses, witness_type_contract=WITNESS_PROPOSITIONS, self_tests=controls, fresh_kernel='PASS', scientific_effect='NONE', source_alignment='NOT_INFERRED', logs={p.name: sha(p) for p in out.glob('*.log')})
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args(argv)
    try:
        require(args.timeout > 0, 'timeout must be positive')
        receipt = execute(args.root.resolve(), args.timeout)
        print(json.dumps(dict(status='PASS_LITERAL_MAIN_ARITHMETIC', proofs=len(receipt['proofs']), constants=len(receipt['all_project_constants']), receipt='formal/.lake/cbz-evidence/receipt.json'), sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as exc:
        print('CBZ_LEAN_FAIL: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
