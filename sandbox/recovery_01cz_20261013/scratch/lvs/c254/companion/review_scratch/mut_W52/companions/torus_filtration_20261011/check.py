#!/usr/bin/env python3
"""Source-bound evidence driver for the C254 torus/closed-superlevel companion (stdlib only).

Not an acceptance engine, not a general Lean parser, not a review. It binds the
companion's bytes to Git HEAD, emits the audit/type/positive/negative Lean files,
refuses negative-control logs that are not the intended diagnostic, and writes a
receipt that records what ran. The companion consumes nothing from formal/ except
the pinned toolchain, lakefile and dependency lock, so it pins no core tree,
manifest or gate. Scientific effect NONE; organizational-independence credit 0.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SIDE = Path(__file__).resolve().parent
ROOT = SIDE.parents[1]
PACKAGE = 'companions/torus_filtration_20261011'
WORKFLOW = '.github/workflows/torus-filtration.yml'
REPO = 'd6g8k5htny-coder/Math-'
NAMESPACE = 'TorusFiltration'
MODULES = ('TorusTopology', 'ClosedSuperlevel')
ROOT_MODULE = 'TorusFiltration'
ROOT_SOURCE = 'import TorusTopology\nimport ClosedSuperlevel\n'
REQUIRED_FILES = tuple(PACKAGE + '/' + n for n in (
    'TorusFiltration.lean', 'TorusTopology.lean', 'ClosedSuperlevel.lean', 'check.py', 'replay.sh',
    'test_torus_filtration.py', 'README.md', 'SCOPE.md')) + (WORKFLOW,)
PINNED_CORE = ('formal/lakefile.toml', 'formal/lake-manifest.json', 'formal/lean-toolchain')
TOOLCHAIN = 'leanprover/lean4:v4.34.1'
LEAN_VERSION = re.compile(r'Lean \(version 4\.34\.1, [^()\r\n]+, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release\)')
CLAIM = '8a5c7a24-784b-4aaa-b951-cd9f37b0dc64'
ORIGINAL_SHA256 = {
    'ResearchFormalCoreR1/TorusTopology.lean': 'efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4',
    'ResearchFormalCoreR1/ClosedSuperlevel.lean': '1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e'}
NAMESPACE_MAPPING = 'ResearchFormalCoreR1.<n> (handoff) -> TorusFiltration.<n> (companion)'
# Own copy of formal/gate.py's audit rules; the companion imports nothing from the core.
ALLOWED = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
NAME = r'[A-Za-z_][A-Za-z0-9_.]*'
AXIOM = re.compile(r"'(" + NAME + r")' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)")
IDENT = r'[A-Za-z][A-Za-z0-9_]*'
HEX40 = re.compile(r'[0-9a-f]{40}')
HEX64 = re.compile(r'[0-9a-f]{64}')
BUILDS = ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types', 'leanchecker')
ADMISSION = ('sorry', 'admit', 'axiom', 'native_decide', 'unsafe', 'opaque', 'partial', 'implemented_by', 'extern',
             'skipKernelTC', 'autoImplicit')
UNEXPECTED_KINDS = re.compile(r'^(?:lemma|structure|inductive|class|example|axiom|opaque|macro|macro_rules|syntax|elab|notation)\b', re.M)
# False statements about the companion's own definitions; Lean must leave exactly `⊢ False`.
FALSE_CONTROLS = {
    'RejectSuperlevelSignFlip': ('() ∈ superlevel_set (fun _ : Unit => (0 : ℝ)) (-1)', 'norm_num [superlevel_set]'),
    'RejectSuperlevelBelowThreshold': ('(3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-4)', 'norm_num [superlevel_set]'),
    'RejectConstantNotPeriodic': ('¬ torus_isPeriodic 1 (1 : ℝ) (fun _ => (0 : ℝ))', 'simp [torus_isPeriodic]')}
# Admission injections; Lean accepts each file (exit 0), the transitive axiom audit must refuse it.
# Diagnostics pinned from Lean 4.34.1 at the Mathlib d13f23b7 pin: the sorry warning uses backticks, and
# native_decide introduces the auxiliary axiom `injected._native.native_decide.ax_1_1` (not Lean.ofReduceBool).
INJECTIONS = {
    'RejectSorry': ("theorem injected : False := by sorry\n", 'sorryAx',
                    r"(?:[^\r\n]*[/\\])?RejectSorry\.lean:[0-9]+:[0-9]+: warning: declaration uses `sorry`\n"),
    'RejectCustomAxiom': ("axiom hiddenPremise : False\ntheorem middle : False := hiddenPremise\n"
                          "theorem injected : False := middle\n", 'hiddenPremise', ''),
    'RejectNativeDecide': ("theorem injected : (2 : Nat) + 2 = 4 := by native_decide\n",
                           'injected._native.native_decide.ax_1_1', '')}
CONTROLS = tuple(FALSE_CONTROLS) + tuple(INJECTIONS)
POSITIVE = '''import TorusFiltration
open TorusFiltration

-- Lattice membership of the literal Field lattice vector (torus_lattice_mem_iff).
example (d : ℕ) (L : ℝ) (hL : 0 < L) (n : Fin d → ℤ) :
    torus_latticeVec d L n ∈ torus_periodLattice d L hL :=
  (torus_lattice_mem_iff d L hL _).2 ⟨n, rfl⟩

-- A concrete lattice translate of the origin has the same quotient image (torus_control_lattice_translate).
example : torus_proj 2 1 one_pos (0 + torus_latticeVec 2 1 ![3, -5]) = torus_proj 2 1 one_pos 0 :=
  torus_control_lattice_translate 2 1 one_pos 0 ![3, -5]

-- Constants are periodic and descend to themselves (torus_descend_proj).
example (d : ℕ) (L : ℝ) (hL : 0 < L) (c : ℝ) (x : torus_E d) :
    torus_descend d L hL (fun _ => c) (fun _ _ => rfl) (torus_proj d L hL x) = c :=
  torus_descend_proj d L hL (fun _ => c) (fun _ _ => rfl) x

-- The path-component map on a representative (superlevel_componentMap_mk).
example {X Y : Type} [TopologicalSpace X] [TopologicalSpace Y] (f : C(X, Y)) (x : X) :
    superlevel_componentMap f (ZerothHomotopy.mk x) = ZerothHomotopy.mk (f x) :=
  superlevel_componentMap_mk f x

-- Closed-superlevel monotonicity at concrete parameters (superlevel_mono).
example : superlevel_set (fun x : ℝ => x) (-2) ⊆ superlevel_set (fun x : ℝ => x) 1 :=
  superlevel_mono (fun x : ℝ => x) (by norm_num)

-- Closed endpoint membership for the identity height: 2 ≤ 3, so 3 ∈ F(-2).
example : (3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-2) := by
  norm_num [superlevel_set]
'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b'blob %d\0' % len(raw) + raw).hexdigest()


def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw), 'git_blob': blob(raw)}


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            pass
            result[key] = value
        return result

    def reject(_):
        raise ValueError('nonfinite or noninteger JSON number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject, parse_float=reject)


def git(*args):
    run = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    require(run.returncode == 0, 'git ' + ' '.join(args) + ' failed: ' + run.stderr.strip())
    return run.stdout.strip()


def metadata():
    meta = strict((SIDE / 'SOURCE_FILES.json').read_bytes())
    require(set(meta) == {'schema', 'scientific_effect', 'alignment_status', 'namespace', 'root_module', 'modules',
                          'definitions', 'theorems', 'files', 'provenance'}, 'SOURCE_FILES schema keys')
    require(type(meta['schema']) is int and meta['schema'] == 1, 'SOURCE_FILES schema version')
    require(meta['scientific_effect'] == 'NONE', 'scientific effect must be NONE')
    require(meta['alignment_status'] == 'PENDING_INDEPENDENT_REVIEW', 'alignment status cannot self-award review')
    require(meta['namespace'] == NAMESPACE and meta['root_module'] == ROOT_MODULE, 'namespace or root module')
    require(meta['modules'] == list(MODULES), 'module list')
    require(set(meta['theorems']) == set(MODULES) and set(meta['definitions']) == set(MODULES), 'per-module inventory keys')
    seen = []
    for module in MODULES:
        theorems = meta['theorems'][module]
        require(isinstance(theorems, list) and theorems, 'empty theorem list: ' + module)
        kinds = meta['definitions'][module]
        require(isinstance(kinds, dict) and set(kinds) == {'abbrev', 'def', 'instance'}, 'definition kinds: ' + module)
        for names in (theorems, kinds['abbrev'], kinds['def'], kinds['instance']):
            require(isinstance(names, list) and all(isinstance(n, str) and re.fullmatch(IDENT, n) for n in names),
                    'invalid declaration name in ' + module)
            seen.extend(names)
    require(len(seen) == len(set(seen)), 'duplicate declaration name')
    files = meta['files']
    require(isinstance(files, dict) and files and set(files) <= set(REQUIRED_FILES), 'files inventory outside the companion')
    for path, entry in files.items():
        require(isinstance(entry, dict) and set(entry) == {'bytes', 'sha256', 'git_blob'}, 'file entry keys: ' + path)
        require(type(entry['bytes']) is int and entry['bytes'] >= 0, 'file bytes: ' + path)
        require(isinstance(entry['sha256'], str) and HEX64.fullmatch(entry['sha256']), 'file sha256: ' + path)
        require(isinstance(entry['git_blob'], str) and HEX40.fullmatch(entry['git_blob']), 'file git blob: ' + path)
    prov = meta['provenance']
    require(isinstance(prov, dict), 'provenance object')
    for key in ('claim', 'handoff_pr', 'handoff_comments', 'original_sha256', 'original_status', 'namespace_mapping',
                'source_author', 'compile_fixes_and_companion', 'organizational_independence_credit', 'human_review'):
        require(key in prov, 'provenance key missing: ' + key)
    require(prov['claim'] == CLAIM and prov['original_sha256'] == ORIGINAL_SHA256, 'provenance identity')
    require(prov['original_status'] == 'NOT_COMPILED' and prov['namespace_mapping'] == NAMESPACE_MAPPING, 'provenance labels')
    require(prov['organizational_independence_credit'] == 0 and prov['human_review'] == 'none', 'credit must stay zero')
    return meta


def theorem_targets(meta):
    return [NAMESPACE + '.' + n for module in MODULES for n in meta['theorems'][module]]


def definition_targets(meta):
    return [NAMESPACE + '.' + n for module in MODULES for kind in ('abbrev', 'def', 'instance')
            for n in meta['definitions'][module][kind]]


def targets(meta):
    result = theorem_targets(meta) + definition_targets(meta)
    require(result and len(result) == len(set(result)), 'duplicate or empty target list')
    return result


def strip_comments(text):
    text = re.sub(r'/-.*?-/', '', text, flags=re.S)
    return re.sub(r'--[^\n]*', '', text)


def scan_module(text, module):
    """Column-0 declaration inventory of one companion module; not a Lean parser."""
    require(len(re.findall(r'^namespace ' + NAMESPACE + r'\s*$', text, re.M)) == 1, 'namespace declaration: ' + module)
    require(len(re.findall(r'^end ' + NAMESPACE + r'\s*$', text, re.M)) == 1, 'namespace end: ' + module)
    require(not re.search(r'^namespace ResearchFormalCoreR1\b', text, re.M), 'handoff namespace left in place: ' + module)
    found = UNEXPECTED_KINDS.search(text)
    require(found is None, 'unexpected column-0 declaration kind in ' + module + ': ' + found.group() if found else '')
    scan = {'theorem': re.findall(r'^theorem\s+(' + IDENT + ')', text, re.M),
            'abbrev': re.findall(r'^abbrev\s+(' + IDENT + ')', text, re.M),
            'def': re.findall(r'^(?:noncomputable\s+)?def\s+(' + IDENT + ')', text, re.M),
            'instance': re.findall(r'^instance\s+(' + IDENT + ')', text, re.M)}
    counts = {'theorem': len(re.findall(r'^theorem\b', text, re.M)), 'abbrev': len(re.findall(r'^abbrev\b', text, re.M)),
              'def': len(re.findall(r'^(?:noncomputable\s+)?def\b', text, re.M)),
              'instance': len(re.findall(r'^instance\b', text, re.M))}
    require(all(counts[k] == len(scan[k]) for k in scan), 'anonymous or unnamed declaration in ' + module)
    code = strip_comments(text)
    for token in ADMISSION:
        require(re.search(r'(?<![A-Za-z0-9_.])' + token + r'(?![A-Za-z0-9_])', code) is None,
                'admission token in ' + module + ': ' + token)
    return scan


def check_modules(meta, side):
    for module in MODULES:
        path = side / (module + '.lean')
        require(path.is_file(), 'module absent: ' + module)
        scan = scan_module(path.read_text(encoding='utf-8'), module)
        require(scan['theorem'] == meta['theorems'][module], 'theorem inventory differs from source: ' + module)
        for kind in ('abbrev', 'def', 'instance'):
            require(scan[kind] == meta['definitions'][module][kind], kind + ' inventory differs from source: ' + module)
    require((side / (ROOT_MODULE + '.lean')).read_text(encoding='utf-8') == ROOT_SOURCE, 'root module must import exactly the two modules')


def environment(root):
    toolchain = (root / 'formal/lean-toolchain').read_text().strip()
    require(toolchain == TOOLCHAIN, 'wrong Lean toolchain: ' + repr(toolchain))
    lock = strict((root / 'formal/lake-manifest.json').read_bytes())
    packages = lock['packages']
    names = [p['name'] for p in packages]
    require(names and len(names) == len(set(names)), 'duplicate or empty dependency list')
    revisions = {p['name']: p['rev'] for p in packages}
    require(all(isinstance(v, str) and HEX40.fullmatch(v) for v in revisions.values()), 'unpinned dependency')
    return {'lean_toolchain': toolchain, 'dependency_revisions': revisions}


def source():
    meta = metadata()
    head = git('rev-parse', 'HEAD')
    require(HEX40.fullmatch(head) is not None, 'invalid HEAD')
    require(set(meta['files']) == set(REQUIRED_FILES), 'files inventory must list every required companion file and the workflow')
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        require(os.environ.get('GITHUB_SHA') == head and os.environ.get('GITHUB_REPOSITORY') == REPO, 'host/source mismatch')
        require(os.environ.get('GITHUB_RUN_ID') and os.environ.get('GITHUB_RUN_ATTEMPT'), 'run identity absent')
    expected = set(REQUIRED_FILES) | {PACKAGE + '/SOURCE_FILES.json'}
    actual = set(git('ls-tree', '-r', '--name-only', head, '--', PACKAGE).splitlines()) | {WORKFLOW}
    require(actual == expected, 'tracked companion membership differs: ' + repr(sorted(actual ^ expected)))
    ids = {}
    for path in sorted(expected):
        parts = Path(path).parts
        require(not any(ROOT.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)), 'symlink source: ' + path)
        raw = (ROOT / path).read_bytes()
        ids[path] = identity(raw)
        require(git('rev-parse', head + ':' + path) == ids[path]['git_blob'], 'dirty execution/source bytes: ' + path)
        if path in meta['files']:
            require(meta['files'][path] == ids[path], 'manifest source mismatch: ' + path)
    diff = subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', PACKAGE, WORKFLOW, *PINNED_CORE],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    require(diff.returncode == 0, 'uncommitted modification in bound paths')
    env = environment(ROOT)
    check_modules(meta, SIDE)
    return {'checked_commit': head, 'repository': REPO, 'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
            'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', 'local'), 'sources': ids,
            'declarations': targets(meta), 'theorems': theorem_targets(meta), 'environment': env,
            'scientific_effect': 'NONE', 'provenance': meta['provenance']}


def lean_version(text):
    version = text.strip()
    require(LEAN_VERSION.fullmatch(version) is not None, 'unexpected running Lean version: ' + repr(version))
    return version


def packages(text, expected):
    records = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([A-Za-z][A-Za-z0-9_.-]*) ([0-9a-f]{40})', line)
        require(match is not None, 'malformed package line: ' + repr(line))
        name, rev = match.groups()
        require(name not in records, 'duplicate installed package: ' + name)
        records[name] = rev
    require(records == expected, 'installed dependency HEADs differ from formal/lake-manifest.json: '
            + repr(sorted(set(records.items()) ^ set(expected.items()))))
    return records


def audit_axioms(text, names):
    require(isinstance(names, list) and names and len(set(names)) == len(names), 'empty or duplicate target list')
    records = {}
    for match in AXIOM.finditer(text):
        name, raw = match.groups()
        require(name in names and name not in records, 'unexpected or duplicate axiom target: ' + name)
        axioms = [] if raw is None or not raw.strip() else [a.strip() for a in raw.split(',')]
        require(all(re.fullmatch(NAME, a) for a in axioms), 'malformed axiom name')
        require(set(axioms) <= ALLOWED, 'forbidden transitive axiom: ' + repr(sorted(set(axioms) - ALLOWED)))
        records[name] = sorted(set(axioms))
    require(not AXIOM.sub('', text).strip(), 'unrecognized audit output')
    require(set(records) == set(names), 'missing target axiom report: ' + repr(sorted(set(names) - set(records))))
    return records


def types(text, names):
    require('error:' not in text and 'warning:' not in text, 'type diagnostic')
    matches = list(re.finditer(r'^' + NAMESPACE + r'\.' + IDENT, text, re.M))
    actual = [m.group() for m in matches]
    require(actual == names and len(set(actual)) == len(actual), 'type inventory mismatch')
    require(not text[:matches[0].start()].strip(), 'unrecognized type preamble')
    return actual


def negative(text, control, status):
    require(control in CONTROLS, 'unknown negative control: ' + repr(control))
    require(type(status) is int, 'negative exit status must be an integer')
    if control in FALSE_CONTROLS:
        require(status == 1, 'false-statement control must exit exactly one')
        pattern = r'(?:[^\r\n]*[/\\])?' + control + r'\.lean:[0-9]+:[0-9]+: error: unsolved goals\s*\n⊢ False\s*'
        require(re.fullmatch(pattern, text) is not None, 'not the intended False-goal diagnostic: ' + control)
        return 'REJECTED_BY_LEAN'
    _, forbidden, prefix = INJECTIONS[control]
    require(status == 0, 'injection control must be accepted by Lean (exit zero) so that only the audit refuses it')
    line = "'injected' depends on axioms: [" + forbidden + "]\n"
    require(re.fullmatch(prefix + re.escape(line), text) is not None, 'not the intended injected axiom report: ' + control)
    try:
        audit_axioms(line, ['injected'])
    except ValueError as refusal:
        require(str(refusal).startswith('forbidden transitive axiom'), 'audit refused for the wrong reason')
        return 'REJECTED_BY_AXIOM_AUDIT'
    raise ValueError('negative control escaped the axiom audit: ' + control)


def emit(out):
    meta = metadata()
    header = 'import ' + ROOT_MODULE + '\n'
    (out / 'Audit.lean').write_text(header + '\n'.join('#print axioms ' + n for n in targets(meta)) + '\n')
    (out / 'Types.lean').write_text(header + 'set_option pp.explicit true\n'
                                    + '\n'.join('#check ' + n for n in theorem_targets(meta)) + '\n')
    (out / 'Positive.lean').write_text(POSITIVE)
    for name, (statement, tactic) in FALSE_CONTROLS.items():
        (out / (name + '.lean')).write_text(header + 'open ' + NAMESPACE + '\nexample : ' + statement + ' := by\n  ' + tactic + '\n')
    for name, (body, _, _) in INJECTIONS.items():
        (out / (name + '.lean')).write_text(header + body + '#print axioms injected\n')


def finish(out):
    initial = strict((out / 'source.json').read_bytes())
    current = source()
    require(initial == current, 'source changed across execution')
    meta = metadata()
    version = lean_version((out / 'toolchain.log').read_text())
    installed = packages((out / 'packages.log').read_text(), current['environment']['dependency_revisions'])
    for label in BUILDS:
        require((out / (label + '.status')).read_bytes() == b'0\n', 'missing successful process: ' + label)
    axioms = audit_axioms((out / 'axioms.log').read_text(), targets(meta))
    types((out / 'types.log').read_text(), theorem_targets(meta))
    outcomes = {}
    for control in CONTROLS:
        status = int((out / (control + '.status')).read_text().strip())
        outcomes[control] = negative((out / (control + '.log')).read_text(), control, status)
    definitions = {kind: sum(len(meta['definitions'][m][kind]) for m in MODULES) for kind in ('abbrev', 'def', 'instance')}
    receipt = {
        'schema': 1, 'scientific_effect': 'NONE', 'formalization_status': 'kernel-checked',
        'alignment_status': 'PENDING_INDEPENDENT_REVIEW', 'scientific_acceptance': False,
        'checked_commit': current['checked_commit'], 'repository': current['repository'],
        'run_id': current['run_id'], 'run_attempt': current['run_attempt'], 'lean_version': version,
        'dependency_revisions': installed, 'sources': current['sources'], 'declarations': current['declarations'],
        'theorem_count': len(current['theorems']), 'definition_count': definitions, 'axioms': axioms,
        'negative_controls': outcomes, 'provenance': current['provenance'],
        'scope_note': 'kernel-checked records that Lean elaborated and leanchecker rechecked these declarations at the '
                      'pinned toolchain; it is not alignment with the handoff scope, not review, and not acceptance.',
        'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('source', 'emit', 'negative', 'finish'))
    parser.add_argument('path', type=Path, nargs='?')
    parser.add_argument('--control', choices=CONTROLS)
    parser.add_argument('--status', type=int)
    args = parser.parse_args()
    try:
        if args.command == 'source':
            result = source()
        elif args.command == 'emit':
            require(args.path is not None and args.path.is_dir(), 'emit needs an existing output directory')
            emit(args.path)
            result = {'emitted': True, 'controls': list(CONTROLS)}
        elif args.command == 'finish':
            require(args.path is not None, 'finish needs the evidence directory')
            result = finish(args.path)
        else:
            require(args.path is not None and args.control is not None and args.status is not None,
                    'negative needs <log> --control C --status N')
            result = {'negative': args.control, 'outcome': negative(args.path.read_text(), args.control, args.status)}
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as failure:
        print('TORUS_FILTRATION_FAIL: ' + str(failure), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
