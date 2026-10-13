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

def sha(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_lean_version(text):
    version = text.strip()
    require(re.fullmatch(r'Lean \(version 4\.34\.1, [^()\r\n]+, commit 5045d0056413266e57c625dcd7c365b10e377c52, Release\)', version) is not None,
            'unexpected running Lean version: ' + repr(version))
    return version

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
    required = {'gate.py', 'tests/test_gate.py', 'lean-toolchain', 'lakefile.toml', 'lake-manifest.json', 'ResearchFormalCoreR1.lean', 'SCOPE.md', 'GLOSSARY.md', 'README.md', 'blueprint/src/content.tex', 'tests/test_alignment_lineage.py', 'LINEAGE_VALIDATION.md'}
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
    names = []
    for path in m['source_modules']:
        require(path in m['files'], 'unbound source module')
        text = (ROOT / path).read_text()
        names.extend('ResearchFormalCoreR1.' + n for n in re.findall(r'^(?:theorem|lemma)\s+(' + NAME + ')', text, re.M))
    require(names == m['targets'], 'target inventory differs from source declarations')
    check_blueprint((ROOT / 'blueprint/src/content.tex').read_text(), names)
    lean_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.lean') if '.lake' not in p.relative_to(ROOT).parts}
    require(lean_files == set(m['source_modules']) | {'ResearchFormalCoreR1.lean'}, 'unregistered Lean module')
    return m, sha(raw)

def run(command, label, out, expect_success=True):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (out / (label + '.log')).write_text(result.stdout)
    print(label + ': exit ' + str(result.returncode), flush=True)
    require((result.returncode == 0) == expect_success, 'unexpected process outcome: ' + label + '\n' + result.stdout[-6000:])
    return result.stdout

def execute(m, digest):
    out = ROOT / '.lake' / 'formal-evidence'
    out.mkdir(parents=True, exist_ok=True)
    require(not (ROOT / '.lake').is_symlink() and not (ROOT / '.lake/build').is_symlink(), 'symlink build directory')
    check_lean_version(run(['lake', 'env', 'lean', '--version'], 'version-preflight', out))
    if (ROOT / '.lake/build').exists():
        shutil.rmtree(ROOT / '.lake/build')  # fresh local-package build; dependency cache is untouched
    run(['lake', 'build'], 'build', out)
    (out / 'leanchecker.log').write_text('SKIPPED IN PARTIAL LOCAL RUN: leanchecker was OOM-killed (exit 137, anon-rss 13.49 GB) by this container memory cgroup on 2026-10-12T22:42Z; this partial run is NOT a gate pass.\n')
    print('leanchecker: SKIPPED (partial local run; not a gate pass)', flush=True)
    audit = out / 'Audit.lean'
    audit.write_text('import ResearchFormalCoreR1\n' + '\n'.join('#print axioms ' + n for n in m['targets']) + '\n')
    text = run(['lake', 'env', 'lean', str(audit)], 'axioms', out)
    axioms = audit_axioms(text, m['targets'])
    types = out / 'Types.lean'
    types.write_text('import ResearchFormalCoreR1\nset_option pp.explicit true\n' + '\n'.join('#check ' + n for n in m['targets']) + '\n')
    run(['lake', 'env', 'lean', str(types)], 'elaborated-types', out)
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
    version = check_lean_version(run(['lake', 'env', 'lean', '--version'], 'version', out))
    actual_dependencies = {name: subprocess.check_output(['git', '-C', str(ROOT / '.lake/packages' / name), 'rev-parse', 'HEAD'], text=True).strip() for name in m['dependency_revisions']}
    require(actual_dependencies == m['dependency_revisions'], 'installed dependency HEAD differs from lock')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(COMMIT.fullmatch(head), 'missing exact checked commit')
    receipt = dict(schema_version=1, scientific_effect='NONE', formalization_status='kernel-checked',
                   alignment_status='PENDING_INDEPENDENT_REVIEW', manifest_sha256=digest,
                   checked_commit=head, repository=os.environ.get('GITHUB_REPOSITORY'),
                   workflow_run_id=os.environ.get('GITHUB_RUN_ID'), lean_version=version,
                   dependency_revisions=m['dependency_revisions'], axioms=axioms, negative_controls=outcomes)
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
