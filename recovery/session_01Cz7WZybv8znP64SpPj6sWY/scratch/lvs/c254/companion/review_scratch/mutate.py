#!/usr/bin/env python3
"""Review driver: weaken one check in a COPY of check.py, regenerate SOURCE_FILES.json for the copy,
run test_torus_filtration.py against the copy, and report whether the suite catches the weakening.
Lives outside the worktree; writes only under review_scratch/."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

S = Path('<SCRATCH>')
W = S / 'repos/math_c254'
R = S / 'lvs/c254/companion/review_scratch'
GEN = S / 'lvs/c254/companion/gen_source_files.py'
PKG = 'companions/torus_filtration_20261011'
FILES = ('TorusFiltration.lean', 'TorusTopology.lean', 'ClosedSuperlevel.lean', 'check.py', 'replay.sh',
         'test_torus_filtration.py', 'README.md', 'SCOPE.md')

# (label, old, new) -- exact single-occurrence string replacements in check.py
MUTATIONS = [
    ('W01 audit: allow forbidden axioms', "require(set(axioms) <= ALLOWED, 'forbidden transitive axiom: ' + repr(sorted(set(axioms) - ALLOWED)))", "pass"),
    ('W02 audit: drop missing-target check', "require(set(records) == set(names), 'missing target axiom report: ' + repr(sorted(set(names) - set(records))))", "pass"),
    ('W03 audit: drop unexpected/duplicate target', "require(name in names and name not in records, 'unexpected or duplicate axiom target: ' + name)", "pass"),
    ('W04 audit: drop unrecognized-output check', "require(not AXIOM.sub('', text).strip(), 'unrecognized audit output')", "pass"),
    ('W05 audit: drop malformed-name check', "require(all(re.fullmatch(NAME, a) for a in axioms), 'malformed axiom name')", "pass"),
    ('W06 types: drop error/warning check', "require('error:' not in text and 'warning:' not in text, 'type diagnostic')", "pass"),
    ('W07 types: order-insensitive', "require(actual == names and len(set(actual)) == len(actual), 'type inventory mismatch')", "require(set(actual) == set(names) and len(set(actual)) == len(actual), 'type inventory mismatch')"),
    ('W08 types: drop preamble check', "require(not text[:matches[0].start()].strip(), 'unrecognized type preamble')", "pass"),
    ('W09 negative: drop unknown-control check', "require(control in CONTROLS, 'unknown negative control: ' + repr(control))", "pass"),
    ('W10 negative: false control accepts exit 0 or 1', "require(status == 1, 'false-statement control must exit exactly one')", "require(status in (0, 1), 'false-statement control must exit exactly one')"),
    ('W11 negative: false control search not fullmatch', "require(re.fullmatch(pattern, text) is not None, 'not the intended False-goal diagnostic: ' + control)", "require(re.search(pattern, text) is not None, 'not the intended False-goal diagnostic: ' + control)"),
    ('W12 negative: injection search not fullmatch', "require(re.fullmatch(prefix + re.escape(line), text) is not None, 'not the intended injected axiom report: ' + control)", "require(re.search(prefix + re.escape(line), text) is not None, 'not the intended injected axiom report: ' + control)"),
    ('W13 negative: injection accepts exit 0 or 1', "require(status == 0, 'injection control must be accepted by Lean (exit zero) so that only the audit refuses it')", "require(status in (0, 1), 'injection control must be accepted by Lean (exit zero) so that only the audit refuses it')"),
    ('W14 negative: audit refusal not required', "    try:\n        audit_axioms(line, ['injected'])\n    except ValueError as refusal:\n        require(str(refusal).startswith('forbidden transitive axiom'), 'audit refused for the wrong reason')\n        return 'REJECTED_BY_AXIOM_AUDIT'\n    raise ValueError('negative control escaped the axiom audit: ' + control)", "    return 'REJECTED_BY_AXIOM_AUDIT'"),
    ('W15 negative: drop int-type check', "require(type(status) is int, 'negative exit status must be an integer')", "pass"),
    ('W16 lean_version: accept any 4.34.x', "LEAN_VERSION = re.compile(r'Lean \\(version 4\\.34\\.1, ", "LEAN_VERSION = re.compile(r'Lean \\(version 4\\.34\\.[0-9], "),
    ('W17 packages: subset instead of equality', "require(records == expected, 'installed dependency HEADs differ from formal/lake-manifest.json: '", "require(set(records.items()) <= set(expected.items()), 'installed dependency HEADs differ from formal/lake-manifest.json: '"),
    ('W18 finish: drop build-status check', "require((out / (label + '.status')).read_bytes() == b'0\\n', 'missing successful process: ' + label)", "pass"),
    ('W19 finish: drop source-stability check', "require(initial == current, 'source changed across execution')", "pass"),
    ('W20 finish: BUILDS without leanchecker', "BUILDS = ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types', 'leanchecker')", "BUILDS = ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types')"),
    ('W21 source: drop membership check', "require(actual == expected, 'tracked companion membership differs: ' + repr(sorted(actual ^ expected)))", "pass"),
    ('W22 source: drop dirty-bytes check', "require(git('rev-parse', head + ':' + path) == ids[path]['git_blob'], 'dirty execution/source bytes: ' + path)", "pass"),
    ('W23 source: drop manifest-identity check', "require(meta['files'][path] == ids[path], 'manifest source mismatch: ' + path)", "pass"),
    ('W24 source: drop git diff check', "require(diff.returncode == 0, 'uncommitted modification in bound paths')", "pass"),
    ('W25 source: drop files==REQUIRED check', "require(set(meta['files']) == set(REQUIRED_FILES), 'files inventory must list every required companion file and the workflow')", "pass"),
    ('W26 source: drop GITHUB_SHA/REPOSITORY check', "require(os.environ.get('GITHUB_SHA') == head and os.environ.get('GITHUB_REPOSITORY') == REPO, 'host/source mismatch')", "pass"),
    ('W27 source: drop GITHUB_REPOSITORY only', "require(os.environ.get('GITHUB_SHA') == head and os.environ.get('GITHUB_REPOSITORY') == REPO, 'host/source mismatch')", "require(os.environ.get('GITHUB_SHA') == head, 'host/source mismatch')"),
    ('W28 source: drop run-identity check', "require(os.environ.get('GITHUB_RUN_ID') and os.environ.get('GITHUB_RUN_ATTEMPT'), 'run identity absent')", "pass"),
    ('W29 source: drop check_modules', "    check_modules(meta, SIDE)\n    return {'checked_commit'", "    return {'checked_commit'"),
    ('W30 scan: drop admission token scan', "        require(re.search(r'(?<![A-Za-z0-9_.])' + token + r'(?![A-Za-z0-9_])', code) is None,\n                'admission token in ' + module + ': ' + token)", "        pass"),
    ('W31 scan: drop unexpected-kind check', "require(found is None, 'unexpected column-0 declaration kind in ' + module + ': ' + found.group() if found else '')", "pass"),
    ('W32 scan: drop namespace check', "require(len(re.findall(r'^namespace ' + NAMESPACE + r'\\s*$', text, re.M)) == 1, 'namespace declaration: ' + module)", "pass"),
    ('W33 scan: drop handoff-namespace check', "require(not re.search(r'^namespace ResearchFormalCoreR1\\b', text, re.M), 'handoff namespace left in place: ' + module)", "pass"),
    ('W34 scan: drop anonymous-declaration check', "require(all(counts[k] == len(scan[k]) for k in scan), 'anonymous or unnamed declaration in ' + module)", "pass"),
    ('W35 check_modules: drop root-module text check', "require((side / (ROOT_MODULE + '.lean')).read_text(encoding='utf-8') == ROOT_SOURCE, 'root module must import exactly the two modules')", "pass"),
    ('W36 environment: drop toolchain text check', "require(toolchain == TOOLCHAIN, 'wrong Lean toolchain: ' + repr(toolchain))", "pass"),
    ('W37 environment: drop unpinned-dependency check', "require(all(isinstance(v, str) and HEX40.fullmatch(v) for v in revisions.values()), 'unpinned dependency')", "pass"),
    ('W38 environment: drop duplicate-package check', "require(names and len(names) == len(set(names)), 'duplicate or empty dependency list')", "pass"),
    ('W39 metadata: drop scientific_effect NONE', "require(meta['scientific_effect'] == 'NONE', 'scientific effect must be NONE')", "pass"),
    ('W40 metadata: drop alignment PENDING', "require(meta['alignment_status'] == 'PENDING_INDEPENDENT_REVIEW', 'alignment status cannot self-award review')", "pass"),
    ('W41 metadata: drop credit==0/human_review none', "require(prov['organizational_independence_credit'] == 0 and prov['human_review'] == 'none', 'credit must stay zero')", "pass"),
    ('W42 metadata: drop provenance identity', "require(prov['claim'] == CLAIM and prov['original_sha256'] == ORIGINAL_SHA256, 'provenance identity')", "pass"),
    ('W43 metadata: drop NOT_COMPILED/namespace_mapping', "require(prov['original_status'] == 'NOT_COMPILED' and prov['namespace_mapping'] == NAMESPACE_MAPPING, 'provenance labels')", "pass"),
    ('W44 metadata: drop schema-keys check', "require(set(meta) == {'schema', 'scientific_effect', 'alignment_status', 'namespace', 'root_module', 'modules',\n                          'definitions', 'theorems', 'files', 'provenance'}, 'SOURCE_FILES schema keys')", "pass"),
    ('W45 metadata: drop files<=REQUIRED check', "require(isinstance(files, dict) and files and set(files) <= set(REQUIRED_FILES), 'files inventory outside the companion')", "pass"),
    ('W46 metadata: drop duplicate-declaration check', "require(len(seen) == len(set(seen)), 'duplicate declaration name')", "pass"),
    ('W47 metadata: drop per-module inventory key check', "require(set(meta['theorems']) == set(MODULES) and set(meta['definitions']) == set(MODULES), 'per-module inventory keys')", "pass"),
    ('W48 ALLOWED gains sorryAx', "ALLOWED = frozenset({'propext', 'Classical.choice', 'Quot.sound'})", "ALLOWED = frozenset({'propext', 'Classical.choice', 'Quot.sound', 'sorryAx'})"),
    ('W49 emit: Types.lean without pp.explicit', "(out / 'Types.lean').write_text(header + 'set_option pp.explicit true\\n'", "(out / 'Types.lean').write_text(header"),
    ('W50 finish: scientific_acceptance True', "'alignment_status': 'PENDING_INDEPENDENT_REVIEW', 'scientific_acceptance': False,", "'alignment_status': 'PENDING_INDEPENDENT_REVIEW', 'scientific_acceptance': True,"),
    ('W51 finish: receipt written before checks', "    initial = strict((out / 'source.json').read_bytes())\n    current = source()", "    (out / 'receipt.json').write_text('{}')\n    initial = strict((out / 'source.json').read_bytes())\n    current = source()"),
    ('W52 strict: drop duplicate-key check', "require(key not in result, 'duplicate JSON key: ' + str(key))", "pass"),
    ('W53 source: drop symlink check', "require(not any(ROOT.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)), 'symlink source: ' + path)", "pass"),
    ('W54 negative: goal pattern any', "error: unsolved goals\\s*\\n⊢ False\\s*'", "error: unsolved goals\\s*\\n⊢ .*'"),
    ('W55 finish: control status not read (always 1 for false)', "status = int((out / (control + '.status')).read_text().strip())", "status = 1 if control in FALSE_CONTROLS else 0"),
    ('W56 targets: drop duplicate/empty check', "require(result and len(result) == len(set(result)), 'duplicate or empty target list')", "pass"),
    ('W57 audit: drop empty/duplicate target-list check', "require(isinstance(names, list) and names and len(set(names)) == len(names), 'empty or duplicate target list')", "pass"),
    ('W58 finish: log_sha256 only *.log', "'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}", "'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log')) if p.is_file()}}"),
    ('W59 finish: formalization_status proved', "'schema': 1, 'scientific_effect': 'NONE', 'formalization_status': 'kernel-checked',", "'schema': 1, 'scientific_effect': 'NONE', 'formalization_status': 'proved',"),
    ('W60 metadata: drop schema==1 int check', "require(type(meta['schema']) is int and meta['schema'] == 1, 'SOURCE_FILES schema version')", "pass"),
]


def base_copy(dest):
    if dest.exists():
        shutil.rmtree(dest)
    side = dest / PKG
    side.mkdir(parents=True)
    (dest / '.github/workflows').mkdir(parents=True)
    for f in FILES:
        shutil.copy(W / PKG / f, side / f)
    shutil.copy(W / PKG / 'SOURCE_FILES.json', side / 'SOURCE_FILES.json')
    shutil.copy(W / '.github/workflows/torus-filtration.yml', dest / '.github/workflows/torus-filtration.yml')
    shutil.copy(W / '.github/workflows/d2-moment-bridge.yml', dest / '.github/workflows/d2-moment-bridge.yml')


def run_suite(dest):
    r = subprocess.run([sys.executable, '-B', '-S', str(dest / PKG / 'test_torus_filtration.py')],
                       capture_output=True, text=True, timeout=600)
    failed = re.findall(r'^(?:FAIL|ERROR): (\w+)', r.stdout + r.stderr, re.M)
    return r.returncode, sorted(set(failed))


def main():
    only = sys.argv[1:]
    results = []
    base = R / 'mut_base'
    base_copy(base)
    gen = subprocess.run([sys.executable, '-B', '-S', str(GEN), '--worktree', str(base)], capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    rc, failed = run_suite(base)
    print('BASELINE (unmodified copy): exit %d failed=%s' % (rc, failed), flush=True)
    assert rc == 0, 'baseline must pass'
    for label, old, new in MUTATIONS:
        if only and not any(label.startswith(o) for o in only):
            continue
        dest = R / ('mut_' + label.split()[0])
        base_copy(dest)
        path = dest / PKG / 'check.py'
        text = path.read_text()
        n = text.count(old)
        if n != 1:
            results.append((label, 'MUTATION-NOT-APPLIED (occurrences=%d)' % n, []))
            print('%-55s %s' % (label, results[-1][1]), flush=True)
            continue
        text = text.replace(old, new)
        if new == 'pass' and old.startswith('require('):
            pass
        path.write_text(text)
        gen = subprocess.run([sys.executable, '-B', '-S', str(GEN), '--worktree', str(dest)], capture_output=True, text=True)
        if gen.returncode != 0:
            results.append((label, 'CAUGHT-BY-GENERATOR', [gen.stderr.strip()[:200]]))
            print('%-55s CAUGHT by generator: %s' % (label, gen.stderr.strip()[:120]), flush=True)
            continue
        rc, failed = run_suite(dest)
        verdict = 'CAUGHT' if rc != 0 else 'ESCAPED'
        results.append((label, verdict, failed))
        print('%-55s %s %s' % (label, verdict, ','.join(failed)), flush=True)
    (R / 'mutation_results.json').write_text(json.dumps(results, indent=1))
    esc = [r[0] for r in results if r[1] == 'ESCAPED']
    print('\nESCAPED (%d): %s' % (len(esc), '; '.join(esc)))


if __name__ == '__main__':
    main()
