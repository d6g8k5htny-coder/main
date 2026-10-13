#!/usr/bin/env python3
"""Apply-pass mutation sweep: weaken one check in a COPY of check.py, regenerate SOURCE_FILES.json for the
copy, run test_torus_filtration.py there, and report CAUGHT/ESCAPED. Writes only under apply_scratch/."""
import json, re, shutil, subprocess, sys
from pathlib import Path

S = Path('<SCRATCH>')
W = S / 'repos/math_c254'
R = S / 'lvs/c254/companion/apply_scratch/sweep'
GEN = S / 'lvs/c254/companion/gen_source_files.py'
PKG = 'companions/torus_filtration_20261011'
FILES = ('TorusFiltration.lean', 'TorusTopology.lean', 'ClosedSuperlevel.lean', 'check.py', 'replay.sh',
         'test_torus_filtration.py', 'README.md', 'SCOPE.md', 'SOURCE_FILES.json')
P = 'pass'
MUTATIONS = [
    ('W05 audit malformed-name', "require(all(re.fullmatch(NAME, a) for a in axioms), 'malformed axiom name')", P),
    ('W06 types error/warning', "require('error:' not in text and 'warning:' not in text, 'type diagnostic')", P),
    ('W14 negative audit try/except (documented redundancy)', "    try:\n        audit_axioms(line, ['injected'])\n    except ValueError as refusal:\n        require(str(refusal).startswith('forbidden transitive axiom'), 'audit refused for the wrong reason')\n        return 'REJECTED_BY_AXIOM_AUDIT'\n    raise ValueError('negative control escaped the axiom audit: ' + control)", "    return 'REJECTED_BY_AXIOM_AUDIT'"),
    ('W27 GITHUB_REPOSITORY only', "require(os.environ.get('GITHUB_SHA') == head and os.environ.get('GITHUB_REPOSITORY') == REPO, 'host/source mismatch')", "require(os.environ.get('GITHUB_SHA') == head, 'host/source mismatch')"),
    ('W33 handoff namespace', "    require(not re.search(r'^namespace ResearchFormalCoreR1(?![A-Za-z0-9_])', code_text, re.M),\n            'handoff namespace left in place: ' + module)", "    pass"),
    ('W36 toolchain text', "require(toolchain == TOOLCHAIN, 'wrong Lean toolchain: ' + repr(toolchain))", P),
    ('W37 unpinned dependency', "require(all(isinstance(v, str) and HEX40.fullmatch(v) for v in revisions.values()), 'unpinned dependency')", P),
    ('W38 duplicate package', "require(names and len(names) == len(set(names)), 'duplicate or empty dependency list')", P),
    ('W39 scientific_effect', "require(meta['scientific_effect'] == 'NONE', 'scientific effect must be NONE')", P),
    ('W40 alignment', "require(meta['alignment_status'] == 'PENDING_INDEPENDENT_REVIEW', 'alignment status cannot self-award review')", P),
    ('W41 credit/human_review', "require(prov['organizational_independence_credit'] == 0 and prov['human_review'] == 'none', 'credit must stay zero')", P),
    ('W42 provenance identity', "require(prov['claim'] == CLAIM and prov['original_sha256'] == ORIGINAL_SHA256, 'provenance identity')", P),
    ('W43 provenance labels', "require(prov['original_status'] == 'NOT_COMPILED' and prov['namespace_mapping'] == NAMESPACE_MAPPING, 'provenance labels')", P),
    ('W44 schema keys', "    require(set(meta) == {'schema', 'scientific_effect', 'alignment_status', 'namespace', 'root_module', 'modules',\n                          'definitions', 'theorems', 'files', 'provenance'}, 'SOURCE_FILES schema keys')", "    pass"),
    ('W45 files subset', "require(isinstance(files, dict) and files and set(files) <= set(REQUIRED_FILES), 'files inventory outside the companion')", P),
    ('W46 duplicate declaration', "require(len(seen) == len(set(seen)), 'duplicate declaration name')", P),
    ('W47 per-module keys', "require(set(meta['theorems']) == set(MODULES) and set(meta['definitions']) == set(MODULES), 'per-module inventory keys')", P),
    ('W52 strict duplicate key', "require(key not in result, 'duplicate JSON key: ' + str(key))", P),
    ('W53 symlink', "require(not any(ROOT.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)), 'symlink source: ' + path)", P),
    ('W56 targets duplicate', "require(result and len(result) == len(set(result)), 'duplicate or empty target list')", P),
    ('W57 audit target list', "require(isinstance(names, list) and names and len(set(names)) == len(names), 'empty or duplicate target list')", P),
    ('W58 log_sha256 only *.log', "'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}", "'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.glob('*.log')) if p.is_file()}}"),
    ('W60 schema int', "require(type(meta['schema']) is int and meta['schema'] == 1, 'SOURCE_FILES schema version')", P),
    ('N01 finish emitted-file comparison', "            require(consumed.is_file() and consumed.read_bytes() == emitted.read_bytes(),\n                    'emitted Lean file changed across execution: ' + emitted.name)", "            pass"),
    ('N02 LEANCHECKER extra form', "LEANCHECKER = {'leanchecker TorusTopology ClosedSuperlevel TorusFiltration': 'modules',", "LEANCHECKER = {'leanchecker TorusTopology ClosedSuperlevel TorusFiltration': 'modules', 'leanchecker TorusFiltration': 'modules',"),
    ('N03 leanchecker log', "require(mode == 'fresh' or not (out / 'leanchecker.log').read_text().strip(), 'leanchecker reported diagnostics')", P),
    ('N04 leanchecker.cmd presence', "require((out / 'leanchecker.cmd').is_file(), 'leanchecker invocation record absent')", P),
    ('N05 scanner command token', "        require(found is None, 'command token outside a column-0 declaration in ' + where + ': ' + (found.group() if found else ''))", "        pass"),
    ('N06 scanner #/@[', "require('#' not in rest and '@[' not in rest, 'attribute or #-command in ' + where)", P),
    ('N07 scanner string literal', "require('\"' not in code, 'string literal in code: ' + module + ':' + str(number))", P),
    ('N08 scanner declaration outside namespace', "require(state == 'body', 'declaration outside namespace ' + NAMESPACE + ': ' + where)", P),
    ('N09 scanner code after end', "require(state != 'closed', 'code after end ' + NAMESPACE + ': ' + where)", P),
    ('N10 scanner import in namespace', "require(keyword != 'import' or state == 'prelude', 'import inside the namespace: ' + where)", P),
    ('N11 scanner whitelist', "                require(keyword is not None and PRELUDE[keyword].fullmatch(line),\n                        'unexpected column-0 line in ' + where + ': ' + line[:60])", "                keyword = keyword or 'variable'"),
    ('N12 DECLARATION name lookahead', "DECLARATION = re.compile(r'(theorem|abbrev|instance|def|noncomputable def) (' + IDENT + r')(?=$|[\\s:(\\[{⦃])')", "DECLARATION = re.compile(r'(theorem|abbrev|instance|def|noncomputable def) (' + IDENT + r')')"),
    ('N13 open lowercase', "'open': re.compile(r'open(?: [A-Z][A-Za-z0-9_.]*)+'),", "'open': re.compile(r'open(?: [A-Za-z][A-Za-z0-9_.]*)+'),"),
    ('N14 ADMISSION without sorryAx', "ADMISSION = ('sorry', 'sorryAx', 'admit',", "ADMISSION = ('sorry', 'admit',"),
    ('N15 lexer no nesting', "                if two == '/-':\n                    depth += 1\n                    i += 2\n                elif two == '-/':", "                if two == '-/':"),
    ('N16 second namespace', "require(state == 'prelude', 'namespace declaration: ' + where)", P),
    ('N17 missing end', "require(state == 'closed', 'namespace ' + NAMESPACE + ' not opened and closed exactly once: ' + module)", P),
    ('N18 receipt leanchecker field', "        'leanchecker': {'command': (out / 'leanchecker.cmd').read_text().strip(), 'mode': mode},\n", ""),
    ('N19 provenance object', "require(isinstance(prov, dict), 'provenance object')", P),
    ('N20 COMMANDS without in', "'open', 'export', 'include', 'omit', 'run_cmd', 'run_tac', 'run_elab', 'in')", "'open', 'export', 'include', 'omit', 'run_cmd', 'run_tac', 'run_elab')"),
    ('N21 variable pattern loosened to anything', "'variable': re.compile(r'variable .*')}", "'variable': re.compile(r'variable.*')}"),
    ('N22 import any module', "'import': re.compile(r'import Mathlib(?:\\.[A-Za-z0-9_]+)*'),", "'import': re.compile(r'import [A-Za-z0-9_.]+'),"),
    ('N23 inventory kind mismatch message', "require(scan['theorem'] == meta['theorems'][module], 'theorem inventory differs from source: ' + module)", "require(set(scan['theorem']) == set(meta['theorems'][module]), 'theorem inventory differs from source: ' + module)"),
]


def base_copy(dest):
    if dest.exists():
        shutil.rmtree(dest)
    side = dest / PKG
    side.mkdir(parents=True)
    (dest / '.github/workflows').mkdir(parents=True)
    for f in FILES:
        shutil.copy(W / PKG / f, side / f)
    shutil.copy(W / '.github/workflows/torus-filtration.yml', dest / '.github/workflows/torus-filtration.yml')
    shutil.copy(W / '.github/workflows/d2-moment-bridge.yml', dest / '.github/workflows/d2-moment-bridge.yml')


def run_suite(dest):
    r = subprocess.run([sys.executable, '-B', '-S', str(dest / PKG / 'test_torus_filtration.py')],
                       capture_output=True, text=True, timeout=900)
    failed = re.findall(r'^(?:FAIL|ERROR): (\w+)', r.stdout + r.stderr, re.M)
    return r.returncode, sorted(set(failed))


def main():
    results = []
    base = R / 'mut_base'
    base_copy(base)
    gen = subprocess.run([sys.executable, '-B', '-S', str(GEN), '--worktree', str(base)], capture_output=True, text=True)
    assert gen.returncode == 0, gen.stderr
    rc, failed = run_suite(base)
    print('BASELINE (unmodified copy): exit %d failed=%s' % (rc, failed), flush=True)
    assert rc == 0, 'baseline must pass'
    for label, old, new in MUTATIONS:
        dest = R / ('mut_' + label.split()[0])
        base_copy(dest)
        path = dest / PKG / 'check.py'
        text = path.read_text()
        n = text.count(old)
        if n != 1:
            results.append((label, 'MUTATION-NOT-APPLIED (occurrences=%d)' % n, []))
            print('%-55s %s' % (label, results[-1][1]), flush=True)
            continue
        path.write_text(text.replace(old, new))
        gen = subprocess.run([sys.executable, '-B', '-S', str(GEN), '--worktree', str(dest)], capture_output=True, text=True)
        if gen.returncode != 0:
            results.append((label, 'CAUGHT-BY-GENERATOR', [gen.stderr.strip()[:200]]))
            print('%-55s CAUGHT by generator: %s' % (label, gen.stderr.strip()[:120]), flush=True)
            continue
        rc, failed = run_suite(dest)
        verdict = 'CAUGHT' if rc != 0 else 'ESCAPED'
        results.append((label, verdict, failed))
        print('%-55s %s %s' % (label, verdict, ','.join(failed)), flush=True)
        shutil.rmtree(dest)
    (R / 'results.json').write_text(json.dumps(results, indent=1))
    esc = [r[0] for r in results if r[1] == 'ESCAPED']
    print('\nESCAPED (%d): %s' % (len(esc), '; '.join(esc)))


if __name__ == '__main__':
    main()
