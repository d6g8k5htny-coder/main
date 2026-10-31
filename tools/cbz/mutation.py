#!/usr/bin/env python3
"""Mutation self-test of every certifying CBZ-0 tool: gate G0 (stdlib only). Scientific effect: NONE.

Targets (TARGETS): the Layer A scanner (G1), the empirical fuzzing harness (G6)
and the federation drift crawler (G7). For each target the engine builds
mutants of the tool's source and runs that tool's test suite against each one,
through the target's environment variable. A mutant is killed when the suite
fails. The gate passes only when every unmutated tool passes its own suite and
every mutant of every target is killed (threshold 100%).

Mutants are literal replacements whose target text must occur exactly once in
the tool, plus, for the scanner, one deletion per rule-table entry. A missing or
ambiguous target is an error, so a refactor cannot quietly turn a mutant into a
no-op. Each mutant runs against a temporary copy of the whole tools/cbz
directory, so sibling imports resolve and only the target file differs.
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'tools/cbz'

SCANNER_LOGIC = [
    ('no-line-comments', "if c == '-' and t.startswith('--', i):", "if False and c == '-' and t.startswith('--', i):"),
    ('no-block-comments', "elif c == '/' and t.startswith('/-', i):", "elif False and c == '/' and t.startswith('/-', i):"),
    ('flat-block-comments', "            if t.startswith('/-', i):\n                depth += 1",
     "            if t.startswith('/-', i) and depth == 0:\n                depth += 1"),
    ('silent-unterminated-comment', "self.errors.append(('unterminated block comment', start))", "pass"),
    ('unmasked-plain-strings', "        end = i + 1\n        self.blank(start, end)", "        end = i + 1"),
    ('no-plain-brace-scan', "                self.restore(j + 1, close)", "                pass"),
    ('plain-escape-one-char', "            if c == '\\\\':\n                i += 2\n            elif c == '\"':\n                break",
     "            if c == '\\\\':\n                i += 1\n            elif c == '\"':\n                break"),
    ('silent-unterminated-string', "        if i >= n:\n            self.errors.append(('unterminated string', start))",
     "        if i >= n:\n            pass"),
    ('no-interpolation', "        if self.interpolated(start):", "        if False and self.interpolated(start):"),
    ('no-bang-prefix', "        if i >= 2 and t[i - 1] == '!':", "        if False and i >= 2 and t[i - 1] == '!':"),
    ('no-macro-interpolation', "        return INTERPOLATING_MACRO.search(t, line_start, i) is not None", "        return False"),
    ('unmasked-interpolated-text', "                elif c == '\"':\n                    self.blank(text_from, i + 1)\n                    return i + 1",
     "                elif c == '\"':\n                    return i + 1"),
    ('no-raw-strings', "        if k >= 0 and t[k] == 'r' and (k == 0 or not is_id_rest(t[k - 1])):", "        if False:"),
    ('no-char-literals', "                m = CHAR_LITERAL.match(t, i)", "                m = None"),
    ('char-literal-after-identifier', "            elif c == \"'\" and not (i > 0 and is_id_rest(t[i - 1])):",
     "            elif c == \"'\":"),
    ('lambda-is-letter', "(0x3b1 <= v <= 0x3c9 and v != 0x3bb)", "(0x3b1 <= v <= 0x3c9)"),
    ('pi-sigma-are-letters', "(0x391 <= v <= 0x3a9 and v != 0x3a0 and v != 0x3a3)", "(0x391 <= v <= 0x3a9)"),
    ('digits-absorb-names', "            i += 1\n    return out, errors",
     "            i += 1\n            while i < n and masked[i].isalnum():\n                i += 1\n    return out, errors"),
    ('no-exit-command', "COMMANDS = {'#exit': 'A1.exit'}", "COMMANDS = {}"),
    ('silent-unterminated-quoted-name', "                        errors.append(('unterminated quoted identifier', i))", "                        pass"),
    ('always-exit-zero', "    return 0 if report['verdict'] == 'PASS' else 1", "    return 0"),
    ('violations-pass', "    clean = not violations and not report['errors']", "    clean = not report['errors']"),
    ('no-count-check', "        if actual != row['count']:", "        if False:"),
    ('no-path-check', "        if row['path'] not in texts:", "        if False:"),
    ('no-duplicate-check', "        if key in exempt_keys:", "        if False:"),
    ('no-isolation-check', "                    if module in candidates:", "                    if False:"),
    ('no-submodule-error', "        if kind == b'commit':", "        if False:"),
    ('no-symlink-error', "            if mode == b'120000':", "            if False:"),
    ('lossy-decode', "                text = raw.decode('utf-8')", "                text = raw.decode('utf-8', 'replace')"),
    ('short-reasons-accepted', "len(row['reason'].strip()) < 20", "len(row['reason'].strip()) < 0"),
    ('working-tree-read', "    commit = git(path, 'rev-parse', '--verify', rev + '^{commit}').decode().strip()",
     "    commit = git(path, 'rev-parse', '--verify', rev + '^{commit}').decode().strip()\n"
     "    return commit, {p: (Path(path) / p).read_bytes() for p in git(path, 'ls-files', '*.lean').decode().split()}, []"),
]

FUZZ_LOGIC = [
    ('nat-subtraction-not-truncated', "        return max(a - b, 0)", "        return a - b"),
    ('int-division-truncating', "        r = a % abs(b)\n        return (a - r) // b", "        return int(a / b)"),
    ('division-by-zero-raises', "    if b == 0:\n        return to_type(0, type_name)", "    if False:\n        return to_type(0, type_name)"),
    ('int-modulo-python-sign', "    return a % abs(b) if type_name == 'int' else a % b", "    return a % b"),
    ('power-left-associative', "    '^': (75, 76, 75),", "    '^': (75, 75, 76),"),
    ('subtraction-right-associative', "    '+': (65, 65, 66), '-': (65, 65, 66),", "    '+': (65, 65, 66), '-': (65, 66, 65),"),
    ('negation-binds-tightly', "            operand, _ = self.expr(75)\n            return ('neg', operand), 75",
     "            operand = self.atom_expr()\n            return ('neg', operand), 75"),
    ('no-maximal-type', "    return max(known, key=RANK.get) if known else 'nat'", "    return 'nat'"),
    ('edge-value-dropped', "2 ** 32 + 1,\n             2 ** 32 + 7,", "2 ** 32 + 1,"),
    ('vacuity-never-flagged', "            if hypothesis_hits == 0:\n                record['status'] = 'VACUITY_SUSPECT'",
     "            if False:\n                record['status'] = 'VACUITY_SUSPECT'"),
    ('oversized-sample-counts-as-pass', "            except TooLarge:\n                skipped += 1\n                continue",
     "            except TooLarge:\n                skipped += 1\n                holds = True"),
    ('no-witness-repair', "                    repaired = repair(env, hypotheses, infinite, ctx, rng) if infinite else None",
     "                    repaired = None"),
    ('bounded-range-off-by-one', "    return 'nat', list(range(top))", "    return 'nat', list(range(top - 1))"),
    ('exists-never-witnessed', "        if kind == 'exists' and result:\n            return True",
     "        if False:\n            return True"),
    ('index-range-off-by-one', "        return 'nat', list(range(n))", "        return 'nat', list(range(n + 1))"),
    ('wrong-factorial', "    'Nat.factorial': (('nat',), 'nat', _factorial),", "    'Nat.factorial': (('nat',), 'nat', lambda n: n),"),
    ('open-interval-closed', "'Ioo': ('<', '<')", "'Ioo': ('≤', '≤')"),
    ('acknowledgement-ignores-statement', "        if record['status'] == 'VACUITY_SUSPECT' and key in acknowledged:",
     "        if record['status'] == 'VACUITY_SUSPECT' and any(k[:2] == key[:2] for k in acknowledged):"),
    ('counterexample-not-blocking', "    failing = [r for r in report['records'] if r['status'] in ('FAIL', 'ERROR')",
     "    failing = [r for r in report['records'] if r['status'] in ('ERROR',)"),
    ('decimal-via-float', "    return Fraction(text)", "    return Fraction(float(text))"),
    ('internal-fault-as-na', "        record = {'status': 'ERROR', 'reason': f'{type(exc).__name__}: {exc}'}",
     "        record = {'status': 'NA', 'reason': f'{type(exc).__name__}: {exc}'}"),
    ('negation-as-hypothesis', "        if node[0] == 'bin' and node[1] == '→' and node[3] != ('var', 'False'):",
     "        if node[0] == 'bin' and node[1] == '→':"),
    ('type-binders-as-hypotheses', "            if is_type_expression(node):", "            if False:"),
    ('raw-identifier-arguments', "                args.append(self.atom_expr())", "                args.append(self.atom())"),
]

CRAWL_LOGIC = [
    ('missing-object-not-blocking',
     "            report['findings'].append({'check': 'P1', 'blocking': True, **where, 'error': 'pinned object does not exist'})",
     "            report['findings'].append({'check': 'P1', 'blocking': False, **where, 'error': 'pinned object does not exist'})"),
    ('every-commit-on-default', "                if fed.on_default_branch(target, edge['commit']):", "                if True:"),
    ('archive-ignored', "                    resolved = ('archived', tags) if tags else 'off-default'",
     "                    resolved = 'off-default'"),
    ('no-blob-check', "            if actual is None or actual[2] != edge['blob']:", "            if False:"),
    ('no-test-data-exclusion', "            test_data = bool(TEST_DATA_DIRS & set(p.split('/')[:-1]))", "            test_data = False"),
    ('no-mathlib-skew', "    if len(mathlib) > 1:", "    if False:"),
    ('no-toolchain-skew', "    if len(toolchains) > 1:", "    if False:"),
    ('lag-never-blocks', "'blocking': bool(require_current_root_pins and lag)", "'blocking': False"),
    ('no-snapshot-provenance', "                if snapshot_of(fed, history, a, b) or snapshot_of(fed, history, b, a):",
     "                if False:"),
    ('everything-a-snapshot', "                if snapshot_of(fed, history, a, b) or snapshot_of(fed, history, b, a):",
     "                if True:"),
    ('allowlist-by-name-only', "                if pair in allowlist:", "                if any(x[0] == name for x in allowlist):"),
    ('no-transitive-discovery', "                if target['key'] not in seen and target['key'] not in queue:",
     "                if edge['root'] and target['key'] not in seen and target['key'] not in queue:"),
    ('no-json-records', "    stack = [value]", "    stack = []"),
    ('gitlink-without-url-allowed',
     "                    report['findings'].append({'check': 'P1', 'blocking': True, 'repository': info['url'],",
     "                    report['findings'].append({'check': 'P1', 'blocking': False, 'repository': info['url'],"),
    ('bare-names-compared', "if d['qualified'] and '.' in d['qualified'] and d['kind'] not in ('instance', 'example'):",
     "if d['qualified'] and d['kind'] not in ('instance', 'example'):"),
    ('unreachable-root-allowed', "            if key == root['key']:\n                report['findings'].append({'check': 'P1', 'blocking': True,",
     "            if key == root['key']:\n                report['findings'].append({'check': 'P1', 'blocking': False,"),
    ('verdict-always-pass', "    report['verdict'] = 'PASS' if report['summary']['blocking'] == 0 else 'FAIL'",
     "    report['verdict'] = 'PASS'"),
    ('unreachable-repository-allowed',
     "            report['findings'].append({'check': 'P1', 'blocking': True, **where, 'error': target['error']})",
     "            report['findings'].append({'check': 'P1', 'blocking': False, **where, 'error': target['error']})"),
]

RULE_LINE = re.compile(r"^    '([^']+)': 'A[0-9][^']*',\n", re.M)


def rule_deletions(source):
    out = [('delete-rule:' + m.group(1), source[:m.start()] + source[m.end():]) for m in RULE_LINE.finditer(source)]
    if len(out) < 20:
        raise RuntimeError(f'rule table not found: only {len(out)} rule lines matched')
    return out


TARGETS = {
    'layer_a_scan': {'file': 'layer_a_scan.py', 'suite': 'test_cbz_layer_a_scan.py', 'env': 'CBZ_LAYER_A_SCANNER',
                     'logic': SCANNER_LOGIC, 'extra': rule_deletions},
    'fuzz_bounds': {'file': 'fuzz_bounds.py', 'suite': 'test_cbz_fuzz_bounds.py', 'env': 'CBZ_FUZZ_BOUNDS',
                    'logic': FUZZ_LOGIC, 'extra': None},
    'drift_crawler': {'file': 'drift_crawler.py', 'suite': 'test_cbz_drift_crawler.py', 'env': 'CBZ_DRIFT_CRAWLER',
                      'logic': CRAWL_LOGIC, 'extra': None},
}


def mutants(target, source):
    """Return [(name, mutated_source)] for one target; raise on a missing or ambiguous text."""
    spec = TARGETS[target]
    out = spec['extra'](source) if spec['extra'] else []
    for name, old, new in spec['logic']:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f'{target} mutant {name}: target occurs {count} times, expected exactly once')
        mutated = source.replace(old, new)
        if mutated == source:
            raise RuntimeError(f'{target} mutant {name} does not change the source')
        out.append((name, mutated))
    names = [n for n, _ in out]
    if len(names) != len(set(names)):
        raise RuntimeError(f'{target}: duplicate mutant names')
    return out


def stage(tmp, target, text):
    """Copy tools/cbz into tmp and write `text` as the target file; return the target path."""
    directory = Path(tmp)
    for item in TOOLS.glob('*.py'):
        shutil.copy2(item, directory / item.name)
    for item in TOOLS.glob('*.json'):
        shutil.copy2(item, directory / item.name)
    path = directory / TARGETS[target]['file']
    path.write_text(text, encoding='utf-8')
    return path


def run_suite(target, path):
    spec = TARGETS[target]
    env = {**os.environ, spec['env']: str(path), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'),
                             '-p', spec['suite']], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=900)
    return result.returncode, result.stdout.decode('utf-8', 'replace')


def run_target(target, jobs):
    source = (TOOLS / TARGETS[target]['file']).read_text(encoding='utf-8')
    cases = mutants(target, source)
    with tempfile.TemporaryDirectory() as tmp:
        code, log = run_suite(target, stage(tmp, target, source))
    if code != 0:
        raise RuntimeError(f'the unmutated {target} fails its own suite:\n' + log[-3000:])

    def one(case):
        name, mutated = case
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = run_suite(target, stage(tmp, target, mutated))
        return name, 'killed' if code != 0 else 'SURVIVED'
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = dict(pool.map(one, cases))
    survived = sorted(n for n, state in results.items() if state != 'killed')
    return {'mutants': len(results), 'killed': len(results) - len(survived), 'survived': survived}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--target', action='append', choices=sorted(TARGETS), help='default: every target')
    parser.add_argument('--json', help='write the result to this file')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args(argv)
    report = {'schema': 1, 'scientific_effect': 'NONE', 'threshold': 1.0, 'targets': {}}
    try:
        for target in args.target or sorted(TARGETS):
            report['targets'][target] = run_target(target, args.jobs)
    except RuntimeError as exc:
        print('MUTATION_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    survived = [f'{t}:{n}' for t, r in report['targets'].items() for n in r['survived']]
    report['verdict'] = 'PASS' if not survived else 'FAIL'
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for name in survived:
        print('SURVIVED: ' + name)
    for target, r in report['targets'].items():
        print(f"{target}: {r['killed']}/{r['mutants']} killed")
    print(f"G0 mutation {report['verdict']} (threshold 100%)")
    return 0 if not survived else 1


if __name__ == '__main__':
    sys.exit(main())
