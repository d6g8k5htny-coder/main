#!/usr/bin/env python3
"""Mutation self-test of the Layer A scanner: CBZ-0 gate G0 (stdlib only). Scientific effect: NONE.

Builds mutants of tools/cbz/layer_a_scan.py and runs the scanner's test suite
(tests/test_cbz_layer_a_scan.py) against each one through CBZ_LAYER_A_SCANNER.
A mutant is killed when the suite fails. The gate passes only when the
unmutated scanner passes the suite and every mutant is killed (threshold 100%).

Mutants: one per rule-table entry (the entry is deleted), plus the targeted
logic mutants in LOGIC below, each a literal replacement whose target text must
occur exactly once in the scanner. A target that is missing or ambiguous is an
error, so a refactor cannot quietly turn a mutant into a no-op.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / 'tools/cbz/layer_a_scan.py'
SUITE = 'test_cbz_layer_a_scan.py'

# (name, old, new): literal replacements; `old` must occur exactly once.
LOGIC = [
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

RULE_LINE = re.compile(r"^    '([^']+)': 'A[0-9][^']*',\n", re.M)


def mutants(source):
    """Return [(name, mutated_source)], raising on a missing or ambiguous target."""
    out = []
    for m in RULE_LINE.finditer(source):
        out.append(('delete-rule:' + m.group(1), source[:m.start()] + source[m.end():]))
    if len(out) < 20:
        raise RuntimeError(f'rule table not found: only {len(out)} rule lines matched')
    for name, old, new in LOGIC:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f'mutant {name}: target occurs {count} times, expected exactly once')
        mutated = source.replace(old, new)
        if mutated == source:
            raise RuntimeError(f'mutant {name} does not change the source')
        out.append((name, mutated))
    return out


def run_suite(scanner_path):
    env = {**os.environ, 'CBZ_LAYER_A_SCANNER': str(scanner_path), 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'),
                             '-p', SUITE], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=600)
    return result.returncode, result.stdout.decode('utf-8', 'replace')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--json', help='write the result to this file')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args(argv)
    source = SCANNER.read_text(encoding='utf-8')
    try:
        cases = mutants(source)
    except RuntimeError as exc:
        print('SCANNER_MUTATION_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    baseline, log = run_suite(SCANNER)
    if baseline != 0:
        print(log[-4000:])
        print('SCANNER_MUTATION_ERROR: the unmutated scanner fails its own suite', file=sys.stderr)
        return 2
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for index, (name, mutated) in enumerate(cases):
            path = Path(tmp) / f'mutant_{index:03d}.py'
            path.write_text(mutated, encoding='utf-8')
            paths[name] = path
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(run_suite, path): name for name, path in paths.items()}
            for future in concurrent.futures.as_completed(futures):
                code, _ = future.result()
                results[futures[future]] = 'killed' if code != 0 else 'SURVIVED'
    survived = sorted(name for name, state in results.items() if state != 'killed')
    report = {'schema': 1, 'scientific_effect': 'NONE', 'mutants': len(results),
              'killed': len(results) - len(survived), 'survived': survived, 'threshold': 1.0,
              'baseline': 'pass', 'verdict': 'PASS' if not survived else 'FAIL'}
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for name in survived:
        print('SURVIVED: ' + name)
    print(f"G0 scanner mutation {report['verdict']}: {report['killed']}/{report['mutants']} killed (threshold 100%)")
    return 0 if not survived else 1


if __name__ == '__main__':
    sys.exit(main())
