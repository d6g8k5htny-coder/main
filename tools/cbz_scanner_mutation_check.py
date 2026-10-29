#!/usr/bin/env python3
"""Mutation self-test for the Layer A scanner (CBZ-0 gate G0).

Each mutant below disables one behaviour that tools/cbz_layer_a_scan.py
relies on to certify Layer A.  For every mutant this script writes the
mutated scanner to a temporary file and runs tests/test_cbz_layer_a_scan.py
against it (via CBZ_SCANNER_PATH).  The gate passes only if:

  * the unmutated scanner passes the tests, and
  * every mutation applies exactly once to the current scanner source, and
  * the tests fail for every mutant ("killed").

A mutation whose target text no longer occurs exactly once is itself a
failure, so the mutant list cannot silently go stale.

Usage: cbz_scanner_mutation_check.py [--json]
Exit status: 0 if all mutants are killed, 1 otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / 'tools' / 'cbz_layer_a_scan.py'
TEST_MODULE = 'tests.test_cbz_layer_a_scan'

# (name, exact source text, replacement)
MUTANTS = [
    # lexer structure
    ('no-line-comments', "if text.startswith('--', self.i):", "if False and text.startswith('--', self.i):"),
    ('no-block-comments', "            if text.startswith('/-', self.i):\n                self._block_comment()",
     "            if False and text.startswith('/-', self.i):\n                self._block_comment()"),
    ('no-comment-nesting', "            if self.text.startswith('/-', self.i):\n                depth += 1",
     "            if False:\n                depth += 1"),
    ('no-plain-strings', "                else:\n                    self._plain_string()",
     "                else:\n                    self._advance()"),
    ('no-raw-strings', "            if self._raw_string():", "            if False:"),
    ('no-char-literals', "if ch == \"'\" and self._char_literal():", "if False:"),
    # interpolation (014K, #687 review 5701985906)
    ('no-interpolation', "                    self._interpolated_string()", "                    self._plain_string()"),
    ('no-plain-brace-scan', "        self._scan_brace_segments(content_start, content_end, start_line)", "        pass"),
    ('no-hole-nesting', "            if until_brace and ch == '{':\n                depth += 1",
     "            if False:\n                depth += 1"),
    # fail closed
    ('fail-open-block-comment', "                raise LexError(f'unterminated block comment starting at line {start_line}')",
     "                return"),
    ('fail-open-raw-string', "            raise LexError(f'unterminated raw string starting at line {self.line}')",
     "            return False"),
    ('fail-open-plain-string', "                raise LexError(f'unterminated string literal starting at line {start_line}')",
     "                return"),
    ('fail-open-interpolated', "                raise LexError(f'unterminated interpolated string starting at line {start_line}')",
     "                return"),
    ('lex-error-not-recorded', "except (OSError, UnicodeDecodeError, LexError) as exc:",
     "except (OSError, UnicodeDecodeError) as exc:"),
    # tokens
    ('drop-sorry', "LAYER_A_TOKENS = ('sorry', ", "LAYER_A_TOKENS = ("),
    ('drop-admit', "'admit', 'sorryAx'", "'sorryAx'"),
    ('drop-sorryAx', "'sorryAx', 'axiom'", "'axiom'"),
    ('drop-axiom', "'axiom', 'native_decide'", "'native_decide'"),
    ('drop-native-decide', "'native_decide', '+native'", "'+native'"),
    ('drop-plus-native', "                        name = '+native'", "                        pass"),
    ('drop-native-config', "                        name = 'native:='", "                        pass"),
    ('native-config-anywhere', "\n                          and self._prev_code_char(start) in ('{', ','))", ")"),
    ('drop-ofReduceBool', "'ofReduceBool', 'ofReduceNat'", "'ofReduceNat'"),
    ('drop-ofReduceNat', "'ofReduceNat', 'trustCompiler'", "'trustCompiler'"),
    ('drop-trustCompiler', "'trustCompiler',\n", "\n"),
    ('drop-skipKernelTC', "'skipKernelTC', 'addDeclWithoutChecking'", "'addDeclWithoutChecking'"),
    ('drop-addDeclWithoutChecking', ", 'addDeclWithoutChecking')", ")"),
    ('substring-match', "if tok in LAYER_A_TOKENS]", "if any(t in tok for t in LAYER_A_TOKENS)]"),
    ('no-dotted-split', "for part in text[start:self.i].split('.'):", "for part in [text[start:self.i]]:"),
    # tree walking and the command line
    ('no-recursion', "dirnames[:] = sorted(d for d in dirnames if d not in ('.git', '.lake'))", "dirnames[:] = []"),
    ('scan-lake-dir', "if d not in ('.git', '.lake'))", "if d not in ('.git',))"),
    ('silent-error-skip', "errors.append({'path': path, 'error': f'{type(exc).__name__}: {exc}'})", "pass"),
    ('always-exit-zero', "return 0 if not blocking and not result['errors'] else 1", "return 0"),
    ('allow-list-covers-everything', "if not (h['token'] == 'axiom'\n                        and ",
     "if not (True\n                        and "),
]


def run_tests(scanner_path: Path) -> int:
    env = dict(os.environ, CBZ_SCANNER_PATH=str(scanner_path))
    proc = subprocess.run([sys.executable, '-B', '-S', '-m', 'unittest', TEST_MODULE],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    return proc.returncode


def main(argv) -> int:
    source = SCANNER.read_text(encoding='utf-8')
    results = {'baseline_passes': run_tests(SCANNER) == 0, 'mutants': []}
    with tempfile.TemporaryDirectory() as tmp:
        for name, old, new in MUTANTS:
            count = source.count(old)
            entry = {'name': name, 'applies_exactly_once': count == 1}
            if count == 1:
                mutant = Path(tmp, f'mutant_{name}.py')
                mutant.write_text(source.replace(old, new), encoding='utf-8')
                entry['killed'] = run_tests(mutant) != 0
            else:
                entry['killed'] = False
            results['mutants'].append(entry)
    survivors = [m['name'] for m in results['mutants'] if not (m['applies_exactly_once'] and m['killed'])]
    results['survivors'] = survivors
    results['passed'] = results['baseline_passes'] and not survivors
    if '--json' in argv:
        print(json.dumps(results, indent=1, sort_keys=True))
    else:
        for m in results['mutants']:
            state = 'KILLED' if m['killed'] else ('STALE' if not m['applies_exactly_once'] else 'SURVIVED')
            print(f"{state:9} {m['name']}")
        print(f"baseline_passes={results['baseline_passes']} mutants={len(MUTANTS)} "
              f"survivors={len(survivors)} passed={results['passed']}")
    return 0 if results['passed'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
