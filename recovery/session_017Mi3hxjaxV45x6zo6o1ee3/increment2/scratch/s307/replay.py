#!/usr/bin/env python3
"""Verify the stored QS A3-A3.6 records and replay their published author controls (standard library only).

Every file in this packet except replay.py, README.md and SOURCES.json is one of two things:
  - an exact copy of a public comment body on main#229 (a note, claim, controls comment, review, readback,
    erratum, successor text or author response);
  - a control script or its stdout, extracted verbatim from a stored controls comment by the extraction rule that
    comment states (the rule of main#229 comment 5971055189).
SOURCES.json pins each file by byte count and SHA-256 and lists each replay. This script performs four checks.

  1. The packet tree equals SOURCES.json's file list, with no symlinks, and every stored file has its pinned
     identity.
  2. Every extracted script and stdout equals the fenced payload inside its stored controls comment, under the
     extraction rule.
  3. Every control script runs under the current interpreter's -O/-S flags and reproduces its published stdout byte
     for byte (none prints an interpreter version). Every listed mutant invocation exits 1, and every listed
     invalid invocation exits 2.
  4. Negative controls must be rejected:
     - a one-byte change to a stored note (A3_6/PROOF.md);
     - a control script with one changed constant (A3.6's hard-gap integral coefficient 5/4 replaced by 1/4).

Reproducibility only. The analytic notes and their reads carry the mathematics. These finite checks support
algebra, as the reads say. Nothing here reads or grades a proof.

Usage: python3 -B -S frontiers/qs_d3_soft_layer_chain_20261005/replay.py   (exit 0 iff every check passes)
"""
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
OWN = {'README.md', 'SOURCES.json', 'replay.py'}


def identity(data):
    return len(data), hashlib.sha256(data).hexdigest()


def interpreter_flags():
    flags = ['-B']
    if sys.flags.optimize:
        flags.append('-O')
    if sys.flags.no_site:
        flags.append('-S')
    return flags


def tree_failures(root, manifest):
    failures = []
    listed = {}
    for obj in manifest['objects']:
        for f in obj['stored']:
            listed[f['path']] = (f['bytes'], f['sha256'])
    present = set()
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append('symlink: ' + rel)
        elif path.is_file():
            present.add(rel)
    for rel in sorted(present ^ (set(listed) | OWN)):
        failures.append(('unlisted file: ' if rel in present else 'missing file: ') + rel)
    for rel, ident in sorted(listed.items()):
        if rel in present and identity((root / rel).read_bytes()) != ident:
            failures.append('identity mismatch: ' + rel)
    return failures


def extraction_failures(root, manifest):
    """Re-extract every script and stdout from its stored controls comment and compare with the stored payloads."""
    failures = []
    for obj in manifest['objects']:
        extracted = [f for f in obj['stored'] if 'extracted_from' in f]
        for script in [f for f in extracted if f['path'].endswith('.py')]:
            stdout = [f for f in extracted if f['extracted_from'] == script['extracted_from']
                      and f['heading'] == script['heading'] and f['path'].endswith('.json')]
            if len(stdout) != 1:
                failures.append('%s: no single stdout record for %s' % (obj['tag'], script['path']))
                continue
            stdout = stdout[0]
            body = (root / script['extracted_from']).read_text(encoding='utf-8')
            if body.count(script['heading']) < 1:
                failures.append('%s: heading %r not found in %s' % (obj['tag'], script['heading'], script['extracted_from']))
                continue
            section = body[body.index(script['heading']):]
            m1 = re.search(r'```python\n(.*?)\n```\n', section, re.S)
            m2 = re.search(r'```json\n(.*?)\n```\n', section[m1.end():], re.S) if m1 else None
            if not (m1 and m2):
                failures.append('%s: fences not found after %r' % (obj['tag'], script['heading']))
                continue
            if (m1.group(1) + '\n').encode('utf-8') != (root / script['path']).read_bytes():
                failures.append('%s: %s differs from the payload in %s' % (obj['tag'], script['path'], script['extracted_from']))
            if (m2.group(1) + '\n').encode('utf-8') != (root / stdout['path']).read_bytes():
                failures.append('%s: %s differs from the payload in %s' % (obj['tag'], stdout['path'], stdout['extracted_from']))
    return failures


def run(script, *args):
    proc = subprocess.run([sys.executable, *interpreter_flags(), str(script), *args], capture_output=True,
                          timeout=1800, cwd=str(script.parent))
    return proc.returncode, proc.stdout


def replay_failures(root, manifest):
    failures = []
    for obj in manifest['objects']:
        for replay in obj['replays']:
            label = '%s %s' % (obj['tag'], replay['name'])
            script = root / replay['script']
            code, out = run(script, *replay['args'])
            if code != 0:
                failures.append('%s: exit code %s' % (label, code))
            if out != (root / replay['stdout']).read_bytes():
                failures.append('%s: stdout differs from the published bytes' % label)
            for mutant in replay['mutants']:
                code, _ = run(script, *mutant)
                if code != 1:
                    failures.append('%s: mutant %s exited %s, expected 1' % (label, ' '.join(mutant), code))
            for invalid in replay['invalid']:
                code, _ = run(script, *invalid)
                if code != 2:
                    failures.append('%s: invalid invocation %r exited %s, expected 2' % (label, ' '.join(invalid), code))
    return failures


def negative_control_failures(root, manifest):
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        copy = pathlib.Path(tmp) / 'packet'
        shutil.copytree(root, copy, symlinks=True)
        target = copy / 'A3_6' / 'PROOF.md'
        data = bytearray(target.read_bytes())
        data[100] ^= 0x01
        target.write_bytes(bytes(data))
        if 'identity mismatch: A3_6/PROOF.md' not in tree_failures(copy, manifest):
            failures.append('negative control: a one-byte change to A3_6/PROOF.md was not detected')
        script = copy / 'A3_6' / 'author_controls.py'
        text = script.read_text(encoding='utf-8')
        old = 'coef = F(1, 4) if MUT == "M5" else F(5, 4)'
        if text.count(old) != 1:
            failures.append('negative control: the hard-gap coefficient line was not found exactly once')
        else:
            script.write_text(text.replace(old, 'coef = F(1, 4) if MUT == "M5" else F(1, 4)'), encoding='utf-8')
            code, out = run(script)
            if code == 0 or out == (root / 'A3_6' / 'author_controls_stdout.json').read_bytes():
                failures.append('negative control: a checker with 5/4 replaced by 1/4 was not rejected')
    return failures


def main():
    manifest = json.loads((HERE / 'SOURCES.json').read_text(encoding='utf-8'))
    failures = (tree_failures(HERE, manifest) + extraction_failures(HERE, manifest) + replay_failures(HERE, manifest)
                + negative_control_failures(HERE, manifest))
    n_scripts = sum(len(o['replays']) for o in manifest['objects'])
    n_mut = sum(len(r['mutants']) for o in manifest['objects'] for r in o['replays'])
    n_inv = sum(len(r['invalid']) for o in manifest['objects'] for r in o['replays'])
    n_files = sum(len(o['stored']) for o in manifest['objects'])
    mode = ' '.join(interpreter_flags())
    if failures:
        for f in failures:
            print('FAIL ' + f)
        print('QS d = 3 chain (A3-A3.6) incorporation replay (%s): %d failure(s)' % (mode, len(failures)))
        return 1
    print('QS d = 3 chain (A3-A3.6) incorporation replay (%s): %d stored identities, %d extractions, %d checker stdouts, '
          '%d mutants, %d invalid invocations and 2 negative controls PASS' % (mode, n_files, 2 * n_scripts, n_scripts, n_mut, n_inv))
    return 0


if __name__ == '__main__':
    sys.exit(main())
