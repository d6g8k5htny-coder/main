#!/usr/bin/env python3
"""CBZ-0 finite static auditor (Python standard library only).

Usage:
  python -B tools/cbz_audit.py selftest
  python -B tools/cbz_audit.py scan --root . [--registry FILE] [--quarantine FILE]

Both commands print JSON. Scan exits 1 on findings, 2 on incomplete coverage or
missing semantic work. It NEVER certifies overall Layer A or mathematical
soundness. static_result=PASS means only the declared finite static checks pass.

Registry schema: {"version":1,"files":[{"path":"bounds.py","sha256":"...",
"role":"certification"}]}. Roles: certification, noncertifying, proof-source,
other. Exact bytes must match; registry completeness is a separate semantic duty.
Certification detection currently supports Python float AST nodes/calls and Lean
Float tokens only. Aliasing, FFI, dependencies and dynamic code need semantic work.

Quarantine schema: {"version":1,"entries":[{"path":"Q.lean","sha256":"...",
"statement":"...","reason":"...","attempts":["..."],"owner":"none"}]}.
A valid entry annotates findings; it NEVER suppresses them or proves isolation.

Inventory is the entire Git tracked working tree, including missing files, stage
conflicts, symlinks and submodules. Archives are explicitly unsupported coverage
gaps; their contents are not claimed scanned. Untracked proof/archive candidates
are reported as gaps. No archive extraction or third-party execution occurs.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata

VERSION = 'cbz-static-1'
PROOF_SUFFIXES = {'.lean', '.olean', '.ilean', '.v', '.agda', '.thy', '.tex', '.typ', '.ipynb', '.smt2', '.pvs', '.dfy', '.fst', '.fsti', '.idr', '.lagda'}
ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tgz', '.tar.gz', '.tar.bz2', '.tar.xz', '.gz', '.bz2', '.xz', '.7z', '.rar', '.zst')
LEAN_FORBIDDEN = {'sorry', 'admit', 'axiom', 'native_decide'}
LEAN_TRUST = {'sorryAx', 'unsafe', 'implemented_by', 'extern', 'ofReduceBool', 'ofReduceNat', 'trustCompiler', 'skipKernelTC'}
REQUIRED_SEMANTICS = ['kernel_and_declared_axioms', 'status_index_and_nonvacuity', 'statement_alignment_and_unweakened_bounds', 'test_preservation_and_regression', 'certification_scope', 'dependency_and_dynamic_syntax_review']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def is_archive(path, raw=b''):
    return (path.lower().endswith(ARCHIVE_SUFFIXES) or raw.startswith((b'PK\x03\x04', b'PK\x05\x06', b'\x1f\x8b', b'BZh', b'\xfd7zXZ\x00', b'7z\xbc\xaf\x27\x1c', b'Rar!', b'\x28\xb5\x2f\xfd')) or raw[257:262] == b'ustar')


def proof_candidate(path):
    return PurePosixPath(path).suffix.lower() in PROOF_SUFFIXES


def ident_char(c):
    return c in "_'" or c.isalnum() or unicodedata.category(c).startswith(('L', 'M'))


def lean_tokens(source):
    """Return (kind, value, line) tokens and conservative lexical gaps.

    Nested comments, ordinary/raw strings and quoted identifiers are recognized.
    Interpolated strings are gaps because embedded Lean expressions need parsing.
    This is not a Lean parser or a semantic dependency/axiom checker.
    """
    tokens, gaps = [], []
    i, line, size = 0, 1, len(source)
    while i < size:
        if source.startswith('--', i):
            end = source.find('\n', i)
            i = size if end < 0 else end
        elif source.startswith('/-', i):
            start, depth = line, 1
            i += 2
            while i < size and depth:
                if source.startswith('/-', i):
                    depth += 1
                    i += 2
                elif source.startswith('-/', i):
                    depth -= 1
                    i += 2
                else:
                    line += source[i] == '\n'
                    i += 1
            if depth:
                gaps.append((start, 'unterminated nested comment'))
        elif source[i] == '«':
            start = line
            end = source.find('»', i + 1)
            if end < 0:
                gaps.append((start, 'unterminated quoted identifier'))
                break
            tokens.append(('quoted', source[i + 1:end], line))
            line += source[i:end + 1].count('\n')
            i = end + 1
        elif source.startswith('s!"', i):
            gaps.append((line, 'interpolated string requires Lean parser'))
            i += 2
        elif source[i] == 'r' and (match := re.match(r'r(#+)?"', source[i:])):
            start = line
            marker = '"' + (match.group(1) or '')
            end = source.find(marker, i + len(match.group()))
            if end < 0:
                gaps.append((start, 'unterminated raw string'))
                break
            line += source[i:end + len(marker)].count('\n')
            i = end + len(marker)
        elif source[i] == '"':
            start = line
            i += 1
            closed = False
            while i < size:
                if source[i] == '\\':
                    line += i + 1 < size and source[i + 1] == '\n'
                    i += 2
                elif source[i] == '"':
                    i += 1
                    closed = True
                    break
                else:
                    line += source[i] == '\n'
                    i += 1
            if not closed:
                gaps.append((start, 'unterminated string'))
        elif source[i] == "'" and (match := re.match(r"'(?:[^'\\\n]|\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.))'", source[i:])):
            i += len(match.group())
        elif ident_char(source[i]):
            start = i
            while i < size and ident_char(source[i]):
                i += 1
            tokens.append(('identifier', source[start:i], line))
        else:
            line += source[i] == '\n'
            i += 1
    return tokens, gaps


def lean_findings(tokens):
    return [('lean.' + value, line, 'lexical trust/placeholder token requires review')
            for kind, value, line in tokens
            if (kind == 'identifier' and value in LEAN_FORBIDDEN) or value in LEAN_TRUST]


def certification_findings(path, source, tokens):
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == '.lean':
        return ([('certification.float', line, 'Float token in explicitly classified certifier')
                 for _, value, line in tokens if value in {'Float', 'Float32', 'Float64'}], [])
    if suffix != '.py':
        return [], [('certification.unsupported', 0, 'certifier language is outside finite detector scope')]
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError) as error:
        return [], [('certification.parse', 0, str(error))]
    findings = []
    for node in ast.walk(tree):
        literal = isinstance(node, ast.Constant) and type(node.value) in {float, complex}
        call = isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == 'float') or (isinstance(node.func, ast.Attribute) and node.func.attr in {'float', 'float16', 'float32', 'float64'}))
        if literal or call:
            findings.append(('certification.float', node.lineno, 'floating point in explicitly classified certifier'))
    return findings, []


def validate_manifest(value, kind, files):
    if value is None:
        return {}
    key = 'files' if kind == 'registry' else 'entries'
    if type(value) is not dict or set(value) != {'version', key} or type(value['version']) is not int or value['version'] != 1 or type(value[key]) is not list:
        raise ValueError('invalid manifest envelope')
    entries = {}
    for row in value[key]:
        fields = {'path', 'sha256', 'role'} if kind == 'registry' else {'path', 'sha256', 'statement', 'reason', 'attempts', 'owner'}
        if type(row) is not dict or set(row) != fields:
            raise ValueError('invalid manifest row fields')
        path = row['path']
        if type(path) is not str or not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or path in entries or path not in files:
            raise ValueError('unknown, unsafe or duplicate manifest path')
        if type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', row['sha256']) or row['sha256'] != files[path].get('sha256'):
            raise ValueError('manifest bytes do not match tracked working file')
        if kind == 'registry':
            if row['role'] not in {'certification', 'noncertifying', 'proof-source', 'other'}:
                raise ValueError('invalid classification role')
        elif row['owner'] != 'none' or any(type(row[k]) is not str or not row[k].strip() for k in ('statement', 'reason')) or type(row['attempts']) is not list or not row['attempts'] or any(type(a) is not str or not a.strip() for a in row['attempts']):
            raise ValueError('quarantine requires statement, reason, attempts and owner:none')
        entries[path] = row
    return entries


def scan(root, registry=None, quarantine=None):
    root = Path(root).resolve()
    result = {'schema_version': 1, 'tool': VERSION, 'scanner_sha256': digest(Path(__file__).read_bytes()), 'root': str(root), 'scope': 'Git tracked working-tree bytes, not commit or kernel semantics', 'files': [], 'tracked_count': 0, 'read_count': 0, 'proof_candidates': [], 'untracked': [], 'findings': [], 'coverage_gaps': [], 'quarantine': [], 'semantic_receipts_required': list(REQUIRED_SEMANTICS), 'mathematical_soundness_certified': False}
    def gap(code, path, detail, line=0):
        result['coverage_gaps'].append({'code': code, 'path': path, 'line': line, 'detail': detail})
    raw_files, by_path = {}, {}
    try:
        top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], check=True, capture_output=True)
        root = Path(os.fsdecode(top.stdout.removesuffix(b'\n'))).resolve()
        result['root'] = str(root)
        proc = subprocess.run(['git', '-C', str(root), 'ls-files', '--stage', '-z'], check=True, capture_output=True)
        records = proc.stdout.split(b'\0')
        for record in records:
            if not record:
                continue
            metadata, raw_path = record.split(b'\t', 1)
            mode, oid, stage = metadata.decode('ascii').split()
            path = raw_path.decode('utf-8', errors='surrogateescape')
            if path in by_path:
                gap('git.conflict', path, 'multiple index stages')
                continue
            entry = {'path': path, 'index_mode': mode, 'index_object': oid, 'index_stage': stage, 'read': False}
            by_path[path] = entry
            result['files'].append(entry)
            if stage != '0':
                gap('git.conflict', path, 'unmerged index entry')
            if mode == '160000':
                gap('file.submodule', path, 'submodule requires independent complete scan')
                continue
            if mode == '120000' or any((root / parent).is_symlink() for parent in (PurePosixPath(path), *PurePosixPath(path).parents) if str(parent) != '.'):
                gap('file.symlink', path, 'symlink not followed')
                continue
            try:
                if not stat.S_ISREG((root / path).stat().st_mode):
                    gap('file.non_regular', path, 'tracked file replaced by non-regular file')
                    continue
                raw = (root / path).read_bytes()
            except OSError as error:
                gap('file.unreadable', path, type(error).__name__)
                continue
            entry.update(read=True, sha256=digest(raw), bytes=len(raw))
            raw_files[path] = raw
        proc = subprocess.run(['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard', '-z'], check=True, capture_output=True)
        result['untracked'] = [p.decode('utf-8', errors='surrogateescape') for p in proc.stdout.split(b'\0') if p]
        for path in result['untracked']:
            if proof_candidate(path) or is_archive(path):
                gap('file.untracked_proof', path, 'untracked proof/archive candidate is outside tracked scan')
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        gap('git.enumeration', '', type(error).__name__)
    result['tracked_count'] = len(by_path)
    result['read_count'] = len(raw_files)
    manifests = {}
    for kind, value in [('registry', registry), ('quarantine', quarantine)]:
        try:
            manifests[kind] = validate_manifest(value, kind, by_path)
        except (ValueError, TypeError) as error:
            gap(kind + '.invalid', '', str(error))
            manifests[kind] = {}
    result['quarantine'] = list(manifests['quarantine'].values())
    if quarantine is not None:
        result['semantic_receipts_required'].append('quarantine_isolation_and_closed_dependencies')
    candidates = {p for p in by_path if proof_candidate(p)}
    candidates.update(p for p, row in manifests['registry'].items() if row['role'] == 'proof-source')
    result['proof_candidates'] = sorted(candidates)
    for path, raw in raw_files.items():
        suffix = PurePosixPath(path).suffix.lower()
        classified = manifests['registry'].get(path, {}).get('role')
        by_path[path]['classification'] = classified or 'unclassified'
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            gap('file.lfs_pointer', path, 'LFS pointer bytes do not include payload')
            continue
        if is_archive(path, raw):
            gap('archive.unsupported', path, 'archive contents not scanned; independent extraction inventory required')
            continue
        if path in candidates and suffix != '.lean':
            gap('proof.unsupported', path, 'proof/document language is outside Lean lexical scope')
        if suffix != '.lean' and classified != 'certification':
            continue
        try:
            source = raw.decode('utf-8')
        except UnicodeDecodeError:
            gap('file.encoding', path, 'source is not strict UTF-8')
            continue
        tokens = []
        findings = []
        if suffix == '.lean':
            tokens, lex_gaps = lean_tokens(source)
            for line, detail in lex_gaps:
                gap('lean.lexical_gap', path, detail, line)
            findings.extend(lean_findings(tokens))
        if classified == 'certification':
            cert_findings, cert_gaps = certification_findings(path, source, tokens)
            findings.extend(cert_findings)
            for code, line, detail in cert_gaps:
                gap(code, path, detail, line)
        for code, line, detail in findings:
            result['findings'].append({'code': code, 'path': path, 'line': line, 'detail': detail, 'sha256': by_path[path]['sha256'], 'quarantine_requested': path in manifests['quarantine']})
    for item in result['coverage_gaps']:
        entry = by_path.get(item['path'], {})
        item.update(sha256=entry.get('sha256'), index_object=entry.get('index_object'), kind=item['code'].split('.', 1)[0])
    result['inventory_sha256'] = digest(json.dumps(result['files'], sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())
    result['static_result'] = 'FAIL' if result['findings'] else ('INCOMPLETE' if result['coverage_gaps'] else 'PASS')
    result['result'] = 'FAIL' if result['findings'] else 'INCOMPLETE'
    return result


# Each mutation changes the actual scanner source. A killed mutant must complete
# the whole test process with assertion failures, not syntax/import errors.
MUTATIONS = [
    ('disable_placeholder_detector', "LEAN_FORBIDDEN = {'sorry', 'admit', 'axiom', 'native_decide'}", 'LEAN_FORBIDDEN = set()'),
    ('disable_trust_detector', "LEAN_TRUST = {'sorryAx', 'unsafe', 'implemented_by', 'extern', 'ofReduceBool', 'ofReduceNat', 'trustCompiler', 'skipKernelTC'}", 'LEAN_TRUST = set()'),
    ('ignore_hidden_files', "if not record:\n                continue", "if not record or b'\\t.' in record:\n                continue"),
    ('suppress_coverage_failures', "result['coverage_gaps'].append({'code': code, 'path': path, 'line': line, 'detail': detail})", 'pass'),
    ('suppress_active_findings', "for code, line, detail in findings:", 'for code, line, detail in []:'),
    ('disable_float_detector', 'if literal or call:', 'if False:'),
    ('ignore_quarantine_hash', "or row['sha256'] != files[path].get('sha256')", ''),
    ('accept_empty_semantic_closure', "result['result'] = 'FAIL' if result['findings'] else 'INCOMPLETE'", "result['result'] = 'FAIL' if result['findings'] else 'PASS'"),
    ('disable_archive_detection', 'if is_archive(path, raw):', 'if False:'),
    ('ignore_untracked_proofs', 'if proof_candidate(path) or is_archive(path):', 'if False:'),
    ('misclassify_all_python_as_certification', "classified = manifests['registry'].get(path, {}).get('role')", "classified = 'certification' if suffix == '.py' else manifests['registry'].get(path, {}).get('role')"),
    ('hide_quarantined_findings', 'for code, line, detail in findings:', "for code, line, detail in ([] if path in manifests['quarantine'] else findings):"),
]


def selftest(tests):
    source_path = Path(__file__).resolve()
    source = source_path.read_text()
    helper = source_path.with_name('cbz_mutation.py')
    helper_before = digest(helper.read_bytes())
    tests_before = digest(tests.read_bytes())
    optimization = ['-OO'] if sys.flags.optimize == 2 else ['-O'] if sys.flags.optimize == 1 else []
    def run(path, expected=None):
        env = dict(os.environ, CBZ_AUDIT_UNDER_TEST=str(path), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', *optimization, str(helper), '--tests', str(tests)]
        with tempfile.TemporaryDirectory(prefix='cbz-audit-population-') as temp:
            if expected is not None:
                population = Path(temp) / 'expected.json'
                population.write_text(json.dumps(expected))
                command.extend(['--expected', str(population)])
            proc = subprocess.run(command, capture_output=True, text=True, env=env)
        try:
            evidence = json.loads(proc.stdout)
        except (ValueError, TypeError):
            evidence = {'body_kill': False, 'baseline_success': False, 'output': proc.stdout + proc.stderr}
        return proc.returncode, evidence
    code, baseline = run(source_path)
    result = {'schema_version': 1, 'tool': VERSION, 'scope': 'finite behavioral and source-mutation suite; no mathematical soundness claim', 'scanner_sha256': digest(source.encode()), 'tests_sha256': tests_before, 'lifecycle_helper_sha256': helper_before, 'optimization': sys.flags.optimize, 'baseline': {'exit_code': code, 'output': baseline.get('output', ''), 'lifecycle': baseline}, 'mutants': [], 'threshold': {'numerator': 1, 'denominator': 1}, 'result': 'FAIL'}
    if code != 0 or not baseline.get('baseline_success') or baseline.get('optimization') != sys.flags.optimize:
        return result
    with tempfile.TemporaryDirectory(prefix='cbz-audit-mutants-') as directory:
        for name, before, after in MUTATIONS:
            # Restrict replacement to production source before mutation definitions.
            head, tail = source.split('\nMUTATIONS = [', 1)
            if head.count(before) != 1:
                result['mutants'].append({'name': name, 'killed': False, 'error': 'mutation target not unique'})
                continue
            path = Path(directory) / (name + '.py')
            path.write_text(head.replace(before, after, 1) + '\nMUTATIONS = [' + tail)
            code, evidence = run(path, baseline['population'])
            killed = code == 0 and evidence.get('body_kill') is True and evidence.get('optimization') == sys.flags.optimize
            result['mutants'].append({'name': name, 'killed': killed, 'exit_code': code, 'output': evidence.get('output', ''), 'lifecycle': evidence})
    result['killed'] = sum(item['killed'] for item in result['mutants'])
    result['total'] = len(MUTATIONS)
    result['inputs_unchanged'] = (digest(source_path.read_bytes()) == result['scanner_sha256']
                                  and digest(tests.read_bytes()) == tests_before
                                  and digest(helper.read_bytes()) == helper_before)
    result['result'] = 'PASS' if result['inputs_unchanged'] and result['total'] > 0 and result['killed'] * result['threshold']['denominator'] >= result['total'] * result['threshold']['numerator'] else 'FAIL'
    return result


def load_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    def bad_number(value):
        raise ValueError('non-integer JSON number forbidden: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_float=bad_number, parse_constant=bad_number)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    scan_args = commands.add_parser('scan')
    scan_args.add_argument('--root', type=Path, default=Path.cwd())
    scan_args.add_argument('--registry', type=Path)
    scan_args.add_argument('--quarantine', type=Path)
    self_args = commands.add_parser('selftest')
    self_args.add_argument('--tests', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/test_cbz_audit.py')
    args = parser.parse_args(argv)
    try:
        if args.command == 'selftest':
            result = selftest(args.tests.resolve())
        else:
            result = scan(args.root, registry=load_json(args.registry) if args.registry else None, quarantine=load_json(args.quarantine) if args.quarantine else None)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        result = {'result': 'FAIL', 'error': type(error).__name__ + ': ' + str(error), 'mathematical_soundness_certified': False}
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if result['result'] == 'PASS' else (2 if result['result'] == 'INCOMPLETE' else 1)


if __name__ == '__main__':
    sys.exit(main())
