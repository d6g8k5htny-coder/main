#!/usr/bin/env python3
"""Layer A static scanner: CBZ-0 gate G1 (stdlib only). Scientific effect: NONE.

Reads every Git-tracked `.lean` file at an exact commit of each named
repository and reports every code token that can put an unproved, axiomatized
or kernel-unchecked declaration into a Lean environment: Layer A invariants
A1 (placeholders), A2 (axiom or trust injection) and A3 (floating point) of
GOVERNANCE.md. It reads committed bytes through `git cat-file`, never the
working tree, so a report names exactly what it scanned.

Masking. Comments (`--` and nested `/- -/`), string-literal text, raw strings
and character literals are masked before matching. Code inside `{...}` within
a string literal is kept, because `s!`/`m!`/`f!` and the `throwError` family
interpolate it; a plain string's braces are kept too, conservatively.
Identifiers follow Lean 4's own `isIdFirst`/`isIdRest` rules (ASCII letters,
the letter-like Unicode blocks without λ, Π and Σ, subscripts, and `_'!?`).
A rule matches one whole dot-separated name component, so `sorry_free` and
`sorry₁` are other names while `Foo.sorry`, `«sorry»` and `sorryλ` all match.

Exemptions. Every hit is a violation unless the registry
(tools/cbz/layer_a_exemptions.json) lists that exact repository, path and
token with the exact current count and a reason. A count that no longer
matches, or a missing path, is an error and exempts nothing, so a registry
entry can never cover a new hit. An entry marked `isolated` must not be
imported by any other scanned module of its repository.

Fails closed. An undecodable file, an unterminated comment, string or quoted
identifier, a symlinked `.lean` path or a submodule is an error, never a skip.
Out of scope, by design: semantic checks (vacuous hypotheses, weakened
bounds), non-Lean certificates, and files that are not tracked at the commit.
The kernel axiom audit (gate G2) remains the authority on axioms.

Exit status: 0 only when there is no unexempted hit and no error.
"""
import argparse
import bisect
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = 1

# token -> rule. Keys are whole name components after masking.
RULES = {
    # A1: logical placeholders and placeholder constructors
    'sorry': 'A1.sorry',
    'admit': 'A1.admit',
    'sorryAx': 'A1.sorry',
    'stop': 'A1.stop',
    'mkSorry': 'A1.sorry',
    'mkSyntheticSorry': 'A1.sorry',
    'mkLabeledSorry': 'A1.sorry',
    # A1: proof-search tactics that are not final proof scripts
    'exact?': 'A1.search',
    'apply?': 'A1.search',
    'rw?': 'A1.search',
    'simp?': 'A1.search',
    'simp_all?': 'A1.search',
    'aesop?': 'A1.search',
    # A2: axioms and kernel or compiler trust injection
    'axiom': 'A2.axiom',
    'native_decide': 'A2.native',
    'native': 'A2.native',
    'ofReduceBool': 'A2.native',
    'ofReduceNat': 'A2.native',
    'reduceBool': 'A2.native',
    'reduceNat': 'A2.native',
    'implemented_by': 'A2.trusted-code',
    'extern': 'A2.trusted-code',
    'skipKernelTC': 'A2.kernel-bypass',
    'addDecl': 'A2.kernel-bypass',
    'addAndCompile': 'A2.kernel-bypass',
    # A3: floating point inside Lean sources
    'Float': 'A3.float',
    'Float32': 'A3.float',
}
# Commands matched on the masked text (not identifiers).
COMMANDS = {'#exit': 'A1.exit'}

INTERPOLATING_MACRO = re.compile(
    r'(?<![A-Za-z0-9_.])(?:throwError|throwErrorAt|logInfo|logInfoAt|logWarning|logWarningAt|'
    r'logError|logErrorAt|dbg_trace|trace\[[^\]\n]*\])\s+(?:[^\s"]+\s+)?$')
CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^\\'\n])'")
IMPORT_LINE = re.compile(r'^[ \t]*import[ \t]+([^\n]+)', re.M)


def is_letter_like(c):
    v = ord(c)
    return ((0x3b1 <= v <= 0x3c9 and v != 0x3bb) or
            (0x391 <= v <= 0x3a9 and v != 0x3a0 and v != 0x3a3) or
            0x3ca <= v <= 0x3fb or 0x1f00 <= v <= 0x1ffe or
            0x2100 <= v <= 0x214f or 0x1d49c <= v <= 0x1d59f)


def is_subscript_alnum(c):
    v = ord(c)
    return 0x2080 <= v <= 0x2089 or 0x2090 <= v <= 0x209c or 0x1d62 <= v <= 0x1d6a


def is_id_first(c):
    return 'a' <= c <= 'z' or 'A' <= c <= 'Z' or c == '_' or is_letter_like(c)


def is_id_rest(c):
    return is_id_first(c) or '0' <= c <= '9' or c in "'!?" or is_subscript_alnum(c)


class Masker:
    """Replace comment, string-text and char-literal characters by spaces (newlines kept)."""

    def __init__(self, text):
        self.t = text
        self.n = len(text)
        self.out = list(text)
        self.errors = []

    def run(self):
        self.code(0, False)
        return ''.join(self.out), self.errors

    def blank(self, a, b):
        for k in range(a, min(b, self.n)):
            if self.out[k] != '\n':
                self.out[k] = ' '

    def restore(self, a, b):
        for k in range(a, min(b, self.n)):
            self.out[k] = self.t[k]

    def code(self, i, interpolant):
        t, n = self.t, self.n
        depth = 0
        while i < n:
            c = t[i]
            if c == '-' and t.startswith('--', i):
                j = t.find('\n', i)
                j = n if j < 0 else j
                self.blank(i, j)
                i = j
            elif c == '/' and t.startswith('/-', i):
                i = self.block_comment(i)
            elif c == '"':
                i = self.string(i)
            elif c == "'" and not (i > 0 and is_id_rest(t[i - 1])):
                m = CHAR_LITERAL.match(t, i)
                if m:
                    self.blank(i, m.end())
                    i = m.end()
                else:
                    i += 1
            elif interpolant and c == '{':
                depth += 1
                i += 1
            elif interpolant and c == '}':
                if depth == 0:
                    return i
                depth -= 1
                i += 1
            else:
                i += 1
        if interpolant:
            self.errors.append(('unterminated string interpolation', n))
        return n

    def block_comment(self, start):
        t, n = self.t, self.n
        depth, i = 0, start
        while i < n:
            if t.startswith('/-', i):
                depth += 1
                i += 2
            elif t.startswith('-/', i):
                depth -= 1
                i += 2
                if depth == 0:
                    self.blank(start, i)
                    return i
            else:
                i += 1
        self.errors.append(('unterminated block comment', start))
        self.blank(start, n)
        return n

    def interpolated(self, i):
        t = self.t
        if i >= 2 and t[i - 1] == '!':
            k = i - 1
            while k > 0 and is_id_rest(t[k - 1]):
                k -= 1
            if k < i - 1 and is_id_first(t[k]):
                return True
        line_start = t.rfind('\n', 0, i) + 1
        return INTERPOLATING_MACRO.search(t, line_start, i) is not None

    def string(self, start):
        t, n = self.t, self.n
        hashes, k = 0, start - 1
        while k >= 0 and t[k] == '#':
            hashes += 1
            k -= 1
        if k >= 0 and t[k] == 'r' and (k == 0 or not is_id_rest(t[k - 1])):
            close = '"' + '#' * hashes
            end = t.find(close, start + 1)
            if end < 0:
                self.errors.append(('unterminated raw string', start))
                self.blank(start, n)
                return n
            self.blank(start, end + len(close))
            return end + len(close)
        if self.interpolated(start):
            text_from, i = start, start + 1
            while i < n:
                c = t[i]
                if c == '\\':
                    i += 2
                elif c == '"':
                    self.blank(text_from, i + 1)
                    return i + 1
                elif c == '{':
                    self.blank(text_from, i + 1)
                    j = self.code(i + 1, True)
                    if j >= n:
                        return n
                    text_from, i = j, j + 1
                else:
                    i += 1
            self.errors.append(('unterminated string', start))
            self.blank(text_from, n)
            return n
        i = start + 1
        while i < n:
            c = t[i]
            if c == '\\':
                i += 2
            elif c == '"':
                break
            else:
                i += 1
        if i >= n:
            self.errors.append(('unterminated string', start))
            self.blank(start, n)
            return n
        end = i + 1
        self.blank(start, end)
        # Conservative: a plain string's brace segments are scanned as code.
        j = start + 1
        while j < i:
            if t[j] == '\\':
                j += 2
                continue
            if t[j] == '{':
                close = t.find('}', j + 1, i)
                if close < 0:
                    break
                self.restore(j + 1, close)
                j = close + 1
                continue
            j += 1
        return end


def names(masked):
    """Yield (component, offset) for every name component in masked code."""
    i, n = 0, len(masked)
    errors = []
    out = []
    while i < n:
        c = masked[i]
        if c == '«' or is_id_first(c):
            while True:
                if i < n and masked[i] == '«':
                    j = masked.find('»', i + 1)
                    if j < 0:
                        errors.append(('unterminated quoted identifier', i))
                        out.append((masked[i + 1:], i))
                        return out, errors
                    out.append((masked[i + 1:j], i))
                    i = j + 1
                elif i < n and is_id_first(masked[i]):
                    j = i + 1
                    while j < n and is_id_rest(masked[j]):
                        j += 1
                    out.append((masked[i:j], i))
                    i = j
                else:
                    break
                if i + 1 < n and masked[i] == '.' and (masked[i + 1] == '«' or is_id_first(masked[i + 1])):
                    i += 1
                    continue
                break
        else:
            # Digits are skipped one at a time: Lean lexes `2sorry` as `2` then `sorry`.
            i += 1
    return out, errors


class Located:
    def __init__(self, text):
        self.starts = [0] + [m.end() for m in re.finditer('\n', text)]

    def __call__(self, offset):
        line = bisect.bisect_right(self.starts, offset)
        return line, offset - self.starts[line - 1] + 1


def scan_text(text):
    """Return (hits, imports, errors) for one Lean source text."""
    masked, errors = Masker(text).run()
    found, name_errors = names(masked)
    errors = errors + name_errors
    where = Located(text)
    hits = []
    for comp, offset in found:
        rule = RULES.get(comp)
        if rule:
            line, col = where(offset)
            hits.append({'line': line, 'col': col, 'token': comp, 'rule': rule})
    for command, rule in COMMANDS.items():
        for m in re.finditer(re.escape(command), masked):
            after = m.end()
            if after < len(masked) and is_id_rest(masked[after]):
                continue
            line, col = where(m.start())
            hits.append({'line': line, 'col': col, 'token': command, 'rule': rule})
    hits.sort(key=lambda h: (h['line'], h['col'], h['token']))
    imports = []
    for m in IMPORT_LINE.finditer(masked):
        imports.extend(m.group(1).split())
    located_errors = []
    for message, offset in errors:
        line, col = where(min(offset, max(len(text) - 1, 0)))
        located_errors.append({'line': line, 'col': col, 'error': message})
    return hits, imports, located_errors


def git(path, *args, data=None):
    return subprocess.run(['git', '-C', str(path), *args], input=data, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def read_repository(path, rev):
    """Return (commit, files, errors): files maps tracked .lean path -> bytes."""
    commit = git(path, 'rev-parse', '--verify', rev + '^{commit}').decode().strip()
    entries = git(path, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0')
    blobs, errors = {}, []
    for entry in entries:
        if not entry:
            continue
        meta, _, name = entry.partition(b'\t')
        mode, kind, sha = meta.split(b' ')
        name = name.decode('utf-8', 'surrogateescape')
        if kind == b'commit':
            errors.append({'path': name, 'error': 'submodule not scanned'})
        elif name.endswith('.lean'):
            if mode == b'120000':
                errors.append({'path': name, 'error': 'symlinked Lean path not scanned'})
            elif kind == b'blob':
                blobs[name] = sha.decode()
    files = {}
    if blobs:
        ordered = sorted(blobs)
        raw = git(path, 'cat-file', '--batch', data=''.join(blobs[p] + '\n' for p in ordered).encode())
        offset = 0
        for name in ordered:
            header_end = raw.index(b'\n', offset)
            sha, kind, size = raw[offset:header_end].split(b' ')
            if sha.decode() != blobs[name] or kind != b'blob':
                raise RuntimeError('unexpected cat-file record for ' + name)
            size = int(size)
            files[name] = raw[header_end + 1:header_end + 1 + size]
            offset = header_end + 1 + size + 1
    return commit, files, errors


def module_candidates(path):
    parts = path[:-len('.lean')].split('/')
    return {'.'.join(parts[k:]) for k in range(len(parts))}


def load_exemptions(path):
    if path is None or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or not isinstance(data.get('exemptions'), list):
        raise ValueError('invalid exemption registry schema')
    rows = []
    for row in data['exemptions']:
        keys = {'repository', 'path', 'token', 'count', 'reason', 'isolated'}
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError('invalid exemption entry: ' + json.dumps(row, sort_keys=True))
        if (not isinstance(row['count'], int) or row['count'] < 1 or row['token'] not in {**RULES, **COMMANDS}
                or not isinstance(row['reason'], str) or len(row['reason'].strip()) < 20
                or not isinstance(row['isolated'], bool)):
            raise ValueError('invalid exemption entry: ' + json.dumps(row, sort_keys=True))
        rows.append(row)
    return rows


def scan(repositories, exemptions):
    """repositories: list of (name, path, rev). Returns the report dict."""
    report = {'schema': SCHEMA, 'scientific_effect': 'NONE', 'repositories': [], 'hits': [],
              'errors': [], 'exemption_errors': [], 'isolation_errors': []}
    per_repo = {}
    for name, path, rev in repositories:
        commit, files, read_errors = read_repository(path, rev)
        texts, imports = {}, {}
        for error in read_errors:
            report['errors'].append({'repository': name, **error})
        for file_path, raw in files.items():
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                report['errors'].append({'repository': name, 'path': file_path, 'error': 'not UTF-8'})
                continue
            hits, file_imports, errors = scan_text(text)
            texts[file_path] = hits
            imports[file_path] = file_imports
            for error in errors:
                report['errors'].append({'repository': name, 'path': file_path, **error})
        per_repo[name] = (texts, imports)
        report['repositories'].append({'name': name, 'commit': commit, 'lean_files': len(files),
                                       'bytes': sum(len(v) for v in files.values())})
    exempt_keys = {}
    for row in exemptions:
        key = (row['repository'], row['path'], row['token'])
        if key in exempt_keys:
            report['exemption_errors'].append({**row, 'error': 'duplicate exemption'})
            continue
        texts = per_repo.get(row['repository'], ({}, {}))[0]
        if row['repository'] not in per_repo:
            continue  # registry rows for repositories not scanned in this run are not evaluated
        if row['path'] not in texts:
            report['exemption_errors'].append({**row, 'error': 'exempted path is not a scanned Lean file'})
            continue
        actual = sum(1 for h in texts[row['path']] if h['token'] == row['token'])
        if actual != row['count']:
            report['exemption_errors'].append({**row, 'error': f'count mismatch: registry {row["count"]}, file {actual}'})
            continue
        exempt_keys[key] = row
    for name, (texts, imports) in per_repo.items():
        for file_path, hits in sorted(texts.items()):
            for hit in hits:
                exempt = (name, file_path, hit['token']) in exempt_keys
                report['hits'].append({'repository': name, 'path': file_path, **hit, 'exempt': exempt})
        isolated = {row['path'] for (repo, _, _), row in exempt_keys.items() if repo == name and row['isolated']}
        for fixture in sorted(isolated):
            candidates = module_candidates(fixture)
            for file_path, file_imports in sorted(imports.items()):
                if file_path == fixture or file_path in isolated:
                    continue
                for module in file_imports:
                    if module in candidates:
                        report['isolation_errors'].append({'repository': name, 'path': file_path,
                                                           'imports': module, 'fixture': fixture})
    violations = [h for h in report['hits'] if not h['exempt']]
    report['summary'] = {
        'lean_files': sum(r['lean_files'] for r in report['repositories']),
        'hits': len(report['hits']),
        'exempt_hits': len(report['hits']) - len(violations),
        'violations': len(violations),
        'errors': len(report['errors']),
        'exemption_errors': len(report['exemption_errors']),
        'isolation_errors': len(report['isolation_errors']),
    }
    clean = not violations and not report['errors'] and not report['exemption_errors'] and not report['isolation_errors']
    report['verdict'] = 'PASS' if clean else 'FAIL'
    return report


def parse_repo(spec):
    name, sep, rest = spec.partition('=')
    if not sep or not name:
        raise argparse.ArgumentTypeError('expected NAME=PATH[@REV]')
    path, at, rev = rest.rpartition('@')
    if not at:
        path, rev = rest, 'HEAD'
    return name, Path(path), rev


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', action='append', type=parse_repo, default=[],
                        help='NAME=PATH[@REV]; repeatable. Default: main=<this repository>@HEAD')
    parser.add_argument('--exemptions', default=str(here / 'layer_a_exemptions.json'))
    parser.add_argument('--json', help='write the full report to this file')
    args = parser.parse_args(argv)
    repositories = args.repo or [('main', here.parent.parent, 'HEAD')]
    try:
        report = scan(repositories, load_exemptions(args.exemptions))
    except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print('LAYER_A_SCAN_ERROR: ' + str(exc), file=sys.stderr)
        return 2
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    for repo in report['repositories']:
        print(f"scanned {repo['name']} @ {repo['commit']}: {repo['lean_files']} Lean files, {repo['bytes']} bytes")
    for hit in report['hits']:
        mark = 'exempt' if hit['exempt'] else 'VIOLATION'
        print(f"{mark}: {hit['repository']}:{hit['path']}:{hit['line']}:{hit['col']} {hit['token']} ({hit['rule']})")
    for kind in ('errors', 'exemption_errors', 'isolation_errors'):
        for row in report[kind]:
            print(kind.upper()[:-1] + ': ' + json.dumps(row, sort_keys=True))
    print('LAYER_A ' + report['verdict'] + ' ' + json.dumps(report['summary'], sort_keys=True))
    return 0 if report['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
