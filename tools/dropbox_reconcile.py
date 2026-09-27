"""Reconcile a Dropbox inventory against the account's Git repositories and Drive.

Scientific effect: NONE. This tool moves no status and edits no source. It
builds indexes, classifies Dropbox files, extracts text from fetched bytes,
and stages copies for review. Every stage writes compact JSON so that no
model has to read file contents to decide what is already preserved.

Stages (each is a subcommand):

  index-git   Every blob reachable from any ref of each given repository,
              with size, sha256, Dropbox content_hash and every path it
              appeared under.
  index-drive Drive-side identities from the public source catalog
              (docs/public-math/sources*.json) and, optionally, a CSV export
              of the Drive source map (columns name,size,sha256[,id,path]).
  classify    Match a Dropbox inventory (list_folder entries, optionally
              augmented with content_hash values) against both indexes.
  fetch       Download single-use Dropbox URLs into a content-addressed store.
  extract     Per-type text/metadata extraction for fetched files.
  stage       Build a GitHub review packet (text only, intake-lane limits) and
              a Drive upload list for originals, for files classified MISSING.

Match tiers, strongest first:
  GIT_EXACT / DRIVE_EXACT   byte-identical (Dropbox content_hash or sha256)
  GIT_NAME_SIZE / DRIVE_NAME_SIZE   same basename and byte count (probable)
  DUP_IN_DROPBOX            another Dropbox file with the same hash, or the
                            same basename and size
  SKIP_*                    software bundles, caches and lock/run debris
  MISSING                   no match in any tier

Only the standard library is required. pdftotext is used when installed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

BLOCK = 4 * 1024 * 1024

SKIP_DIR_PARTS = {'Notes.app', '__pycache__', '.git', 'node_modules'}
SKIP_SUFFIXES = {'.pyc', '.nib', '.loctable', '.car', '.icns', '.appex',
                 '.storyboardc', '.lock', '.done', '.live', '.exit', '.time',
                 '.elapsed', '.rc', '.dmg', '.pkg', '.exe', '.msi'}
TEXT_SUFFIXES = {'.txt', '.md', '.json', '.jsonl', '.py', '.tex', '.csv',
                 '.tsv', '.log', '.out', '.err', '.stdout', '.sh', '.lean',
                 '.gs', '.xml', '.sha256', '.receipt', '.bib', '.yaml',
                 '.yml', '.toml', '.ini', '.cfg', '.html', '.htm'}
INTAKE_SUFFIXES = {'.md', '.txt', '.json', '.csv'}
INTAKE_MAX_FILE = 262144
INTAKE_MAX_FILES = 50
PII = [re.compile(p) for p in (
    r'\b\d{3}-\d{2}-\d{4}\b',                        # SSN-shaped
    r'\b(?:\d[ -]?){13,16}\b',                       # card-shaped
    r'\b\+?1?[ .-]?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b',  # phone-shaped
    r'(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]',
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    r'\bgh[pousr]_[A-Za-z0-9]{36,}\b',
    r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b',
)]


def dbx_hash_bytes(data: bytes) -> str:
    """Dropbox content_hash: sha256 over the concatenated sha256 of 4 MiB blocks."""
    blocks = b''.join(hashlib.sha256(data[i:i + BLOCK]).digest()
                      for i in range(0, len(data), BLOCK))
    return hashlib.sha256(blocks).hexdigest()


def dbx_hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(BLOCK)
            if not chunk:
                break
            h.update(hashlib.sha256(chunk).digest())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel_of(entry):
    """Dropbox path without the namespace prefix."""
    p = entry.get('path_display') or entry['path']
    return p.split('//', 1)[1] if '//' in p else p.lstrip('/')


# ---------------------------------------------------------------- index-git

def git_blobs(repo: Path):
    """Yield (blob_sha1, size, path) for every blob reachable from any ref."""
    rev = subprocess.Popen(['git', '-C', str(repo), 'rev-list', '--objects', '--all'],
                           stdout=subprocess.PIPE)
    check = subprocess.Popen(['git', '-C', str(repo), 'cat-file',
                              '--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)'],
                             stdin=rev.stdout, stdout=subprocess.PIPE)
    rev.stdout.close()
    for line in check.stdout:
        parts = line.decode('utf-8', 'surrogateescape').rstrip('\n').split(' ', 3)
        if len(parts) >= 3 and parts[1] == 'blob':
            yield parts[0], int(parts[2]), (parts[3] if len(parts) == 4 else '')
    check.wait()


def cmd_index_git(args):
    index = {}
    for spec in args.repos:
        name, _, path = spec.partition('=')
        repo = Path(path or name)
        name = name if path else repo.name
        seen = {}
        for sha1, size, p in git_blobs(repo):
            rec = seen.get(sha1)
            if rec is None:
                rec = seen[sha1] = {'size': size, 'paths': set()}
            if p:
                rec['paths'].add(p)
        cat = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        for sha1, rec in seen.items():
            cat.stdin.write((sha1 + '\n').encode())
            cat.stdin.flush()
            header = cat.stdout.readline().split()
            data = cat.stdout.read(int(header[2]))
            cat.stdout.read(1)
            key = f'{name}:{sha1}'
            index[key] = {'repo': name, 'blob': sha1, 'size': rec['size'],
                          'sha256': hashlib.sha256(data).hexdigest(),
                          'dbx': dbx_hash_bytes(data),
                          'paths': sorted(rec['paths'])[:20]}
        cat.stdin.close()
        cat.wait()
        print(f'{name}: {len(seen)} blobs', file=sys.stderr)
    Path(args.out).write_text(json.dumps(index, separators=(',', ':')))
    print(json.dumps({'blobs': len(index), 'out': args.out}))


# -------------------------------------------------------------- index-drive

def cmd_index_drive(args):
    rows = []
    for src in args.catalog or []:
        data = json.loads(Path(src).read_text())
        items = data.get('artifacts') or data.get('rows') or data if isinstance(data, list) else data.get('artifacts', [])
        for a in items if isinstance(items, list) else []:
            if not isinstance(a, dict):
                continue
            rows.append({'name': PurePosixPath(a.get('path', '')).name, 'size': a.get('bytes'),
                         'sha256': a.get('sha256'), 'where': f"{a.get('repository', '')}:{a.get('path', '')}",
                         'origin': 'catalog'})
    for src in args.drive_csv or []:
        with open(src, newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                low = {k.strip().lower(): (v or '').strip() for k, v in r.items() if k}
                size = low.get('size') or low.get('bytes') or ''
                path = low.get('original path') or low.get('member path') or low.get('path') or ''
                sha = next((low[k] for k in ('source sha-256', 'payload sha-256', 'sha-256', 'sha256')
                            if re.fullmatch(r'[0-9a-f]{64}', low.get(k, '').lower())), '')
                carrier = low.get('carrier title')
                rows.append({'name': PurePosixPath(path).name if path else (low.get('title') or low.get('name') or ''),
                             'size': int(float(size)) if re.fullmatch(r'\d+(\.0+)?', size) else None,
                             'sha256': sha.lower() or None,
                             'where': (f'drive-archive:{carrier}!{path}' if carrier else
                                       f"drive:{low.get('drive id') or low.get('id') or ''}:{path}"),
                             'origin': PurePosixPath(src).stem})
    for r in rows:
        # A file of at most one 4 MiB block has content_hash = sha256(sha256(bytes)),
        # so Drive's SHA-256 gives an exact Dropbox match without any download.
        if r.get('sha256') and r.get('size') is not None and 0 < r['size'] <= BLOCK:
            r['dbx'] = hashlib.sha256(bytes.fromhex(r['sha256'])).hexdigest()
        elif r.get('size') == 0:
            r['dbx'] = hashlib.sha256(b'').hexdigest()
    Path(args.out).write_text(json.dumps(rows, separators=(',', ':')))
    print(json.dumps({'drive_rows': len(rows), 'with_dbx': sum('dbx' in r for r in rows), 'out': args.out}))


# ----------------------------------------------------------------- classify

def skip_reason(rel: str, name: str):
    parts = set(PurePosixPath(rel).parts)
    if parts & SKIP_DIR_PARTS:
        return 'SKIP_SOFTWARE_BUNDLE' if 'Notes.app' in parts else 'SKIP_CACHE'
    suf = PurePosixPath(name).suffix.lower()
    if suf in SKIP_SUFFIXES:
        return 'SKIP_RUN_DEBRIS'
    return None


def cmd_classify(args):
    inv = json.loads(Path(args.inventory).read_text())
    hashes = json.loads(Path(args.dbx_hashes).read_text()) if args.dbx_hashes else {}
    git = json.loads(Path(args.git_index).read_text())
    drive = json.loads(Path(args.drive_index).read_text()) if args.drive_index else []
    g_dbx, g_ns = defaultdict(list), defaultdict(list)
    for k, v in git.items():
        g_dbx[v['dbx']].append(k)
        for p in v['paths']:
            g_ns[(PurePosixPath(p).name.lower(), v['size'])].append(f"{v['repo']}:{p}")
    d_dbx, d_ns = defaultdict(list), defaultdict(list)
    for r in drive:
        if r.get('dbx'):
            d_dbx[r['dbx']].append(r['where'])
        if r.get('size') is not None:
            d_ns[(r['name'].lower(), r['size'])].append(r['where'])
    files = [e for e in inv if e.get('object_type') == 'file']
    first_by_hash, first_by_ns = {}, {}
    out = []
    for e in sorted(files, key=lambda x: rel_of(x)):
        rel, name = rel_of(e), e['name']
        size = (e.get('file') or {}).get('size', 0)
        h = hashes.get(e['file_id']) or e.get('content_hash')
        row = {'id': e['file_id'], 'path': rel, 'size': size, 'dbx': h}
        why = skip_reason(rel, name)
        if why:
            row['status'] = why
        elif h and h in g_dbx:
            row['status'], row['match'] = 'GIT_EXACT', g_dbx[h][:3]
        elif h and h in d_dbx:
            row['status'], row['match'] = 'DRIVE_EXACT', d_dbx[h][:3]
        elif (name.lower(), size) in g_ns:
            row['status'], row['match'] = 'GIT_NAME_SIZE', g_ns[(name.lower(), size)][:3]
        elif (name.lower(), size) in d_ns:
            row['status'], row['match'] = 'DRIVE_NAME_SIZE', d_ns[(name.lower(), size)][:3]
        elif h and h in first_by_hash:
            row['status'], row['match'] = 'DUP_IN_DROPBOX', [first_by_hash[h]]
        elif (name.lower(), size) in first_by_ns:
            row['status'], row['match'] = 'DUP_IN_DROPBOX', [first_by_ns[(name.lower(), size)]]
        else:
            row['status'] = 'MISSING'
        if h:
            first_by_hash.setdefault(h, rel)
        first_by_ns.setdefault((name.lower(), size), rel)
        out.append(row)
    summary = defaultdict(lambda: [0, 0])
    for r in out:
        summary[r['status']][0] += 1
        summary[r['status']][1] += r['size']
    Path(args.out).write_text(json.dumps(out, separators=(',', ':')))
    print(json.dumps({k: {'files': v[0], 'bytes': v[1]} for k, v in sorted(summary.items())}, indent=1))


# -------------------------------------------------------------------- fetch

def cmd_fetch(args):
    """urls.json: [{id, path, download_url, content_hash?}] from download_link."""
    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    got = []
    for rec in json.loads(Path(args.urls).read_text()):
        tmp = store / (rec['id'].replace(':', '_') + '.part')
        try:
            with urllib.request.urlopen(rec['download_url'], timeout=120) as r, open(tmp, 'wb') as fh:
                shutil.copyfileobj(r, fh)
        except Exception as exc:  # single-use URLs cannot be retried; report and continue
            got.append({'id': rec['id'], 'path': rec.get('path'), 'error': type(exc).__name__})
            continue
        dh = dbx_hash_file(tmp)
        if rec.get('content_hash') and rec['content_hash'] != dh:
            got.append({'id': rec['id'], 'path': rec.get('path'), 'error': 'content_hash mismatch'})
            tmp.unlink()
            continue
        sha = sha256_file(tmp)
        final = store / sha
        tmp.replace(final)
        got.append({'id': rec['id'], 'path': rec.get('path'), 'sha256': sha, 'dbx': dh,
                    'bytes': final.stat().st_size})
    Path(args.out).write_text(json.dumps(got, indent=1))
    print(json.dumps({'fetched': sum('sha256' in g for g in got), 'errors': sum('error' in g for g in got)}))


# ------------------------------------------------------------------ extract

def _xml_text(data: bytes, tag_break=('</w:p>', '</a:p>', '</text:p>')) -> str:
    s = data.decode('utf-8', 'replace')
    for t in tag_break:
        s = s.replace(t, t + '\n')
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s)


def _rtf_text(s: str) -> str:
    s = re.sub(r'\\par[d]?', '\n', s)
    s = re.sub(r"\\'[0-9a-f]{2}", '', s)
    s = re.sub(r'\\[a-zA-Z]+-?\d* ?', '', s)
    return re.sub(r'[{}]', '', s)


def extract_bytes(name: str, data: bytes, depth=0):
    """Return (extractor, text, members) for one file's bytes."""
    suf = PurePosixPath(name).suffix.lower()
    members = []
    if suf in TEXT_SUFFIXES or suf == '':
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            text = data.decode('latin-1')
        if suf == '.json':
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, separators=(',', ':'))
            except ValueError:
                pass
        return 'text', text, members
    if suf == '.url':
        m = re.search(rb'URL=(\S+)', data)
        return 'url', (m.group(1).decode('utf-8', 'replace') if m else ''), members
    if suf == '.rtf':
        return 'rtf', _rtf_text(data.decode('latin-1')), members
    if suf == '.ipynb':
        nb = json.loads(data)
        cells = ['\n'.join(c.get('source', [])) if isinstance(c.get('source'), list) else c.get('source', '')
                 for c in nb.get('cells', [])]
        return 'ipynb', '\n\n'.join(cells), members
    if suf in {'.docx', '.pptx', '.xlsx', '.odt'}:
        z = zipfile.ZipFile(io.BytesIO(data))
        pick = [n for n in z.namelist() if n.startswith(('word/document', 'ppt/slides/slide', 'xl/sharedStrings',
                                                          'xl/worksheets/sheet', 'content.xml'))]
        return suf[1:], '\n'.join(_xml_text(z.read(n)) for n in sorted(pick)), members
    if suf == '.pdf':
        if shutil.which('pdftotext'):
            p = subprocess.run(['pdftotext', '-layout', '-', '-'], input=data, capture_output=True)
            if p.returncode == 0:
                return 'pdftotext', p.stdout.decode('utf-8', 'replace'), members
        return 'pdf-unextracted', '', members
    if suf == '.zip' and depth < 3:
        z = zipfile.ZipFile(io.BytesIO(data))
        for info in z.infolist():
            if info.is_dir() or info.file_size > 64 * 1024 * 1024:
                continue
            b = z.read(info)
            ex, tx, _ = extract_bytes(info.filename, b, depth + 1)
            members.append({'member': info.filename, 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
                            'dbx': dbx_hash_bytes(b), 'extractor': ex, 'chars': len(tx)})
        return 'zip', '', members
    if suf in {'.png', '.jpg', '.jpeg', '.gif', '.webp'}:
        return 'image', '', members
    return 'binary', '', members


def cmd_extract(args):
    store, text_dir = Path(args.store), Path(args.text_out)
    text_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    for rec in json.loads(Path(args.fetched).read_text()):
        if 'sha256' not in rec:
            continue
        data = (store / rec['sha256']).read_bytes()
        ex, text, members = extract_bytes(rec['path'], data)
        row = dict(rec, extractor=ex, chars=len(text), members=members)
        if text:
            tb = text.encode('utf-8')
            row['text_sha256'] = hashlib.sha256(tb).hexdigest()
            (text_dir / (rec['sha256'] + '.txt')).write_bytes(tb)
            row['pii_flags'] = [p.pattern for p in PII if p.search(text)]
        catalog.append(row)
    Path(args.out).write_text(json.dumps(catalog, indent=1))
    print(json.dumps({'extracted': len(catalog), 'with_text': sum(1 for c in catalog if c['chars']),
                      'pii_flagged': sum(1 for c in catalog if c.get('pii_flags'))}))


# -------------------------------------------------------------------- stage

def cmd_stage(args):
    """Stage MISSING files: text copies for a GitHub review packet, originals for Drive."""
    catalog = {c['id']: c for c in json.loads(Path(args.catalog).read_text())}
    allow = set(json.loads(Path(args.allow).read_text())) if args.allow else None
    store, text_dir, gh, drive = Path(args.store), Path(args.text_dir), Path(args.github_out), Path(args.drive_out)
    gh.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    artifacts, uploads = [], []
    for row in json.loads(Path(args.classified).read_text()):
        c = catalog.get(row['id'])
        if row['status'] != 'MISSING' or not c or 'sha256' not in c:
            continue
        uploads.append({'dropbox_path': row['path'], 'sha256': c['sha256'], 'bytes': c['bytes'],
                        'source': str(store / c['sha256'])})
        if c.get('pii_flags') or (allow is not None and row['path'] not in allow):
            continue  # never auto-publish flagged or unapproved material
        tpath = text_dir / (c['sha256'] + '.txt')
        if not tpath.exists():
            continue
        body = tpath.read_bytes()
        if len(body) > INTAKE_MAX_FILE:
            continue
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', PurePosixPath(row['path']).stem)[:80]
        out = gh / f'{stem}.{c["sha256"][:8]}.txt'
        out.write_bytes(body)
        artifacts.append({'path': out.name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                          'from_dropbox': row['path'], 'original_sha256': c['sha256'],
                          'original_bytes': c['bytes'], 'extractor': c['extractor']})
    Path(drive / 'UPLOADS.json').write_text(json.dumps(uploads, indent=1))
    Path(gh / 'PACKET_ARTIFACTS.json').write_text(json.dumps(artifacts, indent=1))
    over = max(0, len(artifacts) - INTAKE_MAX_FILES)
    print(json.dumps({'github_text_copies': len(artifacts), 'drive_uploads': len(uploads),
                      'packets_needed': -(-len(artifacts) // INTAKE_MAX_FILES), 'over_single_packet': over}))


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r'[A-Z]+', ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def xlsx_sheets(path: Path):
    """Yield (sheet_name, rows) from an .xlsx using only the standard library."""
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in re.findall(r'<si>(.*?)</si>', z.read('xl/sharedStrings.xml').decode('utf-8'), re.S):
            shared.append(html.unescape(''.join(re.findall(r'<t[^>]*>(.*?)</t>', si, re.S))))
    wb = z.read('xl/workbook.xml').decode('utf-8')
    rels = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', z.read('xl/_rels/workbook.xml.rels').decode('utf-8')))
    for name, rid in re.findall(r'<sheet [^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
        xml = z.read('xl/' + rels[rid].lstrip('/').replace('xl/', '')).decode('utf-8')
        rows = []
        for row in re.findall(r'<row[^>]*>(.*?)</row>', xml, re.S):
            cells = {}
            for attrs, body in re.findall(r'<c ([^>]*?)(?:/>|>(.*?)</c>)', row, re.S):
                ref = re.search(r'r="([A-Z]+)\d+"', attrs)
                typ = re.search(r't="(\w+)"', attrs)
                v = re.search(r'<v>(.*?)</v>', body or '', re.S)
                if typ and typ.group(1) == 's' and v:
                    val = shared[int(v.group(1))]
                elif typ and typ.group(1) == 'inlineStr':
                    val = html.unescape(''.join(re.findall(r'<t[^>]*>(.*?)</t>', body or '', re.S)))
                else:
                    val = html.unescape(v.group(1)) if v else ''
                if ref:
                    cells[_col(ref.group(1))] = val
            rows.append([cells.get(i, '') for i in range(max(cells) + 1)] if cells else [])
        yield html.unescape(name), rows


def cmd_xlsx_csv(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rows in xlsx_sheets(Path(args.xlsx)):
        fn = out / (re.sub(r'[^A-Za-z0-9_-]+', '_', name) + '.csv')
        with open(fn, 'w', newline='', encoding='utf-8') as fh:
            csv.writer(fh).writerows(rows)
        report[name] = len(rows)
    print(json.dumps(report))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('index-git'); s.add_argument('repos', nargs='+', help='name=path or path'); s.add_argument('--out', required=True); s.set_defaults(f=cmd_index_git)
    s = sub.add_parser('xlsx-csv'); s.add_argument('xlsx'); s.add_argument('--out-dir', required=True); s.set_defaults(f=cmd_xlsx_csv)
    s = sub.add_parser('index-drive'); s.add_argument('--catalog', nargs='*'); s.add_argument('--drive-csv', nargs='*'); s.add_argument('--out', required=True); s.set_defaults(f=cmd_index_drive)
    s = sub.add_parser('classify'); s.add_argument('--inventory', required=True); s.add_argument('--git-index', required=True); s.add_argument('--drive-index'); s.add_argument('--dbx-hashes'); s.add_argument('--out', required=True); s.set_defaults(f=cmd_classify)
    s = sub.add_parser('fetch'); s.add_argument('--urls', required=True); s.add_argument('--store', required=True); s.add_argument('--out', required=True); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser('extract'); s.add_argument('--fetched', required=True); s.add_argument('--store', required=True); s.add_argument('--text-out', required=True); s.add_argument('--out', required=True); s.set_defaults(f=cmd_extract)
    s = sub.add_parser('stage'); s.add_argument('--classified', required=True); s.add_argument('--catalog', required=True); s.add_argument('--store', required=True); s.add_argument('--text-dir', required=True); s.add_argument('--github-out', required=True); s.add_argument('--drive-out', required=True); s.add_argument('--allow'); s.set_defaults(f=cmd_stage)
    a = ap.parse_args(argv)
    return a.f(a) or 0


if __name__ == '__main__':
    sys.exit(main())
