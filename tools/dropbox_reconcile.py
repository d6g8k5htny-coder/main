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
  harvest-fetch  Without downloads: collect connector `fetch` text from
              transcripts; a text whose bytes reproduce the known content_hash
              is restored as the exact original, anything else stays text-only.
  extract     Per-type text/metadata extraction for fetched files.
  stage       Build a GitHub review packet (text only, intake-lane limits) and
              a Drive upload list for originals, for files classified MISSING.
              GitHub copies require --allow; without it none are staged.
  pack        Deflate the Drive upload list into a few size-capped zips that
              keep Dropbox paths, so each upload is one call, not one per file.
  report      Private Markdown summary of a classification: tiers, missing
              files by folder, largest files and personal-looking names.

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
import zlib
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
# File names that suggest personal paperwork; such files are never staged for GitHub.
PERSONAL_NAME = re.compile(
    r'(?i)r[eé]sum[eé]|curriculum.?vitae|(?<![a-z])cv(?![a-z])|(?<![a-z])tax(?:es)?(?![a-z])|'
    r'(?<![a-z0-9])(?:w-?2|1099)(?![a-z0-9])|bank|pay.?(?:stub|slip)|passport|driv\w*.?licen[cs]e|'
    r'invoice|medical|insurance|(?<![a-z])lease(?![a-z])|(?<![a-z])ssn(?![a-z])')


def personal_name(path: str) -> bool:
    return bool(PERSONAL_NAME.search(PurePosixPath(path).name))


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


# ------------------------------------------------------------ harvest-links

def _json_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_strings(v)


def cmd_harvest_links(args):
    """Collect Dropbox download_link results from a session transcript (JSONL).

    The links are single-use and short-lived, so only records newer than
    --since are emitted as fetchable; every content_hash is kept for classify.
    """
    urls, hashes = {}, {}
    with open(args.transcript, encoding='utf-8') as fh:
        for line in fh:
            if '"download_url' not in line and 'download_url\\"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            stamp = rec.get('timestamp', '')
            for s in _json_strings(rec):
                if 'download_url' not in s or not s.lstrip().startswith('{'):
                    continue
                try:
                    payload = json.loads(s)
                except ValueError:
                    continue
                for e in payload.get('entries', []) if isinstance(payload, dict) else []:
                    if not isinstance(e, dict) or 'download_url' not in e:
                        continue
                    if e.get('content_hash'):
                        hashes[e['id']] = e['content_hash']
                    if stamp >= (args.since or ''):
                        urls[e['id']] = {'id': e['id'], 'path': e.get('path_display') or e.get('path'),
                                         'download_url': e['download_url'], 'content_hash': e.get('content_hash'),
                                         'size': e.get('size'), 'issued': stamp}
    Path(args.urls_out).write_text(json.dumps(list(urls.values()), indent=1))
    Path(args.hashes_out).write_text(json.dumps(hashes, indent=1))
    print(json.dumps({'fetchable_links': len(urls), 'content_hashes': len(hashes)}))


def _fetch_payloads(obj):
    """Yield Dropbox fetch results ({id, text, ...}) found anywhere in a JSON value."""
    if isinstance(obj, dict):
        if isinstance(obj.get('id'), str) and isinstance(obj.get('text'), str) and 'title' in obj:
            yield obj
        for v in obj.values():
            yield from _fetch_payloads(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _fetch_payloads(v)
    elif isinstance(obj, str) and '"text"' in obj and obj.lstrip().startswith('{'):
        try:
            yield from _fetch_payloads(json.loads(obj))
        except ValueError:
            pass


def recover_original(text: str, content_hash: str):
    """Bytes whose Dropbox content_hash matches, from connector-extracted text, or None.

    The connector returns text files with one extra trailing newline; a CRLF
    original comes back with LF endings. Each candidate is accepted only on an
    exact content_hash match, so a wrong guess can never pass as an original.
    """
    if not content_hash:
        return None
    seen = set()
    for t in (text, text[:-1] if text.endswith('\n') else None):
        if t is None:
            continue
        for cand in (t, t.replace('\n', '\r\n')):
            for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
                try:
                    b = cand.encode(enc)
                except UnicodeEncodeError:
                    continue
                if b not in seen:
                    seen.add(b)
                    if dbx_hash_bytes(b) == content_hash:
                        return b
    return None


def cmd_harvest_fetch(args):
    """Collect Dropbox `fetch` results from transcripts and saved tool results.

    Writes a catalog in the shape `stage` reads. Exact originals (content_hash
    match) go to the content-addressed store; every text goes to --text-out.
    """
    want = {r['id']: r for r in json.loads(Path(args.classified).read_text())}
    store, text_dir = Path(args.store), Path(args.text_out)
    store.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    files = []
    for root in args.roots:
        rp = Path(root)
        files += [rp] if rp.is_file() else sorted(p for p in rp.rglob('*') if p.suffix in ('.jsonl', '.txt', '.json'))
    for f in files:
        with open(f, encoding='utf-8', errors='replace') as fh:
            chunks = fh if f.suffix == '.jsonl' else [fh.read()]
            for line in chunks:
                if '"text' not in line and '\\"text' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                for p in _fetch_payloads(obj):
                    if p['id'] in want and len(p['text']) >= len(found.get(p['id'], {}).get('text', '')):
                        found[p['id']] = p
    catalog = []
    for fid, p in sorted(found.items(), key=lambda kv: want[kv[0]]['path']):
        row = want[fid]
        text = p['text']
        tb = text.encode('utf-8')
        tsha = hashlib.sha256(tb).hexdigest()
        (text_dir / (tsha + '.txt')).write_bytes(tb)
        rec = {'id': fid, 'path': row['path'], 'dbx': row.get('dbx'), 'extractor': 'dropbox-fetch',
               'chars': len(text), 'text_sha256': tsha, 'text_path': str(text_dir / (tsha + '.txt')),
               'pii_flags': [q.pattern for q in PII if q.search(text)], 'members': []}
        orig = recover_original(text, row.get('dbx'))
        if orig is not None:
            sha = hashlib.sha256(orig).hexdigest()
            (store / sha).write_bytes(orig)
            rec.update(sha256=sha, bytes=len(orig), exact_original=True)
        else:
            rec.update(exact_original=False, original_bytes=row.get('size'))
        catalog.append(rec)
    Path(args.out).write_text(json.dumps(catalog, indent=1))
    print(json.dumps({'scanned_files': len(files), 'fetched': len(catalog),
                      'exact_originals': sum(c['exact_original'] for c in catalog),
                      'text_only': sum(not c['exact_original'] for c in catalog),
                      'pii_flagged': sum(bool(c['pii_flags']) for c in catalog)}))


def cmd_batches(args):
    """Emit file-id batches (25 each, the download_link maximum) for a given status."""
    rows = [r for r in json.loads(Path(args.classified).read_text()) if r['status'] in set(args.status)]
    rows.sort(key=lambda r: r['path'])
    batches = [[r['id'] for r in rows[i:i + 25]] for i in range(0, len(rows), 25)]
    Path(args.out).write_text(json.dumps(batches))
    print(json.dumps({'files': len(rows), 'batches': len(batches), 'bytes': sum(r['size'] for r in rows)}))


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
    # Fail closed: without an explicit allow-list nothing is staged for GitHub.
    # The Drive upload list is still written, since originals go to Drive regardless.
    allow = set(json.loads(Path(args.allow).read_text())) if args.allow else set()
    store, text_dir, gh, drive = Path(args.store), Path(args.text_dir), Path(args.github_out), Path(args.drive_out)
    gh.mkdir(parents=True, exist_ok=True)
    drive.mkdir(parents=True, exist_ok=True)
    artifacts, uploads = [], []
    rows = [r for r in json.loads(Path(args.classified).read_text())
            if r['status'] == 'MISSING' and ({'sha256', 'text_path'} & set(catalog.get(r['id'], {})))]
    # A loose file whose bytes are a member of a zip that is itself uploaded
    # travels inside that zip; uploading it again would only cost tokens.
    in_zip = {m['dbx']: row['path'] for row in rows for m in catalog[row['id']].get('members', [])}
    for row in rows:
        c = catalog[row['id']]
        up = None
        if 'sha256' in c:  # original bytes are available (fetched, or recovered exactly)
            up = {'dropbox_path': row['path'], 'sha256': c['sha256'], 'bytes': c['bytes'],
                  'source': str(store / c['sha256'])}
            if c.get('dbx') in in_zip and in_zip[c['dbx']] != row['path']:
                up['covered_by_zip'] = in_zip[c['dbx']]
            if personal_name(row['path']):
                up['personal_name'] = True
            uploads.append(up)
        if c.get('pii_flags') or personal_name(row['path']) or row['path'] not in allow:
            continue  # never auto-publish flagged, personal-looking or unapproved material
        tpath = Path(c['text_path']) if c.get('text_path') else text_dir / (c['sha256'] + '.txt')
        if not tpath.exists():
            continue
        body = tpath.read_bytes()
        if len(body) > INTAKE_MAX_FILE:
            continue
        key = c.get('sha256') or c['text_sha256']
        stem = re.sub(r'[^A-Za-z0-9_.-]+', '_', PurePosixPath(row['path']).stem)[:80]
        out = gh / f'{stem}.{key[:8]}.txt'
        out.write_bytes(body)
        artifacts.append({'path': out.name, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                          'from_dropbox': row['path'], 'original_sha256': c.get('sha256'),
                          'original_bytes': c.get('bytes', c.get('original_bytes')),
                          'exact_original': c.get('exact_original', 'sha256' in c), 'extractor': c['extractor']})
    Path(drive / 'UPLOADS.json').write_text(json.dumps(uploads, indent=1))
    Path(gh / 'PACKET_ARTIFACTS.json').write_text(json.dumps(artifacts, indent=1))
    over = max(0, len(artifacts) - INTAKE_MAX_FILES)
    print(json.dumps({'github_text_copies': len(artifacts), 'drive_uploads': len(uploads),
                      'covered_by_zip': sum('covered_by_zip' in u for u in uploads),
                      'personal_name': sum('personal_name' in u for u in uploads),
                      'packets_needed': -(-len(artifacts) // INTAKE_MAX_FILES), 'over_single_packet': over,
                      'allow_list': 'present' if args.allow else 'absent: nothing staged for GitHub'}))


# --------------------------------------------------------------------- pack

def cmd_pack(args):
    """Deflate UPLOADS.json entries into zips of at most --part-bytes compressed.

    Every upload to Drive through a chat connector passes the bytes through the
    model as base64, so the cost is proportional to compressed size. Entries
    covered by an uploaded zip, personal-looking names (unless
    --include-personal) and single files over --max-file-bytes are left out
    and listed, never silently dropped.
    """
    uploads = json.loads(Path(args.uploads).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    held, parts, members = [], [], []
    todo = []
    for u in sorted(uploads, key=lambda u: u['dropbox_path']):
        why = ('covered_by_zip' if u.get('covered_by_zip') else
               'personal_name' if u.get('personal_name') and not args.include_personal else
               'over_max_file_bytes' if u['bytes'] > args.max_file_bytes else None)
        if why:
            held.append({'dropbox_path': u['dropbox_path'], 'bytes': u['bytes'], 'reason': why})
        else:
            todo.append(u)
    z, zpath, n = None, None, 0

    def close():
        if z:
            z.close()
            parts.append({'part': zpath.name, 'bytes': zpath.stat().st_size, 'sha256': sha256_file(zpath)})

    for u in todo:
        data = Path(u['source']).read_bytes()
        est = len(zlib.compress(data, 9))
        if z is None or (zpath.stat().st_size + est > args.part_bytes and n):
            close()
            zpath = out / f'{args.prefix}_part{len(parts) + 1:02d}.zip'
            z, n = zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED, compresslevel=9), 0
        z.writestr(u['dropbox_path'], data)
        z.fp.flush()
        n += 1
        members.append({'part': zpath.name, 'dropbox_path': u['dropbox_path'], 'sha256': u['sha256'],
                        'bytes': u['bytes']})
    close()
    manifest = {'parts': parts, 'members': members, 'held': held}
    (out / f'{args.prefix}_MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    packed = sum(p['bytes'] for p in parts)
    print(json.dumps({'parts': len(parts), 'members': len(members), 'packed_bytes': packed,
                      'base64_chars': 4 * -(-packed // 3), 'held': len(held),
                      'held_bytes': sum(h['bytes'] for h in held)}))


# ------------------------------------------------------------------- report

def cmd_report(args):
    """Private Markdown summary of a classify output. Never commit its output."""
    rows = json.loads(Path(args.classified).read_text())
    tiers = defaultdict(lambda: [0, 0])
    for r in rows:
        tiers[r['status']][0] += 1
        tiers[r['status']][1] += r['size']
    miss = [r for r in rows if r['status'] == 'MISSING']
    folders = defaultdict(lambda: [0, 0])
    for r in miss:
        parent = str(PurePosixPath(r['path']).parent)
        key = '/'.join(parent.split('/')[:args.depth]) if parent != '.' else '(root)'
        folders[key][0] += 1
        folders[key][1] += r['size']
    kind = defaultdict(lambda: [0, 0])
    for r in miss:
        k = 'text' if PurePosixPath(r['path']).suffix.lower() in TEXT_SUFFIXES else \
            (PurePosixPath(r['path']).suffix.lower() or '(none)')
        kind[k][0] += 1
        kind[k][1] += r['size']
    mb = lambda b: f'{b / 1e6:.2f} MB'
    lines = ['# Dropbox reconciliation (private)', '', '| status | files | size |', '|---|---:|---:|']
    lines += [f'| {k} | {v[0]} | {mb(v[1])} |' for k, v in sorted(tiers.items())]
    lines += ['', f'## Missing by folder (depth {args.depth})', '', '| folder | files | size |', '|---|---:|---:|']
    lines += [f'| {k} | {v[0]} | {mb(v[1])} |' for k, v in sorted(folders.items(), key=lambda kv: -kv[1][1])]
    lines += ['', '## Missing by kind', '', '| kind | files | size |', '|---|---:|---:|']
    lines += [f'| {k} | {v[0]} | {mb(v[1])} |' for k, v in sorted(kind.items(), key=lambda kv: -kv[1][1])]
    lines += ['', f'## Largest {args.top} missing files', '']
    lines += [f'- {mb(r["size"])} `{r["path"]}`' for r in sorted(miss, key=lambda r: -r['size'])[:args.top]]
    flagged = [r for r in miss if personal_name(r['path'])]
    lines += ['', '## Personal-looking names (never staged for GitHub)', '']
    lines += [f'- `{r["path"]}`' for r in flagged] or ['- none']
    Path(args.out).write_text('\n'.join(lines) + '\n')
    print(json.dumps({'missing': len(miss), 'missing_bytes': sum(r['size'] for r in miss),
                      'personal_name': len(flagged), 'out': args.out}))


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
    s = sub.add_parser('harvest-links'); s.add_argument('--transcript', required=True); s.add_argument('--since', default=''); s.add_argument('--urls-out', required=True); s.add_argument('--hashes-out', required=True); s.set_defaults(f=cmd_harvest_links)
    s = sub.add_parser('harvest-fetch'); s.add_argument('roots', nargs='+', help='transcript files or directories'); s.add_argument('--classified', required=True); s.add_argument('--store', required=True); s.add_argument('--text-out', required=True); s.add_argument('--out', required=True); s.set_defaults(f=cmd_harvest_fetch)
    s = sub.add_parser('batches'); s.add_argument('--classified', required=True); s.add_argument('--status', nargs='+', default=['MISSING']); s.add_argument('--out', required=True); s.set_defaults(f=cmd_batches)
    s = sub.add_parser('fetch'); s.add_argument('--urls', required=True); s.add_argument('--store', required=True); s.add_argument('--out', required=True); s.set_defaults(f=cmd_fetch)
    s = sub.add_parser('extract'); s.add_argument('--fetched', required=True); s.add_argument('--store', required=True); s.add_argument('--text-out', required=True); s.add_argument('--out', required=True); s.set_defaults(f=cmd_extract)
    s = sub.add_parser('stage'); s.add_argument('--classified', required=True); s.add_argument('--catalog', required=True); s.add_argument('--store', required=True); s.add_argument('--text-dir', required=True); s.add_argument('--github-out', required=True); s.add_argument('--drive-out', required=True); s.add_argument('--allow'); s.set_defaults(f=cmd_stage)
    s = sub.add_parser('pack'); s.add_argument('--uploads', required=True); s.add_argument('--out-dir', required=True); s.add_argument('--prefix', default='dropbox_import'); s.add_argument('--part-bytes', type=int, default=1_000_000); s.add_argument('--max-file-bytes', type=int, default=4_000_000); s.add_argument('--include-personal', action='store_true'); s.set_defaults(f=cmd_pack)
    s = sub.add_parser('report'); s.add_argument('--classified', required=True); s.add_argument('--out', required=True); s.add_argument('--depth', type=int, default=2); s.add_argument('--top', type=int, default=25); s.set_defaults(f=cmd_report)
    a = ap.parse_args(argv)
    return a.f(a) or 0


if __name__ == '__main__':
    sys.exit(main())
