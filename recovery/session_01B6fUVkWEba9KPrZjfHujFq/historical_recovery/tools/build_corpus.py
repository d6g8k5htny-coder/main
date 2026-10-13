"""Build a searchable text corpus from every git object of every mirror, the hardening custody
store, and the recovered Dropbox texts. Every item keeps its exact identity (sha1 blob / sha256)."""
import hashlib, io, json, os, re, subprocess, sys, zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
sys.path.insert(0, '/home/user/main/tools')
import dropbox_reconcile as dr

S = Path('/tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad')
OUT = S / 'corpus'
REPOS = {'main': Path('/home/user/main')}
for p in sorted((S / 'repos').iterdir()):
    REPOS[p.name] = p
MAXB = 30_000_000
seen = {}  # sha1 -> relative text path
idx = open(OUT / 'index.tsv', 'w')
idx.write('source\tid\tsha256\tkind\tbytes\ttext_file\tnames\n')
log = open(OUT / 'build.log', 'w')

def clean(s): return s.replace('\t', ' ').replace('\n', ' ')

def extract(name, data, depth=0):
    """-> list of (member_name, kind, text)"""
    out = []
    suf = PurePosixPath(name).suffix.lower()
    try:
        if data[:4] == b'PK\x03\x04' and suf not in {'.docx', '.pptx', '.xlsx', '.odt'}:
            if depth >= 4: return out
            z = zipfile.ZipFile(io.BytesIO(data))
            for info in z.infolist():
                if info.is_dir() or info.file_size > MAXB: continue
                try: b = z.read(info)
                except Exception as e: out.append((info.filename, 'error:' + type(e).__name__, '')); continue
                for mn, k, t in extract(info.filename, b, depth + 1):
                    out.append((info.filename + ('!' + mn if mn else ''), k, t))
            return out
        if data[:5] == b'%PDF-': suf = '.pdf'
        if suf in {'.docx', '.pptx', '.xlsx', '.odt', '.pdf', '.ipynb', '.rtf'}:
            k, t, _ = dr.extract_bytes(name if PurePosixPath(name).suffix else name + suf, data)
            return [('', k, t)]
        if b'\x00' in data[:8000]: return [('', 'binary', '')]
        try: t = data.decode('utf-8')
        except UnicodeDecodeError: t = data.decode('latin-1')
        return [('', 'text', t)]
    except Exception as e:
        return [('', 'error:' + type(e).__name__, '')]

def emit(source, ident, sha256, names, data, sub):
    n = 0
    for mn, kind, text in extract(names[0] if names else '', data):
        if not text.strip():
            idx.write(f'{source}\t{ident}\t{sha256}\t{kind}\t{len(data)}\t\t{clean(" | ".join(names[:5]))}{(" !" + clean(mn)) if mn else ""}\n'); continue
        rel = f'{sub}/{ident}{("__m" + str(n)) if mn else ""}.txt'; n += 1
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_text(text[:MAXB], encoding='utf-8', errors='replace')
        idx.write(f'{source}\t{ident}\t{sha256}\t{kind}\t{len(data)}\t{rel}\t{clean(" | ".join(names[:5]))}{(" !" + clean(mn)) if mn else ""}\n')

for repo, path in REPOS.items():
    paths = defaultdict(set)
    rev = subprocess.run(['git', '-C', str(path), 'rev-list', '--objects', '--all'], capture_output=True).stdout
    for line in rev.decode('utf-8', 'surrogateescape').splitlines():
        sha, _, p = line.partition(' ')
        if p: paths[sha].add(p)
    lst = subprocess.run(['git', '-C', str(path), 'cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype) %(objectsize)'],
                         capture_output=True).stdout.decode().splitlines()
    shas = [l.split()[0] for l in lst if ' blob ' in l and int(l.split()[2]) <= MAXB]
    cat = subprocess.Popen(['git', '-C', str(path), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    done = 0
    for sha in shas:
        cat.stdin.write(sha.encode() + b'\n'); cat.stdin.flush()
        size = int(cat.stdout.readline().split()[2]); data = cat.stdout.read(size); cat.stdout.read(1)
        names = sorted(paths.get(sha, []))
        if sha in seen:
            idx.write(f'git:{repo}\t{sha}\t\tdup\t{size}\t{seen[sha]}\t{clean(" | ".join(names[:5]))}\n'); continue
        seen[sha] = f'git/{sha[:2]}/{sha}.txt'
        emit(f'git:{repo}', sha, hashlib.sha256(data).hexdigest(), names, data, f'git/{sha[:2]}')
        done += 1
    cat.stdin.close(); cat.wait()
    print(f'{repo}: blobs={len(shas)} new={done}', file=log, flush=True)

# hardening custody store(s)
inv = {}
for l in open(S / 'wt/hardening/drive/inventory.jsonl'):
    r = json.loads(l); inv[r['id']] = r
for objs in (S / 'wt/hardening/drive/deltas').glob('*/*/objects.json'):
    D = objs.parent
    o = json.loads(objs.read_text())
    cov = {}
    if (D / 'coverage.jsonl').exists():
        for l in open(D / 'coverage.jsonl'):
            r = json.loads(l)
            if r.get('sha256'): cov.setdefault(r['sha256'], []).append(r)
    zcache = {}
    for sha, rec in o['objects'].items():
        data = b''
        for c in rec['chunks']:
            pk = D / o['chunks'][c]['pack']
            z = zcache.get(pk) or zcache.setdefault(pk, zipfile.ZipFile(pk))
            data += z.read(c)
        if len(zcache) > 8: zcache.clear()
        names = []
        for r in cov.get(sha, []):
            i = inv.get(r['id'], {})
            nm = i.get('path') or i.get('title') or r['id']
            mime = r.get('export_mime_type') or i.get('mimeType', '')
            if 'wordprocessingml' in mime and not nm.endswith('.docx'): nm += '.docx'
            elif 'spreadsheetml' in mime and not nm.endswith('.xlsx'): nm += '.xlsx'
            elif 'presentationml' in mime and not nm.endswith('.pptx'): nm += '.pptx'
            names.append(f'{nm} [drive:{r["id"]}; {r.get("status")}; modified {r.get("drive_modified_time") or r.get("listed_modified_time")}]')
        emit(f'custody:{D.name}', sha, sha, names or [sha], data, f'custody/{sha[:2]}')
    print(f'custody {D.name}: objects={len(o["objects"])}', file=log, flush=True)

# Dropbox recovered texts
dbx = S / 'dbx'
for n in ('catalog_fetch.json', 'catalog_fetch_retry.json', 'catalog_fetch_trial.json'):
    if not (dbx / n).exists(): continue
    for c in json.loads((dbx / n).read_text()):
        tp = c.get('text_path')
        if not tp or not Path(tp).exists(): continue
        data = Path(tp).read_bytes()
        emit('dropbox', c['id'].replace(':', '_'), c.get('sha256') or hashlib.sha256(data).hexdigest(),
             [(c.get('path') or c.get('title') or '') + ' [dropbox text' + ('; exact original' if c.get('exact_original') else '; connector extract') + ']'], data, 'dropbox')
print('done', file=log, flush=True)
idx.close()
