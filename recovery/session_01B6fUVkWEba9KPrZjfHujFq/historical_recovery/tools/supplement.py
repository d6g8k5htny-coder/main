import hashlib, io, json, re, subprocess, sys, zipfile
from pathlib import Path
S = Path('/tmp/claude-0/-home-user-main/790ebd9d-259f-58e4-b82b-54c1c6b107a2/scratchpad')
sys.path.insert(0, str(S / 'pylib')); sys.path.insert(0, '/home/user/main/tools')
sys.modules['cryptography'] = None
import pypdf, dropbox_reconcile as dr
OUT = S / 'corpus'
rows = [l.rstrip('\n').split('\t') for l in open(OUT / 'index.tsv')][1:]
add = open(OUT / 'index.tsv', 'a')
REPOS = {'main': Path('/home/user/main')}
for p in sorted((S / 'repos').iterdir()): REPOS[p.name] = p

def pdftext(b):
    try:
        r = pypdf.PdfReader(io.BytesIO(b)); return '\n'.join((pg.extract_text() or '') for pg in r.pages)
    except Exception as e: return ''

def gitblob(repo, sha):
    return subprocess.run(['git', '-C', str(REPOS[repo]), 'cat-file', 'blob', sha], capture_output=True).stdout

custody_cache = {}
def custody_obj(dname, sha):
    D = next((S / 'wt/hardening/drive/deltas').glob(f'*/{dname}'))
    o = custody_cache.get(dname) or custody_cache.setdefault(dname, json.loads((D / 'objects.json').read_text()))
    rec = o['objects'][sha]
    return b''.join(zipfile.ZipFile(D / o['chunks'][c]['pack']).read(c) for c in rec['chunks'])

def member(data, mpath):
    parts = mpath.split('!')
    for p in parts:
        data = zipfile.ZipFile(io.BytesIO(data)).read(p)
    return data

nx = npdf = 0
for r in rows:
    if len(r) < 7: continue
    source, ident, sha256, kind, size, textfile, names = r
    # 1. Office XML members stored raw: convert to flowing text
    m = re.search(r' !(.*)$', names)
    mpath = m.group(1) if m else ''
    if textfile and re.search(r'(word/document\d*\.xml|ppt/slides/slide\d+\.xml|xl/sharedStrings\.xml|(^|/)content\.xml)$', mpath):
        p = OUT / textfile
        raw = p.read_bytes()
        if b'<' in raw[:200]:
            p.write_text(dr._xml_text(raw), encoding='utf-8'); nx += 1
    # 2. PDFs not extracted
    if kind == 'pdf-unextracted':
        try:
            if source.startswith('git:'):
                data = gitblob(source[4:], ident)
            elif source.startswith('custody:'):
                data = custody_obj(source[8:], ident)
            else: continue
            if mpath: data = member(data, mpath)
            t = pdftext(data)
            if t.strip():
                rel = f'pdf/{ident}__{hashlib.sha1(mpath.encode()).hexdigest()[:8]}.txt'
                (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
                (OUT / rel).write_text(t, encoding='utf-8')
                add.write(f'{source}\t{ident}\t{sha256}\tpdf-pypdf\t{size}\t{rel}\t{names}\n'); npdf += 1
        except Exception as e:
            print('pdf fail', source, ident[:12], mpath[:60], type(e).__name__)
# 3. Dropbox texts
nd = 0
dbx = S / 'dbx'
for n in ('catalog_fetch.json', 'catalog_fetch_retry.json', 'catalog_fetch_trial.json'):
    if not (dbx / n).exists(): continue
    for c in json.loads((dbx / n).read_text()):
        tp = c.get('text_path')
        if not tp: continue
        f = (dbx / tp) if not Path(tp).is_absolute() else Path(tp)
        if not f.exists(): continue
        rel = f'dropbox/{f.name}'
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        (OUT / rel).write_bytes(f.read_bytes())
        tag = 'exact original' if str(c.get('exact_original')) == 'True' else 'connector extract'
        add.write(f'dropbox\t{c["id"]}\t{c.get("sha256") or ""}\t{tag}\t{c.get("bytes")}\t{rel}\t{c.get("path")} [dropbox content_hash {c.get("dbx","")}]\n'); nd += 1
add.close()
print('xml', nx, 'pdf', npdf, 'dropbox', nd)
