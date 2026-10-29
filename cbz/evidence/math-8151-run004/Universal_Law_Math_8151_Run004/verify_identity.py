"""Read-only file identity checks; no theorem or Layer-A certification claim."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def require(value,message):
    if not value:raise RuntimeError(message)

def check_manifest(path,root):
    doc=json.loads(path.read_text());rows=doc['files'];seen=set()
    for row in rows:
        rel=Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts,'unsafe path')
        require(row['path'] not in seen,'duplicate manifest path');seen.add(row['path'])
        p=root/rel
        require(p.is_file()and not p.is_symlink(),'missing or symlink file: '+row['path'])
        require(p.stat().st_size==row['bytes'],'size: '+row['path'])
        require(hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],'hash: '+row['path'])
    return seen

paths=check_manifest(ROOT/'PACKAGE_MANIFEST.json',ROOT)
actual={p.relative_to(ROOT).as_posix()for p in ROOT.rglob('*')if p.is_file()}
require(actual==paths|{'PACKAGE_MANIFEST.json'},'package membership differs')
seal=ROOT/'sealed/MANIFEST.json'
require(hashlib.sha256(seal.read_bytes()).hexdigest()=='63d0c9e0268ffec38eeb2fe83a0d31850e740aaba39dbdf3caf76ef6970f6acf','original seal identity')
sealed=check_manifest(seal,ROOT/'sealed');require(len(sealed)==315,'original seal count')
receipt=ROOT/'sealed/formal-evidence/receipt.json'
require(hashlib.sha256(receipt.read_bytes()).hexdigest()=='7711c528031f26bb094718db736f960c546fcbb5c00f345fae1b201d4d85f561','receipt identity')
print('PASS file identity only:',len(paths),'package files;',len(sealed),'original sealed entries; no fresh kernel run')
