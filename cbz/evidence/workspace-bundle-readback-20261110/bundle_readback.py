"""Read and hash a committed Workspace evidence bundle without modifying it."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

commit='abe9604f8fe7cebc72fb4673c9c47fa645e7d362'
blob='ae5b379433ab3d78bf8b853c2e53a0808da0ec23'
expected='cf78bbf3afad769f78f8c04ee207cfc99b51c2494d8cbda3539fbff8f0680a4b'
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
if out.exists():raise ValueError('output already exists')
if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=commit:
    raise ValueError('candidate commit mismatch')
path=root/'evidence/vector-origin-20261110/workspace-vector-hosted-20261110.zip'
if path.is_symlink() or not path.is_file():raise ValueError('unsafe bundle path')
if subprocess.check_output(['git','hash-object',str(path)],text=True).strip()!=blob:
    raise ValueError('native bundle blob mismatch')
raw=path.read_bytes()
if len(raw)!=1209336 or hashlib.sha256(raw).hexdigest()!=expected:
    raise ValueError('native bundle bytes mismatch')
with zipfile.ZipFile(path) as z:
    names=z.namelist()
    if len(names)!=98 or len(names)!=len(set(names)) or z.testzip() is not None:
        raise ValueError('native bundle structure or CRC mismatch')
    manifest=json.loads(z.read('SHA256.json'))
    if set(names)-{'SHA256.json'}!=set(manifest):raise ValueError('manifest coverage mismatch')
    for name,digest in manifest.items():
        if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('member mismatch: '+name)
out.mkdir(parents=True)
shutil.copyfile(path,out/'original-bundle.zip')
summary={'source_commit':commit,'git_blob':blob,'sha256':expected,'bytes':len(raw),
         'unique_members':len(names),'member_hashes_and_crc':'matched',
         'scope':'read-only native artifact byte custody, not final federation qualification'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
