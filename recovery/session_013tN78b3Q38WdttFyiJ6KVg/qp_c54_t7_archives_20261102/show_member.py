#!/usr/bin/env python3
"""Print one member of a committed archive, read in memory (nothing written, nothing executed), with line numbers.
Usage: python3 -I -B show_member.py CLONE REF ARCHIVE_PATH MEMBER [NESTED_MEMBER] [--raw]
MEMBER may name a nested archive; then NESTED_MEMBER names the file inside it. --raw prints without line numbers."""
import gzip, io, re, subprocess, sys, tarfile, zipfile
args = [a for a in sys.argv[1:] if a != '--raw']
raw = '--raw' in sys.argv
clone, ref, path, member = args[:4]
nested = args[4] if len(args) > 4 else None
def read(name, data, m):
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.read(m)
    if re.search(r'\.(tgz|tar\.gz|tar)$', name, re.I):
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            return t.extractfile(t.getmember(m)).read()
    if name.lower().endswith('.gz'):
        return gzip.decompress(data)
    raise SystemExit('not an archive')
data = subprocess.run(['git', '-C', clone, 'cat-file', 'blob', f'{ref}:{path}'], capture_output=True, check=True).stdout
b = read(path, data, member)
if nested:
    b = read(member, b, nested)
text = b.decode('utf-8', errors='replace')
if raw:
    sys.stdout.write(text)
else:
    for i, line in enumerate(text.split('\n'), 1):
        print(f'{i:6d}\t{line}')
