#!/usr/bin/env python3
"""Read-only verification of the RETURN-05 root manifest; no source is executed."""
from pathlib import Path, PurePosixPath
import hashlib, json, sys

def verify(root):
    root = Path(root).resolve()
    seen = set()
    errors = []
    manifest = root / 'MANIFEST.sha256'
    for line_no, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        try:
            digest, size, name = line.split('  ', 2)
            p = PurePosixPath(name)
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid SHA-256')
            if p.is_absolute() or '..' in p.parts or '\\' in name or name in seen:
                raise ValueError('unsafe or duplicate path')
            seen.add(name)
            f = root.joinpath(*p.parts)
            if f.is_symlink() or not f.is_file():
                raise ValueError('missing/nonregular member')
            data = f.read_bytes()
            if len(data) != int(size) or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('size/hash mismatch')
        except (ValueError, OSError) as exc:
            errors.append({'line':line_no, 'error':str(exc)})
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != manifest}
    if actual != seen:
        errors.append({'unlisted': sorted(actual-seen), 'missing': sorted(seen-actual)})
    result = {'status':'PASS_BYTE_IDENTITY_ONLY' if not errors else 'FAIL',
              'members':len(seen), 'errors':errors,
              'mathematical_promotion':False, 'remote_writes':0}
    return result

if __name__ == '__main__':
    try:
        result = verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1])
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if not result['errors'] else 1)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status':'INPUT_ERROR', 'error':str(exc)}))
        sys.exit(2)
