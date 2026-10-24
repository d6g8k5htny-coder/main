import sys, re, hashlib, os
d = sys.argv[1]
for fn in sys.argv[2:]:
    p = os.path.join(d, fn)
    raw = open(p, 'rb').read().decode('utf-8-sig')
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    lines = raw.split('\n')
    b = [i for i, l in enumerate(lines) if re.fullmatch(r'BEGIN_[A-Z0-9_]*FROZEN_BODY', l)]
    e = [i for i, l in enumerate(lines) if re.fullmatch(r'END_[A-Z0-9_]*FROZEN_BODY', l)]
    m = re.search(r'BODY_SHA256:\s*([0-9a-f]{64})', raw)
    mb = re.search(r'BODY_BYTES:\s*([0-9,]+)', raw)
    if len(b) != 1 or len(e) != 1:
        print(fn, 'markers', len(b), len(e), 'declared', m.group(1)[:12] if m else None); continue
    body = '\n'.join(lines[b[0]+1:e[0]]).rstrip('\n') + '\n'
    bb = body.encode('utf-8')
    h = hashlib.sha256(bb).hexdigest()
    print(fn, 'body lines %d-%d' % (b[0]+2, e[0]), 'bytes', len(bb), 'declared_bytes', mb.group(1) if mb else None,
          'sha_ok' if (m and m.group(1) == h) else ('MISMATCH declared=%s got=%s' % (m.group(1)[:12] if m else None, h[:12])))
