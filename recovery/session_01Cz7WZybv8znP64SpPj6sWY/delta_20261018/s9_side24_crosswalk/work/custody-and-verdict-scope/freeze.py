import sys, re, hashlib, os
D = sys.argv[1]
names = sys.argv[2:]
def extract(raw, begin, end):
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    t = raw.decode('utf-8')
    t = t.replace('\r\n','\n').replace('\r','\n')
    lines = t.split('\n')
    bi = [i for i,l in enumerate(lines) if l == begin]
    ei = [i for i,l in enumerate(lines) if l == end]
    return lines, bi, ei
for n in names:
    p = os.path.join(D, n + '.txt')
    raw = open(p,'rb').read()
    t0 = raw[3:].decode('utf-8') if raw.startswith(b'\xef\xbb\xbf') else raw.decode('utf-8')
    tn = t0.replace('\r\n','\n').replace('\r','\n')
    mb = re.search(r'^BODY_BYTES:\s*(\d+)\s*$', tn, re.M)
    ms = re.search(r'^BODY_SHA256:\s*([0-9a-f]{64})\s*$', tn, re.M)
    mbeg = re.search(r'^(BEGIN_[A-Z0-9_]+_FROZEN_BODY)$', tn, re.M)
    if not (mb and ms and mbeg):
        print(f"{n}\tNO_FREEZE_DECLARATION\tbytes_decl={mb and mb.group(1)}\tsha_decl={ms and ms.group(1)}\tbegin={mbeg and mbeg.group(1)}")
        continue
    begin = mbeg.group(1); end = 'END_' + begin[len('BEGIN_'):]
    lines, bi, ei = extract(raw, begin, end)
    if len(bi)!=1 or len(ei)!=1:
        print(f"{n}\tMARKER_COUNT begin={len(bi)} end={len(ei)}"); continue
    body = '\n'.join(lines[bi[0]+1:ei[0]])
    body = body.rstrip('\n') + '\n'
    b = body.encode('utf-8')
    h = hashlib.sha256(b).hexdigest()
    db, ds = int(mb.group(1)), ms.group(1)
    print(f"{n}\tbegin_line={bi[0]+1}\tend_line={ei[0]+1}\tbytes={len(b)}\tdecl={db}\t{'BYTES_MATCH' if len(b)==db else 'BYTES_MISMATCH'}\tsha={h}\tdecl_sha={ds}\t{'SHA_MATCH' if h==ds else 'SHA_MISMATCH'}")
    # diagnostics: collapse doubled blank lines
    var = {}
    var['collapse_blank_runs'] = re.sub(r'\n\n\n+', '\n\n', body)
    var['all_blank_removed'] = re.sub(r'\n\n+', '\n', body)
    var['halve_blank'] = re.sub(r'\n(\n+)', lambda m: '\n' + '\n'*(len(m.group(1))//2), body)
    var['rstrip_lines'] = '\n'.join(l.rstrip() for l in body.split('\n'))
    for k,v in var.items():
        vb = v.encode('utf-8')
        print(f"   diag {k}: bytes={len(vb)} sha_match={hashlib.sha256(vb).hexdigest()==ds} bytes_match={len(vb)==db}")
    # non-ascii char census
    na = {}
    for ch in body:
        if ord(ch)>127: na[ch]=na.get(ch,0)+1
    print("   nonascii:", sorted(((hex(ord(k)),v) for k,v in na.items()))[:20])
