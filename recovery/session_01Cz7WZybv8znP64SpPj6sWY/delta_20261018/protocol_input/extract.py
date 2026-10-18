import hashlib, re, sys, os
out = sys.argv[1]
for src in sys.argv[2:]:
    t = open(src, encoding='utf-8').read()
    if t.endswith('\n'): t = t[:-1]  # jq adds one LF
    lines = t.split('\n')
    i = 0
    while i < len(lines):
        m = re.match(r'^### FULL (\S+)', lines[i])
        if not m: i += 1; continue
        name = m.group(1); j = i + 1; decl = None
        while j < len(lines) and not re.match(r'^(`{3,}|~{3,})', lines[j]):
            mm = re.search(r'Bytes\s*([0-9,]+);\s*SHA256\s*([0-9a-f]{64})', lines[j])
            if mm: decl = (int(mm.group(1).replace(',', '')), mm.group(2))
            j += 1
        fence = re.match(r'^(`{3,}|~{3,})', lines[j]).group(1); k = j + 1
        while k < len(lines) and lines[k] != fence: k += 1
        body = '\n'.join(lines[j+1:k])
        cands = [body.encode(), (body + '\n').encode()]
        pick = None
        for b in cands:
            if decl and len(b) == decl[0] and hashlib.sha256(b).hexdigest() == decl[1]: pick = b
        b = pick or cands[1]
        p = os.path.join(out, name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'wb').write(b)
        print(f"{name}\t{len(b)}\t{hashlib.sha256(b).hexdigest()[:16]}\tdecl={decl[0] if decl else None}/{decl[1][:16] if decl else None}\t{'MATCH' if pick else 'NO-MATCH'}")
        i = k + 1
