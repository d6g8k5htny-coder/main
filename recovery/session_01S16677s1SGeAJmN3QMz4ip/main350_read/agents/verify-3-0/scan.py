import hashlib, sys
p = sys.argv[1]
data = open(p,'rb').read()
lines = data.split(b'\n')
if lines[-1] == b'': lines = lines[:-1]
lines = [l+b'\n' for l in lines]
n = len(lines)
TARGET_B, TARGET_L = 32809, 696
TARGET_H = '001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6'
CORE_H = 'd4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213'
print("file lines", n, "bytes", len(data))
# core check
core = b''.join(lines[35:655])
print("core 36-655:", len(lines[35:655]), "lines", len(core), "bytes", hashlib.sha256(core).hexdigest(), "MATCH" if hashlib.sha256(core).hexdigest()==CORE_H else "NO")
# exhaustive contiguous ranges
pref = [0]
for l in lines: pref.append(pref[-1]+len(l))
hits_b = 0; hits_h = 0
for i in range(n):
    for j in range(i+1, n+1):
        if pref[j]-pref[i] == TARGET_B:
            hits_b += 1
            h = hashlib.sha256(b''.join(lines[i:j])).hexdigest()
            print("contiguous bytes hit", i+1, j, "lines", j-i, h)
            if h == TARGET_H: hits_h += 1
print("contiguous ranges with 32809 bytes:", hits_b, "hash hits:", hits_h)
# prefix [0..k) + core [35..655) + suffix [s..e) combos with 696 lines & 32809 bytes
combo_hits = 0
for k in range(0, 36):
    for s in range(655, n):
        for e in range(s+1, n+1):
            L = k + 620 + (e-s)
            if L != TARGET_L: continue
            B = pref[k] + (pref[655]-pref[35]) + (pref[e]-pref[s])
            if B != TARGET_B: continue
            h = hashlib.sha256(b''.join(lines[:k])+core+b''.join(lines[s:e])).hexdigest()
            print("combo hit prefix1-%d core suffix %d-%d" % (k, s+1, e), h, "MATCH" if h==TARGET_H else "NO")
            combo_hits += 1
print("prefix+core+suffix combos with 696 lines / 32809 bytes:", combo_hits)
# also try with CRLF or no trailing newline variants of whole file / core
for name, blob in [("whole", data), ("whole-no-trailing-nl", data.rstrip(b'\n')), ("core-crlf", core.replace(b'\n', b'\r\n'))]:
    print(name, len(blob), hashlib.sha256(blob).hexdigest())
