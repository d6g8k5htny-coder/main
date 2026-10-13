import json, re, sys, collections
def leaves(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items(): yield from leaves(v, p+(k,))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, p+(i,))
    elif isinstance(o, str):
        yield p, o
heads = {
 '38116277597': ('00d2ac6e', 'c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c'),
 '38121713840': ('910809d6', '5b198699308e0585904c21d12bf540cf2475d2f0'),
 '38122386459': ('d9b8dd33', '866136cca66b46e6dbd3186b47c9340f7e93b4f6'),
 '38150363063': ('12e072d6', '86a6e8c876f7904cadda2d006a56f8792dc5c1af'),
 '38180705952': ('48bfcbae', '9833d6d9f9facc335e1a56e243c3144adefe6225'),
 '38187538744': ('be014794', '79d3964cf70e2eadad5c9629eef8f4b47f218dd0'),
}
for fn in sys.argv[1:]:
    d = json.load(open(fn))
    cnt = collections.Counter()
    bad = []
    for p, s in leaves(d):
        for m in re.finditer(r'\b(3[0-9]{10})\b', s):
            rid = m.group(1); cnt[rid] += 1
            # window: from previous ';' or '. ' to next ';' (approx clause)
            st = m.start()
            win = s[max(0, st-400): st+400]
            if rid in heads:
                h, mg = heads[rid]
                if h not in s or mg not in s:
                    bad.append((p, rid, 'leaf-missing'))
                elif mg not in win:
                    bad.append((p, rid, 'merge-not-within-400'))
                elif h not in win:
                    bad.append((p, rid, 'head-not-within-400'))
            else:
                bad.append((p, rid, 'unknown-run'))
    print(fn, dict(cnt))
    for b in bad: print('  ', b)
