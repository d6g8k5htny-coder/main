import re, sys, pathlib, collections
# Find command-category syntax/macro/elab declarations and extract their leading atom(s).
decl = re.compile(r'(?m)^\s*(?:@\[[^\]]*\]\s*)*(?:scoped |local )?(?:syntax|macro|elab|notation)\b(.*?):\s*command\b', re.S)
found = collections.defaultdict(set)
for root in sys.argv[1:]:
    for p in pathlib.Path(root).rglob('*.lean'):
        try: text = p.read_text(encoding='utf-8')
        except Exception: continue
        for m in decl.finditer(text):
            head = m.group(1)
            atoms = re.findall(r'"([^"]+)"', head)
            atoms = [a.strip() for a in atoms if a.strip()]
            if not atoms: continue
            first = atoms[0].split()[0] if atoms[0].split() else ''
            if re.fullmatch(r'[#A-Za-z_][A-Za-z0-9_?!]*', first):
                found[first].add(str(p.relative_to(root)) if False else root.split('/')[-1] + ':' + str(p.relative_to(root)))
for k in sorted(found):
    print(k, '|', sorted(found[k])[:3])
