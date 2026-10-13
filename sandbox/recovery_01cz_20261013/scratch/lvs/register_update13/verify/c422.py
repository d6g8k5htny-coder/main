"""C422-E1 check on the inserted text only: for every inserted/replaced span in the changed leaves
(and every added leaf), list each run id mentioned and the nearest PR head / actual tested merge /
base in the same clause; flag a run mention lacking either, or bound to a wrong hash."""
import json, re, sys, difflib
out = json.load(open(sys.argv[1]))
FACT = {'38195810463': ('c6ce0256', '1320c5b217e491d301ea140e5cc25b07ff5a21b7', 'be01479435f98f34b802258ee9ec3c3578f37169')}
def toks(s): return re.findall(r'\S+|\s+', s)
spans = []
for path, o, n in out['changed']:
    a, b = toks(o), toks(n)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('insert', 'replace'):
            # widen to the clause: take full new text and the char offsets of the span
            start = len(''.join(b[:j1])); end = len(''.join(b[:j2]))
            spans.append((path, n, start, end))
for path, v in out['added']:
    if isinstance(v, str): spans.append((path, v, 0, len(v)))
bad = 0
for path, text, s, e in spans:
    ins = text[s:e]
    for m in re.finditer(r'\b(3\d{10})\b', ins):
        run = m.group(1); pos = s + m.start()
        # clause = from previous ';' or '. ' (outside nothing fancy) to next ';' or '). ' boundary
        lo = max(text.rfind(';', 0, pos), text.rfind('. ', 0, pos)); lo = 0 if lo < 0 else lo
        hi_c = [x for x in (text.find(';', pos), text.find('. ', pos)) if x >= 0]; hi = min(hi_c) if hi_c else len(text)
        clause = text[lo:hi]
        heads = re.findall(r'PR head ([0-9a-f]{8})', clause)
        merges = re.findall(r'actual tested (?:merge|checkout the merge commit) ([0-9a-f]{40})', clause)
        bases = re.findall(r'(?:with base|into base) ([0-9a-f]{40})', clause)
        ok = run in FACT and FACT[run][0] in heads and FACT[run][1] in merges and (not bases or FACT[run][2] in bases)
        wrong = run in FACT and (any(h != FACT[run][0] for h in heads) or any(x != FACT[run][1] for x in merges))
        print(('OK ' if ok and not wrong else 'BAD'), path, run, 'heads', heads, 'merges', [x[:8] for x in merges], 'bases', [x[:8] for x in bases])
        if not ok or wrong:
            bad += 1; print('   CLAUSE:', clause[:600])
print('bad', bad)
