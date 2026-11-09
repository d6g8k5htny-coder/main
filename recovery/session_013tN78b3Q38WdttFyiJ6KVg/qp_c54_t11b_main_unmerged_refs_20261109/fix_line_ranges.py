#!/usr/bin/env python3
"""Mechanical line-range repair for verified statements. For each statement whose text is not exactly the file's lines
line_start..line_end at the object's commit, look for the text, unchanged, at every range within WINDOW lines of the stated one;
if exactly one range matches, set line_start/line_end to it and log the change. The text is never changed, and a statement with
no match or several matches is left as it is (the assembler then fails closed on it).
Usage: python3 -I -B fix_line_ranges.py CLONE FACTS_DIR VERDICTS_IN VERDICTS_OUT"""
import glob, json, os, subprocess, sys
clone, facts_dir, vin, vout = sys.argv[1:]
WINDOW = 3
commit = {}
for f in glob.glob(os.path.join(facts_dir, 'qbatch*.json')):
    for o in json.load(open(f))['objects']:
        commit[o['object']] = o['commit']
cache = {}


def lines(c, p):
    if (c, p) not in cache:
        r = subprocess.run(['git', '-C', clone, 'show', f'{c}:{p}'], capture_output=True)
        cache[(c, p)] = r.stdout.decode('utf-8').split('\n') if r.returncode == 0 else None
    return cache[(c, p)]


d = json.load(open(vin))
log = []
for b in d['batches']:
    for v in b['verdicts'] or []:
        c = commit[v['object']]
        for it in v['items']:
            for st in it['statements']:
                ls = lines(c, st['file'])
                if ls is None:
                    continue
                a, e = st['line_start'], st['line_end']
                if 1 <= a <= e <= len(ls) and '\n'.join(ls[a - 1:e]) == st['text']:
                    continue
                n = st['text'].count('\n') + 1
                hits = [s for s in range(max(1, a - WINDOW), a + WINDOW + 1) if s + n - 1 <= len(ls)
                        and '\n'.join(ls[s - 1:s + n - 1]) == st['text']]
                if len(hits) == 1:
                    s = hits[0]
                    log.append(f'{v["object"]}#{it["item_id"]} {st["file"]}: {a}-{e} -> {s}-{s + n - 1}')
                    st['line_start'], st['line_end'] = s, s + n - 1
                else:
                    log.append(f'{v["object"]}#{it["item_id"]} {st["file"]}: {a}-{e} left as is ({len(hits)} matches)')
with open(vout, 'x') as f:
    json.dump(d, f, indent=1, ensure_ascii=False)
print('\n'.join(log))
print(len(log), 'line-range changes or misses')
