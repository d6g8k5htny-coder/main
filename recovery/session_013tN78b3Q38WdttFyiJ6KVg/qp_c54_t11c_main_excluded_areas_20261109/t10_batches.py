#!/usr/bin/env python3
"""Pack tranche-10 objects (t10_objects.py output) into workflow batches. Objects keep their order (sorted by object id, so one
ref's objects stay together). A math batch holds at most 45 files and 12 objects; an engineering batch at most 160 files and 30
objects; an object larger than the limit gets a batch of its own.
Usage: python3 -I -B t10_batches.py OBJECTS_JSON FACTS_DIR"""
import json, os, sys
objs, out = sys.argv[1:]
b = json.load(open(objs))
os.makedirs(out, exist_ok=True)
n = 0
for kind, (fmax, omax) in (('math', (45, 12)), ('eng', (160, 30))):
    cur, files = [], 0
    def flush():
        global n, cur, files
        if cur:
            n += 1
            with open(os.path.join(out, f'qbatch{n}.json'), 'x') as f:
                json.dump({'batch': n, 'kind': kind, 'objects': cur}, f, indent=1, ensure_ascii=False)
            print(n, kind, len(cur), 'objects', files, 'files', sum(x['path'].endswith('.lean') for o in cur for x in o['files']), 'lean')
        cur, files = [], 0
    for o in sorted(b[kind], key=lambda o: (o['ref'], o['directory'])):
        k = len(o['files'])
        if cur and (files + k > fmax or len(cur) >= omax):
            flush()
        cur.append(o); files += k
    flush()
