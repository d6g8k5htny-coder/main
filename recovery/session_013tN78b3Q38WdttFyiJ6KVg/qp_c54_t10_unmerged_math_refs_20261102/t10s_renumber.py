#!/usr/bin/env python3
"""Renumber tranche-10 supplement batches (t10_batches.py output for t10s_objects.py) after tranche 10's 22 batches, refusing any
object id that tranche 10 already uses. Usage: python3 -I -B t10s_renumber.py SUPPLEMENT_BATCH_DIR T10_FACTS_DIR OUT_DIR OFFSET"""
import glob, json, os, re, sys
src, t10, dst, off = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t10ids = {o['object'] for f in glob.glob(os.path.join(t10, 'qbatch*.json')) for o in json.load(open(f))['objects']
          if json.load(open(f))['batch'] <= off}
os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, 'qbatch*.json')), key=lambda x: int(re.findall(r'(\d+)\.json', x)[0])):
    d = json.load(open(f))
    d['batch'] += off
    clash = [o['object'] for o in d['objects'] if o['object'] in t10ids]
    if clash:
        raise SystemExit(f'object ids already in tranche 10: {clash}')
    with open(os.path.join(dst, f'qbatch{d["batch"]}.json'), 'x') as g:
        json.dump(d, g, indent=1, ensure_ascii=False)
    print('qbatch', d['batch'], d['kind'], len(d['objects']), 'objects')
