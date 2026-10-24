#!/usr/bin/env python3
"""Merge saved list_folder pages into dbx/inventory.tsv (path, name, size, client_modified, modified_time, object_type). No ids kept."""
import glob, json, os, sys
S='<SCRATCH>/dbx'
rows={}
for p in sorted(glob.glob(S+'/pages/page*.json')):
    d=json.load(open(p))
    for e in d['entries']:
        path=e['path'].split('//',1)[1]   # strip the namespace prefix: path relative to the mount
        rows[path]=(path, e['name'], str(e.get('file',{}).get('size','')), e.get('client_modified',''), e.get('modified_time',''), e['object_type'])
with open(S+'/inventory.tsv','w',encoding='utf-8') as f:
    for r in sorted(rows.values()): f.write('\t'.join(r)+'\n')
files=[r for r in rows.values() if r[5]=='file']
print('rows',len(rows),'files',len(files),'folders',len(rows)-len(files))
import collections
print(collections.Counter(r[0].split('/')[1] if r[0].count('/')>=2 else '(Github root)' for r in files).most_common(12))
print('modified after 2026-09-27:', sum(1 for r in files if r[3]>'2026-09-27T23:59:59Z'))
