#!/usr/bin/env python3
"""Merge download_link tool-result files into dbx/content_hashes.json {rel_path: {size, content_hash}}; URLs are never stored."""
import glob, json, os, sys
S='<SCRATCH>'
TR='/root/.claude/projects/-home-user-main/4dd532d5-b4fa-5c8e-a137-a0ec5074f318/tool-results'
out=json.load(open(S+'/dbx/content_hashes.json')) if os.path.exists(S+'/dbx/content_hashes.json') else {}
n=0
for p in sorted(glob.glob(TR+'/mcp-Dropbox-download_link-*.txt')):
    try: d=json.load(open(p))
    except Exception: continue
    for e in d.get('entries',[]):
        rel=e['path'].split('//',1)[1]
        if rel not in out: n+=1
        out[rel]={'size':e.get('size'),'content_hash':e.get('content_hash')}
json.dump(out, open(S+'/dbx/content_hashes.json','w'), indent=0, sort_keys=True)
print('new',n,'total',len(out))
