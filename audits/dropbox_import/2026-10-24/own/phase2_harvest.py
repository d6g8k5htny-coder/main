#!/usr/bin/env python3
"""Harvest download_link tool results from this session's transcript into dbx/content_hashes.json
{rel_path: {size, content_hash}}.  download_url and Dropbox ids are never stored.  Idempotent."""
import json, os, sys
S='<SCRATCH>'
T='<SESSION_DIR>.jsonl'
OUT=S+'/dbx/content_hashes.json'
out=json.load(open(OUT)) if os.path.exists(OUT) else {}
new=0; calls=0
with open(T,'rb') as f:
    for line in f:
        if b'"content_hash"' not in line or b'download_url' not in line: continue
        try: j=json.loads(line)
        except Exception: continue
        for c in (j.get('message',{}).get('content') or []):
            if not (isinstance(c,dict) and c.get('type')=='tool_result'): continue
            t=c.get('content')
            if isinstance(t,list): t=''.join(x.get('text','') for x in t if isinstance(x,dict))
            if not isinstance(t,str): continue
            try: d=json.loads(t)
            except Exception: continue
            ents=d.get('entries') if isinstance(d,dict) else None
            if not ents: continue
            calls+=1
            for e in ents:
                p=e.get('path','')
                if '//' not in p or 'content_hash' not in e: continue
                rel=p.split('//',1)[1]
                if rel not in out: new+=1
                out[rel]={'size':e.get('size'),'content_hash':e['content_hash']}  # size may be absent for empty files
json.dump(out, open(OUT,'w'), indent=0, sort_keys=True, ensure_ascii=False)
print('calls',calls,'new',new,'total',len(out))
