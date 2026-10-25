#!/usr/bin/env python3
"""Merge every listing outside Github/ into inventory_root.tsv (path, name, size, client_modified, folder, personal, hashable) and write hashing batches of 25 ns_paths."""
import json, glob, os, re, csv
S='<SCRATCH>'; D=S+'/dbxroot'
LOC=json.load(open(D+'/labels_local.json'))   # real folder names and the personal-name token live in this local file, never published
PERSONAL=re.compile(r'id.?card|passport|licen[cs]e|insurance|tax|invoice|receipt|bank|statement|medical|resume|cv\b|ssn|social.?security|birth.?cert|mortgage|loan|credit.?card|visa|vehicle|policy\d|\.dmg$|magnet|photo|selfie|IMG_|\.heic$|\.mov$|\.url$|'+LOC['extra_personal_pattern'], re.I)
rows={}
SHARED=open(D+'/shared_ns.txt').read().strip()   # the namespace id of the shared folder whose Github/ subtree main#594 covered; kept in a local file, not in this script
def folder_of(path):
    ns,rel=path.split('//',1)[0],path.split('//',1)[1]
    if ns==SHARED:
        if rel.startswith('Github/'): return None                      # covered by main#594; not re-inventoried
        return rel.split('/')[0] if '/' in rel else '(shared namespace root)'
    return rel.split('/')[0] if '/' in rel else '(root)'
def add(path,name,size,cm,folder,otype='file'):
    if otype!='file': return
    folder=folder_of(path)
    if folder is None: return
    rows[path]={'path':path,'name':name,'size':int(size) if str(size).isdigit() else 0,'client_modified':cm or '','folder':folder}
for r in json.load(open(D+'/root_misc_files.json')): add(r['path'],r['name'],r['size'],r['client_modified'],r['folder'])
for fn in ('list_2026.json','list_2026_p2.json','list_Ai.json','list_Notes.json'):
    p=D+'/'+fn
    if not os.path.exists(p): continue
    for e in json.load(open(p))['entries']:
        rel=e['path'].split('//',1)[1]; add(e['path'],e['name'],e.get('file',{}).get('size',0),e.get('client_modified'),rel.split('/')[0],e['object_type'])
for fn in glob.glob(D+'/parts/*.tsv'):
    for l in open(fn,encoding='utf-8'):
        c=l.rstrip('\n').split('\t')
        if len(c)>=3 and c[2]=='file':
            rel=c[0].split('//',1)[1]; ns=c[0].split('//')[0]
            if ns==SHARED and rel.startswith('Github/'): continue          # the Github/ shared tree: covered by main#594, not re-inventoried here
            folder=('(shared namespace root)' if (ns==SHARED and '/' not in rel) else rel.split('/')[0] if '/' in rel else '(root)')
            add(c[0],c[1],c[3] if len(c)>3 else 0,c[4] if len(c)>4 else '',folder)
inv=sorted(rows.values(),key=lambda r:r['path'])
for r in inv:
    r['personal']=bool(PERSONAL.search(r['path'].split('//',1)[1])) or r['folder'] in LOC['counted_only_folders']   # two root folders are the owner's other material: counted only
    r['hashable']=(not r['personal']) and r['size']>0
with open(D+'/inventory_root.tsv','w',encoding='utf-8') as f:
    f.write('path\tname\tsize\tclient_modified\tfolder\tpersonal\thashable\n')
    for r in inv: f.write('\t'.join([r['path'],r['name'],str(r['size']),r['client_modified'],r['folder'],'Y' if r['personal'] else 'N','Y' if r['hashable'] else 'N'])+'\n')
os.makedirs(D+'/batches',exist_ok=True)
for fn in glob.glob(D+'/batches/*.json'): os.remove(fn)
todo=[r['path'] for r in inv if r['hashable']]
for i in range(0,len(todo),25): json.dump(todo[i:i+25],open(D+f'/batches/b{i//25:03d}.json','w'))
import collections
print('inventory files',len(inv),'personal',sum(r['personal'] for r in inv),'zero-size',sum(1 for r in inv if r['size']==0),'hashable',len(todo),'batches',(len(todo)+24)//25)
print('by folder:',collections.Counter(r['folder'] for r in inv).most_common())
