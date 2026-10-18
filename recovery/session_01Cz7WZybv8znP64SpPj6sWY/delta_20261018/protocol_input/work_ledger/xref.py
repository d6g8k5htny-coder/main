import re, json, sys
base=sys.argv[1]
sb=json.load(open(base+'/files/SOURCE_BINDINGS_V1.json'))
fz=json.load(open(base+'/freeze/author/FINAL_FREEZE_V2.json'))
known={}
for b in sb['bindings']: known[b['sha256']]=(b['bytes'],b['path'].split('/')[-1],'SB')
for k,v in fz.items():
    if isinstance(v,list):
        for b in v: known.setdefault(b['sha256'],(b['bytes'],b['path'].split('/')[-1],'FZ'))
    if isinstance(v,dict) and 'sha256' in v: known.setdefault(v['sha256'],(v['bytes'],v['path'].split('/')[-1],'FZ'))
for fn in ['PROSPECTIVE_CORRECTIVE_PROTOCOL_V1.md','ADDENDUM_V2.md','AUTHOR_REPORT_V1.md','DESIGN_AND_PLAN_V1.md']:
    t=open(base+'/files/'+fn).read()
    for m in re.finditer(r'(\d{3,6})B/(?:SHA(?:256)?)?([0-9a-f]{6,64})',t):
        by,h=int(m.group(1)),m.group(2)
        hit=[(k,v) for k,v in known.items() if k.startswith(h)]
        st='OK' if hit and hit[0][1][0]==by else ('BYTES_MISMATCH '+str(hit[0][1]) if hit else 'UNBOUND')
        print(fn[:12],by,h[:12],st,hit[0][1][1] if hit else '')
    for m in re.finditer(r'([A-Za-z_]+(?:\.md|\.json|\.py))?\s?(\d{3,6})B/(?:SHA256)?([0-9a-f]{64})',t):
        pass
