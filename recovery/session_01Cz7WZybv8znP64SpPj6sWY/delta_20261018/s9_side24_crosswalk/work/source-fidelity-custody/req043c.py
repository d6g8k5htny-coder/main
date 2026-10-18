import sys,re,hashlib,itertools
raw=open(sys.argv[1],'rb').read().decode('utf-8-sig').replace('\r\n','\n')
L=raw.split('\n'); b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
body='\n'.join(L[b+1:e]).rstrip('\n')+'\n'
T='71754d660866721c037743579eaf2f7df282262ad29491faae912e8e587029fe'
c=re.sub(r'\n\n\n+','\n\n',body)
urls=[m for m in re.finditer(r'https?://\S+',c)]
print('collapse bytes',len(c.encode()),'url count',len(urls),'url lens',[len(u.group(0)) for u in urls])
# long hex-ish ids
ids=re.findall(r'(?m)^[0-9a-f]{64}$',c); print('64hex lines',len(ids))
# candidate edits
cands={}
cands['drop_url_lines']=re.sub(r'(?m)^https?://\S+\n','',c)
cands['drop_url_and_Drive_label']=re.sub(r'(?m)^(Drive:\n)?https?://\S+\n','',c)
# drop URL lines and the blank lines between ledger entries
x=re.sub(r'(?m)^https?://\S+\n','',c)
cands['drop_url_then_collapse']=re.sub(r'\n\n\n+','\n\n',x)
for k,v in cands.items():
    bb=v.encode(); print(k,len(bb),hashlib.sha256(bb).hexdigest()==T, len(bb)-3056)
# what is in body before 'You are' line 44 -> reference to CL-AUD id masked
for i,l in enumerate(c.split('\n')):
    if re.search(r'[A-Za-z0-9_-]{25,}',l) and not l.startswith('http'):
        print('LONGTOKEN line', i, len(l), re.sub(r'[A-Za-z0-9_-]{25,}','<TOK%d>'%len(re.search(r'[A-Za-z0-9_-]{25,}',l).group(0)),l)[:120])
