import sys,re,hashlib
raw=open(sys.argv[1],'rb').read().decode('utf-8-sig').replace('\r\n','\n')
L=raw.split('\n'); b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
body='\n'.join(L[b+1:e]).rstrip('\n')+'\n'
T='71754d660866721c037743579eaf2f7df282262ad29491faae912e8e587029fe'
c=re.sub(r'\n\n\n+','\n\n',body)
cands={}
cands['collapse']=c
cands['no_blank_before_list']=re.sub(r'\n\n(?=\s*- )','\n',c)
cands['no_blank_before_list_or_url']=re.sub(r'\n\n(?=(\s*- |https?://|SHA-256|Frozen body|Drive:|\d+\. |\s+- ))','\n',c)
cands['drop_url_lines']=re.sub(r'(?m)^https?://\S+\n','',c)
for k,v in cands.items():
    print(k,len(v.encode()),hashlib.sha256(v.encode()).hexdigest()==T)
