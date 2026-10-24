import sys,re,hashlib,itertools
t=open(sys.argv[1],'rb').read().decode('utf-8-sig').replace('\r\n','\n')
L=t.split('\n'); b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
body='\n'.join(L[b+1:e]).rstrip('\n')+'\n'
T='71754d660866721c037743579eaf2f7df282262ad29491faae912e8e587029fe'; N=3056
c=re.sub(r'\n\n\n+','\n\n',body)
print('collapsed',len(c.encode()))
url_lines=[l for l in c.split('\n') if l.startswith('https://')]
print('url lines',len(url_lines),'bytes incl LF',sum(len(l.encode())+1 for l in url_lines))
# candidate transforms
def tr(s,ops):
    for o in ops: s=o(s)
    return s
ops={
 'drop_url':lambda s: re.sub(r'(?m)^https://\S+\n',''  ,s),
 'drop_url_and_blank':lambda s: re.sub(r'(?m)^https://\S+\n\n?',''  ,s),
 'drop_drive_label':lambda s: re.sub(r'(?m)^Drive:\n',''  ,s),
 'dedent_sub':lambda s: re.sub(r'(?m)^   - ','- ',s),
 'dashdash_to_endash':lambda s: s.replace('--','–'),
 'no_blank_after_heading':lambda s: re.sub(r'(?m)^([A-Z0-9][A-Z0-9 \-—]+)\n\n',r'\1\n',s),
 'single_newlines':lambda s: re.sub(r'\n\n+','\n',s),
}
names=list(ops)
for r in range(1,5):
    for combo in itertools.combinations(names,r):
        s=tr(c,[ops[k] for k in combo])
        n=len(s.encode()); h=hashlib.sha256(s.encode()).hexdigest()
        if abs(n-N)<=40 or h==T: print(combo,n,h==T)
