import sys,re,hashlib
raw=open(sys.argv[1],'rb').read().decode('utf-8-sig').replace('\r\n','\n')
L=raw.split('\n'); b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
body=re.sub(r'\n\n\n+','\n\n','\n'.join(L[b+1:e]).rstrip('\n')+'\n')
ls=body.split('\n')
rm={17,18,19,20,21,25,26,28}
for j in range(14,31):
    t=re.sub(r'https?://\S+','<DRIVE-URL>',ls[j]); t=re.sub(r'^[0-9a-f]{64}$','<64-hex>',t)
    print(('REMOVED ' if j in rm else '        ')+str(j), repr(t))
v='\n'.join(l for j,l in enumerate(ls) if j not in rm).encode()
print(len(v), hashlib.sha256(v).hexdigest())
# does the 64-hex in body equal LS-DER-076 declared?
h=[l for l in ls if re.fullmatch(r'[0-9a-f]{64}',l)]
print('64hex == LS-DER-076 decl:', h[0]=='8250c80a0c034be676f221ec6ecc913bdec767aff9995e845bd6fc66ac92ea1f')
