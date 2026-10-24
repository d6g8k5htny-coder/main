import sys, re, hashlib
raw=open(sys.argv[1],'rb').read()
t=raw[3:].decode() if raw.startswith(b'\xef\xbb\xbf') else raw.decode()
t=t.replace('\r\n','\n').replace('\r','\n')
L=t.split('\n')
b=L.index('BEGIN_LSREQ043_FROZEN_BODY'); e=L.index('END_LSREQ043_FROZEN_BODY')
body='\n'.join(L[b+1:e]).rstrip('\n')+'\n'
c=re.sub(r'\n\n\n+','\n\n',body)
print(len(c.encode()))
# try removing sections one at a time (by header) to see if any yields 3056
secs=re.split(r'(?m)^(?=[A-Z][A-Z0-9 \-—]+$)', c)
print([ (s.split('\n')[0][:40], len(s.encode())) for s in secs])
target='71754d660866721c037743579eaf2f7df282262ad29491faae912e8e587029fe'
import itertools
for i in range(len(secs)):
    v=''.join(secs[:i]+secs[i+1:])
    if hashlib.sha256(v.encode()).hexdigest()==target: print('match removing',i)
