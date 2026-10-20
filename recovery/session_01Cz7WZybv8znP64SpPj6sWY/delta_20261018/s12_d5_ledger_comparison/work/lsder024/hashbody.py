import sys, hashlib
txt=open(sys.argv[1],encoding='utf-8').read()
if txt.startswith('﻿'): txt=txt[1:]
lines=txt.replace('\r\n','\n').replace('\r','\n').split('\n')
i=lines.index('BEGIN_TBG3B_FROZEN_BODY'); j=lines.index('END_TBG3B_FROZEN_BODY')
body='\n'.join(lines[i+1:j])
for name,s in [('after-marker-line',body),('incl-newline-after-marker','\n'+body)]:
    s=s.rstrip('\n')+'\n'
    e=s.encode('utf-8'); print(name,len(e),hashlib.sha256(e).hexdigest())
