import sys,html,hashlib,re
for fn,b,e in [(sys.argv[1],'BEGIN_TBG3B_FROZEN_BODY','END_TBG3B_FROZEN_BODY'),(sys.argv[2],'BEGIN_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY','END_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY')]:
    t=html.unescape(open(fn,encoding='utf-8').read())
    i=t.index(b)+len(b); j=t.index(e,i)
    body=t[i:j].replace('\r\n','\n').replace('\r','\n')
    body=body.lstrip('\n') if False else body
    for variant,bb in [('raw',body),('lstripnl',body[1:] if body.startswith('\n') else body)]:
        x=bb.rstrip('\n')+'\n'
        d=x.encode('utf-8')
        print(fn,variant,len(d),hashlib.sha256(d).hexdigest())
