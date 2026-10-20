import sys, hashlib, html, re
txt = open(sys.argv[1], encoding='utf-8').read()
b0='BEGIN_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY'; b1='END_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY'
body = txt[txt.rindex(b0)+len(b0):txt.rindex(b1)]
def h(s):
    s = s.replace('\r\n','\n').replace('\r','\n').rstrip('\n')+'\n'
    bb = s.encode('utf-8'); return len(bb), hashlib.sha256(bb).hexdigest()[:16]
u = html.unescape(body).lstrip('\n')
print('plain unescaped:', h(u))
# each blank line read as two blank lines
u2 = re.sub(r'\n\n', '\n\n\n', u)
print('blank line doubled:', h(u2))
