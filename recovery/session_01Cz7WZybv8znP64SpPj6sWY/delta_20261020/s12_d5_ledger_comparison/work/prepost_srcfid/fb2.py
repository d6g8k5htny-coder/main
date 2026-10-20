import sys,html,hashlib,re,zipfile
def fromdocx(fn):
    z=zipfile.ZipFile(fn)
    x=z.read('word/document.xml').decode('utf-8')
    paras=re.findall(r'<w:p[ >].*?</w:p>',x,flags=re.S)
    out=[]
    for p in paras:
        # handle breaks
        p2=re.sub(r'<w:br[^>]*/>','\n',p)
        p2=re.sub(r'<w:tab[^>]*/>','\t',p2)
        texts=re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>|(\n)|(\t)',p2,flags=re.S)
        s=''.join(a or b or c for a,b,c in texts)
        out.append(html.unescape(s))
    return '\n'.join(out)
for fn,b,e in [(sys.argv[1],'BEGIN_TBG3B_FROZEN_BODY','END_TBG3B_FROZEN_BODY'),(sys.argv[2],'BEGIN_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY','END_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY')]:
  for src in ['txt','docx']:
    if src=='txt':
        t=html.unescape(open(fn,encoding='utf-8').read())
    else:
        t=fromdocx(fn.replace('.txt','.docx'))
    m=re.search(r'^'+b+r'$',t,flags=re.M); i=m.end()
    m2=re.search(r'^'+e+r'$',t[i:],flags=re.M); j=i+m2.start()
    body=t[i:j].replace('\r\n','\n').replace('\r','\n')
    for variant,bb in [('raw',body),('lstrip1',body[1:] if body.startswith('\n') else body)]:
        x=bb.rstrip('\n')+'\n'
        d=x.encode('utf-8')
        print(fn,src,variant,len(d),hashlib.sha256(d).hexdigest()[:16])
