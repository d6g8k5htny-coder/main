import sys, zipfile, re, hashlib, html
z=zipfile.ZipFile(sys.argv[1])
x=z.read('word/document.xml').decode('utf-8')
paras=re.findall(r'<w:p[ >].*?</w:p>|<w:p/>',x,flags=re.S)
def ptext(p):
    # include tabs and breaks
    out=[]
    for m in re.finditer(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>|<w:tab/>|<w:br/>',p,flags=re.S):
        if m.group(0).startswith('<w:tab'): out.append('\t')
        elif m.group(0).startswith('<w:br'): out.append('\n')
        else: out.append(html.unescape(m.group(1)))
    return ''.join(out)
P=[ptext(p) for p in paras]
print('paragraphs',len(P),'empty',sum(1 for p in P if p==''))
def body(lines):
    i=lines.index('BEGIN_TBG3B_FROZEN_BODY'); j=lines.index('END_TBG3B_FROZEN_BODY')
    s='\n'.join(lines[i+1:j]).replace('\r\n','\n').replace('\r','\n').rstrip('\n')+'\n'
    e=s.encode('utf-8'); return len(e),hashlib.sha256(e).hexdigest()
print('plain LF join      ',body(P))
D=[]
for p in P:
    D.append(p)
    if p=='': D.append('')
print('double empty paras ',body(D))
# compare against supplied txt (unescaped)
t=open(sys.argv[2],encoding='utf-8').read()
tl=html.unescape(t).split('\n')
print('txt unescaped      ',body(tl))
print('txt raw            ',body(t.split('\n')))
