import sys,hashlib,html,zipfile,re
p_txt,p_docx=sys.argv[1],sys.argv[2]
B='BEGIN_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY';E='END_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY'
def H(s):
    s=s.replace('\r\n','\n').replace('\r','\n').rstrip('\n')+'\n'; bb=s.encode(); return len(bb),hashlib.sha256(bb).hexdigest()[:16]
t=open(p_txt,encoding='utf-8').read()
body=t[t.rindex(B)+len(B):t.rindex(E)]
print('txt raw',H(body)); print('txt unescaped',H(html.unescape(body)))
print('txt unescaped lstrip',H(html.unescape(body).lstrip('\n')))
# doubled blank lines
u=html.unescape(body)
print('txt unesc, blank lines doubled',H(re.sub(r'\n\n','\n\n\n',u)))
z=zipfile.ZipFile(p_docx); print(z.namelist())
x=z.read('word/document.xml').decode()
paras=re.findall(r'<w:p[ >].*?</w:p>|<w:p/>',x,flags=re.S)
L=[]
for p in paras:
    p=re.sub(r'<w:br/>','\n',p); p=re.sub(r'<w:tab/>','\t',p)
    L.append(html.unescape(''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>',p,flags=re.S))))
d='\n'.join(L)
bd=d[d.rindex(B)+len(B):d.rindex(E)]
print('docx paragraphs',H(bd), 'lstrip',H(bd.lstrip('\n')))
print('nparas',len(L),'empty paras',sum(1 for l in L if l==''))
for nm,s in (('doubled+lstrip',re.sub(r'\n\n','\n\n\n',u).lstrip('\n')),('docx doubled+lstrip',re.sub(r'\n\n','\n\n\n',bd).lstrip('\n'))):
    s2=s.replace('\r\n','\n').rstrip('\n')+'\n'; print(nm,len(s2.encode()),hashlib.sha256(s2.encode()).hexdigest())
