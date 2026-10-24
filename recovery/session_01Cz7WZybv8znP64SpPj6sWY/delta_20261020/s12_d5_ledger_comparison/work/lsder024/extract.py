import sys, zipfile, re, hashlib, html
p=sys.argv[1]
z=zipfile.ZipFile(p)
print(z.namelist())
x=z.read('word/document.xml').decode('utf-8')
paras=re.findall(r'<w:p[ >].*?</w:p>',x,flags=re.S)
out=[]
for para in paras:
    texts=re.findall(r'<w:t(?:\s[^>]*)?>(.*?)</w:t>',para,flags=re.S)
    # handle tabs/breaks
    out.append(html.unescape(''.join(texts)))
txt='\n'.join(out)
open('docx_extract.txt','w').write(txt)
b=txt.split('BEGIN_TBG3B_FROZEN_BODY',1)[1].split('END_TBG3B_FROZEN_BODY',1)[0]
b=b.replace('\r\n','\n').replace('\r','\n')
for lead in [b, b.lstrip('\n')]:
    s=lead.rstrip('\n')+'\n'
    e=s.encode('utf-8')
    print(len(e), hashlib.sha256(e).hexdigest())
