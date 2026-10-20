import sys, hashlib, html, zipfile, re
txt = open(sys.argv[1], encoding='utf-8').read()
b0 = 'BEGIN_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY'; b1='END_TBG3B_ENDPOINT_DENSITY_FROZEN_BODY'
def h(s):
    s = s.replace('\r\n','\n').replace('\r','\n').rstrip('\n') + '\n'
    if s.startswith('\n'): pass
    bb = s.encode('utf-8')
    return len(bb), hashlib.sha256(bb).hexdigest()
i = txt.rindex(b0)+len(b0); j = txt.rindex(b1)
raw = txt[i:j]
for name, s in [('raw', raw), ('raw-lstrip1', raw.lstrip('\n')), ('unesc', html.unescape(raw)), ('unesc-lstrip1', html.unescape(raw).lstrip('\n'))]:
    print(name, *h(s))
# docx
z = zipfile.ZipFile(sys.argv[2]); x = z.read('word/document.xml').decode('utf-8')
paras = re.findall(r'<w:p[ >].*?</w:p>', x, flags=re.S)
lines=[]
for p in paras:
    p2 = re.sub(r'<w:br/>', '\n', p)
    p2 = re.sub(r'<w:tab/>', '\t', p2)
    t = ''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p2, flags=re.S))
    lines.append(html.unescape(t))
d = '\n'.join(lines)
i = d.rindex(b0)+len(b0); j = d.rindex(b1)
raw = d[i:j]
for name, s in [('docx', raw), ('docx-lstrip1', raw.lstrip('\n'))]:
    print(name, *h(s))
