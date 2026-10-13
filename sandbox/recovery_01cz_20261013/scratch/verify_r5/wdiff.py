import json,sys,difflib,re
o=json.load(open('old.json'));n=json.load(open('new.json'))
paths=[(0,'kernel_evidence',1,'receipt_or_workflow'),(0,'kernel_evidence',1,'note'),(0,'not_established'),(0,'next_step'),(2,'kernel_evidence',0,'note'),(5,'kernel_evidence',1,'note'),(6,'kernel_evidence',1,'note'),(7,'kernel_evidence',1,'note'),(8,'kernel_evidence',1,'note'),(16,'not_established'),(17,'kernel_evidence',0,'note'),(25,'kernel_evidence',1,'note'),(26,'kernel_evidence',1,'note'),(54,'kernel_evidence',0,'note')]
sel=[int(a) for a in sys.argv[1:]] if len(sys.argv)>1 else range(len(paths))
def get(d,p):
    x=d['claims']
    for k in p: x=x[k]
    return x
for i in sel:
    p=paths[i]
    a=get(o,p);b=get(n,p)
    ta=re.findall(r'\S+|\s+',a);tb=re.findall(r'\S+|\s+',b)
    sm=difflib.SequenceMatcher(None,ta,tb,autojunk=False)
    print('=====',i,p, len(a),'->',len(b))
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal': continue
        print(f'  [{tag}] -<<{"".join(ta[i1:i2])}>>\n         +<<{"".join(tb[j1:j2])}>>')
