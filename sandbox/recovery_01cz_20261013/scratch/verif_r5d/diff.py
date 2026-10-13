import json,sys
d='<SCRATCH>/verif_r5d/'
o=json.load(open(d+'old.json'));n=json.load(open(d+'new.json'))
def leaves(x,p=''):
    if isinstance(x,dict):
        for k,v in x.items(): yield from leaves(v,p+'/'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from leaves(v,p+'/'+str(i))
    else: yield p,x
lo=dict(leaves(o)); ln=dict(leaves(n))
ch=[k for k in lo if k in ln and lo[k]!=ln[k]]
rm=[k for k in lo if k not in ln]; ad=[k for k in ln if k not in lo]
print('changed',len(ch)); 
for k in ch: print(' C',k)
print('removed',len(rm)); 
for k in rm: print(' R',k)
print('added',len(ad));
for k in ad: print(' A',k)
# key order check
def keyorder(x,p=''):
    if isinstance(x,dict):
        yield p,list(x.keys())
        for k,v in x.items(): yield from keyorder(v,p+'/'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x): yield from keyorder(v,p+'/'+str(i))
ko=dict(keyorder(o)); kn=dict(keyorder(n))
for k in ko:
    if k in kn and ko[k]!=kn[k]: print('KEYORDER',k)
json.dump({'changed':{k:[lo[k],ln[k]] for k in ch},'added':{k:ln[k] for k in ad}},open(d+'diff.json','w'),indent=1,ensure_ascii=False)
