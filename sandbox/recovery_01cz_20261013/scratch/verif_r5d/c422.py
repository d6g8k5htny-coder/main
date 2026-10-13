import json,re
d=json.load(open('<SCRATCH>/verif_r5d/diff.json'))
o=json.load(open('<SCRATCH>/verif_r5d/old.json'))
heads={'38116277597':'00d2ac6e','38121713840':'910809d6','38122386459':'d9b8dd33','38150363063':'12e072d6','38180705952':'48bfcbae','38187538744':'be014794'}
merges={'38116277597':'c4c66bb8','38121713840':'5b198699','38122386459':'866136cc','38150363063':'86a6e8c8','38180705952':'9833d6d9','38187538744':'79d3964c'}
def check(text,label):
    for m in re.finditer(r'\b(38\d{9})\b',text):
        r=m.group(1)
        win=text[max(0,m.start()-260):m.end()+260]
        h=heads.get(r); mg=merges.get(r)
        if h is None: print(label,'UNKNOWN RUN',r); continue
        okh = h in win; okm = mg in win
        if not (okh and okm):
            print(label,r,'head' if okh else 'NOHEAD','merge' if okm else 'NOMERGE','|',text[max(0,m.start()-160):m.end()+160])
for k,(a,b) in d['changed'].items():
    check(b,'NEW '+k)
for k,v in d['added'].items():
    if isinstance(v,str): check(v,'ADD '+k)
