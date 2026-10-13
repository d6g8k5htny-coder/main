import json,re,sys
d=json.load(open('all2.json'))
pat=re.compile(sys.argv[1],re.I); since=sys.argv[2]
rows=[]
for k,v in d.items():
    for c in v.get('comments',[])+v.get('reviews',[]):
        b=c.get('body') or ''
        t=c.get('created_at') or ''
        if t>=since and pat.search(b): rows.append((t,k,c['id'],re.sub(r'\s+',' ',b[:int(sys.argv[3]) if len(sys.argv)>3 else 220])))
for r in sorted(rows): print(r[0][5:16],r[1],r[2],'|',r[3])
