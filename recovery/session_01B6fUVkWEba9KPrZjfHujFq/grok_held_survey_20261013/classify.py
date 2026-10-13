import json,re,collections
d=json.load(open('all2.json'))
ACT=re.compile(r'(GROK-H0\d|bc-[0-9a-f]{8}-[0-9a-f-]{27,})',re.I)
def isx(b):
    h=b[:500]
    return bool(re.search(r'(performer|actor|worker|from|session|agent)[^\n]{0,100}(GROK-H0\d|Grok|xAI|Cursor|bc-[0-9a-f]{8})',h,re.I) or re.match(r'\W*(GROK-H0\d|\[GROK|Grok|xAI|Cursor)',b,re.I))
TYPES=[('RELEASE',r'\b(own release|released|RELEASE[D]?\b|releasing)'),('DELIVERY',r'\b(DELIVER(Y|ED)|COMPLETE[D]?\b|PASS\b|delivered)'),('BLOCKED',r'\bBLOCK(ED|ER)\b'),('CHECKPOINT',r'\bCHECKPOINT\b'),('RESUME',r'\bRESUME'),('PICKUP',r'\b(PICKUP|CLAIM(ED)?|picks? up|claiming)\b'),('HOLD',r'\bHOLD\b'),('OFFER',r'\bOFFER'),('ACK',r'\bACK\b')]
ev=[]
for k,v in d.items():
    for c in v.get('comments',[])+v.get('reviews',[]):
        b=c.get('body') or ''
        if not isx(b): continue
        m=ACT.search(b[:1500])
        actor=m.group(1) if m else ('GROK?' if re.search('grok|xai',b[:500],re.I) else 'CURSOR?')
        if actor.lower().startswith('bc-'): actor=actor[:11]
        first=b[:300]
        t='OTHER'
        for name,p in TYPES:
            if re.search(p,first,re.I): t=name;break
        ck=re.findall(r'\b([A-Z][A-Z0-9]{1,12}(?:-[A-Z0-9]{1,12}){1,4})\b',b[:600])
        ck=[x for x in ck if not x.startswith('GROK-H') and x not in ('NONE','UTC')]
        ev.append(dict(t_=c.get('created_at') or '',where=k,id=c['id'],actor=actor.upper() if 'GROK' in actor.upper() else actor,type=t,keys=ck[:4],head=first.replace('\n',' ')[:220]))
ev.sort(key=lambda e:e['t_'])
json.dump(ev,open('xai_events.json','w'))
print(len(ev), collections.Counter(e['actor'] for e in ev).most_common())
