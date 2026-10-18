import json,re,sys
pats=[r'LS-DER-0?26',r'LS-DER-0?32',r'LS-DER-0?38',r'RV-DQ-0?35',r'RV-DQ-0?53',r'RV-DQ-0?61',r'TB-G6',r'TB-G7',r'RV-DQ-0?23',r'\bDQ-023',r'\bA4\b',r'A2.{0,20}(uniform )?gap',r'second[- ]provider',r'§\s?12\b',r'X4',r'RECONCILIATION',r'LS-REQ-033',r'LS-REQ-022']
for r in ['main','Math-']:
  for l in open(sys.argv[1]+'/'+r+'.jsonl'):
    c=json.loads(l)
    if c['created_at']<'2026-10-17T18:00:00Z': continue
    b=c['body'] or ''
    hits=[p for p in pats if re.search(p,b)]
    if hits:
      print(r, c['id'], c['created_at'], c['issue_url'].split('/')[-1], hits, '|', b[:160].replace('\n',' '))
