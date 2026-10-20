import json,sys
d=json.load(open(sys.argv[1]))
print('=====',sys.argv[1])
if 'carrier_claim' in d:
  print(d['carrier_claim']); print('FLAGS'); [print('-',x) for x in d['status_flags']]
  for r in d['rows']:
    print('ROW', r['ingredient'],'|',r['carrier_locus'],'|',r['p2_or_component'],'|',r['classification']); print('   ',r['evidence'])
  print('SUMMARY', d['summary']); print('CAV'); [print('-',x) for x in d['caveats']]
else:
  for k,v in d.items():
    print('##',k)
    if isinstance(v,list):
      for x in v: print('-', json.dumps(x,indent=1) if isinstance(x,dict) else x)
    else: print(v)
