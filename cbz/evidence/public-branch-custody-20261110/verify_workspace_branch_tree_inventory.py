"""Independently compare one captured Workspace tree inventory to Git diff-tree."""
import argparse,base64,json,pathlib,re,subprocess
P=argparse.ArgumentParser();P.add_argument('--inventory',required=True);P.add_argument('--source',required=True);P.add_argument('--repo',required=True);args=P.parse_args()
r=json.loads(pathlib.Path(args.inventory).read_text());s=json.loads(pathlib.Path(args.source).read_text());src=[x for x in s['records'] if x['repository']=='d6g8k5htny-coder/Universal-Law-Workspace'];
if len(src)!=12 or len({x['ref'] for x in src})!=12:raise ValueError('source refs')
if r.get('schema')!='cbz.workspace-branch-tree-inventory.v1' or r.get('observed_default')!=src[0]['observed_default']:raise ValueError('source identity')
rows=r['records']; expected_refs={x['ref']:x for x in src}
if len(rows)!=12 or len({x['ref'] for x in rows})!=12 or {x['ref'] for x in rows}!=set(expected_refs):raise ValueError('ref set')
def git(*xs):return subprocess.check_output(['git','--no-lazy-fetch','-C',args.repo,*xs],stderr=subprocess.PIPE)
def entry(mode,oid):
 if mode=='000000' and oid=='0'*40:return None
 if not re.fullmatch('[0-9a-f]{40}',oid):raise ValueError('object id')
 if mode not in ('100644','100755','120000','160000'):raise ValueError('mode')
 return {'mode':mode,'type':'commit' if mode=='160000' else 'blob','oid':oid}
for row in rows:
 original=expected_refs[row['ref']]
 if any(row[k]!=original[k] for k in ('sha','tree','classification')):raise ValueError('ref identity')
 if git('rev-parse',row['sha']+'^{tree}').strip().decode()!=row['tree']:raise ValueError('tree identity')
 output=git('diff-tree','--no-commit-id','--no-renames','--raw','--no-abbrev','-r','-z',r['observed_default'],row['sha']).split(b'\0')
 if output[-1:]!=[b''] or len(output)%2!=1:raise ValueError('diff encoding')
 actual={}
 for i in range(0,len(output)-1,2):
  head,path=output[i:i+2]; m=re.fullmatch(rb':([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([ADMT])',head)
  if not m or not path or path in actual:raise ValueError('diff record')
  om,nm,oo,no,_=(v.decode() for v in m.groups());actual[path]={'default':entry(om,oo),'head':entry(nm,no)}
 claimed={}
 for c in row['changes']:
  try:path=base64.b64decode(c['path_b64'],validate=True)
  except Exception as e:raise ValueError('invalid base64 path') from e
  if not path or path in claimed:raise ValueError('duplicate or empty path')
  try:display=path.decode('utf-8')
  except UnicodeDecodeError:display=None
  if c.get('path_utf8')!=display:raise ValueError('path display mismatch')
  claimed[path]={'default':c['default'],'head':c['head']}
 if claimed!=actual:raise ValueError('diff mismatch')
print('verified 12 captured refs and each changed path/mode/object identity against Git diff-tree; source blob bytes not examined')
