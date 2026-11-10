"""Cold-restore and specified hostile inputs for selected Workspace blob evidence."""
import copy,json,pathlib,shutil,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;d=r/'small-branch-tree-20261110'/'query-';out=d/'blob-adverse';out.mkdir(exist_ok=False);repo=out/'cold.git'
p=subprocess.run(['git','init','--bare',str(repo)],capture_output=True,text=True)
if p.returncode:raise RuntimeError('init')
packdir=repo/'objects'/'pack'
for stem in ('heads-and-trees','changed-head-blobs'):
 for ext in ('pack','idx'):shutil.copy2(d/(stem+'.'+ext),packdir/(stem+'.'+ext))
 p=subprocess.run(['git','verify-pack','-v',str(packdir/(stem+'.idx'))],capture_output=True,text=True);(out/(stem+'-verify.stdout')).write_text(p.stdout);(out/(stem+'-verify.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError('bad original pack')
m=json.loads((d/'changed-head-blobs-manifest.json').read_text());cases={}
def change(name,fn):
 v=copy.deepcopy(m);fn(v);q=out/(name+'.json');q.write_text(json.dumps(v,indent=2)+'\n');cases[name]=q
change('missing-object',lambda x:x['objects'].pop())
change('duplicate-object',lambda x:x['objects'].append(copy.deepcopy(x['objects'][0])))
change('wrong-sha256',lambda x:x['objects'][0].__setitem__('sha256','0'*64))
change('wrong-size',lambda x:x['objects'][0].__setitem__('bytes',x['objects'][0]['bytes']+1))
change('wrong-id',lambda x:x['objects'][0].__setitem__('git_blob','1'*40))
change('wrong-total',lambda x:x.__setitem__('blob_bytes',x['blob_bytes']+1))
script=r/'verify_small_changed_blobs.py';code=script.read_text();needle="if hashlib.sha1(b'blob '+str(n).encode()+b'\\0'+b).hexdigest()!=oid or hashlib.sha256(b).hexdigest()!=by[oid]['sha256'] or n!=by[oid]['bytes']:raise ValueError('blob identity mismatch')"
if code.count(needle)!=1:raise RuntimeError('mutant anchor')
mutant=out/'verify-mutant.py';mutant.write_text(code.replace(needle,"if False:raise ValueError('blob identity mismatch')"));results=[]
def run(label,script,manifest,mode,expected):
 cmd=[sys.executable,'-B']+(['-O'] if mode=='optimized' else [])+['-S',str(script),'--inventory',str(d/'inventory.json'),'--manifest',str(manifest),'--repo',str(repo)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(out/(label+'-'+mode+'.stdout')).write_text(p.stdout);(out/(label+'-'+mode+'.stderr')).write_text(p.stderr);ok=(p.returncode==0)==expected;results.append({'case':label,'mode':mode,'exit':p.returncode,'expected_accept':expected,'matched':ok})
 if not ok:raise RuntimeError(label+' '+mode)
for mode in ('normal','optimized'):
 run('clean',script,d/'changed-head-blobs-manifest.json',mode,True)
 for name,path in cases.items():run(name,script,path,mode,False)
 run('mutant-wrong-sha256',mutant,cases['wrong-sha256'],mode,True)
 # independently recheck path identities against the restored tree pack
 cmd=[sys.executable,'-B']+(['-O'] if mode=='optimized' else [])+['-S',str(r/'verify_small_repository_trees.py'),'--name','query-','--inventory',str(d/'inventory.json'),'--source',str(r/'federation-branch-head-trees-20261110.json'),'--repo',str(repo)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(out/('tree-'+mode+'.stdout')).write_text(p.stdout);(out/('tree-'+mode+'.stderr')).write_text(p.stderr);results.append({'case':'cold-tree','mode':mode,'exit':p.returncode,'expected_accept':True,'matched':p.returncode==0})
 if p.returncode:raise RuntimeError('cold tree failure')
for name in ('missing','corrupt'):
 sub=out/(name+'-pack');sub.mkdir();shutil.copy2(d/'changed-head-blobs.idx',sub/'bad.idx')
 if name=='corrupt':
  b=bytearray((d/'changed-head-blobs.pack').read_bytes());b[len(b)//2]^=1;(sub/'bad.pack').write_bytes(b)
 p=subprocess.run(['git','verify-pack','-v',str(sub/'bad.idx')],capture_output=True,text=True,timeout=30);(out/(name+'-pack.stdout')).write_text(p.stdout);(out/(name+'-pack.stderr')).write_text(p.stderr);results.append({'case':name+'-pack','exit':p.returncode,'expected_accept':False,'matched':p.returncode!=0})
 if p.returncode==0:raise RuntimeError('bad pack accepted')
(out/'results.json').write_text(json.dumps({'scope':'selected query blobs only; specified hostile manifests, cold Git restore and pack corruption','results':results},indent=2)+'\n');print(len(results),'children: 4 cold tree/blob passes, 12 hostile manifest rejections, 2 detected mutant executions, 2 damaged pack rejections')
