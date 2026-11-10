"""Bounded adverse controls for the captured Workspace tree readback."""
import base64,copy,json,pathlib,shutil,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;d=r/'workspace-branch-trees-20261110';out=d/'adverse';out.mkdir(exist_ok=False);base=json.loads((d/'inventory.json').read_text());source=r/'federation-branch-head-trees-20261110.json'; verifier=r/'verify_workspace_branch_tree_inventory.py'
cold=out/'cold.git';p=subprocess.run(['git','init','--bare',str(cold)],capture_output=True,text=True)
if p.returncode:raise RuntimeError('cold init failed')
packdir=cold/'objects'/'pack';shutil.copy2(d/'head-commits-and-trees.pack',packdir/'observed.pack');shutil.copy2(d/'head-commits-and-trees.idx',packdir/'observed.idx')
vp=subprocess.run(['git','verify-pack','-v',str(packdir/'observed.idx')],capture_output=True,text=True)
(out/'cold-verify-pack.stdout').write_text(vp.stdout);(out/'cold-verify-pack.stderr').write_text(vp.stderr)
if vp.returncode:raise RuntimeError('cold pack failed')
# A selected real changed leaf supports the malformed report controls.
idx=next(i for i,x in enumerate(base['records']) if x['changes']); x=base['records'][idx]['changes'][0]; cases={}
def case(name,fn):
 v=copy.deepcopy(base);fn(v);p=out/(name+'.json');p.write_text(json.dumps(v,indent=2)+'\n');cases[name]=p
case('missing-ref',lambda x:x['records'].pop())
case('duplicate-ref',lambda x:x['records'].append(copy.deepcopy(x['records'][0])))
case('wrong-tree',lambda x:x['records'][idx].__setitem__('tree','0'*40))
case('missing-change',lambda x:x['records'][idx]['changes'].pop(0))
case('duplicate-change',lambda x:x['records'][idx]['changes'].append(copy.deepcopy(x['records'][idx]['changes'][0])))
case('invalid-path',lambda x:x['records'][idx]['changes'][0].__setitem__('path_b64','!!!'))
case('wrong-oid',lambda x:x['records'][idx]['changes'][0].__setitem__('head',{'mode':'100644','type':'blob','oid':'1'*40}))
case('wrong-display',lambda x:x['records'][idx]['changes'][0].__setitem__('path_utf8','incorrect'))
code=verifier.read_text();needle="if claimed!=actual:raise ValueError('diff mismatch')"
if code.count(needle)!=1:raise RuntimeError('mutant anchor')
mutant=out/'verify-mutant.py';mutant.write_text(code.replace(needle,"if False:raise ValueError('diff mismatch')"))
results=[]
def run(label,script,inv,mode,should_pass):
 cmd=[sys.executable,'-B']+(['-O'] if mode=='optimized' else [])+['-S',str(script),'--inventory',str(inv),'--source',str(source),'--repo',str(cold)];p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 (out/(label+'-'+mode+'.stdout')).write_text(p.stdout);(out/(label+'-'+mode+'.stderr')).write_text(p.stderr)
 ok=(p.returncode==0)==should_pass;results.append({'label':label,'mode':mode,'exit':p.returncode,'expected_accept':should_pass,'matched':ok})
 if not ok:raise RuntimeError('unexpected outcome '+label+' '+mode)
for mode in ('normal','optimized'):
 run('clean',verifier,d/'inventory.json',mode,True)
 for name,inv in cases.items():run(name,verifier,inv,mode,False)
 # The intentionally unsound verifier accepts the dropped change; that is a detected mutant, not a pass.
 run('mutant-missing-change',mutant,cases['missing-change'],mode,True)
# Git's pack-level integrity check must reject both a missing and altered pack.
for name in ('missing','corrupt'):
 sub=out/(name+'-pack');sub.mkdir();shutil.copy2(d/'head-commits-and-trees.idx',sub/'bad.idx')
 if name=='corrupt':
  b=bytearray((d/'head-commits-and-trees.pack').read_bytes());b[len(b)//2]^=1;(sub/'bad.pack').write_bytes(b)
 p=subprocess.run(['git','verify-pack','-v',str(sub/'bad.idx')],capture_output=True,text=True,timeout=30);(out/(name+'-pack.stdout')).write_text(p.stdout);(out/(name+'-pack.stderr')).write_text(p.stderr);results.append({'label':name+'-pack','exit':p.returncode,'expected_accept':False,'matched':p.returncode!=0})
 if p.returncode==0:raise RuntimeError('damaged pack accepted')
(out/'results.json').write_text(json.dumps({'scope':'Specified report corruptions, one verifier mutant, cold restore, and Git pack damage; no exhaustive soundness claim','results':results},indent=2)+'\n')
print(len(results),'actual children; 2 clean passes, 16 invalid report rejections, 2 detected verifier mutant executions, 2 damaged pack rejections')
