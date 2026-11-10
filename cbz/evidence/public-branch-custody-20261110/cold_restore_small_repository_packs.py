"""Independently restore and check six captured tree and selected-blob packs."""
import hashlib,json,pathlib,shutil,subprocess,sys
root=pathlib.Path(__file__).resolve().parent;names=['meta-framework','query-','google-drive','governance-','sandbox','d6g8k5htny-coder'];results=[]
for name in names:
 d=root/'small-branch-tree-20261110'/name;out=d/'cold-check';out.mkdir(exist_ok=False);repo=out/'cold.git';p=subprocess.run(['git','init','--bare',str(repo)],capture_output=True,text=True)
 if p.returncode:raise ValueError('init '+name)
 packdir=repo/'objects'/'pack'
 for stem,manifest in [('heads-and-trees','pack-manifest.json'),('changed-head-blobs','changed-head-blobs-manifest.json')]:
  m=json.loads((d/manifest).read_text())
  for ext,key in [('pack','pack_sha256'),('idx','index_sha256')]:
   src=d/(stem+'.'+ext)
   if hashlib.sha256(src.read_bytes()).hexdigest()!=m[key]:raise ValueError('pack digest '+name)
   shutil.copy2(src,packdir/(stem+'.'+ext))
  p=subprocess.run(['git','verify-pack','-v',str(packdir/(stem+'.idx'))],capture_output=True,text=True,timeout=30);(out/(stem+'.stdout')).write_text(p.stdout);(out/(stem+'.stderr')).write_text(p.stderr)
  if p.returncode:raise ValueError('verify pack '+name)
 for mode in ('normal','optimized'):
  prefix=[sys.executable,'-B']+(['-O'] if mode=='optimized' else [])+['-S']
  commands=[('tree',prefix+[str(root/'verify_small_repository_trees.py'),'--name',name,'--inventory',str(d/'inventory.json'),'--source',str(root/'federation-branch-head-trees-20261110.json'),'--repo',str(repo)]),('blob',prefix+[str(root/'verify_small_changed_blobs.py'),'--inventory',str(d/'inventory.json'),'--manifest',str(d/'changed-head-blobs-manifest.json'),'--repo',str(repo)])]
  for kind,cmd in commands:
   p=subprocess.run(cmd,capture_output=True,text=True,timeout=45);(out/(kind+'-'+mode+'.stdout')).write_text(p.stdout);(out/(kind+'-'+mode+'.stderr')).write_text(p.stderr);results.append({'repo':name,'mode':mode,'kind':kind,'exit':p.returncode})
   if p.returncode:raise ValueError('readback '+name+' '+mode+' '+kind)
(root/'small-branch-tree-20261110'/'cold-restore-results.json').write_text(json.dumps({'scope':'six public repos, captured tree metadata and only changed head-side blobs; no branch disposition','results':results},indent=2)+'\n');print(len(results),'cold tree/blob readbacks passed across six repositories and both modes')
