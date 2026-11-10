"""Cold restore selected text blobs and previously captured head/tree pack."""
import hashlib,json,pathlib,shutil,subprocess,sys
root=pathlib.Path(__file__).resolve().parent;name=sys.argv[1]
if name not in ('main','Math-'):raise ValueError('unsupported')
d=root/'large-nonancestor-tree-20261110'/name;sel=d/'selected-text-sources';m=json.loads((d/'inventory.json').read_text());b=json.loads((sel/'blob-manifest.json').read_text());out=sel/'cold-check';out.mkdir(exist_ok=False);repo=out/'cold.git';subprocess.run(['git','init','--bare','-q',str(repo)],check=True);dest=repo/'objects'/'pack'
for folder,stem,manifest in [(d,'heads-and-trees',m),(sel,'selected-blobs',b)]:
 for ext,key in [('pack','pack_sha256'),('idx','index_sha256')]:
  source=folder/(stem+'.'+ext)
  if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest[key]:raise ValueError('pack hash')
  shutil.copy2(source,dest/(stem+'.'+ext))
 p=subprocess.run(['git','verify-pack','-v',str(dest/(stem+'.idx'))],capture_output=True,text=True,timeout=45);(out/(stem+'.stdout')).write_text(p.stdout);(out/(stem+'.stderr')).write_text(p.stderr)
 if p.returncode:raise ValueError('pack verification')
results=[]
for mode,flags in [('normal',[]),('optimized',['-O'])]:
 p=subprocess.run([sys.executable,'-B',*flags,'-S',str(root/'verify_captured_text_blobs.py'),'--inventory',str(d/'inventory.json'),'--selection-dir',str(sel),'--repo',str(repo),'--source',str(root/'source.json')],capture_output=True,text=True,timeout=180);(out/(mode+'.stdout')).write_text(p.stdout);(out/(mode+'.stderr')).write_text(p.stderr);results.append({'mode':mode,'exit':p.returncode,'stdout':p.stdout.strip()})
 if p.returncode:raise ValueError('selected source readback '+mode)
(out/'results.json').write_text(json.dumps({'repository':name,'scope':'explicit suffix selection only','results':results},indent=2)+'\n');print(name,'normal and optimized cold selection/blob readbacks passed')
