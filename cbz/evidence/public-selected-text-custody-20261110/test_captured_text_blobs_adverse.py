"""Specified negative controls on one actual captured Math head and blobs."""
import base64,copy,gzip,hashlib,json,pathlib,shutil,subprocess,sys
r=pathlib.Path(__file__).resolve().parent;d=r/'large-nonancestor-tree-20261110'/'Math-';sel=d/'selected-text-sources';out=r/'captured-text-adverse-20261110';out.mkdir(exist_ok=False);inventory=json.loads((d/'inventory.json').read_text());selection=json.loads((sel/'selection.json').read_text());blobm=json.loads((sel/'blob-manifest.json').read_text());rows=[json.loads(x) for x in gzip.decompress((sel/'bindings.jsonl.gz').read_bytes()).splitlines()];head=rows[0]['head'];rows=[x for x in rows if x['head']==head];ids=sorted({x['blob'] for x in rows});obj={x['git_blob']:x for x in blobm['objects']}
if not rows or not ids:raise ValueError('empty actual sample')
inv=copy.deepcopy(inventory);inv['heads']={head:inventory['heads'][head]};inv['refs']=[x for x in inventory['refs'] if x['sha']==head];invpath=out/'inventory.json';invpath.write_text(json.dumps(inv,indent=2)+'\n');source=json.loads((r/'source.json').read_text());names={x['ref'] for x in inv['refs']};source['records']=[x for x in source['records'] if x['repository']==inventory['repository'] and x['ref'] in names];srcpath=out/'source.json';srcpath.write_text(json.dumps(source,indent=2)+'\n');base=out/'clean';base.mkdir();selbase=copy.deepcopy(selection);selbase['bindings']=len(rows);selbase['unique_blob_ids']=len(ids);bm=copy.deepcopy(blobm);bm['objects']=[obj[x] for x in ids];bm['blob_count']=len(ids);bm['raw_bytes']=sum(x['bytes'] for x in bm['objects'])
def write_case(path,rs,ids,s,b):
 path.mkdir();binding=gzip.compress(b''.join((json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode() for x in rs),compresslevel=6,mtime=0);idbytes=''.join(x+'\n' for x in ids).encode();s['bindings_gzip_sha256']=hashlib.sha256(binding).hexdigest();s['object_ids_sha256']=hashlib.sha256(idbytes).hexdigest();(path/'bindings.jsonl.gz').write_bytes(binding);(path/'object-ids.txt').write_bytes(idbytes);(path/'selection.json').write_text(json.dumps(s,indent=2)+'\n');(path/'blob-manifest.json').write_text(json.dumps(b,indent=2)+'\n')
base.rmdir();write_case(base,rows,ids,selbase,bm);cases=[]
def case(name,fn):
 rs=copy.deepcopy(rows);is_=ids.copy();s=copy.deepcopy(selbase);b=copy.deepcopy(bm);fn(rs,is_,s,b);p=out/name;write_case(p,rs,is_,s,b);cases.append((name,p))
case('missing-binding',lambda x,i,s,b:(x.pop(),s.__setitem__('bindings',len(x))))
case('duplicate-binding',lambda x,i,s,b:(x.append(copy.deepcopy(x[0])),s.__setitem__('bindings',len(x))))
case('wrong-path',lambda x,i,s,b:x[0].__setitem__('path_b64',base64.b64encode(b'wrong/path.lean').decode()))
case('wrong-head',lambda x,i,s,b:x[0].__setitem__('head','0'*40))
case('wrong-mode',lambda x,i,s,b:x[0].__setitem__('mode','100755' if x[0]['mode']!='100755' else '100644'))
case('wrong-blob',lambda x,i,s,b:x[0].__setitem__('blob','1'*40))
case('wrong-selector',lambda x,i,s,b:s.__setitem__('selector','all proofs'))
case('wrong-default',lambda x,i,s,b:s.__setitem__('source_default','0'*40))
case('wrong-blob-sha',lambda x,i,s,b:b['objects'][0].__setitem__('sha256','0'*64))
case('wrong-blob-size',lambda x,i,s,b:b['objects'][0].__setitem__('bytes',b['objects'][0]['bytes']+1))
case('wrong-blob-total',lambda x,i,s,b:b.__setitem__('raw_bytes',b['raw_bytes']+1))
case('missing-id',lambda x,i,s,b:i.pop())
verifier=r/'verify_captured_text_blobs.py';code=verifier.read_text();needle="if row!=expected:raise ValueError('binding mismatch')"
if code.count(needle)!=1:raise ValueError('mutant anchor')
mutant=out/'verify-mutant.py';mutant.write_text(code.replace(needle,"if False:raise ValueError('binding mismatch')"));source_needle="if not sr or len(expected_refs)!=len(sr) or len(actual_refs)!=len(m['refs']) or actual_refs!=expected_refs or {x['observed_default'] for x in sr}!={m['observed_default']} or set(m['heads'])!={x[0] for x in expected_refs.values()}:raise ValueError('captured source refs')"
if code.count(source_needle)!=1:raise ValueError('source mutant anchor')
source_mutant=out/'verify-source-mutant.py';source_mutant.write_text(code.replace(source_needle,"if False:raise ValueError('captured source refs')"));sources={}
for label,edit in [('missing-source',lambda x:x.pop()),('duplicate-source',lambda x:x.append(copy.deepcopy(x[0]))),('wrong-source-tree',lambda x:x[0].__setitem__('tree','0'*40))]:
 v=copy.deepcopy(source);edit(v['records']);sp=out/(label+'.json');sp.write_text(json.dumps(v,indent=2)+'\n');sources[label]=sp
repo=sel/'cold-check'/'cold.git';results=[]
def run(label,script,p,mode,expected,source_path=srcpath):
 cmd=[sys.executable,'-B']+(['-O'] if mode=='optimized' else [])+['-S',str(script),'--inventory',str(invpath),'--selection-dir',str(p),'--repo',str(repo),'--source',str(source_path)];p1=subprocess.run(cmd,capture_output=True,text=True,timeout=30);(out/(label+'-'+mode+'.stdout')).write_text(p1.stdout);(out/(label+'-'+mode+'.stderr')).write_text(p1.stderr);ok=(p1.returncode==0)==expected;results.append({'case':label,'mode':mode,'exit':p1.returncode,'expected_accept':expected,'matched':ok})
 if not ok:raise ValueError('unexpected '+label+' '+mode)
for mode in ('normal','optimized'):
 run('clean',verifier,base,mode,True)
 for label,p in cases:run(label,verifier,p,mode,False)
 run('mutant-wrong-path',mutant,dict(cases)['wrong-path'],mode,True)
 for label,sp in sources.items():run(label,verifier,base,mode,False,sp)
 run('mutant-missing-source',source_mutant,base,mode,True,sources['missing-source'])
for label in ('missing','corrupt'):
 sub=out/(label+'-pack');sub.mkdir();shutil.copy2(sel/'selected-blobs.idx',sub/'bad.idx')
 if label=='corrupt':
  b=bytearray((sel/'selected-blobs.pack').read_bytes());b[len(b)//2]^=1;(sub/'bad.pack').write_bytes(b)
 p=subprocess.run(['git','verify-pack','-v',str(sub/'bad.idx')],capture_output=True,text=True,timeout=30);(out/(label+'-pack.stdout')).write_text(p.stdout);(out/(label+'-pack.stderr')).write_text(p.stderr);results.append({'case':label+'-pack','exit':p.returncode,'expected_accept':False,'matched':p.returncode!=0})
 if p.returncode==0:raise ValueError('damaged pack accepted')
(out/'results.json').write_text(json.dumps({'scope':'one actual captured Math head and selected blobs; specified malformed selections and one verifier mutant','results':results},indent=2)+'\n');print(len(results),'children: 2 clean passes, 30 altered-input rejections, 4 detected mutant executions, 2 damaged-pack rejections')
