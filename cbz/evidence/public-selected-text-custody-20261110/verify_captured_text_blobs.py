"""Independently reselect captured text paths and verify exact native blob bytes."""
import argparse,base64,gzip,hashlib,io,json,pathlib,re,subprocess
p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--selection-dir',required=True);p.add_argument('--repo',required=True);p.add_argument('--source',required=True);a=p.parse_args();d=pathlib.Path(a.selection_dir);m=json.loads(pathlib.Path(a.inventory).read_text());s=json.loads((d/'selection.json').read_text());bman=json.loads((d/'blob-manifest.json').read_text());name=m['repository'].split('/')[-1];source=json.loads(pathlib.Path(a.source).read_text());sr=[x for x in source['records'] if x['repository']==m['repository'] and x['classification']=='nonancestor'];expected_refs={x['ref']:(x['sha'],x['tree']) for x in sr};actual_refs={x['ref']:(x['sha'],x['tree']) for x in m['refs']}
if not sr or len(expected_refs)!=len(sr) or len(actual_refs)!=len(m['refs']) or actual_refs!=expected_refs or {x['observed_default'] for x in sr}!={m['observed_default']} or set(m['heads'])!={x[0] for x in expected_refs.values()}:raise ValueError('captured source refs')
if name not in ('main','Math-') or s['repository']!=m['repository'] or s['source_default']!=m['observed_default']:raise ValueError('source identity')
suffixes=(b'.lean',) if name=='main' else (b'.lean',b'.md');expected_selector='case-sensitive head-side changed blob paths ending '+', '.join(x.decode() for x in suffixes)
if s['selector']!=expected_selector or bman['selector']!=expected_selector:raise ValueError('selector')
ids=(d/'object-ids.txt').read_text().splitlines();blobmap={x['git_blob']:x for x in bman['objects']}
if not ids or ids!=sorted(set(ids)) or len(ids)!=s['unique_blob_ids'] or len(ids)!=bman['blob_count'] or len(blobmap)!=len(ids) or set(ids)!=set(blobmap) or hashlib.sha256((d/'object-ids.txt').read_bytes()).hexdigest()!=s['object_ids_sha256']:raise ValueError('blob ID set')
compressed=(d/'bindings.jsonl.gz').read_bytes()
if hashlib.sha256(compressed).hexdigest()!=s['bindings_gzip_sha256']:raise ValueError('binding digest')
stream=io.BytesIO(gzip.decompress(compressed));seen=set();count=0;pattern=re.compile(rb':([0-7]{6}) ([0-7]{6}) ([0-9a-f]{40}) ([0-9a-f]{40}) ([ADMT])')
for head,v in sorted(m['heads'].items()):
 if not re.fullmatch('[0-9a-f]{40}',head):raise ValueError('head syntax')
 if subprocess.check_output(['git','--no-lazy-fetch','-C',a.repo,'rev-parse',head+'^{tree}'],stderr=subprocess.PIPE).strip().decode()!=v['tree']:raise ValueError('head tree identity')
 raw=subprocess.check_output(['git','--no-lazy-fetch','-C',a.repo,'diff-tree','--no-commit-id','--no-renames','--raw','--no-abbrev','-r','-z',m['observed_default'],head],stderr=subprocess.PIPE,timeout=45)
 if len(raw)!=v['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=v['raw_sha256']:raise ValueError('source diff mismatch')
 parts=raw.split(b'\0')
 if parts[-1:]!=[b''] or len(parts)%2!=1:raise ValueError('diff framing')
 for i in range(0,len(parts)-1,2):
  z=pattern.fullmatch(parts[i]);path=parts[i+1]
  if not z or not path:raise ValueError('diff entry')
  mode,oid,status=z.group(2),z.group(4),z.group(5)
  if mode not in (b'100644',b'100755',b'120000') or not path.endswith(suffixes):continue
  expected={'head':head,'path_b64':base64.b64encode(path).decode(),'mode':mode.decode(),'blob':oid.decode(),'status':status.decode()};line=stream.readline()
  if not line:raise ValueError('missing binding')
  row=json.loads(line)
  if row!=expected:raise ValueError('binding mismatch')
  seen.add(oid.decode());count+=1
if stream.read() or count!=s['bindings'] or seen!=set(ids):raise ValueError('selection mismatch')
p=subprocess.Popen(['git','--no-lazy-fetch','-C',a.repo,'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);raw,err=p.communicate(''.join(x+'\n' for x in ids).encode(),timeout=45)
if p.returncode:raise ValueError('blob read')
f=io.BytesIO(raw);total=0
for oid in ids:
 h=f.readline().decode().strip().split()
 if len(h)!=3 or h[0]!=oid or h[1]!='blob':raise ValueError('missing blob')
 n=int(h[2]);data=f.read(n)
 if f.read(1)!=b'\n' or hashlib.sha1(b'blob '+str(n).encode()+b'\0'+data).hexdigest()!=oid or hashlib.sha256(data).hexdigest()!=blobmap[oid]['sha256'] or n!=blobmap[oid]['bytes']:raise ValueError('blob identity')
 total+=n
if f.read() or total!=bman['raw_bytes']:raise ValueError('blob total')
print(name,'verified',count,'path bindings and',len(ids),'selected blob objects,',total,'raw bytes; suffix selection is not proof classification')
