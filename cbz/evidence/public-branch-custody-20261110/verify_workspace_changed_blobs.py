"""Check exact selected blob IDs and SHA-256 bytes from a no-lazy-fetch Git store."""
import argparse,hashlib,io,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--manifest',required=True);p.add_argument('--repo',required=True);a=p.parse_args();j=json.loads(pathlib.Path(a.inventory).read_text());m=json.loads(pathlib.Path(a.manifest).read_text())
ids={c['head']['oid'] for x in j['records'] if x['classification']=='nonancestor' for c in x['changes'] if c['head'] and c['head']['type']=='blob'};objs=m['objects'];by={x['git_blob']:x for x in objs}
if len(ids)!=47 or len(objs)!=47 or len(by)!=47 or set(by)!=ids or m['blob_count']!=47:raise ValueError('blob set mismatch')
p=subprocess.Popen(['git','--no-lazy-fetch','-C',a.repo,'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE);raw,err=p.communicate(''.join(x+'\n' for x in sorted(ids)).encode(),timeout=45)
if p.returncode:raise ValueError('git batch failed')
s=io.BytesIO(raw);total=0
for oid in sorted(ids):
 h=s.readline().decode().strip().split()
 if len(h)!=3 or h[0]!=oid or h[1]!='blob':raise ValueError('missing object')
 n=int(h[2]);b=s.read(n)
 if s.read(1)!=b'\n':raise ValueError('batch framing')
 if hashlib.sha1(b'blob '+str(n).encode()+b'\0'+b).hexdigest()!=oid or hashlib.sha256(b).hexdigest()!=by[oid]['sha256'] or n!=by[oid]['bytes']:raise ValueError('blob identity mismatch')
 total+=n
if s.read() or total!=m['blob_bytes']:raise ValueError('total mismatch')
print('47 selected changed head-side blob identities verified; bytes',total,'not complete branch snapshot')
